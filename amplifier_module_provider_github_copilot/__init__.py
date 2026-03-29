"""GitHub Copilot Provider for Amplifier.

Three-Medium Architecture:
- Python for mechanism (~300 lines)
- YAML for policy (~200 lines)
- Markdown for contracts (~400 lines)

Contract: contracts/provider-protocol.md
"""

from __future__ import annotations

import logging as _logging
import os as _os_logging

# Configure debug logging if GHCP_DEBUG environment variable is set
# Usage: GHCP_DEBUG=1 amplifier run "your prompt"
#
# SECURITY: Logs go to stderr ONLY (captured by Amplifier's logging infrastructure).
# File-based logging is intentionally NOT supported to prevent:
# - Sensitive data (prompts, tokens, tool outputs) persisting on disk
# - World-readable log files in /tmp/ or other locations
# - Compliance violations (GDPR, SOC2, etc.)
if _os_logging.environ.get("GHCP_DEBUG"):
    _stderr_handler = _logging.StreamHandler()
    _stderr_handler.setFormatter(
        _logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s", "%H:%M:%S")
    )

    _logging.basicConfig(
        level=_logging.DEBUG,
        handlers=[_stderr_handler],
    )

    # Enable debug for all our modules
    for _logger_name in [
        "amplifier_module_provider_github_copilot",
        "amplifier_module_provider_github_copilot.provider",
        "amplifier_module_provider_github_copilot.request_adapter",
        "amplifier_module_provider_github_copilot.streaming",
        "amplifier_module_provider_github_copilot.sdk_adapter",
        "amplifier_module_provider_github_copilot.sdk_adapter.client",
        "amplifier_module_provider_github_copilot.sdk_adapter.event_helpers",
        "amplifier_module_provider_github_copilot.sdk_adapter.tool_capture",
    ]:
        _logging.getLogger(_logger_name).setLevel(_logging.DEBUG)

    _logging.getLogger(__name__).debug("GHCP_DEBUG enabled - verbose logging to stderr")

# Eager dependency check: ensure github-copilot-sdk is installed.
# All SDK imports in this module are lazy (inside function bodies) so the module
# would otherwise import successfully without the SDK. That tricks Amplifier's
# provider discovery into thinking the module is fully functional, which prevents
# the automatic dependency-installation fallback from ever running.
# Using importlib.metadata avoids importing the SDK itself at module load time.
# Contract: sdk-boundary.md MUST:5
#
# P1-6 Security Fix: SDK check bypass only allowed in pytest context.
# The env var alone is not sufficient - pytest must be loaded in sys.modules.
# This prevents production misuse while preserving test functionality.
import asyncio
import os as _os
import sys as _sys
import threading
from importlib.metadata import PackageNotFoundError as _PkgNotFoundError
from importlib.metadata import version as _pkg_version


def _is_pytest_running() -> bool:
    """Check if pytest is running (for test-only SDK check bypass)."""
    return "pytest" in _sys.modules


# Only skip SDK check if BOTH conditions are met:
# 1. SKIP_SDK_CHECK env var is set
# 2. pytest is actually running (prevents production misuse)
_skip_sdk_check = _os.environ.get("SKIP_SDK_CHECK") and _is_pytest_running()

if not _skip_sdk_check:
    try:
        _pkg_version("github-copilot-sdk")
    except _PkgNotFoundError as _e:
        raise ImportError(
            "Required dependency 'github-copilot-sdk' is not installed. "
            "Install with:  pip install 'github-copilot-sdk>=0.2.0,<0.3.0'"
        ) from _e

# E402: These imports are intentionally after SDK check - we verify SDK
# installation before importing modules that depend on it (Three-Medium).
from collections.abc import Awaitable, Callable  # noqa: E402
from typing import Any  # noqa: E402

from amplifier_core import ModelInfo, ModuleCoordinator, ProviderInfo  # noqa: E402

from .provider import GitHubCopilotProvider  # noqa: E402

# Contract: sdk-boundary:Membrane:MUST:1 — import from sdk_adapter package, not submodules
from .sdk_adapter import CopilotClientWrapper  # noqa: E402

__version__ = "2.0.0"

# Amplifier module metadata
__amplifier_module_type__ = "provider"

# Type alias for cleanup function
CleanupFn = Callable[[], Awaitable[None]]

# ============================================================================
# Process-Level Singleton State
# ============================================================================
# The Copilot SDK subprocess consumes ~500MB (Electron-based). Without a
# process-level singleton, N sub-agents spawned by Amplifier's task tool
# each create their own CopilotClientWrapper → N × ~500MB memory.
#
# This singleton pattern ensures all providers share a single client.

_shared_client: CopilotClientWrapper | None = None
_shared_client_refcount: int = 0
_shared_client_lock: asyncio.Lock | None = None
_shared_client_lock_guard = threading.Lock()  # Guards lazy lock creation

# Lock timeout in seconds - hardcoded mechanism, not YAML policy
# 30s timeout prevents indefinite blocking from deadlock
_LOCK_TIMEOUT_SECONDS: float = 30.0


def _get_lock() -> asyncio.Lock:
    """Get or create the shared client lock (lazy initialization).

    asyncio.Lock() requires an event loop; import-time creation
    fails in test environments that don't have a loop yet.

    Uses threading.Lock to prevent TOCTOU race where two coroutines
    both see _shared_client_lock=None and create independent locks.

    Returns:
        The shared asyncio.Lock for singleton access.

    """
    global _shared_client_lock
    if _shared_client_lock is None:
        with _shared_client_lock_guard:
            # Double-check after acquiring threading lock
            if _shared_client_lock is None:
                _shared_client_lock = asyncio.Lock()
    return _shared_client_lock


async def _acquire_shared_client() -> CopilotClientWrapper:
    """Acquire a reference to the shared client, creating if needed.

    Implements process-level singleton with refcounting.

    Returns:
        The shared CopilotClientWrapper instance.

    Raises:
        TimeoutError: If lock cannot be acquired within 30 seconds.

    """
    global _shared_client, _shared_client_refcount

    lock = _get_lock()

    # 30s lock timeout prevents deadlock
    try:
        await asyncio.wait_for(lock.acquire(), timeout=_LOCK_TIMEOUT_SECONDS)
    except TimeoutError as e:
        raise TimeoutError(
            f"Failed to acquire shared client lock within {_LOCK_TIMEOUT_SECONDS}s"
        ) from e

    try:
        # Check if existing client is healthy
        if _shared_client is not None:
            if _shared_client.is_healthy():
                _shared_client_refcount += 1
                return _shared_client
            else:
                # Unhealthy client - close and replace
                import logging

                logger = logging.getLogger(__name__)
                logger.warning("[SINGLETON] Existing client unhealthy, replacing...")
                try:
                    await _shared_client.close()
                except Exception as close_err:
                    from .security_redaction import redact_sensitive_text

                    logger.warning(
                        "[SINGLETON] Error closing unhealthy client: %s",
                        redact_sensitive_text(close_err),
                    )
                _shared_client = None
                _shared_client_refcount = 0

        # Create new client - wrap in try/except to ensure clean state on failure
        try:
            new_client = CopilotClientWrapper()
            _shared_client = new_client
            _shared_client_refcount = 1
            return new_client
        except Exception:
            # Ensure clean state on failure
            _shared_client = None
            _shared_client_refcount = 0
            raise
    finally:
        lock.release()


async def _release_shared_client() -> None:
    """Release a reference to the shared client, closing when count reaches 0.

    Safe to call multiple times - refcount floors at 0.
    """
    global _shared_client, _shared_client_refcount

    lock = _get_lock()

    # Don't use timeout for release - always complete cleanup
    async with lock:
        if _shared_client_refcount > 0:
            _shared_client_refcount -= 1

            if _shared_client_refcount == 0 and _shared_client is not None:
                import logging

                logger = logging.getLogger(__name__)
                logger.info("[SINGLETON] Last reference released, closing shared client...")
                try:
                    await _shared_client.close()
                except Exception as close_err:
                    from .security_redaction import redact_sensitive_text

                    logger.warning(
                        "[SINGLETON] Error closing shared client: %s",
                        redact_sensitive_text(close_err),
                    )
                finally:
                    _shared_client = None


async def mount(
    coordinator: ModuleCoordinator,
    config: dict[str, Any] | None = None,
) -> CleanupFn | None:
    """Mount the GitHub Copilot provider.

    Contract: provider-protocol.md

    Uses a process-level singleton for CopilotClientWrapper to prevent
    O(N) memory consumption from N concurrent sub-agents.

    Args:
        coordinator: Amplifier kernel coordinator.
        config: Optional provider configuration.

    Returns:
        Cleanup callable, or None. Returns None on failure (graceful degradation).

    """
    import logging

    logger = logging.getLogger(__name__)

    shared_client: CopilotClientWrapper | None = None
    try:
        shared_client = await _acquire_shared_client()
        logger.info("[MOUNT] Acquired shared client (singleton)")
    except TimeoutError as e:
        from .security_redaction import redact_sensitive_text

        logger.error("[MOUNT] Failed to acquire shared client: %s", redact_sensitive_text(e))
        # P2 Fix: Raise instead of return None — framework must distinguish failure from opt-out.
        raise
    except Exception as e:
        from .security_redaction import redact_sensitive_text

        logger.error("[MOUNT] Error acquiring shared client: %s", redact_sensitive_text(e))
        # P2 Fix: Raise instead of return None — framework must distinguish failure from opt-out.
        raise

    try:
        logger.info("[MOUNT] Creating GitHubCopilotProvider...")
        provider = GitHubCopilotProvider(config, coordinator, client=shared_client)
        logger.info(f"[MOUNT] Provider created: {provider.name}")

        logger.info("[MOUNT] Mounting to coordinator...")
        await coordinator.mount("providers", provider, name="github-copilot")
        logger.info("[MOUNT] Provider mounted successfully")

        async def cleanup() -> None:
            # Release our reference to the shared client.
            # provider.close() is NOT called here because the shared
            # client lifecycle is managed by the singleton, not the provider.
            await _release_shared_client()

        return cleanup
    except Exception as e:
        # Release our reference if mount fails
        await _release_shared_client()

        # Graceful degradation: log error and return None instead of crashing
        # This matches the production provider pattern
        # Contract: security — Only log exception type/message at ERROR, not full traceback
        # (traceback may contain sensitive request data)
        from .security_redaction import redact_sensitive_text

        logger.error(
            "[MOUNT] Failed to mount GitHubCopilotProvider: %s: %s",
            type(e).__name__,
            redact_sensitive_text(e),
        )
        # Full traceback at DEBUG level only (security: avoid leaking sensitive data)
        logger.debug("[MOUNT] Mount failure traceback", exc_info=True)
        # P2 Fix: Raise instead of return None — framework must distinguish failure from opt-out.
        raise


__all__ = ["mount", "GitHubCopilotProvider"]


# =============================================================================
# Deprecation Shims for v2.0.0 Migration
# =============================================================================
# Provides helpful ImportError messages when users import symbols removed in v2.0.0.
# Contract: Follows Amplifier ecosystem's "additive evolution" philosophy.
# See MIGRATION.md for complete migration guide.


def __getattr__(name: str) -> None:
    """Raise ImportError with helpful migration message for removed v1.x symbols.

    This enables users upgrading from v1.x to get clear guidance on replacements
    instead of a generic "cannot import name" error.

    Example:
        >>> from amplifier_module_provider_github_copilot import CopilotSdkProvider
        ImportError: CopilotSdkProvider was removed in v2.0.0. Use GitHubCopilotProvider instead.
    """
    from ._deprecated import REMOVED_SYMBOLS

    if name in REMOVED_SYMBOLS:
        raise ImportError(REMOVED_SYMBOLS[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
