"""GitHub Copilot Provider for Amplifier.

Three-Medium Architecture:
- Python for mechanism (~300 lines)
- YAML for policy (~200 lines)
- Markdown for contracts (~400 lines)

Contract: contracts/provider-protocol.md
"""

from __future__ import annotations

# Eager dependency check: ensure github-copilot-sdk is installed.
# All SDK imports in this module are lazy (inside function bodies) so the module
# would otherwise import successfully without the SDK. That tricks Amplifier's
# provider discovery into thinking the module is fully functional, which prevents
# the automatic dependency-installation fallback from ever running.
# Using importlib.metadata avoids importing the SDK itself at module load time.
# Contract: sdk-boundary.md MUST:5
#
# SDK check bypass: test-only convenience guard — NOT a security boundary.
# Requires both env var AND pytest in sys.modules to reduce accidental misuse.
import asyncio
import os as _os
import threading
from importlib.metadata import PackageNotFoundError as _PkgNotFoundError
from importlib.metadata import version as _pkg_version

# Single source of truth for pytest detection — defined in _platform.py.
# Both __init__.py and sdk_adapter/_imports.py import from there.
from ._platform import is_pytest_running  # noqa: E402 (before SDK check block)

# Only skip SDK check if BOTH conditions are met:
# 1. SKIP_SDK_CHECK env var is set
# 2. pytest is actually running (convenience guard, NOT a security boundary)
_SKIP_SDK_CHECK = _os.environ.get("SKIP_SDK_CHECK") and is_pytest_running()


def _check_sdk_version(version_str: str) -> None:
    """Raise ImportError if SDK version does not satisfy >=0.2.0.

    Extracted for testability — module-level code that runs under SKIP_SDK_CHECK
    cannot be reached by unit tests; this function can be imported and tested
    directly.

    Contract: sdk-boundary:Membrane:MUST:5
    """
    try:
        ver_parts = tuple(int(x) for x in version_str.split(".")[:2] if x.isdigit())
    except (ValueError, TypeError):  # pragma: no cover — malformed version string
        ver_parts = (0, 0)
    if ver_parts < (0, 2):
        raise ImportError(
            f"github-copilot-sdk=={version_str} is installed but >=0.2.0 is required. "
            "Upgrade with: pip install 'github-copilot-sdk>=0.2.0,<0.3.0' "
            "or reinstall the provider: amplifier provider install --force github-copilot"
        )


if not _SKIP_SDK_CHECK:  # pragma: no cover
    try:
        _sdk_version = _pkg_version("github-copilot-sdk")
    except _PkgNotFoundError as _e:
        # SDK required; tests only run with SDK installed
        raise ImportError(
            "Required dependency 'github-copilot-sdk' is not installed. "
            "Install with:  pip install 'github-copilot-sdk>=0.2.0,<0.3.0'"
        ) from _e
    # Contract: sdk-boundary:Membrane:MUST:5 — fail at import time on wrong version.
    # Presence-only check passes silently for SDK 0.1.x which lacks SubprocessConfig,
    # causing a cryptic ConfigurationError deep in the init flow instead.
    _check_sdk_version(_sdk_version)

# E402: These imports are intentionally after SDK check - we verify SDK
# installation before importing modules that depend on it (Three-Medium).
import logging  # noqa: E402
from collections.abc import Awaitable, Callable  # noqa: E402
from typing import Any, NoReturn  # noqa: E402

from amplifier_core import ModuleCoordinator  # noqa: E402

from .config_loader import load_sdk_protection_config  # noqa: E402
from .provider import GitHubCopilotProvider  # noqa: E402

# Contract: sdk-boundary:Membrane:MUST:1 — import from sdk_adapter package, not submodules
from .sdk_adapter import AUTH_ENV_VARS, CopilotClientWrapper  # noqa: E402

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
# threading.Lock (not asyncio.Lock) — safe across event loops
# asyncio.Lock is event-loop-scoped; awaiting it from a different loop raises
# RuntimeError. threading.Lock works correctly in all asyncio / multi-loop scenarios.
_state_lock = threading.Lock()
_prewarm_task: asyncio.Task[None] | None = None  # Track prewarm task for cleanup


async def _acquire_shared_client() -> CopilotClientWrapper:
    """Acquire a reference to the shared client, creating if needed.

    Implements process-level singleton with refcounting.

    Uses threading.Lock (not asyncio.Lock) so multiple event loops can share
    the same singleton without triggering cross-loop RuntimeError.
    CopilotClientWrapper() construction is synchronous, so all state mutations
    fit inside the threading.Lock section. Async cleanup of an old unhealthy
    client happens OUTSIDE the lock to avoid holding it during I/O.

    Returns:
        The shared CopilotClientWrapper instance.

    Raises:
        TimeoutError: If lock cannot be acquired within 30 seconds.

    """
    global _shared_client, _shared_client_refcount

    # Contract: sdk-protection:Singleton:MUST:8 — timeout sourced from YAML
    lock_timeout = load_sdk_protection_config().singleton.lock_timeout_seconds
    acquired = _state_lock.acquire(timeout=lock_timeout)
    if not acquired:
        raise TimeoutError(f"Failed to acquire shared client lock within {lock_timeout}s")

    result_client: CopilotClientWrapper | None = None
    old_client: CopilotClientWrapper | None = None

    try:
        if _shared_client is not None:
            if _shared_client.is_healthy():
                _shared_client_refcount += 1
                result_client = _shared_client
            else:
                # Unhealthy — stash for async close OUTSIDE lock
                import logging

                logging.getLogger(__name__).warning(
                    "[SINGLETON] Existing client unhealthy, replacing..."
                )
                old_client = _shared_client
                _shared_client = None
                _shared_client_refcount = 0

        if result_client is None:
            # Create new client — constructor is sync; safe inside threading.Lock
            try:
                new_client = CopilotClientWrapper()
                _shared_client = new_client
                _shared_client_refcount = 1
                result_client = new_client
            except Exception:
                _shared_client = None
                _shared_client_refcount = 0
                raise
    finally:
        _state_lock.release()

    # Close old unhealthy client outside lock (async operation)
    if old_client is not None:
        try:
            await old_client.close()
        except Exception as close_err:
            import logging

            from .security_redaction import redact_sensitive_text

            logging.getLogger(__name__).warning(
                "[SINGLETON] Error closing unhealthy client: %s",
                redact_sensitive_text(close_err),
            )

    assert result_client is not None  # guaranteed by the logic above
    return result_client


async def _release_shared_client() -> None:
    """Release a reference to the shared client, closing when count reaches 0.

    Safe to call multiple times - refcount floors at 0.
    """
    global _shared_client, _shared_client_refcount

    client_to_close: CopilotClientWrapper | None = None

    with _state_lock:
        if _shared_client_refcount > 0:
            _shared_client_refcount -= 1

            if _shared_client_refcount == 0 and _shared_client is not None:
                import logging

                logging.getLogger(__name__).info(
                    "[SINGLETON] Last reference released, closing shared client..."
                )
                client_to_close = _shared_client
                _shared_client = None

    # Close outside lock (async operation)
    if client_to_close is not None:
        try:
            await client_to_close.close()
        except Exception as close_err:
            import logging

            from .security_redaction import redact_sensitive_text

            logging.getLogger(__name__).warning(
                "[SINGLETON] Error closing shared client: %s",
                redact_sensitive_text(close_err),
            )


def _log_auth_source(logger: logging.Logger) -> None:
    """Emit a single INFO line identifying the active auth source at mount time.

    Resolution priority mirrors the SDK's token resolver. If no env var is
    present, the SDK defers to the logged-in user's cached OAuth (VS Code /
    gh CLI login), so this path is still valid — we just surface it.

    Never logs token values. Only names (env var) or a fixed fallback label.
    """
    for var in AUTH_ENV_VARS:
        if _os.environ.get(var):
            logger.info("[MOUNT] Auth source: %s (env var)", var)
            return
    logger.info(
        "[MOUNT] Auth source: no auth env var set; SDK will attempt cached OAuth"
        " / logged-in user auth (checked %s)",
        ", ".join(AUTH_ENV_VARS),
    )


async def mount(
    coordinator: ModuleCoordinator,
    config: dict[str, Any] | None = None,
) -> CleanupFn | None:
    """Mount the GitHub Copilot provider.

    Contract: provider-protocol.md

    Uses a process-level singleton for CopilotClientWrapper to prevent
    O(N) memory consumption from N concurrent sub-agents.

    Pre-warming (optional):
        When `sdk.prewarm_subprocess: true` in sdk_protection.yaml, the SDK
        subprocess is spawned at mount() time rather than first request time.
        This moves ~2s latency from user-visible first-request to invisible
        mount time. Fire-and-forget — mount() doesn't wait for completion.

    Args:
        coordinator: Amplifier kernel coordinator.
        config: Optional provider configuration.

    Returns:
        Cleanup callable on success.

    Raises:
        Exception: On mount failure (framework distinguishes failure from opt-out).

    """
    import logging

    logger = logging.getLogger(__name__)

    # Eager auth-source resolution: emit one INFO line naming the active auth
    # source at mount time. Absence of all env vars is still a valid path
    # because the SDK falls back to the logged-in user's cached OAuth
    # (copilot-sdk sets use_logged_in_user = not bool(github_token)),
    # so we surface the active source rather than bailing.
    # Guarded: a logging failure on any platform must never block mount.
    try:
        _log_auth_source(logger)
    except Exception:  # pragma: no cover  # diagnostic only — never propagate out of mount()
        pass

    shared_client: CopilotClientWrapper | None = None
    try:
        shared_client = await _acquire_shared_client()
        logger.info("[MOUNT] Acquired shared client (singleton)")
    except TimeoutError as e:
        from .security_redaction import redact_sensitive_text

        logger.error("[MOUNT] Failed to acquire shared client: %s", redact_sensitive_text(e))
        raise
    except Exception as e:
        from .security_redaction import redact_sensitive_text

        logger.error("[MOUNT] Error acquiring shared client: %s", redact_sensitive_text(e))
        raise

    # Pre-warming: spawn SDK subprocess early if enabled
    # Contract: sdk-protection:Subprocess:MUST:5 — Track prewarm task for cleanup
    global _prewarm_task
    try:
        sdk_config = load_sdk_protection_config()
        if sdk_config.sdk.prewarm_subprocess:
            logger.info("[MOUNT] Pre-warming SDK subprocess (fire-and-forget)...")

            async def _prewarm() -> None:
                try:
                    # Use public prewarm() API instead of internal method
                    await shared_client.prewarm()  # type: ignore[union-attr]
                    logger.info("[MOUNT] Pre-warming complete")
                except Exception as prewarm_err:  # pragma: no cover
                    from .security_redaction import redact_sensitive_text

                    # Pre-warm failure is not fatal — first request will retry
                    logger.warning(
                        "[MOUNT] Pre-warming failed (will retry on first request): %s",
                        redact_sensitive_text(prewarm_err),
                    )

            _prewarm_task = asyncio.create_task(_prewarm())
    except Exception as config_err:  # pragma: no cover
        from .security_redaction import redact_sensitive_text

        # Config load failure during prewarm check is not fatal
        logger.warning(
            "[MOUNT] Failed to check prewarm config: %s",
            redact_sensitive_text(config_err),
        )

    try:
        logger.info("[MOUNT] Creating GitHubCopilotProvider...")
        provider = GitHubCopilotProvider(config, coordinator, client=shared_client)
        logger.info(f"[MOUNT] Provider created: {provider.name}")

        logger.info("[MOUNT] Mounting to coordinator...")
        await coordinator.mount("providers", provider, name="github-copilot")
        logger.info("[MOUNT] Provider mounted successfully")

        async def cleanup() -> None:
            # Contract: sdk-protection:Subprocess:MUST:5 — Cancel prewarm task
            global _prewarm_task
            if _prewarm_task is not None and not _prewarm_task.done():  # pragma: no cover
                # Prewarm cancelled during shutdown — unlikely in tests
                _prewarm_task.cancel()
                try:
                    await _prewarm_task
                except asyncio.CancelledError:
                    pass  # Expected
                _prewarm_task = None

            # Contract: streaming-contract:ProgressiveStreaming:SHOULD:3 — cancel tasks
            await provider.cancel_emit_tasks()

            # Release our reference to the shared client.
            # provider.close() is NOT called here because the shared
            # client lifecycle is managed by the singleton, not the provider.
            await _release_shared_client()

        return cleanup
    except Exception as e:
        # Release our reference if mount fails
        await _release_shared_client()

        # Contract: provider-protocol:mount:MUST:2 — raise on failure so the
        # framework can distinguish "provider broke" (exception) from "provider
        # chose not to load" (return None).  Do NOT silently return None here.
        from .security_redaction import redact_sensitive_text

        logger.error(
            "[MOUNT] Failed to mount GitHubCopilotProvider: %s: %s",
            type(e).__name__,
            redact_sensitive_text(e),
        )
        # Full traceback at DEBUG level — formatted and redacted before emission.
        # Contract: behaviors:Security:MUST:2 — raw exc_info logging bypasses redaction;
        # format to string first so redact_sensitive_text() can scrub tokens.
        import logging as _logging
        import traceback as _tb

        if logger.isEnabledFor(_logging.DEBUG):
            formatted_tb = "".join(_tb.format_exception(type(e), e, e.__traceback__))
            logger.debug(
                "[MOUNT] Mount failure traceback:\n%s",
                redact_sensitive_text(formatted_tb),
            )
        raise


# Contract: provider-protocol:public_api:MUST:1
__all__ = ["mount", "GitHubCopilotProvider"]


# =============================================================================
# Deprecation Shims for v2.0.0 Migration
# =============================================================================
# Provides helpful ImportError messages when users import symbols removed in v2.0.0.
# Contract: Follows Amplifier ecosystem's "additive evolution" philosophy.
# See MIGRATION.md for complete migration guide.


def __getattr__(name: str) -> NoReturn:
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
