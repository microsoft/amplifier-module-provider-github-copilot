"""
Amplifier Provider Module: GitHub Copilot CLI SDK

This module provides access to Claude Opus 4.5 and other models
through GitHub Copilot subscription via the Copilot CLI SDK.

Pattern: Stateless Provider (Pattern A)
- Each complete() creates ephemeral Copilot session
- Amplifier maintains all state (history, compaction, agents)
- Provider is a pure LLM bridge

This preserves Amplifier's full potential:
✅ FIC context compaction (handled by Context Manager)
✅ Session persistence (handled by Amplifier's hooks)
✅ Agent delegation (handled by Orchestrator)
✅ Tool execution (handled by Orchestrator)

Usage:
    The module is loaded via Amplifier's module system:

    ```yaml
    # In amplifier.yaml
    providers:
      github-copilot:
        model: claude-opus-4.5
        # timeout: 3600        # 1 hour default (override if needed)
        # thinking_timeout: 3600  # Same as regular (override if needed)
        debug: false
    ```

    Or via CLI:
    ```bash
    amplifier chat --provider github-copilot
    ```

Prerequisites:
    - Copilot CLI installed and in PATH
    - Authenticated to GitHub Copilot (run 'copilot auth login')
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from collections.abc import Awaitable, Callable
from typing import Any

from amplifier_core import ChatResponse, ToolCall

from ._constants import DEFAULT_TIMEOUT
from .client import AuthStatus, CopilotClientWrapper, SessionInfo, SessionListResult
from .exceptions import (
    CopilotAbortError,
    CopilotAuthenticationError,
    CopilotConnectionError,
    CopilotModelNotFoundError,
    CopilotProviderError,
    CopilotRateLimitError,
    CopilotSdkLoopError,
    CopilotSessionError,
    CopilotTimeoutError,
)
from .model_naming import (
    KNOWN_THINKING_PATTERNS,
    ModelIdPattern,
    has_version_period,
    is_thinking_model,
    parse_model_id,
    uses_dash_for_version,
    validate_model_id_format,
)
from .provider import CopilotSdkProvider, ProviderInfo
from .sdk_driver import (
    CapturedToolCall,
    CircuitBreaker,
    LoopController,
    SdkEventHandler,
    ToolCaptureStrategy,
)

# Module exports
__all__ = [
    # Main exports
    "mount",
    "CopilotSdkProvider",
    "ProviderInfo",
    # SDK Driver components (for advanced usage and testing)
    "SdkEventHandler",
    "LoopController",
    "ToolCaptureStrategy",
    "CircuitBreaker",
    "CapturedToolCall",
    # Response types (re-exported from amplifier_core)
    "ChatResponse",
    "ToolCall",
    # Client types
    "AuthStatus",
    "SessionInfo",
    "SessionListResult",
    # Model naming utilities (for validation and debugging)
    "KNOWN_THINKING_PATTERNS",
    "ModelIdPattern",
    "has_version_period",
    "is_thinking_model",
    "parse_model_id",
    "uses_dash_for_version",
    "validate_model_id_format",
    # Exceptions
    "CopilotProviderError",
    "CopilotAuthenticationError",
    "CopilotConnectionError",
    "CopilotRateLimitError",
    "CopilotModelNotFoundError",
    "CopilotSessionError",
    "CopilotSdkLoopError",
    "CopilotAbortError",
    "CopilotTimeoutError",
]

# Amplifier module metadata
__amplifier_module_type__ = "provider"

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Process-Level Singleton State
# ═══════════════════════════════════════════════════════════════════════════════
#
# Sub-agents spawned by the task tool run as async coroutines in the SAME
# Python process and asyncio event loop as the parent session (kernel-guaranteed).
# Each sub-agent gets its own fresh ModuleCoordinator — coordinators are not shared.
#
# This singleton ensures all mounts in a process share ONE CopilotClientWrapper
# (one copilot CLI subprocess) regardless of how many sub-agents are spawned.
# Without this, N sub-agents spawn N processes × ~500 MB each.
#
# Reference: docs/plans/2026-02-23-process-singleton-design.md

_shared_client: CopilotClientWrapper | None = None
_shared_client_refcount: int = 0
_shared_client_lock: asyncio.Lock | None = None


def _get_lock() -> asyncio.Lock:
    """Return the singleton lock, creating it lazily on first call.

    Lazy initialization avoids creating asyncio.Lock at import time,
    which can fail if no event loop exists yet (common in test environments).
    """
    global _shared_client_lock
    if _shared_client_lock is None:
        _shared_client_lock = asyncio.Lock()
    return _shared_client_lock


async def _acquire_shared_client(
    config: dict[str, Any],
    timeout: float,
) -> CopilotClientWrapper:
    """Acquire the shared CopilotClientWrapper, creating it if this is the first mount.

    Increments the reference count. Call _release_shared_client() in cleanup
    to decrement. The subprocess is shut down when the count reaches zero.

    If a second caller passes a different timeout than the first, a DEBUG warning
    is logged and the existing client is returned unchanged — the second caller's
    timeout is silently ignored. Sub-agents inherit bundle config from the parent,
    so all callers typically pass the same values.
    """
    global _shared_client, _shared_client_refcount
    async with _get_lock():
        if _shared_client is None:
            logger.info("[MOUNT] Creating shared Copilot subprocess (first mount in process)")
            _shared_client = CopilotClientWrapper(config=config, timeout=timeout)
        else:
            existing_timeout = getattr(_shared_client, "_timeout", timeout)
            if existing_timeout != timeout:
                logger.debug(
                    f"[MOUNT] Ignoring timeout={timeout} for shared client "
                    f"(already created with timeout={existing_timeout})"
                )
        _shared_client_refcount += 1
        logger.debug(f"[MOUNT] Shared client refcount: {_shared_client_refcount}")
        return _shared_client


async def _release_shared_client() -> None:
    """Release one reference to the shared client.

    When the count reaches zero, closes and destroys the shared subprocess.
    Safe to call if already at zero (safety floor prevents negative counts).

    NOTE: If the Python process is killed (SIGKILL/crash), the refcount
    never reaches zero and close() is never called. This is acceptable —
    the OS reclaims the copilot subprocess when the parent process exits.
    There is no mitigation needed for this case.
    """
    global _shared_client, _shared_client_refcount
    async with _get_lock():
        _shared_client_refcount -= 1
        logger.debug(f"[MOUNT] Shared client refcount after release: {_shared_client_refcount}")
        if _shared_client_refcount <= 0:
            if _shared_client is not None:
                logger.info(
                    "[MOUNT] Last mount cleaned up — shutting down shared Copilot subprocess"
                )
                await _shared_client.close()
                _shared_client = None
            _shared_client_refcount = 0  # safety floor: prevent negative counts


async def mount(
    coordinator: Any,  # ModuleCoordinator
    config: dict[str, Any] | None = None,
) -> Callable[[], Awaitable[None]] | None:
    """
    Mount the Copilot SDK provider.

    This is the entry point called by Amplifier's module loading system.
    It validates prerequisites, creates the provider, and registers it
    with the coordinator.

    Args:
        coordinator: Amplifier's ModuleCoordinator for registration
        config: Provider configuration dict with optional keys:
            - default_model: Default model ID (default: "claude-opus-4.5")
            - model: Alias for default_model (backward compatibility)
            - timeout: Request timeout in seconds (default: 3600 / 1 hour)
            - thinking_timeout: Timeout for thinking models (default: 3600 / 1 hour)
            - debug: Enable debug logging (default: False)
            - cli_path: Path to Copilot CLI executable

    Returns:
        Cleanup function to unmount the provider, or None if
        prerequisites are not met (graceful degradation)

    Example config:
        ```python
        {
            "model": "claude-opus-4.5",
            "timeout": 3600,
            "thinking_timeout": 3600,
            "debug": False,
        }
        ```

    Graceful Degradation:
        If the Copilot CLI is not installed or not authenticated,
        the provider returns None instead of raising an error.
        This allows Amplifier to continue with other providers.
    """
    config = config or {}
    logger.info("[MOUNT] Mounting CopilotSdkProvider...")

    # Check prerequisites: find CLI path
    cli_path = _find_copilot_cli(config)

    if not cli_path:
        logger.warning(
            "[MOUNT] Copilot SDK prerequisites not met - provider not mounted. "
            "Ensure 'copilot' CLI is installed and authenticated."
        )
        return None  # Graceful degradation

    # Set CLI path in config for provider to use
    config["cli_path"] = cli_path

    # Track whether this call acquired a shared client reference,
    # so the error path only releases what this call actually acquired.
    acquired_client: CopilotClientWrapper | None = None

    try:
        timeout = float(config.get("timeout", DEFAULT_TIMEOUT))

        # Acquire (or reuse) the process-level shared client.
        # All mounts in this Python process share one CopilotClientWrapper instance.
        acquired_client = await _acquire_shared_client(config, timeout)

        # Create provider, injecting the shared client
        provider = CopilotSdkProvider(None, config, coordinator, client=acquired_client)

        # Register with coordinator
        await coordinator.mount("providers", provider, name="github-copilot")

        logger.info("[MOUNT] CopilotSdkProvider mounted successfully")

        # Return cleanup function — releases the shared reference, not the provider
        async def cleanup() -> None:
            """Cleanup function called when unmounting."""
            logger.info("[MOUNT] Unmounting CopilotSdkProvider...")
            await _release_shared_client()
            logger.info("[MOUNT] CopilotSdkProvider unmounted")

        return cleanup

    except Exception as e:
        logger.error(f"[MOUNT] Failed to mount CopilotSdkProvider: {e}")
        # Only release if this call successfully acquired a reference before the failure
        if acquired_client is not None:
            await _release_shared_client()
        return None


def _ensure_executable(path: str) -> None:
    """Ensure a binary has execute permission.

    Some package managers (notably uv) don't preserve the execute bit on
    bundled binaries.  Detect and fix this so subprocess.Popen won't fail
    with PermissionError.
    """
    import os
    import stat

    if not os.access(path, os.X_OK):
        current = os.stat(path).st_mode
        os.chmod(path, current | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        logger.info(f"[MOUNT] Fixed missing execute permission on {path}")


def _find_copilot_cli(config: dict[str, Any]) -> str | None:
    """
    Find the Copilot CLI executable path.

    Resolution order:
    1. SDK bundled binary (import copilot -> copilot/bin/copilot)
    2. System PATH fallback (shutil.which)

    The SDK bundles its own CLI binary which is version-matched,
    avoiding potential version mismatches with separately installed CLIs.

    Args:
        config: Provider configuration (unused, kept for API compatibility)

    Returns:
        Resolved CLI path, or None if not found
    """
    try:
        try:
            from pathlib import Path

            import copilot as _copilot_mod  # type: ignore[import-untyped]

            _mod_file = _copilot_mod.__file__
            if _mod_file is None:
                raise ImportError("copilot module has no __file__")
            _cli_bin = Path(_mod_file).parent / "bin" / "copilot"
            if _cli_bin.exists():
                cli_path = str(_cli_bin)
                _ensure_executable(cli_path)
                logger.debug(f"[MOUNT] Found SDK bundled CLI at: {cli_path}")
                return cli_path
            else:
                logger.debug("[MOUNT] SDK bundled CLI binary not found on disk")
        except ImportError:
            logger.debug("[MOUNT] copilot SDK not installed, trying PATH")

        found = shutil.which("copilot") or shutil.which("copilot.exe")
        if found:
            _ensure_executable(found)
            logger.debug(f"[MOUNT] Found Copilot CLI in PATH at: {found}")
            return found

        logger.debug("[MOUNT] Copilot CLI not found via SDK or PATH")
        return None

    except Exception as e:
        logger.debug(f"[MOUNT] CLI path discovery failed: {e}")
        return None


# For direct module usage (testing)
def get_provider_class() -> type[CopilotSdkProvider]:
    """
    Get the provider class for direct instantiation.

    This is useful for testing or when not using Amplifier's
    module system.

    Returns:
        CopilotSdkProvider class
    """
    return CopilotSdkProvider
