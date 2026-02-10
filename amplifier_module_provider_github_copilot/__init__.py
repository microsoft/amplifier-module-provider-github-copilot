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
      copilot-sdk:
        model: claude-opus-4.5
        # timeout: 3600        # 1 hour default (override if needed)
        # thinking_timeout: 3600  # Same as regular (override if needed)
        debug: false
    ```

    Or via CLI:
    ```bash
    amplifier chat --provider copilot-sdk
    ```

Prerequisites:
    - Copilot CLI installed and in PATH
    - Authenticated to GitHub Copilot (run 'copilot auth login')
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Awaitable, Callable
from typing import Any

from amplifier_core import ChatResponse, ToolCall

from .client import AuthStatus, SessionInfo, SessionListResult
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

    try:
        # Create provider (api_key is None for Copilot - uses GitHub auth)
        provider = CopilotSdkProvider(None, config, coordinator)

        # Register with coordinator
        await coordinator.mount("providers", provider, name="github-copilot")

        logger.info("[MOUNT] CopilotSdkProvider mounted successfully")

        # Return cleanup function
        async def cleanup() -> None:
            """Cleanup function called when unmounting."""
            logger.info("[MOUNT] Unmounting CopilotSdkProvider...")
            await provider.close()
            logger.info("[MOUNT] CopilotSdkProvider unmounted")

        return cleanup

    except Exception as e:
        logger.error(f"[MOUNT] Failed to mount CopilotSdkProvider: {e}")
        return None


def _find_copilot_cli(config: dict[str, Any]) -> str | None:
    """
    Find the Copilot CLI executable path.

    Searches in order:
    1. config["cli_path"] (explicit configuration)
    2. COPILOT_CLI_PATH environment variable
    3. System PATH via shutil.which()

    Args:
        config: Provider configuration (read-only)

    Returns:
        Resolved CLI path, or None if not found
    """
    import os

    try:
        # Get CLI path from config or environment
        cli_path = config.get("cli_path")

        if not cli_path:
            cli_path = os.environ.get("COPILOT_CLI_PATH")

        # If still not found, try to locate it
        if not cli_path:
            found = shutil.which("copilot") or shutil.which("copilot.exe")
            if found:
                cli_path = found

        if not cli_path:
            logger.debug("[MOUNT] Copilot CLI not found in PATH or known locations")
            return None

        # Verify the path exists
        if os.path.isabs(cli_path):
            if not os.path.isfile(cli_path):
                logger.debug(f"[MOUNT] Copilot CLI not found at: {cli_path}")
                return None
            logger.debug(f"[MOUNT] Found Copilot CLI at absolute path: {cli_path}")
            return cli_path

        found_path = shutil.which(cli_path)
        if found_path is None:
            logger.debug(f"[MOUNT] '{cli_path}' CLI not found in PATH")
            return None
        logger.debug(f"[MOUNT] Found Copilot CLI at: {found_path}")
        return found_path

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
