"""SDK Client Wrapper with lifecycle management.

Wraps copilot.CopilotClient for Amplifier integration.

SDK imports are quarantined in _imports.py per sdk-boundary.md.
This file imports from _imports.py, not directly from the SDK.

Contract: contracts/sdk-boundary.md
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from ..error_translation import ErrorConfig, translate_sdk_error
from ..security_redaction import safe_log_message

if TYPE_CHECKING:
    from .types import SessionHandle

logger = logging.getLogger(__name__)

# Deny hook constant - aligned with _make_deny_hook_config minimal reason strategy
DENY_ALL: dict[str, Any] = {
    "permissionDecision": "deny",
    # Minimal reason - align with _make_deny_hook_config to avoid teaching model tools are blocked
    "permissionDecisionReason": "Processing",
    # Suppress output to prevent denial from reaching conversation context
    "suppressOutput": True,
}


def _make_deny_hook_config() -> dict[str, Any]:
    """Create hooks config dict for session creation.

    The correct SDK API passes hooks via session config,
    not via the deprecated hook registration method.

    The SDK expects:
        session_config = {
            "hooks": {
                "on_pre_tool_use": deny_hook_fn
            }
        }

    Returns:
        Dict with 'on_pre_tool_use' key for session config.

    Contract: deny-destroy:DenyHook:MUST:1

    """

    def deny_hook(input_data: dict[str, Any], context: Any) -> dict[str, Any]:
        """Deny ALL tools — sovereignty mode.

        Contract: deny-destroy:DenyHook:MUST:1

        CRITICAL: The permissionDecisionReason is shown to the model/user.
        We MUST NOT include explanatory text because the model learns from
        denial reasons and will stop trying tools.

        Solution: Minimal reason + suppressOutput to prevent the denial
        message from polluting the conversation context.
        """
        tool_name = input_data.get("toolName", "unknown")
        logger.debug("[CLIENT] preToolUse deny: %s", tool_name)
        return DENY_ALL

    return {"on_pre_tool_use": deny_hook}


def deny_permission_request(request: Any) -> Any:
    """Deny all permission requests at source.

    SDK requires on_permission_request handler.

    The SDK asks: "May I do X?"
    Amplifier's answer: "No. Return the request to Amplifier's orchestrator."

    Tool capture happens via streaming events (ASSISTANT_MESSAGE), not hooks.
    This is the FIRST line of defense. preToolUse deny hook is the second.

    Contract: contracts/deny-destroy.md
    """
    # Import from quarantine module
    from ._imports import PermissionRequestResult

    if PermissionRequestResult is not None:  # type: ignore[truthy-function]
        return PermissionRequestResult(  # type: ignore[return-value]
            kind="denied-by-rules",
            message="Amplifier orchestrator controls all operations",
        )
    # SDK < 0.1.28 doesn't have PermissionRequestResult
    # Return dict fallback
    return {
        "kind": "denied-by-rules",
        "message": "Amplifier orchestrator controls all operations",
    }


def _load_error_config_once() -> ErrorConfig:
    """Load error config with fallback path resolution.

    Delegates to load_error_config() which handles:
    - importlib.resources loading (installed wheel)
    - File path fallback (dev/test)
    - context_extraction parsing
    """
    from ..error_translation import load_error_config

    # Delegate to unified load_error_config() which handles both
    # importlib.resources and file path scenarios with proper context_extraction
    try:
        return load_error_config()  # None = use importlib.resources or fallback
    except Exception as e:
        from ..security_redaction import redact_sensitive_text

        logger.warning("Failed to load error config: %s", redact_sensitive_text(e))
        return ErrorConfig()


def _resolve_token() -> str | None:
    """Resolve auth token from environment (SDK priority order).

    Official SDK priority from docs/auth/index.md:
    1. COPILOT_AGENT_TOKEN - Copilot agent mode
    2. COPILOT_GITHUB_TOKEN - Official recommended
    3. GH_TOKEN - GitHub CLI compatible
    4. GITHUB_TOKEN - GitHub Actions compatible
    """
    for var in ("COPILOT_AGENT_TOKEN", "COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
        token = os.environ.get(var)
        if token:
            return token
    return None


class CopilotClientWrapper:
    """Wrapper around copilot.CopilotClient with lifecycle management.

    Supports two modes:
    - Injected: pass sdk_client directly (for testing, owned=False)
    - Auto-init: no sdk_client, wrapper creates one lazily (owned=True)

    Only the auto-initialized client is owned and stopped on close().
    """

    def __init__(self, *, sdk_client: Any = None) -> None:
        """Initialize wrapper with optional pre-existing SDK client."""
        self._sdk_client: Any = sdk_client
        self._owned_client: Any = None  # Only set when we created the client ourselves
        self._error_config: Any = None
        # Lock prevents race conditions on lazy init
        self._client_lock: asyncio.Lock = asyncio.Lock()
        self._disconnect_failures: int = 0  # Track disconnect failures for escalation
        self._stopped: bool = False  # Track whether client has been stopped

    def is_healthy(self) -> bool:
        """Check if the client is healthy and usable.

        Note: This performs a lightweight check without network probes.
        A True return indicates the client is not stopped and can attempt
        SDK operations. Actual network connectivity is verified during
        session creation, not here.

        Returns:
            True if the client is usable (not stopped).
            False if the client has been explicitly stopped.

        Contract: sdk-boundary:Config:MUST:1

        """
        # If we've been explicitly stopped, we're unhealthy
        if self._stopped:
            return False

        # If we have an owned client that exists, we're healthy
        # (we don't do network probes here - that happens at session creation)
        return True

    def _get_error_config(self) -> Any:
        if self._error_config is None:
            self._error_config = _load_error_config_once()
        return self._error_config

    def _get_client(self) -> Any | None:
        """Return the active SDK client (injected or owned)."""
        return self._sdk_client or self._owned_client

    async def _ensure_client_initialized(self, caller: str = "session") -> Any:
        """Ensure SDK client is initialized, creating lazily if needed.

        Extracts common lazy-init logic to avoid duplication between
        session() and list_models().

        Args:
            caller: Name of calling method for logging context.

        Returns:
            Initialized SDK client.

        Raises:
            ProviderUnavailableError: When SDK not installed.
            LLMError: When SDK initialization fails.
        """
        client = self._get_client()
        if client is not None:
            return client

        async with self._client_lock:
            # Double-check after acquiring lock
            client = self._get_client()
            if client is not None:
                return client

            try:
                from ._imports import CopilotClient, SubprocessConfig

                token = _resolve_token()

                # SDK v0.2.0: Use SubprocessConfig instead of options dict
                if SubprocessConfig is not None and token:
                    config = SubprocessConfig(github_token=token)
                    self._owned_client = CopilotClient(config)  # type: ignore[arg-type]
                elif token and SubprocessConfig is None:
                    # Security P1-6: Fail closed when explicit token cannot be applied.
                    # An explicitly-provided token MUST NEVER be silently ignored.
                    # This prevents unintended privilege escalation from falling back
                    # to ambient/default authentication with potentially broader permissions.
                    # OWASP A07: Identification and Authentication Failures
                    from ..error_translation import ConfigurationError

                    raise ConfigurationError(
                        "Explicit GitHub token provided via environment variable, but SDK's "
                        "SubprocessConfig is unavailable (SDK version mismatch). Cannot apply "
                        "token - failing closed to prevent unintended default authentication.",
                        provider="github-copilot",
                    )
                else:
                    # No token provided - use SDK default authentication
                    self._owned_client = CopilotClient()  # type: ignore[arg-type]

                logger.debug("[CLIENT] CopilotClient created for %s", caller)

                # Start client - clear on failure for retry
                try:
                    await self._owned_client.start()  # type: ignore[union-attr]
                except Exception:
                    self._owned_client = None
                    raise

                logger.info("[CLIENT] Copilot client initialized for %s", caller)
                return self._owned_client

            except ImportError as e:
                from ..error_translation import ProviderUnavailableError
                from ..security_redaction import redact_sensitive_text

                raise ProviderUnavailableError(
                    f"Copilot SDK not installed: {redact_sensitive_text(e)}",
                    provider="github-copilot",
                ) from e
            except Exception as e:
                error_config = self._get_error_config()
                raise translate_sdk_error(e, error_config) from e

    @asynccontextmanager
    async def session(
        self,
        model: str | None = None,
        *,
        system_message: str | None = None,
        tools: list[Any] | None = None,
    ) -> AsyncIterator[SessionHandle]:
        """Create an ephemeral session with proper cleanup.

        P2-11: Returns SessionHandle façade instead of raw SDK session.
        Contract: sdk-boundary:TypeTranslation:MUST:4 — Use opaque handles

        Sessions are always destroyed on exit (success or error).
        Streaming is always enabled (required for event-based tool capture).

        Args:
            model: Model ID to use
            system_message: Optional system message (mode: replace)
            tools: Optional list of tool definitions (dicts or ToolSpec objects).
                   Accepts dicts with 'name', 'description', 'parameters' keys
                   or objects with those attributes (e.g., amplifier_core.ToolSpec).
                   Contract: sdk-boundary:ToolForwarding:MUST:1

        Yields:
            Raw SDK session (opaque Any)

        Raises:
            Domain errors (from error_translation) on creation failure

        """
        # Use extracted helper for lazy initialization (prevents duplication)
        client = await self._ensure_client_initialized(caller="session")

        session_config: dict[str, Any] = {}
        # Contract: deny-destroy:ToolSuppression:MUST:1
        # ALLOWLIST APPROACH: Set available_tools to Amplifier tool names only.
        # This blocks ALL SDK built-ins (list_agents, bash, edit, etc.) proactively.
        # The model never sees them, so it can't call them.
        #
        # Also set overrides_built_in_tool=True on tools (in convert_tools_for_sdk)
        # to handle name conflicts if an Amplifier tool shares a name with a built-in.
        if model:
            session_config["model"] = model
        if system_message:
            # Use replace mode to ensure Amplifier bundle persona takes precedence.
            # The SDK's default "GitHub Copilot CLI" prompt interferes with bundle instructions.
            # Replace mode gives full control over agent identity.
            session_config["system_message"] = {"mode": "replace", "content": system_message}
        if tools:
            from .types import convert_tools_for_sdk

            sdk_tools = convert_tools_for_sdk(tools)
            session_config["tools"] = sdk_tools

            # ALLOWLIST: Only Amplifier tools visible to model
            # SDK built-ins (list_agents, bash, edit, etc.) are blocked because
            # they're not in the allowlist. This is Layer 1 of defense-in-depth.
            tool_names = [t.name for t in sdk_tools]
            session_config["available_tools"] = tool_names

            logger.debug("[CLIENT] Forwarding %d tool(s) to SDK session", len(tools))
            logger.debug("[CLIENT] available_tools ALLOWLIST set: %s", tool_names)
            logger.debug(
                "[CLIENT] SDK built-ins (list_agents, bash, edit, etc.) blocked by allowlist"
            )
        else:
            # Contract: deny-destroy:ToolSuppression:MUST:1 — MUST NOT omit available_tools
            # When no Amplifier tools are provided, set available_tools=[] to prevent
            # SDK built-ins (list_agents, bash, edit) from appearing to the model.
            # This is safe because there are no Amplifier tools to accidentally disable.
            session_config["available_tools"] = []
            logger.debug("[CLIENT] No tools provided, available_tools=[] blocks SDK built-ins")
        # Streaming MUST always be enabled for event-based tool capture
        session_config["streaming"] = True
        # SDK v0.2.0: on_permission_request passed to create_session (not client constructor)
        session_config["on_permission_request"] = deny_permission_request

        # Pass deny hook via session config 'hooks' key (correct SDK API).
        # The SDK does NOT have a register_pre_tool_use_hook() method.
        # Hooks are passed via session config at creation time.
        # Contract: deny-destroy:DenyHook:MUST:1
        session_config["hooks"] = _make_deny_hook_config()
        logger.debug("[CLIENT] Deny hook configured via session_config['hooks']")

        sdk_session = None
        try:
            logger.debug("[CLIENT] Creating session with model=%r", model)
            # SDK v0.2.0: create_session uses kwargs, unpack config dict
            sdk_session = await client.create_session(**session_config)  # type: ignore[union-attr]
            logger.debug("[CLIENT] Session created: %s", getattr(sdk_session, "session_id", "?"))
        except Exception as e:
            error_config = self._get_error_config()
            raise translate_sdk_error(e, error_config) from e

        try:
            # P2-11: Return SessionHandle façade instead of raw SDK session
            # Contract: sdk-boundary:TypeTranslation:MUST:4 — Use opaque handles
            from .types import SessionHandle

            session_id = str(getattr(sdk_session, "session_id", "unknown"))
            yield SessionHandle(sdk_session, session_id)
        finally:
            if sdk_session is not None:
                try:
                    await sdk_session.disconnect()  # type: ignore[union-attr]
                    logger.debug("[CLIENT] Session disconnected")
                    self._disconnect_failures = 0  # Reset on success
                except Exception as disconnect_err:
                    # Track disconnect failures and escalate after threshold
                    self._disconnect_failures += 1
                    msg = "[CLIENT] Error disconnecting session: %s"
                    logger.warning(*safe_log_message(msg, disconnect_err))
                    if self._disconnect_failures > 3:
                        logger.error(
                            "[CLIENT] Multiple disconnect failures (%d) — potential resource leak",
                            self._disconnect_failures,
                        )

    async def close(self) -> None:
        """Clean up owned client resources. Safe to call multiple times."""
        self._stopped = True  # Mark as stopped so is_healthy() returns False
        if self._owned_client is not None:
            try:
                logger.info("[CLIENT] Stopping owned Copilot client...")
                await self._owned_client.stop()
                logger.info("[CLIENT] Copilot client stopped")
            except Exception as e:
                logger.warning(*safe_log_message("[CLIENT] Error stopping client: %s", e))
            finally:
                self._owned_client = None

    async def list_models(self) -> list[Any]:
        """Fetch available models from SDK backend.

        Contract: sdk-boundary:Models:MUST:1
        - SDK CopilotClient.list_models() returns list[ModelInfo]

        Returns:
            List of SDK ModelInfo objects (translation to domain types
            happens in models.py).

        Raises:
            ProviderUnavailableError: When SDK call fails.
        """
        # Use extracted helper for lazy initialization (prevents duplication)
        client = await self._ensure_client_initialized(caller="list_models")

        try:
            models = await client.list_models()  # type: ignore[union-attr]
            logger.debug("[CLIENT] Fetched %d models from SDK", len(models))
            return list(models)
        except Exception as e:
            error_config = self._get_error_config()
            raise translate_sdk_error(e, error_config) from e
