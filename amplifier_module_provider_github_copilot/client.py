"""
Copilot SDK client wrapper with lifecycle management.

This module provides a wrapper around the Copilot SDK's CopilotClient
with proper error handling, lifecycle management, and logging.

The wrapper implements Pattern A: Stateless Provider where each
session is ephemeral and Amplifier maintains all conversation state.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ._constants import DEFAULT_TIMEOUT, SDK_TIMEOUT_BUFFER_SECONDS, VALID_REASONING_EFFORTS
from .exceptions import (
    CopilotAuthenticationError,
    CopilotConnectionError,
    CopilotModelNotFoundError,
    CopilotProviderError,
    CopilotSessionError,
    CopilotTimeoutError,
)


@dataclass(frozen=True, slots=True)
class AuthStatus:
    """
    Authentication status from Copilot SDK.

    Attributes:
        is_authenticated: True if authenticated, False if not, None if unknown (error)
        github_user: GitHub username (login) if authenticated
        auth_type: Authentication method (e.g., "oauth", "token")
        host: GitHub host URL
        status_message: Human-readable status message from SDK
        error: Error message if the status check failed, None otherwise
    """

    is_authenticated: bool | None
    github_user: str | None
    auth_type: str | None = None
    host: str | None = None
    status_message: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class SessionInfo:
    """
    Information about a Copilot session.

    Attributes:
        session_id: Unique session identifier
        summary: Optional session summary/title
        start_time: ISO 8601 timestamp when session was created
        modified_time: ISO 8601 timestamp when session was last modified
        is_remote: Whether the session is remote
    """

    session_id: str
    start_time: str
    modified_time: str
    is_remote: bool
    summary: str | None = None


@dataclass(frozen=True, slots=True)
class SessionListResult:
    """
    Result of listing sessions.

    Attributes:
        sessions: List of SessionInfo objects
        error: Error message if the list failed, None otherwise
    """

    sessions: tuple[SessionInfo, ...]
    error: str | None = None


if TYPE_CHECKING:
    from copilot import CopilotClient, CopilotSession
    from copilot.types import ModelInfo

logger = logging.getLogger(__name__)


class CopilotClientWrapper:
    """
    Wrapper around CopilotClient for Amplifier integration.

    This class provides:
    - Lazy initialization of the Copilot client
    - Error translation to domain-specific exceptions
    - Comprehensive logging and diagnostics
    - Graceful cleanup on shutdown
    - Thread-safe client initialization

    Pattern A Implementation:
    - Each session is ephemeral (created per complete() call)
    - Sessions are destroyed after use
    - Amplifier maintains all conversation state externally

    Attributes:
        timeout: Default timeout for requests in seconds

    Example:
        >>> wrapper = CopilotClientWrapper(config={}, timeout=300.0)
        >>> async with wrapper.create_session(model="claude-opus-4.5") as session:
        ...     response = await wrapper.send_and_wait(session, "Hello!")
        >>> await wrapper.close()
    """

    __slots__ = ("_config", "_timeout", "_client", "_lock", "_started")

    def __init__(
        self,
        config: dict[str, Any],
        timeout: float = DEFAULT_TIMEOUT,
    ):
        """
        Initialize the Copilot client wrapper.

        Args:
            config: Configuration dict with optional keys:
                - cli_path: Path to Copilot CLI executable
                - log_level: Logging level for CLI
                - auto_restart: Whether to auto-restart CLI on crash
            timeout: Default request timeout in seconds (must be > 0)

        Raises:
            ValueError: If timeout is not positive
        """
        if timeout <= 0:
            raise ValueError(f"timeout must be positive, got {timeout}")

        self._config = config
        self._timeout = timeout
        self._client: CopilotClient | None = None
        self._lock = asyncio.Lock()
        self._started = False

        logger.debug(f"[CLIENT] CopilotClientWrapper initialized, timeout={timeout}s")

    async def ensure_client(self) -> CopilotClient:
        """
        Lazily initialize and return the Copilot client.

        Uses double-checked locking to ensure thread-safe
        initialization while minimizing lock contention.

        Returns:
            Initialized CopilotClient instance

        Raises:
            CopilotConnectionError: If client initialization fails
            CopilotAuthenticationError: If authentication fails
        """
        if self._client is not None and self._started:
            logger.debug("[CLIENT] Returning existing client")
            return self._client

        async with self._lock:
            # Double-check after acquiring lock
            if self._client is not None and self._started:
                return self._client

            # Use local variable to ensure atomic assignment after full initialization
            client: CopilotClient | None = None
            try:
                # Import here to allow graceful degradation if SDK not installed
                from copilot import CopilotClient

                # Build client options from config
                client_options = self._build_client_options()
                logger.debug(f"[CLIENT] Client options: {client_options}")

                logger.info("[CLIENT] Initializing Copilot client...")
                client = CopilotClient(client_options)
                await client.start()

                # Only assign to instance after successful start
                self._client = client
                self._started = True

                # Verify authentication
                await self._verify_authentication()

                logger.info("[CLIENT] Copilot client initialized successfully")
                return self._client

            except asyncio.CancelledError:
                # Clean up partially initialized client on cancellation
                if client is not None:
                    try:
                        await client.stop()
                    except Exception:
                        pass
                raise
            except ImportError as e:
                raise CopilotConnectionError(
                    "Copilot SDK not installed. Install with: pip install copilot-sdk"
                ) from e
            except Exception as e:
                # Clean up on any error
                if client is not None and self._client is None:
                    try:
                        await client.stop()
                    except Exception:
                        pass
                # Log the full exception for debugging
                # logger.exception() logs at ERROR level and includes traceback automatically
                logger.exception(f"[CLIENT] Client initialization failed: {type(e).__name__}: {e}")
                error_msg = str(e).lower()
                # Heuristic detection - SDK doesn't expose typed auth exceptions
                if "auth" in error_msg or "token" in error_msg or "login" in error_msg:
                    raise CopilotAuthenticationError(
                        f"Copilot authentication failed: {e}. "
                        "Run 'copilot auth login' to authenticate."
                    ) from e
                raise CopilotConnectionError(
                    f"Failed to initialize Copilot client: {type(e).__name__}: {e}"
                ) from e

    def _build_client_options(self) -> dict[str, Any]:
        """Build CopilotClientOptions from configuration.

        Note: We intentionally do NOT auto-discover copilot CLI from PATH.
        The SDK bundles its own CLI binary which is version-matched to the SDK.
        Using a system-installed CLI can cause version mismatches and auth issues.
        Only override cli_path if explicitly configured.
        """
        options: dict[str, Any] = {}

        # Only use cli_path if explicitly configured (config or env var)
        # Otherwise let SDK use its bundled CLI binary
        cli_path = self._config.get("cli_path") or os.environ.get("COPILOT_CLI_PATH")

        if cli_path:
            options["cli_path"] = cli_path
            logger.debug(f"[CLIENT] Using explicit CLI path: {cli_path}")
        else:
            logger.debug("[CLIENT] Using SDK's bundled CLI")

        if self._config.get("log_level"):
            options["log_level"] = self._config["log_level"]

        if self._config.get("auto_restart") is not None:
            options["auto_restart"] = self._config["auto_restart"]

        if self._config.get("cwd"):
            options["cwd"] = self._config["cwd"]

        return options

    async def _verify_authentication(self) -> None:
        """
        Verify that the client is authenticated.

        Raises:
            CopilotAuthenticationError: If not authenticated
        """
        try:
            if self._client is None:
                return

            auth_status = await self._client.get_auth_status()
            if not auth_status.isAuthenticated:
                raise CopilotAuthenticationError(
                    "Not authenticated to GitHub Copilot. Run 'copilot auth login' to authenticate."
                )
            logger.debug(f"[CLIENT] Authenticated as: {auth_status.login}")
        except CopilotAuthenticationError:
            raise
        except Exception as e:
            logger.warning(f"[CLIENT] Could not verify authentication: {e}")
            # Don't fail - authentication might still work

    @asynccontextmanager
    async def create_session(
        self,
        model: str,
        system_message: str | None = None,
        streaming: bool = True,
        reasoning_effort: str | None = None,
        tools: list[Any] | None = None,
        excluded_tools: list[str] | None = None,
        hooks: dict[str, Any] | None = None,
    ) -> AsyncIterator[CopilotSession]:
        """
        Create an ephemeral Copilot session.

        This is the key to Pattern A: Stateless Provider.
        Each session is short-lived and destroyed after use.
        Amplifier maintains all conversation state externally.

        Args:
            model: Model ID to use (e.g., "claude-opus-4.5")
            system_message: Optional system message to configure the session
            streaming: Enable streaming mode for delta events (default: True)
            reasoning_effort: Reasoning effort level ("low", "medium", "high", "xhigh")
                              Only used if model supports reasoning.
            tools: Optional list of SDK Tool objects for structured tool calling.
                   When provided, the LLM receives tool definitions and can
                   return tool_requests in responses.
            excluded_tools: Optional list of built-in tool names to disable.
                   When user-defined tools share names with Copilot built-ins,
                   the built-in handler shadows the user handler (causing hangs).
                   Pass conflicting names here to disable the built-ins.
                   See COPILOT_BUILTIN_TOOL_NAMES in _constants.py.
            hooks: Optional dict of session hooks (e.g., preToolUse deny hook).
                   Keys: 'on_pre_tool_use', 'on_post_tool_use', etc.
                   Used by dumb-pipe pattern to prevent CLI tool execution.

        Yields:
            CopilotSession instance for making requests

        Raises:
            CopilotSessionError: If session creation or destruction fails
            CopilotConnectionError: If not connected to Copilot

        Example:
            >>> async with wrapper.create_session("claude-opus-4.5", streaming=True) as session:
            ...     response = await wrapper.send_and_wait(session, "Hello!")
        """
        # Input validation
        if not model or not model.strip():
            raise ValueError("model must be a non-empty string")
        model = model.strip()

        if reasoning_effort is not None:
            if reasoning_effort not in VALID_REASONING_EFFORTS:
                raise ValueError(
                    f"reasoning_effort must be one of {sorted(VALID_REASONING_EFFORTS)}, "
                    f"got '{reasoning_effort}'"
                )

        client = await self.ensure_client()
        session: CopilotSession | None = None

        # Build session configuration
        session_config: dict[str, Any] = {"model": model}

        if system_message:
            # Use append mode to add to default system message
            session_config["system_message"] = {
                "mode": "append",
                "content": system_message,
            }

        # Enable streaming for delta events
        session_config["streaming"] = streaming

        # Add reasoning effort if provided (for models that support extended thinking)
        if reasoning_effort:
            session_config["reasoning_effort"] = reasoning_effort
            logger.debug(f"[CLIENT] Reasoning effort set to: {reasoning_effort}")

        # Disable infinite sessions for ephemeral pattern
        session_config["infinite_sessions"] = {"enabled": False}

        # Add tools for structured tool calling (Option A: capture-and-abort)
        if tools:
            session_config["tools"] = tools
            logger.debug(f"[CLIENT] Registering {len(tools)} tool(s) with session")

            # NOTE: We previously used available_tools to whitelist our tool names,
            # but the SDK docs say: "excluded_tools is IGNORED if available_tools is set".
            # This caused "Tool names must be unique" errors because both user-defined
            # tools (e.g., "grep") and built-in tools with the same name were included.
            #
            # SOLUTION: Use ONLY excluded_tools, not available_tools. The model will
            # see both our user tools and any non-excluded built-ins, but that's OK
            # because we exclude built-ins that share names with our tools.

        # Exclude built-in tools that would shadow user-defined tools
        if excluded_tools:
            session_config["excluded_tools"] = excluded_tools
            logger.debug(
                f"[CLIENT] Excluding {len(excluded_tools)} built-in tool(s): {excluded_tools}"
            )

        # Add session hooks (e.g., preToolUse deny hook for dumb-pipe pattern)
        if hooks:
            session_config["hooks"] = hooks
            logger.debug(f"[CLIENT] Session hooks configured: {list(hooks.keys())}")

        # Session creation - separated from yield to avoid exception masking
        try:
            logger.debug(
                f"[CLIENT] Creating session with model: {model}, "
                f"streaming: {streaming}, reasoning: {reasoning_effort}, "
                f"tools: {len(tools) if tools else 0}"
            )
            session = await client.create_session(session_config)
            logger.debug(f"[CLIENT] Session created: {session.session_id}")
        except Exception as e:
            error_msg = str(e).lower()
            # Heuristic detection - SDK doesn't expose typed model exceptions
            if "model" in error_msg and ("not found" in error_msg or "invalid" in error_msg):
                raise CopilotModelNotFoundError(model=model) from e
            raise CopilotSessionError(f"Failed to create session: {e}") from e

        # Yield session and ensure cleanup - exceptions from caller pass through unchanged
        try:
            yield session
        finally:
            # Always destroy the session to clean up
            try:
                await session.destroy()
                logger.debug(f"[CLIENT] Session destroyed: {session.session_id}")
            except Exception as destroy_error:
                # Log but don't raise - don't mask the original exception
                logger.warning(
                    f"[CLIENT] Error destroying session {session.session_id}: {destroy_error}"
                )

    async def send_and_wait(
        self,
        session: CopilotSession,
        prompt: str,
        timeout: float | None = None,
    ) -> Any:
        """
        Send prompt and wait for response with timeout.

        This method sends a message to the Copilot session and blocks
        until the assistant finishes responding or timeout occurs.

        Args:
            session: Active CopilotSession instance
            prompt: Prompt text to send (must be non-empty)
            timeout: Request timeout in seconds (uses default if not specified)

        Returns:
            SessionEvent with assistant response, or None if no response

        Raises:
            ValueError: If prompt is empty
            CopilotTimeoutError: If request times out
            CopilotProviderError: If request fails for other reasons
        """
        # Input validation
        if not prompt:
            raise ValueError("prompt must be a non-empty string")

        effective_timeout = timeout if timeout is not None else self._timeout

        logger.debug(f"[CLIENT] Sending prompt ({len(prompt)} chars), timeout={effective_timeout}s")

        try:
            # Use single timeout control at our layer; SDK timeout set slightly higher
            # to let our timeout win and provide consistent error handling
            async with asyncio.timeout(effective_timeout):
                response = await session.send_and_wait(
                    {"prompt": prompt},
                    timeout=effective_timeout + SDK_TIMEOUT_BUFFER_SECONDS,
                )
                logger.debug("[CLIENT] Received response from Copilot")
                return response

        except TimeoutError as e:
            # Attempt to abort the in-flight request to clean up session state
            try:
                await asyncio.shield(session.abort())
                logger.debug("[CLIENT] Aborted request after timeout")
            except Exception as abort_error:
                logger.warning(f"[CLIENT] Failed to abort request after timeout: {abort_error}")
            raise CopilotTimeoutError(
                timeout=effective_timeout,
                message=f"Request timed out after {effective_timeout}s",
            ) from e
        except CopilotTimeoutError:
            raise
        except Exception as e:
            raise CopilotProviderError(f"Request failed: {e}") from e

    async def list_models(self) -> list[ModelInfo]:
        """
        List available models from Copilot SDK.

        Returns:
            List of ModelInfo objects describing available models

        Raises:
            CopilotConnectionError: If not connected
            CopilotProviderError: If list fails
        """
        client = await self.ensure_client()

        try:
            logger.debug("[CLIENT] Fetching available models...")
            models = await client.list_models()
            logger.debug(f"[CLIENT] Found {len(models)} models")
            return models
        except Exception as e:
            raise CopilotProviderError(f"Failed to list models: {e}") from e

    async def close(self) -> None:
        """
        Cleanup client resources.

        Stops the Copilot CLI server if we spawned it.
        Safe to call multiple times. Thread-safe via lock.
        """
        async with self._lock:
            if self._client is not None:
                try:
                    logger.info("[CLIENT] Stopping Copilot client...")
                    # Use shield to prevent cancellation during cleanup
                    await asyncio.shield(self._client.stop())
                    logger.info("[CLIENT] Copilot client stopped")
                except Exception as e:
                    logger.warning(f"[CLIENT] Error stopping client: {e}")
                finally:
                    self._client = None
                    self._started = False

    async def get_auth_status(self) -> AuthStatus:
        """
        Get authentication status from the Copilot SDK.

        Returns:
            AuthStatus with authentication details. If an error occurs,
            is_authenticated will be None (unknown) and error will be set.

        Example:
            >>> status = await wrapper.get_auth_status()
            >>> if status.error:
            ...     print(f"Check failed: {status.error}")
            >>> elif status.is_authenticated:
            ...     print(f"Logged in as {status.github_user}")
        """
        client = await self.ensure_client()

        try:
            logger.debug("[CLIENT] Getting auth status...")
            auth_status = await client.get_auth_status()
            result = AuthStatus(
                is_authenticated=auth_status.isAuthenticated,
                github_user=auth_status.login,
                auth_type=auth_status.authType,
                host=auth_status.host,
                status_message=auth_status.statusMessage,
                error=None,
            )
            logger.debug(
                f"[CLIENT] Auth status: authenticated={result.is_authenticated}, "
                f"user={result.github_user}"
            )
            return result
        except Exception as e:
            logger.warning(f"[CLIENT] Failed to get auth status: {e}")
            return AuthStatus(
                is_authenticated=None,  # Unknown, not False
                github_user=None,
                auth_type=None,
                host=None,
                status_message=None,
                error=str(e),
            )

    async def list_sessions(self) -> SessionListResult:
        """
        List all sessions from the Copilot SDK.

        Returns:
            SessionListResult with sessions list. If an error occurs,
            sessions will be empty and error will be set.

        Example:
            >>> result = await wrapper.list_sessions()
            >>> if result.error:
            ...     print(f"List failed: {result.error}")
            >>> else:
            ...     for session in result.sessions:
            ...         print(f"{session.session_id}: {session.summary}")
        """
        client = await self.ensure_client()

        try:
            logger.debug("[CLIENT] Listing sessions...")
            sessions = await client.list_sessions()
            session_infos = tuple(
                SessionInfo(
                    session_id=s.sessionId,
                    summary=s.summary,
                    start_time=s.startTime,
                    modified_time=s.modifiedTime,
                    is_remote=s.isRemote,
                )
                for s in sessions
            )
            logger.debug(f"[CLIENT] Found {len(session_infos)} sessions")
            return SessionListResult(sessions=session_infos, error=None)
        except Exception as e:
            logger.warning(f"[CLIENT] Failed to list sessions: {e}")
            return SessionListResult(sessions=(), error=str(e))

    @property
    def is_connected(self) -> bool:
        """
        Check if client is connected and started.

        Note:
            This is a point-in-time check without locking.
            The connection state may change immediately after this returns.
            Use for diagnostic/logging purposes, not for control flow.
        """
        return self._client is not None and self._started

    async def __aenter__(self) -> CopilotClientWrapper:
        """Async context manager entry."""
        await self.ensure_client()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()
