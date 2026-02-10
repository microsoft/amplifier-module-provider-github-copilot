"""
Custom exceptions for Copilot SDK provider.

This module defines a hierarchy of domain-specific exceptions that provide
clear error context and enable appropriate error handling strategies.

Exception Hierarchy:
    CopilotProviderError (base)
    ├── CopilotAuthenticationError - Authentication/authorization failures
    ├── CopilotConnectionError - Network/connection issues
    ├── CopilotRateLimitError - Rate limiting (with retry_after)
    ├── CopilotModelNotFoundError - Invalid model requested
    ├── CopilotSessionError - Session lifecycle issues
    └── CopilotTimeoutError - Request timeouts

SFI Compliance:
    - All exceptions provide context without exposing sensitive data
    - Errors are designed for safe logging
"""

from __future__ import annotations


class CopilotProviderError(Exception):
    """
    Base exception for all Copilot SDK provider errors.

    All provider-specific exceptions inherit from this class,
    enabling catch-all handling when needed.

    Example:
        try:
            response = await provider.complete(request)
        except CopilotProviderError as e:
            logger.error(f"Provider error: {e}")
            # Handle gracefully
    """

    pass


class CopilotAuthenticationError(CopilotProviderError):
    """
    Raised when Copilot authentication fails.

    This indicates the user needs to authenticate with GitHub Copilot.

    Example:
        raise CopilotAuthenticationError(
            "Copilot authentication required. Run 'copilot auth login'"
        )
    """

    pass


class CopilotConnectionError(CopilotProviderError):
    """
    Raised when connection to Copilot CLI server fails.

    This typically indicates:
    - Copilot CLI is not running
    - Network issues
    - CLI crashed or became unresponsive

    Example:
        raise CopilotConnectionError(
            "Failed to connect to Copilot CLI server"
        )
    """

    pass


class CopilotRateLimitError(CopilotProviderError):
    """
    Raised when Copilot rate limits are exceeded.

    Includes optional retry_after hint for backoff strategies.

    Attributes:
        retry_after: Suggested wait time in seconds before retry.
                    None if not provided by the API.

    Example:
        raise CopilotRateLimitError(retry_after=30.0)
    """

    def __init__(self, retry_after: float | None = None, message: str | None = None):
        self.retry_after = retry_after
        if message:
            super().__init__(message)
        elif retry_after is not None:
            super().__init__(f"Rate limited. Retry after: {retry_after}s")
        else:
            super().__init__("Rate limited. Please retry later.")


class CopilotModelNotFoundError(CopilotProviderError):
    """
    Raised when requested model is not available.

    Includes the requested model and optionally the list of available models
    to help users correct their configuration.

    Attributes:
        model: The model ID that was requested but not found.
        available: List of available model IDs, if known.

    Example:
        raise CopilotModelNotFoundError(
            model="unknown-model",
            available=["claude-opus-4.5", "claude-sonnet-4"]
        )
    """

    def __init__(self, model: str, available: list[str] | None = None):
        self.model = model
        self.available = available or []
        if self.available:
            super().__init__(
                f"Model '{model}' not found. Available models: {', '.join(self.available)}"
            )
        else:
            super().__init__(f"Model '{model}' not found.")


class CopilotSessionError(CopilotProviderError):
    """
    Raised when session creation or management fails.

    This covers errors in the session lifecycle:
    - Session creation failures
    - Session destruction failures
    - Invalid session state

    Example:
        raise CopilotSessionError("Failed to create session: invalid config")
    """

    pass


class CopilotTimeoutError(CopilotProviderError):
    """
    Raised when Copilot request times out.

    Indicates the request took longer than the configured timeout.
    Users may want to:
    - Increase timeout configuration
    - Simplify the request
    - Retry with exponential backoff

    Attributes:
        timeout: The timeout value that was exceeded, in seconds.

    Example:
        raise CopilotTimeoutError(timeout=300.0)
    """

    def __init__(self, timeout: float | None = None, message: str | None = None):
        self.timeout = timeout
        if message:
            super().__init__(message)
        elif timeout is not None:
            super().__init__(f"Request timed out after {timeout}s")
        else:
            super().__init__("Request timed out")


class CopilotSdkLoopError(CopilotProviderError):
    """
    SDK internal loop exceeded limits.

    Raised when circuit breaker trips due to:
    - Too many turns (denial retry loop)
    - Timeout during loop
    - Anomalous SDK behavior

    Evidence: Session a1a0af17 documented 305 turns
    from a single request due to SDK denial retry behavior.

    Attributes:
        turn_count: Number of turns before error
        max_turns: Configured maximum
        tool_calls_captured: Tools captured before error

    Example:
        raise CopilotSdkLoopError(
            message="Circuit breaker tripped",
            turn_count=4,
            max_turns=3,
            tool_calls_captured=2,
        )
    """

    def __init__(
        self,
        message: str,
        turn_count: int,
        max_turns: int,
        tool_calls_captured: int = 0,
    ):
        super().__init__(message)
        self.turn_count = turn_count
        self.max_turns = max_turns
        self.tool_calls_captured = tool_calls_captured


class CopilotAbortError(CopilotProviderError):
    """
    Session abort failed or was interrupted.

    Raised when session.abort() fails to cleanly exit the SDK loop.

    Example:
        raise CopilotAbortError("abort() timed out after 5s")
    """

    pass
