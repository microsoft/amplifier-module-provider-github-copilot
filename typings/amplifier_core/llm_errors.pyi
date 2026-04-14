"""Type stubs for amplifier_core.llm_errors module."""

from typing import Any


class LLMError(Exception):
    """Base class for all LLM errors."""
    message: str
    provider: str | None
    retryable: bool
    delay_multiplier: float

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        retryable: bool = False,
        **kwargs: Any,
    ) -> None: ...


class AuthenticationError(LLMError):
    """Authentication failed."""
    ...


class RateLimitError(LLMError):
    """Rate limit exceeded."""
    retry_after: float | None
    
    def __init__(
        self,
        message: str,
        *,
        retry_after: float | None = None,
        **kwargs: Any,
    ) -> None: ...


class LLMTimeoutError(LLMError):
    """Request timed out."""
    timeout: float | None
    
    def __init__(
        self,
        message: str,
        *,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> None: ...


class NetworkError(LLMError):
    """Network connectivity error."""
    ...


class ProviderUnavailableError(LLMError):
    """Provider service unavailable."""
    ...


class ConfigurationError(LLMError):
    """Configuration error."""
    ...


class ContentFilterError(LLMError):
    """Content filtered by safety systems."""
    ...


class ContextLengthError(LLMError):
    """Context length exceeded."""
    ...


class InvalidRequestError(LLMError):
    """Invalid request parameters."""
    ...


class InvalidToolCallError(LLMError):
    """Invalid tool call."""
    ...


class QuotaExceededError(LLMError):
    """Quota exceeded."""
    ...


class AccessDeniedError(LLMError):
    """Access denied."""
    ...


class NotFoundError(LLMError):
    """Resource not found."""
    ...


class StreamError(LLMError):
    """Streaming error."""
    ...


class AbortError(LLMError):
    """Operation aborted."""
    ...


__all__ = [
    "LLMError",
    "AuthenticationError",
    "RateLimitError",
    "LLMTimeoutError",
    "NetworkError",
    "ProviderUnavailableError",
    "ConfigurationError",
    "ContentFilterError",
    "ContextLengthError",
    "InvalidRequestError",
    "InvalidToolCallError",
    "QuotaExceededError",
    "AccessDeniedError",
    "NotFoundError",
    "StreamError",
    "AbortError",
]
