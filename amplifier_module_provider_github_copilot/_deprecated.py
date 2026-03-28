"""Deprecation shims for v2.0.0 migration.

This module provides helpful ImportError messages when users try to import
symbols that were removed in v2.0.0.
"""

REMOVED_SYMBOLS: dict[str, str] = {
    # Exceptions
    "CopilotProviderError": (
        "CopilotProviderError was removed in v2.0.0. Use ProviderError instead."
    ),
    "CopilotAuthenticationError": (
        "CopilotAuthenticationError was removed in v2.0.0. Use AuthenticationError instead."
    ),
    "CopilotConnectionError": (
        "CopilotConnectionError was removed in v2.0.0. Use ConnectionError instead."
    ),
    "CopilotRateLimitError": (
        "CopilotRateLimitError was removed in v2.0.0. Use RateLimitError instead."
    ),
    "CopilotModelNotFoundError": (
        "CopilotModelNotFoundError was removed in v2.0.0. Use ModelNotFoundError instead."
    ),
    "CopilotSessionError": ("CopilotSessionError was removed in v2.0.0. Use SessionError instead."),
    "CopilotSdkLoopError": ("CopilotSdkLoopError was removed in v2.0.0. Use SdkLoopError instead."),
    "CopilotAbortError": ("CopilotAbortError was removed in v2.0.0. Use AbortError instead."),
    "CopilotTimeoutError": ("CopilotTimeoutError was removed in v2.0.0. Use TimeoutError instead."),
    # Provider class
    "CopilotSdkProvider": (
        "CopilotSdkProvider was removed in v2.0.0. Use GitHubCopilotProvider instead."
    ),
    # Internal classes
    "SdkEventHandler": (
        "SdkEventHandler was removed in v2.0.0. This was an internal implementation detail."
    ),
    "LoopController": (
        "LoopController was removed in v2.0.0. This was an internal implementation detail."
    ),
    "ToolCaptureStrategy": (
        "ToolCaptureStrategy was removed in v2.0.0. This was an internal implementation detail."
    ),
    "CircuitBreaker": (
        "CircuitBreaker was removed in v2.0.0. This was an internal implementation detail."
    ),
    "CapturedToolCall": (
        "CapturedToolCall was removed in v2.0.0. This was an internal implementation detail."
    ),
    "AuthStatus": ("AuthStatus was removed in v2.0.0. SDK v0.2.0 changed authentication patterns."),
    "SessionInfo": (
        "SessionInfo was removed in v2.0.0. SDK v0.2.0 changed authentication patterns."
    ),
    "SessionListResult": (
        "SessionListResult was removed in v2.0.0. SDK v0.2.0 changed authentication patterns."
    ),
    "ModelIdPattern": ("ModelIdPattern was removed in v2.0.0. Use model name strings directly."),
}


def __getattr__(name: str) -> None:
    """Raise ImportError with a helpful migration message for removed symbols."""
    if name in REMOVED_SYMBOLS:
        raise ImportError(REMOVED_SYMBOLS[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
