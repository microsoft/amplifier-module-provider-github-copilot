"""Type stubs for amplifier-core package.

These stubs satisfy pyright strict mode.
The actual types come from the amplifier-core package at runtime.
"""

from typing import Any, Protocol
from dataclasses import dataclass

# Re-export submodule stubs
from .llm_errors import (
    AbortError as AbortError,
    AccessDeniedError as AccessDeniedError,
    AuthenticationError as AuthenticationError,
    ConfigurationError as ConfigurationError,
    ContentFilterError as ContentFilterError,
    ContextLengthError as ContextLengthError,
    InvalidRequestError as InvalidRequestError,
    InvalidToolCallError as InvalidToolCallError,
    LLMError as LLMError,
    LLMTimeoutError as LLMTimeoutError,
    NetworkError as NetworkError,
    NotFoundError as NotFoundError,
    ProviderUnavailableError as ProviderUnavailableError,
    QuotaExceededError as QuotaExceededError,
    RateLimitError as RateLimitError,
    StreamError as StreamError,
)


@dataclass
class ModelInfo:
    """Information about an LLM model."""
    id: str
    name: str | None = None
    display_name: str | None = None
    context_window: int | None = None
    max_output_tokens: int | None = None
    supports_tools: bool = False
    supports_vision: bool = False
    supports_streaming: bool = True
    family: str | None = None
    vendor: str | None = None
    capabilities: list[str] | None = None
    defaults: dict[str, Any] | None = None


@dataclass
class ProviderInfo:
    """Information about an LLM provider."""
    id: str
    display_name: str
    credential_env_vars: list[str]
    capabilities: list[str]
    defaults: dict[str, Any]
    config_fields: list["ConfigField"]


@dataclass
class ConfigField:
    """Configuration field definition."""
    id: str
    display_name: str
    field_type: str
    prompt: str
    env_var: str | None = None
    required: bool = False
    description: str | None = None


@dataclass
class ChatRequest:
    """Request for chat completion."""
    model: str
    messages: list[Any]
    tools: list[Any] | None = None
    tool_choice: str | Any | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    routing_model: str | None = None
    attachments: list[Any] | None = None
    

@dataclass
class ChatResponse:
    """Response from chat completion."""
    content: list[Any]
    tool_calls: list["ToolCall"] | None = None
    usage: "Usage | None" = None
    finish_reason: str | None = None
    metadata: dict[str, Any] | None = None
    # Additional fields used by tests/streaming
    content_blocks: list[Any] | None = None
    text: str | None = None


@dataclass
class ToolCall:
    """Tool call in a response."""
    id: str
    name: str
    arguments: str | dict[str, Any]


class ToolCallBlock:
    """Tool call block for ChatResponse.content (ContentBlockUnion member, Pydantic BaseModel).

    NOTE: Field is 'input', NOT 'arguments'. ToolCall has 'arguments'.
    These are two separate types with different field names and different purposes.
    """
    type: str  # Literal["tool_call"] discriminator
    id: str
    name: str
    input: dict[str, Any]

    def __init__(self, *, type: str = "tool_call", id: str, name: str, input: dict[str, Any], **kwargs: Any) -> None: ...


class TextBlock:
    """Text block for Pydantic context persistence (Pydantic BaseModel)."""
    type: str
    text: str
    
    def __init__(self, *, text: str, type: str = "text", **kwargs: Any) -> None: ...


class ThinkingBlock:
    """Thinking block for Pydantic context persistence (Pydantic BaseModel)."""
    type: str
    thinking: str
    signature: str | None
    
    def __init__(self, *, thinking: str, type: str = "thinking", signature: str | None = None, **kwargs: Any) -> None: ...


@dataclass
class Usage:
    """Token usage statistics."""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    reasoning_tokens: int | None
    cache_read_tokens: int | None
    cache_write_tokens: int | None
    def __init__(
        self,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        reasoning_tokens: int | None = ...,
        cache_read_tokens: int | None = ...,
        cache_write_tokens: int | None = ...,
    ) -> None: ...


@dataclass
class TextContent:
    """Text content block for streaming UI."""
    text: str


@dataclass
class ThinkingContent:
    """Thinking/reasoning content block for streaming UI."""
    text: str
    signature: str | None = None


@dataclass
class ToolCallContent:
    """Tool call content block for streaming UI."""
    id: str
    name: str
    arguments: str | dict[str, Any]


class HooksProtocol(Protocol):
    """Protocol for hook emission."""
    async def emit(self, event: str, data: dict[str, Any]) -> None: ...


class ModuleCoordinator:
    """Coordinator for provider module lifecycle."""
    
    hooks: HooksProtocol
    
    async def mount(
        self,
        category: str,
        module: Any,
        *,
        name: str | None = None,
    ) -> None: ...
    async def unmount(self) -> None: ...


__all__ = [
    # Core types
    "ModelInfo",
    "ProviderInfo", 
    "ConfigField",
    "ChatRequest",
    "ChatResponse",
    "ToolCall",
    "ToolCallBlock",
    "TextBlock",
    "ThinkingBlock",
    "Usage",
    "TextContent",
    "ThinkingContent",
    "ToolCallContent",
    "ModuleCoordinator",
    "HooksProtocol",
    # Error types (re-exported from llm_errors)
    "AbortError",
    "AccessDeniedError",
    "AuthenticationError",
    "ConfigurationError",
    "ContentFilterError",
    "ContextLengthError",
    "InvalidRequestError",
    "InvalidToolCallError",
    "LLMError",
    "LLMTimeoutError",
    "NetworkError",
    "NotFoundError",
    "ProviderUnavailableError",
    "QuotaExceededError",
    "RateLimitError",
    "StreamError",
]
