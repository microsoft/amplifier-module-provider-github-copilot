"""Type stubs for amplifier_core.message_models module."""

from typing import Any, Literal


class Message:
    """Message in a chat conversation (Pydantic model)."""
    role: str
    content: str | list[Any]
    
    def __init__(self, *, role: str, content: str | list[Any], **kwargs: Any) -> None: ...


class TextBlock:
    """Text block in a message (Pydantic model)."""
    type: Literal["text"]
    text: str
    visibility: Literal["internal", "developer", "user"] | None
    
    def __init__(self, *, text: str, type: str = "text", **kwargs: Any) -> None: ...


class ThinkingBlock:
    """Thinking/reasoning block in a message (Pydantic model)."""
    type: Literal["thinking"]
    thinking: str
    signature: str | None
    
    def __init__(self, *, thinking: str, type: str = "thinking", signature: str | None = None, **kwargs: Any) -> None: ...


class ToolCallBlock:
    """Tool call block in a message (ContentBlockUnion member)."""
    type: str  # Literal["tool_call"] discriminator
    id: str
    name: str
    input: dict[str, Any]  # NOTE: 'input', not 'arguments' (that's ToolCall)

    def __init__(self, *, type: str = "tool_call", id: str, name: str, input: dict[str, Any], **kwargs: Any) -> None: ...


class ToolSpec:
    """Tool specification for LLM."""
    name: str
    description: str
    parameters: dict[str, Any] | None
    
    def __init__(
        self,
        *,
        name: str,
        description: str,
        parameters: dict[str, Any] | None = None,
    ) -> None: ...


class ToolCall:
    """Tool call object (goes in ChatResponse.tool_calls, NOT in content).

    NOTE: Field is 'arguments', NOT 'input'. ToolCallBlock has 'input'.
    """
    id: str
    name: str
    arguments: dict[str, Any]

    def __init__(self, *, id: str, name: str, arguments: dict[str, Any], **kwargs: Any) -> None: ...


__all__ = [
    "Message",
    "TextBlock",
    "ThinkingBlock", 
    "ToolCallBlock",
    "ToolCall",
    "ToolSpec",
]
