"""Type stubs for amplifier_core.message_models module."""

from typing import Any, Literal


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
    """Tool call block in a message."""
    type: str
    id: str
    name: str
    arguments: str | dict[str, Any]


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


__all__ = [
    "TextBlock",
    "ThinkingBlock", 
    "ToolCallBlock",
    "ToolSpec",
]
