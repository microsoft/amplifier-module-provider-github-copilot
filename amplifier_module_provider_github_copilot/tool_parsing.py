"""Tool parsing module.

Extracts tool calls from SDK response and returns kernel ToolCall types.

Contract: provider-protocol.md (parse_tool_calls method)

Logs a WARNING for empty tool arguments (LLM may have hallucinated).
"""

import json
import logging
from typing import Any, Protocol, cast

from amplifier_core import ToolCall
from amplifier_core.llm_errors import InvalidToolCallError

logger = logging.getLogger(__name__)

__all__ = [
    "HasToolCalls",
    "parse_tool_arguments",
    "parse_tool_identity",
    "parse_tool_calls",
]


class HasToolCalls(Protocol):
    """Protocol for objects that may contain tool calls."""

    @property
    def tool_calls(self) -> list[Any] | None:
        """Return tool calls if present."""
        ...


def parse_tool_identity(tool_id: Any, tool_name: Any) -> tuple[str, str]:
    """Require the model's original correlation ID and exact tool name."""
    if not isinstance(tool_id, str) or not tool_id:
        raise InvalidToolCallError("Tool call ID must be a nonempty string")
    if not isinstance(tool_name, str) or not tool_name:
        raise InvalidToolCallError("Tool call name must be a nonempty string")
    return tool_id, tool_name


def parse_tool_arguments(arguments: Any) -> dict[str, Any]:
    """Preserve object arguments; reject malformed input before tool dispatch."""
    if arguments is None:
        return {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as e:
            raise InvalidToolCallError(f"Invalid JSON in tool call arguments: {e}") from e
        if not isinstance(arguments, dict):
            raise InvalidToolCallError(
                "Tool call arguments JSON must be an object (dict),"
                f" got {type(arguments).__name__!r}"
            )
    elif not isinstance(arguments, dict):
        raise InvalidToolCallError(
            f"Tool call arguments must be dict, got {type(arguments).__name__!r}"
        )
    return cast(dict[str, Any], arguments)


def parse_tool_calls(response: Any) -> list[ToolCall]:
    """Extract tool calls from response.

    Contract: provider-protocol.md

    - MUST return ToolCall with `arguments` field (not `input`)
    - MUST handle empty/missing tool_calls gracefully
    - MUST parse JSON string arguments if needed

    Args:
        response: ChatResponse or any object with tool_calls attribute

    Returns:
        List of ToolCall objects (may be empty)

    Raises:
        InvalidToolCallError: If the call identity or object arguments are malformed

    """
    tool_calls = getattr(response, "tool_calls", None)

    if not tool_calls:
        return []

    result: list[ToolCall] = []
    for tc in tool_calls:
        # Get arguments - handle both dict and string
        args = getattr(tc, "arguments", {})
        args_was_none = args is None
        args = parse_tool_arguments(args)

        tool_id, tool_name = parse_tool_identity(
            getattr(tc, "id", None), getattr(tc, "name", None)
        )

        # Warn on empty arguments (LLM may have hallucinated)
        # Note: Only warn for explicit empty dict {}, not for None
        if args == {} and not args_was_none:
            logger.warning(
                "[TOOL_PARSING] Empty arguments for tool '%s' (id=%s) - LLM may have hallucinated",
                tool_name,
                tool_id,
            )

        result.append(
            ToolCall(
                id=tool_id,
                name=tool_name,
                arguments=args,
            )
        )

    return result
