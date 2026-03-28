"""Tool parsing module.

Extracts tool calls from SDK response and returns kernel ToolCall types.

Contract: provider-protocol.md (parse_tool_calls method)

Logs a WARNING for empty tool arguments (LLM may have hallucinated).
"""

import json
import logging
from typing import Any, Protocol

from amplifier_core import ToolCall

logger = logging.getLogger(__name__)

__all__ = [
    "HasToolCalls",
    "parse_tool_calls",
]


class HasToolCalls(Protocol):
    """Protocol for objects that may contain tool calls."""

    @property
    def tool_calls(self) -> list[Any] | None:
        """Return tool calls if present."""
        ...


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
        ValueError: If tool call has invalid JSON arguments

    """
    tool_calls = getattr(response, "tool_calls", None)

    if not tool_calls:
        return []

    result: list[ToolCall] = []
    for tc in tool_calls:
        # Get arguments - handle both dict and string
        args = getattr(tc, "arguments", {})
        args_was_none = args is None
        if args_was_none:
            args = {}  # Convert None to empty dict for kernel ToolCall compatibility
        elif isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in tool call arguments: {e}") from e

        tool_id = getattr(tc, "id", "")
        tool_name = getattr(tc, "name", "")

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
