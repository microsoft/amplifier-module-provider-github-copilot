"""Tool parsing module.

Extracts tool calls from SDK response and returns kernel ToolCall types.

Contract: provider-protocol.md (parse_tool_calls method)

Logs a WARNING for empty tool arguments (LLM may have hallucinated).
"""

import json
import logging
import uuid
from typing import Any, Protocol, cast

from amplifier_core import ToolCall
from amplifier_core.llm_errors import InvalidToolCallError

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
        InvalidToolCallError: If tool call has invalid JSON arguments or non-dict arguments

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
                raise InvalidToolCallError(f"Invalid JSON in tool call arguments: {e}") from e
            # json.loads succeeds but may return list, int, etc. — must be object (dict)
            if not isinstance(args, dict):
                raise InvalidToolCallError(
                    "Tool call arguments JSON must be an object (dict),"
                    f" got {type(args).__name__!r}"
                )
        elif not isinstance(args, dict):
            raise InvalidToolCallError(
                f"Tool call arguments must be dict, got {type(args).__name__!r}"
            )

        tool_id = getattr(tc, "id", "") or str(uuid.uuid4())
        tool_name = getattr(tc, "name", "")

        # Warn on empty arguments (LLM may have hallucinated)
        # Note: Only warn for explicit empty dict {}, not for None
        if args == {} and not args_was_none:
            logger.warning(
                "[TOOL_PARSING] Empty arguments for tool '%s' (id=%s) - LLM may have hallucinated",
                tool_name,
                tool_id,
            )

        # Pattern: isinstance guards above guarantee dict; cast narrows Any to dict[str, Any].
        # json.loads returns Any — following established codebase pattern (event_helpers.py:L30).
        args_dict = cast(dict[str, Any], args)

        result.append(
            ToolCall(
                id=tool_id,
                name=tool_name,
                arguments=args_dict,
            )
        )

    return result
