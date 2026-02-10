"""
Message and response conversion utilities.

This module handles the critical task of converting between Amplifier's
ChatRequest/ChatResponse format and the Copilot SDK's message format.

CRITICAL DESIGN DECISION:
Amplifier passes FULL conversation history in ChatRequest.messages.
Copilot session expects a prompt string. We serialize the message array
into a format the LLM understands.

Amplifier's Context Manager has ALREADY:
1. Applied FIC compaction if needed
2. Managed conversation history
3. Handled agent context switching

We just need to serialize to the format Copilot SDK expects.
"""

from __future__ import annotations

import json
import logging
from typing import Any

# Import official Amplifier types - MUST use these, not custom dataclasses
from amplifier_core import (
    ChatResponse,
    TextBlock,
    ToolCall,
    ToolCallBlock,
    Usage,
)

logger = logging.getLogger(__name__)


def convert_messages_to_prompt(messages: list[dict[str, Any]]) -> str:
    """
    Convert Amplifier messages array to Copilot prompt format.

    Amplifier's Context Manager has already:
    1. Applied FIC compaction if needed
    2. Managed conversation history
    3. Handled agent context switching

    We serialize to a multi-turn conversation format that the LLM
    understands within the Copilot session context.

    Args:
        messages: List of message dicts with 'role' and 'content' keys

    Returns:
        Formatted prompt string for Copilot SDK

    Example:
        >>> messages = [
        ...     {"role": "user", "content": "Hello"},
        ...     {"role": "assistant", "content": "Hi there!"},
        ...     {"role": "user", "content": "How are you?"}
        ... ]
        >>> prompt = convert_messages_to_prompt(messages)
        >>> print(prompt)
        Human: Hello

        Assistant: Hi there!

        Human: How are you?
    """
    if not messages:
        return ""

    parts: list[str] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = _extract_content(msg)

        if role == "system":
            # System messages are handled separately via session config
            continue
        elif role == "developer":
            # Developer messages contain developer-provided context (file contents, etc.)
            # Wrap in XML tags per Amplifier convention (aligned with Anthropic/OpenAI providers)
            # See: amplifier-module-provider-anthropic line 689, openai line 1392
            if content:
                wrapped = f"<context_file>\n{content}\n</context_file>"
                parts.append(f"Human: {wrapped}")
        elif role == "user":
            parts.append(f"Human: {content}")
        elif role == "assistant":
            # Handle assistant messages, including tool calls
            assistant_text = content
            tool_calls = msg.get("tool_calls", [])

            if tool_calls:
                # Include tool call information
                tool_parts = []
                for tc in tool_calls:
                    tool_name = tc.get("name", tc.get("function", {}).get("name", "unknown"))
                    tool_args = tc.get("arguments", tc.get("function", {}).get("arguments", {}))
                    if isinstance(tool_args, str):
                        try:
                            tool_args = json.loads(tool_args)
                        except json.JSONDecodeError:
                            pass
                    tool_parts.append(f"[Tool Call: {tool_name}({json.dumps(tool_args)})]")
                if assistant_text:
                    parts.append(f"Assistant: {assistant_text}\n" + "\n".join(tool_parts))
                else:
                    parts.append("Assistant: " + "\n".join(tool_parts))
            elif assistant_text:
                parts.append(f"Assistant: {assistant_text}")
        elif role == "tool":
            # Tool results from Amplifier's tool execution
            tool_name = msg.get("tool_name", msg.get("name", "tool"))
            parts.append(f"Tool Result ({tool_name}): {content}")
        elif role == "function":
            # Legacy function role (deprecated by OpenAI in favor of 'tool')
            # Handle as alias for tool result for backward compatibility
            func_name = msg.get("name", "function")
            parts.append(f"Tool Result ({func_name}): {content}")
        else:
            # Unknown role, treat as user
            logger.warning(f"[CONVERTER] Unknown message role: {role}")
            parts.append(f"Human: {content}")

    return "\n\n".join(parts)


def _extract_content(msg: dict[str, Any]) -> str:
    """
    Extract content from a message, handling various formats.

    Messages can have content as:
    - Simple string
    - List of content blocks (OpenAI format)
    - Complex nested structure

    Args:
        msg: Message dict

    Returns:
        Extracted content as string
    """
    content = msg.get("content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        # Handle list of content blocks (OpenAI-style)
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "image_url":
                    text_parts.append("[Image]")
                else:
                    # Unknown block type
                    text_parts.append(str(block.get("text", block.get("content", ""))))
        return "\n".join(text_parts)

    # Fallback
    return str(content) if content else ""


def extract_system_message(messages: list[dict[str, Any]]) -> str | None:
    """
    Extract system message(s) for Copilot session config.

    The system message is handled specially in Copilot - it's passed
    to the session configuration rather than included in the prompt.

    If multiple system messages exist, they are joined with double
    newlines (consistent with Anthropic/OpenAI/Gemini providers).

    Args:
        messages: List of message dicts

    Returns:
        Combined system message content, or None if not present
    """
    system_parts = []
    for msg in messages:
        if msg.get("role") == "system":
            content = _extract_content(msg)
            if content:
                system_parts.append(content)

    if not system_parts:
        return None

    if len(system_parts) > 1:
        logger.debug(f"[CONVERTER] Joining {len(system_parts)} system messages into one")

    return "\n\n".join(system_parts)


def convert_copilot_response_to_chat_response(
    response: Any,
    model: str,
) -> ChatResponse:
    """
    Convert Copilot SDK response to Amplifier ChatResponse.

    The response from send_and_wait() is a SessionEvent with:
    - type: SessionEventType.ASSISTANT_MESSAGE
    - data: Data object with content, tool_requests, etc.

    ChatResponse includes:
    - content: List of content blocks (TextBlock, ToolCallBlock, etc.)
    - tool_calls: List of tool call requests
    - usage: Token usage statistics

    Args:
        response: SessionEvent from Copilot SDK send_and_wait()
        model: Model ID used for the completion

    Returns:
        ChatResponse in Amplifier format (using amplifier_core types)
    """
    content_blocks: list[Any] = []
    tool_calls: list[ToolCall] = []
    usage = Usage(input_tokens=0, output_tokens=0, total_tokens=0)
    finish_reason = None

    if response is None:
        logger.warning("[CONVERTER] Received None response from Copilot")
        return ChatResponse(
            content=[TextBlock(type="text", text="")],
            tool_calls=[],
            usage=usage,
            finish_reason="none",
        )

    # Extract data from SessionEvent
    data = getattr(response, "data", response)

    # Extract content
    content = None
    if hasattr(data, "content"):
        content = data.content
    elif isinstance(data, dict):
        content = data.get("content")

    if content:
        # Use Amplifier's TextBlock
        content_blocks.append(TextBlock(type="text", text=str(content)))

    # Extract tool requests
    tool_requests = None
    if hasattr(data, "tool_requests"):
        tool_requests = data.tool_requests
    elif isinstance(data, dict):
        tool_requests = data.get("tool_requests", data.get("toolRequests"))

    if tool_requests:
        for tr in tool_requests:
            tool_call = _convert_tool_request(tr)
            if tool_call:
                tool_calls.append(tool_call)
                # Also add to content as ToolCallBlock
                content_blocks.append(
                    ToolCallBlock(
                        type="tool_call",
                        id=tool_call.id,
                        name=tool_call.name,
                        input=tool_call.arguments,
                    )
                )

    # Extract usage info
    usage = _extract_usage(data)

    # Determine finish reason
    if tool_calls:
        finish_reason = "tool_use"
    else:
        finish_reason = "end_turn"

    return ChatResponse(
        content=content_blocks,
        tool_calls=tool_calls if tool_calls else None,
        usage=usage,
        finish_reason=finish_reason,
    )


def _convert_tool_request(tr: Any) -> ToolCall | None:
    """
    Convert a Copilot tool request to Amplifier ToolCall.

    Args:
        tr: Tool request from Copilot SDK (ToolRequest dataclass or dict)

    Returns:
        ToolCall instance (amplifier_core type), or None if conversion fails
    """
    try:
        # Extract fields based on type
        if hasattr(tr, "tool_call_id"):
            # Copilot SDK ToolRequest dataclass
            tool_id = tr.tool_call_id
            name = tr.name
            arguments = tr.arguments
        elif isinstance(tr, dict):
            tool_id = tr.get("toolCallId", tr.get("tool_call_id", ""))
            name = tr.get("name", "")
            arguments = tr.get("arguments", {})
        else:
            logger.warning(f"[CONVERTER] Unknown tool request format: {type(tr)}")
            return None

        # Parse arguments if string
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"raw": arguments}

        return ToolCall(
            id=str(tool_id),
            name=str(name),
            arguments=arguments if isinstance(arguments, dict) else {},
        )
    except Exception as e:
        logger.warning(f"[CONVERTER] Failed to convert tool request: {e}")
        return None


def _extract_usage(data: Any) -> Usage:
    """
    Extract usage information from Copilot response data.

    Args:
        data: Response data from Copilot SDK

    Returns:
        Usage (amplifier_core type) with extracted token counts
    """
    input_tokens = 0
    output_tokens = 0

    # Try to extract from various possible locations
    if hasattr(data, "input_tokens"):
        input_tokens = int(data.input_tokens or 0)
    elif isinstance(data, dict):
        input_tokens = int(data.get("input_tokens", data.get("inputTokens", 0)) or 0)

    if hasattr(data, "output_tokens"):
        output_tokens = int(data.output_tokens or 0)
    elif isinstance(data, dict):
        output_tokens = int(data.get("output_tokens", data.get("outputTokens", 0)) or 0)

    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )


def format_tool_result_message(
    tool_call_id: str,
    tool_name: str,
    result: str | dict[str, Any],
) -> dict[str, Any]:
    """
    Format a tool result for inclusion in messages.

    Args:
        tool_call_id: ID of the tool call this is responding to
        tool_name: Name of the tool
        result: Result content (string or dict)

    Returns:
        Message dict in tool result format
    """
    content = result if isinstance(result, str) else json.dumps(result)
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
        "content": content,
    }
