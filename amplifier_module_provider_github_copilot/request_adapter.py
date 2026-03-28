"""Request adapter for ChatRequest to CompletionRequest conversion.

Contract: provider-protocol:complete:MUST:1

This module handles the adaptation of Amplifier kernel ChatRequest
to internal CompletionRequest, preserving multi-turn context fidelity.

Separation of Concerns:
- Provider (provider.py) orchestrates the completion lifecycle
- This module handles request transformation (domain logic)
"""

from __future__ import annotations

import logging
from typing import Any, cast

# Contract: sdk-boundary:Membrane:MUST:1 — import from sdk_adapter package, not submodules
from .sdk_adapter import CompletionRequest, extract_attachments_from_chat_request

logger = logging.getLogger(__name__)

__all__ = [
    "convert_chat_request",
    "extract_prompt_from_chat_request",
    "extract_system_message",
    # Private exports for backward compatibility with tests
    "_extract_message_content",
    "_extract_content_block",
]


def convert_chat_request(
    request: Any,
    *,
    default_model: str | None = None,
) -> CompletionRequest:
    """Convert kernel ChatRequest to internal CompletionRequest.

    Contract: provider-protocol:complete:MUST:1
    Contract: provider-protocol:complete:MUST:7 — Extract images from last user message
    Contract: provider-protocol:complete:MUST:8 — Include attachments in request

    Args:
        request: Kernel ChatRequest (or CompletionRequest passthrough).
        default_model: Default model to use if not specified in request.

    Returns:
        CompletionRequest suitable for SDK execution.
    """
    # Passthrough if already CompletionRequest
    if isinstance(request, CompletionRequest):
        return request

    # DEBUG: Log incoming request structure
    messages: list[Any] = getattr(request, "messages", [])
    logger.debug(
        "[REQUEST_ADAPTER] Received ChatRequest with %d messages, roles=%s",
        len(messages),
        [getattr(m, "role", "?") for m in messages[:5]],  # First 5 roles
    )

    # Extract prompt preserving multi-turn context
    prompt = extract_prompt_from_chat_request(request)

    # Extract model and tools
    model = getattr(request, "model", None) or default_model
    tools = getattr(request, "tools", []) or []

    # Extract attachments from last user message (images only)
    # Contract: sdk-boundary:ImagePassthrough:MUST:1
    attachments = extract_attachments_from_chat_request(request)

    # Extract system message for SDK session config (mode: replace)
    # CRITICAL: Without system_message, SDK uses default persona instead of bundle
    system_message = extract_system_message(request)

    return CompletionRequest(
        prompt=prompt,
        model=model,
        tools=tools,
        attachments=attachments,
        system_message=system_message,
    )


def extract_prompt_from_chat_request(request: Any) -> str:
    """Extract prompt from kernel ChatRequest preserving context fidelity.

    Preserves role information, tool calls, tool results, and all
    content types in multi-turn conversations.

    Contract: provider-protocol:complete:MUST:1

    Args:
        request: Kernel ChatRequest with messages attribute.

    Returns:
        Formatted prompt string with role markers and all content types.
    """
    messages: list[Any] = getattr(request, "messages", [])
    if not messages:
        return ""

    formatted_parts: list[str] = []

    for msg in messages:
        role: str = getattr(msg, "role", "user")
        content: Any = getattr(msg, "content", "")

        # Format role marker
        role_marker = f"[{role.upper()}]"

        # Extract content based on type
        content_text = _extract_message_content(content)

        if content_text:
            formatted_parts.append(f"{role_marker}\n{content_text}")

    return "\n\n".join(formatted_parts)


def _extract_message_content(content: Any) -> str:
    """Extract text from message content of various types.

    Handles string, list of content blocks, and individual blocks.

    Args:
        content: Message content (string, list, or content block).

    Returns:
        Extracted text content.
    """
    if content is None:
        return ""

    # String content
    if isinstance(content, str):
        return content

    # List of content blocks
    if isinstance(content, list):
        parts: list[str] = []
        for block in cast(list[Any], content):
            block_text = _extract_content_block(block)
            if block_text:
                parts.append(block_text)
        return "\n".join(parts)

    # Single content block
    return _extract_content_block(content)


def _extract_content_block(block: Any) -> str:
    """Extract text from a single content block.

    Handles TextContent, ThinkingContent, ToolCallContent,
    ToolResultContent, and dict representations.

    Args:
        block: Content block (dataclass or dict).

    Returns:
        Extracted text representation of the block.
    """
    if block is None:
        return ""

    # Helper to safely get from dict with proper typing
    def _get(key: str, default: Any = None) -> Any:
        if isinstance(block, dict):
            return cast(dict[str, Any], block).get(key, default)
        return default

    # Get block type
    block_type: str | None = getattr(block, "type", None) or _get("type")

    # CRITICAL: Check block_type FIRST before hasattr fallbacks.
    # ThinkingContent has a "text" attribute, so hasattr(block, "text") is True.
    # Bug fix: Check explicit types before hasattr fallbacks to prevent
    # ThinkingContent from being misclassified as TextContent.

    # ThinkingContent - extract thinking attribute with marker
    # Must check BEFORE TextContent because ThinkingContent also has text attr
    if block_type == "thinking" or (block_type is None and hasattr(block, "thinking")):
        thinking: str | None = getattr(block, "thinking", None) or _get("thinking")
        if thinking:
            return f"[Thinking: {thinking}]"
        return ""

    # TextContent - extract text attribute
    if block_type == "text" or (block_type is None and hasattr(block, "text")):
        text: str | None = getattr(block, "text", None) or _get("text")
        return str(text) if text else ""

    # Skip ToolCallContent blocks entirely — they are handled via tool_calls field.
    # This prevents fake tool call detection from triggering on prior turns.
    if block_type == "tool_call" or hasattr(block, "tool_name"):
        return ""

    # ToolResultContent - format tool result
    if block_type == "tool_result" or hasattr(block, "output"):
        output: str | None = getattr(block, "output", None) or _get("output")
        if output:
            return f"[Tool Result: {output}]"
        return ""

    # Fallback - try common text attributes
    for attr in ("text", "content", "value"):
        val: Any = getattr(block, attr, None) or _get(attr)
        if val:
            return str(val)

    return ""


def extract_system_message(request: Any) -> str | None:
    """Extract system message(s) from ChatRequest for SDK session config.

    The SDK handles system messages specially — they're passed to the session
    config with mode="replace" rather than included in the prompt.

    This is CRITICAL for Amplifier agent behavior:
    - Without system_message, SDK uses default "GitHub Copilot CLI" persona
    - With system_message, the Amplifier bundle persona takes precedence
    - Missing system_message causes agent to not follow bundle instructions

    If multiple system messages exist, they are joined with double newlines
    (consistent with Anthropic/OpenAI/Gemini providers).

    Args:
        request: Kernel ChatRequest with messages attribute.

    Returns:
        Combined system message content, or None if not present.
    """
    messages: list[Any] = getattr(request, "messages", [])
    if not messages:
        return None

    system_parts: list[str] = []

    for msg in messages:
        role: str = getattr(msg, "role", "user")
        if role != "system":
            continue

        content: Any = getattr(msg, "content", "")
        content_text = _extract_message_content(content)
        if content_text:
            system_parts.append(content_text)

    if not system_parts:
        return None

    if len(system_parts) > 1:
        logger.debug("[REQUEST_ADAPTER] Joining %d system messages into one", len(system_parts))

    system_message = "\n\n".join(system_parts)
    logger.debug("[REQUEST_ADAPTER] Extracted system_message (length=%d)", len(system_message))

    return system_message
