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
import re
from types import SimpleNamespace
from typing import Any, cast

# Single canonical import for ConfigurationError. Convention from config_loader.py
# / models.py / streaming.py: top-level only, never re-imported in function bodies.
from ._compat import ConfigurationError
from ._identity import PROVIDER_ID

# Contract: sdk-boundary:Membrane:MUST:1 — import from sdk_adapter package, not submodules
from .sdk_adapter import CompletionRequest, CopilotModelInfo, extract_attachments_from_chat_request
from .security_redaction import redact_sensitive_text

logger = logging.getLogger(__name__)

# Fallback allowlist used when the resolved CopilotModelInfo is unavailable
# (cache miss / unknown model). Without this, any short string would be
# forwarded to the SDK. Mirrors the SDK ReasoningEffort Literal as of v0.3.0.
_REASONING_EFFORT_FALLBACK_ALLOWLIST: frozenset[str] = frozenset({"low", "medium", "high", "xhigh"})

# Static allowlist for context_tier. Mirrors the public SDK annotation
# ``copilot.session.ContextTier = typing.Literal["default", "long_context"]``
# (verified against installed github-copilot-sdk 1.0.2). Unlike reasoning_effort
# there is NO per-model capability descriptor, so this membership set is the only
# validation possible. Evergreen note: if the SDK Literal grows a tier, update
# this frozenset — pinned by test_context_tier.TestSDKSourceShape.
_CONTEXT_TIER_ALLOWLIST: frozenset[str] = frozenset({"default", "long_context"})

# Echo-redaction guard for ConfigurationError messages. Rejected values are
# rendered verbatim only if they match this token shape; anything else is
# replaced with "<redacted; len=N>". The pattern excludes every known
# credential shape (uppercase, hyphens, dots, '/', '=', '+', '%' all fail),
# so machine secrets cannot leak through error reflection. Digits are allowed
# inside the token to admit future SDK literal variants like "high2" without
# false-redacting them; a leading digit is still rejected so PEM-style and
# numeric-prefix secrets are out. Human passphrases that happen to fit are
# out of scope: ChatRequest.reasoning_effort is typed Literal[...] at the
# kernel boundary; routing a secret through it is a caller bug the type
# system already rejects.
_REASONING_EFFORT_TOKEN_RE = re.compile(r"^[a-z][a-z0-9_]{0,15}$")

# Pattern matching synthetic role-marker format in user-controlled content.
# Matches [WORD] where WORD is 2+ uppercase letters/underscores.
# Contract: behaviors:Security:MUST:1 — OWASP A03: Injection prevention
_ROLE_INJECTION_PATTERN = re.compile(r"\[([A-Z][A-Z_]{1,})\]")

# Text inserted into synthetic tool-result messages when repair is needed.
# Verified safe: contains no uppercase bracket sequences ([WORD]) so
# _sanitize_content_for_injection is a clean no-op on this string.
_TOOL_SEQUENCE_REPAIR_MESSAGE = (
    "Tool result unavailable — the result for this tool call was lost. "
    "Please acknowledge this and continue."
)


def _sanitize_content_for_injection(text: str) -> str:
    """Escape role-marker sequences in user-controlled text.

    Prevents prompt injection via synthetic role delimiters such as
    [USER], [ASSISTANT], [SYSTEM]. Escapes [WORD] → \\[WORD\\] so
    the LLM does not interpret the sequence as a role boundary.

    Content is preserved — only the brackets are escaped.

    Contract: behaviors:Security:MUST:1
    """
    return _ROLE_INJECTION_PATTERN.sub(r"\\[\1\\]", text)


__all__ = [
    "convert_chat_request",
    "extract_prompt_from_chat_request",
    "extract_system_message",
    "build_request_payload_for_observability",
    "build_response_payload_for_observability",
    # Private exports for backward compatibility with tests
    "_extract_message_content",
    "_extract_content_block",
]


def _repair_tool_sequence(
    messages: list[Any],
) -> tuple[list[Any], int]:
    """Detect and repair malformed tool sequences by inserting synthetic results.

    Scans assistant messages for tool_call blocks that have no corresponding
    tool_result in any subsequent message. For each unmatched call, inserts a
    synthetic user message containing a tool_result block immediately after the
    offending assistant message. Returns a new list; original is not mutated.

    Detection is scoped to the current message list only (stateless). No
    cross-request state is needed because repair operates on a local copy.

    SDK kernel result format: role='tool' Message with tool_call_id attribute.
    Each Message.tool_call_id identifies the specific ToolCallBlock it answers.
    This is the Amplifier canonical format (verified: sessions 5fc69faf, 39ad7e88).

    Contract: provider-protocol:complete:MUST:9

    Args:
        messages: Ordered message list from the incoming ChatRequest.

    Returns:
        (repaired_messages, repair_count) — repaired list and number of
        synthetic results inserted. repair_count == 0 means no repair needed.
    """
    # Phase 1: collect call IDs (with source msg_index) and result IDs.
    # dict-fallbacks are mandatory: getattr({"key": "v"}, "key", None) returns None.
    tool_calls: dict[str, tuple[int, str]] = {}  # call_id → (msg_index, tool_name)
    unnamed_calls: list[tuple[int, str]] = []  # (msg_index, tool_name) — no call_id
    tool_result_ids: set[str] = set()

    for idx, msg in enumerate(messages):
        role: str = getattr(msg, "role", "")
        content: Any = getattr(msg, "content", None)
        if content is None:
            continue

        # SDK kernel format: role='tool' Message carries a single tool result.
        # The tool_call_id is on the Message itself, not in content blocks.
        if role == "tool":
            result_id: str | None = getattr(msg, "tool_call_id", None)
            if isinstance(msg, dict):
                result_id = result_id or cast(dict[str, Any], msg).get("tool_call_id")
            if result_id:
                tool_result_ids.add(result_id)
            continue  # String content — no blocks to iterate.

        # Only iterate content blocks when content is actually a list.
        # String content has no tool blocks.
        if not isinstance(content, list):
            continue
        for block in content:
            if block is None:
                continue
            block_type: str | None = getattr(block, "type", None)
            if isinstance(block, dict):
                block_type = block_type or cast(dict[str, Any], block).get("type")

            # Tool call block — only count those from assistant messages.
            # block_type == "tool_call" is the canonical check: ToolCallBlock.type is
            # always "tool_call" (verified against amplifier_core 1.3.3).
            if role == "assistant" and block_type == "tool_call":
                # ToolCallBlock (SDK kernel type) uses .id — always a non-None str.
                # Legacy ToolCallContent uses .tool_call_id.
                # Prefer .id (SDK canonical) then .tool_call_id (legacy fallback).
                call_id: str | None = getattr(block, "id", None) or getattr(
                    block, "tool_call_id", None
                )
                if isinstance(block, dict):
                    call_id = call_id or (
                        cast(dict[str, Any], block).get("id")
                        or cast(dict[str, Any], block).get("tool_call_id")
                    )
                tool_name: str = (
                    getattr(block, "name", None)
                    or getattr(block, "tool_name", None)
                    or (
                        cast(dict[str, Any], block).get("name")
                        or cast(dict[str, Any], block).get("tool_name")
                        if isinstance(block, dict)
                        else None
                    )
                    or "unknown"
                )
                if call_id:
                    tool_calls[call_id] = (idx, str(tool_name))
                else:
                    unnamed_calls.append((idx, str(tool_name)))

    # Phase 2: group unmatched calls by their source assistant message index.
    missing_by_idx: dict[int, list[str | None]] = {}
    for call_id, (msg_idx, _) in tool_calls.items():
        if call_id not in tool_result_ids:
            missing_by_idx.setdefault(msg_idx, []).append(call_id)
    for msg_idx, _ in unnamed_calls:
        missing_by_idx.setdefault(msg_idx, []).append(None)

    if not missing_by_idx:
        return list(messages), 0

    # Phase 3: insert synthetic results in reverse index order so earlier
    # insertions don't shift the indices of later ones.
    repaired = list(messages)
    repair_count = 0

    for msg_idx in sorted(missing_by_idx.keys(), reverse=True):
        synthetic_blocks = [
            {"type": "tool_result", "tool_call_id": cid, "output": _TOOL_SEQUENCE_REPAIR_MESSAGE}
            if cid
            else {"type": "tool_result", "output": _TOOL_SEQUENCE_REPAIR_MESSAGE}
            for cid in missing_by_idx[msg_idx]
        ]
        repair_count += len(synthetic_blocks)
        repaired.insert(
            msg_idx + 1,
            SimpleNamespace(role="user", content=synthetic_blocks),
        )

    return repaired, repair_count


def convert_chat_request(
    request: Any,
    *,
    default_model: str | None = None,
) -> CompletionRequest:
    """Convert kernel ChatRequest to internal CompletionRequest.

    Contract: provider-protocol:complete:MUST:1
    Contract: provider-protocol:complete:MUST:7 — Extract images from last user message
    Contract: provider-protocol:complete:MUST:8 — Include attachments in request
    Contract: provider-protocol:complete:MUST:9 — Repair malformed tool sequences

    Args:
        request: Kernel ChatRequest (or CompletionRequest passthrough).
        default_model: Default model to use if not specified in request.

    Returns:
        CompletionRequest suitable for SDK execution.
    """
    # Passthrough if already CompletionRequest
    if isinstance(request, CompletionRequest):
        return request

    messages: list[Any] = getattr(request, "messages", [])

    # Contract: provider-protocol:complete:MUST:9 — repair malformed tool sequences
    # before prompt extraction so the LLM receives coherent message history.
    repaired_messages, repair_count = _repair_tool_sequence(messages)
    if repair_count:
        logger.warning(
            "Malformed tool sequence repaired: %d tool call(s) without matching "
            "tool result — synthetic results inserted before LLM call",
            repair_count,
        )

    # Extract prompt using repaired messages (system/attachments use original request)
    prompt = _extract_prompt_from_messages(repaired_messages)

    # Extract model and tools
    model = getattr(request, "model", None) or default_model
    tools = getattr(request, "tools", []) or []

    # Extract attachments from last user message (images only)
    # Contract: sdk-boundary:ImagePassthrough:MUST:1
    attachments = extract_attachments_from_chat_request(request)

    # Extract system message for SDK session config (mode: replace)
    # CRITICAL: Without system_message, SDK uses default persona instead of bundle
    system_message = extract_system_message(request)

    # Extract max_output_tokens (per-session output cap, forwarded via SDK
    # ModelCapabilitiesOverride). Contract: provider-protocol:complete:MUST:10
    # NOTE: The canonical kernel field name is max_output_tokens (ChatRequest),
    # not max_tokens. Both Anthropic and OpenAI providers use request.max_output_tokens.
    max_tokens = getattr(request, "max_output_tokens", None)

    # reasoning_effort is forwarded verbatim to the SDK; capability validation
    # happens in provider.complete() via validate_reasoning_effort() once
    # CopilotModelInfo is resolved. Contract: provider-protocol:complete:MUST:11.
    # Empty string normalizes to None so downstream consumers can rely on `is
    # None` semantics. Non-str non-None raises: kernel ChatRequest types this
    # field as `str | None`, so other types are caller bugs that must surface.
    raw_reasoning_effort = getattr(request, "reasoning_effort", None)
    if raw_reasoning_effort is None or raw_reasoning_effort == "":
        reasoning_effort: str | None = None
    elif isinstance(raw_reasoning_effort, str):
        reasoning_effort = raw_reasoning_effort
    else:
        raise ConfigurationError(
            f"reasoning_effort must be str or None; got {type(raw_reasoning_effort).__name__}.",
            provider=PROVIDER_ID,
        )

    # context_tier is read defensively via getattr: amplifier-core 1.6.0 ChatRequest
    # has no context_tier field, so this returns None today. When None, the provider
    # falls through to the enable_long_context config default (per MUST:13).
    # Membership validation happens in provider.complete() via validate_context_tier().
    # Contract: provider-protocol:complete:MUST:12. Empty string normalizes to
    # None (no tier requested); a non-str, non-None value is a caller bug and raises.
    raw_context_tier = getattr(request, "context_tier", None)
    if raw_context_tier is None or raw_context_tier == "":
        context_tier: str | None = None
    elif isinstance(raw_context_tier, str):
        context_tier = raw_context_tier
    else:
        raise ConfigurationError(
            f"context_tier must be str or None; got {type(raw_context_tier).__name__}.",
            provider=PROVIDER_ID,
        )

    return CompletionRequest(
        prompt=prompt,
        model=model,
        tools=tools,
        attachments=attachments,
        system_message=system_message,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        context_tier=context_tier,
    )


def _redact_token_for_log(value: str) -> str:
    """Render a short enum-like ``value`` safely for error messages and logs.

    Returns the value verbatim only when it matches the well-formed token
    pattern (lowercase ASCII letters / underscores, <= 16 chars). Anything
    else is replaced with ``<redacted; len=N>`` so an injected secret
    fragment cannot leak via a ConfigurationError text or log line. Shared by
    the reasoning_effort and context_tier validators.

    Contract: provider-protocol:complete:MUST:11 (defense in depth)
    """
    if _REASONING_EFFORT_TOKEN_RE.match(value):
        return repr(value)
    return f"<redacted; len={len(value)}>"


def validate_reasoning_effort(
    reasoning_effort: str | None,
    model_info: CopilotModelInfo | None,
    *,
    model_id: str,
) -> str | None:
    """Validate caller-supplied reasoning_effort against model capability.

    Contract: provider-protocol:complete:MUST:11

    Layer-1 pre-flight gate, called from ``provider.complete()`` after
    ``CopilotModelInfo`` lookup but before any SDK call. Returns the value
    unchanged on success; raises ``ConfigurationError`` on policy violation.

    Validation lives here (not in the membrane) because the membrane
    deliberately knows nothing about model capability metadata.

    Behavior:
        * empty/None → returns ``None``
        * value not in the SDK literal allowlist
          ``{"low","medium","high","xhigh"}`` → raises (universal shape gate;
          rejects mixed-case and unknown tokens regardless of model_info;
          value redacted in error message when it doesn't match the safe
          token shape)
        * model_info missing, value passes shape gate → returns value, defers
          final per-model policy to SDK Layer-2 backstop (``errors.yaml:P4``)
        * model_info present, ``supports_reasoning_effort=False`` → raises
        * model_info present, non-empty
          ``supported_reasoning_efforts`` excludes the value → raises
        * model_info present, value accepted → returns value verbatim

    Args:
        reasoning_effort: Verbatim ``ChatRequest.reasoning_effort``.
        model_info: Resolved capability descriptor; ``None`` on cache miss.
        model_id: Effective model id, used in error messages.

    Returns:
        The validated string, or ``None`` when no effort was requested.

    Raises:
        ConfigurationError: Capability mismatch or value not in the active
            allowlist.
    """
    if not reasoning_effort:
        return None

    safe = _redact_token_for_log(reasoning_effort)

    # Universal shape gate. The SDK accepts only the fixed lowercase Literal
    # set; any other token (mixed-case "High", typos, novel values, overlong
    # strings) is rejected here regardless of model_info presence so the
    # contract's mixed-case rejection clause holds even when
    # supported_reasoning_efforts is empty. The redactor above ensures the
    # error message never echoes a non-token-shape value verbatim.
    if reasoning_effort not in _REASONING_EFFORT_FALLBACK_ALLOWLIST:
        allowed = ", ".join(repr(v) for v in sorted(_REASONING_EFFORT_FALLBACK_ALLOWLIST))
        raise ConfigurationError(
            f"reasoning_effort={safe} for model {model_id!r} is not in the "
            f"SDK literal allowlist (case-sensitive). Allowed values: {allowed}.",
            provider=PROVIDER_ID,
        )

    if model_info is None:
        # Defense-in-depth: redact model_id before logging. ChatRequest.model is
        # caller-controlled; a misrouted secret would otherwise reach the log.
        logger.info(
            "[REQUEST_ADAPTER] No CopilotModelInfo for model=%s; deferring "
            "final reasoning_effort validation to SDK backstop (Layer 2). "
            "reasoning_effort=%s",
            redact_sensitive_text(model_id),
            safe,
        )
        return reasoning_effort

    if not model_info.supports_reasoning_effort:
        raise ConfigurationError(
            f"Model {model_id!r} does not support reasoning_effort. "
            f"Caller passed reasoning_effort={safe}; remove the "
            f"field or pick a model whose capability descriptor advertises "
            f"supports_reasoning_effort=True.",
            provider=PROVIDER_ID,
        )

    allowlist = model_info.supported_reasoning_efforts
    if allowlist and reasoning_effort not in allowlist:
        allowed = ", ".join(repr(v) for v in allowlist)
        raise ConfigurationError(
            f"Model {model_id!r} does not support "
            f"reasoning_effort={safe}. Allowed values: {allowed}.",
            provider=PROVIDER_ID,
        )

    return reasoning_effort


def validate_context_tier(
    context_tier: str | None,
    *,
    model_id: str,
) -> str | None:
    """Validate caller-supplied context_tier against the static SDK allowlist.

    Contract: provider-protocol:complete:MUST:12

    Pre-flight gate, called from ``provider.complete()`` before any SDK call.
    Returns the value unchanged on success; raises ``ConfigurationError`` on a
    membership violation.

    Unlike :func:`validate_reasoning_effort` there is no per-model capability
    descriptor for context tier (the SDK exposes no ``supports_context_tier`` /
    ``supported_context_tiers`` on any model type), so validation is membership
    against the static SDK literal allowlist
    ``{"default", "long_context"}`` only. The membrane forwards the verbatim
    string; the SDK ``ContextTier`` enum is never constructed here because it is
    a plain ``enum.Enum`` and is not JSON-serializable by the SDK's JSON-RPC
    sender.

    Behavior:
        * empty/None → returns ``None``
        * value not in ``{"default","long_context"}`` (including mixed-case and
          overlong values) → raises ``ConfigurationError`` with the rejected
          value redacted when it doesn't match the safe token shape

    Args:
        context_tier: Verbatim ``ChatRequest.context_tier``.
        model_id: Effective model id, used in error messages.

    Returns:
        The validated string, or ``None`` when no tier was requested.

    Raises:
        ConfigurationError: Value not in the SDK literal allowlist.
    """
    if not context_tier:
        return None

    if context_tier not in _CONTEXT_TIER_ALLOWLIST:
        safe = _redact_token_for_log(context_tier)
        allowed = ", ".join(repr(v) for v in sorted(_CONTEXT_TIER_ALLOWLIST))
        raise ConfigurationError(
            f"context_tier={safe} for model {model_id!r} is not in the SDK "
            f"literal allowlist (case-sensitive). Allowed values: {allowed}.",
            provider=PROVIDER_ID,
        )

    return context_tier


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
    return _extract_prompt_from_messages(messages)


def _extract_prompt_from_messages(messages: list[Any]) -> str:
    """Format an ordered message list into a prompt string.

    Internal helper called by both extract_prompt_from_chat_request (public API)
    and convert_chat_request (which passes repaired messages instead of originals).

    Args:
        messages: Ordered message list (may include synthetic repair entries).

    Returns:
        Formatted prompt string with role markers and all content types.
    """
    if not messages:
        return ""

    formatted_parts: list[str] = []

    for msg in messages:
        role: str = getattr(msg, "role", "user")

        # C-4: Skip system messages — they are forwarded via SDK session_config
        # (system_message=..., mode="replace") to avoid dual-path injection.
        # Contract: sdk-boundary:Config:MUST:2
        if role == "system":
            continue

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
        return _sanitize_content_for_injection(content)

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
            # Contract: behaviors:Security:MUST:1 — sanitize thinking content
            # Thinking blocks are in scope: adversarial tool results can cause the model
            # to echo role markers in reasoning, which embeds them in subsequent prompt turns.
            return f"[Thinking: {_sanitize_content_for_injection(str(thinking))}]"
        return ""

    # TextContent - extract text attribute
    if block_type == "text" or (block_type is None and hasattr(block, "text")):
        text: str | None = getattr(block, "text", None) or _get("text")
        # Contract: behaviors:Security:MUST:1 — sanitize user content
        return _sanitize_content_for_injection(str(text)) if text else ""

    # Skip tool_call blocks entirely — they are handled via tool_calls field.
    # This prevents fake tool call detection from triggering on prior turns.
    # ToolCallBlock.type is always "tool_call" (amplifier_core 1.3.3 verified).
    if block_type == "tool_call":
        return ""

    # ToolResultContent - format tool result including tool_call_id for correlation
    # L-2: MUST include tool_call_id so the model can correlate results to calls.
    # Contract: provider-protocol:complete:MUST — preserve tool call IDs
    # ToolResultBlock.type is always "tool_result" (amplifier_core 1.3.3 verified).
    if block_type == "tool_result":
        output: str | None = getattr(block, "output", None) or _get("output")
        if output:
            tool_result_call_id: str | None = getattr(block, "tool_call_id", None) or _get(
                "tool_call_id"
            )
            # Contract: behaviors:Security:MUST:1 — sanitize tool result output AND
            # tool_call_id. Both are interpolated into the prompt. output is
            # user-controlled (external service response); tool_call_id originates
            # from the LLM but defense-in-depth requires sanitizing all interpolated
            # values — repair paths (P0-4) now route assistant IDs through this code.
            sanitized_output = _sanitize_content_for_injection(str(output))
            if tool_result_call_id:
                sanitized_id = _sanitize_content_for_injection(str(tool_result_call_id))
                return f"[Tool Result (id={sanitized_id}): {sanitized_output}]"
            return f"[Tool Result: {sanitized_output}]"
        return ""

    # Image block — SDK cannot re-process images from prior turns; replace with placeholder.
    # S3 Fix: image blocks fell through to the fallback and returned "" silently,
    # losing the model's context that an image existed in that turn.
    # Contract: behaviors:Security:MUST:1 (content integrity in history)
    if block_type == "image" or (
        block_type is None
        and (hasattr(block, "url") or hasattr(block, "source"))
        and not hasattr(block, "text")
        and not hasattr(block, "output")
    ):
        mime: str | None = (
            getattr(block, "media_type", None)
            or getattr(block, "mime_type", None)
            or _get("media_type")
            or _get("mime_type")
        )
        placeholder = f"[Image: {mime}]" if mime else "[Image]"
        logger.warning(
            "Image block in conversation history replaced with %r "
            "(SDK cannot re-process images from prior turns)",
            placeholder,
        )
        return placeholder

    # Fallback - try common text attributes
    for attr in ("text", "content", "value"):
        val: Any = getattr(block, attr, None) or _get(attr)
        if val:
            # Contract: behaviors:Security:MUST:1 — sanitize all interpolated values.
            # This fallback covers unknown block types whose string representation
            # reaches the prompt. Defense-in-depth: any attacker-controlled content
            # block must go through the injection sanitizer regardless of type.
            return _sanitize_content_for_injection(str(val))

    return ""


def extract_system_message(request: Any) -> str | None:
    """Extract system message(s) from ChatRequest for SDK session config.

    The SDK handles system messages specially — they're passed to the session
    config with mode="replace" rather than included in the prompt.

    This is CRITICAL for Amplifier agent behavior:
    - Without system_message, SDK uses default "GitHub Copilot CLI" persona
    - With system_message, the Amplifier bundle persona takes precedence
    - Missing system_message causes agent to not follow bundle instructions

    If multiple system messages exist, they are joined with double newlines.

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


# =============================================================================
# OBSERVABILITY PAYLOAD BUILDERS
# =============================================================================
# Contract: observability:Verbosity:MUST:1 — raw flag controls payload inclusion
# Separation of Concerns: Payload construction belongs in request_adapter.py,
# not inline in provider.py. Provider stays thin, calling these helpers.
# =============================================================================


def build_request_payload_for_observability(
    model: str,
    request: Any,
    internal_request: CompletionRequest,
) -> dict[str, Any]:
    """Build observability payload for llm:request event (raw=true debug mode).

    Contract: observability:Verbosity:MUST:1
    Contract: observability:Debug:MUST:1 — tool_schemas (full schema per tool)
    Contract: observability:Debug:MUST:2 — system_message_length
    Contract: observability:Debug:MUST:3 — prompt_length (NOT full prompt text)

    Tool Format Support:
        This function handles two tool specification formats:
        1. **Nested format** (OpenAI-style): {"function": {"name": "...", ...}}
           - Used by kernel ChatRequest when tools come from OpenAI-style schemas
        2. **Flat format** (Amplifier-native): {"name": "...", "parameters": {...}}
           - Used by Amplifier's internal ToolSpec model

        Both formats are valid and may appear depending on how the request
        was constructed. The SDK accepts either format.

    Args:
        model: The model being used for the request.
        request: The original kernel ChatRequest.
        internal_request: The converted internal CompletionRequest.

    Returns:
        Dict written under the 'raw' key in llm:request when raw=true.
        Passed through security_redaction.redact_dict() before emission.
    """
    tool_names: list[str] = []
    tool_schemas: list[dict[str, Any]] = []

    if internal_request.tools:
        for tool in internal_request.tools:
            name: str | None = None
            description: str | None = None
            parameters: dict[str, Any] | None = None

            # Format 1: ToolSpec object (Amplifier kernel passes these)
            # Has .name attribute directly (Pydantic BaseModel / dataclass / SimpleNamespace)
            # Contract: observability:Payload:SHOULD:2 — Type-safe tool name extraction
            if hasattr(tool, "name") and not isinstance(tool, dict):
                name_raw = getattr(tool, "name", None)
                if isinstance(name_raw, str):
                    name = name_raw
                desc_raw = getattr(tool, "description", None)
                if isinstance(desc_raw, str):
                    description = desc_raw[:300]  # truncate — descriptions can be large
                params_raw = getattr(tool, "parameters", None) or getattr(
                    tool, "input_schema", None
                )
                if isinstance(params_raw, dict):
                    parameters = params_raw

            # Format 2: Dict — check isinstance before .get()
            elif isinstance(tool, dict):
                # Subformat 2a: Nested (OpenAI-style) {"function": {"name": "..."}}
                func_raw = tool.get("function")  # type: ignore[reportUnknownMemberType,reportUnknownVariableType]
                if isinstance(func_raw, dict):
                    name_raw = func_raw.get("name")  # type: ignore[reportUnknownVariableType]
                    if isinstance(name_raw, str):
                        name = name_raw
                    desc_raw = func_raw.get("description")  # type: ignore[reportUnknownVariableType]
                    if isinstance(desc_raw, str):
                        description = desc_raw[:300]
                    params_raw = func_raw.get("parameters") or func_raw.get("input_schema")  # type: ignore[reportUnknownVariableType]
                    if isinstance(params_raw, dict):
                        parameters = params_raw
                # Subformat 2b: Flat (Amplifier-native) {"name": "..."}
                elif "name" in tool:
                    name_raw = tool.get("name")  # type: ignore[reportUnknownVariableType]
                    if isinstance(name_raw, str):
                        name = name_raw
                    desc_raw = tool.get("description")  # type: ignore[reportUnknownVariableType]
                    if isinstance(desc_raw, str):
                        description = desc_raw[:300]
                    params_raw = tool.get("parameters") or tool.get("input_schema")  # type: ignore[reportUnknownVariableType]
                    if isinstance(params_raw, dict):
                        parameters = params_raw

            if name:
                tool_names.append(name)
                schema: dict[str, Any] = {"name": name}
                if description:
                    schema["description"] = description
                if parameters is not None:
                    schema["parameters"] = parameters
                tool_schemas.append(schema)

    sys_msg = internal_request.system_message
    prompt = internal_request.prompt or ""

    return {
        # Analytics fields (always present — useful for summary views)
        "model": model,
        "message_count": len(getattr(request, "messages", [])),
        "tool_names": tool_names,
        "has_system_message": bool(sys_msg),
        # Correlates llm:request with truncation WARNING in streaming (finish_reason=length).
        # None when caller did not set max_output_tokens.
        "max_output_tokens": internal_request.max_tokens,
        # Debug fields (Contract: observability:Debug:MUST:1/2/3)
        # These are the reason raw=true exists: actual schemas, not just names.
        "tool_schemas": tool_schemas,
        "system_message_length": len(sys_msg) if sys_msg else 0,
        "prompt_length": len(prompt),
    }


def build_response_payload_for_observability(
    response: Any,
    tool_calls: int,
) -> dict[str, Any]:
    """Build observability payload for llm:response event.

    Contract: observability:Verbosity:MUST:1

    Args:
        response: The ChatResponse object.
        tool_calls: Number of tool calls in response.

    Returns:
        Dict suitable for raw_response parameter of emit_response_ok().
    """
    # Count content blocks safely
    # Contract: observability:Payload:SHOULD:1 — Type-safe content counting
    # Defense: content may be string in edge cases, which would return char count
    content_block_count = 0
    content = getattr(response, "content", None)
    if content is not None and isinstance(content, (list, tuple)) and not isinstance(content, str):
        content_block_count = len(content)  # type: ignore[arg-type]

    return {
        "text_length": len(response.text) if hasattr(response, "text") and response.text else 0,
        "content_block_count": content_block_count,
        "tool_calls": tool_calls,
        "finish_reason": getattr(response, "finish_reason", None),
    }
