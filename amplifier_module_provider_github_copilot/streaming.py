"""Event translation / streaming module. Contract: event-vocabulary.md."""

import fnmatch
import functools
import json
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, cast

import yaml

# Import ChatResponse at module level for subclass definition
# Contract: streaming-contract:StreamingResponse:MUST:1
from amplifier_core import (
    ChatResponse,
    TextContent,
    ThinkingContent,
    ToolCallContent,
)

# Use _compat.py for consistent ConfigurationError import (W-03 code review)
from ._compat import ConfigurationError

# Contract: sdk-boundary:Membrane:MUST:1 — import from sdk_adapter package, not submodules
from .sdk_adapter import extract_event_fields

logger = logging.getLogger(__name__)

__all__ = [
    "DomainEventType",
    "EventClassification",
    "DomainEvent",
    "AccumulatedResponse",
    "StreamingAccumulator",
    "StreamingChatResponse",
    "EventConfig",
    "MAX_EXTRACTION_DEPTH",
    "load_event_config",
    "classify_event",
    "extract_response_content",
    "translate_event",
]


def _parse_tool_arguments(arguments: Any) -> dict[str, Any]:
    """Parse tool call arguments from str | dict SDK responses.

    The Copilot SDK may deliver ``arguments`` as a JSON string or a dict.
    S4 Fix: The original guard ``isinstance(args, dict) else {}`` silently
    discarded ALL tool arguments when the SDK sent them as a JSON string.

    Contract: streaming-contract:ToolCallBlock:MUST:1

    Args:
        arguments: Raw arguments value from the SDK tool_call block.
            May be ``None`` (no args), ``dict`` (already parsed), or
            ``str`` (JSON-encoded dict).

    Returns:
        Parsed argument dict. Returns ``{}`` for None/unparseable inputs.
    """
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            logger.warning(
                "Tool call arguments are not valid JSON; falling back to {}: %r",
                arguments[:200],
            )
            return {}
        if not isinstance(parsed, dict):
            logger.warning(
                "Tool call arguments parsed to %s (expected dict); falling back to {}",
                type(parsed).__name__,
            )
            return {}
        return parsed
    # None or unexpected type — treat as empty
    return {}


class StreamingChatResponse(ChatResponse):
    """ChatResponse with content_blocks for streaming UI compatibility.

    The loop-streaming orchestrator checks `response.content_blocks` to emit
    CONTENT_BLOCK_START and CONTENT_BLOCK_END events for real-time UI updates.

    Attributes:
        content_blocks: List of streaming content types for UI event emission.
            Uses content_models types (TextContent, ThinkingContent, ToolCallContent).
        text: Convenience field with combined text content.

    Contract: streaming-contract:StreamingResponse:MUST:1-4
    """

    content_blocks: list[TextContent | ThinkingContent | ToolCallContent] | None = None
    text: str | None = None


class DomainEventType(Enum):
    """Domain event types per event-vocabulary.md.

    NOTE: Thinking/reasoning content uses CONTENT_DELTA with block_type="THINKING"
    per events.yaml bridge mappings. A separate THINKING_DELTA type was removed
    as dead code in H-3 fix.
    """

    CONTENT_DELTA = "CONTENT_DELTA"
    TOOL_CALL = "TOOL_CALL"
    USAGE_UPDATE = "USAGE_UPDATE"
    TURN_COMPLETE = "TURN_COMPLETE"
    SESSION_IDLE = "SESSION_IDLE"  # Reserved for future extensibility
    ERROR = "ERROR"


class EventClassification(Enum):
    """How to handle SDK events."""

    BRIDGE = "bridge"
    CONSUME = "consume"
    DROP = "drop"


@dataclass
class DomainEvent:
    """Domain event emitted from SDK event translation."""

    type: DomainEventType
    data: dict[str, Any] = field(default_factory=lambda: {})
    block_type: str | None = None


@dataclass
class AccumulatedResponse:
    """Accumulated response from streaming events."""

    text_content: str = ""
    thinking_content: str = ""
    # Opaque signature for extended thinking (Anthropic models)
    # Contract: streaming-contract:ThinkingBlock:MUST:1
    reasoning_opaque: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=lambda: [])
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    error: dict[str, Any] | None = None
    is_complete: bool = False


@dataclass
class StreamingAccumulator:
    """Accumulates streaming domain events into final response."""

    # H-4: Per-block text storage for streaming-contract:Accumulation:MUST:2
    # A new block is started when a TOOL_CALL event arrives, so pre-tool and
    # post-tool text land in separate TextBlocks in the final response.
    _text_blocks: list[str] = field(default_factory=lambda: [""])
    thinking_content: str = ""
    # Opaque signature for extended thinking (Anthropic models)
    # Contract: streaming-contract:ThinkingBlock:MUST:1
    reasoning_opaque: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=lambda: [])
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    error: dict[str, Any] | None = None
    is_complete: bool = False
    # SDK session ID for observability correlation
    sdk_session_id: str | None = None
    # SDK subprocess PID for log file correlation (~/.copilot/logs/process-*-{pid}.log)
    # Contract: observability:Events:SHOULD:3
    sdk_pid: str | None = None
    # Ordered block sequence for content ordering preservation.
    # Each entry: {"type": "text"|"thinking"|"tool_call", ...data}
    # Contract: streaming-contract:StreamingResponse:MUST:2 (P1-3 fix)
    _ordered_blocks: list[dict[str, Any]] = field(default_factory=lambda: [])

    @property
    def text_content(self) -> str:
        """Backward-compat joined text.  Prefer iterating text_blocks directly."""
        return "".join(self._text_blocks)

    def add(self, event: DomainEvent) -> None:
        """Add domain event to accumulator.

        Usage events are processed even after completion since SDK may send
        ASSISTANT_USAGE after TURN_COMPLETE.
        Contract: streaming-contract:completion:MUST:1
        """
        # USAGE_UPDATE events must be processed even after completion.
        # SDK sends assistant.usage AFTER assistant.turn_end, so we can't
        # block usage events with the is_complete guard.
        # Bug fix: Zero usage reported when no tool calls
        if event.type == DomainEventType.USAGE_UPDATE:
            self.usage = event.data
            return

        # Guard against other events after completion
        if self.is_complete:
            return

        if event.type == DomainEventType.CONTENT_DELTA:
            text = event.data.get("text", "")
            if event.block_type == "THINKING":
                self.thinking_content += text
                # Capture reasoning_opaque for ThinkingBlock.signature
                # Contract: streaming-contract:ThinkingBlock:MUST:1
                if event.data.get("reasoning_opaque"):
                    self.reasoning_opaque = event.data["reasoning_opaque"]
                # Track ordered: extend last thinking block or start new one
                if self._ordered_blocks and self._ordered_blocks[-1]["type"] == "thinking":
                    self._ordered_blocks[-1]["text"] += text
                    if event.data.get("reasoning_opaque"):
                        self._ordered_blocks[-1]["opaque"] = event.data["reasoning_opaque"]
                else:
                    self._ordered_blocks.append(
                        {
                            "type": "thinking",
                            "text": text,
                            "opaque": event.data.get("reasoning_opaque"),
                        }
                    )
            else:
                self._text_blocks[-1] += text
                # Track ordered: extend last text block or start new one
                # Only track non-empty text to preserve thinking block consolidation.
                # Empty text deltas (e.g. assistant.message_delta with empty delta_content)
                # must not create empty text entries that break consecutive thinking consolidation.
                # Contract: streaming-contract:StreamingResponse:MUST:2
                if text:
                    if self._ordered_blocks and self._ordered_blocks[-1]["type"] == "text":
                        self._ordered_blocks[-1]["text"] += text
                    else:
                        self._ordered_blocks.append({"type": "text", "text": text})
        elif event.type == DomainEventType.TOOL_CALL:
            self.tool_calls.append(event.data)
            # H-4: Start a new text block after a tool call so that any
            # post-tool text lands in a separate TextBlock.
            # Contract: streaming-contract:Accumulation:MUST:2
            self._text_blocks.append("")
            # Track ordered: tool_call is a block in insertion order
            self._ordered_blocks.append({"type": "tool_call", "data": event.data})
        elif event.type == DomainEventType.TURN_COMPLETE:
            self.finish_reason = event.data.get("finish_reason", "stop")
            self.is_complete = True
        elif event.type == DomainEventType.ERROR:
            self.error = event.data
            self.is_complete = True

    def get_result(self) -> AccumulatedResponse:
        """Get accumulated response."""
        return AccumulatedResponse(
            text_content=self.text_content,
            thinking_content=self.thinking_content,
            reasoning_opaque=self.reasoning_opaque,
            tool_calls=self.tool_calls,
            usage=self.usage,
            finish_reason=self.finish_reason,
            error=self.error,
            is_complete=self.is_complete,
        )

    def to_chat_response(self) -> "StreamingChatResponse":
        """Convert accumulated response to StreamingChatResponse.

        Builds two content representations:
        - content: Uses TextBlock/ThinkingBlock (Pydantic from message_models)
          for context persistence.
        - content_blocks: Uses TextContent/ThinkingContent/ToolCallContent
          (dataclass from content_models) for streaming UI events.

        Returns:
            StreamingChatResponse with content, content_blocks, tool_calls,
            usage, finish_reason, and text convenience field.

        Contract: streaming-contract:StreamingResponse:MUST:1-4
        """
        from amplifier_core import (
            TextBlock,
            ThinkingBlock,
            ToolCall,
            ToolCallBlock,
            ToolCallContent,
            Usage,
        )

        # Build content list (Pydantic types for context persistence)
        # Narrowed to actual kernel types — eliminates arg-type suppression at StreamingChatResponse
        # Contract: streaming-contract:StreamingResponse:MUST:3
        # ToolCallBlock (not ToolCall) is the ContentBlockUnion member for tool calls in content.
        # ToolCall goes in tool_calls; ToolCallBlock goes in content.
        # streaming-contract:ToolCallBlock:MUST:1
        # P1-3 fix: iterate _ordered_blocks to preserve block arrival order.
        # streaming-contract:StreamingResponse:MUST:2
        content: list[TextBlock | ThinkingBlock | ToolCallBlock] = []
        # Build content_blocks list (dataclass types for streaming UI)
        content_blocks: list[TextContent | ThinkingContent | ToolCallContent] = []
        # tool_calls list built from ordered_blocks tool_call entries
        tool_calls: list[Any] | None = None
        _tool_calls_list: list[Any] = []

        for block in self._ordered_blocks:
            btype = block["type"]
            if btype == "text":
                block_text: str = block["text"]
                if block_text:
                    content.append(TextBlock(text=block_text))
                    content_blocks.append(TextContent(text=block_text))
            elif btype == "thinking":
                thinking_text: str = block["text"]
                if thinking_text:
                    # Contract: streaming-contract:ThinkingBlock:MUST:1
                    content.append(
                        ThinkingBlock(
                            thinking=thinking_text,
                            signature=block.get("opaque") or self.reasoning_opaque,
                        )
                    )
                    content_blocks.append(ThinkingContent(text=thinking_text))
            elif btype == "tool_call":
                tc = block["data"]
                tool_call_id = tc.get("id", "") or str(uuid.uuid4())
                tc_name = tc.get("name", "")
                tc_args: dict[str, Any] = _parse_tool_arguments(tc.get("arguments"))
                # ToolCallBlock in content (ContentBlockUnion member); field is `input`
                # Contract: streaming-contract:ToolCallBlock:MUST:1
                content.append(ToolCallBlock(id=tool_call_id, name=tc_name, input=tc_args))
                _tool_calls_list.append(ToolCall(id=tool_call_id, name=tc_name, arguments=tc_args))
                content_blocks.append(
                    ToolCallContent(id=tool_call_id, name=tc_name, arguments=tc_args)
                )

        if _tool_calls_list:
            tool_calls = _tool_calls_list

        # NOTE: _ordered_blocks is the SOLE source of truth for to_chat_response().
        # All content, thinking, and tool_call blocks are added to _ordered_blocks
        # exclusively via add(). Direct field mutation of _text_blocks/thinking_content/
        # tool_calls bypasses _ordered_blocks and will produce empty content — that is
        # correct behaviour since direct mutation violates the accumulator contract.
        # streaming-contract:StreamingResponse:MUST:2

        # Convert usage - all three fields REQUIRED
        usage: Any | None = None
        if self.usage:
            usage = Usage(
                input_tokens=self.usage.get("input_tokens", 0),
                output_tokens=self.usage.get("output_tokens", 0),
                total_tokens=self.usage.get("total_tokens", 0),
            )

        # Normalize finish_reason for orchestrator
        # Contract: streaming-contract:FinishReason:MUST:5
        # In abort-on-capture flow, TURN_COMPLETE may not arrive.
        # The orchestrator relies on finish_reason to continue the agent loop.
        #
        # CRITICAL: tool_calls ALWAYS override SDK finish_reason
        # The SDK may send "stop" even when there are tool calls (deny flow),
        # but we MUST tell the orchestrator to execute them.
        if tool_calls:
            # Tool calls present: orchestrator must execute them
            # Overrides any SDK-provided finish_reason
            # Valid values per amplifier-core proto:
            #   "stop", "tool_calls", "length", "content_filter"
            normalized_finish_reason = "tool_calls"
        elif not self.finish_reason:
            # No tool calls and no SDK finish_reason: normal completion
            # Use "stop" per amplifier-core proto (not "end_turn" which is an SDK input key)
            normalized_finish_reason = "stop"
        else:
            # No tool calls but SDK provided finish_reason: preserve it
            normalized_finish_reason = self.finish_reason

        # Contract: content_blocks is None when empty (not empty list)
        # streaming-contract:StreamingResponse:MUST:4
        return StreamingChatResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=normalized_finish_reason,
            content_blocks=content_blocks if content_blocks else None,
            text=self.text_content or None,
        )


def _empty_str_to_str_dict() -> dict[str, str]:
    """Return an empty string-to-string dict."""
    return {}


def _empty_str_set() -> set[str]:
    """Return an empty string set."""
    return set()


@dataclass
class EventConfig:
    """Configuration for event translation.

    Contract: event-vocabulary.md, streaming-contract:ProgressiveStreaming
    """

    bridge_mappings: dict[str, tuple[DomainEventType, str | None]] = field(
        default_factory=lambda: {}
    )
    consume_patterns: list[str] = field(default_factory=lambda: [])
    drop_patterns: list[str] = field(default_factory=lambda: [])
    finish_reason_map: dict[str, str] = field(default_factory=_empty_str_to_str_dict)
    # Content event types for TTFT tracking (behaviors:Streaming:MUST:1)
    content_event_types: set[str] = field(default_factory=_empty_str_set)
    # Streaming emission types (streaming-contract:ProgressiveStreaming:SHOULD:1)
    text_content_types: set[str] = field(default_factory=_empty_str_set)
    thinking_content_types: set[str] = field(default_factory=_empty_str_set)
    # Session lifecycle event types (loaded from events.yaml session_lifecycle)
    idle_event_types: set[str] = field(default_factory=_empty_str_set)
    error_event_types: set[str] = field(default_factory=_empty_str_set)
    usage_event_types: set[str] = field(default_factory=_empty_str_set)


def _validate_no_classification_overlap(
    bridge_mappings: dict[str, tuple[DomainEventType, str | None]],
    consume_patterns: list[str],
    drop_patterns: list[str],
) -> None:
    """Validate no overlap between BRIDGE, CONSUME, and DROP categories.

    Event classification must be unambiguous.
    Contract: event-vocabulary:Classification:MUST:1

    Raises:
        ConfigurationError: If any event type appears in multiple categories.

    """
    bridge_types = set(bridge_mappings.keys())

    # Check bridge vs consume (exact match)
    for pattern in consume_patterns:
        if pattern in bridge_types:
            raise ConfigurationError(
                f"Event classification overlap: '{pattern}' is in both BRIDGE and CONSUME. "
                "Each event type must have exactly one classification.",
                provider="github-copilot",
            )

    # Check bridge vs drop (exact match)
    for pattern in drop_patterns:
        if pattern in bridge_types:
            raise ConfigurationError(
                f"Event classification overlap: '{pattern}' is in both BRIDGE and DROP. "
                "Each event type must have exactly one classification.",
                provider="github-copilot",
            )

    # Check consume vs drop (exact match)
    consume_set = set(consume_patterns)
    for pattern in drop_patterns:
        if pattern in consume_set:
            raise ConfigurationError(
                f"Event classification overlap: '{pattern}' is in both CONSUME and DROP. "
                "Each event type must have exactly one classification.",
                provider="github-copilot",
            )

    # Check wildcard patterns against explicit types
    # Bridge types vs wildcards
    for bridge_type in bridge_types:
        for pattern in drop_patterns:
            if "*" in pattern and fnmatch.fnmatch(bridge_type, pattern):
                raise ConfigurationError(
                    f"Event classification overlap: BRIDGE type '{bridge_type}' "
                    f"matches DROP wildcard pattern '{pattern}'. "
                    "Each event type must have exactly one classification.",
                    provider="github-copilot",
                )
        for pattern in consume_patterns:
            if "*" in pattern and fnmatch.fnmatch(bridge_type, pattern):
                raise ConfigurationError(
                    f"Event classification overlap: BRIDGE type '{bridge_type}' "
                    f"matches CONSUME wildcard pattern '{pattern}'. "
                    "Each event type must have exactly one classification.",
                    provider="github-copilot",
                )

    # Consume explicit entries vs drop wildcards
    for consume_entry in consume_patterns:
        if "*" not in consume_entry:  # Only check explicit entries
            for drop_pattern in drop_patterns:
                if "*" in drop_pattern and fnmatch.fnmatch(consume_entry, drop_pattern):
                    raise ConfigurationError(
                        f"Event classification overlap: CONSUME entry '{consume_entry}' "
                        f"matches DROP wildcard pattern '{drop_pattern}'. "
                        "Each event type must have exactly one classification.",
                        provider="github-copilot",
                    )

    # Drop explicit entries vs consume wildcards
    for drop_entry in drop_patterns:
        if "*" not in drop_entry:  # Only check explicit entries
            for consume_pattern in consume_patterns:
                if "*" in consume_pattern and fnmatch.fnmatch(drop_entry, consume_pattern):
                    raise ConfigurationError(
                        f"Event classification overlap: DROP entry '{drop_entry}' "
                        f"matches CONSUME wildcard pattern '{consume_pattern}'. "
                        "Each event type must have exactly one classification.",
                        provider="github-copilot",
                    )


@functools.lru_cache(maxsize=4)
def _load_event_config_cached(config_path_str: str) -> EventConfig:
    """Internal cached loader - takes string path for hashability."""
    path = Path(config_path_str)
    if not path.exists():
        return EventConfig()  # Graceful fallback on missing file

    with open(config_path_str, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not raw:
        return EventConfig()

    classifications = raw.get("event_classifications", {})

    bridge_mappings: dict[str, tuple[DomainEventType, str | None]] = {}
    for i, mapping in enumerate(classifications.get("bridge", [])):
        # Defensive parsing with clear error messages
        try:
            sdk_type = mapping.get("sdk_type")
            if sdk_type is None:
                raise ValueError(f"Bridge mapping {i} missing required 'sdk_type' key")

            domain_type_str = mapping.get("domain_type")
            if domain_type_str is None:
                raise ValueError(
                    f"Bridge mapping {i} (sdk_type={sdk_type}) missing required 'domain_type' key"
                )

            try:
                domain_type = DomainEventType[domain_type_str]
            except KeyError as err:
                raise ValueError(
                    f"Bridge mapping {i} (sdk_type={sdk_type}) has unknown "
                    f"domain_type '{domain_type_str}'. "
                    f"Valid types: {[t.name for t in DomainEventType]}"
                ) from err

            bridge_mappings[sdk_type] = (domain_type, mapping.get("block_type"))
        except ValueError as e:
            # Re-raise as ConfigurationError with context
            raise ConfigurationError(
                f"Invalid event config in events.yaml: {e}",
                provider="github-copilot",
            ) from e

    # Load finish_reason_map
    finish_reason_map = raw.get("finish_reason_map", {})

    consume_patterns = classifications.get("consume", [])
    drop_patterns = classifications.get("drop", [])

    # Load content event types for TTFT tracking (Three-Medium: policy from YAML)
    # Contract: behaviors:Streaming:MUST:1
    content_event_types = set(raw.get("content_event_types", []))

    # Load streaming emission types (Three-Medium: policy from YAML)
    # Contract: streaming-contract:ProgressiveStreaming:SHOULD:1
    streaming_emission = raw.get("streaming_emission", {})
    text_content_types = set(streaming_emission.get("text_content_types", []))
    thinking_content_types = set(streaming_emission.get("thinking_content_types", []))

    # Load session lifecycle event types (Three-Medium: policy from YAML)
    # Used by event_helpers.py for SDK event type detection
    session_lifecycle = raw.get("session_lifecycle", {})
    idle_event_types = set(session_lifecycle.get("idle_events", []))
    error_event_types = set(session_lifecycle.get("error_events", []))
    usage_event_types = set(session_lifecycle.get("usage_events", []))

    # CORE CONFIG VALIDATION: session_lifecycle is structural, not observability
    # Contract: streaming-contract:SessionLifecycle:MUST:1
    # If idle_events is empty, provider cannot detect session completion → infinite hang
    # This is fail-fast at load time, not silent degradation
    if not idle_event_types:
        raise ConfigurationError(
            "session_lifecycle.idle_events is empty or missing in events.yaml. "
            "Provider cannot detect session completion without this configuration. "
            "Expected: ['session.idle', 'idle'] or similar.",
            provider="github-copilot",
        )

    # Validate no overlap between BRIDGE, CONSUME, and DROP categories
    # Contract: event-vocabulary:Classification:MUST:1
    # Each event type has exactly one classification
    _validate_no_classification_overlap(bridge_mappings, consume_patterns, drop_patterns)

    return EventConfig(
        bridge_mappings=bridge_mappings,
        consume_patterns=consume_patterns,
        drop_patterns=drop_patterns,
        finish_reason_map=finish_reason_map,
        content_event_types=content_event_types,
        text_content_types=text_content_types,
        thinking_content_types=thinking_content_types,
        idle_event_types=idle_event_types,
        error_event_types=error_event_types,
        usage_event_types=usage_event_types,
    )


def load_event_config(config_path: str | Path | None = None) -> EventConfig:
    """Load event classification config from YAML. Defaults to config/events.yaml.

    Gracefully handles missing files by returning default config.
    Config lives inside the wheel at amplifier_module_provider_github_copilot/config/
    """
    if config_path is None:
        config_path = str(Path(__file__).parent / "config" / "data" / "events.yaml")
    else:
        config_path = str(config_path)
    config = _load_event_config_cached(config_path)
    # DEBUG: Log config loading for diagnostics
    logger.debug(
        "[EVENT_CONFIG] Loaded from %s: idle_events=%s, error_events=%s, usage_events=%s",
        config_path,
        config.idle_event_types,
        config.error_event_types,
        config.usage_event_types,
    )
    return config


def _matches_pattern(event_type: str, patterns: list[str]) -> bool:
    """Check if event type matches any pattern (supports wildcards)."""
    return any(fnmatch.fnmatch(event_type, p) for p in patterns)


def classify_event(sdk_event_type: str, config: EventConfig) -> EventClassification:
    """Classify SDK event type using config."""
    if sdk_event_type in config.bridge_mappings:
        return EventClassification.BRIDGE
    if _matches_pattern(sdk_event_type, config.consume_patterns):
        return EventClassification.CONSUME
    if _matches_pattern(sdk_event_type, config.drop_patterns):
        return EventClassification.DROP
    logger.warning("Unknown SDK event type: %s", sdk_event_type)
    return EventClassification.DROP


def _extract_event_data(sdk_event: dict[str, Any]) -> dict[str, Any]:
    """Extract data from SDK event dict.

    Handles nested 'data' objects (SessionEventData shape).
    Flattens data attributes into the result dict for accumulator consumption.
    """
    result: dict[str, Any] = {}
    for k, v in sdk_event.items():
        if k == "type":
            continue
        if k == "data" and v is not None:
            # Flatten nested data object (SessionEventData or dict)
            if isinstance(v, dict):
                nested_dict = cast(dict[str, Any], v)
                for dk, dv in nested_dict.items():
                    if dv is not None:
                        result[dk] = dv
            elif hasattr(v, "__dict__"):
                # SessionEventData object - delegate to unified extraction.
                # extract_event_fields handles delta_content→text, tool_call_id→id,
                # tool_name→name, and reasoning_text normalization in one place.
                extracted = extract_event_fields(v)
                for ek, ev in extracted.items():
                    if ek != "type" and ev is not None:
                        result[ek] = ev
        else:
            result[k] = v
    return result


# ============================================================================
# Response Extraction
# ============================================================================

# Maximum recursion depth for extract_response_content
_MAX_EXTRACTION_DEPTH = 5
MAX_EXTRACTION_DEPTH = _MAX_EXTRACTION_DEPTH  # Public alias for re-export


def extract_response_content(response: Any, _depth: int = 0) -> str:
    """Extract text content from SDK response.

    Contract: sdk-response.md

    The SDK returns Data dataclass objects with .content attribute.
    This function handles all response shapes:
    1. Data object with .content attribute
    2. Dict with 'content' key
    3. Response wrapper with .data attribute (recurses)
    4. None (returns empty string)

    Recursion is bounded by _MAX_EXTRACTION_DEPTH to prevent
    stack overflow from deeply nested or circular .data chains.

    Args:
        response: SDK response (Data object, dict, wrapper, or None)
        _depth: Internal recursion depth counter (do not pass externally)

    Returns:
        Extracted text content as string.

    """
    if response is None:
        return ""

    # Guard against infinite recursion
    if _depth > _MAX_EXTRACTION_DEPTH:
        return ""

    # P0 Fix (C1): Added null guard to .data check. Previous code recursed into
    # .data even when None, returning "" silently when .content was valid.
    # SDK design: .data wraps a nested Data object with .content - unwrap first.
    if hasattr(response, "data") and response.data is not None:
        return extract_response_content(response.data, _depth + 1)

    # Fall back to direct .content if .data doesn't exist or is None
    if hasattr(response, "content"):
        content = response.content  # type: ignore[union-attr]
        return str(content) if content is not None else ""

    # Handle dict response
    if isinstance(response, dict):
        return str(cast(dict[str, Any], response).get("content", ""))

    # Fallback for unknown types (shouldn't reach here normally)
    return ""


def translate_event(sdk_event: dict[str, Any], config: EventConfig) -> DomainEvent | None:
    """Translate SDK event to domain event. Contract: event-vocabulary.md."""
    raw_type = sdk_event.get("type", "")
    # Handle both enum objects and strings (SDK uses enums, tests may use strings)
    event_type: str = raw_type.value if hasattr(raw_type, "value") else str(raw_type)
    classification = classify_event(event_type, config)

    if classification != EventClassification.BRIDGE:
        return None

    domain_type, block_type = config.bridge_mappings[event_type]
    data = _extract_event_data(sdk_event)

    # Apply finish_reason_map for TURN_COMPLETE events
    if domain_type == DomainEventType.TURN_COMPLETE and config.finish_reason_map:
        sdk_finish_reason = data.get("finish_reason", "")
        # Map SDK finish_reason to domain finish_reason, with fallback to _default
        mapped_reason = config.finish_reason_map.get(
            sdk_finish_reason,
            config.finish_reason_map.get("_default", sdk_finish_reason),
        )
        data["finish_reason"] = mapped_reason

    return DomainEvent(
        type=domain_type,
        data=data,
        block_type=block_type,
    )
