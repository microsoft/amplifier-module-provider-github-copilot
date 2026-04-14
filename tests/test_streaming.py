"""
Tests for event translation / streaming module.

Contract: event-vocabulary.md

Type annotations for pyright strict mode compliance.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest


class TestEventClassification:
    """Tests for classify_event function."""

    def test_text_delta_classified_as_bridge(self):
        """text_delta is a BRIDGE event."""
        from amplifier_module_provider_github_copilot.streaming import (
            EventClassification,
            classify_event,
            load_event_config,
        )

        config = load_event_config()
        result = classify_event("assistant.message_delta", config)
        assert result == EventClassification.BRIDGE

    def test_thinking_delta_classified_as_bridge(self):
        """assistant.reasoning_delta is a BRIDGE event."""
        from amplifier_module_provider_github_copilot.streaming import (
            EventClassification,
            classify_event,
            load_event_config,
        )

        config = load_event_config()
        result = classify_event("assistant.reasoning_delta", config)
        assert result == EventClassification.BRIDGE

    def test_tool_use_start_classified_as_consume(self):
        """tool_use_start is a CONSUME event."""
        from amplifier_module_provider_github_copilot.streaming import (
            EventClassification,
            classify_event,
            load_event_config,
        )

        config = load_event_config()
        result = classify_event("tool_use_start", config)
        assert result == EventClassification.CONSUME

    def test_heartbeat_classified_as_drop(self):
        """heartbeat is a DROP event."""
        from amplifier_module_provider_github_copilot.streaming import (
            EventClassification,
            classify_event,
            load_event_config,
        )

        config = load_event_config()
        result = classify_event("heartbeat", config)
        assert result == EventClassification.DROP

    def test_wildcard_pattern_tool_result(self):
        """tool_result_* pattern matches tool_result_success."""
        from amplifier_module_provider_github_copilot.streaming import (
            EventClassification,
            classify_event,
            load_event_config,
        )

        config = load_event_config()
        result = classify_event("tool_result_success", config)
        assert result == EventClassification.DROP

    def test_wildcard_pattern_debug(self):
        """debug_* pattern matches debug_log."""
        from amplifier_module_provider_github_copilot.streaming import (
            EventClassification,
            classify_event,
            load_event_config,
        )

        config = load_event_config()
        result = classify_event("debug_log", config)
        assert result == EventClassification.DROP

    def test_unknown_event_dropped_with_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Unknown events are dropped with warning."""
        from amplifier_module_provider_github_copilot.streaming import (
            EventClassification,
            classify_event,
            load_event_config,
        )

        config = load_event_config()
        with caplog.at_level(logging.WARNING):
            result = classify_event("completely_unknown_event_xyz", config)
        assert result == EventClassification.DROP
        assert "Unknown SDK event type" in caplog.text

    def test_system_notification_classified_as_consume(self) -> None:
        """system_notification must be classified as CONSUME (not unknown).

        # Contract: event-vocabulary:classification:system_notification:MUST:1
        """
        from amplifier_module_provider_github_copilot.streaming import (
            EventClassification,
            classify_event,
            load_event_config,
        )

        config = load_event_config()
        result = classify_event("system_notification", config)
        assert result == EventClassification.CONSUME, (
            "system_notification should be CONSUME, not unknown"
        )

    def test_system_notification_no_warning_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """system_notification should not produce 'Unknown SDK event type' warning.

        # Contract: event-vocabulary:classification:system_notification:MUST:1
        """
        from amplifier_module_provider_github_copilot.streaming import (
            classify_event,
            load_event_config,
        )

        config = load_event_config()
        with caplog.at_level(logging.WARNING):
            classify_event("system_notification", config)

        assert "Unknown SDK event type: system_notification" not in caplog.text


class TestTranslateEvent:
    """Tests for translate_event function."""

    def test_text_delta_bridges_to_content_delta(self):
        """text_delta SDK event → CONTENT_DELTA domain event.

        Contract: event-vocabulary:Bridge:MUST:1
        """
        from amplifier_module_provider_github_copilot.streaming import (
            DomainEvent,
            DomainEventType,
            load_event_config,
            translate_event,
        )

        config = load_event_config()
        sdk_event = {"type": "assistant.message_delta", "text": "Hello"}
        result = translate_event(sdk_event, config)
        assert isinstance(result, DomainEvent)
        assert result.type == DomainEventType.CONTENT_DELTA
        assert result.block_type == "TEXT"

    def test_thinking_delta_has_thinking_block_type(self):
        """assistant.reasoning_delta → CONTENT_DELTA with block_type=THINKING.

        # Contract: streaming-contract:ThinkingBlock:MUST:1
        """
        from amplifier_module_provider_github_copilot.streaming import (
            DomainEvent,
            DomainEventType,
            load_event_config,
            translate_event,
        )

        config = load_event_config()
        sdk_event = {"type": "assistant.reasoning_delta", "text": "Let me think..."}
        result = translate_event(sdk_event, config)
        assert isinstance(result, DomainEvent)
        assert result.type == DomainEventType.CONTENT_DELTA
        assert result.block_type == "THINKING"

    def test_tool_use_complete_bridges_to_tool_call(self):
        """tool_use_complete → TOOL_CALL domain event.

        # Contract: streaming-contract:ToolCallBlock:MUST:1
        """
        from amplifier_module_provider_github_copilot.streaming import (
            DomainEvent,
            DomainEventType,
            load_event_config,
            translate_event,
        )

        config = load_event_config()
        sdk_event = {"type": "tool_use_complete", "id": "tc1", "name": "read_file"}
        result = translate_event(sdk_event, config)
        assert isinstance(result, DomainEvent)
        assert result.type == DomainEventType.TOOL_CALL

    def test_consume_event_returns_none(self):
        """Contract: event-vocabulary:Consume:MUST:1 — CONSUME events return None."""
        from amplifier_module_provider_github_copilot.streaming import (
            load_event_config,
            translate_event,
        )

        config = load_event_config()
        sdk_event = {"type": "tool_use_start", "id": "tc1"}
        result = translate_event(sdk_event, config)
        assert result is None

    def test_drop_event_returns_none(self):
        """DROP events return None."""
        from amplifier_module_provider_github_copilot.streaming import (
            load_event_config,
            translate_event,
        )

        config = load_event_config()
        sdk_event = {"type": "heartbeat"}
        result = translate_event(sdk_event, config)
        assert result is None

    def test_event_data_preserved(self):
        """Event data is preserved in domain event."""
        from amplifier_module_provider_github_copilot.streaming import (
            DomainEvent,
            load_event_config,
            translate_event,
        )

        config = load_event_config()
        sdk_event = {"type": "assistant.message_delta", "text": "Hello world", "index": 0}
        result = translate_event(sdk_event, config)
        assert isinstance(result, DomainEvent)
        assert result.data["text"] == "Hello world"

    def test_translate_event_flattens_nested_data(self) -> None:
        """translate_event flattens the SDK event's nested data dict into DomainEvent.data.

        The SDK wraps event payload in a ``data`` sub-key. ``_extract_event_data``
        must flatten those fields one level up so consumers can access them as
        ``domain_event.data["delta_content"]`` rather than
        ``domain_event.data["data"]["delta_content"]``.

        This is a regression guard: if the flatten logic is removed or broken,
        downstream streaming consumers silently receive empty data dicts and
        produce no visible exception — only silent content loss.
        """
        from amplifier_module_provider_github_copilot.streaming import (
            DomainEvent,
            load_event_config,
            translate_event,
        )

        config = load_event_config()
        # SDK shape: nested data dict (SessionEventData-style)
        sdk_event = {
            "type": "assistant.message_delta",
            "data": {"delta_content": "Hello from SDK", "index": 1},
        }
        result = translate_event(sdk_event, config)

        assert isinstance(result, DomainEvent)
        # Flattened: fields from data.* must be top-level in result.data
        assert result.data["delta_content"] == "Hello from SDK"
        assert result.data["index"] == 1
        # The nested "data" key itself must NOT appear as a value in result.data
        assert "data" not in result.data


class TestEventConfig:
    """Tests for event config loading."""

    def test_config_loads_successfully(self):
        """Config file loads without errors.

        Contract: event-vocabulary:Bridge:MUST:2
        """
        from amplifier_module_provider_github_copilot.streaming import load_event_config

        config = load_event_config()
        # Contract: event-vocabulary:Bridge:MUST:2
        assert "assistant.message_delta" in config.bridge_mappings

    def test_config_has_bridge_mappings(self):
        """Config contains bridge mappings."""
        from amplifier_module_provider_github_copilot.streaming import load_event_config

        config = load_event_config()
        assert len(config.bridge_mappings) > 0
        assert "assistant.message_delta" in config.bridge_mappings

    def test_config_has_consume_patterns(self):
        """Config contains consume patterns."""
        from amplifier_module_provider_github_copilot.streaming import load_event_config

        config = load_event_config()
        assert len(config.consume_patterns) > 0

    def test_config_has_drop_patterns(self):
        """Config contains drop patterns."""
        from amplifier_module_provider_github_copilot.streaming import load_event_config

        config = load_event_config()
        assert len(config.drop_patterns) > 0


class TestDomainEventType:
    """Tests for DomainEventType enum."""

    def test_all_domain_types_exist(self):
        """All 6 domain event types exist.

        Contract: event-vocabulary:Events:MUST:1
        """
        from amplifier_module_provider_github_copilot.streaming import DomainEventType

        expected_types = {
            "CONTENT_DELTA",
            "TOOL_CALL",
            "USAGE_UPDATE",
            "TURN_COMPLETE",
            "SESSION_IDLE",
            "ERROR",
        }
        # Contract: event-vocabulary:Events:MUST:1
        actual_names = {m.name for m in DomainEventType}
        assert actual_names == expected_types


class TestStreamingAccumulator:
    """Tests for StreamingAccumulator class."""

    def test_accumulator_starts_empty(self):
        """New accumulator has empty state.

        Contract: streaming-contract:Accumulation:MUST:1
        """
        from amplifier_module_provider_github_copilot.streaming import (
            StreamingAccumulator,
        )

        accumulator = StreamingAccumulator()
        result = accumulator.get_result()
        assert result.text_content == ""
        assert result.thinking_content == ""
        assert result.tool_calls == []
        assert not result.is_complete

    def test_content_delta_accumulates_text(self):
        """CONTENT_DELTA events accumulate text."""
        from amplifier_module_provider_github_copilot.streaming import (
            DomainEvent,
            DomainEventType,
            StreamingAccumulator,
        )

        accumulator = StreamingAccumulator()
        accumulator.add(
            DomainEvent(
                type=DomainEventType.CONTENT_DELTA,
                data={"text": "Hello "},
                block_type="TEXT",
            )
        )
        accumulator.add(
            DomainEvent(
                type=DomainEventType.CONTENT_DELTA,
                data={"text": "world"},
                block_type="TEXT",
            )
        )
        result = accumulator.get_result()
        assert result.text_content == "Hello world"

    def test_thinking_delta_accumulates_separately(self):
        """THINKING block_type accumulates to thinking_content."""
        from amplifier_module_provider_github_copilot.streaming import (
            DomainEvent,
            DomainEventType,
            StreamingAccumulator,
        )

        accumulator = StreamingAccumulator()
        accumulator.add(
            DomainEvent(
                type=DomainEventType.CONTENT_DELTA,
                data={"text": "Let me think"},
                block_type="THINKING",
            )
        )
        result = accumulator.get_result()
        assert result.thinking_content == "Let me think"
        assert result.text_content == ""

    def test_tool_call_collected(self):
        """TOOL_CALL events collected in list."""
        from amplifier_module_provider_github_copilot.streaming import (
            DomainEvent,
            DomainEventType,
            StreamingAccumulator,
        )

        accumulator = StreamingAccumulator()
        accumulator.add(
            DomainEvent(
                type=DomainEventType.TOOL_CALL,
                data={"id": "tc1", "name": "read_file", "arguments": {"path": "x.py"}},
            )
        )
        result = accumulator.get_result()
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "read_file"

    def test_usage_update_stored(self):
        """USAGE_UPDATE event stored.

        # Contract: streaming-contract:usage:MUST:1
        """
        from amplifier_module_provider_github_copilot.streaming import (
            DomainEvent,
            DomainEventType,
            StreamingAccumulator,
        )

        accumulator = StreamingAccumulator()
        accumulator.add(
            DomainEvent(
                type=DomainEventType.USAGE_UPDATE,
                data={"input_tokens": 100, "output_tokens": 50},
            )
        )
        result = accumulator.get_result()
        assert isinstance(result.usage, dict)
        assert result.usage["input_tokens"] == 100

    def test_turn_complete_marks_done(self):
        """TURN_COMPLETE marks accumulator complete."""
        from amplifier_module_provider_github_copilot.streaming import (
            DomainEvent,
            DomainEventType,
            StreamingAccumulator,
        )

        accumulator = StreamingAccumulator()
        accumulator.add(
            DomainEvent(
                type=DomainEventType.TURN_COMPLETE,
                data={"finish_reason": "stop"},
            )
        )
        result = accumulator.get_result()
        assert result.is_complete
        assert result.finish_reason == "stop"

    def test_error_marks_complete_with_error(self):
        """ERROR event marks complete with error data.

        # Contract: streaming-contract:completion:MUST:2
        """
        from amplifier_module_provider_github_copilot.streaming import (
            DomainEvent,
            DomainEventType,
            StreamingAccumulator,
        )

        accumulator = StreamingAccumulator()
        accumulator.add(
            DomainEvent(
                type=DomainEventType.ERROR,
                data={"message": "Rate limit exceeded"},
            )
        )
        result = accumulator.get_result()
        assert result.is_complete
        assert isinstance(result.error, dict)
        assert "Rate limit" in result.error["message"]

    def test_interleaved_content_handled(self):
        """Interleaved text and thinking accumulate correctly."""
        from amplifier_module_provider_github_copilot.streaming import (
            DomainEvent,
            DomainEventType,
            StreamingAccumulator,
        )

        accumulator = StreamingAccumulator()
        accumulator.add(DomainEvent(DomainEventType.CONTENT_DELTA, {"text": "A"}, "TEXT"))
        accumulator.add(DomainEvent(DomainEventType.CONTENT_DELTA, {"text": "T"}, "THINKING"))
        accumulator.add(DomainEvent(DomainEventType.CONTENT_DELTA, {"text": "B"}, "TEXT"))
        result = accumulator.get_result()
        assert result.text_content == "AB"
        assert result.thinking_content == "T"

    def test_multiple_tool_calls_collected(self):
        """Multiple TOOL_CALL events all collected."""
        from amplifier_module_provider_github_copilot.streaming import (
            DomainEvent,
            DomainEventType,
            StreamingAccumulator,
        )

        accumulator = StreamingAccumulator()
        accumulator.add(
            DomainEvent(
                type=DomainEventType.TOOL_CALL,
                data={"id": "tc1", "name": "read_file"},
            )
        )
        accumulator.add(
            DomainEvent(
                type=DomainEventType.TOOL_CALL,
                data={"id": "tc2", "name": "write_file"},
            )
        )
        result = accumulator.get_result()
        assert len(result.tool_calls) == 2

    def test_is_complete_property(self):
        """is_complete property reflects accumulator state."""
        from amplifier_module_provider_github_copilot.streaming import (
            DomainEvent,
            DomainEventType,
            StreamingAccumulator,
        )

        accumulator = StreamingAccumulator()
        assert not accumulator.is_complete
        accumulator.add(DomainEvent(type=DomainEventType.TURN_COMPLETE, data={}))
        assert accumulator.is_complete

    def test_content_delta_with_none_block_type_goes_to_text(self):
        """CONTENT_DELTA with None block_type accumulates to text."""
        from amplifier_module_provider_github_copilot.streaming import (
            DomainEvent,
            DomainEventType,
            StreamingAccumulator,
        )

        accumulator = StreamingAccumulator()
        accumulator.add(
            DomainEvent(
                type=DomainEventType.CONTENT_DELTA,
                data={"text": "No block type"},
                block_type=None,
            )
        )
        result = accumulator.get_result()
        assert result.text_content == "No block type"

    def test_text_block_boundaries_preserved_around_tool_call(self) -> None:
        """H-4: streaming-contract:Accumulation:MUST:2 — block boundaries maintained.

        When the SDK emits text → tool_call → text in one turn, the two text
        segments MUST end up in SEPARATE TextBlocks in the final ChatResponse.
        content.  Before the fix, all text is concatenated into one TextBlock,
        losing the boundary that separates pre-tool text from post-tool text.
        """
        from amplifier_core.message_models import TextBlock

        from amplifier_module_provider_github_copilot.streaming import (
            DomainEvent,
            DomainEventType,
            StreamingAccumulator,
        )

        accumulator = StreamingAccumulator()
        # First text block
        accumulator.add(DomainEvent(DomainEventType.CONTENT_DELTA, {"text": "Hello "}, "TEXT"))
        accumulator.add(DomainEvent(DomainEventType.CONTENT_DELTA, {"text": "world"}, "TEXT"))
        # Tool call event — signals boundary between text blocks
        accumulator.add(
            DomainEvent(
                type=DomainEventType.TOOL_CALL,
                data={"id": "tc1", "name": "read_file", "arguments": {}},
            )
        )
        # Second text block (post-tool commentary)
        accumulator.add(DomainEvent(DomainEventType.CONTENT_DELTA, {"text": "Done."}, "TEXT"))
        accumulator.add(DomainEvent(DomainEventType.TURN_COMPLETE, {}))

        response = accumulator.to_chat_response()

        # Count TextBlock instances in content
        text_blocks = [b for b in response.content if isinstance(b, TextBlock)]

        # MUST have two separate TextBlocks — one before tool call, one after
        assert len(text_blocks) == 2, (
            f"Expected 2 TextBlocks (pre-tool and post-tool), got {len(text_blocks)}. "
            f"Content: {response.content}"
        )
        assert text_blocks[0].text == "Hello world"
        assert text_blocks[1].text == "Done."

    def test_content_block_order_preserved_text_then_thinking(self) -> None:
        """Contract: streaming-contract:StreamingResponse:MUST:2 — content block order
        MUST match the order in which blocks arrived from the SDK.

        P1-3: Extended thinking models (Anthropic) may send thinking BEFORE text.
        The accumulator was emitting all text blocks first, then thinking — regardless
        of actual arrival order. This breaks multi-turn context reconstruction for
        models that interleave reasoning and output.

        Sequence tested: TEXT → THINKING → TEXT
        Expected order:  TextBlock("A") → ThinkingBlock("T") → TextBlock("B")
        Wrong order:     TextBlock("A") → TextBlock("B") → ThinkingBlock("T")
        """
        from amplifier_core.message_models import TextBlock, ThinkingBlock

        from amplifier_module_provider_github_copilot.streaming import (
            DomainEvent,
            DomainEventType,
            StreamingAccumulator,
        )

        acc = StreamingAccumulator()
        acc.add(DomainEvent(DomainEventType.CONTENT_DELTA, {"text": "A"}, "TEXT"))
        acc.add(DomainEvent(DomainEventType.CONTENT_DELTA, {"text": "T"}, "THINKING"))
        acc.add(DomainEvent(DomainEventType.CONTENT_DELTA, {"text": "B"}, "TEXT"))
        acc.add(DomainEvent(DomainEventType.TURN_COMPLETE, {}))

        response = acc.to_chat_response()

        # Verify content block order matches arrival order: TEXT→THINKING→TEXT
        assert len(response.content) == 3
        assert isinstance(response.content[0], TextBlock)
        assert isinstance(response.content[1], ThinkingBlock)
        assert isinstance(response.content[2], TextBlock)
        text_blocks = [b for b in response.content if isinstance(b, TextBlock)]
        thinking_blocks = [b for b in response.content if isinstance(b, ThinkingBlock)]
        assert text_blocks[0].text == "A"
        assert thinking_blocks[0].thinking == "T"
        assert text_blocks[1].text == "B"

    def test_content_block_order_thinking_first(self) -> None:
        """Contract: streaming-contract:StreamingResponse:MUST:2 — thinking-first order.

        Some Anthropic models (extended thinking) send THINKING before TEXT.
        Sequence: THINKING → TEXT
        Expected: ThinkingBlock → TextBlock
        """
        from amplifier_core.message_models import TextBlock, ThinkingBlock

        from amplifier_module_provider_github_copilot.streaming import (
            DomainEvent,
            DomainEventType,
            StreamingAccumulator,
        )

        acc = StreamingAccumulator()
        acc.add(DomainEvent(DomainEventType.CONTENT_DELTA, {"text": "My reasoning"}, "THINKING"))
        acc.add(DomainEvent(DomainEventType.CONTENT_DELTA, {"text": "My answer"}, "TEXT"))
        acc.add(DomainEvent(DomainEventType.TURN_COMPLETE, {}))

        response = acc.to_chat_response()

        # Verify thinking-first order preserved
        assert len(response.content) == 2
        assert isinstance(response.content[0], ThinkingBlock)
        assert isinstance(response.content[1], TextBlock)


class TestAccumulatedResponse:
    """Tests for AccumulatedResponse dataclass."""

    def test_accumulated_response_defaults(self):
        """AccumulatedResponse has correct defaults."""
        from amplifier_module_provider_github_copilot.streaming import (
            AccumulatedResponse,
        )

        response = AccumulatedResponse()
        assert response.text_content == ""
        assert response.thinking_content == ""
        assert response.tool_calls == []
        assert response.usage is None
        assert response.finish_reason is None
        assert response.error is None
        assert not response.is_complete


# ============================================================================
# _extract_content_block fix tests
# Contract: provider-protocol:complete:MUST:5
# ============================================================================


class TestExtractContentBlockSkipsToolCalls:
    """Tests Fix 1: _extract_content_block skips tool_call blocks.

    Contract: provider-protocol:complete:MUST:5
    This ensures prior tool calls in conversation history don't get
    serialized as '[Tool Call: ...]' text that triggers fake detection.
    """

    def test_extract_content_block_skips_tool_call_dict(self) -> None:
        """Dict with type=tool_call returns empty string.

        Contract: provider-protocol:complete:MUST:5
        """
        from amplifier_module_provider_github_copilot.provider import (
            _extract_content_block,  # pyright: ignore[reportPrivateUsage]
        )

        block = {"type": "tool_call", "tool_name": "bash", "arguments": {"cmd": "ls"}}
        result = _extract_content_block(block)
        assert result == ""

    def test_extract_content_block_skips_tool_call_object(self) -> None:
        """Object with tool_name attribute returns empty string.

        Contract: provider-protocol:complete:MUST:5
        """
        from dataclasses import dataclass

        from amplifier_module_provider_github_copilot.provider import (
            _extract_content_block,  # pyright: ignore[reportPrivateUsage]
        )

        @dataclass
        class ToolCallBlock:
            type: str = "tool_call"
            tool_name: str = "read_file"
            arguments: dict[str, Any] | None = None

            def __post_init__(self) -> None:
                if self.arguments is None:
                    self.arguments = {}

        block = ToolCallBlock(tool_name="read_file", arguments={"path": "test.py"})
        result = _extract_content_block(block)
        assert result == ""

    def test_extract_content_block_preserves_text(self) -> None:
        """Text blocks are still extracted correctly.

        Contract: provider-protocol:complete:MUST:5
        """
        from amplifier_module_provider_github_copilot.provider import (
            _extract_content_block,  # pyright: ignore[reportPrivateUsage]
        )

        block = {"type": "text", "text": "Hello, world!"}
        result = _extract_content_block(block)
        assert result == "Hello, world!"

    def test_extract_content_block_preserves_thinking(self) -> None:
        """Thinking blocks are still extracted correctly.

        Contract: provider-protocol:complete:MUST:5
        """
        from amplifier_module_provider_github_copilot.provider import (
            _extract_content_block,  # pyright: ignore[reportPrivateUsage]
        )

        block = {"type": "thinking", "thinking": "Let me analyze..."}
        result = _extract_content_block(block)
        assert result == "[Thinking: Let me analyze...]"

    def test_no_tool_call_text_in_serialized_output(self) -> None:
        """Conversation with tool calls doesn't produce fake tool call text.

        Contract: provider-protocol:complete:MUST:5

        This is the integration test that proves Fix 1 works end-to-end.
        """
        from amplifier_module_provider_github_copilot.provider import (
            _extract_message_content,  # pyright: ignore[reportPrivateUsage]
        )

        # Mixed content with text and tool_call
        content = [
            {"type": "text", "text": "I'll run a command"},
            {"type": "tool_call", "tool_name": "bash", "arguments": {"cmd": "ls"}},
            {"type": "text", "text": "Command completed"},
        ]

        result = _extract_message_content(content)

        # Should NOT contain [Tool Call: pattern
        assert "[Tool Call:" not in result
        # Should preserve text
        assert "I'll run a command" in result
        assert "Command completed" in result


# ============================================================================
# StreamingChatResponse tests
# Contract: streaming-contract:StreamingResponse:MUST:1-4
# ============================================================================


class TestStreamingChatResponse:
    """Tests for StreamingChatResponse class.

    Contract: streaming-contract:StreamingResponse:MUST:1-4
    """

    def test_streaming_response_extends_chat_response(self) -> None:
        """StreamingChatResponse is a subclass of ChatResponse.

        Contract: streaming-contract:StreamingResponse:MUST:1
        """
        from amplifier_core import ChatResponse

        from amplifier_module_provider_github_copilot.streaming import (
            StreamingChatResponse,
        )

        assert issubclass(StreamingChatResponse, ChatResponse), (
            "StreamingChatResponse must extend ChatResponse"
        )

    def test_streaming_response_instance_is_chat_response(self) -> None:
        """StreamingChatResponse instance passes ChatResponse isinstance check.

        Contract: streaming-contract:StreamingResponse:MUST:1
        """
        from amplifier_core import ChatResponse

        from amplifier_module_provider_github_copilot.streaming import (
            StreamingChatResponse,
        )

        response = StreamingChatResponse(content=[])
        assert isinstance(response, ChatResponse), (
            "StreamingChatResponse instance must be ChatResponse instance"
        )

    def test_content_blocks_populated_with_text(self) -> None:
        """content_blocks contains TextContent when text accumulated.

        Contract: streaming-contract:StreamingResponse:MUST:2
        """
        from amplifier_core import TextContent

        from amplifier_module_provider_github_copilot.streaming import (
            DomainEvent,
            DomainEventType,
            StreamingAccumulator,
        )

        accumulator = StreamingAccumulator()
        accumulator.add(
            DomainEvent(
                type=DomainEventType.CONTENT_DELTA,
                data={"text": "Hello world"},
                block_type="TEXT",
            )
        )
        accumulator.add(
            DomainEvent(type=DomainEventType.TURN_COMPLETE, data={"finish_reason": "stop"})
        )

        response = accumulator.to_chat_response()

        assert isinstance(response.content_blocks, list)
        assert len(response.content_blocks) == 1, "Should have one content block"
        assert isinstance(response.content_blocks[0], TextContent)
        assert response.content_blocks[0].text == "Hello world"

    def test_content_blocks_populated_with_thinking(self) -> None:
        """content_blocks contains ThinkingContent when thinking accumulated.

        Contract: streaming-contract:StreamingResponse:MUST:2
        """
        from amplifier_core import ThinkingContent

        from amplifier_module_provider_github_copilot.streaming import (
            DomainEvent,
            DomainEventType,
            StreamingAccumulator,
        )

        accumulator = StreamingAccumulator()
        accumulator.add(
            DomainEvent(
                type=DomainEventType.CONTENT_DELTA,
                data={"text": "Let me think..."},
                block_type="THINKING",
            )
        )
        accumulator.add(
            DomainEvent(type=DomainEventType.TURN_COMPLETE, data={"finish_reason": "stop"})
        )

        response = accumulator.to_chat_response()

        assert isinstance(response.content_blocks, list)
        assert len(response.content_blocks) == 1, "Should have one content block"
        assert isinstance(response.content_blocks[0], ThinkingContent)
        assert response.content_blocks[0].text == "Let me think..."

    def test_content_blocks_populated_with_tool_calls(self) -> None:
        """content_blocks contains ToolCallContent when tools captured.

        Contract: streaming-contract:StreamingResponse:MUST:2
        """
        from amplifier_core import ToolCallContent

        from amplifier_module_provider_github_copilot.streaming import (
            DomainEvent,
            DomainEventType,
            StreamingAccumulator,
        )

        accumulator = StreamingAccumulator()
        accumulator.add(
            DomainEvent(
                type=DomainEventType.TOOL_CALL,
                data={"id": "tc1", "name": "read_file", "arguments": {"path": "test.py"}},
            )
        )
        accumulator.add(
            DomainEvent(type=DomainEventType.TURN_COMPLETE, data={"finish_reason": "tool_use"})
        )

        response = accumulator.to_chat_response()

        assert isinstance(response.content_blocks, list)
        assert len(response.content_blocks) == 1, "Should have one content block"
        assert isinstance(response.content_blocks[0], ToolCallContent)
        assert response.content_blocks[0].name == "read_file"

    def test_text_field_convenience(self) -> None:
        """text field contains combined text content.

        Contract: streaming-contract:StreamingResponse:MUST:2
        """
        from amplifier_module_provider_github_copilot.streaming import (
            DomainEvent,
            DomainEventType,
            StreamingAccumulator,
        )

        accumulator = StreamingAccumulator()
        accumulator.add(
            DomainEvent(
                type=DomainEventType.CONTENT_DELTA,
                data={"text": "Hello "},
                block_type="TEXT",
            )
        )
        accumulator.add(
            DomainEvent(
                type=DomainEventType.CONTENT_DELTA,
                data={"text": "world"},
                block_type="TEXT",
            )
        )
        accumulator.add(
            DomainEvent(type=DomainEventType.TURN_COMPLETE, data={"finish_reason": "stop"})
        )

        response = accumulator.to_chat_response()

        assert response.text == "Hello world", f"Expected 'Hello world', got '{response.text}'"

    def test_content_blocks_none_when_empty(self) -> None:
        """content_blocks is None when no content (not empty list).

        Contract: streaming-contract:StreamingResponse:MUST:4
        """
        from amplifier_module_provider_github_copilot.streaming import (
            DomainEvent,
            DomainEventType,
            StreamingAccumulator,
        )

        accumulator = StreamingAccumulator()
        accumulator.add(
            DomainEvent(type=DomainEventType.TURN_COMPLETE, data={"finish_reason": "stop"})
        )

        response = accumulator.to_chat_response()

        assert response.content_blocks is None, (
            f"content_blocks should be None when empty, got {response.content_blocks}"
        )

    def test_mixed_content_blocks(self) -> None:
        """content_blocks contains all types when mixed content accumulated.

        Contract: streaming-contract:StreamingResponse:MUST:2
        """
        from amplifier_core import TextContent, ThinkingContent, ToolCallContent

        from amplifier_module_provider_github_copilot.streaming import (
            DomainEvent,
            DomainEventType,
            StreamingAccumulator,
        )

        accumulator = StreamingAccumulator()
        accumulator.add(
            DomainEvent(
                type=DomainEventType.CONTENT_DELTA,
                data={"text": "Response text"},
                block_type="TEXT",
            )
        )
        accumulator.add(
            DomainEvent(
                type=DomainEventType.CONTENT_DELTA,
                data={"text": "Thinking..."},
                block_type="THINKING",
            )
        )
        accumulator.add(
            DomainEvent(
                type=DomainEventType.TOOL_CALL,
                data={"id": "tc1", "name": "bash", "arguments": {"cmd": "ls"}},
            )
        )
        accumulator.add(
            DomainEvent(type=DomainEventType.TURN_COMPLETE, data={"finish_reason": "tool_use"})
        )

        response = accumulator.to_chat_response()

        assert isinstance(response.content_blocks, list)
        assert len(response.content_blocks) == 3

        # Check types using isinstance
        assert any(isinstance(b, TextContent) for b in response.content_blocks)
        assert any(isinstance(b, ThinkingContent) for b in response.content_blocks)
        assert any(isinstance(b, ToolCallContent) for b in response.content_blocks)

    def test_content_list_contains_tool_call_block_for_kernel_context(self) -> None:
        """Contract: streaming-contract:StreamingResponse:MUST:2 — ChatResponse.content
        MUST contain ToolCallBlock (not ToolCall) for kernel context persistence.

        ChatResponse.content is typed as list[ContentBlockUnion] (message_models.py:266).
        ContentBlockUnion is a discriminated union on the `type` field. ToolCallBlock
        (type="tool_call", field=`input: dict`) is a member. ToolCall (field=`arguments`)
        is NOT in ContentBlockUnion and belongs only in ChatResponse.tool_calls.

        This test was RED before the fix: streaming.py used the wrong type (ToolCall)
        in content, which cannot satisfy list[ContentBlockUnion].

        Evidence:
        - amplifier_core/message_models.py:61  — ToolCallBlock(type, id, name, input)
        - amplifier_core/message_models.py:215 — ToolCall(id, name, arguments) — NOT in union
        - amplifier_core/message_models.py:266 — ChatResponse.content: list[ContentBlockUnion]
        """
        from amplifier_core.message_models import ToolCallBlock

        from amplifier_module_provider_github_copilot.streaming import (
            DomainEvent,
            DomainEventType,
            StreamingAccumulator,
        )

        accumulator = StreamingAccumulator()
        accumulator.add(
            DomainEvent(
                type=DomainEventType.TOOL_CALL,
                data={"id": "tc1", "name": "read_file", "arguments": {"path": "test.py"}},
            )
        )
        accumulator.add(
            DomainEvent(type=DomainEventType.TURN_COMPLETE, data={"finish_reason": "tool_use"})
        )

        response = accumulator.to_chat_response()

        # content MUST contain a ToolCallBlock instance (ContentBlockUnion member)
        tool_call_blocks = [b for b in response.content if isinstance(b, ToolCallBlock)]
        assert len(tool_call_blocks) == 1
        tcb = tool_call_blocks[0]
        assert tcb.type == "tool_call"
        assert tcb.id == "tc1"
        assert tcb.name == "read_file"
        assert tcb.input == {"path": "test.py"}

    def test_tool_calls_field_uses_tool_call_type_not_tool_call_block(self) -> None:
        """Contract: streaming-contract:StreamingResponse:MUST:2 — tool_calls field
        MUST contain ToolCall objects (with `arguments` field), not ToolCallBlock.

        ChatResponse.tool_calls: list[ToolCall] | None — this is the kernel's
        'execute this tool' signal. ToolCall.arguments maps to SDK tool request arguments.
        """
        from amplifier_core.message_models import ToolCall

        from amplifier_module_provider_github_copilot.streaming import (
            DomainEvent,
            DomainEventType,
            StreamingAccumulator,
        )

        accumulator = StreamingAccumulator()
        accumulator.add(
            DomainEvent(
                type=DomainEventType.TOOL_CALL,
                data={"id": "tc1", "name": "read_file", "arguments": {"path": "test.py"}},
            )
        )
        accumulator.add(
            DomainEvent(type=DomainEventType.TURN_COMPLETE, data={"finish_reason": "tool_use"})
        )

        response = accumulator.to_chat_response()

        assert isinstance(response.tool_calls, list)
        assert len(response.tool_calls) == 1
        tc = response.tool_calls[0]
        assert isinstance(tc, ToolCall)
        assert tc.arguments == {"path": "test.py"}


# ============================================================================
# Contract: streaming-contract:ProgressiveStreaming:SHOULD:1-4
# ============================================================================


class TestProgressiveStreamingEmission:
    """Tests for progressive streaming event emission.

    Contract: streaming-contract:ProgressiveStreaming:SHOULD:1-4

    These tests verify that the provider emits llm:content_block events
    for real-time UI updates as content arrives from the SDK.
    """

    def test_emit_streaming_content_fires_hook(self) -> None:
        """streaming-contract:ProgressiveStreaming:SHOULD:1.

        Provider SHOULD emit llm:content_block events when coordinator has hooks.
        """
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from amplifier_core import ModuleCoordinator, TextContent

        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        # Create provider with mock coordinator
        coordinator = MagicMock(spec=ModuleCoordinator)
        coordinator.hooks = MagicMock()
        coordinator.hooks.emit = AsyncMock()

        provider = GitHubCopilotProvider(
            config={},
            coordinator=coordinator,
        )

        content = TextContent(text="Hello")

        # Must run in event loop for fire-and-forget task creation
        async def run_test() -> None:
            provider._emit_streaming_content(content)  # pyright: ignore[reportPrivateUsage]
            # Give task time to run
            await asyncio.sleep(0.01)

        asyncio.run(run_test())

        # Verify hook was called with llm:content_block
        coordinator.hooks.emit.assert_called()
        call_args = coordinator.hooks.emit.call_args
        assert call_args[0][0] == "llm:content_block"
        assert call_args[0][1]["provider"] == "github-copilot"
        # Content is serialized to JSON-compatible dict (enums converted to values)
        content_data = call_args[0][1]["content"]
        assert content_data["text"] == content.text
        assert content_data["type"] == "text"  # Enum value, not ContentBlockType.TEXT

    def test_emit_streaming_content_skips_without_coordinator(self) -> None:
        """streaming-contract:ProgressiveStreaming:SHOULD:4.

        Provider SHOULD gracefully skip emission when no coordinator.
        """
        from amplifier_core import TextContent

        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        # Create provider without coordinator
        provider = GitHubCopilotProvider(
            config={},
            coordinator=None,
        )

        content = TextContent(text="Hello")

        # Should not raise
        provider._emit_streaming_content(content)  # pyright: ignore[reportPrivateUsage]
        assert provider._pending_emit_tasks == set()  # pyright: ignore[reportPrivateUsage]

    def test_emit_streaming_content_skips_without_hooks(self) -> None:
        """streaming-contract:ProgressiveStreaming:SHOULD:4.

        Provider SHOULD gracefully skip emission when coordinator has no hooks.
        """
        from unittest.mock import MagicMock

        from amplifier_core import TextContent

        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        # Create provider with coordinator but no hooks attribute
        coordinator = MagicMock(spec=[])  # No hooks attribute

        provider = GitHubCopilotProvider(
            config={},
            coordinator=coordinator,
        )

        content = TextContent(text="Hello")

        # Should not raise
        provider._emit_streaming_content(content)  # pyright: ignore[reportPrivateUsage]
        assert provider._pending_emit_tasks == set()  # pyright: ignore[reportPrivateUsage]

    def test_emit_streaming_content_handles_emit_error(self) -> None:
        """streaming-contract:ProgressiveStreaming:SHOULD:2.

        Async emit should handle errors gracefully without blocking.
        """
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from amplifier_core import ModuleCoordinator, TextContent

        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        # Create provider with mock coordinator that raises
        coordinator = MagicMock(spec=ModuleCoordinator)
        coordinator.hooks = MagicMock()
        coordinator.hooks.emit = AsyncMock(side_effect=Exception("Hook broke"))

        provider = GitHubCopilotProvider(
            config={},
            coordinator=coordinator,
        )

        content = TextContent(text="Hello")

        async def run_test() -> None:
            # Should not raise even when hook fails
            provider._emit_streaming_content(content)  # pyright: ignore[reportPrivateUsage]
            await asyncio.sleep(0.01)

        # Should not raise
        asyncio.run(run_test())
        assert coordinator.hooks.emit.await_count == 1
        assert provider._pending_emit_tasks == set()  # pyright: ignore[reportPrivateUsage]

    def test_pending_emit_tasks_tracked(self) -> None:
        """streaming-contract:ProgressiveStreaming:SHOULD:3.

        Provider SHOULD track pending emit tasks for cleanup.
        """
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from amplifier_core import ModuleCoordinator, TextContent

        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        coordinator = MagicMock(spec=ModuleCoordinator)
        coordinator.hooks = MagicMock()

        # Make emit wait so task stays pending
        async def _slow_emit(*a: object, **kw: object) -> None:
            await asyncio.sleep(0.1)

        coordinator.hooks.emit = AsyncMock(side_effect=_slow_emit)

        provider = GitHubCopilotProvider(
            config={},
            coordinator=coordinator,
        )

        content = TextContent(text="Hello")

        async def run_test() -> None:
            provider._emit_streaming_content(content)  # pyright: ignore[reportPrivateUsage]
            # Task should be tracked
            assert len(provider._pending_emit_tasks) == 1  # pyright: ignore[reportPrivateUsage]

        asyncio.run(run_test())


# ============================================================================
# Fail-Fast Guard Tests (P2-10)
# Contract: streaming-contract:SessionLifecycle:MUST:1
# ============================================================================


class TestConfigurationFailFast:
    """Test fail-fast behavior for missing critical config.

    P2-10: Add missing module tests.
    Contract: streaming-contract:SessionLifecycle:MUST:1
    """

    def test_config_has_required_idle_events(self) -> None:
        """Config MUST have idle_events for session completion detection.

        Contract: streaming-contract:SessionLifecycle:MUST:1
        If idle_events is empty, provider cannot detect session completion.
        """
        from amplifier_module_provider_github_copilot.streaming import load_event_config

        config = load_event_config()

        # idle_events MUST be populated
        assert config.idle_event_types, (
            "idle_events must be populated for session completion detection"
        )
        # Should contain expected idle event types
        assert any("idle" in t.lower() for t in config.idle_event_types), (
            "idle_events should contain an idle-like event type"
        )

    def test_config_has_required_error_events(self) -> None:
        """Config MUST have error_events for error detection.

        Contract: streaming-contract:SessionLifecycle:MUST:1
        """
        from amplifier_module_provider_github_copilot.streaming import load_event_config

        config = load_event_config()

        # error_events MUST be populated
        assert config.error_event_types, "error_events must be populated for error detection"

    def test_valid_config_loads_successfully(self) -> None:
        """Valid config loads without error.

        Baseline test to ensure fail-fast doesn't break normal operation.
        """
        from amplifier_module_provider_github_copilot.streaming import load_event_config

        # Should not raise
        config = load_event_config()

        # Config should have required fields populated
        assert config.idle_event_types, "idle_events should be populated"
        assert "session.idle" in config.bridge_mappings


class TestParseToolArguments:
    """Tests for S4 fix: _parse_tool_arguments handles str|dict|other SDK argument variance.

    Contract: streaming-contract:ToolCallBlock:MUST:1
    """

    def test_dict_args_returned_as_is(self) -> None:
        """dict arguments are returned unchanged (no-op, regression-safe).

        Contract: streaming-contract:ToolCallBlock:MUST:1
        """
        from amplifier_module_provider_github_copilot.streaming import _parse_tool_arguments

        args = {"key": "value", "count": 42}
        result = _parse_tool_arguments(args)
        assert result == args

    def test_json_string_args_parsed(self) -> None:
        """JSON string arguments are parsed to dict."""
        from amplifier_module_provider_github_copilot.streaming import _parse_tool_arguments

        result = _parse_tool_arguments('{"file": "main.py", "line": 10}')
        assert result == {"file": "main.py", "line": 10}

    def test_malformed_json_returns_empty_dict(self) -> None:
        """Malformed JSON string returns {} without raising."""
        from amplifier_module_provider_github_copilot.streaming import _parse_tool_arguments

        result = _parse_tool_arguments("not json at all {{")
        assert result == {}

    def test_none_returns_empty_dict(self) -> None:
        """None (missing arguments) returns {}."""
        from amplifier_module_provider_github_copilot.streaming import _parse_tool_arguments

        result = _parse_tool_arguments(None)
        assert result == {}

    def test_json_array_returns_empty_dict(self) -> None:
        """JSON array (not a dict) returns {} — non-dict parsed values are malformed."""
        from amplifier_module_provider_github_copilot.streaming import _parse_tool_arguments

        result = _parse_tool_arguments("[1, 2, 3]")
        assert result == {}

    def test_empty_dict_args_returned(self) -> None:
        """Empty dict is a valid (no-arg) tool call — returned as-is."""
        from amplifier_module_provider_github_copilot.streaming import _parse_tool_arguments

        result = _parse_tool_arguments({})
        assert result == {}

    def test_nested_dict_preserved(self) -> None:
        """Nested dict values are preserved when input is a dict."""
        from amplifier_module_provider_github_copilot.streaming import _parse_tool_arguments

        args = {"query": "test", "options": {"limit": 10, "offset": 0}}
        result = _parse_tool_arguments(args)
        assert result == args

    # -----------------------------------------------------------------------
    # Integration tests: JSON-string arguments through the full accumulator →
    # to_chat_response() pipeline.  These tests exercise the CALL SITE in
    # StreamingAccumulator.to_chat_response(), NOT just the helper in isolation.
    # They MUST fail if the call site still uses the old isinstance guard.
    # Contract: streaming-contract:ToolCallBlock:MUST:1
    # -----------------------------------------------------------------------

    def test_json_string_args_parsed_end_to_end(self) -> None:
        """JSON-string arguments are parsed through the full accumulator pipeline.

        S4 Fix integration test: proves the call site in to_chat_response() uses
        _parse_tool_arguments(), not the old isinstance guard that silently returned {}.

        The SDK sends arguments as a JSON string. This test feeds that exact shape
        through StreamingAccumulator.add() → to_chat_response() and asserts the
        parsed dict appears in BOTH tool_calls AND content (ToolCallBlock.input).

        Contract: streaming-contract:ToolCallBlock:MUST:1
        """
        from amplifier_core import ToolCall, ToolCallBlock

        from amplifier_module_provider_github_copilot.streaming import (
            DomainEvent,
            DomainEventType,
            StreamingAccumulator,
        )

        accumulator = StreamingAccumulator()
        # SDK sends arguments as a JSON string — the shape that triggered the bug.
        accumulator.add(
            DomainEvent(
                type=DomainEventType.TOOL_CALL,
                data={
                    "id": "tc-integration-01",
                    "name": "read_file",
                    "arguments": '{"path": "src/main.py", "encoding": "utf-8"}',
                },
            )
        )
        accumulator.add(
            DomainEvent(type=DomainEventType.TURN_COMPLETE, data={"finish_reason": "tool_use"})
        )

        response = accumulator.to_chat_response()

        expected_args = {"path": "src/main.py", "encoding": "utf-8"}

        # Assert tool_calls list — ToolCall.arguments must be the parsed dict.
        assert isinstance(response.tool_calls, list)
        assert len(response.tool_calls) == 1, "Expected exactly 1 tool call"
        tc = response.tool_calls[0]
        assert isinstance(tc, ToolCall)
        assert tc.arguments == expected_args

        # Assert content list — ToolCallBlock.input must also be the parsed dict.
        assert response.content, "content must not be empty"
        tcb = response.content[0]
        assert isinstance(tcb, ToolCallBlock)
        assert tcb.input == expected_args

    def test_json_string_args_malformed_end_to_end(self) -> None:
        """Malformed JSON-string arguments produce {} through the full pipeline.

        When the SDK sends a malformed JSON string, the accumulator must NOT raise.
        It must fall back to {} and log a warning, preserving the tool call with
        empty arguments rather than crashing the entire request.

        Contract: streaming-contract:ToolCallBlock:MUST:1
        """
        from amplifier_core import ToolCall

        from amplifier_module_provider_github_copilot.streaming import (
            DomainEvent,
            DomainEventType,
            StreamingAccumulator,
        )

        accumulator = StreamingAccumulator()
        accumulator.add(
            DomainEvent(
                type=DomainEventType.TOOL_CALL,
                data={
                    "id": "tc-malformed-01",
                    "name": "bash",
                    "arguments": "not valid json {{{{",
                },
            )
        )
        accumulator.add(
            DomainEvent(type=DomainEventType.TURN_COMPLETE, data={"finish_reason": "tool_use"})
        )

        # Must not raise, even with malformed JSON
        response = accumulator.to_chat_response()

        assert isinstance(response.tool_calls, list)
        assert len(response.tool_calls) == 1
        tc = response.tool_calls[0]
        assert isinstance(tc, ToolCall)
        assert tc.arguments == {}
