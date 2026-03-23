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


class TestTranslateEvent:
    """Tests for translate_event function."""

    def test_text_delta_bridges_to_content_delta(self):
        """text_delta SDK event → CONTENT_DELTA domain event."""
        from amplifier_module_provider_github_copilot.streaming import (
            DomainEventType,
            load_event_config,
            translate_event,
        )

        config = load_event_config()
        sdk_event = {"type": "assistant.message_delta", "text": "Hello"}
        result = translate_event(sdk_event, config)
        assert result is not None
        assert result.type == DomainEventType.CONTENT_DELTA
        assert result.block_type == "TEXT"

    def test_thinking_delta_has_thinking_block_type(self):
        """assistant.reasoning_delta → CONTENT_DELTA with block_type=THINKING."""
        from amplifier_module_provider_github_copilot.streaming import (
            DomainEventType,
            load_event_config,
            translate_event,
        )

        config = load_event_config()
        sdk_event = {"type": "assistant.reasoning_delta", "text": "Let me think..."}
        result = translate_event(sdk_event, config)
        assert result is not None
        assert result.type == DomainEventType.CONTENT_DELTA
        assert result.block_type == "THINKING"

    def test_tool_use_complete_bridges_to_tool_call(self):
        """tool_use_complete → TOOL_CALL domain event."""
        from amplifier_module_provider_github_copilot.streaming import (
            DomainEventType,
            load_event_config,
            translate_event,
        )

        config = load_event_config()
        sdk_event = {"type": "tool_use_complete", "id": "tc1", "name": "read_file"}
        result = translate_event(sdk_event, config)
        assert result is not None
        assert result.type == DomainEventType.TOOL_CALL

    def test_consume_event_returns_none(self):
        """CONSUME events return None."""
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
            load_event_config,
            translate_event,
        )

        config = load_event_config()
        sdk_event = {"type": "assistant.message_delta", "text": "Hello world", "index": 0}
        result = translate_event(sdk_event, config)
        assert result is not None
        assert result.data["text"] == "Hello world"


class TestEventConfig:
    """Tests for event config loading."""

    def test_config_loads_successfully(self):
        """Config file loads without errors."""
        from amplifier_module_provider_github_copilot.streaming import load_event_config

        config = load_event_config()
        assert config is not None

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
        """All 6 domain event types exist."""
        from amplifier_module_provider_github_copilot.streaming import DomainEventType

        expected_types = [
            "CONTENT_DELTA",
            "TOOL_CALL",
            "USAGE_UPDATE",
            "TURN_COMPLETE",
            "SESSION_IDLE",
            "ERROR",
        ]
        for type_name in expected_types:
            assert hasattr(DomainEventType, type_name)


class TestStreamingAccumulator:
    """Tests for StreamingAccumulator class."""

    def test_accumulator_starts_empty(self):
        """New accumulator has empty state."""
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
        """USAGE_UPDATE event stored."""
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
        assert result.usage is not None
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
        """ERROR event marks complete with error data."""
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
        assert result.error is not None
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
