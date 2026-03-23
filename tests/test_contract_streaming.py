"""
Contract Compliance Tests: Streaming.

Contract: contracts/streaming-contract.md

Tests streaming behavior compliance.
"""

from __future__ import annotations

from amplifier_module_provider_github_copilot.streaming import (
    AccumulatedResponse,
    DomainEvent,
    DomainEventType,
    StreamingAccumulator,
)


class TestStreamingAccumulator:
    """streaming-contract:Accumulation:MUST:1-2"""

    def test_preserves_event_order(self) -> None:
        """streaming-contract:Accumulation:MUST:1 — Deltas accumulated in order."""
        accumulator = StreamingAccumulator()

        # Add text deltas in order
        accumulator.add(
            DomainEvent(
                type=DomainEventType.CONTENT_DELTA, data={"text": "Hello ", "block_type": "text"}
            )
        )
        accumulator.add(
            DomainEvent(
                type=DomainEventType.CONTENT_DELTA, data={"text": "world", "block_type": "text"}
            )
        )

        result = accumulator.get_result()
        assert result.text_content == "Hello world"

    def test_produces_complete_response_on_turn_complete(self) -> None:
        """streaming-contract:Accumulation:MUST:2 — Complete response on TURN_COMPLETE."""
        accumulator = StreamingAccumulator()

        accumulator.add(
            DomainEvent(
                type=DomainEventType.CONTENT_DELTA,
                data={"text": "Test response", "block_type": "text"},
            )
        )
        accumulator.add(
            DomainEvent(type=DomainEventType.TURN_COMPLETE, data={"finish_reason": "STOP"})
        )

        assert accumulator.is_complete
        result = accumulator.get_result()
        assert result.finish_reason == "STOP"

    def test_separates_text_and_thinking_content(self) -> None:
        """streaming-contract:ContentTypes:MUST:1 — Separates text and thinking."""
        accumulator = StreamingAccumulator()

        # block_type is on the DomainEvent itself, not in data
        accumulator.add(
            DomainEvent(
                type=DomainEventType.CONTENT_DELTA,
                data={"text": "Thinking..."},
                block_type="THINKING",
            )
        )
        accumulator.add(
            DomainEvent(
                type=DomainEventType.CONTENT_DELTA,
                data={"text": "Response"},
                block_type="text",
            )
        )

        result = accumulator.get_result()
        assert result.text_content == "Response"
        assert result.thinking_content == "Thinking..."


class TestToolCallCapture:
    """streaming-contract:ToolCapture:MUST:1,2"""

    def test_captures_tool_calls(self) -> None:
        """streaming-contract:ToolCapture:MUST:1 — Tool calls captured."""
        accumulator = StreamingAccumulator()

        accumulator.add(
            DomainEvent(
                type=DomainEventType.TOOL_CALL,
                data={"id": "call_123", "name": "read_file", "arguments": {"path": "test.py"}},
            )
        )

        result = accumulator.get_result()
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["id"] == "call_123"
        assert result.tool_calls[0]["name"] == "read_file"

    def test_tool_calls_in_final_response(self) -> None:
        """streaming-contract:ToolCapture:MUST:2 — Tool calls in final response."""
        accumulator = StreamingAccumulator()

        accumulator.add(
            DomainEvent(
                type=DomainEventType.TOOL_CALL,
                data={"id": "call_1", "name": "tool1", "arguments": {}},
            )
        )
        accumulator.add(
            DomainEvent(
                type=DomainEventType.TOOL_CALL,
                data={"id": "call_2", "name": "tool2", "arguments": {}},
            )
        )
        accumulator.add(
            DomainEvent(type=DomainEventType.TURN_COMPLETE, data={"finish_reason": "TOOL_USE"})
        )

        result = accumulator.get_result()
        assert len(result.tool_calls) == 2
        assert result.finish_reason == "TOOL_USE"


class TestAccumulatedResponse:
    """streaming-contract:Response:MUST:1"""

    def test_accumulated_response_structure(self) -> None:
        """streaming-contract:Response:MUST:1 — Response has expected structure."""
        response = AccumulatedResponse(
            text_content="Hello",
            thinking_content="",
            tool_calls=[],
            finish_reason="STOP",
            usage=None,
        )

        assert hasattr(response, "text_content")
        assert hasattr(response, "thinking_content")
        assert hasattr(response, "tool_calls")
        assert hasattr(response, "finish_reason")

    def test_empty_accumulator_returns_defaults(self) -> None:
        """Empty accumulator returns sensible defaults."""
        accumulator = StreamingAccumulator()
        result = accumulator.get_result()

        assert result.text_content == ""
        assert result.tool_calls == []
