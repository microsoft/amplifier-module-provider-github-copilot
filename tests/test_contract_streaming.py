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


class TestFinishReasonNormalization:
    """streaming-contract:FinishReason:MUST:5"""

    def test_finish_reason_tool_calls_when_tools_captured(self) -> None:
        """streaming-contract:FinishReason:MUST:5.

        finish_reason="tool_calls" when tool_calls present.

        In abort-on-capture flow, TURN_COMPLETE may not arrive before session termination.
        The provider MUST set finish_reason="tool_calls" when tool_calls is non-empty,
        regardless of whether TURN_COMPLETE event was received.

        Note: amplifier-core proto defines valid values as:
              "stop", "tool_calls", "length", "content_filter"

        This ensures the orchestrator continues the agent loop instead of dropping
        to interactive mode.
        """
        accumulator = StreamingAccumulator()

        # Add tool calls WITHOUT TURN_COMPLETE (simulates abort-on-capture)
        accumulator.add(
            DomainEvent(
                type=DomainEventType.TOOL_CALL,
                data={"id": "call_1", "name": "delegate", "arguments": {"target": "subagent"}},
            )
        )

        # Convert to ChatResponse - finish_reason should be normalized
        response = accumulator.to_chat_response()

        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        # CRITICAL: finish_reason must be "tool_calls" even without TURN_COMPLETE
        assert response.finish_reason == "tool_calls", (
            f"Expected finish_reason='tool_calls' for tool calls, got '{response.finish_reason}'. "
            "This causes the orchestrator to drop to interactive mode prematurely."
        )

    def test_finish_reason_end_turn_when_no_tools(self) -> None:
        """streaming-contract:FinishReason:MUST:5.

        finish_reason="end_turn" when no tool_calls.

        When no tools are captured and no TURN_COMPLETE arrives (edge case),
        finish_reason should default to "end_turn" for text-only responses.
        """
        accumulator = StreamingAccumulator()

        # Add text content only - no tools
        accumulator.add(
            DomainEvent(
                type=DomainEventType.CONTENT_DELTA,
                data={"text": "Hello, world!"},
                block_type="text",
            )
        )

        response = accumulator.to_chat_response()

        assert response.tool_calls is None or len(response.tool_calls) == 0
        # Should default to end_turn when no tools
        assert response.finish_reason in ("end_turn", "stop"), (
            f"Expected finish_reason='end_turn' or 'stop', got '{response.finish_reason}'"
        )

    def test_finish_reason_preserved_when_turn_complete_received(self) -> None:
        """TURN_COMPLETE finish_reason is preserved when received.

        This tests the normal flow where TURN_COMPLETE arrives with a finish_reason.
        The SDK-provided finish_reason should be used.
        """
        accumulator = StreamingAccumulator()

        accumulator.add(
            DomainEvent(
                type=DomainEventType.CONTENT_DELTA,
                data={"text": "Response text"},
                block_type="text",
            )
        )
        accumulator.add(
            DomainEvent(type=DomainEventType.TURN_COMPLETE, data={"finish_reason": "stop"})
        )

        response = accumulator.to_chat_response()

        # SDK-provided finish_reason should be preserved for text-only responses
        assert response.finish_reason == "stop"

    def test_finish_reason_overridden_to_tool_calls_when_sdk_sends_stop_with_tools(self) -> None:
        """streaming-contract:FinishReason:MUST:5 — tool_calls overrides SDK finish_reason.

        Critical bug fix: When SDK sends TURN_COMPLETE with finish_reason="stop"
        but there are tool_calls, we MUST override to "tool_calls".

        Note: amplifier-core proto defines valid values as:
              "stop", "tool_calls", "length", "content_filter"

        This scenario happens when:
        1. SDK returns tool calls in ASSISTANT_MESSAGE
        2. preToolUse hook denies execution
        3. SDK sends TURN_COMPLETE with finish_reason="stop" (normal completion)
        4. We capture the tool calls and need to tell orchestrator to execute them

        If we don't override to "tool_calls", Amplifier's orchestrator thinks the
        conversation is complete and drops to interactive mode prematurely.
        """
        accumulator = StreamingAccumulator()

        # Simulate: SDK returns tool call, then TURN_COMPLETE with "stop"
        accumulator.add(
            DomainEvent(
                type=DomainEventType.TOOL_CALL,
                data={"id": "call_1", "name": "delegate", "arguments": {"target": "subagent"}},
            )
        )
        accumulator.add(
            DomainEvent(type=DomainEventType.TURN_COMPLETE, data={"finish_reason": "stop"})
        )

        response = accumulator.to_chat_response()

        # CRITICAL: Must be "tool_calls" even though SDK sent "stop"
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.finish_reason == "tool_calls", (
            f"Expected finish_reason='tool_calls' to override SDK's 'stop' "
            f"when tool_calls present. Got '{response.finish_reason}'. "
            f"This causes premature exit to interactive mode."
        )
