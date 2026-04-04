"""
Contract Compliance Tests: Streaming.

Contract: contracts/streaming-contract.md

Tests streaming behavior compliance.
"""

from __future__ import annotations

from typing import Any

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


class TestThinkingBlockSignaturePreservation:
    """streaming-contract:ThinkingBlock:MUST:1

    Tests that reasoning_opaque is preserved as ThinkingBlock.signature.
    """

    def test_thinking_block_preserves_signature(self) -> None:
        """streaming-contract:ThinkingBlock:MUST:1 — reasoning_opaque → signature.

        Anthropic models send encrypted extended thinking data in `reasoning_opaque`.
        This MUST be preserved as ThinkingBlock.signature for multi-turn extended thinking.
        """
        accumulator = StreamingAccumulator()

        # Add thinking delta WITH signature (extracted from SDK reasoning_opaque)
        accumulator.add(
            DomainEvent(
                type=DomainEventType.CONTENT_DELTA,
                data={
                    "text": "Let me think about this...",
                    "reasoning_opaque": "encrypted_signature_abc123xyz",
                },
                block_type="THINKING",
            )
        )
        accumulator.add(
            DomainEvent(type=DomainEventType.TURN_COMPLETE, data={"finish_reason": "stop"})
        )

        response = accumulator.to_chat_response()

        # Should have thinking content
        assert response.content is not None
        assert len(response.content) > 0

        # Find the ThinkingBlock
        thinking_block = None
        for block in response.content:
            if hasattr(block, "thinking"):
                thinking_block = block
                break

        assert thinking_block is not None, "ThinkingBlock not found in response.content"

        # CRITICAL: signature MUST be preserved for multi-turn extended thinking
        assert hasattr(thinking_block, "signature"), "ThinkingBlock missing 'signature' attribute"
        assert thinking_block.signature == "encrypted_signature_abc123xyz", (
            f"Expected signature='encrypted_signature_abc123xyz', "
            f"got '{thinking_block.signature}'. "
            f"This breaks multi-turn extended thinking with Anthropic models."
        )

    def test_thinking_block_handles_no_signature(self) -> None:
        """ThinkingBlock handles missing signature gracefully.

        Non-extended-thinking responses may not have reasoning_opaque.
        """
        accumulator = StreamingAccumulator()

        # Add thinking delta WITHOUT signature
        accumulator.add(
            DomainEvent(
                type=DomainEventType.CONTENT_DELTA,
                data={"text": "Simple thinking..."},
                block_type="THINKING",
            )
        )
        accumulator.add(
            DomainEvent(type=DomainEventType.TURN_COMPLETE, data={"finish_reason": "stop"})
        )

        response = accumulator.to_chat_response()

        # Should have thinking content
        thinking_block = None
        for block in response.content:
            if hasattr(block, "thinking"):
                thinking_block = block
                break

        assert thinking_block is not None
        # signature should be None (not missing, not error)
        assert thinking_block.signature is None


# ---------------------------------------------------------------------------
# T-4 Contract Test Gaps — EventRouter integration tests
# ---------------------------------------------------------------------------


def _make_event_router(
    *,
    capture_handler: Any | None = None,
    on_capture: Any | None = None,
) -> tuple[Any, Any, Any, list[Exception]]:
    """Build a minimal EventRouter for integration testing."""
    import asyncio

    from amplifier_module_provider_github_copilot.event_router import EventRouter
    from amplifier_module_provider_github_copilot.sdk_adapter.tool_capture import ToolCaptureHandler
    from amplifier_module_provider_github_copilot.streaming import EventConfig

    queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=256)
    idle_event = asyncio.Event()
    error_holder: list[Exception] = []
    usage_holder: list[dict[str, int]] = []

    if capture_handler is None:
        capture_handler = ToolCaptureHandler(on_capture_complete=on_capture)

    event_config = EventConfig(
        content_event_types={"assistant.delta", "delta"},
        text_content_types={"assistant.delta"},
        idle_event_types={"session.idle"},
        error_event_types={"session.error", "error"},
    )

    router = EventRouter(
        queue=queue,
        idle_event=idle_event,
        error_holder=error_holder,
        usage_holder=usage_holder,
        capture_handler=capture_handler,
        ttft_state={"checked": False, "start_time": 0.0},
        ttft_threshold_ms=500,
        event_config=event_config,
        emit_streaming_content=lambda _: None,
    )

    return router, queue, idle_event, error_holder


class TestEventRouterToolCaptureAbortIntegration:
    """streaming-contract:abort-on-capture:MUST:1

    EventRouter routes ASSISTANT_MESSAGE → ToolCaptureHandler → abort signal.
    """

    def test_abort_signal_fires_when_tool_captured(self) -> None:
        """streaming-contract:abort-on-capture:MUST:1 — abort when tools captured.

        When EventRouter receives an ASSISTANT_MESSAGE with tool_requests,
        the ToolCaptureHandler fires `on_capture_complete` (the abort callback).
        """

        signal_fired: list[bool] = []

        def abort() -> None:
            signal_fired.append(True)

        router, _, _, _ = _make_event_router(on_capture=abort)

        tool_event: dict[str, Any] = {
            "type": "assistant.message",
            "data": {
                "tool_requests": [{"tool_call_id": "call_1", "name": "read_file", "arguments": {}}]
            },
        }

        router(tool_event)

        assert signal_fired, (
            "abort signal (on_capture_complete) must fire when EventRouter "
            "receives ASSISTANT_MESSAGE with tool_requests"
        )

    def test_tools_are_captured_via_event_router(self) -> None:
        """streaming-contract:abort-on-capture:MUST:1 — tools forwarded to capture handler."""

        from amplifier_module_provider_github_copilot.sdk_adapter.tool_capture import (
            ToolCaptureHandler,
        )

        handler = ToolCaptureHandler()
        router, _, _, _ = _make_event_router(capture_handler=handler)

        tool_event: dict[str, Any] = {
            "type": "assistant.message",
            "data": {
                "tool_requests": [
                    {"tool_call_id": "tc1", "name": "search", "arguments": {"q": "test"}}
                ]
            },
        }
        router(tool_event)

        assert len(handler.captured_tools) == 1
        assert handler.captured_tools[0]["name"] == "search"


class TestEventRouterTTFTWarning:
    """behaviors:Streaming:MUST:1 — TTFT warning emission."""

    def test_ttft_warning_emitted_when_slow(self) -> None:
        """behaviors:Streaming:MUST:1 — logger.warning emitted on slow TTFT.

        When the first content event arrives after more than ttft_threshold_ms,
        the EventRouter MUST emit a warning log.
        """
        import time
        from unittest.mock import patch

        router, _, _, _ = _make_event_router()
        # Set start_time far in the past to guarantee threshold exceeded
        router._ttft["start_time"] = time.time() - 100.0  # 100 seconds ago

        content_event: dict[str, Any] = {"type": "assistant.delta", "data": {}}

        with patch("amplifier_module_provider_github_copilot.event_router.logger") as mock_logger:
            router(content_event)

        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "TTFT" in warning_msg or "first token" in warning_msg.lower()

    def test_ttft_warning_not_emitted_on_second_event(self) -> None:
        """TTFT warning fires at most once per session."""
        import time
        from unittest.mock import patch

        router, _, _, _ = _make_event_router()
        router._ttft["start_time"] = time.time() - 100.0

        content_event: dict[str, Any] = {"type": "assistant.delta", "data": {}}

        with patch("amplifier_module_provider_github_copilot.event_router.logger") as mock_logger:
            router(content_event)  # first event — fires warning
            router(content_event)  # second event — must NOT fire again

        # warning called exactly once
        assert mock_logger.warning.call_count == 1


class TestEventRouterErrorHandling:
    """streaming-contract:completion:MUST:2 — Error event handling."""

    def test_error_event_sets_idle_and_records_error(self) -> None:
        """streaming-contract:completion:MUST:2 — Error event → idle_event set + error captured.

        When EventRouter receives a session.error event, it MUST:
        1. Call _handle_error (record Exception in error_holder)
        2. Set idle_event (unblock session wait loop)
        """

        router, _, idle_event, error_holder = _make_event_router()

        error_event: dict[str, Any] = {
            "type": "session.error",
            "data": {"message": "Model returned an error"},
        }
        router(error_event)

        assert idle_event.is_set(), "idle_event must be set when error event is received"
        assert len(error_holder) == 1, "error_holder must contain one error"
        assert isinstance(error_holder[0], Exception)

    def test_error_event_not_queued(self) -> None:
        """streaming-contract:completion:MUST:2 — Error events must not enter the event queue.

        Error events are handled on the CRITICAL PATH and must not be queued
        (prevents orphan events after error signals idle completion).
        """

        router, queue, _, _ = _make_event_router()

        error_event: dict[str, Any] = {
            "type": "error",
            "data": {"message": "Connection dropped"},
        }
        router(error_event)

        assert queue.empty(), "error events must NOT be queued — critical path only"


# ---------------------------------------------------------------------------
# to_chat_response() kernel output layer tests
# Contract: streaming-contract:StreamingResponse:MUST:1-4
# ---------------------------------------------------------------------------


class TestToChatResponse:
    """streaming-contract:StreamingResponse:MUST:1-4

    Tests that to_chat_response() produces correctly-typed kernel output.
    These verify the kernel output layer (message_models Pydantic types) that
    get_result() tests cannot cover (get_result() returns AccumulatedResponse,
    not kernel types).
    """

    def test_text_content_produces_textblock_in_content(self) -> None:
        """streaming-contract:StreamingResponse:MUST:3

        Text deltas produce TextBlock (Pydantic from message_models) in response.content.
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
                data={"text": "Hello world"},
                block_type="TEXT",
            )
        )
        accumulator.add(
            DomainEvent(type=DomainEventType.TURN_COMPLETE, data={"finish_reason": "stop"})
        )

        response = accumulator.to_chat_response()

        assert response.content is not None
        assert len(response.content) >= 1
        # Must be a TextBlock (Pydantic from message_models), not a plain string
        text_block = response.content[0]
        assert hasattr(text_block, "text"), (
            f"Expected TextBlock with .text attribute, got {type(text_block)}"
        )
        assert text_block.text == "Hello world"

    def test_content_blocks_is_none_when_no_content(self) -> None:
        """streaming-contract:StreamingResponse:MUST:4

        content_blocks is None (not empty list) when no content events received.
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

        # content_blocks MUST be None, not [], when there is no content
        assert response.content_blocks is None, (
            f"Expected content_blocks=None for empty response, "
            f"got {response.content_blocks!r}. "
            f"Streaming UI should not receive an empty list."
        )

    def test_content_blocks_populated_with_text_content_type(self) -> None:
        """streaming-contract:StreamingResponse:MUST:2

        content_blocks is populated with TextContent (dataclass from content_models)
        when text content is present.
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
                data={"text": "Streaming response"},
                block_type="TEXT",
            )
        )
        accumulator.add(
            DomainEvent(type=DomainEventType.TURN_COMPLETE, data={"finish_reason": "stop"})
        )

        response = accumulator.to_chat_response()

        assert response.content_blocks is not None, (
            "content_blocks must be populated when text present"
        )
        assert len(response.content_blocks) >= 1
        # Must be TextContent (dataclass from content_models) — has .text attribute
        # Filter to TextContent blocks (exclude ToolCallContent which has no .text)
        from amplifier_core import TextContent

        text_blocks = [b for b in response.content_blocks if isinstance(b, TextContent)]
        block_types = [type(b).__name__ for b in response.content_blocks]
        assert len(text_blocks) >= 1, f"Expected at least one TextContent block, got {block_types}"
        block = text_blocks[0]
        assert hasattr(block, "text"), (
            f"Expected TextContent dataclass with .text attribute, got {type(block)}"
        )
        assert block.text == "Streaming response"

    def test_tool_call_produces_pydantic_toolcall_in_response(self) -> None:
        """streaming-contract:StreamingResponse:MUST:3

        Tool calls in the accumulator produce ToolCall Pydantic objects
        with correct .id, .name, .arguments fields in response.tool_calls.
        """
        from amplifier_module_provider_github_copilot.streaming import (
            DomainEvent,
            DomainEventType,
            StreamingAccumulator,
        )

        accumulator = StreamingAccumulator()
        accumulator.add(
            DomainEvent(
                type=DomainEventType.TOOL_CALL,
                data={"id": "tc_abc", "name": "read_file", "arguments": {"path": "main.py"}},
            )
        )

        response = accumulator.to_chat_response()

        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        tc = response.tool_calls[0]
        # Must use .arguments (not .input — per provider-protocol:parse_tool_calls:MUST:4)
        assert hasattr(tc, "id"), f"ToolCall missing .id, got {type(tc)}"
        assert hasattr(tc, "name"), f"ToolCall missing .name, got {type(tc)}"
        assert hasattr(tc, "arguments"), (
            "ToolCall missing .arguments (not .input) — provider-protocol:parse_tool_calls:MUST:4"
        )
        assert tc.id == "tc_abc"
        assert tc.name == "read_file"
        assert tc.arguments == {"path": "main.py"}
