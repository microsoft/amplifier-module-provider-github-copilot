"""
Contract Compliance Tests: Streaming.

Contract: contracts/streaming-contract.md

Tests streaming behavior compliance.
"""

from __future__ import annotations

import logging
from typing import Any

from amplifier_module_provider_github_copilot.streaming import (
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
        """Contract: streaming-contract:Accumulation:MUST:2 — Complete response on TURN_COMPLETE."""
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
        """Contract: streaming-contract:ContentTypes:MUST:1 — Separates text and thinking."""
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
        """Contract: streaming-contract:ToolCapture:MUST:2 — Tool calls in final response."""
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

        # Contract: streaming-contract:FinishReason:MUST:5
        assert isinstance(response.tool_calls, list)
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
        # Should default to "stop" when no tools (streaming.py:372 returns exact string "stop")
        assert response.finish_reason == "stop"

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
        assert isinstance(response.tool_calls, list)
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
        assert isinstance(response.content, list)
        assert len(response.content) == 1

        # Find the ThinkingBlock
        from amplifier_core import ThinkingBlock

        thinking_block = None
        for block in response.content:
            if hasattr(block, "thinking"):
                thinking_block = block
                break

        assert isinstance(thinking_block, ThinkingBlock), (
            "ThinkingBlock not found in response.content"
        )

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
        from amplifier_core import ThinkingBlock

        thinking_block = None
        for block in response.content:
            if hasattr(block, "thinking"):
                thinking_block = block
                break

        assert isinstance(thinking_block, ThinkingBlock)
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
    usage_holder: list[dict[str, int | None]] = []

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
    """streaming-contract:ToolCapture:MUST:1

    EventRouter routes ASSISTANT_MESSAGE → ToolCaptureHandler → abort signal.
    (Abort callback is implementation behavior, not contracted.)
    """

    def test_abort_signal_fires_when_tool_captured(self) -> None:
        """streaming-contract:ToolCapture:MUST:1 — tool capture triggers abort callback.

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
        """streaming-contract:ToolCapture:MUST:1 — tools forwarded to capture handler."""

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

        with patch(
            "amplifier_module_provider_github_copilot.event_router.logger", spec=logging.Logger
        ) as mock_logger:
            router(content_event)

        mock_logger.warning.assert_called_once()
        mock_logger.warn.assert_not_called()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "TTFT" in warning_msg or "first token" in warning_msg.lower()

    def test_ttft_warning_not_emitted_on_second_event(self) -> None:
        """TTFT warning fires at most once per session."""
        import time
        from unittest.mock import patch

        router, _, _, _ = _make_event_router()
        router._ttft["start_time"] = time.time() - 100.0

        content_event: dict[str, Any] = {"type": "assistant.delta", "data": {}}

        with patch(
            "amplifier_module_provider_github_copilot.event_router.logger", spec=logging.Logger
        ) as mock_logger:
            router(content_event)  # first event — fires warning
            router(content_event)  # second event — must NOT fire again

        # warning called exactly once
        assert mock_logger.warning.call_count == 1
        mock_logger.warn.assert_not_called()


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


class TestThinkingDeltaNotEmittedPerToken:
    """streaming-contract:ProgressiveStreaming:SHOULD:5

    Thinking deltas MUST NOT be emitted per-token as separate ThinkingContent objects.
    Per-token emission causes the CLI to render one 🧠 box per streaming chunk.
    Granularity is controlled by events.yaml thinking_content_types.
    """

    def test_thinking_content_types_empty_in_events_yaml(self) -> None:
        """streaming-contract:ProgressiveStreaming:SHOULD:5.

        events.yaml must suppress per-token thinking emission.
        The events.yaml streaming_emission.thinking_content_types MUST be empty
        so that EventRouter._emit_progressive_content never emits ThinkingContent
        per delta. Verified by loading real EventConfig from events.yaml.
        """
        from amplifier_module_provider_github_copilot.streaming import load_event_config

        event_config = load_event_config()

        assert event_config.thinking_content_types == set(), (
            "streaming_emission.thinking_content_types must be empty in events.yaml "
            "to prevent per-token ThinkingContent emission (SHOULD:5). "
            f"Got: {event_config.thinking_content_types}"
        )

    def test_thinking_delta_does_not_trigger_emit_callback(self) -> None:
        """streaming-contract:ProgressiveStreaming:SHOULD:5 — reasoning delta must not call emit.

        When EventRouter receives an assistant.reasoning_delta event and
        thinking_content_types is empty (per events.yaml policy), the
        emit_streaming_content callback MUST NOT be called.
        """
        import asyncio

        from amplifier_module_provider_github_copilot.event_router import EventRouter
        from amplifier_module_provider_github_copilot.sdk_adapter.tool_capture import (
            ToolCaptureHandler,
        )
        from amplifier_module_provider_github_copilot.streaming import load_event_config

        emitted: list[Any] = []
        event_config = load_event_config()  # real config — thinking_content_types must be empty

        router = EventRouter(
            queue=asyncio.Queue(maxsize=256),
            idle_event=asyncio.Event(),
            error_holder=[],
            usage_holder=[],
            capture_handler=ToolCaptureHandler(on_capture_complete=None),
            ttft_state={"checked": False, "start_time": 0.0},
            ttft_threshold_ms=500,
            event_config=event_config,
            emit_streaming_content=emitted.append,
        )

        # Simulate a reasoning delta — the per-token thinking chunk the SDK sends
        reasoning_event: dict[str, Any] = {
            "type": "assistant.reasoning_delta",
            "data": {"delta_content": "The"},
        }
        router(reasoning_event)

        assert emitted == [], (
            "emit_streaming_content must NOT be called for reasoning deltas "
            "when thinking_content_types is empty (SHOULD:5). "
            f"Got {len(emitted)} emission(s)."
        )


class TestThinkingBlockConsolidation:
    """streaming-contract:Accumulation:MUST:3

    Empty text deltas interleaved with reasoning deltas MUST NOT produce
    fragmented thinking blocks. The GitHub Copilot SDK emits empty
    assistant.streaming_delta events between each assistant.reasoning_delta.
    Without the guard, each empty text creates a separate text-type entry
    in _ordered_blocks, causing one ThinkingContent per token in content_blocks.
    """

    def test_empty_text_deltas_do_not_fragment_thinking_blocks(self) -> None:
        """Contract: streaming-contract:Accumulation:MUST:3

        Empty text events must not fragment thinking.

        Simulates the SDK pattern: assistant.streaming_delta (empty) interleaved
        with assistant.reasoning_delta events. The accumulator MUST consolidate all
        reasoning tokens into a single thinking block in _ordered_blocks.
        """
        accumulator = StreamingAccumulator()

        T = "TEXT"
        K = "THINKING"
        CD = DomainEventType.CONTENT_DELTA
        # Simulate the SDK pattern: empty streaming_delta before each reasoning_delta
        events = [
            DomainEvent(type=CD, data={"text": ""}, block_type=T),
            DomainEvent(type=CD, data={"text": "The"}, block_type=K),
            DomainEvent(type=CD, data={"text": ""}, block_type=T),
            DomainEvent(type=CD, data={"text": " user wants"}, block_type=K),
            DomainEvent(type=CD, data={"text": ""}, block_type=T),
            DomainEvent(type=CD, data={"text": " to know"}, block_type=K),
            DomainEvent(type=CD, data={"text": "Answer"}, block_type=T),
        ]
        for event in events:
            accumulator.add(event)

        # Must have exactly 2 blocks: 1 thinking + 1 text (in that order)
        block_types = [b["type"] for b in accumulator._ordered_blocks]
        assert block_types == ["thinking", "text"], (
            f"streaming-contract:Accumulation:MUST:3 — empty text deltas between "
            f"reasoning deltas must not fragment thinking blocks. "
            f"Expected ['thinking', 'text'], got {block_types}"
        )

        # The single thinking block must have all reasoning token text concatenated
        thinking_block = accumulator._ordered_blocks[0]
        assert thinking_block["text"] == "The user wants to know", (
            f"All reasoning tokens must be concatenated. Got: {thinking_block['text']!r}"
        )

    def test_content_blocks_has_single_thinking_block_despite_interleaving(self) -> None:
        """streaming-contract:Accumulation:MUST:3 — to_chat_response produces one ThinkingContent.

        The loop-streaming orchestrator emits one content_block:start/end pair per
        content_blocks entry. A single ThinkingContent means ONE 🧠 box.
        """
        from amplifier_core import ThinkingContent

        accumulator = StreamingAccumulator()
        CD = DomainEventType.CONTENT_DELTA

        # 3 reasoning tokens with empty text events between them (SDK pattern)
        for i in range(3):
            accumulator.add(DomainEvent(type=CD, data={"text": ""}, block_type="TEXT"))
            accumulator.add(DomainEvent(type=CD, data={"text": f"token{i}"}, block_type="THINKING"))

        accumulator.add(
            DomainEvent(type=DomainEventType.TURN_COMPLETE, data={"finish_reason": "stop"})
        )

        response = accumulator.to_chat_response()

        assert isinstance(response.content_blocks, list)
        thinking_blocks = [b for b in response.content_blocks if isinstance(b, ThinkingContent)]
        assert len(thinking_blocks) == 1, (
            f"streaming-contract:Accumulation:MUST:3 — must produce ONE ThinkingContent "
            f"in content_blocks regardless of SDK interleaving. "
            f"Got {len(thinking_blocks)}."
        )
        assert thinking_blocks[0].text == "token0token1token2"


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
        """Contract: streaming-contract:StreamingResponse:MUST:3

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

        assert isinstance(response.content, list)
        assert len(response.content) == 1
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

        assert isinstance(response.content_blocks, list), (
            "content_blocks must be populated when text present"
        )
        assert len(response.content_blocks) == 1
        # Must be TextContent (dataclass from content_models) — has .text attribute
        # Filter to TextContent blocks (exclude ToolCallContent which has no .text)
        from amplifier_core import TextContent

        text_blocks = [b for b in response.content_blocks if isinstance(b, TextContent)]
        assert len(text_blocks) == 1, "Expected exactly one TextContent block"
        block = text_blocks[0]
        assert hasattr(block, "text"), (
            f"Expected TextContent dataclass with .text attribute, got {type(block)}"
        )
        assert block.text == "Streaming response"

    def test_tool_call_produces_pydantic_toolcall_in_response(self) -> None:
        """Contract: streaming-contract:StreamingResponse:MUST:3

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

        assert isinstance(response.tool_calls, list)
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
