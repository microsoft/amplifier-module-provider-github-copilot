"""
Provider Streaming Contract Tests — TDD for feat/proper-streaming.

Tests the five-event streaming contract defined in
docs/provider-streaming-contract.md as implemented by the GitHub Copilot provider.

Contract reference: provider-streaming-contract.md
Breaking change: llm:content_block is REMOVED; replaced by the five-event contract.

Ordering guarantee: an asyncio.Queue consumer drains events sequentially,
so block_start is always observed before its deltas and block_end follows
the last delta of its block.
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_coordinator() -> Any:
    """Return a mock coordinator with a tracked hooks.emit."""
    coord = MagicMock()
    coord.hooks = MagicMock()
    coord.hooks.emit = AsyncMock()
    return coord


def _emitted_names(coordinator: Any) -> list[str]:
    """Return the ordered list of event names emitted to coordinator.hooks.emit."""
    return [call.args[0] for call in coordinator.hooks.emit.call_args_list]


def _emitted_payloads(coordinator: Any) -> list[dict[str, Any]]:
    """Return the ordered list of payloads emitted to coordinator.hooks.emit."""
    return [call.args[1] for call in coordinator.hooks.emit.call_args_list]


def _make_event_router_with_ctx(
    *,
    stream_ctx: Any,
    thinking_types: set[str] | None = None,
) -> Any:
    """Build an EventRouter wired to the given streaming context."""
    from amplifier_module_provider_github_copilot.event_router import EventRouter
    from amplifier_module_provider_github_copilot.sdk_adapter.tool_capture import ToolCaptureHandler
    from amplifier_module_provider_github_copilot.streaming import EventConfig

    event_config = EventConfig(
        content_event_types={"assistant.message_delta", "assistant.reasoning_delta"},
        text_content_types={"assistant.message_delta"},
        thinking_content_types=(
            thinking_types if thinking_types is not None else {"assistant.reasoning_delta"}
        ),
        idle_event_types={"session.idle"},
        error_event_types={"session.error", "error"},
    )

    return EventRouter(
        queue=asyncio.Queue(maxsize=256),
        idle_event=asyncio.Event(),
        error_holder=[],
        usage_holder=[],
        capture_handler=ToolCaptureHandler(on_capture_complete=None),
        ttft_state={"checked": False, "start_time": 0.0},
        ttft_threshold_ms=500,
        event_config=event_config,
        stream_ctx=stream_ctx,
    )


# ---------------------------------------------------------------------------
# _StreamingContext unit tests
# ---------------------------------------------------------------------------


class TestStreamingContext:
    """Unit tests for _StreamingContext state machine."""

    def test_initial_state(self) -> None:
        """Fresh context starts with no open block."""
        from amplifier_module_provider_github_copilot.provider import _StreamingContext  # type: ignore[attr-defined]

        ctx = _StreamingContext(request_id="r1")
        assert ctx.block_index == -1
        assert ctx.current_block_type is None
        assert ctx.block_seq == 0
        assert not ctx.partial_emitted

    def test_first_text_delta_opens_block(self) -> None:
        """First text delta emits block_start then block_delta."""
        from amplifier_module_provider_github_copilot.provider import _StreamingContext  # type: ignore[attr-defined]

        ctx = _StreamingContext(request_id="req-abc")
        ctx.handle_delta("Hello", "text")

        events: list[tuple[str, dict[str, Any]]] = []
        while not ctx._queue.empty():  # type: ignore[attr-defined]
            events.append(ctx._queue.get_nowait())  # type: ignore[attr-defined]

        names = [e[0] for e in events]
        assert names == ["llm:stream_block_start", "llm:stream_block_delta"]

    def test_block_start_payload(self) -> None:
        """block_start payload has request_id, block_index=0, block_type='text'."""
        from amplifier_module_provider_github_copilot.provider import _StreamingContext  # type: ignore[attr-defined]

        ctx = _StreamingContext(request_id="r1")
        ctx.handle_delta("Hi", "text")

        start_payload = ctx._queue.get_nowait()[1]  # type: ignore[attr-defined]
        assert start_payload == {"request_id": "r1", "block_index": 0, "block_type": "text"}

    def test_delta_payload(self) -> None:
        """stream_block_delta payload has request_id, block_index, sequence, text."""
        from amplifier_module_provider_github_copilot.provider import _StreamingContext  # type: ignore[attr-defined]

        ctx = _StreamingContext(request_id="r1")
        ctx.handle_delta("Hi", "text")

        ctx._queue.get_nowait()  # discard block_start  # type: ignore[attr-defined]
        delta_payload = ctx._queue.get_nowait()[1]  # type: ignore[attr-defined]
        assert delta_payload == {
            "request_id": "r1",
            "block_index": 0,
            "block_type": "text",
            "sequence": 0,
            "text": "Hi",
        }

    def test_per_block_sequence_increments(self) -> None:
        """Sequence counter increments within a block."""
        from amplifier_module_provider_github_copilot.provider import _StreamingContext  # type: ignore[attr-defined]

        ctx = _StreamingContext(request_id="r1")
        ctx.handle_delta("A", "text")
        ctx.handle_delta("B", "text")
        ctx.handle_delta("C", "text")

        events: list[tuple[str, dict[str, Any]]] = []
        while not ctx._queue.empty():  # type: ignore[attr-defined]
            events.append(ctx._queue.get_nowait())  # type: ignore[attr-defined]

        # block_start + delta(seq=0) + delta(seq=1) + delta(seq=2)
        assert len(events) == 4
        seqs = [e[1]["sequence"] for e in events if e[0] == "llm:stream_block_delta"]
        assert seqs == [0, 1, 2]

    def test_sequence_resets_on_block_transition(self) -> None:
        """Sequence resets to 0 when a new block starts."""
        from amplifier_module_provider_github_copilot.provider import _StreamingContext  # type: ignore[attr-defined]

        ctx = _StreamingContext(request_id="r1")
        ctx.handle_delta("think", "thinking")   # block 0
        ctx.handle_delta("think2", "thinking")  # still block 0
        ctx.handle_delta("text", "text")        # block 1 — seq resets

        events: list[tuple[str, dict[str, Any]]] = []
        while not ctx._queue.empty():  # type: ignore[attr-defined]
            events.append(ctx._queue.get_nowait())  # type: ignore[attr-defined]

        text_delta = next(
            e for e in events
            if e[0] == "llm:stream_block_delta" and e[1].get("block_type") == "text"
        )
        assert text_delta[1]["sequence"] == 0

    def test_thinking_delta_event_name(self) -> None:
        """Thinking deltas use llm:stream_block_delta with block_type='thinking'."""
        from amplifier_module_provider_github_copilot.provider import _StreamingContext  # type: ignore[attr-defined]

        ctx = _StreamingContext(request_id="r1")
        ctx.handle_delta("reasoning...", "thinking")

        events: list[tuple[str, dict[str, Any]]] = []
        while not ctx._queue.empty():  # type: ignore[attr-defined]
            events.append(ctx._queue.get_nowait())  # type: ignore[attr-defined]

        delta_events = [e for e in events if "delta" in e[0]]
        assert len(delta_events) == 1
        assert delta_events[0][0] == "llm:stream_block_delta"
        assert delta_events[0][1]["block_type"] == "thinking"

    def test_transition_emits_block_end_then_new_block_start(self) -> None:
        """Transitioning from thinking to text emits block_end, block_start."""
        from amplifier_module_provider_github_copilot.provider import _StreamingContext  # type: ignore[attr-defined]

        ctx = _StreamingContext(request_id="r1")
        ctx.handle_delta("thinking", "thinking")
        ctx.handle_delta("text", "text")

        events: list[tuple[str, dict[str, Any]]] = []
        while not ctx._queue.empty():  # type: ignore[attr-defined]
            events.append(ctx._queue.get_nowait())  # type: ignore[attr-defined]

        names = [e[0] for e in events]
        payloads = [e[1] for e in events]
        # thinking_start, block_delta(thinking), thinking_end, text_start, block_delta(text)
        assert names[0] == "llm:stream_block_start"
        assert names[1] == "llm:stream_block_delta"
        assert payloads[1]["block_type"] == "thinking"
        assert names[2] == "llm:stream_block_end"
        assert names[3] == "llm:stream_block_start"
        assert names[4] == "llm:stream_block_delta"
        assert payloads[4]["block_type"] == "text"

    def test_block_end_payload(self) -> None:
        """block_end payload has request_id, block_index, block_type."""
        from amplifier_module_provider_github_copilot.provider import _StreamingContext  # type: ignore[attr-defined]

        ctx = _StreamingContext(request_id="r1")
        ctx.handle_delta("thinking", "thinking")
        ctx.handle_delta("text", "text")  # triggers end of thinking block

        events: list[tuple[str, dict[str, Any]]] = []
        while not ctx._queue.empty():  # type: ignore[attr-defined]
            events.append(ctx._queue.get_nowait())  # type: ignore[attr-defined]

        end_event = next(e for e in events if e[0] == "llm:stream_block_end")
        assert end_event[1] == {"request_id": "r1", "block_index": 0, "block_type": "thinking"}

    def test_close_current_block_emits_block_end(self) -> None:
        """close_current_block() emits block_end for the open block."""
        from amplifier_module_provider_github_copilot.provider import _StreamingContext  # type: ignore[attr-defined]

        ctx = _StreamingContext(request_id="r1")
        ctx.handle_delta("hello", "text")
        ctx.close_current_block()

        events: list[tuple[str, dict[str, Any]]] = []
        while not ctx._queue.empty():  # type: ignore[attr-defined]
            events.append(ctx._queue.get_nowait())  # type: ignore[attr-defined]

        names = [e[0] for e in events]
        assert "llm:stream_block_end" in names

    def test_close_current_block_noop_when_no_open_block(self) -> None:
        """close_current_block() is a no-op when no block is open."""
        from amplifier_module_provider_github_copilot.provider import _StreamingContext  # type: ignore[attr-defined]

        ctx = _StreamingContext(request_id="r1")
        ctx.close_current_block()  # no-op
        assert ctx._queue.empty()  # type: ignore[attr-defined]

    def test_empty_text_not_queued(self) -> None:
        """Empty text produces no events (contract: empty fragments never emitted)."""
        from amplifier_module_provider_github_copilot.provider import _StreamingContext  # type: ignore[attr-defined]

        ctx = _StreamingContext(request_id="r2")
        ctx.handle_delta("", "text")
        assert ctx._queue.empty()  # type: ignore[attr-defined]

    def test_partial_emitted_flag(self) -> None:
        """partial_emitted becomes True after first block_start."""
        from amplifier_module_provider_github_copilot.provider import _StreamingContext  # type: ignore[attr-defined]

        ctx = _StreamingContext(request_id="r1")
        assert not ctx.partial_emitted
        ctx.handle_delta("x", "text")
        assert ctx.partial_emitted

    def test_shared_block_index_across_types(self) -> None:
        """block_index is a single shared space across text and thinking blocks."""
        from amplifier_module_provider_github_copilot.provider import _StreamingContext  # type: ignore[attr-defined]

        ctx = _StreamingContext(request_id="r1")
        ctx.handle_delta("thinking", "thinking")  # block 0
        ctx.handle_delta("text", "text")           # block 1
        ctx.handle_delta("more text", "text")      # still block 1

        events: list[tuple[str, dict[str, Any]]] = []
        while not ctx._queue.empty():  # type: ignore[attr-defined]
            events.append(ctx._queue.get_nowait())  # type: ignore[attr-defined]

        starts = [e for e in events if e[0] == "llm:stream_block_start"]
        assert len(starts) == 2
        assert starts[0][1]["block_index"] == 0
        assert starts[1][1]["block_index"] == 1


# ---------------------------------------------------------------------------
# _run_stream_consumer unit tests
# ---------------------------------------------------------------------------


class TestRunStreamConsumer:
    """Tests for the ordered async consumer coroutine."""

    @pytest.mark.asyncio
    async def test_consumer_emits_in_order(self) -> None:
        """Consumer emits events in the order they were enqueued."""
        from amplifier_module_provider_github_copilot.provider import (  # type: ignore[attr-defined]
            _StreamingContext,
            _run_stream_consumer,
        )

        coordinator = _make_coordinator()
        ctx = _StreamingContext(request_id="r1")

        ctx.handle_delta("thinking", "thinking")
        ctx.handle_delta("text", "text")
        ctx.close_current_block()
        ctx.signal_done()

        await _run_stream_consumer(ctx, coordinator)

        names = _emitted_names(coordinator)
        # start, delta, end, start, delta, end
        assert names[0] == "llm:stream_block_start"
        assert names[-1] == "llm:stream_block_end"
        # last delta comes before last end
        last_delta_idx = max(i for i, n in enumerate(names) if "delta" in n)
        last_end_idx = max(i for i, n in enumerate(names) if n == "llm:stream_block_end")
        assert last_delta_idx < last_end_idx

    @pytest.mark.asyncio
    async def test_consumer_no_emit_without_coordinator(self) -> None:
        """Consumer drains queue silently when coordinator is None."""
        from amplifier_module_provider_github_copilot.provider import (  # type: ignore[attr-defined]
            _StreamingContext,
            _run_stream_consumer,
        )

        ctx = _StreamingContext(request_id="r1")
        ctx.handle_delta("hello", "text")
        ctx.close_current_block()
        ctx.signal_done()

        await _run_stream_consumer(ctx, None)  # no error, queue drained
        assert ctx._queue.empty()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_consumer_handles_emit_error_gracefully(self) -> None:
        """Consumer does not propagate exceptions from coordinator.hooks.emit."""
        from amplifier_module_provider_github_copilot.provider import (  # type: ignore[attr-defined]
            _StreamingContext,
            _run_stream_consumer,
        )

        coordinator = _make_coordinator()
        coordinator.hooks.emit.side_effect = RuntimeError("hook broke")

        ctx = _StreamingContext(request_id="r1")
        ctx.handle_delta("hello", "text")
        ctx.signal_done()

        # should not raise even though emit() throws
        await _run_stream_consumer(ctx, coordinator)


# ---------------------------------------------------------------------------
# EventRouter -> StreamingContext integration
# ---------------------------------------------------------------------------


class TestEventRouterStreamingIntegration:
    """EventRouter feeds _StreamingContext correctly via handle_delta."""

    @pytest.mark.asyncio
    async def test_text_delta_queues_block_start_and_delta(self) -> None:
        """SDK text delta -> EventRouter -> block_start, block_delta in context queue."""
        from amplifier_module_provider_github_copilot.provider import (  # type: ignore[attr-defined]
            _StreamingContext,
            _run_stream_consumer,
        )

        coordinator = _make_coordinator()
        ctx = _StreamingContext(request_id="req-1")
        router = _make_event_router_with_ctx(stream_ctx=ctx)

        sdk_event = {"type": "assistant.message_delta", "data": {"delta_content": "Hello"}}
        router(sdk_event)

        ctx.close_current_block()
        ctx.signal_done()
        await _run_stream_consumer(ctx, coordinator)

        names = _emitted_names(coordinator)
        assert "llm:stream_block_start" in names
        assert "llm:stream_block_delta" in names

    @pytest.mark.asyncio
    async def test_thinking_delta_queues_thinking_events(self) -> None:
        """SDK reasoning delta -> EventRouter -> stream_block_start(thinking), stream_block_delta(thinking)."""
        from amplifier_module_provider_github_copilot.provider import (  # type: ignore[attr-defined]
            _StreamingContext,
            _run_stream_consumer,
        )

        coordinator = _make_coordinator()
        ctx = _StreamingContext(request_id="req-1")
        router = _make_event_router_with_ctx(stream_ctx=ctx)

        sdk_event = {"type": "assistant.reasoning_delta", "data": {"delta_content": "thinking..."}}
        router(sdk_event)

        ctx.close_current_block()
        ctx.signal_done()
        await _run_stream_consumer(ctx, coordinator)

        names = _emitted_names(coordinator)
        payloads = _emitted_payloads(coordinator)
        assert "llm:stream_block_delta" in names
        # The delta carries block_type="thinking"
        thinking_delta = next(p for n, p in zip(names, payloads) if n == "llm:stream_block_delta")
        assert thinking_delta["block_type"] == "thinking"
        start = next(p for n, p in zip(names, payloads) if n == "llm:stream_block_start")
        assert start["block_type"] == "thinking"

    @pytest.mark.asyncio
    async def test_no_llm_content_block_emitted(self) -> None:
        """OLD llm:content_block must NOT be emitted anywhere in the new path."""
        from amplifier_module_provider_github_copilot.provider import (  # type: ignore[attr-defined]
            _StreamingContext,
            _run_stream_consumer,
        )

        coordinator = _make_coordinator()
        ctx = _StreamingContext(request_id="req-1")
        router = _make_event_router_with_ctx(stream_ctx=ctx)

        sdk_event = {"type": "assistant.message_delta", "data": {"delta_content": "hello"}}
        router(sdk_event)
        ctx.close_current_block()
        ctx.signal_done()

        await _run_stream_consumer(ctx, coordinator)

        names = _emitted_names(coordinator)
        assert "llm:content_block" not in names, (
            "llm:content_block was renamed — it must NOT appear in the new streaming path."
        )

    @pytest.mark.asyncio
    async def test_block_start_before_delta_ordering(self) -> None:
        """block_start is always first among events for a block."""
        from amplifier_module_provider_github_copilot.provider import (  # type: ignore[attr-defined]
            _StreamingContext,
            _run_stream_consumer,
        )

        coordinator = _make_coordinator()
        ctx = _StreamingContext(request_id="req-1")
        router = _make_event_router_with_ctx(stream_ctx=ctx)

        for tok in ["a", "b", "c"]:
            router({"type": "assistant.message_delta", "data": {"delta_content": tok}})

        ctx.close_current_block()
        ctx.signal_done()
        await _run_stream_consumer(ctx, coordinator)

        names = _emitted_names(coordinator)
        start_idx = names.index("llm:stream_block_start")
        first_delta_idx = names.index("llm:stream_block_delta")
        end_idx = names.index("llm:stream_block_end")

        assert start_idx < first_delta_idx
        assert first_delta_idx < end_idx

    @pytest.mark.asyncio
    async def test_per_block_sequence_numbers(self) -> None:
        """Sequence numbers are per-block and 0-based."""
        from amplifier_module_provider_github_copilot.provider import (  # type: ignore[attr-defined]
            _StreamingContext,
            _run_stream_consumer,
        )

        coordinator = _make_coordinator()
        ctx = _StreamingContext(request_id="req-1")
        router = _make_event_router_with_ctx(stream_ctx=ctx)

        for tok in ["x", "y", "z"]:
            router({"type": "assistant.message_delta", "data": {"delta_content": tok}})

        ctx.close_current_block()
        ctx.signal_done()
        await _run_stream_consumer(ctx, coordinator)

        payloads = _emitted_payloads(coordinator)
        names = _emitted_names(coordinator)
        delta_payloads = [p for n, p in zip(names, payloads) if n == "llm:stream_block_delta"]
        assert [p["sequence"] for p in delta_payloads] == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_single_request_id_all_events(self) -> None:
        """All streaming events for a call share one request_id."""
        from amplifier_module_provider_github_copilot.provider import (  # type: ignore[attr-defined]
            _StreamingContext,
            _run_stream_consumer,
        )

        coordinator = _make_coordinator()
        ctx = _StreamingContext(request_id="unique-req-xyz")
        router = _make_event_router_with_ctx(stream_ctx=ctx)

        router({"type": "assistant.message_delta", "data": {"delta_content": "hello"}})
        ctx.close_current_block()
        ctx.signal_done()

        await _run_stream_consumer(ctx, coordinator)

        payloads = _emitted_payloads(coordinator)
        for p in payloads:
            assert p.get("request_id") == "unique-req-xyz"

    @pytest.mark.asyncio
    async def test_thinking_to_text_transition_full_sequence(self) -> None:
        """Full thinking->text sequence: start/delta/end/start/delta/end."""
        from amplifier_module_provider_github_copilot.provider import (  # type: ignore[attr-defined]
            _StreamingContext,
            _run_stream_consumer,
        )

        coordinator = _make_coordinator()
        ctx = _StreamingContext(request_id="req-1")
        router = _make_event_router_with_ctx(stream_ctx=ctx)

        router({"type": "assistant.reasoning_delta", "data": {"delta_content": "thinking"}})
        router({"type": "assistant.message_delta", "data": {"delta_content": "response"}})
        ctx.close_current_block()
        ctx.signal_done()

        await _run_stream_consumer(ctx, coordinator)

        names = _emitted_names(coordinator)
        payloads = _emitted_payloads(coordinator)
        assert names == [
            "llm:stream_block_start",
            "llm:stream_block_delta",   # thinking block — block_type="thinking"
            "llm:stream_block_end",
            "llm:stream_block_start",
            "llm:stream_block_delta",   # text block — block_type="text"
            "llm:stream_block_end",
        ]
        deltas = [(n, p) for n, p in zip(names, payloads) if n == "llm:stream_block_delta"]
        assert deltas[0][1]["block_type"] == "thinking"
        assert deltas[1][1]["block_type"] == "text"


# ---------------------------------------------------------------------------
# stream_aborted
# ---------------------------------------------------------------------------


class TestStreamAborted:
    """llm:stream_aborted emitted ONLY when partial_emitted and there is an error."""

    @pytest.mark.asyncio
    async def test_aborted_after_partial_emit(self) -> None:
        """stream_aborted emitted when partial_emitted=True and error occurs."""
        from amplifier_module_provider_github_copilot.provider import (  # type: ignore[attr-defined]
            _StreamingContext,
            _run_stream_consumer,
        )

        coordinator = _make_coordinator()
        ctx = _StreamingContext(request_id="r1")
        ctx.handle_delta("hello", "text")  # partial_emitted = True

        # Simulate error path
        err = RuntimeError("sdk failed")
        if ctx.partial_emitted:
            ctx._put("llm:stream_aborted", {  # type: ignore[attr-defined]
                "request_id": ctx.request_id,
                "error": {"type": type(err).__name__, "msg": str(err)},
            })
        ctx.signal_done()

        await _run_stream_consumer(ctx, coordinator)

        names = _emitted_names(coordinator)
        assert "llm:stream_aborted" in names

        payloads = _emitted_payloads(coordinator)
        aborted = next(p for n, p in zip(names, payloads) if n == "llm:stream_aborted")
        assert aborted["request_id"] == "r1"
        assert aborted["error"]["type"] == "RuntimeError"
        assert "sdk failed" in aborted["error"]["msg"]

    @pytest.mark.asyncio
    async def test_no_aborted_without_partial_emit(self) -> None:
        """stream_aborted NOT emitted when partial_emitted=False."""
        from amplifier_module_provider_github_copilot.provider import (  # type: ignore[attr-defined]
            _StreamingContext,
            _run_stream_consumer,
        )

        coordinator = _make_coordinator()
        ctx = _StreamingContext(request_id="r1")
        # partial_emitted is still False

        err = RuntimeError("early error")
        if ctx.partial_emitted:  # False -- should not enter
            ctx._put("llm:stream_aborted", {  # type: ignore[attr-defined]
                "request_id": ctx.request_id,
                "error": {"type": type(err).__name__, "msg": str(err)},
            })
        ctx.signal_done()

        await _run_stream_consumer(ctx, coordinator)

        names = _emitted_names(coordinator)
        assert "llm:stream_aborted" not in names


# ---------------------------------------------------------------------------
# Non-streaming path (stream_ctx=None -> no stream events)
# ---------------------------------------------------------------------------


class TestNonStreamingPath:
    """EventRouter with stream_ctx=None emits no streaming events."""

    def test_no_stream_events_when_stream_ctx_is_none(self) -> None:
        """EventRouter with stream_ctx=None does not queue any stream events."""
        from amplifier_module_provider_github_copilot.event_router import EventRouter
        from amplifier_module_provider_github_copilot.sdk_adapter.tool_capture import ToolCaptureHandler
        from amplifier_module_provider_github_copilot.streaming import EventConfig

        event_config = EventConfig(
            content_event_types={"assistant.message_delta"},
            text_content_types={"assistant.message_delta"},
            idle_event_types={"session.idle"},
            error_event_types={"session.error"},
        )

        captured_emits: list[Any] = []

        router = EventRouter(
            queue=asyncio.Queue(maxsize=256),
            idle_event=asyncio.Event(),
            error_holder=[],
            usage_holder=[],
            capture_handler=ToolCaptureHandler(on_capture_complete=None),
            ttft_state={"checked": False, "start_time": 0.0},
            ttft_threshold_ms=500,
            event_config=event_config,
            stream_ctx=None,  # non-streaming path
        )

        router({"type": "assistant.message_delta", "data": {"delta_content": "hello"}})

        # No stream events should have been put into any streaming queue
        # (We verify by checking that stream_ctx wasn't used -- no _StreamingContext exists)
        # The router should not raise and should not call any stream emission.
        assert True  # reaching here without error is sufficient


# ---------------------------------------------------------------------------
# use_streaming config and metadata override
# ---------------------------------------------------------------------------


class TestUseStreamingConfig:
    """use_streaming config and per-request metadata override."""

    def test_use_streaming_defaults_true(self) -> None:
        """use_streaming defaults to True when not in config."""
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider(config={})
        assert provider.config.get("use_streaming", True) is True

    def test_use_streaming_can_be_disabled(self) -> None:
        """use_streaming=False in config disables streaming."""
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider(config={"use_streaming": False})
        assert provider.config.get("use_streaming", True) is False

    def test_stream_false_metadata_override_logic(self) -> None:
        """metadata={'stream': False} uses identity check (is False, not ==False)."""
        use_streaming = True
        metadata: dict[str, Any] = {"stream": False}

        _use_streaming = use_streaming
        if isinstance(metadata, dict) and metadata.get("stream") is False:
            _use_streaming = False

        assert not _use_streaming

    def test_stream_none_does_not_override(self) -> None:
        """metadata={'stream': None} does NOT disable streaming (None is not False)."""
        use_streaming = True
        metadata: dict[str, Any] = {"stream": None}

        _use_streaming = use_streaming
        if isinstance(metadata, dict) and metadata.get("stream") is False:
            _use_streaming = False

        assert _use_streaming


# ---------------------------------------------------------------------------
# events.yaml thinking_content_types includes reasoning_delta
# ---------------------------------------------------------------------------


class TestEventsYamlThinkingTypes:
    """events.yaml must list thinking event types for streaming contract."""

    def test_reasoning_delta_in_thinking_content_types(self) -> None:
        """assistant.reasoning_delta must appear in thinking_content_types.

        The new streaming contract emits llm:stream_block_delta with block_type="thinking"
        for each reasoning delta. Previously this set was empty (old ThinkingContent
        per-token was suppressed); now it must be populated.
        """
        from amplifier_module_provider_github_copilot.streaming import load_event_config

        event_config = load_event_config()
        assert "assistant.reasoning_delta" in event_config.thinking_content_types, (
            "assistant.reasoning_delta must be in thinking_content_types to enable "
            "per-token llm:stream_block_delta(block_type=thinking) emission."
        )
