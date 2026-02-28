"""
Tests for SDK Driver components.

These tests validate the SDK Driver architecture that prevents
the 305-turn retry loop problem documented in Session a1a0af17.

Evidence Base:
- 305 ASSISTANT_TURN_START events from a single request
- 607 tool calls captured (303 × 2 + 1 due to accumulation across all turns)
- 20 minutes runtime for what should have been a 5-second request
- Root cause: SDK feeds denial error to LLM → LLM retries indefinitely

Solution validated here:
- First-turn-only capture → 2 tools, not 607
- Circuit breaker → abort at turn 3, not turn 305
- Deduplication → safety net for any duplicate slippage
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, Mock

import pytest
from copilot.generated.session_events import SessionEventType

from amplifier_module_provider_github_copilot._constants import (
    SDK_MAX_TURNS_HARD_LIMIT,
)
from amplifier_module_provider_github_copilot.exceptions import (
    CopilotRateLimitError,
    CopilotSdkLoopError,
)
from amplifier_module_provider_github_copilot.sdk_driver import (
    CapturedToolCall,
    CircuitBreaker,
    LoopController,
    LoopState,
    SdkEventHandler,
    ToolCaptureStrategy,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _make_mock_event(event_type, data=None):
    """Create a mock SDK event with the given type and data."""
    mock = Mock()
    mock.type = event_type
    mock.data = data or Mock()
    return mock


def _make_mock_tool_request(tool_id, name, args):
    """Create a mock tool request object mimicking SDK's tool_request structure."""
    mock = Mock()
    mock.tool_call_id = tool_id
    mock.name = name
    mock.arguments = args
    return mock


# ═══════════════════════════════════════════════════════════════════════════════
# TestCapturedToolCall
# ═══════════════════════════════════════════════════════════════════════════════


class TestCapturedToolCall:
    """Tests for CapturedToolCall dataclass."""

    def test_hash_by_name_and_arguments(self):
        """Same name+arguments should hash equally for deduplication.

        ID and turn are identity metadata — they should NOT affect
        deduplication hashing. Two tool calls with the same (name, args)
        from different IDs and different turns are considered duplicates.
        """
        t1 = CapturedToolCall(
            id="id-aaa", name="delegate", arguments={"agent": "bug-hunter"}, turn=0
        )
        t2 = CapturedToolCall(
            id="id-bbb", name="delegate", arguments={"agent": "bug-hunter"}, turn=5
        )

        assert hash(t1) == hash(t2)

    def test_different_arguments_different_hash(self):
        """Tool calls with different arguments must hash differently."""
        t1 = CapturedToolCall(id="id1", name="delegate", arguments={"agent": "bug-hunter"}, turn=0)
        t2 = CapturedToolCall(
            id="id2", name="delegate", arguments={"agent": "code-reviewer"}, turn=0
        )

        assert hash(t1) != hash(t2)
        assert t1 != t2

    def test_different_names_different_hash(self):
        """Tool calls with different names must hash differently."""
        t1 = CapturedToolCall(id="id1", name="delegate", arguments={"a": 1}, turn=0)
        t2 = CapturedToolCall(id="id2", name="report_intent", arguments={"a": 1}, turn=0)

        assert hash(t1) != hash(t2)
        assert t1 != t2

    def test_equality_ignores_id_and_turn(self):
        """Equality is based on (name, arguments) only — not id or turn.

        This is critical for deduplication: the SDK assigns new IDs on each
        retry turn, but the logical tool call is the same.
        """
        t1 = CapturedToolCall(id="call_001", name="foo", arguments={"x": 42}, turn=1)
        t2 = CapturedToolCall(id="call_999", name="foo", arguments={"x": 42}, turn=300)

        assert t1 == t2

    def test_not_equal_to_non_tool_call(self):
        """CapturedToolCall should not be equal to arbitrary objects."""
        t1 = CapturedToolCall(id="id1", name="foo", arguments={}, turn=0)

        assert t1 != "not a tool call"
        assert t1 != 42
        assert t1 != {"name": "foo", "arguments": {}}
        assert t1 != None  # noqa: E711


# ═══════════════════════════════════════════════════════════════════════════════
# TestLoopController
# ═══════════════════════════════════════════════════════════════════════════════


class TestLoopController:
    """Tests for LoopController turn tracking and abort logic."""

    def test_initial_state(self):
        """Controller starts with zero turns and no abort."""
        ctrl = LoopController(max_turns=3)

        assert ctrl.state.turn_count == 0
        assert not ctrl.should_abort()

    def test_turn_counting(self):
        """on_turn_start increments turn count and returns True within limit."""
        ctrl = LoopController(max_turns=5)

        assert ctrl.on_turn_start() is True  # Turn 1
        assert ctrl.state.turn_count == 1

        assert ctrl.on_turn_start() is True  # Turn 2
        assert ctrl.state.turn_count == 2

        assert ctrl.on_turn_start() is True  # Turn 3
        assert ctrl.state.turn_count == 3

    def test_circuit_breaker_trips(self):
        """Exceeding max_turns returns False and sets abort flag.

        Evidence: 305 turns observed in incident. Controller must stop
        at max_turns + 1 and signal abort.
        """
        ctrl = LoopController(max_turns=2)

        assert ctrl.on_turn_start() is True  # Turn 1 — OK
        assert ctrl.on_turn_start() is True  # Turn 2 — OK (at limit)
        assert ctrl.on_turn_start() is False  # Turn 3 — EXCEEDED

        assert ctrl.should_abort()
        assert ctrl.state.turn_count == 3

    def test_abort_callback_invoked(self):
        """Abort callback is called exactly once when limit is exceeded."""
        callback = Mock()
        ctrl = LoopController(max_turns=1)
        ctrl.set_abort_callback(callback)

        ctrl.on_turn_start()  # Turn 1 — OK
        callback.assert_not_called()

        ctrl.on_turn_start()  # Turn 2 — exceeds limit, triggers callback
        callback.assert_called_once()

    def test_hard_limit_enforced(self):
        """max_turns is capped at SDK_MAX_TURNS_HARD_LIMIT regardless of input.

        Even if caller requests 1000 turns, the hard limit prevents runaway.
        """
        ctrl = LoopController(max_turns=1000)

        assert ctrl.max_turns == SDK_MAX_TURNS_HARD_LIMIT

    def test_request_abort(self):
        """request_abort sets the abort flag manually."""
        ctrl = LoopController(max_turns=10)

        assert not ctrl.should_abort()
        ctrl.request_abort("test_reason")
        assert ctrl.should_abort()

    def test_multiple_aborts_callback_once(self):
        """Abort callback is invoked only once even with multiple abort triggers.

        The SDK might fire multiple events after the limit is exceeded.
        We must not spam the abort callback.
        """
        callback = Mock()
        ctrl = LoopController(max_turns=1)
        ctrl.set_abort_callback(callback)

        # Turn 1 — OK
        ctrl.on_turn_start()

        # Turns 2, 3, 4 — all exceed limit
        ctrl.on_turn_start()
        ctrl.on_turn_start()
        ctrl.on_turn_start()

        # Also trigger via request_abort
        ctrl.request_abort("extra_abort")

        # Callback should only be called ONCE
        callback.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════════
# TestToolCaptureStrategy
# ═══════════════════════════════════════════════════════════════════════════════


class TestToolCaptureStrategy:
    """Tests for ToolCaptureStrategy capture and filtering logic."""

    def test_first_turn_capture(self):
        """Captures tools from the first turn's ASSISTANT_MESSAGE."""
        strategy = ToolCaptureStrategy(first_turn_only=True)
        strategy.set_current_turn(1)

        reqs = [_make_mock_tool_request("id1", "delegate", {"agent": "tester"})]
        captured = strategy.capture_from_event(reqs)

        assert len(captured) == 1
        assert captured[0].name == "delegate"
        assert captured[0].arguments == {"agent": "tester"}
        assert captured[0].turn == 1

    def test_first_turn_only_ignores_subsequent(self):
        """With first_turn_only=True, tool requests from turn 2+ are ignored.

        Evidence: 607 tools came from capturing ALL 305 turns.
        Only the first turn's 2 tools were valid.
        """
        strategy = ToolCaptureStrategy(first_turn_only=True)

        # First turn — captured
        strategy.set_current_turn(1)
        reqs1 = [_make_mock_tool_request("id1", "delegate", {"agent": "tester"})]
        captured1 = strategy.capture_from_event(reqs1)

        # Second turn — should be ignored
        strategy.set_current_turn(2)
        reqs2 = [_make_mock_tool_request("id2", "report_intent", {"intent": "fix"})]
        captured2 = strategy.capture_from_event(reqs2)

        # Third turn — also ignored
        strategy.set_current_turn(3)
        reqs3 = [_make_mock_tool_request("id3", "delegate", {"agent": "reviewer"})]
        captured3 = strategy.capture_from_event(reqs3)

        assert len(captured1) == 1
        assert len(captured2) == 0
        assert len(captured3) == 0
        assert len(strategy.captured_tools) == 1

    def test_deduplication(self):
        """Duplicate tools (same name+args) are filtered within a turn.

        Even if the SDK emits the same tool call twice in one message,
        deduplication prevents double-counting.
        """
        strategy = ToolCaptureStrategy(first_turn_only=False, deduplicate=True)
        strategy.set_current_turn(1)

        reqs = [
            _make_mock_tool_request("id1", "delegate", {"agent": "bug-hunter"}),
            _make_mock_tool_request("id2", "delegate", {"agent": "bug-hunter"}),  # Exact dup
        ]
        captured = strategy.capture_from_event(reqs)

        assert len(captured) == 1
        assert len(strategy.captured_tools) == 1

    def test_no_deduplication_when_disabled(self):
        """Without deduplication, all tool requests are captured as-is."""
        strategy = ToolCaptureStrategy(first_turn_only=False, deduplicate=False)
        strategy.set_current_turn(1)

        reqs = [
            _make_mock_tool_request("id1", "delegate", {"agent": "bug-hunter"}),
            _make_mock_tool_request("id2", "delegate", {"agent": "bug-hunter"}),
        ]
        captured = strategy.capture_from_event(reqs)

        assert len(captured) == 2
        assert len(strategy.captured_tools) == 2

    def test_arguments_as_string_parsed(self):
        """JSON string arguments are parsed into dict.

        The SDK sometimes returns arguments as a JSON string rather than
        a parsed dict. The strategy must handle this transparently.
        """
        strategy = ToolCaptureStrategy()
        strategy.set_current_turn(1)

        mock_tr = Mock()
        mock_tr.tool_call_id = "id1"
        mock_tr.name = "write_file"
        mock_tr.arguments = '{"path": "/tmp/test.txt", "content": "hello"}'

        captured = strategy.capture_from_event([mock_tr])

        assert len(captured) == 1
        assert captured[0].arguments == {"path": "/tmp/test.txt", "content": "hello"}
        assert isinstance(captured[0].arguments, dict)

    def test_arguments_invalid_json_fallback(self):
        """Invalid JSON string arguments fall back to {"raw": ...} wrapper.

        Malformed arguments should not crash the strategy — they get
        wrapped in a raw container for downstream handling.
        """
        strategy = ToolCaptureStrategy()
        strategy.set_current_turn(1)

        mock_tr = Mock()
        mock_tr.tool_call_id = "id1"
        mock_tr.name = "foo"
        mock_tr.arguments = "this is not valid json {{"

        captured = strategy.capture_from_event([mock_tr])

        assert len(captured) == 1
        assert captured[0].arguments == {"raw": "this is not valid json {{"}

    def test_has_tools_property(self):
        """has_tools is False when empty, True after capture."""
        strategy = ToolCaptureStrategy()

        assert strategy.has_tools is False

        strategy.set_current_turn(1)
        strategy.capture_from_event([_make_mock_tool_request("id1", "foo", {"a": 1})])

        assert strategy.has_tools is True

    def test_empty_tool_requests(self):
        """Empty tool_requests list produces no captured tools."""
        strategy = ToolCaptureStrategy()
        strategy.set_current_turn(1)

        captured = strategy.capture_from_event([])

        assert len(captured) == 0
        assert strategy.has_tools is False


# ═══════════════════════════════════════════════════════════════════════════════
# TestCircuitBreaker
# ═══════════════════════════════════════════════════════════════════════════════


class TestCircuitBreaker:
    """Tests for CircuitBreaker safety mechanism."""

    def test_initial_not_tripped(self):
        """Circuit breaker starts in clean, un-tripped state."""
        cb = CircuitBreaker(max_turns=3, timeout_seconds=60)
        cb.start()

        assert not cb.is_tripped
        assert cb.trip_reason is None

    def test_trips_on_turn_limit(self):
        """Trips when turn count exceeds maximum.

        Evidence: 305 turns in the incident. Circuit breaker at 3 would
        have caught this after turn 4 (> 3).
        """
        cb = CircuitBreaker(max_turns=3)
        cb.start()

        assert cb.check_turn(1) is True  # OK
        assert cb.check_turn(2) is True  # OK
        assert cb.check_turn(3) is True  # OK (at limit)
        assert cb.check_turn(4) is False  # TRIPPED

        assert cb.is_tripped
        assert "turn_count=4" in cb.trip_reason
        assert "max=3" in cb.trip_reason

    def test_trips_on_timeout(self):
        """Trips when elapsed time exceeds timeout.

        Uses a very short timeout (10ms) and sleep (20ms) to test
        without slowing down the test suite.
        """
        cb = CircuitBreaker(timeout_seconds=0.01)
        cb.start()

        time.sleep(0.02)  # Wait 20ms, exceeding 10ms limit

        assert cb.check_timeout() is False
        assert cb.is_tripped
        assert "timeout" in cb.trip_reason

    def test_trip_reason_recorded(self):
        """Trip reason is correctly recorded for diagnostics."""
        cb = CircuitBreaker(max_turns=2)
        cb.start()

        cb.check_turn(3)  # Exceeds limit

        assert cb.is_tripped
        assert cb.trip_reason is not None
        assert "turn_count=3" in cb.trip_reason

    def test_trips_only_once(self):
        """Multiple violations do not change the original trip reason.

        The first violation is the root cause — subsequent violations
        are consequences and should not overwrite the diagnostic.
        """
        cb = CircuitBreaker(max_turns=2, timeout_seconds=0.01)
        cb.start()

        # First trip: turn limit
        cb.check_turn(3)
        original_reason = cb.trip_reason

        # Second violation: timeout (should NOT overwrite)
        time.sleep(0.02)
        cb.check_timeout()

        # Another turn violation
        cb.check_turn(10)

        assert cb.trip_reason == original_reason
        assert "turn_count=3" in cb.trip_reason


# ═══════════════════════════════════════════════════════════════════════════════
# TestSdkEventHandler
# ═══════════════════════════════════════════════════════════════════════════════


class TestSdkEventHandler:
    """Tests for SdkEventHandler event coordination."""

    def test_turn_tracking(self):
        """ASSISTANT_TURN_START increments turn_count."""
        handler = SdkEventHandler(max_turns=5)

        event = _make_mock_event(SessionEventType.ASSISTANT_TURN_START)
        handler.on_event(event)

        assert handler.turn_count == 1

        handler.on_event(event)
        assert handler.turn_count == 2

    def test_tool_capture_from_assistant_message(self):
        """Captures tool calls from ASSISTANT_MESSAGE events.

        This is the CRITICAL PATH: tools arrive via ASSISTANT_MESSAGE
        before the preToolUse hook fires.
        """
        handler = SdkEventHandler(max_turns=5)

        # First: turn start (sets current turn)
        handler.on_event(_make_mock_event(SessionEventType.ASSISTANT_TURN_START))

        # Then: message with tool requests
        data = Mock()
        data.tool_requests = [
            _make_mock_tool_request("call_1", "delegate", {"agent": "bug-hunter"}),
            _make_mock_tool_request("call_2", "report_intent", {"intent": "fix bug"}),
        ]
        handler.on_event(_make_mock_event(SessionEventType.ASSISTANT_MESSAGE, data))

        assert len(handler.captured_tools) == 2
        assert handler.captured_tools[0].name == "delegate"
        assert handler.captured_tools[1].name == "report_intent"

    def test_abort_requested_after_first_capture(self):
        """Abort is requested immediately after first-turn tool capture.

        With first_turn_only=True, the handler signals abort as soon as
        tools are captured from the first turn, preventing the SDK from
        looping 304 more times.
        """
        handler = SdkEventHandler(max_turns=5, first_turn_only=True)

        # Turn start
        handler.on_event(_make_mock_event(SessionEventType.ASSISTANT_TURN_START))

        # Message with tools
        data = Mock()
        data.tool_requests = [
            _make_mock_tool_request("call_1", "foo", {"x": 1}),
        ]
        handler.on_event(_make_mock_event(SessionEventType.ASSISTANT_MESSAGE, data))

        assert handler.should_abort is True

    def test_circuit_breaker_trips_on_many_turns(self):
        """Circuit breaker trips when turn count exceeds max_turns.

        Evidence: 305 turns observed. With max_turns=2, turn 3 should
        trip the breaker.
        """
        handler = SdkEventHandler(max_turns=2)

        # Simulate 3 turns
        for _ in range(3):
            handler.on_event(_make_mock_event(SessionEventType.ASSISTANT_TURN_START))

        assert handler.loop_controller.should_abort()
        assert handler.circuit_breaker.is_tripped
        assert handler.turn_count == 3

    def test_text_content_accumulation(self):
        """ASSISTANT_MESSAGE_DELTA events accumulate text content."""
        handler = SdkEventHandler(max_turns=5)

        data1 = Mock()
        data1.delta_content = "Hello, "
        handler.on_event(_make_mock_event(SessionEventType.ASSISTANT_MESSAGE_DELTA, data1))

        data2 = Mock()
        data2.delta_content = "world!"
        handler.on_event(_make_mock_event(SessionEventType.ASSISTANT_MESSAGE_DELTA, data2))

        assert handler.text_content == ["Hello, ", "world!"]
        assert "".join(handler.text_content) == "Hello, world!"

    def test_thinking_content_accumulation(self):
        """ASSISTANT_REASONING_DELTA events accumulate thinking content."""
        handler = SdkEventHandler(max_turns=5)

        data1 = Mock()
        data1.delta_content = "Let me think about "
        handler.on_event(_make_mock_event(SessionEventType.ASSISTANT_REASONING_DELTA, data1))

        data2 = Mock()
        data2.delta_content = "this problem..."
        handler.on_event(_make_mock_event(SessionEventType.ASSISTANT_REASONING_DELTA, data2))

        assert handler.thinking_content == ["Let me think about ", "this problem..."]
        assert "".join(handler.thinking_content) == "Let me think about this problem..."

    def test_reasoning_full_block(self):
        """ASSISTANT_REASONING adds full thinking blocks to thinking_content."""
        handler = SdkEventHandler(max_turns=5)

        data = Mock()
        data.content = "I analyzed the code and found the issue."
        handler.on_event(_make_mock_event(SessionEventType.ASSISTANT_REASONING, data))

        assert len(handler.thinking_content) == 1
        assert handler.thinking_content[0] == "I analyzed the code and found the issue."

    def test_usage_tracking(self):
        """ASSISTANT_USAGE events update usage_data dict."""
        handler = SdkEventHandler(max_turns=5)

        data = Mock()
        data.input_tokens = 1500
        data.output_tokens = 350
        handler.on_event(_make_mock_event(SessionEventType.ASSISTANT_USAGE, data))

        assert handler.usage_data["input_tokens"] == 1500
        assert handler.usage_data["output_tokens"] == 350

    def test_session_error_sets_error(self):
        """SESSION_ERROR sets internal error state and unblocks waiters."""
        handler = SdkEventHandler(max_turns=5)

        data = Mock()
        data.message = "Rate limit exceeded"
        handler.on_event(_make_mock_event(SessionEventType.SESSION_ERROR, data))

        assert handler._error_event is not None
        assert "Rate limit exceeded" in str(handler._error_event)

    @pytest.mark.asyncio
    async def test_wait_for_capture_returns_on_idle(self):
        """wait_for_capture_or_idle completes when SESSION_IDLE fires.

        In the no-tools scenario, the SDK sends text and then
        SESSION_IDLE. The waiter should return cleanly.
        """
        handler = SdkEventHandler(max_turns=5)

        async def fire_idle():
            await asyncio.sleep(0.01)
            handler.on_event(_make_mock_event(SessionEventType.SESSION_IDLE))

        asyncio.create_task(fire_idle())
        await handler.wait_for_capture_or_idle(timeout=2.0)
        # Should complete without error or timeout

    @pytest.mark.asyncio
    async def test_wait_for_capture_returns_on_tool_capture(self):
        """wait_for_capture_or_idle returns immediately on first tool capture.

        This validates the fast-path: tools captured on first turn
        cause immediate return without waiting for SESSION_IDLE.
        """
        handler = SdkEventHandler(max_turns=5, first_turn_only=True)

        async def fire_events():
            await asyncio.sleep(0.01)
            # Turn start
            handler.on_event(_make_mock_event(SessionEventType.ASSISTANT_TURN_START))
            # Message with tools
            data = Mock()
            data.tool_requests = [
                _make_mock_tool_request("call_1", "delegate", {"agent": "tester"}),
            ]
            handler.on_event(_make_mock_event(SessionEventType.ASSISTANT_MESSAGE, data))

        asyncio.create_task(fire_events())
        await handler.wait_for_capture_or_idle(timeout=2.0)

        assert len(handler.captured_tools) == 1
        assert handler.captured_tools[0].name == "delegate"

    @pytest.mark.asyncio
    async def test_wait_for_capture_timeout(self):
        """wait_for_capture_or_idle raises TimeoutError when no events arrive."""
        handler = SdkEventHandler(max_turns=5)

        with pytest.raises(asyncio.TimeoutError):
            await handler.wait_for_capture_or_idle(timeout=0.05)


# ═══════════════════════════════════════════════════════════════════════════════
# TestObservabilityEvents
# ═══════════════════════════════════════════════════════════════════════════════


class TestObservabilityEvents:
    """Tests for SDK Driver observability event emission.

    TASK-08 requires 4 hookable events:
    1. sdk:turn_start — each ASSISTANT_TURN_START
    2. sdk:capture_complete — after first-turn capture
    3. sdk:circuit_breaker_trip — when limit exceeded
    4. sdk:abort_requested — when abort is triggered

    These events are critical for debugging in a forensic environment.
    Without them, we cannot diagnose issues like the 305-turn incident.
    """

    def test_turn_start_event_emitted(self):
        """sdk:turn_start is emitted on each ASSISTANT_TURN_START."""
        emitted: list[tuple[str, dict]] = []
        handler = SdkEventHandler(
            max_turns=5,
            emit_event=lambda name, data: emitted.append((name, data)),
        )

        handler.on_event(_make_mock_event(SessionEventType.ASSISTANT_TURN_START))

        assert len(emitted) == 1
        assert emitted[0][0] == "sdk:turn_start"
        assert emitted[0][1] == {"turn": 1, "max_turns": 5}

    def test_capture_complete_event_emitted(self):
        """sdk:capture_complete is emitted after first-turn tool capture."""
        emitted: list[tuple[str, dict]] = []
        handler = SdkEventHandler(
            max_turns=5,
            first_turn_only=True,
            emit_event=lambda name, data: emitted.append((name, data)),
        )

        # Turn start
        handler.on_event(_make_mock_event(SessionEventType.ASSISTANT_TURN_START))

        # Message with tools
        data = Mock()
        data.tool_requests = [
            _make_mock_tool_request("call_1", "delegate", {"agent": "tester"}),
        ]
        handler.on_event(_make_mock_event(SessionEventType.ASSISTANT_MESSAGE, data))

        event_names = [e[0] for e in emitted]
        assert "sdk:capture_complete" in event_names

        capture_event = next(e for e in emitted if e[0] == "sdk:capture_complete")
        assert capture_event[1]["turn"] == 1
        assert capture_event[1]["tool_count"] == 1
        assert capture_event[1]["tools"] == ["delegate"]

    def test_circuit_breaker_trip_event_emitted(self):
        """sdk:circuit_breaker_trip is emitted when turn limit exceeded.

        This event is critical for post-incident analysis. Without it,
        we'd have to grep logs to find when the breaker tripped.
        """
        emitted: list[tuple[str, dict]] = []
        handler = SdkEventHandler(
            max_turns=2,
            emit_event=lambda name, data: emitted.append((name, data)),
        )

        # Turn 1, 2 — within limit
        handler.on_event(_make_mock_event(SessionEventType.ASSISTANT_TURN_START))
        handler.on_event(_make_mock_event(SessionEventType.ASSISTANT_TURN_START))

        # Turn 3 — exceeds limit, should trip
        handler.on_event(_make_mock_event(SessionEventType.ASSISTANT_TURN_START))

        event_names = [e[0] for e in emitted]
        assert "sdk:circuit_breaker_trip" in event_names, (
            f"Expected sdk:circuit_breaker_trip in emitted events, got: {event_names}"
        )

        trip_event = next(e for e in emitted if e[0] == "sdk:circuit_breaker_trip")
        assert trip_event[1]["turn"] == 3
        assert "reason" in trip_event[1]

    def test_abort_requested_event_emitted_after_capture(self):
        """sdk:abort_requested is emitted when abort triggered by first-turn capture.

        Without this event, we can't tell WHY the session was aborted
        when reviewing event logs post-incident.
        """
        emitted: list[tuple[str, dict]] = []
        handler = SdkEventHandler(
            max_turns=5,
            first_turn_only=True,
            emit_event=lambda name, data: emitted.append((name, data)),
        )

        # Turn start
        handler.on_event(_make_mock_event(SessionEventType.ASSISTANT_TURN_START))

        # Message with tools → triggers capture → triggers abort
        data = Mock()
        data.tool_requests = [
            _make_mock_tool_request("call_1", "foo", {"x": 1}),
        ]
        handler.on_event(_make_mock_event(SessionEventType.ASSISTANT_MESSAGE, data))

        event_names = [e[0] for e in emitted]
        assert "sdk:abort_requested" in event_names, (
            f"Expected sdk:abort_requested in emitted events, got: {event_names}"
        )

        abort_event = next(e for e in emitted if e[0] == "sdk:abort_requested")
        assert abort_event[1]["turn"] == 1
        assert abort_event[1]["reason"] == "first_turn_capture_complete"

    def test_abort_requested_event_emitted_after_circuit_breaker(self):
        """sdk:abort_requested is emitted when abort triggered by circuit breaker.

        Both circuit_breaker_trip and abort_requested should fire in sequence
        so the event timeline tells the full story.
        """
        emitted: list[tuple[str, dict]] = []
        handler = SdkEventHandler(
            max_turns=2,
            emit_event=lambda name, data: emitted.append((name, data)),
        )

        # Turn 1, 2 — within limit
        handler.on_event(_make_mock_event(SessionEventType.ASSISTANT_TURN_START))
        handler.on_event(_make_mock_event(SessionEventType.ASSISTANT_TURN_START))

        # Turn 3 — exceeds limit
        handler.on_event(_make_mock_event(SessionEventType.ASSISTANT_TURN_START))

        event_names = [e[0] for e in emitted]

        # Both events must be present
        assert "sdk:circuit_breaker_trip" in event_names
        assert "sdk:abort_requested" in event_names

        # circuit_breaker_trip should come BEFORE abort_requested
        trip_idx = event_names.index("sdk:circuit_breaker_trip")
        abort_idx = event_names.index("sdk:abort_requested")
        assert trip_idx < abort_idx, (
            "Circuit breaker trip should be emitted BEFORE abort_requested "
            f"(trip at {trip_idx}, abort at {abort_idx})"
        )

        abort_event = next(e for e in emitted if e[0] == "sdk:abort_requested")
        assert abort_event[1]["reason"] == "circuit_breaker_turn_limit"

    def test_all_four_events_in_tool_capture_scenario(self):
        """All 4 observability events fire in a realistic tool capture scenario.

        Simulates: Turn 1 → capture tools → abort → Turn 2,3,4 → circuit breaker.
        Should see: turn_start, capture_complete, abort_requested, turn_start(×3),
        circuit_breaker_trip, abort_requested.
        """
        emitted: list[tuple[str, dict]] = []
        handler = SdkEventHandler(
            max_turns=3,
            first_turn_only=True,
            emit_event=lambda name, data: emitted.append((name, data)),
        )

        # Turn 1 — tools captured
        handler.on_event(_make_mock_event(SessionEventType.ASSISTANT_TURN_START))
        data = Mock()
        data.tool_requests = [
            _make_mock_tool_request("call_1", "delegate", {"agent": "bug-hunter"}),
        ]
        handler.on_event(_make_mock_event(SessionEventType.ASSISTANT_MESSAGE, data))

        # Turns 2, 3, 4 — SDK keeps going despite abort being requested
        for _ in range(3):
            handler.on_event(_make_mock_event(SessionEventType.ASSISTANT_TURN_START))

        event_names = [e[0] for e in emitted]

        # ALL 4 event types must be present
        assert "sdk:turn_start" in event_names, "Missing sdk:turn_start"
        assert "sdk:capture_complete" in event_names, "Missing sdk:capture_complete"
        assert "sdk:abort_requested" in event_names, "Missing sdk:abort_requested"
        assert "sdk:circuit_breaker_trip" in event_names, "Missing sdk:circuit_breaker_trip"

    def test_no_events_when_emit_callback_none(self):
        """No events are emitted when emit_event callback is None.

        This ensures we don't crash when observability is disabled
        (e.g., no coordinator configured).
        """
        handler = SdkEventHandler(max_turns=2, emit_event=None)

        # This should work without errors
        handler.on_event(_make_mock_event(SessionEventType.ASSISTANT_TURN_START))
        handler.on_event(_make_mock_event(SessionEventType.ASSISTANT_TURN_START))
        handler.on_event(_make_mock_event(SessionEventType.ASSISTANT_TURN_START))

        # No crash, circuit breaker still works
        assert handler.circuit_breaker.is_tripped
        assert handler.turn_count == 3


# ═══════════════════════════════════════════════════════════════════════════════
# TestPreventForensicScenario
# ═══════════════════════════════════════════════════════════════════════════════


class TestPreventForensicScenario:
    """
    Tests that specifically verify we prevent the 305-turn scenario.

    Evidence: Session a1a0af17
    - 305 ASSISTANT_TURN_START events
    - 607 tool calls captured (303 × 2 + 1)
    - 20 minutes runtime
    - Root cause: preToolUse deny → SDK feeds error to LLM → LLM retries

    These tests simulate that exact scenario and verify our protection works.
    """

    def test_prevents_305_turn_accumulation(self):
        """With first-turn-only capture, we get 2 tools — not 607.

        This is THE critical test. It simulates the exact failure mode
        from the forensic incident:
        - 305 turns
        - Each turn produces 2 tool calls (delegate + report_intent)
        - Without protection: 607 tool calls accumulated
        - WITH protection: Only 2 from the first turn

        Also verifies:
        - Circuit breaker trips (turn 4 > max_turns=3)
        - Abort is requested after first-turn capture
        """
        handler = SdkEventHandler(max_turns=3, first_turn_only=True)

        for turn in range(305):
            # ASSISTANT_TURN_START
            handler.on_event(_make_mock_event(SessionEventType.ASSISTANT_TURN_START))

            # ASSISTANT_MESSAGE with 2 tool requests (like the incident)
            data = Mock()
            data.tool_requests = [
                _make_mock_tool_request(
                    f"call_{turn}_1", "report_intent", {"intent": "investigate bug"}
                ),
                _make_mock_tool_request(f"call_{turn}_2", "delegate", {"agent": "bug-hunter"}),
            ]
            handler.on_event(_make_mock_event(SessionEventType.ASSISTANT_MESSAGE, data))

            # After first turn, abort should already be requested
            if turn == 0:
                assert handler.should_abort, "Should request abort after first-turn capture"

        # ═══════════════════════════════════════════════════════════
        # CRITICAL ASSERTIONS
        # ═══════════════════════════════════════════════════════════

        # Only 2 tools captured (from first turn), NOT 607
        assert len(handler.captured_tools) == 2, (
            f"Expected 2 tools from first turn, got {len(handler.captured_tools)}. "
            f"This is the 607-tool accumulation bug!"
        )
        assert handler.captured_tools[0].name == "report_intent"
        assert handler.captured_tools[1].name == "delegate"

        # Circuit breaker tripped (turn 4 > max_turns=3)
        assert handler.circuit_breaker.is_tripped, "Circuit breaker should have tripped at turn 4"

        # Abort was requested
        assert handler.should_abort, "Abort should be requested"

        # Turn count reflects all simulated turns
        assert handler.turn_count == 305

    def test_circuit_breaker_stops_runaway_loop(self):
        """Circuit breaker trips at max_turns, even without tool capture.

        Scenario: Text-only responses that somehow keep triggering
        new turns (e.g., the model keeps trying to use tools but the
        message event has no tool_requests).
        """
        handler = SdkEventHandler(max_turns=3)

        for _turn in range(10):
            handler.on_event(_make_mock_event(SessionEventType.ASSISTANT_TURN_START))

        assert handler.circuit_breaker.is_tripped
        assert handler.turn_count == 10

        # Circuit breaker should have tripped at turn 4 (> max_turns=3)
        assert "turn_count=4" in handler.circuit_breaker.trip_reason


# ═══════════════════════════════════════════════════════════════════════════════
# Coverage gap tests: bind_session, _do_abort, wait_for_capture_or_idle
# error paths, circuit breaker edge cases, LoopState.elapsed_seconds
# ═══════════════════════════════════════════════════════════════════════════════


class TestLoopControllerRequestAbortWithCallback:
    """Tests for LoopController.request_abort() callback invocation (lines 148-149)."""

    def test_request_abort_invokes_callback(self):
        """request_abort should invoke the abort callback."""
        callback = Mock()
        ctrl = LoopController(max_turns=10)
        ctrl.set_abort_callback(callback)

        ctrl.request_abort("test reason")

        callback.assert_called_once()
        assert ctrl.should_abort()

    def test_request_abort_callback_only_once(self):
        """Repeated request_abort should invoke callback only once."""
        callback = Mock()
        ctrl = LoopController(max_turns=10)
        ctrl.set_abort_callback(callback)

        ctrl.request_abort("first abort")
        ctrl.request_abort("second abort")
        ctrl.request_abort("third abort")

        callback.assert_called_once()

    def test_request_abort_without_callback(self):
        """request_abort should work without a callback set."""
        ctrl = LoopController(max_turns=10)

        ctrl.request_abort("no callback")

        assert ctrl.should_abort()


class TestLoopStateElapsedSeconds:
    """Tests for LoopState.elapsed_seconds property."""

    def test_elapsed_seconds_increases(self):
        """elapsed_seconds should reflect real time passage."""
        state = LoopState()

        # Should be > 0 after creation
        assert state.elapsed_seconds >= 0.0

    def test_elapsed_seconds_is_float(self):
        """elapsed_seconds should return a float."""
        state = LoopState()
        assert isinstance(state.elapsed_seconds, float)


class TestCircuitBreakerCheckTimeoutNoStart:
    """Tests for CircuitBreaker.check_timeout() without start() (line 309)."""

    def test_check_timeout_without_start_returns_true(self):
        """check_timeout should return True when start() was never called."""
        cb = CircuitBreaker(max_turns=3, timeout_seconds=0.001)

        # Without calling start(), _start_time is None
        assert cb.check_timeout() is True
        assert not cb.is_tripped


class TestSdkEventHandlerBindSession:
    """Tests for SdkEventHandler.bind_session() and _do_abort() (lines 390-408)."""

    @pytest.mark.asyncio
    async def test_bind_session_enables_abort(self):
        """bind_session should enable session.abort() on limit exceeded."""
        handler = SdkEventHandler(max_turns=1, first_turn_only=True)

        mock_session = AsyncMock()
        handler.bind_session(mock_session)

        # Trigger turn 1 (OK) then turn 2 (exceeds max_turns=1)
        handler.on_event(_make_mock_event(SessionEventType.ASSISTANT_TURN_START))
        handler.on_event(_make_mock_event(SessionEventType.ASSISTANT_TURN_START))

        # Allow the abort task to run
        await asyncio.sleep(0.05)

        mock_session.abort.assert_called_once()

    @pytest.mark.asyncio
    async def test_bind_session_abort_on_first_turn_capture(self):
        """bind_session should enable abort after first-turn tool capture."""
        handler = SdkEventHandler(max_turns=5, first_turn_only=True)

        mock_session = AsyncMock()
        handler.bind_session(mock_session)

        # Turn start + tool capture
        handler.on_event(_make_mock_event(SessionEventType.ASSISTANT_TURN_START))
        data = Mock()
        data.tool_requests = [
            _make_mock_tool_request("call_1", "delegate", {"agent": "tester"}),
        ]
        handler.on_event(_make_mock_event(SessionEventType.ASSISTANT_MESSAGE, data))

        # Allow the abort task to run
        await asyncio.sleep(0.05)

        mock_session.abort.assert_called_once()

    @pytest.mark.asyncio
    async def test_do_abort_handles_exception(self):
        """_do_abort should handle abort() failures gracefully."""
        handler = SdkEventHandler(max_turns=1)

        mock_session = AsyncMock()
        mock_session.abort.side_effect = RuntimeError("Abort failed: connection lost")
        handler.bind_session(mock_session)

        # Trigger limit exceeded
        handler.on_event(_make_mock_event(SessionEventType.ASSISTANT_TURN_START))
        handler.on_event(_make_mock_event(SessionEventType.ASSISTANT_TURN_START))

        # Allow the abort task to run — should not crash
        await asyncio.sleep(0.05)

        mock_session.abort.assert_called_once()

    @pytest.mark.asyncio
    async def test_do_abort_no_session(self):
        """_do_abort should do nothing if session is None."""
        handler = SdkEventHandler(max_turns=5)
        # Don't bind a session — _session is None

        # Should not crash
        await handler._do_abort()


class TestWaitForCaptureOrIdleErrorPaths:
    """Tests for wait_for_capture_or_idle error handling (lines 591, 608, 611)."""

    @pytest.mark.asyncio
    async def test_raises_error_event_after_capture(self):
        """Should raise stored error after capture event is set (line 591)."""
        handler = SdkEventHandler(max_turns=5)

        async def fire_error():
            await asyncio.sleep(0.01)
            data = Mock()
            data.message = "Server closed connection"
            handler.on_event(_make_mock_event(SessionEventType.SESSION_ERROR, data))

        asyncio.create_task(fire_error())

        with pytest.raises(Exception, match="Server closed connection"):
            await handler.wait_for_capture_or_idle(timeout=2.0)

    @pytest.mark.asyncio
    async def test_raises_loop_error_on_circuit_breaker_trip(self):
        """Should raise CopilotSdkLoopError when circuit breaker tripped (lines 608-611)."""
        handler = SdkEventHandler(max_turns=2)

        async def fire_events():
            await asyncio.sleep(0.01)
            # Fire 3 turns to trip circuit breaker
            for _ in range(3):
                handler.on_event(_make_mock_event(SessionEventType.ASSISTANT_TURN_START))
            # Then idle
            handler.on_event(_make_mock_event(SessionEventType.SESSION_IDLE))

        asyncio.create_task(fire_events())

        with pytest.raises(CopilotSdkLoopError) as exc_info:
            await handler.wait_for_capture_or_idle(timeout=2.0)

        assert exc_info.value.turn_count == 3
        assert exc_info.value.max_turns == 2
        assert "limit exceeded" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_timeout_with_circuit_breaker_trip(self):
        """Timeout should raise CopilotSdkLoopError when circuit breaker timeout trips."""
        handler = SdkEventHandler(max_turns=100)

        # Set circuit breaker timeout to very small so check_timeout() trips it
        handler.circuit_breaker.timeout_seconds = 0.001

        # The wait will timeout, then check_timeout() sees elapsed > 0.001s → trips
        with pytest.raises(CopilotSdkLoopError) as exc_info:
            await handler.wait_for_capture_or_idle(timeout=0.05)

        assert "timeout" in str(exc_info.value).lower()


class TestSdkEventHandlerContentFallback:
    """Tests for ASSISTANT_MESSAGE content fallback when no deltas (line 509)."""

    def test_assistant_message_content_fallback_when_no_deltas(self):
        """ASSISTANT_MESSAGE should set text_content if no deltas received."""
        handler = SdkEventHandler(max_turns=5)

        # No ASSISTANT_MESSAGE_DELTA events first
        data = Mock()
        data.content = "Full response text without streaming"
        data.tool_requests = None
        handler.on_event(_make_mock_event(SessionEventType.ASSISTANT_MESSAGE, data))

        assert handler.text_content == ["Full response text without streaming"]

    def test_assistant_message_no_fallback_when_deltas_present(self):
        """ASSISTANT_MESSAGE should NOT overwrite text when deltas were received."""
        handler = SdkEventHandler(max_turns=5)

        # First receive deltas
        delta_data = Mock()
        delta_data.delta_content = "Streamed"
        handler.on_event(_make_mock_event(SessionEventType.ASSISTANT_MESSAGE_DELTA, delta_data))

        # Then full message (should not add content again)
        data = Mock()
        data.content = "Full response"
        data.tool_requests = None
        handler.on_event(_make_mock_event(SessionEventType.ASSISTANT_MESSAGE, data))

        assert handler.text_content == ["Streamed"]

    def test_assistant_message_no_content(self):
        """ASSISTANT_MESSAGE with None content should not crash."""
        handler = SdkEventHandler(max_turns=5)

        data = Mock()
        data.content = None
        data.tool_requests = None
        handler.on_event(_make_mock_event(SessionEventType.ASSISTANT_MESSAGE, data))

        assert handler.text_content == []


class TestSdkEventHandlerNullDeltas:
    """Tests for null delta handling in content streaming (lines 473, 478, 483)."""

    def test_null_message_delta_ignored(self):
        """ASSISTANT_MESSAGE_DELTA with null delta_content should be ignored."""
        handler = SdkEventHandler(max_turns=5)

        data = Mock()
        data.delta_content = None
        handler.on_event(_make_mock_event(SessionEventType.ASSISTANT_MESSAGE_DELTA, data))

        assert handler.text_content == []

    def test_null_reasoning_delta_ignored(self):
        """ASSISTANT_REASONING_DELTA with null delta_content should be ignored."""
        handler = SdkEventHandler(max_turns=5)

        data = Mock()
        data.delta_content = None
        handler.on_event(_make_mock_event(SessionEventType.ASSISTANT_REASONING_DELTA, data))

        assert handler.thinking_content == []

    def test_null_reasoning_content_ignored(self):
        """ASSISTANT_REASONING with null content should be ignored."""
        handler = SdkEventHandler(max_turns=5)

        data = Mock()
        data.content = None
        handler.on_event(_make_mock_event(SessionEventType.ASSISTANT_REASONING, data))

        assert handler.thinking_content == []


class TestSdkEventHandlerUsagePartialData:
    """Tests for partial usage data in ASSISTANT_USAGE (lines 492-494, 494-496)."""

    def test_usage_with_only_input_tokens(self):
        """Usage with only input_tokens should work without output_tokens."""
        handler = SdkEventHandler(max_turns=5)

        data = Mock()
        data.input_tokens = 500
        data.output_tokens = None
        handler.on_event(_make_mock_event(SessionEventType.ASSISTANT_USAGE, data))

        assert handler.usage_data["input_tokens"] == 500
        assert "output_tokens" not in handler.usage_data

    def test_usage_with_only_output_tokens(self):
        """Usage with only output_tokens should work without input_tokens."""
        handler = SdkEventHandler(max_turns=5)

        data = Mock()
        data.input_tokens = None
        data.output_tokens = 200
        handler.on_event(_make_mock_event(SessionEventType.ASSISTANT_USAGE, data))

        assert "input_tokens" not in handler.usage_data
        assert handler.usage_data["output_tokens"] == 200


class TestSdkEventHandlerSessionIdleCapture:
    """Tests for SESSION_IDLE setting both events (line 555)."""

    def test_session_idle_sets_both_events(self):
        """SESSION_IDLE should set both _idle_event and _capture_event."""
        handler = SdkEventHandler(max_turns=5)

        assert not handler._idle_event.is_set()
        assert not handler._capture_event.is_set()

        handler.on_event(_make_mock_event(SessionEventType.SESSION_IDLE))

        assert handler._idle_event.is_set()
        assert handler._capture_event.is_set()


class TestSessionErrorRateLimitDetection:
    """Tests for rate-limit detection in SESSION_ERROR handler."""

    def test_rate_limit_session_error_stores_copilot_rate_limit_error(self):
        """SESSION_ERROR with rate-limit message stores CopilotRateLimitError with retry_after=60.0."""
        handler = SdkEventHandler(max_turns=5)

        data = Mock()
        data.message = "Rate limit exceeded. retry after 60"
        handler.on_event(_make_mock_event(SessionEventType.SESSION_ERROR, data))

        assert isinstance(handler._error_event, CopilotRateLimitError)
        assert handler._error_event.retry_after == 60.0

    def test_non_rate_limit_session_error_stores_generic_exception(self):
        """SESSION_ERROR without rate-limit message stores generic Exception."""
        handler = SdkEventHandler(max_turns=5)

        data = Mock()
        data.message = "Internal server error"
        handler.on_event(_make_mock_event(SessionEventType.SESSION_ERROR, data))

        assert isinstance(handler._error_event, Exception)
        assert not isinstance(handler._error_event, CopilotRateLimitError)
        assert "Internal server error" in str(handler._error_event)

    def test_rate_limit_session_error_unblocks_idle_and_capture(self):
        """Rate-limit SESSION_ERROR still unblocks idle and capture events."""
        handler = SdkEventHandler(max_turns=5)

        assert not handler._idle_event.is_set()
        assert not handler._capture_event.is_set()

        data = Mock()
        data.message = "Rate limit exceeded"
        handler.on_event(_make_mock_event(SessionEventType.SESSION_ERROR, data))

        assert handler._idle_event.is_set()
        assert handler._capture_event.is_set()

    @pytest.mark.asyncio
    async def test_wait_raises_copilot_rate_limit_error(self):
        """wait_for_capture_or_idle raises CopilotRateLimitError for rate-limit SESSION_ERROR."""
        handler = SdkEventHandler(max_turns=5)

        async def fire_rate_limit_error():
            await asyncio.sleep(0.01)
            data = Mock()
            data.message = "Rate limit exceeded"
            handler.on_event(_make_mock_event(SessionEventType.SESSION_ERROR, data))

        asyncio.create_task(fire_rate_limit_error())

        with pytest.raises(CopilotRateLimitError):
            await handler.wait_for_capture_or_idle(timeout=2.0)

    def test_rate_limit_session_error_extracts_retry_after(self):
        """SESSION_ERROR with retry-after header extracts retry_after=15.5."""
        handler = SdkEventHandler(max_turns=5)

        data = Mock()
        data.message = "Too many requests. Retry-After: 15.5"
        handler.on_event(_make_mock_event(SessionEventType.SESSION_ERROR, data))

        assert isinstance(handler._error_event, CopilotRateLimitError)
        assert handler._error_event.retry_after == 15.5
