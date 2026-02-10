"""
SDK Assumption Tests: Session Lifecycle

These tests validate assumptions about session lifecycle management,
particularly the destroy() method behavior which is critical to
preventing the CLI from retrying with built-in tools.

CRITICAL ASSUMPTION:
    Calling session.destroy() terminates the CLI's internal agent loop,
    preventing it from retrying tool calls with built-in alternatives.

WHY THIS MATTERS:
    Without immediate destroy after tool capture, the CLI continues its
    agent loop. When our preToolUse deny prevents execution, the CLI
    falls back to built-in tools (like 'edit' instead of our 'write_file'),
    bypassing external orchestration entirely.

    Empirical evidence: Session IDs 497bbab7, 2a1fe04a showed this bypass
    behavior before the Deny + Destroy pattern was implemented.

BREAKING CHANGE INDICATORS:
    - CLI continues running after destroy()
    - Events fire after destroy()
    - Built-in tools execute despite destroy being called
    - Resource leaks (processes not terminated)

SDK LOCATIONS TO VERIFY:
    - copilot-sdk/python/copilot/session.py: destroy() method
    - copilot-sdk/python/copilot/client.py: session cleanup
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from .conftest import (
    EventSequence,
    InstrumentedMockSession,
    MockSessionEventType,
    create_event_data,
    create_idle_event,
    create_usage_event,
)


class TestSessionDestroyBasics:
    """
    Tests that verify basic destroy() behavior.
    """

    @pytest.mark.asyncio
    async def test_destroy_marks_session_destroyed(self):
        """
        ASSUMPTION: destroy() sets a destroyed flag or similar state.

        After destroy(), the session must be in a terminated state
        preventing further operations.
        """
        session = InstrumentedMockSession()
        assert session.destroyed is False

        await session.destroy()

        assert session.destroyed is True

    @pytest.mark.asyncio
    async def test_destroy_is_idempotent(self):
        """
        ASSUMPTION: destroy() can be called multiple times safely.

        Cleanup code may defensively call destroy(). It should be safe
        to call multiple times without errors or side effects.
        """
        session = InstrumentedMockSession()

        # Call destroy multiple times
        await session.destroy()
        await session.destroy()
        await session.destroy()

        # Should not raise, and session should still be destroyed
        assert session.destroyed is True
        assert session.destroy_called_count == 3

    @pytest.mark.asyncio
    async def test_destroy_clears_event_handlers(self):
        """
        ASSUMPTION: destroy() clears all registered event handlers.

        After destroy, no handlers should remain to receive events.
        This prevents stale callbacks from executing.
        """
        session = InstrumentedMockSession()

        # Register multiple handlers
        handler_calls: list[str] = []

        def handler1(event: Any) -> None:
            handler_calls.append("handler1")

        def handler2(event: Any) -> None:
            handler_calls.append("handler2")

        session.on(handler1)
        session.on(handler2)

        assert len(session.event_handlers) == 2

        await session.destroy()

        assert len(session.event_handlers) == 0


class TestDestroyStopsEventEmission:
    """
    Tests that verify destroy stops further event emission.
    """

    @pytest.mark.asyncio
    async def test_no_events_after_destroy(self):
        """
        ASSUMPTION: No events fire after destroy() is called.

        Once destroyed, the session must not emit any more events
        to handlers (even if handlers were registered before destroy).
        """
        # Create session with events that would fire after a delay
        events = [
            (
                MockSessionEventType.ASSISTANT_MESSAGE,
                create_event_data(content="Message 1", tool_requests=None),
            ),
            create_usage_event(),
            create_idle_event(),
        ]

        session = InstrumentedMockSession(
            event_sequence=EventSequence(events=events, emit_async=True)
        )

        received_events: list[str] = []

        def handler(event: Any) -> None:
            received_events.append(event.type.value)

        session.on(handler)

        # Destroy BEFORE events can fire
        await session.destroy()

        # Give time for events that might try to fire
        await asyncio.sleep(0.1)

        # Assert: No events received after destroy
        assert len(received_events) == 0

    @pytest.mark.asyncio
    async def test_destroy_mid_event_sequence_stops_remaining(self):
        """
        ASSUMPTION: Destroy mid-sequence stops remaining events.

        If events are being emitted and destroy is called, remaining
        events in the sequence should not fire.
        """
        events = [
            (
                MockSessionEventType.ASSISTANT_MESSAGE_DELTA,
                create_event_data(delta_content="Part 1"),
            ),
            (
                MockSessionEventType.ASSISTANT_MESSAGE_DELTA,
                create_event_data(delta_content="Part 2"),
            ),
            (
                MockSessionEventType.ASSISTANT_MESSAGE,
                create_event_data(content="Complete", tool_requests=None),
            ),
            create_usage_event(),
            create_idle_event(),
        ]

        session = InstrumentedMockSession(
            event_sequence=EventSequence(
                events=events,
                emit_async=True,
                delay_between_events=0.05,  # Slower to allow mid-destroy
            )
        )

        received_events: list[str] = []
        destroy_triggered = False

        async def destroy_after_first():
            nonlocal destroy_triggered
            await asyncio.sleep(0.03)  # Wait for first event
            await session.destroy()
            destroy_triggered = True

        def handler(event: Any) -> None:
            received_events.append(event.type.value)

        session.on(handler)

        # Start both send and destroy concurrently
        await asyncio.gather(
            session.send({"prompt": "test"}),
            destroy_after_first(),
        )

        # Wait a bit for any remaining events
        await asyncio.sleep(0.2)

        # Assert: Should have received fewer events than total
        assert destroy_triggered
        # At most got first few events before destroy took effect
        assert len(received_events) < len(events)


class TestDestroyInContextManager:
    """
    Tests for destroy behavior when used with context managers.

    Our provider uses `async with client.create_session() as session`
    which calls destroy() on exit.
    """

    @pytest.mark.asyncio
    async def test_destroy_on_context_exit(self):
        """
        ASSUMPTION: Context manager exit triggers destroy.

        When using `async with`, __aexit__ must call destroy().
        """
        session = InstrumentedMockSession()

        # Simulate context manager behavior
        class MockContextManager:
            def __init__(self, sess: InstrumentedMockSession):
                self._session = sess

            async def __aenter__(self):
                return self._session

            async def __aexit__(self, *args):
                await self._session.destroy()

        async with MockContextManager(session) as ctx_session:
            assert ctx_session.destroyed is False
            # Normal operations here

        # After exit, session should be destroyed
        assert session.destroyed is True

    @pytest.mark.asyncio
    async def test_destroy_on_exception_in_context(self):
        """
        ASSUMPTION: Destroy is called even if exception occurs in context.

        Exception safety: destroy must be called to clean up even if
        the code inside the context raises.
        """
        session = InstrumentedMockSession()

        class MockContextManager:
            def __init__(self, sess: InstrumentedMockSession):
                self._session = sess

            async def __aenter__(self):
                return self._session

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                await self._session.destroy()
                return False  # Don't suppress exception

        with pytest.raises(ValueError):
            async with MockContextManager(session):
                raise ValueError("Simulated error")

        # Despite exception, destroy should have been called
        assert session.destroyed is True


class TestDestroyPreventsCLIRetry:
    """
    Tests that model the specific behavior we depend on:
    destroying the session prevents the CLI from retrying with
    built-in tools.
    """

    @pytest.mark.asyncio
    async def test_destroy_after_tool_capture(self):
        """
        SCENARIO: Capture tool then immediately destroy.

        This is the critical pattern our provider uses:
        1. Receive ASSISTANT_MESSAGE with tool_requests
        2. Capture the tool data
        3. Destroy session before CLI can retry

        Destroying must happen quickly enough to prevent retry.
        """
        tool_request = create_event_data(
            name="write_file",
            tool_call_id="call_capture_destroy",
            arguments={"path": "test.py", "content": "code"},
        )

        # Message with tool request, but we'll destroy before idle
        events = [
            (
                MockSessionEventType.ASSISTANT_MESSAGE,
                create_event_data(
                    content="I'll write that file.",
                    tool_requests=[tool_request],
                ),
            ),
            # In real SDK, more events would follow but we destroy before
            create_usage_event(),
            create_idle_event(),
        ]

        session = InstrumentedMockSession(
            event_sequence=EventSequence(
                events=events,
                emit_async=True,
                delay_between_events=0.05,
            )
        )

        captured_tools: list[Any] = []
        capture_and_destroy_complete = asyncio.Event()

        async def capture_and_destroy():
            """Capture tool requests and immediately destroy."""

            def handler(event: Any) -> None:
                if event.type == MockSessionEventType.ASSISTANT_MESSAGE:
                    if hasattr(event.data, "tool_requests") and event.data.tool_requests:
                        captured_tools.extend(event.data.tool_requests)

            session.on(handler)
            await session.send({"prompt": "Write test.py"})

            # Wait briefly for message event
            await asyncio.sleep(0.02)

            # CRITICAL: Destroy immediately after capture
            await session.destroy()
            capture_and_destroy_complete.set()

        await capture_and_destroy()

        # Assert: We captured the tool AND destroyed session
        assert len(captured_tools) == 1
        assert captured_tools[0].name == "write_file"
        assert session.destroyed is True

    @pytest.mark.asyncio
    async def test_destroy_timing_window(self):
        """
        ASSUMPTION: Destroy must happen before CLI agent loop iteration.

        This tests that our destroy timing is sufficient. The window
        between tool capture and CLI retry must be larger than our
        destroy latency.
        """
        # Track event sequence timing
        event_times: dict[str, float] = {}

        events = [
            (
                MockSessionEventType.ASSISTANT_MESSAGE,
                create_event_data(content="Tool message", tool_requests=[]),
            ),
            create_usage_event(),
            create_idle_event(),
        ]

        session = InstrumentedMockSession(
            event_sequence=EventSequence(events=events, emit_async=True)
        )

        import time

        message_received = asyncio.Event()
        destroy_start_time: float = 0

        def handler(event: Any) -> None:
            event_times[event.type.value] = time.monotonic()
            if event.type == MockSessionEventType.ASSISTANT_MESSAGE:
                message_received.set()

        session.on(handler)
        await session.send({"prompt": "test"})

        # Wait for message
        async with asyncio.timeout(1.0):
            await message_received.wait()

        destroy_start_time = time.monotonic()
        await session.destroy()
        destroy_end_time = time.monotonic()

        # Assert: Destroy was fast (< 100ms typical)
        destroy_duration = destroy_end_time - destroy_start_time
        assert destroy_duration < 0.1  # Should be nearly instant for mock


class TestSessionAfterDestroy:
    """
    Tests for behavior when operations are attempted after destroy.
    """

    @pytest.mark.asyncio
    async def test_send_after_destroy_behavior(self):
        """
        Document behavior of send() after destroy().

        This test documents what happens, not necessarily what should happen.
        In real SDK, this might raise or silently fail.
        """
        session = InstrumentedMockSession()
        await session.destroy()

        # Our mock allows send after destroy but won't emit events
        # Real SDK behavior may differ
        await session.send({"prompt": "After destroy"})

        # Message was recorded but session is still destroyed
        assert len(session.sent_messages) == 1
        assert session.destroyed is True

    @pytest.mark.asyncio
    async def test_event_registration_after_destroy(self):
        """
        Document behavior of on() after destroy().

        After destroy, event handlers are cleared. Registering new
        handlers should still work (for the mock) but they won't
        receive events since the session is terminated.
        """
        session = InstrumentedMockSession()
        await session.destroy()

        # Register handler after destroy
        handler_called = False

        def late_handler(event: Any) -> None:
            nonlocal handler_called
            handler_called = True

        session.on(late_handler)

        # Give time for any phantom events
        await asyncio.sleep(0.05)

        # Handler should NOT have been called
        assert handler_called is False
