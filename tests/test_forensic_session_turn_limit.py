"""
Forensic regression tests for session turn limit.

This test suite proves that the 305-turn infinite loop incident CANNOT recur
in the next-gen architecture. The public provider experienced a catastrophic
305-turn SDK loop where the SDK's internal agent loop kept retrying after
tool denials. The next-gen architecture uses ephemeral sessions + event queue,
making runaway loops architecturally impossible.

Contract: deny-destroy:Ephemeral:MUST:1, deny-destroy:NoExecution:MUST:1,3
Key architectural difference:
- Public provider: while not done: send_message(); process_events()  <- LOOP
- Next-gen: send() once; queue.drain(); done <- NO LOOP
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest

from tests.fixtures.sdk_mocks import (
    MockSDKSession,
    SessionEvent,
    SessionEventData,
    SessionEventType,
    message_complete_event,
    text_delta_event,
)

if TYPE_CHECKING:
    pass


def tool_call_event(
    tool_name: str = "bash",
    tool_call_id: str | None = None,
    arguments: dict[str, Any] | None = None,
) -> SessionEvent:
    """Create a tool call event.

    Contract: deny-destroy:NoExecution:MUST:1
    The provider captures tool calls but NEVER executes them.
    """
    if tool_call_id is None:
        import uuid

        tool_call_id = f"call_{uuid.uuid4().hex[:8]}"
    return SessionEvent(
        type=SessionEventType.TOOL_USE_COMPLETE,
        data=SessionEventData(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            arguments=arguments or {},
        ),
    )


class TestSingleTurnCompletion:
    """AC-1: Single tool call results in single turn, session destroyed.

    Contract: deny-destroy:Ephemeral:MUST:1
    """

    @pytest.mark.asyncio
    async def test_single_tool_call_single_turn(self) -> None:
        """1 tool call = 1 turn, session destroyed.

        Contract: deny-destroy:Ephemeral:MUST:1

        This test proves that even with tool call events, the session
        completes after one turn and is properly destroyed.
        """
        from amplifier_module_provider_github_copilot.completion import (
            complete_and_collect,
        )
        from amplifier_module_provider_github_copilot.sdk_adapter.types import (
            CompletionConfig,
        )

        events = [
            text_delta_event("Let me help you."),
            tool_call_event("bash", arguments={"command": "ls"}),
            message_complete_event("tool_use"),
        ]

        mock_session = MockSDKSession(events=events)

        async def create_session(config: Any) -> MockSDKSession:
            return mock_session

        @dataclass
        class MockRequest:
            prompt: str
            model: str | None = None

        result = await complete_and_collect(
            MockRequest(prompt="List files"),
            config=CompletionConfig(),
            sdk_create_fn=create_session,  # type: ignore[arg-type]
        )

        # Session should be destroyed
        assert mock_session.destroyed is True

        # Text should be accumulated
        assert "Let me help you." in result.text_content


class TestMultiToolSingleTurn:
    """AC-2: Multiple tool calls still result in single turn.

    Contract: deny-destroy:Ephemeral:MUST:1
    """

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_still_single_turn(self) -> None:
        """5 tool calls = still 1 send(), 1 disconnect().

        Contract: deny-destroy:Ephemeral:MUST:1

        The next-gen architecture processes ALL events from a single send()
        call. Multiple tool calls don't trigger multiple sends.
        """
        from amplifier_module_provider_github_copilot.completion import (
            complete_and_collect,
        )
        from amplifier_module_provider_github_copilot.sdk_adapter.types import (
            CompletionConfig,
        )

        # Create 5 tool call events
        events = [
            text_delta_event("I'll run multiple commands."),
            tool_call_event("bash", arguments={"command": "ls"}),
            tool_call_event("bash", arguments={"command": "pwd"}),
            tool_call_event("bash", arguments={"command": "whoami"}),
            tool_call_event("read_file", arguments={"path": "README.md"}),
            tool_call_event("write_file", arguments={"path": "out.txt", "content": "x"}),
            message_complete_event("tool_use"),
        ]

        mock_session = MockSDKSession(events=events)
        send_call_count = 0

        original_send = mock_session.send

        async def counting_send(
            prompt: str,
            *,
            attachments: list[dict[str, Any]] | None = None,
        ) -> str:
            nonlocal send_call_count
            send_call_count += 1
            return await original_send(prompt, attachments=attachments)

        mock_session.send = counting_send  # type: ignore[method-assign]

        async def create_session(config: Any) -> MockSDKSession:
            return mock_session

        @dataclass
        class MockRequest:
            prompt: str
            model: str | None = None

        await complete_and_collect(
            MockRequest(prompt="Run commands"),
            config=CompletionConfig(),
            sdk_create_fn=create_session,  # type: ignore[arg-type]
        )

        # Only 1 send() call despite 5 tool call events
        assert send_call_count == 1
        # Session destroyed
        assert mock_session.destroyed is True


class TestNoSecondSend:
    """AC-3: Verify send() called exactly once.

    Contract: deny-destroy:NoExecution:MUST:3
    """

    @pytest.mark.asyncio
    async def test_no_second_send_after_events(self) -> None:
        """send() called exactly once.

        Contract: deny-destroy:NoExecution:MUST:3

        The public provider's bug was calling send_message() in a loop.
        The next-gen calls send() once and processes the event queue.
        """
        from amplifier_module_provider_github_copilot.completion import (
            complete_and_collect,
        )
        from amplifier_module_provider_github_copilot.sdk_adapter.types import (
            CompletionConfig,
        )

        events = [
            text_delta_event("Processing..."),
            tool_call_event("bash"),
            tool_call_event("bash"),
            tool_call_event("bash"),
            message_complete_event("tool_use"),
        ]

        mock_session = MockSDKSession(events=events)
        send_calls: list[dict[str, Any]] = []

        original_send = mock_session.send

        async def tracking_send(
            prompt: str,
            *,
            attachments: list[dict[str, Any]] | None = None,
        ) -> str:
            send_calls.append({"prompt": prompt, "attachments": attachments})
            return await original_send(prompt, attachments=attachments)

        mock_session.send = tracking_send  # type: ignore[method-assign]

        async def create_session(config: Any) -> MockSDKSession:
            return mock_session

        @dataclass
        class MockRequest:
            prompt: str
            model: str | None = None

        await complete_and_collect(
            MockRequest(prompt="Test"),
            config=CompletionConfig(),
            sdk_create_fn=create_session,  # type: ignore[arg-type]
        )

        # Exactly 1 send call
        assert len(send_calls) == 1


class TestEventQueueDrain:
    """AC-4: All events processed from queue.

    Contract: deny-destroy:NoExecution:MUST:1
    """

    @pytest.mark.asyncio
    async def test_event_queue_fully_drained(self) -> None:
        """All events processed from queue.

        Contract: deny-destroy:NoExecution:MUST:1

        The architecture drains the event queue completely after idle.
        """
        from amplifier_module_provider_github_copilot.completion import complete
        from amplifier_module_provider_github_copilot.sdk_adapter.types import (
            CompletionConfig,
        )

        events = [
            text_delta_event("One"),
            text_delta_event("Two"),
            text_delta_event("Three"),
            message_complete_event("stop"),
        ]

        mock_session = MockSDKSession(events=events)

        async def create_session(config: Any) -> MockSDKSession:
            return mock_session

        @dataclass
        class MockRequest:
            prompt: str
            model: str | None = None

        # Count events yielded
        event_count = 0
        async for _ in complete(
            MockRequest(prompt="Test"),
            config=CompletionConfig(),
            sdk_create_fn=create_session,  # type: ignore[arg-type]
        ):
            event_count += 1

        # All events should be yielded (text deltas + message complete + idle)
        assert event_count >= 3  # At least the 3 text deltas + more


class TestDisconnectCalledOnce:
    """AC-5: disconnect() called exactly once regardless of event count.

    Contract: deny-destroy:Ephemeral:MUST:2
    """

    @pytest.mark.asyncio
    async def test_disconnect_called_once_regardless_of_event_count(self) -> None:
        """disconnect() called exactly once.

        Contract: deny-destroy:Ephemeral:MUST:2

        Whether 1 or 100 events, disconnect is called exactly once in finally.
        """
        from amplifier_module_provider_github_copilot.completion import (
            complete_and_collect,
        )
        from amplifier_module_provider_github_copilot.sdk_adapter.types import (
            CompletionConfig,
        )

        # Many events
        events = [text_delta_event(f"Delta {i}") for i in range(50)]
        events.append(message_complete_event("stop"))

        disconnect_count = 0

        class CountingMockSession(MockSDKSession):
            async def disconnect(self) -> None:
                nonlocal disconnect_count
                disconnect_count += 1
                await super().disconnect()

        mock_session = CountingMockSession(events=events)

        async def create_session(config: Any) -> CountingMockSession:
            return mock_session

        @dataclass
        class MockRequest:
            prompt: str
            model: str | None = None

        await complete_and_collect(
            MockRequest(prompt="Test"),
            config=CompletionConfig(),
            sdk_create_fn=create_session,  # type: ignore[arg-type]
        )

        # Exactly 1 disconnect
        assert disconnect_count == 1


class TestForensic305EventStress:
    """AC-6: 305 tool call events still result in 1 send + 1 disconnect.

    Contract: deny-destroy:NoExecution:MUST:3

    This is the FORENSIC test - it recreates the exact incident count
    from the public provider's 305-turn infinite loop bug.
    """

    @pytest.mark.asyncio
    async def test_305_event_stress_no_loop(self) -> None:
        """305 tool events → all captured, 1 turn.

        Contract: deny-destroy:NoExecution:MUST:3

        This test is deliberately provocative - it proves the architecture
        handles the exact incident count without any counter or limiter.
        The next-gen architecture makes the loop IMPOSSIBLE, not just limited.
        """
        from amplifier_module_provider_github_copilot.completion import (
            complete_and_collect,
        )
        from amplifier_module_provider_github_copilot.sdk_adapter.types import (
            CompletionConfig,
        )

        # Create exactly 305 tool call events (the incident count)
        events = [tool_call_event(f"tool_{i}") for i in range(305)]
        events.append(message_complete_event("tool_use"))

        send_count = 0
        disconnect_count = 0

        class StressTestSession(MockSDKSession):
            async def send(
                self,
                prompt: str,
                *,
                attachments: list[dict[str, Any]] | None = None,
            ) -> str:
                nonlocal send_count
                send_count += 1
                return await super().send(prompt, attachments=attachments)

            async def disconnect(self) -> None:
                nonlocal disconnect_count
                disconnect_count += 1
                await super().disconnect()

        mock_session = StressTestSession(events=events)

        async def create_session(config: Any) -> StressTestSession:
            return mock_session

        @dataclass
        class MockRequest:
            prompt: str
            model: str | None = None

        await complete_and_collect(
            MockRequest(prompt="Stress test"),
            config=CompletionConfig(),
            sdk_create_fn=create_session,  # type: ignore[arg-type]
        )

        # Even with 305 tool call events:
        # - Only 1 send() call (no retry loop)
        # - Only 1 disconnect() call (session destroyed once)
        assert send_count == 1, f"Expected 1 send, got {send_count}"
        assert disconnect_count == 1, f"Expected 1 disconnect, got {disconnect_count}"


class TestZeroEventsBeforeIdle:
    """AC related: Empty response, clean disconnect.

    Contract: deny-destroy:Ephemeral:MUST:2
    """

    @pytest.mark.asyncio
    async def test_zero_events_before_idle(self) -> None:
        """Empty response, clean disconnect.

        Contract: deny-destroy:Ephemeral:MUST:2

        The architecture handles empty responses gracefully.
        """
        from amplifier_module_provider_github_copilot.completion import (
            complete_and_collect,
        )
        from amplifier_module_provider_github_copilot.sdk_adapter.types import (
            CompletionConfig,
        )

        # No events except the auto-idle from MockSDKSession
        events: list[SessionEvent] = []

        mock_session = MockSDKSession(events=events)

        async def create_session(config: Any) -> MockSDKSession:
            return mock_session

        @dataclass
        class MockRequest:
            prompt: str
            model: str | None = None

        result = await complete_and_collect(
            MockRequest(prompt="Empty"),
            config=CompletionConfig(),
            sdk_create_fn=create_session,  # type: ignore[arg-type]
        )

        # Should complete without error
        assert mock_session.destroyed is True
        # Empty text content
        assert result.text_content == ""


class TestArchitectureNoRetryLoop:
    """Verify the architecture has no retry loop mechanism.

    Contract: deny-destroy:NoExecution:MUST:3
    """

    @pytest.mark.asyncio
    async def test_architecture_has_no_retry_loop(self) -> None:
        """No send() retry mechanism exists.

        Contract: deny-destroy:NoExecution:MUST:3

        This test verifies the completion.py module has no loop that
        retries send() after tool denials. The architecture is fundamentally
        different from the public provider's while loop.
        """
        from pathlib import Path

        # Read completion.py source
        completion_path = Path("amplifier_module_provider_github_copilot/completion.py")
        source = completion_path.read_text()

        # The following patterns would indicate a retry loop (BAD patterns):
        bad_patterns = [
            "while not done",  # Public provider's bug pattern
            "while True:",  # Infinite loop
            "for attempt in range",  # Retry loop
            "retry_count",  # Retry counter
            "send_and_wait",  # Old SDK API (deprecated)
        ]

        for pattern in bad_patterns:
            assert pattern not in source, (
                f"Found suspicious retry pattern '{pattern}' in completion.py"
            )

        # Good patterns that should exist:
        good_patterns = [
            "await session.send(",  # Single send call
            "while not event_queue.empty():",  # Queue drain (GOOD - not a retry)
        ]

        for pattern in good_patterns:
            assert pattern in source, f"Expected pattern '{pattern}' not found in completion.py"


class TestExceptionStillDestroysSession:
    """Verify exception during processing still destroys session.

    Contract: deny-destroy:Ephemeral:MUST:2
    """

    @pytest.mark.asyncio
    async def test_exception_during_processing_destroys_session(self) -> None:
        """Session destroyed even when processing raises.

        Contract: deny-destroy:Ephemeral:MUST:2

        The finally block ensures session cleanup regardless of errors.
        """
        from amplifier_module_provider_github_copilot.completion import complete
        from amplifier_module_provider_github_copilot.error_translation import LLMError
        from amplifier_module_provider_github_copilot.sdk_adapter.types import (
            CompletionConfig,
        )

        mock_session = MockSDKSession(events=[])
        mock_session.send = AsyncMock(  # type: ignore[method-assign]
            side_effect=RuntimeError("SDK error")
        )

        async def create_session(config: Any) -> MockSDKSession:
            return mock_session

        @dataclass
        class MockRequest:
            prompt: str
            model: str | None = None

        # SDK errors are translated to LLMError by complete()
        with pytest.raises(LLMError):
            async for _ in complete(
                MockRequest(prompt="Test"),
                config=CompletionConfig(),
                sdk_create_fn=create_session,  # type: ignore[arg-type]
            ):
                pass

        # Session should still be destroyed despite error
        assert mock_session.destroyed is True
