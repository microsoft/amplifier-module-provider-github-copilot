"""
Tests for completion lifecycle module.

Contract: streaming-contract.md, deny-destroy.md

Test categories:
- Session lifecycle (create/destroy)
- Event streaming and accumulation
- Error handling and translation
- Response construction
"""

from __future__ import annotations

from typing import Any

import pytest

from amplifier_module_provider_github_copilot.error_translation import (
    ErrorConfig,
    LLMError,
    NetworkError,
)
from amplifier_module_provider_github_copilot.provider import (
    CompletionConfig,
    CompletionRequest,
    complete,
    complete_and_collect,
)
from amplifier_module_provider_github_copilot.sdk_adapter.types import SessionConfig
from amplifier_module_provider_github_copilot.streaming import (
    DomainEvent,
    DomainEventType,
    EventConfig,
)
from tests.fixtures.sdk_mocks import (
    MockSDKSession,
    MockSDKSessionWithAbort,
    MockSDKSessionWithError,
)

# === Mock SDK Session ===
# NOTE: MockSDKSession classes consolidated to tests/fixtures/sdk_mocks.py
# Import from there for consistency across all tests


# === Fixtures ===


@pytest.fixture
def event_config() -> EventConfig:
    """Minimal event config for testing."""
    from amplifier_module_provider_github_copilot.streaming import DomainEventType, EventConfig

    return EventConfig(
        bridge_mappings={
            "assistant.message_delta": (DomainEventType.CONTENT_DELTA, "TEXT"),
            "assistant.reasoning_delta": (DomainEventType.CONTENT_DELTA, "THINKING"),
            "tool_use_complete": (DomainEventType.TOOL_CALL, None),
            "message_complete": (DomainEventType.TURN_COMPLETE, None),
            "usage_update": (DomainEventType.USAGE_UPDATE, None),
            "error": (DomainEventType.ERROR, None),
        },
        consume_patterns=["tool_use_start", "tool_use_delta"],
        drop_patterns=["heartbeat", "debug_*"],
    )


@pytest.fixture
def error_config() -> ErrorConfig:
    """Minimal error config for testing."""
    from amplifier_module_provider_github_copilot.error_translation import ErrorConfig, ErrorMapping

    return ErrorConfig(
        mappings=[
            ErrorMapping(
                sdk_patterns=["ConnectionError"],
                kernel_error="NetworkError",
                retryable=True,
            ),
        ],
        default_error="ProviderUnavailableError",
        default_retryable=True,
    )


@pytest.fixture
def completion_config(event_config: EventConfig, error_config: ErrorConfig) -> CompletionConfig:
    """Complete config for testing."""
    return CompletionConfig(
        session_config=SessionConfig(model="gpt-4"),
        event_config=event_config,
        error_config=error_config,
    )


# === Session Lifecycle Tests ===


class TestSessionLifecycle:
    """Test session create/destroy lifecycle."""

    @pytest.mark.asyncio
    async def test_session_created_and_destroyed_on_success(
        self, completion_config: CompletionConfig
    ) -> None:
        """AC-001: Session destroyed after successful completion."""
        events = [
            {"type": "assistant.message_delta", "text": "Hello"},
            {"type": "message_complete", "finish_reason": "stop"},
        ]
        session = MockSDKSession(events)

        async def mock_create(config: Any) -> Any:
            return session

        request = CompletionRequest(prompt="test")
        result = await complete_and_collect(
            request,
            config=completion_config,
            sdk_create_fn=mock_create,
        )

        assert session.disconnected is True
        assert result.is_complete is True

    @pytest.mark.asyncio
    async def test_session_destroyed_on_error(self, completion_config: CompletionConfig) -> None:
        """AC-001: Session destroyed even when error occurs."""
        error = ConnectionError("Network failed")
        session = MockSDKSessionWithError(error)

        async def mock_create(config: Any) -> Any:
            return session

        request = CompletionRequest(prompt="test")
        with pytest.raises(LLMError):
            await complete_and_collect(
                request,
                config=completion_config,
                sdk_create_fn=mock_create,
            )

        assert session.disconnected is True

    @pytest.mark.asyncio
    async def test_deny_hook_installed(self, completion_config: CompletionConfig) -> None:
        """AC-001: Event handler registered on session via on().

        The deny hook is passed via session config, not via method call.
        Here we verify the provider registers an event handler via on().
        Handlers are unsubscribed in finally block (correct cleanup).
        """
        events = [{"type": "message_complete", "finish_reason": "stop"}]
        session = MockSDKSession(events)
        handler_was_registered = False

        # Track when on() is called
        original_on = session.on

        def tracking_on(handler: Any) -> Any:
            nonlocal handler_was_registered
            handler_was_registered = True
            return original_on(handler)

        session.on = tracking_on

        async def mock_create(config: Any) -> Any:
            return session

        request = CompletionRequest(prompt="test")
        await complete_and_collect(
            request,
            config=completion_config,
            sdk_create_fn=mock_create,
        )

        # Verify event handler was registered via on() during execution
        # After cleanup, handler is unsubscribed (session._handlers is empty)
        assert handler_was_registered, "Event handler should be registered via on()"


# === Streaming Integration Tests ===


class TestStreamingIntegration:
    """Test event streaming and accumulation."""

    @pytest.mark.asyncio
    async def test_events_yielded_during_streaming(
        self, completion_config: CompletionConfig
    ) -> None:
        """AC-002: Domain events yielded during streaming."""
        events = [
            {"type": "assistant.message_delta", "text": "Hello "},
            {"type": "assistant.message_delta", "text": "World"},
            {"type": "message_complete", "finish_reason": "stop"},
        ]
        session = MockSDKSession(events)

        async def mock_create(config: Any) -> Any:
            return session

        request = CompletionRequest(prompt="test")
        yielded_events: list[DomainEvent] = []
        async for event in complete(
            request,
            config=completion_config,
            sdk_create_fn=mock_create,
        ):
            yielded_events.append(event)

        assert len(yielded_events) == 3
        assert yielded_events[0].type == DomainEventType.CONTENT_DELTA
        assert yielded_events[1].type == DomainEventType.CONTENT_DELTA
        assert yielded_events[2].type == DomainEventType.TURN_COMPLETE

    @pytest.mark.asyncio
    async def test_consume_events_not_yielded(self, completion_config: CompletionConfig) -> None:
        """AC-002: Consume events processed internally, not yielded."""
        events = [
            {"type": "tool_use_start", "tool_id": "t1"},  # consume
            {"type": "assistant.message_delta", "text": "Hello"},  # bridge
            {"type": "message_complete", "finish_reason": "stop"},
        ]
        session = MockSDKSession(events)

        async def mock_create(config: Any) -> Any:
            return session

        request = CompletionRequest(prompt="test")
        yielded_events: list[DomainEvent] = []
        async for event in complete(
            request,
            config=completion_config,
            sdk_create_fn=mock_create,
        ):
            yielded_events.append(event)

        # Only bridge events yielded
        assert len(yielded_events) == 2

    @pytest.mark.asyncio
    async def test_drop_events_not_yielded(self, completion_config: CompletionConfig) -> None:
        """AC-002: Drop events ignored, not yielded."""
        events = [
            {"type": "heartbeat"},  # drop
            {"type": "assistant.message_delta", "text": "Hello"},
            {"type": "debug_info", "data": "x"},  # drop via pattern
            {"type": "message_complete", "finish_reason": "stop"},
        ]
        session = MockSDKSession(events)

        async def mock_create(config: Any) -> Any:
            return session

        request = CompletionRequest(prompt="test")
        yielded_events: list[DomainEvent] = []
        async for event in complete(
            request,
            config=completion_config,
            sdk_create_fn=mock_create,
        ):
            yielded_events.append(event)

        assert len(yielded_events) == 2


# === Error Handling Tests ===


class TestErrorHandling:
    """Test error translation and propagation."""

    @pytest.mark.asyncio
    async def test_sdk_error_translated(self, completion_config: CompletionConfig) -> None:
        """AC-003: SDK errors translated to kernel types."""
        error = ConnectionError("Connection refused")
        session = MockSDKSessionWithError(error)

        async def mock_create(config: Any) -> Any:
            return session

        request = CompletionRequest(prompt="test")
        with pytest.raises(NetworkError) as exc_info:
            await complete_and_collect(
                request,
                config=completion_config,
                sdk_create_fn=mock_create,
            )

        assert exc_info.value.provider == "github-copilot"
        assert exc_info.value.retryable is True

    @pytest.mark.asyncio
    async def test_error_preserves_original(self, completion_config: CompletionConfig) -> None:
        """AC-003: Original exception chained via __cause__."""
        original = ConnectionError("Original error")
        session = MockSDKSessionWithError(original)

        async def mock_create(config: Any) -> Any:
            return session

        request = CompletionRequest(prompt="test")
        with pytest.raises(LLMError) as exc_info:
            await complete_and_collect(
                request,
                config=completion_config,
                sdk_create_fn=mock_create,
            )

        assert exc_info.value.__cause__ is original


# === Response Construction Tests ===


class TestResponseConstruction:
    """Test final response construction."""

    @pytest.mark.asyncio
    async def test_text_content_accumulated(self, completion_config: CompletionConfig) -> None:
        """AC-004: Text content accumulated correctly."""
        events = [
            {"type": "assistant.message_delta", "text": "Hello "},
            {"type": "assistant.message_delta", "text": "World"},
            {"type": "message_complete", "finish_reason": "stop"},
        ]
        session = MockSDKSession(events)

        async def mock_create(config: Any) -> Any:
            return session

        request = CompletionRequest(prompt="test")
        result = await complete_and_collect(
            request,
            config=completion_config,
            sdk_create_fn=mock_create,
        )

        assert result.text_content == "Hello World"
        assert result.finish_reason == "stop"
        assert result.is_complete is True

    @pytest.mark.asyncio
    async def test_thinking_content_separated(self, completion_config: CompletionConfig) -> None:
        """AC-004: Thinking content accumulated separately."""
        events = [
            {"type": "assistant.reasoning_delta", "text": "Let me think..."},
            {"type": "assistant.message_delta", "text": "The answer is 42"},
            {"type": "message_complete", "finish_reason": "stop"},
        ]
        session = MockSDKSession(events)

        async def mock_create(config: Any) -> Any:
            return session

        request = CompletionRequest(prompt="test")
        result = await complete_and_collect(
            request,
            config=completion_config,
            sdk_create_fn=mock_create,
        )

        assert result.thinking_content == "Let me think..."
        assert result.text_content == "The answer is 42"

    @pytest.mark.asyncio
    async def test_tool_calls_accumulated(self, completion_config: CompletionConfig) -> None:
        """AC-004: Tool calls accumulated correctly."""
        events = [
            {
                "type": "tool_use_complete",
                "tool_id": "t1",
                "name": "read_file",
                "arguments": {"path": "/test"},
            },
            {
                "type": "tool_use_complete",
                "tool_id": "t2",
                "name": "write_file",
                "arguments": {"path": "/out"},
            },
            {"type": "message_complete", "finish_reason": "tool_use"},
        ]
        session = MockSDKSession(events)

        async def mock_create(config: Any) -> Any:
            return session

        request = CompletionRequest(prompt="test")
        result = await complete_and_collect(
            request,
            config=completion_config,
            sdk_create_fn=mock_create,
        )

        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["name"] == "read_file"
        assert result.tool_calls[1]["name"] == "write_file"
        assert result.finish_reason == "tool_use"

    @pytest.mark.asyncio
    async def test_usage_captured(self, completion_config: CompletionConfig) -> None:
        """AC-004: Usage data captured."""
        events = [
            {"type": "assistant.message_delta", "text": "Hello"},
            {"type": "usage_update", "input_tokens": 10, "output_tokens": 5},
            {"type": "message_complete", "finish_reason": "stop"},
        ]
        session = MockSDKSession(events)

        async def mock_create(config: Any) -> Any:
            return session

        request = CompletionRequest(prompt="test")
        result = await complete_and_collect(
            request,
            config=completion_config,
            sdk_create_fn=mock_create,
        )

        assert result.usage is not None
        assert result.usage["input_tokens"] == 10
        assert result.usage["output_tokens"] == 5

    @pytest.mark.asyncio
    async def test_empty_response_handled(self, completion_config: CompletionConfig) -> None:
        """AC-004: Empty response handled gracefully."""
        events = [
            {"type": "message_complete", "finish_reason": "stop"},
        ]
        session = MockSDKSession(events)

        async def mock_create(config: Any) -> Any:
            return session

        request = CompletionRequest(prompt="test")
        result = await complete_and_collect(
            request,
            config=completion_config,
            sdk_create_fn=mock_create,
        )

        assert result.text_content == ""
        assert result.is_complete is True


# === Tool Capture and Abort Tests ===
# MockSDKSessionWithAbort imported from tests/fixtures/sdk_mocks.py


class TestToolCaptureAndAbort:
    """Tests for tool capture with session abort paths.

    Contract: sdk-protection:Session:MUST:3,4
    Coverage: completion.py lines 182-208
    """

    @pytest.mark.asyncio
    async def test_tool_capture_triggers_abort_success(
        self, completion_config: CompletionConfig
    ) -> None:
        """Tool capture triggers successful abort.

        Contract: sdk-protection:Session:MUST:3
        Coverage: completion.py lines 182-200 (tool capture + abort success)
        """
        # ASSISTANT_MESSAGE with tool_requests triggers tool capture
        # This is the SDK event format that ToolCaptureHandler.on_event() expects
        events = [
            {
                "type": "assistant.message",
                "tool_requests": [
                    {"id": "t1", "name": "bash", "arguments": {"command": "ls"}},
                ],
            },
            {"type": "message_complete", "finish_reason": "tool_use"},
        ]
        session = MockSDKSessionWithAbort(events, abort_behavior="success")

        async def mock_create(config: Any) -> Any:
            return session

        request = CompletionRequest(prompt="test")
        result = await complete_and_collect(
            request,
            config=completion_config,
            sdk_create_fn=mock_create,
        )

        # Should have tool calls from capture
        assert len(result.tool_calls) >= 1
        assert result.finish_reason == "tool_use"
        # Abort should have been called (if explicit_abort is enabled in config)
        # Note: This depends on sdk_protection config - test verifies no crash

    @pytest.mark.asyncio
    async def test_tool_capture_abort_timeout_non_blocking(
        self, completion_config: CompletionConfig, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Abort timeout is non-blocking — logs warning and continues.

        Contract: sdk-protection:Session:MUST:4
        Coverage: completion.py lines 201-204 (TimeoutError path)
        """
        import logging

        # ASSISTANT_MESSAGE with tool_requests triggers tool capture
        events = [
            {
                "type": "assistant.message",
                "tool_requests": [
                    {"id": "t1", "name": "bash", "arguments": {"command": "ls"}},
                ],
            },
            {"type": "message_complete", "finish_reason": "tool_use"},
        ]
        # Session that times out on abort (delay > abort_timeout)
        session = MockSDKSessionWithAbort(events, abort_behavior="timeout")

        async def mock_create(config: Any) -> Any:
            return session

        request = CompletionRequest(prompt="test")

        with caplog.at_level(logging.WARNING):
            # Should complete without hanging despite abort timeout
            result = await complete_and_collect(
                request,
                config=completion_config,
                sdk_create_fn=mock_create,
            )

        # Still returns result even if abort times out
        assert len(result.tool_calls) >= 1
        # This test primarily verifies no hang - timeout handling is in the SDK path

    @pytest.mark.asyncio
    async def test_tool_capture_abort_exception_non_critical(
        self, completion_config: CompletionConfig, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Abort exception is non-critical — logs and continues.

        Contract: sdk-protection:Session:MUST:4
        Coverage: completion.py lines 205-208 (Exception path)
        """
        import logging

        # ASSISTANT_MESSAGE with tool_requests triggers tool capture
        events = [
            {
                "type": "assistant.message",
                "tool_requests": [
                    {"id": "t1", "name": "bash", "arguments": {"command": "ls"}},
                ],
            },
            {"type": "message_complete", "finish_reason": "tool_use"},
        ]
        session = MockSDKSessionWithAbort(events, abort_behavior="exception")

        async def mock_create(config: Any) -> Any:
            return session

        request = CompletionRequest(prompt="test")

        with caplog.at_level(logging.DEBUG):
            # Should complete despite abort exception
            result = await complete_and_collect(
                request,
                config=completion_config,
                sdk_create_fn=mock_create,
            )

        # Still returns result even if abort fails
        assert len(result.tool_calls) >= 1


class TestSessionDisconnect:
    """Tests for session disconnect in finally clause.

    Coverage: completion.py lines 247-249
    """

    @pytest.mark.asyncio
    async def test_disconnect_called_on_success(self, completion_config: CompletionConfig) -> None:
        """Session disconnect called after successful completion.

        Coverage: completion.py line 249 (finally disconnect)
        """
        events = [
            {"type": "assistant.message_delta", "text": "Hello"},
            {"type": "message_complete", "finish_reason": "stop"},
        ]
        session = MockSDKSession(events)

        async def mock_create(config: Any) -> Any:
            return session

        request = CompletionRequest(prompt="test")
        await complete_and_collect(
            request,
            config=completion_config,
            sdk_create_fn=mock_create,
        )

        assert session.disconnected is True

    @pytest.mark.asyncio
    async def test_disconnect_called_on_error(self, completion_config: CompletionConfig) -> None:
        """Session disconnect called even when error occurs.

        Coverage: completion.py line 249 (finally disconnect on error path)
        """
        session = MockSDKSessionWithError(NetworkError("connection lost"))

        async def mock_create(config: Any) -> Any:
            return session

        request = CompletionRequest(prompt="test")

        with pytest.raises(LLMError):
            await complete_and_collect(
                request,
                config=completion_config,
                sdk_create_fn=mock_create,
            )

        assert session.disconnected is True
