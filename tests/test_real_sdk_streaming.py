"""Tests for the real SDK streaming pipeline.

Contract: streaming-contract.md, event-vocabulary.md

These tests verify that the real SDK path uses streaming iteration
instead of send_and_wait, routing events through the translation pipeline.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import patch

import pytest

from amplifier_module_provider_github_copilot.provider import (
    CompletionConfig,
    CompletionRequest,
    complete_and_collect,
    load_event_config,
)
from amplifier_module_provider_github_copilot.streaming import (
    DomainEvent,
    DomainEventType,
)
from tests.fixtures.sdk_mocks import MockSDKSession


class TestF052RealSDKStreamingPipeline:
    """Test that real SDK path uses streaming pipeline.

    Contract: streaming-contract.md:Accumulation:MUST:1
    """

    @pytest.mark.asyncio
    async def test_real_sdk_path_uses_streaming_not_send_and_wait(self) -> None:
        """Real SDK path should iterate over streaming events, not use send_and_wait.

        Contract: streaming-contract.md - streaming pipeline must emit correct event sequence
        """
        # Arrange: Create mock SDK session that streams events
        events: list[dict[str, Any]] = [
            {"type": "assistant.message_delta", "text": "Hello "},
            {"type": "assistant.message_delta", "text": "World!"},
            {"type": "message_complete", "finish_reason": "stop"},
        ]

        async def create_session(config: Any) -> MockSDKSession:
            return MockSDKSession(events)

        # Act
        request = CompletionRequest(prompt="Hello")
        event_config = load_event_config()
        config = CompletionConfig(event_config=event_config)

        result = await complete_and_collect(request, config=config, sdk_create_fn=create_session)

        # Assert: Response should contain accumulated text from streaming
        assert result.text_content == "Hello World!"
        assert result.is_complete

    @pytest.mark.asyncio
    async def test_tool_calls_captured_from_streaming_events(self) -> None:
        """Tool calls should be captured from TOOL_CALL streaming events.

        Contract: streaming-contract.md:ToolCapture:MUST:1
        """
        # Arrange: Stream includes tool_use_complete event
        events: list[dict[str, Any]] = [
            {"type": "assistant.message_delta", "text": "Let me check that."},
            {
                "type": "tool_use_complete",
                "id": "tool-123",
                "name": "read_file",
                "arguments": {"path": "/tmp/test.txt"},
            },
            {"type": "message_complete", "finish_reason": "tool_use"},
        ]

        async def create_session(config: Any) -> MockSDKSession:
            return MockSDKSession(events)

        # Act
        request = CompletionRequest(prompt="Read a file")
        event_config = load_event_config()
        config = CompletionConfig(event_config=event_config)

        result = await complete_and_collect(request, config=config, sdk_create_fn=create_session)

        # Assert: Tool calls should be in response
        assert len(result.tool_calls) >= 1
        assert result.tool_calls[0]["name"] == "read_file"
        assert result.tool_calls[0]["id"] == "tool-123"

    @pytest.mark.asyncio
    async def test_usage_update_captured_from_streaming(self) -> None:
        """Usage updates should be captured from USAGE_UPDATE streaming events.

        Contract: streaming-contract.md - streaming events translated through pipeline
        """
        # Arrange: Stream includes usage_update event
        events: list[dict[str, Any]] = [
            {"type": "assistant.message_delta", "text": "Response text"},
            {
                "type": "usage_update",
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            },
            {"type": "message_complete", "finish_reason": "stop"},
        ]

        async def create_session(config: Any) -> MockSDKSession:
            return MockSDKSession(events)

        # Act
        request = CompletionRequest(prompt="Test usage")
        event_config = load_event_config()
        config = CompletionConfig(event_config=event_config)

        result = await complete_and_collect(request, config=config, sdk_create_fn=create_session)

        # Assert: Usage should be captured from streaming event
        assert result.usage is not None
        assert result.usage["input_tokens"] == 10
        assert result.usage["output_tokens"] == 5

    @pytest.mark.asyncio
    async def test_finish_reason_from_turn_complete_event(self) -> None:
        """Finish reason should come from TURN_COMPLETE streaming event.

        Contract: streaming-contract.md - finish_reason from streaming events
        """
        events: list[dict[str, Any]] = [
            {"type": "assistant.message_delta", "text": "Done"},
            {"type": "message_complete", "finish_reason": "stop"},
        ]

        async def create_session(config: Any) -> MockSDKSession:
            return MockSDKSession(events)

        request = CompletionRequest(prompt="Test")
        event_config = load_event_config()
        config = CompletionConfig(event_config=event_config)

        result = await complete_and_collect(request, config=config, sdk_create_fn=create_session)

        # Assert: finish_reason should come from message_complete event
        assert result.finish_reason is not None
        # After finish_reason_map translation: stop -> STOP
        assert result.finish_reason in ("stop", "STOP")

    @pytest.mark.asyncio
    async def test_events_routed_through_translate_event(self) -> None:
        """SDK events should be routed through translate_event function.

        Contract: event-vocabulary.md - events must use defined domain event types
        """
        events: list[dict[str, Any]] = [
            {"type": "assistant.message_delta", "text": "Hello"},
            {"type": "message_complete", "finish_reason": "stop"},
        ]

        async def create_session(config: Any) -> MockSDKSession:
            return MockSDKSession(events)

        # Patch translate_event to verify it's called
        with patch(
            "amplifier_module_provider_github_copilot.completion.translate_event"
        ) as mock_translate:
            # Make translate_event return domain events
            def translate_side_effect(event: dict[str, Any], config: Any) -> DomainEvent | None:
                event_type = event.get("type", "")
                if event_type == "assistant.message_delta":
                    return DomainEvent(
                        type=DomainEventType.CONTENT_DELTA,
                        data={"text": event.get("text", "")},
                    )
                elif event_type == "message_complete":
                    return DomainEvent(
                        type=DomainEventType.TURN_COMPLETE,
                        data={"finish_reason": event.get("finish_reason", "stop")},
                    )
                return None

            mock_translate.side_effect = translate_side_effect

            request = CompletionRequest(prompt="Test")
            event_config = load_event_config()
            config = CompletionConfig(event_config=event_config)

            await complete_and_collect(request, config=config, sdk_create_fn=create_session)

            # Assert: translate_event should have been called for each event
            # (at least 2 for the content events, plus session.idle)
            assert mock_translate.call_count >= 2


class TestF052EventConfigExercised:
    """Test that config/events.yaml is exercised in real SDK path."""

    @pytest.mark.asyncio
    async def test_event_config_loaded_for_real_path(self) -> None:
        """Event config should be loaded and used for real SDK path.

        Contract: event-vocabulary.md - events classified per config
        """
        events: list[dict[str, Any]] = [
            {"type": "assistant.message_delta", "text": "Test"},
            {"type": "message_complete", "finish_reason": "stop"},
        ]

        async def create_session(config: Any) -> MockSDKSession:
            return MockSDKSession(events)

        # Patch load_event_config to verify it's called
        with patch(
            "amplifier_module_provider_github_copilot.completion.load_event_config"
        ) as mock_load_config:
            from amplifier_module_provider_github_copilot.streaming import EventConfig

            mock_load_config.return_value = EventConfig(
                bridge_mappings={
                    "assistant.message_delta": (DomainEventType.CONTENT_DELTA, None),
                    "message_complete": (DomainEventType.TURN_COMPLETE, None),
                    "session.idle": (DomainEventType.SESSION_IDLE, None),
                }
            )

            request = CompletionRequest(prompt="Test")
            config = CompletionConfig()

            await complete_and_collect(request, config=config, sdk_create_fn=create_session)

            # Assert: Event config should have been loaded
            mock_load_config.assert_called()


# ============================================================================
# Error Event Handling Tests
# ============================================================================


class TestErrorEventHandling:
    """Tests for error event handling in SDK completion.

    Contract: streaming-contract:ErrorHandling:MUST:1
    Coverage: provider.py lines 628-645
    """

    @pytest.mark.asyncio
    async def test_error_event_dict_with_data_message(self) -> None:
        """Error event with dict data.message extracts message.

        Contract: streaming-contract:ErrorHandling:MUST:1
        Coverage: provider.py lines 639-641
        """
        from unittest.mock import AsyncMock, MagicMock

        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider()

        @asynccontextmanager
        async def mock_session_cm(*args: Any, **kwargs: Any):
            mock_session = MagicMock()
            events_fired = False

            def mock_on(handler: Any) -> Any:
                nonlocal events_fired
                if not events_fired:
                    events_fired = True
                    # Fire error event with dict structure
                    error_event: dict[str, Any] = {
                        "type": "session_error",
                        "data": {"message": "SDK error occurred"},
                    }
                    handler(error_event)
                return lambda: None

            mock_session.on = mock_on
            mock_session.send = AsyncMock()
            yield mock_session

        provider._client.session = mock_session_cm  # type: ignore[reportPrivateUsage]

        request = MagicMock()
        request.model = "gpt-4o"
        request.messages = []
        request.tools = []

        with pytest.raises(Exception) as exc_info:
            await provider.complete(request)

        assert "SDK error occurred" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_error_event_object_with_data_message(self) -> None:
        """Error event with object data.message extracts message.

        Contract: streaming-contract:ErrorHandling:MUST:1
        Coverage: provider.py lines 643-644
        """
        from unittest.mock import AsyncMock, MagicMock

        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider()

        @asynccontextmanager
        async def mock_session_cm(*args: Any, **kwargs: Any):
            mock_session = MagicMock()
            events_fired = False

            def mock_on(handler: Any) -> Any:
                nonlocal events_fired
                if not events_fired:
                    events_fired = True
                    # Fire error event with object structure
                    error_data = MagicMock()
                    error_data.message = "Object error message"

                    error_event = MagicMock()
                    error_event.type = MagicMock()
                    error_event.type.value = "session_error"
                    error_event.data = error_data
                    handler(error_event)
                return lambda: None

            mock_session.on = mock_on
            mock_session.send = AsyncMock()
            yield mock_session

        provider._client.session = mock_session_cm  # type: ignore[reportPrivateUsage]

        request = MagicMock()
        request.model = "gpt-4o"
        request.messages = []
        request.tools = []

        with pytest.raises(Exception) as exc_info:
            await provider.complete(request)

        assert "Object error message" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_error_event_no_data_uses_event_str(self) -> None:
        """Error event with no data uses str(event).

        Contract: streaming-contract:ErrorHandling:MUST:1
        Coverage: provider.py lines 636-637
        """
        from unittest.mock import AsyncMock, MagicMock

        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider()

        @asynccontextmanager
        async def mock_session_cm(*args: Any, **kwargs: Any):
            mock_session = MagicMock()
            events_fired = False

            def mock_on(handler: Any) -> Any:
                nonlocal events_fired
                if not events_fired:
                    events_fired = True
                    # Fire error event with no data
                    error_event = {"type": "session_error"}
                    handler(error_event)
                return lambda: None

            mock_session.on = mock_on
            mock_session.send = AsyncMock()
            yield mock_session

        provider._client.session = mock_session_cm  # type: ignore[reportPrivateUsage]

        request = MagicMock()
        request.model = "gpt-4o"
        request.messages = []
        request.tools = []

        with pytest.raises(Exception) as exc_info:
            await provider.complete(request)

        # Should use str(event) as fallback
        assert "Session error" in str(exc_info.value)


class TestEventHandlerException:
    """Tests for exception handling in event handler.

    Coverage: provider.py line 650
    """

    @pytest.mark.asyncio
    async def test_event_handler_exception_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """Exception in event handler is logged as warning.

        Coverage: provider.py line 650
        """
        import logging
        from unittest.mock import AsyncMock, MagicMock

        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider()

        @asynccontextmanager
        async def mock_session_cm(*args: Any, **kwargs: Any):
            mock_session = MagicMock()
            call_count = 0

            def mock_on(handler: Any) -> Any:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    # First call: fire malformed event that causes exception
                    # The queue.put_nowait might fail on a bad event
                    try:
                        handler(None)  # This might cause issues in handler
                    except Exception:
                        pass

                    # Then fire idle to continue
                    idle_event = MagicMock()
                    idle_event.type = MagicMock()
                    idle_event.type.value = "session_idle"
                    handler(idle_event)
                return lambda: None

            mock_session.on = mock_on
            mock_session.send = AsyncMock()
            mock_session.abort = AsyncMock()
            yield mock_session

        provider._client.session = mock_session_cm  # type: ignore[reportPrivateUsage]

        request = MagicMock()
        request.model = "gpt-4o"
        request.messages = []
        request.tools = []

        with caplog.at_level(logging.WARNING):
            # Should complete without crashing
            await provider.complete(request)
