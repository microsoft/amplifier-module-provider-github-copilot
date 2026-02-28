"""
Tests for streaming functionality in CopilotSdkProvider.

These tests cover the event-based streaming logic, error handling,
timeouts, and tool call parsing during streaming.
"""

import asyncio
from collections.abc import Callable
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from amplifier_module_provider_github_copilot.client import CopilotClientWrapper
from amplifier_module_provider_github_copilot.provider import CopilotSdkProvider


# Mock SessionEventType enum to match copilot SDK
class MockSessionEventType(Enum):
    """Mock of copilot.generated.session_events.SessionEventType."""

    ASSISTANT_TURN_START = "assistant.turn_start"
    ASSISTANT_TURN_END = "assistant.turn_end"
    ASSISTANT_MESSAGE_DELTA = "assistant.message_delta"
    ASSISTANT_MESSAGE = "assistant.message"
    ASSISTANT_REASONING_DELTA = "assistant.reasoning_delta"
    ASSISTANT_REASONING = "assistant.reasoning"
    ASSISTANT_USAGE = "assistant.usage"
    SESSION_IDLE = "session.idle"
    SESSION_ERROR = "session.error"


class MockStreamingSession:
    """
    Mock session that simulates streaming events.

    Allows testing of event-based streaming logic by controlling
    exactly which events are emitted and when.
    """

    def __init__(self, events: list[tuple[MockSessionEventType, Any]] | None = None):
        """
        Initialize with list of (event_type, event_data) tuples.

        Args:
            events: List of (MockSessionEventType, data) tuples to emit
        """
        self.session_id = "mock-streaming-session"
        self.events = events or []
        self.event_handlers: list[Callable] = []
        self.destroyed = False
        self.sent_messages: list[Any] = []

    def on(self, handler: Callable) -> Callable:
        """Subscribe to events. Returns unsubscribe function."""
        self.event_handlers.append(handler)

        def unsubscribe():
            if handler in self.event_handlers:
                self.event_handlers.remove(handler)

        return unsubscribe

    async def send(self, message: dict[str, Any]) -> None:
        """Send a message and trigger events."""
        self.sent_messages.append(message)

        # Emit all events in sequence
        for event_type, event_data in self.events:
            event = Mock()
            event.type = event_type  # Use MockSessionEventType enum
            event.data = event_data

            for handler in self.event_handlers:
                handler(event)

    async def destroy(self) -> None:
        """Destroy the session."""
        self.destroyed = True


def create_event_data(**kwargs) -> Mock:
    """Create mock event data with attributes."""
    data = Mock()
    for key, value in kwargs.items():
        setattr(data, key, value)
    return data


class TestStreamingEvents:
    """Tests for streaming event handling."""

    @pytest.fixture
    def streaming_provider(self, mock_coordinator):
        """Create provider configured for streaming."""
        return CopilotSdkProvider(
            api_key=None,
            config={
                "model": "claude-opus-4-5",
                "timeout": 60.0,
                "use_streaming": True,
                "debug": False,
            },
            coordinator=mock_coordinator,
        )

    @pytest.mark.asyncio
    async def test_streaming_text_deltas(self, streaming_provider):
        """Test handling of ASSISTANT_MESSAGE_DELTA events."""
        # Create session that emits text deltas
        events = [
            (
                MockSessionEventType.ASSISTANT_MESSAGE_DELTA,
                create_event_data(delta_content="Hello "),
            ),
            (
                MockSessionEventType.ASSISTANT_MESSAGE_DELTA,
                create_event_data(delta_content="world!"),
            ),
            (
                MockSessionEventType.ASSISTANT_MESSAGE,
                create_event_data(content="Hello world!", tool_requests=None),
            ),
            (
                MockSessionEventType.ASSISTANT_USAGE,
                create_event_data(input_tokens=10, output_tokens=5),
            ),
            (MockSessionEventType.SESSION_IDLE, create_event_data()),
        ]
        mock_session = MockStreamingSession(events)

        # Mock the _stream_with_session method by calling _process_streaming_response directly
        # We need to patch the session creation
        @asynccontextmanager
        async def mock_create_session(
            self,
            model,
            system_message=None,
            streaming=True,
            reasoning_effort=None,
            tools=None,
            excluded_tools=None,
            hooks=None,
        ):
            yield mock_session
            await mock_session.destroy()

        with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
            with patch.object(CopilotClientWrapper, "ensure_client", new_callable=AsyncMock):
                with patch(
                    "copilot.generated.session_events.SessionEventType", MockSessionEventType
                ):
                    # Call complete with streaming
                    request = {"messages": [{"role": "user", "content": "Hi"}]}
                    response = await streaming_provider.complete(request)

                    # Verify response
                    assert response is not None
                    assert len(response.content) > 0

    @pytest.mark.asyncio
    async def test_streaming_reasoning_deltas(self, streaming_provider):
        """Test handling of ASSISTANT_REASONING_DELTA events for thinking."""
        events = [
            (
                MockSessionEventType.ASSISTANT_REASONING_DELTA,
                create_event_data(delta_content="Let me "),
            ),
            (
                MockSessionEventType.ASSISTANT_REASONING_DELTA,
                create_event_data(delta_content="think..."),
            ),
            (
                MockSessionEventType.ASSISTANT_MESSAGE_DELTA,
                create_event_data(delta_content="The answer is 4."),
            ),
            (
                MockSessionEventType.ASSISTANT_MESSAGE,
                create_event_data(content="The answer is 4.", tool_requests=None),
            ),
            (
                MockSessionEventType.ASSISTANT_USAGE,
                create_event_data(input_tokens=15, output_tokens=10),
            ),
            (MockSessionEventType.SESSION_IDLE, create_event_data()),
        ]
        mock_session = MockStreamingSession(events)

        @asynccontextmanager
        async def mock_create_session(
            self,
            model,
            system_message=None,
            streaming=True,
            reasoning_effort=None,
            tools=None,
            excluded_tools=None,
            hooks=None,
        ):
            yield mock_session
            await mock_session.destroy()

        with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
            with patch.object(CopilotClientWrapper, "ensure_client", new_callable=AsyncMock):
                with patch(
                    "copilot.generated.session_events.SessionEventType", MockSessionEventType
                ):
                    request = {"messages": [{"role": "user", "content": "What is 2+2?"}]}
                    response = await streaming_provider.complete(request, extended_thinking=True)

                    assert response is not None

    @pytest.mark.asyncio
    async def test_streaming_complete_reasoning_block(self, streaming_provider):
        """Test handling of ASSISTANT_REASONING event (complete block)."""
        events = [
            (
                MockSessionEventType.ASSISTANT_REASONING,
                create_event_data(content="I analyzed the problem carefully."),
            ),
            (
                MockSessionEventType.ASSISTANT_MESSAGE,
                create_event_data(content="Here is my answer.", tool_requests=None),
            ),
            (
                MockSessionEventType.ASSISTANT_USAGE,
                create_event_data(input_tokens=20, output_tokens=15),
            ),
            (MockSessionEventType.SESSION_IDLE, create_event_data()),
        ]
        mock_session = MockStreamingSession(events)

        @asynccontextmanager
        async def mock_create_session(
            self,
            model,
            system_message=None,
            streaming=True,
            reasoning_effort=None,
            tools=None,
            excluded_tools=None,
            hooks=None,
        ):
            yield mock_session
            await mock_session.destroy()

        with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
            with patch.object(CopilotClientWrapper, "ensure_client", new_callable=AsyncMock):
                with patch(
                    "copilot.generated.session_events.SessionEventType", MockSessionEventType
                ):
                    request = {"messages": [{"role": "user", "content": "Complex question"}]}
                    response = await streaming_provider.complete(request, extended_thinking=True)

                    assert response is not None

    @pytest.mark.asyncio
    async def test_streaming_with_tool_requests(self, streaming_provider):
        """Test handling of tool requests in streaming mode."""
        # Create tool request mock
        tool_request = Mock()
        tool_request.tool_call_id = "call_123"
        tool_request.name = "read_file"
        tool_request.arguments = {"path": "test.py"}

        events = [
            (
                MockSessionEventType.ASSISTANT_MESSAGE_DELTA,
                create_event_data(delta_content="I'll read that file."),
            ),
            (
                MockSessionEventType.ASSISTANT_MESSAGE,
                create_event_data(content="I'll read that file.", tool_requests=[tool_request]),
            ),
            (
                MockSessionEventType.ASSISTANT_USAGE,
                create_event_data(input_tokens=25, output_tokens=20),
            ),
            (MockSessionEventType.SESSION_IDLE, create_event_data()),
        ]
        mock_session = MockStreamingSession(events)

        # Mock tool spec to trigger the has_tools=True path
        mock_tool_spec = Mock(name="read_file", description="Read a file", parameters={})

        @asynccontextmanager
        async def mock_create_session(
            self,
            model,
            system_message=None,
            streaming=True,
            reasoning_effort=None,
            tools=None,
            excluded_tools=None,
            hooks=None,
        ):
            yield mock_session
            await mock_session.destroy()

        with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
            with patch.object(CopilotClientWrapper, "ensure_client", new_callable=AsyncMock):
                with patch(
                    "copilot.generated.session_events.SessionEventType", MockSessionEventType
                ):
                    with patch(
                        "amplifier_module_provider_github_copilot.provider.convert_tools_for_sdk",
                        return_value=[Mock()],
                    ):
                        with patch(
                            "amplifier_module_provider_github_copilot.provider.make_deny_all_hook",
                            return_value={
                                "on_pre_tool_use": lambda *a: {"permissionDecision": "deny"}
                            },
                        ):
                            request = {
                                "messages": [{"role": "user", "content": "Read test.py"}],
                                "tools": [mock_tool_spec],
                            }
                            response = await streaming_provider.complete(request)

                            assert response is not None
                            # Should have tool_use finish reason
                            assert response.finish_reason == "tool_use"
                            assert response.tool_calls is not None
                            assert len(response.tool_calls) == 1
                            assert response.tool_calls[0].name == "read_file"

    @pytest.mark.asyncio
    async def test_streaming_tool_arguments_as_string(self, streaming_provider):
        """Test handling of tool arguments passed as JSON string."""
        tool_request = Mock()
        tool_request.tool_call_id = "call_456"
        tool_request.name = "search"
        tool_request.arguments = '{"query": "python"}'  # String instead of dict

        events = [
            (
                MockSessionEventType.ASSISTANT_MESSAGE,
                create_event_data(content="Searching...", tool_requests=[tool_request]),
            ),
            (
                MockSessionEventType.ASSISTANT_USAGE,
                create_event_data(input_tokens=10, output_tokens=5),
            ),
            (MockSessionEventType.SESSION_IDLE, create_event_data()),
        ]
        mock_session = MockStreamingSession(events)

        mock_tool_spec = Mock(name="search", description="Search", parameters={})

        @asynccontextmanager
        async def mock_create_session(
            self,
            model,
            system_message=None,
            streaming=True,
            reasoning_effort=None,
            tools=None,
            excluded_tools=None,
            hooks=None,
        ):
            yield mock_session
            await mock_session.destroy()

        with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
            with patch.object(CopilotClientWrapper, "ensure_client", new_callable=AsyncMock):
                with patch(
                    "copilot.generated.session_events.SessionEventType", MockSessionEventType
                ):
                    with patch(
                        "amplifier_module_provider_github_copilot.provider.convert_tools_for_sdk",
                        return_value=[Mock()],
                    ):
                        with patch(
                            "amplifier_module_provider_github_copilot.provider.make_deny_all_hook",
                            return_value={
                                "on_pre_tool_use": lambda *a: {"permissionDecision": "deny"}
                            },
                        ):
                            request = {
                                "messages": [{"role": "user", "content": "Search for python"}],
                                "tools": [mock_tool_spec],
                            }
                            response = await streaming_provider.complete(request)

                            assert response.tool_calls[0].arguments == {"query": "python"}

    @pytest.mark.asyncio
    async def test_streaming_tool_arguments_invalid_json(self, streaming_provider):
        """Test handling of invalid JSON in tool arguments."""
        tool_request = Mock()
        tool_request.tool_call_id = "call_789"
        tool_request.name = "custom"
        tool_request.arguments = "not valid json {"  # Invalid JSON

        events = [
            (
                MockSessionEventType.ASSISTANT_MESSAGE,
                create_event_data(content="Running...", tool_requests=[tool_request]),
            ),
            (
                MockSessionEventType.ASSISTANT_USAGE,
                create_event_data(input_tokens=10, output_tokens=5),
            ),
            (MockSessionEventType.SESSION_IDLE, create_event_data()),
        ]
        mock_session = MockStreamingSession(events)

        mock_tool_spec = Mock(name="custom", description="Custom", parameters={})

        @asynccontextmanager
        async def mock_create_session(
            self,
            model,
            system_message=None,
            streaming=True,
            reasoning_effort=None,
            tools=None,
            excluded_tools=None,
            hooks=None,
        ):
            yield mock_session
            await mock_session.destroy()

        with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
            with patch.object(CopilotClientWrapper, "ensure_client", new_callable=AsyncMock):
                with patch(
                    "copilot.generated.session_events.SessionEventType", MockSessionEventType
                ):
                    with patch(
                        "amplifier_module_provider_github_copilot.provider.convert_tools_for_sdk",
                        return_value=[Mock()],
                    ):
                        with patch(
                            "amplifier_module_provider_github_copilot.provider.make_deny_all_hook",
                            return_value={
                                "on_pre_tool_use": lambda *a: {"permissionDecision": "deny"}
                            },
                        ):
                            request = {
                                "messages": [{"role": "user", "content": "Run custom"}],
                                "tools": [mock_tool_spec],
                            }
                            response = await streaming_provider.complete(request)

                            # Should wrap in raw key
                            assert response.tool_calls[0].arguments == {"raw": "not valid json {"}


class TestStreamingErrors:
    """Tests for error handling in streaming mode."""

    @pytest.fixture
    def streaming_provider(self, mock_coordinator):
        """Create provider configured for streaming with short timeout."""
        return CopilotSdkProvider(
            api_key=None,
            config={
                "model": "claude-opus-4-5",
                "timeout": 1.0,  # Short timeout for testing
                "thinking_timeout": 1.0,  # Also short - opus triggers reasoning path
                "use_streaming": True,
                "debug": False,
            },
            coordinator=mock_coordinator,
        )

    @pytest.mark.asyncio
    async def test_streaming_session_error(self, streaming_provider):
        """Test handling of SESSION_ERROR event."""
        events = [
            (
                MockSessionEventType.ASSISTANT_MESSAGE_DELTA,
                create_event_data(delta_content="Starting..."),
            ),
            (
                MockSessionEventType.SESSION_ERROR,
                create_event_data(message="Backend error occurred"),
            ),
        ]
        mock_session = MockStreamingSession(events)

        @asynccontextmanager
        async def mock_create_session(
            self,
            model,
            system_message=None,
            streaming=True,
            reasoning_effort=None,
            tools=None,
            excluded_tools=None,
            hooks=None,
        ):
            yield mock_session
            await mock_session.destroy()

        with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
            with patch.object(CopilotClientWrapper, "ensure_client", new_callable=AsyncMock):
                with patch(
                    "copilot.generated.session_events.SessionEventType", MockSessionEventType
                ):
                    request = {"messages": [{"role": "user", "content": "Test"}]}

                    with pytest.raises(Exception) as exc_info:
                        await streaming_provider.complete(request)

                    assert "Backend error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_streaming_timeout(self, streaming_provider):
        """Test timeout handling in streaming mode."""
        from amplifier_core.llm_errors import LLMTimeoutError

        # NOTE: Provider wraps CopilotTimeoutError -> LLMTimeoutError for kernel compatibility
        # Session that emits a delta but never reaches idle — will trigger timeout
        # Key: events must be emitted asynchronously to allow timeout to fire
        class NeverIdleSession(MockStreamingSession):
            """Session where send() returns quickly but idle never fires."""

            async def send(self, message):
                self.sent_messages.append(message)

                async def emit_events_async():
                    """Emit events asynchronously to simulate real SDK behavior."""
                    await asyncio.sleep(0.01)  # Yield to event loop
                    for event_type, event_data in self.events:
                        event = Mock()
                        event.type = event_type
                        event.data = event_data
                        for handler in self.event_handlers:
                            handler(event)
                    # No SESSION_IDLE — provider will timeout

                # Start event emission but don't await (fire and forget)
                asyncio.create_task(emit_events_async())
                # Return immediately — timeout will fire on idle_event.wait()

        events = [
            (
                MockSessionEventType.ASSISTANT_MESSAGE_DELTA,
                create_event_data(delta_content="Starting..."),
            ),
            # No SESSION_IDLE — provider's asyncio.timeout will fire
        ]
        mock_session = NeverIdleSession(events)

        @asynccontextmanager
        async def mock_create_session(
            self,
            model,
            system_message=None,
            streaming=True,
            reasoning_effort=None,
            tools=None,
            excluded_tools=None,
            hooks=None,
        ):
            yield mock_session
            await mock_session.destroy()

        with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
            with patch.object(CopilotClientWrapper, "ensure_client", new_callable=AsyncMock):
                with patch(
                    "copilot.generated.session_events.SessionEventType", MockSessionEventType
                ):
                    request = {"messages": [{"role": "user", "content": "Test"}]}

                    with pytest.raises(LLMTimeoutError):
                        await streaming_provider.complete(request)

    @pytest.mark.asyncio
    async def test_streaming_no_deltas_use_final_content(self, streaming_provider):
        """Test fallback to final message content when no deltas received."""
        events = [
            # No delta events, only final message
            (
                MockSessionEventType.ASSISTANT_MESSAGE,
                create_event_data(content="Direct response without deltas", tool_requests=None),
            ),
            (
                MockSessionEventType.ASSISTANT_USAGE,
                create_event_data(input_tokens=10, output_tokens=5),
            ),
            (MockSessionEventType.SESSION_IDLE, create_event_data()),
        ]
        mock_session = MockStreamingSession(events)

        @asynccontextmanager
        async def mock_create_session(
            self,
            model,
            system_message=None,
            streaming=True,
            reasoning_effort=None,
            tools=None,
            excluded_tools=None,
            hooks=None,
        ):
            yield mock_session
            await mock_session.destroy()

        with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
            with patch.object(CopilotClientWrapper, "ensure_client", new_callable=AsyncMock):
                with patch(
                    "copilot.generated.session_events.SessionEventType", MockSessionEventType
                ):
                    request = {"messages": [{"role": "user", "content": "Test"}]}
                    response = await streaming_provider.complete(request)

                    assert response is not None
                    # Should use content from final message
                    assert any("Direct response" in str(block) for block in response.content)


class TestStreamingUsage:
    """Tests for usage tracking in streaming mode."""

    @pytest.fixture
    def streaming_provider(self, mock_coordinator):
        """Create provider configured for streaming."""
        return CopilotSdkProvider(
            api_key=None,
            config={
                "model": "claude-opus-4-5",
                "timeout": 60.0,
                "use_streaming": True,
            },
            coordinator=mock_coordinator,
        )

    @pytest.mark.asyncio
    async def test_streaming_usage_tracking(self, streaming_provider):
        """Test that usage tokens are correctly tracked."""
        events = [
            (
                MockSessionEventType.ASSISTANT_MESSAGE_DELTA,
                create_event_data(delta_content="Response"),
            ),
            (
                MockSessionEventType.ASSISTANT_MESSAGE,
                create_event_data(content="Response", tool_requests=None),
            ),
            (
                MockSessionEventType.ASSISTANT_USAGE,
                create_event_data(input_tokens=150, output_tokens=75),
            ),
            (MockSessionEventType.SESSION_IDLE, create_event_data()),
        ]
        mock_session = MockStreamingSession(events)

        @asynccontextmanager
        async def mock_create_session(
            self,
            model,
            system_message=None,
            streaming=True,
            reasoning_effort=None,
            tools=None,
            excluded_tools=None,
            hooks=None,
        ):
            yield mock_session
            await mock_session.destroy()

        with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
            with patch.object(CopilotClientWrapper, "ensure_client", new_callable=AsyncMock):
                with patch(
                    "copilot.generated.session_events.SessionEventType", MockSessionEventType
                ):
                    request = {"messages": [{"role": "user", "content": "Test"}]}
                    response = await streaming_provider.complete(request)

                    assert response.usage.input_tokens == 150
                    assert response.usage.output_tokens == 75
                    assert response.usage.total_tokens == 225

    @pytest.mark.asyncio
    async def test_streaming_usage_with_none_values(self, streaming_provider):
        """Test handling of None values in usage data."""
        events = [
            (
                MockSessionEventType.ASSISTANT_MESSAGE,
                create_event_data(content="Response", tool_requests=None),
            ),
            (
                MockSessionEventType.ASSISTANT_USAGE,
                create_event_data(input_tokens=None, output_tokens=None),
            ),
            (MockSessionEventType.SESSION_IDLE, create_event_data()),
        ]
        mock_session = MockStreamingSession(events)

        @asynccontextmanager
        async def mock_create_session(
            self,
            model,
            system_message=None,
            streaming=True,
            reasoning_effort=None,
            tools=None,
            excluded_tools=None,
            hooks=None,
        ):
            yield mock_session
            await mock_session.destroy()

        with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
            with patch.object(CopilotClientWrapper, "ensure_client", new_callable=AsyncMock):
                with patch(
                    "copilot.generated.session_events.SessionEventType", MockSessionEventType
                ):
                    request = {"messages": [{"role": "user", "content": "Test"}]}
                    response = await streaming_provider.complete(request)

                    # Should handle None gracefully
                    assert response.usage.input_tokens == 0
                    assert response.usage.output_tokens == 0
