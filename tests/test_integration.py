"""
Integration tests for provider-github-copilot.

End-to-end tests verifying all modules work together with mock SDK.

Contract: contracts/provider-protocol.md, contracts/streaming-contract.md, contracts/deny-destroy.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from amplifier_module_provider_github_copilot.error_translation import (
    AuthenticationError,
    ErrorConfig,
    ErrorMapping,
    RateLimitError,
)
from amplifier_module_provider_github_copilot.provider import (
    CompletionConfig,
    CompletionRequest,
    GitHubCopilotProvider,
    complete_and_collect,
)
from amplifier_module_provider_github_copilot.sdk_adapter.types import (
    SessionConfig,
)
from amplifier_module_provider_github_copilot.streaming import (
    load_event_config,
)

# ============================================================================
# Mock SDK Session (imported from centralized fixtures)
# ============================================================================
from tests.fixtures.sdk_mocks import MockSDKSession


@dataclass
class MockToolCall:
    """Mock tool call object."""

    id: str
    name: str
    arguments: dict[str, Any]


async def mock_create_session(config: SessionConfig) -> MockSDKSession:
    """Factory that creates MockSDKSession - used as sdk_create_fn."""
    # Default events for simple completion
    return MockSDKSession(
        events=[
            {"type": "assistant.message_delta", "text": "Hello "},
            {"type": "assistant.message_delta", "text": "world!"},
            {"type": "message_complete", "finish_reason": "stop"},
        ]
    )


# ============================================================================
# AC-1: Full Completion Lifecycle
# ============================================================================


class TestFullCompletionLifecycle:
    """AC-1: Provider can complete a request end-to-end with mock SDK."""

    @pytest.mark.asyncio
    async def test_complete_simple_request(self) -> None:
        """Full completion lifecycle with text response."""
        events = [
            {"type": "assistant.message_delta", "text": "Hello "},
            {"type": "assistant.message_delta", "text": "world!"},
            {"type": "message_complete", "finish_reason": "stop"},
        ]
        session = MockSDKSession(events)

        async def create_session(config: Any) -> MockSDKSession:
            return session

        request = CompletionRequest(prompt="Say hello")
        event_config = load_event_config()
        config = CompletionConfig(event_config=event_config)

        result = await complete_and_collect(request, config=config, sdk_create_fn=create_session)

        assert result.text_content == "Hello world!"
        # AC: finish_reason is now mapped via finish_reason_map ("stop" -> "STOP")
        assert result.finish_reason == "STOP"
        assert result.is_complete
        assert session.destroyed  # Session cleanup


# ============================================================================
# AC-2: Tool Call Integration
# ============================================================================


class TestToolCallIntegration:
    """AC-2: Provider correctly parses tool calls from completion response."""

    @pytest.mark.asyncio
    async def test_complete_with_tool_calls(self) -> None:
        """Tool calls are extracted from completion response."""
        events = [
            {"type": "assistant.message_delta", "text": "I'll help you with that."},
            {
                "type": "tool_use_complete",
                "id": "call_123",
                "name": "get_weather",
                "arguments": {"location": "Seattle"},
            },
            {"type": "message_complete", "finish_reason": "tool_use"},
        ]
        session = MockSDKSession(events)

        async def create_session(config: Any) -> MockSDKSession:
            return session

        request = CompletionRequest(prompt="What's the weather?")
        event_config = load_event_config()
        config = CompletionConfig(event_config=event_config)

        result = await complete_and_collect(request, config=config, sdk_create_fn=create_session)

        assert result.text_content == "I'll help you with that."
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"
        assert result.tool_calls[0]["arguments"] == {"location": "Seattle"}
        # AC: finish_reason is now mapped via finish_reason_map ("tool_use" -> "TOOL_USE")
        assert result.finish_reason == "TOOL_USE"

    def test_parse_tool_calls_from_response(self) -> None:
        """parse_tool_calls returns ToolCall objects with correct structure."""
        provider = GitHubCopilotProvider()

        @dataclass
        class MockResponse:
            tool_calls: list[MockToolCall]

        response = MockResponse(
            tool_calls=[
                MockToolCall(
                    id="call_456",
                    name="search",
                    arguments={"query": "python"},
                ),
            ]
        )

        tool_calls = provider.parse_tool_calls(response)

        assert len(tool_calls) == 1
        assert tool_calls[0].id == "call_456"
        assert tool_calls[0].name == "search"
        assert tool_calls[0].arguments == {"query": "python"}


# ============================================================================
# AC-3: Error Translation Integration
# ============================================================================


class TestErrorTranslationIntegration:
    """AC-3: SDK errors are translated to domain error types.

    Iteration 2: Force cache invalidation.
    """

    @pytest.mark.asyncio
    async def test_auth_error_translation(self) -> None:
        """Authentication errors are translated correctly."""

        class SDKAuthError(Exception):
            """Mock SDK authentication error."""

            pass

        session = MockSDKSession(events=[], raise_on_send=SDKAuthError("Invalid token"))

        async def create_session(config: Any) -> MockSDKSession:
            return session

        # Create error config that maps SDKAuthError
        error_config = ErrorConfig(
            mappings=[
                ErrorMapping(
                    sdk_patterns=["SDKAuthError"],
                    kernel_error="AuthenticationError",
                    retryable=False,
                ),
            ]
        )

        request = CompletionRequest(prompt="Test")
        config = CompletionConfig(error_config=error_config)

        with pytest.raises(AuthenticationError) as exc_info:
            await complete_and_collect(request, config=config, sdk_create_fn=create_session)

        assert exc_info.value.retryable is False
        assert exc_info.value.provider == "github-copilot"
        assert session.destroyed  # Session still cleaned up

    @pytest.mark.asyncio
    async def test_rate_limit_error_with_retry_after(self) -> None:
        """Rate limit errors have retry_after populated."""

        class SDKRateLimitError(Exception):
            """Mock SDK rate limit error."""

            pass

        session = MockSDKSession(
            events=[],
            raise_on_send=SDKRateLimitError("Rate limited. Retry after 30 seconds"),
        )

        async def create_session(config: Any) -> MockSDKSession:
            return session

        error_config = ErrorConfig(
            mappings=[
                ErrorMapping(
                    sdk_patterns=["SDKRateLimitError"],
                    kernel_error="RateLimitError",
                    retryable=True,
                    extract_retry_after=True,
                ),
            ]
        )

        request = CompletionRequest(prompt="Test")
        config = CompletionConfig(error_config=error_config)

        with pytest.raises(RateLimitError) as exc_info:
            await complete_and_collect(request, config=config, sdk_create_fn=create_session)

        assert exc_info.value.retryable is True
        assert exc_info.value.retry_after == 30.0


# ============================================================================
# AC-4: Streaming Event Flow
# ============================================================================


class TestStreamingEventFlow:
    """AC-4: Events flow from SDK → streaming → accumulator → response."""

    @pytest.mark.asyncio
    async def test_content_delta_accumulation(self) -> None:
        """CONTENT_DELTA events accumulate correctly."""
        events = [
            {"type": "assistant.message_delta", "text": "Part 1. "},
            {"type": "assistant.message_delta", "text": "Part 2. "},
            {"type": "assistant.message_delta", "text": "Part 3."},
            {"type": "message_complete", "finish_reason": "stop"},
        ]
        session = MockSDKSession(events)

        async def create_session(config: Any) -> MockSDKSession:
            return session

        request = CompletionRequest(prompt="Test")
        event_config = load_event_config()
        config = CompletionConfig(event_config=event_config)

        result = await complete_and_collect(request, config=config, sdk_create_fn=create_session)

        assert result.text_content == "Part 1. Part 2. Part 3."

    @pytest.mark.asyncio
    async def test_thinking_content_separate(self) -> None:
        """Thinking content is accumulated separately from text."""
        events = [
            {"type": "assistant.reasoning_delta", "text": "Let me think..."},
            {"type": "assistant.message_delta", "text": "Here's my answer."},
            {"type": "message_complete", "finish_reason": "stop"},
        ]
        session = MockSDKSession(events)

        async def create_session(config: Any) -> MockSDKSession:
            return session

        request = CompletionRequest(prompt="Test")
        event_config = load_event_config()
        config = CompletionConfig(event_config=event_config)

        result = await complete_and_collect(request, config=config, sdk_create_fn=create_session)

        assert result.text_content == "Here's my answer."
        assert result.thinking_content == "Let me think..."

    @pytest.mark.asyncio
    async def test_turn_complete_signals_finish(self) -> None:
        """TURN_COMPLETE signals completion with finish_reason."""
        events = [
            {"type": "assistant.message_delta", "text": "Done"},
            {"type": "message_complete", "finish_reason": "stop"},
        ]
        session = MockSDKSession(events)

        async def create_session(config: Any) -> MockSDKSession:
            return session

        request = CompletionRequest(prompt="Test")
        event_config = load_event_config()
        config = CompletionConfig(event_config=event_config)

        result = await complete_and_collect(request, config=config, sdk_create_fn=create_session)

        assert result.is_complete
        # AC: finish_reason is now mapped via finish_reason_map ("stop" -> "STOP")
        assert result.finish_reason == "STOP"


# ============================================================================
# AC-5: Session Factory Integration
# ============================================================================


class TestSessionFactoryIntegration:
    """AC-5: Session lifecycle is correctly managed."""

    @pytest.mark.asyncio
    async def test_deny_hook_installed(self) -> None:
        """Deny hook is installed on all sessions."""
        events = [
            {"type": "assistant.message_delta", "text": "Response"},
            {"type": "message_complete", "finish_reason": "stop"},
        ]
        session = MockSDKSession(events)

        async def create_session(config: Any) -> MockSDKSession:
            return session

        request = CompletionRequest(prompt="Test")
        event_config = load_event_config()
        config = CompletionConfig(event_config=event_config)

        await complete_and_collect(request, config=config, sdk_create_fn=create_session)

        assert session.deny_hook_installed

    @pytest.mark.asyncio
    async def test_session_destroyed_on_success(self) -> None:
        """Sessions are destroyed after successful completion."""
        events = [
            {"type": "assistant.message_delta", "text": "Response"},
            {"type": "message_complete", "finish_reason": "stop"},
        ]
        session = MockSDKSession(events)

        async def create_session(config: Any) -> MockSDKSession:
            return session

        request = CompletionRequest(prompt="Test")
        event_config = load_event_config()
        config = CompletionConfig(event_config=event_config)

        await complete_and_collect(request, config=config, sdk_create_fn=create_session)

        assert session.destroyed

    @pytest.mark.asyncio
    async def test_session_destroyed_on_error(self) -> None:
        """Sessions are destroyed even on error (try/finally).

        Note: SDK errors are translated to kernel errors (ProviderUnavailableError)
        by the error translation pipeline. The test verifies session cleanup
        happens regardless of error translation.
        """
        from amplifier_core.llm_errors import ProviderUnavailableError

        class SDKError(Exception):
            pass

        session = MockSDKSession(events=[], raise_on_send=SDKError("Boom"))

        async def create_session(config: Any) -> MockSDKSession:
            return session

        request = CompletionRequest(prompt="Test")
        event_config = load_event_config()
        config = CompletionConfig(event_config=event_config)

        # SDK errors get translated to kernel errors
        with pytest.raises(ProviderUnavailableError):
            await complete_and_collect(request, config=config, sdk_create_fn=create_session)

        assert session.destroyed


# ============================================================================
# AC-6: Provider Protocol Compliance
# ============================================================================


class TestProviderProtocolCompliance:
    """AC-6: Provider implements the full protocol."""

    def test_provider_name_property(self) -> None:
        """Provider has name property."""
        provider = GitHubCopilotProvider()
        assert provider.name == "github-copilot"

    def test_parse_tool_calls_returns_tool_call_list(self) -> None:
        """parse_tool_calls() returns list[ToolCall]."""
        provider = GitHubCopilotProvider()

        @dataclass
        class EmptyResponse:
            tool_calls: None = None

        result = provider.parse_tool_calls(EmptyResponse())
        assert result == []
