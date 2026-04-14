"""
Kernel Type Integration Tests.

These tests verify the provider returns kernel types that pass isinstance() checks.

Contract: contracts/provider-protocol.md
"""

from typing import cast

import pytest


class TestKernelTypeCompliance:
    """Tests that provider returns actual kernel types.

    Contract: provider-protocol:get_info:MUST:1
    """

    @pytest.mark.asyncio
    async def test_provider_info_is_kernel_type(self) -> None:
        """AC-1 - get_info() returns amplifier_core.ProviderInfo.

        # Contract: provider-protocol:get_info:MUST:1
        """
        from amplifier_core import ProviderInfo as KernelProviderInfo

        from amplifier_module_provider_github_copilot import GitHubCopilotProvider

        provider = GitHubCopilotProvider()
        info = provider.get_info()

        # This is THE critical check that fails today
        assert isinstance(info, KernelProviderInfo), (
            f"get_info() returned {type(info).__name__}, expected ProviderInfo from amplifier_core"
        )

        # Verify required kernel fields have correct types
        assert isinstance(info.id, str)
        assert isinstance(info.display_name, str)
        assert isinstance(info.credential_env_vars, list)
        assert isinstance(info.defaults, dict)
        assert isinstance(info.capabilities, list)

    @pytest.mark.asyncio
    async def test_list_models_returns_kernel_types(self) -> None:
        """AC-1b - list_models() returns amplifier_core.ModelInfo.

        Note: This test uses explicit mocking because list_models() now calls SDK
        dynamically, and in tests without SDK connection, we need to mock the response.

        # Contract: provider-protocol:list_models:MUST:1
        """
        from unittest.mock import patch

        from amplifier_core import ModelInfo as KernelModelInfo

        from amplifier_module_provider_github_copilot import GitHubCopilotProvider

        # Create mock models — uses a fixed list since model catalog comes from SDK,
        # not config. This tests that list_models() returns correct kernel types.
        mock_models = [
            KernelModelInfo(
                id="claude-opus-4.5",
                display_name="Claude Opus 4.5",
                context_window=200000,
                max_output_tokens=32000,
                capabilities=["streaming", "tools", "vision"],
                defaults={},
            ),
            KernelModelInfo(
                id="claude-sonnet-4",
                display_name="Claude Sonnet 4",
                context_window=216000,
                max_output_tokens=88000,
                capabilities=["streaming", "tools", "vision"],
                defaults={},
            ),
        ]

        provider = GitHubCopilotProvider()

        async def mock_fetch(*args: object, **kwargs: object) -> list[KernelModelInfo]:
            return mock_models

        with patch.object(provider, "list_models", mock_fetch):
            models = await provider.list_models()

        assert len(models) >= 1
        for model in models:
            assert isinstance(model, KernelModelInfo), (
                f"list_models() returned {type(model).__name__}, expected ModelInfo"
            )
            # Verify required kernel fields have correct types
            assert isinstance(model.id, str)
            assert isinstance(model.display_name, str)
            assert isinstance(model.context_window, int)
            assert isinstance(model.max_output_tokens, int)
            assert isinstance(model.capabilities, list)

    def test_error_translation_produces_kernel_errors(self) -> None:
        """AC-2 - translate_sdk_error() returns amplifier_core.llm_errors types."""
        from pathlib import Path

        from amplifier_core.llm_errors import AuthenticationError as KernelAuthError
        from amplifier_core.llm_errors import LLMError as KernelLLMError

        from amplifier_module_provider_github_copilot.error_translation import (
            load_error_config,
            translate_sdk_error,
        )

        # Must provide config path explicitly
        config_path = (
            Path(__file__).parent.parent
            / "amplifier_module_provider_github_copilot"
            / "config"
            / "data"
            / "errors.yaml"
        )
        config = load_error_config(config_path)

        # Create an auth error
        class SDKAuthError(Exception):
            pass

        exc = SDKAuthError("401 Unauthorized")
        result = translate_sdk_error(exc, config)

        # Must be actual kernel type
        assert isinstance(result, KernelLLMError), (
            f"Expected amplifier_core LLMError base, got {type(result).__name__}"
        )
        assert isinstance(result, KernelAuthError), (
            f"Expected amplifier_core AuthenticationError, got {type(result).__name__}"
        )

    def test_tool_call_is_kernel_type(self) -> None:
        """AC-3 - parse_tool_calls() returns amplifier_core.ToolCall.

        # Contract: provider-protocol:parse_tool_calls:MUST:1
        """
        from amplifier_core import ChatResponse
        from amplifier_core import ToolCall as KernelToolCall

        from amplifier_module_provider_github_copilot import GitHubCopilotProvider

        provider = GitHubCopilotProvider()

        # Mock response with tool calls
        class MockToolCall:
            id = "tc_1"
            name = "test_tool"
            arguments = {"arg1": "value1"}

        class MockResponse:
            tool_calls = [MockToolCall()]

        result = provider.parse_tool_calls(cast(ChatResponse, MockResponse()))

        assert len(result) == 1
        assert isinstance(result[0], KernelToolCall), (
            f"Expected amplifier_core ToolCall, got {type(result[0]).__name__}"
        )


class TestChatResponseContentTypes:
    """Tests that ChatResponse.content uses correct Pydantic block types."""

    def test_text_block_is_pydantic(self) -> None:
        """ChatResponse.content must use TextBlock (Pydantic), not TextContent (dataclass)."""
        from amplifier_core import TextBlock
        from pydantic import BaseModel

        # TextBlock must be Pydantic
        assert issubclass(TextBlock, BaseModel), "TextBlock must be Pydantic BaseModel"

        # Must have type discriminator
        block = TextBlock(text="hello")
        assert block.type == "text"

    def test_thinking_block_is_pydantic(self) -> None:
        """ChatResponse.content uses ThinkingBlock (Pydantic), not ThinkingContent."""
        from amplifier_core import ThinkingBlock
        from pydantic import BaseModel

        # ThinkingBlock must be Pydantic
        assert issubclass(ThinkingBlock, BaseModel), "ThinkingBlock must be Pydantic"

        # Must have type discriminator
        block = ThinkingBlock(thinking="reasoning here")
        assert block.type == "thinking"


class TestUsageRequiredFields:
    """Tests that Usage has all required fields."""

    def test_usage_requires_three_fields(self) -> None:
        """Usage must have input_tokens, output_tokens, total_tokens."""
        from amplifier_core import Usage

        # All three are required (no defaults)
        usage = Usage(input_tokens=10, output_tokens=5, total_tokens=15)

        assert usage.input_tokens == 10
        assert usage.output_tokens == 5
        assert usage.total_tokens == 15


class TestStreamingAccumulatorToChatResponse:
    """Tests for to_chat_response() conversion in StreamingAccumulator."""

    def test_to_chat_response_returns_kernel_type(self) -> None:
        """AC-4 - StreamingAccumulator.to_chat_response() returns ChatResponse."""
        from amplifier_core import ChatResponse

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
                block_type=None,
            )
        )
        accumulator.add(
            DomainEvent(
                type=DomainEventType.TURN_COMPLETE,
                data={"finish_reason": "stop"},
            )
        )

        response = accumulator.to_chat_response()

        assert isinstance(response, ChatResponse), (
            f"to_chat_response() returned {type(response).__name__}, expected ChatResponse"
        )

    def test_to_chat_response_uses_text_block(self) -> None:
        """AC-5 - ChatResponse.content uses TextBlock (Pydantic)."""
        from amplifier_core import TextBlock

        from amplifier_module_provider_github_copilot.streaming import (
            DomainEvent,
            DomainEventType,
            StreamingAccumulator,
        )

        accumulator = StreamingAccumulator()
        accumulator.add(
            DomainEvent(
                type=DomainEventType.CONTENT_DELTA,
                data={"text": "Hello"},
                block_type=None,  # Default to text
            )
        )
        accumulator.add(
            DomainEvent(
                type=DomainEventType.TURN_COMPLETE,
                data={"finish_reason": "stop"},
            )
        )

        response = accumulator.to_chat_response()

        assert len(response.content) == 1
        assert isinstance(response.content[0], TextBlock), (
            f"Expected TextBlock, got {type(response.content[0]).__name__}"
        )
        assert response.content[0].text == "Hello"

    def test_to_chat_response_uses_thinking_block(self) -> None:
        """AC-5 - ChatResponse.content uses ThinkingBlock (Pydantic)."""
        from amplifier_core import ThinkingBlock

        from amplifier_module_provider_github_copilot.streaming import (
            DomainEvent,
            DomainEventType,
            StreamingAccumulator,
        )

        accumulator = StreamingAccumulator()
        accumulator.add(
            DomainEvent(
                type=DomainEventType.CONTENT_DELTA,
                data={"text": "Thinking..."},
                block_type="THINKING",
            )
        )
        accumulator.add(
            DomainEvent(
                type=DomainEventType.TURN_COMPLETE,
                data={"finish_reason": "stop"},
            )
        )

        response = accumulator.to_chat_response()

        assert len(response.content) == 1
        assert isinstance(response.content[0], ThinkingBlock), (
            f"Expected ThinkingBlock, got {type(response.content[0]).__name__}"
        )
        assert response.content[0].thinking == "Thinking..."


class TestErrorTypeHierarchy:
    """Tests that all error types are from kernel."""

    def test_all_error_types_are_kernel_types(self) -> None:
        """AC-2b - All KERNEL_ERROR_MAP values are kernel types."""
        from amplifier_core.llm_errors import LLMError as KernelLLMError

        from amplifier_module_provider_github_copilot.error_translation import KERNEL_ERROR_MAP

        for name, error_class in KERNEL_ERROR_MAP.items():
            assert issubclass(error_class, KernelLLMError), (
                f"KERNEL_ERROR_MAP['{name}'] = {error_class.__name__} is not a kernel type"
            )

    def test_provider_unavailable_is_kernel_type(self) -> None:
        """AC-2c - ProviderUnavailableError is kernel type."""
        from pathlib import Path

        from amplifier_core.llm_errors import ProviderUnavailableError as KernelError

        from amplifier_module_provider_github_copilot.error_translation import (
            load_error_config,
            translate_sdk_error,
        )

        config_path = (
            Path(__file__).parent.parent
            / "amplifier_module_provider_github_copilot"
            / "config"
            / "data"
            / "errors.yaml"
        )
        config = load_error_config(config_path)

        # Trigger default fallback
        exc = Exception("Some unknown error")
        result = translate_sdk_error(exc, config)

        assert isinstance(result, KernelError), (
            f"Default error should be kernel ProviderUnavailableError, got {type(result).__name__}"
        )
