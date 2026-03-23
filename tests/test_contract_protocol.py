"""
Contract Compliance Tests: Provider Protocol.

Contract: contracts/provider-protocol.md

Tests every MUST clause in the Provider Protocol contract.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from amplifier_module_provider_github_copilot.provider import (
    GitHubCopilotProvider,
)


@pytest.fixture
def provider() -> GitHubCopilotProvider:
    """Create provider instance for testing."""
    coordinator = MagicMock()
    return GitHubCopilotProvider(config=None, coordinator=coordinator)


class TestProtocolNameProperty:
    """provider-protocol:name:MUST:1,2"""

    def test_returns_github_copilot_string(self, provider: GitHubCopilotProvider) -> None:
        """provider-protocol:name:MUST:1 - Returns 'github-copilot'."""
        assert provider.name == "github-copilot"

    def test_is_a_property_not_method(self) -> None:
        """provider-protocol:name:MUST:2 - Is a property, not a method."""
        assert isinstance(GitHubCopilotProvider.name, property)


class TestProtocolGetInfo:
    """provider-protocol:get_info:MUST:1,2"""

    def test_returns_provider_info(self, provider: GitHubCopilotProvider) -> None:
        """provider-protocol:get_info:MUST:1 - Returns valid ProviderInfo."""
        info = provider.get_info()

        assert info.id == "github-copilot"
        assert info.display_name is not None
        assert info.capabilities is not None

    def test_includes_capabilities(self, provider: GitHubCopilotProvider) -> None:
        """provider-protocol:get_info:MUST:2 - Includes capabilities."""
        info = provider.get_info()

        assert "streaming" in info.capabilities
        assert "tool_use" in info.capabilities


class TestProtocolListModels:
    """provider-protocol:list_models:MUST:1,2"""

    @pytest.mark.asyncio
    async def test_returns_model_list(self, provider: GitHubCopilotProvider) -> None:
        """provider-protocol:list_models:MUST:1 - Returns model list."""
        models = await provider.list_models()

        assert isinstance(models, list)
        assert len(models) >= 2  # gpt-4 and gpt-4o

    @pytest.mark.asyncio
    async def test_includes_context_window(self, provider: GitHubCopilotProvider) -> None:
        """provider-protocol:list_models:MUST:2 - Includes context_window per model."""
        models = await provider.list_models()

        for model in models:
            assert model.context_window is not None
            assert model.context_window > 0
            assert model.max_output_tokens is not None


class TestProtocolComplete:
    """provider-protocol:complete:MUST:1-4"""

    @pytest.mark.asyncio
    async def test_accepts_kwargs(self, provider: GitHubCopilotProvider) -> None:
        """provider-protocol:complete - Uses **kwargs, not named streaming callback."""
        import inspect

        sig = inspect.signature(provider.complete)
        params = sig.parameters

        # Should have **kwargs or similar for extensibility
        has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        assert has_var_keyword, "complete() must accept **kwargs"


class TestProtocolParseToolCalls:
    """provider-protocol:parse_tool_calls:MUST:1-4"""

    def test_extracts_tool_calls(self, provider: GitHubCopilotProvider) -> None:
        """provider-protocol:parse_tool_calls:MUST:1 - Extracts tool calls from response."""
        # Create a mock response with tool calls (objects, not dicts)
        tc1 = MagicMock()
        tc1.id = "call_1"
        tc1.name = "read_file"
        tc1.arguments = {"path": "test.py"}

        response = MagicMock()
        response.tool_calls = [tc1]

        tool_calls = provider.parse_tool_calls(response)

        assert len(tool_calls) == 1
        assert tool_calls[0].id == "call_1"
        assert tool_calls[0].name == "read_file"

    def test_returns_empty_list_when_none(self, provider: GitHubCopilotProvider) -> None:
        """provider-protocol:parse_tool_calls:MUST:2 - Returns empty list when no tool calls."""
        response = MagicMock()
        response.tool_calls = []

        tool_calls = provider.parse_tool_calls(response)

        assert tool_calls == []

    def test_preserves_tool_call_ids(self, provider: GitHubCopilotProvider) -> None:
        """provider-protocol:parse_tool_calls:MUST:3 - Preserves tool call IDs."""
        tc1 = MagicMock()
        tc1.id = "unique_id_123"
        tc1.name = "test_tool"
        tc1.arguments = {}

        response = MagicMock()
        response.tool_calls = [tc1]

        tool_calls = provider.parse_tool_calls(response)

        assert tool_calls[0].id == "unique_id_123"

    def test_uses_arguments_not_input(self, provider: GitHubCopilotProvider) -> None:
        """provider-protocol:parse_tool_calls:MUST:4 - Uses 'arguments' field, not 'input'."""
        tc1 = MagicMock()
        tc1.id = "call_1"
        tc1.name = "test"
        tc1.arguments = {"key": "value"}

        response = MagicMock()
        response.tool_calls = [tc1]

        tool_calls = provider.parse_tool_calls(response)

        # ToolCall should have 'arguments' attribute
        assert hasattr(tool_calls[0], "arguments")
        assert tool_calls[0].arguments == {"key": "value"}
