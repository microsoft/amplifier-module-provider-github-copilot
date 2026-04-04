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
        """provider-protocol:get_info:MUST:2 - Includes capabilities.

        Contract: Capabilities SHOULD use kernel constants from
        amplifier_core.capabilities (PROVIDER_CONTRACT.md:97).
        Using "tools" (not "tool_use") per kernel canonical naming.

        Provider-level capabilities = minimum ALL models support.
        Per-model capabilities (vision, thinking) are reported via list_models().
        """
        info = provider.get_info()

        # Per kernel capabilities.py constants: STREAMING="streaming", TOOLS="tools"
        assert "streaming" in info.capabilities
        assert "tools" in info.capabilities


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
    """Tests for complete() method signature.

    Note: Core complete() behavior (MUST:1-4) is tested in test_behaviors.py.
    This class tests signature requirements only.
    """

    @pytest.mark.asyncio
    async def test_accepts_kwargs(self, provider: GitHubCopilotProvider) -> None:
        """provider-protocol:complete signature — Uses **kwargs for extensibility."""
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

    def test_generates_id_when_sdk_omits_it(self, provider: GitHubCopilotProvider) -> None:
        """provider-protocol:parse_tool_calls:MUST:3 — C-3 regression guard.

        When the SDK emits a tool call with no id or an empty-string id,
        ``parse_tool_calls`` MUST return a ToolCall with a non-empty id so
        the kernel can use it for tool-result correlation.  An empty string silently
        breaks the correlation loop.
        """
        # SDK omitted the id entirely
        tc_no_id = MagicMock(spec=["name", "arguments"])
        tc_no_id.name = "get_weather"
        tc_no_id.arguments = {"city": "Seattle"}

        # SDK provided an empty-string id
        tc_empty_id = MagicMock()
        tc_empty_id.id = ""
        tc_empty_id.name = "list_files"
        tc_empty_id.arguments = {}

        response = MagicMock()
        response.tool_calls = [tc_no_id, tc_empty_id]

        tool_calls = provider.parse_tool_calls(response)

        assert len(tool_calls) == 2
        # MUST have non-empty ids (generated UUID fallback)
        assert tool_calls[0].id, "ToolCall with missing id MUST get a generated id"
        assert tool_calls[1].id, "ToolCall with empty-string id MUST get a generated id"
        # Generated ids must be distinct (two separate UUIDs)
        assert tool_calls[0].id != tool_calls[1].id

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


class TestConcreteProtocolBehavior:
    """P2-7 FIX: Test actual protocol behavior, not just signatures.

    These tests verify the protocol works correctly end-to-end,
    not just that methods exist with correct signatures.
    """

    def test_provider_info_has_all_required_fields(self, provider: GitHubCopilotProvider) -> None:
        """P2-7: get_info() returns ProviderInfo with all required fields populated."""
        info = provider.get_info()

        # Required fields MUST be non-empty
        assert info.id, "id MUST be non-empty"
        assert info.display_name, "display_name MUST be non-empty"
        # Capabilities should be a list of strings
        assert isinstance(info.capabilities, list), "capabilities MUST be a list"

        # Specific capabilities MUST exist
        assert "streaming" in info.capabilities, "streaming MUST be in capabilities"

    def test_parse_tool_calls_handles_none_tool_calls(
        self, provider: GitHubCopilotProvider
    ) -> None:
        """P2-7: parse_tool_calls handles response with tool_calls=None."""
        response = MagicMock()
        response.tool_calls = None

        tool_calls = provider.parse_tool_calls(response)

        assert tool_calls == [], "None tool_calls should return empty list"

    def test_parse_tool_calls_handles_dict_arguments(self, provider: GitHubCopilotProvider) -> None:
        """P2-7: parse_tool_calls handles tool calls with dict arguments."""
        tc = MagicMock()
        tc.id = "call_dict"
        tc.name = "complex_tool"
        args: dict[str, object] = {"nested": {"key": "value"}, "list": [1, 2, 3]}
        tc.arguments = args

        response = MagicMock()
        response.tool_calls = [tc]

        tool_calls = provider.parse_tool_calls(response)

        assert len(tool_calls) == 1
        # Verify nested structure preserved (type ignores for nested access)
        result_args = tool_calls[0].arguments
        assert "nested" in result_args
        assert "list" in result_args
        # Cast explicitly for type checking
        nested_val = result_args["nested"]  # pyright: ignore[reportArgumentType]
        assert isinstance(nested_val, dict)
        assert nested_val["key"] == "value"
        assert result_args["list"] == [1, 2, 3]  # pyright: ignore[reportArgumentType]

    def test_provider_name_immutable(self, provider: GitHubCopilotProvider) -> None:
        """P2-7: Provider name is immutable (read-only property)."""
        with pytest.raises(AttributeError):
            provider.name = "different-name"  # type: ignore[misc]
