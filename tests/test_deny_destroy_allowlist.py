"""
Contract Compliance Tests: Tool Allowlist and Suppression.

Contract: contracts/deny-destroy.md

Test Anchors:
- deny-destroy:ToolSuppression:MUST:1 — available_tools set to Amplifier tool names
- deny-destroy:ToolSuppression:MUST:2 — overrides_built_in_tool=True on each tool
- deny-destroy:ToolSuppression:MUST:3 — available_tools=[] when no tools provided
- deny-destroy:Allowlist:MUST:1 — available_tools contains only Amplifier tool names
- deny-destroy:Allowlist:MUST:2 — SDK built-ins NOT in allowlist
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest


class MockSDKSession:
    """Test double for SDK session."""

    def __init__(self, session_id: str = "allowlist-test-session") -> None:
        self.session_id = session_id
        self.disconnected = False

    async def disconnect(self) -> None:
        """Disconnect stub."""
        self.disconnected = True


def _make_wrapper() -> tuple[AsyncMock, Any]:
    """Return (mock_sdk_client, wrapper) for session-level tests."""
    from amplifier_module_provider_github_copilot.sdk_adapter.client import CopilotClientWrapper

    mock_sdk_client = AsyncMock()
    mock_sdk_client.create_session = AsyncMock(return_value=MockSDKSession())
    return mock_sdk_client, CopilotClientWrapper(sdk_client=mock_sdk_client)


class TestToolSuppressionWithTools:
    """deny-destroy:ToolSuppression:MUST:1 — available_tools set when tools provided."""

    @pytest.mark.asyncio
    async def test_available_tools_set_to_tool_names(self) -> None:
        """deny-destroy:ToolSuppression:MUST:1 — available_tools equals tool names list.

        When tools are provided to session(), available_tools must be set
        to the list of tool names from the provided tools.
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import CopilotClientWrapper

        mock_sdk_client = AsyncMock()
        mock_sdk_client.create_session = AsyncMock(return_value=MockSDKSession())

        wrapper = CopilotClientWrapper(sdk_client=mock_sdk_client)

        tools: list[dict[str, Any]] = [
            {"name": "amplifier_search", "description": "Search", "parameters": {}},
            {"name": "amplifier_execute", "description": "Execute", "parameters": {}},
        ]

        async with wrapper.session(model="gpt-4", tools=tools):
            pass

        kwargs = mock_sdk_client.create_session.call_args.kwargs
        assert "available_tools" in kwargs, (
            "ToolSuppression:MUST:1 — create_session must include available_tools"
        )
        assert kwargs["available_tools"] == ["amplifier_search", "amplifier_execute"], (
            f"ToolSuppression:MUST:1 — available_tools must equal tool names, "
            f"got {kwargs['available_tools']}"
        )

    @pytest.mark.asyncio
    async def test_single_tool_creates_single_item_allowlist(self) -> None:
        """deny-destroy:ToolSuppression:MUST:1 — single tool creates single-item list."""
        from amplifier_module_provider_github_copilot.sdk_adapter.client import CopilotClientWrapper

        mock_sdk_client = AsyncMock()
        mock_sdk_client.create_session = AsyncMock(return_value=MockSDKSession())

        wrapper = CopilotClientWrapper(sdk_client=mock_sdk_client)

        tools: list[dict[str, Any]] = [
            {"name": "my_tool", "description": "My tool", "parameters": {}},
        ]

        async with wrapper.session(model="gpt-4", tools=tools):
            pass

        kwargs = mock_sdk_client.create_session.call_args.kwargs
        assert kwargs["available_tools"] == ["my_tool"], (
            f"ToolSuppression:MUST:1 — single tool should create single-item list, "
            f"got {kwargs['available_tools']}"
        )


class TestToolSuppressionOverridesBuiltIn:
    """deny-destroy:ToolSuppression:MUST:2 — overrides_built_in_tool=True on each tool."""

    def test_convert_tools_sets_overrides_built_in(self) -> None:
        """deny-destroy:ToolSuppression:MUST:2 — SDKToolWrapper has overrides_built_in_tool=True.

        The convert_tools_for_sdk function must set overrides_built_in_tool=True
        on each tool wrapper to handle name conflicts with SDK built-ins.
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.types import convert_tools_for_sdk

        tools: list[dict[str, Any]] = [
            {"name": "bash", "description": "Execute bash", "parameters": {}},
            {"name": "edit", "description": "Edit files", "parameters": {}},
            {"name": "custom_tool", "description": "Custom", "parameters": {}},
        ]

        sdk_tools = convert_tools_for_sdk(tools)

        assert len(sdk_tools) == 3
        for sdk_tool in sdk_tools:
            assert sdk_tool.overrides_built_in_tool is True, (
                f"ToolSuppression:MUST:2 — tool '{sdk_tool.name}' must have "
                f"overrides_built_in_tool=True"
            )

    def test_convert_tools_preserves_tool_name(self) -> None:
        """deny-destroy:ToolSuppression:MUST:2 — tool names preserved in conversion."""
        from amplifier_module_provider_github_copilot.sdk_adapter.types import convert_tools_for_sdk

        tools: list[dict[str, Any]] = [
            {"name": "my_amplifier_tool", "description": "Test", "parameters": {}},
        ]

        sdk_tools = convert_tools_for_sdk(tools)

        assert len(sdk_tools) == 1
        assert sdk_tools[0].name == "my_amplifier_tool"


class TestToolSuppressionNoTools:
    """deny-destroy:ToolSuppression:MUST:3 — available_tools=[] when no tools."""

    @pytest.mark.asyncio
    async def test_empty_allowlist_when_no_tools(self) -> None:
        """deny-destroy:ToolSuppression:MUST:3 — available_tools=[] blocks SDK built-ins.

        When no Amplifier tools are provided, available_tools must be set
        to an empty list to prevent SDK built-ins from appearing to the model.
        """
        mock_sdk_client, wrapper = _make_wrapper()

        # No tools provided
        async with wrapper.session(model="gpt-4"):
            pass

        kwargs = mock_sdk_client.create_session.call_args.kwargs
        assert "available_tools" in kwargs, (
            "ToolSuppression:MUST:3 — available_tools must be set even without tools"
        )
        assert kwargs["available_tools"] == [], (
            f"ToolSuppression:MUST:3 — available_tools must be [] when no tools, "
            f"got {kwargs['available_tools']}"
        )

    @pytest.mark.asyncio
    async def test_empty_allowlist_when_tools_is_none(self) -> None:
        """deny-destroy:ToolSuppression:MUST:3 — tools=None produces available_tools=[]."""
        mock_sdk_client, wrapper = _make_wrapper()

        # Explicitly pass tools=None
        async with wrapper.session(model="gpt-4", tools=None):
            pass

        kwargs = mock_sdk_client.create_session.call_args.kwargs
        assert kwargs["available_tools"] == [], (
            "ToolSuppression:MUST:3 — tools=None must produce available_tools=[]"
        )


class TestAllowlistContainsOnlyAmplifierTools:
    """deny-destroy:Allowlist:MUST:1 — available_tools contains only Amplifier tool names."""

    @pytest.mark.asyncio
    async def test_allowlist_exactly_matches_provided_tools(self) -> None:
        """deny-destroy:Allowlist:MUST:1 — allowlist exactly matches provided tools.

        The available_tools list must contain exactly the tool names
        from the tools parameter, nothing more, nothing less.
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import CopilotClientWrapper

        mock_sdk_client = AsyncMock()
        mock_sdk_client.create_session = AsyncMock(return_value=MockSDKSession())

        wrapper = CopilotClientWrapper(sdk_client=mock_sdk_client)

        tools: list[dict[str, Any]] = [
            {"name": "tool_a", "description": "A", "parameters": {}},
            {"name": "tool_b", "description": "B", "parameters": {}},
            {"name": "tool_c", "description": "C", "parameters": {}},
        ]

        async with wrapper.session(model="gpt-4", tools=tools):
            pass

        kwargs = mock_sdk_client.create_session.call_args.kwargs
        expected_names = ["tool_a", "tool_b", "tool_c"]
        assert kwargs["available_tools"] == expected_names, (
            f"Allowlist:MUST:1 — available_tools must exactly match provided tools, "
            f"expected {expected_names}, got {kwargs['available_tools']}"
        )


class TestAllowlistExcludesSDKBuiltins:
    """deny-destroy:Allowlist:MUST:2 — SDK built-ins NOT in allowlist."""

    # Known SDK built-in tool names that should never appear in allowlist
    SDK_BUILTINS = [
        "bash",
        "edit",
        "list_agents",
        "read_file",
        "write_file",
        "search",
        "glob",
        "grep",
        "view",
        "powershell",
        "web_fetch",
    ]

    @pytest.mark.asyncio
    async def test_sdk_builtins_not_in_allowlist_no_tools(self) -> None:
        """deny-destroy:Allowlist:MUST:2 — SDK built-ins absent when no tools.

        When no tools are provided, available_tools=[] means no SDK
        built-ins can be in the allowlist.
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import CopilotClientWrapper

        mock_sdk_client = AsyncMock()
        mock_sdk_client.create_session = AsyncMock(return_value=MockSDKSession())

        wrapper = CopilotClientWrapper(sdk_client=mock_sdk_client)

        async with wrapper.session(model="gpt-4"):
            pass

        kwargs = mock_sdk_client.create_session.call_args.kwargs
        available = kwargs["available_tools"]

        for builtin in self.SDK_BUILTINS:
            assert builtin not in available, (
                f"Allowlist:MUST:2 — SDK built-in '{builtin}' must NOT be in allowlist"
            )

    @pytest.mark.asyncio
    async def test_sdk_builtins_not_in_allowlist_with_tools(self) -> None:
        """deny-destroy:Allowlist:MUST:2 — SDK built-ins absent when tools provided.

        When Amplifier tools are provided, the allowlist should only
        contain those tool names, not SDK built-ins.
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import CopilotClientWrapper

        mock_sdk_client = AsyncMock()
        mock_sdk_client.create_session = AsyncMock(return_value=MockSDKSession())

        wrapper = CopilotClientWrapper(sdk_client=mock_sdk_client)

        tools: list[dict[str, Any]] = [
            {"name": "amplifier_tool", "description": "Amplifier tool", "parameters": {}},
        ]

        async with wrapper.session(model="gpt-4", tools=tools):
            pass

        kwargs = mock_sdk_client.create_session.call_args.kwargs
        available = kwargs["available_tools"]

        # Should only contain our tool
        assert available == ["amplifier_tool"], (
            f"Allowlist:MUST:2 — only Amplifier tools should be in allowlist, got {available}"
        )

        # Double-check no SDK built-ins leaked in
        for builtin in self.SDK_BUILTINS:
            assert builtin not in available, (
                f"Allowlist:MUST:2 — SDK built-in '{builtin}' leaked into allowlist"
            )

    @pytest.mark.asyncio
    async def test_amplifier_tool_named_like_builtin_still_allowed(self) -> None:
        """deny-destroy:Allowlist:MUST:2 — Amplifier tool named 'bash' is allowed.

        If an Amplifier tool happens to be named like a built-in (e.g., 'bash'),
        it should still appear in the allowlist because it's an Amplifier tool.
        The overrides_built_in_tool=True handles the conflict at SDK level.
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import CopilotClientWrapper

        mock_sdk_client = AsyncMock()
        mock_sdk_client.create_session = AsyncMock(return_value=MockSDKSession())

        wrapper = CopilotClientWrapper(sdk_client=mock_sdk_client)

        # Amplifier provides a tool named 'bash' (overriding the built-in)
        tools: list[dict[str, Any]] = [
            {"name": "bash", "description": "Amplifier bash (not SDK)", "parameters": {}},
        ]

        async with wrapper.session(model="gpt-4", tools=tools):
            pass

        kwargs = mock_sdk_client.create_session.call_args.kwargs
        # This is the Amplifier tool 'bash', not the SDK built-in
        assert "bash" in kwargs["available_tools"], (
            "Allowlist:MUST:2 — Amplifier tool named 'bash' should be in allowlist"
        )
