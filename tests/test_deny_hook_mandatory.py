"""
Contract Compliance Tests: Deny + Destroy — Mandatory Deny Hook.

Contract: contracts/deny-destroy.md

Anchors covered:
- deny-destroy:DenyHook:MUST:1 — preToolUse hook installed on every session
- deny-destroy:DenyHook:MUST:2 — Hook returns DENY for all tool requests

Referenced from test_contract_deny_destroy.py docstring.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest


def _make_mock_sdk_client() -> tuple[AsyncMock, AsyncMock]:
    """Return (mock_sdk_client, mock_sdk_session) for session() tests."""
    mock_sdk_session = AsyncMock()
    mock_sdk_session.session_id = "sess-deny-test"
    mock_sdk_session.disconnect = AsyncMock()

    mock_sdk_client = AsyncMock()
    mock_sdk_client.create_session = AsyncMock(return_value=mock_sdk_session)

    return mock_sdk_client, mock_sdk_session


class TestDenyHookMandatoryClient:
    """deny-destroy:DenyHook:MUST:1,2 — Deny hook present on every session."""

    @pytest.mark.asyncio
    async def test_deny_hook_present_with_model(self) -> None:
        """deny-destroy:DenyHook:MUST:1 — preToolUse hook installed when model is specified."""
        from amplifier_module_provider_github_copilot.sdk_adapter.client import CopilotClientWrapper

        mock_sdk_client, _ = _make_mock_sdk_client()
        wrapper = CopilotClientWrapper(sdk_client=mock_sdk_client)

        async with wrapper.session(model="gpt-4"):
            pass

        kwargs = mock_sdk_client.create_session.call_args.kwargs
        assert "hooks" in kwargs, "Session config must always include 'hooks'"
        assert "on_pre_tool_use" in kwargs["hooks"], "'hooks' must include 'on_pre_tool_use'"
        assert callable(kwargs["hooks"]["on_pre_tool_use"]), "deny hook must be callable"

    @pytest.mark.asyncio
    async def test_deny_hook_present_without_tools(self) -> None:
        """deny-destroy:DenyHook:MUST:1 — hook installed even when no model or tools supplied."""
        from amplifier_module_provider_github_copilot.sdk_adapter.client import CopilotClientWrapper

        mock_sdk_client, _ = _make_mock_sdk_client()
        wrapper = CopilotClientWrapper(sdk_client=mock_sdk_client)

        async with wrapper.session():
            pass

        kwargs = mock_sdk_client.create_session.call_args.kwargs
        assert "hooks" in kwargs, "hooks must be set even with no model or tools"
        assert "on_pre_tool_use" in kwargs["hooks"]

    @pytest.mark.asyncio
    async def test_deny_hook_present_with_tools(self) -> None:
        """deny-destroy:DenyHook:MUST:1 — hook installed when Amplifier tools are forwarded."""
        from amplifier_module_provider_github_copilot.sdk_adapter.client import CopilotClientWrapper

        mock_sdk_client, _ = _make_mock_sdk_client()
        wrapper = CopilotClientWrapper(sdk_client=mock_sdk_client)

        tool: dict[str, Any] = {"name": "search", "description": "Search the web", "parameters": {}}
        async with wrapper.session(model="gpt-4", tools=[tool]):
            pass

        kwargs = mock_sdk_client.create_session.call_args.kwargs
        assert "hooks" in kwargs, "hooks must be set even when tools are forwarded"
        assert "on_pre_tool_use" in kwargs["hooks"]

    @pytest.mark.asyncio
    async def test_deny_hook_denies_any_tool(self) -> None:
        """deny-destroy:DenyHook:MUST:2 — Hook returns DENY for ALL tool requests."""
        from amplifier_module_provider_github_copilot.sdk_adapter.client import CopilotClientWrapper

        mock_sdk_client, _ = _make_mock_sdk_client()
        wrapper = CopilotClientWrapper(sdk_client=mock_sdk_client)

        async with wrapper.session(model="gpt-4"):
            pass

        deny_hook = mock_sdk_client.create_session.call_args.kwargs["hooks"]["on_pre_tool_use"]

        for tool_name in ("bash", "read_file", "write_file", "list_agents", "my_custom_tool"):
            result = deny_hook({"toolName": tool_name}, None)
            assert result.get("permissionDecision") == "deny", (
                f"Tool '{tool_name}' was NOT denied. Got: {result}"
            )

    @pytest.mark.asyncio
    async def test_deny_hook_suppress_output(self) -> None:
        """deny-destroy:DenyHook:MUST:2 — DENY response suppresses output to protect model."""
        from amplifier_module_provider_github_copilot.sdk_adapter.client import CopilotClientWrapper

        mock_sdk_client, _ = _make_mock_sdk_client()
        wrapper = CopilotClientWrapper(sdk_client=mock_sdk_client)

        async with wrapper.session(model="gpt-4"):
            pass

        deny_hook = mock_sdk_client.create_session.call_args.kwargs["hooks"]["on_pre_tool_use"]
        result = deny_hook({"toolName": "anything"}, None)

        # Output suppression prevents denial reason from teaching model that tools are blocked
        assert result.get("suppressOutput") is True


class TestMakeDenyHookConfig:
    """Unit tests for _make_deny_hook_config() standalone function."""

    def test_returns_on_pre_tool_use_key(self) -> None:
        """deny-destroy:DenyHook:MUST:1 — config has 'on_pre_tool_use' key."""
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            _make_deny_hook_config,  # pyright: ignore[reportPrivateUsage]
        )

        config = _make_deny_hook_config()
        assert "on_pre_tool_use" in config

    def test_hook_is_callable(self) -> None:
        """deny-destroy:DenyHook:MUST:1 — hook value must be callable."""
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            _make_deny_hook_config,  # pyright: ignore[reportPrivateUsage]
        )

        config = _make_deny_hook_config()
        assert callable(config["on_pre_tool_use"])

    def test_hook_denies(self) -> None:
        """deny-destroy:DenyHook:MUST:2 — function-level: hook denies tool execution."""
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            _make_deny_hook_config,  # pyright: ignore[reportPrivateUsage]
        )

        config = _make_deny_hook_config()
        result = config["on_pre_tool_use"]({"toolName": "bash"}, None)
        assert result["permissionDecision"] == "deny"


class TestDenyAllConstant:
    """deny-destroy:DenyHook:MUST:2 — DENY_ALL constant correctness."""

    def test_deny_all_decision_is_deny(self) -> None:
        from amplifier_module_provider_github_copilot.sdk_adapter.client import DENY_ALL

        assert DENY_ALL["permissionDecision"] == "deny"

    def test_deny_all_suppresses_output(self) -> None:
        from amplifier_module_provider_github_copilot.sdk_adapter.client import DENY_ALL

        assert DENY_ALL.get("suppressOutput") is True
