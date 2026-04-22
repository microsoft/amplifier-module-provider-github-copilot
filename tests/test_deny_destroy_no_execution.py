"""
Contract Compliance Tests: No Direct Tool Execution.

Contract: contracts/deny-destroy.md

Test Anchors:
- deny-destroy:NoExecution:MUST:1 — Tool requests captured from SDK events
- deny-destroy:NoExecution:MUST:2 — Tool requests returned to orchestrator
- deny-destroy:NoExecution:MUST:3 — SDK never executes tools directly
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest


class MockSDKSession:
    """Test double for SDK session."""

    def __init__(self, session_id: str = "no-exec-test-session") -> None:
        self.session_id = session_id
        self.disconnected = False

    async def disconnect(self) -> None:
        """Disconnect stub."""
        self.disconnected = True


class TestToolCaptureFromEvents:
    """deny-destroy:NoExecution:MUST:1 — Tool requests captured from SDK events."""

    def test_handler_captures_tool_use_event(self) -> None:
        """deny-destroy:NoExecution:MUST:1 — handler captures ASSISTANT_MESSAGE tool events.

        ToolCaptureHandler.on_event() must extract tool_requests from
        ASSISTANT_MESSAGE events emitted by the SDK.
        """
        from amplifier_module_provider_github_copilot.sdk_adapter import ToolCaptureHandler

        handler = ToolCaptureHandler()

        # Simulate SDK ASSISTANT_MESSAGE event with tool_requests
        mock_event = {
            "type": "assistant.message",
            "data": {
                "tool_requests": [
                    {
                        "tool_call_id": "call-123",
                        "name": "search",
                        "arguments": {"query": "test"},
                    },
                ],
            },
        }

        handler.on_event(mock_event)

        assert len(handler.captured_tools) == 1, (
            f"NoExecution:MUST:1 — handler must capture tool from event, "
            f"got {len(handler.captured_tools)}"
        )
        assert handler.captured_tools[0]["name"] == "search", (
            f"NoExecution:MUST:1 — captured tool name must be 'search', "
            f"got {handler.captured_tools[0].get('name')}"
        )

    def test_handler_captures_multiple_tools(self) -> None:
        """deny-destroy:NoExecution:MUST:1 — handler captures multiple tool requests."""
        from amplifier_module_provider_github_copilot.sdk_adapter import ToolCaptureHandler

        handler = ToolCaptureHandler()

        mock_event = {
            "type": "assistant.message",
            "data": {
                "tool_requests": [
                    {"tool_call_id": "call-1", "name": "tool_a", "arguments": {}},
                    {"tool_call_id": "call-2", "name": "tool_b", "arguments": {}},
                ],
            },
        }

        handler.on_event(mock_event)

        assert len(handler.captured_tools) == 2, (
            f"NoExecution:MUST:1 — handler must capture all tools from event, "
            f"got {len(handler.captured_tools)}"
        )


class TestToolRequestsReturnedToOrchestrator:
    """deny-destroy:NoExecution:MUST:2 — Tool requests returned to orchestrator."""

    def test_captured_tools_accessible_via_property(self) -> None:
        """deny-destroy:NoExecution:MUST:2 — captured_tools property exposes tool list.

        After events are processed, the captured tools must be accessible
        via the captured_tools property for return to Amplifier.
        """
        from amplifier_module_provider_github_copilot.sdk_adapter import ToolCaptureHandler

        handler = ToolCaptureHandler()

        mock_event = {
            "type": "assistant.message",
            "data": {
                "tool_requests": [
                    {"tool_call_id": "call-xyz", "name": "execute", "arguments": {"cmd": "ls"}},
                ],
            },
        }

        handler.on_event(mock_event)

        tools = handler.captured_tools
        assert isinstance(tools, list), (
            f"NoExecution:MUST:2 — captured_tools must be a list, got {type(tools)}"
        )
        assert len(tools) == 1
        assert tools[0]["id"] == "call-xyz", (
            f"NoExecution:MUST:2 — tool id must be preserved, got {tools[0].get('id')}"
        )
        assert tools[0]["name"] == "execute"
        assert tools[0]["arguments"] == {"cmd": "ls"}

    def test_handler_deduplicates_by_tool_call_id(self) -> None:
        """deny-destroy:NoExecution:MUST:2 — duplicate tool_call_ids are deduplicated.

        If the SDK sends the same tool_call_id multiple times (e.g., in
        streaming), the handler must deduplicate to prevent double execution.
        """
        from amplifier_module_provider_github_copilot.sdk_adapter import ToolCaptureHandler

        handler = ToolCaptureHandler()

        # Same tool_call_id sent twice
        event1 = {
            "type": "assistant.message",
            "data": {
                "tool_requests": [
                    {"tool_call_id": "dup-id", "name": "tool", "arguments": {}},
                ],
            },
        }
        event2 = {
            "type": "assistant.message",
            "data": {
                "tool_requests": [
                    {"tool_call_id": "dup-id", "name": "tool", "arguments": {}},
                ],
            },
        }

        handler.on_event(event1)
        handler.on_event(event2)

        # Should only capture once (deduplication)
        assert len(handler.captured_tools) == 1, (
            f"NoExecution:MUST:2 — duplicate tool_call_id must be deduplicated, "
            f"got {len(handler.captured_tools)}"
        )


class TestSDKNeverExecutesTools:
    """deny-destroy:NoExecution:MUST:3 — SDK never executes tools directly."""

    def test_deny_hook_denies_all_tools(self) -> None:
        """deny-destroy:NoExecution:MUST:3 — deny hook returns deny for ALL tools.

        The preToolUse deny hook must return permissionDecision="deny"
        for every tool request, preventing SDK from executing any tools.
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            _make_deny_hook_config,  # pyright: ignore[reportPrivateUsage]
        )

        config = _make_deny_hook_config()
        deny_hook = config["on_pre_tool_use"]

        # Test various tool names
        test_tools = [
            "bash",
            "edit",
            "list_agents",
            "read_file",
            "write_file",
            "unknown_tool",
            "my_custom_tool",
            "dangerous_tool",
        ]

        for tool_name in test_tools:
            result = deny_hook({"toolName": tool_name}, None)
            assert result.get("permissionDecision") == "deny", (
                f"NoExecution:MUST:3 — tool '{tool_name}' must be denied, got {result}"
            )

    def test_deny_hook_suppresses_output(self) -> None:
        """deny-destroy:NoExecution:MUST:3 — deny hook suppresses output.

        The suppressOutput flag prevents the denial message from
        appearing in the conversation and teaching the model.
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            _make_deny_hook_config,  # pyright: ignore[reportPrivateUsage]
        )

        config = _make_deny_hook_config()
        deny_hook = config["on_pre_tool_use"]

        result = deny_hook({"toolName": "any_tool"}, None)
        assert result.get("suppressOutput") is True, (
            "NoExecution:MUST:3 — deny hook must suppress output"
        )

    @pytest.mark.asyncio
    async def test_hooks_always_installed_in_session(self) -> None:
        """deny-destroy:NoExecution:MUST:3 — hooks installed on every session.

        The session() method must always include the deny hooks,
        ensuring no session can execute tools directly.
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import CopilotClientWrapper

        mock_sdk_client = AsyncMock()
        mock_sdk_client.create_session = AsyncMock(return_value=MockSDKSession())

        wrapper = CopilotClientWrapper(sdk_client=mock_sdk_client)

        async with wrapper.session(model="gpt-4"):
            pass

        kwargs = mock_sdk_client.create_session.call_args.kwargs
        assert "hooks" in kwargs, "NoExecution:MUST:3 — hooks must be in session config"
        assert "on_pre_tool_use" in kwargs["hooks"], (
            "NoExecution:MUST:3 — on_pre_tool_use deny hook must be installed"
        )


class TestDenyAllConstant:
    """deny-destroy:NoExecution:MUST:3 — DENY_ALL constant structure."""

    def test_deny_all_structure(self) -> None:
        """deny-destroy:NoExecution:MUST:3 — DENY_ALL has correct structure."""
        from amplifier_module_provider_github_copilot.sdk_adapter.client import DENY_ALL

        assert DENY_ALL["permissionDecision"] == "deny", (
            f"NoExecution:MUST:3 — DENY_ALL must have permissionDecision='deny', "
            f"got {DENY_ALL.get('permissionDecision')}"
        )
        assert "permissionDecisionReason" in DENY_ALL, (
            "NoExecution:MUST:3 — DENY_ALL must have permissionDecisionReason"
        )
        assert DENY_ALL["suppressOutput"] is True, (
            "NoExecution:MUST:3 — DENY_ALL must suppress output"
        )


class TestToolCaptureHandlerCaptureFlagComplete:
    """deny-destroy:NoExecution:MUST:1,2 — capture_complete flag behavior."""

    def test_capture_complete_flag_set_after_capture(self) -> None:
        """deny-destroy:NoExecution:MUST:1 — capture_complete set when tools captured."""
        from amplifier_module_provider_github_copilot.sdk_adapter import ToolCaptureHandler

        handler = ToolCaptureHandler()

        assert handler.capture_complete is False, "Should start as not complete"

        mock_event = {
            "type": "assistant.message",
            "data": {
                "tool_requests": [
                    {"tool_call_id": "call-1", "name": "tool", "arguments": {}},
                ],
            },
        }

        handler.on_event(mock_event)

        assert handler.capture_complete is True, (
            "NoExecution:MUST:1 — capture_complete must be True after capturing tools"
        )

    def test_callback_invoked_on_capture(self) -> None:
        """deny-destroy:NoExecution:MUST:2 — on_capture_complete callback invoked."""
        from amplifier_module_provider_github_copilot.sdk_adapter import ToolCaptureHandler

        callback_invoked = False

        def on_complete() -> None:
            nonlocal callback_invoked
            callback_invoked = True

        handler = ToolCaptureHandler(on_capture_complete=on_complete)

        mock_event = {
            "type": "assistant.message",
            "data": {
                "tool_requests": [
                    {"tool_call_id": "call-1", "name": "tool", "arguments": {}},
                ],
            },
        }

        handler.on_event(mock_event)

        assert callback_invoked is True, (
            "NoExecution:MUST:2 — on_capture_complete callback must be invoked"
        )
