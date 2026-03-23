"""
Tests for mandatory deny hook installation.

Contract: deny-destroy:DenyHook:MUST:1

The deny hook is passed via session config 'hooks' key,
not via register_pre_tool_use_hook() method.

These tests verify:
1. Session config includes 'hooks' key with deny hook
2. The correct SDK API is used (send + on, not send_message)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from amplifier_module_provider_github_copilot.error_translation import ProviderUnavailableError
from amplifier_module_provider_github_copilot.provider import (
    CompletionRequest,
    SessionConfig,
    complete,
)
from amplifier_module_provider_github_copilot.sdk_adapter.client import CopilotClientWrapper


class TestDenyHookMandatoryProvider:
    """Tests for mandatory deny hook in provider.py complete() path."""

    @pytest.mark.asyncio
    async def test_raises_when_session_lacks_sdk_methods(self) -> None:
        """deny-destroy:DenyHook:MUST:1 - raises when required SDK methods absent.

        The SDK requires on() and send() methods for streaming.
        When SDK session doesn't have these, complete() MUST raise ProviderUnavailableError.
        """
        # Create mock session WITHOUT on() and send() methods
        mock_session = MagicMock(spec=["disconnect"])

        async def sdk_create_fn(config: SessionConfig) -> MagicMock:
            return mock_session

        request = CompletionRequest(prompt="test")

        with pytest.raises(ProviderUnavailableError) as exc_info:
            async for _ in complete(request, sdk_create_fn=sdk_create_fn):
                pass

        assert "on()" in str(exc_info.value) or "send()" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_uses_correct_sdk_api_when_methods_present(self) -> None:
        """deny-destroy:DenyHook:MUST:1 - uses correct SDK API pattern.

        The correct SDK pattern is:
        1. Register event handler with on()
        2. Send message with send()
        """
        # Create mock session WITH correct SDK methods
        mock_session = MagicMock(spec=["on", "send", "disconnect"])
        mock_session.disconnect = AsyncMock()

        # on() returns unsubscribe function and stores the handler
        unsubscribe = MagicMock()
        handlers: list[Any] = []

        def mock_on(handler: Any) -> MagicMock:
            handlers.append(handler)
            return unsubscribe

        mock_session.on = MagicMock(side_effect=mock_on)

        # send() triggers the handler with a SESSION_IDLE event
        async def mock_send(prompt: str, attachments: list[Any] | None = None) -> str:
            # Simulate SDK behavior: emit SESSION_IDLE after processing
            if handlers:
                idle_event = MagicMock()
                idle_event.type = "SESSION_IDLE"
                handlers[0](idle_event)
            return "message-id"

        mock_session.send = AsyncMock(side_effect=mock_send)

        async def sdk_create_fn(config: SessionConfig) -> MagicMock:
            return mock_session

        request = CompletionRequest(prompt="test")

        # Should not raise
        async for _ in complete(request, sdk_create_fn=sdk_create_fn):
            pass

        # Verify correct SDK methods were called
        mock_session.on.assert_called_once()
        mock_session.send.assert_called_once()


class TestDenyHookMandatoryClient:
    """Tests for mandatory deny hook in client.py session() path.

    Deny hook is now passed via session config 'hooks' key,
    NOT via register_pre_tool_use_hook() method call.
    """

    @pytest.mark.asyncio
    async def test_session_config_includes_hooks_key(self) -> None:
        """deny-destroy:DenyHook:MUST:1 - session config includes hooks.

        The deny hook is passed via session config 'hooks' key at
        session creation time, not via a method call after creation.
        """
        # Track the config passed to create_session
        captured_configs: list[dict[str, Any]] = []

        mock_sdk_client = MagicMock()
        mock_session = MagicMock(spec=["disconnect", "on", "send", "session_id"])
        mock_session.session_id = "test-session-123"
        mock_session.disconnect = AsyncMock()
        mock_session.on = MagicMock(return_value=MagicMock())
        mock_session.send = AsyncMock()

        async def capture_create_session(**config: Any) -> MagicMock:
            captured_configs.append(config)
            return mock_session

        mock_sdk_client.create_session = AsyncMock(side_effect=capture_create_session)

        wrapper = CopilotClientWrapper(sdk_client=mock_sdk_client)

        async with wrapper.session(model="gpt-4"):
            pass

        # Verify session config included 'hooks' key
        assert len(captured_configs) == 1
        config: dict[str, Any] = captured_configs[0]
        assert "hooks" in config, "Session config must include 'hooks' key for deny hook"
        assert "on_pre_tool_use" in config["hooks"], "hooks must contain on_pre_tool_use"

    @pytest.mark.asyncio
    async def test_hooks_deny_function_denies_all(self) -> None:
        """deny-destroy:DenyHook:MUST:2 - hook function denies all tools.

        The on_pre_tool_use hook MUST return a deny decision.
        """
        # Capture the hooks config
        captured_hooks: list[dict[str, Any]] = []

        mock_sdk_client = MagicMock()
        mock_session = MagicMock(spec=["disconnect", "on", "send", "session_id"])
        mock_session.session_id = "test-session-123"
        mock_session.disconnect = AsyncMock()
        mock_session.on = MagicMock(return_value=MagicMock())
        mock_session.send = AsyncMock()

        async def capture_create_session(**config: Any) -> MagicMock:
            if "hooks" in config:
                captured_hooks.append(config["hooks"])
            return mock_session

        mock_sdk_client.create_session = AsyncMock(side_effect=capture_create_session)

        wrapper = CopilotClientWrapper(sdk_client=mock_sdk_client)

        async with wrapper.session(model="gpt-4"):
            pass

        # Verify hooks were captured and deny function works
        assert len(captured_hooks) == 1
        hooks: dict[str, Any] = captured_hooks[0]
        deny_hook: Any = hooks.get("on_pre_tool_use")
        assert deny_hook is not None, "on_pre_tool_use hook must be set"

        # Test the deny hook returns deny decision
        result: dict[str, Any] = deny_hook({"toolName": "test_tool"}, {})
        assert result["permissionDecision"] == "deny", "Hook must deny all tools"
        # Hook is passed via session config, not via register_pre_tool_use_hook()
        # The captured_hooks validation above (lines 171-178) verifies correct behavior
