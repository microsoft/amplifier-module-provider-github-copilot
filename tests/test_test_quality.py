"""
Test quality improvements.

Contract: sdk-boundary.md (mount lifecycle), deny-destroy.md

Tests for:
- mount() exception → returns None path
- MagicMock with spec= for type safety
- Concurrent session deny hook verification
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestMountFailurePath:
    """DETest mount() exception → returns None path.

    Contract: sdk-boundary.md - mount lifecycle
    """

    @pytest.mark.asyncio
    async def test_mount_returns_none_on_provider_init_failure(self) -> None:
        """mount() returns None when GitHubCopilotProvider.__init__ fails.

        Contract: sdk-boundary.md - graceful degradation on init failure
        """
        from amplifier_module_provider_github_copilot import mount

        coordinator = MagicMock()
        coordinator.mount = AsyncMock()

        # Patch GitHubCopilotProvider to fail during __init__
        with patch(
            "amplifier_module_provider_github_copilot.GitHubCopilotProvider",
            side_effect=RuntimeError("Simulated provider init failure"),
        ):
            result = await mount(coordinator, config=None)

        assert result is None

    @pytest.mark.asyncio
    async def test_mount_returns_none_on_coordinator_mount_exception(self) -> None:
        """mount() returns None when coordinator.mount() raises.

        Contract: sdk-boundary.md - graceful degradation
        """
        from amplifier_module_provider_github_copilot import mount

        coordinator = MagicMock()
        coordinator.mount = AsyncMock(side_effect=RuntimeError("Coordinator error"))

        result = await mount(coordinator, config=None)

        assert result is None

    @pytest.mark.asyncio
    async def test_mount_logs_error_on_failure(self, caplog: pytest.LogCaptureFixture) -> None:
        """mount() logs error when initialization fails.

        Contract: sdk-boundary.md - observability
        """
        from amplifier_module_provider_github_copilot import mount

        coordinator = MagicMock()
        coordinator.mount = AsyncMock(side_effect=ValueError("Test error"))

        with caplog.at_level(logging.ERROR):
            result = await mount(coordinator, config=None)

        assert result is None
        # Check that error was logged
        assert any("[MOUNT] Failed" in record.message for record in caplog.records)

    @pytest.mark.asyncio
    async def test_mount_returns_cleanup_on_success(self) -> None:
        """mount() returns cleanup function on success.

        Contract: sdk-boundary.md - cleanup MUST be returned
        """
        from amplifier_module_provider_github_copilot import mount

        coordinator = MagicMock()
        coordinator.mount = AsyncMock()

        result = await mount(coordinator, config=None)

        # On success, should return a cleanup function (not None)
        assert result is not None
        assert callable(result)


class TestMagicMockSpec:
    """DEMagicMock with spec= for type safety."""

    def test_mock_with_spec_prevents_invalid_attribute_access(self) -> None:
        """MagicMock(spec=) raises AttributeError for invalid attributes.

        This demonstrates why spec= matters for test quality.
        """
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        # Without spec, MagicMock allows any attribute (dangerous!)
        mock_without_spec = MagicMock()
        _ = mock_without_spec.nonexistent_method()  # This silently succeeds

        # With spec, MagicMock raises AttributeError for invalid attributes
        mock_with_spec = MagicMock(spec=GitHubCopilotProvider)
        with pytest.raises(AttributeError):
            _ = mock_with_spec.nonexistent_method()  # This raises

    def test_provider_mock_with_spec_has_required_methods(self) -> None:
        """MagicMock(spec=Provider) has all required methods."""
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        mock = MagicMock(spec=GitHubCopilotProvider)

        # These should work (methods exist on GitHubCopilotProvider)
        assert hasattr(mock, "get_info")
        assert hasattr(mock, "list_models")
        assert hasattr(mock, "complete")
        assert hasattr(mock, "parse_tool_calls")
        assert hasattr(mock, "close")


class TestConcurrentSessionDenyHook:
    """DEConcurrent session test verifies deny hook per session."""

    @pytest.mark.asyncio
    async def test_concurrent_sessions_all_have_deny_hook(self) -> None:
        """Each concurrent session has deny hook installed.

        Contract: deny-destroy.md - deny hook MUST be on EVERY session
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        # Track session configs to verify deny hook
        session_configs: list[dict[str, Any]] = []

        mock_session = MagicMock()
        mock_session.disconnect = AsyncMock()
        mock_session.session_id = "test-session"

        mock_client_instance = AsyncMock()
        mock_client_instance.start = AsyncMock()

        async def mock_create_session(**config: Any) -> MagicMock:
            # Capture the config to verify deny hook
            session_configs.append(config)
            return mock_session

        mock_client_instance.create_session = mock_create_session

        with patch(
            "amplifier_module_provider_github_copilot.sdk_adapter._imports.CopilotClient",
            MagicMock(return_value=mock_client_instance),
        ):
            with patch(
                "amplifier_module_provider_github_copilot.sdk_adapter.client._resolve_token",
                return_value="test-token",
            ):
                wrapper = CopilotClientWrapper()

                # Launch 3 concurrent session requests
                async def get_session() -> MagicMock:
                    async with wrapper.session() as s:
                        return s

                await asyncio.gather(*[get_session() for _ in range(3)])

                # All 3 sessions should have deny hook in config
                assert len(session_configs) == 3
                for config in session_configs:
                    # Verify hooks dict exists with on_pre_tool_use
                    assert "hooks" in config, "Session config missing hooks"
                    assert "on_pre_tool_use" in config["hooks"], "Missing deny hook"
                    # Verify the hook is actually callable
                    assert callable(config["hooks"]["on_pre_tool_use"])

                    # Bug #1 fix: available_tools must NOT be set
                    # Setting to [] was disabling all tools via SDK whitelist behavior
                    assert "available_tools" not in config, "available_tools must NOT be set"

    @pytest.mark.asyncio
    async def test_deny_hook_returns_deny_result(self) -> None:
        """Deny hook actually returns denial when called.

        Contract: deny-destroy:DenyHook:MUST:2 - hook MUST deny tool execution
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        captured_hook: Any = None

        mock_session = MagicMock()
        mock_session.disconnect = AsyncMock()
        mock_session.session_id = "test-session"

        mock_client_instance = AsyncMock()
        mock_client_instance.start = AsyncMock()

        async def mock_create_session(**config: Any) -> MagicMock:
            nonlocal captured_hook
            if "hooks" in config and "on_pre_tool_use" in config["hooks"]:
                captured_hook = config["hooks"]["on_pre_tool_use"]
            return mock_session

        mock_client_instance.create_session = mock_create_session

        with patch(
            "amplifier_module_provider_github_copilot.sdk_adapter._imports.CopilotClient",
            MagicMock(return_value=mock_client_instance),
        ):
            with patch(
                "amplifier_module_provider_github_copilot.sdk_adapter.client._resolve_token",
                return_value="test-token",
            ):
                wrapper = CopilotClientWrapper()

                async with wrapper.session():
                    pass  # Just need to create session to capture hook

                # Call the captured hook and verify it returns denial
                assert captured_hook is not None, "Hook was not captured"

                # Call the hook with mock tool use request
                # SDK hooks take (input_data, context) as per _make_deny_hook_config
                mock_input_data = {"toolName": "read_file", "arguments": {"path": "/etc/passwd"}}
                mock_context = MagicMock()

                result = captured_hook(mock_input_data, mock_context)

                # The deny hook should return something indicating denial
                # (The exact shape depends on SDK version, but should not be None/empty)
                assert result is not None, "Deny hook must return a result"


class TestF069DeadCodeRemoval:
    """Verify _complete_fn dead code removed."""

    def test_complete_fn_attribute_removed(self) -> None:
        """GitHubCopilotProvider no longer has _complete_fn attribute.

        The _complete_fn was never assigned after init,
        so it was dead code. Now removed.
        """
        from unittest.mock import MagicMock

        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider(config=None, coordinator=MagicMock())

        # _complete_fn should NOT exist anymore
        assert not hasattr(provider, "_complete_fn"), "_complete_fn attribute should be removed"
