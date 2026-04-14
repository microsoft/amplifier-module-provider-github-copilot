"""
Test quality improvements.

Contract: sdk-boundary.md (mount lifecycle), deny-destroy.md

Tests for:
- mount() exception → returns None path
- MagicMock with spec= for type safety
- Concurrent session deny hook verification

Note: These tests mock SubprocessConfig as non-None to avoid triggering
the P1-6 security fix (fail-closed when token cannot be applied).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from amplifier_core import ModuleCoordinator


# Mock SubprocessConfig that accepts github_token
class MockSubprocessConfig:
    """Mock SubprocessConfig that accepts github_token."""

    def __init__(self, github_token: str | None = None, log_level: str = "info") -> None:
        self.github_token = github_token
        self.log_level = log_level


class TestMountFailurePath:
    """Test mount() exception paths raise properly (P2 Fix).

    Contract: sdk-boundary.md - mount lifecycle
    P2 Fix: mount() raises on failure so framework can distinguish failure from opt-out.
    """

    @pytest.mark.asyncio
    async def test_mount_raises_on_provider_init_failure(self) -> None:
        """mount() raises when GitHubCopilotProvider.__init__ fails.

        Contract: sdk-boundary:BinaryResolution:MUST:8
        Contract: sdk-boundary.md - fail-fast on init failure
        P2 Fix: Raise instead of return None.
        """
        # Contract: provider-protocol:mount:MUST:2
        from amplifier_module_provider_github_copilot import mount

        coordinator = MagicMock(spec=ModuleCoordinator)
        coordinator.mount = AsyncMock()

        # Patch GitHubCopilotProvider to fail during __init__
        with patch(
            "amplifier_module_provider_github_copilot.GitHubCopilotProvider",
            side_effect=RuntimeError("Simulated provider init failure"),
        ):
            with pytest.raises(RuntimeError, match="Simulated provider init failure"):
                await mount(coordinator, config=None)

    @pytest.mark.asyncio
    async def test_mount_raises_on_coordinator_mount_exception(self) -> None:
        """mount() raises when coordinator.mount() raises.

        Contract: sdk-boundary.md - propagate failures
        P2 Fix: Raise instead of return None.
        """
        # Contract: provider-protocol:mount:MUST:2
        from amplifier_module_provider_github_copilot import mount

        coordinator = MagicMock(spec=ModuleCoordinator)
        coordinator.mount = AsyncMock(side_effect=RuntimeError("Coordinator error"))

        with pytest.raises(RuntimeError, match="Coordinator error"):
            await mount(coordinator, config=None)

    @pytest.mark.asyncio
    async def test_mount_logs_error_on_failure(self, caplog: pytest.LogCaptureFixture) -> None:
        """mount() logs error before raising when initialization fails.

        Contract: sdk-boundary.md - observability
        P2 Fix: Log then raise (not return None).
        """
        # Contract: behaviors:Logging:MUST:1
        from amplifier_module_provider_github_copilot import mount

        coordinator = MagicMock(spec=ModuleCoordinator)
        coordinator.mount = AsyncMock(side_effect=ValueError("Test error"))

        with caplog.at_level(logging.ERROR):
            with pytest.raises(ValueError, match="Test error"):
                await mount(coordinator, config=None)

        # Check that error was logged
        assert any("[MOUNT] Failed" in record.message for record in caplog.records)

    @pytest.mark.asyncio
    async def test_mount_returns_cleanup_on_success(self) -> None:
        """mount() returns cleanup function on success.

        Contract: sdk-boundary.md - cleanup MUST be returned
        """
        # Contract: provider-protocol:mount:MUST:2
        from amplifier_module_provider_github_copilot import mount

        coordinator = MagicMock(spec=ModuleCoordinator)
        coordinator.mount = AsyncMock()

        result = await mount(coordinator, config=None)

        # On success, should return a cleanup function (not None)
        assert callable(result)
        assert asyncio.iscoroutinefunction(result), (
            "mount() must return an async cleanup callable (coroutine function). "
            "Got sync callable — cleanup cannot await provider.cancel_emit_tasks() or "
            "_release_shared_client(). See provider-protocol:mount:MUST:2"
        )


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

        mock_session = MagicMock(spec=["disconnect", "session_id"])
        mock_session.disconnect = AsyncMock()
        mock_session.session_id = "test-session"

        # Cannot spec CopilotClient: SDK not installed in test environment
        mock_client_instance = AsyncMock()
        mock_client_instance.start = AsyncMock()

        async def mock_create_session(**config: Any) -> MagicMock:
            # Capture the config to verify deny hook
            session_configs.append(config)
            return mock_session

        mock_client_instance.create_session = mock_create_session

        with (
            patch(
                "amplifier_module_provider_github_copilot.sdk_adapter._imports.CopilotClient",
                MagicMock(return_value=mock_client_instance),
            ),
            patch(
                "amplifier_module_provider_github_copilot.sdk_adapter._imports.SubprocessConfig",
                MockSubprocessConfig,
            ),
            patch(
                "amplifier_module_provider_github_copilot.sdk_adapter.client._resolve_token",
                return_value="test-token",
            ),
        ):
            from amplifier_module_provider_github_copilot.sdk_adapter import SessionHandle

            wrapper = CopilotClientWrapper()

            # Launch 3 concurrent session requests
            async def get_session() -> SessionHandle:
                async with wrapper.session() as s:
                    return s

            await asyncio.gather(*[get_session() for _ in range(3)])

            # Contract: deny-destroy:DenyHook:MUST:1
            # All 3 sessions should have deny hook in config
            assert len(session_configs) == 3
            for config in session_configs:
                # Verify hooks dict exists with on_pre_tool_use
                assert "hooks" in config, "Session config missing hooks"
                assert "on_pre_tool_use" in config["hooks"], "Missing deny hook"
                # Verify the hook is actually callable
                assert callable(config["hooks"]["on_pre_tool_use"])

                # Contract v1.2: available_tools MUST be set (not omitted)
                # When no tools provided, available_tools=[] blocks SDK built-ins
                assert "available_tools" in config, "available_tools must be set"
                assert config["available_tools"] == [], "available_tools must be []"

    @pytest.mark.asyncio
    async def test_deny_hook_returns_deny_result(self) -> None:
        """Deny hook actually returns denial when called.

        Contract: deny-destroy:DenyHook:MUST:2 - hook MUST deny tool execution
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        captured_hook: Any = None

        mock_session = MagicMock(spec=["disconnect", "session_id"])
        mock_session.disconnect = AsyncMock()
        mock_session.session_id = "test-session"

        # Cannot spec CopilotClient: SDK not installed in test environment
        mock_client_instance = AsyncMock()
        mock_client_instance.start = AsyncMock()

        async def mock_create_session(**config: Any) -> MagicMock:
            nonlocal captured_hook
            if "hooks" in config and "on_pre_tool_use" in config["hooks"]:
                captured_hook = config["hooks"]["on_pre_tool_use"]
            return mock_session

        mock_client_instance.create_session = mock_create_session

        with (
            patch(
                "amplifier_module_provider_github_copilot.sdk_adapter._imports.CopilotClient",
                MagicMock(return_value=mock_client_instance),
            ),
            patch(
                "amplifier_module_provider_github_copilot.sdk_adapter._imports.SubprocessConfig",
                MockSubprocessConfig,
            ),
            patch(
                "amplifier_module_provider_github_copilot.sdk_adapter.client._resolve_token",
                return_value="test-token",
            ),
        ):
            wrapper = CopilotClientWrapper()

            async with wrapper.session():
                pass  # Just need to create session to capture hook

            # Call the hook with mock tool use request
            # SDK hooks take (input_data, context) as per _make_deny_hook_config
            mock_input_data = {"toolName": "read_file", "arguments": {"path": "/etc/passwd"}}
            mock_context = MagicMock()

            result = captured_hook(mock_input_data, mock_context)

            # Contract: deny-destroy:DenyHook:MUST:2
            assert isinstance(result, dict), (
                f"Deny hook must return a dict, got {type(result).__name__!r}. "
                "See client.py:DENY_ALL and deny-destroy:DenyHook:MUST:2"
            )
            assert result["permissionDecision"] == "deny", (
                f"permissionDecision is {result.get('permissionDecision')!r} but must be 'deny'. "
                "See deny-destroy:DenyHook:MUST:2"
            )
            assert result["permissionDecisionReason"] == "Processing", (
                f"permissionDecisionReason is {result.get('permissionDecisionReason')!r} "
                "but must be 'Processing'."
            )
            assert result["suppressOutput"] is True, (
                f"suppressOutput is {result.get('suppressOutput')!r} but must be True. "
                "Suppression prevents denial from polluting conversation context."
            )


class TestQualityGates:
    """Contract: provider-protocol:QualityGates:MUST:1,2

    Verify pyright configuration covers tests/ and the full repo.
    Actual pyright runs are enforced by CI; these tests verify the
    infrastructure (pyrightconfig.json) is correctly configured.
    """

    def test_pyright_config_includes_tests(self) -> None:
        """Contract: provider-protocol:QualityGates:MUST:1 — test files must be type-checked."""
        import json
        from pathlib import Path

        config_path = Path(__file__).parent.parent / "pyrightconfig.json"
        assert config_path.exists(), "pyrightconfig.json must exist at repo root"

        config = json.loads(config_path.read_text(encoding="utf-8"))
        include = config.get("include", [])
        assert "tests" in include, (
            f"pyrightconfig.json 'include' must contain 'tests' for type-clean test files. "
            f"Got: {include}"
        )

    def test_pyright_config_includes_full_repo(self) -> None:
        """Contract: provider-protocol:QualityGates:MUST:2 — full repo must be type-checked."""
        import json
        from pathlib import Path

        config_path = Path(__file__).parent.parent / "pyrightconfig.json"
        assert config_path.exists(), "pyrightconfig.json must exist at repo root"

        config = json.loads(config_path.read_text(encoding="utf-8"))
        include = config.get("include", [])
        assert "amplifier_module_provider_github_copilot" in include, (
            f"pyrightconfig.json 'include' must contain source package. Got: {include}"
        )
