"""
Tests for SDK Client Lifecycle.

This test suite covers:
1. Init-once semantics — lazy client creation, double-check locking
2. Auth token resolution — priority chain
3. Session lifecycle invariants — create/yield/disconnect sequence
4. Cancellation cleanup — owned vs injected client semantics
5. Lock contention — concurrent session() calls

Contract: contracts/sdk-boundary.md
Note: These are behavioral tests complementing the structural tests in test_sdk_client.py
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.fixtures.config_capture import ConfigCapturingMock

if TYPE_CHECKING:
    pass


class _MockSDKSession:
    """Minimal stub for the raw SDK session object used in lifecycle tests."""

    session_id: str = "test-session"

    async def disconnect(self) -> None: ...


class _MockSDKClient:
    """Minimal stub for copilot.CopilotClient used in lifecycle tests."""

    async def create_session(self, **kwargs: Any) -> _MockSDKSession: ...
    async def start(self) -> None: ...
    async def stop(self) -> None: ...


class _MockModuleCoordinator:
    """Minimal stub for amplifier_core.ModuleCoordinator used in prewarm tests."""

    async def mount(self, *args: Any, **kwargs: Any) -> None: ...


class TestLazyClientInit:
    """AC-1: Client not created until session() called.

    # Contract: provider-protocol:mount:MUST:5
    """

    def test_lazy_init_no_client_at_construction(self) -> None:
        """Verify no client is created when wrapper is instantiated.

        Contract: sdk-boundary:Config:MUST:1
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        wrapper = CopilotClientWrapper()

        # Neither injected nor owned client should exist at construction
        assert wrapper._sdk_client is None  # pyright: ignore[reportPrivateUsage]
        assert wrapper._owned_client is None  # pyright: ignore[reportPrivateUsage]

    @pytest.mark.asyncio
    async def test_lazy_init_client_created_on_first_session(self) -> None:
        """Verify client is created lazily on first session() call.

        Contract: sdk-boundary:Config:MUST:1
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        # Use injected client to avoid real SDK import
        mock_client = ConfigCapturingMock()
        wrapper = CopilotClientWrapper(sdk_client=mock_client)

        # Before session() - no captures
        assert len(mock_client.captured_configs) == 0

        # After session() - config captured
        async with wrapper.session(model="gpt-4"):
            pass

        assert len(mock_client.captured_configs) == 1


class TestConcurrentSessionInit:
    """AC-2: Concurrent session() calls result in single client init.

    # Contract: provider-protocol:mount:MUST:5
    """

    @pytest.mark.asyncio
    async def test_concurrent_session_single_client_init(self) -> None:
        """Verify lock prevents duplicate client init under concurrent access.

        Contract: deny-destroy:Ephemeral:MUST:1
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        # Track how many times CopilotClient() is called
        init_count = 0
        mock_session = AsyncMock(spec=_MockSDKSession)
        mock_session.session_id = "concurrent-test"
        mock_session.disconnect = AsyncMock()

        class MockCopilotClient:
            def __init__(
                self,
                *,
                base_directory: str,
                github_token: str | None = None,
                log_level: str = "info",
                env: dict[str, str],
                mode: str,
            ) -> None:
                nonlocal init_count
                init_count += 1

            async def start(self) -> None:
                pass

            async def create_session(self, **kwargs: Any) -> Any:
                return mock_session

            async def stop(self) -> None:
                pass

        wrapper = CopilotClientWrapper()

        # Patch CopilotClient using patch() with full path for reliable async behavior
        with patch(
            "amplifier_module_provider_github_copilot.sdk_adapter._imports.CopilotClient",
            MockCopilotClient,
        ):
            # Create multiple concurrent session() calls
            async def create_session() -> None:
                async with wrapper.session(model="gpt-4"):
                    await asyncio.sleep(0.01)  # Small delay to ensure overlap

            # Run 5 concurrent sessions
            await asyncio.gather(*[create_session() for _ in range(5)])

            # Client should only be initialized once despite 5 concurrent sessions
            assert init_count == 1


class TestTokenPriority:
    """AC-3: Token resolution follows 4-variable priority chain.

    Contract: sdk-boundary:Config:MUST:1
    """

    def test_token_priority_copilot_agent_first(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """COPILOT_AGENT_TOKEN has highest priority.

        Contract: sdk-boundary:Config:MUST:1
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            _resolve_token,  # pyright: ignore[reportPrivateUsage]
        )

        monkeypatch.setenv("COPILOT_AGENT_TOKEN", "agent-token")
        monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "copilot-token")
        monkeypatch.setenv("GH_TOKEN", "gh-token")
        monkeypatch.setenv("GITHUB_TOKEN", "github-token")

        assert _resolve_token() == "agent-token"

    def test_token_priority_copilot_github_second(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """COPILOT_GITHUB_TOKEN is second priority.

        Contract: sdk-boundary:Config:MUST:1
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            _resolve_token,  # pyright: ignore[reportPrivateUsage]
        )

        monkeypatch.delenv("COPILOT_AGENT_TOKEN", raising=False)
        monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "copilot-token")
        monkeypatch.setenv("GH_TOKEN", "gh-token")
        monkeypatch.setenv("GITHUB_TOKEN", "github-token")

        assert _resolve_token() == "copilot-token"

    def test_token_priority_gh_token_third(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """GH_TOKEN is third priority.

        Contract: sdk-boundary:Config:MUST:1
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            _resolve_token,  # pyright: ignore[reportPrivateUsage]
        )

        monkeypatch.delenv("COPILOT_AGENT_TOKEN", raising=False)
        monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
        monkeypatch.setenv("GH_TOKEN", "gh-token")
        monkeypatch.setenv("GITHUB_TOKEN", "github-token")

        assert _resolve_token() == "gh-token"

    def test_token_priority_github_token_fourth(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """GITHUB_TOKEN is fourth priority.

        Contract: sdk-boundary:Config:MUST:1
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            _resolve_token,  # pyright: ignore[reportPrivateUsage]
        )

        monkeypatch.delenv("COPILOT_AGENT_TOKEN", raising=False)
        monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.setenv("GITHUB_TOKEN", "github-token")

        assert _resolve_token() == "github-token"

    def test_token_priority_no_token_returns_none(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No token env vars returns None.

        Contract: sdk-boundary:Config:MUST:1
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            _resolve_token,  # pyright: ignore[reportPrivateUsage]
        )

        monkeypatch.delenv("COPILOT_AGENT_TOKEN", raising=False)
        monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)

        assert _resolve_token() is None

    @pytest.mark.asyncio
    async def test_fail_closed_token_without_client_raises_in_test_mode(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Test mode still fails closed when a token cannot reach CopilotClient.

        Contract: sdk-boundary:Auth:MUST:3
        """
        from amplifier_core.llm_errors import ConfigurationError

        from amplifier_module_provider_github_copilot.config._paths import ProviderPaths
        from amplifier_module_provider_github_copilot.sdk_adapter import client as client_mod

        paths = ProviderPaths(
            provider_home=tmp_path / "provider-home",
            cache_home=tmp_path / "cache-home",
        )
        monkeypatch.setenv("SKIP_SDK_CHECK", "1")
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")
        monkeypatch.setattr(client_mod, "CopilotClient", None)

        wrapper = client_mod.CopilotClientWrapper(provider_paths=paths)

        with pytest.raises(ConfigurationError, match="Explicit GitHub token"):
            await wrapper._ensure_client_initialized(caller="test")  # pyright: ignore[reportPrivateUsage]


class TestSessionConfigInvariants:
    """AC-4: Session config always contains required fields.

    Contract: deny-destroy:ToolSuppression:MUST:1
    Contract: sdk-boundary:Config:MUST:2,3,4
    Contract: deny-destroy:DenyHook:MUST:1
    """

    @pytest.mark.asyncio
    async def test_session_config_available_tools_set_empty_without_tools(self) -> None:
        """available_tools MUST be set to [] when no tools provided.

        Contract v1.2 correction: available_tools MUST NOT be omitted.
        When no Amplifier tools are provided, available_tools=[] prevents
        SDK built-in tools (list_agents, bash, edit) from appearing.

        Contract: deny-destroy:ToolSuppression:MUST:1
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        mock_client = ConfigCapturingMock()
        wrapper = CopilotClientWrapper(sdk_client=mock_client)

        async with wrapper.session(model="gpt-4"):
            pass

        # available_tools MUST be set (not omitted) - contract v1.2 fix
        # When no tools provided, empty list blocks SDK built-ins
        assert "available_tools" in mock_client.last_config
        assert mock_client.last_config["available_tools"] == []

    @pytest.mark.asyncio
    async def test_session_config_does_not_forward_temperature(self) -> None:
        """The provider MUST NOT forward a sampling temperature to the SDK.

        github-copilot-sdk v0.3.0 does not expose a temperature kwarg on
        either Client.create_session or CopilotSession.send. The provider
        carries no temperature plumbing of its own. This test pins that
        invariant: a future edit that silently re-introduces a temperature
        key into session_config (e.g., by passing through ChatRequest.
        temperature, or by accepting a v0.4 SDK kwarg without an explicit
        decision) will turn this assertion red.
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        mock_client = ConfigCapturingMock()
        wrapper = CopilotClientWrapper(sdk_client=mock_client)

        async with wrapper.session(model="gpt-4"):
            pass

        assert "temperature" not in mock_client.last_config

    @pytest.mark.asyncio
    async def test_session_config_streaming_always_true(self) -> None:
        """streaming=True on every session.

        Contract: sdk-boundary:Config:MUST:4
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        mock_client = ConfigCapturingMock()
        wrapper = CopilotClientWrapper(sdk_client=mock_client)

        async with wrapper.session(model="gpt-4"):
            pass

        assert mock_client.last_config["streaming"] is True

    @pytest.mark.asyncio
    async def test_session_config_deny_hook_present(self) -> None:
        """hooks.on_pre_tool_use in config.

        Contract: deny-destroy:DenyHook:MUST:1
        Contract: sdk-boundary:Config:MUST:5
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        mock_client = ConfigCapturingMock()
        wrapper = CopilotClientWrapper(sdk_client=mock_client)

        async with wrapper.session(model="gpt-4"):
            pass

        hooks = mock_client.last_config.get("hooks", {})
        assert "on_pre_tool_use" in hooks
        assert callable(hooks["on_pre_tool_use"])

    @pytest.mark.asyncio
    async def test_session_config_permission_handler(self) -> None:
        """on_permission_request always set.

        Contract: sdk-boundary:Config:MUST:3
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        mock_client = ConfigCapturingMock()
        wrapper = CopilotClientWrapper(sdk_client=mock_client)

        async with wrapper.session(model="gpt-4"):
            pass

        assert "on_permission_request" in mock_client.last_config
        assert callable(mock_client.last_config["on_permission_request"])

    @pytest.mark.asyncio
    async def test_session_config_system_message_replace(self) -> None:
        """system_message mode is 'replace'.

        Contract: sdk-boundary:Config:MUST:2
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        mock_client = ConfigCapturingMock()
        wrapper = CopilotClientWrapper(sdk_client=mock_client)

        async with wrapper.session(model="gpt-4", system_message="Test persona"):
            pass

        system_msg = mock_client.last_config["system_message"]
        assert system_msg["mode"] == "replace"
        assert system_msg["content"] == "Test persona"


class TestDisconnectBehavior:
    """AC-5, AC-6: Disconnect is called in finally block and failures are logged.

    Contract: deny-destroy:Ephemeral:MUST:2
    """

    @pytest.mark.asyncio
    async def test_disconnect_on_normal_exit(self) -> None:
        """disconnect called after yield.

        Contract: deny-destroy:Ephemeral:MUST:2
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        mock_session = AsyncMock(spec=_MockSDKSession)
        mock_session.session_id = "normal-exit"
        mock_session.disconnect = AsyncMock()

        mock_client = AsyncMock(spec=_MockSDKClient)
        mock_client.create_session = AsyncMock(return_value=mock_session)

        wrapper = CopilotClientWrapper(sdk_client=mock_client)

        async with wrapper.session(model="gpt-4"):
            pass

        mock_session.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_on_exception(self) -> None:
        """disconnect called when body raises.

        Contract: deny-destroy:Ephemeral:MUST:2
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        mock_session = AsyncMock(spec=_MockSDKSession)
        mock_session.session_id = "exception-exit"
        mock_session.disconnect = AsyncMock()

        mock_client = AsyncMock(spec=_MockSDKClient)
        mock_client.create_session = AsyncMock(return_value=mock_session)

        wrapper = CopilotClientWrapper(sdk_client=mock_client)

        with pytest.raises(RuntimeError, match="user error"):
            async with wrapper.session(model="gpt-4"):
                raise RuntimeError("user error")

        mock_session.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_failure_logged_not_raised(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Warning logged, caller doesn't see error.

        Contract: deny-destroy:Ephemeral:MUST:2
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        mock_session = AsyncMock(spec=_MockSDKSession)
        mock_session.session_id = "disconnect-fail"
        mock_session.disconnect = AsyncMock(side_effect=RuntimeError("disconnect failed"))

        mock_client = AsyncMock(spec=_MockSDKClient)
        mock_client.create_session = AsyncMock(return_value=mock_session)

        wrapper = CopilotClientWrapper(sdk_client=mock_client)

        # Should NOT raise despite disconnect failure
        async with wrapper.session(model="gpt-4"):
            pass

        # Warning should be logged
        assert "disconnect" in caplog.text.lower()

    @pytest.mark.asyncio
    async def test_disconnect_failure_escalation(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """>3 failures logged at error level.

        Contract: behaviors:Logging:MUST:1
        """
        import logging

        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        mock_session = AsyncMock(spec=_MockSDKSession)
        mock_session.session_id = "disconnect-escalate"
        mock_session.disconnect = AsyncMock(side_effect=RuntimeError("disconnect failed"))

        mock_client = AsyncMock(spec=_MockSDKClient)
        mock_client.create_session = AsyncMock(return_value=mock_session)

        wrapper = CopilotClientWrapper(sdk_client=mock_client)

        # Trigger 4 disconnect failures to exceed threshold (>3)
        with caplog.at_level(logging.ERROR):
            for _ in range(4):
                async with wrapper.session(model="gpt-4"):
                    pass

        # After 4 failures (>3), should see ERROR level log
        log_text = caplog.text.lower()
        assert "multiple disconnect failures" in log_text


class TestFailedStartCleanup:
    """AC-7: Failed start() clears _owned_client.

    # Lifecycle invariant — no formal contract anchor; see client.py:358-360
    """

    @pytest.mark.asyncio
    async def test_failed_start_clears_owned_client(self) -> None:
        """_owned_client = None after start() fails.

        Contract: sdk-boundary:Config:MUST:1
        """
        from amplifier_core.llm_errors import ProviderUnavailableError

        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        class FailingCopilotClient:
            def __init__(
                self,
                *,
                base_directory: str,
                github_token: str | None = None,
                log_level: str = "info",
                env: dict[str, str],
                mode: str,
            ) -> None:
                pass

            async def start(self) -> None:
                raise RuntimeError("SDK start failed")

        wrapper = CopilotClientWrapper()

        # Patch CopilotClient using patch() with full path for reliable async behavior
        with patch(
            "amplifier_module_provider_github_copilot.sdk_adapter._imports.CopilotClient",
            FailingCopilotClient,
        ):
            # session() raises because start() fails; translate_sdk_error() maps
            # unrecognised RuntimeError to ProviderUnavailableError (default fallback).
            with pytest.raises(ProviderUnavailableError, match="SDK start failed"):
                async with wrapper.session(model="gpt-4"):
                    pass  # Should not reach here

            # _owned_client should be cleared after failed start
            assert wrapper._owned_client is None  # pyright: ignore[reportPrivateUsage]


class TestCloseOwnershipSemantics:
    """AC-8, AC-9: Close ownership semantics and idempotency.

    # Contract: provider-protocol:mount:MUST:5
    """

    @pytest.mark.asyncio
    async def test_close_stops_owned_client(self) -> None:
        """owned client's stop() called.

        Contract: provider-protocol:complete:MUST:3
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        wrapper = CopilotClientWrapper()
        mock_owned = AsyncMock(spec=_MockSDKClient)
        mock_owned.stop = AsyncMock()
        wrapper._owned_client = mock_owned  # pyright: ignore[reportPrivateUsage]

        await wrapper.close()

        mock_owned.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_does_not_stop_injected_client(self) -> None:
        """injected client's stop() NOT called.

        Contract: provider-protocol:complete:MUST:3
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        mock_injected = AsyncMock(spec=_MockSDKClient)
        mock_injected.stop = AsyncMock()

        wrapper = CopilotClientWrapper(sdk_client=mock_injected)
        await wrapper.close()

        mock_injected.stop.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_idempotent(self) -> None:
        """Multiple close() calls are safe.

        Contract: provider-protocol:complete:MUST:3
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        wrapper = CopilotClientWrapper()
        mock_owned = AsyncMock(spec=_MockSDKClient)
        mock_owned.stop = AsyncMock()
        wrapper._owned_client = mock_owned  # pyright: ignore[reportPrivateUsage]

        # First close should stop
        await wrapper.close()
        assert mock_owned.stop.call_count == 1

        # Second close should not raise or double-stop
        await wrapper.close()
        assert mock_owned.stop.call_count == 1  # Still just 1 call

    @pytest.mark.asyncio
    async def test_close_before_any_session(self) -> None:
        """close() called before any session() is safe.

        Contract: provider-protocol:complete:MUST:3
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        wrapper = CopilotClientWrapper()

        # Should not raise even though no session was ever created
        await wrapper.close()


class TestSDKNotInstalled:
    """AC-10 analog: ImportError translates to ProviderUnavailableError.

    Contract: sdk-boundary:Membrane:MUST:5
    """

    @pytest.mark.asyncio
    async def test_sdk_not_installed_raises_provider_unavailable(self) -> None:
        """ImportError → ProviderUnavailableError.

        Contract: sdk-boundary:Membrane:MUST:5
        """
        from amplifier_module_provider_github_copilot.error_translation import (
            ProviderUnavailableError,
        )
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        wrapper = CopilotClientWrapper()

        # Patch _imports to raise ImportError
        with patch(
            "amplifier_module_provider_github_copilot.sdk_adapter._imports.CopilotClient",
            side_effect=ImportError("No module named 'copilot'"),
        ):
            with pytest.raises(ProviderUnavailableError) as exc_info:
                async with wrapper.session(model="gpt-4"):
                    pass  # Should not reach here

            assert "Copilot SDK not installed" in str(exc_info.value)


class TestHealthCheck:
    """is_healthy() method for singleton pattern.

    # Contract: sdk-protection:Subprocess:MUST:6
    """

    @pytest.mark.asyncio
    async def test_is_healthy_false_after_close(self) -> None:
        """Closed wrapper is unhealthy.

        Contract: sdk-boundary:Config:MUST:1
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        wrapper = CopilotClientWrapper()
        await wrapper.close()
        assert wrapper.is_healthy() is False


class TestDenyHookBehavior:
    """Verify deny hook returns correct denial response.

    Contract: deny-destroy:DenyHook:MUST:2
    """

    def test_deny_hook_returns_denial(self) -> None:
        """Deny hook returns denial dict.

        Contract: deny-destroy:DenyHook:MUST:2
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            _make_deny_hook_config,  # pyright: ignore[reportPrivateUsage]
        )

        hooks_config = _make_deny_hook_config()
        deny_hook = hooks_config["on_pre_tool_use"]

        result = deny_hook({"toolName": "bash"}, None)

        assert result["permissionDecision"] == "deny"

    def test_deny_hook_has_suppress_output(self) -> None:
        """Deny hook includes suppressOutput.

        Contract: deny-destroy:DenyHook:MUST:2
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            _make_deny_hook_config,  # pyright: ignore[reportPrivateUsage]
        )

        hooks_config = _make_deny_hook_config()
        deny_hook = hooks_config["on_pre_tool_use"]

        result = deny_hook({"toolName": "bash"}, None)

        assert result.get("suppressOutput") is True


class TestTokenFallbackSecurity:
    """Test security behavior when token exists but CopilotClient constructor unavailable.

    P1-6 Security Fix: An explicit token MUST NEVER be silently ignored.

    Contract: sdk-boundary:Auth:MUST:3

    When explicit token is provided but SDK can't apply it:
    - ALWAYS fail closed with ConfigurationError
    - No escape hatches - security behavior is unconditional
    - If tests need to avoid this: clear token env vars, don't mask constructor failures

    The SKIP_SDK_CHECK env var controls SDK _imports_ only, NOT auth behavior.
    """

    @pytest.mark.asyncio
    async def test_raises_error_when_token_dropped_in_production(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Raises ConfigurationError when CopilotClient is None and token is set.

        Contract: sdk-boundary:Auth:MUST:3
        """
        from amplifier_core.llm_errors import ConfigurationError

        import amplifier_module_provider_github_copilot.sdk_adapter.client as client_mod
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        # Substitute SDK client with None — test-mode invariant path.
        monkeypatch.setattr(client_mod, "CopilotClient", None)
        monkeypatch.setenv("GITHUB_TOKEN", "test-token-value")

        wrapper = CopilotClientWrapper()
        with pytest.raises(ConfigurationError) as exc_info:
            async with wrapper.session(model="gpt-4"):
                pass

        err = str(exc_info.value)
        assert "CopilotClient" in err
        assert "Cannot apply token" in err


class TestPrewarmSubprocess:
    """Pre-warming: SDK subprocess initialization at mount() time.

    When `sdk.prewarm_subprocess: true` in config/_sdk_protection.py, the SDK
    subprocess should be spawned during mount() rather than during the
    first complete() call. This moves ~2s latency from user-visible
    first-request time to invisible mount time.

    # Contract: sdk-protection:Subprocess:MUST:5
    """

    @pytest.mark.asyncio
    async def test_prewarm_enabled_triggers_early_init(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When prewarm_subprocess=True, mount() triggers client init.

        The client should be initialized before first session() call.
        This is a fire-and-forget task — mount() doesn't wait for it.

        Contract: sdk-boundary:Config:MUST:1
        """
        import amplifier_module_provider_github_copilot as provider_module

        init_timestamps: list[float] = []
        import time

        mock_session = AsyncMock(spec=_MockSDKSession)
        mock_session.session_id = "prewarm-test"
        mock_session.disconnect = AsyncMock()

        class MockCopilotClient:
            def __init__(
                self,
                *,
                base_directory: str,
                github_token: str | None = None,
                log_level: str = "info",
                env: dict[str, str],
                mode: str,
            ) -> None:
                init_timestamps.append(time.perf_counter())

            async def start(self) -> None:
                # Simulate subprocess spawn delay
                await asyncio.sleep(0.1)

            async def create_session(self, **kwargs: Any) -> Any:
                return mock_session

            async def stop(self) -> None:
                pass

        # Reset singleton state
        provider_module._shared_client = None  # pyright: ignore[reportPrivateUsage]
        provider_module._shared_client_refcount = 0  # pyright: ignore[reportPrivateUsage]

        # Mock coordinator
        mock_coordinator = MagicMock(spec=_MockModuleCoordinator)
        mock_coordinator.mount = AsyncMock()

        # Patch config to enable pre-warming via monkeypatch.
        # Targets __init__.py's module-level binding (not config_loader module).
        import amplifier_module_provider_github_copilot as provider_module

        try:
            from amplifier_module_provider_github_copilot.config._sdk_protection import (
                SdkConfig,
                SdkProtectionConfig,
                SingletonConfig,
            )

            def mock_load_config() -> SdkProtectionConfig:
                return SdkProtectionConfig(
                    sdk=SdkConfig(
                        prewarm_subprocess=True,
                        log_level="none",
                        log_level_env_var="COPILOT_SDK_LOG",
                    ),
                    singleton=SingletonConfig(lock_timeout_seconds=30.0),
                )

            monkeypatch.setattr(provider_module, "load_sdk_protection_config", mock_load_config)

            with (
                patch(
                    "amplifier_module_provider_github_copilot.sdk_adapter._imports.CopilotClient",
                    MockCopilotClient,
                ),
            ):
                # Call mount — should trigger pre-warming
                cleanup = await provider_module.mount(mock_coordinator, {})

                # Give fire-and-forget task time to run
                await asyncio.sleep(0.2)

                # Client should have been initialized DURING mount (not after)
                assert len(init_timestamps) == 1, (
                    "Pre-warming should have triggered client initialization during mount()"
                )

                # Cleanup
                if cleanup:
                    await cleanup()
        finally:
            # Reset singleton after test
            provider_module._shared_client = None  # pyright: ignore[reportPrivateUsage]
            provider_module._shared_client_refcount = 0  # pyright: ignore[reportPrivateUsage]

    @pytest.mark.asyncio
    async def test_prewarm_race_uses_existing_client(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Concurrent session() during pre-warm uses the same client.

        If pre-warming is in progress and session() is called, the
        session() should wait for the pre-warm to complete (via lock)
        and use the same client — NOT create a second one.

        Contract: deny-destroy:Ephemeral:MUST:1
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        init_count = 0

        mock_session = AsyncMock(spec=_MockSDKSession)
        mock_session.session_id = "race-test"
        mock_session.disconnect = AsyncMock()

        class MockCopilotClient:
            def __init__(
                self,
                *,
                base_directory: str,
                github_token: str | None = None,
                log_level: str = "info",
                env: dict[str, str],
                mode: str,
            ) -> None:
                nonlocal init_count
                init_count += 1

            async def start(self) -> None:
                # Simulate slow subprocess spawn
                await asyncio.sleep(0.3)

            async def create_session(self, **kwargs: Any) -> Any:
                return mock_session

            async def stop(self) -> None:
                pass

        with (
            patch(
                "amplifier_module_provider_github_copilot.sdk_adapter._imports.CopilotClient",
                MockCopilotClient,
            ),
        ):
            wrapper = CopilotClientWrapper()

            # Start pre-warm (simulated)
            # pyright: ignore[reportPrivateUsage]
            prewarm_task = asyncio.create_task(
                wrapper._ensure_client_initialized(caller="prewarm")  # pyright: ignore[reportPrivateUsage]
            )

            # Immediately start session() — should block on lock
            await asyncio.sleep(0.05)  # Let prewarm acquire lock first
            async with wrapper.session(model="gpt-4"):
                pass

            await prewarm_task

            # Only ONE client should have been created
            assert init_count == 1, (
                f"Expected 1 client init (lock should prevent race), got {init_count}"
            )

    @pytest.mark.asyncio
    async def test_mount_failure_cancels_prewarm_task(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If coordinator.mount() raises, the in-flight prewarm task is
        cancelled and the ``_prewarm_task`` module global is cleared.

        Pre-fix behavior: the global was left referring to a scheduled task
        that would run against a closing shared client, and subsequent
        mount() calls would overwrite the leaked task without cancellation.

        Contract: sdk-protection:Subprocess:MUST:5
        """
        import amplifier_module_provider_github_copilot as provider_module

        mock_session = AsyncMock(spec=_MockSDKSession)
        mock_session.session_id = "prewarm-fail-test"
        mock_session.disconnect = AsyncMock()

        class MockCopilotClient:
            def __init__(self, **kwargs: Any) -> None:
                pass

            async def start(self) -> None:
                await asyncio.sleep(5.0)

            async def prewarm(self) -> None:
                pass

            async def create_session(self, **kwargs: Any) -> Any:
                return mock_session

            async def stop(self) -> None:
                pass

        provider_module._shared_client = None  # pyright: ignore[reportPrivateUsage]
        provider_module._shared_client_refcount = 0  # pyright: ignore[reportPrivateUsage]
        provider_module._prewarm_task = None  # pyright: ignore[reportPrivateUsage]

        mock_coordinator = MagicMock(spec=_MockModuleCoordinator)
        mock_coordinator.mount = AsyncMock(side_effect=RuntimeError("mount kaboom"))

        from amplifier_module_provider_github_copilot.config._sdk_protection import (
            SdkConfig,
            SdkProtectionConfig,
            SingletonConfig,
        )

        def mock_load_config() -> SdkProtectionConfig:
            return SdkProtectionConfig(
                sdk=SdkConfig(
                    prewarm_subprocess=True,
                    log_level="none",
                    log_level_env_var="COPILOT_SDK_LOG",
                ),
                singleton=SingletonConfig(lock_timeout_seconds=30.0),
            )

        monkeypatch.setattr(provider_module, "load_sdk_protection_config", mock_load_config)

        try:
            with patch(
                "amplifier_module_provider_github_copilot.sdk_adapter._imports.CopilotClient",
                MockCopilotClient,
            ):
                with pytest.raises(RuntimeError, match="mount kaboom"):
                    await provider_module.mount(mock_coordinator, {})

                # Behavioral assertion: the cleanup path on mount failure must
                # clear the prewarm-task global — a stale Task here would leak
                # across mount attempts (contract MUST:5).
                assert provider_module._prewarm_task is None, (  # pyright: ignore[reportPrivateUsage]
                    "Stale prewarm task left in module global after mount failure"
                )
        finally:
            provider_module._shared_client = None  # pyright: ignore[reportPrivateUsage]
            provider_module._shared_client_refcount = 0  # pyright: ignore[reportPrivateUsage]
            provider_module._prewarm_task = None  # pyright: ignore[reportPrivateUsage]

    @pytest.mark.asyncio
    async def test_cancel_prewarm_task_does_not_raise_on_inner_exception(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A non-CancelledError surfacing from the awaited prewarm task MUST
        be suppressed by _cancel_prewarm_task and MUST be logged by _prewarm().

        Pins the contract between the two layers:
          * _prewarm() at __init__.py:374-386 catches Exception and emits a
            redacted ``logger.warning`` — the primary observability signal.
          * _cancel_prewarm_task at __init__.py:228-248 suppresses any leaked
            Exception so the cancel path stays clean during mount failures.

        If a future refactor removes _prewarm()'s inner ``except``, this test
        still proves _cancel_prewarm_task doesn't raise — preventing a silent
        regression where prewarm errors crash shutdown. The complementary
        assertion (warning fired) catches the inverse: if someone deletes the
        cancel-site suppression AND the inner handler, both signals vanish.

        Contract: sdk-protection:Subprocess:MUST:5
        """
        import logging

        import amplifier_module_provider_github_copilot as provider_module

        async def failing_prewarm() -> None:
            raise RuntimeError("prewarm exploded")

        # Drive the prewarm coroutine to completion so the task captures the
        # RuntimeError before we await it inside _cancel_prewarm_task. (If we
        # cancel before the task runs, CancelledError wins and the inner
        # except never fires.)
        provider_module._prewarm_task = asyncio.create_task(failing_prewarm())  # pyright: ignore[reportPrivateUsage]
        await asyncio.sleep(0)
        # Let the task observe its own exception
        with contextlib.suppress(RuntimeError):
            await provider_module._prewarm_task  # pyright: ignore[reportPrivateUsage]
        # Reset the global to the now-completed task and exercise cancel path
        # against an already-done task that raised. The function must not
        # propagate the inner RuntimeError.
        try:
            with caplog.at_level(logging.WARNING):
                await provider_module._cancel_prewarm_task()  # pyright: ignore[reportPrivateUsage]
        finally:
            provider_module._prewarm_task = None  # pyright: ignore[reportPrivateUsage]

        # Module-global must be cleared regardless of the inner exception.
        assert provider_module._prewarm_task is None, (  # pyright: ignore[reportPrivateUsage]
            "_cancel_prewarm_task must clear the module global even when the "
            "awaited task raised a non-CancelledError."
        )

    @pytest.mark.asyncio
    async def test_cancel_prewarm_task_suppresses_inner_exception_branch(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The bare ``except Exception`` at __init__.py:244 MUST execute when a
        non-cancelled prewarm task surfaces a non-CancelledError after cancel.

        The companion test above (test_cancel_prewarm_task_does_not_raise_on_
        inner_exception) pre-resolves the task, so _cancel_prewarm_task short-
        circuits at the ``task.done()`` guard and never reaches the suppress
        site. This test exercises the live branch: it installs a task that is
        still pending at cancel time and that catches CancelledError to re-raise
        RuntimeError, then asserts:

          * _cancel_prewarm_task() returns without propagating the RuntimeError
          * the DEBUG suppression log record is emitted (proves the branch ran)
          * _prewarm_task module global is cleared

        Contract: sdk-protection:Subprocess:MUST:5
        """
        import logging

        import amplifier_module_provider_github_copilot as provider_module

        started = asyncio.Event()

        async def cancel_to_runtime_error() -> None:
            try:
                started.set()
                await asyncio.sleep(3600)
            except asyncio.CancelledError as exc:
                raise RuntimeError("cancel-converted-to-runtime-error") from exc

        provider_module._prewarm_task = asyncio.create_task(cancel_to_runtime_error())  # pyright: ignore[reportPrivateUsage]
        await started.wait()
        assert not provider_module._prewarm_task.done(), (  # pyright: ignore[reportPrivateUsage]
            "Setup invariant: task must still be pending so _cancel_prewarm_task "
            "reaches the cancel/await branch (not the task.done() short-circuit)."
        )

        try:
            with caplog.at_level(
                logging.DEBUG, logger="amplifier_module_provider_github_copilot"
            ):
                await provider_module._cancel_prewarm_task()  # pyright: ignore[reportPrivateUsage]
        finally:
            provider_module._prewarm_task = None  # pyright: ignore[reportPrivateUsage]

        assert provider_module._prewarm_task is None, (  # pyright: ignore[reportPrivateUsage]
            "_cancel_prewarm_task must clear the module global after suppressing "
            "the inner RuntimeError."
        )

        debug_records = [
            r
            for r in caplog.records
            if r.levelno == logging.DEBUG
            and r.name == "amplifier_module_provider_github_copilot"
            and "cancel-site suppressed" in r.getMessage()
        ]
        assert len(debug_records) == 1, (
            "The bare ``except Exception`` branch at __init__.py:244 MUST emit "
            "exactly one DEBUG record so a future regression that removes "
            "_prewarm()'s inner handler still leaves a traceable signal. "
            f"Captured records: {[(r.name, r.levelname, r.getMessage()) for r in caplog.records]}"
        )


class TestTokenFallbackSecurityExtended:
    """Extended security tests for token fallback behavior.

    P1-6 Security Fix: An explicit token MUST NEVER be silently ignored.

    Contract: sdk-boundary:Auth:MUST:3

    When explicit token is provided but SDK can't apply it:
    - ALWAYS fail closed with ConfigurationError
    - No escape hatches - security behavior is unconditional
    - If tests need to avoid this: clear token env vars, don't mask constructor failures

    The SKIP_SDK_CHECK env var controls SDK _imports_ only, NOT auth behavior.
    """

    @pytest.mark.asyncio
    async def test_token_always_fails_closed_no_escape_hatch(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Explicit token ALWAYS raises ConfigurationError when CopilotClient is None.

        SKIP_SDK_CHECK does not provide an escape hatch — token failure is
        unconditional when no SDK client is available.

        Contract: sdk-boundary:Auth:MUST:3
        """
        from amplifier_core.llm_errors import ConfigurationError

        import amplifier_module_provider_github_copilot.sdk_adapter.client as client_mod
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        # Substitute SDK client with None — test-mode invariant; SKIP_SDK_CHECK has no effect.
        monkeypatch.setattr(client_mod, "CopilotClient", None)
        monkeypatch.setenv("GITHUB_TOKEN", "test-token-value")

        wrapper = CopilotClientWrapper()
        with pytest.raises(ConfigurationError) as exc_info:
            async with wrapper.session(model="gpt-4"):
                pass

        err = str(exc_info.value)
        assert "CopilotClient" in err
        assert "Cannot apply token" in err

    @pytest.mark.asyncio
    async def test_no_token_with_none_client_raises_runtime_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No token + None CopilotClient => RuntimeError naming the SDK floor.

        Covers the second branch of the ``sdk_client_cls is None`` guard at
        ``sdk_adapter/client.py:374-377``. The token branch fails with
        ``ConfigurationError`` (test above); the no-token branch must surface
        ``RuntimeError`` so production code that loses the SDK symbol does
        not silently proceed into ``None(**kwargs)``.

        Contract: sdk-boundary:Membrane:MUST:5 (production needs >=1.0.0).
        """
        from amplifier_core.llm_errors import ProviderUnavailableError

        import amplifier_module_provider_github_copilot.sdk_adapter.client as client_mod
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        monkeypatch.setattr(client_mod, "CopilotClient", None)
        for var in ("COPILOT_AGENT_TOKEN", "COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
            monkeypatch.delenv(var, raising=False)

        wrapper = CopilotClientWrapper()
        with pytest.raises(ProviderUnavailableError) as exc_info:
            async with wrapper.session(model="gpt-4"):
                pass

        err = str(exc_info.value)
        assert "CopilotClient unavailable" in err
        # Pin the full GA floor phrase (with trailing period) so a regression to a
        # pre-GA beta message like ">= 1.0.0b10." cannot satisfy this assertion.
        assert "requires github-copilot-sdk >= 1.0.0." in err
        # The bare RuntimeError chain must survive translation as the cause.
        assert isinstance(exc_info.value.__cause__, RuntimeError)

    @pytest.mark.asyncio
    async def test_no_error_when_no_token(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No error when no token exists (normal fallback path).

        Contract: sdk-boundary:Config:MUST:1
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        # Clear SKIP_SDK_CHECK to simulate production mode
        monkeypatch.delenv("SKIP_SDK_CHECK", raising=False)

        # Clear all token env vars
        for var in ("COPILOT_AGENT_TOKEN", "COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
            monkeypatch.delenv(var, raising=False)

        mock_session = AsyncMock(spec=_MockSDKSession)
        mock_session.session_id = "no-token-test"
        mock_session.disconnect = AsyncMock()

        class MockCopilotClient:
            def __init__(
                self,
                *,
                base_directory: str,
                github_token: str | None = None,
                log_level: str = "info",
                env: dict[str, str],
                mode: str,
            ) -> None:
                pass

            async def start(self) -> None:
                pass

            async def create_session(self, **kwargs: Any) -> Any:
                return mock_session

            async def stop(self) -> None:
                pass

        with (
            patch(
                "amplifier_module_provider_github_copilot.sdk_adapter._imports.CopilotClient",
                MockCopilotClient,
            ),
        ):
            wrapper = CopilotClientWrapper()
            # Should NOT raise - no token means no security concern
            async with wrapper.session(model="gpt-4"):
                pass


class TestCopilotPidTracking:
    """SDK process ID tracking for log correlation.

    The copilot_pid property enables correlation between provider events.jsonl
    and SDK logs at ~/.copilot/logs/process-{timestamp}-{pid}.log.

    # Contract: behaviors:Logging:MUST:5
    """

    def test_copilot_pid_none_before_init(self) -> None:
        """copilot_pid is None before client initialization."""
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        wrapper = CopilotClientWrapper()
        assert wrapper.copilot_pid is None

    def test_copilot_pid_none_with_injected_client(self) -> None:
        """copilot_pid is None when using injected client (no subprocess)."""
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        mock_client = MagicMock(spec=_MockSDKClient)
        wrapper = CopilotClientWrapper(sdk_client=mock_client)
        # Injected client = no subprocess, no PID
        assert wrapper.copilot_pid is None

    @pytest.mark.asyncio
    async def test_copilot_pid_captured_after_start(self) -> None:
        """copilot_pid is captured after client.start().

        The PID is extracted from client._process.pid after successful start.
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        mock_session = AsyncMock(spec=_MockSDKSession)
        mock_session.session_id = "test-pid"
        mock_session.disconnect = AsyncMock()

        class MockProcess:
            pid = 12345

        class MockCopilotClient:
            def __init__(
                self,
                *,
                base_directory: str,
                github_token: str | None = None,
                log_level: str = "info",
                env: dict[str, str],
                mode: str,
            ) -> None:
                self._process = MockProcess()

            async def start(self) -> None:
                pass

            async def create_session(self, **kwargs: Any) -> Any:
                return mock_session

            async def stop(self) -> None:
                pass

        wrapper = CopilotClientWrapper()

        with patch(
            "amplifier_module_provider_github_copilot.sdk_adapter._imports.CopilotClient",
            MockCopilotClient,
        ):
            async with wrapper.session(model="gpt-4"):
                # PID should be captured after initialization
                assert wrapper.copilot_pid == "12345"


class TestGuardReinitAfterStop:
    """MUST-6: Guard re-initialization after stop.

    Contract: sdk-protection:Subprocess:MUST:6
    """

    @pytest.mark.asyncio
    async def test_session_raises_after_close(self) -> None:
        """Cannot create session after close().

        Contract: sdk-protection:Subprocess:MUST:6
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        wrapper = CopilotClientWrapper()
        await wrapper.close()

        with pytest.raises(RuntimeError, match="stopped"):
            async with wrapper.session(model="gpt-4"):
                pass

    @pytest.mark.asyncio
    async def test_prewarm_raises_after_close(self) -> None:
        """Cannot prewarm after close().

        Contract: sdk-protection:Subprocess:MUST:6
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        wrapper = CopilotClientWrapper()
        await wrapper.close()

        with pytest.raises(RuntimeError, match="stopped"):
            await wrapper.prewarm()

    @pytest.mark.asyncio
    async def test_list_models_raises_after_close(self) -> None:
        """Cannot list models after close().

        Contract: sdk-protection:Subprocess:MUST:6
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        wrapper = CopilotClientWrapper()
        await wrapper.close()

        with pytest.raises(RuntimeError, match="stopped"):
            await wrapper.list_models()


class TestPublicPrewarmAPI:
    """Public prewarm() API for SDK subprocess initialization.

    Contract: sdk-protection:Subprocess:MUST:5
    """

    @pytest.mark.asyncio
    async def test_prewarm_initializes_client(self) -> None:
        """prewarm() triggers client initialization.

        Contract: sdk-protection:Subprocess:MUST:5
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        init_called = False

        class MockCopilotClient:
            def __init__(
                self,
                *,
                base_directory: str,
                github_token: str | None = None,
                log_level: str = "info",
                env: dict[str, str],
                mode: str,
            ) -> None:
                nonlocal init_called
                init_called = True

            async def start(self) -> None:
                pass

            async def stop(self) -> None:
                pass

        wrapper = CopilotClientWrapper()

        with (
            patch(
                "amplifier_module_provider_github_copilot.sdk_adapter._imports.CopilotClient",
                MockCopilotClient,
            ),
        ):
            await wrapper.prewarm()
            assert init_called, "prewarm() should have initialized client"

        await wrapper.close()

    @pytest.mark.asyncio
    async def test_prewarm_idempotent(self) -> None:
        """Multiple prewarm() calls only initialize once.

        Contract: sdk-protection:Subprocess:MUST:5
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        init_count = 0

        class MockCopilotClient:
            def __init__(
                self,
                *,
                base_directory: str,
                github_token: str | None = None,
                log_level: str = "info",
                env: dict[str, str],
                mode: str,
            ) -> None:
                nonlocal init_count
                init_count += 1

            async def start(self) -> None:
                pass

            async def stop(self) -> None:
                pass

        wrapper = CopilotClientWrapper()

        with (
            patch(
                "amplifier_module_provider_github_copilot.sdk_adapter._imports.CopilotClient",
                MockCopilotClient,
            ),
        ):
            await wrapper.prewarm()
            await wrapper.prewarm()  # Second call should be no-op
            assert init_count == 1, "prewarm() should only initialize once"

        await wrapper.close()


class TestSDKLogLevelEnvOverride:
    """L3: SDK log-level env override behavior tests.

    Contract: sdk-protection:Subprocess:MUST:7
    Priority: Environment > YAML default
    """

    def test_resolve_log_level_uses_yaml_default(self) -> None:
        """When no env var set, uses YAML default.

        Contract: sdk-protection:Subprocess:MUST:7
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            _resolve_sdk_log_level,  # pyright: ignore[reportPrivateUsage]
        )

        # Load actual config — assumes valid YAML
        result = _resolve_sdk_log_level()
        # Should be one of the valid log levels from YAML
        assert result in {"none", "error", "warning", "info", "debug", "all"}

    def test_resolve_log_level_env_overrides_yaml(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Env var overrides YAML default (priority: env > yaml).

        Contract: sdk-protection:Subprocess:MUST:7
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_sdk_protection_config,
        )
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            _resolve_sdk_log_level,  # pyright: ignore[reportPrivateUsage]
        )

        # Get the env var name from config
        load_sdk_protection_config.cache_clear()
        config = load_sdk_protection_config()
        env_var_name = config.sdk.log_level_env_var

        # Set env override
        monkeypatch.setenv(env_var_name, "debug")

        result = _resolve_sdk_log_level()
        assert result == "debug"

    def test_resolve_log_level_empty_env_uses_yaml(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Empty env var treated as not set — uses YAML.

        Contract: sdk-protection:Subprocess:MUST:7
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_sdk_protection_config,
        )
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            _resolve_sdk_log_level,  # pyright: ignore[reportPrivateUsage]
        )

        # Get the env var name from config
        load_sdk_protection_config.cache_clear()
        config = load_sdk_protection_config()
        env_var_name = config.sdk.log_level_env_var
        yaml_default = config.sdk.log_level

        # Set empty env var
        monkeypatch.setenv(env_var_name, "")

        result = _resolve_sdk_log_level()
        # Empty string = falsy = use YAML default
        assert result == yaml_default

    def test_resolve_log_level_config_specifies_env_var_name(self) -> None:
        """Config YAML specifies which env var to check.

        # Contract: sdk-protection:Subprocess:MUST:7

        Three-Medium Architecture: policy (env var name) comes from YAML.
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_sdk_protection_config,
        )

        load_sdk_protection_config.cache_clear()
        config = load_sdk_protection_config()

        # env_var name defined in Python config module
        # Contract: sdk-protection:Subprocess:MUST:7
        assert isinstance(config.sdk.log_level_env_var, str)
        assert config.sdk.log_level_env_var == "COPILOT_SDK_LOG_LEVEL"

    def test_invalid_env_log_level_falls_back_to_yaml(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Invalid env log level falls back to YAML default.

        Contract: sdk-protection:Subprocess:MUST:7
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_sdk_protection_config,
        )
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            _resolve_sdk_log_level,  # pyright: ignore[reportPrivateUsage]
        )

        load_sdk_protection_config.cache_clear()
        config = load_sdk_protection_config()
        env_var_name = config.sdk.log_level_env_var
        yaml_default = config.sdk.log_level

        # Set invalid env value
        monkeypatch.setenv(env_var_name, "INVALID_LEVEL")

        result = _resolve_sdk_log_level()
        # Must fall back to YAML default, not use invalid value
        assert result == yaml_default


# ===========================================================================
# Phase 3: ensure_executable() wiring + singleton lock type
# ===========================================================================


class TestEnsureExecutableWiring:
    """ensure_executable() must be called before CopilotClient.start() on Unix.

    Contract: sdk-boundary:BinaryResolution:MUST:6 — MUST set execute bits.
    The uv package manager strips execute permissions; without this call,
    the SDK subprocess silently fails on fresh installs.
    """

    @pytest.mark.asyncio
    async def test_ensure_executable_called_before_start_on_unix(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """ensure_executable called with binary path before CopilotClient.start().

        # Contract: sdk-boundary:BinaryResolution:MUST:6
        """
        from pathlib import Path
        from unittest.mock import patch

        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        call_order: list[str] = []
        binary_path = Path("/usr/local/bin/copilot")

        def mock_locate_cli_binary() -> Path | None:
            return binary_path

        def mock_ensure_executable(path: Path) -> bool:
            call_order.append(f"ensure_executable:{path}")
            return True

        class MockCopilotClient:
            def __init__(
                self,
                *,
                base_directory: str,
                github_token: str | None = None,
                log_level: str = "info",
                env: dict[str, str],
                mode: str,
            ) -> None:
                pass

            async def start(self) -> None:
                call_order.append("start")

            async def create_session(self, **kwargs: Any) -> Any:
                mock_sess = AsyncMock(spec=_MockSDKSession)
                mock_sess.session_id = "exe-test"
                mock_sess.disconnect = AsyncMock()
                return mock_sess

            async def stop(self) -> None:
                pass

        with (
            patch(
                "amplifier_module_provider_github_copilot.sdk_adapter._imports.CopilotClient",
                MockCopilotClient,
            ),
            patch(
                "amplifier_module_provider_github_copilot.sdk_adapter.client.locate_cli_binary",
                mock_locate_cli_binary,
            ),
            patch(
                "amplifier_module_provider_github_copilot.sdk_adapter.client.ensure_executable",
                mock_ensure_executable,
            ),
        ):
            wrapper = CopilotClientWrapper()
            async with wrapper.session(model="gpt-4"):
                pass

        assert f"ensure_executable:{binary_path}" in call_order, (
            "ensure_executable must be called before start()"
        )
        exe_idx = call_order.index(f"ensure_executable:{binary_path}")
        start_idx = call_order.index("start")
        assert exe_idx < start_idx, (
            f"ensure_executable (pos {exe_idx}) must come before start() (pos {start_idx})"
        )

    @pytest.mark.asyncio
    async def test_ensure_executable_failure_raises_before_start(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When ensure_executable returns False (chmod silently no-oped on
        NTFS-mounted /mnt/, FUSE, etc.), the wrapper MUST raise
        ProviderUnavailableError before launching the subprocess.

        Rationale: subprocess.exec() would surface an opaque OS EACCES error
        two layers removed from the actual cause. The provider already knows
        the cause one frame earlier — fail fast with a diagnostic message.

        Contract: sdk-boundary:BinaryResolution:MUST:6
        """
        from pathlib import Path
        from unittest.mock import patch

        from amplifier_core.llm_errors import ProviderUnavailableError

        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        call_order: list[str] = []
        binary_path = Path("/mnt/d/copilot-bundle/bin/copilot")

        def mock_locate_cli_binary() -> Path | None:
            return binary_path

        def mock_ensure_executable(path: Path) -> bool:
            # Simulate NTFS-on-WSL: chmod "succeeded" but didn't persist.
            call_order.append(f"ensure_executable:{path}")
            return False

        class MockCopilotClient:
            def __init__(
                self,
                *,
                base_directory: str,
                github_token: str | None = None,
                log_level: str = "info",
                env: dict[str, str],
                mode: str,
            ) -> None:
                pass

            async def start(self) -> None:
                # MUST NOT be reached — we should raise before this.
                call_order.append("start")

            async def stop(self) -> None:
                pass

        with (
            patch(
                "amplifier_module_provider_github_copilot.sdk_adapter._imports.CopilotClient",
                MockCopilotClient,
            ),
            patch(
                "amplifier_module_provider_github_copilot.sdk_adapter.client.locate_cli_binary",
                mock_locate_cli_binary,
            ),
            patch(
                "amplifier_module_provider_github_copilot.sdk_adapter.client.ensure_executable",
                mock_ensure_executable,
            ),
        ):
            wrapper = CopilotClientWrapper()
            with pytest.raises(ProviderUnavailableError) as exc_info:
                async with wrapper.session(model="gpt-4"):
                    pass

        # Behavioral assertions — every one must turn red if the fix is reverted.
        assert "start" not in call_order, (
            "subprocess MUST NOT be launched when ensure_executable returned False; "
            f"call_order={call_order}"
        )
        assert f"ensure_executable:{binary_path}" in call_order, (
            "ensure_executable must still be invoked"
        )
        msg = str(exc_info.value)
        assert "execute" in msg.lower() or "permission" in msg.lower(), (
            f"error message must mention execute/permission cause; got: {msg!r}"
        )
        assert str(binary_path) in msg, (
            f"error message must include binary path for diagnosability; got: {msg!r}"
        )
        assert exc_info.value.provider == "github-copilot", (
            "ProviderUnavailableError must carry provider tag for routing"
        )

    @pytest.mark.asyncio
    async def test_ensure_executable_failure_message_is_wsl_specific_on_wsl(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """On WSL, the error message MUST mention /mnt/ and DrvFs metadata —
        the precise actionable cause for the most common trigger of this error.

        Contract: sdk-boundary:BinaryResolution:MUST:6
        """
        from pathlib import Path
        from unittest.mock import patch

        from amplifier_core.llm_errors import ProviderUnavailableError

        from amplifier_module_provider_github_copilot._platform import PlatformInfo
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        binary_path = Path("/mnt/d/copilot-bundle/bin/copilot")
        wsl_info = PlatformInfo(
            name="Unix", is_windows=False, is_wsl=True, cli_binary_name="copilot"
        )

        class MockCopilotClient:
            def __init__(
                self,
                *,
                base_directory: str,
                github_token: str | None = None,
                log_level: str = "info",
                env: dict[str, str],
                mode: str,
            ) -> None:
                pass

            async def start(self) -> None:
                pass

            async def stop(self) -> None:
                pass

        with (
            patch(
                "amplifier_module_provider_github_copilot.sdk_adapter._imports.CopilotClient",
                MockCopilotClient,
            ),
            patch(
                "amplifier_module_provider_github_copilot.sdk_adapter.client.locate_cli_binary",
                lambda: binary_path,
            ),
            patch(
                "amplifier_module_provider_github_copilot.sdk_adapter.client.ensure_executable",
                lambda _p: False,
            ),
            patch(
                "amplifier_module_provider_github_copilot.sdk_adapter.client.get_platform_info",
                lambda: wsl_info,
            ),
        ):
            wrapper = CopilotClientWrapper()
            with pytest.raises(ProviderUnavailableError) as exc_info:
                async with wrapper.session(model="gpt-4"):
                    pass

        msg = str(exc_info.value)
        assert "/mnt/" in msg, f"WSL message must mention /mnt/; got: {msg!r}"
        assert "metadata" in msg.lower(), (
            f"WSL message must mention DrvFs 'metadata' option; got: {msg!r}"
        )

    @pytest.mark.asyncio
    async def test_ensure_executable_failure_message_is_generic_on_non_wsl(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """On non-WSL Unix (macOS, native Linux, containers), the message MUST
        NOT use WSL-specific jargon. It must point to network shares / volume
        mounts / FUSE — the actual triggers in those environments.

        Contract: sdk-boundary:BinaryResolution:MUST:6
        """
        from pathlib import Path
        from unittest.mock import patch

        from amplifier_core.llm_errors import ProviderUnavailableError

        from amplifier_module_provider_github_copilot._platform import PlatformInfo
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        binary_path = Path("/Users/dev/.local/copilot/bin/copilot")
        macos_info = PlatformInfo(
            name="Unix", is_windows=False, is_wsl=False, cli_binary_name="copilot"
        )

        class MockCopilotClient:
            def __init__(
                self,
                *,
                base_directory: str,
                github_token: str | None = None,
                log_level: str = "info",
                env: dict[str, str],
                mode: str,
            ) -> None:
                pass

            async def start(self) -> None:
                pass

            async def stop(self) -> None:
                pass

        with (
            patch(
                "amplifier_module_provider_github_copilot.sdk_adapter._imports.CopilotClient",
                MockCopilotClient,
            ),
            patch(
                "amplifier_module_provider_github_copilot.sdk_adapter.client.locate_cli_binary",
                lambda: binary_path,
            ),
            patch(
                "amplifier_module_provider_github_copilot.sdk_adapter.client.ensure_executable",
                lambda _p: False,
            ),
            patch(
                "amplifier_module_provider_github_copilot.sdk_adapter.client.get_platform_info",
                lambda: macos_info,
            ),
        ):
            wrapper = CopilotClientWrapper()
            with pytest.raises(ProviderUnavailableError) as exc_info:
                async with wrapper.session(model="gpt-4"):
                    pass

        msg = str(exc_info.value)
        assert "/mnt/" not in msg, f"non-WSL message must NOT mention WSL /mnt/ paths; got: {msg!r}"
        assert "DrvFs" not in msg, f"non-WSL message must NOT mention WSL DrvFs; got: {msg!r}"
        # Must point to the actual non-WSL triggers.
        assert any(term in msg for term in ("NFS", "SMB", "CIFS", "FUSE", "network", "volume")), (
            f"non-WSL message must mention network shares / volume mounts / FUSE; got: {msg!r}"
        )

    @pytest.mark.asyncio
    async def test_configuration_error_not_translated_to_unavailable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """ConfigurationError raised by CopilotClient constructor MUST propagate as
        ConfigurationError, NOT be re-wrapped as ProviderUnavailableError by the
        catchall translator.

        Rationale: ConfigurationError signals "user misconfigured this" — a
        fail-closed condition that orchestrators MUST NOT treat as transient
        unavailability. Re-wrapping would change retry semantics and obscure
        the user-facing fix.

        Trigger path: CopilotClient constructor raises ConfigurationError →
        except (ProviderUnavailableError, ConfigurationError): raise preserves type.

        Contract: sdk-boundary:BinaryResolution:MUST:10
        """
        from unittest.mock import patch

        from amplifier_core.llm_errors import ConfigurationError

        import amplifier_module_provider_github_copilot.sdk_adapter.client as client_mod
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        class FakeCopilotClient:
            def __init__(self, **_: Any) -> None:
                raise ConfigurationError(
                    "failing closed to prevent unintended default authentication",
                    provider="github-copilot",
                )

        translate_calls: list[Exception] = []

        def spy_translate(exc: Exception, _config: Any, **_kwargs: Any) -> Exception:
            translate_calls.append(exc)
            return ConfigurationError("translated", provider="github-copilot")

        monkeypatch.setattr(client_mod, "CopilotClient", FakeCopilotClient)
        with patch(
            "amplifier_module_provider_github_copilot.error_translation.translate_sdk_error",
            spy_translate,
        ):
            wrapper = CopilotClientWrapper()
            with pytest.raises(ConfigurationError) as exc_info:
                async with wrapper.session(model="gpt-4"):
                    pass

        # Core behavioral invariant: the translator MUST NOT have been invoked.
        # If translate_sdk_error ran, the original ConfigurationError was caught
        # by the catchall — even if the YAML mapping happens to round-trip the
        # type back to ConfigurationError, that's fragile and loses the original
        # message + cause chain.
        assert translate_calls == [], (
            f"translate_sdk_error MUST NOT be called for ConfigurationError; "
            f"called with: {translate_calls!r}"
        )
        assert type(exc_info.value) is ConfigurationError, (
            f"must propagate as ConfigurationError, not re-wrapped; "
            f"got: {type(exc_info.value).__name__}"
        )
        # Verify the ORIGINAL message survived (proves no re-wrapping).
        assert "failing closed" in str(exc_info.value).lower(), (
            f"original fail-closed message must survive; got: {exc_info.value!s}"
        )
        assert exc_info.value.provider == "github-copilot"

    @pytest.mark.asyncio
    async def test_ensure_executable_failure_clears_owned_client_for_retry(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When ensure_executable fails and we raise, the constructed-but-never-
        started CopilotClient MUST be cleared so the NEXT call to session() /
        list_models() retries initialization rather than returning a stale
        unstarted client via the double-check locking.

        Without this guard, the first request gets the diagnostic error, but
        every subsequent request silently returns a never-started client and
        fails opaquely deep inside create_session() — defeating the whole
        purpose of the fail-fast diagnostic.

        Contract: sdk-boundary:BinaryResolution:MUST:9
        """
        from pathlib import Path
        from unittest.mock import patch

        from amplifier_core.llm_errors import ProviderUnavailableError

        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        binary_path = Path("/mnt/d/copilot-bundle/bin/copilot")

        class MockCopilotClient:
            instances: list[Any] = []

            def __init__(
                self,
                *,
                base_directory: str,
                github_token: str | None = None,
                log_level: str = "info",
                env: dict[str, str],
                mode: str,
            ) -> None:
                MockCopilotClient.instances.append(self)
                self.started = False

            async def start(self) -> None:
                self.started = True

            async def stop(self) -> None:
                pass

        MockCopilotClient.instances = []

        with (
            patch(
                "amplifier_module_provider_github_copilot.sdk_adapter._imports.CopilotClient",
                MockCopilotClient,
            ),
            patch(
                "amplifier_module_provider_github_copilot.sdk_adapter.client.locate_cli_binary",
                lambda: binary_path,
            ),
            patch(
                "amplifier_module_provider_github_copilot.sdk_adapter.client.ensure_executable",
                lambda _p: False,
            ),
        ):
            wrapper = CopilotClientWrapper()

            with pytest.raises(ProviderUnavailableError):
                async with wrapper.session(model="gpt-4"):
                    pass

            # Critical invariant: _owned_client must be cleared. Otherwise the
            # next call hits the double-check locking and returns a stale,
            # never-started client.
            assert wrapper._owned_client is None, (
                f"_owned_client MUST be cleared after ensure_executable failure; "
                f"got: {wrapper._owned_client!r}"
            )

            # Second attempt MUST raise diagnostic error (not silently return
            # stale client). Confirms retry path works.
            with pytest.raises(ProviderUnavailableError):
                async with wrapper.session(model="gpt-4"):
                    pass

            # Two CopilotClient instances must have been constructed (one per
            # retry), proving the lazy-init path was re-entered, not skipped.
            assert len(MockCopilotClient.instances) == 2, (
                f"second attempt must re-enter init; "
                f"got {len(MockCopilotClient.instances)} construction(s)"
            )
            assert not any(c.started for c in MockCopilotClient.instances), (
                "no CopilotClient should have been started"
            )


class TestSingletonLock:
    """The module-level singleton lock must be threading.Lock, not asyncio.Lock.

    # Contract: provider-protocol:mount:MUST:5 (prose bullet — table fix pending)
    An asyncio.Lock is event-loop scoped; using one across multiple event loops
    (e.g., in multi-turn test suites or multi-threaded Amplifier environments)
    raises RuntimeError. threading.Lock works reliably across all loops/threads.
    """

    def test_state_lock_is_threading_lock(self) -> None:
        """_state_lock must be threading.Lock, not asyncio.Lock.

        Contract: provider-protocol:mount:MUST:5
        """
        import asyncio
        import threading

        import amplifier_module_provider_github_copilot as mod

        assert hasattr(mod, "_state_lock"), (
            "_state_lock (threading.Lock) must exist in __init__.py for cross-loop safety"
        )
        lock = mod._state_lock  # type: ignore[attr-defined]
        assert isinstance(lock, type(threading.Lock())), (
            f"_state_lock must be threading.Lock, got {type(lock)}. "
            "asyncio.Lock is event-loop scoped and unsafe for cross-loop singleton access."
        )
        assert not isinstance(lock, asyncio.Lock), (
            "Must NOT be asyncio.Lock — it binds to the creating event loop"
        )
