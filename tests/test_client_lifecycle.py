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
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.fixtures.config_capture import ConfigCapturingMock

if TYPE_CHECKING:
    pass


class TestLazyClientInit:
    """AC-1: Client not created until session() called.

    Contract: sdk-boundary:Config:MUST:1
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

    Contract: deny-destroy:Ephemeral:MUST:1
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
        mock_session = AsyncMock()
        mock_session.session_id = "concurrent-test"
        mock_session.disconnect = AsyncMock()

        class MockCopilotClient:
            def __init__(self, config: Any = None) -> None:
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


class TestSessionConfigInvariants:
    """AC-4: Session config always contains required fields.

    Contract: deny-destroy:ToolSuppression:MUST:1
    Contract: sdk-boundary:Config:MUST:2,3,4
    Contract: deny-destroy:DenyHook:MUST:1
    """

    @pytest.mark.asyncio
    async def test_session_config_available_tools_not_set(self) -> None:
        """available_tools must NOT be set to avoid SDK whitelist behavior.

        SDK treats available_tools=[] as "no tools allowed" (empty whitelist).
        By NOT setting available_tools, we let user tools with overrides_built_in_tool=True
        take precedence over SDK built-ins.

        Contract: deny-destroy:ToolSuppression:MUST:1
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        mock_client = ConfigCapturingMock()
        wrapper = CopilotClientWrapper(sdk_client=mock_client)

        async with wrapper.session(model="gpt-4"):
            pass

        # available_tools should NOT be in config (Bug #1 fix)
        # Setting it to [] was disabling all tools via SDK whitelist
        assert "available_tools" not in mock_client.last_config

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

        mock_session = AsyncMock()
        mock_session.session_id = "normal-exit"
        mock_session.disconnect = AsyncMock()

        mock_client = AsyncMock()
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

        mock_session = AsyncMock()
        mock_session.session_id = "exception-exit"
        mock_session.disconnect = AsyncMock()

        mock_client = AsyncMock()
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

        mock_session = AsyncMock()
        mock_session.session_id = "disconnect-fail"
        mock_session.disconnect = AsyncMock(side_effect=RuntimeError("disconnect failed"))

        mock_client = AsyncMock()
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

        mock_session = AsyncMock()
        mock_session.session_id = "disconnect-escalate"
        mock_session.disconnect = AsyncMock(side_effect=RuntimeError("disconnect failed"))

        mock_client = AsyncMock()
        mock_client.create_session = AsyncMock(return_value=mock_session)

        wrapper = CopilotClientWrapper(sdk_client=mock_client)

        # Trigger 4 disconnect failures to exceed threshold (>3)
        with caplog.at_level(logging.ERROR):
            for _ in range(4):
                async with wrapper.session(model="gpt-4"):
                    pass

        # After 4 failures (>3), should see ERROR level log
        log_text = caplog.text.lower()
        assert "resource leak" in log_text or "multiple disconnect" in log_text


class TestFailedStartCleanup:
    """AC-7: Failed start() clears _owned_client.

    Contract: sdk-boundary:Config:MUST:1
    """

    @pytest.mark.asyncio
    async def test_failed_start_clears_owned_client(self) -> None:
        """_owned_client = None after start() fails.

        Contract: sdk-boundary:Config:MUST:1
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        class FailingCopilotClient:
            def __init__(self, config: Any = None) -> None:
                pass

            async def start(self) -> None:
                raise RuntimeError("SDK start failed")

        wrapper = CopilotClientWrapper()

        # Patch CopilotClient using patch() with full path for reliable async behavior
        with patch(
            "amplifier_module_provider_github_copilot.sdk_adapter._imports.CopilotClient",
            FailingCopilotClient,
        ):
            # session() should raise because start() fails
            # Note: The error is translated via translate_sdk_error(), so we catch
            # the base Exception and verify the original message is preserved
            with pytest.raises(Exception, match="SDK start failed"):
                async with wrapper.session(model="gpt-4"):
                    pass  # Should not reach here

            # _owned_client should be cleared after failed start
            assert wrapper._owned_client is None  # pyright: ignore[reportPrivateUsage]


class TestCloseOwnershipSemantics:
    """AC-8, AC-9: Close ownership semantics and idempotency.

    Contract: provider-protocol:complete:MUST:3
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
        mock_owned = AsyncMock()
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

        mock_injected = AsyncMock()
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
        mock_owned = AsyncMock()
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

            assert (
                "SDK not installed" in str(exc_info.value)
                or "copilot" in str(exc_info.value).lower()
            )


class TestHealthCheck:
    """is_healthy() method for singleton pattern.

    Contract: sdk-boundary:Config:MUST:1
    """

    def test_is_healthy_true_initially(self) -> None:
        """Fresh wrapper is healthy.

        Contract: sdk-boundary:Config:MUST:1
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        wrapper = CopilotClientWrapper()
        assert wrapper.is_healthy() is True

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

    @pytest.mark.asyncio
    async def test_is_healthy_with_injected_client(self) -> None:
        """Wrapper with injected client is healthy.

        Contract: sdk-boundary:Config:MUST:1
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        mock_client = MagicMock()
        wrapper = CopilotClientWrapper(sdk_client=mock_client)
        assert wrapper.is_healthy() is True


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


class TestTokenFallbackWarning:
    """Test warning when token exists but SubprocessConfig unavailable.

    This prevents silent token drops when SDK version mismatch causes
    SubprocessConfig to be None.
    """

    @pytest.mark.asyncio
    async def test_logs_warning_when_token_dropped(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Logs warning when SubprocessConfig is None but token exists.

        Contract: sdk-boundary:Config:MUST:1
        Bug fix: Token silently dropped when SubprocessConfig unavailable.
        """
        import logging

        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        # Set a token
        monkeypatch.setenv("GITHUB_TOKEN", "test-token-value")

        # Mock SDK imports with SubprocessConfig = None (simulates SDK version mismatch)
        mock_session = AsyncMock()
        mock_session.session_id = "fallback-test"
        mock_session.disconnect = AsyncMock()

        class MockCopilotClient:
            def __init__(self, config: Any = None) -> None:
                pass

            async def start(self) -> None:
                pass

            async def create_session(self, **kwargs: Any) -> Any:
                return mock_session

            async def stop(self) -> None:
                pass

        # Patch to simulate SubprocessConfig = None
        with (
            patch(
                "amplifier_module_provider_github_copilot.sdk_adapter._imports.CopilotClient",
                MockCopilotClient,
            ),
            patch(
                "amplifier_module_provider_github_copilot.sdk_adapter._imports.SubprocessConfig",
                None,
            ),
            caplog.at_level(logging.WARNING),
        ):
            wrapper = CopilotClientWrapper()
            async with wrapper.session(model="gpt-4"):
                pass

        # Verify warning was logged
        assert any(
            "SubprocessConfig unavailable" in record.message
            and "token will be ignored" in record.message
            for record in caplog.records
        ), f"Expected warning about token being ignored, got: {[r.message for r in caplog.records]}"

    @pytest.mark.asyncio
    async def test_no_warning_when_token_none(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """No warning when no token exists (normal fallback path).

        Contract: sdk-boundary:Config:MUST:1
        """
        import logging

        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        # Clear all token env vars
        for var in ("COPILOT_AGENT_TOKEN", "COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
            monkeypatch.delenv(var, raising=False)

        mock_session = AsyncMock()
        mock_session.session_id = "no-token-test"
        mock_session.disconnect = AsyncMock()

        class MockCopilotClient:
            def __init__(self, config: Any = None) -> None:
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
            patch(
                "amplifier_module_provider_github_copilot.sdk_adapter._imports.SubprocessConfig",
                None,
            ),
            caplog.at_level(logging.WARNING),
        ):
            wrapper = CopilotClientWrapper()
            async with wrapper.session(model="gpt-4"):
                pass

        # No warning should be logged when token is None
        assert not any("token will be ignored" in record.message for record in caplog.records), (
            f"Unexpected warning: {[r.message for r in caplog.records]}"
        )
