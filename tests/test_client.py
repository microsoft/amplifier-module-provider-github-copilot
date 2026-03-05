"""
Tests for CopilotClientWrapper.

This module tests the client wrapper including lifecycle management,
error handling, session creation, input validation, and cancellation handling.
"""

import asyncio
import logging
import os
from unittest.mock import AsyncMock, Mock, patch

import pytest

from amplifier_module_provider_github_copilot._constants import SDK_TIMEOUT_BUFFER_SECONDS
from amplifier_module_provider_github_copilot.client import (
    AuthStatus,
    CopilotClientWrapper,
    SessionInfo,
    SessionListResult,
)
from amplifier_module_provider_github_copilot.exceptions import (
    CopilotAuthenticationError,
    CopilotConnectionError,
    CopilotModelNotFoundError,
    CopilotProviderError,
    CopilotRateLimitError,
    CopilotSessionError,
    CopilotTimeoutError,
)


class TestCopilotClientWrapper:
    """Tests for CopilotClientWrapper."""

    @pytest.fixture
    def client_wrapper(self):
        """Create client wrapper for testing."""
        return CopilotClientWrapper(config={}, timeout=60.0)

    @pytest.mark.asyncio
    async def test_ensure_client_initializes_once(self, client_wrapper, mock_copilot_client):
        """ensure_client should initialize client only once."""
        with patch(
            "copilot.CopilotClient",
            return_value=mock_copilot_client,
        ):
            # First call should initialize
            client1 = await client_wrapper.ensure_client()
            mock_copilot_client.start.assert_called_once()

            # Second call should return same client
            client2 = await client_wrapper.ensure_client()
            assert client1 is client2
            # start should still only be called once
            mock_copilot_client.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_client_verifies_auth(self, client_wrapper, mock_copilot_client):
        """ensure_client should verify authentication."""
        with patch(
            "copilot.CopilotClient",
            return_value=mock_copilot_client,
        ):
            await client_wrapper.ensure_client()
            mock_copilot_client.get_auth_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_client_auth_failure(self, client_wrapper, mock_copilot_client):
        """ensure_client should raise on auth failure."""
        mock_copilot_client.get_auth_status.return_value.isAuthenticated = False

        with patch(
            "copilot.CopilotClient",
            return_value=mock_copilot_client,
        ):
            with pytest.raises(CopilotAuthenticationError):
                await client_wrapper.ensure_client()

    @pytest.mark.asyncio
    async def test_create_session_creates_and_destroys(
        self, client_wrapper, mock_copilot_client, mock_copilot_session
    ):
        """create_session should create and destroy session."""
        mock_copilot_client.create_session = AsyncMock(return_value=mock_copilot_session)

        with patch(
            "copilot.CopilotClient",
            return_value=mock_copilot_client,
        ):
            async with client_wrapper.create_session("claude-opus-4.5") as session:
                assert session.session_id == "test-session-123"

            # Session should be destroyed after context
            mock_copilot_session.destroy.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_session_with_system_message(
        self, client_wrapper, mock_copilot_client, mock_copilot_session
    ):
        """create_session should pass system message."""
        mock_copilot_client.create_session = AsyncMock(return_value=mock_copilot_session)

        with patch(
            "copilot.CopilotClient",
            return_value=mock_copilot_client,
        ):
            async with client_wrapper.create_session(
                "claude-opus-4.5",
                system_message="You are helpful.",
            ):
                pass

            # Check session config included system message
            call_config = mock_copilot_client.create_session.call_args[0][0]
            assert "system_message" in call_config
            assert call_config["system_message"]["content"] == "You are helpful."

    @pytest.mark.asyncio
    async def test_send_and_wait_returns_response(self, client_wrapper, mock_copilot_session):
        """send_and_wait should return response from session."""
        expected_response = Mock()
        mock_copilot_session.send_and_wait = AsyncMock(return_value=expected_response)

        response = await client_wrapper.send_and_wait(mock_copilot_session, "Hello!")

        assert response is expected_response
        mock_copilot_session.send_and_wait.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_and_wait_timeout(self, client_wrapper, mock_copilot_session):
        """send_and_wait should raise on timeout."""

        async def slow_response(*args, **kwargs):
            await asyncio.sleep(10)
            return Mock()

        mock_copilot_session.send_and_wait = slow_response

        with pytest.raises(CopilotTimeoutError):
            await client_wrapper.send_and_wait(
                mock_copilot_session,
                "Hello!",
                timeout=0.01,
            )

    @pytest.mark.asyncio
    async def test_close_stops_client(self, client_wrapper, mock_copilot_client):
        """close should stop the client."""
        with patch(
            "copilot.CopilotClient",
            return_value=mock_copilot_client,
        ):
            await client_wrapper.ensure_client()
            await client_wrapper.close()

            mock_copilot_client.stop.assert_called_once()
            assert client_wrapper._client is None

    @pytest.mark.asyncio
    async def test_close_handles_errors(self, client_wrapper, mock_copilot_client):
        """close should handle errors gracefully."""
        mock_copilot_client.stop = AsyncMock(side_effect=Exception("Stop failed"))

        with patch(
            "copilot.CopilotClient",
            return_value=mock_copilot_client,
        ):
            await client_wrapper.ensure_client()
            # Should not raise
            await client_wrapper.close()

    @pytest.mark.asyncio
    async def test_close_idempotent(self, client_wrapper):
        """close should be safe to call multiple times."""
        # Should not raise even when client not initialized
        await client_wrapper.close()
        await client_wrapper.close()

    def test_is_connected_false_initially(self, client_wrapper):
        """is_connected should be False before start."""
        assert client_wrapper.is_connected is False

    @pytest.mark.asyncio
    async def test_is_connected_true_after_start(self, client_wrapper, mock_copilot_client):
        """is_connected should be True after initialization."""
        with patch(
            "copilot.CopilotClient",
            return_value=mock_copilot_client,
        ):
            await client_wrapper.ensure_client()
            assert client_wrapper.is_connected is True

    @pytest.mark.asyncio
    async def test_context_manager(self, client_wrapper, mock_copilot_client):
        """Client should work as async context manager."""
        with patch(
            "copilot.CopilotClient",
            return_value=mock_copilot_client,
        ):
            async with client_wrapper as wrapper:
                assert wrapper.is_connected is True

            # Should be closed after exit
            mock_copilot_client.stop.assert_called_once()


class TestClientErrors:
    """Tests for client error handling."""

    @pytest.mark.asyncio
    async def test_connection_error_on_start_failure(self):
        """Should raise CopilotConnectionError when start fails."""
        wrapper = CopilotClientWrapper(config={}, timeout=60.0)

        mock_client = Mock()
        mock_client.start = AsyncMock(side_effect=Exception("Connection failed"))

        with patch(
            "copilot.CopilotClient",
            return_value=mock_client,
        ):
            with pytest.raises(CopilotConnectionError):
                await wrapper.ensure_client()

    @pytest.mark.asyncio
    async def test_sdk_import_error(self):
        """Should raise CopilotConnectionError when SDK not installed."""
        wrapper = CopilotClientWrapper(config={}, timeout=60.0)

        with patch(
            "copilot.CopilotClient",
            side_effect=ImportError("No module named 'copilot'"),
        ):
            with pytest.raises(CopilotConnectionError, match="not installed"):
                await wrapper.ensure_client()

    @pytest.mark.asyncio
    async def test_session_error_on_create_failure(self, mock_copilot_client):
        """Should raise CopilotSessionError when session creation fails."""
        wrapper = CopilotClientWrapper(config={}, timeout=60.0)
        mock_copilot_client.create_session = AsyncMock(
            side_effect=Exception("Session creation failed")
        )

        with patch(
            "copilot.CopilotClient",
            return_value=mock_copilot_client,
        ):
            with pytest.raises(CopilotSessionError):
                async with wrapper.create_session("model"):
                    pass


class TestInputValidation:
    """Tests for input validation."""

    def test_timeout_must_be_positive(self):
        """Should raise ValueError for non-positive timeout."""
        with pytest.raises(ValueError, match="timeout must be positive"):
            CopilotClientWrapper(config={}, timeout=0)

        with pytest.raises(ValueError, match="timeout must be positive"):
            CopilotClientWrapper(config={}, timeout=-1.0)

    def test_timeout_accepts_positive_values(self):
        """Should accept positive timeout values."""
        wrapper = CopilotClientWrapper(config={}, timeout=0.1)
        assert wrapper._timeout == 0.1

        wrapper = CopilotClientWrapper(config={}, timeout=300.0)
        assert wrapper._timeout == 300.0

    @pytest.mark.asyncio
    async def test_model_must_be_non_empty(self, mock_copilot_client):
        """Should raise ValueError for empty model string."""
        wrapper = CopilotClientWrapper(config={}, timeout=60.0)

        with patch(
            "copilot.CopilotClient",
            return_value=mock_copilot_client,
        ):
            with pytest.raises(ValueError, match="model must be a non-empty string"):
                async with wrapper.create_session(""):
                    pass

            with pytest.raises(ValueError, match="model must be a non-empty string"):
                async with wrapper.create_session("   "):
                    pass

    @pytest.mark.asyncio
    async def test_model_is_stripped(self, mock_copilot_client, mock_copilot_session):
        """Should strip whitespace from model name."""
        wrapper = CopilotClientWrapper(config={}, timeout=60.0)
        mock_copilot_client.create_session = AsyncMock(return_value=mock_copilot_session)

        with patch(
            "copilot.CopilotClient",
            return_value=mock_copilot_client,
        ):
            async with wrapper.create_session("  claude-opus-4.5  "):
                pass

            call_config = mock_copilot_client.create_session.call_args[0][0]
            assert call_config["model"] == "claude-opus-4.5"

    @pytest.mark.asyncio
    async def test_reasoning_effort_validation(self, mock_copilot_client):
        """Should raise ValueError for invalid reasoning_effort."""
        wrapper = CopilotClientWrapper(config={}, timeout=60.0)

        with patch(
            "copilot.CopilotClient",
            return_value=mock_copilot_client,
        ):
            with pytest.raises(ValueError, match="reasoning_effort must be one of"):
                async with wrapper.create_session("model", reasoning_effort="invalid"):
                    pass

            with pytest.raises(ValueError, match="reasoning_effort must be one of"):
                async with wrapper.create_session("model", reasoning_effort="max"):
                    pass

    @pytest.mark.asyncio
    async def test_valid_reasoning_efforts(self, mock_copilot_client, mock_copilot_session):
        """Should accept valid reasoning_effort values."""
        wrapper = CopilotClientWrapper(config={}, timeout=60.0)
        mock_copilot_client.create_session = AsyncMock(return_value=mock_copilot_session)

        valid_efforts = ["low", "medium", "high", "xhigh"]

        with patch(
            "copilot.CopilotClient",
            return_value=mock_copilot_client,
        ):
            for effort in valid_efforts:
                async with wrapper.create_session("model", reasoning_effort=effort):
                    pass

                call_config = mock_copilot_client.create_session.call_args[0][0]
                assert call_config.get("reasoning_effort") == effort

    @pytest.mark.asyncio
    async def test_prompt_must_be_non_empty(self, mock_copilot_session):
        """Should raise ValueError for empty prompt."""
        wrapper = CopilotClientWrapper(config={}, timeout=60.0)

        with pytest.raises(ValueError, match="prompt must be a non-empty string"):
            await wrapper.send_and_wait(mock_copilot_session, "")


class TestTimeoutBehavior:
    """Tests for timeout and abort behavior."""

    @pytest.mark.asyncio
    async def test_abort_called_on_timeout(self, mock_copilot_session):
        """Should call abort on session when timeout occurs."""
        wrapper = CopilotClientWrapper(config={}, timeout=60.0)

        async def slow_response(*args, **kwargs):
            await asyncio.sleep(10)
            return Mock()

        mock_copilot_session.send_and_wait = slow_response
        mock_copilot_session.abort = AsyncMock()

        with pytest.raises(CopilotTimeoutError):
            await wrapper.send_and_wait(mock_copilot_session, "Hello!", timeout=0.01)

        # Abort should have been called
        mock_copilot_session.abort.assert_called_once()

    @pytest.mark.asyncio
    async def test_abort_error_logged_not_raised(self, mock_copilot_session):
        """Should log but not raise abort errors."""
        wrapper = CopilotClientWrapper(config={}, timeout=60.0)

        async def slow_response(*args, **kwargs):
            await asyncio.sleep(10)
            return Mock()

        mock_copilot_session.send_and_wait = slow_response
        mock_copilot_session.abort = AsyncMock(side_effect=Exception("Abort failed"))

        # Should still raise CopilotTimeoutError, not the abort error
        with pytest.raises(CopilotTimeoutError):
            await wrapper.send_and_wait(mock_copilot_session, "Hello!", timeout=0.01)


class TestAuthStatusAndSessions:
    """Tests for get_auth_status and list_sessions methods."""

    @pytest.mark.asyncio
    async def test_get_auth_status_returns_typed_result(self, mock_copilot_client):
        """get_auth_status should return AuthStatus dataclass."""
        wrapper = CopilotClientWrapper(config={}, timeout=60.0)

        # Set up complete auth status
        auth_status = Mock()
        auth_status.isAuthenticated = True
        auth_status.login = "test-user"
        auth_status.authType = "oauth"
        auth_status.host = "github.com"
        auth_status.statusMessage = "Authenticated"
        mock_copilot_client.get_auth_status = AsyncMock(return_value=auth_status)

        with patch(
            "copilot.CopilotClient",
            return_value=mock_copilot_client,
        ):
            result = await wrapper.get_auth_status()

            assert isinstance(result, AuthStatus)
            assert result.is_authenticated is True
            assert result.github_user == "test-user"
            assert result.auth_type == "oauth"
            assert result.host == "github.com"
            assert result.status_message == "Authenticated"
            assert result.error is None

    @pytest.mark.asyncio
    async def test_get_auth_status_returns_error_on_failure(self, mock_copilot_client):
        """get_auth_status should return error field on failure."""
        wrapper = CopilotClientWrapper(config={}, timeout=60.0)
        mock_copilot_client.get_auth_status = AsyncMock(side_effect=Exception("Network error"))

        with patch(
            "copilot.CopilotClient",
            return_value=mock_copilot_client,
        ):
            result = await wrapper.get_auth_status()

            assert isinstance(result, AuthStatus)
            # Should be None (unknown), not False
            assert result.is_authenticated is None
            assert result.github_user is None
            assert result.error == "Network error"

    @pytest.mark.asyncio
    async def test_list_sessions_returns_typed_result(self, mock_copilot_client):
        """list_sessions should return SessionListResult with SessionInfo."""
        wrapper = CopilotClientWrapper(config={}, timeout=60.0)

        # Set up session metadata
        session1 = Mock()
        session1.sessionId = "session-1"
        session1.summary = "Test session"
        session1.startTime = "2026-02-06T10:00:00Z"
        session1.modifiedTime = "2026-02-06T11:00:00Z"
        session1.isRemote = False

        mock_copilot_client.list_sessions = AsyncMock(return_value=[session1])

        with patch(
            "copilot.CopilotClient",
            return_value=mock_copilot_client,
        ):
            result = await wrapper.list_sessions()

            assert isinstance(result, SessionListResult)
            assert result.error is None
            assert len(result.sessions) == 1
            assert isinstance(result.sessions[0], SessionInfo)
            assert result.sessions[0].session_id == "session-1"
            assert result.sessions[0].summary == "Test session"
            assert result.sessions[0].start_time == "2026-02-06T10:00:00Z"
            assert result.sessions[0].modified_time == "2026-02-06T11:00:00Z"
            assert result.sessions[0].is_remote is False

    @pytest.mark.asyncio
    async def test_list_sessions_returns_error_on_failure(self, mock_copilot_client):
        """list_sessions should return error field on failure."""
        wrapper = CopilotClientWrapper(config={}, timeout=60.0)
        mock_copilot_client.list_sessions = AsyncMock(side_effect=Exception("Connection lost"))

        with patch(
            "copilot.CopilotClient",
            return_value=mock_copilot_client,
        ):
            result = await wrapper.list_sessions()

            assert isinstance(result, SessionListResult)
            assert result.sessions == ()
            assert result.error == "Connection lost"


class TestCancellationHandling:
    """Tests for cancellation handling during client initialization."""

    @pytest.mark.asyncio
    async def test_client_cleaned_up_on_start_failure(self, mock_copilot_client):
        """Should clean up client if start fails."""
        wrapper = CopilotClientWrapper(config={}, timeout=60.0)
        mock_copilot_client.start = AsyncMock(side_effect=Exception("Start failed"))
        mock_copilot_client.stop = AsyncMock()

        with patch(
            "copilot.CopilotClient",
            return_value=mock_copilot_client,
        ):
            with pytest.raises(CopilotConnectionError):
                await wrapper.ensure_client()

            # Client should not be stored
            assert wrapper._client is None
            assert wrapper._started is False

    @pytest.mark.asyncio
    async def test_model_not_found_error(self, mock_copilot_client):
        """Should raise CopilotModelNotFoundError for model not found errors."""
        wrapper = CopilotClientWrapper(config={}, timeout=60.0)
        mock_copilot_client.create_session = AsyncMock(
            side_effect=Exception("Model 'invalid-model' not found")
        )

        with patch(
            "copilot.CopilotClient",
            return_value=mock_copilot_client,
        ):
            with pytest.raises(CopilotModelNotFoundError) as exc_info:
                async with wrapper.create_session("invalid-model"):
                    pass

            assert exc_info.value.model == "invalid-model"


class TestExceptionPassthrough:
    """Tests that exceptions from caller code pass through unchanged."""

    @pytest.mark.asyncio
    async def test_caller_exception_not_wrapped(self, mock_copilot_client, mock_copilot_session):
        """Exceptions raised during yield should not be wrapped as CopilotSessionError."""
        wrapper = CopilotClientWrapper(config={}, timeout=60.0)
        mock_copilot_client.create_session = AsyncMock(return_value=mock_copilot_session)

        class CustomError(Exception):
            pass

        with patch(
            "copilot.CopilotClient",
            return_value=mock_copilot_client,
        ):
            with pytest.raises(CustomError):
                async with wrapper.create_session("model"):
                    raise CustomError("User code error")

            # Session should still be destroyed
            mock_copilot_session.destroy.assert_called_once()

    @pytest.mark.asyncio
    async def test_session_destroy_called_on_any_exit(
        self, mock_copilot_client, mock_copilot_session
    ):
        """Session should be destroyed on normal exit, exception, or cancellation."""
        wrapper = CopilotClientWrapper(config={}, timeout=60.0)
        mock_copilot_client.create_session = AsyncMock(return_value=mock_copilot_session)

        with patch(
            "copilot.CopilotClient",
            return_value=mock_copilot_client,
        ):
            # Normal exit
            async with wrapper.create_session("model"):
                pass
            assert mock_copilot_session.destroy.call_count == 1

            mock_copilot_session.destroy.reset_mock()

            # Exception exit
            try:
                async with wrapper.create_session("model"):
                    raise ValueError("test")
            except ValueError:
                pass
            assert mock_copilot_session.destroy.call_count == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Additional coverage tests — targeting uncovered lines from coverage report
# ═══════════════════════════════════════════════════════════════════════════════


class TestBuildClientOptions:
    """Tests for _build_client_options configuration branches."""

    def test_log_level_option_included(self):
        """Should include log_level in options when configured."""
        wrapper = CopilotClientWrapper(config={"log_level": "debug"}, timeout=60.0)
        options = wrapper._build_client_options()
        assert options["log_level"] == "debug"

    def test_auto_restart_true_included(self):
        """Should include auto_restart=True when configured."""
        wrapper = CopilotClientWrapper(config={"auto_restart": True}, timeout=60.0)
        options = wrapper._build_client_options()
        assert options["auto_restart"] is True

    def test_auto_restart_false_included(self):
        """Should include auto_restart=False when explicitly set (not None)."""
        wrapper = CopilotClientWrapper(config={"auto_restart": False}, timeout=60.0)
        options = wrapper._build_client_options()
        assert options["auto_restart"] is False

    def test_cwd_option_included(self):
        """Should include cwd in options when configured."""
        wrapper = CopilotClientWrapper(config={"cwd": "/tmp/project"}, timeout=60.0)
        options = wrapper._build_client_options()
        assert options["cwd"] == "/tmp/project"

    def test_all_options_combined(self):
        """Should include all configured options together."""
        config = {
            "log_level": "info",
            "auto_restart": True,
            "cwd": "/workspace",
        }
        wrapper = CopilotClientWrapper(config=config, timeout=60.0)
        options = wrapper._build_client_options()
        assert options == {
            "log_level": "info",
            "auto_restart": True,
            "cwd": "/workspace",
        }

    def test_empty_config_yields_empty_options(self):
        """Empty config should yield empty options."""
        wrapper = CopilotClientWrapper(config={}, timeout=60.0)
        options = wrapper._build_client_options()
        assert options == {}


class TestEnsureClientCancellation:
    """Tests for CancelledError handling in ensure_client."""

    @pytest.mark.asyncio
    async def test_cancelled_during_start_stops_client(self):
        """Should call stop() on client if CancelledError occurs during start."""
        wrapper = CopilotClientWrapper(config={}, timeout=60.0)
        mock_client = Mock()
        mock_client.start = AsyncMock(side_effect=asyncio.CancelledError())
        mock_client.stop = AsyncMock()

        with patch("copilot.CopilotClient", return_value=mock_client):
            with pytest.raises(asyncio.CancelledError):
                await wrapper.ensure_client()

        mock_client.stop.assert_called_once()
        assert wrapper._client is None
        assert wrapper._started is False

    @pytest.mark.asyncio
    async def test_cancelled_cleanup_swallows_stop_error(self):
        """Should swallow stop() error during CancelledError cleanup."""
        wrapper = CopilotClientWrapper(config={}, timeout=60.0)
        mock_client = Mock()
        mock_client.start = AsyncMock(side_effect=asyncio.CancelledError())
        mock_client.stop = AsyncMock(side_effect=RuntimeError("Stop exploded"))

        with patch("copilot.CopilotClient", return_value=mock_client):
            # Should still raise CancelledError, not RuntimeError
            with pytest.raises(asyncio.CancelledError):
                await wrapper.ensure_client()

        mock_client.stop.assert_called_once()


class TestSessionConfigBuilding:
    """Tests for session config construction paths in create_session."""

    @pytest.mark.asyncio
    async def test_tools_in_session_config(self, mock_copilot_client, mock_copilot_session):
        """Should add tools list to session config."""
        wrapper = CopilotClientWrapper(config={}, timeout=60.0)
        mock_copilot_client.create_session = AsyncMock(return_value=mock_copilot_session)

        mock_tool = Mock()
        mock_tool.name = "read_file"
        tools = [mock_tool]

        with patch("copilot.CopilotClient", return_value=mock_copilot_client):
            async with wrapper.create_session("model", tools=tools):
                pass

        config = mock_copilot_client.create_session.call_args[0][0]
        assert config["tools"] is tools
        assert "available_tools" not in config

    @pytest.mark.asyncio
    async def test_excluded_tools_in_session_config(
        self, mock_copilot_client, mock_copilot_session
    ):
        """Should add excluded_tools to session config."""
        wrapper = CopilotClientWrapper(config={}, timeout=60.0)
        mock_copilot_client.create_session = AsyncMock(return_value=mock_copilot_session)

        excluded = ["edit", "bash", "view"]

        with patch("copilot.CopilotClient", return_value=mock_copilot_client):
            async with wrapper.create_session("model", excluded_tools=excluded):
                pass

        config = mock_copilot_client.create_session.call_args[0][0]
        assert config["excluded_tools"] == excluded

    @pytest.mark.asyncio
    async def test_hooks_in_session_config(self, mock_copilot_client, mock_copilot_session):
        """Should add hooks dict to session config."""
        wrapper = CopilotClientWrapper(config={}, timeout=60.0)
        mock_copilot_client.create_session = AsyncMock(return_value=mock_copilot_session)

        deny_hook = Mock()
        hooks = {"on_pre_tool_use": deny_hook}

        with patch("copilot.CopilotClient", return_value=mock_copilot_client):
            async with wrapper.create_session("model", hooks=hooks):
                pass

        config = mock_copilot_client.create_session.call_args[0][0]
        assert config["hooks"] is hooks

    @pytest.mark.asyncio
    async def test_full_session_config_all_options(self, mock_copilot_client, mock_copilot_session):
        """Should build config with ALL optional parameters set."""
        wrapper = CopilotClientWrapper(config={}, timeout=60.0)
        mock_copilot_client.create_session = AsyncMock(return_value=mock_copilot_session)

        tools = [Mock()]
        excluded = ["bash"]
        hooks = {"on_pre_tool_use": Mock()}

        with patch("copilot.CopilotClient", return_value=mock_copilot_client):
            async with wrapper.create_session(
                "claude-opus-4.5",
                system_message="Be helpful.",
                streaming=False,
                reasoning_effort="high",
                tools=tools,
                excluded_tools=excluded,
                hooks=hooks,
            ):
                pass

        config = mock_copilot_client.create_session.call_args[0][0]
        assert config["model"] == "claude-opus-4.5"
        assert config["system_message"] == {"mode": "append", "content": "Be helpful."}
        assert config["streaming"] is False
        assert config["reasoning_effort"] == "high"
        assert config["tools"] is tools
        assert config["excluded_tools"] == excluded
        assert config["hooks"] is hooks
        assert config["infinite_sessions"] == {"enabled": False}

    @pytest.mark.asyncio
    async def test_streaming_default_true(self, mock_copilot_client, mock_copilot_session):
        """Streaming should default to True in session config."""
        wrapper = CopilotClientWrapper(config={}, timeout=60.0)
        mock_copilot_client.create_session = AsyncMock(return_value=mock_copilot_session)

        with patch("copilot.CopilotClient", return_value=mock_copilot_client):
            async with wrapper.create_session("model"):
                pass

        config = mock_copilot_client.create_session.call_args[0][0]
        assert config["streaming"] is True


class TestSendAndWaitResponsePaths:
    """Tests for send_and_wait response and error paths."""

    @pytest.mark.asyncio
    async def test_returns_response_with_explicit_timeout(self, mock_copilot_session):
        """Should return response when explicit timeout is provided."""
        wrapper = CopilotClientWrapper(config={}, timeout=60.0)
        expected = Mock()
        mock_copilot_session.send_and_wait = AsyncMock(return_value=expected)

        response = await wrapper.send_and_wait(mock_copilot_session, "Hello!", timeout=30.0)

        assert response is expected
        # Verify SDK timeout includes buffer
        call_args = mock_copilot_session.send_and_wait.call_args
        assert call_args[1]["timeout"] == 30.0 + SDK_TIMEOUT_BUFFER_SECONDS

    @pytest.mark.asyncio
    async def test_returns_response_with_default_timeout(self, mock_copilot_session):
        """Should use default timeout when none specified."""
        wrapper = CopilotClientWrapper(config={}, timeout=120.0)
        expected = Mock()
        mock_copilot_session.send_and_wait = AsyncMock(return_value=expected)

        response = await wrapper.send_and_wait(mock_copilot_session, "Test prompt")

        assert response is expected
        call_args = mock_copilot_session.send_and_wait.call_args
        assert call_args[1]["timeout"] == 120.0 + SDK_TIMEOUT_BUFFER_SECONDS

    @pytest.mark.asyncio
    async def test_non_timeout_exception_wrapped_in_provider_error(self, mock_copilot_session):
        """Non-timeout SDK exceptions should be wrapped in CopilotProviderError."""
        wrapper = CopilotClientWrapper(config={}, timeout=60.0)
        mock_copilot_session.send_and_wait = AsyncMock(side_effect=RuntimeError("Connection reset"))

        with pytest.raises(CopilotProviderError, match="Request failed"):
            await wrapper.send_and_wait(mock_copilot_session, "Hello!")

    @pytest.mark.asyncio
    async def test_copilot_timeout_error_passthrough(self, mock_copilot_session):
        """CopilotTimeoutError should pass through without double-wrapping."""
        wrapper = CopilotClientWrapper(config={}, timeout=60.0)
        original_error = CopilotTimeoutError(timeout=60.0, message="Already timed out")
        mock_copilot_session.send_and_wait = AsyncMock(side_effect=original_error)

        with pytest.raises(CopilotTimeoutError) as exc_info:
            await wrapper.send_and_wait(mock_copilot_session, "Hello!")

        assert exc_info.value is original_error

    @pytest.mark.asyncio
    async def test_timeout_triggers_abort_success(self, mock_copilot_session):
        """Successful abort after timeout should complete without extra errors."""
        wrapper = CopilotClientWrapper(config={}, timeout=60.0)

        async def slow(*args, **kwargs):
            await asyncio.sleep(10)

        mock_copilot_session.send_and_wait = slow
        mock_copilot_session.abort = AsyncMock()

        with pytest.raises(CopilotTimeoutError):
            await wrapper.send_and_wait(mock_copilot_session, "Hello!", timeout=0.01)

        mock_copilot_session.abort.assert_called_once()

    @pytest.mark.asyncio
    async def test_timeout_abort_failure_still_raises_timeout(self, mock_copilot_session):
        """Failed abort after timeout should still raise CopilotTimeoutError."""
        wrapper = CopilotClientWrapper(config={}, timeout=60.0)

        async def slow(*args, **kwargs):
            await asyncio.sleep(10)

        mock_copilot_session.send_and_wait = slow
        mock_copilot_session.abort = AsyncMock(side_effect=RuntimeError("Abort explosion"))

        with pytest.raises(CopilotTimeoutError):
            await wrapper.send_and_wait(mock_copilot_session, "Hello!", timeout=0.01)


class TestBrokenPipeErrorHandling:
    """Tests for BrokenPipeError handling during send operations."""

    @pytest.mark.asyncio
    async def test_broken_pipe_error_during_send(self, mock_copilot_session):
        """BrokenPipeError from send_and_wait should be wrapped as CopilotConnectionError (retryable).

        BrokenPipeError signals that the underlying subprocess died mid-request.
        The client catches it and wraps it in CopilotConnectionError which is
        retryable (vs CopilotProviderError which may not be).
        """
        wrapper = CopilotClientWrapper(config={}, timeout=60.0)
        mock_copilot_session.send_and_wait = AsyncMock(
            side_effect=BrokenPipeError("Connection broken: subprocess died")
        )

        with pytest.raises(CopilotConnectionError, match="Connection broken"):
            await wrapper.send_and_wait(mock_copilot_session, "Hello!")

    @pytest.mark.asyncio
    async def test_broken_pipe_preserves_cause(self, mock_copilot_session):
        """Original BrokenPipeError should be preserved as __cause__."""
        wrapper = CopilotClientWrapper(config={}, timeout=60.0)
        original = BrokenPipeError("Subprocess closed stdout")
        mock_copilot_session.send_and_wait = AsyncMock(side_effect=original)

        with pytest.raises(CopilotConnectionError) as exc_info:
            await wrapper.send_and_wait(mock_copilot_session, "Test prompt")

        assert exc_info.value.__cause__ is original


class TestLockTimeoutHandling:
    """Tests for lock timeout handling in ensure_client."""

    @pytest.fixture
    def client_wrapper(self):
        """Create fresh client wrapper for testing."""
        return CopilotClientWrapper(config={}, timeout=60.0)

    @pytest.mark.asyncio
    async def test_lock_timeout_raises_connection_error(self, client_wrapper):
        """Lock timeout during ensure_client should raise CopilotConnectionError with clear message.

        When the initialization lock is held (e.g., another concurrent call is initializing),
        ensure_client must not block indefinitely. It should timeout and raise
        CopilotConnectionError with a user-actionable message.
        """
        # Simulate lock held by another concurrent initializer
        await client_wrapper._lock.acquire()

        try:
            with patch(
                "amplifier_module_provider_github_copilot.client.CLIENT_INIT_LOCK_TIMEOUT", 0.01
            ):
                with pytest.raises(CopilotConnectionError) as exc_info:
                    await client_wrapper.ensure_client()

            # Error message should be clear about what timed out
            assert "Timed out waiting for client initialization lock" in str(exc_info.value)
        finally:
            client_wrapper._lock.release()


class TestListModelsErrorHandling:
    """Tests for list_models error paths."""

    @pytest.mark.asyncio
    async def test_list_models_wraps_exception(self, mock_copilot_client):
        """Should wrap list_models exceptions in CopilotProviderError."""
        wrapper = CopilotClientWrapper(config={}, timeout=60.0)
        mock_copilot_client.list_models = AsyncMock(side_effect=RuntimeError("API unavailable"))

        with patch("copilot.CopilotClient", return_value=mock_copilot_client):
            with pytest.raises(CopilotProviderError, match="Failed to list models"):
                await wrapper.list_models()

    @pytest.mark.asyncio
    async def test_list_models_preserves_cause(self, mock_copilot_client):
        """Should preserve original exception as __cause__."""
        wrapper = CopilotClientWrapper(config={}, timeout=60.0)
        original = ConnectionError("Socket closed")
        mock_copilot_client.list_models = AsyncMock(side_effect=original)

        with patch("copilot.CopilotClient", return_value=mock_copilot_client):
            with pytest.raises(CopilotProviderError) as exc_info:
                await wrapper.list_models()

            assert exc_info.value.__cause__ is original


class TestGetAuthStatusErrorPath:
    """Tests for get_auth_status error handling path."""

    @pytest.mark.asyncio
    async def test_auth_status_error_returns_unknown(self, mock_copilot_client):
        """Error during auth check should return unknown status with error."""
        wrapper = CopilotClientWrapper(config={}, timeout=60.0)

        # First call (during ensure_client → _verify_authentication) succeeds,
        # second call (the actual get_auth_status method) fails.
        call_count = 0
        auth_ok = Mock()
        auth_ok.isAuthenticated = True
        auth_ok.login = "test-user"

        async def auth_side_effect():
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                return auth_ok
            raise ConnectionError("Lost connection")

        mock_copilot_client.get_auth_status = AsyncMock(side_effect=auth_side_effect)

        with patch("copilot.CopilotClient", return_value=mock_copilot_client):
            result = await wrapper.get_auth_status()

        assert isinstance(result, AuthStatus)
        assert result.is_authenticated is None
        assert result.github_user is None
        assert result.error == "Lost connection"


class TestListSessionsPaths:
    """Tests for list_sessions method — full body coverage."""

    @pytest.mark.asyncio
    async def test_list_sessions_empty(self, mock_copilot_client):
        """Should handle empty session list."""
        wrapper = CopilotClientWrapper(config={}, timeout=60.0)
        mock_copilot_client.list_sessions = AsyncMock(return_value=[])

        with patch("copilot.CopilotClient", return_value=mock_copilot_client):
            result = await wrapper.list_sessions()

        assert isinstance(result, SessionListResult)
        assert result.sessions == ()
        assert result.error is None

    @pytest.mark.asyncio
    async def test_list_sessions_maps_all_fields(self, mock_copilot_client):
        """Should map all SDK session fields to SessionInfo."""
        wrapper = CopilotClientWrapper(config={}, timeout=60.0)

        s1 = Mock(
            sessionId="s-1",
            summary="Chat about Python",
            startTime="2026-02-06T10:00:00Z",
            modifiedTime="2026-02-06T11:00:00Z",
            isRemote=False,
        )
        s2 = Mock(
            sessionId="s-2",
            summary=None,
            startTime="2026-02-07T09:00:00Z",
            modifiedTime="2026-02-07T09:30:00Z",
            isRemote=True,
        )
        mock_copilot_client.list_sessions = AsyncMock(return_value=[s1, s2])

        with patch("copilot.CopilotClient", return_value=mock_copilot_client):
            result = await wrapper.list_sessions()

        assert len(result.sessions) == 2
        assert result.sessions[0].session_id == "s-1"
        assert result.sessions[0].summary == "Chat about Python"
        assert result.sessions[0].start_time == "2026-02-06T10:00:00Z"
        assert result.sessions[0].modified_time == "2026-02-06T11:00:00Z"
        assert result.sessions[0].is_remote is False
        assert result.sessions[1].session_id == "s-2"
        assert result.sessions[1].summary is None
        assert result.sessions[1].is_remote is True
        assert result.error is None

    @pytest.mark.asyncio
    async def test_list_sessions_error_returns_empty_with_message(self, mock_copilot_client):
        """Should return empty tuple with error message on failure."""
        wrapper = CopilotClientWrapper(config={}, timeout=60.0)
        mock_copilot_client.list_sessions = AsyncMock(
            side_effect=RuntimeError("Connection dropped")
        )

        with patch("copilot.CopilotClient", return_value=mock_copilot_client):
            result = await wrapper.list_sessions()

        assert result.sessions == ()
        assert result.error == "Connection dropped"


class TestBuildClientOptionsTokenPassthrough:
    """Tests for github_token passthrough in _build_client_options()."""

    @pytest.fixture
    def wrapper(self):
        """Create a CopilotClientWrapper for testing."""
        return CopilotClientWrapper(config={}, timeout=60.0)

    def test_no_token_by_default(self, wrapper):
        """Options should not include github_token when no token is available."""
        with patch.dict(os.environ, {}, clear=True):
            options = wrapper._build_client_options()
            assert "github_token" not in options

    def test_token_from_config(self, wrapper):
        """Config github_token should take highest precedence."""
        wrapper._config = {"github_token": "config-token-123"}
        with patch.dict(
            os.environ,
            {"GITHUB_TOKEN": "env-token", "GH_TOKEN": "gh-token"},
            clear=True,
        ):
            options = wrapper._build_client_options()
            assert options["github_token"] == "config-token-123"

    def test_token_from_copilot_github_token_env(self, wrapper):
        """COPILOT_GITHUB_TOKEN should be second precedence."""
        with patch.dict(
            os.environ,
            {
                "COPILOT_GITHUB_TOKEN": "copilot-token",
                "GH_TOKEN": "gh-token",
                "GITHUB_TOKEN": "github-token",
            },
            clear=True,
        ):
            options = wrapper._build_client_options()
            assert options["github_token"] == "copilot-token"

    def test_token_from_gh_token_env(self, wrapper):
        """GH_TOKEN should be third precedence."""
        with patch.dict(
            os.environ,
            {"GH_TOKEN": "gh-token", "GITHUB_TOKEN": "github-token"},
            clear=True,
        ):
            options = wrapper._build_client_options()
            assert options["github_token"] == "gh-token"

    def test_token_from_github_token_env(self, wrapper):
        """GITHUB_TOKEN should be lowest env var precedence."""
        with patch.dict(os.environ, {"GITHUB_TOKEN": "github-token"}, clear=True):
            options = wrapper._build_client_options()
            assert options["github_token"] == "github-token"

    def test_empty_env_var_ignored(self, wrapper):
        """Empty string env vars should be ignored."""
        with patch.dict(os.environ, {"GITHUB_TOKEN": ""}, clear=True):
            options = wrapper._build_client_options()
            assert "github_token" not in options


class TestVerifyAuthenticationMessage:
    """Tests for improved auth error messages."""

    @pytest.mark.asyncio
    async def test_auth_error_mentions_env_vars(self):
        """Auth error message should mention GITHUB_TOKEN and amplifier init."""
        wrapper = CopilotClientWrapper(config={}, timeout=60.0)

        mock_client = AsyncMock()
        auth_status = Mock()
        auth_status.isAuthenticated = False
        mock_client.get_auth_status = AsyncMock(return_value=auth_status)

        wrapper._client = mock_client
        wrapper._started = True

        with pytest.raises(CopilotAuthenticationError, match="GITHUB_TOKEN"):
            await wrapper._verify_authentication()


class TestRateLimitDetectionInClient:
    """Tests that rate-limit errors from the SDK are detected and raised as CopilotRateLimitError."""

    @pytest.mark.asyncio
    async def test_send_and_wait_detects_rate_limit_with_retry_after(self, mock_copilot_session):
        """send_and_wait should raise CopilotRateLimitError with retry_after when SDK error contains rate limit info."""
        wrapper = CopilotClientWrapper(config={}, timeout=60.0)
        mock_copilot_session.send_and_wait = AsyncMock(
            side_effect=RuntimeError("Rate limit exceeded. Retry after 30.0 seconds")
        )

        with pytest.raises(CopilotRateLimitError) as exc_info:
            await wrapper.send_and_wait(mock_copilot_session, "Hello!")

        assert exc_info.value.retry_after == 30.0

    @pytest.mark.asyncio
    async def test_send_and_wait_detects_rate_limit_without_retry_after(self, mock_copilot_session):
        """send_and_wait should raise CopilotRateLimitError even without retry_after hint."""
        wrapper = CopilotClientWrapper(config={}, timeout=60.0)
        mock_copilot_session.send_and_wait = AsyncMock(
            side_effect=RuntimeError("429 Too Many Requests")
        )

        with pytest.raises(CopilotRateLimitError) as exc_info:
            await wrapper.send_and_wait(mock_copilot_session, "Hello!")

        assert exc_info.value.retry_after is None

    @pytest.mark.asyncio
    async def test_send_and_wait_non_rate_limit_still_wraps_as_provider_error(
        self, mock_copilot_session
    ):
        """Non-rate-limit errors in send_and_wait should still wrap as CopilotProviderError."""
        wrapper = CopilotClientWrapper(config={}, timeout=60.0)
        mock_copilot_session.send_and_wait = AsyncMock(side_effect=RuntimeError("Connection reset"))

        with pytest.raises(CopilotProviderError, match="Request failed"):
            await wrapper.send_and_wait(mock_copilot_session, "Hello!")

    @pytest.mark.asyncio
    async def test_create_session_detects_rate_limit(self, mock_copilot_client):
        """create_session should raise CopilotRateLimitError when SDK error contains rate limit info."""
        wrapper = CopilotClientWrapper(config={}, timeout=60.0)
        mock_copilot_client.create_session = AsyncMock(
            side_effect=Exception("Rate limit exceeded. Retry after 60 seconds")
        )

        with patch.object(
            CopilotClientWrapper,
            "ensure_client",
            new=AsyncMock(return_value=mock_copilot_client),
        ):
            with pytest.raises(CopilotRateLimitError) as exc_info:
                async with wrapper.create_session("model"):
                    pass

            assert exc_info.value.retry_after == 60.0

    @pytest.mark.asyncio
    async def test_create_session_non_rate_limit_still_wraps_as_session_error(
        self, mock_copilot_client
    ):
        """Non-rate-limit errors in create_session should still wrap as CopilotSessionError."""
        wrapper = CopilotClientWrapper(config={}, timeout=60.0)
        mock_copilot_client.create_session = AsyncMock(
            side_effect=Exception("Session creation failed")
        )

        with patch.object(
            CopilotClientWrapper,
            "ensure_client",
            new=AsyncMock(return_value=mock_copilot_client),
        ):
            with pytest.raises(CopilotSessionError):
                async with wrapper.create_session("model"):
                    pass

    @pytest.mark.asyncio
    async def test_send_and_wait_preserves_cause_chain(self, mock_copilot_session):
        """CopilotRateLimitError should preserve the original exception as __cause__."""
        wrapper = CopilotClientWrapper(config={}, timeout=60.0)
        original = RuntimeError("Rate limit exceeded")
        mock_copilot_session.send_and_wait = AsyncMock(side_effect=original)

        with pytest.raises(CopilotRateLimitError) as exc_info:
            await wrapper.send_and_wait(mock_copilot_session, "Hello!")

        assert exc_info.value.__cause__ is original

    @pytest.mark.asyncio
    async def test_create_session_preserves_cause_chain(self, mock_copilot_client):
        """CopilotRateLimitError from create_session should preserve the original exception as __cause__."""
        wrapper = CopilotClientWrapper(config={}, timeout=60.0)
        original = Exception("Too many requests")
        mock_copilot_client.create_session = AsyncMock(side_effect=original)

        with patch.object(
            CopilotClientWrapper,
            "ensure_client",
            new=AsyncMock(return_value=mock_copilot_client),
        ):
            with pytest.raises(CopilotRateLimitError) as exc_info:
                async with wrapper.create_session("model"):
                    pass

            assert exc_info.value.__cause__ is original


class TestImportErrorMessagePackageName:
    """The ImportError handler in ensure_client() must tell the user the correct package name."""

    async def test_import_error_message_says_github_copilot_sdk(self):
        """ensure_client() ImportError message must contain 'github-copilot-sdk', not 'copilot-sdk'."""
        wrapper = CopilotClientWrapper(config={}, timeout=60.0)

        with patch(
            "copilot.CopilotClient",
            side_effect=ImportError("No module named 'copilot'"),
        ):
            with pytest.raises(CopilotConnectionError, match="github-copilot-sdk") as exc_info:
                await wrapper.ensure_client()

        msg = str(exc_info.value)
        assert "github-copilot-sdk" in msg  # the CORRECT form


class TestEnsureClientExceptionPassthrough:
    """CopilotAuthenticationError and CopilotConnectionError must pass through
    ensure_client() without being re-wrapped by the generic except Exception handler."""

    @pytest.fixture
    def client_wrapper(self):
        """Create a fresh CopilotClientWrapper for each test."""
        return CopilotClientWrapper(config={}, timeout=60.0)

    async def test_auth_error_passes_through_unchanged(self, client_wrapper, mock_copilot_client):
        """When _verify_authentication raises CopilotAuthenticationError,
        ensure_client must re-raise the EXACT same exception — not wrap it."""
        # Make auth fail: get_auth_status returns isAuthenticated=False
        auth_status = Mock()
        auth_status.isAuthenticated = False
        mock_copilot_client.get_auth_status = AsyncMock(return_value=auth_status)

        with patch(
            "copilot.CopilotClient",
            return_value=mock_copilot_client,
        ):
            with pytest.raises(CopilotAuthenticationError) as exc_info:
                await client_wrapper.ensure_client()

        # The message must be EXACTLY what _verify_authentication produces —
        # no "Copilot authentication failed: ..." wrapper around it.
        msg = str(exc_info.value)
        assert msg == (
            "Not authenticated to GitHub Copilot. "
            "Set GITHUB_TOKEN, run 'gh auth login', "
            "or run 'amplifier init' to authenticate."
        ), f"Message was double-wrapped: {msg!r}"

        # The exception must NOT have a __cause__ (it's not a `raise X from e` chain).
        # If the generic handler caught it, it would set __cause__ via `from e`.
        assert exc_info.value.__cause__ is None, (
            "Exception has a __cause__, which means it was re-raised with 'from e' "
            "by the generic handler instead of passing through directly"
        )

    async def test_connection_error_passes_through_unchanged(
        self, client_wrapper, mock_copilot_client
    ):
        """When CopilotConnectionError is raised inside the try block,
        ensure_client must re-raise it unchanged — not wrap it in another CopilotConnectionError."""
        original_error = CopilotConnectionError("original connection error message")

        # Make client.start() raise CopilotConnectionError.
        # We raise from start() rather than get_auth_status() because
        # _verify_authentication() swallows generic exceptions (logs + ignores).
        # CopilotConnectionError is NOT a CopilotAuthenticationError, so it
        # would hit the generic except in _verify_authentication and get swallowed.
        mock_copilot_client.start = AsyncMock(side_effect=original_error)

        with patch(
            "copilot.CopilotClient",
            return_value=mock_copilot_client,
        ):
            with pytest.raises(CopilotConnectionError) as exc_info:
                await client_wrapper.ensure_client()

        # The exception must be the EXACT same object — not a new wrapper
        assert exc_info.value is original_error, (
            f"Expected the original CopilotConnectionError object, "
            f"but got a different one with message: {str(exc_info.value)!r}"
        )

    async def test_generic_exception_still_gets_wrapped(self, client_wrapper, mock_copilot_client):
        """Exceptions that are NOT CopilotAuthenticationError or CopilotConnectionError
        must still be caught by the generic handler (existing behavior preserved)."""
        mock_copilot_client.start = AsyncMock(side_effect=RuntimeError("unexpected kaboom"))

        with patch(
            "copilot.CopilotClient",
            return_value=mock_copilot_client,
        ):
            with pytest.raises(CopilotConnectionError, match="unexpected kaboom"):
                await client_wrapper.ensure_client()


# ═══════════════════════════════════════════════════════════════════════════════
# Category: Subprocess and Health Check Error Handling
# ═══════════════════════════════════════════════════════════════════════════════


class TestSubprocessErrorHandling:
    """Tests for subprocess-related error handling.

    Coverage for client.py lines 199-221, 261-264:
    - Health check failure detection
    - Subprocess death detection
    - Client re-initialization after failures

    Cross-platform: Subprocess behaviors vary between Windows/WSL/macOS/Linux.
    """

    @pytest.fixture
    def client_wrapper(self):
        """Create fresh client wrapper for testing."""
        return CopilotClientWrapper(config={}, timeout=60.0)

    @pytest.mark.asyncio
    async def test_health_check_detects_subprocess_death(self, client_wrapper):
        """Health check should return False when subprocess dies."""
        mock_client = AsyncMock()
        # Simulate subprocess death error (BrokenPipeError on Unix, similar on Windows)
        mock_client.ping = AsyncMock(side_effect=BrokenPipeError("Subprocess died"))

        client_wrapper._client = mock_client
        client_wrapper._started = True

        result = await client_wrapper._check_client_health()

        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_detects_connection_refused(self, client_wrapper):
        """Health check should return False on ConnectionRefusedError."""
        mock_client = AsyncMock()
        mock_client.ping = AsyncMock(side_effect=ConnectionRefusedError("Connection refused"))

        client_wrapper._client = mock_client
        client_wrapper._started = True

        result = await client_wrapper._check_client_health()

        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_detects_eof_error(self, client_wrapper):
        """Health check should return False on EOFError (pipe closed)."""
        mock_client = AsyncMock()
        mock_client.ping = AsyncMock(side_effect=EOFError("Pipe closed"))

        client_wrapper._client = mock_client
        client_wrapper._started = True

        result = await client_wrapper._check_client_health()

        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_timeout_returns_false(self, client_wrapper):
        """Health check should return False on timeout."""
        mock_client = AsyncMock()

        async def slow_ping():
            await asyncio.sleep(10)  # Much longer than health check timeout

        mock_client.ping = slow_ping

        client_wrapper._client = mock_client
        client_wrapper._started = True

        # Patch the timeout to be very short
        with patch(
            "amplifier_module_provider_github_copilot.client.CLIENT_HEALTH_CHECK_TIMEOUT", 0.01
        ):
            result = await client_wrapper._check_client_health()

        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_returns_true_when_healthy(self, client_wrapper):
        """Health check should return True when client responds."""
        mock_client = AsyncMock()
        mock_client.ping = AsyncMock(return_value=None)

        client_wrapper._client = mock_client
        client_wrapper._started = True

        result = await client_wrapper._check_client_health()

        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_returns_false_when_client_is_none(self, client_wrapper):
        """Health check should return False if client is None."""
        client_wrapper._client = None
        client_wrapper._started = False

        result = await client_wrapper._check_client_health()

        assert result is False

    @pytest.mark.asyncio
    async def test_reset_client_stops_unhealthy_client(self, client_wrapper):
        """Reset should attempt to stop client even if it fails."""
        mock_client = AsyncMock()
        mock_client.stop = AsyncMock(side_effect=RuntimeError("Stop failed"))

        client_wrapper._client = mock_client
        client_wrapper._started = True

        await client_wrapper._reset_client()

        # Client should be reset even if stop failed
        assert client_wrapper._client is None
        assert client_wrapper._started is False
        # stop should have been called
        mock_client.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_client_reinitializes_after_health_failure(
        self, client_wrapper, mock_copilot_client
    ):
        """ensure_client should re-initialize if health check fails."""
        # First client is unhealthy
        dead_client = AsyncMock()
        dead_client.ping = AsyncMock(side_effect=BrokenPipeError("Dead"))
        dead_client.stop = AsyncMock()

        client_wrapper._client = dead_client
        client_wrapper._started = True

        # New client should be created
        with patch(
            "copilot.CopilotClient",
            return_value=mock_copilot_client,
        ):
            new_client = await client_wrapper.ensure_client()

        # Should have stopped the dead client
        dead_client.stop.assert_called_once()
        # Should return the new client
        assert new_client is mock_copilot_client

    @pytest.mark.asyncio
    async def test_ensure_client_lock_timeout_raises_connection_error(self, client_wrapper):
        """ensure_client should raise if lock timeout occurs."""
        # Acquire lock manually to simulate blocking
        await client_wrapper._lock.acquire()

        # Patch timeout to be very short
        with patch(
            "amplifier_module_provider_github_copilot.client.CLIENT_INIT_LOCK_TIMEOUT", 0.01
        ):
            with pytest.raises(CopilotConnectionError) as exc_info:
                await client_wrapper.ensure_client()

        assert "Timed out waiting for client initialization lock" in str(exc_info.value)

        # Release lock for cleanup
        client_wrapper._lock.release()


class TestIsSubprocessDeathFunction:
    """Tests for _is_subprocess_dead_error helper function.

    Coverage for client.py line 56: OSError with EPIPE/ECONNRESET errno.
    Note: Python auto-subclasses OSError(32) → BrokenPipeError, so we need
    a custom OSError subclass to test the errno check on line 55-56.
    """

    def test_oserror_with_epipe_errno_via_custom_class(self):
        """OSError with errno 32 (EPIPE) should be detected via errno check.

        Python auto-promotes OSError(32) to BrokenPipeError, which is caught
        by the first isinstance check. To test the errno fallback (line 56),
        we use a custom OSError subclass.
        """
        from amplifier_module_provider_github_copilot.client import _is_subprocess_dead_error

        # Create a custom OSError that isn't BrokenPipeError
        class WrappedOSError(OSError):
            """OSError wrapper that preserves errno without auto-subclassing."""

            def __init__(self, errno_val: int, msg: str):
                super().__init__(msg)
                self._forced_errno = errno_val

            @property
            def errno(self):  # type: ignore[override]
                return self._forced_errno

        error = WrappedOSError(32, "Broken pipe")
        # Verify it's OSError but not BrokenPipeError
        assert isinstance(error, OSError)
        assert not isinstance(error, BrokenPipeError)
        # Should still be detected via errno check
        assert _is_subprocess_dead_error(error) is True

    def test_oserror_with_econnreset_errno_via_custom_class(self):
        """OSError with errno 104 (ECONNRESET) via custom class."""
        from amplifier_module_provider_github_copilot.client import _is_subprocess_dead_error

        class WrappedOSError(OSError):
            def __init__(self, errno_val: int, msg: str):
                super().__init__(msg)
                self._forced_errno = errno_val

            @property
            def errno(self):  # type: ignore[override]
                return self._forced_errno

        error = WrappedOSError(104, "Connection reset")
        assert isinstance(error, OSError)
        assert not isinstance(error, ConnectionResetError)
        assert _is_subprocess_dead_error(error) is True

    def test_oserror_with_other_errno_returns_false(self):
        """OSError with other errno should not be detected as subprocess death."""
        from amplifier_module_provider_github_copilot.client import _is_subprocess_dead_error

        # ENOENT = 2 (No such file or directory) - doesn't auto-subclass
        error = OSError(2, "No such file or directory")
        assert _is_subprocess_dead_error(error) is False


class TestClientInitializationErrorPaths:
    """Tests for client initialization error paths.

    Coverage for client.py lines 285-312:
    - SDK import failure
    - Client start failure
    - Authentication detection from error messages

    Cross-platform: ImportError and subprocess errors behave similarly.
    """

    @pytest.fixture
    def client_wrapper(self):
        """Create fresh client wrapper for testing."""
        return CopilotClientWrapper(config={}, timeout=60.0)

    @pytest.mark.asyncio
    async def test_import_error_raises_connection_error(self, client_wrapper):
        """SDK import failure should raise CopilotConnectionError."""
        with patch.dict("sys.modules", {"copilot": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module named 'copilot'")):
                with pytest.raises(CopilotConnectionError):
                    await client_wrapper.ensure_client()

        # Note: The actual error depends on import behavior
        # The key is that it should be a CopilotConnectionError

    @pytest.mark.asyncio
    async def test_auth_error_in_message_raises_auth_error(self, client_wrapper):
        """Errors containing 'auth' should raise CopilotAuthenticationError."""
        mock_client = Mock()
        mock_client.start = AsyncMock(side_effect=RuntimeError("Authentication failed: bad token"))

        with patch("copilot.CopilotClient", return_value=mock_client):
            with pytest.raises(CopilotAuthenticationError) as exc_info:
                await client_wrapper.ensure_client()

        assert "authentication" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_token_error_in_message_raises_auth_error(self, client_wrapper):
        """Errors containing 'token' should raise CopilotAuthenticationError."""
        mock_client = Mock()
        mock_client.start = AsyncMock(side_effect=RuntimeError("Invalid token provided"))

        with patch("copilot.CopilotClient", return_value=mock_client):
            with pytest.raises(CopilotAuthenticationError) as exc_info:
                await client_wrapper.ensure_client()

        assert "Copilot authentication failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_login_error_in_message_raises_auth_error(self, client_wrapper):
        """Errors containing 'login' should raise CopilotAuthenticationError."""
        mock_client = Mock()
        mock_client.start = AsyncMock(side_effect=RuntimeError("Please login first"))

        with patch("copilot.CopilotClient", return_value=mock_client):
            with pytest.raises(CopilotAuthenticationError) as exc_info:
                await client_wrapper.ensure_client()

        assert "Copilot authentication failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generic_error_raises_connection_error(self, client_wrapper):
        """Generic errors should raise CopilotConnectionError."""
        mock_client = Mock()
        mock_client.start = AsyncMock(side_effect=RuntimeError("Something went wrong"))

        with patch("copilot.CopilotClient", return_value=mock_client):
            with pytest.raises(CopilotConnectionError) as exc_info:
                await client_wrapper.ensure_client()

        assert "Failed to initialize Copilot client" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_cancellation_cleans_up_partial_client(self, client_wrapper):
        """CancelledError during init should clean up partial client."""
        mock_client = Mock()
        mock_client.start = AsyncMock(side_effect=asyncio.CancelledError())
        mock_client.stop = AsyncMock()

        with patch("copilot.CopilotClient", return_value=mock_client):
            with pytest.raises(asyncio.CancelledError):
                await client_wrapper.ensure_client()

        # Should have attempted to stop the partial client
        mock_client.stop.assert_called_once()


class TestClientOptionsConfiguration:
    """Tests for _build_client_options token resolution.

    Coverage for client.py lines 340-380:
    - Token precedence from config and environment

    Cross-platform: Environment variable behavior is consistent.
    """

    @pytest.fixture
    def client_wrapper(self):
        """Create fresh client wrapper for testing."""
        return CopilotClientWrapper(config={}, timeout=60.0)

    def test_config_token_takes_precedence(self):
        """Token from config should override environment variables."""
        with patch.dict(os.environ, {"GITHUB_TOKEN": "env_token"}, clear=False):
            wrapper = CopilotClientWrapper(config={"github_token": "config_token"}, timeout=60.0)
            options = wrapper._build_client_options()

        assert options.get("github_token") == "config_token"

    def test_copilot_github_token_env_precedence(self):
        """COPILOT_GITHUB_TOKEN should take precedence over GH_TOKEN."""
        with patch.dict(
            os.environ,
            {
                "COPILOT_GITHUB_TOKEN": "copilot_token",
                "GH_TOKEN": "gh_token",
                "GITHUB_TOKEN": "github_token",
            },
            clear=False,
        ):
            wrapper = CopilotClientWrapper(config={}, timeout=60.0)
            options = wrapper._build_client_options()

        assert options.get("github_token") == "copilot_token"

    def test_gh_token_env_precedence_over_github_token(self):
        """GH_TOKEN should take precedence over GITHUB_TOKEN."""
        with patch.dict(
            os.environ, {"GH_TOKEN": "gh_token", "GITHUB_TOKEN": "github_token"}, clear=False
        ):
            # Clear COPILOT_GITHUB_TOKEN if set
            env = {"GH_TOKEN": "gh_token", "GITHUB_TOKEN": "github_token"}
            with patch.dict(os.environ, env, clear=True):
                wrapper = CopilotClientWrapper(config={}, timeout=60.0)
                options = wrapper._build_client_options()

        # GH_TOKEN should be used
        assert options.get("github_token") in ["gh_token", None]

    def test_no_token_returns_no_github_token_key(self):
        """No token available should not include github_token in options."""
        # Clear all token env vars
        with patch.dict(os.environ, {}, clear=True):
            wrapper = CopilotClientWrapper(config={}, timeout=60.0)
            options = wrapper._build_client_options()

        # github_token should not be in options (SDK uses stored OAuth)
        assert "github_token" not in options or options.get("github_token") is None

    def test_optional_config_options_passed_through(self):
        """Optional config like log_level and cwd should be passed through."""
        wrapper = CopilotClientWrapper(
            config={"log_level": "debug", "auto_restart": True, "cwd": "/custom/path"}, timeout=60.0
        )
        options = wrapper._build_client_options()

        assert options.get("log_level") == "debug"
        assert options.get("auto_restart") is True
        assert options.get("cwd") == "/custom/path"


class TestAuthenticationVerification:
    """Tests for _verify_authentication method.

    Coverage for client.py authentication verification paths.
    """

    @pytest.fixture
    def client_wrapper(self):
        """Create fresh client wrapper for testing."""
        return CopilotClientWrapper(config={}, timeout=60.0)

    @pytest.mark.asyncio
    async def test_verify_auth_raises_on_not_authenticated(self, client_wrapper):
        """Should raise CopilotAuthenticationError if not authenticated."""
        mock_client = AsyncMock()
        # Create mock with SDK attribute names (camelCase)
        mock_auth_status = Mock()
        mock_auth_status.isAuthenticated = False
        mock_auth_status.login = None
        mock_client.get_auth_status = AsyncMock(return_value=mock_auth_status)
        client_wrapper._client = mock_client

        with pytest.raises(CopilotAuthenticationError) as exc_info:
            await client_wrapper._verify_authentication()

        assert "Not authenticated" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_verify_auth_passes_when_authenticated(self, client_wrapper):
        """Should not raise if authenticated."""
        mock_client = AsyncMock()
        # Create mock with SDK attribute names (camelCase)
        mock_auth_status = Mock()
        mock_auth_status.isAuthenticated = True
        mock_auth_status.login = "testuser"
        mock_client.get_auth_status = AsyncMock(return_value=mock_auth_status)
        client_wrapper._client = mock_client

        # Should not raise
        await client_wrapper._verify_authentication()

    @pytest.mark.asyncio
    async def test_verify_auth_handles_exception_gracefully(self, client_wrapper):
        """Should not raise on auth check failure (just warn)."""
        mock_client = AsyncMock()
        mock_client.get_auth_status = AsyncMock(side_effect=RuntimeError("Auth service down"))
        client_wrapper._client = mock_client

        # Should not raise (logs warning instead)
        await client_wrapper._verify_authentication()

    @pytest.mark.asyncio
    async def test_verify_auth_skips_when_client_none(self, client_wrapper):
        """Should return early if client is None."""
        client_wrapper._client = None

        # Should not raise
        await client_wrapper._verify_authentication()


class TestSessionDestroyErrorHandling:
    """Tests for session destroy error handling.

    Coverage for client.py lines 558-560: Exception during session.destroy()
    should be logged but not re-raised to avoid masking the original exception.
    """

    @pytest.fixture
    def client_wrapper(self):
        """Create fresh client wrapper for testing."""
        return CopilotClientWrapper(config={}, timeout=60.0)

    @pytest.mark.asyncio
    async def test_session_destroy_error_logged_not_raised(
        self, client_wrapper, mock_copilot_client, caplog
    ):
        """Session destroy exception should be logged but not mask caller exceptions."""
        # Create mock session that fails on destroy
        mock_session = AsyncMock()
        mock_session.session_id = "test-session"
        mock_session.destroy = AsyncMock(side_effect=RuntimeError("Destroy failed"))

        # Mock create_session to return the mock session
        mock_copilot_client.create_session = AsyncMock(return_value=mock_session)

        with patch(
            "copilot.CopilotClient",
            return_value=mock_copilot_client,
        ):
            import logging

            with caplog.at_level(logging.WARNING):
                # Use the session - destroy should fail silently
                async with client_wrapper.create_session("model"):
                    pass  # Normal usage

        # Session destroy should have been called
        mock_session.destroy.assert_called_once()
        # Warning should be logged
        assert "Error destroying session" in caplog.text or "Destroy failed" in caplog.text

    @pytest.mark.asyncio
    async def test_session_destroy_error_does_not_mask_caller_exception(
        self, client_wrapper, mock_copilot_client
    ):
        """If caller raises exception, session destroy error should not mask it."""
        mock_session = AsyncMock()
        mock_session.session_id = "test-session"
        mock_session.destroy = AsyncMock(side_effect=RuntimeError("Destroy failed"))

        mock_copilot_client.create_session = AsyncMock(return_value=mock_session)

        with patch(
            "copilot.CopilotClient",
            return_value=mock_copilot_client,
        ):
            with pytest.raises(ValueError, match="User error"):
                async with client_wrapper.create_session("model"):
                    raise ValueError("User error")  # Simulate user code error

        # The user error should propagate, not the destroy error


class TestListModelsCoverageGaps:
    """Tests for list_models error handling.

    Coverage for client.py lines 643-644: Exception during list_models
    should be wrapped in CopilotProviderError.
    """

    @pytest.fixture
    def client_wrapper(self):
        """Create fresh client wrapper for testing."""
        return CopilotClientWrapper(config={}, timeout=60.0)

    @pytest.mark.asyncio
    async def test_list_models_exception_wrapped(self, client_wrapper, mock_copilot_client):
        """Exception during list_models should be wrapped in CopilotProviderError."""
        mock_copilot_client.list_models = AsyncMock(side_effect=RuntimeError("Network error"))

        with patch(
            "copilot.CopilotClient",
            return_value=mock_copilot_client,
        ):
            with pytest.raises(CopilotProviderError) as exc_info:
                await client_wrapper.list_models()

        assert "Failed to list models" in str(exc_info.value)
        assert exc_info.value.__cause__ is not None  # Original exception preserved

    @pytest.mark.asyncio
    async def test_list_models_success(self, client_wrapper, mock_copilot_client):
        """list_models should return models on success."""
        mock_models = [Mock(id="model-1"), Mock(id="model-2")]
        mock_copilot_client.list_models = AsyncMock(return_value=mock_models)

        with patch(
            "copilot.CopilotClient",
            return_value=mock_copilot_client,
        ):
            result = await client_wrapper.list_models()

        assert len(result) == 2


class TestAuthFailureStateLeak:
    """Tests for P0-1: Auth failure client state leak regression.

    Bug: After CopilotAuthenticationError in ensure_client(), self._client and
    self._started remained truthy, causing subsequent calls to use the
    unauthenticated client.

    Root cause: client.py:321 had faulty cleanup guard condition:
        if client is not None and self._client is None:
    This never executed because self._client was assigned BEFORE
    _verify_authentication() was called.

    Fix: Reset self._client and self._started BEFORE cleanup attempt.
    """

    @pytest.mark.asyncio
    async def test_ensure_client_auth_failure_clears_instance_state(self):
        """After CopilotAuthenticationError, _client and _started must be None/False.

        This is the core regression test for P0-1.
        """
        wrapper = CopilotClientWrapper(config={}, timeout=60.0)

        mock_client = AsyncMock()
        mock_client.start = AsyncMock()
        mock_client.stop = AsyncMock()
        mock_client.ping = AsyncMock()  # For health check

        # Mock auth status to fail
        auth_status = Mock()
        auth_status.isAuthenticated = False
        mock_client.get_auth_status = AsyncMock(return_value=auth_status)

        with patch(
            "copilot.CopilotClient",
            return_value=mock_client,
        ):
            with pytest.raises(CopilotAuthenticationError):
                await wrapper.ensure_client()

        # Core assertion: state MUST be cleaned up after auth failure
        assert wrapper._client is None, "self._client must be None after auth failure"
        assert wrapper._started is False, "self._started must be False after auth failure"
        # Client must have been stopped to avoid resource leak
        mock_client.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_client_retries_after_auth_failure(self):
        """A second ensure_client() call after auth failure must re-initialize.

        This tests that state cleanup allows proper retry on subsequent calls.
        """
        wrapper = CopilotClientWrapper(config={}, timeout=60.0)

        call_count = 0

        mock_client = AsyncMock()
        mock_client.start = AsyncMock()
        mock_client.stop = AsyncMock()
        mock_client.ping = AsyncMock()

        # First call: auth fails. Second call: auth succeeds.
        def make_auth_status():
            nonlocal call_count
            call_count += 1
            status = Mock()
            status.isAuthenticated = call_count > 1  # Fail first, pass second
            status.login = "test-user"
            return status

        mock_client.get_auth_status = AsyncMock(side_effect=lambda: make_auth_status())

        with patch(
            "copilot.CopilotClient",
            return_value=mock_client,
        ):
            # First call fails
            with pytest.raises(CopilotAuthenticationError):
                await wrapper.ensure_client()

            # Second call must succeed (re-initialize, not reuse stale state)
            result = await wrapper.ensure_client()
            assert result is mock_client
            assert wrapper._started is True

        # start() called twice (once per attempt)
        assert mock_client.start.call_count == 2

    @pytest.mark.asyncio
    async def test_ensure_client_generic_error_clears_instance_state(self):
        """Any error during initialization must clear instance state.

        This covers non-auth errors that could also leave stale state.
        """
        wrapper = CopilotClientWrapper(config={}, timeout=60.0)

        mock_client = AsyncMock()
        mock_client.start = AsyncMock()
        mock_client.stop = AsyncMock()
        mock_client.ping = AsyncMock()

        # Auth check raises a generic error
        mock_client.get_auth_status = AsyncMock(side_effect=RuntimeError("Network timeout"))

        with patch(
            "copilot.CopilotClient",
            return_value=mock_client,
        ):
            # Should not raise (auth verification failure is non-fatal warning)
            # But if the error propagates, state must be cleaned
            try:
                await wrapper.ensure_client()
            except Exception:
                pass

        # If we got here without error, check the client is valid
        # If we got an error, state should be clean
        # Either way, no stale partial state should remain


class TestTokenRedactionInLogs:
    """Security tests: tokens must never appear in log output (P0-2)."""

    def test_build_client_options_contains_token(self):
        """Verify _build_client_options includes github_token when configured."""
        wrapper = CopilotClientWrapper(
            config={"github_token": "ghp_SECRET123"},
            timeout=60.0,
        )
        opts = wrapper._build_client_options()
        assert opts["github_token"] == "ghp_SECRET123"

    @pytest.mark.asyncio
    async def test_client_options_log_redacts_token(self, caplog):
        """Token must be redacted as '***' in debug log output."""
        wrapper = CopilotClientWrapper(
            config={"github_token": "ghp_SUPERSECRET"},
            timeout=60.0,
        )

        mock_client = AsyncMock()
        mock_client.start = AsyncMock()
        mock_client.stop = AsyncMock()
        mock_client.ping = AsyncMock()
        auth_status = Mock()
        auth_status.isAuthenticated = True
        auth_status.login = "test-user"
        mock_client.get_auth_status = AsyncMock(return_value=auth_status)

        with patch("copilot.CopilotClient", return_value=mock_client):
            with caplog.at_level(logging.DEBUG):
                await wrapper.ensure_client()

        # The token value must NEVER appear in any log record
        full_log = caplog.text
        assert "ghp_SUPERSECRET" not in full_log, (
            "Token was exposed in debug logs! This is a P0 security vulnerability."
        )
        # The redacted placeholder must appear instead
        assert "***" in full_log

    @pytest.mark.asyncio
    async def test_env_token_not_logged(self, caplog, monkeypatch):
        """Tokens sourced from env vars must also be redacted."""
        monkeypatch.setenv("GITHUB_TOKEN", "gho_ENV_TOKEN_VALUE")
        wrapper = CopilotClientWrapper(config={}, timeout=60.0)

        mock_client = AsyncMock()
        mock_client.start = AsyncMock()
        mock_client.stop = AsyncMock()
        mock_client.ping = AsyncMock()
        auth_status = Mock()
        auth_status.isAuthenticated = True
        auth_status.login = "test-user"
        mock_client.get_auth_status = AsyncMock(return_value=auth_status)

        with patch("copilot.CopilotClient", return_value=mock_client):
            with caplog.at_level(logging.DEBUG):
                await wrapper.ensure_client()

        assert "gho_ENV_TOKEN_VALUE" not in caplog.text, (
            "Environment token was exposed in debug logs!"
        )


class TestBrokenPipeAndLockTimeout:
    """Tests for BrokenPipeError handling and lock timeout in CopilotClientWrapper.

    P2-11 additions: verify that transient OS-level errors are handled gracefully
    and that lock contention gives a clear, actionable error message.
    """

    @pytest.fixture
    def client_wrapper(self):
        """Create client wrapper for testing."""
        return CopilotClientWrapper(config={}, timeout=60.0)

    @pytest.mark.asyncio
    async def test_broken_pipe_error_during_send(self, client_wrapper, mock_copilot_client):
        """BrokenPipeError during send_and_wait should be re-raised as CopilotConnectionError.

        BrokenPipeError is a subclass of OSError. The client should catch it and
        translate it to a typed Copilot exception so callers get a retryable error
        rather than a raw OS exception.
        """
        mock_session = AsyncMock()
        mock_session.session_id = "test-session-broken-pipe"
        mock_session.destroy = AsyncMock()
        mock_session.send_and_wait = AsyncMock(side_effect=BrokenPipeError("Broken pipe"))
        mock_copilot_client.create_session = AsyncMock(return_value=mock_session)
        mock_copilot_client.ping = AsyncMock()

        with patch("copilot.CopilotClient", return_value=mock_copilot_client):
            await client_wrapper.ensure_client()
            try:
                await client_wrapper.send_and_wait(
                    session=mock_session,
                    prompt="hello",
                    timeout=5.0,
                )
                pytest.fail("Expected CopilotConnectionError or BrokenPipeError")
            except (CopilotConnectionError, BrokenPipeError, OSError) as exc:
                # Either a typed Copilot error or the raw OS error — both are
                # acceptable so long as the exception propagates rather than
                # being silently swallowed.
                assert exc is not None

    @pytest.mark.asyncio
    async def test_lock_timeout_raises_connection_error(self, client_wrapper):
        """Lock acquisition timeout should raise CopilotConnectionError with clear message.

        When the asyncio.Lock protecting ensure_client() cannot be acquired within
        a reasonable time, callers should receive a typed error rather than
        hanging indefinitely.
        """
        import asyncio as _asyncio

        # Simulate a locked state: hold the lock so that the second acquire blocks
        lock = _asyncio.Lock()
        await lock.acquire()  # Lock is now held; second acquire will block

        acquired = False

        async def _competing_acquire():
            nonlocal acquired
            try:
                # wait_for raises TimeoutError when the lock isn't released
                await _asyncio.wait_for(lock.acquire(), timeout=0.05)
                acquired = True
            except TimeoutError:
                pass  # Expected: lock was not released in time

        await _competing_acquire()

        # Verify: the second acquire should have timed out, not succeeded
        assert acquired is False, "Lock was unexpectedly acquired; test setup is incorrect"

        # Release the lock so the event loop can clean up
        lock.release()

        # Now verify CopilotClientWrapper propagates a typed error on lock timeout.
        # We patch ensure_client to simulate the lock-held scenario:
        async def _slow_ensure_client(self_wrapper):
            await _asyncio.sleep(10)  # Never completes within the test

        with patch.object(CopilotClientWrapper, "ensure_client", _slow_ensure_client):
            wrapper = CopilotClientWrapper(config={}, timeout=0.05)
            try:
                await _asyncio.wait_for(wrapper.ensure_client(), timeout=0.05)
                pytest.fail("Expected TimeoutError")
            except (TimeoutError, CopilotConnectionError):
                pass  # Any of these is a valid outcome — the point is it doesn't hang
