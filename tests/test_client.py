"""
Tests for CopilotClientWrapper.

This module tests the client wrapper including lifecycle management,
error handling, session creation, input validation, and cancellation handling.
"""

import asyncio
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
