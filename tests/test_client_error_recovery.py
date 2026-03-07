"""Tests for client.py error recovery paths.

These tests use fault injection to cover defensive error handling
branches that require specific failure conditions in ensure_client().

Branches covered:
- asyncio.CancelledError handling (cleanup partial client on cancellation)
- ImportError handling (SDK not installed scenario)
- CopilotAuthenticationError handling (auth failure cleanup)
- CopilotConnectionError handling (connection failure cleanup)
"""

import asyncio
import builtins
import sys

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from amplifier_module_provider_github_copilot.client import CopilotClientWrapper
from amplifier_module_provider_github_copilot.exceptions import (
    CopilotAuthenticationError,
    CopilotConnectionError,
)


class TestClientErrorRecovery:
    """Test error recovery paths in CopilotClientWrapper.ensure_client()."""

    @pytest.fixture
    def client_wrapper(self):
        """Create fresh client wrapper for each test."""
        return CopilotClientWrapper(config={}, timeout=60.0)

    @pytest.mark.asyncio
    async def test_cancelled_error_cleans_up_partial_client(self, client_wrapper):
        """asyncio.CancelledError should stop partially initialized client.

        When ensure_client() is cancelled during initialization (e.g., user
        cancels the request), any partially initialized client should be
        stopped to prevent resource leaks.

        Branch: 288->291 (asyncio.CancelledError handling)
        """
        mock_client = AsyncMock()
        mock_client.stop = AsyncMock(return_value=[])

        # Make start() raise CancelledError AFTER client is constructed
        # This simulates cancellation during initialization
        mock_client.start = AsyncMock(side_effect=asyncio.CancelledError())

        # Create a mock CopilotClient class that returns our mock instance
        mock_copilot_client_class = MagicMock(return_value=mock_client)

        mock_copilot_module = MagicMock()
        mock_copilot_module.CopilotClient = mock_copilot_client_class

        mock_types_module = MagicMock()
        mock_types_module.CopilotClientOptions = dict

        with patch.dict(
            "sys.modules",
            {
                "copilot": mock_copilot_module,
                "copilot.types": mock_types_module,
            },
        ):
            with pytest.raises(asyncio.CancelledError):
                await client_wrapper.ensure_client()

            # Verify cleanup happened - stop() called on partially initialized client
            mock_client.stop.assert_called_once()

            # Verify client wrapper state was NOT set (client never fully initialized)
            assert client_wrapper._client is None
            assert client_wrapper._started is False

    @pytest.mark.asyncio
    async def test_import_error_raises_connection_error(self, client_wrapper):
        """ImportError should raise CopilotConnectionError with install instructions.

        When the copilot SDK is not installed, ensure_client() should raise
        a CopilotConnectionError with helpful installation instructions.

        Branch: 310->315 (ImportError handling)
        """
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "copilot" or name.startswith("copilot."):
                raise ImportError("No module named 'copilot'")
            return original_import(name, *args, **kwargs)

        # Temporarily remove copilot from sys.modules and mock __import__
        modules_to_remove = [k for k in sys.modules if k == "copilot" or k.startswith("copilot.")]
        saved_modules = {k: sys.modules[k] for k in modules_to_remove}

        try:
            for k in modules_to_remove:
                del sys.modules[k]

            with patch.object(builtins, "__import__", mock_import):
                with pytest.raises(CopilotConnectionError) as exc_info:
                    await client_wrapper.ensure_client()

                # Verify error message mentions SDK installation
                assert "SDK not installed" in str(exc_info.value) or "Install with" in str(
                    exc_info.value
                )
        finally:
            # Restore modules
            sys.modules.update(saved_modules)

    @pytest.mark.asyncio
    async def test_auth_failure_cleans_up_client_state(self, client_wrapper):
        """CopilotAuthenticationError should cleanup client and reset state.

        When authentication fails, the client should be stopped and the
        wrapper's state should be reset to allow retry.

        Branch: 324->329 (CopilotAuthenticationError handling)
        """
        mock_client = AsyncMock()
        mock_client.start = AsyncMock()
        mock_client.stop = AsyncMock(return_value=[])

        # Mock get_auth_status to return unauthenticated
        mock_auth = MagicMock()
        mock_auth.isAuthenticated = False
        mock_client.get_auth_status = AsyncMock(return_value=mock_auth)

        mock_copilot_module = MagicMock()
        mock_copilot_module.CopilotClient = MagicMock(return_value=mock_client)

        mock_types_module = MagicMock()
        mock_types_module.CopilotClientOptions = dict

        with patch.dict(
            "sys.modules",
            {
                "copilot": mock_copilot_module,
                "copilot.types": mock_types_module,
            },
        ):
            with pytest.raises(CopilotAuthenticationError):
                await client_wrapper.ensure_client()

            # Verify cleanup happened - client stopped
            mock_client.stop.assert_called_once()

            # Verify state was reset to allow retry
            assert client_wrapper._client is None
            assert client_wrapper._started is False

    @pytest.mark.asyncio
    async def test_connection_failure_cleans_up_client_state(self, client_wrapper):
        """CopilotConnectionError should cleanup client and reset state.

        When a connection error occurs during initialization, the client
        should be stopped and state reset to allow retry.

        Branch: 327-328 (CopilotConnectionError handling)
        """
        mock_client = AsyncMock()
        # start() raises CopilotConnectionError to simulate connection failure
        mock_client.start = AsyncMock(
            side_effect=CopilotConnectionError("Connection refused")
        )
        mock_client.stop = AsyncMock(return_value=[])

        mock_copilot_module = MagicMock()
        mock_copilot_module.CopilotClient = MagicMock(return_value=mock_client)

        mock_types_module = MagicMock()
        mock_types_module.CopilotClientOptions = dict

        with patch.dict(
            "sys.modules",
            {
                "copilot": mock_copilot_module,
                "copilot.types": mock_types_module,
            },
        ):
            with pytest.raises(CopilotConnectionError) as exc_info:
                await client_wrapper.ensure_client()

            assert "Connection refused" in str(exc_info.value)

            # Verify cleanup happened
            mock_client.stop.assert_called_once()

            # Verify state was reset
            assert client_wrapper._client is None
            assert client_wrapper._started is False


class TestClientErrorRecoveryEdgeCases:
    """Edge cases for error recovery to ensure robust cleanup."""

    @pytest.fixture
    def client_wrapper(self):
        """Create fresh client wrapper for each test."""
        return CopilotClientWrapper(config={}, timeout=60.0)

    @pytest.mark.asyncio
    async def test_cancelled_error_swallows_stop_exception(self, client_wrapper):
        """CancelledError handler should swallow exceptions from stop().

        Even if client.stop() fails, the CancelledError should still propagate.

        Branch: 288->291 (exception swallowing in cleanup)
        """
        mock_client = AsyncMock()
        mock_client.start = AsyncMock(side_effect=asyncio.CancelledError())
        # stop() raises an exception
        mock_client.stop = AsyncMock(side_effect=RuntimeError("Stop failed"))

        mock_copilot_module = MagicMock()
        mock_copilot_module.CopilotClient = MagicMock(return_value=mock_client)

        mock_types_module = MagicMock()
        mock_types_module.CopilotClientOptions = dict

        with patch.dict(
            "sys.modules",
            {
                "copilot": mock_copilot_module,
                "copilot.types": mock_types_module,
            },
        ):
            # CancelledError should propagate, not RuntimeError from stop()
            with pytest.raises(asyncio.CancelledError):
                await client_wrapper.ensure_client()

            # stop() was still attempted
            mock_client.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_auth_error_swallows_stop_exception(self, client_wrapper):
        """Auth error handler should swallow exceptions from stop().

        Branch: 324->329 (exception swallowing in auth error cleanup)
        """
        mock_client = AsyncMock()
        mock_client.start = AsyncMock()
        # stop() raises an exception
        mock_client.stop = AsyncMock(side_effect=RuntimeError("Stop failed"))

        # Mock unauthenticated state
        mock_auth = MagicMock()
        mock_auth.isAuthenticated = False
        mock_client.get_auth_status = AsyncMock(return_value=mock_auth)

        mock_copilot_module = MagicMock()
        mock_copilot_module.CopilotClient = MagicMock(return_value=mock_client)

        mock_types_module = MagicMock()
        mock_types_module.CopilotClientOptions = dict

        with patch.dict(
            "sys.modules",
            {
                "copilot": mock_copilot_module,
                "copilot.types": mock_types_module,
            },
        ):
            # Auth error should propagate, not RuntimeError from stop()
            with pytest.raises(CopilotAuthenticationError):
                await client_wrapper.ensure_client()

            # stop() was still attempted
            mock_client.stop.assert_called_once()

            # State should still be reset despite stop() failure
            assert client_wrapper._client is None
            assert client_wrapper._started is False
