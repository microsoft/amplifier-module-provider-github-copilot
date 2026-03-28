"""
Tests for concurrent session race conditions.

Contract: contracts/sdk-boundary.md

Note: These tests mock SubprocessConfig as non-None to avoid triggering
the P1-6 security fix (fail-closed when token cannot be applied).
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# Mock SubprocessConfig that accepts github_token
class MockSubprocessConfig:
    """Mock SubprocessConfig that accepts github_token."""

    def __init__(self, github_token: str | None = None) -> None:
        self.github_token = github_token


class TestConcurrentSessions:
    """AC-4: Concurrent session() calls don't race on client init."""

    @pytest.mark.asyncio
    async def test_concurrent_sessions_no_race(self) -> None:
        """Concurrent session() calls don't race on client init."""
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        # Track how many times create_session is called
        create_session_count = 0
        client_start_count = 0

        mock_session = MagicMock()
        mock_session.disconnect = AsyncMock()
        mock_session.session_id = "test-session"

        mock_client_instance = AsyncMock()

        async def mock_start() -> None:
            nonlocal client_start_count
            client_start_count += 1
            # Simulate some startup delay
            await asyncio.sleep(0.01)

        mock_client_instance.start = mock_start

        async def mock_create_session(**config: Any) -> MagicMock:
            nonlocal create_session_count
            create_session_count += 1
            return mock_session

        mock_client_instance.create_session = mock_create_session

        # Patch CopilotClient and SubprocessConfig in _imports.py
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

            # Launch 5 concurrent session requests
            from amplifier_module_provider_github_copilot.sdk_adapter import SessionHandle

            async def get_session() -> SessionHandle:
                async with wrapper.session() as s:
                    return s

            sessions = await asyncio.gather(*[get_session() for _ in range(5)])

            # All should succeed without error
            assert len(sessions) == 5

            # Client should have been started exactly once (not 5 times)
            assert client_start_count == 1

            # All sessions wrap the same underlying mock session
            # (SessionHandle façade wraps raw SDK session)

            for s in sessions:
                assert isinstance(s, SessionHandle)

    @pytest.mark.asyncio
    async def test_lock_prevents_double_init(self) -> None:
        """Lock prevents double client initialization under contention."""
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        init_count = 0

        mock_session = MagicMock()
        mock_session.disconnect = AsyncMock()

        mock_client_instance = AsyncMock()

        async def mock_start() -> None:
            nonlocal init_count
            init_count += 1
            # Add delay to make race condition more likely without lock
            await asyncio.sleep(0.05)

        mock_client_instance.start = mock_start
        mock_client_instance.create_session = AsyncMock(return_value=mock_session)

        # Patch CopilotClient and SubprocessConfig in _imports.py
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

            # Create many concurrent requests
            async def create_one() -> None:
                async with wrapper.session():
                    pass

            # Run 10 concurrent session creations
            await asyncio.gather(*[create_one() for _ in range(10)])

            # Init should have happened exactly once, not 10 times
            assert init_count == 1
