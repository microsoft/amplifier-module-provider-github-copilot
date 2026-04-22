"""
Contract Compliance Tests: Ephemeral Sessions.

Contract: contracts/deny-destroy.md

Test Anchors:
- deny-destroy:Ephemeral:MUST:1 — New session per complete() call
- deny-destroy:Ephemeral:MUST:2 — Session destroyed after first turn
- deny-destroy:Ephemeral:MUST:3 — No session reuse
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest


class MockSDKSession:
    """Test double for SDK session with disconnect tracking."""

    def __init__(self, session_id: str | None = None) -> None:
        if session_id is None:
            session_id = f"ephemeral-session-{uuid4().hex[:8]}"
        self.session_id = session_id
        self.disconnected = False

    async def disconnect(self) -> None:
        """Mark session as disconnected."""
        self.disconnected = True


class TestNewSessionPerCall:
    """deny-destroy:Ephemeral:MUST:1 — New session per complete() call."""

    @pytest.mark.asyncio
    async def test_each_session_call_creates_new_session(self) -> None:
        """deny-destroy:Ephemeral:MUST:1 — Each session() call invokes create_session.

        CopilotClientWrapper.session() must create a fresh SDK session
        on every invocation, never reusing a previous session.
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import CopilotClientWrapper

        mock_sdk_client = AsyncMock()
        create_session_call_count = 0

        async def track_create_session(**kwargs: Any) -> MockSDKSession:
            nonlocal create_session_call_count
            create_session_call_count += 1
            return MockSDKSession()

        mock_sdk_client.create_session = AsyncMock(side_effect=track_create_session)

        wrapper = CopilotClientWrapper(sdk_client=mock_sdk_client)

        # First session call
        async with wrapper.session(model="gpt-4"):
            pass

        assert create_session_call_count == 1, "Ephemeral:MUST:1 — first call should create session"

        # Second session call
        async with wrapper.session(model="gpt-4"):
            pass

        assert create_session_call_count == 2, (
            "Ephemeral:MUST:1 — second call must create NEW session, not reuse"
        )

    @pytest.mark.asyncio
    async def test_three_session_calls_create_three_sessions(self) -> None:
        """deny-destroy:Ephemeral:MUST:1 — N session() calls = N create_session calls."""
        from amplifier_module_provider_github_copilot.sdk_adapter.client import CopilotClientWrapper

        mock_sdk_client = AsyncMock()
        sessions_created: list[MockSDKSession] = []

        async def track_create_session(**kwargs: Any) -> MockSDKSession:
            session = MockSDKSession()
            sessions_created.append(session)
            return session

        mock_sdk_client.create_session = AsyncMock(side_effect=track_create_session)

        wrapper = CopilotClientWrapper(sdk_client=mock_sdk_client)

        # Create 3 sessions
        for _ in range(3):
            async with wrapper.session(model="gpt-4"):
                pass

        assert len(sessions_created) == 3, (
            f"Ephemeral:MUST:1 — 3 calls must create 3 sessions, got {len(sessions_created)}"
        )


class TestSessionDestroyedAfterTurn:
    """deny-destroy:Ephemeral:MUST:2 — Session destroyed after first turn."""

    @pytest.mark.asyncio
    async def test_session_disconnected_on_context_exit(self) -> None:
        """deny-destroy:Ephemeral:MUST:2 — disconnect() called when exiting context.

        After the async with block exits (normally), the session's
        disconnect() method must be invoked.
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import CopilotClientWrapper

        mock_sdk_client = AsyncMock()
        created_session: MockSDKSession | None = None

        async def track_create_session(**kwargs: Any) -> MockSDKSession:
            nonlocal created_session
            created_session = MockSDKSession()
            return created_session

        mock_sdk_client.create_session = AsyncMock(side_effect=track_create_session)

        wrapper = CopilotClientWrapper(sdk_client=mock_sdk_client)

        async with wrapper.session(model="gpt-4") as session:
            # Session should exist but not be disconnected yet
            assert isinstance(created_session, MockSDKSession)
            assert created_session.disconnected is False, (
                "Ephemeral:MUST:2 — session should not be disconnected inside context"
            )
            _ = session.session_id  # Use session

        # After exit, session must be disconnected
        assert isinstance(created_session, MockSDKSession)
        assert created_session.disconnected is True, (
            "Ephemeral:MUST:2 — session MUST be disconnected after context exit"
        )

    @pytest.mark.asyncio
    async def test_session_disconnected_on_exception(self) -> None:
        """deny-destroy:Ephemeral:MUST:2 — disconnect() called even on exception.

        If an exception occurs inside the session context, the session
        must still be destroyed in the finally block.
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import CopilotClientWrapper

        mock_sdk_client = AsyncMock()
        created_session: MockSDKSession | None = None

        async def track_create_session(**kwargs: Any) -> MockSDKSession:
            nonlocal created_session
            created_session = MockSDKSession()
            return created_session

        mock_sdk_client.create_session = AsyncMock(side_effect=track_create_session)

        wrapper = CopilotClientWrapper(sdk_client=mock_sdk_client)

        with pytest.raises(RuntimeError, match="test error"):
            async with wrapper.session(model="gpt-4"):
                raise RuntimeError("test error")

        # Session must still be disconnected despite exception
        assert isinstance(created_session, MockSDKSession)
        assert created_session.disconnected is True, (
            "Ephemeral:MUST:2 — session MUST be disconnected even when exception raised"
        )


class TestNoSessionReuse:
    """deny-destroy:Ephemeral:MUST:3 — No session reuse."""

    @pytest.mark.asyncio
    async def test_two_sessions_have_different_objects(self) -> None:
        """deny-destroy:Ephemeral:MUST:3 — Two sessions are distinct objects.

        Each session() call must return a wrapper around a different
        underlying SDK session object.
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import CopilotClientWrapper

        mock_sdk_client = AsyncMock()
        sessions_created: list[MockSDKSession] = []

        async def track_create_session(**kwargs: Any) -> MockSDKSession:
            session = MockSDKSession()
            sessions_created.append(session)
            return session

        mock_sdk_client.create_session = AsyncMock(side_effect=track_create_session)

        wrapper = CopilotClientWrapper(sdk_client=mock_sdk_client)

        async with wrapper.session(model="gpt-4"):
            pass

        async with wrapper.session(model="gpt-4"):
            pass

        assert len(sessions_created) == 2
        assert sessions_created[0] is not sessions_created[1], (
            "Ephemeral:MUST:3 — sessions must be distinct objects, not reused"
        )

    @pytest.mark.asyncio
    async def test_session_ids_are_unique(self) -> None:
        """deny-destroy:Ephemeral:MUST:3 — Session IDs are unique.

        Each session created by session() must have a unique session_id.
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import CopilotClientWrapper

        mock_sdk_client = AsyncMock()
        session_ids: list[str] = []

        async def track_create_session(**kwargs: Any) -> MockSDKSession:
            session = MockSDKSession()
            session_ids.append(session.session_id)
            return session

        mock_sdk_client.create_session = AsyncMock(side_effect=track_create_session)

        wrapper = CopilotClientWrapper(sdk_client=mock_sdk_client)

        # Create 5 sessions
        for _ in range(5):
            async with wrapper.session(model="gpt-4"):
                pass

        assert len(session_ids) == 5
        assert len(set(session_ids)) == 5, (
            f"Ephemeral:MUST:3 — all session IDs must be unique, got duplicates in {session_ids}"
        )

    @pytest.mark.asyncio
    async def test_closed_session_not_reused(self) -> None:
        """deny-destroy:Ephemeral:MUST:3 — After disconnect, session not reused.

        Once a session is disconnected, subsequent session() calls
        must not return the same session object.
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import CopilotClientWrapper

        mock_sdk_client = AsyncMock()
        first_session: MockSDKSession | None = None
        second_session: MockSDKSession | None = None

        call_count = 0

        async def track_create_session(**kwargs: Any) -> MockSDKSession:
            nonlocal call_count, first_session, second_session
            session = MockSDKSession()
            call_count += 1
            if call_count == 1:
                first_session = session
            elif call_count == 2:
                second_session = session
            return session

        mock_sdk_client.create_session = AsyncMock(side_effect=track_create_session)

        wrapper = CopilotClientWrapper(sdk_client=mock_sdk_client)

        # First session — will be disconnected
        async with wrapper.session(model="gpt-4"):
            pass

        assert isinstance(first_session, MockSDKSession)
        assert first_session.disconnected is True

        # Second session — must be a new one
        async with wrapper.session(model="gpt-4"):
            pass

        assert isinstance(second_session, MockSDKSession)
        assert first_session is not second_session, (
            "Ephemeral:MUST:3 — disconnected session must not be reused"
        )
