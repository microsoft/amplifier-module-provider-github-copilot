"""
Ephemeral Session Invariant Tests.

Contract: contracts/deny-destroy.md

Tests verify the deny-destroy contract's ephemeral session requirements:
- Each session MUST be ephemeral — no session reuse
- Session state is isolated between sessions
- Session IDs are unique per session

Test Anchors:
- deny-destroy:Ephemeral:MUST:1 — New session per complete() call
- deny-destroy:Ephemeral:MUST:2 — Session destroyed after first turn
- deny-destroy:Ephemeral:MUST:3 — No session reuse
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# Import for MagicMock spec= parameter (runtime import required)
from copilot import CopilotClient


class MockSDKSession:
    """Test double for SDK session with unique ID tracking."""

    _session_counter: int = 0

    def __init__(self, session_id: str | None = None) -> None:
        if session_id is None:
            MockSDKSession._session_counter += 1
            session_id = f"session_{MockSDKSession._session_counter}"
        self.session_id = session_id
        self.disconnected = False  # Public for test assertions
        self._internal_state: dict[str, Any] = {}

    async def disconnect(self) -> None:
        self.disconnected = True

    def set_state(self, key: str, value: Any) -> None:
        """Set internal state for isolation testing."""
        self._internal_state[key] = value

    def get_state(self, key: str) -> Any:
        """Get internal state for isolation testing."""
        return self._internal_state.get(key)


class TestSessionsAreDistinct:
    """deny-destroy:Ephemeral:MUST:3 — No session reuse.

    Each session() call must produce a distinct session with unique ID.
    """

    @pytest.mark.asyncio
    async def test_two_sessions_have_distinct_ids(self) -> None:
        """deny-destroy:Ephemeral:MUST:3 — Sessions must have distinct IDs.

        Two successive session() calls must produce sessions with different IDs.
        This verifies no session object is reused.
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        # Create mock SDK client that produces sessions with unique IDs
        mock_client = MagicMock(spec=CopilotClient)
        session_ids: list[str] = []

        async def create_session_mock(**kwargs: Any) -> MockSDKSession:
            session = MockSDKSession()
            session_ids.append(session.session_id)
            return session

        mock_client.create_session = AsyncMock(side_effect=create_session_mock)

        wrapper = CopilotClientWrapper(sdk_client=mock_client)

        # Create first session
        async with wrapper.session(model="gpt-4") as session1:
            id1 = session1.session_id

        # Create second session
        async with wrapper.session(model="gpt-4") as session2:
            id2 = session2.session_id

        # Sessions must have distinct IDs
        assert id1 != id2, f"Sessions must have distinct IDs, got: {id1} == {id2}"
        assert len(session_ids) == 2, "Two sessions should have been created"

    @pytest.mark.asyncio
    async def test_session_ids_are_not_reused_across_many_sessions(self) -> None:
        """Contract: deny-destroy:Ephemeral:MUST:3 — Session IDs never reuse.

        Even across many sessions, no ID should repeat.
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        mock_client = MagicMock(spec=CopilotClient)
        session_ids: list[str] = []

        async def create_session_mock(**kwargs: Any) -> MockSDKSession:
            session = MockSDKSession()
            session_ids.append(session.session_id)
            return session

        mock_client.create_session = AsyncMock(side_effect=create_session_mock)

        wrapper = CopilotClientWrapper(sdk_client=mock_client)

        # Create 5 sessions
        for _ in range(5):
            async with wrapper.session(model="gpt-4"):
                pass

        # All IDs should be unique
        assert len(session_ids) == 5
        assert len(set(session_ids)) == 5, f"Duplicate session IDs found: {session_ids}"


class TestSessionStateIsolation:
    """deny-destroy:Ephemeral:MUST:1 — New session per complete() call.

    Session state must not leak between sessions.
    """

    @pytest.mark.asyncio
    async def test_session_state_is_isolated(self) -> None:
        """deny-destroy:Ephemeral:MUST:1 — Session state does not persist.

        State set in session 1 must not be visible in session 2.
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        mock_client = MagicMock(spec=CopilotClient)
        created_sessions: list[MockSDKSession] = []

        async def create_session_mock(**kwargs: Any) -> MockSDKSession:
            session = MockSDKSession()
            created_sessions.append(session)
            return session

        mock_client.create_session = AsyncMock(side_effect=create_session_mock)

        wrapper = CopilotClientWrapper(sdk_client=mock_client)

        # Session 1: Set state on first session
        async with wrapper.session(model="gpt-4") as session1:
            _ = session1.session_id  # Access to ensure it exists

        # Set state on the underlying session AFTER it's created
        created_sessions[0].set_state("secret_key", "secret_value")

        # Session 2: Create a second session
        async with wrapper.session(model="gpt-4") as session2:
            _ = session2.session_id  # Access to ensure it exists

        # Two separate underlying sessions were created
        assert len(created_sessions) == 2, "Should create separate sessions"

        # Sessions are distinct objects
        assert created_sessions[0] is not created_sessions[1], "Sessions must be distinct objects"
        assert id(created_sessions[0]) != id(created_sessions[1]), (
            "Sessions must have different identities"
        )

        # Sessions have distinct IDs
        assert created_sessions[0].session_id != created_sessions[1].session_id, (
            f"Sessions must have distinct IDs: {created_sessions[0].session_id} vs "
            f"{created_sessions[1].session_id}"
        )

        # State set on session 0 is NOT visible in session 1
        assert created_sessions[0].get_state("secret_key") == "secret_value", (
            "Session 0 should have the state we set"
        )
        assert created_sessions[1].get_state("secret_key") is None, (
            "Session 1 must not see state from session 0 — isolation violated"
        )


class TestSessionDestruction:
    """deny-destroy:Ephemeral:MUST:2 — Session destroyed after first turn."""

    @pytest.mark.asyncio
    async def test_session_disconnected_on_exit(self) -> None:
        """deny-destroy:Ephemeral:MUST:2 — Session disconnect called on exit.

        When context manager exits, session.disconnect() must be called.
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        mock_client = MagicMock(spec=CopilotClient)
        created_sessions: list[MockSDKSession] = []

        async def create_session_mock(**kwargs: Any) -> MockSDKSession:
            session = MockSDKSession()
            created_sessions.append(session)
            return session

        mock_client.create_session = AsyncMock(side_effect=create_session_mock)

        wrapper = CopilotClientWrapper(sdk_client=mock_client)

        async with wrapper.session(model="gpt-4") as session:
            # Session is wrapped in SessionHandle
            assert isinstance(session.session_id, str), "session_id must be a string"
            assert len(session.session_id) > 0, "session_id must be non-empty"

        # After exiting context, the underlying session should be disconnected
        assert len(created_sessions) == 1
        assert created_sessions[0].disconnected, "Session must be disconnected on exit"

    @pytest.mark.asyncio
    async def test_session_disconnected_on_exception(self) -> None:
        """deny-destroy:Ephemeral:MUST:2 — Session disconnect even on error.

        Even if an exception occurs, session must still be destroyed.
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        mock_client = MagicMock(spec=CopilotClient)
        created_sessions: list[MockSDKSession] = []

        async def create_session_mock(**kwargs: Any) -> MockSDKSession:
            session = MockSDKSession()
            created_sessions.append(session)
            return session

        mock_client.create_session = AsyncMock(side_effect=create_session_mock)

        wrapper = CopilotClientWrapper(sdk_client=mock_client)

        with pytest.raises(ValueError, match="test error"):
            async with wrapper.session(model="gpt-4"):
                raise ValueError("test error")

        # Session should still be disconnected despite exception
        assert len(created_sessions) == 1
        assert created_sessions[0].disconnected, (
            "Session must be disconnected even when exception occurs"
        )


class TestNoSessionAccumulation:
    """deny-destroy:Ephemeral:MUST:3 — No state accumulation.

    The wrapper itself must not accumulate session references.
    """

    @pytest.mark.asyncio
    async def test_wrapper_does_not_retain_session_references(self) -> None:
        """deny-destroy:Ephemeral:MUST:3 — Wrapper doesn't hold session refs.

        After session exits, the wrapper should not retain any reference
        to the closed session.
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        mock_client = MagicMock(spec=CopilotClient)

        async def create_session_mock(**kwargs: Any) -> MockSDKSession:
            return MockSDKSession()

        mock_client.create_session = AsyncMock(side_effect=create_session_mock)

        wrapper = CopilotClientWrapper(sdk_client=mock_client)

        # Verify wrapper has no session-related instance attributes that persist
        # (beyond the injected client which is expected)
        initial_attrs = set(vars(wrapper).keys())

        async with wrapper.session(model="gpt-4"):
            pass

        # After session exits, no new persistent session state should remain
        final_attrs = set(vars(wrapper).keys())

        # The only new attributes should be caching-related (error_config)
        # NOT session references
        new_attrs = final_attrs - initial_attrs
        forbidden_session_attrs = {
            attr for attr in new_attrs if "session" in attr.lower() and attr != "_error_config"
        }

        assert not forbidden_session_attrs, (
            f"Wrapper accumulated session-related attributes: {forbidden_session_attrs}"
        )


class TestConcurrentSessionIsolation:
    """deny-destroy:Ephemeral:MUST:3 — Concurrent sessions are isolated."""

    @pytest.mark.asyncio
    async def test_concurrent_sessions_are_independent(self) -> None:
        """deny-destroy:Ephemeral:MUST:3 — Parallel sessions don't interfere.

        Two sessions running concurrently should have distinct IDs
        and not share state.
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        mock_client = MagicMock(spec=CopilotClient)
        created_sessions: list[MockSDKSession] = []
        session_lock = asyncio.Lock()

        async def create_session_mock(**kwargs: Any) -> MockSDKSession:
            async with session_lock:
                session = MockSDKSession()
                created_sessions.append(session)
                return session

        mock_client.create_session = AsyncMock(side_effect=create_session_mock)

        wrapper = CopilotClientWrapper(sdk_client=mock_client)

        session_ids: list[str] = []
        session_lock_for_ids = asyncio.Lock()

        async def use_session() -> None:
            async with wrapper.session(model="gpt-4") as session:
                async with session_lock_for_ids:
                    session_ids.append(session.session_id)
                # Simulate some work
                await asyncio.sleep(0.01)

        # Run two sessions concurrently
        await asyncio.gather(use_session(), use_session())

        # Both sessions should have distinct IDs
        assert len(session_ids) == 2
        assert session_ids[0] != session_ids[1], (
            f"Concurrent sessions must have distinct IDs: {session_ids}"
        )

        # Both should be disconnected
        assert all(s.disconnected for s in created_sessions)
