"""Tests for centralized SDK test mocks.

Contract Reference: sdk-boundary:TypeTranslation:MUST:1 — SDK types must be translated at boundary.
Tests should use realistic shapes to verify translation works.
"""

from typing import Any

import pytest


class TestSessionEventShape:
    """Verify SessionEvent matches SDK shape."""

    def test_session_event_has_type_enum(self) -> None:
        """AC-6: SessionEvent.type is enum with .value attribute.

        Contract: sdk-boundary:TypeTranslation:MUST:1
        """
        from tests.fixtures.sdk_mocks import SessionEvent, SessionEventType

        event = SessionEvent(type=SessionEventType.SESSION_IDLE)
        assert event.type.value == "session.idle"
        assert isinstance(event.type, SessionEventType)

    def test_session_event_type_value_is_string(self) -> None:
        """SessionEventType enum values are SDK-format strings.

        Contract: event-vocabulary:BRIDGE:session.idle
        """
        from tests.fixtures.sdk_mocks import SessionEventType

        assert SessionEventType.SESSION_IDLE.value == "session.idle"
        assert SessionEventType.ASSISTANT_MESSAGE_DELTA.value == "assistant.message_delta"

    def test_session_event_has_data_field(self) -> None:
        """SessionEvent has data field matching SDK shape.

        Contract: sdk-boundary:TypeTranslation:MUST:1
        """
        from tests.fixtures.sdk_mocks import SessionEvent, SessionEventData, SessionEventType

        event = SessionEvent(
            type=SessionEventType.ASSISTANT_MESSAGE_DELTA,
            data=SessionEventData(text="hello"),
        )
        assert event.data.text == "hello"

    def test_session_event_has_id_and_timestamp(self) -> None:
        """SessionEvent has id and timestamp fields.

        Contract: sdk-boundary:TypeTranslation:MUST:1
        """
        from datetime import datetime
        from uuid import UUID

        from tests.fixtures.sdk_mocks import SessionEvent, SessionEventType

        event = SessionEvent(type=SessionEventType.SESSION_IDLE)
        assert isinstance(event.id, UUID)
        assert isinstance(event.timestamp, datetime)


class TestMockSDKSessionBehavior:
    """Verify MockSDKSession delivers events correctly."""

    @pytest.mark.asyncio
    async def test_mock_session_delivers_session_events(self) -> None:
        """Handler receives SessionEvent objects, not dicts.

        Contract: sdk-boundary:TypeTranslation:MUST:2
        """
        from tests.fixtures.sdk_mocks import MockSDKSession, SessionEvent, SessionEventType

        events = [
            SessionEvent(type=SessionEventType.ASSISTANT_MESSAGE_DELTA),
        ]
        session = MockSDKSession(events=events)

        received: list[Any] = []
        session.on(lambda e: received.append(e))

        # SDK v0.2.0: send(prompt, attachments=...)
        await session.send("test")

        # Should receive SessionEvent objects
        assert len(received) >= 1
        assert isinstance(received[0], SessionEvent)
        # Contract: event-vocabulary:Bridge:MUST:1
        assert received[0].type.value == "assistant.message_delta"

    @pytest.mark.asyncio
    async def test_mock_session_accepts_legacy_dicts(self) -> None:
        """Backward compat: accepts dict events, converts to SessionEvent.

        Contract: sdk-boundary:TypeTranslation:SHOULD:1
        """
        from tests.fixtures.sdk_mocks import MockSDKSession, SessionEvent

        # Legacy dict format (backward compatibility)
        events = [
            {"type": "assistant.message_delta", "text": "hello"},
        ]
        session = MockSDKSession(events=events)

        received: list[Any] = []
        session.on(lambda e: received.append(e))

        # SDK v0.2.0: send(prompt, attachments=...)
        await session.send("test")

        # Should still deliver SessionEvent objects (converted)
        assert len(received) >= 1
        assert isinstance(received[0], SessionEvent)

    @pytest.mark.asyncio
    async def test_mock_session_auto_sends_idle_event(self) -> None:
        """MockSDKSession auto-sends session.idle to signal completion.

        Contract: event-vocabulary:BRIDGE:session.idle
        """
        from tests.fixtures.sdk_mocks import MockSDKSession, SessionEvent, SessionEventType

        session = MockSDKSession(events=[])

        received: list[Any] = []
        session.on(lambda e: received.append(e))

        # SDK v0.2.0: send(prompt, attachments=...)
        await session.send("test")

        # Last event should be session.idle
        assert len(received) == 1
        idle_event: Any = received[-1]
        assert isinstance(idle_event, SessionEvent)
        assert idle_event.type == SessionEventType.SESSION_IDLE

    def test_mock_session_on_returns_unsubscribe(self) -> None:
        """on() returns unsubscribe function per SDK API.

        Contract: sdk-boundary:TypeTranslation:MUST:1
        """
        from tests.fixtures.sdk_mocks import MockSDKSession

        session = MockSDKSession(events=[])
        unsubscribe = session.on(lambda e: None)

        assert callable(unsubscribe)
        unsubscribe()  # Should not raise


class TestEventFactories:
    """Verify factory helpers create correct shapes."""

    def test_idle_event_factory(self) -> None:
        """idle_event() returns correct shape.

        Contract: event-vocabulary:BRIDGE:session.idle
        """
        from tests.fixtures.sdk_mocks import SessionEvent, SessionEventType, idle_event

        event = idle_event()
        assert isinstance(event, SessionEvent)
        assert event.type == SessionEventType.SESSION_IDLE
        assert event.type.value == "session.idle"

    def test_text_delta_event_factory(self) -> None:
        """text_delta_event() returns correct shape with delta_content.

        Contract: event-vocabulary:BRIDGE:text_delta
        Contract: sdk-boundary:EventShape:MUST:2
        SDK v0.1.33+ uses data.delta_content for streaming text.
        """
        from tests.fixtures.sdk_mocks import SessionEvent, text_delta_event

        event = text_delta_event("hello")
        assert isinstance(event, SessionEvent)
        assert event.data.delta_content == "hello"

    def test_message_complete_event_factory(self) -> None:
        """message_complete_event() returns correct shape with finish_reason.

        Contract: event-vocabulary:BRIDGE:message_complete
        """
        from tests.fixtures.sdk_mocks import SessionEvent, message_complete_event

        event = message_complete_event("stop")
        assert isinstance(event, SessionEvent)
        assert event.data.finish_reason == "stop"


# TestCentralizationVerification removed - test_integration.py was deleted
# as part of the completion.py removal in Issue #6
