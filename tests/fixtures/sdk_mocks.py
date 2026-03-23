"""Centralized SDK mock classes for testing.

Single source of truth for SDK session mocking.
Contract Reference: sdk-boundary:TypeTranslation:MUST:1 — SDK types must be translated at boundary.

This module provides mock implementations matching the real SDK shapes:
- SessionEventType: Enum matching SDK session event types
- SessionEventData: Dataclass for event payloads
- SessionEvent: Dataclass matching SDK SessionEvent shape
- MockSDKSession: Mock session with on() + send() API pattern

Factory helpers for common event types are also provided.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4


class SessionEventType(Enum):
    """Mock of SDK SessionEventType enum.

    Values match SDK SessionEventType from github-copilot-sdk.
    """

    SESSION_IDLE = "session.idle"
    SESSION_ERROR = "session.error"
    SESSION_START = "session.start"
    ASSISTANT_MESSAGE = "assistant.message"
    ASSISTANT_MESSAGE_DELTA = "assistant.message_delta"
    ASSISTANT_REASONING_DELTA = "assistant.reasoning_delta"
    ASSISTANT_USAGE = "assistant.usage"
    TOOL_EXECUTION_COMPLETE = "tool.execution_complete"
    # Legacy aliases for test compatibility
    MESSAGE_COMPLETE = "message_complete"
    TOOL_USE_COMPLETE = "tool_use_complete"
    USAGE_UPDATE = "usage_update"


@dataclass
class SessionEventData:
    """Mock SDK event data payload.

    Contract: sdk-boundary:EventShape:MUST:2
    Reference: SDK SessionEvent.data structure from github-copilot-sdk

    SDK v0.1.33+ field mapping:
    - delta_content: Streaming text chunks (assistant.message_delta)
    - content: Complete message text (assistant.message)
    - reasoning_text: Extended thinking text
    - text: DEPRECATED - kept for backward compatibility in test dict conversion
    - tool_call_id/tool_name/arguments: Tool execution events
    """

    # SDK v0.1.33+ fields
    delta_content: str | None = None
    content: str | None = None
    reasoning_text: str | None = None
    message_id: str | None = None
    reasoning_id: str | None = None

    # Legacy field for backward compat with dict-based tests
    text: str | None = None

    # Tool call fields (SDK uses tool_call_id, tool_name)
    tool_call_id: str | None = None
    tool_name: str | None = None
    arguments: dict[str, Any] | None = None

    # Other common fields
    message: str | None = None
    finish_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    tool_requests: list[dict[str, Any]] | None = None


@dataclass
class SessionEvent:
    """Mock of SDK SessionEvent matching production shape.

    Reference: copilot-sdk/python/copilot/generated/session_events.py:2697
    """

    type: SessionEventType
    data: SessionEventData = field(default_factory=SessionEventData)
    id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.now)


# Mapping from legacy string types to SessionEventType
_LEGACY_TYPE_MAP: dict[str, SessionEventType] = {
    "session.idle": SessionEventType.SESSION_IDLE,
    "SESSION_IDLE": SessionEventType.SESSION_IDLE,
    "session.error": SessionEventType.SESSION_ERROR,
    "session.start": SessionEventType.SESSION_START,
    "assistant.message": SessionEventType.ASSISTANT_MESSAGE,
    "assistant.message_delta": SessionEventType.ASSISTANT_MESSAGE_DELTA,
    "assistant.reasoning_delta": SessionEventType.ASSISTANT_REASONING_DELTA,
    "assistant.usage": SessionEventType.ASSISTANT_USAGE,
    "tool.execution_complete": SessionEventType.TOOL_EXECUTION_COMPLETE,
    # Legacy underscore aliases used by tests
    "message_complete": SessionEventType.MESSAGE_COMPLETE,
    "tool_use_complete": SessionEventType.TOOL_USE_COMPLETE,
    "usage_update": SessionEventType.USAGE_UPDATE,
}


class MockSDKSession:
    """Centralized mock SDK session for all tests.

    Uses correct SDK API pattern: on() + send()
    Single source of truth for SDK session mocking.
    Iteration 2: Force cache invalidation.

    Accepts both SessionEvent objects and legacy dicts for backward compatibility.
    Handlers always receive SessionEvent objects (dicts are converted).
    """

    def __init__(
        self,
        events: Sequence[SessionEvent | dict[str, Any]] | None = None,
        *,
        raise_on_send: Exception | None = None,
    ) -> None:
        """Initialize mock session.

        Args:
            events: Sequence of SDK events to yield (SessionEvent or dict).
            raise_on_send: Exception to raise during send().
        """
        self.events: Sequence[SessionEvent | dict[str, Any]] = events or []
        self.raise_on_send = raise_on_send
        self.destroyed = False
        self.disconnected = False  # Alias for destroyed (backward compat)
        self.deny_hook_installed = False
        # SDK v0.2.0: Track send() calls for test assertions
        self.last_prompt: str | None = None
        self.last_attachments: list[dict[str, Any]] | None = None
        # Handler accepts Any since we pass both dicts and SessionEvent
        self._handlers: list[Callable[[Any], None]] = []

    def on(self, handler: Callable[[SessionEvent], None]) -> Callable[[], None]:
        """Register event handler (correct API).

        Args:
            handler: Callback that receives SessionEvent objects.

        Returns:
            Unsubscribe function.
        """
        self._handlers.append(handler)
        self.deny_hook_installed = True  # Consider any handler registration as "hook installed"

        def unsubscribe() -> None:
            if handler in self._handlers:
                self._handlers.remove(handler)

        return unsubscribe

    async def send(
        self,
        prompt: str,
        *,
        attachments: list[dict[str, Any]] | None = None,
    ) -> str:
        """Send message and trigger events via handlers.

        SDK v0.2.0 API: send(prompt, attachments=...) replaces send({"prompt": ...})

        Args:
            prompt: The message prompt text.
            attachments: Optional list of attachments (e.g., BlobAttachment for images).

        Returns:
            Message ID string.

        Raises:
            Exception configured in raise_on_send.
        """
        if self.raise_on_send:
            raise self.raise_on_send

        # Store for test inspection
        self.last_prompt = prompt
        self.last_attachments = attachments

        # Deliver events to all registered handlers
        # Convert dict events to SessionEvent for proper SDK simulation
        for event in self.events:
            converted = self._to_session_event(event)
            for handler in self._handlers:
                handler(converted)

        # Auto-send SESSION_IDLE to signal completion (prevents timeout)
        idle = SessionEvent(type=SessionEventType.SESSION_IDLE)
        for handler in self._handlers:
            handler(idle)

        return "message-id"

    def _to_session_event(self, event: SessionEvent | dict[str, Any]) -> SessionEvent:
        """Convert dict to SessionEvent if needed.

        Contract: sdk-boundary:EventShape:MUST:3
        Converts legacy dict events to proper SDK structure.

        Field mapping for legacy test dicts:
        - text -> delta_content (SDK v0.1.33+)
        - id -> tool_call_id (tool events)
        - name -> tool_name (tool events)
        """
        if isinstance(event, SessionEvent):
            return event
        # Convert dict to SessionEvent
        type_str = event.get("type", "session.idle")
        event_type = _LEGACY_TYPE_MAP.get(type_str, SessionEventType.SESSION_IDLE)

        # SDK v0.1.33+: text deltas use delta_content, not text
        # Check for both to support legacy test dicts
        delta_content = event.get("delta_content") or event.get("text")
        tool_requests = event.get("tool_requests")

        # Map legacy field names to SDK field names for tool events
        # Legacy tests use "id" and "name", SDK uses "tool_call_id" and "tool_name"
        tool_call_id = event.get("tool_call_id") or event.get("id")
        tool_name = event.get("tool_name") or event.get("name")
        arguments = event.get("arguments")

        data = SessionEventData(
            delta_content=delta_content,
            content=event.get("content"),
            reasoning_text=event.get("reasoning_text"),
            message_id=event.get("message_id"),
            text=event.get("text"),  # Keep for backward compat
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            arguments=arguments,
            message=event.get("message"),
            finish_reason=event.get("finish_reason"),
            input_tokens=event.get("input_tokens"),
            output_tokens=event.get("output_tokens"),
            total_tokens=event.get("total_tokens"),
            tool_requests=tool_requests,
        )
        return SessionEvent(type=event_type, data=data)

    async def disconnect(self) -> None:
        """Mark session as destroyed/disconnected."""
        self.destroyed = True
        self.disconnected = True  # Alias for backward compat


# Factory helpers for common event types


def idle_event() -> SessionEvent:
    """Create session.idle event matching SDK format."""
    return SessionEvent(type=SessionEventType.SESSION_IDLE)


def text_delta_event(text: str) -> SessionEvent:
    """Create text delta event.

    Contract: sdk-boundary:EventShape:MUST:2
    SDK v0.1.33+ uses data.delta_content for streaming text chunks.
    """
    return SessionEvent(
        type=SessionEventType.ASSISTANT_MESSAGE_DELTA,
        data=SessionEventData(delta_content=text),
    )


def message_complete_event(finish_reason: str = "stop") -> SessionEvent:
    """Create message complete event."""
    return SessionEvent(
        type=SessionEventType.ASSISTANT_MESSAGE,
        data=SessionEventData(finish_reason=finish_reason),
    )


def error_event(message: str) -> SessionEvent:
    """Create error event."""
    return SessionEvent(
        type=SessionEventType.SESSION_ERROR,
        data=SessionEventData(message=message),
    )


def usage_event(
    input_tokens: int = 0,
    output_tokens: int = 0,
    total_tokens: int | None = None,
) -> SessionEvent:
    """Create usage update event."""
    return SessionEvent(
        type=SessionEventType.ASSISTANT_USAGE,
        data=SessionEventData(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens if total_tokens is not None else input_tokens + output_tokens,
        ),
    )


# === MockSDKSession Variants for Specific Test Scenarios ===


class MockSDKSessionWithError(MockSDKSession):
    """Mock SDK session that raises error during streaming.

    Use for testing error handling paths in completion/provider.
    """

    def __init__(
        self,
        error: Exception,
        events_before_error: int = 0,
    ) -> None:
        """Initialize error-raising session.

        Args:
            error: Exception to raise during send().
            events_before_error: Number of events to deliver before raising.
        """
        super().__init__()
        self.error = error
        self.events_before_error = events_before_error

    async def send(
        self,
        prompt: str,
        *,
        attachments: list[dict[str, Any]] | None = None,
    ) -> str:
        """Send message but raise error after some events.

        Args:
            prompt: The message prompt text.
            attachments: Optional list of attachments (unused but matches base).

        Returns:
            Message ID (never reached due to error).
        """
        # Track call for assertions
        self.last_prompt = prompt
        self.last_attachments = attachments
        # Deliver some events first
        for i in range(self.events_before_error):
            event = SessionEvent(
                type=SessionEventType.ASSISTANT_MESSAGE_DELTA,
                data=SessionEventData(delta_content=f"chunk{i}"),
            )
            for handler in self._handlers:
                handler(event)
        # Raise error (no idle signal - error path)
        raise self.error


class MockSDKSessionWithAbort(MockSDKSession):
    """Mock SDK session with abort support for testing abort paths.

    Contract: sdk-protection:Session:MUST:3,4
    Use for testing tool capture + abort behavior.
    """

    def __init__(
        self,
        events: Sequence[SessionEvent | dict[str, Any]] | None = None,
        *,
        abort_behavior: str = "success",
        abort_delay: float = 0.0,
        raise_on_send: Exception | None = None,
    ) -> None:
        """Initialize abort-capable session.

        Args:
            events: Events to deliver during send().
            abort_behavior: "success", "timeout", or "exception".
            abort_delay: Delay before abort completes (seconds).
            raise_on_send: Exception to raise during send().
        """
        super().__init__(events, raise_on_send=raise_on_send)
        self.abort_behavior = abort_behavior
        self.abort_delay = abort_delay
        self.abort_called = False

    async def abort(self) -> None:
        """Mock abort with configurable behavior."""
        import asyncio

        self.abort_called = True
        if self.abort_delay > 0:
            await asyncio.sleep(self.abort_delay)

        if self.abort_behavior == "success":
            return
        elif self.abort_behavior == "timeout":
            # Sleep longer than any reasonable timeout
            await asyncio.sleep(60)
        elif self.abort_behavior == "exception":
            raise RuntimeError("Abort failed")
