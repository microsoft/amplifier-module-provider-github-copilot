"""
Shared fixtures for SDK behavioral assumption tests.

This module provides test infrastructure specifically designed to validate
SDK behavioral assumptions. The fixtures simulate SDK internals with
precise control over event ordering, hook invocation, and session lifecycle.

Design Principles:
    1. Fast execution — all tests use mocks, no live SDK required
    2. Precise control — can inject specific event sequences
    3. Observable — can verify hook invocations and their effects
    4. Isolated — each test gets fresh session state

Based on patterns proven in test_streaming.py (the async event emission
pattern using asyncio.create_task was critical for timeout tests).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from unittest.mock import Mock

import pytest

# =============================================================================
# Mock SDK Types (match copilot.generated.session_events.SessionEventType)
# =============================================================================


class MockSessionEventType(Enum):
    """
    Mock of copilot.generated.session_events.SessionEventType.

    Used to avoid importing the real SDK in unit tests. Values must
    match the real enum exactly for test validity.
    """

    # Assistant message events
    ASSISTANT_MESSAGE_DELTA = "assistant.message_delta"
    ASSISTANT_MESSAGE = "assistant.message"
    ASSISTANT_REASONING_DELTA = "assistant.reasoning_delta"
    ASSISTANT_REASONING = "assistant.reasoning"
    ASSISTANT_USAGE = "assistant.usage"

    # Session lifecycle events
    SESSION_IDLE = "session.idle"
    SESSION_ERROR = "session.error"
    SESSION_START = "session.start"
    SESSION_SHUTDOWN = "session.shutdown"

    # Tool events (from SDK, for reference)
    TOOL_EXECUTION_START = "tool.execution_start"
    TOOL_EXECUTION_COMPLETE = "tool.execution_complete"

    # Hook events
    HOOK_START = "hook.start"
    HOOK_END = "hook.end"


# =============================================================================
# Observable Hook Infrastructure
# =============================================================================


@dataclass
class HookInvocation:
    """Record of a single hook invocation for assertion."""

    hook_type: str
    input_data: dict[str, Any]
    output: dict[str, Any] | None
    timestamp: float


@dataclass
class HookRecorder:
    """
    Records all hook invocations for later assertion.

    Used to verify hook ordering, input data, and output effects.
    """

    invocations: list[HookInvocation] = field(default_factory=list)

    def record(
        self,
        hook_type: str,
        input_data: dict[str, Any],
        output: dict[str, Any] | None,
    ) -> None:
        """Record a hook invocation."""
        import time

        self.invocations.append(
            HookInvocation(
                hook_type=hook_type,
                input_data=input_data.copy(),
                output=output.copy() if output else None,
                timestamp=time.monotonic(),
            )
        )

    def get_by_type(self, hook_type: str) -> list[HookInvocation]:
        """Get all invocations of a specific hook type."""
        return [inv for inv in self.invocations if inv.hook_type == hook_type]

    def was_called(self, hook_type: str) -> bool:
        """Check if a hook type was ever invoked."""
        return any(inv.hook_type == hook_type for inv in self.invocations)

    def clear(self) -> None:
        """Clear all recorded invocations."""
        self.invocations.clear()


# =============================================================================
# Mock Session with Controllable Behavior
# =============================================================================


@dataclass
class EventSequence:
    """
    Defines a sequence of events to emit during a session turn.

    Allows precise control over:
    - Event types and data
    - Whether events fire before/after hooks
    - Timing of event emission (sync vs async)
    """

    events: list[tuple[MockSessionEventType, Any]]
    emit_async: bool = True  # If True, events emit via asyncio.create_task
    delay_between_events: float = 0.001  # Small delay for realistic ordering


class InstrumentedMockSession:
    """
    Mock session with full instrumentation for assumption testing.

    Features:
    - Event handlers with controlled event sequences
    - Hook simulation with invocation recording
    - Tool handler registration tracking
    - Destroy behavior observation

    This is the core test infrastructure for SDK assumption tests.
    """

    def __init__(
        self,
        session_id: str = "test-assumption-session",
        event_sequence: EventSequence | None = None,
    ):
        self.session_id = session_id
        self.event_sequence = event_sequence or EventSequence(events=[])

        # Event handling
        self.event_handlers: list[Callable[[Any], None]] = []

        # Lifecycle tracking
        self.destroyed = False
        self.destroy_called_count = 0
        self.sent_messages: list[dict[str, Any]] = []

        # Hook infrastructure
        self.hook_recorder = HookRecorder()
        self._pre_tool_use_handler: Callable | None = None
        self._post_tool_use_handler: Callable | None = None

        # Tool handler tracking
        self.registered_tool_handlers: dict[str, Callable] = {}
        self.tool_handler_invocations: list[dict[str, Any]] = []

        # For simulating tool execution attempts
        self.pending_tool_calls: list[dict[str, Any]] = []

    def on(self, handler: Callable[[Any], None]) -> Callable[[], None]:
        """
        Subscribe to events. Returns unsubscribe function.

        Matches SDK CopilotSession.on() signature.
        """
        self.event_handlers.append(handler)

        def unsubscribe() -> None:
            if handler in self.event_handlers:
                self.event_handlers.remove(handler)

        return unsubscribe

    def register_pre_tool_use_hook(self, handler: Callable[[dict, dict], dict | None]) -> None:
        """Register preToolUse hook handler."""
        self._pre_tool_use_handler = handler

    def register_post_tool_use_hook(self, handler: Callable[[dict, dict], dict | None]) -> None:
        """Register postToolUse hook handler."""
        self._post_tool_use_handler = handler

    def register_tool_handler(self, name: str, handler: Callable) -> None:
        """Register a tool handler (for tracking, not execution in tests)."""
        self.registered_tool_handlers[name] = handler

    async def send(self, message: dict[str, Any]) -> None:
        """
        Send a message and trigger the event sequence.

        Matches SDK CopilotSession.send() pattern where events are
        emitted asynchronously after send() returns.
        """
        self.sent_messages.append(message)

        if self.event_sequence.emit_async:
            asyncio.create_task(self._emit_events_async())
        else:
            await self._emit_events_sync()

    async def _emit_events_async(self) -> None:
        """Emit events asynchronously (realistic SDK behavior)."""
        await asyncio.sleep(0.001)  # Yield to allow timeout to start

        for event_type, event_data in self.event_sequence.events:
            if self.destroyed:
                break  # Stop emitting if destroyed

            event = self._create_event(event_type, event_data)
            self._dispatch_event(event)

            await asyncio.sleep(self.event_sequence.delay_between_events)

    async def _emit_events_sync(self) -> None:
        """Emit events synchronously (for specific test scenarios)."""
        for event_type, event_data in self.event_sequence.events:
            if self.destroyed:
                break

            event = self._create_event(event_type, event_data)
            self._dispatch_event(event)

    def _create_event(self, event_type: MockSessionEventType, event_data: Any) -> Mock:
        """Create a mock event object."""
        event = Mock()
        event.type = event_type
        event.data = event_data
        return event

    def _dispatch_event(self, event: Any) -> None:
        """Dispatch event to all registered handlers."""
        for handler in self.event_handlers:
            try:
                handler(event)
            except Exception:
                pass  # Handlers shouldn't crash the dispatch

    async def simulate_tool_call(
        self,
        tool_name: str,
        tool_call_id: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Simulate the SDK's tool call flow.

        This method simulates what the SDK does internally:
        1. Fire preToolUse hook
        2. If allowed, invoke tool handler
        3. Fire postToolUse hook
        4. Return result

        Used to test hook behavior assumptions.
        """
        # Step 1: preToolUse hook
        hook_input = {
            "toolName": tool_name,
            "toolArgs": arguments,
            "timestamp": 0,
            "cwd": "/mock",
        }
        context = {"session_id": self.session_id}

        pre_result = None
        if self._pre_tool_use_handler:
            pre_result = self._pre_tool_use_handler(hook_input, context)
            if asyncio.iscoroutine(pre_result):
                pre_result = await pre_result

            self.hook_recorder.record("preToolUse", hook_input, pre_result)

        # Step 2: Check if denied
        permission = pre_result.get("permissionDecision") if pre_result else "allow"

        tool_result: dict[str, Any] = {}
        handler_invoked = False

        if permission != "deny":
            # Step 3: Invoke tool handler
            handler = self.registered_tool_handlers.get(tool_name)
            if handler:
                handler_invoked = True
                invocation = {
                    "tool_name": tool_name,
                    "tool_call_id": tool_call_id,
                    "arguments": arguments,
                }
                self.tool_handler_invocations.append(invocation)

                try:
                    result = handler(invocation)
                    if asyncio.iscoroutine(result):
                        result = await result
                    tool_result = {"content": str(result), "success": True}
                except Exception as e:
                    tool_result = {"error": str(e), "success": False}

        # Step 4: postToolUse hook (if handler was invoked or explicitly testing)
        if handler_invoked and self._post_tool_use_handler:
            post_input = {
                "toolName": tool_name,
                "toolArgs": arguments,
                "toolResult": tool_result,
                "timestamp": 0,
                "cwd": "/mock",
            }
            post_result = self._post_tool_use_handler(post_input, context)
            if asyncio.iscoroutine(post_result):
                post_result = await post_result
            self.hook_recorder.record("postToolUse", post_input, post_result)

        return {
            "permission": permission,
            "handler_invoked": handler_invoked,
            "tool_result": tool_result,
        }

    async def destroy(self) -> None:
        """
        Destroy the session.

        Matches SDK CopilotSession.destroy() behavior:
        - Marks session as destroyed
        - Clears handlers
        - Is idempotent (safe to call multiple times)
        """
        self.destroyed = True
        self.destroy_called_count += 1
        self.event_handlers.clear()


# =============================================================================
# Helper Functions for Creating Test Data
# =============================================================================


def create_event_data(**kwargs: Any) -> Mock:
    """Create mock event data with specified attributes."""
    data = Mock()
    for key, value in kwargs.items():
        setattr(data, key, value)
    return data


def create_tool_request(
    name: str,
    tool_call_id: str,
    arguments: dict[str, Any] | str,
) -> Mock:
    """Create a mock tool request matching SDK ToolRequest structure."""
    request = Mock()
    request.name = name
    request.tool_call_id = tool_call_id
    request.arguments = arguments
    return request


def create_message_event_with_tools(
    content: str,
    tool_requests: list[Mock],
) -> tuple[MockSessionEventType, Mock]:
    """Create an ASSISTANT_MESSAGE event with tool requests."""
    return (
        MockSessionEventType.ASSISTANT_MESSAGE,
        create_event_data(content=content, tool_requests=tool_requests),
    )


def create_idle_event() -> tuple[MockSessionEventType, Mock]:
    """Create a SESSION_IDLE event."""
    return (MockSessionEventType.SESSION_IDLE, create_event_data())


def create_usage_event(
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> tuple[MockSessionEventType, Mock]:
    """Create an ASSISTANT_USAGE event."""
    return (
        MockSessionEventType.ASSISTANT_USAGE,
        create_event_data(input_tokens=input_tokens, output_tokens=output_tokens),
    )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def hook_recorder() -> HookRecorder:
    """Provide a fresh hook recorder for each test."""
    return HookRecorder()


@pytest.fixture
def mock_session_event_type():
    """Provide MockSessionEventType for test patching."""
    return MockSessionEventType


@pytest.fixture
def basic_response_sequence() -> EventSequence:
    """
    Standard response sequence: message + usage + idle.

    Use this for tests that don't need tool calls.
    """
    return EventSequence(
        events=[
            (
                MockSessionEventType.ASSISTANT_MESSAGE,
                create_event_data(content="Test response", tool_requests=None),
            ),
            create_usage_event(),
            create_idle_event(),
        ],
        emit_async=True,
    )


@pytest.fixture
def tool_call_response_sequence() -> EventSequence:
    """
    Response sequence with tool calls: message with tools + usage + idle.

    Use this for tests validating tool call capture.
    """
    tool_request = create_tool_request(
        name="test_tool",
        tool_call_id="call_test_123",
        arguments={"arg1": "value1"},
    )
    return EventSequence(
        events=[
            create_message_event_with_tools("I'll use the test tool.", [tool_request]),
            create_usage_event(),
            create_idle_event(),
        ],
        emit_async=True,
    )


@pytest.fixture
def instrumented_session(basic_response_sequence: EventSequence) -> InstrumentedMockSession:
    """Provide a fresh instrumented session with basic response sequence."""
    return InstrumentedMockSession(event_sequence=basic_response_sequence)


@pytest.fixture
def deny_all_hook() -> Callable[[dict, dict], dict]:
    """
    The standard deny-all hook used by our provider.

    Returns deny for all tool calls.
    """

    def deny_hook(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        return {
            "permissionDecision": "deny",
            "permissionDecisionReason": "Test deny hook",
        }

    return deny_hook


@pytest.fixture
def allow_all_hook() -> Callable[[dict, dict], dict]:
    """
    Hook that allows all tool calls.

    Used for comparison testing with deny hook.
    """

    def allow_hook(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        return {"permissionDecision": "allow"}

    return allow_hook
