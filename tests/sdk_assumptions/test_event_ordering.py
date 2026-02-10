"""
SDK Assumption Tests: Event Ordering

These tests validate assumptions about the order in which SDK events fire.
Our "Deny + Destroy" pattern critically depends on receiving ASSISTANT_MESSAGE
events (with tool_requests) BEFORE the preToolUse hook is invoked.

CRITICAL ASSUMPTION:
    tool_requests are available in ASSISTANT_MESSAGE event data BEFORE
    the SDK's internal tool execution flow begins (which would trigger
    preToolUse hook).

WHY THIS MATTERS:
    We capture tool calls from the ASSISTANT_MESSAGE event. If the SDK
    changed to invoke preToolUse BEFORE emitting the message event,
    we would have no way to capture the tool call data.

BREAKING CHANGE INDICATORS:
    - response.tool_calls is empty when LLM clearly requested tools
    - preToolUse hook fires but we haven't captured tool_requests yet
    - Tool calls silently disappear from responses

SDK LOCATIONS TO VERIFY:
    - copilot-sdk/python/copilot/session.py: Event dispatch logic
    - copilot-sdk/python/copilot/generated/session_events.py: Event types
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from .conftest import (
    EventSequence,
    InstrumentedMockSession,
    MockSessionEventType,
    create_event_data,
    create_idle_event,
    create_tool_request,
    create_usage_event,
)


class TestEventOrderAssumptions:
    """
    Tests that validate event ordering assumptions.

    These tests simulate SDK behavior to verify our implementation
    handles events in the expected order.
    """

    @pytest.mark.asyncio
    async def test_message_event_contains_tool_requests_before_idle(self):
        """
        ASSUMPTION: ASSISTANT_MESSAGE events contain tool_requests data.

        The tool_requests field in ASSISTANT_MESSAGE.data must contain
        the structured tool call data before SESSION_IDLE fires.
        """
        # Arrange: Create a message event with tool requests
        tool_request = create_tool_request(
            name="read_file",
            tool_call_id="call_order_test_001",
            arguments={"path": "test.py"},
        )

        events = [
            (
                MockSessionEventType.ASSISTANT_MESSAGE,
                create_event_data(
                    content="I'll read that file.",
                    tool_requests=[tool_request],
                ),
            ),
            create_usage_event(),
            create_idle_event(),
        ]

        session = InstrumentedMockSession(
            event_sequence=EventSequence(events=events, emit_async=True)
        )

        # Track captured data
        captured_tool_requests: list[Any] = []
        idle_received = asyncio.Event()

        def event_handler(event: Any) -> None:
            if event.type == MockSessionEventType.ASSISTANT_MESSAGE:
                if hasattr(event.data, "tool_requests") and event.data.tool_requests:
                    captured_tool_requests.extend(event.data.tool_requests)
            elif event.type == MockSessionEventType.SESSION_IDLE:
                idle_received.set()

        # Act
        session.on(event_handler)
        await session.send({"prompt": "Read test.py"})

        # Wait for events with timeout
        async with asyncio.timeout(1.0):
            await idle_received.wait()

        # Assert: Tool requests must be captured before idle
        assert len(captured_tool_requests) == 1
        assert captured_tool_requests[0].name == "read_file"
        assert captured_tool_requests[0].tool_call_id == "call_order_test_001"

    @pytest.mark.asyncio
    async def test_streaming_deltas_precede_final_message(self):
        """
        ASSUMPTION: ASSISTANT_MESSAGE_DELTA events fire before ASSISTANT_MESSAGE.

        When streaming is enabled, delta events must precede the final
        message event so we can assemble content progressively.
        """
        # Arrange: Deltas followed by final message
        events = [
            (
                MockSessionEventType.ASSISTANT_MESSAGE_DELTA,
                create_event_data(delta_content="Hello "),
            ),
            (
                MockSessionEventType.ASSISTANT_MESSAGE_DELTA,
                create_event_data(delta_content="world!"),
            ),
            (
                MockSessionEventType.ASSISTANT_MESSAGE,
                create_event_data(content="Hello world!", tool_requests=None),
            ),
            create_usage_event(),
            create_idle_event(),
        ]

        session = InstrumentedMockSession(
            event_sequence=EventSequence(events=events, emit_async=True)
        )

        # Track event order
        event_order: list[str] = []
        idle_received = asyncio.Event()

        def event_handler(event: Any) -> None:
            event_order.append(event.type.value)
            if event.type == MockSessionEventType.SESSION_IDLE:
                idle_received.set()

        # Act
        session.on(event_handler)
        await session.send({"prompt": "Say hello"})

        async with asyncio.timeout(1.0):
            await idle_received.wait()

        # Assert: Deltas must come before final message
        delta_indices = [i for i, e in enumerate(event_order) if e == "assistant.message_delta"]
        message_index = event_order.index("assistant.message")

        assert len(delta_indices) == 2
        assert all(di < message_index for di in delta_indices)

    @pytest.mark.asyncio
    async def test_usage_event_fires_after_message(self):
        """
        ASSUMPTION: ASSISTANT_USAGE event fires after ASSISTANT_MESSAGE.

        Usage data (token counts) must be available after we've captured
        the response content and tool calls.
        """
        events = [
            (
                MockSessionEventType.ASSISTANT_MESSAGE,
                create_event_data(content="Response", tool_requests=None),
            ),
            (
                MockSessionEventType.ASSISTANT_USAGE,
                create_event_data(input_tokens=100, output_tokens=50),
            ),
            create_idle_event(),
        ]

        session = InstrumentedMockSession(
            event_sequence=EventSequence(events=events, emit_async=True)
        )

        event_order: list[str] = []
        idle_received = asyncio.Event()

        def event_handler(event: Any) -> None:
            event_order.append(event.type.value)
            if event.type == MockSessionEventType.SESSION_IDLE:
                idle_received.set()

        session.on(event_handler)
        await session.send({"prompt": "Test"})

        async with asyncio.timeout(1.0):
            await idle_received.wait()

        # Assert: Usage must come after message
        message_index = event_order.index("assistant.message")
        usage_index = event_order.index("assistant.usage")
        assert usage_index > message_index

    @pytest.mark.asyncio
    async def test_idle_event_fires_last(self):
        """
        ASSUMPTION: SESSION_IDLE event fires after all turn processing.

        We wait for SESSION_IDLE to know the response is complete.
        It must be the last event in a successful turn.
        """
        events = [
            (
                MockSessionEventType.ASSISTANT_MESSAGE,
                create_event_data(content="Done", tool_requests=None),
            ),
            create_usage_event(),
            create_idle_event(),
        ]

        session = InstrumentedMockSession(
            event_sequence=EventSequence(events=events, emit_async=True)
        )

        event_order: list[str] = []
        idle_received = asyncio.Event()

        def event_handler(event: Any) -> None:
            event_order.append(event.type.value)
            if event.type == MockSessionEventType.SESSION_IDLE:
                idle_received.set()

        session.on(event_handler)
        await session.send({"prompt": "Test"})

        async with asyncio.timeout(1.0):
            await idle_received.wait()

        # Assert: Idle must be last
        assert event_order[-1] == "session.idle"

    @pytest.mark.asyncio
    async def test_reasoning_events_precede_message(self):
        """
        ASSUMPTION: Reasoning events fire before the final message.

        For thinking models (o3, Claude with extended_thinking), reasoning
        deltas and final reasoning content must precede the assistant message.
        """
        events = [
            (
                MockSessionEventType.ASSISTANT_REASONING_DELTA,
                create_event_data(delta_content="Analyzing... "),
            ),
            (
                MockSessionEventType.ASSISTANT_REASONING,
                create_event_data(content="I analyzed the problem."),
            ),
            (
                MockSessionEventType.ASSISTANT_MESSAGE,
                create_event_data(content="The answer is 4.", tool_requests=None),
            ),
            create_usage_event(),
            create_idle_event(),
        ]

        session = InstrumentedMockSession(
            event_sequence=EventSequence(events=events, emit_async=True)
        )

        event_order: list[str] = []
        idle_received = asyncio.Event()

        def event_handler(event: Any) -> None:
            event_order.append(event.type.value)
            if event.type == MockSessionEventType.SESSION_IDLE:
                idle_received.set()

        session.on(event_handler)
        await session.send({"prompt": "What is 2+2?"})

        async with asyncio.timeout(1.0):
            await idle_received.wait()

        # Assert: Reasoning events before message
        reasoning_delta_idx = event_order.index("assistant.reasoning_delta")
        reasoning_idx = event_order.index("assistant.reasoning")
        message_idx = event_order.index("assistant.message")

        assert reasoning_delta_idx < reasoning_idx < message_idx


class TestMultipleToolRequestOrdering:
    """
    Tests for scenarios with multiple tool calls in a single turn.
    """

    @pytest.mark.asyncio
    async def test_multiple_tool_requests_in_single_message(self):
        """
        ASSUMPTION: Multiple tool_requests in one ASSISTANT_MESSAGE.

        When LLM wants to call multiple tools, all requests must be
        present in the same message event's tool_requests array.
        """
        tool_requests = [
            create_tool_request(
                name="read_file",
                tool_call_id="call_multi_001",
                arguments={"path": "file1.py"},
            ),
            create_tool_request(
                name="read_file",
                tool_call_id="call_multi_002",
                arguments={"path": "file2.py"},
            ),
            create_tool_request(
                name="list_dir",
                tool_call_id="call_multi_003",
                arguments={"path": "."},
            ),
        ]

        events = [
            (
                MockSessionEventType.ASSISTANT_MESSAGE,
                create_event_data(
                    content="I'll read multiple files.",
                    tool_requests=tool_requests,
                ),
            ),
            create_usage_event(),
            create_idle_event(),
        ]

        session = InstrumentedMockSession(
            event_sequence=EventSequence(events=events, emit_async=True)
        )

        captured_requests: list[Any] = []
        idle_received = asyncio.Event()

        def event_handler(event: Any) -> None:
            if event.type == MockSessionEventType.ASSISTANT_MESSAGE:
                if hasattr(event.data, "tool_requests") and event.data.tool_requests:
                    captured_requests.extend(event.data.tool_requests)
            elif event.type == MockSessionEventType.SESSION_IDLE:
                idle_received.set()

        session.on(event_handler)
        await session.send({"prompt": "Read file1.py and file2.py"})

        async with asyncio.timeout(1.0):
            await idle_received.wait()

        # Assert: All tool requests captured
        assert len(captured_requests) == 3
        tool_names = [r.name for r in captured_requests]
        assert tool_names == ["read_file", "read_file", "list_dir"]

    @pytest.mark.asyncio
    async def test_tool_request_order_preserved(self):
        """
        ASSUMPTION: Tool request order in array is preserved.

        The order of tool_requests must match the order intended by LLM,
        as some tools may have implicit dependencies.
        """
        tool_requests = [
            create_tool_request(
                name="first_tool",
                tool_call_id="call_order_001",
                arguments={},
            ),
            create_tool_request(
                name="second_tool",
                tool_call_id="call_order_002",
                arguments={},
            ),
            create_tool_request(
                name="third_tool",
                tool_call_id="call_order_003",
                arguments={},
            ),
        ]

        events = [
            (
                MockSessionEventType.ASSISTANT_MESSAGE,
                create_event_data(
                    content="Sequential tools.",
                    tool_requests=tool_requests,
                ),
            ),
            create_usage_event(),
            create_idle_event(),
        ]

        session = InstrumentedMockSession(
            event_sequence=EventSequence(events=events, emit_async=True)
        )

        captured_ids: list[str] = []
        idle_received = asyncio.Event()

        def event_handler(event: Any) -> None:
            if event.type == MockSessionEventType.ASSISTANT_MESSAGE:
                if hasattr(event.data, "tool_requests") and event.data.tool_requests:
                    captured_ids.extend([r.tool_call_id for r in event.data.tool_requests])
            elif event.type == MockSessionEventType.SESSION_IDLE:
                idle_received.set()

        session.on(event_handler)
        await session.send({"prompt": "Run tools in order"})

        async with asyncio.timeout(1.0):
            await idle_received.wait()

        # Assert: Order preserved
        assert captured_ids == ["call_order_001", "call_order_002", "call_order_003"]


class TestEventDataIntegrity:
    """
    Tests that validate event data is complete and correctly structured.
    """

    @pytest.mark.asyncio
    async def test_tool_request_has_required_fields(self):
        """
        ASSUMPTION: Tool requests have name, tool_call_id, and arguments.

        These three fields are required for our provider to construct
        ToolCall objects for the Amplifier orchestrator.
        """
        tool_request = create_tool_request(
            name="test_tool",
            tool_call_id="call_fields_test",
            arguments={"key": "value"},
        )

        events = [
            (
                MockSessionEventType.ASSISTANT_MESSAGE,
                create_event_data(
                    content="Using tool.",
                    tool_requests=[tool_request],
                ),
            ),
            create_usage_event(),
            create_idle_event(),
        ]

        session = InstrumentedMockSession(
            event_sequence=EventSequence(events=events, emit_async=True)
        )

        captured_request: Any = None
        idle_received = asyncio.Event()

        def event_handler(event: Any) -> None:
            nonlocal captured_request
            if event.type == MockSessionEventType.ASSISTANT_MESSAGE:
                if hasattr(event.data, "tool_requests") and event.data.tool_requests:
                    captured_request = event.data.tool_requests[0]
            elif event.type == MockSessionEventType.SESSION_IDLE:
                idle_received.set()

        session.on(event_handler)
        await session.send({"prompt": "Use tool"})

        async with asyncio.timeout(1.0):
            await idle_received.wait()

        # Assert: Required fields present
        assert captured_request is not None
        assert hasattr(captured_request, "name")
        assert hasattr(captured_request, "tool_call_id")
        assert hasattr(captured_request, "arguments")

        assert captured_request.name == "test_tool"
        assert captured_request.tool_call_id == "call_fields_test"
        assert captured_request.arguments == {"key": "value"}

    @pytest.mark.asyncio
    async def test_arguments_can_be_string_or_dict(self):
        """
        ASSUMPTION: Tool arguments may be dict or JSON string.

        The SDK sometimes passes arguments as a JSON string rather than
        a parsed dict. Our provider must handle both.
        """
        # Test with string arguments
        tool_request_str = create_tool_request(
            name="tool_with_str_args",
            tool_call_id="call_str_args",
            arguments='{"query": "search term"}',  # JSON string
        )

        # Test with dict arguments
        tool_request_dict = create_tool_request(
            name="tool_with_dict_args",
            tool_call_id="call_dict_args",
            arguments={"query": "search term"},  # Dict
        )

        events = [
            (
                MockSessionEventType.ASSISTANT_MESSAGE,
                create_event_data(
                    content="Testing argument types.",
                    tool_requests=[tool_request_str, tool_request_dict],
                ),
            ),
            create_usage_event(),
            create_idle_event(),
        ]

        session = InstrumentedMockSession(
            event_sequence=EventSequence(events=events, emit_async=True)
        )

        captured_args_types: list[type] = []
        idle_received = asyncio.Event()

        def event_handler(event: Any) -> None:
            if event.type == MockSessionEventType.ASSISTANT_MESSAGE:
                if hasattr(event.data, "tool_requests") and event.data.tool_requests:
                    for req in event.data.tool_requests:
                        captured_args_types.append(type(req.arguments))
            elif event.type == MockSessionEventType.SESSION_IDLE:
                idle_received.set()

        session.on(event_handler)
        await session.send({"prompt": "Test"})

        async with asyncio.timeout(1.0):
            await idle_received.wait()

        # Assert: Both types received
        assert str in captured_args_types
        assert dict in captured_args_types
