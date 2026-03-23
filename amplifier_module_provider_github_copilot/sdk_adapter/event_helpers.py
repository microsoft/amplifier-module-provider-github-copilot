"""SDK event type helpers.

Shared utilities for extracting and classifying SDK event types.

Contract: contracts/sdk-boundary.md

These helpers handle two SDK event shapes:
- Dict events: {"type": "session.idle"}  (used in tests)
- Object events: event.type.value or str(event.type)  (real SDK)
"""

from __future__ import annotations

from typing import Any, cast


def extract_event_type(sdk_event: Any) -> str | None:
    """Extract event type from SDK event (dict or object).

    Handles both dict events (tests) and object events (real SDK):
    - Dict events: {"type": "session.idle"}
    - Object events: event.type.value or str(event.type)
    """
    if isinstance(sdk_event, dict):
        typed_dict = cast(dict[str, Any], sdk_event)
        event_type = typed_dict.get("type")
        return str(event_type) if event_type is not None else None
    # Object event — check for .type attribute
    event_type = getattr(sdk_event, "type", None)
    if event_type is None:
        return None
    # SDK events use enum with .value attribute
    if hasattr(event_type, "value"):
        return str(event_type.value)
    return str(event_type)


def is_idle_event(event_type: str | None) -> bool:
    """Check if event signals session idle.

    SDK uses "session.idle", tests may use "SESSION_IDLE".
    Per contracts/event-vocabulary.md: session.idle -> TURN_COMPLETE
    """
    if event_type is None:
        return False
    type_lower = event_type.lower()
    # SDK format: "session.idle" (dot-separated)
    # Config format: "session_idle" (underscore)
    # Legacy format: "SESSION_IDLE" (uppercase)
    return "idle" in type_lower


def is_error_event(event_type: str | None) -> bool:
    """Check if event signals an error.

    Handles various error event formats.
    """
    if event_type is None:
        return False
    type_lower = event_type.lower()
    return "error" in type_lower


def is_assistant_message(event_type: str | None) -> bool:
    """Check if event is an ASSISTANT_MESSAGE (tool capture source).

    SDK uses "assistant.message" for completion with potential tool_requests.
    This is the event that signals first-turn complete for tool capture.
    """
    if event_type is None:
        return False
    type_lower = event_type.lower()
    # SDK format: "assistant.message"
    # Legacy format: "assistant_message", "ASSISTANT_MESSAGE"
    return "assistant" in type_lower and "message" in type_lower and "delta" not in type_lower


def extract_tool_requests(sdk_event: Any) -> list[Any]:
    """Extract tool_requests from SDK event.

    SDK ASSISTANT_MESSAGE events contain tool_requests when the model
    wants to call tools. This is critical for abort-on-capture pattern.

    Args:
        sdk_event: SDK event (dict or object with .data attribute)

    Returns:
        List of tool requests, or empty list if none found.

    Note:
        This function handles dynamic SDK data with unknown structure.
        Type ignores are used for dict access on dynamic data.

    """
    # Handle dict events (tests)
    if isinstance(sdk_event, dict):
        # pyright: ignore[reportUnknownVariableType,reportUnknownMemberType]
        data: Any = sdk_event.get("data", sdk_event)  # type: ignore[union-attr]
        if isinstance(data, dict):
            tool_reqs: Any = data.get("tool_requests")  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
            return list(tool_reqs) if tool_reqs else []  # pyright: ignore[reportUnknownArgumentType]

    # Handle object events (real SDK)
    # pyright: ignore[reportUnknownArgumentType]
    data = getattr(sdk_event, "data", None)  # type: ignore[attr-defined]
    if data is not None:
        tool_reqs = getattr(data, "tool_requests", None)  # pyright: ignore[reportUnknownArgumentType]
        if tool_reqs:
            return list(tool_reqs)

    # Fallback: check directly on event
    tool_reqs = getattr(sdk_event, "tool_requests", None)  # pyright: ignore[reportUnknownArgumentType]
    return list(tool_reqs) if tool_reqs else []


def has_tool_capture_event(sdk_event: Any) -> bool:
    """Check if SDK event contains tool requests (abort-on-capture trigger).

    This helper combines event type check with tool_requests extraction
    to determine if we should abort the SDK's agentic loop.

    Contract: When ASSISTANT_MESSAGE contains tool_requests, we have
    captured the model's tool intentions and should stop waiting.

    Args:
        sdk_event: SDK event to check

    Returns:
        True if this is an ASSISTANT_MESSAGE with tool_requests

    """
    event_type = extract_event_type(sdk_event)
    if not is_assistant_message(event_type):
        return False
    tool_reqs = extract_tool_requests(sdk_event)
    return len(tool_reqs) > 0
