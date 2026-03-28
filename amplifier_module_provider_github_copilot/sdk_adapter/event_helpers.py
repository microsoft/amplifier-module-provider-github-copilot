"""SDK event type helpers.

Shared utilities for extracting and classifying SDK event types.

Contract: contracts/sdk-boundary.md

These helpers handle two SDK event shapes:
- Dict events: {"type": "session.idle"}  (used in tests)
- Object events: event.type.value or str(event.type)  (real SDK)

Three-Medium Architecture: Event type sets loaded from events.yaml session_lifecycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Set


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


def is_idle_event(event_type: str | None, *, idle_events: Set[str] | None = None) -> bool:
    """Check if event signals session idle.

    SDK uses "session.idle", tests may use "idle".
    Per contracts/event-vocabulary.md: session.idle -> TURN_COMPLETE

    Args:
        event_type: The event type string to check
        idle_events: Optional set of idle event types (from EventConfig.idle_event_types).
                     If not provided, uses fallback set for backward compatibility.

    IMPORTANT: Uses explicit set matching, NOT substring matching.
    Substring matching would misclassify "session.idle_timeout" as idle.
    Contract: event-vocabulary:Classification:MUST:1
    """
    if event_type is None:
        return False
    type_lower = event_type.lower()
    # Use provided set or fallback (Three-Medium: prefer caller-provided YAML-backed set)
    # Note: 'if idle_events' checks for both None AND empty set - empty set must fallback
    if idle_events:
        return type_lower in {e.lower() for e in idle_events}
    # Fallback for backward compatibility (matches events.yaml session_lifecycle.idle_events)
    return type_lower in {"session.idle", "session_idle", "idle"}


def is_error_event(event_type: str | None, *, error_events: Set[str] | None = None) -> bool:
    """Check if event signals an error.

    Handles various error event formats.

    Args:
        event_type: The event type string to check
        error_events: Optional set of error event types (from EventConfig.error_event_types).
                      If not provided, uses fallback set for backward compatibility.

    IMPORTANT: Uses explicit set matching, NOT substring matching.
    Substring matching would misclassify "tool_error_recovered" as error.
    Contract: event-vocabulary:Classification:MUST:1
    """
    if event_type is None:
        return False
    type_lower = event_type.lower()
    # Use provided set or fallback (Three-Medium: prefer caller-provided YAML-backed set)
    # Note: 'if error_events' checks for both None AND empty set - empty set must fallback
    if error_events:
        return type_lower in {e.lower() for e in error_events}
    # Fallback for backward compatibility (matches events.yaml session_lifecycle.error_events)
    return type_lower in {"error", "session.error", "session_error", "sdk_error"}


def is_assistant_message(event_type: str | None) -> bool:
    """Check if event is an ASSISTANT_MESSAGE (tool capture source).

    SDK uses "assistant.message" for completion with potential tool_requests.
    This is the event that signals first-turn complete for tool capture.

    IMPORTANT: Uses explicit set matching, NOT substring matching.
    Substring matching is dangerous — "assistant_message_delta" would match.
    P2 Fix: Align with is_idle_event, is_error_event, is_usage_event pattern.
    """
    if event_type is None:
        return False
    type_lower = event_type.lower()
    # P2 Fix: Explicit set matching (consistent with other helpers in this file)
    return type_lower in {"assistant.message", "assistant_message"}


def is_usage_event(event_type: str | None, *, usage_events: Set[str] | None = None) -> bool:
    """Check if event contains usage data (token counts).

    SDK sends assistant.usage event with input_tokens and output_tokens.
    This event may arrive AFTER session.idle, causing race conditions
    if we rely solely on queue draining.

    Args:
        event_type: The event type string to check
        usage_events: Optional set of usage event types (from EventConfig.usage_event_types).
                      If not provided, uses fallback set for backward compatibility.

    Contract: streaming-contract:usage:MUST:1
    """
    if event_type is None:
        return False
    type_lower = event_type.lower()
    # Use provided set or fallback (Three-Medium: prefer caller-provided YAML-backed set)
    # Note: 'if usage_events' checks for both None AND empty set - empty set must fallback
    if usage_events:
        return type_lower in {e.lower() for e in usage_events}
    # Fallback for backward compatibility (matches events.yaml session_lifecycle.usage_events)
    return type_lower in {"assistant.usage", "usage_update"}


def extract_usage_data(sdk_event: Any) -> dict[str, int] | None:
    """Extract usage data (input_tokens, output_tokens) from SDK event.

    Args:
        sdk_event: SDK event (dict or object with .data attribute)

    Returns:
        Dict with input_tokens and output_tokens, or None if not a usage event.

    Note:
        This function handles dynamic SDK data with unknown structure.
        Type ignores are used for dict access on dynamic data.
    """
    # Handle dict events (tests)
    if isinstance(sdk_event, dict):
        typed_dict = cast(dict[str, Any], sdk_event)
        data = typed_dict.get("data", typed_dict)
        if isinstance(data, dict):
            typed_data = cast(dict[str, Any], data)
            input_tokens: Any = typed_data.get("input_tokens")
            output_tokens: Any = typed_data.get("output_tokens")
            if input_tokens is not None or output_tokens is not None:
                return {
                    "input_tokens": int(input_tokens) if input_tokens else 0,
                    "output_tokens": int(output_tokens) if output_tokens else 0,
                }
        return None

    # Handle object events (real SDK)
    data = getattr(sdk_event, "data", None)
    if data is not None:
        input_tokens = getattr(data, "input_tokens", None)
        output_tokens = getattr(data, "output_tokens", None)
        if input_tokens is not None or output_tokens is not None:
            return {
                "input_tokens": int(input_tokens) if input_tokens else 0,
                "output_tokens": int(output_tokens) if output_tokens else 0,
            }

    return None


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
