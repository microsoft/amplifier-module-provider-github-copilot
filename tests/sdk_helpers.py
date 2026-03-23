"""SDK integration test helper functions.

These helpers handle the reality that SDK events may be dicts OR typed objects,
and that SDK versions may change field names or structures.

Contract: sdk-boundary:EventShape
Reference: SDK SessionEvent structure from github-copilot-sdk

SDK v0.1.33+ Structure:
    SessionEvent.data.delta_content  # Streaming text
    SessionEvent.data.content        # Complete message
    SessionEvent.data.message_id     # Message correlation
"""

from typing import Any, cast


def get_event_type(event: Any) -> str:
    """Extract event type from SDK event (handles dict or object).

    SDK events may be:
    - dict with "type" key
    - object with "type" attribute
    - object with "event_type" attribute
    """
    if isinstance(event, dict):
        event_dict = cast(dict[str, Any], event)
        return str(event_dict.get("type", "unknown"))
    result: str = getattr(event, "type", getattr(event, "event_type", "unknown"))
    return result


def get_event_field(event: Any, field: str) -> Any:
    """Extract field from SDK event (handles dict or object).

    SDK v0.1.33+ nests content fields in event.data:
    - event.data.delta_content (streaming)
    - event.data.content (complete)
    - event.data.message_id

    This function checks both top-level and nested data locations.

    Args:
        event: SDK event (dict or typed object)
        field: Field name to extract (e.g., "delta_content", "content")

    Returns:
        Field value or None if not found

    Contract: sdk-boundary:EventShape:MUST:2
    Reference: SDK SessionEvent.data structure from github-copilot-sdk
    """
    if isinstance(event, dict):
        event_dict = cast(dict[str, Any], event)
        # Check top-level first (backward compat)
        if field in event_dict:
            return event_dict.get(field)
        # Check nested data (SDK v0.1.33+)
        data = event_dict.get("data")
        if isinstance(data, dict):
            data_dict = cast(dict[str, Any], data)
            return data_dict.get(field)
        return None

    # Object-style event
    # Check top-level first (backward compat)
    value = getattr(event, field, None)
    if value is not None:
        return value

    # Check nested data (SDK v0.1.33+)
    data = getattr(event, "data", None)
    if data is not None:
        return getattr(data, field, None)

    return None


def describe_event(event: Any) -> str:
    """Human-readable description of an SDK event for debugging.

    Useful for test failure messages and drift detection.
    """
    if isinstance(event, dict):
        return str(cast(dict[str, Any], event))
    cls = type(event).__name__
    try:
        attrs: dict[str, Any] = {
            k: v for k, v in cast(dict[str, Any], vars(event)).items() if not k.startswith("_")
        }
        return f"{cls}({attrs})"
    except TypeError:
        # Some objects don't support vars()
        return f"{cls}(<non-inspectable>)"


def collect_event_types(events: list[Any]) -> list[str]:
    """Extract event types from a list of SDK events.

    Args:
        events: List of SDK events

    Returns:
        List of event type strings
    """
    return [get_event_type(e) for e in events]


def has_event_type(events: list[Any], event_type: str) -> bool:
    """Check if any event in the list has the given type.

    Args:
        events: List of SDK events
        event_type: Event type to search for

    Returns:
        True if at least one event has the type
    """
    return event_type in collect_event_types(events)


def count_event_type(events: list[Any], event_type: str) -> int:
    """Count events of a specific type.

    Args:
        events: List of SDK events
        event_type: Event type to count

    Returns:
        Number of events with that type
    """
    return collect_event_types(events).count(event_type)
