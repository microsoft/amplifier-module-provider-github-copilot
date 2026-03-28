"""SDK Event Field Extraction - Unified extraction logic.

Contract: contracts/sdk-boundary.md (SDK Event Structure v0.1.33+)

This module provides a single extraction function used by both
completion.py (test path) and provider.py (real SDK path).

Change here when SDK event structure changes.
"""

from typing import Any, cast

# Fields to exclude from extraction (envelope fields, not content)
_ENVELOPE_FIELDS = frozenset({"id", "timestamp", "parent_id"})

# Common top-level attributes to extract via getattr
_COMMON_ATTRS = (
    "type",
    "text",
    "name",
    "arguments",
    "finish_reason",
    "input_tokens",
    "output_tokens",
    "total_tokens",
)

# Usage-related fields in nested data
_USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "finish_reason",
)


def extract_event_fields(sdk_event: Any) -> dict[str, Any]:
    """Extract all relevant fields from an SDK event object.

    Contract: sdk-boundary:EventShape:MUST:1-3

    SDK v0.1.33+ uses nested event.data structure:
    - event.data.delta_content for streaming text
    - event.data.reasoning_text for thinking events
    - event.data.tool_call_id, tool_name, arguments for tool events
    - event.data.input_tokens, output_tokens, total_tokens for usage

    Args:
        sdk_event: SDK SessionEvent object (or mock with same structure)

    Returns:
        Dict with extracted fields ready for translate_event()

    """
    event_dict: dict[str, Any] = {}

    # Step 1: Extract all public attributes from SDK event object
    if hasattr(sdk_event, "__dict__"):
        raw_vars = cast(dict[str, Any], vars(sdk_event))
        # Filter envelope fields that conflict with data fields
        # (SessionEvent.id is event UUID, not tool_call_id)
        event_dict = {
            k: v for k, v in raw_vars.items() if not k.startswith("_") and k not in _ENVELOPE_FIELDS
        }

    # Step 2: Extract common event fields via getattr
    for attr in _COMMON_ATTRS:
        if attr not in event_dict and hasattr(sdk_event, attr):
            event_dict[attr] = getattr(sdk_event, attr)

    # Step 3: SDK v0.1.33+ nested data extraction
    sdk_data = getattr(sdk_event, "data", None)
    if sdk_data is not None:
        # delta_content contains streaming text chunks
        delta_content = getattr(sdk_data, "delta_content", None)
        if delta_content and "text" not in event_dict:
            event_dict["text"] = delta_content

        # reasoning_text for thinking/reasoning events
        reasoning_text = getattr(sdk_data, "reasoning_text", None)
        if reasoning_text and "reasoning_text" not in event_dict:
            event_dict["reasoning_text"] = reasoning_text

        # Tool event fields (SDK uses tool_call_id, tool_name)
        # Map to domain's "id" and "name" for translate_event
        tool_call_id = getattr(sdk_data, "tool_call_id", None)
        if tool_call_id and "id" not in event_dict:
            event_dict["id"] = tool_call_id

        tool_name = getattr(sdk_data, "tool_name", None)
        if tool_name and "name" not in event_dict:
            event_dict["name"] = tool_name

        # Tool arguments — MUST preserve empty dict for zero-parameter tools
        # Contract: streaming-contract.md — tools MUST have arguments field
        # Bug fix: `if arguments` dropped {} because bool({}) == False
        arguments = getattr(sdk_data, "arguments", None)
        if arguments is not None and "arguments" not in event_dict:
            event_dict["arguments"] = arguments

        # Usage fields from nested data
        for usage_field in _USAGE_FIELDS:
            field_val = getattr(sdk_data, usage_field, None)
            if field_val is not None and usage_field not in event_dict:
                event_dict[usage_field] = field_val

    # Fallback: if object itself has tool_call_id (no nested data), map to id
    # This handles SessionEventData being passed directly to extract_event_fields
    if "id" not in event_dict:
        tool_call_id = getattr(sdk_event, "tool_call_id", None)
        if tool_call_id:
            event_dict["id"] = tool_call_id

    if "name" not in event_dict:
        tool_name = getattr(sdk_event, "tool_name", None)
        if tool_name:
            event_dict["name"] = tool_name

    return event_dict
