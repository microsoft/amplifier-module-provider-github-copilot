"""Coverage tests for streaming.py missing branches.

Covers:
- Line 569: classify_event returns DROP for unknown event type (+ warning)
- Line 591: _extract_event_data with data object having __dict__ (not dict)

Contract: event-vocabulary:Classification:MUST:1 — each event has exactly one classification
Contract: streaming-contract:SessionLifecycle:MUST:1 — session lifecycle events
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from amplifier_module_provider_github_copilot.streaming import DomainEventType

# ---------------------------------------------------------------------------
# classify_event unknown type → DROP (line 569)
# ---------------------------------------------------------------------------


class TestClassifyEventUnknownType:
    """classify_event returns DROP and logs warning for unknown event types."""

    def test_unknown_event_type_returns_drop(self) -> None:
        """Event type not in any classification list returns EventClassification.DROP.

        Line ~569 in streaming.py — fallback DROP with warning
        Contract: event-vocabulary:Classification:MUST:1 — unknown → DROP
        """
        from amplifier_module_provider_github_copilot.streaming import (
            EventClassification,
            EventConfig,
            classify_event,
        )

        # Use minimal EventConfig with nothing matching the unknown type
        config = EventConfig(
            bridge_mappings=cast(
                dict[str, tuple[DomainEventType, str | None]],
                {"session.idle": (DomainEventType.SESSION_IDLE, None)},
            ),
            consume_patterns=["assistant.usage"],
            drop_patterns=["system.*"],
        )

        result = classify_event("completely.unknown.event.type", config)

        # Unknown events fall back to DROP
        assert result is EventClassification.DROP

    def test_unknown_event_type_logs_warning(self, caplog: Any) -> None:
        """Unknown event type logs a warning message.

        Line ~569 in streaming.py — logger.warning("Unknown SDK event type")
        """
        import logging

        from amplifier_module_provider_github_copilot.streaming import EventConfig, classify_event

        config = EventConfig(
            bridge_mappings={},
            consume_patterns=[],
            drop_patterns=[],
        )

        with caplog.at_level(logging.WARNING):
            classify_event("mystery.event.type", config)

        assert any("Unknown SDK event type" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# _extract_event_data with data object having __dict__ (line 591)
# ---------------------------------------------------------------------------


class TestExtractEventDataWithObjectData:
    """_extract_event_data handles data objects with __dict__ via extract_event_fields."""

    def test_data_object_with_dict_attribute_is_flattened(self) -> None:
        """data field that is an object with __dict__ is extracted via extract_event_fields.

        Line ~591 in streaming.py — elif hasattr(v, '__dict__'):
        Contract: event-vocabulary:Bridge:MUST:3
        """
        from amplifier_module_provider_github_copilot.streaming import (
            _extract_event_data,  # pyright: ignore[reportPrivateUsage]
        )

        @dataclass
        class SessionEventData:
            """Simulates SDK SessionEventData object with __dict__."""

            delta_content: str | None = "Hello from SDK"
            finish_reason: str | None = None
            input_tokens: int | None = None

        # SDK event dict with data as an object (not a plain dict)
        sdk_event = {
            "type": "assistant.message_delta",
            "data": SessionEventData(delta_content="Hello from SDK"),
        }

        result = _extract_event_data(sdk_event)

        # Contract: sdk-boundary:EventShape:MUST:2
        # Data object's delta_content is extracted
        assert result == {"delta_content": "Hello from SDK"}

    def test_data_dict_is_flattened_normally(self) -> None:
        """data field as dict uses the normal dict path (not the __dict__ branch)."""
        from amplifier_module_provider_github_copilot.streaming import (
            _extract_event_data,  # pyright: ignore[reportPrivateUsage]
        )

        sdk_event = {
            "type": "assistant.message_delta",
            "data": {"delta_content": "from dict", "finish_reason": None},
        }

        result = _extract_event_data(sdk_event)

        # Contract: sdk-boundary:EventShape:MUST:2
        assert result["delta_content"] == "from dict"

    def test_non_data_keys_are_preserved(self) -> None:
        """Non-'data', non-'type' keys in SDK event are passed through."""
        from amplifier_module_provider_github_copilot.streaming import (
            _extract_event_data,  # pyright: ignore[reportPrivateUsage]
        )

        sdk_event = {
            "type": "session.idle",  # excluded
            "session_id": "sess-123",  # should be preserved
            "timestamp": 12345,  # should be preserved
        }

        result = _extract_event_data(sdk_event)

        assert result["session_id"] == "sess-123"
        assert result["timestamp"] == 12345
        assert "type" not in result


# ---------------------------------------------------------------------------
# classify_event — DROP pattern matched (line 569)
# ---------------------------------------------------------------------------


class TestClassifyEventDropPatternMatch:
    """classify_event returns DROP when event type matches a drop_pattern."""

    def test_event_matching_drop_pattern_returns_drop(self) -> None:
        """Event type that matches a drop_pattern returns EventClassification.DROP.

        Line 569 in streaming.py — return EventClassification.DROP (pattern matched)
        Contract: event-vocabulary:Classification:MUST:1

        This is DISTINCT from the "unknown event" fallthrough at lines 570-571:
        here the event type explicitly matches a wildcard drop_pattern.
        """
        from amplifier_module_provider_github_copilot.streaming import (
            EventClassification,
            EventConfig,
            classify_event,
        )

        config = EventConfig(
            bridge_mappings={},
            consume_patterns=[],
            drop_patterns=["debug.*"],  # wildcard matches "debug.verbose"
        )

        result = classify_event("debug.verbose", config)

        assert result is EventClassification.DROP


class TestSDKVersionSkewDropEvents:
    """New SDK events from CLI version-skew are explicitly classified as DROP.

    Contract: event-vocabulary:Drop:MUST:2
    """

    def test_session_custom_agents_updated_is_dropped_silently(self, caplog: Any) -> None:
        """session.custom_agents_updated (SDK v0.2.1+) is classified as DROP without warning.

        Contract: event-vocabulary:Drop:MUST:2

        Introduced in SDK v0.2.1 CLI binary. Represents an internal session-state
        notification with no domain value. Must be silently dropped — not cause
        an 'Unknown SDK event type' warning that pollutes production logs.
        """
        import logging

        from amplifier_module_provider_github_copilot.streaming import (
            EventClassification,
            classify_event,
            load_event_config,
        )

        config = load_event_config()

        with caplog.at_level(logging.WARNING):
            result = classify_event("session.custom_agents_updated", config)

        assert result is EventClassification.DROP
        assert not any("Unknown SDK event type" in record.message for record in caplog.records), (
            "session.custom_agents_updated must be silently dropped, not logged as unknown"
        )


# ---------------------------------------------------------------------------
# _extract_event_data — primitive data field (line 591 elif branch → False)
# ---------------------------------------------------------------------------


class TestExtractEventDataPrimitiveData:
    """_extract_event_data silently skips data field when it is a primitive."""

    def test_primitive_data_field_is_silently_skipped(self) -> None:
        """data field that is a primitive (not dict, no __dict__) is silently dropped.

        Line 591 in streaming.py — elif hasattr(v, '__dict__'): evaluates False
        Contract: event-vocabulary:Bridge:MUST:3 — graceful handling

        Both the isinstance(v, dict) and hasattr(v, '__dict__') conditions are False
        for a primitive (e.g., int 42), so neither branch fires and data is skipped.
        """
        from amplifier_module_provider_github_copilot.streaming import (
            _extract_event_data,  # pyright: ignore[reportPrivateUsage]
        )

        # A plain integer has no __dict__ and is not a dict — both branches skip
        sdk_event = {
            "type": "assistant.usage",
            "data": 42,  # primitive: not dict, no __dict__
        }

        result = _extract_event_data(sdk_event)

        # Primitive data is silently skipped — no crash, no output from 'data' key
        assert isinstance(result, dict)
        assert "data" not in result  # data field was skipped, not forwarded raw
