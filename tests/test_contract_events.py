"""
Contract Compliance Tests: Event Vocabulary.

Contract: contracts/event-vocabulary.md

Tests event classification compliance.
"""

from __future__ import annotations

# event-vocabulary:Events:MUST:1 defines exactly 6 domain events
_EXPECTED_DOMAIN_EVENT_COUNT = 6


class TestDomainEventTypes:
    """event-vocabulary:Events:MUST:1 — Test domain event type definitions."""

    def test_domain_event_type_enum_exists(self) -> None:
        """DomainEventType enum must exist with exactly 6 members.

        Contract: event-vocabulary:Events:MUST:1
        """
        from amplifier_module_provider_github_copilot.streaming import DomainEventType

        expected_names = {
            "CONTENT_DELTA",
            "TOOL_CALL",
            "USAGE_UPDATE",
            "TURN_COMPLETE",
            "SESSION_IDLE",
            "ERROR",
        }
        actual_names = {e.name for e in DomainEventType}
        assert actual_names == expected_names, (
            f"DomainEventType members mismatch. Expected {expected_names}, got {actual_names}"
        )
        assert len(DomainEventType) == _EXPECTED_DOMAIN_EVENT_COUNT, (
            f"Expected {_EXPECTED_DOMAIN_EVENT_COUNT} DomainEventType members, "
            f"got {len(DomainEventType)}"
        )

    def test_domain_event_dataclass_exists(self) -> None:
        """DomainEvent dataclass must exist."""
        from amplifier_module_provider_github_copilot.streaming import DomainEvent, DomainEventType

        # Should be able to create a domain event
        event = DomainEvent(type=DomainEventType.CONTENT_DELTA, data={"text": "test"})
        assert event.type == DomainEventType.CONTENT_DELTA
        assert event.data["text"] == "test"


class TestEventTranslation:
    """event-vocabulary:Bridge:MUST:2 — Test event translation."""

    def test_content_delta_mapped_correctly(self) -> None:
        """P2-7: Concrete assertion for content delta translation.

        event-vocabulary:Bridge:MUST:2 — assistant.message_delta → CONTENT_DELTA
        """
        from amplifier_module_provider_github_copilot.streaming import (
            DomainEvent,
            DomainEventType,
            load_event_config,
            translate_event,
        )

        # Use load_event_config() to get actual config with bridge mappings
        config = load_event_config()
        sdk_event = {"type": "assistant.message_delta", "data": {"delta_content": "Hello"}}
        result = translate_event(sdk_event, config)

        # P2-7: Verify SPECIFIC mapping, not just type
        assert isinstance(result, DomainEvent)
        assert result.type == DomainEventType.CONTENT_DELTA, (
            f"Expected CONTENT_DELTA, got {result.type}"
        )

    def test_usage_event_mapped_correctly(self) -> None:
        """P2-7: Concrete assertion for usage event translation.

        event-vocabulary:Bridge:MUST:2 — assistant.usage → USAGE
        """
        from amplifier_module_provider_github_copilot.streaming import (
            DomainEvent,
            DomainEventType,
            load_event_config,
            translate_event,
        )

        # Use load_event_config() to get actual config with bridge mappings
        config = load_event_config()
        sdk_event = {
            "type": "assistant.usage",
            "data": {"input_tokens": 10, "output_tokens": 20},
        }
        result = translate_event(sdk_event, config)

        assert isinstance(result, DomainEvent)
        # DomainEventType is USAGE_UPDATE (not USAGE)
        assert result.type == DomainEventType.USAGE_UPDATE, (
            f"Expected USAGE_UPDATE, got {result.type}"
        )


class TestEventDrop:
    """event-vocabulary:Drop:MUST:1 — Unknown SDK events must be dropped."""

    def test_unknown_events_dropped(self) -> None:
        """Unknown event types must classify as DROP and return None.

        Contract: event-vocabulary:Drop:MUST:1
        """
        from amplifier_module_provider_github_copilot.streaming import (
            EventClassification,
            classify_event,
            load_event_config,
            translate_event,
        )

        # Contract: event-vocabulary:Drop:MUST:1
        config = load_event_config()
        classification = classify_event("completely_unknown_event_xyz", config)
        assert classification == EventClassification.DROP, (
            f"Unknown events must classify as DROP, got {classification}"
        )
        result = translate_event({"type": "completely_unknown_event_xyz"}, config)
        assert result is None, "Unknown events must return None from translate_event"


class TestBridgeEventDataShape:
    """event-vocabulary:Bridge:MUST:3 — BRIDGE event data must be flat and serialisable.

    Verifies that translate_event() promotes all SDK event data fields to the top level
    of DomainEvent.data, leaving no nested 'data' sub-key and no non-serialisable SDK
    objects. Required for StreamingAccumulator and JSON logging to work correctly.
    """

    def test_nested_dict_data_promoted_to_top_level(self) -> None:
        """BRIDGE event with nested dict 'data' must have all fields at the top level.

        Contract: event-vocabulary:Bridge:MUST:3
        """
        from amplifier_module_provider_github_copilot.streaming import (
            DomainEvent,
            DomainEventType,
            load_event_config,
            translate_event,
        )

        config = load_event_config()
        sdk_event = {
            "type": "assistant.message_delta",
            "data": {"delta_content": "Hello"},
        }
        result = translate_event(sdk_event, config)

        assert isinstance(result, DomainEvent)
        assert result.type == DomainEventType.CONTENT_DELTA
        # MUST:3 — nested 'data' key must be consumed, not passed through
        assert "data" not in result.data, (
            "translate_event MUST NOT leave a nested 'data' key in DomainEvent.data"
        )
        # MUST:3 — nested fields must be promoted to the top level
        assert result.data["delta_content"] == "Hello", (
            "translate_event MUST promote nested data fields to the top level of DomainEvent.data"
        )

    def test_usage_token_fields_at_top_level(self) -> None:
        """USAGE_UPDATE DomainEvent.data must have token fields at the top level.

        Contract: event-vocabulary:Bridge:MUST:3
        Rationale: StreamingAccumulator stores event.data directly as self.usage;
        token fields buried under a nested 'data' key produce silent zero usage counts.
        """
        from amplifier_module_provider_github_copilot.streaming import (
            DomainEvent,
            DomainEventType,
            load_event_config,
            translate_event,
        )

        config = load_event_config()
        sdk_event = {
            "type": "assistant.usage",
            "data": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        }
        result = translate_event(sdk_event, config)

        assert isinstance(result, DomainEvent)
        assert result.type == DomainEventType.USAGE_UPDATE
        # MUST:3 — no residual 'data' key; all token fields at top level
        assert "data" not in result.data, (
            "translate_event MUST NOT leave a nested 'data' key in DomainEvent.data"
        )
        assert result.data["input_tokens"] == 10, (
            "translate_event MUST promote nested usage token fields to the top level"
        )
        assert result.data["output_tokens"] == 20
        assert result.data["total_tokens"] == 30

    def test_sdk_object_in_data_not_preserved(self) -> None:
        """When sdk_event contains a raw SDK data object, it must not appear in DomainEvent.data
        and its fields must be promoted to the top level.

        Contract: event-vocabulary:Bridge:MUST:3
        Rationale: EventRouter queues raw SessionEvent objects; extract_event_fields()
        preserves the .data attribute (SessionEventData) in the returned dict.
        _extract_event_data() must strip it so DomainEvent.data remains a plain dict.
        """
        from amplifier_module_provider_github_copilot.streaming import (
            DomainEvent,
            DomainEventType,
            load_event_config,
            translate_event,
        )

        class _FakeSessionEventData:
            """Minimal stand-in for SDK SessionEventData (has __dict__, not JSON-serialisable)."""

            def __init__(self) -> None:
                self.delta_content = "from object"
                self.reasoning_opaque = None

        config = load_event_config()
        # Simulate the dict that extract_event_fields() produces for a real SessionEvent:
        # flat 'text' field already extracted plus the raw .data object still present.
        sdk_event = {
            "type": "assistant.message_delta",
            "text": "streamed text",
            "data": _FakeSessionEventData(),
        }
        result = translate_event(sdk_event, config)

        assert isinstance(result, DomainEvent)
        assert result.type == DomainEventType.CONTENT_DELTA
        # MUST:3 — raw SDK object must be consumed, not passed into DomainEvent.data
        assert "data" not in result.data, (
            "translate_event MUST remove the raw SDK data object from DomainEvent.data"
        )
        assert result.data["text"] == "streamed text"
        # MUST:3 — fields from the SDK object must be promoted to the top level
        assert result.data["delta_content"] == "from object", (
            "translate_event MUST promote fields from SDK data objects to the top "
            "level of DomainEvent.data"
        )

    def test_sdk_object_with_data_attr_does_not_leak_data_key(self) -> None:
        """When a promoted SDK data object itself has a .data attribute, the 'data'
        key must not appear in DomainEvent.data.

        Contract: event-vocabulary:Bridge:MUST:3
        Rationale: extract_event_fields() returns all non-envelope attributes of an
        SDK object.  If the SDK ever adds a .data field to SessionEventData (a
        realistic SDK evolution), the promotion loop must exclude it to honour the
        invariant that 'data' is never a key in DomainEvent.data.
        """
        from amplifier_module_provider_github_copilot.streaming import (
            DomainEvent,
            load_event_config,
            translate_event,
        )

        class _SDKObjectWithDataAttr:
            """Simulates a future SDK SessionEventData that carries its own .data attribute.
            Instance attributes (via __init__) are used so that vars() sees them — class-level
            attrs are invisible to extract_event_fields() which uses vars(obj).
            """

            def __init__(self) -> None:
                self.delta_content = "from nested object"
                self.data = {"should": "never leak"}  # would violate MUST:3 if promoted

        config = load_event_config()
        sdk_event = {"type": "assistant.streaming_delta", "data": _SDKObjectWithDataAttr()}
        result = translate_event(sdk_event, config)

        assert isinstance(result, DomainEvent)
        # MUST:3 — the residual 'data' attribute on the nested object must be excluded
        assert "data" not in result.data, (
            "translate_event MUST NOT allow a .data attribute on the SDK data object "
            "to appear as a 'data' key in DomainEvent.data"
        )
        # The promoted field should still arrive at the top level
        assert result.data.get("delta_content") == "from nested object", (
            "translate_event MUST still promote other fields from the SDK data object"
        )

    def test_none_data_omitted_from_domain_event(self) -> None:
        """When sdk_event has data=None, the 'data' key must not appear in DomainEvent.data.

        Contract: event-vocabulary:Bridge:MUST:3
        Rationale: session.idle events may carry data=None; propagating it as
        {"data": None} in DomainEvent.data creates a spurious key that pollutes
        downstream consumers and breaks serialisation assumptions.
        """
        from amplifier_module_provider_github_copilot.streaming import (
            DomainEvent,
            DomainEventType,
            load_event_config,
            translate_event,
        )

        config = load_event_config()
        sdk_event = {"type": "session.idle", "data": None}
        result = translate_event(sdk_event, config)

        assert isinstance(result, DomainEvent)
        assert result.type == DomainEventType.TURN_COMPLETE
        # MUST:3 — data=None must not propagate into DomainEvent.data
        assert "data" not in result.data, (
            "translate_event MUST NOT propagate data=None as a key in DomainEvent.data"
        )
