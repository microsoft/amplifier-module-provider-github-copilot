"""
Contract Compliance Tests: Event Vocabulary.

Contract: contracts/event-vocabulary.md

Tests event classification compliance.
"""

from __future__ import annotations


class TestDomainEventTypes:
    """event-vocabulary:Events:MUST:1 — Test domain event type definitions."""

    def test_domain_event_type_enum_exists(self) -> None:
        """DomainEventType enum must exist."""
        from amplifier_module_provider_github_copilot.streaming import DomainEventType

        # Should have the 6 domain events
        expected_types = {"CONTENT_DELTA", "TOOL_CALL", "USAGE_UPDATE", "TURN_COMPLETE", "ERROR"}

        actual_types = {e.name for e in DomainEventType}

        # At least these types should exist
        for expected in expected_types:
            assert expected in actual_types, f"Missing domain event type: {expected}"

    def test_domain_event_dataclass_exists(self) -> None:
        """DomainEvent dataclass must exist."""
        from amplifier_module_provider_github_copilot.streaming import DomainEvent, DomainEventType

        # Should be able to create a domain event
        event = DomainEvent(type=DomainEventType.CONTENT_DELTA, data={"text": "test"})
        assert event.type == DomainEventType.CONTENT_DELTA
        assert event.data["text"] == "test"


class TestEventTranslation:
    """event-vocabulary:Bridge:MUST:2 — Test event translation."""

    def test_translate_event_with_real_config(self) -> None:
        """P2-7 FIX: translate_event correctly bridges SDK events to domain events.

        Uses REAL config to verify actual production behavior, not empty config.
        """
        from amplifier_module_provider_github_copilot.streaming import (
            DomainEvent,
            DomainEventType,
            load_event_config,
            translate_event,
        )

        config = load_event_config()

        # Test assistant.message_delta → CONTENT_DELTA (concrete input/output)
        sdk_event = {"type": "assistant.message_delta", "data": {"delta_content": "test"}}
        result = translate_event(sdk_event, config)

        assert result is not None, "assistant.message_delta MUST be bridged"
        assert isinstance(result, DomainEvent)
        assert result.type == DomainEventType.CONTENT_DELTA

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
        assert result is not None, "assistant.message_delta should be bridged"
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

        assert result is not None, "assistant.usage should be bridged"
        assert isinstance(result, DomainEvent)
        # DomainEventType is USAGE_UPDATE (not USAGE)
        assert result.type == DomainEventType.USAGE_UPDATE, (
            f"Expected USAGE_UPDATE, got {result.type}"
        )

    def test_unknown_events_dropped(self) -> None:
        """event-vocabulary:Drop:MUST:1 — Unknown events should be dropped."""
        from amplifier_module_provider_github_copilot.streaming import (
            EventConfig,
            translate_event,
        )

        config = EventConfig()

        # Unknown event type should be dropped (return None)
        sdk_event = {"type": "completely_unknown_event_xyz"}
        result = translate_event(sdk_event, config)

        assert result is None, "Unknown events should be dropped (return None)"
