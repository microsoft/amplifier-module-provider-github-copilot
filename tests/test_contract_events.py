"""
Contract Compliance Tests: Event Vocabulary.

Contract: contracts/event-vocabulary.md

Tests event classification compliance.
"""

from __future__ import annotations

from pathlib import Path

import yaml


class TestEventConfigCompliance:
    """event-vocabulary:Events:MUST:1 — Verify config satisfies contract."""

    def _get_config_path(self) -> Path:
        """Get path to events.yaml in package config."""
        return (
            Path(__file__).parent.parent
            / "amplifier_module_provider_github_copilot"
            / "config"
            / "events.yaml"
        )

    def test_events_yaml_exists(self) -> None:
        """Config file must exist."""
        config_path = self._get_config_path()
        assert config_path.exists(), f"config/events.yaml must exist at {config_path}"

    def test_events_yaml_valid_yaml(self) -> None:
        """Config file must be valid YAML."""
        config_path = self._get_config_path()
        content = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        assert content is not None

    def test_has_bridge_events(self) -> None:
        """event-vocabulary:Bridge:MUST:1 — Must define BRIDGE events."""
        config_path = self._get_config_path()
        content = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        # Check for bridge classification
        has_bridge = (
            "bridge" in content.get("event_classifications", {})
            or any(
                item.get("classification") == "BRIDGE"
                for item in content.get("event_classifications", {}).get("bridge", [])
            )
            if isinstance(content.get("event_classifications"), dict)
            else False
        )

        # Alternative structure check
        if not has_bridge and isinstance(content.get("event_classifications"), dict):
            has_bridge = "bridge" in content["event_classifications"]

        assert has_bridge, "Must define BRIDGE event classifications"

    def test_has_drop_events(self) -> None:
        """event-vocabulary:Drop:MUST:1 — Must define DROP events."""
        config_path = self._get_config_path()
        content = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        if isinstance(content.get("event_classifications"), dict):
            has_drop = "drop" in content["event_classifications"]
            assert has_drop, "Must define DROP event classifications"

    def test_has_finish_reason_map(self) -> None:
        """event-vocabulary:FinishReason:MUST:1 — Must have finish_reason mapping."""
        config_path = self._get_config_path()
        content = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        assert "finish_reason_map" in content, "Must have finish_reason_map"

        finish_map = content["finish_reason_map"]

        # Should map SDK reasons to domain reasons
        assert "stop" in finish_map or "end_turn" in finish_map
        assert "_default" in finish_map, "Must have _default fallback"


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
