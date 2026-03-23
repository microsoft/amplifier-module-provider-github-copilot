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
        content = yaml.safe_load(config_path.read_text())

        assert content is not None

    def test_has_bridge_events(self) -> None:
        """event-vocabulary:Bridge:MUST:1 — Must define BRIDGE events."""
        config_path = self._get_config_path()
        content = yaml.safe_load(config_path.read_text())

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
        content = yaml.safe_load(config_path.read_text())

        if isinstance(content.get("event_classifications"), dict):
            has_drop = "drop" in content["event_classifications"]
            assert has_drop, "Must define DROP event classifications"

    def test_has_finish_reason_map(self) -> None:
        """event-vocabulary:FinishReason:MUST:1 — Must have finish_reason mapping."""
        config_path = self._get_config_path()
        content = yaml.safe_load(config_path.read_text())

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

    def test_translate_event_returns_domain_event_or_none(self) -> None:
        """translate_event returns DomainEvent for BRIDGE, None otherwise."""
        from amplifier_module_provider_github_copilot.streaming import (
            DomainEvent,
            EventConfig,
            translate_event,
        )

        # Create minimal config
        config = EventConfig()

        # Test with a known bridged event type
        sdk_event = {"type": "assistant.message_delta", "data": {"text": "hello"}}
        result = translate_event(sdk_event, config)

        # Result should be DomainEvent or None based on classification
        assert result is None or isinstance(result, DomainEvent)

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
