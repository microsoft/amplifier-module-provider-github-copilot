"""
Regression tests for expert review bug fixes.

Contract: contracts/provider-protocol.md
"""

from pathlib import Path


class TestAC1LoadEventConfigCrash:
    """AC-1: Fix load_event_config crash on missing file."""

    def test_load_event_config_missing_file_returns_default(self):
        """load_event_config with non-existent path returns default config."""
        from amplifier_module_provider_github_copilot.streaming import load_event_config

        result = load_event_config("/nonexistent/path/events.yaml")

        # Should return default config, not crash
        assert result is not None
        assert hasattr(result, "bridge_mappings")
        assert hasattr(result, "consume_patterns")
        assert hasattr(result, "drop_patterns")

    def test_load_event_config_missing_file_has_empty_defaults(self):
        """Default config should have empty collections."""
        from amplifier_module_provider_github_copilot.streaming import load_event_config

        result = load_event_config("/nonexistent/path/events.yaml")

        assert result.bridge_mappings == {}
        assert result.consume_patterns == []
        assert result.drop_patterns == []
        assert result.finish_reason_map == {}  # AC-5: verify finish_reason_map also empty

    def test_load_event_config_accepts_path_object(self):
        """load_event_config accepts Path objects (AC-1 type contract)."""
        from pathlib import Path

        from amplifier_module_provider_github_copilot.streaming import load_event_config

        # Should accept Path without error
        result = load_event_config(Path("/nonexistent/path/events.yaml"))

        assert result is not None
        assert result.bridge_mappings == {}


# TestAC2DeadAsserts removed - the test was for completion.py which is now deleted.
# The production path (provider._execute_sdk_completion) uses CopilotClientWrapper
# which is a properly implemented context manager that never yields None.


class TestAC3RetryAfterRegex:
    """AC-3: Fix retry_after regex to not match unrelated strings."""

    def test_extract_retry_after_standard_format(self):
        """Should extract from 'Retry after 30 seconds' format."""
        from amplifier_module_provider_github_copilot.error_translation import (
            _extract_retry_after,  # pyright: ignore[reportPrivateUsage]
        )

        result = _extract_retry_after("Rate limited. Retry after 30 seconds")
        assert result == 30.0

    def test_extract_retry_after_header_format(self):
        """Should extract from 'retry-after: 60' format."""
        from amplifier_module_provider_github_copilot.error_translation import (
            _extract_retry_after,  # pyright: ignore[reportPrivateUsage]
        )

        result = _extract_retry_after("retry-after: 60")
        assert result == 60.0

    def test_extract_retry_after_ignores_unrelated_seconds(self):
        """Should NOT match generic 'N seconds' without retry context."""
        from amplifier_module_provider_github_copilot.error_translation import (
            _extract_retry_after,  # pyright: ignore[reportPrivateUsage]
        )

        # This is an error message that happens to mention seconds
        # but is NOT a retry-after instruction
        result = _extract_retry_after("Operation timed out after 30 seconds")
        assert result is None, "Should not match 'N seconds' without retry context"

    def test_extract_retry_after_ignores_timestamp_in_message(self):
        """Should NOT match timestamps or durations in general error messages."""
        from amplifier_module_provider_github_copilot.error_translation import (
            _extract_retry_after,  # pyright: ignore[reportPrivateUsage]
        )

        result = _extract_retry_after("Request took 5 seconds and failed")
        assert result is None, "Should not match casual duration mentions"


class TestAC5FinishReasonMap:
    """AC-5: Load finish_reason_map from events.yaml."""

    def test_event_config_has_finish_reason_map(self):
        """EventConfig should have finish_reason_map field."""
        # Check the dataclass has the field
        import dataclasses

        from amplifier_module_provider_github_copilot.streaming import EventConfig

        field_names = [f.name for f in dataclasses.fields(EventConfig)]
        assert "finish_reason_map" in field_names, "EventConfig should have finish_reason_map field"

    def test_load_event_config_loads_finish_reason_map(self):
        """load_event_config should populate finish_reason_map from YAML."""
        from pathlib import Path

        from amplifier_module_provider_github_copilot.streaming import load_event_config

        config_path = (
            Path(__file__).parent.parent
            / "amplifier_module_provider_github_copilot"
            / "config"
            / "events.yaml"
        )

        result = load_event_config(str(config_path))

        assert hasattr(result, "finish_reason_map")
        assert result.finish_reason_map is not None
        # P2 Fix #3: YAML now uses lowercase values per amplifier-core proto
        # Valid values: "stop", "tool_calls", "length", "content_filter"
        assert result.finish_reason_map.get("end_turn") == "stop"
        assert result.finish_reason_map.get("stop") == "stop"
        assert result.finish_reason_map.get("tool_use") == "tool_calls"

    def test_translate_event_uses_finish_reason_map(self):
        """translate_event should map finish reasons per config."""
        from amplifier_module_provider_github_copilot.streaming import (
            DomainEventType,
            EventConfig,
            translate_event,
        )

        config = EventConfig(
            bridge_mappings={
                "message_complete": (DomainEventType.TURN_COMPLETE, None),
            },
            finish_reason_map={
                "end_turn": "STOP",
                "tool_use": "TOOL_USE",
                "_default": "ERROR",
            },
        )

        # Create SDK event with SDK finish_reason
        sdk_event = {"type": "message_complete", "finish_reason": "end_turn"}
        domain_event = translate_event(sdk_event, config)

        assert domain_event is not None
        # Should map "end_turn" to "STOP" per finish_reason_map
        assert domain_event.data["finish_reason"] == "STOP", (
            "Should map SDK finish_reason using finish_reason_map"
        )

    def test_translate_event_finish_reason_map_uses_default(self):
        """translate_event uses _default for unknown finish reasons."""
        from amplifier_module_provider_github_copilot.streaming import (
            DomainEventType,
            EventConfig,
            translate_event,
        )

        config = EventConfig(
            bridge_mappings={
                "message_complete": (DomainEventType.TURN_COMPLETE, None),
            },
            finish_reason_map={
                "end_turn": "STOP",
                "_default": "UNKNOWN",
            },
        )

        # Unknown finish_reason
        sdk_event = {"type": "message_complete", "finish_reason": "unknown_reason"}
        domain_event = translate_event(sdk_event, config)

        assert domain_event is not None
        assert domain_event.data["finish_reason"] == "UNKNOWN"


class TestAC6TombstoneFiles:
    """AC-6: Verify tombstone files are deleted."""

    def test_completion_tombstone_deleted(self):
        """completion.py tombstone should not exist."""
        tombstone = (
            Path(__file__).parent.parent
            / "src"
            / "amplifier_module_provider_github_copilot"
            / "completion.py"
        )
        # AC-6: File MUST be deleted, not just a tombstone
        assert not tombstone.exists(), "completion.py should be deleted (AC-6)"

    def test_session_factory_tombstone_deleted(self):
        """session_factory.py tombstone should not exist."""
        tombstone = (
            Path(__file__).parent.parent
            / "src"
            / "amplifier_module_provider_github_copilot"
            / "session_factory.py"
        )
        # AC-6: File MUST be deleted, not just a tombstone
        assert not tombstone.exists(), "session_factory.py should be deleted (AC-6)"


class TestSessionLifecycleValidation:
    """Fail-fast validation for session_lifecycle config.

    Contract: streaming-contract:SessionLifecycle:MUST:1

    session_lifecycle is CORE config, not observability:
    - If idle_events is empty, provider cannot detect session completion
    - This causes infinite hang with no error message
    - Developer experience: fail loudly at startup, not silently at runtime

    Bug discovery: Session hung for 4+ minutes because load_event_config
    returned empty sets, and is_idle_event() always returned False.
    """

    def test_missing_idle_events_raises_configuration_error(self, tmp_path: Path):
        """load_event_config MUST raise ConfigurationError if idle_events is empty.

        This is fail-fast at load time, not silent degradation.
        Developers need loud failures, not 4-minute debugging sessions.
        """
        import pytest
        from amplifier_core.llm_errors import ConfigurationError

        from amplifier_module_provider_github_copilot.streaming import load_event_config

        # Create events.yaml without session_lifecycle
        events_yaml = tmp_path / "events.yaml"
        events_yaml.write_text(
            """
event_classifications:
  bridge: []
  consume: []
  drop: []
""",
            encoding="utf-8",
        )

        with pytest.raises(ConfigurationError) as exc_info:
            load_event_config(str(events_yaml))

        assert "idle_events" in str(exc_info.value)
        assert "session_lifecycle" in str(exc_info.value)

    def test_empty_idle_events_raises_configuration_error(self, tmp_path: Path):
        """Explicitly empty idle_events list MUST also raise ConfigurationError."""
        import pytest
        from amplifier_core.llm_errors import ConfigurationError

        from amplifier_module_provider_github_copilot.streaming import load_event_config

        # Create events.yaml with empty idle_events
        events_yaml = tmp_path / "events.yaml"
        events_yaml.write_text(
            """
event_classifications:
  bridge: []
  consume: []
  drop: []
session_lifecycle:
  idle_events: []
  error_events: []
  usage_events: []
""",
            encoding="utf-8",
        )

        with pytest.raises(ConfigurationError) as exc_info:
            load_event_config(str(events_yaml))

        assert "idle_events" in str(exc_info.value)

    def test_production_events_yaml_has_valid_session_lifecycle(self):
        """Production events.yaml MUST have valid session_lifecycle config.

        This verifies our shipped YAML is correct and won't cause runtime failures.
        """
        from amplifier_module_provider_github_copilot.streaming import load_event_config

        config_path = (
            Path(__file__).parent.parent
            / "amplifier_module_provider_github_copilot"
            / "config"
            / "events.yaml"
        )

        # Should not raise - validates at load time
        config = load_event_config(str(config_path))

        # Verify session_lifecycle is populated
        assert config.idle_event_types, "idle_event_types must not be empty"
        assert "session.idle" in config.idle_event_types
        assert config.error_event_types, "error_event_types must not be empty"
        assert config.usage_event_types, "usage_event_types must not be empty"
