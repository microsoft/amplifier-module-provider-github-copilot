"""
Regression tests for expert review bug fixes.

Contract: contracts/provider-protocol.md
"""

from pathlib import Path

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
    """AC-5: Verify finish_reason_map in event config."""

    def test_event_config_has_finish_reason_map(self):
        """EventConfig should have finish_reason_map field."""
        # Check the dataclass has the field
        import dataclasses

        from amplifier_module_provider_github_copilot.streaming import EventConfig

        field_names = [f.name for f in dataclasses.fields(EventConfig)]
        assert "finish_reason_map" in field_names, "EventConfig should have finish_reason_map field"

    def test_load_event_config_loads_finish_reason_map(self):
        """load_event_config should populate finish_reason_map."""
        from amplifier_module_provider_github_copilot.streaming import load_event_config

        result = load_event_config()

        assert hasattr(result, "finish_reason_map")
        assert result.finish_reason_map is not None
        # Values MUST be lowercase per amplifier-core proto
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
    """Verify session lifecycle config has required event types.

    Contract: streaming-contract:SessionLifecycle:MUST:1
    """

    def test_production_config_has_valid_session_lifecycle(self):
        """Production event config MUST have valid session_lifecycle config.

        Verifies the hardcoded config has the expected lifecycle event types.
        """
        from amplifier_module_provider_github_copilot.streaming import load_event_config

        config = load_event_config()

        # Verify session_lifecycle is populated
        assert config.idle_event_types, "idle_event_types must not be empty"
        assert "session.idle" in config.idle_event_types
        assert config.error_event_types, "error_event_types must not be empty"
        assert config.usage_event_types, "usage_event_types must not be empty"
