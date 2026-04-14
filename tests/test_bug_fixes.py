"""
Regression tests for expert review bug fixes.

Contract: contracts/provider-protocol.md
"""

# TestAC2DeadAsserts removed - the test was for completion.py which is now deleted.
# The production path (provider._execute_sdk_completion) uses CopilotClientWrapper
# which is a properly implemented context manager that never yields None.


class TestAC3RetryAfterRegex:
    """AC-3: Fix retry_after regex to not match unrelated strings."""

    def test_extract_retry_after_standard_format(self) -> None:
        """Retry-after standard form parses to float.

        # Contract: error-hierarchy:RateLimit:MUST:1
        """
        from amplifier_module_provider_github_copilot.error_translation import (
            _extract_retry_after,  # pyright: ignore[reportPrivateUsage]
        )

        result = _extract_retry_after("Rate limited. Retry after 30 seconds")
        assert result == 30.0

    def test_extract_retry_after_header_format(self) -> None:
        """Retry-after header parses to float.

        # Contract: error-hierarchy:RateLimit:MUST:1
        """
        from amplifier_module_provider_github_copilot.error_translation import (
            _extract_retry_after,  # pyright: ignore[reportPrivateUsage]
        )

        result = _extract_retry_after("retry-after: 60")
        assert result == 60.0

    def test_extract_retry_after_ignores_unrelated_seconds(self) -> None:
        """Should NOT match generic 'N seconds' without retry context.

        # Contract: error-hierarchy:RateLimit:MUST:1
        """
        from amplifier_module_provider_github_copilot.error_translation import (
            _extract_retry_after,  # pyright: ignore[reportPrivateUsage]
        )

        # This is an error message that happens to mention seconds
        # but is NOT a retry-after instruction
        result = _extract_retry_after("Operation timed out after 30 seconds")
        assert result is None, "Should not match 'N seconds' without retry context"

    def test_extract_retry_after_ignores_timestamp_in_message(self) -> None:
        """Should NOT match timestamps or durations in general error messages.

        # Contract: error-hierarchy:RateLimit:MUST:1
        """
        from amplifier_module_provider_github_copilot.error_translation import (
            _extract_retry_after,  # pyright: ignore[reportPrivateUsage]
        )

        result = _extract_retry_after("Request took 5 seconds and failed")
        assert result is None, "Should not match casual duration mentions"

    def test_extract_retry_after_decimal_format(self) -> None:
        """Retry-after header with decimal value parses correctly.

        # Contract: error-hierarchy:RateLimit:MUST:1
        """
        from amplifier_module_provider_github_copilot.error_translation import (
            _extract_retry_after,  # pyright: ignore[reportPrivateUsage]
        )

        result = _extract_retry_after("retry-after: 0.5")
        assert result == 0.5

    def test_extract_retry_after_without_seconds_suffix(self) -> None:
        """Retry-after plain number form (no 'seconds' word) parses correctly.

        # Contract: error-hierarchy:RateLimit:MUST:1
        """
        from amplifier_module_provider_github_copilot.error_translation import (
            _extract_retry_after,  # pyright: ignore[reportPrivateUsage]
        )

        result = _extract_retry_after("Retry after 30")
        assert result == 30.0


class TestAC5FinishReasonMap:
    """AC-5: Verify finish_reason_map in event config."""

    def test_load_event_config_loads_finish_reason_map(self) -> None:
        """load_event_config should populate finish_reason_map.

        # Contract: event-vocabulary:FinishReason:MUST:1
        """
        from amplifier_module_provider_github_copilot.streaming import load_event_config

        result = load_event_config()

        # Values MUST be lowercase per amplifier-core proto
        # Valid values: "stop", "tool_calls", "length", "content_filter"
        assert result.finish_reason_map.get("end_turn") == "stop"
        assert result.finish_reason_map.get("stop") == "stop"
        assert result.finish_reason_map.get("tool_use") == "tool_calls"

    def test_translate_event_uses_finish_reason_map(self) -> None:
        """translate_event should map finish reasons per config.

        Contract: sdk-boundary:Translation:MUST:1
        # Contract: event-vocabulary:FinishReason:MUST:1
        """
        from amplifier_module_provider_github_copilot.streaming import (
            DomainEvent,
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

        assert isinstance(domain_event, DomainEvent)
        # Should map "end_turn" to "STOP" per finish_reason_map
        assert domain_event.data["finish_reason"] == "STOP", (
            "Should map SDK finish_reason using finish_reason_map"
        )

    def test_translate_event_finish_reason_map_uses_default(self) -> None:
        """translate_event uses _default for unknown finish reasons.

        # Contract: event-vocabulary:FinishReason:MUST:1
        """
        from amplifier_module_provider_github_copilot.streaming import (
            DomainEvent,
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

        assert isinstance(domain_event, DomainEvent)
        assert domain_event.data["finish_reason"] == "UNKNOWN"


class TestSessionLifecycleValidation:
    """Verify session lifecycle config has required event types.

    Contract: streaming-contract:SessionLifecycle:MUST:1
    """

    def test_production_config_has_required_session_lifecycle_events(self) -> None:
        """Production event config MUST have valid session_lifecycle config.

        # Contract: streaming-contract:SessionLifecycle:MUST:1

        Verifies the hardcoded config has the expected lifecycle event types.
        """
        from amplifier_module_provider_github_copilot.streaming import load_event_config

        config = load_event_config()

        # Verify session_lifecycle is populated
        assert config.idle_event_types, "idle_event_types must not be empty"
        assert "session.idle" in config.idle_event_types
        assert "session.error" in config.error_event_types, (
            "Production config must include 'session.error' in error_event_types"
        )
        assert "assistant.usage" in config.usage_event_types, (
            "Production config must include 'assistant.usage' in usage_event_types"
        )

    def test_empty_idle_events_raises_configuration_error(self) -> None:
        """Provider raises ConfigurationError when idle_events is empty.

        # Contract: streaming-contract:SessionLifecycle:MUST:1
        """
        import tempfile
        from pathlib import Path

        import pytest
        from amplifier_core.llm_errors import ConfigurationError

        from amplifier_module_provider_github_copilot.streaming import load_event_config

        # Create a minimal YAML config with empty idle_events
        yaml_content = """
session_lifecycle:
  idle_events: []
  error_events:
    - session.error
  usage_events:
    - assistant.usage
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(yaml_content)
            temp_path = Path(f.name)

        try:
            with pytest.raises(ConfigurationError) as exc_info:
                load_event_config(temp_path)

            assert "idle_events" in str(exc_info.value)
        finally:
            temp_path.unlink()
