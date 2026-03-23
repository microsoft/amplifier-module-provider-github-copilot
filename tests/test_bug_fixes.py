"""
Regression tests for expert review bug fixes.

Contract: contracts/provider-protocol.md
"""

from pathlib import Path

import pytest


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


class TestAC2DeadAsserts:
    """AC-2: Remove dead assert statements."""

    def test_complete_with_none_session_raises_proper_error(self):
        """complete() with sdk_create_fn returning None raises ProviderUnavailableError."""
        import asyncio
        from typing import Any

        from amplifier_module_provider_github_copilot.error_translation import (
            ProviderUnavailableError,
        )
        from amplifier_module_provider_github_copilot.provider import (
            CompletionRequest,
            complete,
        )
        from amplifier_module_provider_github_copilot.sdk_adapter.types import SessionConfig

        async def broken_create_fn(_config: SessionConfig) -> Any:
            return None

        async def run_test() -> None:
            events: list[Any] = []
            # AC-2: Should raise ProviderUnavailableError, NOT AssertionError
            with pytest.raises(ProviderUnavailableError) as exc_info:
                async for event in complete(
                    CompletionRequest(prompt="test"),
                    sdk_create_fn=broken_create_fn,
                ):
                    events.append(event)

            # Verify it's the correct error with proper message
            assert "SDK session factory returned None" in str(exc_info.value)
            assert exc_info.value.provider == "github-copilot"

        asyncio.run(run_test())


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
        # Check expected mappings from events.yaml
        assert result.finish_reason_map.get("end_turn") == "STOP"
        assert result.finish_reason_map.get("stop") == "STOP"
        assert result.finish_reason_map.get("tool_use") == "TOOL_USE"

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
