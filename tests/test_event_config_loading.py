"""Tests for defensive event config loading.

Contract: contracts/event-vocabulary.md - event config loading must produce valid domain event types

These tests verify that malformed events.yaml produces clear ConfigurationError messages
instead of cryptic KeyError tracebacks.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest
import yaml

from amplifier_module_provider_github_copilot.error_translation import ConfigurationError
from amplifier_module_provider_github_copilot.streaming import (
    DomainEventType,
    EventConfig,
    load_event_config,
)


class TestDefensiveEventConfigLoading:
    """Tests for Defensive event config loading."""

    def test_valid_config_loads_correctly(self):
        """Valid configuration loads without error.

        Regression guard: ensure defensive changes don't break valid configs.
        """
        # Contract: event-vocabulary:Bridge:MUST:2
        # Create a valid config file
        config_data = {
            "event_classifications": {
                "bridge": [
                    {"sdk_type": "assistant.message_delta", "domain_type": "CONTENT_DELTA"},
                    {"sdk_type": "error", "domain_type": "ERROR"},
                ],
                "consume": ["tool_use_start"],
                "drop": ["heartbeat"],
            },
            "finish_reason_map": {"stop": "STOP"},
            "session_lifecycle": {
                "idle_events": ["session.idle"],
                "error_events": [],
                "usage_events": [],
            },
        }

        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            f.flush()
            config_path = Path(f.name)

        try:
            config = load_event_config(config_path)

            assert isinstance(config, EventConfig)
            assert "assistant.message_delta" in config.bridge_mappings
            assert (
                config.bridge_mappings["assistant.message_delta"][0]
                == DomainEventType.CONTENT_DELTA
            )
        finally:
            config_path.unlink()

    def test_missing_sdk_type_raises_configuration_error(self):
        """Bridge entry missing sdk_type produces ConfigurationError with clear message.

        Missing sdk_type must not cause KeyError
        """
        # Contract: behaviors:ConfigLoading:MUST:2
        config_data = {
            "event_classifications": {
                "bridge": [
                    # Missing sdk_type key!
                    {"domain_type": "CONTENT_DELTA"}
                ],
            },
        }

        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            f.flush()
            config_path = Path(f.name)

        try:
            with pytest.raises(ConfigurationError) as exc_info:
                load_event_config(config_path)

            # Error message should mention the missing key
            assert "sdk_type" in str(exc_info.value)
        finally:
            config_path.unlink()

    def test_missing_domain_type_raises_configuration_error(self):
        """Bridge entry missing domain_type produces ConfigurationError with clear message.

        Missing domain_type must not cause KeyError
        """
        # Contract: behaviors:ConfigLoading:MUST:2
        config_data = {
            "event_classifications": {
                "bridge": [
                    # Missing domain_type key!
                    {"sdk_type": "assistant.message_delta"}
                ],
            },
        }

        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            f.flush()
            config_path = Path(f.name)

        try:
            with pytest.raises(ConfigurationError) as exc_info:
                load_event_config(config_path)

            # Error message should mention the missing key and the sdk_type
            error_msg = str(exc_info.value)
            assert "domain_type" in error_msg
            assert "assistant.message_delta" in error_msg
        finally:
            config_path.unlink()

    def test_unknown_domain_type_raises_configuration_error(self):
        """Bridge entry with unknown domain_type produces ConfigurationError.

        Unknown enum value must not cause cryptic KeyError
        """
        # Contract: behaviors:ConfigLoading:MUST:3
        config_data = {
            "event_classifications": {
                "bridge": [
                    {"sdk_type": "assistant.message_delta", "domain_type": "NONEXISTENT_TYPE"}
                ],
            },
        }

        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            f.flush()
            config_path = Path(f.name)

        try:
            with pytest.raises(ConfigurationError) as exc_info:
                load_event_config(config_path)

            error_msg = str(exc_info.value)
            # Should mention the unknown type and list valid types
            assert error_msg == (
                "Invalid event config in events.yaml: Bridge mapping 0 "
                "(sdk_type=assistant.message_delta) has unknown domain_type "
                "'NONEXISTENT_TYPE'. Valid types: ['CONTENT_DELTA', 'TOOL_CALL', "
                "'USAGE_UPDATE', 'TURN_COMPLETE', 'SESSION_IDLE', 'ERROR']"
            )
        finally:
            config_path.unlink()

    def test_error_message_includes_entry_index(self):
        """Error message includes the index of the problematic entry.

        Debugging aid: know which entry failed
        """
        # Contract: behaviors:ConfigLoading:MUST:2
        config_data = {
            "event_classifications": {
                "bridge": [
                    {
                        "sdk_type": "assistant.message_delta",
                        "domain_type": "CONTENT_DELTA",
                    },  # valid
                    {"sdk_type": "bad_entry"},  # invalid - missing domain_type (index 1)
                ],
            },
        }

        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            f.flush()
            config_path = Path(f.name)

        try:
            with pytest.raises(ConfigurationError) as exc_info:
                load_event_config(config_path)

            # The error should help locate the problem
            assert "bad_entry" in str(exc_info.value)
        finally:
            config_path.unlink()

    def test_empty_config_returns_default(self):
        """Empty config file returns default EventConfig.

        Regression guard: don't break graceful fallback behavior.
        """
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")  # Empty file
            f.flush()
            config_path = Path(f.name)

        try:
            config = load_event_config(config_path)
            assert isinstance(config, EventConfig)
            assert config.bridge_mappings == {}
        finally:
            config_path.unlink()

    def test_missing_file_returns_default(self):
        """Missing config file returns default EventConfig."""
        config = load_event_config("/nonexistent/path/events.yaml")
        assert isinstance(config, EventConfig)
        assert config.bridge_mappings == {}
