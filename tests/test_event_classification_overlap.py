"""Tests for event classification overlap validation.

Contract: contracts/event-vocabulary.md -- event classification must be unambiguous.

The problem: If the same event type appears in both bridge mappings and consume/drop
patterns, BRIDGE wins silently. This is a config error that should be detected at load time.
"""

from pathlib import Path

import pytest

from amplifier_module_provider_github_copilot.streaming import load_event_config


class TestEventClassificationOverlapValidation:
    """Validate no overlap between BRIDGE, CONSUME, and DROP categories."""

    def test_overlapping_bridge_and_consume_raises_error(self, tmp_path: Path) -> None:
        """Overlap between bridge and consume must raise ConfigurationError.

        Contract: event-vocabulary:Classification:MUST:1 -- each event type has exactly one
        classification.
        """
        config_yaml = tmp_path / "events.yaml"
        config_yaml.write_text(
            """
event_classifications:
  bridge:
    - sdk_type: assistant.message_delta
      domain_type: CONTENT_DELTA
      block_type: TEXT
  consume:
    - assistant.message_delta  # OVERLAP -- same as bridge entry
  drop: []
"""
        )

        with pytest.raises(Exception) as exc_info:
            load_event_config(config_yaml)

        # Should be ConfigurationError with details about the conflict
        assert "overlap" in str(exc_info.value).lower() or "assistant.message_delta" in str(
            exc_info.value
        )

    def test_overlapping_bridge_and_drop_raises_error(self, tmp_path: Path) -> None:
        """Overlap between bridge and drop must raise ConfigurationError.

        Contract: event-vocabulary:Classification:MUST:1
        """
        config_yaml = tmp_path / "events.yaml"
        config_yaml.write_text(
            """
event_classifications:
  bridge:
    - sdk_type: error
      domain_type: ERROR
  consume: []
  drop:
    - error  # OVERLAP -- same as bridge entry
"""
        )

        with pytest.raises(Exception) as exc_info:
            load_event_config(config_yaml)

        assert "overlap" in str(exc_info.value).lower() or "error" in str(exc_info.value)

    def test_overlapping_consume_and_drop_raises_error(self, tmp_path: Path) -> None:
        """Overlap between consume and drop must raise ConfigurationError.

        Contract: event-vocabulary:Classification:MUST:1
        """
        config_yaml = tmp_path / "events.yaml"
        config_yaml.write_text(
            """
event_classifications:
  bridge: []
  consume:
    - session_created
  drop:
    - session_created  # OVERLAP -- same as consume entry
"""
        )

        with pytest.raises(Exception) as exc_info:
            load_event_config(config_yaml)

        assert "overlap" in str(exc_info.value).lower() or "session_created" in str(exc_info.value)

    def test_wildcard_overlap_detected(self, tmp_path: Path) -> None:
        """Wildcard patterns that overlap with bridge must be detected.

        Contract: event-vocabulary:Classification:MUST:1
        """
        config_yaml = tmp_path / "events.yaml"
        config_yaml.write_text(
            """
event_classifications:
  bridge:
    - sdk_type: tool_result_success
      domain_type: TOOL_CALL
  consume: []
  drop:
    - tool_result_*  # WILDCARD -- matches tool_result_success in bridge
"""
        )

        with pytest.raises(Exception) as exc_info:
            load_event_config(config_yaml)

        assert "overlap" in str(exc_info.value).lower() or "tool_result" in str(exc_info.value)

    def test_clean_config_loads_without_error(self, tmp_path: Path) -> None:
        """Clean config (no overlaps) loads successfully.

        Regression test to ensure validation doesn't break valid configs.
        """
        config_yaml = tmp_path / "events.yaml"
        config_yaml.write_text(
            """
event_classifications:
  bridge:
    - sdk_type: assistant.message_delta
      domain_type: CONTENT_DELTA
      block_type: TEXT
    - sdk_type: error
      domain_type: ERROR
  consume:
    - session_created
    - session_destroyed
  drop:
    - heartbeat
    - debug_*
"""
        )

        # Should NOT raise
        config = load_event_config(config_yaml)

        assert "assistant.message_delta" in config.bridge_mappings
        assert "session_created" in config.consume_patterns
        assert "heartbeat" in config.drop_patterns

    def test_production_config_loads_without_error(self) -> None:
        """Production config/events.yaml has no overlaps.

        Regression test for actual production config.
        """
        # Should NOT raise -- production config should be valid
        config = load_event_config()

        # Verify config loaded properly
        assert len(config.bridge_mappings) > 0
        assert len(config.consume_patterns) > 0
        assert len(config.drop_patterns) > 0

    def test_error_message_includes_conflicting_entry(self, tmp_path: Path) -> None:
        """Error message must include details about which entry conflicts.

        Contract: error-hierarchy:Context:SHOULD:1 -- errors include actionable context.
        """
        config_yaml = tmp_path / "events.yaml"
        config_yaml.write_text(
            """
event_classifications:
  bridge:
    - sdk_type: usage_update
      domain_type: USAGE_UPDATE
  consume:
    - usage_update
  drop: []
"""
        )

        with pytest.raises(Exception) as exc_info:
            load_event_config(config_yaml)

        error_msg = str(exc_info.value).lower()
        # Error should identify the conflicting event type
        assert "usage_update" in error_msg
        # And ideally which categories conflict
        assert "bridge" in error_msg or "consume" in error_msg
