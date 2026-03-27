"""Tests for SDK protection config loading.

Contract: sdk-protection:ToolCapture:MUST:1,2
Contract: sdk-protection:Session:MUST:3,4
"""

from __future__ import annotations


class TestSdkProtectionConfigLoading:
    """Test loading of SDK protection config from YAML."""

    def test_load_sdk_protection_config_returns_dataclass(self) -> None:
        """Config loader returns SdkProtectionConfig dataclass."""
        from amplifier_module_provider_github_copilot.config_loader import (
            SdkProtectionConfig,
            load_sdk_protection_config,
        )

        config = load_sdk_protection_config()

        assert isinstance(config, SdkProtectionConfig)
        assert hasattr(config, "tool_capture")
        assert hasattr(config, "session")

    def test_tool_capture_config_has_expected_fields(self) -> None:
        """ToolCaptureConfig has first_turn_only, deduplicate, log_capture_events.

        Contract: sdk-protection:ToolCapture:MUST:1,2
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_sdk_protection_config,
        )

        config = load_sdk_protection_config()

        assert hasattr(config.tool_capture, "first_turn_only")
        assert hasattr(config.tool_capture, "deduplicate")
        assert hasattr(config.tool_capture, "log_capture_events")

        # Verify types
        assert isinstance(config.tool_capture.first_turn_only, bool)
        assert isinstance(config.tool_capture.deduplicate, bool)
        assert isinstance(config.tool_capture.log_capture_events, bool)

    def test_session_config_has_expected_fields(self) -> None:
        """SessionConfig has explicit_abort, abort_timeout_seconds, idle_timeout_seconds.

        Contract: sdk-protection:Session:MUST:3,4
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_sdk_protection_config,
        )

        config = load_sdk_protection_config()

        assert hasattr(config.session, "explicit_abort")
        assert hasattr(config.session, "abort_timeout_seconds")
        assert hasattr(config.session, "idle_timeout_seconds")

        # Verify types
        assert isinstance(config.session.explicit_abort, bool)
        assert isinstance(config.session.abort_timeout_seconds, float)
        assert isinstance(config.session.idle_timeout_seconds, float)

    def test_yaml_config_has_expected_defaults(self) -> None:
        """YAML config has expected default values.

        Three-Medium: YAML is authoritative source, no Python fallbacks.
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_sdk_protection_config,
        )

        config = load_sdk_protection_config()

        # Tool capture defaults from YAML
        assert config.tool_capture.first_turn_only is True
        assert config.tool_capture.deduplicate is True
        assert config.tool_capture.log_capture_events is True

        # Session defaults from YAML
        assert config.session.explicit_abort is True
        assert config.session.abort_timeout_seconds == 5.0
        assert config.session.idle_timeout_seconds == 30.0


class TestSdkProtectionConfigValues:
    """Test that loaded config has correct values."""

    def test_tool_capture_defaults_match_contract(self) -> None:
        """Tool capture values match contracts/sdk-protection.md.

        Contract: sdk-protection:ToolCapture:MUST:1 (first_turn_only)
        Contract: sdk-protection:ToolCapture:MUST:2 (deduplicate)
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_sdk_protection_config,
        )

        config = load_sdk_protection_config()

        # Per contract, these should be true by default
        assert config.tool_capture.first_turn_only is True
        assert config.tool_capture.deduplicate is True

    def test_session_abort_timeout_is_reasonable(self) -> None:
        """Abort timeout is reasonable (not too short, not too long).

        Contract: sdk-protection:Session:MUST:4
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_sdk_protection_config,
        )

        config = load_sdk_protection_config()

        # 1-60 seconds is reasonable for abort timeout
        assert 1.0 <= config.session.abort_timeout_seconds <= 60.0

    def test_idle_timeout_is_reasonable(self) -> None:
        """Idle timeout is reasonable for abort/cleanup operations.

        NOTE: This config is NOT used for main SDK wait (uses caller's timeout).
        SDK API calls can take 60+ seconds for complex operations like delegation.
        This value is retained for abort/cleanup operations only.

        Contract: sdk-protection:Session:SHOULD:2
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_sdk_protection_config,
        )

        config = load_sdk_protection_config()

        # 10-300 seconds is reasonable for idle safety timeout
        assert 10.0 <= config.session.idle_timeout_seconds <= 300.0


class TestSdkProtectionYamlExists:
    """Test that the YAML config file exists and is valid."""

    def test_sdk_protection_yaml_exists(self) -> None:
        """config/sdk_protection.yaml exists in package."""
        from pathlib import Path

        config_path = (
            Path(__file__).parent.parent
            / "amplifier_module_provider_github_copilot"
            / "config"
            / "sdk_protection.yaml"
        )
        assert config_path.exists(), f"SDK protection config not found at {config_path}"

    def test_sdk_protection_yaml_is_valid(self) -> None:
        """config/sdk_protection.yaml parses without error."""
        from pathlib import Path

        import yaml

        config_path = (
            Path(__file__).parent.parent
            / "amplifier_module_provider_github_copilot"
            / "config"
            / "sdk_protection.yaml"
        )

        with config_path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert data is not None
        assert "tool_capture" in data
        assert "session" in data
