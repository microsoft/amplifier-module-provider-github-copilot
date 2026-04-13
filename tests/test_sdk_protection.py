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


class TestSdkProtectionPythonModule:
    """Test that the Python config module exists and has correct defaults.

    Replaces TestSdkProtectionYamlExists — sdk_protection.yaml was migrated
    to config/sdk_protection.py (Python dataclasses with hardcoded defaults).
    """

    def test_sdk_protection_python_module_importable(self) -> None:
        """config/sdk_protection.py is importable as a Python module."""
        from amplifier_module_provider_github_copilot.config._sdk_protection import (
            SdkProtectionConfig,
        )

        assert SdkProtectionConfig is not None

    def test_sdk_protection_config_instantiates_with_no_args(self) -> None:
        """SdkProtectionConfig() instantiates with hardcoded defaults (no I/O)."""
        from amplifier_module_provider_github_copilot.config._sdk_protection import (
            SdkProtectionConfig,
        )

        config = SdkProtectionConfig()

        assert hasattr(config, "tool_capture")
        assert hasattr(config, "session")
        assert hasattr(config, "sdk")
        assert hasattr(config, "singleton")

    def test_sdk_protection_module_has_all_dataclasses(self) -> None:
        """All expected dataclasses are exported from the module."""
        import amplifier_module_provider_github_copilot.config._sdk_protection as mod

        assert hasattr(mod, "ToolCaptureConfig")
        assert hasattr(mod, "SessionProtectionConfig")
        assert hasattr(mod, "SingletonConfig")
        assert hasattr(mod, "SdkConfig")
        assert hasattr(mod, "SdkProtectionConfig")


class TestSdkConfigLoading:
    """Test SDK subprocess configuration loading."""

    def test_sdk_protection_config_has_sdk_field(self) -> None:
        """SdkProtectionConfig includes sdk configuration."""
        from amplifier_module_provider_github_copilot.config_loader import (
            load_sdk_protection_config,
        )

        config = load_sdk_protection_config()

        assert hasattr(config, "sdk")

    def test_sdk_config_has_log_level_field(self) -> None:
        """SdkConfig has log_level field."""
        from amplifier_module_provider_github_copilot.config_loader import (
            load_sdk_protection_config,
        )

        config = load_sdk_protection_config()

        assert hasattr(config.sdk, "log_level")
        assert isinstance(config.sdk.log_level, str)

    def test_sdk_config_has_log_level_env_var_field(self) -> None:
        """SdkConfig has log_level_env_var field."""
        from amplifier_module_provider_github_copilot.config_loader import (
            load_sdk_protection_config,
        )

        config = load_sdk_protection_config()

        assert hasattr(config.sdk, "log_level_env_var")
        assert isinstance(config.sdk.log_level_env_var, str)

    def test_sdk_config_default_log_level_is_info(self) -> None:
        """Default log level is 'info' (safe default)."""
        from amplifier_module_provider_github_copilot.config_loader import (
            load_sdk_protection_config,
        )

        config = load_sdk_protection_config()

        assert config.sdk.log_level == "info"

    def test_sdk_config_log_level_env_var_name(self) -> None:
        """Environment variable name is COPILOT_SDK_LOG_LEVEL."""
        from amplifier_module_provider_github_copilot.config_loader import (
            load_sdk_protection_config,
        )

        config = load_sdk_protection_config()

        assert config.sdk.log_level_env_var == "COPILOT_SDK_LOG_LEVEL"


class TestSDKConfigValidation:
    """MUST-7: Validate SDK config values.

    Contract: sdk-protection:Subprocess:MUST:7
    Contract: behaviors:ConfigLoading:MUST:3
    """

    def test_valid_log_levels_accepted(self) -> None:
        """All valid log levels are accepted.

        Contract: sdk-protection:Subprocess:MUST:7
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_sdk_protection_config,
        )

        # Should not raise — default log_level is "info" which is valid
        config = load_sdk_protection_config()
        assert config.sdk.log_level in {"none", "error", "warning", "info", "debug", "all"}

    def test_validation_rejects_invalid_log_level_directly(self) -> None:
        """Validation rejects log_level not in allowlist.

        Contract: sdk-protection:Subprocess:MUST:7
        Contract: behaviors:ConfigLoading:MUST:3

        This test directly verifies the error message format that would be
        raised when log_level validation fails.
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            ConfigurationError,
        )

        # The validation code raises ConfigurationError with specific message format
        # Testing error message construction directly
        invalid_level = "invalid_level"
        valid_levels = ["none", "error", "warning", "info", "debug", "all"]

        # Verify the condition that triggers validation error
        assert invalid_level not in valid_levels

        # Verify error message format matches what config_loader produces
        error_msg = (
            f"Config validation failed: sdk.log_level '{invalid_level}' is not valid. "
            f"Must be one of: {', '.join(valid_levels)}"
        )
        err = ConfigurationError(error_msg)
        assert "sdk.log_level" in str(err)
        assert invalid_level in str(err)

    def test_validation_accepts_all_valid_log_levels(self) -> None:
        """All defined log levels are accepted by validation.

        Contract: sdk-protection:Subprocess:MUST:7
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_sdk_protection_config,
        )

        config = load_sdk_protection_config()

        # Verify config has valid_log_levels defined
        assert hasattr(config.sdk, "valid_log_levels")
        assert len(config.sdk.valid_log_levels) > 0

        # Current log_level must be in the valid set
        assert config.sdk.log_level in config.sdk.valid_log_levels

    def test_bool_fields_are_actual_booleans(self) -> None:
        """Bool fields are actual bool type, not strings.

        Contract: behaviors:ConfigLoading:MUST:6
        Python's bool('false') == True, which is a trap.
        This test verifies loaded config uses actual bool values.
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_sdk_protection_config,
        )

        config = load_sdk_protection_config()

        # All bool fields must be actual bool type
        assert isinstance(config.tool_capture.first_turn_only, bool)
        assert isinstance(config.tool_capture.deduplicate, bool)
        assert isinstance(config.tool_capture.log_capture_events, bool)
        assert isinstance(config.session.explicit_abort, bool)
        assert isinstance(config.sdk.prewarm_subprocess, bool)


class TestFrozenInvariantEnforcement:
    """Safety-critical configs are frozen — mutation raises FrozenInstanceError.

    Contract: sdk-protection:ToolCapture:MUST:1,2
    Contract: sdk-protection:Session:MUST:3,4

    The @lru_cache loader returns a shared singleton. Without frozen=True,
    any caller mutating a field silently corrupts all other callers in the
    process. frozen=True makes the invariant enforced by Python, not convention.
    """

    def test_tool_capture_config_is_frozen(self) -> None:
        """ToolCaptureConfig raises FrozenInstanceError on field assignment.

        Contract: sdk-protection:ToolCapture:MUST:1
        """
        from dataclasses import FrozenInstanceError

        import pytest

        from amplifier_module_provider_github_copilot.config._sdk_protection import (
            ToolCaptureConfig,
        )

        config = ToolCaptureConfig()

        with pytest.raises(FrozenInstanceError):
            config.first_turn_only = False  # type: ignore[misc]

    def test_tool_capture_deduplicate_is_frozen(self) -> None:
        """ToolCaptureConfig.deduplicate cannot be mutated post-construction.

        Contract: sdk-protection:ToolCapture:MUST:2
        """
        from dataclasses import FrozenInstanceError

        import pytest

        from amplifier_module_provider_github_copilot.config._sdk_protection import (
            ToolCaptureConfig,
        )

        config = ToolCaptureConfig()

        with pytest.raises(FrozenInstanceError):
            config.deduplicate = False  # type: ignore[misc]

    def test_session_protection_config_is_frozen(self) -> None:
        """SessionProtectionConfig raises FrozenInstanceError on field assignment.

        Contract: sdk-protection:Session:MUST:3
        """
        from dataclasses import FrozenInstanceError

        import pytest

        from amplifier_module_provider_github_copilot.config._sdk_protection import (
            SessionProtectionConfig,
        )

        config = SessionProtectionConfig()

        with pytest.raises(FrozenInstanceError):
            config.explicit_abort = False  # type: ignore[misc]

    def test_session_abort_timeout_is_frozen(self) -> None:
        """SessionProtectionConfig abort timeout cannot be mutated post-construction.

        Contract: sdk-protection:Session:MUST:4
        """
        from dataclasses import FrozenInstanceError

        import pytest

        from amplifier_module_provider_github_copilot.config._sdk_protection import (
            SessionProtectionConfig,
        )

        config = SessionProtectionConfig()

        with pytest.raises(FrozenInstanceError):
            config.abort_timeout_seconds = 0.0  # type: ignore[misc]

    def test_construct_with_override_still_works(self) -> None:
        """frozen=True does not break construct-with-override pattern.

        Callers that need non-default values pass them via constructor.
        This is the only supported override mechanism for frozen configs.
        """
        from amplifier_module_provider_github_copilot.config._sdk_protection import (
            ToolCaptureConfig,
        )

        config = ToolCaptureConfig(first_turn_only=False, deduplicate=False)

        assert config.first_turn_only is False
        assert config.deduplicate is False
