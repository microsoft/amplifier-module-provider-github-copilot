"""Tests for load_sdk_protection_config — Python defaults.

Migrated from YAML-based validation to Python-module-based validation.
The sdk_protection migration replaced config/sdk_protection.yaml with
config/sdk_protection.py (hardcoded defaults). load_sdk_protection_config()
now returns SdkProtectionConfig() directly — no YAML parsing.

Contract: sdk-protection:ToolCapture:MUST:1,2
Contract: sdk-protection:Session:MUST:3,4
Contract: sdk-protection:Subprocess:MUST:7
Contract: behaviors:ConfigLoading:MUST:6
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Singleton config defaults
# ---------------------------------------------------------------------------


class TestSingletonSectionValidation:
    """Validation: singleton config has correct defaults."""

    def test_singleton_config_has_defaults(self) -> None:
        """SingletonConfig has lock_timeout_seconds default.

        Contract: provider-protocol:mount:MUST:5
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            SingletonConfig,
            load_sdk_protection_config,
        )

        load_sdk_protection_config.cache_clear()
        config = load_sdk_protection_config()
        assert isinstance(config.singleton, SingletonConfig)
        assert config.singleton.lock_timeout_seconds > 0

    def test_singleton_lock_timeout_positive(self) -> None:
        """SingletonConfig lock_timeout_seconds is positive.

        Contract: provider-protocol:mount:MUST:5
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_sdk_protection_config,
        )

        load_sdk_protection_config.cache_clear()
        config = load_sdk_protection_config()
        assert config.singleton.lock_timeout_seconds >= 1.0

    def test_lock_timeout_is_float(self) -> None:
        """lock_timeout_seconds is accessible and valid.

        Contract: provider-protocol:mount:MUST:5 — timeout is defined
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_sdk_protection_config,
        )

        load_sdk_protection_config.cache_clear()
        config = load_sdk_protection_config()
        assert isinstance(config.singleton.lock_timeout_seconds, float)


# ---------------------------------------------------------------------------
# SDK section key validation
# ---------------------------------------------------------------------------


class TestSdkSectionKeyValidation:
    """SDK config attributes have correct default values."""

    def test_sdk_defaults(self) -> None:
        """SDK config has all required keys with correct defaults.

        Contract: sdk-protection:Subprocess:MUST:7
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_sdk_protection_config,
        )

        load_sdk_protection_config.cache_clear()
        config = load_sdk_protection_config()
        assert config.sdk.log_level == "info"
        assert config.sdk.log_level_env_var == "COPILOT_SDK_LOG_LEVEL"
        assert config.sdk.prewarm_subprocess is False
        assert set(config.sdk.valid_log_levels) == {
            "none",
            "error",
            "warning",
            "info",
            "debug",
            "all",
        }


# ---------------------------------------------------------------------------
# valid_log_levels validation
# ---------------------------------------------------------------------------


class TestValidLogLevelsValidation:
    """valid_log_levels must be a list with expected entries."""

    def test_valid_log_levels_is_list(self) -> None:
        """sdk.valid_log_levels is a list (not string).

        Contract: sdk-protection:Subprocess:MUST:7
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_sdk_protection_config,
        )

        load_sdk_protection_config.cache_clear()
        config = load_sdk_protection_config()
        assert isinstance(config.sdk.valid_log_levels, list)

    def test_log_level_in_allowlist(self) -> None:
        """sdk.log_level must be in valid_log_levels allowlist.

        Contract: sdk-protection:Subprocess:MUST:7 — Validate log_level against allowlist
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_sdk_protection_config,
        )

        load_sdk_protection_config.cache_clear()
        config = load_sdk_protection_config()
        assert config.sdk.log_level in config.sdk.valid_log_levels


# ---------------------------------------------------------------------------
# prewarm_subprocess boolean validation
# ---------------------------------------------------------------------------


class TestPrewarmBooleanValidation:
    """prewarm_subprocess must be a literal boolean."""

    def test_prewarm_is_bool_not_string(self) -> None:
        """sdk.prewarm_subprocess is a bool, not a string.

        Contract: behaviors:ConfigLoading:MUST:6 — Reject string booleans
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_sdk_protection_config,
        )

        load_sdk_protection_config.cache_clear()
        config = load_sdk_protection_config()
        assert isinstance(config.sdk.prewarm_subprocess, bool)
        assert not isinstance(config.sdk.prewarm_subprocess, str)

    def test_prewarm_is_literal_bool(self) -> None:
        """sdk.prewarm_subprocess is not an integer.

        Contract: behaviors:ConfigLoading:MUST:6 — Reject string booleans
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_sdk_protection_config,
        )

        load_sdk_protection_config.cache_clear()
        config = load_sdk_protection_config()
        # bool is a subclass of int in Python, but we want the value to be False/True not 0/1
        assert config.sdk.prewarm_subprocess in (True, False)
        assert type(config.sdk.prewarm_subprocess) is bool

    def test_prewarm_as_literal_bool_true_accepted(self) -> None:
        """sdk.prewarm_subprocess has a default bool value.

        Contract: behaviors:ConfigLoading:MUST:6 — Bool is OK
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_sdk_protection_config,
        )

        load_sdk_protection_config.cache_clear()
        config = load_sdk_protection_config()
        # Default is False per sdk_protection.py
        assert config.sdk.prewarm_subprocess is False


# ---------------------------------------------------------------------------
# Tool capture section key validation
# ---------------------------------------------------------------------------


class TestToolCaptureSectionKeyValidation:
    """Tool capture config attributes have correct default values."""

    def test_tool_capture_defaults(self) -> None:
        """ToolCaptureConfig has all required attributes with correct defaults.

        Contract: sdk-protection:ToolCapture:MUST:1,2
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_sdk_protection_config,
        )

        load_sdk_protection_config.cache_clear()
        config = load_sdk_protection_config()
        assert config.tool_capture.first_turn_only is True
        assert config.tool_capture.deduplicate is True
        assert config.tool_capture.log_capture_events is True


# ---------------------------------------------------------------------------
# Session section key validation
# ---------------------------------------------------------------------------


class TestSessionSectionKeyValidation:
    """Session config attributes have correct default values."""

    def test_session_defaults(self) -> None:
        """SessionProtectionConfig has all required attributes with correct defaults.

        Contract: sdk-protection:Session:MUST:3,4
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_sdk_protection_config,
        )

        load_sdk_protection_config.cache_clear()
        config = load_sdk_protection_config()
        assert config.session.explicit_abort is True
        assert config.session.abort_timeout_seconds == 5.0
        assert config.session.idle_timeout_seconds == 30.0
        assert config.session.disconnect_timeout_seconds == 30.0


# ---------------------------------------------------------------------------
# Top-level section validation
# ---------------------------------------------------------------------------


class TestTopLevelSectionValidation:
    """tool_capture, session, sdk, singleton sections are all present."""

    def test_tool_capture_has_correct_type(self) -> None:
        """SdkProtectionConfig has tool_capture attribute."""
        from amplifier_module_provider_github_copilot.config_loader import (
            SdkProtectionConfig,
            ToolCaptureConfig,
            load_sdk_protection_config,
        )

        load_sdk_protection_config.cache_clear()
        config = load_sdk_protection_config()
        assert isinstance(config, SdkProtectionConfig)
        assert isinstance(config.tool_capture, ToolCaptureConfig)

    def test_session_has_correct_type(self) -> None:
        """SdkProtectionConfig has session attribute."""
        from amplifier_module_provider_github_copilot.config_loader import (
            SessionProtectionConfig,
            load_sdk_protection_config,
        )

        load_sdk_protection_config.cache_clear()
        config = load_sdk_protection_config()
        assert isinstance(config.session, SessionProtectionConfig)

    def test_sdk_has_correct_type(self) -> None:
        """SdkProtectionConfig has sdk attribute."""
        from amplifier_module_provider_github_copilot.config_loader import (
            SdkConfig,
            load_sdk_protection_config,
        )

        load_sdk_protection_config.cache_clear()
        config = load_sdk_protection_config()
        assert isinstance(config.sdk, SdkConfig)

    def test_top_level_config_valid(self) -> None:
        """load_sdk_protection_config() returns valid SdkProtectionConfig."""
        from amplifier_module_provider_github_copilot.config_loader import (
            SdkConfig,
            SdkProtectionConfig,
            SessionProtectionConfig,
            SingletonConfig,
            ToolCaptureConfig,
            load_sdk_protection_config,
        )

        load_sdk_protection_config.cache_clear()
        config = load_sdk_protection_config()
        assert isinstance(config, SdkProtectionConfig)
        # All required subsections present with correct types
        assert isinstance(config.tool_capture, ToolCaptureConfig)
        assert isinstance(config.session, SessionProtectionConfig)
        assert isinstance(config.sdk, SdkConfig)
        assert isinstance(config.singleton, SingletonConfig)
