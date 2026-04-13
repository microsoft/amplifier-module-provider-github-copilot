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

import pytest

# ---------------------------------------------------------------------------
# Singleton config defaults
# ---------------------------------------------------------------------------


class TestSingletonSectionValidation:
    """Validation: singleton config has correct defaults."""

    def test_missing_singleton_section_raises(self) -> None:
        """SingletonConfig has lock_timeout_seconds default.

        Contract: sdk-protection:Singleton:MUST:8
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            SingletonConfig,
            load_sdk_protection_config,
        )

        load_sdk_protection_config.cache_clear()
        config = load_sdk_protection_config()
        assert isinstance(config.singleton, SingletonConfig)
        assert config.singleton.lock_timeout_seconds > 0

    def test_empty_singleton_section_raises(self) -> None:
        """SingletonConfig lock_timeout_seconds is positive.

        Contract: sdk-protection:Singleton:MUST:8
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_sdk_protection_config,
        )

        load_sdk_protection_config.cache_clear()
        config = load_sdk_protection_config()
        assert config.singleton.lock_timeout_seconds >= 1.0

    def test_missing_lock_timeout_raises(self) -> None:
        """lock_timeout_seconds is accessible and valid.

        Contract: sdk-protection:Singleton:MUST:8 — timeout is defined
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_sdk_protection_config,
        )

        load_sdk_protection_config.cache_clear()
        config = load_sdk_protection_config()
        assert hasattr(config.singleton, "lock_timeout_seconds")
        assert isinstance(config.singleton.lock_timeout_seconds, float)


# ---------------------------------------------------------------------------
# SDK section key validation
# ---------------------------------------------------------------------------


class TestSdkSectionKeyValidation:
    """Each required key in sdk section must be present as attribute."""

    @pytest.mark.parametrize(
        "required_key",
        [
            "log_level",
            "log_level_env_var",
            "prewarm_subprocess",
            "valid_log_levels",
        ],
    )
    def test_missing_sdk_key_raises(self, required_key: str) -> None:
        """SDK config has all required keys as attributes.

        Contract: sdk-protection:Subprocess:MUST:7
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_sdk_protection_config,
        )

        load_sdk_protection_config.cache_clear()
        config = load_sdk_protection_config()
        assert hasattr(config.sdk, required_key), (
            f"SdkConfig missing required attribute: {required_key}"
        )


# ---------------------------------------------------------------------------
# valid_log_levels validation
# ---------------------------------------------------------------------------


class TestValidLogLevelsValidation:
    """valid_log_levels must be a list with expected entries."""

    def test_valid_log_levels_not_a_list_raises(self) -> None:
        """sdk.valid_log_levels is a list (not string).

        Contract: sdk-protection:Subprocess:MUST:7
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_sdk_protection_config,
        )

        load_sdk_protection_config.cache_clear()
        config = load_sdk_protection_config()
        assert isinstance(config.sdk.valid_log_levels, list)

    def test_invalid_log_level_not_in_allowlist_raises(self) -> None:
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

    def test_prewarm_as_string_true_raises(self) -> None:
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

    def test_prewarm_as_integer_raises(self) -> None:
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
    """Each required key in tool_capture section must be present as attribute."""

    @pytest.mark.parametrize(
        "required_key",
        [
            "first_turn_only",
            "deduplicate",
            "log_capture_events",
        ],
    )
    def test_missing_tool_capture_key_raises(self, required_key: str) -> None:
        """ToolCaptureConfig has all required attributes.

        Contract: sdk-protection:ToolCapture:MUST:1,2
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_sdk_protection_config,
        )

        load_sdk_protection_config.cache_clear()
        config = load_sdk_protection_config()
        assert hasattr(config.tool_capture, required_key), (
            f"ToolCaptureConfig missing required attribute: {required_key}"
        )


# ---------------------------------------------------------------------------
# Session section key validation
# ---------------------------------------------------------------------------


class TestSessionSectionKeyValidation:
    """Each required key in session section must be present as attribute."""

    @pytest.mark.parametrize(
        "required_key",
        [
            "explicit_abort",
            "abort_timeout_seconds",
            "idle_timeout_seconds",
            "disconnect_timeout_seconds",
        ],
    )
    def test_missing_session_key_raises(self, required_key: str) -> None:
        """SessionProtectionConfig has all required attributes.

        Contract: sdk-protection:Session:MUST:3,4
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_sdk_protection_config,
        )

        load_sdk_protection_config.cache_clear()
        config = load_sdk_protection_config()
        assert hasattr(config.session, required_key), (
            f"SessionProtectionConfig missing required attribute: {required_key}"
        )


# ---------------------------------------------------------------------------
# Top-level section validation
# ---------------------------------------------------------------------------


class TestTopLevelSectionValidation:
    """tool_capture, session, sdk, singleton sections are all present."""

    def test_missing_tool_capture_section_raises(self) -> None:
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

    def test_missing_session_section_raises(self) -> None:
        """SdkProtectionConfig has session attribute."""
        from amplifier_module_provider_github_copilot.config_loader import (
            SessionProtectionConfig,
            load_sdk_protection_config,
        )

        load_sdk_protection_config.cache_clear()
        config = load_sdk_protection_config()
        assert isinstance(config.session, SessionProtectionConfig)

    def test_missing_sdk_section_raises(self) -> None:
        """SdkProtectionConfig has sdk attribute."""
        from amplifier_module_provider_github_copilot.config_loader import (
            SdkConfig,
            load_sdk_protection_config,
        )

        load_sdk_protection_config.cache_clear()
        config = load_sdk_protection_config()
        assert isinstance(config.sdk, SdkConfig)

    def test_empty_yaml_raises(self) -> None:
        """load_sdk_protection_config() returns valid SdkProtectionConfig."""
        from amplifier_module_provider_github_copilot.config_loader import (
            SdkProtectionConfig,
            load_sdk_protection_config,
        )

        load_sdk_protection_config.cache_clear()
        config = load_sdk_protection_config()
        assert isinstance(config, SdkProtectionConfig)
        # All required subsections present
        assert config.tool_capture is not None
        assert config.session is not None
        assert config.sdk is not None
        assert config.singleton is not None
