"""Tests for load_sdk_protection_config validation branches.

Coverage target: config_loader.py lines 461, 470-571
(singleton section, sdk section key validation branches)

Contract: sdk-protection:ToolCapture:MUST:1,2
Contract: sdk-protection:Session:MUST:3,4
Contract: sdk-protection:Subprocess:MUST:7
Contract: behaviors:ConfigLoading:MUST:6
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_valid_yaml() -> dict[str, Any]:
    """Minimal valid sdk_protection.yaml data for testing."""
    return {
        "tool_capture": {
            "first_turn_only": True,
            "deduplicate": True,
            "log_capture_events": True,
        },
        "session": {
            "explicit_abort": True,
            "abort_timeout_seconds": 5,
            "idle_timeout_seconds": 30,
            "disconnect_timeout_seconds": 10,
        },
        "singleton": {
            "lock_timeout_seconds": 30,
        },
        "sdk": {
            "log_level": "error",
            "log_level_env_var": "COPILOT_SDK_LOG_LEVEL",
            "prewarm_subprocess": False,
            "valid_log_levels": ["none", "error", "warning", "info", "debug", "all"],
        },
    }


def _load_fresh(yaml_data: dict[str, Any]) -> Any:
    """Clear the lru_cache, patch yaml.safe_load, call load_sdk_protection_config."""
    from amplifier_module_provider_github_copilot.config_loader import load_sdk_protection_config

    load_sdk_protection_config.cache_clear()
    try:
        with patch(
            "amplifier_module_provider_github_copilot.config_loader.yaml.safe_load",
            return_value=yaml_data,
        ):
            return load_sdk_protection_config()
    finally:
        load_sdk_protection_config.cache_clear()


def _load_fresh_raises(yaml_data: dict[str, Any] | None) -> pytest.ExceptionInfo[Any]:
    """Expect ConfigurationError when loading the given data."""
    from amplifier_module_provider_github_copilot.config_loader import load_sdk_protection_config
    from amplifier_module_provider_github_copilot.error_translation import ConfigurationError

    load_sdk_protection_config.cache_clear()
    try:
        with patch(
            "amplifier_module_provider_github_copilot.config_loader.yaml.safe_load",
            return_value=yaml_data,
        ):
            with pytest.raises(ConfigurationError) as exc_info:
                load_sdk_protection_config()
        return exc_info
    finally:
        load_sdk_protection_config.cache_clear()


# ---------------------------------------------------------------------------
# Singleton section validation (lines ~461-470)
# ---------------------------------------------------------------------------


class TestSingletonSectionValidation:
    """Validation: singleton section is required."""

    def test_missing_singleton_section_raises(self) -> None:
        """Missing 'singleton' section raises ConfigurationError.

        Contract: sdk-protection:Singleton:MUST:8
        """
        data = _base_valid_yaml()
        del data["singleton"]

        exc_info = _load_fresh_raises(data)
        assert "singleton" in str(exc_info.value).lower()

    def test_empty_singleton_section_raises(self) -> None:
        """Empty 'singleton' section raises ConfigurationError.

        Contract: sdk-protection:Singleton:MUST:8
        """
        data = _base_valid_yaml()
        data["singleton"] = {}  # empty dict is falsy → "missing singleton section"

        exc_info = _load_fresh_raises(data)
        assert "singleton" in str(exc_info.value).lower()

    def test_missing_lock_timeout_raises(self) -> None:
        """Missing 'singleton.lock_timeout_seconds' raises ConfigurationError.

        Contract: sdk-protection:Singleton:MUST:8 — timeout sourced from YAML
        """
        data = _base_valid_yaml()
        data["singleton"] = {"other_key": 99}  # no lock_timeout_seconds

        exc_info = _load_fresh_raises(data)
        assert "lock_timeout_seconds" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# SDK section key validation (lines ~470-510)
# ---------------------------------------------------------------------------


class TestSdkSectionKeyValidation:
    """Each required key in sdk section must be present."""

    @pytest.mark.parametrize(
        "missing_key",
        [
            "log_level",
            "log_level_env_var",
            "prewarm_subprocess",
            "valid_log_levels",
        ],
    )
    def test_missing_sdk_key_raises(self, missing_key: str) -> None:
        """Missing sdk key raises ConfigurationError.

        Contract: sdk-protection:Subprocess:MUST:7
        """
        data = _base_valid_yaml()
        del data["sdk"][missing_key]

        exc_info = _load_fresh_raises(data)
        assert missing_key in str(exc_info.value)


# ---------------------------------------------------------------------------
# valid_log_levels validation (lines ~510-530)
# ---------------------------------------------------------------------------


class TestValidLogLevelsValidation:
    """valid_log_levels must be a list."""

    def test_valid_log_levels_not_a_list_raises(self) -> None:
        """When sdk.valid_log_levels is a string (not list) raises ConfigurationError.

        Contract: sdk-protection:Subprocess:MUST:7
        """
        data = _base_valid_yaml()
        data["sdk"]["valid_log_levels"] = "none,error,warning"  # string, not list

        exc_info = _load_fresh_raises(data)
        assert "valid_log_levels" in str(exc_info.value).lower()
        assert "list" in str(exc_info.value).lower()

    def test_invalid_log_level_not_in_allowlist_raises(self) -> None:
        """When sdk.log_level is not in valid_log_levels raises ConfigurationError.

        Contract: sdk-protection:Subprocess:MUST:7 — Validate log_level against allowlist
        """
        data = _base_valid_yaml()
        data["sdk"]["log_level"] = "verbose"  # not in the list

        exc_info = _load_fresh_raises(data)
        assert "log_level" in str(exc_info.value).lower()
        assert "verbose" in str(exc_info.value)


# ---------------------------------------------------------------------------
# prewarm_subprocess boolean validation (lines ~545-560)
# ---------------------------------------------------------------------------


class TestPrewarmBooleanValidation:
    """prewarm_subprocess must be a literal boolean, not a string."""

    def test_prewarm_as_string_true_raises(self) -> None:
        """Reject 'true' string for prewarm_subprocess.

        Contract: behaviors:ConfigLoading:MUST:6 — Reject string booleans
        """
        data = _base_valid_yaml()
        data["sdk"]["prewarm_subprocess"] = "true"  # string, not bool

        exc_info = _load_fresh_raises(data)
        assert "prewarm_subprocess" in str(exc_info.value).lower()
        assert "boolean" in str(exc_info.value).lower()

    def test_prewarm_as_integer_raises(self) -> None:
        """Reject integer for prewarm_subprocess.

        Contract: behaviors:ConfigLoading:MUST:6 — Reject string booleans
        """
        data = _base_valid_yaml()
        data["sdk"]["prewarm_subprocess"] = 1  # int, not bool

        exc_info = _load_fresh_raises(data)
        assert "prewarm_subprocess" in str(exc_info.value).lower()

    def test_prewarm_as_literal_bool_true_accepted(self) -> None:
        """Literal True bool for prewarm_subprocess is accepted.

        Contract: behaviors:ConfigLoading:MUST:6 — Bool is OK
        """
        data = _base_valid_yaml()
        data["sdk"]["prewarm_subprocess"] = True

        # Should NOT raise — just load successfully
        config = _load_fresh(data)
        assert config.sdk.prewarm_subprocess is True


# ---------------------------------------------------------------------------
# Tool capture section key validation
# ---------------------------------------------------------------------------


class TestToolCaptureSectionKeyValidation:
    """Each required key in tool_capture section must be present."""

    @pytest.mark.parametrize(
        "missing_key",
        [
            "first_turn_only",
            "deduplicate",
            "log_capture_events",
        ],
    )
    def test_missing_tool_capture_key_raises(self, missing_key: str) -> None:
        """Missing tool_capture key raises ConfigurationError.

        Contract: sdk-protection:ToolCapture:MUST:1,2
        """
        data = _base_valid_yaml()
        del data["tool_capture"][missing_key]

        exc_info = _load_fresh_raises(data)
        assert missing_key in str(exc_info.value)


# ---------------------------------------------------------------------------
# Session section key validation
# ---------------------------------------------------------------------------


class TestSessionSectionKeyValidation:
    """Each required key in session section must be present."""

    @pytest.mark.parametrize(
        "missing_key",
        [
            "explicit_abort",
            "abort_timeout_seconds",
            "idle_timeout_seconds",
            "disconnect_timeout_seconds",
        ],
    )
    def test_missing_session_key_raises(self, missing_key: str) -> None:
        """Missing session key raises ConfigurationError.

        Contract: sdk-protection:Session:MUST:3,4
        """
        data = _base_valid_yaml()
        del data["session"][missing_key]

        exc_info = _load_fresh_raises(data)
        assert missing_key in str(exc_info.value)


# ---------------------------------------------------------------------------
# Top-level section validation
# ---------------------------------------------------------------------------


class TestTopLevelSectionValidation:
    """tool_capture, session, sdk sections are all required."""

    def test_missing_tool_capture_section_raises(self) -> None:
        """Missing 'tool_capture' section raises ConfigurationError."""
        data = _base_valid_yaml()
        del data["tool_capture"]

        exc_info = _load_fresh_raises(data)
        assert "tool_capture" in str(exc_info.value).lower()

    def test_missing_session_section_raises(self) -> None:
        """Missing 'session' section raises ConfigurationError."""
        data = _base_valid_yaml()
        del data["session"]

        exc_info = _load_fresh_raises(data)
        assert "session" in str(exc_info.value).lower()

    def test_missing_sdk_section_raises(self) -> None:
        """Missing 'sdk' section raises ConfigurationError."""
        data = _base_valid_yaml()
        del data["sdk"]

        exc_info = _load_fresh_raises(data)
        assert "'sdk'" in str(exc_info.value).lower() or "sdk" in str(exc_info.value).lower()

    def test_empty_yaml_raises(self) -> None:
        """Empty/None yaml raises ConfigurationError."""
        from amplifier_module_provider_github_copilot.config_loader import (
            load_sdk_protection_config,
        )
        from amplifier_module_provider_github_copilot.error_translation import ConfigurationError

        load_sdk_protection_config.cache_clear()
        try:
            with patch(
                "amplifier_module_provider_github_copilot.config_loader.yaml.safe_load",
                return_value=None,
            ):
                with pytest.raises(ConfigurationError) as exc_info:
                    load_sdk_protection_config()
            err = str(exc_info.value).lower()
            assert "empty" in err or "invalid" in err
        finally:
            load_sdk_protection_config.cache_clear()
