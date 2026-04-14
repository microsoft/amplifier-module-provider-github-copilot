"""Tests for SDK protection config loading.

Contract: sdk-protection:ToolCapture:MUST:1,2
Contract: sdk-protection:Session:MUST:3,4
"""

from __future__ import annotations

import pytest


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

    def test_yaml_config_has_expected_defaults(self) -> None:
        """Config has expected default values.

        Python dataclass is authoritative source (config/_sdk_protection.py).
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

    def test_session_abort_timeout_is_reasonable(self) -> None:
        """Abort timeout is reasonable (not too short, not too long).

        Contract: sdk-protection:Session:MUST:4
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_sdk_protection_config,
        )

        config = load_sdk_protection_config()

        assert config.session.abort_timeout_seconds == 5.0

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

        assert config.session.idle_timeout_seconds == 30.0


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

        assert callable(SdkProtectionConfig)

    def test_sdk_protection_config_instantiates_with_no_args(self) -> None:
        """SdkProtectionConfig() instantiates with hardcoded defaults (no I/O)."""
        from amplifier_module_provider_github_copilot.config._sdk_protection import (
            SdkProtectionConfig,
        )

        config = SdkProtectionConfig()

        assert config.tool_capture.first_turn_only is True
        assert config.session.abort_timeout_seconds == 5.0
        assert config.sdk.log_level == "info"
        assert config.singleton.lock_timeout_seconds == 30.0


class TestSdkConfigLoading:
    """Test SDK subprocess configuration loading."""

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

    def test_validation_rejects_invalid_log_level_directly(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """SDK log level validator falls back to default for invalid env values.

        Contract: sdk-protection:Subprocess:MUST:7
        """
        import logging
        import os
        from unittest.mock import patch

        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            _resolve_sdk_log_level,  # pyright: ignore[reportPrivateUsage]
        )

        with (
            patch.dict(os.environ, {"COPILOT_SDK_LOG_LEVEL": "invalid_level"}, clear=False),
            caplog.at_level(logging.WARNING),
        ):
            result = _resolve_sdk_log_level()

        assert result == "info"
        assert any("Invalid SDK log level" in r.getMessage() for r in caplog.records)

    def test_validation_accepts_all_valid_log_levels(self) -> None:
        """All defined log levels are accepted by validation.

        Contract: sdk-protection:Subprocess:MUST:7
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_sdk_protection_config,
        )

        config = load_sdk_protection_config()

        expected_levels = {"none", "error", "warning", "info", "debug", "all"}
        assert set(config.sdk.valid_log_levels) == expected_levels
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
