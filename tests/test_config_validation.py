"""Tests for config validation fail-fast behavior.

Contract: contracts/observability.md (logging standards)

Tests verify that missing or invalid config/models.py data raises ConfigurationError
at startup rather than silently falling back to hardcoded defaults.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from amplifier_module_provider_github_copilot.error_translation import ConfigurationError

_MODELS_MODULE = "amplifier_module_provider_github_copilot.config._models"


class TestConfigValidationFailFast:
    """Tests for fail-fast config validation.

    The provider should raise ConfigurationError at startup if:
    1. config/models.py has no models defined
    2. config/models.py has no PROVIDER definition
    3. config/models.py is missing required defaults

    This replaces silent fallback behavior with explicit failure.
    """

    def test_empty_models_list_is_allowed(self) -> None:
        """config loader does not validate a MODELS constant — MODELS was removed.

        The loader now validates PROVIDER (id, defaults.model, defaults.timeout),
        not the MODELS list — model discovery is the SDK's job, not config's.
        Contract: behaviors:ModelDiscoveryError:MUST_NOT:1
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_models_config,
        )

        load_models_config.cache_clear()
        # Patch in a temporary MODELS=[] (create=True since the attribute no longer exists)
        # to prove load_models_config() does NOT gate on it
        with patch(f"{_MODELS_MODULE}.MODELS", [], create=True):
            cfg = load_models_config()
        assert cfg.provider_id == "github-copilot"
        assert cfg.defaults["model"] == "claude-opus-4.5"
        load_models_config.cache_clear()

    def test_configuration_error_includes_provider_id(self) -> None:
        """ConfigurationError includes provider='github-copilot' when PROVIDER missing.

        Contract: error-hierarchy.md (MUST set provider on errors)
        Feature: Config validation fail-fast on PROVIDER block, not MODELS list.
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_models_config,
        )

        load_models_config.cache_clear()
        with patch(f"{_MODELS_MODULE}.PROVIDER", {}):
            with pytest.raises(ConfigurationError) as exc_info:
                load_models_config()

        assert exc_info.value.provider == "github-copilot"
        load_models_config.cache_clear()


class TestConfigLoaderMissingKeys:
    """Tests for config validation when required keys are missing.

    Validates that load_models_config() fails fast when PROVIDER data
    is missing required fields.
    """

    def test_missing_provider_section_raises_configuration_error(self) -> None:
        """Missing PROVIDER definition raises ConfigurationError.

        # Contract: behaviors:ConfigLoading:MUST:2
        # Contract: behaviors:ConfigLoading:MUST:6
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_models_config,
        )

        load_models_config.cache_clear()
        with patch(f"{_MODELS_MODULE}.PROVIDER", {}):
            with pytest.raises(ConfigurationError) as exc_info:
                load_models_config()

        assert "provider" in str(exc_info.value).lower()
        load_models_config.cache_clear()

    def test_missing_defaults_model_raises_configuration_error(self) -> None:
        """Missing PROVIDER[defaults][model] raises ConfigurationError.

        # Contract: behaviors:ConfigLoading:MUST:2
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_models_config,
        )

        load_models_config.cache_clear()
        provider_no_model: dict[str, Any] = {
            "id": "github-copilot",
            "display_name": "Test",
            "defaults": {},  # Missing 'model' key
        }
        with patch(f"{_MODELS_MODULE}.PROVIDER", provider_no_model):
            with pytest.raises(ConfigurationError) as exc_info:
                load_models_config()

        assert "model" in str(exc_info.value).lower()
        load_models_config.cache_clear()

    def test_missing_defaults_timeout_raises_configuration_error(self) -> None:
        """Missing PROVIDER[defaults][timeout] raises ConfigurationError.

        # Contract: behaviors:ConfigLoading:MUST:2
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_models_config,
        )

        load_models_config.cache_clear()
        provider_no_timeout: dict[str, Any] = {
            "id": "github-copilot",
            "display_name": "Test",
            "defaults": {"model": "claude-opus-4.5"},  # Missing 'timeout' key
        }
        with patch(f"{_MODELS_MODULE}.PROVIDER", provider_no_timeout):
            with pytest.raises(ConfigurationError) as exc_info:
                load_models_config()

        assert "timeout" in str(exc_info.value).lower()
        load_models_config.cache_clear()


class TestSdkLogLevelValidation:
    """sdk-protection:Subprocess:MUST:7 — log level validation.

    Migrated from test_sdk_protection.py.
    """

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
