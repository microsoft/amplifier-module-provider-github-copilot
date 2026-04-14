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

    def test_empty_models_list_raises_configuration_error(self) -> None:
        """Empty models list raises ConfigurationError, not silent fallback.

        Contract: observability.md (logging standards)
        Feature: Config validation fail-fast
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_models_config,
        )

        load_models_config.cache_clear()
        with patch(f"{_MODELS_MODULE}.MODELS", []):
            with pytest.raises(ConfigurationError) as exc_info:
                load_models_config()

        assert "no models" in str(exc_info.value).lower()
        load_models_config.cache_clear()

    def test_configuration_error_includes_provider_id(self) -> None:
        """ConfigurationError includes provider='github-copilot'.

        Contract: error-hierarchy.md (MUST set provider on errors)
        Feature: Config validation fail-fast
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_models_config,
        )

        load_models_config.cache_clear()
        with patch(f"{_MODELS_MODULE}.MODELS", []):
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
            "defaults": {"model": "gpt-4"},  # Missing 'timeout' key
        }
        with patch(f"{_MODELS_MODULE}.PROVIDER", provider_no_timeout):
            with pytest.raises(ConfigurationError) as exc_info:
                load_models_config()

        assert "timeout" in str(exc_info.value).lower()
        load_models_config.cache_clear()


class TestPythonConfigStructure:
    """Tests validating the Python config structure in config/models.py.

    These tests catch accidental corruption or deletion of required fields
    in the config/models.py data module.
    """

    def test_models_list_is_populated(self) -> None:
        """config/_models.py MODELS list is not empty.

        # Contract: behaviors:ConfigLoading:MUST:3
        """
        from amplifier_module_provider_github_copilot.config import _models as models

        assert len(models.MODELS) > 0, "MODELS list must have at least one model"

    def test_each_model_has_required_fields(self) -> None:
        """Each model in MODELS has required fields.

        # Contract: provider-protocol:list_models:MUST:2
        """
        from amplifier_module_provider_github_copilot.config import _models as models

        # Assert exact values for known models
        opus = next(m for m in models.MODELS if m["id"] == "claude-opus-4.5")
        assert opus["id"] == "claude-opus-4.5"
        assert opus["display_name"] == "Claude Opus 4.5"
        assert opus["context_window"] == 200000
        assert opus["max_output_tokens"] == 32000

        gpt4 = next(m for m in models.MODELS if m["id"] == "gpt-4")
        assert gpt4["id"] == "gpt-4"
        assert gpt4["display_name"] == "GPT-4"
        assert gpt4["context_window"] == 128000
        assert gpt4["max_output_tokens"] == 4096

        gpt4o = next(m for m in models.MODELS if m["id"] == "gpt-4o")
        assert gpt4o["id"] == "gpt-4o"
        assert gpt4o["display_name"] == "GPT-4o"
        assert gpt4o["context_window"] == 128000
        assert gpt4o["max_output_tokens"] == 4096

    def test_provider_has_required_fields(self) -> None:
        """config/_models.py PROVIDER has required top-level and nested fields.

        # Contract: provider-protocol:name:MUST:1
        """
        from amplifier_module_provider_github_copilot.config import _models as models

        assert models.PROVIDER["id"] == "github-copilot"
        assert models.PROVIDER["display_name"] == "GitHub Copilot SDK"
        assert models.PROVIDER["defaults"]["model"] == "claude-opus-4.5"
        assert models.PROVIDER["defaults"]["timeout"] == 3600

    def test_fallbacks_has_required_keys(self) -> None:
        """config/_models.py FALLBACKS has required keys.

        # Contract: provider-protocol:get_info:MUST:1
        """
        from amplifier_module_provider_github_copilot.config import _models as models

        assert models.FALLBACKS["context_window"] == 128000
        assert models.FALLBACKS["max_output_tokens"] == 16384

    def test_load_models_config_returns_valid_config(self) -> None:
        """load_models_config() returns valid ProviderConfig from Python module.

        # Contract: provider-protocol:get_info:MUST:2
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_models_config,
        )

        load_models_config.cache_clear()
        config = load_models_config()
        assert config.provider_id == "github-copilot"
        assert len(config.models) > 0
        load_models_config.cache_clear()
