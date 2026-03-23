"""Tests for config validation fail-fast behavior.

Contract: contracts/observability.md (logging standards)

Tests verify that missing or corrupt config files raise ConfigurationError
at startup rather than silently falling back to hardcoded defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from amplifier_module_provider_github_copilot.error_translation import ConfigurationError


class TestConfigValidationFailFast:
    """Tests for fail-fast config validation.

    The provider should raise ConfigurationError at startup if:
    1. models.yaml is missing
    2. models.yaml is corrupt/unparseable
    3. models.yaml has no models defined

    This replaces silent fallback behavior with explicit failure.
    """

    def test_missing_models_yaml_raises_configuration_error(self) -> None:
        """Missing models.yaml raises ConfigurationError, not silent fallback.

        Contract: observability.md (logging standards)
        Feature: Config validation fail-fast
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_models_config,
        )

        with patch.object(Path, "exists", return_value=False):
            with pytest.raises(ConfigurationError) as exc_info:
                load_models_config()

        assert "models.yaml not found" in str(exc_info.value)
        assert "broken installation" in str(exc_info.value).lower()

    def test_corrupt_models_yaml_raises_configuration_error(self) -> None:
        """Corrupt models.yaml raises ConfigurationError, not silent fallback.

        Contract: observability.md (logging standards)
        Feature: Config validation fail-fast
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_models_config,
        )

        with patch(
            "amplifier_module_provider_github_copilot.config_loader.yaml.safe_load",
            side_effect=Exception("yaml parse error"),
        ):
            with pytest.raises(ConfigurationError) as exc_info:
                load_models_config()

        assert "corrupted" in str(exc_info.value).lower()

    def test_empty_models_list_raises_configuration_error(self) -> None:
        """Empty models list raises ConfigurationError, not silent fallback.

        Contract: observability.md (logging standards)
        Feature: Config validation fail-fast
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_models_config,
        )

        empty_config: dict[str, Any] = {
            "version": "1.0",
            "provider": {"id": "github-copilot"},
            "models": [],
        }

        with patch(
            "amplifier_module_provider_github_copilot.config_loader.yaml.safe_load",
            return_value=empty_config,
        ):
            with pytest.raises(ConfigurationError) as exc_info:
                load_models_config()

        assert "no models" in str(exc_info.value).lower()

    def test_configuration_error_includes_provider_id(self) -> None:
        """ConfigurationError includes provider='github-copilot'.

        Contract: error-hierarchy.md (MUST set provider on errors)
        Feature: Config validation fail-fast
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_models_config,
        )

        with patch.object(Path, "exists", return_value=False):
            with pytest.raises(ConfigurationError) as exc_info:
                load_models_config()

        # ConfigurationError should have provider attribute
        assert exc_info.value.provider == "github-copilot"
