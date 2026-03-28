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


# =============================================================================
# Additional Coverage Tests for config_loader.py
# Coverage targets: lines 34-47, 121-122, 141, 148, 154
# =============================================================================


class TestConfigLoaderMissingKeys:
    """Tests for config validation when required keys are missing.

    Covers lines 121-122, 141, 148, 154 in config_loader.py.
    """

    def test_empty_config_raises_configuration_error(self) -> None:
        """Empty config data raises ConfigurationError.

        Covers line 121-122.
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_models_config,
        )

        # Clear cache
        load_models_config.cache_clear()

        with patch(
            "amplifier_module_provider_github_copilot.config_loader.yaml.safe_load",
            return_value=None,
        ):
            with pytest.raises(ConfigurationError) as exc_info:
                load_models_config()

        assert "empty" in str(exc_info.value).lower() or "invalid" in str(exc_info.value).lower()

        # Clear cache for other tests
        load_models_config.cache_clear()

    def test_missing_provider_section_raises_configuration_error(self) -> None:
        """Missing 'provider' section raises ConfigurationError.

        Covers line 141.
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_models_config,
        )

        # Clear cache
        load_models_config.cache_clear()

        partial_config: dict[str, Any] = {
            "version": "1.0",
            "models": [{"id": "test-model", "display_name": "Test"}],
            # 'provider' section missing
        }

        with patch(
            "amplifier_module_provider_github_copilot.config_loader.yaml.safe_load",
            return_value=partial_config,
        ):
            with pytest.raises(ConfigurationError) as exc_info:
                load_models_config()

        assert "provider" in str(exc_info.value).lower()

        # Clear cache for other tests
        load_models_config.cache_clear()

    def test_missing_defaults_model_raises_configuration_error(self) -> None:
        """Missing 'provider.defaults.model' raises ConfigurationError.

        Covers line 148.
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_models_config,
        )

        # Clear cache
        load_models_config.cache_clear()

        partial_config: dict[str, Any] = {
            "version": "1.0",
            "provider": {
                "id": "github-copilot",
                "display_name": "Test",
                "defaults": {},  # Missing 'model' key
            },
            "models": [{"id": "test-model", "display_name": "Test"}],
        }

        with patch(
            "amplifier_module_provider_github_copilot.config_loader.yaml.safe_load",
            return_value=partial_config,
        ):
            with pytest.raises(ConfigurationError) as exc_info:
                load_models_config()

        assert "model" in str(exc_info.value).lower()

        # Clear cache for other tests
        load_models_config.cache_clear()

    def test_missing_defaults_timeout_raises_configuration_error(self) -> None:
        """Missing 'provider.defaults.timeout' raises ConfigurationError.

        Covers line 154.
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_models_config,
        )

        # Clear cache
        load_models_config.cache_clear()

        partial_config: dict[str, Any] = {
            "version": "1.0",
            "provider": {
                "id": "github-copilot",
                "display_name": "Test",
                "defaults": {
                    "model": "gpt-4",
                    # 'timeout' missing
                },
            },
            "models": [{"id": "test-model", "display_name": "Test"}],
        }

        with patch(
            "amplifier_module_provider_github_copilot.config_loader.yaml.safe_load",
            return_value=partial_config,
        ):
            with pytest.raises(ConfigurationError) as exc_info:
                load_models_config()

        assert "timeout" in str(exc_info.value).lower()

        # Clear cache for other tests
        load_models_config.cache_clear()


class TestRetryConfigValidation:
    """Tests for retry config validation.

    Covers lines 346, 355-356, 363, 370, 378 in config_loader.py.
    """

    def test_missing_retry_yaml_raises_configuration_error(self) -> None:
        """Missing retry.yaml raises ConfigurationError."""
        from amplifier_module_provider_github_copilot.config_loader import (
            load_retry_config,
        )

        # Clear cache
        load_retry_config.cache_clear()

        original_exists = Path.exists

        def mock_exists(self: Path) -> bool:
            if "retry.yaml" in str(self):
                return False
            return original_exists(self)

        with patch.object(Path, "exists", mock_exists):
            with pytest.raises(ConfigurationError) as exc_info:
                load_retry_config()

        assert "retry.yaml not found" in str(exc_info.value)

        # Clear cache for other tests
        load_retry_config.cache_clear()

    def test_corrupt_retry_yaml_raises_configuration_error(self) -> None:
        """Corrupt retry.yaml raises ConfigurationError."""
        from amplifier_module_provider_github_copilot.config_loader import (
            load_retry_config,
        )

        # Clear cache
        load_retry_config.cache_clear()

        # Mock yaml.safe_load to raise exception
        with patch(
            "amplifier_module_provider_github_copilot.config_loader.yaml.safe_load",
            side_effect=Exception("parse error"),
        ):
            with pytest.raises(ConfigurationError) as exc_info:
                load_retry_config()

        assert "corrupted" in str(exc_info.value).lower()

        # Clear cache for other tests
        load_retry_config.cache_clear()

    def test_empty_retry_yaml_raises_configuration_error(self) -> None:
        """Empty retry.yaml raises ConfigurationError."""
        from amplifier_module_provider_github_copilot.config_loader import (
            load_retry_config,
        )

        # Clear cache
        load_retry_config.cache_clear()

        # Mock yaml.safe_load to return None (empty)
        with patch(
            "amplifier_module_provider_github_copilot.config_loader.yaml.safe_load",
            return_value=None,
        ):
            with pytest.raises(ConfigurationError) as exc_info:
                load_retry_config()

        assert "empty" in str(exc_info.value).lower() or "invalid" in str(exc_info.value).lower()

        # Clear cache for other tests
        load_retry_config.cache_clear()


class TestStreamingConfigValidation:
    """Tests for streaming config validation.

    Covers lines 441, 450-451, 458, 466, 473, 481, 488 in config_loader.py.
    """

    def test_missing_streaming_yaml_raises_configuration_error(self) -> None:
        """Missing retry.yaml (for streaming) raises ConfigurationError."""
        from amplifier_module_provider_github_copilot.config_loader import (
            load_streaming_config,
        )

        # Clear cache
        load_streaming_config.cache_clear()

        original_exists = Path.exists

        def mock_exists(self: Path) -> bool:
            if "retry.yaml" in str(self):
                return False
            return original_exists(self)

        with patch.object(Path, "exists", mock_exists):
            with pytest.raises(ConfigurationError) as exc_info:
                load_streaming_config()

        assert "retry.yaml not found" in str(exc_info.value)

        # Clear cache for other tests
        load_streaming_config.cache_clear()

    def test_corrupt_streaming_yaml_raises_configuration_error(self) -> None:
        """Corrupt retry.yaml (for streaming) raises ConfigurationError."""
        from amplifier_module_provider_github_copilot.config_loader import (
            load_streaming_config,
        )

        # Clear cache
        load_streaming_config.cache_clear()

        with patch(
            "amplifier_module_provider_github_copilot.config_loader.yaml.safe_load",
            side_effect=Exception("parse error"),
        ):
            with pytest.raises(ConfigurationError) as exc_info:
                load_streaming_config()

        assert "corrupted" in str(exc_info.value).lower()

        # Clear cache for other tests
        load_streaming_config.cache_clear()

    def test_empty_streaming_yaml_raises_configuration_error(self) -> None:
        """Empty retry.yaml (for streaming) raises ConfigurationError."""
        from amplifier_module_provider_github_copilot.config_loader import (
            load_streaming_config,
        )

        # Clear cache
        load_streaming_config.cache_clear()

        with patch(
            "amplifier_module_provider_github_copilot.config_loader.yaml.safe_load",
            return_value=None,
        ):
            with pytest.raises(ConfigurationError) as exc_info:
                load_streaming_config()

        assert "empty" in str(exc_info.value).lower() or "invalid" in str(exc_info.value).lower()

        # Clear cache for other tests
        load_streaming_config.cache_clear()

    def test_missing_streaming_config_raises_configuration_error(self) -> None:
        """Missing streaming section in retry.yaml raises ConfigurationError."""
        from amplifier_module_provider_github_copilot.config_loader import (
            load_streaming_config,
        )

        # Clear cache
        load_streaming_config.cache_clear()

        partial_config: dict[str, Any] = {
            "retry": {"max_attempts": 3},
            # 'streaming' section missing
        }

        with patch(
            "amplifier_module_provider_github_copilot.config_loader.yaml.safe_load",
            return_value=partial_config,
        ):
            with pytest.raises(ConfigurationError) as exc_info:
                load_streaming_config()

        assert "streaming" in str(exc_info.value).lower()

        # Clear cache for other tests
        load_streaming_config.cache_clear()

    def test_missing_streaming_key_raises_configuration_error(self) -> None:
        """Missing required streaming key raises ConfigurationError."""
        from amplifier_module_provider_github_copilot.config_loader import (
            load_streaming_config,
        )

        # Clear cache
        load_streaming_config.cache_clear()

        partial_config: dict[str, Any] = {
            "streaming": {
                "event_queue_size": 100,
                # Missing other required keys
            }
        }

        with patch(
            "amplifier_module_provider_github_copilot.config_loader.yaml.safe_load",
            return_value=partial_config,
        ):
            with pytest.raises(ConfigurationError) as exc_info:
                load_streaming_config()

        # Should mention missing key
        assert "streaming" in str(exc_info.value).lower()

        # Clear cache for other tests
        load_streaming_config.cache_clear()
