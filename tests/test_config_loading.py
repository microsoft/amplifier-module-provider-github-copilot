"""Tests for config loading functionality.

Contract: contracts/provider-protocol.md

Tests verify that provider identity and model catalog are loaded from
config/models.yaml instead of being hardcoded in Python.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import mock_open, patch

import pytest


class TestLoadModelsConfig:
    """Tests for _load_models_config() function."""

    def test_load_models_config_returns_provider_id(self) -> None:
        """Models config loader returns correct provider id from YAML."""
        from amplifier_module_provider_github_copilot.provider import (
            _load_models_config,  # type: ignore[reportPrivateUsage]  # Testing internal function
        )

        config = _load_models_config()
        assert config.provider_id == "github-copilot"

    def test_load_models_config_returns_display_name(self) -> None:
        """Models config loader returns display_name from YAML."""
        from amplifier_module_provider_github_copilot.provider import (
            _load_models_config,  # type: ignore[reportPrivateUsage]  # Testing internal function
        )

        config = _load_models_config()
        assert config.display_name == "GitHub Copilot SDK"

    def test_load_models_config_returns_credential_env_vars(self) -> None:
        """Models config loader returns credential_env_vars from YAML."""
        from amplifier_module_provider_github_copilot.provider import (
            _load_models_config,  # type: ignore[reportPrivateUsage]  # Testing internal function
        )

        config = _load_models_config()
        assert "COPILOT_GITHUB_TOKEN" in config.credential_env_vars
        assert "GH_TOKEN" in config.credential_env_vars
        assert "GITHUB_TOKEN" in config.credential_env_vars

    def test_load_models_config_returns_capabilities(self) -> None:
        """Models config loader returns capabilities from YAML.

        Provider-level = minimum ALL models support (streaming, tools).
        Per kernel capabilities.py: TOOLS="tools", not "tool_use".
        Per-model capabilities (vision, thinking) come from SDK dynamically.
        """
        from amplifier_module_provider_github_copilot.provider import (
            _load_models_config,  # type: ignore[reportPrivateUsage]  # Testing internal function
        )

        config = _load_models_config()
        assert "streaming" in config.capabilities
        assert "tools" in config.capabilities
        # vision NOT in provider-level - it's per-model via list_models()

    def test_load_models_config_returns_defaults(self) -> None:
        """Models config loader returns defaults from YAML.

        Updated to expect claude-opus-4.5 as default model.
        """
        from amplifier_module_provider_github_copilot.provider import (
            _load_models_config,  # type: ignore[reportPrivateUsage]  # Testing internal function
        )

        config = _load_models_config()
        assert config.defaults["model"] == "claude-opus-4.5"
        assert config.defaults["max_tokens"] == 4096

    def test_load_models_config_returns_models_list(self) -> None:
        """Models config loader returns non-empty models list.

        Updated to expect claude-opus-4.5 as primary model.
        """
        from amplifier_module_provider_github_copilot.provider import (
            _load_models_config,  # type: ignore[reportPrivateUsage]  # Testing internal function
        )

        config = _load_models_config()
        assert len(config.models) >= 2
        model_ids = [m["id"] for m in config.models]
        assert "claude-opus-4.5" in model_ids
        assert "gpt-4" in model_ids

    # NOTE: Fallback tests removed - config validation now uses fail-fast pattern.
    # Missing/empty config raises ConfigurationError instead of silent fallback.
    # See tests/test_config_validation.py for the new fail-fast behavior tests.


class TestProviderUsesYamlConfig:
    """Tests verify provider methods use YAML config, not hardcoded values."""

    def test_get_info_sourced_from_yaml(self) -> None:
        """Provider.get_info() values come from YAML, not hardcoded strings.

        Updated to expect claude-opus-4.5 as default model.
        """
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider()
        info = provider.get_info()
        assert info.id == "github-copilot"
        assert "claude-opus-4.5" in str(info.defaults.get("model", ""))

    @pytest.mark.asyncio
    async def test_list_models_sourced_from_yaml(self) -> None:
        """Provider.list_models() comes from YAML, not hardcoded list.

        Updated to expect claude-opus-4.5 as primary model.
        """
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider()
        models = await provider.list_models()
        model_ids = [m.id for m in models]
        assert "claude-opus-4.5" in model_ids
        assert "gpt-4" in model_ids

    # NOTE: Graceful fallback test removed - config validation now uses fail-fast pattern.
    # Missing config raises ConfigurationError at provider init.
    # See tests/test_config_validation.py for fail-fast behavior tests.


class TestModelsYamlSchemaCompliance:
    """Tests verify models.yaml has correct structure."""

    def test_models_yaml_version_field_present(self) -> None:
        """Models YAML has version field."""
        import yaml

        config_path = (
            Path(__file__).parent.parent
            / "amplifier_module_provider_github_copilot"
            / "config"
            / "models.yaml"
        )
        with config_path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "version" in data
        assert data["version"] == "1.0"

    def test_models_yaml_provider_id(self) -> None:
        """Models YAML provider.id equals github-copilot."""
        import yaml

        config_path = (
            Path(__file__).parent.parent
            / "amplifier_module_provider_github_copilot"
            / "config"
            / "models.yaml"
        )
        with config_path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["provider"]["id"] == "github-copilot"

    def test_models_yaml_models_list_nonempty(self) -> None:
        """Models YAML has non-empty models list."""
        import yaml

        config_path = (
            Path(__file__).parent.parent
            / "amplifier_module_provider_github_copilot"
            / "config"
            / "models.yaml"
        )
        with config_path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data["models"], list)
        assert len(data["models"]) > 0

    def test_models_yaml_each_model_has_required_fields(self) -> None:
        """Each model in YAML has required fields."""
        import yaml

        config_path = (
            Path(__file__).parent.parent
            / "amplifier_module_provider_github_copilot"
            / "config"
            / "models.yaml"
        )
        with config_path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)

        required_fields = ["id", "display_name", "context_window", "max_output_tokens"]
        for model in data["models"]:
            for field in required_fields:
                assert field in model, f"Model {model.get('id', 'unknown')} missing field: {field}"


# ============================================================================
# Merged from test_coverage_gaps_final.py — config_loader.py fallback paths
# ============================================================================


# NOTE: TestLoadModelsConfigFallbackPaths removed - config validation now uses fail-fast pattern.
# The old fallback tests expected silent degradation; now missing/corrupt config raises
# ConfigurationError.
# See tests/test_config_validation.py for the new fail-fast behavior tests.


class TestLoadRetryConfigFailFast:
    """Three-Medium: YAML is authoritative. Missing/invalid YAML raises ConfigurationError."""

    def test_missing_retry_yaml_raises_configuration_error(self) -> None:
        """Missing retry.yaml raises ConfigurationError (fail-fast).

        Contract: behaviors:Config:MUST:2
        Three-Medium: YAML is authoritative.
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_retry_config,
        )

        # Clear cache to allow re-loading
        load_retry_config.cache_clear()

        with patch.object(Path, "exists", return_value=False):
            with pytest.raises(Exception) as exc:
                load_retry_config()

            # Should be ConfigurationError
            assert "retry.yaml not found" in str(exc.value)

    def test_yaml_load_exception_raises_configuration_error(self) -> None:
        """Corrupted retry.yaml raises ConfigurationError (fail-fast).

        Contract: behaviors:Config:MUST:2
        Three-Medium: YAML is authoritative.
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_retry_config,
        )

        load_retry_config.cache_clear()

        with (
            patch.object(Path, "exists", return_value=True),
            patch("builtins.open", mock_open(read_data="invalid: yaml: content:")),
            patch(
                "amplifier_module_provider_github_copilot.config_loader.yaml.safe_load",
                side_effect=Exception("bad yaml"),
            ),
        ):
            with pytest.raises(Exception) as exc:
                load_retry_config()

            assert "corrupted" in str(exc.value).lower()

    def test_empty_retry_yaml_raises_configuration_error(self) -> None:
        """Empty retry.yaml raises ConfigurationError (fail-fast).

        Contract: behaviors:Config:MUST:2
        Three-Medium: YAML is authoritative.
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_retry_config,
        )

        load_retry_config.cache_clear()

        with (
            patch.object(Path, "exists", return_value=True),
            patch("builtins.open", mock_open(read_data="")),
            patch(
                "amplifier_module_provider_github_copilot.config_loader.yaml.safe_load",
                return_value=None,
            ),
        ):
            with pytest.raises(Exception) as exc:
                load_retry_config()

            assert "empty" in str(exc.value).lower() or "invalid" in str(exc.value).lower()

    def test_max_attempts_less_than_one_raises_configuration_error(self) -> None:
        """Invalid max_attempts raises ConfigurationError (fail-fast).

        Contract: behaviors:Retry:MUST:4
        Three-Medium: Validation happens at load time.
        """
        from amplifier_module_provider_github_copilot.config_loader import load_retry_config

        load_retry_config.cache_clear()

        bad_config: dict[str, object] = {
            "retry": {
                "max_attempts": 0,
                "backoff": {
                    "base_delay_ms": 1000,
                    "max_delay_ms": 30000,
                    "jitter_factor": 0.1,
                },
            }
        }

        with (
            patch.object(Path, "exists", return_value=True),
            patch("builtins.open", mock_open(read_data="yaml content")),
            patch(
                "amplifier_module_provider_github_copilot.config_loader.yaml.safe_load",
                return_value=bad_config,
            ),
        ):
            with pytest.raises(Exception) as exc:
                load_retry_config()

            assert "max_attempts" in str(exc.value) or "invalid" in str(exc.value).lower()

    def test_max_attempts_negative_raises_configuration_error(self) -> None:
        """Negative max_attempts raises ConfigurationError (fail-fast).

        Contract: behaviors:Retry:MUST:4
        Three-Medium: Validation happens at load time.
        """
        from amplifier_module_provider_github_copilot.config_loader import load_retry_config

        load_retry_config.cache_clear()

        bad_config: dict[str, object] = {
            "retry": {
                "max_attempts": -5,
                "backoff": {
                    "base_delay_ms": 1000,
                    "max_delay_ms": 30000,
                    "jitter_factor": 0.1,
                },
            }
        }

        with (
            patch.object(Path, "exists", return_value=True),
            patch("builtins.open", mock_open(read_data="yaml content")),
            patch(
                "amplifier_module_provider_github_copilot.config_loader.yaml.safe_load",
                return_value=bad_config,
            ),
        ):
            with pytest.raises(Exception) as exc:
                load_retry_config()

            assert "max_attempts" in str(exc.value) or "invalid" in str(exc.value).lower()

    def test_valid_retry_yaml_loads_values(self) -> None:
        """Valid retry.yaml loads actual values.

        Three-Medium: YAML is authoritative.
        """
        from amplifier_module_provider_github_copilot.config_loader import load_retry_config

        load_retry_config.cache_clear()

        # This test uses the real YAML file
        result = load_retry_config()

        # Values should come from actual retry.yaml
        assert result.max_attempts == 3
        assert result.base_delay_ms == 1000
        assert result.max_delay_ms == 30000
        assert result.jitter_factor == 0.1

    def test_missing_retry_section_raises_configuration_error(self) -> None:
        """Missing 'retry' section raises ConfigurationError (fail-fast).

        Contract: behaviors:Retry:MUST:4
        """
        from amplifier_module_provider_github_copilot.config_loader import load_retry_config

        load_retry_config.cache_clear()

        # Config with no 'retry' key
        bad_config: dict[str, object] = {"other_section": {}}

        with (
            patch.object(Path, "exists", return_value=True),
            patch("builtins.open", mock_open(read_data="yaml content")),
            patch(
                "amplifier_module_provider_github_copilot.config_loader.yaml.safe_load",
                return_value=bad_config,
            ),
        ):
            with pytest.raises(Exception) as exc:
                load_retry_config()

            assert "retry" in str(exc.value).lower()

    def test_missing_backoff_section_raises_configuration_error(self) -> None:
        """Missing 'backoff' section raises ConfigurationError (fail-fast).

        Contract: behaviors:Retry:MUST:4
        """
        from amplifier_module_provider_github_copilot.config_loader import load_retry_config

        load_retry_config.cache_clear()

        # Config with retry but no backoff
        bad_config: dict[str, object] = {
            "retry": {
                "max_attempts": 3,
                # Missing backoff section
            }
        }

        with (
            patch.object(Path, "exists", return_value=True),
            patch("builtins.open", mock_open(read_data="yaml content")),
            patch(
                "amplifier_module_provider_github_copilot.config_loader.yaml.safe_load",
                return_value=bad_config,
            ),
        ):
            with pytest.raises(Exception) as exc:
                load_retry_config()

            assert "backoff" in str(exc.value).lower()

    def test_missing_max_attempts_raises_configuration_error(self) -> None:
        """Missing 'max_attempts' raises ConfigurationError (fail-fast).

        Contract: behaviors:Retry:MUST:4
        """
        from amplifier_module_provider_github_copilot.config_loader import load_retry_config

        load_retry_config.cache_clear()

        # Config with no max_attempts
        bad_config: dict[str, object] = {
            "retry": {
                # Missing max_attempts
                "backoff": {
                    "base_delay_ms": 1000,
                    "max_delay_ms": 30000,
                    "jitter_factor": 0.1,
                },
            }
        }

        with (
            patch.object(Path, "exists", return_value=True),
            patch("builtins.open", mock_open(read_data="yaml content")),
            patch(
                "amplifier_module_provider_github_copilot.config_loader.yaml.safe_load",
                return_value=bad_config,
            ),
        ):
            with pytest.raises(Exception) as exc:
                load_retry_config()

            assert "max_attempts" in str(exc.value).lower()

    def test_missing_backoff_key_raises_configuration_error(self) -> None:
        """Missing backoff key raises ConfigurationError (fail-fast).

        Contract: behaviors:Retry:MUST:4
        """
        from amplifier_module_provider_github_copilot.config_loader import load_retry_config

        load_retry_config.cache_clear()

        # Config with incomplete backoff section
        bad_config: dict[str, object] = {
            "retry": {
                "max_attempts": 3,
                "backoff": {
                    "base_delay_ms": 1000,
                    # Missing max_delay_ms and jitter_factor
                },
            }
        }

        with (
            patch.object(Path, "exists", return_value=True),
            patch("builtins.open", mock_open(read_data="yaml content")),
            patch(
                "amplifier_module_provider_github_copilot.config_loader.yaml.safe_load",
                return_value=bad_config,
            ),
        ):
            with pytest.raises(Exception) as exc:
                load_retry_config()

            # Should mention missing key
            assert "max_delay_ms" in str(exc.value) or "jitter" in str(exc.value).lower()


# ============================================================================
# Fake Tool Detection Config Loading Tests
# Contract: behaviors:Config:MUST:1
# ============================================================================


class TestFakeToolDetectionConfigLoading:
    """Tests for fake tool detection config loading.

    Contract: behaviors:Config:MUST:1
    """

    def test_config_loads_from_yaml(self) -> None:
        """Config loads patterns, max_attempts, message from YAML.

        Contract: behaviors:Config:MUST:1
        """
        from amplifier_module_provider_github_copilot.fake_tool_detection import (
            load_fake_tool_detection_config,
        )

        config = load_fake_tool_detection_config()
        assert len(config.patterns) > 0
        assert config.max_correction_attempts >= 1
        assert len(config.correction_message) > 0

    def test_config_patterns_are_compiled_regex(self) -> None:
        """Patterns are compiled as re.Pattern objects.

        Contract: behaviors:Config:MUST:1
        """
        import re

        from amplifier_module_provider_github_copilot.fake_tool_detection import (
            load_fake_tool_detection_config,
        )

        config = load_fake_tool_detection_config()
        for pattern in config.patterns:
            assert isinstance(pattern, re.Pattern)

    def test_config_fallback_on_missing_file(self) -> None:
        """Defaults used when config file missing.

        Contract: behaviors:Config:MUST:1
        """
        from pathlib import Path

        from amplifier_module_provider_github_copilot.fake_tool_detection import (
            load_fake_tool_detection_config,
        )

        # Use a path that doesn't exist
        nonexistent_path = Path("/nonexistent/fake-tool-detection.yaml")
        config = load_fake_tool_detection_config(config_path=nonexistent_path)

        # Should return defaults, not raise
        assert len(config.patterns) > 0
        assert config.max_correction_attempts == 2

    def test_config_logging_section_loaded(self) -> None:
        """Logging config section loaded from YAML.

        Contract: behaviors:Logging:MUST:1
        """
        from amplifier_module_provider_github_copilot.fake_tool_detection import (
            load_fake_tool_detection_config,
        )

        config = load_fake_tool_detection_config()
        assert config.logging is not None
        assert config.logging.log_matched_pattern is True
        assert config.logging.level_on_detection in ("INFO", "WARNING", "ERROR", "DEBUG")
