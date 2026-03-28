"""Tests for default model configuration.

Contract: contracts/provider-protocol.md

Three-Medium Architecture: Default model comes from YAML, not Python code.
Tests verify that models.yaml has correct values and load_models_config() reads them.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


class TestModelsYamlDefaults:
    """Tests for models.yaml default values.

    Three-Medium: YAML is authoritative source for all policy values.
    """

    def test_models_yaml_has_default_model(self) -> None:
        """models.yaml must have provider.defaults.model.

        Contract: behaviors:Config:MUST:2
        Three-Medium: YAML is authoritative.
        """
        config_path = (
            Path(__file__).parent.parent
            / "amplifier_module_provider_github_copilot"
            / "config"
            / "models.yaml"
        )
        with config_path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert "provider" in data
        assert "defaults" in data["provider"]
        assert "model" in data["provider"]["defaults"]

    def test_models_yaml_has_timeout(self) -> None:
        """models.yaml must have provider.defaults.timeout.

        Contract: behaviors:Config:MUST:2
        Three-Medium: YAML is authoritative.
        """
        config_path = (
            Path(__file__).parent.parent
            / "amplifier_module_provider_github_copilot"
            / "config"
            / "models.yaml"
        )
        with config_path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert data["provider"]["defaults"]["timeout"] == 3600

    def test_load_models_config_reads_yaml(self) -> None:
        """load_models_config() reads values from YAML.

        Three-Medium: Python loads from YAML, no hardcoded fallbacks.
        """
        from amplifier_module_provider_github_copilot.config_loader import load_models_config

        config = load_models_config()

        # Values should match YAML file
        assert config.defaults["model"] == "claude-opus-4.5"
        assert config.defaults["timeout"] == 3600
        assert config.defaults["context_window"] == 200000


class TestModelsYamlClaudeOpus:
    """Tests for models.yaml claude-opus-4.5 configuration."""

    def test_models_yaml_default_model_is_claude_opus_45(self) -> None:
        """models.yaml defaults.model is claude-opus-4.5.

        Contract anchor: provider-protocol:get_info:MUST:1
        """
        config_path = (
            Path(__file__).parent.parent
            / "amplifier_module_provider_github_copilot"
            / "config"
            / "models.yaml"
        )
        with config_path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert data["provider"]["defaults"]["model"] == "claude-opus-4.5"

    def test_models_yaml_context_window_200000(self) -> None:
        """models.yaml defaults.context_window is 200000.

        Contract anchor: provider-protocol:get_info:MUST:2
        """
        config_path = (
            Path(__file__).parent.parent
            / "amplifier_module_provider_github_copilot"
            / "config"
            / "models.yaml"
        )
        with config_path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert data["provider"]["defaults"]["context_window"] == 200000

    def test_models_yaml_max_output_tokens_32000(self) -> None:
        """models.yaml defaults.max_output_tokens is 32000.

        SDK limits: max_context_window=200000, max_prompt_tokens=168000
        Therefore: max_output_tokens = 200000 - 168000 = 32000

        This is the OUTPUT budget used by context manager for budget calculation.
        Contract anchor: provider-protocol:get_info:MUST:1
        """
        config_path = (
            Path(__file__).parent.parent
            / "amplifier_module_provider_github_copilot"
            / "config"
            / "models.yaml"
        )
        with config_path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert data["provider"]["defaults"]["max_output_tokens"] == 32000

    def test_models_yaml_contains_claude_opus_45_model_entry(self) -> None:
        """models.yaml contains claude-opus-4.5 in models list.

        Contract anchor: provider-protocol:list_models:MUST:1
        """
        config_path = (
            Path(__file__).parent.parent
            / "amplifier_module_provider_github_copilot"
            / "config"
            / "models.yaml"
        )
        with config_path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)

        model_ids = [m["id"] for m in data["models"]]
        assert "claude-opus-4.5" in model_ids


class TestProviderIntegration:
    """Tests for provider integration with new defaults."""

    def test_provider_get_info_defaults_model_claude_opus_45(self) -> None:
        """Provider.get_info().defaults['model'] is claude-opus-4.5.

        Contract anchor: provider-protocol:get_info:MUST:1
        """
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        provider = GitHubCopilotProvider()
        info = provider.get_info()
        assert info.defaults.get("model") == "claude-opus-4.5"

    @pytest.mark.asyncio
    async def test_provider_list_models_contains_claude_opus_45(self) -> None:
        """Provider.list_models() includes claude-opus-4.5.

        Contract anchor: provider-protocol:list_models:MUST:1
        """
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        provider = GitHubCopilotProvider()
        models = await provider.list_models()
        model_ids = [m.id for m in models]
        assert "claude-opus-4.5" in model_ids
