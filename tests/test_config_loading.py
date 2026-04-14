"""Tests for config loading functionality.

Contract: contracts/provider-protocol.md

Tests verify that provider identity and model catalog are loaded from
config/_models.py instead of being hardcoded in Python.
"""

from __future__ import annotations

import pytest


class TestLoadModelsConfig:
    """Tests for _load_models_config() function."""

    def test_load_models_config_returns_provider_id(self) -> None:
        """Models config loader returns correct provider id from YAML.

        Contract: provider-protocol:get_info:MUST:1
        """
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

        Contract: provider-protocol:list_models:MUST:1
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
        assert info.defaults.get("model") == "claude-opus-4.5"

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
    """Tests verify config/models.py (_models.py) has correct structure.

    Updated: models.yaml migrated to config/_models.py (Python dataclass).
    """

    def test_models_yaml_version_field_present(self) -> None:
        """Models config has version field."""
        from amplifier_module_provider_github_copilot.config import _models as _models

        assert _models.VERSION == "1.0"

    def test_models_yaml_provider_id(self) -> None:
        """Models config provider.id equals github-copilot."""
        from amplifier_module_provider_github_copilot.config import _models as _models

        assert _models.PROVIDER["id"] == "github-copilot"

    def test_models_yaml_models_list_nonempty(self) -> None:
        """Models config has non-empty models list."""
        from amplifier_module_provider_github_copilot.config import _models as _models

        assert isinstance(_models.MODELS, list)
        assert len(_models.MODELS) > 0

    def test_models_yaml_each_model_has_required_fields(self) -> None:
        """Each model in config has required fields."""
        from amplifier_module_provider_github_copilot.config import _models as _models

        required_fields = ["id", "display_name", "context_window", "max_output_tokens"]
        for model in _models.MODELS:
            for field in required_fields:
                assert field in model, f"Model {model.get('id', 'unknown')} missing field: {field}"


# ============================================================================
# Fake Tool Detection Config Loading Tests
# Contract: behaviors:Config:MUST:1
# ============================================================================


class TestFakeToolDetectionConfigLoading:
    """Tests for fake tool detection config loading.

    Contract: behaviors:Config:MUST:1
    """

    def test_config_loads_defaults(self) -> None:  # Contract: provider-protocol:complete:MUST:6
        """Config loads patterns, max_attempts, message from defaults.

        Contract: behaviors:Config:MUST:1
        """
        from amplifier_module_provider_github_copilot.fake_tool_detection import (
            load_fake_tool_detection_config,
        )

        config = load_fake_tool_detection_config()
        assert len(config.patterns) == 3
        assert config.max_correction_attempts == 2
        expected_msg = (
            "You wrote tool calls as plain text instead of using the structured"
            " tool calling mechanism. Please use actual tool calls, not text"
            " representations of them."
        )
        assert config.correction_message == expected_msg

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

    def test_config_logging_section_loaded(self) -> None:
        """Logging config section loaded from YAML.

        Contract: behaviors:Logging:MUST:1
        """
        from amplifier_module_provider_github_copilot.fake_tool_detection import (
            load_fake_tool_detection_config,
        )

        config = load_fake_tool_detection_config()
        assert config.logging.log_matched_pattern is True
        assert config.logging.level_on_detection == "INFO"


# ============================================================================
# Models YAML Default Values Tests
# Contract: behaviors:Config:MUST:2
# Migrated from test_default_model.py per behaviors:TestFiles:MUST:1
# ============================================================================


class TestModelsYamlDefaultValues:
    """Tests for models config default values.

    Contract: behaviors:Config:MUST:2
    Contract: behaviors:TestFiles:MUST:1,2,3
    Two-Medium Architecture: config/_models.py is authoritative source for policy values.
    """

    def test_models_yaml_has_timeout(self) -> None:
        """config/_models.py must have PROVIDER.defaults.timeout == 3600.

        Contract: behaviors:Config:MUST:2
        """
        from amplifier_module_provider_github_copilot.config import _models

        assert _models.PROVIDER["defaults"]["timeout"] == 3600

    def test_load_models_config_reads_yaml(self) -> None:
        """load_models_config() reads values from config/_models.py.

        Contract: behaviors:Config:MUST:2
        Two-Medium: Python loads from config, no hardcoded fallbacks.
        """
        from amplifier_module_provider_github_copilot.config_loader import load_models_config

        config = load_models_config()

        # Values should match config/_models.py
        assert config.defaults["model"] == "claude-opus-4.5"
        assert config.defaults["timeout"] == 3600
        assert config.defaults["context_window"] == 200000

    def test_models_yaml_default_model_is_claude_opus_45(self) -> None:
        """config/_models.py PROVIDER.defaults.model is claude-opus-4.5.

        Contract: behaviors:Config:MUST:2
        """
        from amplifier_module_provider_github_copilot.config import _models

        assert _models.PROVIDER["defaults"]["model"] == "claude-opus-4.5"

    def test_models_yaml_context_window_200000(self) -> None:
        """config/_models.py PROVIDER.defaults.context_window is 200000.

        Contract: behaviors:Config:MUST:2
        """
        from amplifier_module_provider_github_copilot.config import _models

        assert _models.PROVIDER["defaults"]["context_window"] == 200000

    def test_models_yaml_max_output_tokens_32000(self) -> None:
        """config/_models.py PROVIDER.defaults.max_output_tokens is 32000.

        SDK limits: max_context_window=200000, max_prompt_tokens=168000
        Therefore: max_output_tokens = 200000 - 168000 = 32000

        Contract: behaviors:Config:MUST:2
        """
        from amplifier_module_provider_github_copilot.config import _models

        assert _models.PROVIDER["defaults"]["max_output_tokens"] == 32000

    def test_models_yaml_contains_claude_opus_45_model_entry(self) -> None:
        """config/_models.py MODELS list contains claude-opus-4.5.

        Contract: provider-protocol:list_models:MUST:1
        """
        from amplifier_module_provider_github_copilot.config import _models

        model_ids = [m["id"] for m in _models.MODELS]
        assert "claude-opus-4.5" in model_ids

    def test_provider_get_info_defaults_model_claude_opus_45(self) -> None:
        """Provider.get_info().defaults['model'] is claude-opus-4.5.

        Contract: provider-protocol:get_info:MUST:1
        """
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider()
        info = provider.get_info()
        assert info.defaults.get("model") == "claude-opus-4.5"
