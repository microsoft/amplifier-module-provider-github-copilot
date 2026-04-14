"""Tests for ConfigField conformance.

Contract: provider-protocol.md provider-protocol:get_info:MUST:3

These tests verify that get_info() returns config_fields with proper
token field for init wizard integration.
"""

from __future__ import annotations

from amplifier_core import ConfigField

from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider


class TestConfigFieldConformance:
    """Tests for ConfigField in get_info()."""

    def test_get_info_includes_config_fields(self) -> None:
        """Contract: provider-protocol:get_info:MUST:3 - includes config_fields."""
        provider = GitHubCopilotProvider()
        info = provider.get_info()

        assert isinstance(info.config_fields, list), "config_fields must be a list"
        assert len(info.config_fields) > 0, "config_fields must not be empty"

    def test_config_fields_has_github_token(self) -> None:
        """Contract: provider-protocol:get_info:MUST:3 - includes GitHub token field."""
        provider = GitHubCopilotProvider()
        info = provider.get_info()

        field_ids = [f.id for f in info.config_fields]
        assert "github_token" in field_ids, (
            f"config_fields must include github_token field. Found: {field_ids}"
        )

    def test_github_token_field_is_secret(self) -> None:
        """Contract: ConfigField for token must use field_type='secret'.

        Contract: provider-protocol:get_info:MUST:3
        """
        provider = GitHubCopilotProvider()
        info = provider.get_info()

        token_fields = [f for f in info.config_fields if f.id == "github_token"]
        assert len(token_fields) == 1, "Must have exactly one github_token field"

        token_field = token_fields[0]
        assert token_field.field_type == "secret", (
            f"github_token field_type must be 'secret', got '{token_field.field_type}'"
        )

    def test_github_token_field_has_env_var(self) -> None:
        """Contract: GitHub token field must specify env_var.

        Contract: provider-protocol:get_info:MUST:3
        """
        provider = GitHubCopilotProvider()
        info = provider.get_info()

        token_fields = [f for f in info.config_fields if f.id == "github_token"]
        token_field = token_fields[0]

        assert token_field.env_var == "GITHUB_TOKEN", (
            f"env_var must be 'GITHUB_TOKEN', got '{token_field.env_var}'"
        )

    def test_all_config_fields_are_valid(self) -> None:
        """All config_fields must be valid ConfigField instances.

        Contract: provider-protocol:get_info:MUST:1
        """
        provider = GitHubCopilotProvider()
        info = provider.get_info()

        for field in info.config_fields:
            assert isinstance(field, ConfigField), (
                f"config_field must be ConfigField instance, got {type(field)}"
            )

    def test_github_token_field_display_and_prompt(self) -> None:
        """github_token ConfigField has exact display_name and prompt.

        Contract: provider-protocol:get_info:MUST:3
        """
        provider = GitHubCopilotProvider()
        info = provider.get_info()

        token_fields = [f for f in info.config_fields if f.id == "github_token"]
        assert len(token_fields) == 1, "Must have exactly one github_token field"
        token_field = token_fields[0]

        assert token_field.display_name == "GitHub Token"
        assert token_field.prompt == "Enter your GitHub token (or Copilot agent token)"
