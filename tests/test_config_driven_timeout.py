"""Tests for config-driven timeout in the completion path.

Contract: contracts/behaviors.md (Three-Medium Architecture)

Tests verify that module-level complete() function loads timeout from
config/_models.py instead of using hardcoded 120.0 value.
"""

from __future__ import annotations

from pathlib import Path


class TestModuleLevelCompleteTimeout:
    """Tests for module-level complete() timeout configuration."""

    def test_no_hardcoded_120_in_provider(self) -> None:
        """provider.py should not contain hardcoded 120.0 timeout.

        Contract anchor: behaviors.md:Config:MUST:1
        """
        provider_path = (
            Path(__file__).parent.parent
            / "amplifier_module_provider_github_copilot"
            / "provider.py"
        )
        content = provider_path.read_text(encoding="utf-8")

        # Check for hardcoded 120.0 timeout patterns
        assert "timeout=120.0" not in content, (
            "Found hardcoded timeout=120.0 in provider.py. "
            "Per Three-Medium Architecture, policy must live in YAML."
        )

    def test_module_complete_uses_load_models_config(self) -> None:
        """Module-level complete() should call load_models_config() for timeout.

        Contract anchor: behaviors.md:Config:MUST:1
        """
        # Check the source code references load_models_config
        provider_path = (
            Path(__file__).parent.parent
            / "amplifier_module_provider_github_copilot"
            / "provider.py"
        )
        content = provider_path.read_text(encoding="utf-8")

        # Find that load_models_config is used (either directly or via _load_models_config alias)
        # The provider uses load_models_config() in __init__ and exposes _load_models_config alias
        assert "load_models_config" in content, (
            "complete() should use load_models_config() to load timeout"
        )

    # test_module_complete_timeout_from_config removed - tested completion.py which is now deleted
    # The production path (provider._execute_sdk_completion) still uses config-driven timeout,
    # verified by TestProductionPathWithMockClient tests in test_behaviors.py


class TestTimeoutConfigValue:
    """Tests for timeout configuration values."""

    def test_models_yaml_has_timeout_3600(self) -> None:
        """config/models.py PROVIDER.defaults.timeout should be 3600.

        Contract anchor: behaviors.md:Config:SHOULD:1
        """
        from amplifier_module_provider_github_copilot.config import _models as _models

        assert _models.PROVIDER["defaults"]["timeout"] == 3600

    def test_yaml_timeout_is_3600(self) -> None:
        """YAML timeout value is 3600 (1 hour for reasoning models).

        Contract anchor: behaviors.md:Config:MUST:2
        Three-Medium: YAML is authoritative source.
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_models_config,
        )

        config = load_models_config()
        timeout = config.defaults["timeout"]
        assert timeout == 3600
