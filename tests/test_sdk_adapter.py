"""
Tests for SDK Adapter structure and client exports.

Contract: contracts/sdk-boundary.md
"""

from __future__ import annotations

from dataclasses import fields


class TestSDKAdapterTypes:
    """Test domain types exposed by sdk_adapter."""

    def test_session_config_has_model(self) -> None:
        """AC-4: SessionConfig has model field."""
        from amplifier_module_provider_github_copilot.sdk_adapter.types import SessionConfig

        config = SessionConfig(model="gpt-4")
        assert config.model == "gpt-4"

    def test_session_config_has_system_prompt(self) -> None:
        """AC-4: SessionConfig has system_prompt field with None default."""
        from amplifier_module_provider_github_copilot.sdk_adapter.types import SessionConfig

        config = SessionConfig(model="gpt-4")
        assert config.system_prompt is None

        config_with_prompt = SessionConfig(model="gpt-4", system_prompt="You are helpful.")
        assert config_with_prompt.system_prompt == "You are helpful."

    def test_session_config_has_max_tokens(self) -> None:
        """AC-4: SessionConfig has max_tokens field with None default."""
        from amplifier_module_provider_github_copilot.sdk_adapter.types import SessionConfig

        config = SessionConfig(model="gpt-4")
        assert config.max_tokens is None

        config_with_tokens = SessionConfig(model="gpt-4", max_tokens=1000)
        assert config_with_tokens.max_tokens == 1000

    def test_session_config_is_dataclass(self) -> None:
        """Contract: sdk-boundary:Types:MUST:2 — SessionConfig is a dataclass."""
        from amplifier_module_provider_github_copilot.sdk_adapter.types import SessionConfig

        field_names = {f.name for f in fields(SessionConfig)}
        assert "model" in field_names
        assert "system_prompt" in field_names
        assert "max_tokens" in field_names


class TestSDKAdapterExports:
    """Test sdk_adapter module exports."""

    def test_exports_session_config(self) -> None:
        """sdk_adapter exports SessionConfig."""
        from amplifier_module_provider_github_copilot.sdk_adapter import SessionConfig

        # Contract: sdk-boundary:Membrane:MUST:2
        assert isinstance(SessionConfig, type)

    def test_exports_copilot_client_wrapper(self) -> None:
        """sdk_adapter exports CopilotClientWrapper."""
        from amplifier_module_provider_github_copilot.sdk_adapter import CopilotClientWrapper

        # Contract: sdk-boundary:Membrane:MUST:2
        assert isinstance(CopilotClientWrapper, type)
