"""
Tests for provider protocol compliance.

Contract: provider-protocol.md

These tests verify that the provider implements the full Amplifier Provider Protocol
so it can be mounted by the kernel.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from amplifier_core import ModuleCoordinator


class TestMount:
    """Tests for AC-1: mount() entry point.

    Contract: provider-protocol.md (mount section)
    """

    @pytest.mark.asyncio
    async def test_mount_registers_provider(self) -> None:
        """provider-protocol:mount:MUST:3 — Registers provider on coordinator.

        Verifies mount() calls coordinator.mount() with correct arguments.
        """
        # Contract: provider-protocol:mount:MUST:3
        from amplifier_module_provider_github_copilot import mount

        # Mock coordinator with spec to catch interface mismatches
        coordinator = MagicMock(spec=ModuleCoordinator)
        coordinator.mount = AsyncMock()

        # Call mount
        _ = await mount(coordinator, config=None)

        # Verify provider was mounted
        coordinator.mount.assert_called_once()
        call_args = coordinator.mount.call_args
        assert call_args[0][0] == "providers"  # First positional arg
        assert call_args[1]["name"] == "github-copilot"  # Keyword arg

    @pytest.mark.asyncio
    async def test_mount_returns_cleanup_callable(self) -> None:
        """provider-protocol:mount:MUST:2 — Returns cleanup callable.

        Verifies mount() returns an async cleanup coroutine function.
        """
        # Contract: provider-protocol:mount:MUST:2
        from amplifier_module_provider_github_copilot import mount

        coordinator = MagicMock(spec=ModuleCoordinator)
        coordinator.mount = AsyncMock()

        cleanup = await mount(coordinator, config=None)

        assert asyncio.iscoroutinefunction(cleanup)
        assert callable(cleanup)


class TestGetInfo:
    """Tests for AC-2: get_info() method."""

    def test_get_info_returns_provider_info(self) -> None:
        """get_info() returns ProviderInfo with required fields."""
        # Contract: provider-protocol:get_info:MUST:1
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
            ProviderInfo,
        )

        provider = GitHubCopilotProvider()
        info = provider.get_info()

        assert isinstance(info, ProviderInfo)
        assert info.id == "github-copilot"
        assert info.display_name == "GitHub Copilot SDK"
        assert isinstance(info.capabilities, list)

    def test_get_info_includes_streaming_capability(self) -> None:
        """get_info() includes 'streaming' capability."""
        # Contract: provider-protocol:get_info:MUST:1
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider()
        info = provider.get_info()

        assert "streaming" in info.capabilities

    def test_get_info_includes_tool_use_capability(self) -> None:
        """get_info() includes 'tools' capability.

        Per kernel capabilities.py: TOOLS="tools" (not "tool_use").
        """
        # Contract: provider-protocol:get_info:MUST:1
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider()
        info = provider.get_info()

        assert "tools" in info.capabilities


class TestListModels:
    """Tests for AC-3: list_models() method."""

    @pytest.mark.asyncio
    async def test_list_models_returns_model_info_list(self) -> None:
        """list_models() returns list of ModelInfo."""
        # Contract: provider-protocol:list_models:MUST:1
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
            ModelInfo,
        )

        provider = GitHubCopilotProvider()
        models = await provider.list_models()

        assert isinstance(models, list)
        assert models, "list_models must return at least one model"
        for model in models:
            assert isinstance(model, ModelInfo)

    @pytest.mark.asyncio
    async def test_list_models_has_minimum_models(self) -> None:
        """list_models() returns at least one model with valid id.

        P3-13: Replaced brittle assertion for specific model IDs.
        Model inventory may change; we verify structure, not specific names.
        """
        # Contract: provider-protocol:list_models:MUST:1
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider()
        models = await provider.list_models()

        # Structural check: at least one model exists
        assert models, "list_models must return at least one model"
        # Each model has required fields
        for model in models:
            assert isinstance(model.id, str) and model.id

    @pytest.mark.asyncio
    async def test_list_models_has_required_fields(self) -> None:
        """Each ModelInfo has required fields with correct types."""
        # Contract: provider-protocol:list_models:MUST:2
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider()
        models = await provider.list_models()

        for model in models:
            assert isinstance(model.id, str) and model.id
            assert isinstance(model.display_name, str) and model.display_name
            assert isinstance(model.context_window, int) and model.context_window > 0
            assert isinstance(model.max_output_tokens, int) and model.max_output_tokens > 0


class TestProviderProtocol:
    """Tests for full Provider Protocol compliance."""

    def test_provider_has_name_property(self) -> None:
        """Provider has name property returning correct value."""
        # Contract: provider-protocol:name:MUST:1,2
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider()
        assert provider.name == "github-copilot"
        assert isinstance(GitHubCopilotProvider.name, property)
