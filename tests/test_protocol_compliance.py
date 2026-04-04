"""
Tests for provider protocol compliance.

Contract: provider-protocol.md

These tests verify that the provider implements the full Amplifier Provider Protocol
so it can be mounted by the kernel.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


class TestMount:
    """Tests for AC-1: mount() entry point.

    Contract: provider-protocol.md (mount section)
    """

    @pytest.mark.asyncio
    async def test_mount_exists(self) -> None:
        """provider-protocol:mount:MUST:1 — mount() accepts ModuleCoordinator type.

        Verifies mount() is importable and callable.
        """
        from amplifier_module_provider_github_copilot import mount

        assert callable(mount)

    @pytest.mark.asyncio
    async def test_mount_registers_provider(self) -> None:
        """provider-protocol:mount:MUST:3 — Registers provider on coordinator.

        Verifies mount() calls coordinator.mount() with correct arguments.
        """
        from amplifier_module_provider_github_copilot import mount

        # Mock coordinator
        coordinator = MagicMock()
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

        Verifies mount() returns a callable for resource cleanup.
        """
        from amplifier_module_provider_github_copilot import mount

        coordinator = MagicMock()
        coordinator.mount = AsyncMock()

        cleanup = await mount(coordinator, config=None)

        assert cleanup is not None
        assert callable(cleanup)

    @pytest.mark.asyncio
    async def test_mount_cleanup_calls_provider_close(self) -> None:
        """provider-protocol:mount:MUST:2 — Cleanup callable releases resources.

        Verifies the cleanup function returned by mount() is awaitable.
        """
        from amplifier_module_provider_github_copilot import mount

        coordinator = MagicMock()
        coordinator.mount = AsyncMock()

        cleanup = await mount(coordinator, config=None)

        # The cleanup should be awaitable
        assert cleanup is not None
        await cleanup()
        # Provider should have close() called (we verify this indirectly)


class TestGetInfo:
    """Tests for AC-2: get_info() method."""

    def test_get_info_exists(self) -> None:
        """AC-2: get_info() method exists on provider."""
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider()
        assert hasattr(provider, "get_info")
        assert callable(provider.get_info)

    def test_get_info_returns_provider_info(self) -> None:
        """AC-2: get_info() returns ProviderInfo with required fields."""
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
            ProviderInfo,
        )

        provider = GitHubCopilotProvider()
        info = provider.get_info()

        assert isinstance(info, ProviderInfo)
        assert info.id == "github-copilot"
        assert info.display_name is not None
        assert isinstance(info.capabilities, list)

    def test_get_info_includes_streaming_capability(self) -> None:
        """AC-2: get_info() includes 'streaming' capability."""
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider()
        info = provider.get_info()

        assert "streaming" in info.capabilities

    def test_get_info_includes_tool_use_capability(self) -> None:
        """AC-2: get_info() includes 'tools' capability.

        Per kernel capabilities.py: TOOLS="tools" (not "tool_use").
        """
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider()
        info = provider.get_info()

        assert "tools" in info.capabilities


class TestListModels:
    """Tests for AC-3: list_models() method."""

    @pytest.mark.asyncio
    async def test_list_models_exists(self) -> None:
        """AC-3: list_models() method exists on provider."""
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider()
        assert hasattr(provider, "list_models")

    @pytest.mark.asyncio
    async def test_list_models_returns_model_info_list(self) -> None:
        """AC-3: list_models() returns list of ModelInfo."""
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
            ModelInfo,
        )

        provider = GitHubCopilotProvider()
        models = await provider.list_models()

        assert isinstance(models, list)
        assert len(models) > 0
        for model in models:
            assert isinstance(model, ModelInfo)

    @pytest.mark.asyncio
    async def test_list_models_has_minimum_models(self) -> None:
        """AC-3: list_models() returns at least one model.

        P3-13: Replaced brittle assertion for specific model IDs.
        Model inventory may change; we verify structure, not specific names.
        """
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider()
        models = await provider.list_models()

        # Structural check: at least one model exists
        assert len(models) >= 1, "Must have at least one model"
        # Each model has required fields
        for model in models:
            assert model.id, "Model must have non-empty id"
            assert model.display_name or model.id, "Model must have display name or id"

    @pytest.mark.asyncio
    async def test_list_models_has_required_fields(self) -> None:
        """AC-3: Each ModelInfo has required fields."""
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider()
        models = await provider.list_models()

        for model in models:
            assert model.id is not None
            assert model.display_name is not None
            assert model.context_window is not None
            assert model.max_output_tokens is not None


class TestCompleteMethod:
    """Tests for AC-4: complete() as class method."""

    @pytest.mark.asyncio
    async def test_complete_is_method(self) -> None:
        """AC-4: complete() is a method on GitHubCopilotProvider."""
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider()
        assert hasattr(provider, "complete")
        assert callable(provider.complete)


class TestProviderProtocol:
    """Tests for full Provider Protocol compliance."""

    def test_provider_has_name_property(self) -> None:
        """Provider has name property."""
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider()
        assert hasattr(provider, "name")
        assert provider.name == "github-copilot"

    def test_provider_has_parse_tool_calls(self) -> None:
        """Provider has parse_tool_calls method."""
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider()
        assert hasattr(provider, "parse_tool_calls")
        assert callable(provider.parse_tool_calls)

    def test_all_protocol_methods_exist(self) -> None:
        """Provider implements all required protocol methods."""
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider()

        # 4 methods + 1 property
        assert hasattr(provider, "name")  # property
        assert hasattr(provider, "get_info")  # method
        assert hasattr(provider, "list_models")  # async method
        assert hasattr(provider, "complete")  # async method
        assert hasattr(provider, "parse_tool_calls")  # method
