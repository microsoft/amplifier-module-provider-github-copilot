"""
Tests for module mount function.

This module tests the mount() entry point and prerequisite checking.
"""

from unittest.mock import AsyncMock, patch

import pytest

from amplifier_module_provider_github_copilot import mount


class TestMount:
    """Tests for the mount function."""

    @pytest.mark.asyncio
    async def test_mount_success(self, mock_coordinator):
        """Mount should succeed when prerequisites are met."""
        with patch("shutil.which", return_value="/usr/bin/copilot"):
            with patch("os.path.isfile", return_value=True):
                with patch("os.path.isabs", return_value=True):
                    with patch("amplifier_module_provider_github_copilot._ensure_executable"):
                        cleanup = await mount(mock_coordinator, {"model": "claude-opus-4.5"})

                        # Should return cleanup function
                        assert cleanup is not None
                        assert callable(cleanup)

                        # Provider should be mounted
                        assert "github-copilot" in mock_coordinator.mounted_providers

    @pytest.mark.asyncio
    async def test_mount_missing_cli(self, mock_coordinator):
        """Mount should return None when CLI not found."""
        with patch("shutil.which", return_value=None):
            cleanup = await mount(mock_coordinator, {})

            # Should return None (graceful degradation)
            assert cleanup is None

            # Provider should not be mounted
            assert "github-copilot" not in mock_coordinator.mounted_providers

    @pytest.mark.asyncio
    async def test_mount_cleanup_function(self, mock_coordinator):
        """Cleanup function should close provider."""
        with patch("shutil.which", return_value="/usr/bin/copilot"):
            with patch("os.path.isfile", return_value=True):
                with patch("os.path.isabs", return_value=True):
                    with patch("amplifier_module_provider_github_copilot._ensure_executable"):
                        cleanup = await mount(mock_coordinator, {})

                        assert cleanup is not None

                        # Get the mounted provider
                        provider = mock_coordinator.mounted_providers.get("github-copilot")
                        assert provider is not None

                        # Mock the provider's close method
                        with patch.object(provider, "close", new_callable=AsyncMock) as mock_close:
                            await cleanup()
                            mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_mount_default_config(self, mock_coordinator):
        """Mount should work with no config (uses defaults)."""
        with patch("shutil.which", return_value="/usr/bin/copilot"):
            with patch("os.path.isfile", return_value=True):
                with patch("os.path.isabs", return_value=True):
                    with patch("amplifier_module_provider_github_copilot._ensure_executable"):
                        cleanup = await mount(mock_coordinator, None)

                        assert cleanup is not None
                        provider = mock_coordinator.mounted_providers.get("github-copilot")
                        assert provider._model == "claude-opus-4.5"  # Default model

    @pytest.mark.asyncio
    async def test_mount_registers_with_coordinator(self, mock_coordinator):
        """Mount should call coordinator.mount with correct arguments."""
        with patch("shutil.which", return_value="/usr/bin/copilot"):
            with patch("os.path.isfile", return_value=True):
                with patch("os.path.isabs", return_value=True):
                    with patch("amplifier_module_provider_github_copilot._ensure_executable"):
                        await mount(mock_coordinator, {})

                        # Verify mount was called
                        mock_coordinator.mount.assert_called_once()
                        call_args = mock_coordinator.mount.call_args

                        assert call_args[0][0] == "providers"  # category
                        assert call_args[1]["name"] == "github-copilot"


class TestModuleMetadata:
    """Tests for module metadata."""

    def test_module_type(self):
        """Module should declare correct type."""
        from amplifier_module_provider_github_copilot import __amplifier_module_type__

        assert __amplifier_module_type__ == "provider"

    def test_exports(self):
        """Module should export expected symbols."""
        from amplifier_module_provider_github_copilot import (
            ChatResponse,
            CopilotProviderError,
            CopilotSdkProvider,
            ProviderInfo,
            ToolCall,
            mount,
        )

        assert mount is not None
        assert CopilotSdkProvider is not None
        assert ProviderInfo is not None
        assert ChatResponse is not None
        assert ToolCall is not None
        assert CopilotProviderError is not None

    def test_get_provider_class(self):
        """get_provider_class should return provider class."""
        from amplifier_module_provider_github_copilot import get_provider_class

        cls = get_provider_class()
        assert cls.__name__ == "CopilotSdkProvider"


# ═══════════════════════════════════════════════════════════════════════════
# Coverage gap tests: mount() error path, _find_copilot_cli() branches
# ═══════════════════════════════════════════════════════════════════════════


class TestMountErrorHandling:
    """Tests for mount() exception handling (lines 199-201)."""

    @pytest.mark.asyncio
    async def test_mount_returns_none_on_provider_init_error(self, mock_coordinator):
        """mount() should return None when provider initialization fails."""
        with patch("shutil.which", return_value="/usr/bin/copilot"):
            with patch("os.path.isfile", return_value=True):
                with patch("os.path.isabs", return_value=True):
                    with patch("amplifier_module_provider_github_copilot._ensure_executable"):
                        # Make coordinator.mount raise during provider registration
                        mock_coordinator.mount = AsyncMock(
                            side_effect=RuntimeError("Mount failed: config error")
                        )

                        cleanup = await mount(mock_coordinator, {})

                        # Should return None (graceful degradation)
                        assert cleanup is None


class TestFindCopilotCli:
    """Tests for _find_copilot_cli() functionality."""

    def test_cli_from_shutil_which(self):
        """_find_copilot_cli should find CLI via shutil.which()."""
        from amplifier_module_provider_github_copilot import _find_copilot_cli

        with patch.dict("os.environ", {}, clear=True):
            with patch("shutil.which", return_value="/usr/bin/copilot"):
                with patch("amplifier_module_provider_github_copilot._ensure_executable"):
                    result = _find_copilot_cli({})

                    assert result == "/usr/bin/copilot"

    def test_cli_not_found_returns_none(self):
        """_find_copilot_cli should return None when CLI not found."""
        from amplifier_module_provider_github_copilot import _find_copilot_cli

        with patch.dict("os.environ", {}, clear=True):
            with patch("shutil.which", return_value=None):
                result = _find_copilot_cli({})

                assert result is None

    def test_cli_discovery_exception_returns_none(self):
        """_find_copilot_cli should return None on unexpected exceptions."""
        from amplifier_module_provider_github_copilot import _find_copilot_cli

        with patch.dict("os.environ", {}, clear=True):
            with patch("shutil.which", side_effect=OSError("Permission denied")):
                result = _find_copilot_cli({})

                assert result is None

    def test_cli_finds_copilot_exe_fallback(self):
        """_find_copilot_cli should find copilot.exe when copilot is not found."""
        from amplifier_module_provider_github_copilot import _find_copilot_cli

        def which_side_effect(name):
            if name == "copilot":
                return None
            if name == "copilot.exe":
                return "C:\\Program Files\\copilot\\copilot.exe"
            return None

        with patch.dict("os.environ", {}, clear=True):
            with patch("shutil.which", side_effect=which_side_effect):
                with patch("amplifier_module_provider_github_copilot._ensure_executable"):
                    result = _find_copilot_cli({})
                    assert result == "C:\\Program Files\\copilot\\copilot.exe"
