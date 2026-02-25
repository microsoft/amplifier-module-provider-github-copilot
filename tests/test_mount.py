"""
Tests for module mount function.

This module tests the mount() entry point and prerequisite checking.
"""

from unittest.mock import AsyncMock, Mock, patch

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
        """Cleanup function should release the shared client reference."""
        with patch("shutil.which", return_value="/usr/bin/copilot"):
            with patch("amplifier_module_provider_github_copilot._ensure_executable"):
                with patch(
                    "amplifier_module_provider_github_copilot.CopilotClientWrapper"
                ) as mock_wrapper_cls:
                    mock_wrapper_cls.return_value = AsyncMock()

                    cleanup = await mount(mock_coordinator, {})
                    assert cleanup is not None

                    import amplifier_module_provider_github_copilot as mod

                    assert mod._shared_client_refcount == 1

                    await cleanup()
                    assert mod._shared_client_refcount == 0

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


class TestSingleton:
    """Tests for the process-level singleton CopilotClientWrapper."""

    @pytest.mark.asyncio
    async def test_singleton_creates_one_wrapper(self, mock_coordinator):
        """First mount should create exactly one CopilotClientWrapper."""
        with patch("shutil.which", return_value="/usr/bin/copilot"):
            with patch("amplifier_module_provider_github_copilot._ensure_executable"):
                with patch(
                    "amplifier_module_provider_github_copilot.CopilotClientWrapper"
                ) as mock_wrapper_cls:
                    mock_wrapper_cls.return_value = Mock()

                    await mount(mock_coordinator, {})

                    mock_wrapper_cls.assert_called_once()

    @pytest.mark.asyncio
    async def test_singleton_reuses_wrapper_across_mounts(self):
        """Multiple mounts should reuse the same CopilotClientWrapper instance."""
        with patch("shutil.which", return_value="/usr/bin/copilot"):
            with patch("amplifier_module_provider_github_copilot._ensure_executable"):
                with patch(
                    "amplifier_module_provider_github_copilot.CopilotClientWrapper"
                ) as mock_wrapper_cls:
                    mock_wrapper_cls.return_value = Mock()

                    coordinator_a = Mock()
                    coordinator_a.mount = AsyncMock()
                    coordinator_a.hooks = Mock()
                    coordinator_a.hooks.emit = AsyncMock()

                    coordinator_b = Mock()
                    coordinator_b.mount = AsyncMock()
                    coordinator_b.hooks = Mock()
                    coordinator_b.hooks.emit = AsyncMock()

                    coordinator_c = Mock()
                    coordinator_c.mount = AsyncMock()
                    coordinator_c.hooks = Mock()
                    coordinator_c.hooks.emit = AsyncMock()

                    await mount(coordinator_a, {})
                    await mount(coordinator_b, {})
                    await mount(coordinator_c, {})

                    # Only ONE wrapper should ever be created
                    assert mock_wrapper_cls.call_count == 1

                    import amplifier_module_provider_github_copilot as mod

                    assert mod._shared_client_refcount == 3

    @pytest.mark.asyncio
    async def test_singleton_close_only_on_last_cleanup(self):
        """close() should be called only when the last session's cleanup runs."""
        with patch("shutil.which", return_value="/usr/bin/copilot"):
            with patch("amplifier_module_provider_github_copilot._ensure_executable"):
                with patch(
                    "amplifier_module_provider_github_copilot.CopilotClientWrapper"
                ) as mock_wrapper_cls:
                    mock_client_instance = AsyncMock()
                    mock_client_instance.close = AsyncMock()
                    mock_wrapper_cls.return_value = mock_client_instance

                    coordinator_a = Mock()
                    coordinator_a.mount = AsyncMock()
                    coordinator_a.hooks = Mock()
                    coordinator_a.hooks.emit = AsyncMock()

                    coordinator_b = Mock()
                    coordinator_b.mount = AsyncMock()
                    coordinator_b.hooks = Mock()
                    coordinator_b.hooks.emit = AsyncMock()

                    cleanup_a = await mount(coordinator_a, {})
                    cleanup_b = await mount(coordinator_b, {})

                    assert cleanup_a is not None
                    assert cleanup_b is not None

                    await cleanup_a()
                    # close() must NOT have been called yet — b is still mounted
                    mock_client_instance.close.assert_not_called()

                    await cleanup_b()
                    # Now the last reference is gone — close() must have been called
                    mock_client_instance.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_singleton_concurrent_mounts_create_one_wrapper(self):
        """Concurrent mount() calls must not create more than one CopilotClientWrapper."""
        import asyncio

        with patch("shutil.which", return_value="/usr/bin/copilot"):
            with patch("amplifier_module_provider_github_copilot._ensure_executable"):
                with patch(
                    "amplifier_module_provider_github_copilot.CopilotClientWrapper"
                ) as mock_wrapper_cls:
                    mock_wrapper_cls.return_value = Mock()

                    def make_coordinator():
                        c = Mock()
                        c.mount = AsyncMock()
                        c.hooks = Mock()
                        c.hooks.emit = AsyncMock()
                        return c

                    coordinators = [make_coordinator() for _ in range(5)]
                    await asyncio.gather(*[mount(c, {}) for c in coordinators])

                    # All five concurrent mounts must share ONE wrapper
                    assert mock_wrapper_cls.call_count == 1

                    import amplifier_module_provider_github_copilot as mod

                    assert mod._shared_client_refcount == 5

    @pytest.mark.asyncio
    async def test_singleton_logs_debug_on_timeout_mismatch(self, caplog):
        """Mismatched timeout on second mount emits DEBUG log, does not raise."""
        import logging

        with patch("shutil.which", return_value="/usr/bin/copilot"):
            with patch("amplifier_module_provider_github_copilot._ensure_executable"):
                with patch(
                    "amplifier_module_provider_github_copilot.CopilotClientWrapper"
                ) as mock_wrapper_cls:
                    mock_wrapper_cls.return_value = Mock(_timeout=300.0)

                    coordinator_a = Mock()
                    coordinator_a.mount = AsyncMock()
                    coordinator_a.hooks = Mock()
                    coordinator_a.hooks.emit = AsyncMock()

                    coordinator_b = Mock()
                    coordinator_b.mount = AsyncMock()
                    coordinator_b.hooks = Mock()
                    coordinator_b.hooks.emit = AsyncMock()

                    await mount(coordinator_a, {"timeout": 300.0})

                    with caplog.at_level(logging.DEBUG):
                        cleanup = await mount(coordinator_b, {"timeout": 600.0})

                    assert cleanup is not None  # No exception raised
                    assert "Ignoring timeout" in caplog.text
                    assert mock_wrapper_cls.call_count == 1  # Still only one wrapper
    def test_cli_from_sdk_bundled_binary(self):
        """_find_copilot_cli should find the SDK's bundled binary first."""
        from amplifier_module_provider_github_copilot import _find_copilot_cli

        mock_copilot_mod = Mock()
        mock_copilot_mod.__file__ = "/fake/site-packages/copilot/__init__.py"

        with patch.dict("os.environ", {}, clear=True):
            with patch.dict("sys.modules", {"copilot": mock_copilot_mod}):
                with patch("pathlib.Path.exists", return_value=True):
                    with patch("amplifier_module_provider_github_copilot._ensure_executable"):
                        with patch("shutil.which", return_value=None):
                            result = _find_copilot_cli({})
                            assert result is not None
                            assert "copilot" in result
                            assert "bin" in result

    def test_cli_sdk_binary_preferred_over_path(self):
        """SDK bundled binary should be preferred over PATH binary."""
        from amplifier_module_provider_github_copilot import _find_copilot_cli

        mock_copilot_mod = Mock()
        mock_copilot_mod.__file__ = "/fake/site-packages/copilot/__init__.py"

        with patch.dict("os.environ", {}, clear=True):
            with patch.dict("sys.modules", {"copilot": mock_copilot_mod}):
                with patch("pathlib.Path.exists", return_value=True):
                    with patch("amplifier_module_provider_github_copilot._ensure_executable"):
                        with patch("shutil.which", return_value="/usr/bin/copilot"):
                            result = _find_copilot_cli({})
                            assert result is not None
                            assert "/fake/site-packages/copilot/bin/copilot" in result

    def test_cli_falls_back_to_path_when_sdk_missing(self):
        """When SDK bundled binary doesn't exist, fall back to PATH."""
        from amplifier_module_provider_github_copilot import _find_copilot_cli

        mock_copilot_mod = Mock()
        mock_copilot_mod.__file__ = "/fake/site-packages/copilot/__init__.py"

        with patch.dict("os.environ", {}, clear=True):
            with patch.dict("sys.modules", {"copilot": mock_copilot_mod}):
                with patch("pathlib.Path.exists", return_value=False):
                    with patch("amplifier_module_provider_github_copilot._ensure_executable"):
                        with patch("shutil.which", return_value="/usr/bin/copilot"):
                            result = _find_copilot_cli({})
                            assert result == "/usr/bin/copilot"
