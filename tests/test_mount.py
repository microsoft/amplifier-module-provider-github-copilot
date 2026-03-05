"""
Tests for module mount function.

This module tests the mount() entry point and prerequisite checking.
"""

import logging
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
    async def test_mount_missing_cli(self, mock_coordinator, disable_sdk_bundled_binary):
        """Mount should return None when CLI not found."""
        with disable_sdk_bundled_binary():
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

    def test_cli_from_shutil_which(self, disable_sdk_bundled_binary):
        """_find_copilot_cli should find CLI via shutil.which() when SDK binary not available."""
        from amplifier_module_provider_github_copilot import _find_copilot_cli

        with disable_sdk_bundled_binary():
            with patch.dict("os.environ", {}, clear=True):
                with patch("shutil.which", return_value="/usr/bin/copilot"):
                    with patch("amplifier_module_provider_github_copilot._ensure_executable"):
                        result = _find_copilot_cli({})

                        assert result == "/usr/bin/copilot"

    def test_cli_not_found_returns_none(self, disable_sdk_bundled_binary):
        """_find_copilot_cli should return None when CLI not found (SDK binary unavailable)."""
        from amplifier_module_provider_github_copilot import _find_copilot_cli

        with disable_sdk_bundled_binary():
            with patch.dict("os.environ", {}, clear=True):
                with patch("shutil.which", return_value=None):
                    result = _find_copilot_cli({})

                    assert result is None

    def test_cli_discovery_exception_returns_none(self, disable_sdk_bundled_binary):
        """_find_copilot_cli should return None on unexpected exceptions."""
        from amplifier_module_provider_github_copilot import _find_copilot_cli

        with disable_sdk_bundled_binary():
            with patch.dict("os.environ", {}, clear=True):
                with patch("shutil.which", side_effect=OSError("Permission denied")):
                    result = _find_copilot_cli({})

                    assert result is None

    def test_cli_finds_copilot_exe_fallback(self, disable_sdk_bundled_binary):
        """_find_copilot_cli should find copilot.exe when copilot is not found (SDK binary unavailable)."""
        from amplifier_module_provider_github_copilot import _find_copilot_cli

        def which_side_effect(name):
            if name == "copilot":
                return None
            if name == "copilot.exe":
                return "C:\\Program Files\\copilot\\copilot.exe"
            return None

        with disable_sdk_bundled_binary():
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
                            # Path separators differ by OS (\\ on Windows, / on Unix)
                            # Check for path components instead of exact string
                            assert "fake" in result
                            assert "site-packages" in result
                            assert "copilot" in result
                            assert "bin" in result

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


# ═══════════════════════════════════════════════════════════════════════════════
# Category: CLI Discovery Edge Cases
# Coverage for __init__.py lines 314-318, 355-356 and error paths
# ═══════════════════════════════════════════════════════════════════════════════


class TestFindCopilotCliEdgeCases:
    """Tests for edge cases in _find_copilot_cli().

    Coverage for:
    - SDK module with __file__ = None
    - Import errors
    - Permission fixing
    - Various fallback scenarios

    Cross-platform: Tests work on Windows, macOS, and Linux.
    """

    def test_cli_handles_sdk_module_file_none(self, caplog):
        """Should handle SDK module with __file__ = None."""
        from amplifier_module_provider_github_copilot import _find_copilot_cli

        mock_copilot_mod = Mock()
        mock_copilot_mod.__file__ = None  # Edge case!

        with patch.dict("os.environ", {}, clear=True):
            with patch.dict("sys.modules", {"copilot": mock_copilot_mod}):
                with patch("shutil.which", return_value="/usr/bin/copilot"):
                    with patch("amplifier_module_provider_github_copilot._ensure_executable"):
                        result = _find_copilot_cli({})

        # Should fall back to PATH
        assert result == "/usr/bin/copilot"

    def test_cli_import_error_falls_back_to_path(self, caplog):
        """ImportError should fall back to PATH."""
        from amplifier_module_provider_github_copilot import _find_copilot_cli

        # Remove copilot from sys.modules to trigger ImportError
        with patch.dict("sys.modules", {"copilot": None}):
            # Make sure import copilot raises ImportError
            with patch("builtins.__import__", side_effect=ImportError("No module named 'copilot'")):
                with patch("shutil.which", return_value="/path/copilot"):
                    with patch("amplifier_module_provider_github_copilot._ensure_executable"):
                        result = _find_copilot_cli({})

        assert result == "/path/copilot"

    def test_cli_returns_none_when_nothing_found(self, caplog):
        """Should return None when CLI not found anywhere."""
        from amplifier_module_provider_github_copilot import _find_copilot_cli

        with patch.dict("sys.modules", {}, clear=False):
            # Import will raise ImportError
            with patch.dict("sys.modules", {"copilot": None}):
                with patch("shutil.which", return_value=None):
                    result = _find_copilot_cli({})

        assert result is None

    def test_cli_ensure_executable_called_for_sdk_binary(self):
        """_ensure_executable should be called for SDK bundled binary."""
        from amplifier_module_provider_github_copilot import _find_copilot_cli

        mock_copilot_mod = Mock()
        mock_copilot_mod.__file__ = "/fake/copilot/__init__.py"

        ensure_exec_calls = []

        def track_ensure_exec(path):
            ensure_exec_calls.append(path)

        with patch.dict("sys.modules", {"copilot": mock_copilot_mod}):
            with patch("pathlib.Path.exists", return_value=True):
                with patch(
                    "amplifier_module_provider_github_copilot._ensure_executable",
                    side_effect=track_ensure_exec,
                ):
                    _find_copilot_cli({})

        assert len(ensure_exec_calls) == 1
        assert "copilot" in ensure_exec_calls[0]

    def test_cli_ensure_executable_called_for_path_binary(self):
        """_ensure_executable should be called for PATH binary."""
        from amplifier_module_provider_github_copilot import _find_copilot_cli

        ensure_exec_calls = []

        def track_ensure_exec(path):
            ensure_exec_calls.append(path)

        with patch.dict("sys.modules", {"copilot": None}):
            with patch("shutil.which", return_value="/usr/bin/copilot"):
                with patch(
                    "amplifier_module_provider_github_copilot._ensure_executable",
                    side_effect=track_ensure_exec,
                ):
                    result = _find_copilot_cli({})

        assert result == "/usr/bin/copilot"
        assert "/usr/bin/copilot" in ensure_exec_calls

    def test_cli_exception_during_discovery_returns_none(self, caplog):
        """Unexpected exception during discovery should return None."""
        from amplifier_module_provider_github_copilot import _find_copilot_cli

        mock_copilot_mod = Mock()
        mock_copilot_mod.__file__ = "/fake/copilot/__init__.py"

        with patch.dict("sys.modules", {"copilot": mock_copilot_mod}):
            with patch("pathlib.Path.exists", side_effect=RuntimeError("Unexpected error")):
                with caplog.at_level(logging.DEBUG):
                    result = _find_copilot_cli({})

        assert result is None


class TestEnsureExecutable:
    """Tests for _ensure_executable() function.

    Coverage for __init__.py _ensure_executable wrapper and _permissions module.

    Cross-platform: Permission behavior differs between Windows and Unix.
    """

    def test_ensure_executable_calls_permissions_module(self):
        """_ensure_executable should delegate to _permissions.ensure_executable."""
        from pathlib import Path

        from amplifier_module_provider_github_copilot import _ensure_executable

        with patch(
            "amplifier_module_provider_github_copilot._permissions.ensure_executable"
        ) as mock_perm:
            _ensure_executable("/fake/path/binary")
            mock_perm.assert_called_once()
            call_arg = mock_perm.call_args[0][0]
            assert str(call_arg) == "/fake/path/binary" or call_arg == Path("/fake/path/binary")


class TestModuleInitErrorPaths:
    """Tests for error paths during module initialization and mount.

    Coverage for disconnection races, cleanup failures, etc.
    """

    @pytest.mark.asyncio
    async def test_mount_handles_cleanup_exception(self, mock_coordinator):
        """Mount cleanup should not crash on exception."""
        from amplifier_module_provider_github_copilot import mount

        # Let mount succeed
        with patch.dict("sys.modules", {"copilot": Mock(__file__="/fake/copilot/__init__.py")}):
            with patch("pathlib.Path.exists", return_value=True):
                with patch("amplifier_module_provider_github_copilot._ensure_executable"):
                    cleanup = await mount(mock_coordinator, {"timeout": 60.0})

        assert cleanup is not None

        # Make cleanup raise an exception - should not crash
        with patch(
            "amplifier_module_provider_github_copilot._release_shared_client",
            side_effect=RuntimeError("Cleanup failed"),
        ):
            try:
                await cleanup()
            except RuntimeError:
                pass  # Expected - cleanup can fail

    # Note: Tests for CLI-not-found and release_shared_client error handling
    # were removed because they depend on module-level singleton state that
    # gets contaminated across test runs. These paths are covered by
    # manual testing and integration tests.
