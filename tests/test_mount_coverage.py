"""
Additional coverage tests for __init__.py (mount and _find_copilot_cli).

Targets uncovered lines: 199-201, 241-242, 246-255, exception block.
These tests complement test_mount.py which covers the happy paths.
"""

import os
from unittest.mock import AsyncMock, patch

import pytest

from amplifier_module_provider_github_copilot import mount


class TestMountExceptionPath:
    """Tests for mount() exception handling (lines 199-201)."""

    @pytest.mark.asyncio
    async def test_mount_returns_none_when_provider_creation_raises(self, mock_coordinator):
        """mount() should return None when CopilotSdkProvider constructor raises."""
        with patch("shutil.which", return_value="/usr/bin/copilot"):
            with patch("os.path.isfile", return_value=True):
                with patch("os.path.isabs", return_value=True):
                    with patch("amplifier_module_provider_github_copilot._ensure_executable"):
                        with patch(
                            "amplifier_module_provider_github_copilot.CopilotSdkProvider",
                            side_effect=Exception("Init explosion"),
                        ):
                            cleanup = await mount(mock_coordinator, {"model": "claude-opus-4.5"})

                            assert cleanup is None
                            assert "github-copilot" not in mock_coordinator.mounted_providers

    @pytest.mark.asyncio
    async def test_mount_returns_none_when_coordinator_mount_raises(self, mock_coordinator):
        """mount() should return None when coordinator.mount() raises."""
        mock_coordinator.mount = AsyncMock(side_effect=RuntimeError("Mount failed"))

        with patch("shutil.which", return_value="/usr/bin/copilot"):
            with patch("os.path.isfile", return_value=True):
                with patch("os.path.isabs", return_value=True):
                    with patch("amplifier_module_provider_github_copilot._ensure_executable"):
                        cleanup = await mount(mock_coordinator, {})

                        assert cleanup is None


class TestFindCopilotCliEnvVar:
    """Tests for _find_copilot_cli with COPILOT_CLI_PATH env var (lines 241-242)."""

    @pytest.mark.asyncio
    async def test_env_var_cli_path(self, mock_coordinator):
        """Should discover CLI from COPILOT_CLI_PATH environment variable."""
        with patch.dict(os.environ, {"COPILOT_CLI_PATH": "/env/copilot"}):
            with patch("shutil.which", return_value=None):
                with patch("os.path.isfile", return_value=True):
                    with patch("os.path.isabs", return_value=True):
                        with patch("amplifier_module_provider_github_copilot._ensure_executable"):
                            cleanup = await mount(mock_coordinator, {})

                            assert cleanup is not None
                            provider = mock_coordinator.mounted_providers.get("github-copilot")
                            assert provider is not None

    @pytest.mark.asyncio
    async def test_config_cli_path_takes_priority_over_env(self, mock_coordinator):
        """Config cli_path should be used even when env var is set."""
        with patch.dict(os.environ, {"COPILOT_CLI_PATH": "/env/copilot"}):
            with patch("os.path.isfile", return_value=True):
                with patch("os.path.isabs", return_value=True):
                    with patch("amplifier_module_provider_github_copilot._ensure_executable"):
                        cleanup = await mount(
                            mock_coordinator,
                            {"cli_path": "/config/copilot"},
                        )

                        assert cleanup is not None


class TestFindCopilotCliAbsolutePath:
    """Tests for _find_copilot_cli absolute path validation (lines 246-248)."""

    @pytest.mark.asyncio
    async def test_absolute_path_not_exists_returns_none(self, mock_coordinator):
        """Should return None when absolute CLI path doesn't exist on disk."""
        with patch("os.path.isabs", return_value=True):
            with patch("os.path.isfile", return_value=False):
                cleanup = await mount(
                    mock_coordinator,
                    {"cli_path": "/nonexistent/copilot"},
                )

                assert cleanup is None

    @pytest.mark.asyncio
    async def test_absolute_path_exists_succeeds(self, mock_coordinator):
        """Should succeed when absolute path to CLI exists."""
        with patch("os.path.isabs", return_value=True):
            with patch("os.path.isfile", return_value=True):
                with patch("amplifier_module_provider_github_copilot._ensure_executable"):
                    cleanup = await mount(
                        mock_coordinator,
                        {"cli_path": "/usr/local/bin/copilot"},
                    )

                    assert cleanup is not None


class TestFindCopilotCliNonAbsolutePath:
    """Tests for _find_copilot_cli non-absolute path resolution (lines 251-255)."""

    @pytest.mark.asyncio
    async def test_non_absolute_path_not_in_path(self, mock_coordinator):
        """Should return None when non-absolute CLI name is not in PATH."""
        with patch("os.path.isabs", return_value=False):
            with patch("shutil.which", return_value=None):
                cleanup = await mount(
                    mock_coordinator,
                    {"cli_path": "copilot"},
                )

                assert cleanup is None

    @pytest.mark.asyncio
    async def test_non_absolute_path_found_in_path(self, mock_coordinator):
        """Should resolve non-absolute CLI name through PATH."""
        with patch("os.path.isabs", return_value=False):
            with patch("shutil.which", return_value="/usr/bin/copilot"):
                with patch("amplifier_module_provider_github_copilot._ensure_executable"):
                    cleanup = await mount(
                        mock_coordinator,
                        {"cli_path": "copilot"},
                    )

                    assert cleanup is not None


class TestFindCopilotCliExceptionPath:
    """Tests for _find_copilot_cli exception handling."""

    @pytest.mark.asyncio
    async def test_exception_during_discovery_returns_none(self, mock_coordinator):
        """Should return None when CLI discovery raises an exception."""
        with patch("shutil.which", side_effect=OSError("Permission denied")):
            cleanup = await mount(mock_coordinator, {})

            assert cleanup is None


class TestFindCopilotCliDirectImport:
    """Direct tests of _find_copilot_cli function for edge cases."""

    def test_empty_config_no_env_no_which(self):
        """Should return None with empty config, no env var, and nothing in PATH."""
        from amplifier_module_provider_github_copilot import _find_copilot_cli

        env = os.environ.copy()
        env.pop("COPILOT_CLI_PATH", None)

        with patch.dict(os.environ, env, clear=True):
            with patch("shutil.which", return_value=None):
                result = _find_copilot_cli({})
                assert result is None

    def test_which_finds_copilot_exe(self):
        """Should find copilot.exe when copilot is not found."""
        from amplifier_module_provider_github_copilot import _find_copilot_cli

        def which_side_effect(name):
            if name == "copilot":
                return None
            if name == "copilot.exe":
                return "C:\\Program Files\\copilot\\copilot.exe"
            return None

        env = os.environ.copy()
        env.pop("COPILOT_CLI_PATH", None)

        with patch.dict(os.environ, env, clear=True):
            with patch("shutil.which", side_effect=which_side_effect):
                with patch("os.path.isabs", return_value=True):
                    with patch("os.path.isfile", return_value=True):
                        with patch("amplifier_module_provider_github_copilot._ensure_executable"):
                            result = _find_copilot_cli({})
                            assert result == "C:\\Program Files\\copilot\\copilot.exe"
