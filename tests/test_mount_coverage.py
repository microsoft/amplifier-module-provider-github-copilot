"""
Additional coverage tests for __init__.py (mount and _find_copilot_cli).

Targets uncovered lines: 199-201, exception block.
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


class TestFindCopilotCliExceptionPath:
    """Tests for _find_copilot_cli exception handling."""

    @pytest.mark.asyncio
    async def test_exception_during_discovery_returns_none(
        self, mock_coordinator, disable_sdk_bundled_binary
    ):
        """Should return None when CLI discovery raises an exception."""
        with disable_sdk_bundled_binary():
            with patch("shutil.which", side_effect=OSError("Permission denied")):
                cleanup = await mount(mock_coordinator, {})

                assert cleanup is None


class TestFindCopilotCliDirectImport:
    """Direct tests of _find_copilot_cli function for edge cases."""

    def test_empty_config_no_which(self, disable_sdk_bundled_binary):
        """Should return None with empty config and nothing in PATH (SDK binary unavailable)."""
        from amplifier_module_provider_github_copilot import _find_copilot_cli

        with disable_sdk_bundled_binary():
            with patch.dict(os.environ, {}, clear=True):
                with patch("shutil.which", return_value=None):
                    result = _find_copilot_cli({})
                    assert result is None

    def test_which_finds_copilot_exe(self, disable_sdk_bundled_binary):
        """Should find copilot.exe when copilot is not found (SDK binary unavailable)."""
        from amplifier_module_provider_github_copilot import _find_copilot_cli

        def which_side_effect(name):
            if name == "copilot":
                return None
            if name == "copilot.exe":
                return "C:\\Program Files\\copilot\\copilot.exe"
            return None

        with disable_sdk_bundled_binary():
            with patch.dict(os.environ, {}, clear=True):
                with patch("shutil.which", side_effect=which_side_effect):
                    with patch("amplifier_module_provider_github_copilot._ensure_executable"):
                        result = _find_copilot_cli({})
                        assert result == "C:\\Program Files\\copilot\\copilot.exe"
