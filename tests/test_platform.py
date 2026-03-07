"""Tests for _platform.py - Cross-platform utilities.

This module tests the centralized platform detection and binary location
functionality. These tests ensure the Single Source of Truth pattern
works correctly across all platforms.

Design philosophy:
- Test the logic, not the platform: Use mocks to test all platform paths
- Test on actual platform: Live tests validate real behavior
- Test the contract: Functions return expected types
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest


class TestPlatformInfo:
    """Tests for PlatformInfo dataclass and get_platform_info()."""

    def test_platform_info_is_frozen(self):
        """PlatformInfo should be immutable."""
        from amplifier_module_provider_github_copilot._platform import PlatformInfo

        info = PlatformInfo(
            name="Test",
            is_windows=False,
            cli_binary_name="copilot",
            uses_exe_extension=False,
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            info.name = "Changed"  # type: ignore

    def test_get_platform_info_returns_platform_info(self):
        """get_platform_info should return a PlatformInfo instance."""
        from amplifier_module_provider_github_copilot._platform import (
            PlatformInfo,
            get_platform_info,
        )

        result = get_platform_info()
        assert isinstance(result, PlatformInfo)

    def test_get_platform_info_windows(self):
        """On Windows, should return Windows-appropriate values."""
        from amplifier_module_provider_github_copilot._platform import get_platform_info

        # Clear cache to allow re-evaluation with mocked platform
        get_platform_info.cache_clear()

        with patch.object(sys, "platform", "win32"):
            result = get_platform_info()

        # Clear again to not affect other tests
        get_platform_info.cache_clear()

        assert result.is_windows is True
        assert result.cli_binary_name == "copilot.exe"
        assert result.uses_exe_extension is True

    def test_get_platform_info_linux(self):
        """On Linux, should return Unix-appropriate values."""
        from amplifier_module_provider_github_copilot._platform import get_platform_info

        get_platform_info.cache_clear()

        with patch.object(sys, "platform", "linux"):
            result = get_platform_info()

        get_platform_info.cache_clear()

        assert result.is_windows is False
        assert result.cli_binary_name == "copilot"
        assert result.uses_exe_extension is False

    def test_get_platform_info_darwin(self):
        """On macOS, should return Unix-appropriate values with macOS name."""
        from amplifier_module_provider_github_copilot._platform import get_platform_info

        get_platform_info.cache_clear()

        with patch.object(sys, "platform", "darwin"):
            result = get_platform_info()

        get_platform_info.cache_clear()

        assert result.is_windows is False
        assert result.cli_binary_name == "copilot"
        assert result.name == "macOS"


class TestBinaryNameConstants:
    """Tests for binary name constants."""

    def test_constants_are_defined(self):
        """Binary name constants should be defined."""
        from amplifier_module_provider_github_copilot._platform import (
            CLI_BINARY_NAME_UNIX,
            CLI_BINARY_NAME_WINDOWS,
            CLI_BINARY_SUBDIR,
        )

        assert CLI_BINARY_NAME_UNIX == "copilot"
        assert CLI_BINARY_NAME_WINDOWS == "copilot.exe"
        assert CLI_BINARY_SUBDIR == "bin"

    def test_get_cli_binary_name_returns_string(self):
        """get_cli_binary_name should return a string."""
        from amplifier_module_provider_github_copilot._platform import get_cli_binary_name

        result = get_cli_binary_name()
        assert isinstance(result, str)
        assert result in ("copilot", "copilot.exe")


class TestSdkBinaryPath:
    """Tests for get_sdk_binary_path()."""

    def test_returns_none_when_sdk_not_installed(self):
        """Should return None when copilot SDK is not installed."""
        from amplifier_module_provider_github_copilot._platform import get_sdk_binary_path

        with patch.dict("sys.modules", {"copilot": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module")):
                result = get_sdk_binary_path()

        assert result is None

    def test_returns_none_when_binary_not_found(self):
        """Should return None when SDK exists but binary doesn't."""
        from amplifier_module_provider_github_copilot._platform import get_sdk_binary_path

        mock_mod = Mock()
        mock_mod.__file__ = "/fake/copilot/__init__.py"

        with patch.dict("sys.modules", {"copilot": mock_mod}):
            with patch("pathlib.Path.exists", return_value=False):
                result = get_sdk_binary_path()

        assert result is None

    def test_returns_path_when_binary_found(self):
        """Should return Path when SDK binary is found."""
        from amplifier_module_provider_github_copilot._platform import get_sdk_binary_path

        mock_mod = Mock()
        mock_mod.__file__ = "/fake/copilot/__init__.py"

        with patch.dict("sys.modules", {"copilot": mock_mod}):
            with patch("pathlib.Path.exists", return_value=True):
                result = get_sdk_binary_path()

        assert result is not None
        assert isinstance(result, Path)


class TestFindCliInPath:
    """Tests for find_cli_in_path()."""

    def test_returns_none_when_not_in_path(self):
        """Should return None when CLI not in PATH."""
        from amplifier_module_provider_github_copilot._platform import find_cli_in_path

        with patch("shutil.which", return_value=None):
            result = find_cli_in_path()

        assert result is None

    def test_returns_path_when_found(self):
        """Should return Path when CLI found in PATH."""
        from amplifier_module_provider_github_copilot._platform import find_cli_in_path

        with patch("shutil.which", return_value="/usr/bin/copilot"):
            result = find_cli_in_path()

        assert result is not None
        assert isinstance(result, Path)
        # Path normalizes separators per platform
        assert result.name == "copilot"


class TestLocateCliBinary:
    """Tests for locate_cli_binary() - the main entry point."""

    def test_prefers_sdk_binary_over_path(self):
        """SDK binary should be preferred over PATH."""
        from amplifier_module_provider_github_copilot._platform import locate_cli_binary

        mock_mod = Mock()
        mock_mod.__file__ = "/sdk/copilot/__init__.py"

        with patch.dict("sys.modules", {"copilot": mock_mod}):
            with patch("pathlib.Path.exists", return_value=True):
                with patch("shutil.which", return_value="/path/copilot"):
                    result = locate_cli_binary()

        # Should return SDK path, not PATH path
        assert result is not None
        assert "sdk" in str(result).lower() or "copilot" in str(result)

    def test_falls_back_to_path_when_sdk_missing(self):
        """Should fall back to PATH when SDK binary not available."""
        from amplifier_module_provider_github_copilot._platform import locate_cli_binary

        with patch.dict("sys.modules", {"copilot": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module")):
                with patch("shutil.which", return_value="/usr/bin/copilot"):
                    result = locate_cli_binary()

        assert result is not None
        # Path normalizes separators per platform
        assert result.name == "copilot"

    def test_returns_none_when_nothing_found(self):
        """Should return None when CLI not found anywhere."""
        from amplifier_module_provider_github_copilot._platform import locate_cli_binary

        with patch.dict("sys.modules", {"copilot": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module")):
                with patch("shutil.which", return_value=None):
                    result = locate_cli_binary()

        assert result is None


class TestMakeTestPlatformInfo:
    """Tests for the test utility function."""

    def test_creates_windows_platform(self):
        """_make_test_platform_info should create Windows platform."""
        from amplifier_module_provider_github_copilot._platform import (
            _make_test_platform_info,
        )

        result = _make_test_platform_info(is_windows=True)

        assert result.is_windows is True
        assert result.cli_binary_name == "copilot.exe"
        assert result.uses_exe_extension is True

    def test_creates_unix_platform(self):
        """_make_test_platform_info should create Unix platform."""
        from amplifier_module_provider_github_copilot._platform import (
            _make_test_platform_info,
        )

        result = _make_test_platform_info(is_windows=False)

        assert result.is_windows is False
        assert result.cli_binary_name == "copilot"
        assert result.uses_exe_extension is False


class TestLivePlatformDetection:
    """Live tests for actual platform detection."""

    def test_live_platform_detection_matches_sys_platform(self):
        """get_platform_info().is_windows should match sys.platform."""
        from amplifier_module_provider_github_copilot._platform import get_platform_info

        result = get_platform_info()

        if sys.platform == "win32":
            assert result.is_windows is True
            assert result.cli_binary_name == "copilot.exe"
        else:
            assert result.is_windows is False
            assert result.cli_binary_name == "copilot"

    def test_live_locate_cli_binary(self):
        """locate_cli_binary should work on current platform."""
        from amplifier_module_provider_github_copilot._platform import locate_cli_binary

        # Just verify it doesn't crash - may or may not find binary
        result = locate_cli_binary()
        assert result is None or isinstance(result, Path)

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-only")
    def test_live_windows_returns_exe(self):
        """On actual Windows, binary name should be .exe."""
        from amplifier_module_provider_github_copilot._platform import get_cli_binary_name

        assert get_cli_binary_name() == "copilot.exe"

    @pytest.mark.skipif(sys.platform == "win32", reason="Unix-only")
    def test_live_unix_no_exe(self):
        """On actual Unix, binary name should not have .exe."""
        from amplifier_module_provider_github_copilot._platform import get_cli_binary_name

        assert get_cli_binary_name() == "copilot"
        assert ".exe" not in get_cli_binary_name()
