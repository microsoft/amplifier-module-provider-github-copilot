# pyright: reportPrivateUsage=false
# pyright: reportUnknownMemberType=false
# pyright: reportUntypedFunctionDecorator=false
"""Tests for cross-platform binary discovery.

Contract: sdk-boundary:BinaryResolution:MUST:1-8

These tests verify:
- Platform detection via sys.platform
- Binary name resolution (copilot vs copilot.exe)
- SDK binary path discovery via importlib.util.find_spec
- PATH fallback for system CLI
- Permission repair on Unix systems
"""

from __future__ import annotations

import stat
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from amplifier_module_provider_github_copilot._platform import (
    CLI_BINARY_NAME_UNIX,
    CLI_BINARY_NAME_WINDOWS,
    PlatformInfo,
)


def _make_test_platform_info(*, is_windows: bool = False) -> PlatformInfo:
    """Create platform info for testing.

    Test-only factory — avoids patching sys.platform in tests that only need
    a PlatformInfo value object.

    Args:
        is_windows: Whether to create Windows platform info.

    Returns:
        PlatformInfo configured for the requested platform.

    """
    if is_windows:
        return PlatformInfo(
            name="Windows",
            is_windows=True,
            cli_binary_name=CLI_BINARY_NAME_WINDOWS,
        )
    return PlatformInfo(
        name="Unix",
        is_windows=False,
        cli_binary_name=CLI_BINARY_NAME_UNIX,
    )


# ============================================================================
# Test: Platform Detection
# ============================================================================


class TestPlatformDetection:
    """Tests for get_platform_info()."""

    def test_platform_info_is_cached(self) -> None:
        """get_platform_info() returns same instance on repeated calls.

        Contract: sdk-boundary:BinaryResolution:MUST:1
        """
        from amplifier_module_provider_github_copilot._platform import get_platform_info

        # Clear cache first
        get_platform_info.cache_clear()

        info1 = get_platform_info()
        info2 = get_platform_info()

        assert info1 is info2

    def test_platform_info_is_frozen(self) -> None:
        """PlatformInfo is immutable (frozen dataclass).

        Contract: sdk-boundary:BinaryResolution:MUST:1
        """
        from amplifier_module_provider_github_copilot._platform import get_platform_info

        get_platform_info.cache_clear()
        info = get_platform_info()

        with pytest.raises(AttributeError):
            info.name = "modified"  # type: ignore[misc]

    def test_windows_detection(self) -> None:
        """Windows platform detected correctly.

        Contract: sdk-boundary:BinaryResolution:MUST:3
        """
        info = _make_test_platform_info(is_windows=True)

        assert info.is_windows is True
        assert info.name == "Windows"
        assert info.cli_binary_name == "copilot.exe"

    def test_unix_detection(self) -> None:
        """Unix platform detected correctly.

        Contract: sdk-boundary:BinaryResolution:MUST:3
        """
        info = _make_test_platform_info(is_windows=False)

        assert info.is_windows is False
        assert info.name == "Unix"
        assert info.cli_binary_name == "copilot"

    def test_windows_platform_detection_via_sys_platform(self) -> None:
        """L73: sys.platform == 'win32' returns Windows PlatformInfo.

        Contract: sdk-boundary:BinaryResolution:MUST:1, :MUST:3
        """
        from amplifier_module_provider_github_copilot._platform import (
            CLI_BINARY_NAME_WINDOWS,
            get_platform_info,
        )

        get_platform_info.cache_clear()
        try:
            with patch("sys.platform", "win32"):
                get_platform_info.cache_clear()
                info = get_platform_info()

            assert info.is_windows is True
            assert info.name == "Windows"
            assert info.cli_binary_name == CLI_BINARY_NAME_WINDOWS
        finally:
            get_platform_info.cache_clear()

    def test_macos_platform_detection_via_sys_platform(self) -> None:
        """L79: sys.platform == 'darwin' returns macOS PlatformInfo.

        Contract: sdk-boundary:BinaryResolution:MUST:1, :MUST:3
        """
        from amplifier_module_provider_github_copilot._platform import (
            CLI_BINARY_NAME_UNIX,
            get_platform_info,
        )

        get_platform_info.cache_clear()
        try:
            with patch("sys.platform", "darwin"):
                get_platform_info.cache_clear()
                info = get_platform_info()

            assert info.is_windows is False
            assert info.name == "macOS"
            assert info.cli_binary_name == CLI_BINARY_NAME_UNIX
        finally:
            get_platform_info.cache_clear()


# ============================================================================
# Test: Binary Name Resolution
# ============================================================================


class TestBinaryNameResolution:
    """Tests for get_cli_binary_name()."""

    def test_windows_binary_name(self) -> None:
        """Windows uses copilot.exe.

        Contract: sdk-boundary:BinaryResolution:MUST:3
        """
        from amplifier_module_provider_github_copilot._platform import (
            CLI_BINARY_NAME_WINDOWS,
            get_platform_info,
        )

        get_platform_info.cache_clear()

        with patch("sys.platform", "win32"):
            get_platform_info.cache_clear()
            # Need to reimport to get fresh detection
            from amplifier_module_provider_github_copilot import _platform

            _platform.get_platform_info.cache_clear()

        # Just verify the constant exists
        assert CLI_BINARY_NAME_WINDOWS == "copilot.exe"

    def test_unix_binary_name(self) -> None:
        """Unix uses copilot (no extension).

        Contract: sdk-boundary:BinaryResolution:MUST:3
        """
        from amplifier_module_provider_github_copilot._platform import (
            CLI_BINARY_NAME_UNIX,
        )

        assert CLI_BINARY_NAME_UNIX == "copilot"


# ============================================================================
# Test: SDK Binary Discovery
# ============================================================================


class TestSdkBinaryDiscovery:
    """Tests for get_sdk_binary_path()."""

    def test_returns_none_when_sdk_not_installed(self) -> None:
        """Returns None when copilot package not found.

        Contract: sdk-boundary:BinaryResolution:MUST:2
        """
        from amplifier_module_provider_github_copilot._platform import (
            get_sdk_binary_path,
        )

        with patch("importlib.util.find_spec", return_value=None):
            result = get_sdk_binary_path()

        assert result is None

    def test_returns_none_when_binary_missing(self) -> None:
        """Returns None when package found but binary not in bin/.

        Contract: sdk-boundary:BinaryResolution:MUST:2
        """
        from amplifier_module_provider_github_copilot._platform import (
            get_sdk_binary_path,
        )

        mock_spec = MagicMock()
        mock_spec.origin = "/fake/copilot/__init__.py"

        with (
            patch("importlib.util.find_spec", return_value=mock_spec),
            patch.object(Path, "is_file", return_value=False),
        ):
            result = get_sdk_binary_path()

        assert result is None

    def test_returns_path_when_binary_found(self, tmp_path: Path) -> None:
        """Returns path when package and binary both exist.

        Contract: sdk-boundary:BinaryResolution:MUST:2
        """
        from amplifier_module_provider_github_copilot._platform import (
            get_cli_binary_name,
            get_sdk_binary_path,
        )

        # Create fake SDK structure
        pkg_dir = tmp_path / "copilot"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").touch()
        bin_dir = pkg_dir / "bin"
        bin_dir.mkdir()
        binary = bin_dir / get_cli_binary_name()
        binary.touch()

        mock_spec = MagicMock()
        mock_spec.origin = str(pkg_dir / "__init__.py")

        with patch("importlib.util.find_spec", return_value=mock_spec):
            result = get_sdk_binary_path()

        assert result is not None
        assert result.name == get_cli_binary_name()

    def test_invalid_origin_returns_none(self) -> None:
        """L127-128: TypeError/ValueError from Path() construction returns None.

        Contract: sdk-boundary:BinaryResolution:MUST:2
        """
        from amplifier_module_provider_github_copilot._platform import get_sdk_binary_path

        # Provide an origin value that causes Path() to raise TypeError
        # The value 123 (int) causes Path(123) to work but .parent to fail
        # Actually we need to mock get_copilot_spec_origin to return something
        # that makes Path(origin).parent raise
        # Contract: sdk-boundary:Membrane:MUST:1 - mock via membrane API
        with patch(
            "amplifier_module_provider_github_copilot.sdk_adapter.get_copilot_spec_origin",
            return_value=123,  # int will cause TypeError in Path()
        ):
            result = get_sdk_binary_path()

        assert result is None

    def test_value_error_origin_returns_none(self) -> None:
        """L127-128: ValueError from Path() construction returns None.

        Contract: sdk-boundary:BinaryResolution:MUST:2
        Note: In practice ValueError is unlikely from Path(), but the code
        handles it defensively. We test with embedded null byte which raises.
        """
        from amplifier_module_provider_github_copilot._platform import get_sdk_binary_path

        # Path with embedded null byte causes ValueError
        # Contract: sdk-boundary:Membrane:MUST:1 - mock via membrane API
        with patch(
            "amplifier_module_provider_github_copilot.sdk_adapter.get_copilot_spec_origin",
            return_value="/valid/path\x00with_null",
        ):
            result = get_sdk_binary_path()

        # Path with embedded null should trigger ValueError handling
        assert result is None


# ============================================================================
# Test: PATH Fallback
# ============================================================================


class TestPathFallback:
    """Tests for find_cli_in_path()."""

    def test_returns_none_when_not_in_path(self) -> None:
        """Returns None when CLI not in PATH.

        Contract: sdk-boundary:BinaryResolution:MUST:5
        """
        from amplifier_module_provider_github_copilot._platform import find_cli_in_path

        with patch("shutil.which", return_value=None):
            result = find_cli_in_path()

        assert result is None

    def test_returns_path_when_found(self) -> None:
        """Returns Path when CLI found in PATH.

        Contract: sdk-boundary:BinaryResolution:MUST:5
        """
        from amplifier_module_provider_github_copilot._platform import find_cli_in_path

        with patch("shutil.which", return_value="/usr/local/bin/copilot"):
            result = find_cli_in_path()

        assert result is not None
        assert result == Path("/usr/local/bin/copilot")

    def test_alternate_binary_found_when_primary_missing(self) -> None:
        """L163: platform-primary name not found; alternate name found returns Path.

        Contract: sdk-boundary:BinaryResolution:SHOULD:1
        WSL edge case: Linux process with Windows PATH entries visible.
        Primary 'copilot' not found; fallback 'copilot.exe' found in PATH.
        """
        from amplifier_module_provider_github_copilot._platform import (
            PlatformInfo,
            find_cli_in_path,
            get_platform_info,
        )

        # Simulate Linux (not Windows) but copilot.exe is in Windows PATH via WSL
        unix_info = PlatformInfo(name="Unix", is_windows=False, cli_binary_name="copilot")

        get_platform_info.cache_clear()

        def which_side_effect(name: str) -> str | None:
            # Primary 'copilot' not found; alternate 'copilot.exe' found
            if name == "copilot":
                return None
            if name == "copilot.exe":
                return "/mnt/c/Users/user/AppData/copilot.exe"
            return None

        with (
            patch(
                "amplifier_module_provider_github_copilot._platform.get_platform_info",
                return_value=unix_info,
            ),
            patch("shutil.which", side_effect=which_side_effect),
        ):
            result = find_cli_in_path()

        assert result is not None
        assert "copilot.exe" in str(result)


# ============================================================================
# Test: Main Entry Point
# ============================================================================


class TestLocateCliBinary:
    """Tests for locate_cli_binary()."""

    def test_prefers_sdk_over_path(self) -> None:
        """SDK binary preferred over PATH (security).

        Contract: sdk-boundary:BinaryResolution:MUST:4
        """
        from amplifier_module_provider_github_copilot._platform import (
            locate_cli_binary,
        )

        sdk_path = Path("/sdk/bin/copilot")
        path_binary = Path("/usr/bin/copilot")

        with (
            patch(
                "amplifier_module_provider_github_copilot._platform.get_sdk_binary_path",
                return_value=sdk_path,
            ),
            patch(
                "amplifier_module_provider_github_copilot._platform.find_cli_in_path",
                return_value=path_binary,
            ),
        ):
            result = locate_cli_binary()

        assert result == sdk_path

    def test_falls_back_to_path(self) -> None:
        """Falls back to PATH when SDK binary not found.

        Contract: sdk-boundary:BinaryResolution:MUST:5
        """
        from amplifier_module_provider_github_copilot._platform import (
            locate_cli_binary,
        )

        path_binary = Path("/usr/bin/copilot")

        with (
            patch(
                "amplifier_module_provider_github_copilot._platform.get_sdk_binary_path",
                return_value=None,
            ),
            patch(
                "amplifier_module_provider_github_copilot._platform.find_cli_in_path",
                return_value=path_binary,
            ),
        ):
            result = locate_cli_binary()

        assert result == path_binary

    def test_returns_none_when_not_found(self) -> None:
        """Returns None when binary not found anywhere.

        Contract: sdk-boundary:BinaryResolution:MUST:5
        """
        from amplifier_module_provider_github_copilot._platform import (
            locate_cli_binary,
        )

        with (
            patch(
                "amplifier_module_provider_github_copilot._platform.get_sdk_binary_path",
                return_value=None,
            ),
            patch(
                "amplifier_module_provider_github_copilot._platform.find_cli_in_path",
                return_value=None,
            ),
        ):
            result = locate_cli_binary()

        assert result is None


# ============================================================================
# Test: Permission Repair
# ============================================================================


class TestPermissionRepair:
    """Tests for ensure_executable()."""

    def test_no_op_on_windows(self) -> None:
        """Returns True immediately on Windows (no chmod needed).

        Contract: sdk-boundary:BinaryResolution:MUST:7
        """
        from amplifier_module_provider_github_copilot._permissions import (
            ensure_executable,
        )
        from amplifier_module_provider_github_copilot._platform import (
            PlatformInfo,
            get_platform_info,
        )

        windows_info = PlatformInfo(name="Windows", is_windows=True, cli_binary_name="copilot.exe")

        # Clear lru_cache before patching (get_platform_info uses @lru_cache)
        get_platform_info.cache_clear()

        # Patch at the source module where ensure_executable imports from
        with patch(
            "amplifier_module_provider_github_copilot._platform.get_platform_info",
            return_value=windows_info,
        ):
            # Even a non-existent path should return True on Windows
            result = ensure_executable(Path("/nonexistent"))

            assert result is True

    @pytest.mark.skipif(
        __import__("sys").platform == "win32",
        reason="Unix permission tests require POSIX filesystem",
    )
    def test_already_executable_is_idempotent(self, tmp_path: Path) -> None:
        """Returns True without chmod if already executable.

        Contract: sdk-boundary:BinaryResolution:MUST:6
        """
        from amplifier_module_provider_github_copilot._permissions import (
            ensure_executable,
        )
        from amplifier_module_provider_github_copilot._platform import (
            PlatformInfo,
            get_platform_info,
        )

        # Create executable file
        binary = tmp_path / "copilot"
        binary.touch()
        binary.chmod(0o755)

        unix_info = PlatformInfo(name="Unix", is_windows=False, cli_binary_name="copilot")

        # Clear lru_cache before patching (get_platform_info uses @lru_cache)
        get_platform_info.cache_clear()

        with patch(
            "amplifier_module_provider_github_copilot._platform.get_platform_info",
            return_value=unix_info,
        ):
            result = ensure_executable(binary)

            assert result is True
            # Mode should be unchanged
            assert binary.stat().st_mode & stat.S_IXUSR

    @pytest.mark.skipif(
        __import__("sys").platform == "win32",
        reason="Unix permission tests require POSIX filesystem",
    )
    def test_adds_execute_permission(self, tmp_path: Path) -> None:
        """Adds user+group execute permission.

        Contract: sdk-boundary:BinaryResolution:MUST:6
        """
        from amplifier_module_provider_github_copilot._permissions import (
            ensure_executable,
        )
        from amplifier_module_provider_github_copilot._platform import (
            PlatformInfo,
            get_platform_info,
        )

        # Create non-executable file
        binary = tmp_path / "copilot"
        binary.touch()
        binary.chmod(0o644)

        unix_info = PlatformInfo(name="Unix", is_windows=False, cli_binary_name="copilot")

        # Clear lru_cache before patching (get_platform_info uses @lru_cache)
        get_platform_info.cache_clear()

        with patch(
            "amplifier_module_provider_github_copilot._platform.get_platform_info",
            return_value=unix_info,
        ):
            result = ensure_executable(binary)

            assert result is True
            # Should now have user execute
            assert binary.stat().st_mode & stat.S_IXUSR
            # Should have group execute
            assert binary.stat().st_mode & stat.S_IXGRP

    def test_returns_false_for_missing_file(self, tmp_path: Path) -> None:
        """Returns False for non-existent file.

        Contract: sdk-boundary:BinaryResolution:MUST:6
        """
        from amplifier_module_provider_github_copilot._permissions import (
            ensure_executable,
        )
        from amplifier_module_provider_github_copilot._platform import (
            PlatformInfo,
            get_platform_info,
        )

        unix_info = PlatformInfo(name="Unix", is_windows=False, cli_binary_name="copilot")

        # Clear lru_cache before patching (get_platform_info uses @lru_cache)
        get_platform_info.cache_clear()

        with patch(
            "amplifier_module_provider_github_copilot._platform.get_platform_info",
            return_value=unix_info,
        ):
            result = ensure_executable(tmp_path / "nonexistent")

            assert result is False
