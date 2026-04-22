"""
Cross-platform compatibility tests.

This test suite verifies that the provider works correctly on Windows, macOS, and Linux.
It catches platform-specific issues before they reach CI.

Contract: behaviors.md (cross-platform hygiene)

Note: The Windows event loop policy is set in conftest.py, not here.
"""

from __future__ import annotations


class TestPlatformModuleExists:
    """Platform module exists and is functional.

    Contract: sdk-boundary:BinaryResolution:MUST:1
    """

    def test_platform_info_has_required_fields(self) -> None:
        """PlatformInfo has name, is_windows, and cli_binary_name fields.

        Contract: sdk-boundary:BinaryResolution:MUST:1
        """
        from amplifier_module_provider_github_copilot._platform import get_platform_info

        info = get_platform_info()
        # L138 asserts exact value for info.name; hasattr guard unnecessary
        assert isinstance(info.is_windows, bool)
        # cli_binary_name asserted below based on is_windows value

        # Name should be one of the known platforms
        assert info.name in ("Windows", "macOS", "Unix")

        # cli_binary_name should match is_windows
        if info.is_windows:
            assert info.cli_binary_name == "copilot.exe"
        else:
            assert info.cli_binary_name == "copilot"


class TestWSLDetection:
    """PlatformInfo reports WSL vs native Linux distinctly.

    Contract: sdk-boundary:BinaryResolution:MUST:1 — platform detection must
    accurately detect the running environment. WSL has different binary
    resolution characteristics than native Linux.
    """

    def test_wsl_detected_from_proc_version(self) -> None:
        """is_wsl=True when /proc/version contains 'microsoft'.

        Contract: sdk-boundary:BinaryResolution:MUST:1
        """
        from unittest.mock import mock_open, patch

        from amplifier_module_provider_github_copilot._platform import (
            PlatformInfo,
            get_platform_info,
        )

        get_platform_info.cache_clear()

        wsl_proc_version = "Linux version 5.15.0-microsoft-standard-WSL2"

        with (
            patch("sys.platform", "linux"),
            patch("builtins.open", mock_open(read_data=wsl_proc_version)),
        ):
            info = get_platform_info()

        get_platform_info.cache_clear()

        assert info.is_wsl is True, "Should detect WSL from /proc/version"
        assert isinstance(info, PlatformInfo)

    def test_non_wsl_linux_has_is_wsl_false(self) -> None:
        """is_wsl=False on native Linux (no 'microsoft' in /proc/version).

        Contract: sdk-boundary:BinaryResolution:MUST:1
        """
        from unittest.mock import mock_open, patch

        from amplifier_module_provider_github_copilot._platform import get_platform_info

        get_platform_info.cache_clear()

        native_proc_version = "Linux version 5.15.0-91-generic (ubuntu)"

        with (
            patch("sys.platform", "linux"),
            patch("builtins.open", mock_open(read_data=native_proc_version)),
        ):
            info = get_platform_info()

        get_platform_info.cache_clear()

        assert info.is_wsl is False
        # SCHNEIER: verify Linux branch sets all required fields correctly
        assert info.is_windows is False
        assert info.name == "Unix"
        assert info.cli_binary_name == "copilot"
