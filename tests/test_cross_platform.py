"""
Cross-platform compatibility tests.

This test suite verifies that the provider works correctly on Windows, macOS, and Linux.
It catches platform-specific issues before they reach CI.

Contract: behaviors.md (cross-platform hygiene)

Note: The Windows event loop policy is set in conftest.py, not here.
"""

from __future__ import annotations


class TestConfigLoadingUsesImportlib:
    """AC-5 related: Config loaded via importlib, not file paths.

    Contract: behaviors:ConfigLoading:MUST:1
    """

    def test_config_loading_uses_importlib_resources(self) -> None:
        """Config loaded via importlib, not file paths.

        Contract: behaviors:ConfigLoading:MUST:1

        Using importlib.resources ensures configs work in installed wheels
        and on all platforms.
        """
        # The load functions should work regardless of current directory
        from amplifier_module_provider_github_copilot.config_loader import (
            load_models_config,
        )
        from amplifier_module_provider_github_copilot.error_translation import (
            load_error_config,
        )
        from amplifier_module_provider_github_copilot.streaming import load_event_config

        # These should not raise regardless of cwd
        models_config = load_models_config()
        error_config = load_error_config()
        event_config = load_event_config()

        # Contract: behaviors:ConfigLoading:MUST:1
        assert models_config.provider_id == "github-copilot"
        # Verify error config has RateLimitError mapping (proves real data loaded)
        rate_mappings = [
            m for m in error_config.mappings if "RateLimitError" in (m.sdk_patterns or [])
        ]
        assert len(rate_mappings) >= 1, "Error config must include RateLimitError mapping"
        # Verify event config has core content_delta mapping
        assert "assistant.message_delta" in event_config.bridge_mappings


class TestErrorConfigPlatformIndependent:
    """AC-6: ErrorConfig loads on any platform.

    Contract: behaviors:ConfigLoading:MUST:1
    """

    def test_error_config_loading_platform_independent(self) -> None:
        """ErrorConfig loads on any platform.

        Contract: behaviors:ConfigLoading:MUST:1

        Error config loading must work on Windows, macOS, and Linux.
        """
        from amplifier_module_provider_github_copilot.error_translation import (
            load_error_config,
        )

        config = load_error_config()

        # Contract: behaviors:ConfigLoading:MUST:1
        rate_limit_mappings = [
            m for m in config.mappings if "RateLimitError" in (m.sdk_patterns or [])
        ]
        assert len(rate_limit_mappings) == 1, (
            "Error config must have exactly one RateLimitError mapping"
        )
        assert rate_limit_mappings[0].extract_retry_after is True


class TestEventConfigPlatformIndependent:
    """AC-7: EventConfig loads on any platform.

    Contract: behaviors:ConfigLoading:MUST:1
    """

    def test_event_config_loading_platform_independent(self) -> None:
        """EventConfig loads on any platform.

        Contract: behaviors:ConfigLoading:MUST:1

        Event config loading must work on Windows, macOS, and Linux.
        """
        from amplifier_module_provider_github_copilot.streaming import load_event_config

        config = load_event_config()

        # Contract: behaviors:ConfigLoading:MUST:1
        assert "assistant.message_delta" in config.bridge_mappings, (
            "Event config must include assistant.message_delta bridge mapping"
        )


class TestPlatformModuleExists:
    """Platform module exists and is functional.

    Contract: sdk-boundary:BinaryResolution:MUST:1
    """

    def test_platform_module_exists(self) -> None:
        """_platform.py module exists.

        Contract: sdk-boundary:BinaryResolution:MUST:1
        """
        from amplifier_module_provider_github_copilot._platform import (
            PlatformInfo,
            get_platform_info,
        )

        info = get_platform_info()
        assert isinstance(info, PlatformInfo)

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
