"""
Cross-platform compatibility tests.

This test suite verifies that the provider works correctly on Windows, macOS, and Linux.
It catches platform-specific issues before they reach CI.

Contract: behaviors.md (cross-platform hygiene)

Note: The Windows event loop policy is set in conftest.py, not here.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from tests.fixtures.config_capture import ConfigCapturingMock

if TYPE_CHECKING:
    pass


class TestNoHardcodedPathSeparators:
    """AC-2: No hardcoded path separators in production code.

    Cross-platform hygiene test.
    """

    def test_no_hardcoded_path_separators(self) -> None:
        """No '/' or '\\' in path construction.

        Cross-platform hygiene: Use pathlib.Path for all path operations.
        Hardcoded separators break on different platforms.
        """
        src_root = Path("amplifier_module_provider_github_copilot")

        # Patterns that indicate hardcoded path construction
        # (not string literals that happen to contain slashes)
        bad_patterns = [
            # Path construction with hardcoded separators
            "os.path.join",  # Should use pathlib
            '+ "/"',  # String concatenation with slash
            "+ '/'",  # Single quote variant
            '+ "\\\\"',  # Windows backslash
            "+ '\\\\'",  # Single quote variant
        ]

        violations: list[tuple[str, int, str]] = []
        files_scanned = 0

        for py_file in src_root.rglob("*.py"):
            files_scanned += 1
            source = py_file.read_text(encoding="utf-8")
            for i, line in enumerate(source.splitlines(), 1):
                # Skip comments and docstrings indicators
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue

                for pattern in bad_patterns:
                    if pattern in line:
                        # Check if it's in a comment at end of line
                        if "#" in line and line.index("#") < line.index(pattern):
                            continue
                        violations.append((str(py_file), i, pattern))

        assert files_scanned > 0, "No files found - check path"
        if violations:
            msg = "Hardcoded path patterns found:\n"
            for filepath, lineno, pattern in violations[:10]:  # Show first 10
                msg += f"  {filepath}:{lineno}: {pattern}\n"
            pytest.fail(msg)


class TestNoOsAccessOnMissingPaths:
    """AC-5: No os.access/os.stat on potentially missing paths.

    Cross-platform hygiene test.
    """

    def test_no_os_access_in_production_code(self) -> None:
        """No os.access() in production code.

        Contract: sdk-boundary:Membrane:MUST:1

        The public provider Issue #4 pattern was calling os.access(path, os.X_OK)
        on test fixture paths that don't exist on Windows. Production code should
        use pathlib.Path.exists() or delegate to SDK.
        """
        src_root = Path("amplifier_module_provider_github_copilot")

        # These are problematic for cross-platform compatibility
        bad_functions = [
            "os.access(",
            "os.stat(",
            "os.chmod(",  # Doesn't work the same on Windows
        ]

        violations: list[tuple[str, int, str]] = []
        files_scanned = 0

        for py_file in src_root.rglob("*.py"):
            # Skip _permissions.py which legitimately uses these for executable check
            if py_file.name == "_permissions.py":
                continue

            files_scanned += 1
            source = py_file.read_text(encoding="utf-8")
            for i, line in enumerate(source.splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue

                for func in bad_functions:
                    if func in line:
                        # Skip if in comment
                        if "#" in line and line.index("#") < line.index(func):
                            continue
                        violations.append((str(py_file), i, func))

        assert files_scanned > 0, "No files found - check path"
        if violations:
            msg = "os.access/os.stat/os.chmod found in production code:\n"
            for filepath, lineno, func in violations[:10]:
                msg += f"  {filepath}:{lineno}: {func}\n"
            pytest.fail(msg)


class TestTokenResolutionPlatformIndependent:
    """AC-3: Token resolution uses os.environ only.

    Contract: behaviors:Logging:MUST:4
    """

    def test_token_resolution_no_platform_dependency(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """_resolve_token uses os.environ only.

        Contract: behaviors:Logging:MUST:4

        Token resolution must work the same on all platforms.
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            _resolve_token,  # pyright: ignore[reportPrivateUsage]
        )

        # Clear all token env vars
        for var in ("COPILOT_AGENT_TOKEN", "COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
            monkeypatch.delenv(var, raising=False)

        # Test with a specific token
        monkeypatch.setenv("GITHUB_TOKEN", "test-token-123")

        token = _resolve_token()
        assert token == "test-token-123"

        # Token should be identical regardless of platform
        # (we can't actually change platform, but we verify the function
        # uses only os.environ which is platform-independent)


class TestSessionConfigPlatformIndependent:
    """AC-4: Session config dict is pure Python, no platform calls.

    Contract: sdk-boundary:Config:MUST:1
    """

    @pytest.mark.asyncio
    async def test_session_config_no_platform_dependency(self) -> None:
        """Config dict is pure Python, no platform calls.

        Contract: sdk-boundary:Config:MUST:1

        Session config should be serializable JSON without any platform
        dependencies.
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        mock_client = ConfigCapturingMock()
        wrapper = CopilotClientWrapper(sdk_client=mock_client)

        async with wrapper.session(model="gpt-4", system_message="Test"):
            pass

        config = mock_client.last_config

        # Verify config contains only JSON-serializable types
        # (no Path objects, no platform-specific types)

        def check_serializable(obj: Any, path: str = "root") -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():  # type: ignore[reportUnknownVariableType]
                    check_serializable(v, f"{path}.{k}")
            elif isinstance(obj, list):
                for i, v in enumerate(obj):  # type: ignore[reportUnknownVariableType]
                    check_serializable(v, f"{path}[{i}]")
            elif callable(obj):
                # Callbacks are allowed (deny hook, permission handler)
                pass
            elif isinstance(obj, Path):
                pytest.fail(f"Path object found at {path} - should be string")
            elif not isinstance(obj, str | int | float | bool | type(None)):
                pytest.fail(f"Non-serializable type {type(obj)} at {path}")

        check_serializable(config)


class TestConfigLoadingUsesImportlib:
    """AC-5 related: Config loaded via importlib, not file paths.

    Contract: sdk-boundary:Config:MUST:1
    """

    def test_config_loading_uses_importlib_resources(self) -> None:
        """Config loaded via importlib, not file paths.

        Contract: sdk-boundary:Config:MUST:1

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

        # Basic validation - configs loaded successfully
        assert models_config is not None
        assert error_config is not None
        assert event_config is not None


class TestErrorConfigPlatformIndependent:
    """AC-6: ErrorConfig loads on any platform.

    Contract: sdk-boundary:Translation:MUST:2
    """

    def test_error_config_loading_platform_independent(self) -> None:
        """ErrorConfig loads on any platform.

        Contract: sdk-boundary:Translation:MUST:2

        Error config loading must work on Windows, macOS, and Linux.
        """
        from amplifier_module_provider_github_copilot.error_translation import (
            load_error_config,
        )

        config = load_error_config()

        # Should have error mappings
        assert config.mappings is not None
        assert len(config.mappings) > 0


class TestEventConfigPlatformIndependent:
    """AC-7: EventConfig loads on any platform.

    Contract: sdk-boundary:Translation:MUST:1
    """

    def test_event_config_loading_platform_independent(self) -> None:
        """EventConfig loads on any platform.

        Contract: sdk-boundary:Translation:MUST:1

        Event config loading must work on Windows, macOS, and Linux.
        """
        from amplifier_module_provider_github_copilot.streaming import load_event_config

        config = load_event_config()

        # Should have bridge mappings
        assert config.bridge_mappings is not None
        assert len(config.bridge_mappings) > 0


class TestWindowsEventLoopPolicy:
    """AC-1: Windows event loop policy in conftest.py.

    Contract: provider-protocol:QualityGates:MUST:1
    """

    def test_windows_event_loop_policy_set(self) -> None:
        """WindowsSelectorEventLoopPolicy active on win32.

        Contract: provider-protocol:QualityGates:MUST:1

        On Windows, we must use WindowsSelectorEventLoopPolicy to avoid
        ProactorEventLoop issues with asyncio subprocesses.
        """
        if sys.platform != "win32":
            pytest.skip("Windows-only test")

        import asyncio

        policy = asyncio.get_event_loop_policy()
        assert isinstance(policy, asyncio.WindowsSelectorEventLoopPolicy), (
            f"Expected WindowsSelectorEventLoopPolicy, got {type(policy)}"
        )

    def test_conftest_has_windows_policy(self) -> None:
        """conftest.py contains Windows event loop policy.

        Contract: provider-protocol:QualityGates:MUST:1
        """
        conftest_path = Path("tests/conftest.py")
        source = conftest_path.read_text(encoding="utf-8")

        assert "WindowsSelectorEventLoopPolicy" in source, (
            "conftest.py should set WindowsSelectorEventLoopPolicy for Windows"
        )
        assert 'sys.platform == "win32"' in source, (
            "conftest.py should check sys.platform before setting Windows policy"
        )


class TestPlatformModuleExists:
    """Platform module exists and is functional.

    Contract: sdk-boundary:Membrane:MUST:1
    """

    def test_platform_module_exists(self) -> None:
        """_platform.py module exists.

        Contract: sdk-boundary:Membrane:MUST:1
        """
        from amplifier_module_provider_github_copilot._platform import (
            PlatformInfo,
            get_platform_info,
        )

        info = get_platform_info()
        assert isinstance(info, PlatformInfo)

    def test_platform_info_has_required_fields(self) -> None:
        """PlatformInfo has name, is_windows, and cli_binary_name fields.

        Contract: sdk-boundary:Membrane:MUST:1
        """
        from amplifier_module_provider_github_copilot._platform import get_platform_info

        info = get_platform_info()
        assert hasattr(info, "name")
        assert hasattr(info, "is_windows")
        assert hasattr(info, "cli_binary_name")

        # Name should be one of the known platforms
        assert info.name in ("Windows", "macOS", "Unix")

        # cli_binary_name should match is_windows
        if info.is_windows:
            assert info.cli_binary_name == "copilot.exe"
        else:
            assert info.cli_binary_name == "copilot"


class TestPermissionsModuleExists:
    """Permissions module exists and is functional.

    Contract: sdk-boundary:Membrane:MUST:1
    """

    def test_permissions_module_exists(self) -> None:
        """_permissions.py module exists.

        Contract: sdk-boundary:Membrane:MUST:1
        """
        from amplifier_module_provider_github_copilot._permissions import (
            ensure_executable,
        )

        assert callable(ensure_executable)

    def test_ensure_executable_handles_missing_file(self) -> None:
        """ensure_executable doesn't raise on missing file.

        Contract: sdk-boundary:Membrane:MUST:1

        The function should be safe to call on paths that may not exist.
        """
        from amplifier_module_provider_github_copilot._permissions import (
            ensure_executable,
        )

        # Should not raise even if file doesn't exist
        result = ensure_executable(Path("/nonexistent/path/to/binary"))

        # Result depends on implementation - just verify it doesn't crash
        assert result is not None or result is None  # Either is OK


class TestWSLDetection:
    """PlatformInfo reports WSL vs native Linux distinctly.

    Contract: sdk-boundary:BinaryResolution:MUST:1 — platform detection must
    accurately detect the running environment. WSL has different binary
    resolution characteristics than native Linux.
    """

    def test_platform_info_has_is_wsl_field(self) -> None:
        """PlatformInfo must have an is_wsl field.

        Contract: sdk-boundary:BinaryResolution:MUST:1
        """
        from amplifier_module_provider_github_copilot._platform import get_platform_info

        info = get_platform_info()
        assert hasattr(info, "is_wsl"), (
            "PlatformInfo must expose is_wsl field to distinguish WSL from native Linux"
        )

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
