"""
Entry Point Registration Tests.

Contract: provider-protocol.md

Tests that the provider is discoverable by the Amplifier kernel via entry points.
"""

from __future__ import annotations

import pytest


class TestEntryPointRegistration:
    """Entry point registration tests."""

    def test_entry_point_registered(self) -> None:
        """AC: Kernel can discover provider via entry point.

        # Contract: provider-protocol:public_api:MUST:1
        """
        from importlib.metadata import entry_points

        eps = entry_points(group="amplifier.modules")
        names = [ep.name for ep in eps]
        if not names:
            # Package not installed in editable mode - verify pyproject.toml has entry point
            import tomllib
            from pathlib import Path

            pyproject = Path(__file__).parent.parent / "pyproject.toml"
            with open(pyproject, "rb") as f:
                data = tomllib.load(f)
            entry_points_section = data.get("project", {}).get("entry-points", {})
            amp_modules = entry_points_section.get("amplifier.modules", {})
            assert "provider-github-copilot" in amp_modules, (
                "Entry point 'provider-github-copilot' not declared in pyproject.toml"
            )
        else:
            assert "provider-github-copilot" in names, (
                f"Entry point 'provider-github-copilot' not found. Found: {names}"
            )

    def test_entry_point_loads_mount_function(self) -> None:
        """AC: Entry point loads mount function."""
        # Contract: provider-protocol:public_api:MUST:1
        from importlib.metadata import entry_points

        eps = entry_points(group="amplifier.modules")
        ep = next((ep for ep in eps if ep.name == "provider-github-copilot"), None)

        if ep is None:
            # Package not installed - test direct import instead
            from amplifier_module_provider_github_copilot import mount

            assert callable(mount), "mount should be callable"
            assert mount.__name__ == "mount", f"Expected 'mount', got '{mount.__name__}'"
        else:
            mount_fn = ep.load()
            assert callable(mount_fn), "mount should be callable"
            assert mount_fn.__name__ == "mount", f"Expected 'mount', got '{mount_fn.__name__}'"

    def test_mount_function_signature(self) -> None:
        """AC: mount() has correct signature."""
        # Contract: provider-protocol:mount:MUST:1
        import inspect

        from amplifier_module_provider_github_copilot import mount

        sig = inspect.signature(mount)
        params = list(sig.parameters.keys())

        assert "coordinator" in params, "mount() must accept coordinator parameter"
        assert "config" in params, "mount() must accept config parameter"

    def test_module_type_metadata(self) -> None:
        """AC: Module declares type as 'provider'.

        # Contract: provider-protocol:public_api:MUST:1
        """
        import amplifier_module_provider_github_copilot as module

        assert module.__amplifier_module_type__ == "provider"

    def test_module_exports(self) -> None:
        """AC: Module exports required symbols."""
        # Contract: provider-protocol:public_api:MUST:1
        import amplifier_module_provider_github_copilot as module

        # Behavioral: call mount to verify callable, check GitHubCopilotProvider is a type
        assert callable(module.mount), "Module must export callable mount"
        assert isinstance(module.GitHubCopilotProvider, type), (
            "GitHubCopilotProvider must be a class"
        )
        assert "mount" in module.__all__
        assert "GitHubCopilotProvider" in module.__all__


class TestMountFunction:
    """Tests for mount() behavior."""

    @pytest.mark.asyncio
    async def test_mount_creates_provider(self) -> None:
        """mount() creates and registers provider with coordinator."""
        # Contract: provider-protocol:mount:MUST:3
        from unittest.mock import AsyncMock, MagicMock

        from amplifier_core import ModuleCoordinator

        from amplifier_module_provider_github_copilot import mount

        coordinator = MagicMock(spec=ModuleCoordinator)
        coordinator.mount = AsyncMock()

        cleanup = await mount(coordinator)

        coordinator.mount.assert_called_once()
        call_args = coordinator.mount.call_args
        assert call_args[0][0] == "providers"
        assert call_args[1]["name"] == "github-copilot"
        assert callable(cleanup)

    @pytest.mark.asyncio
    async def test_mount_returns_cleanup_function(self) -> None:
        """mount() returns async cleanup callable."""
        # Contract: provider-protocol:mount:MUST:2
        import inspect
        from unittest.mock import AsyncMock, MagicMock

        from amplifier_core import ModuleCoordinator

        from amplifier_module_provider_github_copilot import mount

        coordinator = MagicMock(spec=ModuleCoordinator)
        coordinator.mount = AsyncMock()

        cleanup = await mount(coordinator)

        assert callable(cleanup)
        assert inspect.iscoroutinefunction(cleanup)


def test_package_has_version() -> None:
    """Verify package exposes __version__.

    Supersedes: test_placeholder.py::test_version_exists
    Contract: provider-protocol:public_api:MUST:1 (package identity)
    """
    from amplifier_module_provider_github_copilot import __version__

    assert isinstance(__version__, str)
    assert len(__version__) > 0


class TestModuleGetattr:
    """Tests for module-level __getattr__ fallback behavior."""

    def test_unknown_attribute_raises_attribute_error(self) -> None:
        """Accessing a truly unknown module attribute raises AttributeError.

        N/A — module __getattr__ upgrade-compatibility path.
        Line 357 in __init__.py — fallback for names NOT in REMOVED_SYMBOLS.
        """
        import amplifier_module_provider_github_copilot as m

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = m.xyz_absolute_nonexistent_attribute_99887766  # type: ignore[attr-defined]

    def test_removed_symbol_raises_import_error(self) -> None:
        """Accessing a v1.x removed symbol raises ImportError with guidance.

        N/A — module __getattr__ upgrade-compatibility path.
        Line 355 in __init__.py — ImportError for names in REMOVED_SYMBOLS.
        """
        from amplifier_module_provider_github_copilot._deprecated import REMOVED_SYMBOLS

        # REMOVED_SYMBOLS currently has symbols; if empty, loop is vacuously true
        if not REMOVED_SYMBOLS:
            assert REMOVED_SYMBOLS == {}  # vacuous but explicit — update when symbols are removed
            return

        removed_name = next(iter(REMOVED_SYMBOLS))  # First entry

        import amplifier_module_provider_github_copilot as m

        with pytest.raises(ImportError):
            _ = getattr(m, removed_name)


class TestSDKVersionCheck:
    """sdk-boundary:Membrane:MUST:5 — fail at import time on wrong SDK version.

    Regression guard for the clean-machine init bug:
    When SDK 0.1.x is installed, SubprocessConfig is absent but the old
    presence-only check passed silently, causing a cryptic ConfigurationError
    deep in the init flow instead of a clear ImportError at module load.

    All tests call the REAL _check_sdk_version() from __init__.py, not a copy.
    This ensures changes to the actual function are caught by the test suite.

    Contract: sdk-boundary:Membrane:MUST:5
    """

    def test_old_sdk_version_raises_import_error(self) -> None:
        """_check_sdk_version MUST raise ImportError for SDK < 0.2.0.

        Contract: sdk-boundary:Membrane:MUST:5
        """
        from amplifier_module_provider_github_copilot import (
            _check_sdk_version,  # type: ignore[reportPrivateUsage]
        )

        for old_ver in ("0.1.0", "0.1.28", "0.0.1", "0.1.99"):
            with pytest.raises(ImportError, match="github-copilot-sdk"):
                _check_sdk_version(old_ver)

        # Correct versions — must NOT raise
        for good_ver in ("0.2.0", "0.2.1", "0.2.99", "1.0.0"):
            _check_sdk_version(good_ver)  # no exception

    def test_correct_sdk_version_installed(self) -> None:
        """SDK installed in test environment MUST be >= 0.2.0.

        Contract: sdk-boundary:Membrane:MUST:5 (negative case)
        """
        import importlib.metadata

        version = importlib.metadata.version("github-copilot-sdk")
        parts = tuple(int(x) for x in version.split(".")[:2] if x.isdigit())
        assert parts >= (0, 2), (
            f"Test environment has SDK {version} which is < 0.2.0. "
            "Install 'github-copilot-sdk>=0.2.0' to run these tests."
        )

    def test_version_check_error_message_is_actionable(self) -> None:
        """ImportError message MUST name the package, installed version, and fix command.

        Users must know how to fix it from the error message alone.
        Contract: sdk-boundary:Membrane:MUST:5
        """
        from amplifier_module_provider_github_copilot import (
            _check_sdk_version,  # type: ignore[reportPrivateUsage]
        )

        with pytest.raises(ImportError) as exc_info:
            _check_sdk_version("0.1.28")

        error_msg = str(exc_info.value)
        assert "0.1.28" in error_msg, "Error must include the installed version"
        assert "0.2.0" in error_msg, "Error must state the required version"
        assert "github-copilot-sdk" in error_msg, "Error must name the package"
        assert "amplifier provider install" in error_msg, (
            "Error must include the amplifier provider install command"
        )

    def test_malformed_version_string_does_not_raise_unhandled(self) -> None:
        """Malformed SDK version string MUST NOT produce an unhandled exception.

        Pre-release suffix, empty string, or non-numeric must be safely handled.
        Contract: sdk-boundary:Membrane:MUST:5 (robustness)
        """
        from amplifier_module_provider_github_copilot import (
            _check_sdk_version,  # type: ignore[reportPrivateUsage]
        )

        # These have no valid semver minor → treated as (0, 0) → raise ImportError
        for weird_ver in ("", "unknown", "0.1.0rc1"):
            with pytest.raises(ImportError):
                _check_sdk_version(weird_ver)

        # "0.2.0b1" → "0" and "2" are digits → ver_parts = (0, 2) → does NOT raise
        _check_sdk_version("0.2.0b1")
