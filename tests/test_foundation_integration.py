"""
Tests for foundation integration.

Verifies bundle.md, exports, and config loading.
"""

from __future__ import annotations

from pathlib import Path


class TestBundleFile:
    """AC-1: bundle.md removed per commit 24e2b48 - test pyproject.toml instead."""

    def test_pyproject_exists(self) -> None:
        """pyproject.toml exists at project root."""
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        assert pyproject_path.exists(), "pyproject.toml should exist at project root"

    def test_pyproject_has_project_section(self) -> None:
        """pyproject.toml contains project metadata."""
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        content = pyproject_path.read_text(encoding="utf-8")
        assert "[project]" in content, "pyproject.toml should have [project] section"
        assert "name" in content, "pyproject.toml should have name field"


class TestModuleExports:
    """AC-2: Module exports are correct."""

    def test_all_exports_defined(self) -> None:
        """__all__ is defined with required exports."""
        import amplifier_module_provider_github_copilot as mod

        assert hasattr(mod, "__all__")
        assert "mount" in mod.__all__
        assert "GitHubCopilotProvider" in mod.__all__

    def test_all_exports_ecosystem_alignment(self) -> None:
        """Contract: provider-protocol:public_api:MUST:1,2

        __all__ exports only mount and provider class.
        Kernel types must be imported from amplifier_core.
        """
        import amplifier_module_provider_github_copilot as mod

        expected = {"mount", "GitHubCopilotProvider"}
        actual = set(mod.__all__)

        assert actual == expected, (
            f"__all__ must export only mount and provider class. Expected {expected}, got {actual}."
        )

    def test_amplifier_module_type(self) -> None:
        """__amplifier_module_type__ is set to 'provider'."""
        import amplifier_module_provider_github_copilot as mod

        assert hasattr(mod, "__amplifier_module_type__")
        assert mod.__amplifier_module_type__ == "provider"

    def test_mount_is_async(self) -> None:
        """mount() is an async function."""
        import inspect

        import amplifier_module_provider_github_copilot as mod

        assert inspect.iscoroutinefunction(mod.mount)


class TestConfigPackage:
    """AC-4: Config path uses importlib.resources."""

    def test_config_init_exists(self) -> None:
        """config/__init__.py exists (makes it a package)."""
        config_init = (
            Path(__file__).parent.parent
            / "amplifier_module_provider_github_copilot"
            / "config"
            / "__init__.py"
        )
        assert config_init.exists(), "config/__init__.py should exist"

    def test_error_config_loads_with_fallback(self) -> None:
        """Error config loading falls back gracefully."""
        from amplifier_module_provider_github_copilot.error_translation import ErrorConfig
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            _load_error_config_once,  # type: ignore[reportPrivateUsage]  # Testing internal loader
        )

        # Should return ErrorConfig (either loaded or default)
        config = _load_error_config_once()
        assert isinstance(config, ErrorConfig)


class TestDeprecationShims:
    """Test _deprecated.py import error messages.

    Coverage: _deprecated.py lines 7-63
    """

    def test_removed_symbol_raises_import_error(self) -> None:
        """Importing removed symbol raises ImportError with helpful message."""
        import pytest

        from amplifier_module_provider_github_copilot import _deprecated

        with pytest.raises(ImportError) as exc_info:
            _deprecated.__getattr__("CopilotProviderError")

        assert "removed in v2.0.0" in str(exc_info.value)
        assert "ProviderError" in str(exc_info.value)

    def test_unknown_symbol_raises_attribute_error(self) -> None:
        """Importing unknown symbol raises AttributeError."""
        import pytest

        from amplifier_module_provider_github_copilot import _deprecated

        with pytest.raises(AttributeError) as exc_info:
            _deprecated.__getattr__("NonExistentSymbol")

        assert "has no attribute" in str(exc_info.value)

    def test_all_removed_symbols_have_messages(self) -> None:
        """All symbols in REMOVED_SYMBOLS have migration messages."""
        from amplifier_module_provider_github_copilot._deprecated import REMOVED_SYMBOLS

        for _symbol, message in REMOVED_SYMBOLS.items():
            assert "removed in v2.0.0" in message.lower() or "v2.0.0" in message
            assert len(message) > 20  # Meaningful message
