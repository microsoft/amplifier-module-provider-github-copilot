"""
Tests for foundation integration.

Verifies exports and config loading.
"""

from __future__ import annotations


class TestModuleExports:
    """AC-2: Module exports are correct."""

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


class TestConfigPackage:
    """AC-4: Config path uses importlib.resources."""

    def test_error_config_loads_with_fallback(self) -> None:
        """Error config loading falls back gracefully.

        Contract: error-hierarchy:config:SHOULD:1, error-hierarchy:config:MUST:2
        """
        from amplifier_module_provider_github_copilot.error_translation import ErrorConfig
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            _load_error_config_once,  # type: ignore[reportPrivateUsage]  # Testing internal loader
        )

        # Should return ErrorConfig (either loaded or default)
        config = _load_error_config_once()
        assert isinstance(config, ErrorConfig)
        assert len(config.mappings) == 17, (
            "error-hierarchy:config:MUST:2 — expected 17 error mappings from errors.yaml, "
            f"got {len(config.mappings)}"
        )


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
            assert "removed in v2.0.0" in message.lower()
            assert len(message) > 20  # Meaningful message
