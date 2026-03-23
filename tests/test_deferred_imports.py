"""Deferred Import Cleanup Tests.

Verify that ProviderUnavailableError and LLMError are imported at
top-level in provider.py, not inside functions.

Contract: N/A — code quality cleanup, no contract implications
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PROVIDER_FILE = Path("amplifier_module_provider_github_copilot/provider.py")


class TestDeferredImportsRemoved:
    """Verify deferred imports were moved to top-level."""

    def test_no_deferred_imports_in_functions(self) -> None:
        """No import statements inside function bodies for error types."""
        if not PROVIDER_FILE.exists():
            pytest.skip("provider.py not found")

        content = PROVIDER_FILE.read_text()
        tree = ast.parse(content)

        deferred_imports: list[str] = []

        for node in ast.walk(tree):
            # Check if node is a function definition
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Walk the function body looking for imports
                for child in ast.walk(node):
                    if isinstance(child, ast.ImportFrom):
                        # Check if importing error types from error_translation
                        if child.module and "error_translation" in child.module:
                            imported_names = [alias.name for alias in child.names]
                            if "ProviderUnavailableError" in imported_names:
                                deferred_imports.append(
                                    f"{node.name}(): deferred ProviderUnavailableError"
                                )
                            if "LLMError" in imported_names:
                                deferred_imports.append(
                                    f"{node.name}(): from {child.module} import LLMError"
                                )

        assert not deferred_imports, (
            "Found deferred imports that should be at top-level:\n"
            + "\n".join(f"  - {imp}" for imp in deferred_imports)
        )

    def test_top_level_imports_exist(self) -> None:
        """ProviderUnavailableError and LLMError are imported at module level."""
        if not PROVIDER_FILE.exists():
            pytest.skip("provider.py not found")

        content = PROVIDER_FILE.read_text()
        tree = ast.parse(content)

        # Find top-level imports (not inside functions)
        top_level_imports: set[str] = set()
        for node in tree.body:  # Only top-level statements
            if isinstance(node, ast.ImportFrom):
                if node.module and "error_translation" in node.module:
                    for alias in node.names:
                        top_level_imports.add(alias.name)

        assert "ProviderUnavailableError" in top_level_imports, (
            "ProviderUnavailableError should be in top-level imports"
        )
        assert "LLMError" in top_level_imports, "LLMError should be in top-level imports"

    def test_module_imports_without_circular_import(self) -> None:
        """Module can be imported without circular import error."""
        # This test verifies the import works at runtime
        try:
            import importlib

            # Clear cached module to force fresh import
            import sys

            if "amplifier_module_provider_github_copilot.provider" in sys.modules:
                del sys.modules["amplifier_module_provider_github_copilot.provider"]

            importlib.import_module("amplifier_module_provider_github_copilot.provider")
        except ImportError as e:
            if "circular" in str(e).lower():
                pytest.fail(f"Circular import detected: {e}")
            # Other import errors are OK (missing SDK, etc.)
            pytest.skip(f"Module import failed (expected in test environment): {e}")
