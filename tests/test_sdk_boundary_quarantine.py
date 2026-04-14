"""SDK boundary quarantine tests.

Contract: contracts/sdk-boundary.md

Tests in this module verify:
- ImportQuarantine:MUST:1 — SDK imports confined to sdk_adapter/
- ImportQuarantine:MUST:5 — ImportError with install instructions if SDK absent
- ImportQuarantine:MUST:6 — Multi-level fallback for moved SDK types
- Membrane:MUST:1 — Import from sdk_adapter package, not submodules
- Membrane:MUST:3 — __init__.py does not expose _imports module
"""

from __future__ import annotations

import ast
import importlib
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from types import ModuleType

# Root path for source code
SDK_ADAPTER_PATH = Path("amplifier_module_provider_github_copilot/sdk_adapter")
IMPORTS_FILE = SDK_ADAPTER_PATH / "_imports.py"


class TestSDKImportQuarantine:
    """Verify SDK imports are quarantined in _imports.py.

    Contract: sdk-boundary:Membrane:MUST:1
    """

    def test_imports_py_exists(self) -> None:
        """_imports.py MUST exist as the single SDK import point.

        Contract: sdk-boundary:Membrane:MUST:1
        """
        # Contract: sdk-boundary:Membrane:MUST:1
        assert IMPORTS_FILE.exists(), (
            f"_imports.py not found at {IMPORTS_FILE}. "
            "SDK imports must be quarantined in _imports.py."
        )


class TestSDKAdapterExports:
    """Verify __init__.py does not expose private quarantine module.

    Contract: sdk-boundary:Membrane:MUST:3
    """

    def test_init_does_not_expose_imports_module(self) -> None:
        """__init__.py MUST NOT re-export the _imports quarantine module.

        Contract: sdk-boundary:Membrane:MUST:3

        Domain code must use the public sdk_adapter API, never reach into
        _imports directly. If _imports is in __all__, domain code could
        accidentally bypass the membrane.
        """
        # Contract: sdk-boundary:Membrane:MUST:3
        init_file = SDK_ADAPTER_PATH / "__init__.py"
        assert init_file.exists(), f"{init_file} must exist in the repository"

        content = init_file.read_text(encoding="utf-8")
        tree = ast.parse(content)

        # Find __all__ list and check it does not contain "_imports"
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, ast.List):
                            exports = [
                                elt.value
                                for elt in node.value.elts
                                if isinstance(elt, ast.Constant)
                            ]
                            assert "_imports" not in exports, (
                                "__all__ exports '_imports' — this violates "
                                "sdk-boundary:Membrane:MUST:3. Domain code must not "
                                "reach into private quarantine module."
                            )


class TestSDKImportsRealPath:
    """Cover sdk_adapter/_imports.py real SDK import paths.

    Contract: sdk-boundary:Membrane:MUST:5
    Contract: sdk-boundary:ImportQuarantine:MUST:6
    """

    def _save_import_state(self) -> tuple[str | None, ModuleType | None]:
        """Save environment and module state before import tests."""
        original_skip = os.environ.get("SKIP_SDK_CHECK")
        original_module = sys.modules.pop(
            "amplifier_module_provider_github_copilot.sdk_adapter._imports", None
        )
        return original_skip, original_module

    def _restore_import_state(
        self, original_skip: str | None, original_module: ModuleType | None
    ) -> None:
        """Restore environment and module state after import tests."""
        if original_skip is not None:
            os.environ["SKIP_SDK_CHECK"] = original_skip
        else:
            os.environ["SKIP_SDK_CHECK"] = "1"

        sys.modules.pop("amplifier_module_provider_github_copilot.sdk_adapter._imports", None)

        if original_module is not None:
            sys.modules["amplifier_module_provider_github_copilot.sdk_adapter._imports"] = (
                original_module
            )

    def test_sdk_import_failure_raises_import_error(self) -> None:
        """copilot import failing MUST raise ImportError with install instructions.

        Contract: sdk-boundary:Membrane:MUST:5

        When SKIP_SDK_CHECK is not set and copilot is not importable,
        _imports.py must raise ImportError with a message containing
        'github-copilot-sdk not installed'.
        """
        # Contract: sdk-boundary:Membrane:MUST:5
        original_skip, original_module = self._save_import_state()

        try:
            os.environ.pop("SKIP_SDK_CHECK", None)

            # Clear any cached copilot modules
            copilot_modules = [k for k in sys.modules if k.startswith("copilot")]
            for k in copilot_modules:
                sys.modules.pop(k, None)

            with patch.dict("sys.modules", {"copilot": None}):
                with pytest.raises(ImportError, match="github-copilot-sdk not installed"):
                    importlib.import_module(
                        "amplifier_module_provider_github_copilot.sdk_adapter._imports"
                    )
        finally:
            self._restore_import_state(original_skip, original_module)

    def test_permission_request_result_missing_sets_none(self) -> None:
        """PermissionRequestResult MUST be None when absent from all SDK locations.

        Contract: sdk-boundary:ImportQuarantine:MUST:6

        Simulates SDK < 0.1.28 which predates PermissionRequestResult entirely.
        All three fallback levels (copilot.types, copilot root, copilot.session)
        fail, resulting in PermissionRequestResult = None.
        """
        # Contract: sdk-boundary:ImportQuarantine:MUST:6
        original_skip, original_module = self._save_import_state()

        try:
            os.environ.pop("SKIP_SDK_CHECK", None)

            # Mock copilot with CopilotClient but no PermissionRequestResult anywhere
            mock_copilot = MagicMock(spec=["CopilotClient"])
            mock_copilot.CopilotClient = MagicMock(name="CopilotClient")

            with patch.dict(
                "sys.modules",
                {
                    "copilot": mock_copilot,
                    "copilot.types": None,  # ImportError on from-import
                    "copilot.session": None,  # ImportError on from-import
                },
            ):
                mod = importlib.import_module(
                    "amplifier_module_provider_github_copilot.sdk_adapter._imports"
                )
                # PermissionRequestResult must be None when all fallbacks fail
                assert mod.PermissionRequestResult is None, (
                    "PermissionRequestResult should be None when absent from all SDK locations"
                )
        finally:
            self._restore_import_state(original_skip, original_module)

    def test_subprocess_config_loads_when_copilot_types_absent(self) -> None:
        """SubprocessConfig MUST resolve even when copilot.types does not exist.

        Contract: sdk-boundary:ImportQuarantine:MUST:6

        Regression test for SDK 0.2.1 breaking change: copilot.types was removed.
        SubprocessConfig moved to copilot.client, re-exported from copilot root.

        Before fix: from copilot.types import SubprocessConfig → ModuleNotFoundError
                    → SubprocessConfig = None → ConfigurationError with any GitHub token
        After fix:  fallback to from copilot import SubprocessConfig → succeeds
        """
        # Contract: sdk-boundary:ImportQuarantine:MUST:6
        original_skip, original_module = self._save_import_state()

        try:
            os.environ.pop("SKIP_SDK_CHECK", None)

            # Simulate SDK 0.2.1: copilot root has SubprocessConfig,
            # but copilot.types does NOT exist.
            mock_subprocess_config = MagicMock(name="SubprocessConfig")
            mock_copilot = MagicMock(spec=["CopilotClient", "SubprocessConfig"])
            mock_copilot.CopilotClient = MagicMock(name="CopilotClient")
            mock_copilot.SubprocessConfig = mock_subprocess_config

            with patch.dict(
                "sys.modules",
                {
                    "copilot": mock_copilot,
                    "copilot.types": None,  # None → ModuleNotFoundError on import
                },
            ):
                mod = importlib.import_module(
                    "amplifier_module_provider_github_copilot.sdk_adapter._imports"
                )

            # SubprocessConfig MUST be the mock from copilot root, not None
            assert mod.SubprocessConfig is mock_subprocess_config, (
                f"SubprocessConfig is {mod.SubprocessConfig!r} but should be "
                f"{mock_subprocess_config!r}. _imports.py must fall back to "
                "'from copilot import SubprocessConfig' when copilot.types is absent."
            )

        finally:
            self._restore_import_state(original_skip, original_module)

    def test_permission_request_result_falls_back_to_copilot_session(self) -> None:
        """PermissionRequestResult MUST resolve from copilot.session when copilot.types absent.

        Contract: sdk-boundary:ImportQuarantine:MUST:6

        Regression guard for SDK v0.2.1 breaking change: copilot/types.py was deleted
        (PR #871). PermissionRequestResult moved to copilot.session and is NOT
        re-exported from the copilot root package.

        Before fix:  copilot.types fails → copilot root fails → PermissionRequestResult = None
                     → deny_permission_request() returns dict → SDK calls .kind → AttributeError
                     → wrong deny reason: 'denied-no-approval-rule-and-could-not-request-from-user'

        After fix:   copilot.types fails → copilot root fails → copilot.session succeeds
                     → PermissionRequestResult is the real class → .kind works → 'denied-by-rules'
        """
        # Contract: sdk-boundary:ImportQuarantine:MUST:6
        original_skip, original_module = self._save_import_state()

        try:
            os.environ.pop("SKIP_SDK_CHECK", None)

            # Simulate SDK v0.2.1:
            # - copilot root has CopilotClient but NOT PermissionRequestResult
            # - copilot.types is deleted (None → ImportError on from-import)
            # - copilot.session has PermissionRequestResult
            mock_prr = MagicMock(name="PermissionRequestResult")
            mock_copilot = MagicMock(spec=["CopilotClient"])
            mock_copilot.CopilotClient = MagicMock(name="CopilotClient")

            mock_copilot_session = MagicMock(spec=["PermissionRequestResult"])
            mock_copilot_session.PermissionRequestResult = mock_prr

            with patch.dict(
                "sys.modules",
                {
                    "copilot": mock_copilot,
                    "copilot.types": None,  # None → ImportError on from-import
                    "copilot.session": mock_copilot_session,
                },
            ):
                mod = importlib.import_module(
                    "amplifier_module_provider_github_copilot.sdk_adapter._imports"
                )

            # PermissionRequestResult MUST be the mock from copilot.session
            assert mod.PermissionRequestResult is mock_prr, (
                f"PermissionRequestResult is {mod.PermissionRequestResult!r} but should be "
                f"{mock_prr!r}. _imports.py must fall back to "
                "'from copilot.session import PermissionRequestResult'."
            )

        finally:
            self._restore_import_state(original_skip, original_module)


class TestMembraneAPIPattern:
    """Verify domain code uses sdk_adapter package API, not submodule imports.

    Contract: sdk-boundary:Membrane:MUST:1, sdk-boundary:Membrane:MUST:3

    Domain modules (provider.py, streaming.py, request_adapter.py, __init__.py)
    MUST import from .sdk_adapter package, NOT from .sdk_adapter.client,
    .sdk_adapter.types, etc. directly.

    This ensures encapsulation: internal restructuring doesn't break domain code.
    """

    # Domain files that should use membrane API
    DOMAIN_FILES = [
        "amplifier_module_provider_github_copilot/provider.py",
        "amplifier_module_provider_github_copilot/streaming.py",
        "amplifier_module_provider_github_copilot/request_adapter.py",
        "amplifier_module_provider_github_copilot/__init__.py",
    ]

    # Forbidden submodule import patterns (should use .sdk_adapter not .sdk_adapter.X)
    FORBIDDEN_PATTERNS = [
        ".sdk_adapter.client",
        ".sdk_adapter.event_helpers",
        ".sdk_adapter.extract",
        ".sdk_adapter.tool_capture",
        ".sdk_adapter.types",
        ".sdk_adapter._imports",
        ".sdk_adapter._spec_utils",
        ".sdk_adapter.model_translation",
    ]

    @pytest.mark.parametrize("file_path", DOMAIN_FILES)
    def test_domain_file_uses_membrane_api(self, file_path: str) -> None:
        """Domain file MUST import from sdk_adapter package, not submodules.

        Contract: sdk-boundary:Membrane:MUST:1, sdk-boundary:Membrane:MUST:3

        Example of WRONG (bypasses membrane):
            from .sdk_adapter.client import CopilotClientWrapper

        Example of RIGHT (uses membrane):
            from .sdk_adapter import CopilotClientWrapper
        """
        # Contract: sdk-boundary:Membrane:MUST:1, sdk-boundary:Membrane:MUST:3
        py_file = Path(file_path)
        assert py_file.exists(), f"{file_path} must exist in the repository"

        content = py_file.read_text(encoding="utf-8")
        tree = ast.parse(content)

        violations: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                # Check if import is from sdk_adapter submodule
                for pattern in self.FORBIDDEN_PATTERNS:
                    if node.module.endswith(pattern) or pattern in node.module:
                        names = ", ".join(alias.name for alias in node.names)
                        violations.append(
                            f"from {node.module} import {names} "
                            f"(line {node.lineno}) — should use from .sdk_adapter import"
                        )

        assert violations == [], (
            f"{file_path} bypasses sdk_adapter membrane:\n"
            + "\n".join(f"  - {v}" for v in violations)
            + "\n\nFix: import from .sdk_adapter package, not submodules. "
            "See sdk-boundary:Membrane:MUST:1"
        )
