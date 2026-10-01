"""SDK boundary quarantine tests.

Contract: contracts/sdk-boundary.md

Tests in this module verify:
- ImportQuarantine:MUST:1 — SDK imports confined to sdk_adapter/
- ImportQuarantine:MUST:5 — ImportError with install instructions if SDK absent
- ImportQuarantine:MUST:6 — Direct imports for pinned SDK version (no fallback chains)
- ImportQuarantine:MUST:8 — SDK constructor calls encapsulated via factory
- Membrane:MUST:1 — Import from sdk_adapter package, not submodules
- Membrane:MUST:3 — __init__.py does not expose _imports module
"""

from __future__ import annotations

import ast
import importlib
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest

from tests._sdk_version_gate import require_sdk

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

    def _save_import_state(self) -> tuple[str | None, ModuleType | None, Any]:
        """Save environment and module state before import tests.

        Captures client.CopilotClient and the sdk_adapter._imports package
        attribute so _restore_import_state can fully undo any mock leakage
        caused by re-importing _imports inside a patch.dict context.
        """
        original_skip = os.environ.get("SKIP_SDK_CHECK")
        original_module = sys.modules.pop(
            "amplifier_module_provider_github_copilot.sdk_adapter._imports", None
        )
        # Option B fix: capture the client.py snapshot binding and the
        # sdk_adapter._imports package attribute so _restore_import_state can
        # write both back after the test. The core issue is that
        # importlib.import_module("..._imports") inside a patch.dict context
        # sets sdk_adapter._imports to M2 (mock). patch.dict only restores
        # sys.modules keys on exit, not package-level attributes, so
        # sdk_adapter._imports stays as M2 and from . import _imports in
        # client._get_or_create_client resolves to M2 instead of M1, making
        # patch("...sdk_adapter._imports.CopilotClient", ...) unreachable.
        client_mod = sys.modules.get("amplifier_module_provider_github_copilot.sdk_adapter.client")
        original_client_cc: Any = client_mod.CopilotClient if client_mod is not None else None
        return original_skip, original_module, original_client_cc

    def _restore_import_state(
        self,
        original_skip: str | None,
        original_module: ModuleType | None,
        original_client_cc: Any = None,
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

        # Write back the client.py snapshot (belt-and-suspenders). If _imports was
        # re-imported under a mock environment, client.CopilotClient may have
        # been set to a MagicMock. Restoring it here ensures subsequent tests
        # that patch _imports.CopilotClient still reach the fallback branch in
        # client.py:L347 (_imports.CopilotClient) rather than a stale mock.
        client_mod = sys.modules.get("amplifier_module_provider_github_copilot.sdk_adapter.client")
        if client_mod is not None:
            client_mod_any: Any = client_mod
            client_mod_any.CopilotClient = original_client_cc

        # Sync the sdk_adapter PACKAGE ATTRIBUTE back to the restored module.
        # importlib.import_module("..._imports") inside a patch.dict context sets
        # sdk_adapter._imports (the package-level attribute) to the mock M2.
        # patch.dict only restores sys.modules dict keys on exit — it does NOT
        # revert the attribute on the parent package object. Consequently,
        # from . import _imports in client._get_or_create_client resolves to M2
        # via sdk_adapter._imports, not to M1 in sys.modules, so any
        # patch("..._imports.CopilotClient", ...) applied to M1 is silently
        # ignored. Syncing the attribute here closes that gap.
        sdk_adapter_mod = sys.modules.get("amplifier_module_provider_github_copilot.sdk_adapter")
        if sdk_adapter_mod is not None:
            restored_imports = sys.modules.get(
                "amplifier_module_provider_github_copilot.sdk_adapter._imports"
            )
            sdk_adapter_any: Any = sdk_adapter_mod
            if restored_imports is not None:
                sdk_adapter_any._imports = restored_imports
            elif hasattr(sdk_adapter_mod, "_imports"):
                delattr(sdk_adapter_mod, "_imports")

    def test_sdk_import_failure_raises_import_error(self) -> None:
        """copilot import failing MUST raise ImportError with install instructions.

        Contract: sdk-boundary:Membrane:MUST:5

        When SKIP_SDK_CHECK is not set and copilot is not importable,
        _imports.py must raise ImportError with a message containing
        'github-copilot-sdk not installed'.
        """
        # Contract: sdk-boundary:Membrane:MUST:5
        original_skip, original_module, original_client_cc = self._save_import_state()

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
            self._restore_import_state(original_skip, original_module, original_client_cc)

    def test_permission_request_result_loads_from_copilot_session(self) -> None:
        """PermissionRequestResult MUST resolve directly from copilot.session.

        Contract: sdk-boundary:ImportQuarantine:MUST:6

        SDK v1.0.0b10 canonical location: copilot.session.
        Direct import — no fallback chain.

        Note: this test uses ``sys.modules`` ``MagicMock`` patching to verify
        the import *structure* of the quarantine membrane. The companion
        live-SDK signature/path validation lives in
        ``tests/test_sdk_assumptions.py`` (``@pytest.mark.sdk_assumption``),
        which exercises the real ``copilot.*`` modules end-to-end. Both files
        together form the full boundary contract.
        """
        # Contract: sdk-boundary:ImportQuarantine:MUST:6
        original_skip, original_module, original_client_cc = self._save_import_state()

        try:
            os.environ.pop("SKIP_SDK_CHECK", None)

            mock_prr = MagicMock(name="PermissionRequestResult")
            mock_copilot = MagicMock(
                spec=[
                    "CopilotClient",
                    "ModelCapabilitiesOverride",
                    "ModelLimitsOverride",
                ]
            )
            mock_copilot.CopilotClient = MagicMock(name="CopilotClient")
            mock_copilot.ModelCapabilitiesOverride = MagicMock(name="ModelCapabilitiesOverride")
            mock_copilot.ModelLimitsOverride = MagicMock(name="ModelLimitsOverride")

            mock_copilot_session = MagicMock(spec=["PermissionRequestResult"])
            mock_copilot_session.PermissionRequestResult = mock_prr

            mock_copilot_generated_rpc = MagicMock()
            mock_copilot_generated_rpc.PermissionDecisionReject = MagicMock(
                name="PermissionDecisionReject"
            )
            mock_copilot_rpc = MagicMock(spec=["ModelsListRequest"])
            mock_copilot_rpc.ModelsListRequest = MagicMock(name="ModelsListRequest")
            mock_copilot_generated = MagicMock()

            with patch.dict(
                "sys.modules",
                {
                    "copilot": mock_copilot,
                    "copilot.session": mock_copilot_session,
                    "copilot.rpc": mock_copilot_rpc,
                    "copilot.generated": mock_copilot_generated,
                    "copilot.generated.rpc": mock_copilot_generated_rpc,
                },
            ):
                mod = importlib.import_module(
                    "amplifier_module_provider_github_copilot.sdk_adapter._imports"
                )

            assert mod.PermissionRequestResult is mock_prr, (
                f"PermissionRequestResult is {mod.PermissionRequestResult!r} but should be "
                f"{mock_prr!r}. _imports.py must import directly from copilot.session."
            )

        finally:
            self._restore_import_state(original_skip, original_module, original_client_cc)

    def test_subprocess_config_is_quarantined_from_sdk_surface(self) -> None:
        """SubprocessConfig MUST stay absent from the b10 import surface.

        Contract: sdk-boundary:ImportQuarantine:MUST:6

        Contract: sdk-boundary:SDKSurface:MUST:8

        SDK v1.0.0b10 (since b9) keeps SubprocessConfig absent from the public API.
        The quarantine module
        MUST expose a sentinel ``None`` instead of a working config object.
        """
        # Contract: sdk-boundary:ImportQuarantine:MUST:6
        # Contract: sdk-boundary:SDKSurface:MUST:8
        original_skip, original_module, original_client_cc = self._save_import_state()

        try:
            os.environ.pop("SKIP_SDK_CHECK", None)

            mock_copilot = MagicMock(
                spec=[
                    "CopilotClient",
                    "ModelCapabilitiesOverride",
                    "ModelLimitsOverride",
                ]
            )
            mock_copilot.CopilotClient = MagicMock(name="CopilotClient")
            mock_copilot.ModelCapabilitiesOverride = MagicMock(name="ModelCapabilitiesOverride")
            mock_copilot.ModelLimitsOverride = MagicMock(name="ModelLimitsOverride")

            mock_copilot_session = MagicMock(spec=["PermissionRequestResult"])
            mock_copilot_session.PermissionRequestResult = MagicMock(name="PermissionRequestResult")

            mock_copilot_generated_rpc = MagicMock()
            mock_copilot_generated_rpc.PermissionDecisionReject = MagicMock(
                name="PermissionDecisionReject"
            )
            mock_copilot_rpc = MagicMock(spec=["ModelsListRequest"])
            mock_copilot_rpc.ModelsListRequest = MagicMock(name="ModelsListRequest")
            mock_copilot_generated = MagicMock()

            with patch.dict(
                "sys.modules",
                {
                    "copilot": mock_copilot,
                    "copilot.session": mock_copilot_session,
                    "copilot.rpc": mock_copilot_rpc,
                    "copilot.generated": mock_copilot_generated,
                    "copilot.generated.rpc": mock_copilot_generated_rpc,
                },
            ):
                mod = importlib.import_module(
                    "amplifier_module_provider_github_copilot.sdk_adapter._imports"
                )

            assert mod.SubprocessConfig is None, (
                "SubprocessConfig must remain quarantined as None for b9+ "
                "instead of exposing a working config object."
            )

        finally:
            self._restore_import_state(original_skip, original_module, original_client_cc)

    def test_subprocess_config_absent_from_sdk(self) -> None:
        """The b10 runtime removes SubprocessConfig from the root SDK module.

        Contract: sdk-boundary:ImportQuarantine:MUST:6
        """
        copilot = require_sdk()
        sentinel = object()

        assert getattr(copilot, "SubprocessConfig", sentinel) is sentinel


class TestSDKConstructorEncapsulation:
    """Verify SDK constructor calls are encapsulated in _imports.py.

    Contract: sdk-boundary:ImportQuarantine:MUST:8
    """

    def test_make_permission_denied_exists_in_imports(self) -> None:
        """_imports.py MUST expose make_permission_denied factory.

        Contract: sdk-boundary:ImportQuarantine:MUST:8

        The factory encapsulates SDK constructor field knowledge so
        that client.py expresses intent only.
        """
        # Contract: sdk-boundary:ImportQuarantine:MUST:8
        content = IMPORTS_FILE.read_text(encoding="utf-8")
        tree = ast.parse(content)

        function_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        assert "make_permission_denied" in function_names, (
            "ImportQuarantine:MUST:8 — _imports.py must define make_permission_denied factory"
        )

    def test_make_permission_denied_returns_reject(self) -> None:
        """make_permission_denied MUST return result with kind='reject'.

        Contract: sdk-boundary:ImportQuarantine:MUST:8

        In test mode (SKIP_SDK_CHECK=1), PermissionRequestResult is None
        and the factory returns a dict. Verify exact field value.
        """
        # Contract: sdk-boundary:ImportQuarantine:MUST:8
        from amplifier_module_provider_github_copilot.sdk_adapter._imports import (
            make_permission_denied,
        )

        result = make_permission_denied()

        # Handle both real SDK object and dict fallback
        kind = result.kind if hasattr(result, "kind") else result.get("kind")
        assert kind == "reject", (
            f"ImportQuarantine:MUST:8 — make_permission_denied must return kind='reject', "
            f"got {kind!r}"
        )
        assert isinstance(kind, str), (
            f"ImportQuarantine:MUST:8 — kind must be str, got {type(kind).__name__}"
        )

    def test_client_does_not_call_permission_constructor_directly(self) -> None:
        """client.py MUST NOT call PermissionRequestResult(...) directly.

        Contract: sdk-boundary:ImportQuarantine:MUST:8

        SDK constructor calls must be encapsulated in _imports.py via factory.
        client.py must express intent only (call make_permission_denied),
        never instantiate SDK types directly.
        """
        # Contract: sdk-boundary:ImportQuarantine:MUST:8
        client_file = SDK_ADAPTER_PATH / "client.py"
        assert client_file.exists(), f"{client_file} must exist in the repository"

        content = client_file.read_text(encoding="utf-8")
        tree = ast.parse(content)

        violations: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Check for direct PermissionRequestResult(...) calls
                if isinstance(node.func, ast.Name) and node.func.id == "PermissionRequestResult":
                    violations.append(
                        f"PermissionRequestResult(...) called directly at line {node.lineno}"
                    )
                elif (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "PermissionRequestResult"
                ):
                    violations.append(
                        f"*.PermissionRequestResult(...) called directly at line {node.lineno}"
                    )

        assert violations == [], (
            "client.py calls SDK constructor directly — violates "
            "sdk-boundary:ImportQuarantine:MUST:8:\n"
            + "\n".join(f"  - {v}" for v in violations)
            + "\n\nFix: use make_permission_denied() factory from _imports.py"
        )

    def test_make_permission_denied_returns_decision_reject_no_kwargs(self) -> None:
        """make_permission_denied MUST call PermissionDecisionReject with zero arguments.

        Contract: sdk-boundary:SDKSurface:MUST:1b

        In b10, PermissionDecisionReject has kind as ClassVar; passing kind= or any
        other kwarg causes TypeError at runtime. Zero-arg call is the only safe form.
        """
        # Contract: sdk-boundary:SDKSurface:MUST:1b
        content = IMPORTS_FILE.read_text(encoding="utf-8")
        tree = ast.parse(content)

        call_found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "make_permission_denied":
                for call_node in ast.walk(node):
                    if (
                        isinstance(call_node, ast.Call)
                        and isinstance(call_node.func, ast.Name)
                        and call_node.func.id == "PermissionDecisionReject"
                    ):
                        call_found = True
                        assert len(call_node.args) == 0, (
                            "PermissionDecisionReject must be called with no positional args"
                        )
                        assert len(call_node.keywords) == 0, (
                            f"PermissionDecisionReject must be called with no kwargs in b10 — "
                            f"kind is ClassVar; got {[kw.arg for kw in call_node.keywords]!r}"
                        )

        assert call_found, (
            "sdk-boundary:SDKSurface:MUST:1b — make_permission_denied must call "
            "PermissionDecisionReject with no arguments"
        )

        # Behavioral: test-mode fallback returns SimpleNamespace with correct shape.
        from amplifier_module_provider_github_copilot.sdk_adapter._imports import (
            make_permission_denied,
        )

        result = make_permission_denied()
        assert result.kind == "reject", f"kind must be 'reject', got {result.kind!r}"
        assert result.feedback is None, f"feedback must be None, got {result.feedback!r}"

    def test_generated_rpc_import_documented(self) -> None:
        """The generated-rpc carve-out is isolated and visibly marked as debt.

        Contract: sdk-boundary:ImportQuarantine:MUST:7
        """
        import amplifier_module_provider_github_copilot.sdk_adapter._imports as imports_mod

        module_file = imports_mod.__file__
        if module_file is None:
            pytest.fail("sdk_adapter._imports must be backed by a source file")
        source_path = Path(module_file)
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        matching_imports = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module == "copilot.generated.rpc"
            and any(alias.name == "PermissionDecisionReject" for alias in node.names)
        ]

        assert len(matching_imports) == 1
        import_line = matching_imports[0].lineno
        preceding_line = source.splitlines()[import_line - 2]
        assert preceding_line.lstrip().startswith("# TODO(maintainer 2026):")


class TestSDKNoFallbackChains:
    """Verify _imports.py has no fallback chains or deleted-module imports.

    Contract: sdk-boundary:ImportQuarantine:MUST:6
    """

    def test_imports_py_has_no_copilot_types_import(self) -> None:
        """_imports.py MUST NOT import from copilot.types.

        Contract: sdk-boundary:ImportQuarantine:MUST:6

        copilot.types was deleted in SDK v0.2.1. Any import from it fails at
        import time with SDK >= v0.2.1. With pin >=0.3.0, this module is always
        absent. Direct imports from canonical locations only — no fallback chains.
        """
        # Contract: sdk-boundary:ImportQuarantine:MUST:6
        content = IMPORTS_FILE.read_text(encoding="utf-8")
        tree = ast.parse(content)

        violations: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "copilot.types":
                names = ", ".join(alias.name for alias in node.names)
                violations.append(f"from copilot.types import {names} at line {node.lineno}")

        assert violations == [], (
            "_imports.py imports from copilot.types (deleted in SDK v0.2.1):\n"
            + "\n".join(f"  - {v}" for v in violations)
            + "\n\nFix: use canonical b10 locations — "
            "PermissionRequestResult: copilot.session | CopilotClient: copilot root."
        )


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


class TestTestModeBindingInvariants:
    """Pin the test-mode bindings that other tests rely on for SDK substitution.

    conftest.py:26 globally sets `SKIP_SDK_CHECK=1` so the suite never imports
    the real SDK; this leaves both `client.CopilotClient` (re-exported from
    `_imports`) and `_imports.CopilotClient` bound to `None`. The fallback
    chain at `sdk_adapter/client.py:343-345` depends on this invariant. If a
    future change resolves either binding to a non-`None` value under pytest,
    every test that patches `_imports.CopilotClient` would silently run
    against the wrong target.
    """

    def test_skip_sdk_check_leaves_client_copilotclient_none(self) -> None:
        """client.CopilotClient must be None when SKIP_SDK_CHECK is active."""
        assert os.environ.get("SKIP_SDK_CHECK") == "1", (
            "conftest.py:26 must set SKIP_SDK_CHECK=1 globally; this test "
            "depends on that invariant."
        )
        from amplifier_module_provider_github_copilot.sdk_adapter import client as client_mod

        assert client_mod.CopilotClient is None, (
            "sdk_adapter.client.CopilotClient must be None under "
            "SKIP_SDK_CHECK=1 so the fallback to _imports.CopilotClient at "
            "client.py:343-345 remains the active resolution path."
        )

    def test_skip_sdk_check_leaves_imports_copilotclient_none(self) -> None:
        """_imports.CopilotClient must be None when SKIP_SDK_CHECK is active."""
        from amplifier_module_provider_github_copilot.sdk_adapter import _imports

        assert _imports.CopilotClient is None, (
            "_imports.CopilotClient must be None under SKIP_SDK_CHECK=1 so "
            "tests substituting it via monkeypatch hit the canonical anchor."
        )
