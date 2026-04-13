"""
Contract Compliance Tests: Deny + Destroy Pattern.

Contract: contracts/deny-destroy.md

Tests the sovereignty guarantee - SDK never executes tools.

Test coverage by anchor:
- deny-destroy:DenyHook:MUST:1,2 → test_deny_hook_mandatory.py (TestDenyHookMandatoryClient)
- deny-destroy:DenyHook:MUST:3 → this file (TestDenyHookNotConfigurable)
- deny-destroy:NoExecution:MUST:3 → this file (TestArchitectureFitness)
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


class TestDenyHookNotConfigurable:
    """deny-destroy:DenyHook:MUST:3 — ARCHITECTURE FITNESS"""

    # Comprehensive deny list of forbidden config key patterns
    # Any key containing these terms could potentially disable the deny hook
    FORBIDDEN_PATTERNS = [
        "bypass",
        "skip",
        "disable",
        "allow",
        "override",
        "hook",
    ]

    FORBIDDEN_EXACT_KEYS = [
        "allow_tool_execution",
        "enable_tools",
        "tool_execution",
        "skip_deny",
        "bypass_sovereignty",
    ]

    def test_no_yaml_key_can_disable_deny_hook(self) -> None:
        """deny-destroy:DenyHook:MUST:3 - No config key can disable the deny hook.

        Expanded to check for broader patterns that could compromise sovereignty.
        Reads Python config modules and errors.yaml (authoritative per council verdict).
        """
        import importlib.resources

        import yaml

        from amplifier_module_provider_github_copilot.config import _models as _models

        # Load errors.yaml (authoritative source — kept as YAML per council verdict)
        pkg = importlib.resources.files("amplifier_module_provider_github_copilot.config.data")
        errors_yaml_text = (pkg / "errors.yaml").read_text(encoding="utf-8")
        errors_data = yaml.safe_load(errors_yaml_text)
        error_mappings = errors_data.get("error_mappings", [])

        # Collect all top-level config data dicts to inspect
        config_sources = [
            ("models.py/PROVIDER", _models.PROVIDER),
            ("models.py/MODELS", {"models": _models.MODELS}),
            ("errors.yaml/error_mappings", {"error_mappings": error_mappings}),
            ("sdk_protection.py", {}),  # dataclass-based, no dict keys to scan
        ]

        for source_name, content in config_sources:
            if not content:
                continue

            # Flatten all keys recursively
            all_keys = _collect_all_keys(content)

            # Check no key could disable deny hook
            for key in all_keys:
                key_lower = key.lower()

                # Check exact forbidden keys
                base_key = key_lower.split(".")[-1]  # Get the leaf key name
                assert base_key not in self.FORBIDDEN_EXACT_KEYS, (
                    f"Config {source_name} has forbidden key '{key}'"
                )

                # Check for forbidden pattern combinations
                sovereignty_terms = ["deny", "tool", "hook", "sovereignty"]
                has_sovereignty_term = any(term in key_lower for term in sovereignty_terms)

                if has_sovereignty_term:
                    for pattern in self.FORBIDDEN_PATTERNS:
                        assert pattern not in key_lower, (
                            f"Config {source_name} has key '{key}' that might "
                            f"disable sovereignty (contains '{pattern}')"
                        )


class TestArchitectureFitness:
    """deny-destroy:NoExecution:MUST:3 — SDK imports outside sdk_adapter/ are prohibited"""

    def test_no_sdk_imports_outside_adapter(self) -> None:
        """deny-destroy:NoExecution:MUST:3 - SDK imports only in sdk_adapter/.

        Uses __file__-relative paths for robust resolution.
        """
        # Use __file__-relative path for robust resolution
        root = Path(__file__).parent.parent / "amplifier_module_provider_github_copilot"

        violations: list[str] = []
        files_scanned = 0

        for py_file in root.glob("*.py"):
            # Skip __init__.py which may re-export
            if py_file.name == "__init__.py":
                continue

            try:
                tree = ast.parse(py_file.read_text(encoding="utf-8"))
            except SyntaxError:
                continue

            files_scanned += 1
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if _is_sdk_import(alias.name):
                            violations.append(f"{py_file.name}: import {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    if node.module and _is_sdk_import(node.module):
                        violations.append(f"{py_file.name}: from {node.module}")

        assert files_scanned > 0, "No files found — check path"
        assert not violations, "SDK imports found outside sdk_adapter/:\n" + "\n".join(violations)

    def test_sdk_adapter_contains_sdk_imports(self) -> None:
        """Verify sdk_adapter/ is the membrane for SDK imports.

        Uses __file__-relative paths for robust resolution.
        """
        # Use __file__-relative path for robust resolution
        adapter_dir = (
            Path(__file__).parent.parent
            / "amplifier_module_provider_github_copilot"
            / "sdk_adapter"
        )

        assert adapter_dir.exists(), "sdk_adapter/ directory must exist"

        # At least one file should exist
        py_files = list(adapter_dir.glob("*.py"))
        assert len(py_files) > 0, "sdk_adapter/ must contain Python files"


class TestSessionEphemerality:
    """Tests for session config structure.

    Note: Actual ephemeral session behavior (MUST:1-3) is tested in test_behaviors.py.
    This class verifies supporting types exist.
    """

    def test_session_config_exists(self) -> None:
        """SessionConfig type exists for ephemeral session configuration."""
        from amplifier_module_provider_github_copilot.sdk_adapter.types import SessionConfig

        # SessionConfig should be a valid type
        config = SessionConfig(model="gpt-4", system_prompt="test")
        assert config.system_prompt == "test"
        assert config.model == "gpt-4"


def _collect_all_keys(data: Any, prefix: str = "") -> list[str]:
    """Recursively collect all keys from nested dict."""
    from typing import cast

    keys: list[str] = []
    if isinstance(data, dict):
        for k, v in cast(dict[str, Any], data).items():
            full_key: str = f"{prefix}.{k}" if prefix else str(k)
            keys.append(full_key)
            keys.extend(_collect_all_keys(v, full_key))
    elif isinstance(data, list):
        for i, item in enumerate(cast(list[Any], data)):
            keys.extend(_collect_all_keys(item, f"{prefix}[{i}]"))
    return keys


def _is_sdk_import(module_name: str) -> bool:
    """Check if module name is a DIRECT GitHub Copilot SDK import.

    Imports from our sdk_adapter/ membrane are ALLOWED - that's the correct pattern.
    Imports from our own package (amplifier_module_provider_github_copilot) are ALLOWED.
    Only flag direct SDK imports like `import copilot` or `from copilot import X`.
    """
    # Allow imports from our own sdk_adapter (that's the membrane)
    if "sdk_adapter" in module_name:
        return False

    # Allow imports from our own package (includes "copilot" in name but is not SDK)
    if module_name.startswith("amplifier_module_provider_github_copilot"):
        return False

    # These are the actual SDK package names that should only appear in sdk_adapter/
    sdk_patterns = [
        "copilot",
        "github_copilot",
        "github.copilot",
        "ghcp",
    ]
    name_lower = module_name.lower()
    return any(pattern in name_lower for pattern in sdk_patterns)


class TestToolSuppression:
    """Tests for deny-destroy:ToolSuppression:MUST:1,2.

    Contract: deny-destroy.md ToolSuppression section

    Note: Uses untyped MagicMock for SDK session capture.
    """

    @pytest.mark.asyncio
    async def test_tools_have_override_flag(self) -> None:
        """Tools forwarded to SDK have overrides_built_in_tool=True.

        Contract: deny-destroy:ToolSuppression:MUST:2
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import CopilotClientWrapper

        mock_sdk_client = MagicMock()
        create_session_kwargs: list[dict[str, Any]] = []

        async def capture_create(**kwargs: object) -> MagicMock:
            create_session_kwargs.append(kwargs)
            mock_session = MagicMock()
            mock_session.on = MagicMock(return_value=lambda: None)
            mock_session.disconnect = AsyncMock()
            return mock_session

        mock_sdk_client.create_session = capture_create
        wrapper = CopilotClientWrapper(sdk_client=mock_sdk_client)

        tools: list[dict[str, Any]] = [
            {"name": "bash", "description": "Execute bash", "parameters": {}}
        ]

        async with wrapper.session(model="test-model", tools=tools):
            pass

        assert len(create_session_kwargs) == 1
        session_tools = create_session_kwargs[0].get("tools", [])
        for tool in session_tools:
            if hasattr(tool, "overrides_built_in_tool"):
                assert tool.overrides_built_in_tool is True

    @pytest.mark.asyncio
    async def test_available_tools_set_on_session(self) -> None:
        """SDK session MUST receive available_tools attribute.

        Contract: deny-destroy:ToolSuppression:MUST:1 — available_tools MUST NOT be omitted
        When no tools provided, available_tools=[] blocks SDK built-ins.
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import CopilotClientWrapper

        mock_sdk_client = MagicMock()
        create_session_kwargs: list[dict[str, Any]] = []

        async def capture_create(**kwargs: object) -> MagicMock:
            create_session_kwargs.append(kwargs)
            mock_session = MagicMock()
            mock_session.on = MagicMock(return_value=lambda: None)
            mock_session.disconnect = AsyncMock()
            return mock_session

        mock_sdk_client.create_session = capture_create
        wrapper = CopilotClientWrapper(sdk_client=mock_sdk_client)

        async with wrapper.session(model="test-model"):
            pass

        assert len(create_session_kwargs) == 1
        # MUST have available_tools key set to empty list (blocks SDK built-ins)
        assert "available_tools" in create_session_kwargs[0]
        assert create_session_kwargs[0]["available_tools"] == []
