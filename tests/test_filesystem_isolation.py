"""Isolation + Wiring:MUST:1-3 anchors for filesystem-layout V1.0.

Contract: contracts/filesystem-layout.md
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from amplifier_module_provider_github_copilot.config._paths import (
    ProviderPaths,
    load_provider_paths,
)

PKG_ROOT = Path(__file__).parent.parent / "amplifier_module_provider_github_copilot"
ENV_VAR_NAME = "AMPLIFIER_PROVIDER_GITHUB_COPILOT_HOME"
FORBIDDEN_ENV_READS = {"COPILOT_HOME", "COPILOT_CLI_PATH"}


def _iter_py_files() -> list[Path]:
    return sorted(p for p in PKG_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for k in (
        ENV_VAR_NAME, "XDG_DATA_HOME", "XDG_CACHE_HOME", "LOCALAPPDATA",
        "AMPLIFIER_HOME", "AMPLIFIER_APP_CLI_HOME", "AMPLIFIER_DISTRO_HOME",
        "AMPLIFIER_RUNTIME_HOME", "AMPLIFIER_FOUNDATION_HOME",
        "COPILOT_HOME", "COPILOT_CLI_PATH",
    ):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


# Contract: filesystem-layout:Isolation:MUST:1 — behavioural sentinel
@pytest.mark.parametrize(
    "env_name",
    [
        "AMPLIFIER_HOME",
        "AMPLIFIER_APP_CLI_HOME",
        "AMPLIFIER_DISTRO_HOME",
        "AMPLIFIER_RUNTIME_HOME",
        "AMPLIFIER_FOUNDATION_HOME",
        "COPILOT_HOME",
        "COPILOT_CLI_PATH",
    ],
)
def test_host_env_sentinel_never_appears_in_resolved_paths(
    clean_env: pytest.MonkeyPatch, tmp_path: Path, env_name: str
) -> None:
    sentinel = str(tmp_path / "DO-NOT-USE-host-injection")
    clean_env.setenv(env_name, sentinel)
    paths = load_provider_paths()
    assert str(paths.provider_home) != sentinel, f"{env_name} sentinel leaked into provider_home"
    assert str(paths.cache_home) != sentinel, f"{env_name} sentinel leaked into cache_home"
    assert sentinel not in str(paths.provider_home), f"{env_name} substring leak in provider_home"
    assert sentinel not in str(paths.cache_home), f"{env_name} substring leak in cache_home"


# Contract: filesystem-layout:Isolation:MUST:1 — AST scan supplement
def test_no_forbidden_env_reads_in_package() -> None:
    """Belt-and-suspenders: any future regression that imports COPILOT_HOME
    or COPILOT_CLI_PATH via os.environ/os.getenv fails this scan.
    """
    bad: list[str] = []
    for py in _iter_py_files():
        # _paths.py owns the env reads it needs; other modules MUST NOT touch
        # COPILOT_HOME or COPILOT_CLI_PATH from os.environ.
        rel = py.relative_to(PKG_ROOT).as_posix()
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            # os.environ["X"] subscript
            if isinstance(node, ast.Subscript):
                if (
                    isinstance(node.value, ast.Attribute)
                    and isinstance(node.value.value, ast.Name)
                    and node.value.value.id == "os"
                    and node.value.attr == "environ"
                ):
                    sl = node.slice
                    if isinstance(sl, ast.Constant) and sl.value in FORBIDDEN_ENV_READS:
                        bad.append(f"{rel}:{node.lineno} os.environ[{sl.value!r}]")
            # os.environ.get("X") / os.getenv("X")
            if isinstance(node, ast.Call):
                func = node.func
                name = None
                if isinstance(func, ast.Attribute):
                    if (
                        isinstance(func.value, ast.Attribute)
                        and isinstance(func.value.value, ast.Name)
                        and func.value.value.id == "os"
                        and func.value.attr == "environ"
                        and func.attr == "get"
                    ):
                        name = "os.environ.get"
                    elif (
                        isinstance(func.value, ast.Name)
                        and func.value.id == "os"
                        and func.attr == "getenv"
                    ):
                        name = "os.getenv"
                if name and node.args:
                    first = node.args[0]
                    if isinstance(first, ast.Constant) and first.value in FORBIDDEN_ENV_READS:
                        bad.append(f"{rel}:{node.lineno} {name}({first.value!r})")
    assert bad == [], "Forbidden env reads found:\n" + "\n".join(bad)


# Contract: filesystem-layout:Isolation:MUST:2
def test_dot_amplifier_and_dot_copilot_never_probed_or_resolved(
    clean_env: pytest.MonkeyPatch,
) -> None:
    paths = load_provider_paths()
    bad_prefixes = (
        Path.home() / ".amplifier",
        Path.home() / ".copilot",
    )
    for bad in bad_prefixes:
        # Forbid equal or proper-child relationship — substring would falsely
        # flag the legitimate ~/.amplifier-provider-github-copilot dotdir.
        for resolved in (paths.provider_home, paths.cache_home):
            assert resolved != bad, f"path equals forbidden {bad!s}: {resolved!s}"
            try:
                resolved.relative_to(bad)
            except ValueError:
                continue
            pytest.fail(f"{resolved!s} is contained in forbidden prefix {bad!s}")
    # AST scan: no module references these forbidden paths as literals.
    forbidden_literals = (".amplifier/", ".copilot/", "~/.amplifier", "~/.copilot")
    found: list[str] = []
    for py in _iter_py_files():
        rel = py.relative_to(PKG_ROOT).as_posix()
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        # Walk only ast.Constant string nodes that are NOT docstrings
        docstrings: set[int] = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                body = getattr(n, "body", [])
                if (
                    body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                ):
                    docstrings.add(id(body[0].value))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in docstrings
                and any(f in node.value for f in forbidden_literals)
            ):
                found.append(f"{rel}:{node.lineno} {node.value!r}")
    assert found == [], (
        "Forbidden ~/.amplifier or ~/.copilot literal references:\n" + "\n".join(found)
    )


# Contract: filesystem-layout:Isolation:MUST:3
@pytest.mark.asyncio
async def test_constructor_injection_overrides_env_resolution(
    clean_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Host-injected ProviderPaths through CopilotClientWrapper MUST win
    over any env-based resolution and the wiring MUST use those values
    verbatim when constructing SubprocessConfig.
    """
    from unittest.mock import patch

    from amplifier_module_provider_github_copilot.sdk_adapter.client import (
        CopilotClientWrapper,
    )

    injected = ProviderPaths(
        provider_home=tmp_path / "injected-data",
        cache_home=tmp_path / "injected-cache",
    )
    # Sentinel env that, if read, would visibly point elsewhere.
    clean_env.setenv(ENV_VAR_NAME, str(tmp_path / "env-target"))

    cfg_calls: list[dict] = []

    class _FakeCfg:
        def __init__(self, **kwargs):
            cfg_calls.append(kwargs)

    class _FakeClient:
        def __init__(self, cfg):
            self.cfg = cfg

        async def start(self):
            raise RuntimeError("intentional — only probing SubprocessConfig kwargs")

    with patch(
        "amplifier_module_provider_github_copilot.sdk_adapter._imports.SubprocessConfig",
        _FakeCfg,
    ), patch(
        "amplifier_module_provider_github_copilot.sdk_adapter._imports.CopilotClient",
        _FakeClient,
    ):
        wrapper = CopilotClientWrapper(provider_paths=injected)
        try:
            await wrapper._ensure_client_initialized("isolation-must-3")
        except Exception:
            pass

    assert cfg_calls, (
        "Wiring must construct SubprocessConfig (injection-or-load path)"
    )
    assert cfg_calls[0]["copilot_home"] == str(injected.provider_home), (
        f"Injected provider_home MUST flow to copilot_home verbatim; "
        f"got {cfg_calls[0]['copilot_home']!r}, expected {injected.provider_home!s}"
    )


# Contract: filesystem-layout:Wiring:MUST:3
def test_load_provider_paths_is_uncached_across_env_change(
    clean_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No @lru_cache on load_provider_paths — env change between calls
    MUST be reflected in the next call.
    """
    first = tmp_path / "first-home"
    second = tmp_path / "second-home"
    clean_env.setenv(ENV_VAR_NAME, str(first))
    p1 = load_provider_paths()
    assert p1.provider_home == first

    clean_env.setenv(ENV_VAR_NAME, str(second))
    p2 = load_provider_paths()
    assert p2.provider_home == second, (
        "load_provider_paths must re-read env on every call (no lru_cache)"
    )
