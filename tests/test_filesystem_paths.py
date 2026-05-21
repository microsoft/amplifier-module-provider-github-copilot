"""Identity, Paths, Portability anchors for filesystem-layout V1.0.

Contract: contracts/filesystem-layout.md
"""

from __future__ import annotations

import ast
import dataclasses
import sys
import tomllib
from pathlib import Path

import pytest

from amplifier_module_provider_github_copilot._identity import PROVIDER_ID
from amplifier_module_provider_github_copilot.config._paths import (
    PROVIDER_DISTRIBUTION_NAME,
    ProviderPaths,
    load_provider_paths,
)

PKG_ROOT = Path(__file__).parent.parent / "amplifier_module_provider_github_copilot"
PKG_NAME = "amplifier_module_provider_github_copilot"
ALLOWED_PROVIDER_ID_FILES = {"config/_models.py"}
ALLOWED_DISTRIBUTION_NAME_FILES = {"config/_paths.py"}


def _iter_py_files() -> list[Path]:
    return sorted(p for p in PKG_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def _docstring_ids(tree: ast.AST) -> set[int]:
    out: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                out.add(id(body[0].value))
    return out


def _find_literal_sites(literal: str, allow_rel: set[str]) -> list[str]:
    violations: list[str] = []
    for py in _iter_py_files():
        rel = py.relative_to(PKG_ROOT).as_posix()
        if rel in allow_rel:
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        docs = _docstring_ids(tree)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value == literal
                and id(node) not in docs
            ):
                violations.append(f"{rel}:{node.lineno}")
    return violations


# Contract: filesystem-layout:Identity:MUST:1
def test_provider_id_equals_name_equals_get_info_id() -> None:
    from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider as Provider

    p = Provider.__new__(Provider)
    # Bypass __init__ for the property check (no SDK / coordinator needed).
    assert Provider.name.fget(p) == PROVIDER_ID, (  # type: ignore[union-attr]
        "Provider.name property must return _identity.PROVIDER_ID"
    )
    # get_info needs a real instance — construct via __init__ (no SDK call yet).
    real = Provider()
    assert real.get_info().id == PROVIDER_ID, (
        f"Provider.get_info().id MUST equal PROVIDER_ID; got {real.get_info().id!r}"
    )


# Contract: filesystem-layout:Identity:MUST:1 (AST scan — supplements behavioral test above)
def test_provider_id_literal_appears_only_in_provider_config() -> None:
    sites = _find_literal_sites(PROVIDER_ID, ALLOWED_PROVIDER_ID_FILES)
    assert sites == [], (
        f"PROVIDER_ID literal {PROVIDER_ID!r} duplicated outside config/_models.py:\n"
        + "\n".join(sites)
    )


# Contract: filesystem-layout:Identity:MUST:2 (AST scan)
def test_distribution_name_literal_appears_only_in_paths_module() -> None:
    sites = _find_literal_sites(PROVIDER_DISTRIBUTION_NAME, ALLOWED_DISTRIBUTION_NAME_FILES)
    assert sites == [], (
        "PROVIDER_DISTRIBUTION_NAME literal duplicated outside config/_paths.py:\n"
        + "\n".join(sites)
    )


# Contract: filesystem-layout:Identity:MUST:3 — behavioural derivation
def test_env_var_dotdir_xdg_subdir_all_derived_from_distribution_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify resolver derives the env var, dotdir, and XDG subdir names
    mechanically from PROVIDER_DISTRIBUTION_NAME (no separate literals).
    """
    expected_env = f"{PROVIDER_DISTRIBUTION_NAME.upper().replace('-', '_')}_HOME"
    expected_dotdir = f".{PROVIDER_DISTRIBUTION_NAME}"

    # (1) The derived env var name must drive the override path.
    for k in (expected_env, "XDG_DATA_HOME", "XDG_CACHE_HOME", "LOCALAPPDATA"):
        monkeypatch.delenv(k, raising=False)
    sentinel = tmp_path / "id-must-3-sentinel"
    monkeypatch.setenv(expected_env, str(sentinel))
    paths = load_provider_paths()
    assert paths.provider_home == sentinel, (
        f"derived env var name {expected_env!r} must drive override resolution"
    )

    # (2) With no env, the dotdir form must be the default.
    monkeypatch.delenv(expected_env, raising=False)
    paths2 = load_provider_paths()
    assert paths2.provider_home == Path.home() / expected_dotdir, (
        f"default fallback must be ~/{expected_dotdir}"
    )

    # (3) XDG subdir name equals the distribution name verbatim.
    xdg_root = tmp_path / "xdg-root"
    monkeypatch.setenv("XDG_DATA_HOME", str(xdg_root))
    paths3 = load_provider_paths()
    assert paths3.provider_home == xdg_root / PROVIDER_DISTRIBUTION_NAME, (
        "XDG subdir must equal PROVIDER_DISTRIBUTION_NAME verbatim"
    )


# ---------- Paths:MUST:1 — provider_home resolution ----------

ENV_VAR_NAME = "AMPLIFIER_PROVIDER_GITHUB_COPILOT_HOME"


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Strip every env var the resolver might inspect."""
    for k in (
        ENV_VAR_NAME,
        "XDG_DATA_HOME",
        "XDG_CACHE_HOME",
        "LOCALAPPDATA",
        "AMPLIFIER_HOME",
        "AMPLIFIER_APP_CLI_HOME",
        "AMPLIFIER_DISTRO_HOME",
        "AMPLIFIER_RUNTIME_HOME",
        "AMPLIFIER_FOUNDATION_HOME",
        "COPILOT_HOME",
        "COPILOT_CLI_PATH",
    ):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


# Contract: filesystem-layout:Paths:MUST:1
@pytest.mark.parametrize(
    "row_id,env_value,expectation",
    [
        pytest.param("override-abs-wins", "ABS", "override", id="override-abs-wins"),
        pytest.param("override-empty-falls", "", "fallthrough", id="override-empty-falls"),
        pytest.param("override-ws-falls", "   \t  ", "fallthrough", id="override-ws-falls"),
        pytest.param(
            "override-relative-raises", "relative/path", "ValueError",
            id="override-relative-raises",
        ),
        pytest.param(
            "override-tilde-expands", "~/sentinel-tilde", "tilde",
            id="override-tilde-expands",
        ),
    ],
)
def test_provider_home_resolves_per_precedence_override(
    clean_env: pytest.MonkeyPatch, tmp_path: Path, row_id: str, env_value: str, expectation: str
) -> None:
    abs_value = str(tmp_path / "override-target") if env_value == "ABS" else env_value
    clean_env.setenv(ENV_VAR_NAME, abs_value)
    if expectation == "ValueError":
        with pytest.raises(ValueError, match="absolute"):
            load_provider_paths()
        return
    paths = load_provider_paths()
    if expectation == "override":
        assert paths.provider_home == Path(abs_value), f"row {row_id}: override should win"
    elif expectation == "tilde":
        assert paths.provider_home == Path(env_value).expanduser(), (
            f"row {row_id}: tilde must expand"
        )
    elif expectation == "fallthrough":
        # Must NOT equal the empty/whitespace value
        assert str(paths.provider_home) != env_value, f"row {row_id}: empty/ws must fall through"
        assert paths.provider_home.is_absolute()


# Contract: filesystem-layout:Paths:MUST:1 (XDG branch)
@pytest.mark.parametrize(
    "row_id,xdg_value,expectation",
    [
        pytest.param("xdg-abs-wins", "ABS", "xdg", id="xdg-abs-wins"),
        pytest.param("xdg-empty-falls", "", "fallthrough", id="xdg-empty-falls"),
        pytest.param("xdg-ws-falls", "   ", "fallthrough", id="xdg-ws-falls"),
        pytest.param("xdg-relative-raises", "data/xdg", "ValueError", id="xdg-relative-raises"),
        pytest.param("xdg-tilde-expands", "~/xdg-tilde", "tilde", id="xdg-tilde-expands"),
    ],
)
def test_provider_home_resolves_per_precedence_xdg(
    clean_env: pytest.MonkeyPatch, tmp_path: Path, row_id: str, xdg_value: str, expectation: str
) -> None:
    abs_value = str(tmp_path / "xdg-data") if xdg_value == "ABS" else xdg_value
    clean_env.setenv("XDG_DATA_HOME", abs_value)
    if expectation == "ValueError":
        with pytest.raises(ValueError, match="absolute"):
            load_provider_paths()
        return
    paths = load_provider_paths()
    if expectation == "xdg":
        assert paths.provider_home == Path(abs_value) / PROVIDER_DISTRIBUTION_NAME
    elif expectation == "tilde":
        assert paths.provider_home == Path(xdg_value).expanduser() / PROVIDER_DISTRIBUTION_NAME
    elif expectation == "fallthrough":
        # Must equal default ~/.<distribution-name>
        assert paths.provider_home == Path.home() / f".{PROVIDER_DISTRIBUTION_NAME}"


# Contract: filesystem-layout:Paths:MUST:1 (default fallback)
def test_provider_home_all_unset_default(clean_env: pytest.MonkeyPatch) -> None:
    paths = load_provider_paths()
    assert paths.provider_home == Path.home() / f".{PROVIDER_DISTRIBUTION_NAME}"


# Contract: filesystem-layout:Paths:MUST:3
def test_provider_home_and_cache_home_are_disjoint(clean_env: pytest.MonkeyPatch) -> None:
    paths = load_provider_paths()
    assert paths.provider_home != paths.cache_home, "homes must not be equal"
    try:
        paths.provider_home.relative_to(paths.cache_home)
        pytest.fail("provider_home must not be contained in cache_home")
    except ValueError:
        pass
    try:
        paths.cache_home.relative_to(paths.provider_home)
        pytest.fail("cache_home must not be contained in provider_home")
    except ValueError:
        pass


# Contract: filesystem-layout:Paths:MUST:3 — adversarial env causing overlap
@pytest.mark.parametrize(
    ("provider_env", "xdg_cache_env", "localappdata_env", "shape"),
    [
        # cache_home (XDG_CACHE_HOME/<dist>) nests inside provider_home
        ("/tmp/v2amp-overlap", "/tmp/v2amp-overlap", None, "nested-cache-under-provider"),
        # provider_home equals cache_home exactly (override + XDG_CACHE_HOME tail-named dist)
        (
            f"/tmp/v2amp-equal/{PROVIDER_DISTRIBUTION_NAME}",
            "/tmp/v2amp-equal",
            None,
            "exactly-equal",
        ),
        # provider_home nests inside cache_home (XDG path is parent of provider override)
        (
            f"/tmp/v2amp-pin/{PROVIDER_DISTRIBUTION_NAME}/inner",
            "/tmp/v2amp-pin",
            None,
            "provider-under-cache",
        ),
    ],
    ids=["nested-cache", "exactly-equal", "nested-provider"],
)
def test_overlapping_env_raises_value_error(
    provider_env: str,
    xdg_cache_env: str,
    localappdata_env: str | None,
    shape: str,
    clean_env: pytest.MonkeyPatch,
) -> None:
    """Adversarial env producing overlap MUST raise ValueError, not silently
    chmod overlapping subtrees.

    Without the runtime guard, a host setting both
    `AMPLIFIER_PROVIDER_GITHUB_COPILOT_HOME` and `XDG_CACHE_HOME` (or
    `LOCALAPPDATA`) to the same root or to nested roots would let the
    SDK's session/auth state and the provider's regenerable cache write
    into a shared subtree — collapsing the data/cache separation
    promised by `Paths:MUST:3`.
    """
    if sys.platform == "win32":
        pytest.skip("Adversarial overlap test uses POSIX-style /tmp paths")
    clean_env.setenv("AMPLIFIER_PROVIDER_GITHUB_COPILOT_HOME", provider_env)
    clean_env.setenv("XDG_CACHE_HOME", xdg_cache_env)
    if localappdata_env is not None:
        clean_env.setenv("LOCALAPPDATA", localappdata_env)
    with pytest.raises(ValueError, match=r"(disjoint|contained|MUST:3)"):
        load_provider_paths()


# Contract: filesystem-layout:Paths:MUST:5
def test_provider_paths_is_immutable(tmp_path: Path) -> None:
    pp = ProviderPaths(provider_home=tmp_path / "a", cache_home=tmp_path / "b")
    with pytest.raises(dataclasses.FrozenInstanceError):
        pp.provider_home = tmp_path / "c"  # type: ignore[misc]


# Contract: filesystem-layout:Portability:MUST:1 (AST scan over _paths.py)
def test_paths_module_imports_only_stdlib() -> None:
    paths_py = PKG_ROOT / "config" / "_paths.py"
    tree = ast.parse(paths_py.read_text(encoding="utf-8"))
    stdlib = set(sys.stdlib_module_names) | {"__future__"}
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in stdlib:
                    bad.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            if node.level == 0 and mod not in stdlib:
                bad.append(f"from {node.module} import ...")
    assert bad == [], f"_paths.py imports non-stdlib symbols: {bad}"


# Contract: filesystem-layout:Portability:MUST:2
def test_pyproject_dependencies_have_no_host_packages_at_runtime() -> None:
    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    deps = data["project"]["dependencies"]
    forbidden_prefixes = (
        "amplifier-core", "amplifier-cli", "amplifier-distro", "amplifier-runtime",
    )
    bad = [d for d in deps if any(d.startswith(p) for p in forbidden_prefixes)]
    assert bad == [], f"Runtime deps must not include host packages: {bad}"
