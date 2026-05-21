"""Wiring:MUST:4 + cache_home (Paths:MUST:2) anchors.

Contract: contracts/filesystem-layout.md
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from amplifier_module_provider_github_copilot.config._paths import (
    PROVIDER_DISTRIBUTION_NAME,
    ProviderPaths,
    load_provider_paths,
)

PKG_ROOT = Path(__file__).parent.parent / "amplifier_module_provider_github_copilot"
ENV_VAR_NAME = "AMPLIFIER_PROVIDER_GITHUB_COPILOT_HOME"
FORBIDDEN_CACHE_ENV_KEYS = {"LOCALAPPDATA", "XDG_CACHE_HOME", "XDG_DATA_HOME"}


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for k in (
        ENV_VAR_NAME, "XDG_DATA_HOME", "XDG_CACHE_HOME", "LOCALAPPDATA",
        "COPILOT_HOME", "COPILOT_CLI_PATH",
    ):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


# ---------- Paths:MUST:2 — cache_home resolution matrix ----------


# Contract: filesystem-layout:Paths:MUST:2
def test_cache_home_xdg_absolute_wins(clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    clean_env.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))
    paths = load_provider_paths()
    assert paths.cache_home == tmp_path / "xdg-cache" / PROVIDER_DISTRIBUTION_NAME


# Contract: filesystem-layout:Paths:MUST:2 (empty falls through)
def test_cache_home_xdg_empty_falls_through(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("XDG_CACHE_HOME", "")
    paths = load_provider_paths()
    # Falls to platform default — never equals empty string
    assert str(paths.cache_home) != ""
    assert paths.cache_home.is_absolute()


# Contract: filesystem-layout:Paths:MUST:2 (whitespace falls through)
def test_cache_home_xdg_whitespace_falls_through(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("XDG_CACHE_HOME", "   ")
    paths = load_provider_paths()
    assert paths.cache_home.is_absolute()
    assert "   " not in str(paths.cache_home)


# Contract: filesystem-layout:Paths:MUST:2 + MUST:4 (tilde expands)
def test_cache_home_xdg_tilde_expands(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("XDG_CACHE_HOME", "~/xdg-cache-tilde")
    paths = load_provider_paths()
    assert paths.cache_home == Path("~/xdg-cache-tilde").expanduser() / PROVIDER_DISTRIBUTION_NAME


# Contract: filesystem-layout:Paths:MUST:2 (relative raises)
def test_cache_home_xdg_relative_raises(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("XDG_CACHE_HOME", "relative/cache")
    with pytest.raises(ValueError, match="absolute"):
        load_provider_paths()


# Contract: filesystem-layout:Paths:MUST:2 (Windows LOCALAPPDATA relative raises)
def test_cache_home_localappdata_relative_raises(
    clean_env: pytest.MonkeyPatch, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("sys.platform", "win32")
    clean_env.setenv("LOCALAPPDATA", "relative/path")
    with pytest.raises(ValueError, match="absolute"):
        load_provider_paths()


# Contract: filesystem-layout:Paths:MUST:2 (Windows LOCALAPPDATA absolute)
def test_cache_home_localappdata_absolute_wins(
    clean_env: pytest.MonkeyPatch, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("sys.platform", "win32")
    clean_env.setenv("LOCALAPPDATA", str(tmp_path / "AppData" / "Local"))
    paths = load_provider_paths()
    assert paths.cache_home == tmp_path / "AppData" / "Local" / PROVIDER_DISTRIBUTION_NAME / "Cache"


# Contract: filesystem-layout:Paths:MUST:2 (Windows LOCALAPPDATA unset)
def test_cache_home_localappdata_unset_appdata_local_fallback(
    clean_env: pytest.MonkeyPatch, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("sys.platform", "win32")
    paths = load_provider_paths()
    expected = Path.home() / "AppData" / "Local" / PROVIDER_DISTRIBUTION_NAME / "Cache"
    assert paths.cache_home == expected


# Contract: filesystem-layout:Paths:MUST:2 (darwin default)
def test_cache_home_darwin_default_library_caches(
    clean_env: pytest.MonkeyPatch, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("sys.platform", "darwin")
    paths = load_provider_paths()
    assert paths.cache_home == Path.home() / "Library" / "Caches" / PROVIDER_DISTRIBUTION_NAME


# Contract: filesystem-layout:Paths:MUST:2 (linux default)
def test_cache_home_linux_default_dot_cache(
    clean_env: pytest.MonkeyPatch, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("sys.platform", "linux")
    paths = load_provider_paths()
    assert paths.cache_home == Path.home() / ".cache" / PROVIDER_DISTRIBUTION_NAME


# ---------- Wiring:MUST:4 — model_cache binds to cache_home ----------


# Contract: filesystem-layout:Wiring:MUST:4
def test_models_cache_file_resolves_under_cache_home(
    clean_env: pytest.MonkeyPatch, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Sentinel monkeypatch: replace load_provider_paths to point cache_home
    at an absolute tmp sentinel; assert model_cache resolves its file under it.
    The test fails if model_cache recomputes precedence (synthesizes its own).
    """
    sentinel_cache = tmp_path / "SENTINEL-cache-home"
    sentinel = ProviderPaths(
        provider_home=tmp_path / "data",
        cache_home=sentinel_cache,
    )
    # Override at the binding site model_cache.get_cache_file_path actually
    # consults (it imports load_provider_paths at module load time —
    # model_cache.py:19 — so we must patch the symbol on that module). This
    # also overrides the autouse fixture's binding for this test.
    monkeypatch.setattr(
        "amplifier_module_provider_github_copilot.model_cache.load_provider_paths",
        lambda: sentinel,
    )
    from amplifier_module_provider_github_copilot.model_cache import get_cache_file_path
    result = get_cache_file_path()
    assert result.parent == sentinel_cache, (
        f"models_cache.json must resolve directly under cache_home; got {result}"
    )
    assert result.name == "models_cache.json", (
        f"Cache file must be named 'models_cache.json'; got {result.name}"
    )


# Contract: filesystem-layout:Wiring:MUST:4 (AST scan — no cache-base synthesis)
def test_no_cache_base_synthesis_outside_paths_module() -> None:
    bad: list[str] = []
    for py in PKG_ROOT.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        rel = py.relative_to(PKG_ROOT).as_posix()
        if rel == "config/_paths.py":
            continue  # this module owns precedence
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            # os.environ["LOCALAPPDATA" | "XDG_CACHE_HOME" | "XDG_DATA_HOME"]
            if isinstance(node, ast.Subscript):
                if (
                    isinstance(node.value, ast.Attribute)
                    and isinstance(node.value.value, ast.Name)
                    and node.value.value.id == "os"
                    and node.value.attr == "environ"
                    and isinstance(node.slice, ast.Constant)
                    and node.slice.value in FORBIDDEN_CACHE_ENV_KEYS
                ):
                    bad.append(f"{rel}:{node.lineno} os.environ[{node.slice.value!r}]")
            # os.environ.get(...) / os.getenv(...) with forbidden key
            if isinstance(node, ast.Call):
                fn = node.func
                is_environ_get = (
                    isinstance(fn, ast.Attribute) and fn.attr == "get"
                    and isinstance(fn.value, ast.Attribute) and fn.value.attr == "environ"
                    and isinstance(fn.value.value, ast.Name) and fn.value.value.id == "os"
                )
                is_getenv = (
                    isinstance(fn, ast.Attribute) and fn.attr == "getenv"
                    and isinstance(fn.value, ast.Name) and fn.value.id == "os"
                )
                if (is_environ_get or is_getenv) and node.args:
                    first = node.args[0]
                    if isinstance(first, ast.Constant) and first.value in FORBIDDEN_CACHE_ENV_KEYS:
                        bad.append(f"{rel}:{node.lineno} cache-env-read({first.value!r})")
            # sys.platform == "..." comparisons outside _paths.py
            if isinstance(node, ast.Compare):
                left = node.left
                if (
                    isinstance(left, ast.Attribute)
                    and isinstance(left.value, ast.Name)
                    and left.value.id == "sys"
                    and left.attr == "platform"
                ):
                    bad.append(
                        f"{rel}:{node.lineno} sys.platform comparison "
                        "(cache-base synthesis suspected)"
                    )
    # Wiring:MUST:4 contract anchor mandates BOTH:
    #   (1) no os.environ reads with cache-env keys outside _paths.py
    #   (2) no sys.platform comparisons outside _paths.py and _platform.py
    # _platform.py is allowlisted because the contract explicitly carves out
    # event-loop selection (see contracts/filesystem-layout.md §Wiring:MUST:4
    # acceptance note). Any other sys.platform compare suggests cache-base
    # synthesis bleeding back into client/provider code.
    allowlist_for_platform = {"_platform.py"}
    env_violations = [b for b in bad if "sys.platform" not in b]
    platform_violations = [
        b for b in bad
        if "sys.platform" in b
        and not any(b.startswith(f"{name}:") or f"\\{name}:" in b or f"/{name}:" in b
                    for name in allowlist_for_platform)
    ]
    assert env_violations == [], (
        "Cache-base env-key reads outside _paths.py:\n" + "\n".join(env_violations)
    )
    assert platform_violations == [], (
        "Forbidden sys.platform comparisons (cache-base synthesis suspected):\n"
        + "\n".join(platform_violations)
    )
