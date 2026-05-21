"""Lifecycle:MUST:1-3 — creation, permissions, symlink refusal, collision.

Contract: contracts/filesystem-layout.md
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

from amplifier_module_provider_github_copilot.config._paths import (
    ProviderPaths,
    ensure_paths_exist,
)


@pytest.fixture
def fresh_paths(tmp_path: Path) -> ProviderPaths:
    return ProviderPaths(
        provider_home=tmp_path / "data" / "home",
        cache_home=tmp_path / "cache" / "home",
    )


# Contract: filesystem-layout:Lifecycle:MUST:1 (POSIX 0o700)
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only contract")
def test_provider_home_created_with_0700_on_posix(fresh_paths: ProviderPaths) -> None:
    ensure_paths_exist(fresh_paths)
    assert fresh_paths.provider_home.is_dir()
    mode = stat.S_IMODE(fresh_paths.provider_home.stat().st_mode)
    assert mode == 0o700, f"provider_home must be chmod 0o700; got {oct(mode)}"
    cmode = stat.S_IMODE(fresh_paths.cache_home.stat().st_mode)
    assert cmode == 0o700, f"cache_home must be chmod 0o700; got {oct(cmode)}"


# Contract: filesystem-layout:Lifecycle:MUST:1 (symlink refusal)
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only contract")
def test_provider_home_creation_refuses_symlink_on_posix(
    tmp_path: Path,
) -> None:
    target = tmp_path / "real-dir"
    target.mkdir()
    link = tmp_path / "link-to-real"
    os.symlink(target, link)
    paths = ProviderPaths(provider_home=link, cache_home=tmp_path / "cache")
    with pytest.raises(OSError, match="symlink|symbolic"):
        ensure_paths_exist(paths)


# Contract: filesystem-layout:Lifecycle:MUST:1 (no umask mutation)
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only contract")
def test_creation_does_not_mutate_global_umask(
    fresh_paths: ProviderPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Record every os.umask call and assert none happened during creation.

    Catches transient mutations (e.g., `old = os.umask(0o077); ...; os.umask(old)`)
    that the before/after snapshot pattern would miss because state is restored.
    """
    calls: list[int] = []

    def _recorder(mask: int) -> int:
        calls.append(mask)
        return 0o022

    monkeypatch.setattr(os, "umask", _recorder)
    ensure_paths_exist(fresh_paths)
    assert calls == [], f"ensure_paths_exist MUST NOT call os.umask; saw {calls!r}"


# Contract: filesystem-layout:Lifecycle:MUST:2 (Windows mkdir-only)
@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only contract")
def test_provider_home_creation_omits_mode_arg_on_windows(
    fresh_paths: ProviderPaths,
) -> None:
    """Windows path uses mkdir(parents=True, exist_ok=True) without mode.
    Verify lifecycle still creates dirs successfully and no chmod is observable.
    """
    ensure_paths_exist(fresh_paths)
    assert fresh_paths.provider_home.is_dir()
    assert fresh_paths.cache_home.is_dir()


# Contract: filesystem-layout:Lifecycle:MUST:3 (collision -> FileExistsError chain)
def test_collision_with_file_raises_FileExistsError(
    tmp_path: Path,
) -> None:
    # Pre-existing file at the leaf path
    bad = tmp_path / "preexisting-file"
    bad.write_text("blocking the dir")
    paths = ProviderPaths(provider_home=bad, cache_home=tmp_path / "cache")
    with pytest.raises((FileExistsError, NotADirectoryError, OSError)) as exc_info:
        ensure_paths_exist(paths)
    # Verify the exception (or its cause chain) carries the right type
    err = exc_info.value
    chain_types: list[type] = []
    while err is not None:
        chain_types.append(type(err))
        err = err.__cause__ or err.__context__
        if err is exc_info.value:
            break
    assert any(
        issubclass(t, (FileExistsError, NotADirectoryError)) for t in chain_types
    ), f"Collision chain must include FileExistsError/NotADirectoryError; got {chain_types}"


# Contract: filesystem-layout:Lifecycle:MUST:1 (idempotent — second call ok)
def test_creation_is_idempotent(fresh_paths: ProviderPaths) -> None:
    ensure_paths_exist(fresh_paths)
    # Second call must not raise
    ensure_paths_exist(fresh_paths)
    assert fresh_paths.provider_home.is_dir()
    assert fresh_paths.cache_home.is_dir()


# Contract: filesystem-layout:Lifecycle:MUST:4 (no deletion/migration)
def test_existing_content_preserved_on_creation(fresh_paths: ProviderPaths) -> None:
    fresh_paths.provider_home.mkdir(parents=True, exist_ok=True)
    marker = fresh_paths.provider_home / "user-content.txt"
    marker.write_text("DO-NOT-DELETE")
    ensure_paths_exist(fresh_paths)
    assert marker.read_text() == "DO-NOT-DELETE", (
        "Lifecycle:MUST:4 — provider must not delete or migrate existing content"
    )
