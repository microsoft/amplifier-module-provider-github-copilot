"""Provider path resolution.

Contract: contracts/filesystem-layout.md (V1.0)

Single source of truth for `provider_home` (XDG-DATA: persistent state)
and `cache_home` (XDG-CACHE: regenerable artifacts). Stdlib-only so the
provider loads identically under every host distribution.
"""

from __future__ import annotations

import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path

# Identity:MUST:2 — filesystem-facing identity, defined exactly once.
PROVIDER_DISTRIBUTION_NAME: str = "amplifier-provider-github-copilot"

# Identity:MUST:3 — mechanical derivations.
_ENV_OVERRIDE: str = f"{PROVIDER_DISTRIBUTION_NAME.upper().replace('-', '_')}_HOME"
_DOT_DIR_NAME: str = f".{PROVIDER_DISTRIBUTION_NAME}"


@dataclass(frozen=True)
class ProviderPaths:
    """Resolved on-disk locations owned by the provider.

    Contract: filesystem-layout:Paths:MUST:5 (frozen).

    Note: Instances built through `load_provider_paths()` are
    guaranteed disjoint (`_enforce_disjoint` runs on the env-resolution
    path). Callers constructing `ProviderPaths` directly — i.e. host
    wiring via the `Isolation:MUST:3` injection escape hatch — own the
    disjointness invariant themselves; `_enforce_disjoint` is not
    invoked on the injection path by design.
    """

    provider_home: Path
    cache_home: Path


def _resolve_env_value(raw: str | None) -> str | None:
    """Return stripped value if non-empty after strip, else None."""
    if raw is None:
        return None
    stripped = raw.strip()
    return stripped if stripped else None


def _resolve_absolute(value: str, env_name: str) -> Path:
    """Expand `~` if present, then enforce absoluteness.

    Raises ValueError per Paths:MUST:1 / MUST:2 fail-closed rule.
    """
    expanded = Path(value).expanduser()
    if not expanded.is_absolute():
        raise ValueError(
            f"{env_name}={value!r} must be absolute after Path.expanduser(); "
            f"got {expanded!s}"
        )
    return expanded


def _resolve_provider_home() -> Path:
    """Contract: filesystem-layout:Paths:MUST:1."""
    override = _resolve_env_value(os.environ.get(_ENV_OVERRIDE))
    if override is not None:
        return _resolve_absolute(override, _ENV_OVERRIDE)

    xdg_data = _resolve_env_value(os.environ.get("XDG_DATA_HOME"))
    if xdg_data is not None:
        base = _resolve_absolute(xdg_data, "XDG_DATA_HOME")
        return base / PROVIDER_DISTRIBUTION_NAME

    return Path.home() / _DOT_DIR_NAME


def _resolve_cache_home() -> Path:
    """Contract: filesystem-layout:Paths:MUST:2."""
    xdg_cache = _resolve_env_value(os.environ.get("XDG_CACHE_HOME"))
    if xdg_cache is not None:
        base = _resolve_absolute(xdg_cache, "XDG_CACHE_HOME")
        return base / PROVIDER_DISTRIBUTION_NAME

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / PROVIDER_DISTRIBUTION_NAME

    if sys.platform == "win32":
        localappdata = _resolve_env_value(os.environ.get("LOCALAPPDATA"))
        if localappdata is not None:
            base = _resolve_absolute(localappdata, "LOCALAPPDATA")
            return base / PROVIDER_DISTRIBUTION_NAME / "Cache"
        return Path.home() / "AppData" / "Local" / PROVIDER_DISTRIBUTION_NAME / "Cache"

    return Path.home() / ".cache" / PROVIDER_DISTRIBUTION_NAME


def load_provider_paths() -> ProviderPaths:
    """Resolve `provider_home` and `cache_home` per V1.0 contract.

    Contract: filesystem-layout:Paths:MUST:1, MUST:2, MUST:3;
    filesystem-layout:Wiring:MUST:3 (uncached — env reads on every call).

    Raises ValueError when the resolved paths violate Paths:MUST:3
    (equal or one contained in the other) — typically because the
    operator set overlapping `AMPLIFIER_PROVIDER_GITHUB_COPILOT_HOME`
    and `XDG_CACHE_HOME` / `LOCALAPPDATA` values.
    """
    provider_home = _resolve_provider_home()
    cache_home = _resolve_cache_home()
    _enforce_disjoint(provider_home, cache_home)
    return ProviderPaths(provider_home=provider_home, cache_home=cache_home)


def _enforce_disjoint(provider_home: Path, cache_home: Path) -> None:
    """Reject overlap between `provider_home` and `cache_home`.

    Contract: filesystem-layout:Paths:MUST:3.
    """
    if provider_home == cache_home:
        raise ValueError(
            f"provider_home and cache_home must be disjoint but resolved "
            f"to the same path {provider_home!s}; check "
            f"AMPLIFIER_PROVIDER_GITHUB_COPILOT_HOME, XDG_DATA_HOME, "
            f"XDG_CACHE_HOME, and LOCALAPPDATA env values."
        )
    try:
        cache_home.relative_to(provider_home)
    except ValueError:
        pass
    else:
        raise ValueError(
            f"cache_home {cache_home!s} must not be contained within "
            f"provider_home {provider_home!s} (filesystem-layout:Paths:MUST:3)."
        )
    try:
        provider_home.relative_to(cache_home)
    except ValueError:
        pass
    else:
        raise ValueError(
            f"provider_home {provider_home!s} must not be contained within "
            f"cache_home {cache_home!s} (filesystem-layout:Paths:MUST:3)."
        )


def _create_one(leaf: Path) -> None:
    """Materialize a single leaf with the contract's lifecycle semantics."""
    leaf.mkdir(parents=True, exist_ok=True)

    if sys.platform == "win32":
        # Lifecycle:MUST:2 — mkdir-only; no mode, no chmod.
        return

    # Lifecycle:MUST:1 — POSIX/macOS: refuse symlinks then chmod 0o700.
    st = os.lstat(leaf)
    if stat.S_ISLNK(st.st_mode):
        raise OSError(
            f"refusing to use symbolic link at {leaf!s}: "
            f"provider directories must be regular directories"
        )
    os.chmod(leaf, 0o700)


def ensure_paths_exist(paths: ProviderPaths) -> None:
    """Materialize `provider_home` and `cache_home` lazily and idempotently.

    Contract: filesystem-layout:Lifecycle:MUST:1 (POSIX/macOS chmod 0o700 +
    symlink refuse), MUST:2 (Windows mkdir-only), MUST:3 (collision chain).
    """
    _create_one(paths.provider_home)
    _create_one(paths.cache_home)
