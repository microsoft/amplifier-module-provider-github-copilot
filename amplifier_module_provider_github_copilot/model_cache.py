"""Model cache for disk persistence.

Contract: contracts/behaviors.md (ModelCache section)

TTL policy values come from config/policy.py (CacheConfig dataclass).
Philosophy: Fail clearly rather than fail silently with stale data.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from .config._policy import load_cache_config

if TYPE_CHECKING:
    from .models import CopilotModelInfo

logger = logging.getLogger(__name__)

# Supported cache schema version (S8 Fix: version was written but never checked on read).
# Old caches without a version field are treated as "1.0" (backward compat).
_SUPPORTED_CACHE_VERSION = "1.0"


def get_cache_ttl_seconds() -> int:
    """Get cache TTL in seconds from config.

    Contract: behaviors:ModelCache:SHOULD:2
    """
    return load_cache_config().disk_ttl_seconds


def get_cache_filename() -> str:
    """Get cache filename from config."""
    return load_cache_config().cache_filename


# =============================================================================
# Cross-Platform Cache Directory
# Contract: behaviors:ModelCache:SHOULD:1, Cross-platform requirements
# =============================================================================


def get_cache_dir() -> Path:
    """Get cross-platform cache directory.

    Follows platform conventions:
    - Windows: %LOCALAPPDATA%/amplifier/provider-github-copilot/
    - macOS: ~/Library/Caches/amplifier/provider-github-copilot/
    - Linux: $XDG_CACHE_HOME/amplifier/provider-github-copilot/ or ~/.cache/...

    Contract: behaviors:ModelCache:SHOULD:1

    Returns:
        Path to cache directory (may not exist yet).
    """
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:  # Linux/BSD
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))

    return base / "amplifier" / "provider-github-copilot"


def get_cache_file_path() -> Path:
    """Get full path to cache file."""
    return get_cache_dir() / get_cache_filename()


# =============================================================================
# Write Cache
# Contract: behaviors:ModelCache:SHOULD:1
# =============================================================================


def write_cache(
    models: list[CopilotModelInfo],
    cache_file: Path | None = None,
) -> None:
    """Write models to disk cache.

    Contract: behaviors:ModelCache:SHOULD:1
    - SHOULD cache SDK models to disk for session persistence

    Args:
        models: List of CopilotModelInfo to cache.
        cache_file: Optional path override (for testing). Uses default if None.
    """
    if cache_file is None:
        cache_file = get_cache_file_path()

    # Create parent directories if needed
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    # Build cache data structure
    cache_data = {
        "version": "1.0",
        "timestamp": time.time(),
        "models": [
            {
                "id": m.id,
                "name": m.name,
                "context_window": m.context_window,
                "max_output_tokens": m.max_output_tokens,
                "supports_vision": m.supports_vision,
                "supports_reasoning_effort": m.supports_reasoning_effort,
                "supported_reasoning_efforts": list(m.supported_reasoning_efforts),
                "default_reasoning_effort": m.default_reasoning_effort,
            }
            for m in models
        ],
    }

    # Write atomically: write to temp file, then rename
    # Contract: Cross-platform requirements - UTF-8 encoding
    temp_file = cache_file.with_suffix(".tmp")
    try:
        temp_file.write_text(
            json.dumps(cache_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temp_file.replace(cache_file)
        logger.debug("Cached %d models to %s", len(models), cache_file)
    except Exception as e:
        from .security_redaction import redact_sensitive_text

        logger.warning("Failed to write cache: %s", redact_sensitive_text(e))
        # Clean up temp file if it exists
        if temp_file.exists():
            try:
                temp_file.unlink()
            except Exception:
                pass


# =============================================================================
# Read Cache
# Contract: behaviors:ModelCache:SHOULD:1, SHOULD:2
# =============================================================================


def read_cache(
    cache_file: Path | None = None,
    max_age_seconds: int | None = None,
) -> list[CopilotModelInfo] | None:
    """Read models from disk cache.

    Contract: behaviors:ModelCache:SHOULD:1, SHOULD:2
    - SHOULD cache SDK models to disk for session persistence
    - SHOULD respect TTL from config/policy.py (CacheConfig)

    Args:
        cache_file: Optional path override (for testing). Uses default if None.
        max_age_seconds: Optional TTL override. Uses config value if None.

    Returns:
        List of CopilotModelInfo if cache valid, None otherwise.
    """
    # Import here to avoid circular import
    from .models import CopilotModelInfo

    if cache_file is None:
        cache_file = get_cache_file_path()

    if not cache_file.exists():
        logger.debug("Cache file not found: %s", cache_file)
        return None

    try:
        content = cache_file.read_text(encoding="utf-8")
        data = json.loads(content)
    except (json.JSONDecodeError, OSError) as e:
        from .security_redaction import redact_sensitive_text

        logger.warning("Failed to read cache: %s", redact_sensitive_text(e))
        return None

    # S8: Validate cache schema version before parsing.
    # Missing version treated as "1.0" (backward compat for pre-version caches).
    # Mismatched version forces cache miss to avoid parsing unknown schemas.
    cache_version = data.get("version", _SUPPORTED_CACHE_VERSION)
    if cache_version != _SUPPORTED_CACHE_VERSION:
        logger.debug(
            "Cache version %r unsupported (expected %r); ignoring",
            cache_version,
            _SUPPORTED_CACHE_VERSION,
        )
        return None

    # Check timestamp / TTL
    if max_age_seconds is None:
        max_age_seconds = get_cache_ttl_seconds()

    # P1 Fix: Handle null timestamp. dict.get() returns None if key exists with null value.
    # Using `or 0` handles both missing key and explicit null.
    timestamp = data.get("timestamp") or 0
    age = time.time() - timestamp

    if age > max_age_seconds:
        logger.debug("Cache stale: age=%.0f seconds, max=%d", age, max_age_seconds)
        return None

    # Parse models per-entry: preserves valid entries even when some are malformed.
    # S5 Fix: list comprehension raised on first bad entry, discarding all valid entries.
    raw_models = data.get("models", [])
    models: list[CopilotModelInfo] = []
    for idx, m in enumerate(raw_models):
        try:
            models.append(
                CopilotModelInfo(
                    id=m["id"],
                    name=m["name"],
                    context_window=m["context_window"],
                    max_output_tokens=m["max_output_tokens"],
                    supports_vision=m.get("supports_vision", False),
                    supports_reasoning_effort=m.get("supports_reasoning_effort", False),
                    supported_reasoning_efforts=tuple(m.get("supported_reasoning_efforts", [])),
                    default_reasoning_effort=m.get("default_reasoning_effort"),
                )
            )
        except (KeyError, TypeError) as e:
            from .security_redaction import redact_sensitive_text

            logger.warning(
                "Cache entry %d malformed (%s); skipping",
                idx,
                redact_sensitive_text(e),
            )
    if not models and raw_models:
        # All entries malformed — treat as cache miss so live API is called.
        logger.debug("All %d cache entries invalid; forcing live API call", len(raw_models))
        return None
    logger.debug(
        "Read %d models from cache (%d skipped)",
        len(models),
        len(raw_models) - len(models),
    )
    return models


# =============================================================================
# Cache Operations for Provider
# =============================================================================


def invalidate_cache(cache_file: Path | None = None) -> None:
    """Remove cache file.

    Args:
        cache_file: Optional path override (for testing).
    """
    if cache_file is None:
        cache_file = get_cache_file_path()

    if cache_file.exists():
        try:
            cache_file.unlink()
            logger.debug("Cache invalidated: %s", cache_file)
        except Exception as e:
            from .security_redaction import redact_sensitive_text

            logger.warning("Failed to invalidate cache: %s", redact_sensitive_text(e))
