"""Model cache for disk persistence.

Contract: contracts/behaviors.md (ModelCache section)

Three-Medium Architecture:
- Python: Cache mechanism (this module)
- YAML: TTL policy values (config/model_cache.yaml)
- Markdown: Requirements (contracts/behaviors.md)

Philosophy: Fail clearly rather than fail silently with stale data.
No hardcoded fallback dicts — cache is transparent layer only.
"""

from __future__ import annotations

import functools
import importlib.resources
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from .models import CopilotModelInfo

logger = logging.getLogger(__name__)


# =============================================================================
# Cache Policy Loading (Three-Medium Architecture: YAML = policy)
# =============================================================================


@functools.lru_cache(maxsize=1)
def load_cache_config() -> dict[str, Any]:
    """Load cache policy from config/model_cache.yaml.

    Contract: behaviors:ModelCache:SHOULD:2
    Three-Medium Architecture: Policy values come from YAML

    Returns:
        Dictionary with cache policy settings.
    """
    try:
        # Try package resources first (installed wheel)
        files = importlib.resources.files("amplifier_module_provider_github_copilot")
        config_path = files.joinpath("config", "model_cache.yaml")
        content = config_path.read_text(encoding="utf-8")
        return yaml.safe_load(content)  # type: ignore[no-any-return]
    except Exception:
        # Fallback to file path (development)
        config_file = Path(__file__).parent / "config" / "model_cache.yaml"
        if config_file.exists():
            content = config_file.read_text(encoding="utf-8")
            return yaml.safe_load(content)  # type: ignore[no-any-return]

        # Return sensible defaults if config missing
        logger.warning("model_cache.yaml not found, using defaults")
        return {
            "cache": {
                "disk_ttl_seconds": 86400,
                "max_stale_seconds": 604800,
                "cache_filename": "models_cache.json",
            }
        }


def get_cache_ttl_seconds() -> int:
    """Get cache TTL in seconds from config.

    Contract: behaviors:ModelCache:SHOULD:2
    """
    config = load_cache_config()
    cache_config = config.get("cache", {})
    return int(cache_config.get("disk_ttl_seconds", 86400))


def get_cache_filename() -> str:
    """Get cache filename from config."""
    config = load_cache_config()
    cache_config = config.get("cache", {})
    return str(cache_config.get("cache_filename", "models_cache.json"))


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
    - SHOULD respect TTL from config/model_cache.yaml

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

    # Parse models
    try:
        models = [
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
            for m in data.get("models", [])
        ]
        logger.debug("Read %d models from cache", len(models))
        return models
    except (KeyError, TypeError) as e:
        from .security_redaction import redact_sensitive_text

        logger.warning("Invalid cache data: %s", redact_sensitive_text(e))
        return None


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
