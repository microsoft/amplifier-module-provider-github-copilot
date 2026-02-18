"""
Model info cache for the GitHub Copilot provider.

This module handles persistence of model metadata (context_window, max_output_tokens)
to enable instant startup without API calls. The cache is written during
`amplifier init` (via list_models()) and read during provider initialization.

Cross-platform Considerations:
- Uses pathlib.Path for all path handling
- Cache directory: ~/.amplifier/cache/
- UTF-8 encoding for all file I/O
- Atomic writes via temp file + rename

Error Handling:
- All errors are caught and logged — cache is best-effort
- Returns None on read failure, False on write failure
- Caller should fall back to BUNDLED_MODEL_LIMITS on cache miss

Target Platforms (tested):
- WSL (primary development)
- Linux
- macOS
- Windows
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ._constants import (
    CACHE_FILE_NAME,
    CACHE_FORMAT_VERSION,
    CACHE_STALE_DAYS,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Bundled Model Limits (Hardcoded Fallback)
# ═══════════════════════════════════════════════════════════════════════════════

# Known context window limits from SDK (as of 2026-02-16)
# These are fallback values used when disk cache is unavailable.
# Real values are fetched from SDK and cached in list_models().
# Values derived: max_output = max_context_window - max_prompt_tokens
#
# This dict serves as "cache tier 2" (bundled) when "tier 1" (disk) misses.
#
# ORDERING: Alphabetical by model ID (matches SDK list_models() order).
# SDK sorts lexicographically: 'c' < 'g' (gemini before gpt), '-' < '.' < digits.
# Example: claude-opus-4.6 < claude-opus-4.6-1m < claude-opus-4.6-fast
#          (because '-' ASCII 45 < 'f' ASCII 102, and '1' ASCII 49 < 'f')
BUNDLED_MODEL_LIMITS: dict[str, tuple[int, int]] = {
    # (context_window, max_output_tokens)
    "claude-haiku-4.5": (144000, 16000),
    "claude-opus-4.5": (200000, 32000),
    "claude-opus-4.6": (200000, 32000),
    "claude-opus-4.6-1m": (1000000, 64000),
    "claude-opus-4.6-fast": (200000, 32000),
    "claude-sonnet-4": (216000, 88000),
    "claude-sonnet-4.5": (200000, 32000),
    # NOTE: SDK returns max_output=0 for gemini - likely SDK bug.
    # Using 65536 as max_output_tokens to keep budget positive.
    "gemini-3-pro-preview": (128000, 65536),
    "gpt-4.1": (128000, 64000),
    "gpt-5": (400000, 272000),
    "gpt-5-mini": (264000, 136000),
    "gpt-5.1": (264000, 136000),
    "gpt-5.1-codex": (400000, 272000),
    "gpt-5.1-codex-max": (400000, 272000),
    "gpt-5.1-codex-mini": (400000, 272000),
    "gpt-5.2": (264000, 136000),
    "gpt-5.2-codex": (400000, 128000),
    "gpt-5.3-codex": (400000, 128000),
}


def get_fallback_limits(model_id: str) -> tuple[int, int] | None:
    """
    Get bundled context limits for a model.

    This is the "tier 2" fallback when disk cache is unavailable.
    Returns hardcoded limits derived from SDK data at build time.

    Args:
        model_id: Model identifier (e.g., "claude-opus-4.5")

    Returns:
        Tuple of (context_window, max_output_tokens) if model is known,
        None if model is not in bundled limits.

    Example:
        >>> get_fallback_limits("claude-opus-4.5")
        (200000, 32000)
        >>> get_fallback_limits("unknown-model")
        None
    """
    return BUNDLED_MODEL_LIMITS.get(model_id)


# ═══════════════════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class CacheEntry:
    """
    Single model's cached limits.

    Attributes:
        context_window: Maximum context window in tokens
        max_output_tokens: Maximum output tokens per response
    """

    context_window: int
    max_output_tokens: int


@dataclass
class ModelCache:
    """
    Complete cache file contents.

    Attributes:
        format_version: Schema version for forward compatibility
        cached_at: When the cache was created
        sdk_version: SDK version that produced this cache
        models: Map of model_id → CacheEntry
    """

    format_version: int
    cached_at: datetime
    sdk_version: str
    models: dict[str, CacheEntry]


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════


def get_cache_path() -> Path:
    """
    Return path to model info cache file.

    Uses pathlib.Path.home() for cross-platform compatibility:
    - Linux/WSL: /home/<user>/.amplifier/cache/github-copilot-models.json
    - macOS: /Users/<user>/.amplifier/cache/github-copilot-models.json
    - Windows: C:\\Users\\<user>\\.amplifier\\cache\\github-copilot-models.json

    Returns:
        Path: Absolute path to cache file

    Raises:
        RuntimeError: If home directory cannot be determined
        OSError: If home directory is inaccessible
    """
    return Path.home() / ".amplifier" / "cache" / CACHE_FILE_NAME


def load_cache() -> ModelCache | None:
    """
    Load model info cache from disk.

    Reads the cache file synchronously. This is called during provider
    initialization to populate _model_info_cache without API calls.

    Returns:
        ModelCache if file exists and is valid, None otherwise.

    Error Handling:
        - Missing file: Returns None, logs debug
        - Corrupted JSON: Returns None, logs warning
        - Missing required fields: Returns None, logs warning
        - Permission errors: Returns None, logs warning
        - Home directory unavailable: Returns None, logs warning
    """
    try:
        cache_path = get_cache_path()
    except (OSError, RuntimeError) as e:
        logger.warning(f"[MODEL_CACHE] Could not determine cache path: {e}")
        return None

    if not cache_path.exists():
        logger.debug(
            f"[MODEL_CACHE] No cache file found at {cache_path}. "
            f"Will use BUNDLED_MODEL_LIMITS fallback."
        )
        return None

    try:
        raw_data = cache_path.read_text(encoding="utf-8")
        data = json.loads(raw_data)

        # Validate required fields
        if "models" not in data:
            logger.warning(
                "[MODEL_CACHE] Cache file missing 'models' key. "
                "Treating as corrupted."
            )
            return None

        if "format_version" not in data:
            logger.warning(
                "[MODEL_CACHE] Cache file missing 'format_version'. "
                "Treating as corrupted."
            )
            return None

        # Parse cached_at timestamp
        cached_at = datetime.now(UTC)
        if "cached_at" in data:
            try:
                # Handle both Z suffix and +00:00
                ts_str = data["cached_at"]
                if ts_str.endswith("Z"):
                    ts_str = ts_str[:-1] + "+00:00"
                cached_at = datetime.fromisoformat(ts_str)
            except (ValueError, TypeError) as e:
                logger.debug(f"[MODEL_CACHE] Could not parse cached_at: {e}")

        # Parse models
        models: dict[str, CacheEntry] = {}
        raw_models = data.get("models", {})
        for model_id, limits in raw_models.items():
            if not isinstance(limits, dict):
                logger.debug(f"[MODEL_CACHE] Skipping invalid model entry: {model_id}")
                continue

            context_window = limits.get("context_window")
            max_output_tokens = limits.get("max_output_tokens")

            # Validate values are positive integers
            if not isinstance(context_window, int) or context_window <= 0:
                logger.debug(
                    f"[MODEL_CACHE] Invalid context_window for {model_id}: {context_window}"
                )
                continue

            if not isinstance(max_output_tokens, int) or max_output_tokens <= 0:
                logger.debug(
                    f"[MODEL_CACHE] Invalid max_output_tokens for {model_id}: {max_output_tokens}"
                )
                continue

            models[model_id] = CacheEntry(
                context_window=context_window,
                max_output_tokens=max_output_tokens,
            )

        cache = ModelCache(
            format_version=data.get("format_version", CACHE_FORMAT_VERSION),
            cached_at=cached_at,
            sdk_version=data.get("sdk_version", "unknown"),
            models=models,
        )

        logger.info(
            f"[MODEL_CACHE] Loaded {len(models)} model(s) from cache "
            f"(SDK v{cache.sdk_version}, cached at {cache.cached_at.isoformat()})"
        )

        return cache

    except json.JSONDecodeError as e:
        logger.warning(f"[MODEL_CACHE] Cache file is corrupted JSON: {e}")
        return None
    except OSError as e:
        logger.warning(f"[MODEL_CACHE] Failed to read cache file: {e}")
        return None
    except Exception as e:
        logger.warning(f"[MODEL_CACHE] Unexpected error loading cache: {e}")
        return None


def write_cache(models: dict[str, CacheEntry], sdk_version: str) -> bool:
    """
    Write model info cache to disk.

    Uses atomic write pattern (temp file + rename) for crash safety.
    Creates parent directories if they don't exist.

    Args:
        models: Map of model_id → CacheEntry
        sdk_version: SDK version that provided this data

    Returns:
        True if write succeeded, False otherwise.

    Error Handling:
        - Permission errors: Returns False, logs warning
        - Disk full: Returns False, logs warning
        - Other I/O errors: Returns False, logs warning
        - Home directory unavailable: Returns False, logs warning
    """
    try:
        cache_path = get_cache_path()
    except (OSError, RuntimeError) as e:
        logger.warning(f"[MODEL_CACHE] Failed to write cache file: {e}")
        return False

    try:
        # Ensure parent directory exists
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Build cache data
        cache_data: dict[str, Any] = {
            "format_version": CACHE_FORMAT_VERSION,
            "cached_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "sdk_version": sdk_version,
            "models": {
                model_id: {
                    "context_window": entry.context_window,
                    "max_output_tokens": entry.max_output_tokens,
                }
                for model_id, entry in models.items()
            },
        }

        # Serialize to JSON with nice formatting for debugging
        json_data = json.dumps(cache_data, indent=2, ensure_ascii=False)

        # Atomic write using tempfile.mkstemp for unique temp file
        # IMPORTANT: temp file must be in same directory for atomic rename
        import os
        import tempfile

        fd, temp_path_str = tempfile.mkstemp(
            suffix=".tmp",
            prefix="github-copilot-models.",
            dir=cache_path.parent,
        )
        temp_path = Path(temp_path_str)

        try:
            # Write data to file descriptor and sync to disk
            os.write(fd, json_data.encode("utf-8"))
            os.fsync(fd)  # Ensure data reaches disk before rename
            os.close(fd)
            fd = -1  # Mark as closed

            # Atomic replace: works on Windows + Unix
            # NOTE: Use replace() not rename() - Windows doesn't allow rename to existing file
            temp_path.replace(cache_path)

        except Exception:
            # Clean up temp file on any failure
            if fd != -1:
                try:
                    os.close(fd)
                except OSError:
                    pass
            temp_path.unlink(missing_ok=True)
            raise

        logger.info(
            f"[MODEL_CACHE] Wrote {len(models)} model(s) to cache "
            f"(SDK v{sdk_version})"
        )
        return True

    except OSError as e:
        logger.warning(f"[MODEL_CACHE] Failed to write cache file: {e}")
        return False
    except Exception as e:
        logger.warning(f"[MODEL_CACHE] Unexpected error writing cache: {e}")
        return False


def is_cache_stale(cache: ModelCache, days: int = CACHE_STALE_DAYS) -> bool:
    """
    Check if cache is older than threshold.

    This is informational only — stale cache is still used, but a debug
    message suggests re-running 'amplifier init'.

    Args:
        cache: ModelCache to check
        days: Staleness threshold in days (default from constants)

    Returns:
        True if cache is older than threshold, False otherwise.
    """
    now = datetime.now(UTC)

    # Ensure cached_at is timezone-aware for comparison
    cached_at = cache.cached_at
    if cached_at.tzinfo is None:
        cached_at = cached_at.replace(tzinfo=UTC)

    age = now - cached_at
    is_stale = age > timedelta(days=days)

    if is_stale:
        logger.debug(
            f"[MODEL_CACHE] Cache is {age.days} days old (threshold: {days} days). "
            f"Consider running 'amplifier init' to refresh."
        )

    return is_stale


def get_sdk_version_from_cache() -> str | None:
    """
    Get SDK version from cache without loading full cache.

    Useful for staleness checks comparing against current SDK version.

    Returns:
        SDK version string if cache exists and is valid, None otherwise.
    """
    cache = load_cache()
    return cache.sdk_version if cache else None
