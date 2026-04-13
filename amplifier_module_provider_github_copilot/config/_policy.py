"""Retry, streaming, and cache policy defaults.

Python policy module — replaces the former config/retry.yaml YAML file.
Contract: behaviors.md
"""

from __future__ import annotations

import functools
from dataclasses import dataclass


@dataclass(frozen=True)
class RetryPolicy:
    """Retry policy configuration.

    Contract: behaviors:Retry:MUST:1,2,3
    """

    max_attempts: int = 3
    base_delay_ms: int = 1000
    max_delay_ms: int = 30000
    jitter_factor: float = 0.1


@dataclass(frozen=True)
class StreamingConfig:
    """Streaming policy configuration.

    Contract: behaviors:Streaming:MUST:1,2

    Note: max_gap_warning_ms and max_gap_error_ms were previously defined in
    _policy.py but never read by any caller. They are intentionally omitted.
    """

    event_queue_size: int = 10000
    ttft_warning_ms: int = 15000


@dataclass(frozen=True)
class CacheConfig:
    """Model cache policy configuration.

    Contract: behaviors:ModelCache:SHOULD:2

    Note: max_stale_seconds is defined in the contract but not currently
    read by any caller. Included for contract compliance.
    """

    disk_ttl_seconds: int = 86400  # 24 hours
    max_stale_seconds: int = 604800  # 7 days
    cache_filename: str = "models_cache.json"


@functools.lru_cache(maxsize=1)
def load_cache_config() -> CacheConfig:
    """Load cache policy (hardcoded defaults).

    Returns frozen CacheConfig. Cached for performance parity
    with the former YAML loader. Tests call .cache_clear().
    """
    return CacheConfig()


@functools.lru_cache(maxsize=1)
def load_retry_config() -> RetryPolicy:
    """Load retry policy (hardcoded defaults).

    Returns frozen RetryPolicy. Cached for performance parity
    with the former YAML loader. Tests call .cache_clear().
    """
    return RetryPolicy()


@functools.lru_cache(maxsize=1)
def load_streaming_config() -> StreamingConfig:
    """Load streaming policy (hardcoded defaults).

    Returns frozen StreamingConfig. Cached for performance parity
    with the former YAML loader. Tests call .cache_clear().
    """
    return StreamingConfig()


__all__ = [
    "CacheConfig",
    "RetryPolicy",
    "StreamingConfig",
    "load_cache_config",
    "load_retry_config",
    "load_streaming_config",
]
