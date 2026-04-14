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

    Contract: behaviors:Retry:MUST:1,2,3,7,8

    Field naming note: `max_attempts` is the total call count including the initial
    attempt. The user-facing config key is `max_retries` (= max_attempts - 1).
    Example: max_retries=2 → max_attempts=3 → one initial call + two retries.
    """

    max_attempts: int = 3  # total attempts including first; user-facing key: max_retries
    base_delay_ms: int = 1000
    max_delay_ms: int = 30000
    jitter_factor: float = 0.1
    overloaded_delay_multiplier: float = 10.0

    def __post_init__(self) -> None:
        """Validate field invariants on construction.

        frozen=True means fields cannot be mutated after __init__;
        __post_init__ can read fields and raise to reject invalid state.
        """
        if self.overloaded_delay_multiplier < 1.0:
            raise ValueError(
                f"overloaded_delay_multiplier must be >= 1.0, "
                f"got {self.overloaded_delay_multiplier!r}"
            )


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
