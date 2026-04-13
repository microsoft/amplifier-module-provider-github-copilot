"""Retry mechanics — backoff calculation and error classification.

Separated from config loading per single-responsibility principle.
Contract: behaviors:Retry:MUST:2,3,4,5,6
"""

from __future__ import annotations

import random


def calculate_backoff_delay(
    attempt: int,
    base_delay_ms: int = 1000,
    max_delay_ms: int = 30000,
    jitter_factor: float = 0.1,
) -> float:
    """Calculate exponential backoff delay with jitter.

    Contract: behaviors:Retry:MUST:2, behaviors:Retry:MUST:3

    Args:
        attempt: 0-indexed attempt number (0 = first retry)
        base_delay_ms: Base delay in milliseconds
        max_delay_ms: Maximum delay cap in milliseconds
        jitter_factor: Jitter factor (0.1 = ±10%)

    Returns:
        Delay in milliseconds with jitter applied (always >= 0).

    """
    # Clamp inputs to valid ranges
    base_delay_ms = max(0, base_delay_ms)
    max_delay_ms = max(0, max_delay_ms)
    jitter_factor = max(0.0, min(1.0, jitter_factor))  # Clamp to [0, 1]

    # Exponential: 2^attempt * base
    delay = min(base_delay_ms * (2**attempt), max_delay_ms)

    # Apply jitter (±jitter_factor)
    # S311: random is appropriate here - this is for retry jitter, not cryptography
    jitter = delay * jitter_factor * (2 * random.random() - 1)  # noqa: S311
    return max(0.0, delay + jitter)  # Never return negative delay


def is_retryable_error(error: Exception) -> bool:
    """Check if error should be retried.

    Contract: behaviors:Retry:MUST:4, behaviors:Retry:MUST:5

    Checks the `retryable` attribute on LLMError subclasses.
    """
    return getattr(error, "retryable", False)


def get_retry_after(error: Exception) -> float | None:
    """Extract retry_after from error if present.

    Contract: behaviors:Retry:MUST:6
    """
    retry_after = getattr(error, "retry_after", None)
    if retry_after is not None:
        return float(retry_after)
    return None


__all__ = [
    "calculate_backoff_delay",
    "is_retryable_error",
    "get_retry_after",
]
