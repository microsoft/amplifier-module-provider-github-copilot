"""Fake tool call detection module.

Contract: provider-protocol.md (complete:MUST:5)

LLMs sometimes emit tool calls as plain text instead of structured calls.
This module detects such patterns and provides correction retry logic.

Two-Medium Architecture:
- Python: Regex matching and retry orchestration (mechanism)
- Markdown: provider-protocol.md anchors the behavior (contract)
"""

from __future__ import annotations

import functools
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LoggingConfig:
    """Logging configuration for fake tool call detection."""

    log_matched_pattern: bool = True
    # P1 Fix (C4): Secure default (false). Fail closed.
    log_response_text: bool = False
    log_response_text_limit: int = 500
    # P1 Fix: Secure defaults (false). Fail closed.
    log_tool_calls: bool = False
    log_correction_message: bool = False
    level_on_detection: str = "INFO"
    level_on_retry: str = "INFO"
    level_on_success: str = "INFO"
    level_on_exhausted: str = "WARNING"


def _default_patterns() -> list[re.Pattern[str]]:
    """Return default detection patterns."""
    return [
        re.compile(r"\[Tool Call:\s*\w+", re.IGNORECASE),
        re.compile(r"<tool_used\s+name=", re.IGNORECASE),
        re.compile(r"<tool_result\s+name=", re.IGNORECASE),
    ]


@dataclass
class FakeToolDetectionConfig:
    """Policy for fake tool call detection (hardcoded defaults).

    Contract: behaviors:Config:MUST:1
    """

    patterns: list[re.Pattern[str]] = field(default_factory=_default_patterns)
    max_correction_attempts: int = 2
    correction_message: str = (
        "You wrote tool calls as plain text instead of using the "
        "structured tool calling mechanism. Please use actual tool "
        "calls, not text representations of them."
    )
    logging: LoggingConfig = field(default_factory=LoggingConfig)


@functools.lru_cache(maxsize=1)
def load_fake_tool_detection_config() -> FakeToolDetectionConfig:
    """Load fake tool call detection policy (hardcoded defaults).

    Returns FakeToolDetectionConfig with compiled detection patterns.
    Cached for performance. Tests call .cache_clear().
    """
    return FakeToolDetectionConfig()


def contains_fake_tool_calls(
    text: str,
    config: FakeToolDetectionConfig,
) -> tuple[bool, str | None]:
    """Check if text contains fake tool call patterns.

    Contract: provider-protocol:complete:MUST:5

    Args:
        text: Response text to check.
        config: Detection configuration with compiled patterns.

    Returns:
        Tuple of (detected, matched_pattern_str).
        detected is True if any configured pattern matches.
        matched_pattern_str is the pattern that matched, or None.

    """
    if not text:
        return False, None

    for pattern in config.patterns:
        if pattern.search(text):
            return True, pattern.pattern

    return False, None


def _truncate_text(text: str, limit: int) -> str:
    """Truncate text to limit, adding ellipsis if needed."""
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + "..."


def log_detection(
    config: FakeToolDetectionConfig,
    text: str,
    matched_pattern: str | None,
    tool_calls: list[Any],
) -> None:
    """Log fake tool call detection per config."""
    log_cfg = config.logging
    level = getattr(logging, log_cfg.level_on_detection.upper(), logging.INFO)

    parts: list[str] = ["[FAKE_TOOL_CALL] Detected fake tool call in response"]

    if log_cfg.log_matched_pattern and matched_pattern:
        parts.append(f"pattern='{matched_pattern}'")

    if log_cfg.log_response_text:
        truncated = _truncate_text(text, log_cfg.log_response_text_limit)
        parts.append(f"text='{truncated}'")

    if log_cfg.log_tool_calls:
        parts.append(f"tool_calls={tool_calls}")

    logger.log(level, " ".join(parts))


def log_retry(config: FakeToolDetectionConfig, attempt: int, max_attempts: int) -> None:
    """Log retry attempt."""
    log_cfg = config.logging
    level = getattr(logging, log_cfg.level_on_retry.upper(), logging.INFO)

    msg = f"[FAKE_TOOL_CALL] Retrying with correction (attempt {attempt + 1}/{max_attempts})"
    if log_cfg.log_correction_message:
        msg += f" message='{config.correction_message}'"

    logger.log(level, msg)


def log_exhausted(config: FakeToolDetectionConfig, attempts: int) -> None:
    """Log when max attempts exhausted."""
    log_cfg = config.logging
    level = getattr(logging, log_cfg.level_on_exhausted.upper(), logging.WARNING)
    logger.log(
        level,
        "[FAKE_TOOL_CALL] Max correction attempts (%d) exhausted, returning last response",
        attempts,
    )


def log_success(config: FakeToolDetectionConfig, attempt: int) -> None:
    """Log successful correction."""
    log_cfg = config.logging
    level = getattr(logging, log_cfg.level_on_success.upper(), logging.INFO)
    logger.log(
        level,
        "[FAKE_TOOL_CALL] Correction succeeded on attempt %d",
        attempt + 1,
    )


def should_retry_for_fake_tool_calls(
    response_text: str,
    tool_calls: list[Any] | None,
    tools_available: bool,
    config: FakeToolDetectionConfig,
) -> tuple[bool, str | None]:
    """Check if we should retry due to fake tool calls in response.

    Contract: provider-protocol:complete:MUST:5

    Conditions for retry:
    - Fake tool call patterns detected in response text
    - No structured tool_calls in response
    - Tools were available in the request

    Args:
        response_text: The accumulated text content from the response.
        tool_calls: The structured tool_calls from the response (may be None or empty).
        tools_available: Whether tools were provided in the original request.
        config: Detection configuration.

    Returns:
        Tuple of (should_retry, matched_pattern).

    """
    # No retry if real tool calls were returned - LLM used tools correctly
    if tool_calls:
        return False, None

    # No retry if no tools were available in request - text-only completion
    if not tools_available:
        return False, None

    # Check for fake tool call patterns
    detected, matched_pattern = contains_fake_tool_calls(response_text, config)

    return detected, matched_pattern
