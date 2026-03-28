"""Fake tool call detection module.

Contract: provider-protocol.md (complete:MUST:5)

LLMs sometimes emit tool calls as plain text instead of structured calls.
This module detects such patterns and provides correction retry logic.

Three-Medium Architecture:
- YAML: Detection patterns, correction message, max attempts (config)
- Python: Regex matching and retry orchestration (mechanism)
- Markdown: provider-protocol.md anchors the behavior (contract)
"""

from __future__ import annotations

import functools
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from importlib.abc import Traversable

logger = logging.getLogger(__name__)


@dataclass
class LoggingConfig:
    """Logging configuration for fake tool call detection."""

    log_matched_pattern: bool = True
    # P1 Fix (C4): Secure default matches YAML (false). If YAML fails to load,
    # we fail closed (don't log potentially sensitive LLM response text).
    log_response_text: bool = False
    log_response_text_limit: int = 500
    # P1 Fix: Secure defaults match YAML (false). If YAML fails to load,
    # we fail closed (don't log potentially sensitive tool arguments/correction text).
    log_tool_calls: bool = False
    log_correction_message: bool = False
    level_on_detection: str = "INFO"
    level_on_retry: str = "INFO"
    level_on_success: str = "INFO"
    level_on_exhausted: str = "WARNING"


@dataclass
class FakeToolDetectionConfig:
    """Policy loaded from config/fake-tool-detection.yaml.

    Contract: behaviors:Config:MUST:1 - loaded from YAML.
    """

    patterns: list[re.Pattern[str]] = field(default_factory=lambda: [])
    max_correction_attempts: int = 2
    correction_message: str = (
        "You wrote tool calls as plain text instead of using the "
        "structured tool calling mechanism. Please use actual tool "
        "calls, not text representations of them."
    )
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _default_patterns() -> list[re.Pattern[str]]:
    """Return default detection patterns.

    Used when config file is missing.
    """
    return [
        re.compile(r"\[Tool Call:\s*\w+", re.IGNORECASE),
        re.compile(r"<tool_used\s+name=", re.IGNORECASE),
        re.compile(r"<tool_result\s+name=", re.IGNORECASE),
    ]


@functools.lru_cache(maxsize=4)
def _load_fake_tool_detection_config_cached(
    config_path_str: str | None,
) -> FakeToolDetectionConfig:
    """Internal cached loader."""
    yaml_text: str | None = None

    if config_path_str is None:
        # Use importlib.resources for package config
        try:
            from importlib import resources

            config_files: Traversable = resources.files(
                "amplifier_module_provider_github_copilot.config"
            )
            config_file = config_files.joinpath("fake-tool-detection.yaml")
            yaml_text = config_file.read_text(encoding="utf-8")
        except (ModuleNotFoundError, FileNotFoundError, TypeError):
            # Fallback to filesystem path
            fallback_path = Path(__file__).parent / "config" / "fake-tool-detection.yaml"
            if fallback_path.exists():
                yaml_text = fallback_path.read_text(encoding="utf-8")
            else:
                # Return defaults
                logger.debug("Fake tool detection config not found, using defaults")
                return FakeToolDetectionConfig(patterns=_default_patterns())
    else:
        config_path = Path(config_path_str)
        if not config_path.exists():
            logger.debug(
                "Fake tool detection config not found at %s, using defaults",
                config_path,
            )
            return FakeToolDetectionConfig(patterns=_default_patterns())
        yaml_text = config_path.read_text(encoding="utf-8")

    if not yaml_text:  # Empty file or whitespace-only
        return FakeToolDetectionConfig(patterns=_default_patterns())

    try:
        data = yaml.safe_load(yaml_text)
        if not data:
            return FakeToolDetectionConfig(patterns=_default_patterns())

        # Compile patterns
        raw_patterns: list[str] = data.get("patterns", [])
        compiled_patterns: list[re.Pattern[str]] = []
        for pattern_str in raw_patterns:
            try:
                compiled_patterns.append(re.compile(pattern_str, re.IGNORECASE))
            except re.error as e:
                from .security_redaction import redact_sensitive_text

                logger.warning(
                    "Invalid regex pattern '%s': %s",
                    pattern_str,
                    redact_sensitive_text(e),
                )

        if not compiled_patterns:
            compiled_patterns = _default_patterns()

        # Load logging config
        logging_data = data.get("logging", {})
        # Security: All sensitive logging defaults to False (fail-closed).
        # If YAML is readable, it can explicitly enable; if YAML is missing,
        # we don't inadvertently log secrets.
        logging_config = LoggingConfig(
            log_matched_pattern=logging_data.get("log_matched_pattern", True),
            log_response_text=logging_data.get("log_response_text", False),
            log_response_text_limit=logging_data.get("log_response_text_limit", 500),
            log_tool_calls=logging_data.get("log_tool_calls", False),
            log_correction_message=logging_data.get("log_correction_message", False),
            level_on_detection=logging_data.get("level_on_detection", "INFO"),
            level_on_retry=logging_data.get("level_on_retry", "INFO"),
            level_on_success=logging_data.get("level_on_success", "INFO"),
            level_on_exhausted=logging_data.get("level_on_exhausted", "WARNING"),
        )

        return FakeToolDetectionConfig(
            patterns=compiled_patterns,
            max_correction_attempts=data.get("max_correction_attempts", 2),
            correction_message=data.get(
                "correction_message",
                FakeToolDetectionConfig.correction_message,
            ),
            logging=logging_config,
        )
    except yaml.YAMLError as e:
        from .security_redaction import redact_sensitive_text

        # P3 Fix: Re-raise to prevent lru_cache from caching the failure.
        # Returning a fallback here would poison the cache for the process lifetime.
        logger.warning("Error parsing fake tool detection config: %s", redact_sensitive_text(e))
        raise


def load_fake_tool_detection_config(
    config_path: Path | None = None,
) -> FakeToolDetectionConfig:
    """Load fake tool call detection policy from config.

    Contract: behaviors:Config:MUST:1 - policy from YAML.

    Falls back to sensible defaults if config missing.

    Args:
        config_path: Optional explicit path to config file.
                    If None, uses package config directory.

    Returns:
        FakeToolDetectionConfig with patterns, max attempts, and message.

    """
    # Convert Path to str for caching (or keep None)
    path_str = str(config_path) if config_path is not None else None
    return _load_fake_tool_detection_config_cached(path_str)


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
