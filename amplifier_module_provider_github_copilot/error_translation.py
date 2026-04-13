"""Config-driven error translation using kernel types.

All SDK errors are translated to kernel LLMError types from amplifier_core.llm_errors.
Mappings are driven by config/errors.yaml - no hardcoded error mappings.

Contract: contracts/error-hierarchy.md

MUST constraints (from contract):
- MUST use kernel error types from amplifier_core.llm_errors
- MUST NOT create custom error classes
- MUST set provider="github-copilot" on all errors
- MUST preserve original exception via chaining
- MUST use config-driven pattern matching
- MUST fall through to ProviderUnavailableError(retryable=False) for unknown errors
"""

from __future__ import annotations

import functools
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Import ALL error types from amplifier_core.llm_errors
from amplifier_core.llm_errors import (
    AbortError,
    AccessDeniedError,
    AuthenticationError,
    ConfigurationError,
    ContentFilterError,
    ContextLengthError,
    InvalidRequestError,
    InvalidToolCallError,
    LLMError,
    LLMTimeoutError,
    NetworkError,
    NotFoundError,
    ProviderUnavailableError,
    QuotaExceededError,
    RateLimitError,
    StreamError,
)

logger = logging.getLogger(__name__)

# Re-export kernel error types for use by other modules in this package
__all__ = [
    # Kernel error types (re-exported from amplifier_core.llm_errors)
    "AbortError",
    "AuthenticationError",
    "ConfigurationError",
    "LLMError",
    "LLMTimeoutError",
    "NetworkError",
    "ProviderUnavailableError",
    "RateLimitError",
    # Module exports
    "ErrorConfig",
    "ErrorMapping",
    "ContextExtraction",
    "load_error_config",
    "translate_sdk_error",
    "KERNEL_ERROR_MAP",
]


# Mapping from config names to kernel error classes
KERNEL_ERROR_MAP: dict[str, type[LLMError]] = {
    "AuthenticationError": AuthenticationError,
    "RateLimitError": RateLimitError,
    "QuotaExceededError": QuotaExceededError,
    "LLMTimeoutError": LLMTimeoutError,
    "ContentFilterError": ContentFilterError,
    "NetworkError": NetworkError,
    "NotFoundError": NotFoundError,
    "ProviderUnavailableError": ProviderUnavailableError,
    "ContextLengthError": ContextLengthError,
    "InvalidRequestError": InvalidRequestError,
    "StreamError": StreamError,
    "InvalidToolCallError": InvalidToolCallError,
    "ConfigurationError": ConfigurationError,
    "AccessDeniedError": AccessDeniedError,
    "AbortError": AbortError,
}


def _str_list() -> list[str]:
    """Return an empty string list (typed)."""
    return []


@dataclass
class ContextExtraction:
    """A context extraction pattern for enhanced error messages.

    Extracts structured context from error messages.

    Attributes:
        pattern: Regex pattern with a capture group.
        field: Name of the field to extract (e.g., "tool_name").

    """

    pattern: str
    field: str


def _context_list() -> list[ContextExtraction]:
    """Return an empty ContextExtraction list (typed)."""
    return []


@dataclass
class ErrorMapping:
    """A single error mapping from SDK to kernel error.

    Attributes:
        sdk_patterns: Exception type names to match (e.g., ["AuthenticationError"]).
        string_patterns: Patterns to match in exception message.
        kernel_error: Target kernel error type name.
        retryable: Whether the error is retryable.
        extract_retry_after: Whether to extract retry_after from message.
        context_extraction: Optional list of context extraction patterns.

    """

    sdk_patterns: list[str] = field(default_factory=_str_list)
    string_patterns: list[str] = field(default_factory=_str_list)
    kernel_error: str = "ProviderUnavailableError"
    retryable: bool = True
    extract_retry_after: bool = False
    context_extraction: list[ContextExtraction] = field(default_factory=_context_list)


def _mapping_list() -> list[ErrorMapping]:
    """Return an empty ErrorMapping list (typed)."""
    return []


@dataclass
class ErrorConfig:
    """Error translation configuration.

    Attributes:
        mappings: List of error mappings to try in order.
        default_error: Default kernel error type for unmatched errors.
        default_retryable: Default retryable flag for unmatched errors.

    """

    mappings: list[ErrorMapping] = field(default_factory=_mapping_list)
    default_error: str = "ProviderUnavailableError"
    # Three-Medium: Must match YAML default (config/errors.yaml:128) and loader fallback
    default_retryable: bool = False


@functools.lru_cache(maxsize=4)
def _load_error_config_cached(config_path_str: str | None) -> ErrorConfig:
    """Internal cached loader."""
    data: dict[str, Any] | None = None

    if config_path_str is None:
        # Load via importlib.resources (installed wheel scenario)
        try:
            from importlib import resources

            config_text = (
                resources.files("amplifier_module_provider_github_copilot.config.data")
                .joinpath("errors.yaml")
                .read_text(encoding="utf-8")
            )
            data = yaml.safe_load(config_text)
        except Exception:
            # importlib.resources failed, fall back to file path
            config_path_str = str(Path(__file__).parent / "config" / "data" / "errors.yaml")

    if data is None and config_path_str is not None:
        # Load from file path
        path = Path(config_path_str)
        if not path.exists():
            return ErrorConfig()

        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)

    if not data:
        return ErrorConfig()

    mappings: list[ErrorMapping] = []
    for mapping_data in data.get("error_mappings", []):
        # Load context extraction patterns
        context_extraction: list[ContextExtraction] = []
        for ce_data in mapping_data.get("context_extraction", []):
            context_extraction.append(
                ContextExtraction(
                    pattern=ce_data.get("pattern", ""),
                    field=ce_data.get("field", ""),
                )
            )

        mappings.append(
            ErrorMapping(
                sdk_patterns=mapping_data.get("sdk_patterns", []),
                string_patterns=mapping_data.get("string_patterns", []),
                kernel_error=mapping_data.get("kernel_error", "ProviderUnavailableError"),
                retryable=mapping_data.get("retryable", True),
                extract_retry_after=mapping_data.get("extract_retry_after", False),
                context_extraction=context_extraction,
            )
        )

    default = data.get("default", {})
    return ErrorConfig(
        mappings=mappings,
        default_error=default.get("kernel_error", "ProviderUnavailableError"),
        # Python fallback MUST match YAML default (Three-Medium Architecture)
        default_retryable=default.get("retryable", False),
    )


def load_error_config(config_path: str | Path | None = None) -> ErrorConfig:
    """Load error configuration from YAML file or importlib.resources.

    Single source of truth for error config parsing.
    Supports both file path and importlib.resources loading.

    Args:
        config_path: Path to the YAML config file. If None, uses importlib.resources
            to load from the package's config directory.

    Returns:
        ErrorConfig with loaded mappings.

    """
    # Convert Path to str for caching (or keep None)
    path_str = str(config_path) if config_path is not None else None
    return _load_error_config_cached(path_str)


def _extract_retry_after(message: str) -> float | None:
    """Extract retry_after seconds from error message.

    Looks for patterns like "Retry after 30 seconds" or "retry-after: 60".

    Args:
        message: The exception message to parse.

    Returns:
        Seconds to wait, or None if not found.

    """
    # Only match retry-specific patterns, not generic "N seconds"
    patterns = [
        r"[Rr]etry[- ]?after[:\s]+(\d+(?:\.\d+)?)",
        # Removed overly broad r"(\d+(?:\.\d+)?)\s*seconds?" pattern
    ]
    for pattern in patterns:
        match = re.search(pattern, message)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return None


def _matches_mapping(exc: Exception, mapping: ErrorMapping) -> bool:
    """Check if exception matches a mapping's patterns.

    Uses exact type name matching to avoid false positives.
    Substring matching caused false positives in the past:
    - Pattern 'Error' matched every exception
    - Pattern 'TimeoutError' matched 'LLMTimeoutError'

    Args:
        exc: The exception to check.
        mapping: The mapping to match against.

    Returns:
        True if the exception matches.

    """
    exc_type_name = type(exc).__name__
    exc_message = str(exc).lower()

    # Check SDK type patterns with EXACT match (not substring)
    for pattern in mapping.sdk_patterns:
        if pattern == exc_type_name:
            return True

    # Check string patterns in message (substring match is intentional here)
    for pattern in mapping.string_patterns:
        if pattern.lower() in exc_message:
            return True

    return False


def _extract_context(message: str, extractions: list[ContextExtraction]) -> dict[str, str]:
    """Extract context fields from error message using regex patterns.

    Args:
        message: The error message to extract from.
        extractions: List of context extraction patterns.

    Returns:
        Dictionary of field_name -> extracted_value.

    """
    context: dict[str, str] = {}
    for extraction in extractions:
        try:
            match = re.search(extraction.pattern, message)
            if match:
                context[extraction.field] = match.group(1)
        except (re.error, IndexError):
            # Invalid regex or no capture group - silently skip
            pass
    return context


def _format_context_suffix(context: dict[str, str]) -> str:
    """Format extracted context as a message suffix.

    Appends context in [context: key=value, ...] format.

    Args:
        context: Dictionary of field_name -> value.

    Returns:
        Formatted suffix string, or empty string if no context.

    """
    if not context:
        return ""
    parts = [f"{k}={v}" for k, v in context.items()]
    return f" [context: {', '.join(parts)}]"


def _create_kernel_error_safely(
    error_class: type[LLMError],
    message: str,
    *,
    provider: str,
    model: str | None,
    retryable: bool,
    retry_after: float | None,
) -> LLMError:
    """Create kernel error with constructor safety.

    Different kernel error classes have varying constructor signatures.
    Some accept retry_after, others don't.
    Try with full args first, fall back to minimal args on TypeError.

    H-5 Fix: Log debug messages when fallback paths are taken, aiding debugging
    without spamming logs. Contract: behaviors:Logging:SHOULD:1

    Args:
        error_class: The kernel error class to instantiate.
        message: Error message.
        provider: Provider name.
        model: Model name (optional).
        retryable: Whether the error is retryable.
        retry_after: Retry delay in seconds (optional).

    Returns:
        Instantiated kernel error.

    """
    error_class_name = error_class.__name__

    # First, try with retry_after (most error classes support it)
    try:
        return error_class(
            message,
            provider=provider,
            model=model,
            retryable=retryable,
            retry_after=retry_after,
        )
    except TypeError as e:
        logger.debug(
            "[ERROR_TRANSLATION] %s constructor failed with retry_after: %s",
            error_class_name,
            e,
        )

    # Fall back without retry_after (e.g., InvalidToolCallError, AbortError)
    try:
        return error_class(
            message,
            provider=provider,
            model=model,
            retryable=retryable,
        )
    except TypeError as e:
        logger.debug(
            "[ERROR_TRANSLATION] %s constructor failed without retry_after: %s",
            error_class_name,
            e,
        )

    # Final fallback: minimal constructor
    try:
        return error_class(
            message,
            provider=provider,
        )
    except TypeError as e:
        # Last resort: return ProviderUnavailableError
        logger.warning(
            "[ERROR_TRANSLATION] %s constructor failed with minimal args, "
            "falling back to ProviderUnavailableError: %s",
            error_class_name,
            e,
        )
        return ProviderUnavailableError(
            message,
            provider=provider,
            model=model,
            retryable=retryable,
        )


def translate_sdk_error(
    exc: Exception,
    config: ErrorConfig,
    *,
    provider: str = "github-copilot",
    model: str | None = None,
) -> LLMError:
    """Translate SDK exception to kernel LLMError.

    Contract: error-hierarchy.md

    - MUST NOT raise (always returns)
    - MUST use config patterns (no hardcoded mappings)
    - MUST chain original via `raise X from exc`
    - MUST set provider attribute

    Security: Redacts secrets from error messages before including in kernel errors.
    Pattern matching uses original message (for accuracy); kernel error uses
    redacted message (for security).

    Args:
        exc: The SDK exception to translate.
        config: Error translation configuration.
        provider: Provider name to set on error.
        model: Model name to set on error.

    Returns:
        Kernel LLMError with original exception chained.

    """
    from .security_redaction import redact_sensitive_text

    original_message = str(exc)
    safe_message = redact_sensitive_text(original_message)
    retry_after: float | None = None

    # Try each mapping in order
    for mapping in config.mappings:
        if _matches_mapping(exc, mapping):
            # Get the kernel error class
            error_class = KERNEL_ERROR_MAP.get(mapping.kernel_error, ProviderUnavailableError)

            # Extract retry_after if configured
            if mapping.extract_retry_after:
                retry_after = _extract_retry_after(original_message)

            # Extract context from message (use original for pattern matching)
            context = _extract_context(original_message, mapping.context_extraction)
            # P0 Fix (C3): Redact extracted context values - they come from the
            # original (unredacted) message and may contain secrets like tokens.
            context = {k: redact_sensitive_text(v) for k, v in context.items()}
            # Use redacted message for output, append context
            message = safe_message + _format_context_suffix(context)

            # Create kernel error with constructor safety.
            # Different kernel error classes have varying constructor signatures.
            # Try with retry_after first, fall back without if TypeError.
            kernel_error = _create_kernel_error_safely(
                error_class,
                message,
                provider=provider,
                model=model,
                retryable=mapping.retryable,
                retry_after=retry_after,
            )
            kernel_error.__cause__ = exc

            # Log translation with sanitized type name (security)
            sanitized_type = type(exc).__name__
            logger.debug(
                "[ERROR_TRANSLATION] %s -> %s (retryable=%s, context=%s)",
                sanitized_type,
                kernel_error.__class__.__name__,
                kernel_error.retryable,
                context if context else "none",
            )

            return kernel_error

    # No mapping matched - use default with constructor safety
    # Use redacted message for output
    default_class = KERNEL_ERROR_MAP.get(config.default_error, ProviderUnavailableError)
    kernel_error = _create_kernel_error_safely(
        default_class,
        safe_message,
        provider=provider,
        model=model,
        retryable=config.default_retryable,
        retry_after=None,
    )
    kernel_error.__cause__ = exc

    # Log default translation with sanitized type name
    logger.debug(
        "[ERROR_TRANSLATION] %s -> %s (retryable=%s, default)",
        type(exc).__name__,
        kernel_error.__class__.__name__,
        kernel_error.retryable,
    )

    return kernel_error
