"""Provider configuration loading from YAML.

Contract: provider-protocol.md (get_info, list_models)

MUST constraints:
- MUST load from wheel-packaged config/models.yaml
- MUST raise ConfigurationError if config missing or invalid (fail-fast)
- MUST include context_window per provider-protocol.md

Performance: Config loaders use @lru_cache since YAML files are packaged
in the wheel and never change at runtime. This avoids disk I/O on every request.
"""

from __future__ import annotations

import functools
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    pass

# Single import point for ConfigurationError (Three-Medium Architecture)
# Contract: sdk-boundary:Membrane:MUST:1 — Single import point for runtime dependencies
from amplifier_module_provider_github_copilot._compat import ConfigurationError

# Three-Medium Architecture: All policy values come from YAML. No Python fallbacks.
# If YAML is missing required keys, fail fast with ConfigurationError.

logger = logging.getLogger(__name__)

__all__ = [
    "ProviderConfig",
    "RetryConfig",
    "StreamingConfig",
    "SdkProtectionConfig",
    "load_models_config",
    "load_retry_config",
    "load_streaming_config",
    "load_sdk_protection_config",
    "calculate_backoff_delay",
    "is_retryable_error",
    "get_retry_after",
    # Model fallbacks (moved here to avoid circular import A-03)
    "get_default_context_window",
    "get_default_max_output_tokens",
]


# ============================================================================
# Provider Config Loading
# ============================================================================


@dataclass
class ProviderConfig:
    """Policy data loaded from config/models.yaml."""

    provider_id: str
    display_name: str
    credential_env_vars: list[str]
    capabilities: list[str]
    defaults: dict[str, Any]
    models: list[dict[str, Any]]


@functools.lru_cache(maxsize=1)
def load_models_config() -> ProviderConfig:
    """Load provider and model policy from config/models.yaml.

    Config lives inside the wheel at amplifier_module_provider_github_copilot/config/

    Raises ConfigurationError if config is missing or invalid (fail-fast).
    The config file ships with the package — if it's missing, the installation is broken.
    """
    config_path = Path(__file__).parent / "config" / "models.yaml"

    # Fail-fast: missing config = broken installation
    if not config_path.exists():
        logger.debug("[CONFIG_VALIDATION] models.yaml not found at: %s", config_path)
        raise ConfigurationError(
            f"Config validation failed: models.yaml not found. "
            f"Expected at: {config_path}. "
            "This indicates a broken installation - reinstall the package.",
            provider="github-copilot",
        )

    try:
        with config_path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        from .security_redaction import redact_sensitive_text

        logger.warning(
            "[CONFIG_VALIDATION] Failed to parse models.yaml: %s",
            redact_sensitive_text(e),
        )
        raise ConfigurationError(
            f"Config validation failed: models.yaml is corrupted ({type(e).__name__}). "
            "Reinstall the package to restore config files.",
            provider="github-copilot",
        ) from e

    # Fail-fast: empty or None config = broken file
    if not data:
        logger.warning("[CONFIG_VALIDATION] models.yaml is empty or returned None")
        raise ConfigurationError(
            "Config validation failed: models.yaml is empty or invalid. "
            "Reinstall the package to restore config files.",
            provider="github-copilot",
        )

    # Fail-fast: no models defined = incomplete config
    models = data.get("models", [])
    if not models:
        logger.warning("[CONFIG_VALIDATION] models.yaml has no models defined")
        raise ConfigurationError(
            "Config validation failed: models.yaml contains no models. "
            "At least one model must be defined.",
            provider="github-copilot",
        )

    # Three-Medium: Validate required keys exist (fail-fast)
    p = data.get("provider")
    if not p:
        raise ConfigurationError(
            "Config validation failed: models.yaml missing 'provider' section.",
            provider="github-copilot",
        )

    defaults = p.get("defaults")
    if not defaults or "model" not in defaults:
        raise ConfigurationError(
            "Config validation failed: models.yaml missing 'provider.defaults.model'.",
            provider="github-copilot",
        )

    if "timeout" not in defaults:
        raise ConfigurationError(
            "Config validation failed: models.yaml missing 'provider.defaults.timeout'.",
            provider="github-copilot",
        )

    # All required keys validated - use direct access
    return ProviderConfig(
        provider_id=p["id"],
        display_name=p["display_name"],
        credential_env_vars=p.get("credential_env_vars", []),
        capabilities=p.get("capabilities", []),
        defaults=defaults,
        models=models,
    )


# ============================================================================
# Retry Config Loading
# ============================================================================


@dataclass
class RetryConfig:
    """Policy data loaded from config/retry.yaml.

    Contract: behaviors.md
    """

    max_attempts: int
    base_delay_ms: int
    max_delay_ms: int
    jitter_factor: float


@functools.lru_cache(maxsize=1)
def load_retry_config() -> RetryConfig:
    """Load retry policy from config/retry.yaml.

    Three-Medium: YAML is authoritative. Fail-fast if missing/invalid.
    Validates max_attempts >= 1.
    """
    config_path = Path(__file__).parent / "config" / "retry.yaml"
    if not config_path.exists():
        raise ConfigurationError(
            f"Config validation failed: retry.yaml not found at {config_path}. "
            "This indicates a broken installation - reinstall the package.",
            provider="github-copilot",
        )

    try:
        with config_path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        raise ConfigurationError(
            f"Config validation failed: retry.yaml is corrupted ({type(e).__name__}). "
            "Reinstall the package to restore config files.",
            provider="github-copilot",
        ) from e

    if not data:
        raise ConfigurationError(
            "Config validation failed: retry.yaml is empty or invalid.",
            provider="github-copilot",
        )

    retry = data.get("retry")
    if not retry:
        raise ConfigurationError(
            "Config validation failed: retry.yaml missing 'retry' section.",
            provider="github-copilot",
        )

    backoff = retry.get("backoff")
    if not backoff:
        raise ConfigurationError(
            "Config validation failed: retry.yaml missing 'retry.backoff' section.",
            provider="github-copilot",
        )

    # Validate max_attempts (must be at least 1)
    max_attempts = retry.get("max_attempts")
    if max_attempts is None:
        raise ConfigurationError(
            "Config validation failed: retry.yaml missing 'retry.max_attempts'.",
            provider="github-copilot",
        )
    if max_attempts < 1:
        raise ConfigurationError(
            f"Config validation failed: retry.max_attempts={max_attempts} invalid (must be >= 1).",
            provider="github-copilot",
        )

    # Validate required backoff keys
    for key in ["base_delay_ms", "max_delay_ms", "jitter_factor"]:
        if key not in backoff:
            raise ConfigurationError(
                f"Config validation failed: retry.yaml missing 'retry.backoff.{key}'.",
                provider="github-copilot",
            )

    return RetryConfig(
        max_attempts=max_attempts,
        base_delay_ms=backoff["base_delay_ms"],
        max_delay_ms=backoff["max_delay_ms"],
        jitter_factor=backoff["jitter_factor"],
    )


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


# ============================================================================
# Streaming Config Loading
# Contract: behaviors:Streaming:MUST:1,2
# ============================================================================


@dataclass
class StreamingConfig:
    """Streaming policy from config/retry.yaml (streaming section).

    Contract: behaviors:Streaming:MUST:1,2,3,4
    """

    event_queue_size: int
    ttft_warning_ms: int
    max_gap_warning_ms: int
    max_gap_error_ms: int


@functools.lru_cache(maxsize=1)
def load_streaming_config() -> StreamingConfig:
    """Load streaming policy from config/retry.yaml (streaming section).

    Contract: behaviors:Streaming:MUST:1,2,3,4

    Three-Medium: YAML is authoritative. Fail-fast if missing/invalid.
    """
    config_path = Path(__file__).parent / "config" / "retry.yaml"
    if not config_path.exists():
        raise ConfigurationError(
            f"Config validation failed: retry.yaml not found at {config_path}. "
            "This indicates a broken installation - reinstall the package.",
            provider="github-copilot",
        )

    try:
        with config_path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        raise ConfigurationError(
            f"Config validation failed: retry.yaml is corrupted ({type(e).__name__}). "
            "Reinstall the package to restore config files.",
            provider="github-copilot",
        ) from e

    if not data:
        raise ConfigurationError(
            "Config validation failed: retry.yaml is empty or invalid.",
            provider="github-copilot",
        )

    streaming = data.get("streaming")
    if not streaming:
        raise ConfigurationError(
            "Config validation failed: retry.yaml missing 'streaming' section.",
            provider="github-copilot",
        )

    # Validate required keys
    for key in ["event_queue_size", "ttft_warning_ms", "max_gap_warning_ms", "max_gap_error_ms"]:
        if key not in streaming:
            raise ConfigurationError(
                f"Config validation failed: retry.yaml missing 'streaming.{key}'.",
                provider="github-copilot",
            )

    return StreamingConfig(
        event_queue_size=streaming["event_queue_size"],
        ttft_warning_ms=streaming["ttft_warning_ms"],
        max_gap_warning_ms=streaming["max_gap_warning_ms"],
        max_gap_error_ms=streaming["max_gap_error_ms"],
    )


# ============================================================================
# SDK Protection Config Loading
# ============================================================================


@dataclass
class ToolCaptureConfig:
    """Tool capture policy from config/sdk_protection.yaml.

    Contract: sdk-protection:ToolCapture:MUST:1,2
    """

    first_turn_only: bool
    deduplicate: bool
    log_capture_events: bool


@dataclass
class SessionProtectionConfig:
    """Session protection policy from config/sdk_protection.yaml.

    Contract: sdk-protection:Session:MUST:3,4

    Note: Named SessionProtectionConfig (not SessionConfig) to avoid collision
    with sdk_adapter.types.SessionConfig which configures SDK session creation.
    This class configures session lifecycle protection (abort, idle timeouts).
    """

    explicit_abort: bool
    abort_timeout_seconds: float
    idle_timeout_seconds: float


@dataclass
class SdkProtectionConfig:
    """SDK protection policy from config/sdk_protection.yaml.

    Contract: contracts/sdk-protection.md
    """

    tool_capture: ToolCaptureConfig
    session: SessionProtectionConfig


@functools.lru_cache(maxsize=1)
def load_sdk_protection_config() -> SdkProtectionConfig:
    """Load SDK protection policy from config/sdk_protection.yaml.

    Contract: sdk-protection:ToolCapture:MUST:1,2, sdk-protection:Session:MUST:3,4

    Three-Medium: YAML is authoritative. Fail-fast if missing/invalid.
    """
    config_path = Path(__file__).parent / "config" / "sdk_protection.yaml"
    if not config_path.exists():
        raise ConfigurationError(
            f"Config validation failed: sdk_protection.yaml not found at {config_path}. "
            "This indicates a broken installation - reinstall the package.",
            provider="github-copilot",
        )

    try:
        with config_path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        raise ConfigurationError(
            f"Config validation failed: sdk_protection.yaml is corrupted ({type(e).__name__}). "
            "Reinstall the package to restore config files.",
            provider="github-copilot",
        ) from e

    if not data:
        raise ConfigurationError(
            "Config validation failed: sdk_protection.yaml is empty or invalid.",
            provider="github-copilot",
        )

    # Validate required sections
    tc = data.get("tool_capture")
    if not tc:
        raise ConfigurationError(
            "Config validation failed: sdk_protection.yaml missing 'tool_capture' section.",
            provider="github-copilot",
        )

    sess = data.get("session")
    if not sess:
        raise ConfigurationError(
            "Config validation failed: sdk_protection.yaml missing 'session' section.",
            provider="github-copilot",
        )

    # Validate required keys
    for key in ["first_turn_only", "deduplicate", "log_capture_events"]:
        if key not in tc:
            raise ConfigurationError(
                f"Config validation failed: sdk_protection.yaml missing 'tool_capture.{key}'.",
                provider="github-copilot",
            )

    for key in ["explicit_abort", "abort_timeout_seconds", "idle_timeout_seconds"]:
        if key not in sess:
            raise ConfigurationError(
                f"Config validation failed: sdk_protection.yaml missing 'session.{key}'.",
                provider="github-copilot",
            )

    tool_capture = ToolCaptureConfig(
        first_turn_only=tc["first_turn_only"],
        deduplicate=tc["deduplicate"],
        log_capture_events=tc["log_capture_events"],
    )

    session = SessionProtectionConfig(
        explicit_abort=sess["explicit_abort"],
        abort_timeout_seconds=float(sess["abort_timeout_seconds"]),
        idle_timeout_seconds=float(sess["idle_timeout_seconds"]),
    )

    return SdkProtectionConfig(
        tool_capture=tool_capture,
        session=session,
    )


# =============================================================================
# Model Fallback Values (moved from models.py to avoid circular import A-03)
# Contract: behaviors:ConfigLoading:MUST:1 — fail-fast on missing config
# =============================================================================


@functools.lru_cache(maxsize=1)
def _load_model_fallback_values() -> dict[str, int]:
    """Load fallback values from config/models.yaml.

    Three-Medium Architecture: Python loads policy from YAML.
    Contract: behaviors:ConfigLoading:MUST:1 — fail-fast on missing config.

    Returns:
        Dict with context_window and max_output_tokens from YAML.

    Raises:
        ConfigurationError: If models.yaml is missing or lacks required keys.
    """
    import importlib.resources

    try:
        files = importlib.resources.files("amplifier_module_provider_github_copilot.config")
        content = (files / "models.yaml").read_text()
    except (FileNotFoundError, TypeError) as exc:
        raise ConfigurationError(
            "models.yaml not found in config/. "
            "Three-Medium Architecture: YAML is authoritative for fallback values."
        ) from exc

    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Failed to parse models.yaml: {exc}") from exc

    if data is None or "fallbacks" not in data:
        raise ConfigurationError(
            "models.yaml must contain 'fallbacks' section with context_window "
            "and max_output_tokens. "
            "Three-Medium Architecture: YAML is authoritative for policy values."
        )

    fallbacks = data["fallbacks"]
    required_keys = ["context_window", "max_output_tokens"]
    missing = [k for k in required_keys if k not in fallbacks]
    if missing:
        raise ConfigurationError(
            f"models.yaml fallbacks section missing required keys: {missing}. "
            "Three-Medium Architecture: YAML is authoritative for policy values."
        )

    return fallbacks


def get_default_context_window() -> int:
    """Get default context window from YAML config.

    Contract: behaviors:ConfigLoading:MUST:1 — fail-fast on missing config.

    Returns:
        Default context window value from models.yaml.

    Raises:
        ConfigurationError: If models.yaml is missing or lacks required keys.
    """
    return _load_model_fallback_values()["context_window"]


def get_default_max_output_tokens() -> int:
    """Get default max output tokens from YAML config.

    Contract: behaviors:ConfigLoading:MUST:1 — fail-fast on missing config.

    Returns:
        Default max_output_tokens value from models.yaml.

    Raises:
        ConfigurationError: If models.yaml is missing or lacks required keys.
    """
    return _load_model_fallback_values()["max_output_tokens"]
