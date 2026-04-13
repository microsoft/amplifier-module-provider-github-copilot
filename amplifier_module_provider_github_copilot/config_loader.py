"""Provider configuration — loading and re-exports.

Contract: provider-protocol.md (get_info, list_models)

MUST constraints:
- MUST load from wheel-packaged config/models.py
- MUST raise ConfigurationError if config missing or invalid (fail-fast)
- MUST include context_window per provider-protocol.md

This module is both a loader (load_models_config, load_sdk_protection_config,
get_default_context_window) and a re-export hub that gives callers a single
import point for policy dataclasses (RetryConfig, StreamingConfig, SdkProtectionConfig)
and retry utilities (calculate_backoff_delay, is_retryable_error, get_retry_after).

Performance: Config loaders use @lru_cache since Python config modules are packaged
in the wheel and never change at runtime. This avoids repeated module attribute access.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from typing import Any

# Single import point for ConfigurationError (Three-Medium Architecture)
# Contract: sdk-boundary:Membrane:MUST:1 — Single import point for runtime dependencies
from amplifier_module_provider_github_copilot._compat import ConfigurationError

# Retry/streaming policy (hardcoded defaults — lives in config/_policy.py)
from .config._policy import RetryPolicy as RetryConfig  # Public alias preserved
from .config._policy import StreamingConfig, load_retry_config, load_streaming_config

# SDK protection dataclasses (SoC: data lives in config/_sdk_protection.py)
from .config._sdk_protection import (
    SdkConfig,
    SdkProtectionConfig,
    SessionProtectionConfig,
    SingletonConfig,
    ToolCaptureConfig,
)

# Retry mechanics (separated from config loading)
from .retry_utils import calculate_backoff_delay, get_retry_after, is_retryable_error

# Three-Medium Architecture: Policy values from Python (models, retry, streaming, events).
# Fail fast with ConfigurationError if invalid.

logger = logging.getLogger(__name__)

__all__ = [
    "ProviderConfig",
    "RetryConfig",
    "StreamingConfig",
    "SdkProtectionConfig",
    "SdkConfig",
    "SessionProtectionConfig",
    "ToolCaptureConfig",
    "load_models_config",
    "load_retry_config",
    "load_streaming_config",
    "load_sdk_protection_config",
    "calculate_backoff_delay",
    "is_retryable_error",
    "get_retry_after",
    # Singleton config
    "SingletonConfig",
    # Model fallbacks (moved here to avoid circular import A-03)
    "get_default_context_window",
    "get_default_max_output_tokens",
]


# ============================================================================
# Provider Config Loading
# ============================================================================


@dataclass
class ProviderConfig:
    """Policy data loaded from config/models.py."""

    provider_id: str
    display_name: str
    credential_env_vars: list[str]
    capabilities: list[str]
    defaults: dict[str, Any]
    models: list[dict[str, Any]]


@functools.lru_cache(maxsize=1)
def load_models_config() -> ProviderConfig:
    """Load provider and model policy from config/models.py.

    Config lives inside the wheel at amplifier_module_provider_github_copilot/config/

    Raises ConfigurationError if config is missing or invalid (fail-fast).
    The config ships with the package — if it's missing, the installation is broken.
    """
    try:
        from .config import _models as _models_data
    except (ImportError, ModuleNotFoundError) as e:
        raise ConfigurationError(
            "config/_models.py not found — broken installation. "
            "Package is missing required configuration module: config/_models.py",
            provider="github-copilot",
        ) from e

    # Fail-fast: no models defined = incomplete config
    models = _models_data.MODELS
    if not models:
        raise ConfigurationError(
            "Config validation failed: config/models.py contains no models. "
            "At least one model must be defined.",
            provider="github-copilot",
        )

    # Three-Medium: Validate required keys exist (fail-fast)
    p = _models_data.PROVIDER
    if not p:
        raise ConfigurationError(
            "Config validation failed: config/models.py missing 'PROVIDER' definition.",
            provider="github-copilot",
        )

    defaults = p.get("defaults")
    if not defaults or "model" not in defaults:
        raise ConfigurationError(
            "Config validation failed: config/models.py missing 'PROVIDER[defaults][model]'.",
            provider="github-copilot",
        )

    if "timeout" not in defaults:
        raise ConfigurationError(
            "Config validation failed: config/models.py missing 'PROVIDER[defaults][timeout]'.",
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
# SDK Protection Config Loading
# ============================================================================


@functools.lru_cache(maxsize=1)
def load_sdk_protection_config() -> SdkProtectionConfig:
    """Load SDK protection policy from config/_sdk_protection.py.

    Contract: sdk-protection:ToolCapture:MUST:1,2, sdk-protection:Session:MUST:3,4

    Three-Medium: Python module is authoritative. Returns hardcoded defaults.
    """
    return SdkProtectionConfig()


# =============================================================================
# Model Fallback Values (moved from models.py to avoid circular import A-03)
# Contract: behaviors:ConfigLoading:MUST:1 — fail-fast on missing config
# =============================================================================


@functools.lru_cache(maxsize=1)
def _load_model_fallback_values() -> dict[str, int]:
    """Load fallback values from config/models.py.

    Three-Medium Architecture: Python module is authoritative for fallback values.
    Contract: behaviors:ConfigLoading:MUST:1 — fail-fast on missing config.

    Returns:
        Dict with context_window and max_output_tokens from config/models.py.

    Raises:
        ConfigurationError: If FALLBACKS section is missing or lacks required keys.
    """
    try:
        from .config import _models as _models_data
    except (ImportError, ModuleNotFoundError) as e:
        raise ConfigurationError(
            "config/_models.py not found — broken installation. "
            "Package is missing required configuration module: config/_models.py",
        ) from e

    fallbacks = _models_data.FALLBACKS
    if not isinstance(fallbacks, dict):
        raise ConfigurationError(
            "config/_models.py: FALLBACKS must be a dict. "
            "Broken installation — config/_models.py is corrupted.",
        )
    required_keys = ["context_window", "max_output_tokens"]
    missing = [k for k in required_keys if k not in fallbacks]
    if missing:
        raise ConfigurationError(
            f"config/models.py fallbacks section missing required keys: {missing}. "
            "Three-Medium Architecture: Python module is authoritative for policy values."
        )

    return fallbacks


def get_default_context_window() -> int:
    """Get default context window from config/models.py.

    Contract: behaviors:ConfigLoading:MUST:1 — fail-fast on missing config.

    Returns:
        Default context window value from config/models.py.

    Raises:
        ConfigurationError: If config/models.py is missing or lacks required keys.
    """
    return _load_model_fallback_values()["context_window"]


def get_default_max_output_tokens() -> int:
    """Get default max output tokens from config/models.py.

    Contract: behaviors:ConfigLoading:MUST:1 — fail-fast on missing config.

    Returns:
        Default max_output_tokens value from config/models.py.

    Raises:
        ConfigurationError: If config/models.py is missing or lacks required keys.
    """
    return _load_model_fallback_values()["max_output_tokens"]
