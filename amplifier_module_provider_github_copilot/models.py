"""Model discovery and type translation.

Contract: contracts/sdk-boundary.md (ModelDiscovery section)

Three-Medium Architecture:
- Python: Type translation logic (this module)
- YAML: Fallback policy values (config/models.yaml)
- Markdown: Requirements (contracts/sdk-boundary.md)

Type Translation Chain:
    SDK ModelInfo → CopilotModelInfo → amplifier_core.ModelInfo
    (copilot.types)   (isolation layer)   (kernel contract)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

# Import amplifier_core.ModelInfo (provided by Amplifier runtime)
try:
    from amplifier_core import (
        ModelInfo as AmplifierModelInfo,  # pyright: ignore[reportAssignmentType]
    )
    from amplifier_core import ProviderUnavailableError  # pyright: ignore[reportAssignmentType]
except ImportError:
    # Fallback for standalone testing
    from pydantic import BaseModel, Field

    class AmplifierModelInfo(BaseModel):  # type: ignore[no-redef]
        """Fallback when amplifier_core unavailable."""

        id: str
        display_name: str
        context_window: int
        max_output_tokens: int
        capabilities: list[str] = Field(default_factory=list)
        defaults: dict[str, Any] = Field(default_factory=dict)

    class ProviderUnavailableError(Exception):  # type: ignore[no-redef]
        """Fallback when amplifier_core unavailable."""

        def __init__(self, message: str, *, provider: str = "github-copilot") -> None:
            super().__init__(message)
            self.provider = provider


# =============================================================================
# Fallback Policy Values (from models.yaml)
# Three-Medium Architecture: Python calls YAML for policy values
# Contract: behaviors:ConfigLoading:MUST:1 — YAML authoritative, fail-fast on missing
# =============================================================================


# Single import point for ConfigurationError (Three-Medium Architecture)
# Contract: sdk-boundary:Membrane:MUST:1 — Single import point for runtime dependencies
from amplifier_module_provider_github_copilot._compat import ConfigurationError

# Re-export fallback functions from config_loader for backward compatibility
# (They were moved there to fix circular import A-03)
from amplifier_module_provider_github_copilot.config_loader import (
    get_default_context_window,
    get_default_max_output_tokens,
)

# =============================================================================
# Re-export from sdk_adapter membrane (NOT direct module import)
# R6 Fix: Import via membrane __init__.py, not bypassing to model_translation.py
# Contract: sdk-boundary:ModelDiscovery:MUST:2
# =============================================================================
from .sdk_adapter import (  # noqa: E402
    CopilotModelInfo,
    sdk_model_to_copilot_model,
)

# Re-export for backward compatibility
__all__ = [
    "CopilotModelInfo",
    "sdk_model_to_copilot_model",
    "copilot_model_to_amplifier_model",
    "fetch_models",
    "get_default_context_window",
    "get_default_max_output_tokens",
    "ConfigurationError",
]


# =============================================================================
# CopilotModelInfo → amplifier_core.ModelInfo Translation
# Contract: sdk-boundary:ModelDiscovery:MUST:3
# =============================================================================


def copilot_model_to_amplifier_model(model: CopilotModelInfo) -> AmplifierModelInfo:
    """Translate CopilotModelInfo to amplifier_core.ModelInfo.

    Contract: sdk-boundary:ModelDiscovery:MUST:3
    - MUST translate CopilotModelInfo to amplifier_core.ModelInfo (kernel contract)
    - MUST map: id, display_name, context_window, max_output_tokens, capabilities

    Args:
        model: CopilotModelInfo domain type

    Returns:
        amplifier_core.ModelInfo (what kernel expects from provider.list_models())
    """
    # Build capabilities list
    capabilities: list[str] = ["streaming", "tools"]  # All Copilot models support these

    if model.supports_vision:
        capabilities.append("vision")

    if model.supports_reasoning_effort:
        capabilities.append("thinking")

    # Build defaults dict (model-specific config)
    defaults: dict[str, Any] = {}

    if model.default_reasoning_effort is not None:
        defaults["reasoning_effort"] = model.default_reasoning_effort

    if model.supported_reasoning_efforts:
        defaults["supported_reasoning_efforts"] = list(model.supported_reasoning_efforts)

    return AmplifierModelInfo(
        id=model.id,
        display_name=model.name,
        context_window=model.context_window,
        max_output_tokens=model.max_output_tokens,
        capabilities=capabilities,
        defaults=defaults,
    )


# =============================================================================
# SDK Fetch → CopilotModelInfo List
# Contract: sdk-boundary:ModelDiscovery:MUST:1
# =============================================================================


async def fetch_models(client: Any) -> list[CopilotModelInfo]:
    """Fetch models from SDK and translate to CopilotModelInfo.

    Contract: sdk-boundary:ModelDiscovery:MUST:1
    - MUST fetch models from SDK list_models() API

    Contract: behaviors:ModelDiscoveryError:MUST:1
    - MUST raise ProviderUnavailableError when SDK unavailable AND no cache

    Args:
        client: SDK CopilotClient or CopilotClientWrapper with list_models() method

    Returns:
        List of CopilotModelInfo domain types

    Raises:
        ProviderUnavailableError: When SDK call fails (behaviors:ModelDiscoveryError:MUST:1)
    """
    try:
        sdk_models: Sequence[Any] = await client.list_models()
        return [sdk_model_to_copilot_model(m) for m in sdk_models]
    except Exception as exc:
        # Contract: behaviors:ModelDiscoveryError:MUST:2
        # Error message MUST include reason for failure
        # Contract: behaviors:Logging:MUST:4 - Redact sensitive text
        from .security_redaction import redact_sensitive_text

        raise ProviderUnavailableError(
            f"Failed to fetch models from SDK: {redact_sensitive_text(exc)}. "
            "SDK connection unavailable, no cached models available.",
            provider="github-copilot",
        ) from exc


# =============================================================================
# Convenience: Full Translation Chain
# =============================================================================


async def fetch_and_map_models(
    client: Any,
) -> tuple[list[AmplifierModelInfo], list[CopilotModelInfo]]:
    """Fetch models from SDK and translate to amplifier_core.ModelInfo.

    Convenience function that chains:
        SDK ModelInfo → CopilotModelInfo → amplifier_core.ModelInfo

    Also returns raw CopilotModelInfo for caching to avoid double SDK fetch.

    Args:
        client: SDK CopilotClient or CopilotClientWrapper

    Returns:
        Tuple of:
        - List of amplifier_core.ModelInfo (what kernel expects)
        - List of CopilotModelInfo (for caching to disk)
    """
    copilot_models = await fetch_models(client)
    amplifier_models = [copilot_model_to_amplifier_model(m) for m in copilot_models]
    return amplifier_models, copilot_models
