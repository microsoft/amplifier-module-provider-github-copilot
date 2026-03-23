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

from dataclasses import dataclass
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

import functools
import importlib.resources


class ConfigurationError(Exception):
    """Raised when required configuration is missing or invalid.

    Contract: behaviors:ConfigLoading:MUST:1 — fail-fast on missing config.
    """

    pass


@functools.lru_cache(maxsize=1)
def _load_fallback_values() -> dict[str, int]:
    """Load fallback values from config/models.yaml.

    Three-Medium Architecture: Python loads policy from YAML.
    Contract: behaviors:ConfigLoading:MUST:1 — fail-fast on missing config.

    Returns:
        Dict with context_window and max_output_tokens from YAML.

    Raises:
        ConfigurationError: If models.yaml is missing or lacks required keys.
    """
    import yaml

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
    return _load_fallback_values()["context_window"]


def get_default_max_output_tokens() -> int:
    """Get default max output tokens from YAML config.

    Contract: behaviors:ConfigLoading:MUST:1 — fail-fast on missing config.

    Returns:
        Default max_output_tokens value from models.yaml.

    Raises:
        ConfigurationError: If models.yaml is missing or lacks required keys.
    """
    return _load_fallback_values()["max_output_tokens"]


# =============================================================================
# Domain Type: CopilotModelInfo (Isolation Layer)
# Contract: sdk-boundary:ModelDiscovery:MUST:2
# =============================================================================


@dataclass(frozen=True)
class CopilotModelInfo:
    """Internal representation between SDK and Amplifier domains.

    Isolates SDK type structure from Amplifier ModelInfo.
    Enables independent evolution of both interfaces.

    Contract: sdk-boundary:ModelDiscovery:MUST:2
    - MUST translate SDK ModelInfo to domain CopilotModelInfo (isolation layer)

    Attributes:
        id: Model identifier (e.g., "claude-opus-4.5")
        name: Human-readable display name
        context_window: Maximum context window in tokens
        max_output_tokens: Maximum output tokens per response
        supports_vision: Whether the model supports image inputs
        supports_reasoning_effort: Whether the model supports reasoning effort
        supported_reasoning_efforts: Tuple of supported reasoning effort levels
        default_reasoning_effort: Default reasoning effort level
    """

    id: str
    name: str
    context_window: int
    max_output_tokens: int
    supports_vision: bool = False
    supports_reasoning_effort: bool = False
    supported_reasoning_efforts: tuple[str, ...] = ()
    default_reasoning_effort: str | None = None


# =============================================================================
# SDK ModelInfo → CopilotModelInfo Translation
# Contract: sdk-boundary:ModelDiscovery:MUST:2
# =============================================================================


def sdk_model_to_copilot_model(sdk_model: Any) -> CopilotModelInfo:
    """Translate SDK ModelInfo to domain CopilotModelInfo.

    Contract: sdk-boundary:ModelDiscovery:MUST:2
    - MUST extract context_window from SDK capabilities.limits.max_context_window_tokens
    - MUST derive max_output_tokens as context_window - max_prompt_tokens

    Args:
        sdk_model: SDK ModelInfo object (from copilot.types)

    Returns:
        CopilotModelInfo domain type

    Note:
        Uses duck-typing for SDK type access (no SDK import outside sdk_adapter/).
        Fallback values from policy config when SDK returns None.
    """
    # Extract capabilities using duck-typing (SDK type structure)
    capabilities = sdk_model.capabilities
    limits = capabilities.limits
    supports = capabilities.supports

    # Extract limits with fallback to policy defaults from YAML
    # Contract: behaviors:ConfigLoading:MUST:1 — YAML authoritative
    context_window = limits.max_context_window_tokens
    max_prompt_tokens = limits.max_prompt_tokens

    if context_window is None:
        context_window = get_default_context_window()

    # Derive max_output_tokens: context_window - max_prompt_tokens
    if max_prompt_tokens is not None:
        max_output_tokens = context_window - max_prompt_tokens
    else:
        max_output_tokens = get_default_max_output_tokens()

    # Ensure max_output_tokens is positive (safety check)
    if max_output_tokens <= 0:
        max_output_tokens = get_default_max_output_tokens()

    # Extract reasoning effort fields
    supported_efforts = sdk_model.supported_reasoning_efforts
    supported_reasoning_efforts: tuple[str, ...] = ()
    if supported_efforts is not None:
        supported_reasoning_efforts = tuple(supported_efforts)

    return CopilotModelInfo(
        id=sdk_model.id,
        name=sdk_model.name,
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        supports_vision=supports.vision,
        supports_reasoning_effort=supports.reasoning_effort,
        supported_reasoning_efforts=supported_reasoning_efforts,
        default_reasoning_effort=sdk_model.default_reasoning_effort,
    )


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
        raise ProviderUnavailableError(
            f"Failed to fetch models from SDK: {exc}. "
            "SDK connection unavailable, no cached models available.",
            provider="github-copilot",
        ) from exc


# =============================================================================
# Convenience: Full Translation Chain
# =============================================================================


async def fetch_and_map_models(client: Any) -> list[AmplifierModelInfo]:
    """Fetch models from SDK and translate to amplifier_core.ModelInfo.

    Convenience function that chains:
        SDK ModelInfo → CopilotModelInfo → amplifier_core.ModelInfo

    Args:
        client: SDK CopilotClient or CopilotClientWrapper

    Returns:
        List of amplifier_core.ModelInfo (what kernel expects)
    """
    copilot_models = await fetch_models(client)
    return [copilot_model_to_amplifier_model(m) for m in copilot_models]
