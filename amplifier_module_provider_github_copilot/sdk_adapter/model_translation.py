"""SDK Model Translation — Inside the Membrane.

Contract: sdk-boundary:ModelDiscovery:MUST:2

This module lives inside the SDK membrane and handles translation of
SDK ModelInfo objects to domain CopilotModelInfo objects.

Three-Medium Architecture:
- Python: Translation logic (this module)
- YAML: Fallback policy values (config/models.yaml)
- Markdown: Requirements (contracts/sdk-boundary.md)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Import from config_loader to avoid circular import with models.py (A-03 fix)
from ..config_loader import get_default_context_window, get_default_max_output_tokens

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

    This function lives inside the membrane (sdk_adapter/) because it directly
    accesses SDK object structure via duck-typing.

    Args:
        sdk_model: SDK ModelInfo object (from copilot.types)

    Returns:
        CopilotModelInfo domain type

    Note:
        Uses duck-typing for SDK type access (SDK objects have this structure).
        Fallback values from policy config when SDK returns None.
    """
    # Extract capabilities using duck-typing (SDK type structure)
    # Guard against None capabilities (SDK may return partial model info)
    capabilities = sdk_model.capabilities
    if capabilities is None:
        # SDK returned no capabilities - use all defaults
        context_window = get_default_context_window()
        max_output_tokens = get_default_max_output_tokens()
        supports_vision = False
        supports_reasoning_effort = False
    else:
        limits = capabilities.limits
        supports = capabilities.supports

        # Guard against None limits/supports within capabilities
        if limits is None:
            context_window = get_default_context_window()
            max_prompt_tokens = None
        else:
            # Extract limits with fallback to policy defaults from YAML
            # Contract: behaviors:ConfigLoading:MUST:1 — YAML authoritative
            context_window = limits.max_context_window_tokens
            max_prompt_tokens = limits.max_prompt_tokens

            if context_window is None:
                context_window = get_default_context_window()

        # Derive max_output_tokens: context_window - max_prompt_tokens
        if limits is not None and max_prompt_tokens is not None:
            max_output_tokens = context_window - max_prompt_tokens
        else:
            max_output_tokens = get_default_max_output_tokens()

        # Ensure max_output_tokens is positive (safety check)
        if max_output_tokens <= 0:
            max_output_tokens = get_default_max_output_tokens()

        # Extract supports flags with None guard
        if supports is None:
            supports_vision = False
            supports_reasoning_effort = False
        else:
            supports_vision = supports.vision if supports.vision is not None else False
            supports_reasoning_effort = (
                supports.reasoning_effort if supports.reasoning_effort is not None else False
            )

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
        supports_vision=supports_vision,
        supports_reasoning_effort=supports_reasoning_effort,
        supported_reasoning_efforts=supported_reasoning_efforts,
        default_reasoning_effort=sdk_model.default_reasoning_effort,
    )
