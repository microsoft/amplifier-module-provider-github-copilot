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
        context_window: SDK display ceiling (limits.max_context_window_tokens).
            Use for the model-list UI surface; this is NOT a compaction budget.
        max_output_tokens: Maximum output tokens per response
        supports_vision: Whether the model supports image inputs
        supports_reasoning_effort: Whether the model supports reasoning effort
        supported_reasoning_efforts: Tuple of supported reasoning effort levels
        default_reasoning_effort: Default reasoning effort level
        context_window_default: Default-tier prompt budget (max_prompt_tokens),
            from billing.token_prices.context_max ELSE limits.max_prompt_tokens.
            Feeds the compaction budget when the long tier is off. 0 => unknown.
        context_window_long: Long-tier prompt budget, from
            billing.token_prices.long_context.context_max ELSE context_window_default.
            0 => no long tier (enable_long_context becomes a no-op for the model).
    """

    id: str
    name: str
    context_window: int
    max_output_tokens: int
    supports_vision: bool = False
    supports_reasoning_effort: bool = False
    supported_reasoning_efforts: tuple[str, ...] = ()
    default_reasoning_effort: str | None = None
    context_window_default: int = 0
    context_window_long: int = 0


def resolve_effective_window(info: CopilotModelInfo, enable_long_context: bool) -> int:
    """Return the prompt-budget window for the active context tier.

    The long-tier budget is selected when ``enable_long_context`` is set,
    otherwise the default-tier budget. Returns 0 when the budget is unknown
    (a pre-tier cache read); the caller treats 0 as "keep the static policy
    window" so the display ceiling is never reported as a compaction budget.

    Contract: provider-protocol:get_info:MUST:5
    """
    return info.context_window_long if enable_long_context else info.context_window_default


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
        sdk_model: SDK ModelInfo object (from copilot.client)

    Returns:
        CopilotModelInfo domain type

    Note:
        Uses duck-typing for SDK type access (SDK objects have this structure).
        Fallback values from policy config when SDK returns None.
    """
    # Extract capabilities using duck-typing (SDK type structure)
    # Guard against None capabilities (SDK may return partial model info)
    capabilities = sdk_model.capabilities
    max_prompt_tokens: int | None = None
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

    # Extract reasoning effort fields. ``getattr`` guards against SDK
    # versions/mocks/partial-shape objects that may omit these attributes;
    # without it an AttributeError on ``list_models()`` would kill model
    # discovery for the entire process and silently fall the cache to the
    # shape-only Layer-1 gate.
    supported_efforts = getattr(sdk_model, "supported_reasoning_efforts", None)
    supported_reasoning_efforts: tuple[str, ...] = ()
    if supported_efforts is not None:
        supported_reasoning_efforts = tuple(supported_efforts)
    default_reasoning_effort = getattr(sdk_model, "default_reasoning_effort", None)

    # Per-tier PROMPT budgets from the SDK billing surface. The SDK separates
    # the display ceiling (context_window, above) from the billed prompt budget
    # per tier; compaction needs the latter. billing is Optional and present
    # only on tiered models — None-walk each level (mirrors the limits reads).
    billing = getattr(sdk_model, "billing", None)
    billing_default: int | None = None
    billing_long: int | None = None
    if billing is not None:
        token_prices = billing.token_prices
        if token_prices is not None:
            billing_default = token_prices.context_max
            long_prices = token_prices.long_context
            if long_prices is not None:
                billing_long = long_prices.context_max

    # Default-tier budget: billed prompt budget ELSE the limits' max_prompt_tokens
    # ELSE the static policy fallback. Long-tier budget collapses to default when
    # the model has no long_context billing entry.
    context_window_default = billing_default
    if context_window_default is None:
        context_window_default = max_prompt_tokens
    if context_window_default is None:
        context_window_default = get_default_context_window()
    context_window_long = billing_long if billing_long is not None else context_window_default

    return CopilotModelInfo(
        id=sdk_model.id,
        name=sdk_model.name,
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        supports_vision=supports_vision,
        supports_reasoning_effort=supports_reasoning_effort,
        supported_reasoning_efforts=supported_reasoning_efforts,
        default_reasoning_effort=default_reasoning_effort,
        context_window_default=context_window_default,
        context_window_long=context_window_long,
    )
