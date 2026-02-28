"""
Model mapping and metadata for Copilot SDK provider.

This module handles the conversion between Copilot SDK's ModelInfo
and Amplifier's model representation format.

The module provides:
- Internal model metadata representation
- Known models registry for fallback/caching
- Conversion utilities for Amplifier compatibility

Phase 2: Uses official ModelInfo from amplifier_core with proper capabilities.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

# Use official ModelInfo from amplifier_core
from amplifier_core import ModelInfo

from ._constants import DEFAULT_MODEL

if TYPE_CHECKING:
    from .client import CopilotClientWrapper

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CopilotModelInfo:
    """
    Internal model metadata representation.

    This dataclass captures model information from the Copilot SDK
    in a normalized format suitable for Amplifier integration.

    Attributes:
        id: Unique model identifier (e.g., "claude-opus-4.5")
        name: Human-readable display name
        provider: Upstream provider ("anthropic", "openai", etc.)
        context_window: Maximum context window in tokens
        max_output_tokens: Maximum output tokens per response
        supports_tools: Whether the model supports tool/function calling
        supports_vision: Whether the model supports image inputs
        supports_extended_thinking: Whether the model supports extended reasoning
        supported_reasoning_efforts: List of supported reasoning effort levels
        default_reasoning_effort: Default reasoning effort level
    """

    id: str
    name: str
    provider: str
    context_window: int
    max_output_tokens: int
    supports_tools: bool = True
    supports_vision: bool = False
    supports_extended_thinking: bool = False
    supported_reasoning_efforts: tuple[str, ...] = field(default_factory=tuple)
    default_reasoning_effort: str | None = None


# NOTE: No hardcoded KNOWN_MODELS registry.
# Models are fetched at runtime from the Copilot SDK.
# If SDK call fails, we error out rather than pretending to know what's available.


def copilot_model_to_internal(raw_model: Any) -> CopilotModelInfo:
    """
    Convert Copilot SDK ModelInfo to internal CopilotModelInfo.

    Args:
        raw_model: ModelInfo from Copilot SDK (copilot.types.ModelInfo)

    Returns:
        CopilotModelInfo with normalized metadata
    """
    # Extract capabilities
    context_window = 128000  # Default
    max_output_tokens = 8192  # Default
    supports_vision = False
    supports_reasoning = False

    if hasattr(raw_model, "capabilities") and raw_model.capabilities:
        caps = raw_model.capabilities
        if hasattr(caps, "limits") and caps.limits:
            limits = caps.limits
            if hasattr(limits, "max_context_window_tokens") and limits.max_context_window_tokens:
                context_window = limits.max_context_window_tokens
            if hasattr(limits, "max_prompt_tokens") and limits.max_prompt_tokens:
                # Derive max output from context window minus prompt allocation
                # NOTE: Do NOT cap this value. The SDK provides authoritative limits.
                # Capping causes incorrect budget calculation in context manager.
                max_output_tokens = context_window - limits.max_prompt_tokens

        if hasattr(caps, "supports") and caps.supports:
            supports = caps.supports
            if hasattr(supports, "vision"):
                supports_vision = bool(supports.vision)
            if hasattr(supports, "reasoning_effort"):
                supports_reasoning = bool(supports.reasoning_effort)

    # Determine provider - prefer SDK field, fall back to inference
    model_id = str(raw_model.id) if hasattr(raw_model, "id") else "unknown"
    provider = _get_provider(raw_model, model_id)

    # Get reasoning effort info
    supported_efforts: tuple[str, ...] = ()
    default_effort: str | None = None
    if hasattr(raw_model, "supported_reasoning_efforts") and raw_model.supported_reasoning_efforts:
        supported_efforts = tuple(raw_model.supported_reasoning_efforts)
    if hasattr(raw_model, "default_reasoning_effort"):
        default_effort = raw_model.default_reasoning_effort

    return CopilotModelInfo(
        id=model_id,
        name=str(raw_model.name) if hasattr(raw_model, "name") else model_id,
        provider=provider,
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        supports_tools=True,  # All Copilot models support tools
        supports_vision=supports_vision,
        supports_extended_thinking=supports_reasoning,
        supported_reasoning_efforts=supported_efforts,
        default_reasoning_effort=default_effort,
    )


def _get_provider(raw_model: Any, model_id: str) -> str:
    """
    Get provider from SDK metadata.

    The SDK is the authoritative source for provider information.
    If the SDK doesn't provide it, we return "unknown" rather than
    maintaining fragile pattern-matching that requires updates
    whenever new providers are added.

    Args:
        raw_model: Raw ModelInfo from Copilot SDK
        model_id: Model identifier string

    Returns:
        Provider name from SDK, or "unknown" if not provided
    """
    # Try to get provider directly from SDK
    if hasattr(raw_model, "provider") and raw_model.provider:
        provider = str(raw_model.provider).lower()
        logger.debug(f"[MODELS] Got provider '{provider}' from SDK for {model_id}")
        return provider

    # Check vendor field (alternative SDK naming)
    if hasattr(raw_model, "vendor") and raw_model.vendor:
        vendor = str(raw_model.vendor).lower()
        logger.debug(f"[MODELS] Got vendor '{vendor}' from SDK for {model_id}")
        return vendor

    # SDK didn't provide provider - log at debug level and return unknown
    # This is informational, not actionable by users. The provider field
    # is used for capability inference but the module works without it.
    logger.debug(
        f"[MODELS] SDK did not provide provider for model '{model_id}'. "
        f"Provider field will be 'unknown'."
    )
    return "unknown"


def to_amplifier_model_info(model: CopilotModelInfo) -> ModelInfo:
    """
    Convert CopilotModelInfo to Amplifier's official ModelInfo.

    Uses the official ModelInfo class from amplifier_core which Amplifier
    expects for proper capability display (streaming, tools, thinking, etc).

    Args:
        model: Internal CopilotModelInfo instance

    Returns:
        ModelInfo from amplifier_core with proper capabilities
    """
    # Build capabilities list matching official provider patterns
    capabilities = ["streaming"]  # All Copilot models support streaming

    if model.supports_tools:
        capabilities.append("tools")
    if model.supports_vision:
        capabilities.append("vision")

    # Extended thinking support - ONLY if SDK explicitly reports it
    # Do NOT infer thinking capability - the SDK knows which models support it
    # and inferring causes errors when Amplifier enables extended_thinking
    # for models that don't actually support reasoning_effort
    if model.supports_extended_thinking:
        # Use "thinking" for Claude, "reasoning" for OpenAI naming
        if "claude" in model.id.lower():
            capabilities.append("thinking")
        else:
            capabilities.append("reasoning")

    # Mark fast models (check for actual patterns from SDK evidence)
    model_lower = model.id.lower()
    fast_patterns = ["-haiku", "-mini", "-flash"]
    if any(pattern in model_lower for pattern in fast_patterns):
        capabilities.append("fast")

    logger.debug(f"[MODELS] {model.id}: capabilities={capabilities}")

    result = ModelInfo(
        id=model.id,
        display_name=model.name,
        context_window=model.context_window,
        max_output_tokens=model.max_output_tokens,
        capabilities=capabilities,
        defaults={
            "temperature": 0.7,
            "max_tokens": min(16384, model.max_output_tokens),
        },
    )
    logger.debug(f"[MODELS] {model.id}: type={type(result).__name__}, caps={result.capabilities}")
    return result


async def fetch_and_map_models(client: CopilotClientWrapper) -> list[ModelInfo]:
    """
    Fetch models from Copilot SDK and map to Amplifier format.

    This function:
    1. Calls the Copilot SDK to get available models
    2. Converts each to internal format
    3. Maps to official Amplifier ModelInfo format

    Raises an error if the SDK call fails - no fallback to hardcoded models.

    Args:
        client: CopilotClientWrapper instance

    Returns:
        List of ModelInfo objects (official amplifier_core type)

    Raises:
        CopilotProviderError: If SDK call fails or returns no models
    """
    from .exceptions import CopilotProviderError

    try:
        copilot_client = await client.ensure_client()
        raw_models = await copilot_client.list_models()
        logger.debug(f"[MODELS] Got {len(raw_models)} raw models from SDK")

        result = []
        for raw_model in raw_models:
            try:
                internal = copilot_model_to_internal(raw_model)
                model_info = to_amplifier_model_info(internal)
                result.append(model_info)
            except Exception as e:
                logger.warning(f"[MODELS] Failed to convert model: {e}")
                continue

        if not result:
            raise CopilotProviderError(
                "Copilot SDK returned no models. Check authentication and connectivity."
            )

        logger.info(f"[MODELS] Fetched {len(result)} models from Copilot SDK")
        return result

    except CopilotProviderError:
        raise
    except Exception as e:
        raise CopilotProviderError(
            f"Failed to fetch models from Copilot SDK: {e}. "
            "Ensure Copilot CLI is running and authenticated."
        ) from e


def get_default_model() -> str:
    """
    Get the default model ID.

    Returns:
        Default model identifier
    """
    return DEFAULT_MODEL
