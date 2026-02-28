"""Tests for cost tier mapping and capability fixes in Copilot provider.

Validates:
1. _copilot_cost_tier maps model IDs to correct tiers
2. to_amplifier_model_info populates metadata with cost_tier
3. Fast pattern detection does NOT match '-fast' (expensive variant)
4. Vision capability is preserved from SDK data
"""

import pytest

from amplifier_module_provider_github_copilot.models import (
    CopilotModelInfo,
    _copilot_cost_tier,
    to_amplifier_model_info,
)


# --- 1. Cost tier mapping ---


class TestCopilotCostTier:
    """Verify _copilot_cost_tier returns correct tiers for known models."""

    # 0× free models
    def test_gpt41_is_free(self):
        assert _copilot_cost_tier("gpt-4.1") == "free"

    def test_gpt4o_is_free(self):
        assert _copilot_cost_tier("gpt-4o") == "free"

    def test_gpt5_mini_is_free(self):
        assert _copilot_cost_tier("gpt-5-mini") == "free"

    # 0.25-0.33× low models
    def test_haiku_is_low(self):
        assert _copilot_cost_tier("claude-haiku-4.5") == "low"

    def test_gemini_flash_is_low(self):
        assert _copilot_cost_tier("gemini-flash-2.5") == "low"

    def test_codex_mini_is_low(self):
        assert _copilot_cost_tier("codex-mini-latest") == "low"

    # 1× medium models
    def test_sonnet_is_medium(self):
        assert _copilot_cost_tier("claude-sonnet-4.5") == "medium"

    def test_gpt51_is_medium(self):
        assert _copilot_cost_tier("gpt-5.1") == "medium"

    def test_gpt52_is_medium(self):
        assert _copilot_cost_tier("gpt-5.2") == "medium"

    # 3× high models
    def test_opus_45_is_high(self):
        assert _copilot_cost_tier("claude-opus-4.5") == "high"

    def test_opus_46_is_high(self):
        assert _copilot_cost_tier("claude-opus-4.6") == "high"

    # 30× extreme — fast mode
    def test_opus_46_fast_is_extreme(self):
        assert _copilot_cost_tier("claude-opus-4.6-fast") == "extreme"

    # Default for unknown models
    def test_unknown_model_defaults_to_medium(self):
        assert _copilot_cost_tier("some-future-model-xyz") == "medium"


# --- 2. to_amplifier_model_info includes cost_tier ---


class TestModelInfoCostTier:
    """Verify to_amplifier_model_info populates metadata with cost_tier."""

    def _make_model(self, model_id: str, **kwargs) -> CopilotModelInfo:
        defaults = {
            "id": model_id,
            "name": model_id,
            "provider": "test",
            "context_window": 128000,
            "max_output_tokens": 8192,
        }
        defaults.update(kwargs)
        return CopilotModelInfo(**defaults)

    def test_sonnet_model_has_medium_tier(self):
        model = self._make_model("claude-sonnet-4.5")
        info = to_amplifier_model_info(model)
        assert info.metadata["cost_tier"] == "medium"

    def test_opus_model_has_high_tier(self):
        model = self._make_model("claude-opus-4.6")
        info = to_amplifier_model_info(model)
        assert info.metadata["cost_tier"] == "high"

    def test_haiku_model_has_low_tier(self):
        model = self._make_model("claude-haiku-4.5")
        info = to_amplifier_model_info(model)
        assert info.metadata["cost_tier"] == "low"

    def test_cost_per_token_is_none(self):
        """Copilot doesn't charge per-token."""
        model = self._make_model("claude-sonnet-4.5")
        info = to_amplifier_model_info(model)
        assert info.cost_per_input_token is None
        assert info.cost_per_output_token is None


# --- 3. Fast pattern detection ---


class TestFastPatternDetection:
    """Verify fast_patterns correctly identifies budget models, not premium -fast variants."""

    def test_haiku_is_fast(self):
        model = self._make_model("claude-haiku-4.5")
        info = to_amplifier_model_info(model)
        assert "fast" in info.capabilities

    def test_mini_is_fast(self):
        model = self._make_model("gpt-5-mini")
        info = to_amplifier_model_info(model)
        assert "fast" in info.capabilities

    def test_flash_is_fast(self):
        model = self._make_model("gemini-flash-2.5")
        info = to_amplifier_model_info(model)
        assert "fast" in info.capabilities

    def test_opus_fast_is_NOT_fast(self):
        """The 30× fast mode model should NOT get the 'fast' capability tag."""
        model = self._make_model("claude-opus-4.6-fast")
        info = to_amplifier_model_info(model)
        # '-fast' does not match '-haiku', '-mini', or '-flash'
        assert "fast" not in info.capabilities

    def test_opus_is_NOT_fast(self):
        model = self._make_model("claude-opus-4.6")
        info = to_amplifier_model_info(model)
        assert "fast" not in info.capabilities

    def test_sonnet_is_NOT_fast(self):
        model = self._make_model("claude-sonnet-4.5")
        info = to_amplifier_model_info(model)
        assert "fast" not in info.capabilities

    def _make_model(self, model_id: str, **kwargs) -> CopilotModelInfo:
        defaults = {
            "id": model_id,
            "name": model_id,
            "provider": "test",
            "context_window": 128000,
            "max_output_tokens": 8192,
        }
        defaults.update(kwargs)
        return CopilotModelInfo(**defaults)


# --- 4. Vision from SDK ---


class TestVisionCapability:
    """Verify vision is passed through from SDK data."""

    def _make_model(self, model_id: str, supports_vision: bool = False) -> CopilotModelInfo:
        return CopilotModelInfo(
            id=model_id,
            name=model_id,
            provider="test",
            context_window=128000,
            max_output_tokens=8192,
            supports_vision=supports_vision,
        )

    def test_vision_model_has_vision_cap(self):
        model = self._make_model("claude-sonnet-4.5", supports_vision=True)
        info = to_amplifier_model_info(model)
        assert "vision" in info.capabilities

    def test_non_vision_model_lacks_vision_cap(self):
        model = self._make_model("some-text-model", supports_vision=False)
        info = to_amplifier_model_info(model)
        assert "vision" not in info.capabilities
