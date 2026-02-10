"""
Tests for model mapping and metadata.

This module tests the model conversion between Copilot SDK format
and Amplifier format.
"""

from unittest.mock import AsyncMock, Mock

import pytest
from amplifier_core import ModelInfo

from amplifier_module_provider_github_copilot.exceptions import CopilotProviderError
from amplifier_module_provider_github_copilot.models import (
    CopilotModelInfo,
    copilot_model_to_internal,
    fetch_and_map_models,
    get_default_model,
    to_amplifier_model_info,
)


class TestCopilotModelInfo:
    """Tests for CopilotModelInfo dataclass."""

    def test_create_model_info(self):
        """Should create CopilotModelInfo with required fields."""
        model = CopilotModelInfo(
            id="test-model",
            name="Test Model",
            provider="test-provider",
            context_window=100000,
            max_output_tokens=8192,
        )

        assert model.id == "test-model"
        assert model.name == "Test Model"
        assert model.provider == "test-provider"
        assert model.context_window == 100000
        assert model.max_output_tokens == 8192
        assert model.supports_tools is True  # Default
        assert model.supports_vision is False  # Default
        assert model.supports_extended_thinking is False  # Default

    def test_model_info_with_all_fields(self):
        """Should create CopilotModelInfo with all optional fields."""
        model = CopilotModelInfo(
            id="advanced-model",
            name="Advanced Model",
            provider="advanced",
            context_window=200000,
            max_output_tokens=16384,
            supports_tools=True,
            supports_vision=True,
            supports_extended_thinking=True,
            supported_reasoning_efforts=("low", "medium", "high"),
            default_reasoning_effort="medium",
        )

        assert model.supports_tools is True
        assert model.supports_vision is True
        assert model.supports_extended_thinking is True
        assert model.supported_reasoning_efforts == ("low", "medium", "high")
        assert model.default_reasoning_effort == "medium"

    def test_model_info_is_frozen(self):
        """CopilotModelInfo should be immutable."""
        model = CopilotModelInfo(
            id="test",
            name="Test",
            provider="test",
            context_window=100000,
            max_output_tokens=8192,
        )

        with pytest.raises(AttributeError):
            model.id = "changed"


class TestCopilotModelToInternal:
    """Tests for copilot_model_to_internal converter."""

    def test_convert_basic_model(self):
        """Should convert basic model info."""
        raw_model = Mock()
        raw_model.id = "test-model"
        raw_model.name = "Test Model"
        raw_model.capabilities = Mock()
        raw_model.capabilities.supports = Mock()
        raw_model.capabilities.supports.vision = True
        raw_model.capabilities.supports.reasoning_effort = False
        raw_model.capabilities.limits = Mock()
        raw_model.capabilities.limits.max_context_window_tokens = 128000
        raw_model.capabilities.limits.max_prompt_tokens = 100000
        raw_model.supported_reasoning_efforts = None
        raw_model.default_reasoning_effort = None

        result = copilot_model_to_internal(raw_model)

        assert result.id == "test-model"
        assert result.name == "Test Model"
        assert result.context_window == 128000
        assert result.supports_vision is True
        assert result.supports_extended_thinking is False

    def test_provider_from_sdk_field(self):
        """Should use provider field from SDK when available."""
        raw_model = Mock()
        raw_model.id = "custom-model"
        raw_model.name = "Custom Model"
        raw_model.provider = "anthropic"  # SDK provides this directly
        raw_model.vendor = None
        raw_model.capabilities = None
        raw_model.supported_reasoning_efforts = None
        raw_model.default_reasoning_effort = None

        result = copilot_model_to_internal(raw_model)

        assert result.provider == "anthropic"

    def test_provider_from_vendor_field(self):
        """Should use vendor field from SDK when provider is not available."""
        raw_model = Mock()
        raw_model.id = "custom-model"
        raw_model.name = "Custom Model"
        raw_model.provider = None
        raw_model.vendor = "google"  # SDK provides vendor instead
        raw_model.capabilities = None
        raw_model.supported_reasoning_efforts = None
        raw_model.default_reasoning_effort = None

        result = copilot_model_to_internal(raw_model)

        assert result.provider == "google"

    def test_provider_prefers_provider_over_vendor(self):
        """Should prefer provider field over vendor field."""
        raw_model = Mock()
        raw_model.id = "some-model"
        raw_model.name = "Some Model"
        raw_model.provider = "openai"
        raw_model.vendor = "google"  # Should be ignored
        raw_model.capabilities = None
        raw_model.supported_reasoning_efforts = None
        raw_model.default_reasoning_effort = None

        result = copilot_model_to_internal(raw_model)

        assert result.provider == "openai"

    def test_unknown_when_sdk_lacks_provider(self):
        """Should return 'unknown' when SDK doesn't provide provider info."""
        raw_model = Mock()
        raw_model.id = "some-new-model-xyz"
        raw_model.name = "Some New Model"
        raw_model.provider = None
        raw_model.vendor = None
        raw_model.capabilities = None
        raw_model.supported_reasoning_efforts = None
        raw_model.default_reasoning_effort = None

        result = copilot_model_to_internal(raw_model)

        assert result.provider == "unknown"


class TestToAmplifierModelInfo:
    """Tests for to_amplifier_model_info converter."""

    def test_convert_to_amplifier_format(self):
        """Should convert to Amplifier ModelInfo format."""
        model = CopilotModelInfo(
            id="test-model",
            name="Test Model",
            provider="test",
            context_window=100000,
            max_output_tokens=8192,
            supports_tools=True,
            supports_vision=True,
            supports_extended_thinking=True,
            supported_reasoning_efforts=("low", "medium"),
            default_reasoning_effort="low",
        )

        result = to_amplifier_model_info(model)

        # Now returns official ModelInfo from amplifier_core
        assert isinstance(result, ModelInfo)
        assert result.id == "test-model"
        assert result.display_name == "Test Model"
        assert result.context_window == 100000
        assert result.max_output_tokens == 8192
        # Capabilities are list of strings
        assert "streaming" in result.capabilities
        assert "tools" in result.capabilities
        assert "vision" in result.capabilities


class TestFetchAndMapModels:
    """Tests for fetch_and_map_models function."""

    @pytest.mark.asyncio
    async def test_fetch_models_from_sdk(self, mock_copilot_client):
        """Should fetch models from SDK and convert."""
        mock_wrapper = Mock()
        mock_wrapper.ensure_client = AsyncMock(return_value=mock_copilot_client)

        result = await fetch_and_map_models(mock_wrapper)

        assert isinstance(result, list)
        assert len(result) > 0
        # Returns ModelInfo objects now
        assert isinstance(result[0], ModelInfo)
        assert result[0].id == "claude-opus-4.5"

    @pytest.mark.asyncio
    async def test_raises_error_on_sdk_failure(self):
        """Should raise CopilotProviderError when SDK fails (no fallback)."""
        mock_wrapper = Mock()
        mock_wrapper.ensure_client = AsyncMock(side_effect=Exception("SDK error"))

        with pytest.raises(CopilotProviderError) as exc_info:
            await fetch_and_map_models(mock_wrapper)

        assert "Failed to fetch models" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_raises_error_on_empty_models(self, mock_copilot_client):
        """Should raise CopilotProviderError when SDK returns no models."""
        mock_copilot_client.list_models = AsyncMock(return_value=[])
        mock_wrapper = Mock()
        mock_wrapper.ensure_client = AsyncMock(return_value=mock_copilot_client)

        with pytest.raises(CopilotProviderError) as exc_info:
            await fetch_and_map_models(mock_wrapper)

        assert "returned no models" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_skips_unconvertible_models(self):
        """Should skip models that fail conversion and continue with valid ones."""
        from unittest.mock import patch

        # Create two mock raw models
        bad_model = Mock()
        bad_model.id = "bad-model"
        bad_model.name = "Bad Model"

        good_model = Mock()
        good_model.id = "good-model"
        good_model.name = "Good Model"

        mock_client = Mock()
        mock_client.list_models = AsyncMock(return_value=[bad_model, good_model])

        mock_wrapper = Mock()
        mock_wrapper.ensure_client = AsyncMock(return_value=mock_client)

        # Patch copilot_model_to_internal to fail on first call, succeed on second
        call_count = 0

        def mock_convert(raw_model):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Simulated conversion error")
            # Return valid internal model for second call
            return CopilotModelInfo(
                id="good-model",
                name="Good Model",
                provider="test",
                context_window=100000,
                max_output_tokens=4096,
            )

        with patch(
            "amplifier_module_provider_github_copilot.models.copilot_model_to_internal",
            side_effect=mock_convert,
        ):
            result = await fetch_and_map_models(mock_wrapper)

        # Should have only the good model
        assert len(result) == 1
        assert result[0].id == "good-model"


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_get_default_model(self):
        """Should return default model ID."""
        default = get_default_model()
        assert default == "claude-opus-4.5"


class TestCapabilityInferenceRegression:
    """
    Regression tests for capability inference.

    These tests ensure we NEVER advertise capabilities that the SDK
    doesn't explicitly support. The Amplifier orchestrator uses
    capabilities to decide which kwargs to pass to complete().

    Bug fixed: We were inferring "thinking" capability from model name
    (e.g., "claude-opus" → thinking) but Copilot SDK said model doesn't
    support reasoning_effort. When Amplifier passed extended_thinking=True,
    the SDK rejected: "Model does not support reasoning effort configuration."
    """

    def test_no_thinking_capability_when_sdk_says_no(self):
        """
        REGRESSION: Don't advertise 'thinking' if SDK says reasoning_effort=False.

        This is the exact bug that was found in production via Amplifier.
        """
        # SDK says Claude Opus 4.5 does NOT support reasoning
        raw_model = Mock()
        raw_model.id = "claude-opus-4.5"
        raw_model.name = "Claude Opus 4.5"
        raw_model.provider = None
        raw_model.vendor = None
        raw_model.capabilities = Mock()
        raw_model.capabilities.supports = Mock()
        raw_model.capabilities.supports.vision = True
        raw_model.capabilities.supports.reasoning_effort = False  # SDK says NO
        raw_model.capabilities.limits = Mock()
        raw_model.capabilities.limits.max_context_window_tokens = 200000
        raw_model.capabilities.limits.max_prompt_tokens = 150000
        raw_model.supported_reasoning_efforts = None
        raw_model.default_reasoning_effort = None

        internal = copilot_model_to_internal(raw_model)
        result = to_amplifier_model_info(internal)

        # CRITICAL: "thinking" should NOT be in capabilities
        assert "thinking" not in result.capabilities, (
            "BUG REGRESSION: 'thinking' capability advertised but SDK says "
            "reasoning_effort=False. This will cause Amplifier to pass "
            "extended_thinking=True and crash the session."
        )

    def test_no_thinking_capability_for_any_model_without_sdk_support(self):
        """Don't infer thinking for ANY model if SDK doesn't confirm support."""
        test_cases = [
            ("claude-opus-4.5", "Claude Opus 4.5"),
            ("claude-sonnet-4-5", "Claude Sonnet 4.5"),
            ("gpt-5.2", "GPT 5.2"),
            ("o3-preview", "O3 Preview"),
            ("gemini-3-pro", "Gemini 3 Pro"),
        ]

        for model_id, model_name in test_cases:
            raw_model = Mock()
            raw_model.id = model_id
            raw_model.name = model_name
            raw_model.provider = None
            raw_model.vendor = None
            raw_model.capabilities = Mock()
            raw_model.capabilities.supports = Mock()
            raw_model.capabilities.supports.vision = True
            raw_model.capabilities.supports.reasoning_effort = False  # SDK says NO
            raw_model.capabilities.limits = Mock()
            raw_model.capabilities.limits.max_context_window_tokens = 200000
            raw_model.capabilities.limits.max_prompt_tokens = 150000
            raw_model.supported_reasoning_efforts = None
            raw_model.default_reasoning_effort = None

            internal = copilot_model_to_internal(raw_model)
            result = to_amplifier_model_info(internal)

            assert "thinking" not in result.capabilities, (
                f"Model {model_id}: 'thinking' should not be in capabilities "
                f"when SDK says reasoning_effort=False"
            )
            assert "reasoning" not in result.capabilities, (
                f"Model {model_id}: 'reasoning' should not be in capabilities "
                f"when SDK says reasoning_effort=False"
            )

    def test_thinking_capability_when_sdk_supports_it(self):
        """DO advertise 'thinking' when SDK explicitly says yes."""
        raw_model = Mock()
        raw_model.id = "claude-opus-4.5"
        raw_model.name = "Claude Opus 4.5"
        raw_model.provider = None
        raw_model.vendor = None
        raw_model.capabilities = Mock()
        raw_model.capabilities.supports = Mock()
        raw_model.capabilities.supports.vision = True
        raw_model.capabilities.supports.reasoning_effort = True  # SDK says YES
        raw_model.capabilities.limits = Mock()
        raw_model.capabilities.limits.max_context_window_tokens = 200000
        raw_model.capabilities.limits.max_prompt_tokens = 150000
        raw_model.supported_reasoning_efforts = ["low", "medium", "high"]
        raw_model.default_reasoning_effort = "medium"

        internal = copilot_model_to_internal(raw_model)
        result = to_amplifier_model_info(internal)

        # Should have thinking because SDK says yes
        assert "thinking" in result.capabilities, (
            "Model with SDK reasoning_effort=True should have 'thinking' capability"
        )

    def test_reasoning_capability_for_openai_models_when_supported(self):
        """OpenAI-prefixed models should get 'reasoning' not 'thinking' when supported."""
        raw_model = Mock()
        raw_model.id = "o3-reasoning"
        raw_model.name = "O3 Reasoning"
        raw_model.provider = "openai"
        raw_model.vendor = None
        raw_model.capabilities = Mock()
        raw_model.capabilities.supports = Mock()
        raw_model.capabilities.supports.vision = True
        raw_model.capabilities.supports.reasoning_effort = True  # SDK says YES
        raw_model.capabilities.limits = Mock()
        raw_model.capabilities.limits.max_context_window_tokens = 200000
        raw_model.capabilities.limits.max_prompt_tokens = 150000
        raw_model.supported_reasoning_efforts = ["low", "medium", "high"]
        raw_model.default_reasoning_effort = "medium"

        internal = copilot_model_to_internal(raw_model)
        result = to_amplifier_model_info(internal)

        # OpenAI models get "reasoning" instead of "thinking"
        assert "reasoning" in result.capabilities, (
            "OpenAI model with SDK reasoning_effort=True should have 'reasoning' capability"
        )
