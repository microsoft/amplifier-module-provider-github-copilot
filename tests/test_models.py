"""
Tests for model discovery and type translation.

Contract: contracts/sdk-boundary.md (ModelDiscovery section)
Contract: contracts/behaviors.md (ModelDiscoveryError section)

Three-Medium Architecture:
- Python: Type translation logic (models.py)
- Python: Cache policy values (config/_policy.py)
- Markdown: Invariants (contracts/*.md)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# =============================================================================
# SDK Mock Types (replicating SDK type structure for tests)
# These match the actual SDK types from github-copilot-sdk
# =============================================================================


@dataclass
class MockModelVisionLimits:
    """Mock SDK ModelVisionLimits."""

    supported_media_types: list[str] | None = None
    max_prompt_images: int | None = None
    max_prompt_image_size: int | None = None


@dataclass
class MockModelLimits:
    """Mock SDK ModelLimits."""

    max_prompt_tokens: int | None = None
    max_context_window_tokens: int | None = None
    vision: MockModelVisionLimits | None = None


@dataclass
class MockModelSupports:
    """Mock SDK ModelSupports."""

    vision: bool = False
    reasoning_effort: bool = False


@dataclass
class MockModelCapabilities:
    """Mock SDK ModelCapabilities."""

    supports: MockModelSupports
    limits: MockModelLimits


@dataclass
class MockSDKModelInfo:
    """Mock SDK ModelInfo - replicates copilot.types.ModelInfo structure."""

    id: str
    name: str
    capabilities: MockModelCapabilities
    policy: Any = None
    billing: Any = None
    supported_reasoning_efforts: list[str] | None = None
    default_reasoning_effort: str | None = None


class _MockSDKClient:
    """Minimal stub for SDK client used in model discovery tests."""

    async def list_models(self) -> list[MockSDKModelInfo]: ...


# =============================================================================
# Test Fixtures
# =============================================================================


def make_sdk_model_info(
    model_id: str = "claude-sonnet-4-5",
    name: str = "Claude Sonnet 4.5",
    context_window: int = 200000,
    max_prompt_tokens: int = 168000,
    supports_vision: bool = True,
    supports_reasoning_effort: bool = False,
    supported_reasoning_efforts: list[str] | None = None,
    default_reasoning_effort: str | None = None,
) -> MockSDKModelInfo:
    """Create a mock SDK ModelInfo for testing."""
    return MockSDKModelInfo(
        id=model_id,
        name=name,
        capabilities=MockModelCapabilities(
            supports=MockModelSupports(
                vision=supports_vision,
                reasoning_effort=supports_reasoning_effort,
            ),
            limits=MockModelLimits(
                max_prompt_tokens=max_prompt_tokens,
                max_context_window_tokens=context_window,
            ),
        ),
        supported_reasoning_efforts=supported_reasoning_efforts,
        default_reasoning_effort=default_reasoning_effort,
    )


# =============================================================================
# Phase 1a: SDK ModelInfo → CopilotModelInfo (Isolation Layer)
# Contract: sdk-boundary:ModelDiscovery:MUST:2
# =============================================================================


class TestSDKToCopilotModelInfoTranslation:
    """Test translation from SDK ModelInfo to CopilotModelInfo.

    Contract: sdk-boundary:ModelDiscovery:MUST:2
    - MUST translate SDK ModelInfo to domain CopilotModelInfo
    """

    def test_copilot_model_to_internal_extracts_limits(self) -> None:
        """Contract: sdk-boundary:ModelDiscovery:MUST:2

        MUST extract context_window from SDK capabilities.limits.max_context_window_tokens.
        MUST derive max_output_tokens as context_window - max_prompt_tokens.
        """
        from amplifier_module_provider_github_copilot.models import (
            CopilotModelInfo,
            sdk_model_to_copilot_model,
        )

        sdk_model = make_sdk_model_info(
            model_id="claude-sonnet-4-5",
            name="Claude Sonnet 4.5",
            context_window=200000,
            max_prompt_tokens=168000,
            supports_vision=True,
        )

        result = sdk_model_to_copilot_model(sdk_model)

        assert isinstance(result, CopilotModelInfo)
        assert result.id == "claude-sonnet-4-5"
        assert result.name == "Claude Sonnet 4.5"
        assert result.context_window == 200000
        # max_output_tokens = context_window - max_prompt_tokens
        assert result.max_output_tokens == 200000 - 168000  # 32000
        assert result.supports_vision is True

    def test_copilot_model_to_internal_handles_missing_limits(self) -> None:
        """Contract: sdk-boundary:ModelDiscovery:MUST:2

        MUST handle None values in SDK limits gracefully with sensible defaults.
        """
        from amplifier_module_provider_github_copilot.models import (
            CopilotModelInfo,
            sdk_model_to_copilot_model,
        )

        # Create model with None limits
        sdk_model = MockSDKModelInfo(
            id="test-model",
            name="Test Model",
            capabilities=MockModelCapabilities(
                supports=MockModelSupports(vision=False),
                limits=MockModelLimits(
                    max_prompt_tokens=None,
                    max_context_window_tokens=None,
                ),
            ),
        )

        result = sdk_model_to_copilot_model(sdk_model)

        assert isinstance(result, CopilotModelInfo)
        assert result.id == "test-model"
        # Should use policy defaults from config/models.yaml
        assert result.context_window > 0
        assert result.max_output_tokens > 0

    def test_copilot_model_info_is_frozen(self) -> None:
        """CopilotModelInfo MUST be immutable (frozen dataclass)."""
        from amplifier_module_provider_github_copilot.models import CopilotModelInfo

        model = CopilotModelInfo(
            id="test",
            name="Test",
            context_window=100000,
            max_output_tokens=32000,
        )

        with pytest.raises(AttributeError):  # Frozen dataclass
            model.id = "modified"  # type: ignore[misc]

    def test_extracts_vision_capability(self) -> None:
        """Contract: sdk-boundary:ModelDiscovery:MUST:2

        MUST extract vision capability from SDK supports.vision.
        """
        from amplifier_module_provider_github_copilot.models import (
            sdk_model_to_copilot_model,
        )

        sdk_model_with_vision = make_sdk_model_info(supports_vision=True)
        sdk_model_without_vision = make_sdk_model_info(supports_vision=False)

        result_with = sdk_model_to_copilot_model(sdk_model_with_vision)
        result_without = sdk_model_to_copilot_model(sdk_model_without_vision)

        assert result_with.supports_vision is True
        assert result_without.supports_vision is False

    def test_extracts_reasoning_effort_capability(self) -> None:
        """Contract: sdk-boundary:ModelDiscovery:MUST:2

        MUST extract reasoning_effort capability from SDK supports.reasoning_effort.
        """
        from amplifier_module_provider_github_copilot.models import (
            sdk_model_to_copilot_model,
        )

        sdk_model = make_sdk_model_info(
            supports_reasoning_effort=True,
            supported_reasoning_efforts=["low", "medium", "high"],
            default_reasoning_effort="medium",
        )

        result = sdk_model_to_copilot_model(sdk_model)

        assert result.supports_reasoning_effort is True
        assert result.supported_reasoning_efforts == ("low", "medium", "high")
        assert result.default_reasoning_effort == "medium"


# =============================================================================
# Phase 1a: CopilotModelInfo → amplifier_core.ModelInfo (Kernel Contract)
# Contract: sdk-boundary:ModelDiscovery:MUST:3
# =============================================================================


class TestCopilotToAmplifierModelInfoTranslation:
    """Test translation from CopilotModelInfo to amplifier_core.ModelInfo.

    Contract: sdk-boundary:ModelDiscovery:MUST:3
    - MUST translate CopilotModelInfo to amplifier_core.ModelInfo (kernel contract)
    """

    def test_to_amplifier_model_info_maps_all_fields(self) -> None:
        """Contract: sdk-boundary:ModelDiscovery:MUST:3

        MUST translate CopilotModelInfo to amplifier_core.ModelInfo.
        MUST map: id, display_name, context_window, max_output_tokens, capabilities.
        """
        from amplifier_core import ModelInfo as AmplifierModelInfo

        from amplifier_module_provider_github_copilot.models import (
            CopilotModelInfo,
            copilot_model_to_amplifier_model,
        )

        copilot_model = CopilotModelInfo(
            id="claude-sonnet-4-5",
            name="Claude Sonnet 4.5",
            context_window=200000,
            max_output_tokens=32000,
            supports_vision=True,
            supports_reasoning_effort=False,
        )

        result = copilot_model_to_amplifier_model(copilot_model)

        assert isinstance(result, AmplifierModelInfo)
        assert result.id == "claude-sonnet-4-5"
        assert result.display_name == "Claude Sonnet 4.5"
        assert result.context_window == 200000
        assert result.max_output_tokens == 32000

    def test_capabilities_include_vision_when_supported(self) -> None:
        """Contract: sdk-boundary:ModelDiscovery:MUST:3

        MUST include 'vision' in capabilities list when model supports vision.
        """
        from amplifier_module_provider_github_copilot.models import (
            CopilotModelInfo,
            copilot_model_to_amplifier_model,
        )

        model_with_vision = CopilotModelInfo(
            id="test",
            name="Test",
            context_window=100000,
            max_output_tokens=32000,
            supports_vision=True,
        )
        model_without_vision = CopilotModelInfo(
            id="test2",
            name="Test 2",
            context_window=100000,
            max_output_tokens=32000,
            supports_vision=False,
        )

        result_with = copilot_model_to_amplifier_model(model_with_vision)
        result_without = copilot_model_to_amplifier_model(model_without_vision)

        assert "vision" in result_with.capabilities
        assert "vision" not in result_without.capabilities

    def test_capabilities_include_thinking_when_reasoning_supported(self) -> None:
        """Contract: sdk-boundary:ModelDiscovery:MUST:3

        MUST include 'thinking' in capabilities when model supports reasoning effort.
        """
        from amplifier_module_provider_github_copilot.models import (
            CopilotModelInfo,
            copilot_model_to_amplifier_model,
        )

        model_with_reasoning = CopilotModelInfo(
            id="test",
            name="Test",
            context_window=100000,
            max_output_tokens=32000,
            supports_reasoning_effort=True,
        )

        result = copilot_model_to_amplifier_model(model_with_reasoning)

        assert "thinking" in result.capabilities

    def test_all_models_have_streaming_and_tools_capabilities(self) -> None:
        """Contract: sdk-boundary:ModelDiscovery:MUST:3

        All GitHub Copilot models support streaming and tools.
        """
        from amplifier_module_provider_github_copilot.models import (
            CopilotModelInfo,
            copilot_model_to_amplifier_model,
        )

        model = CopilotModelInfo(
            id="any-model",
            name="Any Model",
            context_window=100000,
            max_output_tokens=32000,
        )

        result = copilot_model_to_amplifier_model(model)

        assert "streaming" in result.capabilities
        assert "tools" in result.capabilities


# =============================================================================
# Phase 3: Provider.list_models() Dynamic Integration
# Contract: sdk-boundary:ModelDiscovery:MUST:1
# =============================================================================


class TestProviderListModelsDynamic:
    """Test that provider.list_models() uses SDK dynamically.

    Contract: sdk-boundary:ModelDiscovery:MUST:1
    - MUST fetch models from SDK list_models() API
    """

    @pytest.mark.asyncio
    async def test_provider_list_models_calls_fetch_models(self) -> None:
        """Contract: sdk-boundary:ModelDiscovery:MUST:1

        provider.list_models() MUST call models.fetch_and_map_models().
        This test verifies the provider uses dynamic SDK fetch, not static YAML.
        """
        from unittest.mock import patch

        from amplifier_core import ModelInfo as AmplifierModelInfo

        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        # Create provider
        provider = GitHubCopilotProvider()

        # Create a unique model that would only come from SDK
        # (different from any YAML config models)
        sdk_only_model = AmplifierModelInfo(
            id="sdk-unique-model-xyz",  # Not in any YAML config
            display_name="SDK-Only Model XYZ",
            context_window=999999,
            max_output_tokens=88888,
            capabilities=["streaming", "tools"],
            defaults={},
        )

        # Patch fetch_and_map_models to return our unique model
        # fetch_and_map_models now returns (amplifier_models, copilot_models) tuple
        with patch(
            "amplifier_module_provider_github_copilot.provider.fetch_and_map_models"
        ) as mock_fetch:
            # Create matching CopilotModelInfo for the tuple
            from amplifier_module_provider_github_copilot.sdk_adapter.model_translation import (
                CopilotModelInfo,
            )

            copilot_model = CopilotModelInfo(
                id="sdk-unique-model-xyz",
                name="SDK-Only Model XYZ",
                context_window=999999,
                max_output_tokens=88888,
            )
            mock_fetch.return_value = ([sdk_only_model], [copilot_model])

            result = await provider.list_models()

        # Verify fetch was called
        mock_fetch.assert_called_once()

        # Verify we got the SDK model, not YAML config
        assert len(result) == 1
        assert result[0].id == "sdk-unique-model-xyz"
        assert result[0].context_window == 999999

    @pytest.mark.asyncio
    async def test_provider_list_models_uses_cache_on_sdk_failure(self) -> None:
        """Contract: behaviors:ModelCache:SHOULD:1

        When SDK fails, provider.list_models() SHOULD use disk cache.
        """
        from unittest.mock import patch

        from amplifier_core import ModelInfo as AmplifierModelInfo

        from amplifier_module_provider_github_copilot.models import CopilotModelInfo
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        provider = GitHubCopilotProvider()

        # Cached model
        cached_model = CopilotModelInfo(
            id="cached-model",
            name="Cached Model",
            context_window=100000,
            max_output_tokens=16000,
        )

        # Simulate SDK failure + cache hit
        with (
            patch(
                "amplifier_module_provider_github_copilot.provider.fetch_and_map_models",
                side_effect=Exception("SDK unavailable"),
            ),
            patch(
                "amplifier_module_provider_github_copilot.provider.read_cache",
                return_value=[cached_model],
            ),
            patch(
                "amplifier_module_provider_github_copilot.provider.copilot_model_to_amplifier_model"
            ) as mock_convert,
        ):
            mock_convert.return_value = AmplifierModelInfo(
                id="cached-model",
                display_name="Cached Model",
                context_window=100000,
                max_output_tokens=16000,
                capabilities=["streaming", "tools"],
                defaults={},
            )

            result = await provider.list_models()

        # Should have used cache
        assert len(result) == 1
        assert result[0].id == "cached-model"

    @pytest.mark.asyncio
    async def test_provider_list_models_writes_cache_on_success(self) -> None:
        """Contract: provider-protocol:list_models:SHOULD:1

        Successful SDK fetch should write to cache for future fallback.
        """
        from unittest.mock import patch

        from amplifier_core import ModelInfo as AmplifierModelInfo

        from amplifier_module_provider_github_copilot.models import CopilotModelInfo
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        provider = GitHubCopilotProvider()

        copilot_model = CopilotModelInfo(
            id="test-model",
            name="Test Model",
            context_window=200000,
            max_output_tokens=32000,
        )

        amplifier_model = AmplifierModelInfo(
            id="test-model",
            display_name="Test Model",
            context_window=200000,
            max_output_tokens=32000,
            capabilities=["streaming", "tools"],
            defaults={},
        )

        write_cache_called = False

        def mock_write_cache(models: Any) -> None:
            nonlocal write_cache_called
            write_cache_called = True

        # fetch_and_map_models now returns (amplifier_models, copilot_models) tuple
        # The provider uses copilot_models for caching directly
        with (
            patch(
                "amplifier_module_provider_github_copilot.provider.fetch_and_map_models",
                return_value=([amplifier_model], [copilot_model]),
            ),
            patch(
                "amplifier_module_provider_github_copilot.provider.write_cache",
                side_effect=mock_write_cache,
            ),
        ):
            result = await provider.list_models()

        assert len(result) == 1
        assert write_cache_called, "write_cache should be called on SDK success"

    @pytest.mark.asyncio
    async def test_provider_list_models_raises_when_both_sdk_and_cache_fail(self) -> None:
        """Contract: behaviors:ModelDiscoveryError:MUST:1

        When SDK fails AND cache is empty, MUST raise ProviderUnavailableError.
        """
        from unittest.mock import patch

        from amplifier_core import ProviderUnavailableError

        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        provider = GitHubCopilotProvider()

        with (
            patch(
                "amplifier_module_provider_github_copilot.provider.fetch_and_map_models",
                side_effect=Exception("SDK unavailable"),
            ),
            patch(
                "amplifier_module_provider_github_copilot.provider.read_cache",
                return_value=None,  # Empty cache
            ),
        ):
            with pytest.raises(ProviderUnavailableError) as exc_info:
                await provider.list_models()

        assert "cached models" in str(exc_info.value).lower()


# =============================================================================
# Phase 1e: CopilotClientWrapper.list_models()
# =============================================================================


class TestCopilotClientWrapperListModels:
    """Test CopilotClientWrapper.list_models() method.

    Contract: sdk-boundary:Models:MUST:1
    - SDK CopilotClient.list_models() returns list[ModelInfo]
    """

    @pytest.mark.asyncio
    async def test_client_wrapper_list_models_calls_sdk(self) -> None:
        """Contract: sdk-boundary:Models:MUST:1

        CopilotClientWrapper.list_models() MUST call SDK client.list_models().
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        # Mock SDK client
        mock_sdk_client = MagicMock()
        mock_sdk_client.list_models = AsyncMock(return_value=[make_sdk_model_info()])

        wrapper = CopilotClientWrapper(sdk_client=mock_sdk_client)
        result = await wrapper.list_models()

        mock_sdk_client.list_models.assert_called_once()
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_client_wrapper_list_models_returns_sdk_models(self) -> None:
        """Contract: sdk-boundary:Models:MUST:1

        list_models() returns SDK ModelInfo objects (translation happens elsewhere).
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        sdk_model = make_sdk_model_info(model_id="claude-opus-4.5", name="Claude Opus 4.5")
        mock_sdk_client = MagicMock()
        mock_sdk_client.list_models = AsyncMock(return_value=[sdk_model])

        wrapper = CopilotClientWrapper(sdk_client=mock_sdk_client)
        result = await wrapper.list_models()

        assert len(result) == 1
        assert result[0].id == "claude-opus-4.5"
        assert result[0].name == "Claude Opus 4.5"

    @pytest.mark.asyncio
    async def test_client_wrapper_list_models_lazy_init(self) -> None:
        """Contract: sdk-boundary:Models:MUST:1

        list_models() should work with lazy-initialized client.
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        # Create wrapper without injected client
        wrapper = CopilotClientWrapper()

        # Inject a mock client directly to test the path
        mock_sdk_client = MagicMock()
        mock_sdk_client.list_models = AsyncMock(return_value=[make_sdk_model_info()])
        wrapper._owned_client = mock_sdk_client  # type: ignore[attr-defined]

        result = await wrapper.list_models()

        mock_sdk_client.list_models.assert_called_once()
        assert len(result) == 1


# =============================================================================
# Phase 1a: SDK list_models() Integration
# Contract: sdk-boundary:ModelDiscovery:MUST:1
# =============================================================================


class TestFetchModelsFromSDK:
    """Test that models are fetched from SDK list_models() API.

    Contract: sdk-boundary:ModelDiscovery:MUST:1
    - MUST fetch models from SDK list_models() API
    """

    @pytest.mark.asyncio
    async def test_fetch_calls_sdk_list_models(self) -> None:
        """Contract: sdk-boundary:ModelDiscovery:MUST:1

        MUST call SDK client.list_models() to get available models.
        """
        from amplifier_module_provider_github_copilot.models import fetch_models

        # Mock SDK client
        mock_client = MagicMock()
        mock_client.list_models = AsyncMock(
            return_value=[
                make_sdk_model_info(
                    model_id="claude-sonnet-4-5",
                    name="Claude Sonnet 4.5",
                    context_window=200000,
                    max_prompt_tokens=168000,
                ),
                make_sdk_model_info(
                    model_id="gpt-4o",
                    name="GPT-4o",
                    context_window=128000,
                    max_prompt_tokens=100000,
                ),
            ]
        )

        result = await fetch_models(mock_client)

        mock_client.list_models.assert_called_once()
        assert len(result) == 2
        assert result[0].id == "claude-sonnet-4-5"
        assert result[1].id == "gpt-4o"

    @pytest.mark.asyncio
    async def test_fetch_returns_copilot_model_info_list(self) -> None:
        """Contract: sdk-boundary:ModelDiscovery:MUST:2

        fetch_models MUST return list[CopilotModelInfo], not SDK types.
        """
        from amplifier_module_provider_github_copilot.models import (
            CopilotModelInfo,
            fetch_models,
        )

        mock_client = MagicMock()
        mock_client.list_models = AsyncMock(return_value=[make_sdk_model_info()])

        result = await fetch_models(mock_client)

        assert len(result) == 1
        assert isinstance(result[0], CopilotModelInfo)


# =============================================================================
# Phase 1a: No Hardcoded Model Lists
# Contract: sdk-boundary:ModelDiscovery:MUST_NOT:1
# =============================================================================


class TestNoHardcodedModelLists:
    """Verify no hardcoded model lists exist in production code.

    Contract: sdk-boundary:ModelDiscovery:MUST_NOT:1
    - MUST NOT use hardcoded model lists in production code
    """

    def test_no_bundled_model_limits_dict(self) -> None:
        """Contract: sdk-boundary:ModelDiscovery:MUST_NOT:1

        There MUST NOT be a BUNDLED_MODEL_LIMITS dict or similar hardcoded fallback.
        This is hardcoding disguised as a variable.
        """
        import amplifier_module_provider_github_copilot.provider as provider_module

        # These names indicate hardcoded model data
        forbidden_names = [
            "BUNDLED_MODEL_LIMITS",
            "MODEL_LIMITS",
            "HARDCODED_MODELS",
            "FALLBACK_MODELS",
            "STATIC_MODELS",
            "DEFAULT_MODELS_LIST",
        ]

        for name in forbidden_names:
            assert not hasattr(provider_module, name), (
                f"Found '{name}' in provider.py — contract "
                "sdk-boundary:ModelDiscovery:MUST_NOT:1 violated. "
                "Models must come from SDK, not hardcoded dicts."
            )

    def test_models_yaml_contains_policy_not_catalog(self) -> None:
        """Contract: ARCHITECTURE.md line 71

        models.yaml contains default model POLICY, not a model catalog.
        It should have defaults.model but NOT expose a models list field.
        """
        from amplifier_module_provider_github_copilot.config_loader import load_models_config

        models_config = load_models_config()

        # Should have defaults (policy) with correct default model
        assert models_config.defaults["model"] == "claude-opus-4.5"

        # ProviderConfig no longer exposes a .models field — catalog comes from SDK
        assert not hasattr(models_config, "models"), (
            "ProviderConfig.models was removed — model catalog must come from SDK, "
            "not config. Contract: behaviors:ModelDiscoveryError:MUST_NOT:1"
        )


# =============================================================================
# Error Handling Tests
# Contract: behaviors:ModelDiscoveryError
# =============================================================================


class TestModelDiscoveryErrors:
    """Test error handling when SDK model discovery fails.

    Contract: behaviors:ModelDiscoveryError:MUST:1-2
    Contract: behaviors:ModelDiscoveryError:MUST_NOT:1
    """

    @pytest.mark.asyncio
    async def test_list_models_raises_when_sdk_fails_and_no_cache(self) -> None:
        """Contract: behaviors:ModelDiscoveryError:MUST:1

        MUST raise ProviderUnavailableError when SDK unavailable AND disk cache empty.
        """
        from amplifier_core import ProviderUnavailableError

        from amplifier_module_provider_github_copilot.models import fetch_models

        mock_client = MagicMock(spec=_MockSDKClient)
        mock_client.list_models = AsyncMock(side_effect=Exception("SDK connection failed"))

        with pytest.raises(ProviderUnavailableError) as exc_info:
            await fetch_models(mock_client)

        # Check error includes the exact documented message
        # Contract: behaviors:ModelDiscoveryError:MUST:1
        assert "Failed to fetch models from SDK" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_error_message_includes_reason(self) -> None:
        """Contract: behaviors:ModelDiscoveryError:MUST:2

        Error message MUST include reason for failure.
        """
        from amplifier_core import ProviderUnavailableError

        from amplifier_module_provider_github_copilot.models import fetch_models

        mock_client = MagicMock(spec=_MockSDKClient)
        mock_client.list_models = AsyncMock(side_effect=ConnectionError("Network unreachable"))

        with pytest.raises(ProviderUnavailableError) as exc_info:
            await fetch_models(mock_client)

        error_msg = str(exc_info.value)
        # Should contain the documented error message prefix
        # Contract: behaviors:ModelDiscoveryError:MUST:2
        assert "Failed to fetch models from SDK" in error_msg
        assert "SDK connection unavailable" in error_msg


# =============================================================================
# Three-Medium Architecture: Fail-Fast Configuration Loading
# Contract: behaviors:ConfigLoading:MUST:1
# =============================================================================


class TestConfigurationFailFast:
    """Test fail-fast behavior for configuration loading.

    Contract: behaviors:ConfigLoading:MUST:1
    Three-Medium Architecture: YAML is authoritative, fail-fast on missing.
    """

    def test_load_fallback_values_succeeds_with_valid_yaml(self) -> None:
        """Contract: behaviors:ConfigLoading:MUST:1

        Valid models.yaml should load fallback values successfully.
        """
        from amplifier_module_provider_github_copilot.models import (
            get_default_context_window,
            get_default_max_output_tokens,
        )

        # Should return values from models.yaml fallbacks section
        context_window = get_default_context_window()
        max_output_tokens = get_default_max_output_tokens()

        assert context_window == 128000, "Expected fallback context_window from YAML"
        assert max_output_tokens == 16384, "Expected fallback max_output_tokens from YAML"

    def test_configuration_error_raised_when_yaml_missing(self) -> None:
        """Contract: behaviors:ConfigLoading:MUST:1

        ConfigurationError raised when config/models.py is corrupted/missing data.
        Three-Medium Architecture: fail-fast, no hardcoded Python fallbacks.

        Updated: patches FALLBACKS = None to simulate broken installation.
        sys.modules patching does not intercept already-loaded relative imports;
        patch.object on the module attribute is the correct approach here.
        """
        from unittest.mock import patch

        from amplifier_module_provider_github_copilot._compat import ConfigurationError
        from amplifier_module_provider_github_copilot.config import (
            _models as _config_models,
        )
        from amplifier_module_provider_github_copilot.config_loader import (
            _load_model_fallback_values,  # pyright: ignore[reportPrivateUsage]
        )

        # Simulate broken installation: FALLBACKS is not a dict (e.g. None)
        with patch.object(_config_models, "FALLBACKS", None):
            _load_model_fallback_values.cache_clear()  # pyright: ignore[reportPrivateUsage]
            with pytest.raises(ConfigurationError) as exc_info:
                _load_model_fallback_values()  # pyright: ignore[reportPrivateUsage]

            assert "config/_models.py" in str(exc_info.value)

        # Clear cache after test to restore normal behavior
        _load_model_fallback_values.cache_clear()  # pyright: ignore[reportPrivateUsage]

    def test_configuration_error_raised_when_fallbacks_missing(self) -> None:
        """Contract: behaviors:ConfigLoading:MUST:1

        ConfigurationError raised when FALLBACKS dict has no required keys.

        Updated: patch config.models.FALLBACKS directly (not importlib.resources).
        """
        from unittest.mock import patch

        from amplifier_module_provider_github_copilot._compat import ConfigurationError
        from amplifier_module_provider_github_copilot.config import (
            _models as _config_models,
        )
        from amplifier_module_provider_github_copilot.config_loader import (
            _load_model_fallback_values,  # pyright: ignore[reportPrivateUsage]
        )

        # Clear cache before test
        _load_model_fallback_values.cache_clear()  # pyright: ignore[reportPrivateUsage]

        # Patch FALLBACKS to empty dict — all required keys missing
        with patch.object(_config_models, "FALLBACKS", {}):
            with pytest.raises(ConfigurationError) as exc_info:
                _load_model_fallback_values()  # pyright: ignore[reportPrivateUsage]

            assert "fallbacks" in str(exc_info.value)

        # Clear cache after test
        _load_model_fallback_values.cache_clear()  # pyright: ignore[reportPrivateUsage]

    def test_configuration_error_raised_when_required_keys_missing(self) -> None:
        """Contract: behaviors:ConfigLoading:MUST:1

        ConfigurationError raised when required key missing from FALLBACKS.

        Updated: patch config.models.FALLBACKS directly (not importlib.resources).
        """
        from unittest.mock import patch

        from amplifier_module_provider_github_copilot._compat import ConfigurationError
        from amplifier_module_provider_github_copilot.config import (
            _models as _config_models,
        )
        from amplifier_module_provider_github_copilot.config_loader import (
            _load_model_fallback_values,  # pyright: ignore[reportPrivateUsage]
        )

        # Clear cache before test
        _load_model_fallback_values.cache_clear()  # pyright: ignore[reportPrivateUsage]

        # Patch FALLBACKS with only context_window — max_output_tokens missing
        with patch.object(_config_models, "FALLBACKS", {"context_window": 128000}):
            with pytest.raises(ConfigurationError) as exc_info:
                _load_model_fallback_values()  # pyright: ignore[reportPrivateUsage]

            assert "max_output_tokens" in str(exc_info.value)

        # Clear cache after test
        _load_model_fallback_values.cache_clear()  # pyright: ignore[reportPrivateUsage]

    def test_configuration_error_raised_when_yaml_invalid(self) -> None:
        """Contract: behaviors:ConfigLoading:MUST:1

        ConfigurationError raised when FALLBACKS is not a dict (corrupted config).

        Updated: patch config.models.FALLBACKS to a non-dict value to trigger
        the isinstance(fallbacks, dict) guard in config_loader.py.
        """
        from unittest.mock import patch

        from amplifier_module_provider_github_copilot._compat import ConfigurationError
        from amplifier_module_provider_github_copilot.config import (
            _models as _config_models,
        )
        from amplifier_module_provider_github_copilot.config_loader import (
            _load_model_fallback_values,  # pyright: ignore[reportPrivateUsage]
        )

        # Clear cache before test
        _load_model_fallback_values.cache_clear()  # pyright: ignore[reportPrivateUsage]

        # Patch FALLBACKS to a non-dict (simulates corrupted/invalid config)
        with patch.object(_config_models, "FALLBACKS", "invalid_string"):
            with pytest.raises(ConfigurationError) as exc_info:
                _load_model_fallback_values()  # pyright: ignore[reportPrivateUsage]

            assert "config/_models.py" in str(exc_info.value)

        # Clear cache after test
        _load_model_fallback_values.cache_clear()  # pyright: ignore[reportPrivateUsage]


# =============================================================================
# Edge Cases: SDK Model Translation
# Contract: sdk-boundary:ModelDiscovery:MUST:2
# =============================================================================


class TestSDKModelEdgeCases:
    """Test edge cases in SDK model translation.

    Contract: sdk-boundary:ModelDiscovery:MUST:2
    """

    def test_negative_max_output_tokens_uses_fallback(self) -> None:
        """Contract: sdk-boundary:ModelDiscovery:MUST:2

        When derived max_output_tokens is <= 0, use fallback from YAML.
        Edge case: context_window < max_prompt_tokens
        """
        from amplifier_module_provider_github_copilot.models import (
            get_default_max_output_tokens,
            sdk_model_to_copilot_model,
        )

        # Create SDK model where context_window < max_prompt_tokens
        sdk_model = MockSDKModelInfo(
            id="test-model",
            name="Test Model",
            capabilities=MockModelCapabilities(
                supports=MockModelSupports(vision=False),
                limits=MockModelLimits(
                    max_context_window_tokens=100000,
                    max_prompt_tokens=150000,  # Greater than context_window!
                ),
            ),
        )

        result = sdk_model_to_copilot_model(sdk_model)

        # Should use fallback because 100000 - 150000 = -50000 <= 0
        assert result.max_output_tokens == get_default_max_output_tokens()
        assert result.max_output_tokens > 0

    def test_reasoning_effort_defaults_populated(self) -> None:
        """Contract: sdk-boundary:ModelDiscovery:MUST:3

        Reasoning effort fields should be included in defaults when present.
        """
        from amplifier_module_provider_github_copilot.models import (
            CopilotModelInfo,
            copilot_model_to_amplifier_model,
        )

        model = CopilotModelInfo(
            id="claude-thinking",
            name="Claude Thinking",
            context_window=200000,
            max_output_tokens=32000,
            supports_vision=False,
            supports_reasoning_effort=True,
            supported_reasoning_efforts=("low", "medium", "high"),
            default_reasoning_effort="medium",
        )

        result = copilot_model_to_amplifier_model(model)

        assert "thinking" in result.capabilities
        assert result.defaults["reasoning_effort"] == "medium"
        assert result.defaults["supported_reasoning_efforts"] == ["low", "medium", "high"]


# =============================================================================
# Full Translation Chain: fetch_and_map_models
# Contract: sdk-boundary:ModelDiscovery:MUST:1,2,3
# =============================================================================


class TestFetchAndMapModels:
    """Test the full translation chain via fetch_and_map_models.

    Contract: sdk-boundary:ModelDiscovery:MUST:1,2,3

    Note: Uses real_model_discovery fixture to bypass autouse mock.
    """

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("real_model_discovery")
    async def test_fetch_and_map_returns_amplifier_models(self) -> None:
        """Contract: sdk-boundary:ModelDiscovery:MUST:1,2,3

        fetch_and_map_models chains: SDK → CopilotModelInfo → AmplifierModelInfo
        Returns tuple of (amplifier_models, copilot_models) for caching.
        """
        from amplifier_module_provider_github_copilot.models import fetch_and_map_models

        mock_client = MagicMock()
        mock_client.list_models = AsyncMock(
            return_value=[
                make_sdk_model_info(
                    model_id="claude-opus-4.5",
                    name="Claude Opus 4.5",
                    context_window=200000,
                    max_prompt_tokens=168000,
                    supports_vision=True,
                ),
            ]
        )

        amplifier_models, copilot_models = await fetch_and_map_models(mock_client)

        # Verify amplifier models
        assert len(amplifier_models) == 1
        model = amplifier_models[0]
        assert model.id == "claude-opus-4.5"
        assert model.display_name == "Claude Opus 4.5"
        assert model.context_window == 200000
        assert model.max_output_tokens == 32000  # 200000 - 168000
        assert "vision" in model.capabilities
        assert "streaming" in model.capabilities
        assert "tools" in model.capabilities

        # Verify copilot models (for caching)
        assert len(copilot_models) == 1
        assert copilot_models[0].id == "claude-opus-4.5"

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("real_model_discovery")
    async def test_fetch_and_map_propagates_errors(self) -> None:
        """Contract: behaviors:ModelDiscoveryError:MUST:1

        fetch_and_map_models propagates SDK errors as ProviderUnavailableError.
        """
        from amplifier_core import ProviderUnavailableError

        from amplifier_module_provider_github_copilot.models import fetch_and_map_models

        mock_client = MagicMock()
        mock_client.list_models = AsyncMock(side_effect=ConnectionError("SDK unavailable"))

        with pytest.raises(ProviderUnavailableError):
            await fetch_and_map_models(mock_client)


# =============================================================================
# Test: model_translation edge cases
# Coverage: sdk_adapter/model_translation.py lines 90-93, 100-101, 123-124
# =============================================================================


class TestModelTranslationEdgeCases:
    """Test sdk_model_to_copilot_model with malformed SDK data.

    The SDK may return partial model info. Translation must handle None values.
    """

    def test_translate_sdk_model_with_none_capabilities(self) -> None:
        """SDK model with capabilities=None uses defaults.

        Coverage: model_translation.py lines 90-93
        """
        from types import SimpleNamespace

        from amplifier_module_provider_github_copilot.sdk_adapter.model_translation import (
            sdk_model_to_copilot_model,
        )

        # SDK model with no capabilities (complete shape)
        sdk_model = SimpleNamespace(
            id="test-model",
            name="Test Model",
            capabilities=None,
            supported_reasoning_efforts=None,
            default_reasoning_effort=None,
        )

        result = sdk_model_to_copilot_model(sdk_model)

        assert result.id == "test-model"
        # Uses defaults from config when capabilities is None
        assert result.context_window > 0
        assert result.max_output_tokens > 0

    def test_translate_sdk_model_with_none_limits(self) -> None:
        """SDK model with capabilities.limits=None uses defaults.

        Coverage: model_translation.py lines 100-101
        """
        from types import SimpleNamespace

        from amplifier_module_provider_github_copilot.sdk_adapter.model_translation import (
            sdk_model_to_copilot_model,
        )

        # SDK model with capabilities but no limits
        sdk_model = SimpleNamespace(
            id="test-model",
            name="Test Model",
            capabilities=SimpleNamespace(
                limits=None,
                supports=SimpleNamespace(vision=True, reasoning_effort=False),
            ),
            supported_reasoning_efforts=None,
            default_reasoning_effort=None,
        )

        result = sdk_model_to_copilot_model(sdk_model)

        assert result.id == "test-model"
        assert result.context_window > 0

    def test_translate_sdk_model_with_none_supports(self) -> None:
        """SDK model with capabilities.supports=None defaults to False.

        Coverage: model_translation.py lines 123-124
        """
        from types import SimpleNamespace

        from amplifier_module_provider_github_copilot.sdk_adapter.model_translation import (
            sdk_model_to_copilot_model,
        )

        # SDK model with capabilities but no supports
        sdk_model = SimpleNamespace(
            id="test-model",
            name="Test Model",
            capabilities=SimpleNamespace(
                limits=SimpleNamespace(
                    max_context_window_tokens=100000,
                    max_prompt_tokens=80000,
                ),
                supports=None,
            ),
            supported_reasoning_efforts=None,
            default_reasoning_effort=None,
        )

        result = sdk_model_to_copilot_model(sdk_model)

        assert result.id == "test-model"
        # Vision/reasoning should default to False when supports is None
        assert result.supports_vision is False
        assert result.supports_reasoning_effort is False
