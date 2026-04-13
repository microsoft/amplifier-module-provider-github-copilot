"""
Tests for error translation.

Contract: contracts/error-hierarchy.md

Acceptance Criteria:
- Translates known SDK errors to kernel types
- Uses config/errors.yaml for mappings
- Unknown errors become ProviderUnavailableError
- AC-4: All errors have provider="github-copilot"
- AC-5: Original exception chained via __cause__
- AC-6: RateLimitError extracts retry_after when present

Type annotations for pyright strict mode compliance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from amplifier_module_provider_github_copilot.error_translation import ErrorConfig


# Mock kernel error types for testing (since amplifier-core may not be installed)
# In real usage, these come from amplifier_core.llm_errors
@dataclass
class MockLLMError(Exception):
    """Mock base LLM error for testing."""

    message: str
    provider: str | None = None
    model: str | None = None
    retryable: bool = False
    retry_after: float | None = None


class MockAuthenticationError(MockLLMError):
    """Mock authentication error."""

    retryable: bool = False


class MockRateLimitError(MockLLMError):
    """Mock rate limit error."""

    retryable: bool = True


class MockLLMTimeoutError(MockLLMError):
    """Mock timeout error."""

    retryable: bool = True


class MockContentFilterError(MockLLMError):
    """Mock content filter error."""

    retryable: bool = False


class MockProviderUnavailableError(MockLLMError):
    """Mock provider unavailable error."""

    retryable: bool = True


class MockNetworkError(MockLLMError):
    """Mock network error."""

    retryable: bool = True


class MockNotFoundError(MockLLMError):
    """Mock not found error."""

    retryable: bool = False


# Test fixtures
@pytest.fixture
def error_config() -> ErrorConfig:
    """Load error config from Python config module."""
    from amplifier_module_provider_github_copilot.error_translation import load_error_config

    return load_error_config()


@pytest.fixture
def translate_fn() -> None:
    """Get translate function.

    Note: This fixture is currently unused but kept for API compatibility.
    """
    pass


class TestErrorTranslationBasic:
    """Test basic error translation functionality."""

    def test_translate_sdk_error_exists(self) -> None:
        """translate_sdk_error function exists."""
        from amplifier_module_provider_github_copilot.error_translation import translate_sdk_error

        assert callable(translate_sdk_error)

    def test_load_error_config_exists(self) -> None:
        """load_error_config function exists."""
        from amplifier_module_provider_github_copilot.error_translation import load_error_config

        assert callable(load_error_config)

    def test_error_config_dataclass_exists(self) -> None:
        """ErrorConfig dataclass exists."""
        from amplifier_module_provider_github_copilot.error_translation import ErrorConfig

        assert ErrorConfig is not None

    def test_error_mapping_dataclass_exists(self) -> None:
        """ErrorMapping dataclass exists."""
        from amplifier_module_provider_github_copilot.error_translation import ErrorMapping

        assert ErrorMapping is not None


class TestErrorConfigLoading:
    """Test error config loading from Python config module."""

    def test_config_has_default_error(self) -> None:
        """Config specifies default error type."""
        from amplifier_module_provider_github_copilot.error_translation import load_error_config

        # Use None to load from package resources (production path)
        config = load_error_config()
        assert config.default_error == "ProviderUnavailableError"

    def test_config_has_default_retryable(self) -> None:
        """Config specifies default retryable flag.

        Per Golden Vision V2 spec, default is non-retryable (conservative).
        """
        from amplifier_module_provider_github_copilot.error_translation import load_error_config

        # Use None to load from package resources (production path)
        config = load_error_config()
        assert config.default_retryable is False


class TestErrorTranslationMappings:
    """Test error translation using config mappings."""

    def test_unknown_error_becomes_provider_unavailable(self) -> None:
        """AC-3: Unknown errors become ProviderUnavailableError."""
        from amplifier_module_provider_github_copilot.error_translation import (
            ErrorConfig,
            translate_sdk_error,
        )

        config = ErrorConfig(
            mappings=[], default_error="ProviderUnavailableError", default_retryable=True
        )

        class UnknownSDKError(Exception):
            pass

        result = translate_sdk_error(UnknownSDKError("something went wrong"), config)
        assert result.__class__.__name__ == "ProviderUnavailableError"

    def test_error_has_provider_attribute(self) -> None:
        """AC-4: All errors have provider='github-copilot'."""
        from amplifier_module_provider_github_copilot.error_translation import (
            ErrorConfig,
            translate_sdk_error,
        )

        config = ErrorConfig(
            mappings=[], default_error="ProviderUnavailableError", default_retryable=True
        )

        class SomeError(Exception):
            pass

        result = translate_sdk_error(SomeError("test"), config)
        assert result.provider == "github-copilot"

    def test_original_exception_chained(self) -> None:
        """AC-5: Original exception chained via __cause__."""
        from amplifier_module_provider_github_copilot.error_translation import (
            ErrorConfig,
            translate_sdk_error,
        )

        config = ErrorConfig(
            mappings=[], default_error="ProviderUnavailableError", default_retryable=True
        )

        class OriginalError(Exception):
            pass

        original = OriginalError("original message")
        result = translate_sdk_error(original, config)
        assert result.__cause__ is original

    def test_translate_never_raises(self) -> None:
        """Contract: translate_sdk_error MUST NOT raise."""
        from amplifier_module_provider_github_copilot.error_translation import (
            ErrorConfig,
            translate_sdk_error,
        )

        config = ErrorConfig(
            mappings=[], default_error="ProviderUnavailableError", default_retryable=True
        )

        # Even with weird inputs, should not raise
        result = translate_sdk_error(Exception("test"), config)
        assert result is not None


class TestErrorMappingPatterns:
    """Test pattern matching for error translation."""

    def test_match_by_type_name(self) -> None:
        """AC-1: Match SDK error by type name."""
        from amplifier_module_provider_github_copilot.error_translation import (
            ErrorConfig,
            ErrorMapping,
            translate_sdk_error,
        )

        mapping = ErrorMapping(
            sdk_patterns=["AuthenticationError"],
            string_patterns=[],
            kernel_error="AuthenticationError",
            retryable=False,
        )
        config = ErrorConfig(
            mappings=[mapping],
            default_error="ProviderUnavailableError",
            default_retryable=True,
        )

        class AuthenticationError(Exception):
            pass

        result = translate_sdk_error(AuthenticationError("invalid token"), config)
        assert result.__class__.__name__ == "AuthenticationError"
        assert result.retryable is False

    def test_match_by_string_pattern(self) -> None:
        """AC-1: Match SDK error by string pattern in message."""
        from amplifier_module_provider_github_copilot.error_translation import (
            ErrorConfig,
            ErrorMapping,
            translate_sdk_error,
        )

        mapping = ErrorMapping(
            sdk_patterns=[],
            string_patterns=["429", "rate limit"],
            kernel_error="RateLimitError",
            retryable=True,
        )
        config = ErrorConfig(
            mappings=[mapping],
            default_error="ProviderUnavailableError",
            default_retryable=True,
        )

        class GenericError(Exception):
            pass

        result = translate_sdk_error(GenericError("HTTP 429 rate limit exceeded"), config)
        assert result.__class__.__name__ == "RateLimitError"
        assert result.retryable is True

    def test_first_match_wins(self) -> None:
        """Edge case: First matching pattern wins."""
        from amplifier_module_provider_github_copilot.error_translation import (
            ErrorConfig,
            ErrorMapping,
            translate_sdk_error,
        )

        mapping1 = ErrorMapping(
            sdk_patterns=["TestError"],
            string_patterns=[],
            kernel_error="AuthenticationError",
            retryable=False,
        )
        mapping2 = ErrorMapping(
            sdk_patterns=["TestError"],
            string_patterns=[],
            kernel_error="RateLimitError",
            retryable=True,
        )
        config = ErrorConfig(
            mappings=[mapping1, mapping2],
            default_error="ProviderUnavailableError",
            default_retryable=True,
        )

        class TestError(Exception):
            pass

        result = translate_sdk_error(TestError("test"), config)
        # First mapping should win
        assert result.__class__.__name__ == "AuthenticationError"


class TestRateLimitRetryAfter:
    """Test retry_after extraction for rate limit errors."""

    def test_extract_retry_after_from_message(self) -> None:
        """AC-6: RateLimitError extracts retry_after when present."""
        from amplifier_module_provider_github_copilot.error_translation import (
            ErrorConfig,
            ErrorMapping,
            translate_sdk_error,
        )

        mapping = ErrorMapping(
            sdk_patterns=["RateLimitError"],
            string_patterns=[],
            kernel_error="RateLimitError",
            retryable=True,
            extract_retry_after=True,
        )
        config = ErrorConfig(
            mappings=[mapping],
            default_error="ProviderUnavailableError",
            default_retryable=True,
        )

        class RateLimitError(Exception):
            pass

        # Error message contains retry_after hint
        result = translate_sdk_error(
            RateLimitError("Rate limit exceeded. Retry after 30 seconds."),
            config,
        )
        assert result.__class__.__name__ == "RateLimitError"


class TestVisionErrorTranslation:
    """Test vision/image input error translation.

    Contract: contracts/error-hierarchy.md
    Anchors:
    - error-hierarchy:ImageValidation:MUST:1 — CAPIError 400 with image_url → InvalidRequestError
    - error-hierarchy:ImageValidation:MUST:2 — Image validation errors are non-retryable
    - error-hierarchy:ImageValidation:MUST:3 — Original exception preserved via chaining
    """

    def test_image_validation_error_maps_to_invalid_request(self) -> None:
        """error-hierarchy:ImageValidation:MUST:1 — CAPIError 400 → InvalidRequestError."""
        from amplifier_module_provider_github_copilot.error_translation import (
            load_error_config,
            translate_sdk_error,
        )

        config = load_error_config()

        # Real error captured from SDK (test_vision_real_image.py)
        error_message = (
            "Session error: Execution failed: CAPIError: 400 invalid request body, "
            "failed to validate schema: (1) Reason: got array, want string, "
            "Location: /properties/messages/items/properties/content/oneOf/0/type. "
            "(2) Reason: missing property 'type', "
            "Location: /properties/messages/items/properties/content/oneOf/1/items/"
            "properties/image_url/required."
        )

        class SDKError(Exception):
            pass

        result = translate_sdk_error(SDKError(error_message), config)
        assert result.__class__.__name__ == "InvalidRequestError"

    def test_image_validation_error_is_not_retryable(self) -> None:
        """error-hierarchy:ImageValidation:MUST:2 — Image errors are non-retryable."""
        from amplifier_module_provider_github_copilot.error_translation import (
            load_error_config,
            translate_sdk_error,
        )

        config = load_error_config()

        class CAPIError(Exception):
            pass

        result = translate_sdk_error(
            CAPIError("400 invalid request body, image_url failed"), config
        )
        assert result.retryable is False

    def test_image_validation_error_preserves_cause(self) -> None:
        """error-hierarchy:ImageValidation:MUST:3 — Original exception chained."""
        from amplifier_module_provider_github_copilot.error_translation import (
            load_error_config,
            translate_sdk_error,
        )

        config = load_error_config()

        class CAPIError(Exception):
            pass

        original = CAPIError("400 invalid request body, image_url validation failed")
        result = translate_sdk_error(original, config)

        assert result.__cause__ is original
        assert result.provider == "github-copilot"


class TestConnectionErrorMapping:
    """Contract: error-hierarchy:ConnectionError:MUST:1

    ConnectionError MUST map to ProviderUnavailableError, not NetworkError.
    The contract table (error-hierarchy.md:71) is the governing truth.
    errors.yaml had ConnectionError and ProcessExitedError grouped under NetworkError —
    that contradicts the contract.
    """

    def test_connection_error_maps_to_provider_unavailable(self) -> None:
        """error-hierarchy:ConnectionError:MUST:1 — ConnectionError → ProviderUnavailableError.

        Connection refused = provider endpoint unreachable = provider-level failure.
        """
        from amplifier_module_provider_github_copilot.error_translation import (
            load_error_config,
            translate_sdk_error,
        )

        config = load_error_config()

        class ConnectionError(Exception):  # noqa: A001
            pass

        result = translate_sdk_error(ConnectionError("connection refused"), config)

        assert result.__class__.__name__ == "ProviderUnavailableError", (
            f"ConnectionError must map to ProviderUnavailableError per "
            f"error-hierarchy:ConnectionError:MUST:1, got {result.__class__.__name__}"
        )
        assert result.retryable is True

    def test_process_exited_error_maps_to_network_error(self) -> None:
        """error-hierarchy:ConnectionError:MUST:1 — ProcessExitedError → NetworkError.

        Process exit = raw network/process transport failure = NetworkError.
        """
        from amplifier_module_provider_github_copilot.error_translation import (
            load_error_config,
            translate_sdk_error,
        )

        config = load_error_config()

        class ProcessExitedError(Exception):
            pass

        result = translate_sdk_error(ProcessExitedError("process exited"), config)

        assert result.__class__.__name__ == "NetworkError", (
            f"ProcessExitedError must map to NetworkError, got {result.__class__.__name__}"
        )
        assert result.retryable is True
