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

from typing import TYPE_CHECKING

import pytest
from amplifier_core.llm_errors import (
    AbortError as KernelAbortError,
)
from amplifier_core.llm_errors import (
    AuthenticationError as KernelAuthenticationError,
)
from amplifier_core.llm_errors import (
    InvalidRequestError as KernelInvalidRequestError,
)
from amplifier_core.llm_errors import (
    NetworkError as KernelNetworkError,
)
from amplifier_core.llm_errors import (
    ProviderUnavailableError as KernelProviderUnavailableError,
)
from amplifier_core.llm_errors import (
    RateLimitError as KernelRateLimitError,
)

if TYPE_CHECKING:
    from amplifier_module_provider_github_copilot.error_translation import ErrorConfig


# Test fixtures
@pytest.fixture
def error_config() -> ErrorConfig:
    """Load error config from Python config module."""
    from amplifier_module_provider_github_copilot.error_translation import load_error_config

    return load_error_config()


class TestErrorConfigLoading:
    """Test error config loading from Python config module.

    Contract: error-hierarchy:Default:MUST:1
    """

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
        """Unknown errors become ProviderUnavailableError.

        Contract: error-hierarchy:Default:MUST:1
        """
        # Contract: error-hierarchy:Default:MUST:1
        from amplifier_module_provider_github_copilot.error_translation import (
            load_error_config,
            translate_sdk_error,
        )

        config = load_error_config()

        class UnknownSDKError(Exception):
            pass

        result = translate_sdk_error(UnknownSDKError("something went wrong"), config)
        # Contract: error-hierarchy:Default:MUST:1
        assert type(result) is KernelProviderUnavailableError

    def test_error_has_provider_attribute(self) -> None:
        """All errors have provider='github-copilot'.

        Contract: error-hierarchy:Kernel:MUST:2
        """
        # Contract: error-hierarchy:Kernel:MUST:2
        from amplifier_module_provider_github_copilot.error_translation import (
            load_error_config,
            translate_sdk_error,
        )

        config = load_error_config()

        class SomeError(Exception):
            pass

        result = translate_sdk_error(SomeError("test"), config)
        assert result.provider == "github-copilot"

    def test_original_exception_chained(self) -> None:
        """Original exception chained via __cause__.

        Contract: error-hierarchy:Translation:MUST:3
        """
        # Contract: error-hierarchy:Translation:MUST:3
        from amplifier_module_provider_github_copilot.error_translation import (
            load_error_config,
            translate_sdk_error,
        )

        config = load_error_config()

        class OriginalError(Exception):
            pass

        original = OriginalError("original message")
        result = translate_sdk_error(original, config)
        assert result.__cause__ is original

    def test_translate_never_raises(self) -> None:
        """translate_sdk_error MUST NOT raise.

        Contract: error-hierarchy:Translation:MUST:1
        """
        # Contract: error-hierarchy:Translation:MUST:1
        from amplifier_core.llm_errors import LLMError

        from amplifier_module_provider_github_copilot.error_translation import (
            load_error_config,
            translate_sdk_error,
        )

        config = load_error_config()

        # Even with weird inputs, should not raise
        result = translate_sdk_error(Exception("test"), config)
        assert isinstance(result, LLMError)
        assert result.provider == "github-copilot"


class TestErrorMappingPatterns:
    """Test pattern matching for error translation."""

    def test_match_by_type_name(self) -> None:
        """Match SDK error by type name.

        Contract: error-hierarchy:Translation:MUST:2
        """
        # Contract: error-hierarchy:Translation:MUST:2
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

        # Local class named AuthenticationError to match sdk_patterns
        class AuthenticationError(Exception):  # noqa: A001
            pass

        result = translate_sdk_error(AuthenticationError("invalid token"), config)
        # Contract: error-hierarchy:Translation:MUST:2
        assert type(result) is KernelAuthenticationError
        assert result.retryable is False

    def test_match_by_string_pattern(self) -> None:
        """Match SDK error by string pattern in message.

        Contract: error-hierarchy:Translation:MUST:2
        """
        # Contract: error-hierarchy:Translation:MUST:2
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
        # Contract: error-hierarchy:RateLimit:MUST:1
        assert type(result) is KernelRateLimitError
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
        # Contract: error-hierarchy:Translation:MUST:2
        assert type(result) is KernelAuthenticationError


class TestRateLimitRetryAfter:
    """Test retry_after extraction for rate limit errors."""

    def test_extract_retry_after_from_message(self) -> None:
        """RateLimitError extracts retry_after when present.

        Contract: error-hierarchy:RateLimit:MUST:1
        """
        # Contract: error-hierarchy:RateLimit:MUST:1
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

        # Local class named RateLimitError to match sdk_patterns
        class RateLimitError(Exception):  # noqa: A001
            pass

        # Error message contains retry_after hint
        result = translate_sdk_error(
            RateLimitError("Rate limit exceeded. Retry after 30 seconds."),
            config,
        )
        # Contract: error-hierarchy:RateLimit:MUST:1
        assert type(result) is KernelRateLimitError
        expected_retry = 30.0
        assert result.retry_after == expected_retry, (
            f"RateLimit:MUST:1 — retry_after must be extracted, got {result.retry_after}"
        )


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
        # Contract: error-hierarchy:ImageValidation:MUST:1
        assert type(result) is KernelInvalidRequestError

    def test_image_validation_error_is_not_retryable(self) -> None:
        """Contract: error-hierarchy:ImageValidation:MUST:2 — Image errors are non-retryable."""
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
        """Contract: error-hierarchy:ImageValidation:MUST:3 — Original exception chained."""
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

        # Contract: error-hierarchy:ConnectionError:MUST:1
        assert type(result) is KernelProviderUnavailableError, (
            f"ConnectionError must map to ProviderUnavailableError per "
            f"error-hierarchy:ConnectionError:MUST:1, got {type(result).__name__}"
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

        # Contract: error-hierarchy:ConnectionError:MUST:1
        assert type(result) is KernelNetworkError, (
            f"ProcessExitedError must map to NetworkError, got {type(result).__name__}"
        )
        assert result.retryable is True


class TestAbortErrorTranslation:
    """error-hierarchy:AbortError:MUST:1 — SDK abort/cancel errors must translate to AbortError."""

    def test_abort_error_translation(self) -> None:
        """AbortError or CancelledError must translate to kernel AbortError(retryable=False).

        Contract: error-hierarchy:AbortError:MUST:1
        """
        # Contract: error-hierarchy:AbortError:MUST:1
        from amplifier_module_provider_github_copilot.error_translation import (
            load_error_config,
            translate_sdk_error,
        )

        config = load_error_config()

        # Local class named AbortError to match sdk_patterns
        class AbortError(Exception):  # noqa: A001
            pass

        result = translate_sdk_error(AbortError("operation aborted"), config)
        # Contract: error-hierarchy:AbortError:MUST:1
        assert type(result) is KernelAbortError, (
            f"AbortError:MUST:1 — expected AbortError, got {type(result).__name__}"
        )
        assert result.retryable is False, "AbortError:MUST:1 — abort errors must not be retryable"


class TestKernelTypesOnly:
    """error-hierarchy:Kernel:MUST:1 — All translations must produce kernel LLMError subtypes."""

    def test_kernel_types_only(self) -> None:
        """translate_sdk_error must always return a kernel LLMError subtype.

        Contract: error-hierarchy:Kernel:MUST:1
        """
        # Contract: error-hierarchy:Kernel:MUST:1
        from amplifier_core.llm_errors import LLMError

        from amplifier_module_provider_github_copilot.error_translation import (
            load_error_config,
            translate_sdk_error,
        )

        config = load_error_config()
        result = translate_sdk_error(RuntimeError("some error"), config)
        assert isinstance(result, LLMError), (
            f"Kernel:MUST:1 — result must be LLMError subtype, got {type(result)}"
        )
        assert result.provider == "github-copilot", (
            f"Kernel:MUST:1 — provider must be 'github-copilot', got {result.provider!r}"
        )


class TestConfigLoadingIdentical:
    """error-hierarchy:config:MUST:3 — Both config loading paths must produce identical results."""

    def test_config_loading_identical(self) -> None:
        """load_error_config() called twice must produce identical config objects.

        Contract: error-hierarchy:config:MUST:3
        """
        # Contract: error-hierarchy:config:MUST:3
        from amplifier_module_provider_github_copilot.error_translation import (
            load_error_config,
        )

        config1 = load_error_config()
        config2 = load_error_config()
        assert config1.default_error == config2.default_error
        assert config1.default_retryable == config2.default_retryable
        assert len(config1.mappings) == len(config2.mappings), (
            f"config:MUST:3 — mapping counts differ: "
            f"{len(config1.mappings)} vs {len(config2.mappings)}"
        )
