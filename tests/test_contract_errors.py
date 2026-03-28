"""
Contract Compliance Tests: Error Hierarchy.

Contract: contracts/error-hierarchy.md

Tests error mapping compliance with kernel types.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from amplifier_module_provider_github_copilot.error_translation import (
    ErrorConfig,
    load_error_config,
)


@pytest.fixture
def error_config() -> ErrorConfig:
    """Load error config from YAML."""
    from pathlib import Path

    config_path = (
        Path(__file__).parent.parent
        / "amplifier_module_provider_github_copilot"
        / "config"
        / "errors.yaml"
    )
    return load_error_config(config_path)


# Valid kernel error types from amplifier_core.llm_errors
VALID_KERNEL_ERRORS = {
    "AuthenticationError",
    "RateLimitError",
    "LLMTimeoutError",
    "ContentFilterError",
    "ProviderUnavailableError",
    "NetworkError",
    "NotFoundError",
    "QuotaExceededError",
    "StreamError",
    "AbortError",
    "InvalidRequestError",
    "ContextLengthError",
    "InvalidToolCallError",
    "ConfigurationError",
}


class TestErrorConfigCompliance:
    """error-hierarchy:Kernel:MUST:1,2 — Verify config satisfies contract."""

    def test_auth_errors_not_retryable(self, error_config: ErrorConfig) -> None:
        """error-hierarchy:Translation:MUST - AuthenticationError MUST be retryable=False."""
        auth_mappings = [
            m
            for m in error_config.mappings
            if "Authentication" in str(m.kernel_error) or "Auth" in str(m.kernel_error)
        ]

        # Should have at least one auth mapping
        assert len(auth_mappings) >= 1, "Must have AuthenticationError mapping"

        for mapping in auth_mappings:
            assert not mapping.retryable, (
                f"AuthenticationError mapping must have retryable=False, got {mapping.retryable}"
            )

    def test_rate_limit_retryable(self, error_config: ErrorConfig) -> None:
        """error-hierarchy:RateLimit:MUST:1 - RateLimitError MUST be retryable=True."""
        rate_mappings = [m for m in error_config.mappings if "RateLimit" in str(m.kernel_error)]

        assert len(rate_mappings) >= 1, "Must have RateLimitError mapping"

        for mapping in rate_mappings:
            assert mapping.retryable, "RateLimitError mapping must have retryable=True"

    def test_timeout_retryable(self, error_config: ErrorConfig) -> None:
        """error-hierarchy:Translation - LLMTimeoutError MUST be retryable=True."""
        timeout_mappings = [m for m in error_config.mappings if "Timeout" in str(m.kernel_error)]

        assert len(timeout_mappings) >= 1, "Must have timeout error mapping"

        for mapping in timeout_mappings:
            assert mapping.retryable, "Timeout errors should be retryable"

    def test_content_filter_not_retryable(self, error_config: ErrorConfig) -> None:
        """error-hierarchy:Translation - ContentFilterError MUST be retryable=False."""
        filter_mappings = [
            m for m in error_config.mappings if "ContentFilter" in str(m.kernel_error)
        ]

        for mapping in filter_mappings:
            assert not mapping.retryable, "ContentFilter errors should not be retryable"

    def test_has_default_fallback(self, error_config: ErrorConfig) -> None:
        """error-hierarchy:Default:MUST:1 - Has default fallback mapping."""
        # ErrorConfig has default_error and default_retryable attributes
        assert error_config.default_error is not None, "Must have default error type"
        assert (
            error_config.default_error in VALID_KERNEL_ERRORS
            or error_config.default_error == "ProviderUnavailableError"
        )


class TestErrorTranslationFunction:
    """error-hierarchy:Translation:MUST:1-3"""

    def test_translation_never_raises(self, error_config: ErrorConfig) -> None:
        """error-hierarchy:Translation:MUST:1 - translate_sdk_error never raises."""
        from amplifier_module_provider_github_copilot.error_translation import (
            translate_sdk_error,
        )

        # Test with various exception types
        test_exceptions = [
            ValueError("test error"),
            RuntimeError("runtime error"),
            Exception("generic exception"),
            TypeError("type error"),
        ]

        for exc in test_exceptions:
            result = translate_sdk_error(exc, error_config)
            # Should always return an LLMError, never raise
            assert result is not None
            assert hasattr(result, "provider")

    def test_sets_provider_attribute(self, error_config: ErrorConfig) -> None:
        """error-hierarchy:Kernel:MUST:2 - Sets provider='github-copilot'."""
        from amplifier_module_provider_github_copilot.error_translation import (
            translate_sdk_error,
        )

        result = translate_sdk_error(ValueError("test"), error_config)

        assert result.provider == "github-copilot"


class TestErrorConfigFile:
    """Test that config/errors.yaml exists and is valid.

    Uses __file__-relative paths for robust resolution.
    """

    def test_errors_yaml_exists(self) -> None:
        """Config file must exist."""
        # Use __file__-relative path for robust resolution
        config_path = (
            Path(__file__).parent.parent
            / "amplifier_module_provider_github_copilot"
            / "config"
            / "errors.yaml"
        )
        assert config_path.exists(), f"config/errors.yaml must exist at {config_path}"

    def test_errors_yaml_valid_yaml(self) -> None:
        """Config file must be valid YAML."""
        # Use __file__-relative path for robust resolution
        config_path = (
            Path(__file__).parent.parent
            / "amplifier_module_provider_github_copilot"
            / "config"
            / "errors.yaml"
        )
        content = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        assert content is not None
        assert "error_mappings" in content or "mappings" in content

    def test_errors_yaml_has_version(self) -> None:
        """Config file should have version field."""
        # Use __file__-relative path for robust resolution
        config_path = (
            Path(__file__).parent.parent
            / "amplifier_module_provider_github_copilot"
            / "config"
            / "errors.yaml"
        )
        content = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        assert "version" in content, "errors.yaml should have version field"


class TestConcreteErrorTranslation:
    """P2-7 FIX: Test actual error translation behavior with concrete inputs/outputs.

    These tests verify the error translation produces specific kernel error types,
    not just that returned values have correct structure.
    """

    def test_rate_limit_error_translation(self, error_config: ErrorConfig) -> None:
        """P2-7: RateLimitException → RateLimitError with retryable=True."""
        from amplifier_module_provider_github_copilot.error_translation import (
            RateLimitError,
            translate_sdk_error,
        )

        # Create exception matching SDK rate limit pattern
        class RateLimitException(Exception):
            """Mock SDK rate limit exception."""

            pass

        exc = RateLimitException("429 rate limit exceeded")
        result = translate_sdk_error(exc, error_config)

        assert isinstance(result, RateLimitError), f"Expected RateLimitError, got {type(result)}"
        assert result.retryable is True, "RateLimitError MUST be retryable"

    def test_auth_error_translation(self, error_config: ErrorConfig) -> None:
        """P2-7: Authentication errors → AuthenticationError with retryable=False."""
        from amplifier_module_provider_github_copilot.error_translation import (
            AuthenticationError,
            translate_sdk_error,
        )

        # Create exception matching SDK auth error pattern
        class AuthError(Exception):
            """Mock SDK auth exception."""

            pass

        exc = AuthError("401 Unauthorized")
        result = translate_sdk_error(exc, error_config)

        assert isinstance(result, AuthenticationError), (
            f"Expected AuthenticationError, got {type(result)}"
        )
        assert result.retryable is False, "AuthenticationError MUST NOT be retryable"

    def test_timeout_error_translation(self, error_config: ErrorConfig) -> None:
        """P2-7: TimeoutException → LLMTimeoutError with retryable=True."""
        from amplifier_module_provider_github_copilot.error_translation import (
            LLMTimeoutError,
            translate_sdk_error,
        )

        # Create exception matching SDK timeout pattern
        class TimeoutException(Exception):
            """Mock SDK timeout exception."""

            pass

        exc = TimeoutException("Request timed out after 3600 seconds")
        result = translate_sdk_error(exc, error_config)

        assert isinstance(result, LLMTimeoutError), f"Expected LLMTimeoutError, got {type(result)}"
        assert result.retryable is True, "LLMTimeoutError MUST be retryable"

    def test_unknown_error_fallback(self, error_config: ErrorConfig) -> None:
        """P2-7: Unknown errors → default ProviderUnavailableError."""
        from amplifier_module_provider_github_copilot.error_translation import (
            ProviderUnavailableError,
            translate_sdk_error,
        )

        exc = Exception("Some completely unknown error XYZ123")
        result = translate_sdk_error(exc, error_config)

        # Default fallback should be ProviderUnavailableError
        assert isinstance(result, ProviderUnavailableError), (
            f"Expected ProviderUnavailableError, got {type(result)}"
        )
        assert result.provider == "github-copilot"
