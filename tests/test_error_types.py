"""
Tests for error type expansion.

Contract: contracts/error-hierarchy.md

Acceptance Criteria:
- P0: Circuit breaker errors MUST NOT match LLMTimeoutError
- P1: Token/context errors MUST map to ContextLengthError
- P2: Stream interruption errors MUST map to StreamError
- P3: Tool errors MUST map to InvalidToolCallError
- P4: Config errors MUST map to ConfigurationError
"""

from collections.abc import Callable
from pathlib import Path

import pytest

from amplifier_module_provider_github_copilot.error_translation import (
    ErrorConfig,
    LLMError,
    load_error_config,
    translate_sdk_error,
)


@pytest.fixture
def error_config() -> ErrorConfig:
    """Load error config from YAML."""
    config_path = (
        Path(__file__).parent.parent
        / "amplifier_module_provider_github_copilot"
        / "config"
        / "data"
        / "errors.yaml"
    )
    return load_error_config(config_path)


@pytest.fixture
def translate_fn() -> Callable[..., LLMError]:
    """Get translate function."""
    return translate_sdk_error


class TestF035ErrorClassesExist:
    """New error classes must exist in error_translation module."""

    @pytest.mark.parametrize(
        "class_name",
        [
            "ContextLengthError",
            "InvalidRequestError",
            "StreamError",
            "InvalidToolCallError",
            "ConfigurationError",
        ],
    )
    def test_error_class_exists(self, class_name: str) -> None:
        """Classes - {class_name} must be importable."""
        import amplifier_module_provider_github_copilot.error_translation as et

        assert hasattr(et, class_name), f"{class_name} not found in error_translation"


class TestF035KernelErrorMap:
    """KERNEL_ERROR_MAP must include new types."""

    def test_kernel_error_map_has_new_types(self) -> None:
        """Map - All new error types must be in KERNEL_ERROR_MAP."""
        from amplifier_module_provider_github_copilot.error_translation import KERNEL_ERROR_MAP

        required_types = [
            "ContextLengthError",
            "InvalidRequestError",
            "StreamError",
            "InvalidToolCallError",
            "ConfigurationError",
        ]

        for error_type in required_types:
            assert error_type in KERNEL_ERROR_MAP, f"{error_type} missing from KERNEL_ERROR_MAP"


class TestF035P0CircuitBreaker:
    """P0: Circuit breaker false positive fix (CRITICAL)."""

    def test_circuit_breaker_pattern_exists(self, error_config: ErrorConfig) -> None:
        """P0:Exists - Circuit breaker pattern must exist in config."""
        circuit_patterns = [
            m
            for m in error_config.mappings
            if any("circuit breaker" in p.lower() for p in m.string_patterns)
        ]
        assert len(circuit_patterns) >= 1, "Must have circuit breaker pattern"
        assert circuit_patterns[0].retryable is False, "Circuit breaker must NOT be retryable"

    def test_circuit_breaker_before_timeout(self, error_config: ErrorConfig) -> None:
        """P0:Order - Circuit breaker MUST come before timeout pattern."""
        circuit_idx = None
        timeout_idx = None

        for i, m in enumerate(error_config.mappings):
            if any("circuit breaker" in p.lower() for p in m.string_patterns):
                circuit_idx = i
            if m.kernel_error == "LLMTimeoutError":
                timeout_idx = i

        assert circuit_idx is not None, "Circuit breaker pattern not found"
        assert timeout_idx is not None, "Timeout pattern not found"
        assert circuit_idx < timeout_idx, (
            f"Circuit breaker (idx={circuit_idx}) must come before timeout (idx={timeout_idx})"
        )

    def test_circuit_breaker_not_retryable(
        self, error_config: ErrorConfig, translate_fn: Callable[..., LLMError]
    ) -> None:
        """P0:Retryable - Circuit breaker MUST NOT be retryable."""
        exc = Exception("Circuit breaker TRIPPED: timeout=3720.0s > max=60.0s")
        result = translate_fn(exc, error_config)

        assert result.__class__.__name__ == "ProviderUnavailableError"
        assert result.retryable is False

    def test_circuit_breaker_not_timeout_error(
        self, error_config: ErrorConfig, translate_fn: Callable[..., LLMError]
    ) -> None:
        """P0:FalsePositive - Circuit breaker MUST NOT match LLMTimeoutError."""
        exc = Exception("Circuit breaker TRIPPED: timeout=3720.0s > max=60.0s")
        result = translate_fn(exc, error_config)

        assert result.__class__.__name__ != "LLMTimeoutError", (
            "Circuit breaker matched LLMTimeoutError - this causes infinite retry loops!"
        )


class TestF035P1ContextLength:
    """P1: ContextLengthError mappings."""

    @pytest.mark.parametrize(
        "message",
        [
            "CAPIError: 400 prompt token count of 140535 exceeds the limit of 128000",
            "CAPIError: 413 Request Entity Too Large",
            "context length exceeded",
            "token count 50000 exceeds limit",
        ],
    )
    def test_context_length_patterns(
        self, error_config: ErrorConfig, translate_fn: Callable[..., LLMError], message: str
    ) -> None:
        """P1 - Token/context errors MUST map to ContextLengthError."""
        result = translate_fn(Exception(message), error_config)

        assert result.__class__.__name__ == "ContextLengthError", (
            f"Expected ContextLengthError for '{message[:50]}...', got {result.__class__.__name__}"
        )
        assert result.retryable is False

    def test_400_without_token_not_context_error(
        self, error_config: ErrorConfig, translate_fn: Callable[..., LLMError]
    ) -> None:
        """P1:Negative - Generic 400 should NOT match ContextLengthError."""
        exc = Exception("HTTP 400 Bad Request: invalid JSON syntax")
        result = translate_fn(exc, error_config)

        assert result.__class__.__name__ != "ContextLengthError", (
            "Generic 400 error should not match ContextLengthError"
        )


class TestF035P2StreamError:
    """P2: StreamError mappings."""

    @pytest.mark.parametrize(
        "message",
        [
            "HTTP/2 GOAWAY: NO_ERROR (server gracefully closing connection)",
            "[Errno 32] Broken pipe",
            "Connection reset by peer",
            "stream terminated unexpectedly",
        ],
    )
    def test_stream_error_patterns(
        self, error_config: ErrorConfig, translate_fn: Callable[..., LLMError], message: str
    ) -> None:
        """P2 - Stream errors MUST map to StreamError."""
        result = translate_fn(Exception(message), error_config)

        assert result.__class__.__name__ == "StreamError", (
            f"Expected StreamError for '{message[:50]}...', got {result.__class__.__name__}"
        )
        assert result.retryable is True  # Streams are retryable


class TestF035P3InvalidToolCall:
    """P3: InvalidToolCallError mappings."""

    @pytest.mark.parametrize(
        "message",
        [
            'External tool "apply_patch" conflicts with a built-in tool',
            "Detected fake tool call text in response",
            "Detected 3 missing tool result(s)",
            "tool conflict detected",
        ],
    )
    def test_tool_error_patterns(
        self, error_config: ErrorConfig, translate_fn: Callable[..., LLMError], message: str
    ) -> None:
        """P3 - Tool errors MUST map to InvalidToolCallError."""
        result = translate_fn(Exception(message), error_config)

        assert result.__class__.__name__ == "InvalidToolCallError", (
            f"Expected InvalidToolCallError for '{message[:50]}', got {result.__class__.__name__}"
        )
        assert result.retryable is False


class TestF035P4ConfigurationError:
    """P4: ConfigurationError mappings."""

    @pytest.mark.parametrize(
        "message",
        [
            "gpt-3.5-turbo does not support reasoning effort configuration",
            "Model configuration error: invalid parameter",
            "does not support extended thinking",
        ],
    )
    def test_config_error_patterns(
        self, error_config: ErrorConfig, translate_fn: Callable[..., LLMError], message: str
    ) -> None:
        """P4 - Config errors MUST map to ConfigurationError."""
        result = translate_fn(Exception(message), error_config)

        assert result.__class__.__name__ == "ConfigurationError", (
            f"Expected ConfigurationError for '{message[:50]}...', got {result.__class__.__name__}"
        )
        assert result.retryable is False


class TestF035EdgeCases:
    """Edge cases error translation."""

    def test_empty_message_fallthrough(
        self, error_config: ErrorConfig, translate_fn: Callable[..., LLMError]
    ) -> None:
        """Edge - Empty message falls through to default."""
        exc = Exception("")
        result = translate_fn(exc, error_config)
        # Should get default ProviderUnavailableError
        assert result.__class__.__name__ == "ProviderUnavailableError"

    def test_timeout_without_circuit_breaker(
        self, error_config: ErrorConfig, translate_fn: Callable[..., LLMError]
    ) -> None:
        """Edge - Regular timeout still matches LLMTimeoutError."""
        exc = Exception("Request timed out after 30 seconds")
        result = translate_fn(exc, error_config)
        assert result.__class__.__name__ == "LLMTimeoutError"
        assert result.retryable is True

    def test_generic_connection_error_maps_to_provider_unavailable_error(
        self, error_config: ErrorConfig, translate_fn: Callable[..., LLMError]
    ) -> None:
        """Edge - Generic 'connection error' message maps to ProviderUnavailableError.

        Contract: error-hierarchy:ConnectionError:MUST:1 — connection failures
        indicate the provider endpoint is unreachable, not a transient network event.
        """
        exc = Exception("connection error occurred")
        result = translate_fn(exc, error_config)
        assert result.__class__.__name__ == "ProviderUnavailableError"
        assert result.retryable is True
