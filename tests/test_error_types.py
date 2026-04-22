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
    ConfigurationError,
    ContextLengthError,
    ErrorConfig,
    InvalidToolCallError,
    LLMError,
    LLMTimeoutError,
    ProviderUnavailableError,
    StreamError,
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


class TestF035P0CircuitBreaker:
    """P0: Circuit breaker false positive fix (CRITICAL).

    # Contract: unanchored — see contracts/error-hierarchy.md mapping table
    """

    def test_circuit_breaker_pattern_exists(self, error_config: ErrorConfig) -> None:
        """P0:Exists - Circuit breaker pattern must exist in config."""
        circuit_patterns = [
            m
            for m in error_config.mappings
            if any("circuit breaker" in p.lower() for p in m.string_patterns)
        ]
        assert len(circuit_patterns) == 1, (
            "error-hierarchy — expected 1 circuit breaker pattern in config, "
            f"got {len(circuit_patterns)}"
        )
        assert circuit_patterns[0].retryable is False, "Circuit breaker must NOT be retryable"

    def test_circuit_breaker_before_timeout(self, error_config: ErrorConfig) -> None:
        """P0:Order - Circuit breaker MUST come before timeout pattern.

        # Contract: error-hierarchy:Translation:MUST:3
        """
        circuit_idx = None
        timeout_idx = None

        for i, m in enumerate(error_config.mappings):
            if any("circuit breaker" in p.lower() for p in m.string_patterns):
                circuit_idx = i
            if m.kernel_error == "LLMTimeoutError":
                timeout_idx = i

        assert isinstance(circuit_idx, int), "Circuit breaker pattern not found"
        assert isinstance(timeout_idx, int), "Timeout pattern not found"
        assert circuit_idx < timeout_idx, (
            f"Circuit breaker (idx={circuit_idx}) must come before timeout (idx={timeout_idx})"
        )

    def test_circuit_breaker_not_retryable(
        self, error_config: ErrorConfig, translate_fn: Callable[..., LLMError]
    ) -> None:
        """P0:Retryable - Circuit breaker MUST NOT be retryable.

        # Contract: error-hierarchy:Translation:MUST:3
        """
        exc = Exception("Circuit breaker TRIPPED: timeout=3720.0s > max=60.0s")
        result = translate_fn(exc, error_config)

        assert isinstance(result, ProviderUnavailableError)
        assert result.retryable is False

    def test_circuit_breaker_not_timeout_error(
        self, error_config: ErrorConfig, translate_fn: Callable[..., LLMError]
    ) -> None:
        """P0:FalsePositive - Circuit breaker MUST NOT match LLMTimeoutError.

        # Contract: error-hierarchy:Translation:MUST:3
        """
        exc = Exception("Circuit breaker TRIPPED: timeout=3720.0s > max=60.0s")
        result = translate_fn(exc, error_config)

        assert not isinstance(result, LLMTimeoutError)


class TestF035P1ContextLength:
    """P1: ContextLengthError mappings.

    # Contract: error-hierarchy:Translation:MUST:3
    """

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

        assert isinstance(result, ContextLengthError)
        assert result.retryable is False

    def test_400_without_token_not_context_error(
        self, error_config: ErrorConfig, translate_fn: Callable[..., LLMError]
    ) -> None:
        """P1:Negative - Generic 400 should NOT match ContextLengthError."""
        exc = Exception("HTTP 400 Bad Request: invalid JSON syntax")
        result = translate_fn(exc, error_config)

        assert not isinstance(result, ContextLengthError)


class TestF035P2StreamError:
    """P2: StreamError mappings.

    # Contract: error-hierarchy:Translation:MUST:3
    """

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

        assert isinstance(result, StreamError)
        assert result.retryable is True  # Streams are retryable


class TestF035P3InvalidToolCall:
    """P3: InvalidToolCallError mappings.

    # Contract: error-hierarchy:Translation:MUST:3
    """

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

        assert isinstance(result, InvalidToolCallError)
        assert result.retryable is False


class TestF035P4ConfigurationError:
    """P4: ConfigurationError mappings.

    # Contract: error-hierarchy:Translation:MUST:3
    """

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

        assert isinstance(result, ConfigurationError)
        assert result.retryable is False


class TestF035EdgeCases:
    """Edge cases error translation."""

    def test_empty_message_fallthrough(
        self, error_config: ErrorConfig, translate_fn: Callable[..., LLMError]
    ) -> None:
        """Edge - Empty message falls through to default.

        Contract: error-hierarchy:Default:MUST:1
        """
        exc = Exception("")
        result = translate_fn(exc, error_config)
        # Should get default ProviderUnavailableError
        assert isinstance(result, ProviderUnavailableError)
        assert result.retryable is False

    def test_timeout_without_circuit_breaker(
        self, error_config: ErrorConfig, translate_fn: Callable[..., LLMError]
    ) -> None:
        """Edge - Regular timeout still matches LLMTimeoutError."""
        exc = Exception("Request timed out after 30 seconds")
        result = translate_fn(exc, error_config)
        assert isinstance(result, LLMTimeoutError)
        assert result.retryable is True



