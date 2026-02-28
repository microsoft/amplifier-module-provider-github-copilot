"""
Tests for Copilot SDK provider exceptions.

This module tests the exception classes, their attributes,
and message formatting.
"""

import pytest

from amplifier_module_provider_github_copilot.exceptions import (
    CopilotAuthenticationError,
    CopilotConnectionError,
    CopilotModelNotFoundError,
    CopilotProviderError,
    CopilotRateLimitError,
    CopilotSessionError,
    CopilotTimeoutError,
    detect_rate_limit_error,
)


class TestCopilotProviderError:
    """Tests for base CopilotProviderError."""

    def test_base_exception(self):
        """Base exception should be catchable."""
        with pytest.raises(CopilotProviderError):
            raise CopilotProviderError("Test error")

    def test_inheritance(self):
        """All specific exceptions should inherit from base."""
        assert issubclass(CopilotAuthenticationError, CopilotProviderError)
        assert issubclass(CopilotConnectionError, CopilotProviderError)
        assert issubclass(CopilotRateLimitError, CopilotProviderError)
        assert issubclass(CopilotModelNotFoundError, CopilotProviderError)
        assert issubclass(CopilotSessionError, CopilotProviderError)
        assert issubclass(CopilotTimeoutError, CopilotProviderError)


class TestCopilotAuthenticationError:
    """Tests for CopilotAuthenticationError."""

    def test_basic_message(self):
        """Should accept basic message."""
        err = CopilotAuthenticationError("Login required")
        assert str(err) == "Login required"

    def test_caught_by_base(self):
        """Should be caught by base exception."""
        with pytest.raises(CopilotProviderError):
            raise CopilotAuthenticationError("Auth failed")


class TestCopilotConnectionError:
    """Tests for CopilotConnectionError."""

    def test_basic_message(self):
        """Should accept basic message."""
        err = CopilotConnectionError("Connection refused")
        assert str(err) == "Connection refused"


class TestCopilotRateLimitError:
    """Tests for CopilotRateLimitError."""

    def test_with_retry_after(self):
        """Should format message with retry_after."""
        err = CopilotRateLimitError(retry_after=30.0)
        assert err.retry_after == 30.0
        assert "30.0s" in str(err)

    def test_without_retry_after(self):
        """Should provide default message without retry_after."""
        err = CopilotRateLimitError()
        assert err.retry_after is None
        assert "retry later" in str(err).lower()

    def test_with_custom_message(self):
        """Should use custom message when provided."""
        err = CopilotRateLimitError(message="Custom rate limit message")
        assert str(err) == "Custom rate limit message"

    def test_retry_after_with_message(self):
        """Custom message should take precedence over retry_after."""
        err = CopilotRateLimitError(retry_after=60.0, message="Please wait")
        assert err.retry_after == 60.0
        assert str(err) == "Please wait"


class TestCopilotModelNotFoundError:
    """Tests for CopilotModelNotFoundError."""

    def test_with_model_only(self):
        """Should format message with just model name."""
        err = CopilotModelNotFoundError(model="unknown-model")
        assert err.model == "unknown-model"
        assert err.available == []
        assert "unknown-model" in str(err)
        assert "not found" in str(err).lower()

    def test_with_available_models(self):
        """Should include available models in message."""
        available = ["claude-opus-4-5", "claude-sonnet-4"]
        err = CopilotModelNotFoundError(model="bad-model", available=available)
        assert err.model == "bad-model"
        assert err.available == available
        assert "claude-opus-4-5" in str(err)
        assert "claude-sonnet-4" in str(err)

    def test_with_none_available(self):
        """Should handle None available gracefully."""
        err = CopilotModelNotFoundError(model="test", available=None)
        assert err.available == []

    def test_with_empty_available(self):
        """Should handle empty available list."""
        err = CopilotModelNotFoundError(model="test", available=[])
        assert "Available models" not in str(err)


class TestCopilotSessionError:
    """Tests for CopilotSessionError."""

    def test_basic_message(self):
        """Should accept basic message."""
        err = CopilotSessionError("Session creation failed")
        assert str(err) == "Session creation failed"


class TestCopilotTimeoutError:
    """Tests for CopilotTimeoutError."""

    def test_with_timeout(self):
        """Should format message with timeout value."""
        err = CopilotTimeoutError(timeout=300.0)
        assert err.timeout == 300.0
        assert "300.0s" in str(err)

    def test_without_timeout(self):
        """Should provide default message without timeout."""
        err = CopilotTimeoutError()
        assert err.timeout is None
        assert "timed out" in str(err).lower()

    def test_with_custom_message(self):
        """Should use custom message when provided."""
        err = CopilotTimeoutError(message="Request exceeded time limit")
        assert str(err) == "Request exceeded time limit"

    def test_timeout_with_message(self):
        """Custom message should take precedence."""
        err = CopilotTimeoutError(timeout=60.0, message="Too slow")
        assert err.timeout == 60.0
        assert str(err) == "Too slow"


class TestExceptionCatchPatterns:
    """Tests for common exception catch patterns."""

    def test_catch_all_provider_errors(self):
        """Should catch all provider errors with base class."""
        errors = [
            CopilotAuthenticationError("auth"),
            CopilotConnectionError("conn"),
            CopilotRateLimitError(retry_after=10),
            CopilotModelNotFoundError("model"),
            CopilotSessionError("session"),
            CopilotTimeoutError(timeout=30),
        ]

        for error in errors:
            try:
                raise error
            except CopilotProviderError as e:
                # All should be caught
                assert e is error

    def test_specific_catch_before_general(self):
        """Specific catches should work before general catch."""
        try:
            raise CopilotRateLimitError(retry_after=5.0)
        except CopilotRateLimitError as e:
            assert e.retry_after == 5.0
        except CopilotProviderError:
            pytest.fail("Should have been caught by specific exception")


class TestRateLimitDetection:
    """Tests for detect_rate_limit_error() helper function."""

    def test_normal_error_returns_none(self):
        """Non-rate-limit error messages should return None."""
        result = detect_rate_limit_error("Connection refused")
        assert result is None

    def test_empty_string_returns_none(self):
        """Empty string should return None."""
        result = detect_rate_limit_error("")
        assert result is None

    def test_detects_rate_limit_phrase(self):
        """Should detect 'rate limit' phrase."""
        result = detect_rate_limit_error("You have exceeded the rate limit")
        assert isinstance(result, CopilotRateLimitError)

    def test_detects_too_many_requests(self):
        """Should detect 'too many requests' phrase."""
        result = detect_rate_limit_error("Error: too many requests")
        assert isinstance(result, CopilotRateLimitError)

    def test_detects_429_status_code(self):
        """Should detect '429' status code."""
        result = detect_rate_limit_error("HTTP 429 response")
        assert isinstance(result, CopilotRateLimitError)

    def test_detects_throttle_variants(self):
        """Should detect 'throttl' variants (throttle, throttled, throttling)."""
        for msg in ["Request throttled", "Throttling in effect", "throttle limit"]:
            result = detect_rate_limit_error(msg)
            assert isinstance(result, CopilotRateLimitError), f"Failed for: {msg}"

    def test_detects_quota_exceeded(self):
        """Should detect 'quota exceeded' phrase."""
        result = detect_rate_limit_error("API quota exceeded for project")
        assert isinstance(result, CopilotRateLimitError)

    def test_case_insensitivity(self):
        """Detection should be case-insensitive."""
        result = detect_rate_limit_error("RATE LIMIT exceeded")
        assert isinstance(result, CopilotRateLimitError)

    def test_extracts_retry_after_integer(self):
        """Should extract integer retry_after seconds."""
        result = detect_rate_limit_error("Rate limited. Retry after 30 seconds")
        assert isinstance(result, CopilotRateLimitError)
        assert result.retry_after == 30.0

    def test_extracts_retry_after_float(self):
        """Should extract float retry_after seconds."""
        result = detect_rate_limit_error("Rate limited. retry-after: 2.5")
        assert isinstance(result, CopilotRateLimitError)
        assert result.retry_after == 2.5

    def test_retry_after_none_when_absent(self):
        """retry_after should be None when no retry info in message."""
        result = detect_rate_limit_error("rate limit exceeded")
        assert isinstance(result, CopilotRateLimitError)
        assert result.retry_after is None

    def test_preserves_original_message(self):
        """Error message should contain the original message text."""
        original = "rate limit exceeded, please slow down"
        result = detect_rate_limit_error(original)
        assert isinstance(result, CopilotRateLimitError)
        assert original in str(result)

    def test_extracts_retry_after_equals_syntax(self):
        """Should extract retry_after from 'Retry_After=60' syntax."""
        result = detect_rate_limit_error("rate_limit exceeded. Retry_After=60")
        assert isinstance(result, CopilotRateLimitError)
        assert result.retry_after == 60.0

    def test_no_false_positive_on_limit_alone(self):
        """Should NOT match 'limit' alone without 'rate'."""
        result = detect_rate_limit_error("You have reached the limit")
        assert result is None

    def test_no_false_positive_on_rate_alone(self):
        """Should NOT match 'rate' alone without 'limit'."""
        result = detect_rate_limit_error("The rate of requests is high")
        assert result is None
