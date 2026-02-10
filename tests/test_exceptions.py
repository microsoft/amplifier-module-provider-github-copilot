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
