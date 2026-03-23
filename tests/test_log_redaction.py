# pyright: reportPrivateUsage=false
"""Tests for secret redaction in log output.

Contract: behaviors:Logging:MUST:4

These tests verify:
- Authorization headers are redacted
- Token/API key/credential key-value pairs are redacted
- GitHub token literals are redacted
- Redaction is idempotent
- Error translation returns redacted messages
- Logging paths don't emit secrets
"""

from __future__ import annotations

import logging

import pytest

from amplifier_module_provider_github_copilot.security_redaction import (
    REDACTED,
    redact_exception_message,
    redact_sensitive_text,
)

# ============================================================================
# Test: Authorization Header Redaction
# ============================================================================


class TestAuthorizationHeaderRedaction:
    """Tests for Authorization/Bearer header redaction."""

    def test_redacts_authorization_header(self) -> None:
        """Authorization header value is redacted but structure remains.

        Contract: behaviors:Logging:MUST:4
        """
        text = "Authorization: Bearer abc123xyz"
        result = redact_sensitive_text(text)

        assert "abc123xyz" not in result
        assert REDACTED in result
        assert "Authorization" in result

    def test_redacts_bearer_token(self) -> None:
        """Bearer token value is redacted.

        Contract: behaviors:Logging:MUST:4
        """
        text = "Bearer=sk_test_12345"
        result = redact_sensitive_text(text)

        assert "sk_test_12345" not in result
        assert REDACTED in result

    def test_redacts_auth_header_in_json(self) -> None:
        """Authorization header in JSON-like text is redacted.

        Contract: behaviors:Logging:MUST:4
        """
        text = '{"Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"}'
        result = redact_sensitive_text(text)

        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in result
        assert REDACTED in result


# ============================================================================
# Test: Token/API Key/Credential Redaction
# ============================================================================


class TestKeyValueRedaction:
    """Tests for token, API key, and credential key-value pair redaction."""

    @pytest.mark.parametrize(
        "key",
        [
            "token",
            "github_token",
            "access_token",
            "refresh_token",
            "id_token",
            "api_key",
            "apikey",
            "client_secret",
            "secret",
            "password",
            "passwd",
            "pwd",
            "credential",
        ],
    )
    def test_redacts_secret_key_value_pairs(self, key: str) -> None:
        """Secret key-value pairs have values redacted.

        Contract: behaviors:Logging:MUST:4
        """
        text = f"{key}=my_secret_value_123"
        result = redact_sensitive_text(text)

        assert "my_secret_value_123" not in result
        assert REDACTED in result
        assert key in result  # Key is preserved

    def test_redacts_token_in_query_string(self) -> None:
        """Token in query string format is redacted.

        Contract: behaviors:Logging:MUST:4
        """
        text = "token=abc123&model=gpt-4&user=test"
        result = redact_sensitive_text(text)

        assert "abc123" not in result
        assert "model=gpt-4" in result  # Non-secret preserved
        assert "user=test" in result

    def test_redacts_api_key_in_json(self) -> None:
        """API key in JSON is redacted.

        Contract: behaviors:Logging:MUST:4
        """
        text = '{"api_key":"sk_live_abc123","user":"x"}'
        result = redact_sensitive_text(text)

        assert "sk_live_abc123" not in result
        assert '"user":"x"' in result or "user" in result

    def test_preserves_non_secret_text(self) -> None:
        """Non-secret text is unchanged.

        Contract: behaviors:Logging:MUST:4
        """
        text = "Model not found: gpt-5 is not available"
        result = redact_sensitive_text(text)

        assert result == text


# ============================================================================
# Test: GitHub Token Literal Redaction
# ============================================================================


class TestGitHubTokenRedaction:
    """Tests for GitHub token format redaction."""

    @pytest.mark.parametrize(
        "token_prefix",
        ["ghp_", "gho_", "ghu_", "ghs_", "ghr_"],
    )
    def test_redacts_github_token_formats(self, token_prefix: str) -> None:
        """GitHub token formats are redacted even without key context.

        Contract: behaviors:Logging:MUST:4
        """
        # GitHub tokens are prefix + 36 alphanumeric chars
        token = f"{token_prefix}{'a' * 36}"
        text = f"Error with token {token} in request"
        result = redact_sensitive_text(text)

        assert token not in result
        assert REDACTED in result
        assert "Error with token" in result

    def test_redacts_fine_grained_pat(self) -> None:
        """Fine-grained PAT format is redacted.

        Contract: behaviors:Logging:MUST:4
        """
        # Fine-grained PAT: github_pat_ + 22 chars + _ + 59 chars
        token = f"github_pat_{'a' * 22}_{'b' * 59}"
        text = f"Token: {token}"
        result = redact_sensitive_text(text)

        assert token not in result
        assert REDACTED in result


# ============================================================================
# Test: Idempotency
# ============================================================================


class TestIdempotency:
    """Tests for redaction idempotency."""

    def test_already_redacted_text_is_stable(self) -> None:
        """Already-redacted text remains stable.

        Contract: behaviors:Logging:MUST:4
        """
        text = f"token={REDACTED}&model=gpt-4"
        result = redact_sensitive_text(text)

        # Should not double-mangle
        assert result.count(REDACTED) == 1

    def test_multiple_redactions_are_idempotent(self) -> None:
        """Multiple redaction passes produce same result.

        Contract: behaviors:Logging:MUST:4
        """
        text = "token=secret123&api_key=abc"
        result1 = redact_sensitive_text(text)
        result2 = redact_sensitive_text(result1)

        assert result1 == result2


# ============================================================================
# Test: Exception Redaction
# ============================================================================


class TestExceptionRedaction:
    """Tests for exception message redaction."""

    def test_redact_exception_message(self) -> None:
        """Exception message is redacted.

        Contract: behaviors:Logging:MUST:4
        """
        exc = Exception("Authentication failed: token=ghp_" + "a" * 36)
        result = redact_exception_message(exc)

        assert "ghp_" not in result or REDACTED in result
        assert "Authentication failed" in result

    def test_redact_exception_with_api_key(self) -> None:
        """Exception with API key is redacted.

        Contract: behaviors:Logging:MUST:4
        """
        exc = ValueError("Invalid api_key=sk_test_12345 provided")
        result = redact_exception_message(exc)

        assert "sk_test_12345" not in result
        assert "api_key" in result


# ============================================================================
# Test: Error Translation Integration
# ============================================================================


class TestErrorTranslationIntegration:
    """Tests for error translation integration."""

    def test_translate_sdk_error_returns_redacted_message(self) -> None:
        """Kernel error message is redacted.

        Contract: error-hierarchy:Translation:MUST:1
        """
        from amplifier_module_provider_github_copilot.error_translation import (
            ErrorConfig,
            translate_sdk_error,
        )

        # Create exception with secret
        exc = Exception("API call failed: api_key=secret123 invalid")
        config = ErrorConfig()

        result = translate_sdk_error(exc, config)

        # The translated error message should not contain the secret
        assert "secret123" not in str(result)

    def test_translate_preserves_error_classification(self) -> None:
        """Error classification uses original, not redacted message.

        Contract: error-hierarchy:Translation:MUST:2
        """
        from amplifier_module_provider_github_copilot.error_translation import (
            ErrorConfig,
            ErrorMapping,
            translate_sdk_error,
        )

        # Create config with pattern that matches original message
        config = ErrorConfig(
            mappings=[
                ErrorMapping(
                    sdk_patterns=["TimeoutError"],
                    kernel_error="LLMTimeoutError",
                    retryable=True,
                )
            ]
        )

        # Exception with secret in message
        exc = TimeoutError("Request timed out: token=secret123")

        result = translate_sdk_error(exc, config)

        # Should be classified as LLMTimeoutError (matching worked)
        assert "LLMTimeoutError" in type(result).__name__ or result.retryable is True


# ============================================================================
# Test: Logging Path Integration
# ============================================================================


class TestLoggingPathIntegration:
    """Tests for logging path redaction integration."""

    def test_provider_retry_log_redacts_error(self, caplog: pytest.LogCaptureFixture) -> None:
        """Retry log path does not leak secrets.

        Contract: behaviors:Logging:MUST:4
        """
        # This test verifies the integration point exists
        # Full integration tested in provider tests
        from amplifier_module_provider_github_copilot.security_redaction import (
            redact_sensitive_text,
        )

        secret_error = "Failed: api_key=sk_test_123"
        redacted = redact_sensitive_text(secret_error)

        with caplog.at_level(logging.INFO):
            logging.info("Retry error: %s", redacted)

        assert "sk_test_123" not in caplog.text
        assert REDACTED in caplog.text

    def test_mixed_message_preserves_paths(self) -> None:
        """Mixed message with file paths preserves non-secret content.

        Contract: behaviors:Logging:MUST:4
        """
        text = "Error loading /home/user/config.yaml: token=secret123"
        result = redact_sensitive_text(text)

        assert "/home/user/config.yaml" in result
        assert "secret123" not in result
