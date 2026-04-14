"""Tests for error translation safety improvements.

Contract: contracts/error-hierarchy.md

These tests verify:
1. Pattern matching uses exact type name match (no false positives)
2. Kernel error construction handles varying constructor signatures
3. Exception messages are logged without sensitive data leakage
"""

from unittest.mock import patch

from amplifier_core.llm_errors import (
    AbortError,
    ProviderUnavailableError,
)

from amplifier_module_provider_github_copilot.error_translation import (
    KERNEL_ERROR_MAP,
    ErrorConfig,
    ErrorMapping,
    _matches_mapping,  # type: ignore[reportPrivateUsage]  # Testing internal matching logic
    translate_sdk_error,
)


class TestExactTypeMatching:
    """Tests for exact type name matching.

    Contract: error-hierarchy:Translation:MUST:1 — errors must map to correct types.
    """

    def test_error_matches_exact_type_name(self) -> None:
        """Pattern 'TimeoutError' matches exception named 'TimeoutError' exactly."""
        mapping = ErrorMapping(
            sdk_patterns=["TimeoutError"],
            kernel_error="LLMTimeoutError",
            retryable=True,
        )

        class TimeoutError(Exception):
            pass

        exc = TimeoutError("timed out")
        assert _matches_mapping(exc, mapping) is True

    def test_error_does_not_match_substring(self) -> None:
        """Pattern 'TimeoutError' should NOT match 'LLMTimeoutError' (substring).

        DESubstring matching causes false positives. Pattern 'Error'
        would match every exception. Pattern 'TimeoutError' matching
        'LLMTimeoutError' causes incorrect classification.
        """
        mapping = ErrorMapping(
            sdk_patterns=["TimeoutError"],
            kernel_error="LLMTimeoutError",
            retryable=True,
        )

        class LLMTimeoutError(Exception):
            pass

        exc = LLMTimeoutError("some timeout")
        # After fix: should NOT match due to exact matching
        assert _matches_mapping(exc, mapping) is False

    def test_error_pattern_does_not_match_every_exception(self) -> None:
        """Pattern 'Error' should NOT match arbitrary exceptions.

        DEWithout exact matching, 'Error' in 'AuthenticationError'
        would cause incorrect classification.
        """
        mapping = ErrorMapping(
            sdk_patterns=["Error"],  # Generic pattern
            kernel_error="ProviderUnavailableError",
            retryable=True,
        )

        class AuthenticationError(Exception):
            pass

        exc = AuthenticationError("auth failed")
        # After fix: should NOT match arbitrary exceptions
        assert _matches_mapping(exc, mapping) is False

    def test_authentication_error_does_not_match_network_auth_error(self) -> None:
        """Pattern 'AuthenticationError' should NOT match 'NetworkAuthenticationError'.

        AC: Negative matching test — ensure no false positives.
        """
        mapping = ErrorMapping(
            sdk_patterns=["AuthenticationError"],
            kernel_error="AuthenticationError",
            retryable=False,
        )

        class NetworkAuthenticationError(Exception):
            pass

        exc = NetworkAuthenticationError("network auth failed")
        # After fix: should NOT match (different type)
        assert _matches_mapping(exc, mapping) is False

    def test_string_patterns_still_work(self) -> None:
        """String patterns in message body should still match."""
        mapping = ErrorMapping(
            sdk_patterns=[],
            string_patterns=["connection refused"],
            kernel_error="NetworkError",
            retryable=True,
        )

        exc = Exception("Connection refused by server")
        assert _matches_mapping(exc, mapping) is True


class TestConstructorSafety:
    """Tests for safe kernel error construction.

    Contract: error-hierarchy:Translation:MUST:2 — translation must not crash.
    """

    def test_abort_error_construction_does_not_crash(self) -> None:
        """AbortError may not accept retry_after — construction must not raise.

        DEOnly InvalidToolCallError was special-cased. Other kernel
        errors with different signatures would cause TypeError.
        """
        config = ErrorConfig(
            mappings=[
                ErrorMapping(
                    sdk_patterns=["AbortError", "CancelledError"],
                    kernel_error="AbortError",
                    retryable=False,
                )
            ]
        )

        class CancelledError(Exception):
            pass

        exc = CancelledError("user cancelled")

        # Should NOT raise TypeError
        result = translate_sdk_error(exc, config)

        # Should return AbortError
        assert isinstance(result, AbortError)
        assert result.retryable is False

    def test_error_with_unknown_constructor_signature_falls_back_safely(self) -> None:
        """If kernel error rejects retry_after, fall back gracefully.

        AC: Construction handles varying constructor signatures safely.
        """
        config = ErrorConfig(
            mappings=[
                ErrorMapping(
                    sdk_patterns=["CustomError"],
                    kernel_error="AbortError",  # AbortError may not accept retry_after
                    retryable=False,
                    extract_retry_after=True,  # Would try to pass retry_after
                )
            ]
        )

        class CustomError(Exception):
            pass

        exc = CustomError("Retry after 30 seconds")

        # Should NOT crash even with extract_retry_after=True
        result = translate_sdk_error(exc, config)
        # Contract: error-hierarchy:AbortError:MUST:1 — CustomError maps to AbortError per config
        assert type(result) is AbortError


class TestLogSanitization:
    """Tests for log message sanitization (security-guardian Finding #5).

    Contract: security — error messages must not leak tokens/credentials.
    """

    def test_token_not_logged_in_debug_output(self) -> None:
        """Error messages containing tokens should NOT appear in debug logs.

        AC: Exception messages are not included in debug log output.
        The debug log only includes the exception type name and translation
        result, not the raw message which may contain sensitive data.
        """
        config = ErrorConfig(
            mappings=[
                ErrorMapping(
                    sdk_patterns=["AuthenticationError"],
                    kernel_error="AuthenticationError",
                    retryable=False,
                )
            ]
        )

        class AuthenticationError(Exception):
            pass

        # Exception message contains a realistic PAT-shaped token (36 alphanumeric chars).
        # Concatenation avoids triggering the gate's secrets scanner on source itself.
        exc = AuthenticationError("Invalid token: ghp_" + "a" * 36)

        with patch(
            "amplifier_module_provider_github_copilot.error_translation.logger"
        ) as mock_logger:
            translate_sdk_error(exc, config)

            # Check that logger.debug was called
            assert mock_logger.debug.called

            # The logged message must not contain the raw PAT-shaped token.
            # Debug log only includes the exception type name, not raw message content.
            call_args = str(mock_logger.debug.call_args)
            assert "ghp_" + "a" * 36 not in call_args


class TestMappingIntegration:
    """Integration tests for error mapping with safety fixes."""

    def test_full_translation_with_exact_matching(self) -> None:
        """End-to-end test of translation with exact type matching."""
        config = ErrorConfig(
            mappings=[
                ErrorMapping(
                    sdk_patterns=["TimeoutError"],
                    kernel_error="LLMTimeoutError",
                    retryable=True,
                ),
                ErrorMapping(
                    sdk_patterns=["AuthenticationError"],
                    kernel_error="AuthenticationError",
                    retryable=False,
                ),
            ],
            default_error="ProviderUnavailableError",
            default_retryable=True,
        )

        # Create an exception that shouldn't match TimeoutError pattern
        class SomeOtherTimeoutError(Exception):
            pass

        exc = SomeOtherTimeoutError("some other timeout")
        result = translate_sdk_error(exc, config)

        # Should fall through to default (no exact match)
        assert isinstance(result, ProviderUnavailableError)

    def test_all_kernel_error_types_constructible(self) -> None:
        """All kernel error types in KERNEL_ERROR_MAP should be constructible.

        AC: Constructor safety — no TypeError on any mapped type.
        Contract: error-hierarchy:Translation:MUST:2 — translation must not crash.
        """
        from amplifier_module_provider_github_copilot.error_translation import (
            _create_kernel_error_safely,  # type: ignore[reportPrivateUsage]
        )

        for name, error_class in KERNEL_ERROR_MAP.items():
            # Use _create_kernel_error_safely to test the actual code path
            err = _create_kernel_error_safely(
                error_class,
                f"Test message for {name}",
                provider="github-copilot",
                model="gpt-4o",
                retryable=False,
                retry_after=None,
            )
            # Assert exact type match, not isinstance
            # — error_class or ProviderUnavailableError fallback
            assert type(err) is error_class or type(err) is ProviderUnavailableError
