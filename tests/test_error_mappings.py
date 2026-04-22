"""Tests for error config missing mappings.

Contract: contracts/error-hierarchy.md

Tests verify that AbortError, session lifecycle errors, and the
conservative default fallback are correctly mapped.
"""

from __future__ import annotations


class TestAbortErrorMapping:
    """Tests for AbortError mapping."""

    def test_abort_error_produces_abort_error(self) -> None:
        """SDK AbortError should produce kernel AbortError(retryable=False).

        Contract anchor: error-hierarchy:AbortError:MUST:1
        """
        from amplifier_module_provider_github_copilot.error_translation import (
            AbortError,
            load_error_config,
            translate_sdk_error,
        )

        class MockAbortError(Exception):
            pass

        # Rename to match SDK pattern
        MockAbortError.__name__ = "AbortError"

        config = load_error_config()
        exc = MockAbortError("User aborted the operation")

        result = translate_sdk_error(exc, config)

        assert isinstance(result, AbortError)
        assert result.retryable is False

    def test_cancelled_error_produces_abort_error(self) -> None:
        """CancelledError should produce kernel AbortError.

        Contract anchor: error-hierarchy:AbortError:MUST:1
        """
        from amplifier_module_provider_github_copilot.error_translation import (
            AbortError,
            load_error_config,
            translate_sdk_error,
        )

        class MockCancelledError(Exception):
            pass

        MockCancelledError.__name__ = "CancelledError"

        config = load_error_config()
        exc = MockCancelledError("Task was cancelled")

        result = translate_sdk_error(exc, config)

        assert isinstance(result, AbortError)
        assert result.retryable is False

    def test_abort_string_pattern_match(self) -> None:
        """String 'abort' in message should produce AbortError.

        Contract anchor: error-hierarchy:AbortError:MUST:1
        """
        from amplifier_module_provider_github_copilot.error_translation import (
            AbortError,
            load_error_config,
            translate_sdk_error,
        )

        config = load_error_config()
        exc = RuntimeError("User requested abort")

        result = translate_sdk_error(exc, config)

        assert isinstance(result, AbortError)
        assert result.retryable is False


class TestSessionLifecycleErrors:
    """Tests for session lifecycle error mappings."""

    def test_session_create_error_produces_provider_unavailable(self) -> None:
        """SessionCreateError should produce ProviderUnavailableError(retryable=True).

        Contract anchor: error-hierarchy:SessionLifecycle:SHOULD:1
        """
        from amplifier_module_provider_github_copilot.error_translation import (
            ProviderUnavailableError,
            load_error_config,
            translate_sdk_error,
        )

        class MockSessionCreateError(Exception):
            pass

        MockSessionCreateError.__name__ = "SessionCreateError"

        config = load_error_config()
        exc = MockSessionCreateError("Failed to initialize session")

        result = translate_sdk_error(exc, config)

        assert isinstance(result, ProviderUnavailableError)
        assert result.retryable is True

    def test_session_destroy_error_produces_provider_unavailable(self) -> None:
        """SessionDestroyError should produce ProviderUnavailableError(retryable=True).

        Contract anchor: error-hierarchy:SessionLifecycle:SHOULD:1
        """
        from amplifier_module_provider_github_copilot.error_translation import (
            ProviderUnavailableError,
            load_error_config,
            translate_sdk_error,
        )

        class MockSessionDestroyError(Exception):
            pass

        MockSessionDestroyError.__name__ = "SessionDestroyError"

        config = load_error_config()
        exc = MockSessionDestroyError("Failed to cleanup session")

        result = translate_sdk_error(exc, config)

        assert isinstance(result, ProviderUnavailableError)
        assert result.retryable is True




