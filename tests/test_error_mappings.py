"""Tests for error config missing mappings.

Contract: contracts/error-hierarchy.md

Tests verify that AbortError, session lifecycle errors, and the
conservative default fallback are correctly mapped.
"""

from __future__ import annotations

from pathlib import Path

import yaml


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


class TestDefaultFallback:
    """Tests for default fallback behavior."""

    def test_unknown_error_is_not_retryable(self) -> None:
        """Unknown errors should default to retryable=False per Golden Vision.

        Contract anchor: error-hierarchy:Default:MUST:1
        """
        from amplifier_module_provider_github_copilot.error_translation import (
            ProviderUnavailableError,
            load_error_config,
            translate_sdk_error,
        )

        config = load_error_config()
        exc = Exception("Some completely unknown error type")

        result = translate_sdk_error(exc, config)

        assert isinstance(result, ProviderUnavailableError)
        # Default changed from retryable=True to retryable=False
        # per Golden Vision V2 conservative default
        assert result.retryable is False


class TestErrorsYamlSchema:
    """Tests for errors.yaml schema compliance."""

    def test_abort_error_mapping_exists(self) -> None:
        """errors.yaml must have AbortError mapping.

        Contract anchor: error-hierarchy:AbortError:MUST:1
        """
        config_path = (
            Path(__file__).parent.parent
            / "amplifier_module_provider_github_copilot"
            / "config"
            / "data"
            / "errors.yaml"
        )
        with config_path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Find AbortError mapping
        mappings = data.get("error_mappings", [])
        abort_mappings = [m for m in mappings if "AbortError" in m.get("sdk_patterns", [])]

        assert len(abort_mappings) == 1
        abort_mapping = abort_mappings[0]
        assert abort_mapping["kernel_error"] == "AbortError"
        assert abort_mapping["retryable"] is False

    def test_session_lifecycle_mappings_exist(self) -> None:
        """errors.yaml must have session lifecycle error mappings.

        Contract anchor: error-hierarchy:SessionLifecycle:SHOULD:1
        """
        config_path = (
            Path(__file__).parent.parent
            / "amplifier_module_provider_github_copilot"
            / "config"
            / "data"
            / "errors.yaml"
        )
        with config_path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)

        mappings = data.get("error_mappings", [])

        # Check SessionCreateError mapping
        create_mappings = [m for m in mappings if "SessionCreateError" in m.get("sdk_patterns", [])]
        assert len(create_mappings) == 1
        assert create_mappings[0]["retryable"] is True

        # Check SessionDestroyError mapping
        destroy_mappings = [
            m for m in mappings if "SessionDestroyError" in m.get("sdk_patterns", [])
        ]
        assert len(destroy_mappings) == 1
        assert destroy_mappings[0]["retryable"] is True

    def test_default_retryable_is_false(self) -> None:
        """Default fallback must be retryable=False per Golden Vision V2.

        Contract anchor: error-hierarchy:Default:MUST:1
        """
        config_path = (
            Path(__file__).parent.parent
            / "amplifier_module_provider_github_copilot"
            / "config"
            / "data"
            / "errors.yaml"
        )
        with config_path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)

        default = data.get("default", {})
        assert default.get("retryable") is False
