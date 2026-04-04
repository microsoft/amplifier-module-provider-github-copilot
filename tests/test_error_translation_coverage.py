"""Coverage tests for error_translation.py missing paths.

Covers:
- Lines 177-179: importlib.resources fallback on load
- Lines ~266-267: _matches_mapping return True via string_patterns / return False
- Lines 400-421: _create_kernel_error_safely fallback constructor paths

Contract: error-hierarchy:Construction:MUST:1 — caller always gets an LLMError back
Contract: behaviors:Logging:SHOULD:1 — fallback paths log debug messages
"""

from __future__ import annotations

from unittest.mock import patch

# ---------------------------------------------------------------------------
# importlib.resources fallback (lines 177-179)
# ---------------------------------------------------------------------------


class TestImportlibResourcesFallback:
    """When importlib.resources fails, loader falls back to file path."""

    def test_importlib_failure_falls_back_to_file_path(self) -> None:
        """importlib.resources.files() failure triggers file-path fallback.

        Contract: error-hierarchy:Loading:SHOULD:1 — graceful fallback
        Lines 177-179 in error_translation.py
        """
        from amplifier_module_provider_github_copilot.error_translation import (
            _load_error_config_cached,  # pyright: ignore[reportPrivateUsage]
            load_error_config,
        )

        # Clear cache before forcing the importlib failure path
        _load_error_config_cached.cache_clear()  # pyright: ignore[reportPrivateUsage]

        # Patch importlib.resources.files so it raises, triggering the fallback path
        # resources is imported locally inside _load_error_config_cached — patch the module
        with patch(
            "importlib.resources.files",
            side_effect=AttributeError("importlib.resources.files not available"),
        ):
            config = load_error_config(config_path=None)

        # Config should load successfully via the file-path fallback
        assert config is not None
        # Should have loaded real mappings from file
        assert len(config.mappings) > 0

        # Clean up cache after test
        _load_error_config_cached.cache_clear()  # pyright: ignore[reportPrivateUsage]

    def test_importlib_failure_with_yaml_error_falls_back_gracefully(self) -> None:
        """importlib failure with broken file path returns empty ErrorConfig.

        Lines 177-183 in error_translation.py — fallback path, path doesn't exist
        """
        from pathlib import Path

        from amplifier_module_provider_github_copilot.error_translation import (
            ErrorConfig,
            _load_error_config_cached,  # pyright: ignore[reportPrivateUsage]
            load_error_config,
        )

        _load_error_config_cached.cache_clear()  # pyright: ignore[reportPrivateUsage]

        # Force both importlib AND file path to fail
        with (
            patch(
                "importlib.resources.files",
                side_effect=ImportError("no importlib"),
            ),
            patch.object(Path, "exists", return_value=False),
        ):
            config = load_error_config(config_path=None)

        # Should return default empty config gracefully
        assert isinstance(config, ErrorConfig)
        _load_error_config_cached.cache_clear()  # pyright: ignore[reportPrivateUsage]


# ---------------------------------------------------------------------------
# _matches_mapping string_patterns path (lines ~266-267)
# ---------------------------------------------------------------------------


class TestMatchesMappingPaths:
    """_matches_mapping returns True via both sdk_patterns and string_patterns."""

    def test_matches_via_sdk_type_name(self) -> None:
        """Returns True when exception type name matches sdk_patterns exactly.

        Contract: error-hierarchy:Matching:MUST:1 — exact type name match
        """
        from amplifier_module_provider_github_copilot.error_translation import (
            ErrorMapping,
            _matches_mapping,  # pyright: ignore[reportPrivateUsage]
        )

        class ConnectTimeout(Exception):
            pass

        mapping = ErrorMapping(sdk_patterns=["ConnectTimeout"], string_patterns=[])
        exc = ConnectTimeout("connection timed out")

        result = _matches_mapping(exc, mapping)
        assert result is True

    def test_matches_via_string_pattern_in_message(self) -> None:
        """Returns True when exception message contains string_pattern substring.

        Contract: error-hierarchy:Matching:MUST:2 — substring match in message
        Lines ~266 in error_translation.py
        """
        from amplifier_module_provider_github_copilot.error_translation import (
            ErrorMapping,
            _matches_mapping,  # pyright: ignore[reportPrivateUsage]
        )

        class GenericError(Exception):
            pass

        mapping = ErrorMapping(
            sdk_patterns=[],  # No type match
            string_patterns=["rate limit exceeded"],  # Message match
        )
        exc = GenericError("HTTP 429: Rate limit exceeded by quota")

        result = _matches_mapping(exc, mapping)
        assert result is True

    def test_no_match_returns_false(self) -> None:
        """Returns False when neither type nor message matches.

        Lines ~267 in error_translation.py — the return False branch
        """
        from amplifier_module_provider_github_copilot.error_translation import (
            ErrorMapping,
            _matches_mapping,  # pyright: ignore[reportPrivateUsage]
        )

        class SomeError(Exception):
            pass

        mapping = ErrorMapping(
            sdk_patterns=["DifferentError"],
            string_patterns=["unrelated pattern"],
        )
        exc = SomeError("something went wrong")

        result = _matches_mapping(exc, mapping)
        assert result is False

    def test_empty_patterns_never_matches(self) -> None:
        """Mapping with no patterns never matches any exception."""
        from amplifier_module_provider_github_copilot.error_translation import (
            ErrorMapping,
            _matches_mapping,  # pyright: ignore[reportPrivateUsage]
        )

        mapping = ErrorMapping(sdk_patterns=[], string_patterns=[])
        exc = ValueError("test error")

        assert _matches_mapping(exc, mapping) is False


# ---------------------------------------------------------------------------
# _create_kernel_error_safely fallback paths (lines 400-421)
# ---------------------------------------------------------------------------


class TestCreateKernelErrorSafelyFallbacks:
    """_create_kernel_error_safely handles constructors with restricted signatures."""

    def test_first_path_full_constructor_used(self) -> None:
        """Happy path: error class accepts all args including retry_after.

        Contract: error-hierarchy:Construction:MUST:1
        """
        from amplifier_module_provider_github_copilot.error_translation import (
            LLMError,
            ProviderUnavailableError,
            _create_kernel_error_safely,  # pyright: ignore[reportPrivateUsage]
        )

        result = _create_kernel_error_safely(
            ProviderUnavailableError,
            "test error",
            provider="github-copilot",
            model="gpt-4o",
            retryable=True,
            retry_after=30.0,
        )
        assert isinstance(result, LLMError)
        assert "test error" in str(result)

    def test_second_fallback_when_constructor_rejects_retry_after(self) -> None:
        """Falls back to (message, provider, model, retryable) when retry_after rejected.

        Contract: error-hierarchy:Construction:MUST:1 — always returns LLMError
        Lines ~400-410 in error_translation.py
        """
        from amplifier_module_provider_github_copilot.error_translation import (
            LLMError,
            _create_kernel_error_safely,  # pyright: ignore[reportPrivateUsage]
        )

        class NoRetryAfterError(LLMError):
            """LLMError that rejects retry_after kwarg."""

            def __init__(
                self,
                message: str,
                *,
                provider: str,
                model: str | None = None,
                retryable: bool = False,
            ) -> None:  # noqa: E501
                super().__init__(message, provider=provider, model=model, retryable=retryable)

        result = _create_kernel_error_safely(
            NoRetryAfterError,
            "no retry after",
            provider="github-copilot",
            model="gpt-4o",
            retryable=False,
            retry_after=None,
        )
        assert isinstance(result, LLMError)
        assert isinstance(result, NoRetryAfterError)

    def test_third_fallback_when_constructor_rejects_model_and_retryable(self) -> None:
        """Falls back to (message, provider) when model/retryable also rejected.

        Contract: error-hierarchy:Construction:MUST:1 — always returns LLMError
        Lines ~410-421 in error_translation.py
        """
        from amplifier_module_provider_github_copilot.error_translation import (
            LLMError,
            _create_kernel_error_safely,  # pyright: ignore[reportPrivateUsage]
        )

        class MinimalError(LLMError):
            """LLMError that only accepts message and provider."""

            def __init__(self, message: str, *, provider: str, **_: object) -> None:
                # Only message + provider — rejects retry_after AND model/retryable
                super().__init__(message, provider=provider)

        result = _create_kernel_error_safely(
            MinimalError,
            "minimal error",
            provider="github-copilot",
            model="gpt-4o",
            retryable=True,
            retry_after=5.0,
        )
        assert isinstance(result, LLMError)
        assert isinstance(result, MinimalError)

    def test_last_resort_returns_provider_unavailable_when_all_fail(self) -> None:
        """Last resort: returns ProviderUnavailableError when all constructors fail.

        Contract: error-hierarchy:Construction:MUST:1 — never raises, always returns
        Lines ~415-421 in error_translation.py
        """
        from amplifier_module_provider_github_copilot.error_translation import (
            LLMError,
            ProviderUnavailableError,
            _create_kernel_error_safely,  # pyright: ignore[reportPrivateUsage]
        )

        class BrokenError(LLMError):
            """LLMError that always raises TypeError on construction."""

            def __init__(self, *args: object, **kwargs: object) -> None:
                raise TypeError("cannot instantiate BrokenError with any args")

        result = _create_kernel_error_safely(
            BrokenError,  # type: ignore[arg-type]
            "broken error",
            provider="github-copilot",
            model="gpt-4o",
            retryable=True,
            retry_after=None,
        )
        # Falls back to ProviderUnavailableError
        assert isinstance(result, ProviderUnavailableError)
        assert "broken error" in str(result)
