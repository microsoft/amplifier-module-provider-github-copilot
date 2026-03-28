"""
Tests for observability improvements.

Contract: contracts/observability.md

Tests verify that observability logging is added to error translation
and tool parsing without changing provider behavior.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pytest

from amplifier_module_provider_github_copilot.error_translation import (
    ErrorConfig,
    ErrorMapping,
    translate_sdk_error,
)
from amplifier_module_provider_github_copilot.tool_parsing import (
    parse_tool_calls,
)


class TestErrorTranslationLogging:
    """AC-1, AC-2, AC-3: Logger imported and DEBUG log emitted for translations."""

    def test_logger_is_imported(self) -> None:
        """AC-1: logger imported in error_translation.py."""
        from amplifier_module_provider_github_copilot import error_translation

        assert hasattr(error_translation, "logger")
        assert isinstance(error_translation.logger, logging.Logger)

    def test_debug_log_emitted_for_translation(self, caplog: pytest.LogCaptureFixture) -> None:
        """AC-2: DEBUG log emitted for every translate_sdk_error() call."""
        mapping = ErrorMapping(
            sdk_patterns=["AuthenticationError"],
            kernel_error="AuthenticationError",
            retryable=False,
        )
        config = ErrorConfig(mappings=[mapping])

        class AuthenticationError(Exception):
            pass

        exc = AuthenticationError("Invalid token")

        with caplog.at_level(logging.DEBUG):
            translate_sdk_error(exc, config)

        # Check DEBUG log was emitted
        assert any(record.levelno == logging.DEBUG for record in caplog.records)
        assert "[ERROR_TRANSLATION]" in caplog.text

    def test_log_includes_error_types_and_retryable(self, caplog: pytest.LogCaptureFixture) -> None:
        """AC-3: Log includes original error type, kernel error type, retryable flag."""
        mapping = ErrorMapping(
            sdk_patterns=["RateLimitError"],
            kernel_error="RateLimitError",
            retryable=True,
        )
        config = ErrorConfig(mappings=[mapping])

        class RateLimitError(Exception):
            pass

        exc = RateLimitError("Rate limit exceeded")

        with caplog.at_level(logging.DEBUG):
            translate_sdk_error(exc, config)

        # Check log content
        assert "RateLimitError" in caplog.text
        assert "retryable=True" in caplog.text

    def test_log_emitted_for_default_translation(self, caplog: pytest.LogCaptureFixture) -> None:
        """DEBUG log emitted even when no mapping matches (default path)."""
        config = ErrorConfig(
            mappings=[],
            default_error="ProviderUnavailableError",
            default_retryable=True,
        )

        exc = Exception("Unknown error")

        with caplog.at_level(logging.DEBUG):
            translate_sdk_error(exc, config)

        assert "[ERROR_TRANSLATION]" in caplog.text
        assert "ProviderUnavailableError" in caplog.text
        assert "default" in caplog.text


class TestToolParsingLogging:
    """AC-4, AC-5, AC-6: WARNING log for empty tool arguments."""

    def test_warning_logged_for_empty_arguments(self, caplog: pytest.LogCaptureFixture) -> None:
        """AC-4: WARNING log emitted when tool_call.arguments == {}."""

        @dataclass
        class MockToolCall:
            id: str
            name: str
            arguments: dict[str, Any]

        @dataclass
        class MockResponse:
            tool_calls: list[MockToolCall]

        response = MockResponse(
            tool_calls=[MockToolCall(id="tc_1", name="apply_patch", arguments={})]
        )

        with caplog.at_level(logging.WARNING):
            result = parse_tool_calls(response)

        # Verify parse succeeds
        assert len(result) == 1
        assert result[0].arguments == {}

        # Check WARNING log was emitted
        assert any(record.levelno == logging.WARNING for record in caplog.records)
        assert "[TOOL_PARSING]" in caplog.text

    def test_warning_includes_tool_name_and_id(self, caplog: pytest.LogCaptureFixture) -> None:
        """AC-5: Warning log includes tool name and ID."""

        @dataclass
        class MockToolCall:
            id: str
            name: str
            arguments: dict[str, Any]

        @dataclass
        class MockResponse:
            tool_calls: list[MockToolCall]

        response = MockResponse(
            tool_calls=[MockToolCall(id="tc_123", name="dangerous_tool", arguments={})]
        )

        with caplog.at_level(logging.WARNING):
            parse_tool_calls(response)

        assert "dangerous_tool" in caplog.text
        assert "tc_123" in caplog.text

    def test_log_format_uses_correct_tag(self, caplog: pytest.LogCaptureFixture) -> None:
        """AC-6: Log format uses [TOOL_PARSING] tag."""

        @dataclass
        class MockToolCall:
            id: str
            name: str
            arguments: dict[str, Any]

        @dataclass
        class MockResponse:
            tool_calls: list[MockToolCall]

        response = MockResponse(
            tool_calls=[MockToolCall(id="tc_1", name="test_tool", arguments={})]
        )

        with caplog.at_level(logging.WARNING):
            parse_tool_calls(response)

        assert "[TOOL_PARSING]" in caplog.text


class TestNoWarningForNonEmptyArguments:
    """Edge cases: no warning for non-empty arguments."""

    def test_no_warning_for_none_arguments(self, caplog: pytest.LogCaptureFixture) -> None:
        """No warning when arguments are None (different from empty dict)."""

        @dataclass
        class MockToolCall:
            id: str
            name: str
            arguments: dict[str, Any] | None

        @dataclass
        class MockResponse:
            tool_calls: list[MockToolCall]

        response = MockResponse(
            tool_calls=[MockToolCall(id="tc_1", name="test_tool", arguments=None)]  # type: ignore
        )

        with caplog.at_level(logging.WARNING):
            result = parse_tool_calls(response)

        # Parse succeeds with empty dict
        assert len(result) == 1
        # No warning for None (only for explicit {})
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_records) == 0

    def test_no_warning_for_populated_arguments(self, caplog: pytest.LogCaptureFixture) -> None:
        """No warning when arguments have content."""

        @dataclass
        class MockToolCall:
            id: str
            name: str
            arguments: dict[str, Any]

        @dataclass
        class MockResponse:
            tool_calls: list[MockToolCall]

        response = MockResponse(
            tool_calls=[MockToolCall(id="tc_1", name="test_tool", arguments={"path": "/test/path"})]
        )

        with caplog.at_level(logging.WARNING):
            result = parse_tool_calls(response)

        # Parse succeeds
        assert len(result) == 1
        assert result[0].arguments == {"path": "/test/path"}
        # No warning
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_records) == 0

    def test_no_warning_for_empty_string_values(self, caplog: pytest.LogCaptureFixture) -> None:
        """No warning when arguments have keys with empty string values."""

        @dataclass
        class MockToolCall:
            id: str
            name: str
            arguments: dict[str, Any]

        @dataclass
        class MockResponse:
            tool_calls: list[MockToolCall]

        response = MockResponse(
            tool_calls=[MockToolCall(id="tc_1", name="test_tool", arguments={"key": ""})]
        )

        with caplog.at_level(logging.WARNING):
            result = parse_tool_calls(response)

        # Parse succeeds
        assert len(result) == 1
        # No warning (has keys, just empty values)
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_records) == 0


class TestMultipleToolCalls:
    """Edge case: multiple tool calls with some empty args."""

    def test_warning_for_each_empty_args_tool(self, caplog: pytest.LogCaptureFixture) -> None:
        """Warning logged for each tool call with empty arguments."""

        @dataclass
        class MockToolCall:
            id: str
            name: str
            arguments: dict[str, Any]

        @dataclass
        class MockResponse:
            tool_calls: list[MockToolCall]

        response = MockResponse(
            tool_calls=[
                MockToolCall(id="tc_1", name="tool_a", arguments={}),
                MockToolCall(id="tc_2", name="tool_b", arguments={"x": 1}),
                MockToolCall(id="tc_3", name="tool_c", arguments={}),
            ]
        )

        with caplog.at_level(logging.WARNING):
            result = parse_tool_calls(response)

        # All three parsed
        assert len(result) == 3

        # Two warnings (for tc_1 and tc_3)
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_records) == 2
        assert "tool_a" in caplog.text
        assert "tool_c" in caplog.text


class TestNoRegressions:
    """AC-7: All existing tool parsing behavior preserved."""

    def test_basic_parsing_still_works(self) -> None:
        """Basic tool parsing without logging concerns."""

        @dataclass
        class MockToolCall:
            id: str
            name: str
            arguments: dict[str, Any]

        @dataclass
        class MockResponse:
            tool_calls: list[MockToolCall]

        response = MockResponse(
            tool_calls=[
                MockToolCall(id="tc_1", name="read_file", arguments={"path": "/workspace/test.py"})
            ]
        )

        result = parse_tool_calls(response)

        assert len(result) == 1
        assert result[0].id == "tc_1"
        assert result[0].name == "read_file"
        assert result[0].arguments == {"path": "/workspace/test.py"}

    def test_json_string_arguments_still_parsed(self) -> None:
        """JSON string arguments are still parsed correctly."""

        @dataclass
        class MockToolCall:
            id: str
            name: str
            arguments: str

        @dataclass
        class MockResponse:
            tool_calls: list[MockToolCall]

        response = MockResponse(
            tool_calls=[MockToolCall(id="tc_1", name="test", arguments='{"key": "value"}')]
        )

        result = parse_tool_calls(response)

        assert len(result) == 1
        assert result[0].arguments == {"key": "value"}


class TestObservabilityConfigLoading:
    """Tests for observability config loading edge cases.

    Contract: behaviors:Observability:SHOULD:1 — graceful degradation on config errors.
    """

    def test_load_observability_config_returns_config(self) -> None:
        """load_observability_config returns ObservabilityConfig."""
        from amplifier_module_provider_github_copilot.observability import (
            ObservabilityConfig,
            load_observability_config,
        )

        config = load_observability_config()
        assert isinstance(config, ObservabilityConfig)

    def test_default_observability_config_has_expected_defaults(self) -> None:
        """_default_observability_config returns sane defaults."""
        from amplifier_module_provider_github_copilot.observability import (
            _default_observability_config,  # pyright: ignore[reportPrivateUsage]
        )

        config = _default_observability_config()
        assert config.provider_name == "github-copilot"
        assert config.events_enabled is True
        assert config.raw_payloads is False

    def test_observability_config_has_event_names(self) -> None:
        """Loaded config has event_names from observability.yaml."""
        from amplifier_module_provider_github_copilot.observability import (
            load_observability_config,
        )

        config = load_observability_config()
        assert config.event_names is not None
        # Check key event name attributes exist
        assert hasattr(config.event_names, "llm_request")
        assert hasattr(config.event_names, "llm_response")

    def test_observability_config_has_status_values(self) -> None:
        """Loaded config has status values from observability.yaml."""
        from amplifier_module_provider_github_copilot.observability import (
            load_observability_config,
        )

        config = load_observability_config()
        assert config.status is not None
        assert hasattr(config.status, "ok")
        assert hasattr(config.status, "error")


class TestLlmLifecycleContext:
    """Tests for llm_lifecycle context manager."""

    @pytest.mark.asyncio
    async def test_llm_lifecycle_yields_context(self) -> None:
        """llm_lifecycle yields LlmLifecycleContext."""
        from amplifier_module_provider_github_copilot.observability import (
            LlmLifecycleContext,
            llm_lifecycle,
        )

        async with llm_lifecycle(coordinator=None, model="gpt-4o") as ctx:
            assert isinstance(ctx, LlmLifecycleContext)
            assert ctx.model == "gpt-4o"

    @pytest.mark.asyncio
    async def test_llm_lifecycle_accepts_config(self) -> None:
        """llm_lifecycle accepts pre-loaded config."""
        from amplifier_module_provider_github_copilot.observability import (
            ObservabilityConfig,
            llm_lifecycle,
        )

        custom_config = ObservabilityConfig(provider_name="test-provider")

        async with llm_lifecycle(coordinator=None, model="gpt-4o", config=custom_config) as ctx:
            assert ctx.config.provider_name == "test-provider"

    @pytest.mark.asyncio
    async def test_llm_lifecycle_loads_config_when_none(self) -> None:
        """llm_lifecycle loads config when config=None."""
        from amplifier_module_provider_github_copilot.observability import (
            llm_lifecycle,
        )

        async with llm_lifecycle(coordinator=None, model="gpt-4o", config=None) as ctx:
            # Config should be loaded from yaml
            assert ctx.config is not None
            assert ctx.config.provider_name == "github-copilot"


# =============================================================================
# Additional Coverage Tests
# Coverage targets: lines 111-120, 123, 129, 168-170
# =============================================================================


class TestObservabilityConfigFallbacks:
    """Tests for config loading fallback paths.

    Covers lines 111-120, 123, 129, 168-170 in observability.py.
    """

    def test_config_fallback_to_filesystem_path(self) -> None:
        """load_observability_config falls back to filesystem on importlib failure.

        Covers lines 111-120.
        """
        from unittest.mock import patch

        from amplifier_module_provider_github_copilot.observability import (
            load_observability_config,
        )

        # Clear cache
        load_observability_config.cache_clear()

        # Mock importlib.resources to fail
        with patch("importlib.resources.files") as mock_files:
            mock_files.side_effect = ModuleNotFoundError("No module")

            config = load_observability_config()

        # Should still return valid config from filesystem fallback (or defaults if not found)
        assert config is not None
        assert config.provider_name == "github-copilot"

        # Clear cache for other tests
        load_observability_config.cache_clear()

    def test_config_returns_default_on_empty_yaml(self) -> None:
        """load_observability_config returns defaults when YAML is empty.

        Covers line 129.
        """
        from unittest.mock import MagicMock, patch

        from amplifier_module_provider_github_copilot.observability import (
            load_observability_config,
        )

        # Clear cache
        load_observability_config.cache_clear()

        # Mock importlib to return empty file content
        MagicMock()
        mock_config_file = MagicMock()
        mock_config_file.read_text.return_value = ""  # Empty content

        with patch("importlib.resources.files") as mock_resources:
            mock_resources.return_value.joinpath.return_value = mock_config_file

            config = load_observability_config()

        # Should return default config
        assert config is not None
        assert config.provider_name == "github-copilot"

        # Clear cache for other tests
        load_observability_config.cache_clear()

    def test_config_raises_on_yaml_parse_error(self) -> None:
        """load_observability_config raises when YAML parsing fails.

        P3 Fix: Re-raise prevents lru_cache from caching failure result.
        Three-Medium Architecture: YAML errors should fail-fast.
        Covers lines 168-170.
        """
        from unittest.mock import MagicMock, patch

        import yaml

        from amplifier_module_provider_github_copilot.observability import (
            load_observability_config,
        )

        # Clear cache
        load_observability_config.cache_clear()

        # Mock importlib to return invalid YAML
        mock_config_file = MagicMock()
        mock_config_file.read_text.return_value = "invalid: yaml: content: [[[["

        with patch("importlib.resources.files") as mock_resources:
            mock_resources.return_value.joinpath.return_value = mock_config_file

            # P3 Fix: Now raises instead of returning fallback
            with pytest.raises(yaml.YAMLError):
                load_observability_config()

        # Clear cache for other tests
        load_observability_config.cache_clear()

    def test_config_returns_default_when_both_sources_fail(self) -> None:
        """load_observability_config returns defaults when both importlib and path fail.

        Covers lines 123 (warning log and calling _default_observability_config).
        """
        from pathlib import Path
        from unittest.mock import patch

        from amplifier_module_provider_github_copilot.observability import (
            load_observability_config,
        )

        # Clear cache
        load_observability_config.cache_clear()

        # Mock both sources to fail
        with patch("importlib.resources.files") as mock_files:
            mock_files.side_effect = ModuleNotFoundError("No module")

            original_exists = Path.exists

            def mock_exists(self: Path) -> bool:
                if "observability.yaml" in str(self):
                    return False
                return original_exists(self)

            with patch.object(Path, "exists", mock_exists):
                config = load_observability_config()

        # Should return default config
        assert config is not None
        assert config.provider_name == "github-copilot"
        assert config.events_enabled is True

        # Clear cache for other tests
        load_observability_config.cache_clear()

    def test_config_handles_none_yaml_data(self) -> None:
        """load_observability_config handles None from yaml.safe_load.

        Covers line 129.
        """
        from unittest.mock import MagicMock, patch

        from amplifier_module_provider_github_copilot.observability import (
            load_observability_config,
        )

        # Clear cache
        load_observability_config.cache_clear()

        # Mock importlib to return empty YAML (which safe_load returns as None)
        MagicMock()
        mock_config_file = MagicMock()
        mock_config_file.read_text.return_value = "# just a comment"

        with patch("importlib.resources.files") as mock_resources:
            mock_resources.return_value.joinpath.return_value = mock_config_file

            config = load_observability_config()

        # Should return default config
        assert config is not None
        assert config.provider_name == "github-copilot"

        # Clear cache for other tests
        load_observability_config.cache_clear()
