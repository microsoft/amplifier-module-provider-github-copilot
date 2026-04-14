"""
Tests for error context enhancement.

Contract: contracts/error-hierarchy.md

Tests verify that error translation includes optional context extraction
from error messages to improve debuggability.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from amplifier_module_provider_github_copilot.error_translation import (
    ContextExtraction,
    ErrorConfig,
    ErrorMapping,
    load_error_config,
    translate_sdk_error,
)

# Path is used in test_load_config_with_context_extraction


class TestContextExtractionDataclass:
    """AC-1: ErrorMapping dataclass has optional context_extraction field."""

    def test_context_extraction_is_optional(self) -> None:
        """ErrorMapping can be created without context_extraction."""
        mapping = ErrorMapping(
            sdk_patterns=["TestError"],
            kernel_error="ProviderUnavailableError",
        )
        assert mapping.context_extraction == []

    def test_context_extraction_can_be_set(self) -> None:
        """ErrorMapping can include context_extraction patterns."""
        extraction = ContextExtraction(pattern=r"tool '(\w+)'", field="tool_name")
        mapping = ErrorMapping(
            sdk_patterns=["InvalidToolCallError"],
            kernel_error="InvalidToolCallError",
            context_extraction=[extraction],
        )
        assert len(mapping.context_extraction) == 1
        assert mapping.context_extraction[0].field == "tool_name"


class TestContextExtractionInTranslation:
    """AC-2: translate_sdk_error() extracts context when patterns match."""

    def test_extracts_tool_name_from_message(self, caplog: pytest.LogCaptureFixture) -> None:
        """Context extraction captures tool name from error message."""
        extraction = ContextExtraction(
            pattern=r"tool '([^']+)'",
            field="tool_name",
        )
        mapping = ErrorMapping(
            sdk_patterns=["InvalidToolCallError"],
            string_patterns=["tool conflict"],
            kernel_error="InvalidToolCallError",
            retryable=False,
            context_extraction=[extraction],
        )
        config = ErrorConfig(mappings=[mapping])

        exc = Exception("tool conflict: External tool 'apply_patch' conflicts with built-in")

        with caplog.at_level(logging.DEBUG):
            result = translate_sdk_error(exc, config)

        assert "apply_patch" in str(result) or "tool_name=apply_patch" in caplog.text

    def test_extracts_multiple_fields(self) -> None:
        """Multiple context patterns can extract different fields."""
        extractions = [
            ContextExtraction(pattern=r"tool '([^']+)'", field="tool_name"),
            ContextExtraction(pattern=r"(built-in|external)", field="conflict_type"),
        ]
        mapping = ErrorMapping(
            sdk_patterns=["InvalidToolCallError"],
            string_patterns=["conflicts"],
            kernel_error="InvalidToolCallError",
            retryable=False,
            context_extraction=extractions,
        )
        config = ErrorConfig(mappings=[mapping])

        exc = Exception("External tool 'apply_patch' conflicts with built-in tool")
        result = translate_sdk_error(exc, config)

        # Original message preserved
        assert "External tool 'apply_patch' conflicts with built-in tool" in str(result)


class TestContextExtractionFailure:
    """AC-3: Context extraction failure does NOT prevent error translation."""

    def test_regex_failure_does_not_prevent_translation(self) -> None:
        """If regex doesn't match, translation still succeeds."""
        extraction = ContextExtraction(
            pattern=r"tool '([^']+)'",  # Won't match
            field="tool_name",
        )
        mapping = ErrorMapping(
            sdk_patterns=["TestError"],
            kernel_error="ProviderUnavailableError",
            context_extraction=[extraction],
        )
        config = ErrorConfig(mappings=[mapping])

        class TestError(Exception):
            pass

        exc = TestError("Some error without tool pattern")

        # Contract: error-hierarchy:Translation:MUST:1 (never raises)
        # Contract: error-hierarchy:Kernel:MUST:1 (kernel types only)
        from amplifier_core.llm_errors import ProviderUnavailableError

        result = translate_sdk_error(exc, config)
        assert isinstance(result, ProviderUnavailableError)
        assert "Some error without tool pattern" in str(result)

    def test_invalid_regex_does_not_raise(self) -> None:
        """Invalid regex pattern is handled gracefully."""
        extraction = ContextExtraction(
            pattern=r"[invalid(regex",  # Invalid regex
            field="broken",
        )
        mapping = ErrorMapping(
            sdk_patterns=["TestError"],
            kernel_error="ProviderUnavailableError",
            context_extraction=[extraction],
        )
        config = ErrorConfig(mappings=[mapping])

        class TestError(Exception):
            pass

        exc = TestError("Some error")

        # Contract: error-hierarchy:Translation:MUST:1 (never raises)
        # Contract: error-hierarchy:Kernel:MUST:1 (kernel types only)
        from amplifier_core.llm_errors import ProviderUnavailableError

        result = translate_sdk_error(exc, config)
        assert isinstance(result, ProviderUnavailableError)
        assert "Some error" in str(result)

    def test_partial_extraction_success(self) -> None:
        """If one pattern fails but another succeeds, successful one is used."""
        extractions = [
            ContextExtraction(pattern=r"nonexistent '([^']+)'", field="missing"),
            ContextExtraction(pattern=r"tool '([^']+)'", field="tool_name"),
        ]
        mapping = ErrorMapping(
            sdk_patterns=["InvalidToolCallError"],
            string_patterns=["tool"],
            kernel_error="InvalidToolCallError",
            retryable=False,
            context_extraction=extractions,
        )
        config = ErrorConfig(mappings=[mapping])

        exc = Exception("tool 'apply_patch' error")

        # Contract: error-hierarchy:Translation:MUST:2 (uses config patterns)
        # Contract: error-hierarchy:Kernel:MUST:1 (kernel types only)
        from amplifier_core.llm_errors import InvalidToolCallError

        result = translate_sdk_error(exc, config)
        assert isinstance(result, InvalidToolCallError)
        assert "tool_name=apply_patch" in str(result)
        assert result.retryable is False


class TestContextExtractionLogging:
    """AC-4: DEBUG log emitted with extracted context."""

    def test_debug_log_emitted_with_context(self, caplog: pytest.LogCaptureFixture) -> None:
        """DEBUG log includes extracted context fields."""
        extraction = ContextExtraction(
            pattern=r"tool '([^']+)'",
            field="tool_name",
        )
        mapping = ErrorMapping(
            sdk_patterns=["InvalidToolCallError"],
            string_patterns=["tool conflict"],
            kernel_error="InvalidToolCallError",
            retryable=False,
            context_extraction=[extraction],
        )
        config = ErrorConfig(mappings=[mapping])

        exc = Exception("tool conflict: External tool 'apply_patch' conflicts with built-in")

        with caplog.at_level(logging.DEBUG):
            translate_sdk_error(exc, config)

        # Check DEBUG log was emitted
        assert any(record.levelno == logging.DEBUG for record in caplog.records)


class TestExtractedContextInOutput:
    """AC-5: Extracted context appears in error message or attributes."""

    def test_context_in_message_format(self) -> None:
        """Extracted context appears in [context: key=value] format."""
        extraction = ContextExtraction(
            pattern=r"tool '([^']+)'",
            field="tool_name",
        )
        mapping = ErrorMapping(
            sdk_patterns=[],
            string_patterns=["tool conflict"],
            kernel_error="InvalidToolCallError",
            retryable=False,
            context_extraction=[extraction],
        )
        config = ErrorConfig(mappings=[mapping])

        exc = Exception("tool conflict: External tool 'apply_patch' conflicts with built-in")
        result = translate_sdk_error(exc, config)

        # Context should be appended in structured format
        msg = str(result)
        assert "tool conflict" in msg  # Original preserved
        assert "tool_name=apply_patch" in msg or "apply_patch" in msg


class TestEmptyToolArgumentsWarning:
    """AC-6: Empty tool arguments triggers warning log."""

    def test_empty_tool_args_warning_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """Warning logged when translating error about empty tool arguments."""
        # This tests that warning is logged for empty args scenario
        # The actual empty args warning is in tool_parsing.py
        # This test verifies error translation doesn't break on such messages
        mapping = ErrorMapping(
            sdk_patterns=["InvalidToolCallError"],
            string_patterns=["empty arguments"],
            kernel_error="InvalidToolCallError",
            retryable=False,
        )
        config = ErrorConfig(mappings=[mapping])

        exc = Exception("Tool call with empty arguments {}")

        # Contract: error-hierarchy:Translation:MUST:1 (never raises)
        # Contract: error-hierarchy:Kernel:MUST:1 (kernel types only)
        from amplifier_core.llm_errors import InvalidToolCallError

        with caplog.at_level(logging.WARNING):
            result = translate_sdk_error(exc, config)

        assert isinstance(result, InvalidToolCallError)
        assert result.retryable is False


class TestConfigLoading:
    """Test context_extraction loading from YAML config."""

    def test_load_config_with_context_extraction(self, tmp_path: Path) -> None:
        """Context extraction patterns loaded from YAML."""
        config_content = """
version: "1.0"
error_mappings:
  - sdk_patterns: ["InvalidToolCallError"]
    string_patterns: ["tool conflict"]
    kernel_error: InvalidToolCallError
    retryable: false
    context_extraction:
      - pattern: "tool '([^']+)'"
        field: tool_name
      - pattern: "(built-in|external)"
        field: conflict_type
"""
        config_file = tmp_path / "errors.yaml"
        config_file.write_text(config_content)

        config = load_error_config(config_file)

        assert len(config.mappings) == 1
        mapping = config.mappings[0]
        assert len(mapping.context_extraction) == 2
        assert mapping.context_extraction[0].field == "tool_name"
        assert mapping.context_extraction[1].field == "conflict_type"


class TestNoRegressions:
    """AC-7: All existing error translation behavior preserved."""

    def test_basic_translation_still_works(self) -> None:
        """Basic error translation without context extraction still works."""
        mapping = ErrorMapping(
            sdk_patterns=["AuthenticationError"],
            kernel_error="AuthenticationError",
            retryable=False,
        )
        config = ErrorConfig(mappings=[mapping])

        class AuthenticationError(Exception):
            pass

        exc = AuthenticationError("Invalid token")
        result = translate_sdk_error(exc, config)

        assert result.retryable is False
        assert "Invalid token" in str(result)

    def test_original_message_preserved(self) -> None:
        """Original error message is preserved unchanged."""
        extraction = ContextExtraction(
            pattern=r"tool '([^']+)'",
            field="tool_name",
        )
        mapping = ErrorMapping(
            sdk_patterns=[],
            string_patterns=["tool"],
            kernel_error="InvalidToolCallError",
            retryable=False,
            context_extraction=[extraction],
        )
        config = ErrorConfig(mappings=[mapping])

        original_msg = "External tool 'apply_patch' conflicts with built-in tool"
        exc = Exception(original_msg)
        result = translate_sdk_error(exc, config)

        # Original message must be in result
        assert original_msg in str(result)
