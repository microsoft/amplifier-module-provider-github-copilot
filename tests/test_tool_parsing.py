"""
Tests for tool parsing module.

Contract: provider-protocol.md (parse_tool_calls method)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pytest

from amplifier_module_provider_github_copilot.fake_tool_detection import (
    FakeToolDetectionConfig,
    LoggingConfig,
    _truncate_text,  # type: ignore[reportPrivateUsage]
    contains_fake_tool_calls,
    load_fake_tool_detection_config,
    log_detection,
    log_exhausted,
    log_retry,
    log_success,
    should_retry_for_fake_tool_calls,
)
from amplifier_module_provider_github_copilot.tool_parsing import ToolCall, parse_tool_calls

_fake_tool_config = load_fake_tool_detection_config()


# Mock types for testing (before implementation exists)
@dataclass
class MockToolCall:
    """Mock tool call from SDK response."""

    id: str
    name: str
    arguments: dict[str, Any] | str


@dataclass
class MockChatResponse:
    """Mock ChatResponse for testing."""

    content: list[Any]
    tool_calls: list[MockToolCall] | None = None


class TestParseToolCalls:
    """Tests for parse_tool_calls function.

    Contract: provider-protocol:parse_tool_calls:MUST:1
    """

    def test_empty_tool_calls_returns_empty_list(self) -> None:
        """No tool_calls in response → empty list.

        Contract: provider-protocol:parse_tool_calls:MUST:1
        """
        response = MockChatResponse(content=[], tool_calls=None)
        result = parse_tool_calls(response)
        assert result == []

    def test_empty_list_tool_calls_returns_empty_list(self) -> None:
        """Empty tool_calls list → empty list.

        Contract: provider-protocol:parse_tool_calls:MUST:1
        """

        response = MockChatResponse(content=[], tool_calls=[])
        result = parse_tool_calls(response)
        assert result == []

    def test_single_tool_call_parsed(self) -> None:
        """Single tool call extracted correctly.

        Contract: provider-protocol:parse_tool_calls:MUST:1
        """
        response = MockChatResponse(
            content=[],
            tool_calls=[MockToolCall(id="tc1", name="read_file", arguments={"path": "test.py"})],
        )
        result = parse_tool_calls(response)
        assert len(result) == 1
        assert result[0].id == "tc1"
        assert result[0].name == "read_file"
        assert result[0].arguments == {"path": "test.py"}

    def test_multiple_tool_calls_parsed(self) -> None:
        """Multiple tool calls all extracted.

        Contract: provider-protocol:parse_tool_calls:MUST:1
        """
        response = MockChatResponse(
            content=[],
            tool_calls=[
                MockToolCall(id="tc1", name="read_file", arguments={"path": "a.py"}),
                MockToolCall(
                    id="tc2", name="write_file", arguments={"path": "b.py", "content": "hello"}
                ),
                MockToolCall(id="tc3", name="bash", arguments={"command": "ls"}),
            ],
        )
        result = parse_tool_calls(response)
        assert len(result) == 3
        assert result[0].name == "read_file"
        assert result[1].name == "write_file"
        assert result[2].name == "bash"

    def test_string_arguments_parsed_as_json(self) -> None:
        """String arguments are JSON-parsed.

        Contract: provider-protocol:parse_tool_calls:MUST:1
        """
        response = MockChatResponse(
            content=[],
            tool_calls=[MockToolCall(id="tc1", name="bash", arguments='{"command": "ls -la"}')],
        )
        result = parse_tool_calls(response)
        assert result[0].arguments == {"command": "ls -la"}

    def test_invalid_json_raises_invalid_tool_call_error(self) -> None:
        """Invalid JSON arguments raise InvalidToolCallError (error-hierarchy.md:MUST:1)."""
        from amplifier_core.llm_errors import InvalidToolCallError

        response = MockChatResponse(
            content=[],
            tool_calls=[MockToolCall(id="tc1", name="bash", arguments="{invalid json}")],
        )
        with pytest.raises(InvalidToolCallError, match="Invalid JSON"):
            parse_tool_calls(response)

    def test_non_dict_direct_arguments_raise_invalid_tool_call_error(self) -> None:
        """Non-dict arguments (list, int) raise InvalidToolCallError.

        Contract: error-hierarchy.md:MUST:1 — malformed tool calls use kernel error type.
        provider-protocol.md:parse_tool_calls — arguments MUST be dict.
        """
        from amplifier_core.llm_errors import InvalidToolCallError

        class BadArgsToolCall:
            id = "tc1"
            name = "bash"
            arguments: list[str] = ["not", "a", "dict"]  # type: ignore[assignment]

        class BadArgsResponse:
            tool_calls = [BadArgsToolCall()]

        with pytest.raises(InvalidToolCallError, match="must be dict"):
            parse_tool_calls(BadArgsResponse())

    def test_json_array_arguments_raise_invalid_tool_call_error(self) -> None:
        """JSON string parsing to array raises InvalidToolCallError.

        Contract: error-hierarchy.md:MUST:1
        provider-protocol.md:parse_tool_calls — arguments MUST be dict (object), not array.
        Regression guard: json.loads('[1,2]') returns list, not dict — must be caught.
        """
        from amplifier_core.llm_errors import InvalidToolCallError

        response = MockChatResponse(
            content=[],
            tool_calls=[MockToolCall(id="tc1", name="bash", arguments='["not", "an", "object"]')],
        )
        with pytest.raises(InvalidToolCallError, match="object"):
            parse_tool_calls(response)


class TestToolCallType:
    """Tests for ToolCall dataclass.

    Contract: provider-protocol:parse_tool_calls:MUST:4
    """

    def test_tool_call_has_arguments_not_input(self) -> None:
        """ToolCall uses 'arguments' field per kernel contract (E3).

        Contract: provider-protocol:parse_tool_calls:MUST:4
        """
        tc = ToolCall(id="1", name="test", arguments={"key": "value"})
        assert tc.arguments == {"key": "value"}
        with pytest.raises(AttributeError):
            _ = tc.input  # type: ignore[attr-defined]

    def test_tool_call_fields(self) -> None:
        """ToolCall has required fields: id, name, arguments.

        Contract: provider-protocol:parse_tool_calls:MUST:4
        """
        tc = ToolCall(id="tc-123", name="read_file", arguments={"path": "/etc/hosts"})
        assert tc.id == "tc-123"
        assert tc.name == "read_file"
        assert tc.arguments == {"path": "/etc/hosts"}

    def test_tool_call_empty_arguments(self) -> None:
        """ToolCall can have empty arguments dict.

        Contract: provider-protocol:parse_tool_calls:MUST:4
        """
        tc = ToolCall(id="1", name="get_time", arguments={})
        assert tc.arguments == {}


class TestEdgeCases:
    """Edge case tests for tool parsing.

    Contract: provider-protocol:parse_tool_calls:MUST:1
    """

    def test_nested_arguments(self) -> None:
        """Nested dict arguments are preserved.

        Contract: provider-protocol:parse_tool_calls:MUST:1
        """
        nested = {"config": {"nested": {"deep": "value"}}, "list": [1, 2, 3]}
        response = MockChatResponse(
            content=[],
            tool_calls=[MockToolCall(id="tc1", name="complex", arguments=nested)],
        )
        result = parse_tool_calls(response)
        assert result[0].arguments == nested

    def test_unicode_in_arguments(self) -> None:
        """Unicode characters in arguments are preserved.

        Contract: provider-protocol:parse_tool_calls:MUST:1
        """
        response = MockChatResponse(
            content=[],
            tool_calls=[MockToolCall(id="tc1", name="write", arguments={"text": "Hello 世界 🌍"})],
        )
        result = parse_tool_calls(response)
        # pyright: ignore[reportArgumentType] - arguments is dict in this test
        assert result[0].arguments["text"] == "Hello 世界 🌍"  # type: ignore[index]

    def test_special_characters_in_tool_name(self) -> None:
        """Tool names with special characters are preserved.

        Contract: provider-protocol:parse_tool_calls:MUST:1
        """
        response = MockChatResponse(
            content=[],
            tool_calls=[MockToolCall(id="tc1", name="mcp_server:read_file", arguments={})],
        )
        result = parse_tool_calls(response)
        assert result[0].name == "mcp_server:read_file"


# ============================================================================
# Fake Tool Call Detection Tests
# Contract: provider-protocol:complete:MUST:6
# ============================================================================


class TestFakeToolCallDetection:
    """Tests for fake tool call detection patterns.

    Contract: provider-protocol:complete:MUST:6
    These tests verify detection of LLM-generated text that looks like
    tool calls but isn't actual structured tool calls.
    """

    def test_detects_bracket_style_fake_call(self) -> None:
        """[Tool Call: bash(...)] detected as fake.

        Contract: provider-protocol:complete:MUST:6
        """
        config = _fake_tool_config
        text = 'I will run this command: [Tool Call: bash(command="ls -la")]'
        detected, pattern = contains_fake_tool_calls(text, config)
        assert detected is True
        assert pattern == r"\[Tool Call:\s*\w+"

    def test_detects_xml_style_fake_call(self) -> None:
        """<tool_used name="bash"> detected as fake.

        Contract: provider-protocol:complete:MUST:6
        """
        config = _fake_tool_config
        text = '<tool_used name="bash"><command>ls -la</command></tool_used>'
        detected, pattern = contains_fake_tool_calls(text, config)
        assert detected is True
        assert pattern == r"<tool_used\s+name="

    def test_detects_xml_result_fake_call(self) -> None:
        """<tool_result name="bash"> detected as fake.

        Contract: provider-protocol:complete:MUST:6
        """
        config = _fake_tool_config
        text = '<tool_result name="bash">output here</tool_result>'
        detected, pattern = contains_fake_tool_calls(text, config)
        assert detected is True
        assert pattern == r"<tool_result\s+name="

    def test_ignores_normal_text(self) -> None:
        """Clean text without fake patterns returns False.

        Contract: provider-protocol:complete:MUST:6
        """
        config = _fake_tool_config
        text = "Hello, how can I help you today? Let me know what you need."
        detected, pattern = contains_fake_tool_calls(text, config)
        assert detected is False
        assert pattern is None

    def test_ignores_empty_text(self) -> None:
        """Empty string returns False.

        Contract: provider-protocol:complete:MUST:6
        """
        config = _fake_tool_config
        detected, pattern = contains_fake_tool_calls("", config)
        assert detected is False
        assert pattern is None

    def test_mention_of_tool_call_not_detected(self) -> None:
        """Normal mention of 'tool call' in conversation not detected.

        Contract: provider-protocol:complete:MUST:6
        """
        config = _fake_tool_config
        text = "When you need to use a tool call, make sure it's structured properly."
        detected, _ = contains_fake_tool_calls(text, config)
        assert detected is False

    def test_case_insensitive_detection(self) -> None:
        """Detection is case insensitive.

        Contract: provider-protocol:complete:MUST:6
        """
        config = _fake_tool_config
        text = '[TOOL CALL: Bash(command="ls")]'
        detected, _ = contains_fake_tool_calls(text, config)
        assert detected is True


class TestShouldRetryForFakeToolCalls:
    """Tests for retry decision logic.

    Contract: provider-protocol:complete:MUST:6
    """

    def test_no_retry_when_real_tool_calls_present(self) -> None:
        """Structured tool calls suppress retry even if fake text present.

        Contract: provider-protocol:complete:MUST:6
        """
        config = _fake_tool_config
        # Fake text in response BUT real tool_calls present
        should_retry, _ = should_retry_for_fake_tool_calls(
            response_text="[Tool Call: bash(command='ls')]",
            tool_calls=[{"name": "bash", "arguments": {"command": "ls"}}],
            tools_available=True,
            config=config,
        )
        assert should_retry is False

    def test_no_retry_when_no_tools_available(self) -> None:
        """Text-only completion skips detection entirely.

        Contract: provider-protocol:complete:MUST:6
        """
        config = _fake_tool_config
        # Fake text BUT no tools in original request
        should_retry, _ = should_retry_for_fake_tool_calls(
            response_text="[Tool Call: bash(command='ls')]",
            tool_calls=None,
            tools_available=False,
            config=config,
        )
        assert should_retry is False

    def test_retry_when_fake_detected_no_real_tools_tools_available(self) -> None:
        """Fake text + no real calls + tools available = retry.

        Contract: provider-protocol:complete:MUST:6
        """
        config = _fake_tool_config
        should_retry, pattern = should_retry_for_fake_tool_calls(
            response_text="[Tool Call: bash(command='ls')]",
            tool_calls=[],  # Empty - no real tool calls
            tools_available=True,
            config=config,
        )
        assert should_retry is True
        assert pattern == r"\[Tool Call:\s*\w+"


class TestFakeToolLogging:
    """Tests for fake tool call logging functions.

    Contract: provider-protocol:complete:MUST:6
    Contract: behaviors:Logging:MUST:4
    """

    def test_log_detection_logs_at_configured_level(self, caplog: pytest.LogCaptureFixture) -> None:
        """log_detection emits log at configured level.

        Contract: behaviors:Logging:MUST:4
        """
        config = FakeToolDetectionConfig(
            patterns=[],
            logging=LoggingConfig(
                log_matched_pattern=True,
                log_response_text=True,
                log_response_text_limit=100,
                log_tool_calls=True,
                level_on_detection="WARNING",
            ),
        )
        with caplog.at_level(logging.WARNING):
            log_detection(config, "test text", "pattern.*", [{"name": "tool1"}])

        assert "[FAKE_TOOL_CALL] Detected" in caplog.text
        assert "pattern.*" in caplog.text
        assert "test text" in caplog.text

    def test_log_detection_truncates_long_text(self, caplog: pytest.LogCaptureFixture) -> None:
        """log_detection truncates text to configured limit.

        Contract: behaviors:Logging:MUST:4
        """
        config = FakeToolDetectionConfig(
            patterns=[],
            logging=LoggingConfig(
                log_response_text=True,
                log_response_text_limit=10,
                level_on_detection="INFO",
            ),
        )
        long_text = "a" * 50
        with caplog.at_level(logging.INFO):
            log_detection(config, long_text, None, [])

        # Should truncate to 10 chars + "..."
        assert "aaaaaaaaaa..." in caplog.text

    def test_log_detection_skips_pattern_when_disabled(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """log_detection skips pattern when log_matched_pattern=False.

        Contract: behaviors:Logging:MUST:4
        """
        config = FakeToolDetectionConfig(
            patterns=[],
            logging=LoggingConfig(
                log_matched_pattern=False,
                log_response_text=False,
                log_tool_calls=False,
                level_on_detection="INFO",
            ),
        )
        with caplog.at_level(logging.INFO):
            log_detection(config, "text", "my_pattern", [{"name": "t"}])

        assert "my_pattern" not in caplog.text

    def test_log_retry_includes_attempt_info(self, caplog: pytest.LogCaptureFixture) -> None:
        """log_retry emits attempt count.

        Contract: provider-protocol:complete:MUST:5
        """
        config = FakeToolDetectionConfig(
            patterns=[],
            logging=LoggingConfig(
                log_correction_message=True,
                level_on_retry="INFO",
            ),
        )
        with caplog.at_level(logging.INFO):
            log_retry(config, attempt=1, max_attempts=3)

        assert "[FAKE_TOOL_CALL] Retrying" in caplog.text
        assert "2/3" in caplog.text  # attempt+1 / max

    def test_log_retry_includes_correction_message(self, caplog: pytest.LogCaptureFixture) -> None:
        """log_retry includes correction message when enabled.

        Contract: provider-protocol:complete:MUST:5
        """
        config = FakeToolDetectionConfig(
            patterns=[],
            correction_message="Please use structured tool calls",
            logging=LoggingConfig(
                log_correction_message=True,
                level_on_retry="DEBUG",
            ),
        )
        with caplog.at_level(logging.DEBUG):
            log_retry(config, attempt=0, max_attempts=2)

        assert "structured tool calls" in caplog.text

    def test_log_exhausted_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """log_exhausted emits warning when attempts exhausted.

        Contract: provider-protocol:complete:MUST:5
        """
        config = FakeToolDetectionConfig(
            patterns=[],
            logging=LoggingConfig(level_on_exhausted="WARNING"),
        )
        with caplog.at_level(logging.WARNING):
            log_exhausted(config, attempts=3)

        assert "[FAKE_TOOL_CALL] Max correction attempts" in caplog.text
        assert "(3)" in caplog.text

    def test_log_success_logs_correction_success(self, caplog: pytest.LogCaptureFixture) -> None:
        """log_success emits success on correction.

        Contract: provider-protocol:complete:MUST:5
        """
        config = FakeToolDetectionConfig(
            patterns=[],
            logging=LoggingConfig(level_on_success="INFO"),
        )
        with caplog.at_level(logging.INFO):
            log_success(config, attempt=2)

        assert "[FAKE_TOOL_CALL] Correction succeeded" in caplog.text
        assert "attempt 3" in caplog.text  # attempt+1

    def test_log_retry_omits_correction_message_when_disabled(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """log_retry with log_correction_message=False omits correction text.

        Line ~264 in fake_tool_detection.py — branch not taken.
        Contract: behaviors:Logging:MUST:4 — logging is controllable via config.
        """
        config = FakeToolDetectionConfig(
            patterns=[],
            correction_message="SECRET_CORRECTION_TEXT",
            logging=LoggingConfig(
                log_correction_message=False,
                level_on_retry="INFO",
            ),
        )
        with caplog.at_level(logging.INFO):
            log_retry(config, attempt=0, max_attempts=2)

        assert "SECRET_CORRECTION_TEXT" not in caplog.text
        assert "[FAKE_TOOL_CALL] Retrying" in caplog.text


class TestTruncateText:
    """Tests for _truncate_text helper.

    Contract: behaviors:Logging:MUST:4
    """

    def test_truncate_text_short_unchanged(self) -> None:
        """Short text unchanged.

        Contract: behaviors:Logging:MUST:4
        """
        result = _truncate_text("hello", 100)
        assert result == "hello"

    def test_truncate_text_at_limit_unchanged(self) -> None:
        """Text at exact limit unchanged.

        Contract: behaviors:Logging:MUST:4
        """
        result = _truncate_text("hello", 5)
        assert result == "hello"

    def test_truncate_text_over_limit_truncated(self) -> None:
        """Long text truncated with ellipsis.

        Contract: behaviors:Logging:MUST:4
        """
        result = _truncate_text("hello world", 5)
        assert result == "hello..."

    def test_truncate_text_zero_limit(self) -> None:
        """Zero limit returns original.

        Contract: behaviors:Logging:MUST:4
        """
        result = _truncate_text("hello", 0)
        assert result == "hello"

    def test_truncate_text_negative_limit(self) -> None:
        """Negative limit returns original.

        Contract: behaviors:Logging:MUST:4
        """
        result = _truncate_text("hello", -5)
        assert result == "hello"


class TestLoggingConfigDefaults:
    """Tests for LoggingConfig default values.

    Contract: behaviors:Logging:MUST:4
    """

    def test_logging_config_secure_defaults(self) -> None:
        """Secure defaults: sensitive logging disabled by default.

        Contract: behaviors:Logging:MUST:4
        """
        config = LoggingConfig()
        assert config.log_response_text is False
        assert config.log_tool_calls is False
        assert config.log_correction_message is False
        assert config.log_matched_pattern is True
        assert config.log_response_text_limit == 500

    def test_logging_config_custom_values(self) -> None:
        """LoggingConfig accepts custom values.

        Contract: behaviors:Logging:MUST:4
        """
        config = LoggingConfig(
            log_matched_pattern=False, log_response_text=True, level_on_exhausted="ERROR"
        )
        assert config.log_matched_pattern is False
        assert config.log_response_text is True
        assert config.level_on_exhausted == "ERROR"
