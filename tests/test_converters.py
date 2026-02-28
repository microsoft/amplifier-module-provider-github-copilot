"""
Tests for message converters.

This module tests the conversion between Amplifier message format
and Copilot SDK format.
"""

import json
from unittest.mock import Mock

from amplifier_core import ChatResponse

from amplifier_module_provider_github_copilot.converters import (
    convert_copilot_response_to_chat_response,
    convert_messages_to_prompt,
    extract_system_message,
    format_tool_result_message,
)


class TestConvertMessagesToPrompt:
    """Tests for convert_messages_to_prompt function."""

    def test_empty_messages(self):
        """Should return empty string for empty messages."""
        result = convert_messages_to_prompt([])
        assert result == ""

    def test_single_user_message(self):
        """Should format single user message correctly."""
        messages = [{"role": "user", "content": "Hello, world!"}]
        result = convert_messages_to_prompt(messages)
        assert result == "Human: Hello, world!"

    def test_single_assistant_message(self):
        """Should format single assistant message correctly."""
        messages = [{"role": "assistant", "content": "Hi there!"}]
        result = convert_messages_to_prompt(messages)
        assert result == "Assistant: Hi there!"

    def test_conversation_flow(self, sample_messages):
        """Should format multi-turn conversation correctly."""
        result = convert_messages_to_prompt(sample_messages)

        # System message should be excluded (handled separately)
        assert "You are a helpful assistant" not in result

        # Check conversation flow
        assert "Human: Hello, how are you?" in result
        assert "Assistant: I'm doing well, thank you!" in result
        assert "Human: Can you help me with Python?" in result

        # Check ordering (Human before Assistant in the output)
        human_pos = result.find("Human: Hello")
        assistant_pos = result.find("Assistant: I'm doing well")
        assert human_pos < assistant_pos

    def test_system_message_excluded(self):
        """System messages should be excluded from prompt."""
        messages = [
            {"role": "system", "content": "System instructions here"},
            {"role": "user", "content": "User message"},
        ]
        result = convert_messages_to_prompt(messages)

        assert "System instructions" not in result
        assert "Human: User message" in result

    def test_tool_result_message(self):
        """Should format tool result messages correctly."""
        messages = [
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "tool_name": "read_file",
                "content": "file contents here",
            }
        ]
        result = convert_messages_to_prompt(messages)
        assert '<tool_result name="read_file">file contents here</tool_result>' in result

    def test_assistant_with_tool_calls(self):
        """Should include tool call information in assistant message."""
        messages = [
            {
                "role": "assistant",
                "content": "Let me check that.",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "name": "read_file",
                        "arguments": {"path": "test.py"},
                    }
                ],
            }
        ]
        result = convert_messages_to_prompt(messages)

        assert "Assistant:" in result
        assert "Let me check that." in result
        assert '<tool_used name="read_file">' in result

    def test_list_content_blocks(self):
        """Should handle OpenAI-style list content blocks."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Part 1"},
                    {"type": "text", "text": "Part 2"},
                ],
            }
        ]
        result = convert_messages_to_prompt(messages)
        assert "Part 1" in result
        assert "Part 2" in result

    def test_image_content_blocks(self):
        """Should handle image content blocks gracefully."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look at this:"},
                    {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
                ],
            }
        ]
        result = convert_messages_to_prompt(messages)
        assert "Look at this:" in result
        assert "[Image]" in result


class TestExtractSystemMessage:
    """Tests for extract_system_message function."""

    def test_no_system_message(self):
        """Should return None when no system message present."""
        messages = [{"role": "user", "content": "Hello"}]
        result = extract_system_message(messages)
        assert result is None

    def test_extract_system_message(self):
        """Should extract system message content."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        result = extract_system_message(messages)
        assert result == "You are helpful."

    def test_multiple_system_messages_joined(self):
        """Should join multiple system messages with double newline."""
        messages = [
            {"role": "system", "content": "First system"},
            {"role": "system", "content": "Second system"},
        ]
        result = extract_system_message(messages)
        assert result == "First system\n\nSecond system"


class TestConvertCopilotResponseToChatResponse:
    """Tests for convert_copilot_response_to_chat_response function."""

    def test_none_response(self):
        """Should handle None response gracefully."""
        result = convert_copilot_response_to_chat_response(None, "test-model")

        assert isinstance(result, ChatResponse)
        assert len(result.content) == 1
        assert result.content[0].text == ""
        assert result.tool_calls == []

    def test_basic_text_response(self):
        """Should convert basic text response."""
        mock_response = Mock()
        mock_response.data = Mock()
        mock_response.data.content = "Hello, I'm Claude!"
        mock_response.data.tool_requests = None
        mock_response.data.input_tokens = 100
        mock_response.data.output_tokens = 50

        result = convert_copilot_response_to_chat_response(mock_response, "claude-opus-4-5")

        assert isinstance(result, ChatResponse)
        assert len(result.content) >= 1

        # Find text content block
        text_blocks = [b for b in result.content if b.type == "text"]
        assert len(text_blocks) >= 1
        assert text_blocks[0].text == "Hello, I'm Claude!"

        assert result.finish_reason == "end_turn"

    def test_response_with_tool_calls(self, mock_tool_response):
        """Should convert response with tool calls."""
        result = convert_copilot_response_to_chat_response(mock_tool_response, "claude-opus-4-5")

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "read_file"
        assert result.tool_calls[0].id == "call_456"
        assert result.tool_calls[0].arguments == {"path": "example.py"}
        assert result.finish_reason == "tool_use"

    def test_dict_response(self):
        """Should handle dict-style response."""
        dict_response = {
            "content": "Response text",
            "tool_requests": None,
            "input_tokens": 50,
            "output_tokens": 25,
        }

        result = convert_copilot_response_to_chat_response(dict_response, "test-model")

        assert len(result.content) >= 1
        text_blocks = [b for b in result.content if b.type == "text"]
        assert text_blocks[0].text == "Response text"


class TestUsageInfo:
    """Tests for usage info extraction."""

    def test_usage_extraction(self):
        """Should extract usage information from response."""
        mock_response = Mock()
        mock_response.data = Mock()
        mock_response.data.content = "Test"
        mock_response.data.tool_requests = None
        mock_response.data.input_tokens = 100
        mock_response.data.output_tokens = 50

        result = convert_copilot_response_to_chat_response(mock_response, "test-model")

        assert result.usage.input_tokens == 100
        assert result.usage.output_tokens == 50
        assert result.usage.total_tokens == 150


class TestFormatToolResultMessage:
    """Tests for format_tool_result_message function."""

    def test_string_result(self):
        """Should format string tool result."""
        result = format_tool_result_message(
            tool_call_id="call_123",
            tool_name="read_file",
            result="file contents",
        )

        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_123"
        assert result["tool_name"] == "read_file"
        assert result["content"] == "file contents"

    def test_dict_result(self):
        """Should serialize dict tool result to JSON."""
        result = format_tool_result_message(
            tool_call_id="call_456",
            tool_name="list_files",
            result={"files": ["a.py", "b.py"]},
        )

        assert result["role"] == "tool"
        content = json.loads(result["content"])
        assert content == {"files": ["a.py", "b.py"]}


class TestEdgeCases:
    """Tests for edge cases in message conversion."""

    def test_tool_call_arguments_as_json_string(self):
        """Should parse JSON string arguments in tool calls."""
        messages = [
            {
                "role": "assistant",
                "content": "I'll search for that.",
                "tool_calls": [
                    {
                        "id": "call_789",
                        "name": "search",
                        "arguments": '{"query": "python", "limit": 10}',  # JSON string
                    }
                ],
            }
        ]
        result = convert_messages_to_prompt(messages)

        assert '<tool_used name="search">' in result
        # Arguments should be parsed and re-serialized
        assert "python" in result

    def test_tool_call_arguments_invalid_json_string(self):
        """Should handle invalid JSON in tool call arguments gracefully."""
        messages = [
            {
                "role": "assistant",
                "content": "Trying something.",
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "name": "custom",
                        "arguments": "not valid json {",  # Invalid JSON
                    }
                ],
            }
        ]
        result = convert_messages_to_prompt(messages)

        # Should not crash, include the tool call somehow
        assert '<tool_used name="custom">' in result

    def test_assistant_with_tool_calls_no_content(self):
        """Should handle assistant message with tool calls but no text."""
        messages = [
            {
                "role": "assistant",
                "content": "",  # Empty content
                "tool_calls": [
                    {
                        "id": "call_empty",
                        "name": "read_file",
                        "arguments": {"path": "file.py"},
                    }
                ],
            }
        ]
        result = convert_messages_to_prompt(messages)

        assert "Assistant:" in result
        assert '<tool_used name="read_file">' in result

    def test_unknown_role_message(self):
        """Should handle unknown message roles gracefully."""
        messages = [
            {"role": "custom_role", "content": "Some content"},
        ]
        result = convert_messages_to_prompt(messages)

        # Should fall back to Human: prefix
        assert "Human: Some content" in result

    def test_unknown_content_block_type(self):
        """Should handle unknown content block types gracefully."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Normal text"},
                    {"type": "unknown_type", "data": "some data"},  # Unknown type
                ],
            }
        ]
        result = convert_messages_to_prompt(messages)

        # Should include the known text at least
        assert "Normal text" in result

    def test_string_content_in_list(self):
        """Should handle plain strings in content list."""
        messages = [
            {
                "role": "user",
                "content": ["String one", "String two"],  # Plain strings
            }
        ]
        result = convert_messages_to_prompt(messages)

        assert "String one" in result
        assert "String two" in result

    def test_tool_name_fallback(self):
        """Should use fallback for tool name in tool result."""
        messages = [
            {
                "role": "tool",
                "tool_call_id": "call_xyz",
                # No tool_name, should fallback
                "content": "result content",
            }
        ]
        result = convert_messages_to_prompt(messages)

        assert "<tool_result" in result
        assert "result content" in result

    def test_tool_call_with_function_format(self):
        """Should handle OpenAI function format for tool calls."""
        messages = [
            {
                "role": "assistant",
                "content": "Calling function.",
                "tool_calls": [
                    {
                        "id": "call_func",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "Seattle"},
                        },
                    }
                ],
            }
        ]
        result = convert_messages_to_prompt(messages)

        assert '<tool_used name="get_weather">' in result
        assert "Seattle" in result


class TestToolRequestConversion:
    """Tests for _convert_tool_request function."""

    def test_dict_tool_request(self):
        """Should convert dict-style tool request."""
        from amplifier_module_provider_github_copilot.converters import _convert_tool_request

        tr = {
            "toolCallId": "call_dict_123",
            "name": "search",
            "arguments": {"query": "test"},
        }
        result = _convert_tool_request(tr)

        assert result is not None
        assert result.id == "call_dict_123"
        assert result.name == "search"
        assert result.arguments == {"query": "test"}

    def test_dict_tool_request_alternate_keys(self):
        """Should handle alternate key names in dict request."""
        from amplifier_module_provider_github_copilot.converters import _convert_tool_request

        tr = {
            "tool_call_id": "call_alt_456",  # snake_case
            "name": "read",
            "arguments": "{}",  # String
        }
        result = _convert_tool_request(tr)

        assert result is not None
        assert result.id == "call_alt_456"

    def test_unknown_tool_request_format(self):
        """Should return None for unknown tool request format."""
        from amplifier_module_provider_github_copilot.converters import _convert_tool_request

        # Pass something that's not a dict or has tool_call_id
        result = _convert_tool_request("just a string")

        assert result is None

    def test_tool_request_string_arguments_invalid(self):
        """Should wrap invalid JSON arguments in raw key."""
        from amplifier_module_provider_github_copilot.converters import _convert_tool_request

        tr = {
            "tool_call_id": "call_bad",
            "name": "test",
            "arguments": "not valid { json",
        }
        result = _convert_tool_request(tr)

        assert result is not None
        assert result.arguments == {"raw": "not valid { json"}


class TestDeveloperRoleHandling:
    """Tests for developer role message handling.

    Developer role is defined in amplifier-core as a valid message role.
    Both Anthropic and OpenAI providers wrap developer messages in XML tags.
    See: amplifier-module-provider-anthropic line 689, openai line 1392
    """

    def test_developer_message_xml_wrapped(self):
        """Developer messages should be XML-wrapped as user messages."""
        messages = [{"role": "developer", "content": "This is file context from the IDE."}]
        result = convert_messages_to_prompt(messages)

        # Should be wrapped in context_file XML tags per Anthropic/OpenAI pattern
        assert "<context_file>" in result
        assert "</context_file>" in result
        assert "This is file context from the IDE." in result
        # Should be presented as Human (user) message
        assert "Human:" in result

    def test_developer_message_multiline_content(self):
        """Developer messages should preserve multiline content in XML wrapper."""
        multiline_content = """def hello():
    print("world")
    return True"""
        messages = [{"role": "developer", "content": multiline_content}]
        result = convert_messages_to_prompt(messages)

        assert "<context_file>\n" in result
        assert "\n</context_file>" in result
        assert 'print("world")' in result

    def test_developer_message_empty_content_skipped(self):
        """Empty developer messages should be skipped."""
        messages = [{"role": "developer", "content": ""}, {"role": "user", "content": "Hello"}]
        result = convert_messages_to_prompt(messages)

        # Empty developer message should not add context_file tags
        assert "<context_file>" not in result
        # User message should still appear
        assert "Human: Hello" in result

    def test_developer_messages_before_conversation(self):
        """Multiple developer messages should all be included with XML wrapping."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "developer", "content": "File 1 content"},
            {"role": "developer", "content": "File 2 content"},
            {"role": "user", "content": "What's in these files?"},
        ]
        result = convert_messages_to_prompt(messages)

        # System should be excluded (handled separately)
        assert "You are helpful" not in result
        # Both developer messages should have XML wrapping
        assert result.count("<context_file>") == 2
        assert result.count("</context_file>") == 2
        assert "File 1 content" in result
        assert "File 2 content" in result
        assert "Human: What's in these files?" in result

    def test_developer_message_list_content(self):
        """Developer messages with list content should be extracted and wrapped."""
        messages = [
            {
                "role": "developer",
                "content": [{"type": "text", "text": "Part 1"}, {"type": "text", "text": "Part 2"}],
            }
        ]
        result = convert_messages_to_prompt(messages)

        assert "<context_file>" in result
        assert "Part 1" in result
        assert "Part 2" in result


class TestFunctionRoleHandling:
    """Tests for function role message handling.

    Function role is defined in amplifier-core as a valid message role
    (legacy OpenAI format, deprecated in favor of 'tool').
    """

    def test_function_message_formatted_as_tool_result(self):
        """Function messages should be handled like tool results (backward compat)."""
        messages = [{"role": "function", "name": "get_weather", "content": "72 degrees, sunny"}]
        result = convert_messages_to_prompt(messages)

        # Should be formatted as tool result for consistency
        assert '<tool_result name="get_weather">' in result
        assert "72 degrees, sunny" in result

    def test_function_message_no_name_fallback(self):
        """Function messages without name should use fallback."""
        messages = [{"role": "function", "content": "result content"}]
        result = convert_messages_to_prompt(messages)

        # Should use 'function' as fallback name
        assert '<tool_result name="function">' in result
        assert "result content" in result

    def test_mixed_tool_and_function_roles(self):
        """Both tool and function roles should be handled consistently."""
        messages = [
            {"role": "tool", "name": "read_file", "content": "file A contents"},
            {"role": "function", "name": "write_file", "content": "file B written"},
        ]
        result = convert_messages_to_prompt(messages)

        # Both should appear as Tool Results
        assert '<tool_result name="read_file">file A contents</tool_result>' in result
        assert '<tool_result name="write_file">file B written</tool_result>' in result


class TestAllSixRolesIntegration:
    """Integration test for all 6 amplifier-core message roles.

    Roles from amplifier-core: system, developer, user, assistant, function, tool
    """

    def test_all_six_roles_in_conversation(self):
        """All 6 message roles should be handled without errors."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "developer", "content": "Context: user is working on Python."},
            {"role": "user", "content": "What time is it?"},
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [{"id": "call_1", "name": "get_time", "arguments": {}}],
            },
            {"role": "tool", "name": "get_time", "content": "10:30 AM"},
            {"role": "function", "name": "legacy_func", "content": "legacy result"},
        ]
        result = convert_messages_to_prompt(messages)

        # System excluded (handled separately)
        assert "You are a helpful assistant" not in result

        # Developer wrapped in XML
        assert "<context_file>" in result
        assert "Context: user is working on Python" in result

        # User as Human
        assert "Human: What time is it?" in result

        # Assistant with tool call
        assert "Assistant:" in result
        assert '<tool_used name="get_time">' in result

        # Tool result
        assert '<tool_result name="get_time">10:30 AM</tool_result>' in result

        # Function as legacy tool result
        assert '<tool_result name="legacy_func">legacy result</tool_result>' in result
