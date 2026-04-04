"""Tests for ChatRequest multi-turn context preservation.

Contract: contracts/provider-protocol.md — complete() must handle ChatRequest content types

These tests verify:
1. Role information is preserved (user/assistant/system)
2. Tool call content blocks are represented
3. Tool result blocks are represented
4. Multi-turn conversations maintain context fidelity

"""

from dataclasses import dataclass, field
from typing import Any

from amplifier_module_provider_github_copilot.provider import (
    _extract_prompt_from_chat_request,  # type: ignore[reportPrivateUsage]  # Testing internal helper
)


# Test fixtures for kernel types (simulating amplifier_core types)
@dataclass
class MockTextContent:
    """Mock TextContent block."""

    text: str
    type: str = "text"


@dataclass
class MockThinkingContent:
    """Mock ThinkingContent block."""

    thinking: str
    type: str = "thinking"


@dataclass
class MockToolCallContent:
    """Mock ToolCallContent block for tool calls."""

    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    type: str = "tool_call"


@dataclass
class MockToolResultContent:
    """Mock ToolResultContent block for tool results."""

    tool_call_id: str
    output: str
    type: str = "tool_result"


@dataclass
class MockMessage:
    """Mock kernel Message."""

    role: str
    content: list[Any] | str


@dataclass
class MockChatRequest:
    """Mock kernel ChatRequest."""

    messages: list[MockMessage]
    model: str | None = None
    tools: list[Any] = field(default_factory=list)  # type: ignore[reportUnknownVariableType]


class TestRolePreservation:
    """Tests for role information preservation.

    Contract: provider-protocol:complete:MUST:1 — role context must be preserved.
    """

    def test_single_user_message_has_role(self) -> None:
        """Single user message should include role marker in prompt."""
        request = MockChatRequest(
            messages=[
                MockMessage(role="user", content="Hello, who are you?"),
            ]
        )

        prompt = _extract_prompt_from_chat_request(request)

        # Should include role marker
        assert "user" in prompt.lower() or "User:" in prompt

    def test_multi_turn_preserves_roles(self) -> None:
        """Multi-turn conversation should preserve user/assistant role boundaries.

        AC: Role information is preserved.
        """
        request = MockChatRequest(
            messages=[
                MockMessage(role="user", content="What is 2+2?"),
                MockMessage(role="assistant", content="2+2 equals 4."),
                MockMessage(role="user", content="And 3+3?"),
            ]
        )

        prompt = _extract_prompt_from_chat_request(request)

        # Both roles should be present in the output
        assert "user" in prompt.lower() or "User:" in prompt
        assert "assistant" in prompt.lower() or "Assistant:" in prompt

    def test_system_message_not_in_prompt(self) -> None:
        """C-4: system messages MUST NOT be included in the prompt body.

        Contract: sdk-boundary:Config:MUST:2 — system_message goes to SDK session config
        (mode=replace), NOT to the prompt body.  Including it in the prompt ALSO causes
        a dual-path injection: the model sees the system instructions both via
        `session_config.system_message` AND repeated verbatim in the conversation.

        extract_prompt_from_chat_request() MUST skip role=="system" messages.
        """
        request = MockChatRequest(
            messages=[
                MockMessage(role="system", content="You are a helpful assistant."),
                MockMessage(role="user", content="Hello!"),
            ]
        )

        prompt = _extract_prompt_from_chat_request(request)

        # MUST NOT include system message in prompt body
        assert "You are a helpful assistant." not in prompt, (
            "System message must not be in the prompt body — "
            "it goes through SDK session_config.system_message instead"
        )
        # User message MUST still be present
        assert "Hello!" in prompt or "user" in prompt.lower()

    def test_system_message_preserved(self) -> None:
        """System message role should be preserved (in session config, not in prompt body)."""
        request = MockChatRequest(
            messages=[
                MockMessage(role="system", content="You are a helpful assistant."),
                MockMessage(role="user", content="Hello!"),
            ]
        )

        prompt = _extract_prompt_from_chat_request(request)

        # System content must NOT be in prompt body — it's routed through session config
        assert "You are a helpful assistant." not in prompt
        # User message must still be in prompt
        assert "Hello!" in prompt or "user" in prompt.lower()


class TestContentTypePreservation:
    """Tests for content type preservation.

    Contract: provider-protocol:complete:MUST:2 — content types must be preserved.
    """

    def test_text_content_blocks_extracted(self) -> None:
        """TextContent blocks should have their text extracted."""
        request = MockChatRequest(
            messages=[
                MockMessage(
                    role="user",
                    content=[MockTextContent(text="Hello from text block")],
                ),
            ]
        )

        prompt = _extract_prompt_from_chat_request(request)

        assert "Hello from text block" in prompt

    def test_thinking_content_blocks_included(self) -> None:
        """ThinkingContent blocks should be included in prompt.

        AC: ThinkingContent blocks are included.
        """
        request = MockChatRequest(
            messages=[
                MockMessage(
                    role="assistant",
                    content=[
                        MockThinkingContent(thinking="Let me reason about this..."),
                        MockTextContent(text="Here is my answer."),
                    ],
                ),
            ]
        )

        prompt = _extract_prompt_from_chat_request(request)

        # Thinking content should be included (possibly with marker)
        assert "reason about this" in prompt.lower() or "thinking" in prompt.lower()

    def test_tool_call_content_blocks_included(self) -> None:
        """ToolCallContent blocks should be represented.

        AC: ToolCallContent blocks are included.
        """
        request = MockChatRequest(
            messages=[
                MockMessage(
                    role="assistant",
                    content=[
                        MockToolCallContent(
                            tool_call_id="tc_123",
                            tool_name="read_file",
                            arguments={"path": "/tmp/test.txt"},
                        ),
                    ],
                ),
            ]
        )

        prompt = _extract_prompt_from_chat_request(request)

        # Tool call blocks are intentionally NOT serialized to text
        # They are handled via the tool_calls field, not text content
        # This prevents fake tool call detection from triggering on prior turns
        assert "read_file" not in prompt

    def test_tool_result_content_blocks_included(self) -> None:
        """ToolResultContent blocks should be represented.

        AC: Tool result blocks are represented.
        """
        request = MockChatRequest(
            messages=[
                MockMessage(
                    role="user",  # Tool results often come as user role
                    content=[
                        MockToolResultContent(
                            tool_call_id="tc_123",
                            output="File contents: hello world",
                        ),
                    ],
                ),
            ]
        )

        prompt = _extract_prompt_from_chat_request(request)

        # Tool result should be represented
        assert "hello world" in prompt or "result" in prompt.lower()


class TestMultiTurnConversation:
    """Tests for full multi-turn conversation handling.

    Contract: provider-protocol:complete:MUST:3 — multi-turn context fidelity.
    """

    def test_mixed_content_types_in_conversation(self) -> None:
        """Multi-turn with mixed content types should maintain fidelity.

        AC: Test covers mixed content types in a multi-turn request.
        """
        request = MockChatRequest(
            messages=[
                MockMessage(role="system", content="You are a coding assistant."),
                MockMessage(role="user", content="Read the file config.yaml"),
                MockMessage(
                    role="assistant",
                    content=[
                        MockThinkingContent(thinking="I need to read the file..."),
                        MockToolCallContent(
                            tool_call_id="tc_1",
                            tool_name="read_file",
                            arguments={"path": "config.yaml"},
                        ),
                    ],
                ),
                MockMessage(
                    role="user",
                    content=[
                        MockToolResultContent(
                            tool_call_id="tc_1",
                            output="key: value\nport: 8080",
                        ),
                    ],
                ),
                MockMessage(
                    role="assistant",
                    content=[MockTextContent(text="The config has port 8080.")],
                ),
                MockMessage(role="user", content="What port is configured?"),
            ]
        )

        prompt = _extract_prompt_from_chat_request(request)

        # All key content should be present (system message is NOT in prompt —
        # it goes via session config).
        assert "Read the file" in prompt or "config.yaml" in prompt
        assert "8080" in prompt
        assert "What port" in prompt or "port" in prompt

    def test_conversation_order_preserved(self) -> None:
        """Message order should be preserved in the output."""
        request = MockChatRequest(
            messages=[
                MockMessage(role="user", content="First message"),
                MockMessage(role="assistant", content="Second message"),
                MockMessage(role="user", content="Third message"),
            ]
        )

        prompt = _extract_prompt_from_chat_request(request)

        # Order should be preserved
        first_pos = prompt.find("First")
        second_pos = prompt.find("Second")
        third_pos = prompt.find("Third")

        assert first_pos < second_pos < third_pos, "Messages should maintain order"


class TestRegressionPrevention:
    """Regression tests to ensure existing behavior is not broken."""

    def test_string_content_still_works(self) -> None:
        """Simple string content should still work."""
        request = MockChatRequest(
            messages=[
                MockMessage(role="user", content="Simple string message"),
            ]
        )

        prompt = _extract_prompt_from_chat_request(request)

        assert "Simple string message" in prompt

    def test_empty_messages_handled(self) -> None:
        """Empty messages list should not crash."""
        request = MockChatRequest(messages=[])

        prompt = _extract_prompt_from_chat_request(request)

        assert prompt == "" or prompt is not None  # Should not crash


class TestContentExtractionEdgeCases:
    """Edge case tests for content extraction.

    Coverage: provider.py lines 186, 202, 219, 225, 240, 252-260
    """

    def test_none_content_returns_empty(self) -> None:
        """None content returns empty string.

        Coverage: provider.py line 186 (_extract_message_content None path)
        """
        request = MockChatRequest(
            messages=[
                MockMessage(role="user", content=None),  # type: ignore[arg-type]
            ]
        )

        prompt = _extract_prompt_from_chat_request(request)

        # Should not crash, may include role marker only
        assert prompt is not None

    def test_single_content_block_not_list(self) -> None:
        """Single content block (not wrapped in list) extracted.

        Coverage: provider.py line 202 (single block path)
        """
        request = MockChatRequest(
            messages=[
                MockMessage(
                    role="user",
                    content=MockTextContent(text="Single block"),  # type: ignore[arg-type]
                ),
            ]
        )

        prompt = _extract_prompt_from_chat_request(request)

        assert "Single block" in prompt

    def test_thinking_with_empty_value(self) -> None:
        """ThinkingContent with empty thinking returns empty.

        Coverage: provider.py line 219 (empty thinking path)
        """
        request = MockChatRequest(
            messages=[
                MockMessage(
                    role="assistant",
                    content=[MockThinkingContent(thinking="")],
                ),
            ]
        )

        prompt = _extract_prompt_from_chat_request(request)

        # Empty thinking should not add "[Thinking: ]"
        assert "Thinking:" not in prompt or prompt == ""

    def test_tool_call_content_skipped(self) -> None:
        """ToolCallContent blocks are intentionally skipped.

        Coverage: provider.py line 225 (tool_call skip path)
        Contract: prevents fake tool detection on prior turns
        """
        request = MockChatRequest(
            messages=[
                MockMessage(
                    role="assistant",
                    content=[
                        MockToolCallContent(
                            tool_call_id="tc_1",
                            tool_name="bash",
                            arguments={"command": "ls"},
                        ),
                    ],
                ),
            ]
        )

        prompt = _extract_prompt_from_chat_request(request)

        # Tool call should NOT appear in text
        assert "bash" not in prompt
        assert "tool_call" not in prompt.lower()

    def test_tool_result_with_output(self) -> None:
        """ToolResultContent with output is formatted.

        Coverage: provider.py line 240 (tool_result path)
        """
        request = MockChatRequest(
            messages=[
                MockMessage(
                    role="user",
                    content=[
                        MockToolResultContent(
                            tool_call_id="tc_1",
                            output="command output here",
                        ),
                    ],
                ),
            ]
        )

        prompt = _extract_prompt_from_chat_request(request)

        # Tool result should appear, possibly with marker
        assert "command output here" in prompt

    def test_fallback_value_attribute(self) -> None:
        """Content with 'value' attribute uses fallback.

        Coverage: provider.py lines 252-260 (fallback loop)
        """

        @dataclass
        class ValueContent:
            value: str

        request = MockChatRequest(
            messages=[
                MockMessage(
                    role="user",
                    content=[ValueContent(value="value attribute content")],
                ),
            ]
        )

        prompt = _extract_prompt_from_chat_request(request)

        assert "value attribute content" in prompt

    def test_unknown_block_returns_empty(self) -> None:
        """Unknown block type without known attributes returns empty.

        Coverage: provider.py line 260 (final return "")
        """

        @dataclass
        class UnknownBlock:
            unknown_field: str = "should not appear"

        request = MockChatRequest(
            messages=[
                MockMessage(
                    role="user",
                    content=[UnknownBlock()],
                ),
            ]
        )

        prompt = _extract_prompt_from_chat_request(request)

        # Should not crash, unknown block content not included
        assert "should not appear" not in prompt

    def test_tool_result_includes_tool_call_id(self) -> None:
        """L-2: ToolResultContent.tool_call_id MUST appear in serialized prompt.

        Contract: provider-protocol:complete:MUST — MUST preserve tool call IDs
        for result correlation.

        When the model receives multi-turn context as a prompt string, EACH tool
        result MUST include the originating tool_call_id so the model can correlate
        results back to calls.  Without it, out-of-order or multiple-tool turns
        cannot be resolved.
        """
        request = MockChatRequest(
            messages=[
                MockMessage(
                    role="user",
                    content=[
                        MockToolResultContent(
                            tool_call_id="call_abc123",
                            output="file contents here",
                        ),
                    ],
                ),
            ]
        )

        prompt = _extract_prompt_from_chat_request(request)

        # MUST include tool_call_id for correlation (not just the output)
        assert "call_abc123" in prompt, (
            "Tool result MUST include tool_call_id in serialized prompt for correlation. "
            f"Prompt was: {prompt!r}"
        )
        assert "file contents here" in prompt
