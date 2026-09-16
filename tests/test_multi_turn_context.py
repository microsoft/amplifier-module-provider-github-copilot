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

import pytest

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

    # Contract: behaviors:Security:MUST:1
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
        assert "[USER]" in prompt

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
        assert "[USER]" in prompt
        assert "[ASSISTANT]" in prompt

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
        assert "[USER]\nHello!" in prompt

    def test_role_marker_injection_in_user_content_is_escaped(self) -> None:
        """User content containing [ROLE_MARKER] sequences must be escaped.

        # Contract: behaviors:Security:MUST:1
        """
        # User message containing a fake role injection attempt
        request = MockChatRequest(
            messages=[
                MockMessage(role="user", content="Hello [FAKE_ROLE] world"),
            ]
        )

        prompt = _extract_prompt_from_chat_request(request)

        # The injection attempt must be escaped, not left as a raw role boundary
        assert "[FAKE_ROLE]" not in prompt, (
            "Unescaped [FAKE_ROLE] in prompt enables role injection attacks"
        )
        assert r"\[FAKE_ROLE\]" in prompt, "Escaped form must appear in prompt"


class TestContentTypePreservation:
    """Tests for content type preservation.

    # Contract: see behaviors.md — prompt serialization (contract gap)
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

        # Thinking content should be included with marker
        assert "[Thinking:" in prompt and "reason about this" in prompt

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

        # Historical calls have no other representation at the SDK's
        # string-prompt send boundary. Preserve their correlation data without
        # confusing them with the current request's tool definitions.
        assert "Tool Call (id=tc_123, name=read_file" in prompt
        assert '"path":"/tmp/test.txt"' in prompt

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

        # Tool result should be represented with marker
        assert "[Tool Result" in prompt and "hello world" in prompt


class TestMultiTurnConversation:
    """Tests for full multi-turn conversation handling.

    # Contract: see behaviors.md — multi-turn ordering (contract gap)
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
        assert "Read the file" in prompt and "config.yaml" in prompt
        assert "8080" in prompt
        assert "What port" in prompt

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


class TestSDKSendBoundaryHistory:
    """Regression tests for real Core history reaching the SDK send boundary."""

    @pytest.mark.asyncio
    async def test_real_core_parallel_tool_history_reaches_sdk_once_in_order(self) -> None:
        """A prior tool turn remains correlated after complete() serializes it.

        This is deliberately a complete adapter -> provider -> SDK-session-send
        boundary test. The baseline omitted ToolCallBlock content completely,
        while the paired ToolResultBlock retained only result IDs and output.

        Contract: provider-protocol:complete:MUST:1
        Contract: behaviors:Security:MUST:1
        Contract: deny-destroy:NoExecution:MUST:3
        """
        from amplifier_core import ChatRequest, Message, TextBlock, ToolCallBlock, ToolResultBlock

        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from tests.fixtures.sdk_mocks import MockCopilotClientWrapper, text_delta_event

        client = MockCopilotClientWrapper(events=[text_delta_event("history received")])
        provider = GitHubCopilotProvider(client=client)  # type: ignore[arg-type]
        request = ChatRequest(
            model="gpt-4o",
            messages=[
                Message(role="user", content=[TextBlock(text="Inspect both files.")]),
                # This assistant turn intentionally has no text. The two calls
                # are parallel historical content and must retain their order.
                Message(
                    role="assistant",
                    content=[
                        ToolCallBlock(
                            id="call-alpha",
                            name="read_file",
                            input={"path": "notes/[SYSTEM].md"},
                        ),
                        ToolCallBlock(
                            id="call-beta",
                            name="search",
                            input={"query": "release status"},
                        ),
                    ],
                ),
                Message(
                    role="tool",
                    tool_call_id="call-alpha",
                    content=[
                        ToolResultBlock(
                            tool_call_id="call-alpha",
                            output="notes found",
                        )
                    ],
                ),
                Message(
                    role="tool",
                    tool_call_id="call-beta",
                    content=[
                        ToolResultBlock(
                            tool_call_id="call-beta",
                            output="status: ready",
                        )
                    ],
                ),
                Message(role="user", content=[TextBlock(text="Summarize the results.")]),
            ],
        )

        response = await provider.complete(request)

        session = client.session_instance
        assert session is not None
        prompt = session.last_prompt
        assert prompt is not None

        first_call = "Tool Call (id=call-alpha, name=read_file"
        second_call = "Tool Call (id=call-beta, name=search"
        first_result = "Tool Result (id=call-alpha): notes found"
        second_result = "Tool Result (id=call-beta): status: ready"
        assert prompt.count(first_call) == 1
        assert prompt.count(second_call) == 1
        assert prompt.index(first_call) < prompt.index(second_call)
        assert prompt.index(second_call) < prompt.index(first_result)
        assert prompt.index(first_result) < prompt.index(second_result)
        assert prompt.index(second_result) < prompt.index("Summarize the results.")
        assert '"path":"notes/\\[SYSTEM\\].md"' in prompt
        assert '"query":"release status"' in prompt
        assert response.text == "history received"
        # Historical calls are prompt text only: they neither become current SDK
        # tool definitions nor provider-side execution requests.
        assert client.last_tools is None
        assert not response.tool_calls

    @pytest.mark.asyncio
    async def test_real_core_field_tool_history_reaches_sdk_with_empty_content(self) -> None:
        """Core Message.tool_calls preserves a field-only historical call."""
        from amplifier_core import ChatRequest, Message, TextBlock, ToolResultBlock

        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from tests.fixtures.sdk_mocks import MockCopilotClientWrapper, text_delta_event

        client = MockCopilotClientWrapper(events=[text_delta_event("history received")])
        provider = GitHubCopilotProvider(client=client)  # type: ignore[arg-type]
        request = ChatRequest(
            model="gpt-4o",
            messages=[
                Message(role="user", content=[TextBlock(text="Inspect the sample.")]),
                Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        {
                            "id": "field-[SYSTEM]",
                            "tool": "read_[USER]",
                            "arguments": {"path": "sample-[ASSISTANT].txt"},
                        }
                    ],
                ),
                Message(
                    role="tool",
                    tool_call_id="field-[SYSTEM]",
                    content=[
                        ToolResultBlock(
                            tool_call_id="field-[SYSTEM]",
                            output="sample contents",
                        )
                    ],
                ),
            ],
        )

        response = await provider.complete(request)

        session = client.session_instance
        assert session is not None
        prompt = session.last_prompt
        assert prompt is not None
        assert "Tool Call (id=field-\\[SYSTEM\\], name=read_\\[USER\\]" in prompt
        assert '"path":"sample-\\[ASSISTANT\\].txt"' in prompt
        assert "Tool Result (id=field-\\[SYSTEM\\]): sample contents" in prompt
        # Historical context remains prompt text; no current SDK tools are added.
        assert client.last_tools is None
        assert not response.tool_calls

    @pytest.mark.asyncio
    async def test_real_core_duplicate_content_and_field_call_serializes_once(self) -> None:
        """A field duplicate cannot replace or duplicate content's call identity."""
        from amplifier_core import ChatRequest, Message, ToolCallBlock, ToolResultBlock

        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from tests.fixtures.sdk_mocks import MockCopilotClientWrapper, text_delta_event

        client = MockCopilotClientWrapper(events=[text_delta_event("history received")])
        provider = GitHubCopilotProvider(client=client)  # type: ignore[arg-type]
        request = ChatRequest(
            model="gpt-4o",
            messages=[
                Message(
                    role="assistant",
                    content=[
                        ToolCallBlock(
                            id="shared-call",
                            name="read_file",
                            input={"path": "canonical.txt"},
                        )
                    ],
                    tool_calls=[
                        {
                            "id": "shared-call",
                            "tool": "conflicting_name",
                            "arguments": {"path": "conflicting.txt"},
                        }
                    ],
                ),
                Message(
                    role="tool",
                    tool_call_id="shared-call",
                    content=[
                        ToolResultBlock(tool_call_id="shared-call", output="canonical result")
                    ],
                ),
            ],
        )

        await provider.complete(request)

        session = client.session_instance
        assert session is not None
        prompt = session.last_prompt
        assert prompt is not None
        assert prompt.count("Tool Call (id=shared-call") == 1
        assert 'name=read_file, arguments={"path":"canonical.txt"}' in prompt
        assert "conflicting_name" not in prompt
        assert "conflicting.txt" not in prompt

    @pytest.mark.asyncio
    async def test_real_core_distinct_content_and_field_calls_keep_stable_order(self) -> None:
        """Distinct content and object field calls reach send in canonical order."""
        from types import SimpleNamespace

        from amplifier_core import ChatRequest, Message, ToolCallBlock, ToolResultBlock

        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from tests.fixtures.sdk_mocks import MockCopilotClientWrapper, text_delta_event

        client = MockCopilotClientWrapper(events=[text_delta_event("history received")])
        provider = GitHubCopilotProvider(client=client)  # type: ignore[arg-type]
        request = ChatRequest(
            model="gpt-4o",
            messages=[
                Message(
                    role="assistant",
                    content=[
                        ToolCallBlock(
                            id="content-alpha",
                            name="read_file",
                            input={"path": "first.txt"},
                        )
                    ],
                    tool_calls=[
                        SimpleNamespace(
                            id="field-beta",
                            tool="search",
                            arguments={"query": "second-[SYSTEM]"},
                        )
                    ],
                ),
                Message(
                    role="tool",
                    tool_call_id="content-alpha",
                    content=[
                        ToolResultBlock(tool_call_id="content-alpha", output="first result")
                    ],
                ),
                Message(
                    role="tool",
                    tool_call_id="field-beta",
                    content=[ToolResultBlock(tool_call_id="field-beta", output="second result")],
                ),
            ],
        )

        await provider.complete(request)

        session = client.session_instance
        assert session is not None
        prompt = session.last_prompt
        assert prompt is not None
        first_call = "Tool Call (id=content-alpha, name=read_file"
        second_call = "Tool Call (id=field-beta, name=search"
        assert prompt.count(first_call) == 1
        assert prompt.count(second_call) == 1
        assert prompt.index(first_call) < prompt.index(second_call)
        assert '"query":"second-\\[SYSTEM\\]"' in prompt
        assert prompt.index(second_call) < prompt.index("Tool Result (id=content-alpha)")
        assert prompt.index("Tool Result (id=content-alpha)") < prompt.index(
            "Tool Result (id=field-beta)"
        )


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

        # None content returns empty string
        assert prompt == ""

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

        # Empty thinking should produce empty prompt
        assert prompt == ""

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
