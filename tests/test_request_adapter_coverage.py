"""Coverage tests for request_adapter.py missing branches.

Covers:
- Line ~176: ThinkingContent with empty/None thinking → return ""
- Lines ~221-222: ToolResultContent without tool_call_id / with no output
- Line ~273: Multiple system messages joined
- Line ~346: Nested OpenAI-style tool dict {"function": {"name": "..."}}

Contract: provider-protocol:complete:MUST:1
Contract: observability:Payload:SHOULD:2 — Type-safe tool name extraction
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# _extract_content_block: thinking block with empty thinking (line ~176)
# ---------------------------------------------------------------------------


class TestExtractContentBlockThinkingEmpty:
    """ThinkingContent with empty thinking text returns empty string."""

    def test_thinking_block_with_none_thinking_returns_empty(self) -> None:
        """ThinkingContent with thinking=None returns ''.

        Line ~176 in request_adapter.py — the return "" after if thinking: check
        Contract: provider-protocol:complete:MUST:1 — content extraction
        """
        from amplifier_module_provider_github_copilot.request_adapter import (
            _extract_content_block,  # pyright: ignore[reportPrivateUsage]
        )

        @dataclass
        class ThinkingBlock:
            type: str = "thinking"
            thinking: str | None = None

        result = _extract_content_block(ThinkingBlock(thinking=None))
        assert result == ""

    def test_thinking_block_with_empty_string_thinking_returns_empty(self) -> None:
        """ThinkingContent with thinking='' returns ''.

        Line ~176 in request_adapter.py
        """
        from amplifier_module_provider_github_copilot.request_adapter import (
            _extract_content_block,  # pyright: ignore[reportPrivateUsage]
        )

        @dataclass
        class ThinkingBlock:
            type: str = "thinking"
            thinking: str = ""

        result = _extract_content_block(ThinkingBlock(thinking=""))
        assert result == ""

    def test_thinking_block_with_text_returns_formatted(self) -> None:
        """ThinkingContent with valid text returns '[Thinking: ...]'."""
        from amplifier_module_provider_github_copilot.request_adapter import (
            _extract_content_block,  # pyright: ignore[reportPrivateUsage]
        )

        @dataclass
        class ThinkingBlock:
            type: str = "thinking"
            thinking: str = "step 1: analyze the problem"

        result = _extract_content_block(ThinkingBlock())
        assert result == "[Thinking: step 1: analyze the problem]"

    def test_thinking_type_none_with_hasattr_fallback(self) -> None:
        """Block with no type but thinking attr uses hasattr fallback path."""
        from amplifier_module_provider_github_copilot.request_adapter import (
            _extract_content_block,  # pyright: ignore[reportPrivateUsage]
        )

        @dataclass
        class UnknownBlockWithThinking:
            thinking: str = ""  # no 'type' attr, has 'thinking'

        result = _extract_content_block(UnknownBlockWithThinking(thinking=""))
        assert result == ""


# ---------------------------------------------------------------------------
# _extract_content_block: tool result without tool_call_id (lines ~221-222)
# ---------------------------------------------------------------------------


class TestExtractContentBlockToolResult:
    """ToolResultContent handles missing tool_call_id and empty output."""

    def test_tool_result_with_output_no_id_returns_simple_format(self) -> None:
        """ToolResult with output but no tool_call_id returns '[Tool Result: ...]'.

        Line ~221 in request_adapter.py — return f"[Tool Result: {output}]"
        Contract: provider-protocol:complete:MUST — preserve tool results
        """
        from amplifier_module_provider_github_copilot.request_adapter import (
            _extract_content_block,  # pyright: ignore[reportPrivateUsage]
        )

        @dataclass
        class ToolResultNoId:
            type: str = "tool_result"
            output: str = "{'result': 42}"
            tool_call_id: str | None = None

        result = _extract_content_block(ToolResultNoId())
        assert result == "[Tool Result: {'result': 42}]"
        assert "id=" not in result  # no ID in the output

    def test_tool_result_with_no_output_returns_empty(self) -> None:
        """ToolResult with output=None returns ''.

        Line ~222 in request_adapter.py — return ""
        """
        from amplifier_module_provider_github_copilot.request_adapter import (
            _extract_content_block,  # pyright: ignore[reportPrivateUsage]
        )

        @dataclass
        class ToolResultNoOutput:
            type: str = "tool_result"
            output: str | None = None
            tool_call_id: str = "call_abc"

        result = _extract_content_block(ToolResultNoOutput())
        assert result == ""

    def test_tool_result_with_output_and_id_returns_id_format(self) -> None:
        """ToolResult with output + tool_call_id returns '[Tool Result (id=...): ...]'.

        This is the existing covered path — verify it still works.
        """
        from amplifier_module_provider_github_copilot.request_adapter import (
            _extract_content_block,  # pyright: ignore[reportPrivateUsage]
        )

        @dataclass
        class ToolResultFull:
            type: str = "tool_result"
            output: str = "success"
            tool_call_id: str = "call_xyz"

        result = _extract_content_block(ToolResultFull())
        assert "id=call_xyz" in result
        assert "success" in result


# ---------------------------------------------------------------------------
# extract_system_message: multiple system messages (line ~273)
# ---------------------------------------------------------------------------


class TestExtractSystemMessageMultiple:
    """Multiple system messages are joined with double newlines."""

    def test_multiple_system_messages_joined_with_double_newline(self) -> None:
        """Two system messages → joined with '\\n\\n'.

        Line ~273 in request_adapter.py — the len(system_parts) > 1 branch
        Contract: provider-protocol:complete:MUST:2 — system message forwarding
        """
        from amplifier_module_provider_github_copilot.request_adapter import extract_system_message

        @dataclass
        class Msg:
            role: str
            content: str

        @dataclass
        class Request:
            messages: list[Any]

        request = Request(
            messages=[
                Msg(role="system", content="You are a helpful assistant."),
                Msg(role="system", content="Always be concise."),
                Msg(role="user", content="Hello"),
            ]
        )

        result = extract_system_message(request)

        assert result is not None
        assert "\n\n" in result
        assert "You are a helpful assistant." in result
        assert "Always be concise." in result

    def test_single_system_message_no_join_log(self) -> None:
        """Single system message is returned as-is without join."""
        from amplifier_module_provider_github_copilot.request_adapter import extract_system_message

        @dataclass
        class Msg:
            role: str
            content: str

        @dataclass
        class Request:
            messages: list[Any]

        request = Request(
            messages=[
                Msg(role="system", content="You are a concise assistant."),
            ]
        )

        result = extract_system_message(request)
        assert result == "You are a concise assistant."

    def test_no_system_messages_returns_none(self) -> None:
        """No system messages → returns None."""
        from amplifier_module_provider_github_copilot.request_adapter import extract_system_message

        @dataclass
        class Msg:
            role: str
            content: str

        @dataclass
        class Request:
            messages: list[Any]

        request = Request(messages=[Msg(role="user", content="Hello")])
        assert extract_system_message(request) is None


# ---------------------------------------------------------------------------
# build_request_payload_for_observability: nested OpenAI-style tool (line ~346)
# ---------------------------------------------------------------------------


class TestBuildRequestPayloadToolNameExtraction:
    """build_request_payload handles nested {'function': {'name': '...'}} format."""

    def test_nested_openai_style_tool_name_extracted(self) -> None:
        """Nested {'function': {'name': '...'}} format tool name is extracted.

        Line ~346 in request_adapter.py — Subformat 2a: OpenAI-style nested dict
        Contract: observability:Payload:SHOULD:2 — Type-safe tool name extraction
        """
        from amplifier_module_provider_github_copilot.request_adapter import (
            build_request_payload_for_observability,
        )
        from amplifier_module_provider_github_copilot.sdk_adapter import CompletionRequest

        # OpenAI-style nested format: {"function": {"name": "...", "description": "..."}}
        tools_nested: list[Any] = [
            {"function": {"name": "search_web", "description": "Search the web"}},
            {"function": {"name": "read_file", "description": "Read a file"}},
        ]
        internal_request = CompletionRequest(
            prompt="test",
            model="gpt-4o",
            tools=tools_nested,
        )

        result = build_request_payload_for_observability(
            model="gpt-4o",
            request=object(),
            internal_request=internal_request,
        )

        assert result["tool_names"] == ["search_web", "read_file"]

    def test_flat_amplifier_native_tool_name_extracted(self) -> None:
        """Flat {'name': '...'} format tool name is extracted.

        Subformat 2b — should already be covered but verify
        """
        from amplifier_module_provider_github_copilot.request_adapter import (
            build_request_payload_for_observability,
        )
        from amplifier_module_provider_github_copilot.sdk_adapter import CompletionRequest

        tools_flat: list[Any] = [
            {"name": "list_agents", "description": "List available agents"},
        ]
        internal_request = CompletionRequest(
            prompt="test",
            model="gpt-4o",
            tools=tools_flat,
        )

        result = build_request_payload_for_observability(
            model="gpt-4o",
            request=object(),
            internal_request=internal_request,
        )

        assert result["tool_names"] == ["list_agents"]

    def test_mixed_tool_formats_both_extracted(self) -> None:
        """Mix of nested and flat tool formats — both names extracted.

        Exercises multiple branches in the tool extraction loop.
        """
        from amplifier_module_provider_github_copilot.request_adapter import (
            build_request_payload_for_observability,
        )
        from amplifier_module_provider_github_copilot.sdk_adapter import CompletionRequest

        @dataclass
        class ToolSpecLike:
            name: str
            description: str

        tools_mixed: list[Any] = [
            ToolSpecLike(name="tool_a", description="First tool"),  # Format 1: object with .name
            {"name": "tool_b"},  # Format 2b: flat dict
            {"function": {"name": "tool_c"}},  # Format 2a: nested OpenAI-style
        ]
        internal_request = CompletionRequest(
            prompt="test",
            model="gpt-4o",
            tools=tools_mixed,
        )

        result = build_request_payload_for_observability(
            model="gpt-4o",
            request=object(),
            internal_request=internal_request,
        )

        assert "tool_a" in result["tool_names"]
        assert "tool_b" in result["tool_names"]
        assert "tool_c" in result["tool_names"]
