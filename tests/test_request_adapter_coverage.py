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

from dataclasses import dataclass, field
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

    def test_block_with_both_thinking_and_text_attrs_takes_thinking_path(self) -> None:
        """Block with explicit type="thinking" AND .text attribute classifies as thinking.

        Contract: streaming-contract:Accumulation:MUST:2 — block boundaries maintained.

        Regression guard: if the explicit block_type == "thinking" branch is removed
        entirely, the hasattr(block, "text") fallback would match and return the text
        path. This canary turns red when the thinking type-check is deleted.

        Note: this test does NOT catch a simple branch reorder of the elif chain because
        the explicit block_type == "thinking" check fires before any elif. The untyped
        test below (type=None) is the canary for hasattr ordering.

        Mutation check: remove the explicit `block_type == "thinking"` branch from
        request_adapter._extract_content_block — this assertion turns red.
        """
        from amplifier_module_provider_github_copilot.request_adapter import (
            _extract_content_block,  # pyright: ignore[reportPrivateUsage]
        )

        @dataclass
        class ThinkingWithShadowText:
            type: str = "thinking"
            thinking: str = "internal reasoning payload"
            text: str = "SHOULD_NOT_APPEAR_IN_OUTPUT"

        result = _extract_content_block(ThinkingWithShadowText())
        assert result == "[Thinking: internal reasoning payload]"
        assert "SHOULD_NOT_APPEAR_IN_OUTPUT" not in result

    def test_block_type_none_with_both_thinking_and_text_takes_thinking_path(
        self,
    ) -> None:
        """type=None + BOTH .thinking AND .text → hasattr fallback picks thinking.

        Contract: streaming-contract:Accumulation:MUST:2

        Exercises the `block_type is None` fallback path where both hasattr
        checks would match. The thinking hasattr check is first in the if/elif
        chain — this guards that ordering.
        """
        from amplifier_module_provider_github_copilot.request_adapter import (
            _extract_content_block,  # pyright: ignore[reportPrivateUsage]
        )

        @dataclass
        class UntypedThinkingAndText:
            thinking: str = "untyped reasoning"
            text: str = "SHOULD_NOT_APPEAR"

        result = _extract_content_block(UntypedThinkingAndText())
        assert result == "[Thinking: untyped reasoning]"
        assert "SHOULD_NOT_APPEAR" not in result


# ---------------------------------------------------------------------------
# _extract_content_block: tool result without tool_call_id (lines ~221-222)
# ---------------------------------------------------------------------------


class TestExtractContentBlockToolResult:
    """ToolResultContent handles missing tool_call_id and empty output."""

    def test_tool_result_with_output_no_id_returns_simple_format(self) -> None:
        """ToolResult with output but no tool_call_id returns '[Tool Result: ...]'.

        Line ~221 in request_adapter.py — return f"[Tool Result: {output}]"
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
        """
        # Contract: sdk-boundary:Config:MUST:2
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

        assert result == "You are a helpful assistant.\n\nAlways be concise."

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
        """
        # Contract: observability:Payload:SHOULD:2
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
        # Contract: observability:Payload:SHOULD:2
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
        # Contract: observability:Payload:SHOULD:2
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


# ---------------------------------------------------------------------------
# _extract_content_block: None block guard (request_adapter.py line ~197)
# _extract_message_content: None content guard (request_adapter.py line ~164)
# extract_system_message: empty messages guard (request_adapter.py line ~285)
# ---------------------------------------------------------------------------


class TestNullGuardBranches:
    """Null/empty guard branches in request_adapter that must be covered.

    Contract: provider-protocol:complete:MUST:1 — content extraction is
    fault-tolerant. These are real return paths, not dead code.
    """

    def test_extract_content_block_with_none_returns_empty(self) -> None:
        """_extract_content_block(None) returns ''.

        Line ~197 in request_adapter.py: `if block is None: return ""`
        Contract: provider-protocol:complete:MUST:1 — null block guard
        Caller: _extract_message_content iterates list items; None can appear
        in a malformed content list from a future kernel version.
        """
        from amplifier_module_provider_github_copilot.request_adapter import (
            _extract_content_block,  # pyright: ignore[reportPrivateUsage]
        )

        result = _extract_content_block(None)
        assert result == "", (
            "_extract_content_block(None) must return '' not raise. "
            "Contract: provider-protocol:complete:MUST:1"
        )

    def test_extract_message_content_with_none_returns_empty(self) -> None:
        """_extract_message_content(None) returns ''.

        Line ~164 in request_adapter.py: `if content is None: return ""`
        Contract: provider-protocol:complete:MUST:1 — null content guard
        A message with content=None is permissible (role-only message).
        """
        from amplifier_module_provider_github_copilot.request_adapter import (
            _extract_message_content,  # pyright: ignore[reportPrivateUsage]
        )

        result = _extract_message_content(None)
        assert result == "", (
            "_extract_message_content(None) must return '' not raise. "
            "Contract: provider-protocol:complete:MUST:1"
        )

    def test_extract_system_message_with_empty_messages_returns_none(self) -> None:
        """extract_system_message with messages=[] returns None.

        Line ~285 in request_adapter.py: `if not messages: return None`
        Contract: provider-protocol:complete:MUST:1 — empty request guard
        The kernel MAY send a request with no messages (edge case in orchestrator).
        """
        from amplifier_module_provider_github_copilot.request_adapter import extract_system_message

        @dataclass
        class EmptyRequest:
            messages: list[Any] = field(default_factory=list)

        result = extract_system_message(EmptyRequest())
        assert result is None, (
            "extract_system_message with empty messages must return None. "
            "Contract: provider-protocol:complete:MUST:1"
        )
