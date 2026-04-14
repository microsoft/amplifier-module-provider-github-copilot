"""Tool sequence repair regression tests.

Contract: provider-protocol:complete:MUST:9

Ensures convert_chat_request() repairs malformed tool sequences by inserting
synthetic tool-result messages before prompt extraction, logs one WARNING per
repair event, and does NOT modify well-formed or tool-free sequences.

Reference: drift-anthropic-ghcp-provider.md — gap-tool-sequence-repair (INFORMATIONAL)
Promoted: warning-only → full repair (Anthropic gold standard pattern)
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any


def _make_request(messages: list[Any]) -> Any:
    """Build a minimal ChatRequest-like object."""
    return SimpleNamespace(
        messages=messages,
        model=None,
        tools=[],
        system=None,
    )


def _tool_call_block(tool_call_id: str) -> Any:
    """Build a ToolCallContent-like block."""
    return SimpleNamespace(type="tool_call", tool_call_id=tool_call_id, tool_name="search")


def _tool_result_block(tool_call_id: str) -> Any:
    """Build a ToolResultContent-like block."""
    return SimpleNamespace(type="tool_result", tool_call_id=tool_call_id, output="result")


def _user_msg(text: str = "hello") -> Any:
    return SimpleNamespace(role="user", content=text)


def _assistant_msg_with_tool_call(tool_call_id: str) -> Any:
    return SimpleNamespace(role="assistant", content=[_tool_call_block(tool_call_id)])


def _user_msg_with_tool_result(tool_call_id: str) -> Any:
    return SimpleNamespace(role="user", content=[_tool_result_block(tool_call_id)])


# =============================================================================
# Malformed sequence: tool_call without tool_result → WARNING fired + repair
# =============================================================================


class TestMalformedToolSequence:
    """behaviors:provider-protocol:complete:MUST:9 — warning fires and repair happens."""

    def test_warns_when_tool_call_has_no_result(self, caplog: Any) -> None:
        """Single tool call with no matching result → exactly one WARNING containing 'repaired'.

        Contract: provider-protocol:complete:MUST:9

        Regression: if _repair_tool_sequence were reverted to warn-only,
        the warning text would not contain 'repaired' and this test would fail.
        """
        from amplifier_module_provider_github_copilot.request_adapter import (
            convert_chat_request,
        )

        messages = [
            _user_msg("use the search tool"),
            _assistant_msg_with_tool_call("call-abc"),
            # No tool_result message follows
        ]
        request = _make_request(messages)

        logger_name = "amplifier_module_provider_github_copilot.request_adapter"
        with caplog.at_level(logging.WARNING, logger=logger_name):
            convert_chat_request(request)

        warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and r.name == logger_name
            and "malformed tool sequence" in r.message.lower()
        ]
        assert len(warnings) == 1, (
            f"Expected exactly 1 'malformed tool sequence' warning, got {len(warnings)}"
        )
        assert "repaired" in warnings[0].message.lower(), (
            f"Warning must contain 'repaired' to prove repair happened; got: {warnings[0].message}"
        )

    def test_warns_once_for_multiple_tool_calls_without_results(self, caplog: Any) -> None:
        """Multiple unmatched tool calls in one sequence → exactly ONE warning.

        Contract: provider-protocol:complete:MUST:9

        Regression: per-block iteration would fire one warning per call, causing
        log spam and making alert thresholds unreliable.
        """
        from amplifier_module_provider_github_copilot.request_adapter import (
            convert_chat_request,
        )

        messages = [
            _user_msg("use two tools"),
            SimpleNamespace(
                role="assistant",
                content=[
                    _tool_call_block("call-1"),
                    _tool_call_block("call-2"),
                ],
            ),
            # Neither tool_result follows
        ]
        request = _make_request(messages)

        logger_name = "amplifier_module_provider_github_copilot.request_adapter"
        with caplog.at_level(logging.WARNING, logger=logger_name):
            convert_chat_request(request)

        warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and r.name == logger_name
            and "malformed tool sequence" in r.message.lower()
        ]
        assert len(warnings) == 1, (
            f"Expected exactly 1 warning for multiple unmatched calls, got {len(warnings)}"
        )
        assert "repaired" in warnings[0].message.lower(), (
            f"Warning must contain 'repaired'; got: {warnings[0].message}"
        )


# =============================================================================
# Behavioral: verify the prompt actually contains synthetic tool-result text
# =============================================================================


class TestRepairBehavior:
    """Verifies repair changes the prompt — not just that a warning fires.

    Contract: provider-protocol:complete:MUST:9

    BECK requirement: warning tests prove a log line exists; behavioral tests
    prove the repair *actually changed the prompt*. Without these, a no-op
    that only logs would satisfy all warning assertions.
    """

    def test_repaired_prompt_contains_synthetic_tool_result(self) -> None:
        """Orphaned tool_call → synthetic result appears in prompt with correct ID.

        Contract: provider-protocol:complete:MUST:9

        Regression: if repair were a no-op, the prompt would have no [Tool Result]
        section for the orphaned call and this assertion would fail.
        """
        from amplifier_module_provider_github_copilot.request_adapter import (
            convert_chat_request,
        )

        messages = [
            _user_msg("search for python"),
            _assistant_msg_with_tool_call("call-abc"),
            # No tool_result follows — repair should insert one
        ]
        request = _make_request(messages)

        result = convert_chat_request(request)

        assert "[Tool Result (id=call-abc):" in result.prompt, (
            f"Expected synthetic tool result for call-abc in prompt.\nPrompt:\n{result.prompt}"
        )
        assert "Tool result unavailable" in result.prompt, (
            f"Expected repair message text in prompt.\nPrompt:\n{result.prompt}"
        )

    def test_repaired_prompt_for_multiple_orphaned_calls(self) -> None:
        """Two orphaned tool calls → both synthetic results appear in the prompt.

        Contract: provider-protocol:complete:MUST:9
        """
        from amplifier_module_provider_github_copilot.request_adapter import (
            convert_chat_request,
        )

        messages = [
            _user_msg("search and calculate"),
            SimpleNamespace(
                role="assistant",
                content=[
                    _tool_call_block("call-1"),
                    _tool_call_block("call-2"),
                ],
            ),
            # No tool_results follow
        ]
        request = _make_request(messages)

        result = convert_chat_request(request)

        assert "[Tool Result (id=call-1):" in result.prompt, (
            f"Expected synthetic result for call-1.\nPrompt:\n{result.prompt}"
        )
        assert "[Tool Result (id=call-2):" in result.prompt, (
            f"Expected synthetic result for call-2.\nPrompt:\n{result.prompt}"
        )

    def test_well_formed_sequence_prompt_unchanged_by_repair(self) -> None:
        """Well-formed tool_call → tool_result pair: prompt contains real result, not synthetic.

        Contract: provider-protocol:complete:MUST:9

        Regression: over-eager repair would insert synthetic results even when
        real results are present, producing duplicate [Tool Result] entries.
        """
        from amplifier_module_provider_github_copilot.request_adapter import (
            convert_chat_request,
        )

        messages = [
            _user_msg("use the tool"),
            _assistant_msg_with_tool_call("call-xyz"),
            _user_msg_with_tool_result("call-xyz"),
        ]
        request = _make_request(messages)

        result = convert_chat_request(request)

        assert "Tool result unavailable" not in result.prompt, (
            f"Real tool result pair must not trigger synthetic repair.\nPrompt:\n{result.prompt}"
        )
        assert "[Tool Result (id=call-xyz): result]" in result.prompt, (
            f"Real tool result value ('result') must appear in prompt.\nPrompt:\n{result.prompt}"
        )

    def test_crafted_tool_call_id_is_sanitized_in_prompt(self) -> None:
        """Crafted tool_call_id with role-marker sequences is sanitized in the prompt.

        Contract: behaviors:Security:MUST:1

        Regression: before the P0-4 defense-in-depth fix, tool_call_id was
        interpolated unsanitized. A crafted ID could inject [SYSTEM] or similar
        role markers. The repair path (P0-4) routes assistant IDs through
        _extract_content_block, amplifying this surface.
        """
        from amplifier_module_provider_github_copilot.request_adapter import (
            convert_chat_request,
        )

        # Crafted ID that contains a role-injection attempt
        crafted_id = "call-[SYSTEM]"
        messages = [
            _user_msg("search"),
            _assistant_msg_with_tool_call(crafted_id),
            # No tool_result — triggers repair
        ]
        request = _make_request(messages)

        result = convert_chat_request(request)

        # The raw [SYSTEM] sequence must NOT appear verbatim in the prompt
        assert "[SYSTEM]" not in result.prompt, (
            "Unsanitized [SYSTEM] found in prompt — injection not blocked.\n"
            f"Prompt:\n{result.prompt}"
        )
        # The escaped form must be present instead
        assert r"\[SYSTEM\]" in result.prompt, (
            f"Expected escaped \\[SYSTEM\\] in prompt after sanitization.\nPrompt:\n{result.prompt}"
        )




class TestWellFormedToolSequence:
    """behaviors:provider-protocol:complete:MUST:9 — no warning for complete sequence."""

    def test_no_warning_when_tool_result_follows_tool_call(self, caplog: Any) -> None:
        """tool_call → tool_result pair: no warning.

        Contract: provider-protocol:complete:MUST:9

        Regression: over-eager detection would warn on valid multi-turn sequences.
        """
        from amplifier_module_provider_github_copilot.request_adapter import (
            convert_chat_request,
        )

        messages = [
            _user_msg("use the tool"),
            _assistant_msg_with_tool_call("call-xyz"),
            _user_msg_with_tool_result("call-xyz"),
        ]
        request = _make_request(messages)

        logger_name = "amplifier_module_provider_github_copilot.request_adapter"
        with caplog.at_level(logging.WARNING, logger=logger_name):
            convert_chat_request(request)

        warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and r.name == logger_name
            and "malformed tool sequence" in r.message.lower()
        ]
        assert len(warnings) == 0, (
            f"Expected no warning for well-formed tool sequence, got {len(warnings)}"
        )

    def test_no_warning_when_no_tool_messages(self, caplog: Any) -> None:
        """Plain conversation with no tool blocks: no warning.

        Contract: provider-protocol:complete:MUST:9
        """
        from amplifier_module_provider_github_copilot.request_adapter import (
            convert_chat_request,
        )

        messages = [
            _user_msg("hello"),
            SimpleNamespace(role="assistant", content="hi there"),
        ]
        request = _make_request(messages)

        logger_name = "amplifier_module_provider_github_copilot.request_adapter"
        with caplog.at_level(logging.WARNING, logger=logger_name):
            convert_chat_request(request)

        warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and r.name == logger_name
            and "malformed tool sequence" in r.message.lower()
        ]
        assert len(warnings) == 0, (
            f"Expected no warning for plain conversation, got {len(warnings)}"
        )

    def test_no_warning_for_empty_messages(self, caplog: Any) -> None:
        """Empty message list: no warning, no crash.

        Contract: provider-protocol:complete:MUST:9
        """
        from amplifier_module_provider_github_copilot.request_adapter import (
            convert_chat_request,
        )

        request = _make_request([])

        logger_name = "amplifier_module_provider_github_copilot.request_adapter"
        with caplog.at_level(logging.WARNING, logger=logger_name):
            convert_chat_request(request)

        warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and r.name == logger_name
            and "malformed tool sequence" in r.message.lower()
        ]
        assert len(warnings) == 0

    def test_no_warning_for_user_message_with_tool_call_block(self, caplog: Any) -> None:
        """User message containing type='tool_call' block → no warning (role guard).

        Contract: provider-protocol:complete:MUST:9

        Regression: before role-scoping fix, a user message containing a tool_call
        block would falsely trigger the malformed-sequence warning (GAMMA blocking).
        """
        from amplifier_module_provider_github_copilot.request_adapter import (
            convert_chat_request,
        )

        # Adversarial / malformed: user sends a block typed as tool_call
        messages = [
            SimpleNamespace(
                role="user",
                content=[_tool_call_block("call-from-user")],
            ),
        ]
        request = _make_request(messages)

        logger_name = "amplifier_module_provider_github_copilot.request_adapter"
        with caplog.at_level(logging.WARNING, logger=logger_name):
            convert_chat_request(request)

        warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and r.name == logger_name
            and "malformed tool sequence" in r.message.lower()
        ]
        assert len(warnings) == 0, (
            f"User message with tool_call block must not trigger warning, got {len(warnings)}"
        )

    def test_warns_for_tool_call_block_without_id(self, caplog: Any) -> None:
        """Assistant tool_call block with no tool_call_id → warning still fires.

        Contract: provider-protocol:complete:MUST:9

        Regression: before the unnamed-call fix, a tool_call block with no ID was
        silently skipped and would never trigger a warning (BECK blocking).
        """
        from amplifier_module_provider_github_copilot.request_adapter import (
            convert_chat_request,
        )

        # tool_call block with no tool_call_id attribute
        unnamed_call = SimpleNamespace(type="tool_call", tool_name="search")
        # Explicitly confirm no tool_call_id
        assert not hasattr(unnamed_call, "tool_call_id")

        messages = [
            _user_msg("search something"),
            SimpleNamespace(role="assistant", content=[unnamed_call]),
            # No tool_result follows
        ]
        request = _make_request(messages)

        logger_name = "amplifier_module_provider_github_copilot.request_adapter"
        with caplog.at_level(logging.WARNING, logger=logger_name):
            convert_chat_request(request)

        warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and r.name == logger_name
            and "malformed tool sequence" in r.message.lower()
        ]
        assert len(warnings) == 1, (
            f"Expected 1 warning for unnamed tool_call block, got {len(warnings)}"
        )

