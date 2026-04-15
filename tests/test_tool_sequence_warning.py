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


def _user_msg(text: str = "hello") -> Any:
    return SimpleNamespace(role="user", content=text)


def _assistant_msg_with_tool_call(tool_call_id: str) -> Any:
    return SimpleNamespace(role="assistant", content=[_tool_call_block(tool_call_id)])


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
            r
            for r in caplog.records
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
            r
            for r in caplog.records
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
            r
            for r in caplog.records
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
            r
            for r in caplog.records
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
            r
            for r in caplog.records
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
            r
            for r in caplog.records
            if r.levelno == logging.WARNING
            and r.name == logger_name
            and "malformed tool sequence" in r.message.lower()
        ]
        assert len(warnings) == 1, (
            f"Expected 1 warning for unnamed tool_call block, got {len(warnings)}"
        )


# =============================================================================
# Real kernel type tests — ToolCallBlock uses .id not .tool_call_id
# =============================================================================


class TestRealKernelTypes:
    """Verify _repair_tool_sequence handles actual amplifier_core kernel types.

    Contract: provider-protocol:complete:MUST:9

    Regression guard: ToolCallBlock uses `.id` (not `.tool_call_id`). If
    _repair_tool_sequence reads only `.tool_call_id`, every real tool call
    falls into unnamed_calls and repair fires unconditionally.
    """

    def test_toolcallblock_with_matching_toolresultblock_no_repair(self, caplog: Any) -> None:
        """ToolCallBlock(id=X) + role='tool' Message(tool_call_id=X) → NO repair triggered.

        This is the normal Amplifier multi-turn case: turn N emits tool calls,
        turn N+1 includes their results as role='tool' Messages. Repair must NOT fire.

        Regression: before the .id fallback fix, this would fire because
        ToolCallBlock.tool_call_id does not exist (returns None), so the block
        was treated as unnamed and always triggered repair.
        """
        from amplifier_core.message_models import Message, ToolCallBlock

        from amplifier_module_provider_github_copilot.request_adapter import (
            convert_chat_request,
        )

        tool_id = "toolu_01Xyz"
        messages = [
            SimpleNamespace(role="user", content="search for something"),
            SimpleNamespace(
                role="assistant",
                content=[
                    ToolCallBlock(
                        type="tool_call",
                        id=tool_id,
                        name="bash",
                        input={"command": "ls"},
                    )
                ],
            ),
            # Tool result in canonical kernel format: role='tool' Message
            Message(role="tool", content='["file1.py", "file2.py"]', tool_call_id=tool_id),
            SimpleNamespace(role="user", content="what files did you find?"),
        ]
        request = _make_request(messages)

        logger_name = "amplifier_module_provider_github_copilot.request_adapter"
        with caplog.at_level(logging.WARNING, logger=logger_name):
            convert_chat_request(request)

        warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING
            and r.name == logger_name
            and "malformed tool sequence" in r.message.lower()
        ]
        assert len(warnings) == 0, (
            f"Repair must NOT fire when ToolCallBlock.id has a matching role='tool' Message. "
            f"Got {len(warnings)} warning(s). This means ToolCallBlock.id is not being "
            f"read — only .tool_call_id is checked, causing false-positive repairs every turn."
        )

    def test_toolcallblock_without_result_fires_repair(self, caplog: Any) -> None:
        """ToolCallBlock(id=X) with NO matching result → repair fires exactly once.

        Contract: provider-protocol:complete:MUST:9

        This verifies the repair still works correctly for real kernel types
        even after adding the .id fallback.
        """
        from amplifier_core.message_models import ToolCallBlock

        from amplifier_module_provider_github_copilot.request_adapter import (
            convert_chat_request,
        )

        messages = [
            SimpleNamespace(role="user", content="run a command"),
            SimpleNamespace(
                role="assistant",
                content=[
                    ToolCallBlock(
                        type="tool_call",
                        id="toolu_orphaned_01",
                        name="bash",
                        input={"command": "pwd"},
                    )
                ],
            ),
            # No tool result — repair required
        ]
        request = _make_request(messages)

        logger_name = "amplifier_module_provider_github_copilot.request_adapter"
        with caplog.at_level(logging.WARNING, logger=logger_name):
            result = convert_chat_request(request)

        warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING
            and r.name == logger_name
            and "malformed tool sequence" in r.message.lower()
        ]
        assert len(warnings) == 1, (
            f"Expected 1 repair warning for orphaned ToolCallBlock, got {len(warnings)}"
        )
        # Synthetic result must reference the actual ID from ToolCallBlock.id
        assert (
            "[Tool Result (id=toulu_orphaned_01):" in result.prompt
            or "[Tool Result (id=toolu_orphaned_01):" in result.prompt
        ), f"Synthetic result must reference ToolCallBlock.id. Prompt:\n{result.prompt}"


# =============================================================================
# Amplifier kernel transcript format — role='tool' with Message.tool_call_id
# The real Amplifier transcript uses role='tool' Messages where tool_call_id
# is set at the Message level (not inside content blocks).
# Verified live: session 5fc69faf transcript shows:
#   msg role=tool tool_call_id='toolu_vrtx_01LNhVwZNNrrmeTjzw9pQ53E'
#   content='{"error": null, "output": {...}}'
# Reference: amplifier-module-provider-anthropic _find_missing_tool_results():
#   elif msg.role == "tool" and hasattr(msg, "tool_call_id") and msg.tool_call_id:
#       tool_results.add(msg.tool_call_id)
# =============================================================================


class TestKernelRoleToolFormat:
    """Verify _repair_tool_sequence handles role='tool' Messages (Amplifier kernel format).

    Contract: provider-protocol:complete:MUST:9

    Regression guard: Amplifier's kernel transcript stores tool results as
    role='tool' Messages with tool_call_id on the Message itself — NOT as
    role='user' + type='tool_result' blocks. The Message.tool_call_id is the
    canonical link back to ToolCallBlock.id.

    Each role='tool' Message satisfies exactly one ToolCallBlock by ID.
    Positional inference is wrong because it marks all calls in an assistant
    message satisfied from the first result — creating false negatives when
    only some of multiple parallel tool calls receive results.
    """

    def _role_tool_msg(self, tool_call_id: str, tool_name: str = "read_file") -> Any:
        """Build a role='tool' Message as Amplifier kernel produces.

        Uses real Message type so tool_call_id is an attribute.
        SimpleNamespace without tool_call_id would mask detection regressions.
        """
        import json

        from amplifier_core.message_models import Message

        return Message(
            role="tool",
            tool_call_id=tool_call_id,
            name=tool_name,
            content=json.dumps({"error": None, "output": {"content": "file contents"}}),
        )

    def test_role_tool_message_satisfies_preceding_tool_call_no_repair(self, caplog: Any) -> None:
        """ToolCallBlock(id=X) + Message(role='tool', tool_call_id=X) → NO repair.

        Contract: provider-protocol:complete:MUST:9

        Message.tool_call_id must be read directly — NOT inferred positionally.
        This is the exact pattern session 5fc69faf produced on every tool turn.

        Mutation check: remove the `elif role == "tool"` branch →
        tool_result_ids stays empty → repair fires → assertion turns red. ✓
        """
        from amplifier_core.message_models import ToolCallBlock

        from amplifier_module_provider_github_copilot.request_adapter import (
            convert_chat_request,
        )

        tool_id = "toolu_vrtx_01LNhVwZNNrrmeTjzw9pQ53E"
        messages = [
            _user_msg("read the file"),
            SimpleNamespace(
                role="assistant",
                content=[
                    ToolCallBlock(
                        type="tool_call",
                        id=tool_id,
                        name="read_file",
                        input={"file_path": "README.md"},
                    ),
                ],
            ),
            self._role_tool_msg(tool_call_id=tool_id),
        ]
        request = _make_request(messages)

        logger_name = "amplifier_module_provider_github_copilot.request_adapter"
        with caplog.at_level(logging.WARNING, logger=logger_name):
            convert_chat_request(request)

        warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING
            and r.name == logger_name
            and "malformed tool sequence" in r.message.lower()
        ]
        assert len(warnings) == 0, (
            f"role='tool' Message(tool_call_id={tool_id!r}) must satisfy "
            f"ToolCallBlock(id={tool_id!r}). "
            f"Repair must NOT fire. Got {len(warnings)} warning(s)."
        )

    def test_sequential_tool_calls_with_role_tool_results_no_repair(self, caplog: Any) -> None:
        """Two sequential tool calls each with matching role='tool' result → NO repair.

        Contract: provider-protocol:complete:MUST:9

        Matches session 5fc69faf:
          msg[1] assistant ToolCallBlock(id=id1) → msg[2] Message(role='tool', tool_call_id=id1)
          msg[3] assistant ToolCallBlock(id=id2) → msg[4] Message(role='tool', tool_call_id=id2)

        Mutation check: swap id1/id2 on the role='tool' messages →
        neither ID matches its call → repair fires → assertion turns red. ✓
        """
        from amplifier_core.message_models import ToolCallBlock

        from amplifier_module_provider_github_copilot.request_adapter import (
            convert_chat_request,
        )

        id1 = "toolu_vrtx_011hAodwuPZ5wVE7SJcQCWec"
        id2 = "toolu_vrtx_01KoSZNbabhPqvFRp2emV1P1"
        messages = [
            _user_msg("read the file twice"),
            SimpleNamespace(
                role="assistant",
                content=[
                    ToolCallBlock(
                        type="tool_call", id=id1, name="read_file", input={"path": "a.py"}
                    )
                ],
            ),
            self._role_tool_msg(tool_call_id=id1),
            SimpleNamespace(
                role="assistant",
                content=[
                    ToolCallBlock(
                        type="tool_call", id=id2, name="read_file", input={"path": "b.py"}
                    )
                ],
            ),
            self._role_tool_msg(tool_call_id=id2),
        ]
        request = _make_request(messages)

        logger_name = "amplifier_module_provider_github_copilot.request_adapter"
        with caplog.at_level(logging.WARNING, logger=logger_name):
            convert_chat_request(request)

        warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING
            and r.name == logger_name
            and "malformed tool sequence" in r.message.lower()
        ]
        assert len(warnings) == 0, (
            f"Both sequential tool calls have matching role='tool' results — "
            f"repair must NOT fire. Got {len(warnings)} warning(s)."
        )

    def test_partial_role_tool_results_fires_repair_for_missing(self, caplog: Any) -> None:
        """3 parallel tool calls, only 2 role='tool' results → repair fires for the 1 missing.

        Contract: provider-protocol:complete:MUST:9

        SCHNEIER regression gap: positional backward-walk would mark ALL 3 calls
        satisfied when ANY role='tool' result arrives (walks back to the assistant
        message and adds all 3 IDs at once — false negative).
        Message.tool_call_id approach adds only the specific answered ID.

        Mutation check: add a third self._role_tool_msg(tool_call_id=id_c) →
        all 3 are matched → repair doesn't fire → assertion turns red. ✓
        """
        from amplifier_core.message_models import ToolCallBlock

        from amplifier_module_provider_github_copilot.request_adapter import (
            convert_chat_request,
        )

        id_a = "toolu_tc_A"
        id_b = "toolu_tc_B"
        id_c = "toolu_tc_C"
        messages = [
            _user_msg("read three files"),
            SimpleNamespace(
                role="assistant",
                content=[
                    ToolCallBlock(
                        type="tool_call", id=id_a, name="read_file", input={"path": "a.py"}
                    ),
                    ToolCallBlock(
                        type="tool_call", id=id_b, name="read_file", input={"path": "b.py"}
                    ),
                    ToolCallBlock(
                        type="tool_call", id=id_c, name="read_file", input={"path": "c.py"}
                    ),
                ],
            ),
            self._role_tool_msg(tool_call_id=id_a),  # A answered
            self._role_tool_msg(tool_call_id=id_b),  # B answered
            # C has no result — genuine orphan
        ]
        request = _make_request(messages)

        logger_name = "amplifier_module_provider_github_copilot.request_adapter"
        with caplog.at_level(logging.WARNING, logger=logger_name):
            result = convert_chat_request(request)

        warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING
            and r.name == logger_name
            and "malformed tool sequence" in r.message.lower()
        ]
        assert len(warnings) == 1, (
            f"Exactly 1 of 3 tool calls is missing — repair must fire once. "
            f"Got {len(warnings)} warning(s). Positional detection would give 0 "
            f"(false negative: all 3 marked satisfied from position)."
        )
        assert "[Tool Result (id=toolu_tc_C):" in result.prompt, (
            f"Synthetic result must reference id_c specifically. Prompt:\n{result.prompt}"
        )
        assert result.prompt.count("Tool result unavailable") == 1, (
            f"Only id_c should be synthetically repaired. Prompt:\n{result.prompt}"
        )

    def test_role_tool_without_tool_call_id_does_not_satisfy_calls(self, caplog: Any) -> None:
        """role='tool' Message with tool_call_id=None does not satisfy any ToolCallBlock.

        Contract: provider-protocol:complete:MUST:9

        If a role='tool' Message arrives with tool_call_id=None (malformed),
        the tool call that preceded it is still unmatched and repair fires.

        Mutation check: set tool_call_id=tool_id on the Message →
        call is matched → repair doesn't fire → assertion turns red. ✓
        """
        from amplifier_core.message_models import Message, ToolCallBlock

        from amplifier_module_provider_github_copilot.request_adapter import (
            convert_chat_request,
        )

        tool_id = "toolu_abc_123"
        messages = [
            _user_msg("run a tool"),
            SimpleNamespace(
                role="assistant",
                content=[ToolCallBlock(type="tool_call", id=tool_id, name="bash", input={})],
            ),
            Message(role="tool", content="some result", tool_call_id=None),
        ]
        request = _make_request(messages)

        logger_name = "amplifier_module_provider_github_copilot.request_adapter"
        with caplog.at_level(logging.WARNING, logger=logger_name):
            convert_chat_request(request)

        warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING
            and r.name == logger_name
            and "malformed tool sequence" in r.message.lower()
        ]
        assert len(warnings) == 1, (
            f"role='tool'(tool_call_id=None) must NOT satisfy {tool_id!r}. "
            f"Repair must fire. Got {len(warnings)} warning(s)."
        )

    def test_missing_role_tool_result_still_fires_repair(self, caplog: Any) -> None:
        """ToolCallBlock with no role='tool' result → repair fires exactly once.

        Contract: provider-protocol:complete:MUST:9

        Verifies the role='tool' fix doesn't suppress genuine orphan detection.

        Mutation check: add self._role_tool_msg(tool_call_id='toolu_orphaned_99') →
        call is matched → repair doesn't fire → assertion turns red. ✓
        """
        from amplifier_core.message_models import ToolCallBlock

        from amplifier_module_provider_github_copilot.request_adapter import (
            convert_chat_request,
        )

        messages = [
            _user_msg("run a tool"),
            SimpleNamespace(
                role="assistant",
                content=[
                    ToolCallBlock(type="tool_call", id="toolu_orphaned_99", name="bash", input={})
                ],
            ),
            # No role='tool' result — genuine orphan
        ]
        request = _make_request(messages)

        logger_name = "amplifier_module_provider_github_copilot.request_adapter"
        with caplog.at_level(logging.WARNING, logger=logger_name):
            convert_chat_request(request)

        warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING
            and r.name == logger_name
            and "malformed tool sequence" in r.message.lower()
        ]
        assert len(warnings) == 1, (
            f"Genuine orphaned ToolCallBlock must trigger repair. Got {len(warnings)} warning(s)."
        )
