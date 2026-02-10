"""
SDK Assumption Tests: preToolUse Hook Deny Behavior

These tests validate that the SDK's preToolUse hook correctly prevents
tool handler execution when returning {"permissionDecision": "deny"}.

CRITICAL ASSUMPTION:
    When a preToolUse hook returns {"permissionDecision": "deny"}, the SDK
    MUST NOT invoke the tool handler. This is essential for our "dumb-pipe"
    pattern where tools are captured but never executed locally.

WHY THIS MATTERS:
    Our provider registers no-op tool handlers purely to satisfy the SDK's
    Tool constructor requirement. These handlers should NEVER execute. If
    the SDK ignored the deny decision and invoked handlers anyway, tools
    would fail silently in production.

BREAKING CHANGE INDICATORS:
    - Tool handlers execute despite returning deny
    - Built-in tools (edit, create) execute after deny
    - Deny decision is logged as allowed
    - Hook invocations missing expected arguments

SDK LOCATIONS TO VERIFY:
    - copilot-sdk/python/copilot/session.py: _handle_hooks_invoke method
    - copilot-sdk/python/copilot/types.py: PreToolUseHookOutput type
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from .conftest import (
    InstrumentedMockSession,
)


class TestDenyHookPreventExecution:
    """
    Tests that verify deny hook prevents tool handler execution.
    """

    @pytest.mark.asyncio
    async def test_deny_hook_prevents_handler_invocation(self, deny_all_hook):
        """
        ASSUMPTION: Deny decision prevents tool handler invocation.

        When preToolUse returns permissionDecision="deny", the registered
        tool handler must NOT be called.
        """
        session = InstrumentedMockSession()
        session.register_pre_tool_use_hook(deny_all_hook)

        # Register a tool handler that records if it was called
        handler_called = False

        def test_handler(args: Any) -> str:
            nonlocal handler_called
            handler_called = True
            return "Handler executed"

        session.register_tool_handler("test_tool", test_handler)

        # Simulate tool call
        result = await session.simulate_tool_call(
            tool_name="test_tool",
            tool_call_id="call_deny_test",
            arguments={"param": "value"},
        )

        # Assert: Handler was NOT called
        assert handler_called is False
        assert result["permission"] == "deny"
        assert result["handler_invoked"] is False
        assert len(session.tool_handler_invocations) == 0

    @pytest.mark.asyncio
    async def test_allow_hook_permits_handler_invocation(self, allow_all_hook):
        """
        CONTROL TEST: Allow decision permits tool handler invocation.

        This validates our test infrastructure correctly simulates the
        SDK's tool execution flow.
        """
        session = InstrumentedMockSession()
        session.register_pre_tool_use_hook(allow_all_hook)

        handler_called = False

        def test_handler(args: Any) -> str:
            nonlocal handler_called
            handler_called = True
            return "Handler executed"

        session.register_tool_handler("test_tool", test_handler)

        result = await session.simulate_tool_call(
            tool_name="test_tool",
            tool_call_id="call_allow_test",
            arguments={"param": "value"},
        )

        # Assert: Handler WAS called
        assert handler_called is True
        assert result["permission"] == "allow"
        assert result["handler_invoked"] is True
        assert len(session.tool_handler_invocations) == 1

    @pytest.mark.asyncio
    async def test_deny_hook_receives_tool_name(self, hook_recorder):
        """
        ASSUMPTION: Hook receives toolName in input data.

        We use toolName for logging and debugging. It must be present.
        """

        def recording_deny_hook(input_data: dict, context: dict) -> dict:
            hook_recorder.record("preToolUse", input_data, {"permissionDecision": "deny"})
            return {"permissionDecision": "deny"}

        session = InstrumentedMockSession()
        session.register_pre_tool_use_hook(recording_deny_hook)

        await session.simulate_tool_call(
            tool_name="my_special_tool",
            tool_call_id="call_name_test",
            arguments={},
        )

        # Assert: Hook received correct tool name
        invocations = hook_recorder.get_by_type("preToolUse")
        assert len(invocations) == 1
        assert invocations[0].input_data["toolName"] == "my_special_tool"

    @pytest.mark.asyncio
    async def test_deny_hook_receives_tool_arguments(self, hook_recorder):
        """
        ASSUMPTION: Hook receives toolArgs in input data.

        We log arguments for debugging. They must be present.
        """

        def recording_deny_hook(input_data: dict, context: dict) -> dict:
            hook_recorder.record("preToolUse", input_data, {"permissionDecision": "deny"})
            return {"permissionDecision": "deny"}

        session = InstrumentedMockSession()
        session.register_pre_tool_use_hook(recording_deny_hook)

        test_args = {"path": "/test/file.py", "content": "# Python code"}

        await session.simulate_tool_call(
            tool_name="write_file",
            tool_call_id="call_args_test",
            arguments=test_args,
        )

        # Assert: Hook received correct arguments
        invocations = hook_recorder.get_by_type("preToolUse")
        assert len(invocations) == 1
        assert invocations[0].input_data["toolArgs"] == test_args


class TestDenyHookForDifferentToolTypes:
    """
    Tests that verify deny hook works for all tool types.
    """

    @pytest.mark.asyncio
    async def test_deny_works_for_user_defined_tools(self, deny_all_hook):
        """
        ASSUMPTION: Deny works for user-defined tools.

        Tools registered via the tools parameter must be deniable.
        """
        session = InstrumentedMockSession()
        session.register_pre_tool_use_hook(deny_all_hook)

        user_tool_called = False

        def user_tool_handler(args: Any) -> str:
            nonlocal user_tool_called
            user_tool_called = True
            return "User tool result"

        session.register_tool_handler("my_custom_tool", user_tool_handler)

        result = await session.simulate_tool_call(
            tool_name="my_custom_tool",
            tool_call_id="call_user_tool",
            arguments={"custom_arg": "value"},
        )

        assert user_tool_called is False
        assert result["permission"] == "deny"

    @pytest.mark.asyncio
    async def test_deny_hook_called_for_each_tool_in_batch(self, hook_recorder):
        """
        ASSUMPTION: Hook is called once per tool in a multi-tool batch.

        When LLM requests multiple tools, each should trigger a separate
        hook invocation (not a single batch call).
        """

        def counting_deny_hook(input_data: dict, context: dict) -> dict:
            hook_recorder.record("preToolUse", input_data, {"permissionDecision": "deny"})
            return {"permissionDecision": "deny"}

        session = InstrumentedMockSession()
        session.register_pre_tool_use_hook(counting_deny_hook)

        # Simulate three tool calls
        tools = [
            ("tool_a", "call_batch_1", {}),
            ("tool_b", "call_batch_2", {"arg": 1}),
            ("tool_c", "call_batch_3", {"arg": 2}),
        ]

        for name, call_id, args in tools:
            await session.simulate_tool_call(
                tool_name=name,
                tool_call_id=call_id,
                arguments=args,
            )

        # Assert: Hook called three times
        invocations = hook_recorder.get_by_type("preToolUse")
        assert len(invocations) == 3

        tool_names = [inv.input_data["toolName"] for inv in invocations]
        assert tool_names == ["tool_a", "tool_b", "tool_c"]


class TestDenyHookReturnValues:
    """
    Tests that verify the hook return value format is correct.
    """

    @pytest.mark.asyncio
    async def test_deny_with_reason(self):
        """
        ASSUMPTION: Deny can include a reason string for logging.

        permissionDecisionReason should be accepted and preserved.
        """
        reason_text = "Tool denied by external orchestrator policy"

        def deny_with_reason(input_data: dict, context: dict) -> dict:
            return {
                "permissionDecision": "deny",
                "permissionDecisionReason": reason_text,
            }

        session = InstrumentedMockSession()
        session.register_pre_tool_use_hook(deny_with_reason)

        # This should not raise - reason should be accepted
        result = await session.simulate_tool_call(
            tool_name="test_tool",
            tool_call_id="call_reason_test",
            arguments={},
        )

        assert result["permission"] == "deny"

    @pytest.mark.asyncio
    async def test_hook_returning_none_defaults_to_allow(self):
        """
        ASSUMPTION: Hook returning None defaults to allow.

        This tests what happens if a hook doesn't return a decision.
        Per SDK behavior, None should be treated as allow.
        """

        def hook_returns_none(input_data: dict, context: dict) -> None:
            return None

        session = InstrumentedMockSession()
        session.register_pre_tool_use_hook(hook_returns_none)

        handler_called = False

        def test_handler(args: Any) -> str:
            nonlocal handler_called
            handler_called = True
            return "result"

        session.register_tool_handler("test_tool", test_handler)

        await session.simulate_tool_call(
            tool_name="test_tool",
            tool_call_id="call_none_test",
            arguments={},
        )

        # None is treated as no hook registered -> allow
        # (permission will be None from the result extraction)
        assert handler_called is True

    @pytest.mark.asyncio
    async def test_hook_can_be_async(self):
        """
        ASSUMPTION: Hook handler can be async function.

        Our deny hook is synchronous, but the SDK should support async.
        This validates the infrastructure handles both.
        """

        async def async_deny_hook(input_data: dict, context: dict) -> dict:
            await asyncio.sleep(0.001)  # Simulate async work
            return {"permissionDecision": "deny"}

        session = InstrumentedMockSession()
        session.register_pre_tool_use_hook(async_deny_hook)

        handler_called = False

        def test_handler(args: Any) -> str:
            nonlocal handler_called
            handler_called = True
            return "result"

        session.register_tool_handler("test_tool", test_handler)

        result = await session.simulate_tool_call(
            tool_name="test_tool",
            tool_call_id="call_async_hook",
            arguments={},
        )

        assert handler_called is False
        assert result["permission"] == "deny"


class TestHookInvocationContext:
    """
    Tests that verify hook receives correct context information.
    """

    @pytest.mark.asyncio
    async def test_hook_receives_session_id_in_context(self, hook_recorder):
        """
        ASSUMPTION: Hook context includes session_id.

        We may use session_id for logging correlation.
        """

        def context_recording_hook(input_data: dict, context: dict) -> dict:
            # Store context for assertion
            hook_recorder.record("preToolUse", {"context": context, **input_data}, None)
            return {"permissionDecision": "deny"}

        session = InstrumentedMockSession(session_id="my-test-session-789")
        session.register_pre_tool_use_hook(context_recording_hook)

        await session.simulate_tool_call(
            tool_name="test_tool",
            tool_call_id="call_context_test",
            arguments={},
        )

        invocations = hook_recorder.get_by_type("preToolUse")
        assert len(invocations) == 1

        context = invocations[0].input_data.get("context", {})
        assert context.get("session_id") == "my-test-session-789"


class TestEdgeCases:
    """
    Tests for edge cases in hook behavior.
    """

    @pytest.mark.asyncio
    async def test_deny_with_empty_arguments(self, deny_all_hook):
        """
        ASSUMPTION: Deny works even when tool has no arguments.

        Some tools take no parameters. Deny should still work.
        """
        session = InstrumentedMockSession()
        session.register_pre_tool_use_hook(deny_all_hook)

        handler_called = False

        def no_args_handler(args: Any) -> str:
            nonlocal handler_called
            handler_called = True
            return "result"

        session.register_tool_handler("no_args_tool", no_args_handler)

        result = await session.simulate_tool_call(
            tool_name="no_args_tool",
            tool_call_id="call_empty_args",
            arguments={},
        )

        assert handler_called is False
        assert result["permission"] == "deny"

    @pytest.mark.asyncio
    async def test_deny_with_complex_nested_arguments(self, deny_all_hook, hook_recorder):
        """
        ASSUMPTION: Deny works with complex nested argument structures.

        Tool arguments can be deeply nested. Hook should receive full structure.
        """

        def recording_deny(input_data: dict, context: dict) -> dict:
            hook_recorder.record("preToolUse", input_data, None)
            return {"permissionDecision": "deny"}

        session = InstrumentedMockSession()
        session.register_pre_tool_use_hook(recording_deny)

        complex_args = {
            "config": {
                "nested": {
                    "deeply": {
                        "value": [1, 2, 3],
                    },
                },
            },
            "items": [{"id": 1}, {"id": 2}],
        }

        await session.simulate_tool_call(
            tool_name="complex_tool",
            tool_call_id="call_complex",
            arguments=complex_args,
        )

        invocations = hook_recorder.get_by_type("preToolUse")
        received_args = invocations[0].input_data["toolArgs"]

        # Arguments should be preserved exactly
        assert received_args == complex_args
        assert received_args["config"]["nested"]["deeply"]["value"] == [1, 2, 3]
