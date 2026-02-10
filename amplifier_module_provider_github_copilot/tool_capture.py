# =============================================================================
# ARCHITECTURE NOTE: External Orchestration via Deny + Destroy
# =============================================================================
#
# CONTEXT:
# The Copilot SDK is designed for agentic workflows where the CLI server
# executes tools internally and returns final results. This provider has a
# different requirement: return tool calls to an external orchestrator
# (Amplifier) which handles execution, context management, and agent delegation.
#
# APPROACH:
# Force "capture-only" mode using the SDK's extensibility points:
#   1. Register tools with no-op handlers (LLM sees structured definitions)
#   2. Deny ALL execution via preToolUse hook (prevents CLI from running tools)
#   3. Capture tool_requests from ASSISTANT_MESSAGE streaming events
#   4. Destroy session after capture (prevents CLI retry with built-in tools)
#
# WHY SESSION DESTROY IS REQUIRED:
# Without immediate session destruction after tool capture, the CLI's internal
# agent loop retries with its built-in tools (e.g., 'edit' instead of our
# 'write_file'), bypassing the external orchestrator entirely. Empirically
# validated: session IDs 497bbab7, 2a1fe04a showed this bypass behavior.
#
# FRAGILITY NOTES:
# This pattern depends on SDK internal behavior that is not part of the
# public API contract:
#   - ASSISTANT_MESSAGE events fire before preToolUse hook
#   - preToolUse deny prevents handler invocation but allows event capture
#   - Session destroy reliably terminates the CLI's agent loop
# Changes to SDK event ordering or hook semantics could break this pattern.
#
# FUTURE SIMPLIFICATION:
# If the SDK adds an official "capture-only" or "external orchestration" mode
# that returns tool calls without attempting execution, this workaround can
# be replaced with that cleaner approach.
#
# =============================================================================

"""
Tool bridge for dumb-pipe pattern: Deny + Destroy.

This module bridges Amplifier's tool system with the Copilot SDK's
tool calling mechanism. It converts Amplifier ToolSpec objects to SDK
Tool objects and provides a preToolUse deny hook to prevent the CLI
from executing any tools internally.

Architecture (empirically validated via test_dumb_pipe_strategies.py):

1. Convert Amplifier ToolSpec → SDK Tool with no-op handler
2. Register preToolUse deny hook as safety belt
3. Register tools with SDK session → LLM sees structured definitions
4. Provider's event handler captures ASSISTANT_MESSAGE.tool_requests
5. Provider returns ToolCall objects to Amplifier's orchestrator
6. Session is destroyed by context manager (prevents CLI retry loop)

Why this works (Strategy 5: Deny + Destroy):
- Event capture from ASSISTANT_MESSAGE provides structured tool data
  (name, tool_call_id, arguments) BEFORE the CLI attempts execution
- preToolUse deny prevents handler execution during the gap between
  event capture and session destroy
- Session destroy prevents the CLI from retrying with built-in tools
  (empirically: without destroy, CLI falls back to built-in 'create')

What this replaces:
- Previous capture-and-abort pattern used handler blocking + abort()
- That was vulnerable to CLI bypass (built-in tools ran internally)
- The deny + destroy pattern eliminates both bypass and race conditions
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _noop_tool_handler(args: Any) -> str:
    """
    No-op handler for SDK Tool objects.

    Execution is prevented by the preToolUse deny hook. This handler
    exists only because the SDK Tool constructor requires one. If
    somehow called (deny hook failed AND destroy was slow), it returns
    a harmless error result without side effects.

    Args:
        args: Tool arguments (ignored)

    Returns:
        JSON error message
    """
    return json.dumps({"error": "Tool execution denied by provider policy"})


def convert_tools_for_sdk(tool_specs: list[Any]) -> list[Any]:
    """
    Convert Amplifier ToolSpec objects to SDK Tool objects.

    Each SDK Tool uses a no-op handler since actual tool execution
    is prevented by the preToolUse deny hook and handled by Amplifier's
    orchestrator after this provider returns.

    Deduplicates by tool name (keeps first occurrence) since the
    Copilot API rejects duplicate tool names with a 400 error.

    Args:
        tool_specs: List of ToolSpec objects from ChatRequest.tools.
            Each has: name (str), description (str | None), parameters (dict).

    Returns:
        List of SDK Tool objects ready for session registration.
    """
    from copilot.types import Tool

    sdk_tools: list[Tool] = []
    seen_names: set[str] = set()

    for spec in tool_specs:
        # Extract fields from ToolSpec (Pydantic model or dict)
        if hasattr(spec, "name"):
            name = spec.name
            description = getattr(spec, "description", "") or ""
            parameters = getattr(spec, "parameters", None)
        elif isinstance(spec, dict):
            name = spec.get("name", "")
            description = spec.get("description", "")
            parameters = spec.get("parameters")
        else:
            logger.warning(f"[TOOL_BRIDGE] Skipping unknown tool spec type: {type(spec)}")
            continue

        if not name:
            continue

        # Deduplicate: Copilot API returns 400 if tool names are not unique
        if name in seen_names:
            logger.debug(f"[TOOL_BRIDGE] Skipping duplicate tool: {name}")
            continue
        seen_names.add(name)

        sdk_tools.append(
            Tool(
                name=name,
                description=description,
                handler=_noop_tool_handler,
                parameters=parameters,
            )
        )

    logger.debug(
        f"[TOOL_BRIDGE] Converted {len(sdk_tools)} tool(s) for SDK: {[t.name for t in sdk_tools]}"
    )

    return sdk_tools


def make_deny_all_hook() -> dict[str, Any]:
    """
    Create a preToolUse hook that denies all tool execution.

    This hook is a safety belt: it prevents the CLI from executing
    any tool handler during the gap between the ASSISTANT_MESSAGE
    event (where we capture tool_requests) and session destroy.

    The hook returns a 'deny' decision for every tool, regardless
    of whether it's a user-defined tool or a CLI built-in.

    CRITICAL: The permissionDecisionReason is shown to the model/user.
    We use a minimal reason and suppressOutput to prevent the denial
    from polluting the conversation context and causing the model to
    stop trying to use tools.

    Returns:
        Dict with 'on_pre_tool_use' key for session config hooks.

    Example:
        >>> hooks = make_deny_all_hook()
        >>> session = await client.create_session({
        ...     "model": "claude-sonnet-4",
        ...     "tools": sdk_tools,
        ...     "hooks": hooks,
        ... })
    """

    def deny_hook(input_data: dict[str, Any], context: Any) -> dict[str, Any]:
        """Deny ALL tools — dumb-pipe mode.

        CRITICAL: The permissionDecisionReason is shown to the model/user.
        We MUST NOT include explanatory text like "dumb-pipe mode" because
        the model learns from denial reasons and will stop trying tools.

        Solution: Minimal reason + suppressOutput to prevent the denial
        message from polluting the conversation context.
        """
        tool_name = input_data.get("toolName", "unknown")
        logger.debug(f"[TOOL_BRIDGE] preToolUse deny: {tool_name}")
        return {
            "permissionDecision": "deny",
            # Minimal reason - model shouldn't learn tools are "blocked"
            "permissionDecisionReason": "Processing",
            # Suppress output to prevent denial from reaching conversation
            "suppressOutput": True,
        }

    return {"on_pre_tool_use": deny_hook}
