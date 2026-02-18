"""
Regression Test: 305-Turn SDK Loop — The Bug-Hunter Forensic Scenario.

=======================================================================
THIS IS THE EXACT SCENARIO THAT CAUSED THE ARCHITECTURE CHANGE.
=======================================================================

Original Failure (Session a1a0af17):
  - Prompt: "Use the bug-hunter agent to check if there are any obvious
    issues in the models.py file."
  - Result: 305 SDK turns, 607 tool calls (303 pairs × delegate + report_intent),
    20 minutes of spinning, 303 bug-hunter agent spawns.
  - Root Cause: SDK's preToolUse deny returned error → LLM retried →
    infinite loop accumulating tool_calls.

This test REPLAYS that exact scenario with the new SDK Driver
architecture and validates:
  1. Tool calls ARE captured (bug-hunter delegation works)
  2. Turn count stays ≤ 3 (not 305)
  3. Tool call count stays ≤ 10 (not 607)
  4. Completes in < 60 seconds (not 20 minutes)
  5. Session is aborted after first-turn capture

Prerequisites:
  - Copilot CLI in PATH
  - Valid GitHub Copilot authentication
  - Network access

Run:
  RUN_LIVE_TESTS=1 python -m pytest tests/integration/test_regression_305_loop.py -v -s

Architecture:
  - SDK Driver with first-turn capture
  - Deny+Destroy pattern prevents runaway loops
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

logger = logging.getLogger(__name__)

# Skip unless explicitly enabled — these make real API calls
pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_LIVE_TESTS"),
    reason="Live tests disabled. Set RUN_LIVE_TESTS=1 to run.",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Exact Amplifier Tool Definitions (from production)
# ═══════════════════════════════════════════════════════════════════════════════


def build_bug_hunter_tool_catalog() -> list[Any]:
    """
    Build the EXACT tool catalog that triggers the 305-turn loop.

    This includes the delegate tool with full agent descriptions,
    bash, todo, read_file, grep, and glob — exactly as the Amplifier
    orchestrator constructs them from foundation bundle tool modules.
    """
    from amplifier_core.message_models import ToolSpec

    delegate_tool = ToolSpec(
        name="delegate",
        description=(
            "Spawn a specialized agent to handle tasks autonomously.\n\n"
            "CRITICAL: Delegation is your PRIMARY operating mode, not an optimization.\n\n"
            "ALWAYS use this tool when:\n"
            "- Task requires reading more than 2 files\n"
            "- Task requires exploration or investigation\n"
            "- Task matches any agent's specialty\n"
            "- Task would benefit from specialized context or tools\n\n"
            "Available agents:\n"
            "  - foundation:bug-hunter: Specialized debugging expert\n"
            "  - foundation:explorer: Deep codebase exploration specialist\n"
            "  - foundation:zen-architect: Architecture design specialist\n"
            "  - foundation:modular-builder: Implementation specialist\n"
            "  - foundation:test-coverage: Test creation specialist\n"
            "  - foundation:git-ops: Git operations specialist"
        ),
        parameters={
            "type": "object",
            "properties": {
                "agent": {
                    "type": "string",
                    "description": "Agent to delegate to (e.g., 'foundation:bug-hunter')",
                },
                "instruction": {
                    "type": "string",
                    "description": "Clear instruction for the agent",
                },
                "context_depth": {
                    "type": "string",
                    "enum": ["none", "recent", "all"],
                },
            },
            "required": ["instruction"],
        },
    )

    bash_tool = ToolSpec(
        name="bash",
        description="Execute shell commands directly. Use for quick, simple commands only.",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
            },
            "required": ["command"],
        },
    )

    todo_tool = ToolSpec(
        name="todo",
        description="Self-accountability tracking for complex tasks.",
        parameters={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["create", "update", "list"]},
            },
            "required": ["action"],
        },
    )

    read_file_tool = ToolSpec(
        name="read_file",
        description="Read file contents. Supports line ranges with offset/limit.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to read"},
            },
            "required": ["path"],
        },
    )

    grep_tool = ToolSpec(
        name="grep",
        description="Search for text patterns in files.",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Search pattern"},
                "path": {"type": "string", "description": "Path to search in"},
            },
            "required": ["pattern"],
        },
    )

    glob_tool = ToolSpec(
        name="glob",
        description="Find files matching a glob pattern.",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern"},
            },
            "required": ["pattern"],
        },
    )

    return [delegate_tool, bash_tool, todo_tool, read_file_tool, grep_tool, glob_tool]


def build_mock_request(prompt: str, tools: list[Any]) -> Mock:
    """Create a mock ChatRequest matching Amplifier's structure."""
    request = Mock()
    request.messages = [
        Mock(
            role="system",
            content=(
                "You are Amplifier, a modular AI agent framework. "
                "You have access to powerful tools including delegation to specialist agents. "
                "ALWAYS delegate tasks to the most appropriate specialist agent. "
                "NEVER try to do investigation or complex analysis yourself."
            ),
        ),
        Mock(role="user", content=prompt),
    ]
    request.tools = tools
    request.stream = None

    # Messages need model_dump for converter compatibility
    for msg in request.messages:
        msg.model_dump = lambda m=msg: {"role": m.role, "content": m.content}

    return request


# ═══════════════════════════════════════════════════════════════════════════════
# Forensic Analysis Harness
# ═══════════════════════════════════════════════════════════════════════════════


class ForensicCollector:
    """
    Collect SDK Driver observability events for forensic analysis.

    Captures all sdk_driver:* events emitted during a complete() call
    for post-mortem comparison against the original 305-turn incident.
    """

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.start_time: float = 0.0

    def reset(self) -> None:
        self.events.clear()
        self.start_time = time.time()

    async def capture_event(self, event_name: str, data: dict[str, Any]) -> None:
        """Mock hooks.emit that captures events."""
        elapsed = time.time() - self.start_time if self.start_time else 0.0
        self.events.append(
            {
                "event": event_name,
                "data": data,
                "elapsed_s": round(elapsed, 3),
            }
        )

    # ── Analysis Properties ─────────────────────────────────────────

    @property
    def turn_start_events(self) -> list[dict]:
        return [e for e in self.events if e["event"] == "sdk_driver:sdk:turn_start"]

    @property
    def capture_events(self) -> list[dict]:
        return [e for e in self.events if e["event"] == "sdk_driver:sdk:capture_complete"]

    @property
    def abort_events(self) -> list[dict]:
        return [e for e in self.events if e["event"] == "sdk_driver:sdk:abort_requested"]

    @property
    def circuit_breaker_events(self) -> list[dict]:
        return [e for e in self.events if e["event"] == "sdk_driver:sdk:circuit_breaker_trip"]

    @property
    def turn_count(self) -> int:
        return len(self.turn_start_events)

    @property
    def total_tools_captured(self) -> int:
        for e in self.capture_events:
            return e["data"].get("tool_count", 0)
        return 0

    def print_forensic_report(self, scenario_name: str) -> None:
        """Print detailed forensic analysis report."""
        print(f"\n{'=' * 72}")
        print(f"FORENSIC REPORT: {scenario_name}")
        print(f"{'=' * 72}")

        last_elapsed = self.events[-1]["elapsed_s"] if self.events else 0
        print(f"  Total events:     {len(self.events)}")
        print(f"  Total elapsed:    {last_elapsed:.2f}s")
        print(f"  Turn count:       {self.turn_count}")
        print(f"  Tools captured:   {self.total_tools_captured}")
        print(f"  Aborts requested: {len(self.abort_events)}")
        print(f"  CB trips:         {len(self.circuit_breaker_events)}")

        print("\n  Event Timeline:")
        for ev in self.events:
            data_summary = ""
            d = ev["data"]
            if "turn" in d:
                data_summary += f" turn={d['turn']}"
            if "tool_count" in d:
                data_summary += f" tools={d['tool_count']}"
            if "reason" in d:
                data_summary += f" reason={d['reason']}"
            if "tools" in d:
                data_summary += f" names={d['tools']}"
            print(f"    [{ev['elapsed_s']:>7.3f}s] {ev['event']}{data_summary}")

        # Comparison against incident
        print("\n  ── Comparison vs. Original Incident ──")
        print(f"  {'Metric':<25} {'Original':>12} {'Now':>12} {'Status':>10}")
        print(f"  {'-' * 60}")

        turn_status = "PASS" if self.turn_count <= 3 else "FAIL"
        print(f"  {'Turn count':<25} {'305':>12} {self.turn_count:>12} {turn_status:>10}")

        tools_status = "PASS" if self.total_tools_captured <= 10 else "FAIL"
        print(
            f"  {'Tool calls captured':<25} {'607':>12} {self.total_tools_captured:>12} {tools_status:>10}"
        )

        time_status = "PASS" if last_elapsed < 60 else "FAIL"
        print(f"  {'Duration (seconds)':<25} {'~1200':>12} {last_elapsed:>12.1f} {time_status:>10}")

        abort_status = "PASS" if len(self.abort_events) > 0 else "WARN"
        print(
            f"  {'Abort requested':<25} {'No':>12} {'Yes' if self.abort_events else 'No':>12} {abort_status:>10}"
        )

        print(f"{'=' * 72}")


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def forensic_collector() -> ForensicCollector:
    """Fresh forensic event collector."""
    return ForensicCollector()


@pytest.fixture
async def live_provider(forensic_collector: ForensicCollector):
    """
    Create provider with forensic event collection.

    Uses claude-opus-4.5 (same model as the original incident).
    """
    from amplifier_module_provider_github_copilot import CopilotSdkProvider

    coordinator = Mock()
    coordinator.hooks = Mock()
    coordinator.hooks.emit = AsyncMock(side_effect=forensic_collector.capture_event)

    provider = CopilotSdkProvider(
        api_key=None,
        config={
            "model": "claude-opus-4.5",
            "timeout": 120,
            "debug": True,
            "use_streaming": True,
            "sdk_max_turns": 3,  # Guard: trip at 3 turns max
        },
        coordinator=coordinator,
    )
    yield provider
    await provider.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Test: THE Bug-Hunter Scenario (305-Turn Regression)
# ═══════════════════════════════════════════════════════════════════════════════


class TestBugHunter305Regression:
    """
    Replay the exact scenario that caused the 305-turn incident.

    Original prompt: "Use the bug-hunter agent to check if there are any
    obvious issues in the models.py file."

    This is the SINGLE MOST CRITICAL test in the entire test suite.
    If this test fails, the SDK Driver architecture has regressed.
    """

    THE_PROMPT = (
        "Use the bug-hunter agent to check if there are any obvious issues in the models.py file."
    )

    @pytest.mark.asyncio
    async def test_bug_hunter_delegation_completes_fast(
        self,
        live_provider,
        forensic_collector: ForensicCollector,
    ) -> None:
        """
        CRITICAL REGRESSION TEST: The exact prompt that triggered 305 turns.

        Acceptance criteria (all must pass):
        - [x] Completes in < 60 seconds (original: ~1200s)
        - [x] Turn count ≤ 3 (original: 305)
        - [x] Tool calls ≤ 10 (original: 607)
        - [x] Contains 'delegate' tool call to 'foundation:bug-hunter'
        - [x] Has valid tool_call_id
        - [x] Abort was requested after capture
        """
        tools = build_bug_hunter_tool_catalog()
        request = build_mock_request(self.THE_PROMPT, tools)

        forensic_collector.reset()
        start = time.time()
        response = await live_provider.complete(request)
        elapsed = time.time() - start

        # Print forensic report regardless of pass/fail
        forensic_collector.print_forensic_report("Bug-Hunter 305-Turn Regression")

        logger.info(
            f"Bug-hunter regression: elapsed={elapsed:.2f}s, "
            f"finish_reason={response.finish_reason}, "
            f"tool_calls={len(response.tool_calls) if response.tool_calls else 0}"
        )

        # ── CRITICAL ASSERTIONS ─────────────────────────────────────

        # 1. Time: Must complete in < 60s (original took ~1200s)
        assert elapsed < 60, (
            f"Request took {elapsed:.1f}s — REGRESSION ALERT! "
            f"Original incident: ~1200s. SDK Driver should abort quickly."
        )

        # 2. Turn count: Must be ≤ 3 (original was 305)
        assert forensic_collector.turn_count <= 3, (
            f"Turn count: {forensic_collector.turn_count} — REGRESSION ALERT! "
            f"Original incident: 305 turns. Circuit breaker should trip."
        )

        # 3. Tool calls present
        assert response.tool_calls, (
            "No tool calls captured — bug-hunter delegation should produce "
            "at least a 'delegate' tool call."
        )

        # 4. Tool call count: Must be ≤ 10 (original was 607)
        tc_count = len(response.tool_calls)
        assert tc_count <= 10, (
            f"Tool calls: {tc_count} — REGRESSION ALERT! "
            f"Original incident: 607. First-turn-only capture should limit this."
        )

        # 5. Must contain 'delegate' tool call
        tool_names = [tc.name for tc in response.tool_calls]
        assert "delegate" in tool_names, (
            f"Expected 'delegate' tool call, got: {tool_names}. "
            f"LLM should delegate to bug-hunter for code investigation."
        )

        # 6. Delegate should target bug-hunter
        delegate_calls = [tc for tc in response.tool_calls if tc.name == "delegate"]
        for dc in delegate_calls:
            args = dc.arguments if isinstance(dc.arguments, dict) else {}
            agent = args.get("agent", "")
            assert "bug-hunter" in agent, (
                f"Delegate target: '{agent}' — expected 'foundation:bug-hunter'. "
                f"This was the exact agent in the original incident."
            )

        # 7. Each tool call must have valid ID
        for tc in response.tool_calls:
            assert tc.id, f"Tool call '{tc.name}' missing id"

        # 8. finish_reason must be tool_use
        assert response.finish_reason == "tool_use", (
            f"Expected finish_reason='tool_use', got '{response.finish_reason}'"
        )

        # 9. Abort was requested (SDK Driver stopped the loop)
        assert len(forensic_collector.abort_events) > 0, (
            "No abort event emitted — SDK Driver should abort after first-turn capture"
        )

    @pytest.mark.asyncio
    async def test_zen_architect_delegation(
        self,
        live_provider,
        forensic_collector: ForensicCollector,
    ) -> None:
        """
        Second regression scenario: architecture review delegation.

        A different prompt that should also trigger delegation without looping.
        This validates the fix works for various delegation patterns, not just
        the exact bug-hunter prompt.
        """
        prompt = (
            "We need to redesign the data pipeline architecture. Analyze the "
            "current codebase structure and propose a modular event-driven design "
            "with proper separation of concerns."
        )
        tools = build_bug_hunter_tool_catalog()
        request = build_mock_request(prompt, tools)

        forensic_collector.reset()
        start = time.time()
        response = await live_provider.complete(request)
        elapsed = time.time() - start

        forensic_collector.print_forensic_report("Zen-Architect Delegation")

        # Must complete quickly
        assert elapsed < 60, f"Took {elapsed:.1f}s — potential loop issue"

        # Must have tool calls
        if response.tool_calls:
            assert len(response.tool_calls) <= 10, (
                f"Got {len(response.tool_calls)} tool calls — possible accumulation"
            )
            # If delegate was called, good
            tool_names = [tc.name for tc in response.tool_calls]
            logger.info(f"Architect tools: {tool_names}")
        else:
            # Text response is acceptable for architecture questions
            logger.info("Architect scenario: text response (no delegation)")

        # Turn count must be reasonable
        assert forensic_collector.turn_count <= 3

    @pytest.mark.asyncio
    async def test_simple_text_no_loop(
        self,
        live_provider,
        forensic_collector: ForensicCollector,
    ) -> None:
        """
        Baseline: Simple prompt should NOT trigger tools or loops.

        Validates that non-tool-triggering prompts complete cleanly
        as a sanity check that the SDK Driver doesn't interfere with
        normal operation.
        """
        prompt = "What is 2 + 2? Reply with just the number."
        tools = build_bug_hunter_tool_catalog()
        request = build_mock_request(prompt, tools)

        forensic_collector.reset()
        start = time.time()
        response = await live_provider.complete(request)
        elapsed = time.time() - start

        forensic_collector.print_forensic_report("Simple Text (Baseline)")

        assert elapsed < 30, f"Simple prompt took {elapsed:.1f}s — too slow"
        assert response.content, "Expected text content"

        # May or may not have tool calls (model decides)
        if not response.tool_calls:
            assert response.finish_reason == "end_turn"


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Circuit Breaker Stress Test
# ═══════════════════════════════════════════════════════════════════════════════


class TestCircuitBreakerStress:
    """
    Validate circuit breaker behavior under adversarial conditions.

    Uses a stricter max_turns=1 to force the circuit breaker to trip
    even if first-turn capture doesn't succeed.
    """

    @pytest.fixture
    async def strict_provider(self, forensic_collector: ForensicCollector):
        from amplifier_module_provider_github_copilot import CopilotSdkProvider

        coordinator = Mock()
        coordinator.hooks = Mock()
        coordinator.hooks.emit = AsyncMock(side_effect=forensic_collector.capture_event)

        provider = CopilotSdkProvider(
            api_key=None,
            config={
                "model": "claude-sonnet-4",
                "timeout": 120,
                "debug": True,
                "use_streaming": True,
                "sdk_max_turns": 1,
            },
            coordinator=coordinator,
        )
        yield provider
        await provider.close()

    @pytest.mark.asyncio
    async def test_circuit_breaker_prevents_runaway(
        self,
        strict_provider,
        forensic_collector: ForensicCollector,
    ) -> None:
        """
        With max_turns=1, circuit breaker should trip immediately on second turn.

        This validates that even if the SDK retries (which it does when
        tools are denied), the circuit breaker catches it.
        """
        from amplifier_module_provider_github_copilot.exceptions import CopilotSdkLoopError

        tools = build_bug_hunter_tool_catalog()
        prompt = "Read the file README.md and summarize its contents"
        request = build_mock_request(prompt, tools)

        forensic_collector.reset()
        start = time.time()

        try:
            response = await strict_provider.complete(request)
            elapsed = time.time() - start

            forensic_collector.print_forensic_report("Circuit Breaker Strict")

            # If we got a response, first-turn capture succeeded before CB trip
            logger.info(
                f"Got response before CB trip: elapsed={elapsed:.2f}s, "
                f"tool_calls={len(response.tool_calls) if response.tool_calls else 0}"
            )
            if response.tool_calls:
                assert len(response.tool_calls) <= 5

        except CopilotSdkLoopError as e:
            elapsed = time.time() - start
            forensic_collector.print_forensic_report("Circuit Breaker Strict (Tripped)")

            logger.info(f"CB tripped as expected in {elapsed:.2f}s: {e}")
            assert e.max_turns == 1
            assert elapsed < 60
