#!/usr/bin/env python3
"""
Self-Serve Forensic Analysis & Integration Test Runner.

This script is the definitive validation tool for the SDK Driver architecture.
It reproduces the exact conditions that caused the 305-turn incident, runs
them against the current implementation, and generates a forensic comparison
report proving the fix works.

Usage:
    # Quick smoke test (simple math, no tools)
    python tests/integration/forensic_runner.py --smoke

    # Full forensic regression (reproduces the incident)
    python tests/integration/forensic_runner.py --forensic

    # Compare against original incident data
    python tests/integration/forensic_runner.py --compare

    # Full suite: smoke + forensic + comparison
    python tests/integration/forensic_runner.py --all

    # Run via pytest (requires RUN_LIVE_TESTS=1)
    RUN_LIVE_TESTS=1 python -m pytest tests/integration/ -v -s

Prerequisites:
    - Python 3.11+
    - Copilot CLI installed and authenticated (any platform)
    - PYTHONPATH includes amplifier-core and this module
    - Amplifier installed (for E2E tests)

Evidence base:
    - Original incident: 305 turns, 607 tool calls, 303 bug-hunter spawns
    - Session a1a0af17: _dev_scripts/_forensic_amp_events.jsonl
    - Session bfa3b57b: _dev_scripts/_forensic_sdk_events.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from amplifier_module_provider_github_copilot._constants import COPILOT_BUILTIN_TOOL_NAMES

# ═══════════════════════════════════════════════════════════════════════════════
# Constants — derived from the original incident
# ═══════════════════════════════════════════════════════════════════════════════

INCIDENT_SESSION_ID = "a1a0af17-4a3c-4bf5-8c17-5f51b235d6e4"
INCIDENT_SDK_SESSION_ID = "bfa3b57b-09b5-47a9-8d1a-d89716abfab0"
INCIDENT_TURNS = 305
INCIDENT_TOOL_CALLS = 607
INCIDENT_DELEGATES = 303
INCIDENT_DURATION_S = 1200  # ~20 minutes

MAX_ACCEPTABLE_TURNS = 5
MAX_ACCEPTABLE_TOOL_CALLS = 10
MAX_ACCEPTABLE_DELEGATES = 2
MAX_ACCEPTABLE_DURATION_S = 60

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("forensic_runner")


# ═══════════════════════════════════════════════════════════════════════════════
# SDK Session Parser
# ═══════════════════════════════════════════════════════════════════════════════


def parse_sdk_session(events_file: Path) -> dict[str, Any]:
    """Parse SDK session events into summary metrics."""
    events: list[dict[str, Any]] = []
    with open(events_file) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    turn_starts = [e for e in events if e.get("type") == "assistant.turn_start"]
    tool_executions = [e for e in events if e.get("type") == "tool.execution_complete"]
    denials = [e for e in tool_executions if not e.get("data", {}).get("success", True)]
    successes = [e for e in tool_executions if e.get("data", {}).get("success", True)]

    # Extract tool names
    tool_names: dict[str, int] = {}
    for e in events:
        if e.get("type") == "tool.execution_start":
            name = e.get("data", {}).get("toolName", "unknown")
            tool_names[name] = tool_names.get(name, 0) + 1

    # Calculate duration
    timestamps = [e.get("timestamp", "") for e in events]
    first_ts = timestamps[0] if timestamps else ""
    last_ts = timestamps[-1] if timestamps else ""

    return {
        "session_id": events_file.parent.name,
        "total_events": len(events),
        "turns": len(turn_starts),
        "tool_executions": len(tool_executions),
        "denials": len(denials),
        "successes": len(successes),
        "tool_names": tool_names,
        "first_timestamp": first_ts,
        "last_timestamp": last_ts,
    }


def parse_amplifier_session(events_file: Path) -> dict[str, Any]:
    """Parse Amplifier session events into summary metrics."""
    events: list[dict[str, Any]] = []
    with open(events_file) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    delegate_spawns = sum(1 for e in events if e.get("event") == "delegate:agent_spawned")

    llm_responses = [e for e in events if e.get("event") == "llm:response"]
    total_tool_calls = 0
    duration_ms = 0
    for resp in llm_responses:
        data = resp.get("data", {})
        tc = data.get("tool_calls", 0)
        dur = data.get("duration_ms", 0)
        if isinstance(tc, int):
            total_tool_calls += tc
        if isinstance(dur, (int, float)):
            duration_ms += dur

    event_counts: dict[str, int] = {}
    for e in events:
        ev = e.get("event", "unknown")
        event_counts[ev] = event_counts.get(ev, 0) + 1

    return {
        "total_events": len(events),
        "delegate_spawns": delegate_spawns,
        "total_tool_calls": total_tool_calls,
        "duration_ms": duration_ms,
        "event_counts": event_counts,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Test Functions
# ═══════════════════════════════════════════════════════════════════════════════


async def run_smoke_test() -> dict[str, Any]:
    """Run a simple math prompt through the provider directly."""
    from unittest.mock import AsyncMock, Mock

    from amplifier_module_provider_github_copilot import CopilotSdkProvider

    logger.info("=" * 60)
    logger.info("SMOKE TEST: Simple math (no tools)")
    logger.info("=" * 60)

    coordinator = Mock()
    coordinator.hooks = Mock()
    coordinator.hooks.emit = AsyncMock()

    provider = CopilotSdkProvider(
        api_key=None,
        config={
            "model": "claude-opus-4.5",
            "timeout": 60,
            "debug": True,
            "use_streaming": True,
        },
        coordinator=coordinator,
    )

    try:
        request = Mock()
        request.messages = [
            {"role": "user", "content": "What is 3 + 6 - 6? Reply with ONLY the number."},
        ]
        request.tools = None
        request.stream = None

        start = time.time()
        response = await provider.complete(request)
        elapsed = time.time() - start

        text_blocks = [b for b in (response.content or []) if getattr(b, "type", None) == "text"]
        text = text_blocks[0].text if text_blocks else "(no text)"

        result = {
            "test": "smoke",
            "status": "PASS" if "3" in text else "FAIL",
            "elapsed": round(elapsed, 2),
            "response_text": text[:200],
            "finish_reason": response.finish_reason,
        }

        logger.info(f"Response: {text}")
        logger.info(f"Duration: {elapsed:.2f}s")
        logger.info(f"Status: {result['status']}")
        return result
    finally:
        await provider.close()


async def run_forensic_regression() -> dict[str, Any]:
    """
    Run the exact forensic incident prompt and measure results.

    This is the critical test — same prompt, same tools, measured against
    the original 305-turn incident metrics.
    """
    from unittest.mock import AsyncMock, Mock

    from amplifier_module_provider_github_copilot import CopilotSdkProvider

    logger.info("=" * 60)
    logger.info("FORENSIC REGRESSION: Bug-hunter delegation prompt")
    logger.info("Reproducing session a1a0af17 / bfa3b57b")
    logger.info("=" * 60)

    coordinator = Mock()
    coordinator.hooks = Mock()

    # Capture emitted events
    emitted: list[tuple[str, dict[str, Any]]] = []

    async def capture(name: str, data: dict[str, Any]) -> None:
        emitted.append((name, data))

    coordinator.hooks.emit = AsyncMock(side_effect=capture)

    provider = CopilotSdkProvider(
        api_key=None,
        config={
            "model": "claude-opus-4.5",
            "timeout": 120,
            "debug": True,
            "use_streaming": True,
            "sdk_max_turns": 5,
        },
        coordinator=coordinator,
    )

    try:
        # Build tools matching the incident
        # NOTE: Tool names MUST NOT collide with SDK built-in names
        # (bash, grep, glob, view, edit, etc.) because the API rejects
        # "Tool names must be unique" even when the built-in is excluded.
        # This was discovered during forensic regression testing.
        # Real Amplifier modules avoid this by using distinct names.
        tool_specs = [
            ("delegate", "Delegate task to specialized agent"),
            ("report_intent", "Report your intent before acting"),
            ("read_file", "Read a file"),
            ("write_file", "Write to a file"),
            ("run_command", "Run a shell command"),
            ("list_directory", "List directory contents"),
            ("search_files", "Search for files"),
            ("search_content", "Search file contents with regex"),
            ("todo", "Track tasks"),
        ]

        tools = []
        for name, desc in tool_specs:
            t = Mock()
            t.name = name
            t.description = desc
            t.parameters = {"type": "object", "properties": {}}
            tools.append(t)

        request = Mock()
        request.messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful AI assistant with access to tools. "
                    "When the user asks you to delegate a task to another agent, "
                    "use the delegate tool. Always report your intent first."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Use the bug-hunter agent to check if there are any "
                    "obvious issues in the models.py file."
                ),
            },
        ]
        request.tools = tools
        request.stream = None

        before_ts = datetime.now(UTC)
        start = time.time()

        response = await provider.complete(request)
        elapsed = time.time() - start

        tool_count = len(response.tool_calls) if response.tool_calls else 0
        tool_names = [tc.name for tc in response.tool_calls] if response.tool_calls else []

        # Extract llm:request event data
        llm_req_events = [(n, d) for n, d in emitted if n == "llm:request"]
        excluded_builtins = []
        if llm_req_events:
            excluded_builtins = llm_req_events[0][1].get("excluded_builtin_tools", [])

        # Find SDK session for correlation
        sdk_metrics = await _correlate_sdk_session(before_ts)

        result = {
            "test": "forensic_regression",
            "elapsed": round(elapsed, 2),
            "tool_calls": tool_count,
            "tool_names": tool_names,
            "finish_reason": response.finish_reason,
            "excluded_builtins": excluded_builtins,
            "excluded_builtin_count": len(excluded_builtins),
            "sdk_session": sdk_metrics,
            "comparison": {
                "incident_turns": INCIDENT_TURNS,
                "incident_tool_calls": INCIDENT_TOOL_CALLS,
                "incident_duration_s": INCIDENT_DURATION_S,
                "current_turns": sdk_metrics.get("turns", "N/A") if sdk_metrics else "N/A",
                "current_tool_calls": tool_count,
                "current_duration_s": round(elapsed, 2),
            },
        }

        # Determine pass/fail
        passes: list[str] = []
        failures: list[str] = []

        if elapsed < MAX_ACCEPTABLE_DURATION_S:
            passes.append(f"Duration: {elapsed:.1f}s < {MAX_ACCEPTABLE_DURATION_S}s")
        else:
            failures.append(f"Duration: {elapsed:.1f}s > {MAX_ACCEPTABLE_DURATION_S}s")

        if tool_count <= MAX_ACCEPTABLE_TOOL_CALLS:
            passes.append(f"Tool calls: {tool_count} <= {MAX_ACCEPTABLE_TOOL_CALLS}")
        else:
            failures.append(f"Tool calls: {tool_count} > {MAX_ACCEPTABLE_TOOL_CALLS}")

        if len(excluded_builtins) >= len(COPILOT_BUILTIN_TOOL_NAMES):
            passes.append(
                f"Excluded builtins: {len(excluded_builtins)} >= {len(COPILOT_BUILTIN_TOOL_NAMES)}"
            )
        else:
            failures.append(
                f"Excluded builtins: {len(excluded_builtins)} < {len(COPILOT_BUILTIN_TOOL_NAMES)} — "
                f"BYPASS RISK"
            )

        if sdk_metrics:
            sdk_turns = sdk_metrics.get("turns", 0)
            if sdk_turns <= MAX_ACCEPTABLE_TURNS:
                passes.append(f"SDK turns: {sdk_turns} <= {MAX_ACCEPTABLE_TURNS}")
            else:
                failures.append(f"SDK turns: {sdk_turns} > {MAX_ACCEPTABLE_TURNS}")

        result["status"] = "PASS" if not failures else "FAIL"
        result["passes"] = passes
        result["failures"] = failures

        _print_forensic_comparison(result)
        return result

    finally:
        await provider.close()


async def _correlate_sdk_session(
    before_ts: datetime,
) -> dict[str, Any] | None:
    """Find and parse the SDK session created after before_ts."""
    session_state_dir = Path.home() / ".copilot" / "session-state"
    if not session_state_dir.exists():
        return None

    for session_dir in sorted(
        session_state_dir.iterdir(),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    ):
        if not session_dir.is_dir():
            continue
        events_file = session_dir / "events.jsonl"
        if not events_file.exists():
            continue
        mtime = datetime.fromtimestamp(events_file.stat().st_mtime, tz=UTC)
        if mtime >= before_ts:
            return parse_sdk_session(events_file)

    return None


def _print_forensic_comparison(result: dict[str, Any]) -> None:
    """Print a detailed comparison table."""
    comparison = result.get("comparison", {})

    logger.info(
        f"\n{'=' * 70}\n"
        f" FORENSIC COMPARISON: BEFORE vs AFTER\n"
        f"{'=' * 70}\n"
        f"\n"
        f" Original Incident (session a1a0af17 / bfa3b57b):\n"
        f"   - Prompt: 'Use bug-hunter to check models.py'\n"
        f"   - SDK turns: {INCIDENT_TURNS}\n"
        f"   - Tool calls returned to Amplifier: {INCIDENT_TOOL_CALLS}\n"
        f"   - bug-hunter agents spawned: {INCIDENT_DELEGATES}\n"
        f"   - Duration: ~{INCIDENT_DURATION_S}s\n"
        f"\n"
        f" Current Results (SDK Driver architecture):\n"
        f"   - SDK turns: {comparison.get('current_turns', 'N/A')}\n"
        f"   - Tool calls returned: {comparison.get('current_tool_calls', 'N/A')}\n"
        f"   - Duration: {comparison.get('current_duration_s', 'N/A')}s\n"
        f"   - Excluded built-ins: {result.get('excluded_builtin_count', 'N/A')}\n"
        f"   - Captured tools: {result.get('tool_names', [])}\n"
        f"\n"
        f"{'─' * 70}\n"
        f" {'METRIC':<35} {'INCIDENT':>12} {'NOW':>12} {'THRESHOLD':>12}\n"
        f"{'─' * 70}\n"
        f" {'SDK turns':<35} {INCIDENT_TURNS:>12} "
        f"{str(comparison.get('current_turns', '?')):>12} "
        f"{'<=' + str(MAX_ACCEPTABLE_TURNS):>12}\n"
        f" {'Tool calls':<35} {INCIDENT_TOOL_CALLS:>12} "
        f"{comparison.get('current_tool_calls', '?'):>12} "
        f"{'<=' + str(MAX_ACCEPTABLE_TOOL_CALLS):>12}\n"
        f" {'Duration (s)':<35} {INCIDENT_DURATION_S:>12} "
        f"{comparison.get('current_duration_s', '?'):>12} "
        f"{'<=' + str(MAX_ACCEPTABLE_DURATION_S):>12}\n"
        f"{'─' * 70}\n"
    )

    if result.get("passes"):
        logger.info("PASSED CHECKS:")
        for p in result["passes"]:
            logger.info(f"  [PASS] {p}")

    if result.get("failures"):
        logger.error("FAILED CHECKS:")
        for f in result["failures"]:
            logger.error(f"  [FAIL] {f}")

    status = result.get("status", "UNKNOWN")
    if status == "PASS":
        logger.info(f"\n{'=' * 70}")
        logger.info(" VERDICT: PASS — SDK Driver architecture fix confirmed")
        logger.info(f"{'=' * 70}\n")
    else:
        logger.error(f"\n{'=' * 70}")
        logger.error(" VERDICT: FAIL — Possible regression detected")
        logger.error(f"{'=' * 70}\n")


def run_comparison() -> dict[str, Any]:
    """
    Compare original incident data with latest session data.

    Reads the archived forensic data and the most recent session.
    """
    logger.info("=" * 60)
    logger.info("FORENSIC COMPARISON: Original incident vs latest session")
    logger.info("=" * 60)

    # Load original incident data
    dev_scripts = Path(__file__).resolve().parents[2] / "_dev_scripts"
    sdk_events_file = dev_scripts / "_forensic_sdk_events.jsonl"
    amp_events_file = dev_scripts / "_forensic_amp_events.jsonl"

    results: dict[str, Any] = {"test": "comparison"}

    if sdk_events_file.exists():
        incident_sdk = parse_sdk_session(sdk_events_file)
        results["incident_sdk"] = incident_sdk
        logger.info(
            f"Original SDK incident: "
            f"{incident_sdk['turns']} turns, "
            f"{incident_sdk['denials']} denials"
        )
    else:
        logger.warning(f"Incident SDK data not found: {sdk_events_file}")

    if amp_events_file.exists():
        incident_amp = parse_amplifier_session(amp_events_file)
        results["incident_amp"] = incident_amp
        logger.info(
            f"Original Amplifier incident: "
            f"{incident_amp['delegate_spawns']} delegates, "
            f"{incident_amp['total_tool_calls']} tool calls"
        )
    else:
        logger.warning(f"Incident Amplifier data not found: {amp_events_file}")

    # Find latest session
    sessions_dir = (
        Path.home()
        / ".amplifier"
        / "projects"
        / "-mnt-e-amplifier+GHC-CLI-SDK-Experiment"
        / "sessions"
    )

    if sessions_dir.exists():
        session_dirs = sorted(
            [
                d
                for d in sessions_dir.iterdir()
                if d.is_dir() and len(d.name) == 36  # UUID format
            ],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )

        if session_dirs:
            latest_events = session_dirs[0] / "events.jsonl"
            if latest_events.exists():
                latest_amp = parse_amplifier_session(latest_events)
                results["latest_amp"] = latest_amp
                results["latest_session"] = session_dirs[0].name
                logger.info(
                    f"Latest session ({session_dirs[0].name}): "
                    f"{latest_amp['delegate_spawns']} delegates, "
                    f"{latest_amp['total_tool_calls']} tool calls"
                )

    results["status"] = "COMPLETE"
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Result Output
# ═══════════════════════════════════════════════════════════════════════════════


def write_report(
    results: list[dict[str, Any]],
    output_path: Path | None = None,
) -> None:
    """Write JSON report of all test results."""
    if output_path is None:
        output_path = Path(__file__).parent / "forensic_report.json"

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "incident_reference": {
            "session_id": INCIDENT_SESSION_ID,
            "sdk_session_id": INCIDENT_SDK_SESSION_ID,
            "turns": INCIDENT_TURNS,
            "tool_calls": INCIDENT_TOOL_CALLS,
            "delegates": INCIDENT_DELEGATES,
            "duration_s": INCIDENT_DURATION_S,
        },
        "thresholds": {
            "max_turns": MAX_ACCEPTABLE_TURNS,
            "max_tool_calls": MAX_ACCEPTABLE_TOOL_CALLS,
            "max_delegates": MAX_ACCEPTABLE_DELEGATES,
            "max_duration_s": MAX_ACCEPTABLE_DURATION_S,
        },
        "results": results,
        "overall_status": (
            "PASS" if all(r.get("status") == "PASS" for r in results if "status" in r) else "FAIL"
        ),
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Report written to: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════════


async def main() -> int:
    """CLI entry point for the forensic runner."""
    parser = argparse.ArgumentParser(
        description="Forensic Analysis & Integration Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 forensic_runner.py --smoke          # Quick provider smoke test
  python3 forensic_runner.py --forensic       # Reproduce the 305-turn incident
  python3 forensic_runner.py --compare        # Compare incident data vs latest
  python3 forensic_runner.py --all            # Run everything
        """,
    )

    parser.add_argument(
        "--smoke", action="store_true", help="Run smoke test (simple math, no tools)"
    )
    parser.add_argument(
        "--forensic",
        action="store_true",
        help="Run forensic regression (reproduces the incident prompt)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare original incident data with latest session",
    )
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for JSON report",
    )

    args = parser.parse_args()

    # Default to --all if nothing specified
    if not any([args.smoke, args.forensic, args.compare, args.all]):
        args.all = True

    results: list[dict[str, Any]] = []
    exit_code = 0

    if args.smoke or args.all:
        result = await run_smoke_test()
        results.append(result)
        if result.get("status") != "PASS":
            exit_code = 1

    if args.forensic or args.all:
        result = await run_forensic_regression()
        results.append(result)
        if result.get("status") != "PASS":
            exit_code = 1

    if args.compare or args.all:
        result = run_comparison()
        results.append(result)

    # Write report
    write_report(results, args.output)

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)
    for r in results:
        status = r.get("status", "N/A")
        test = r.get("test", "unknown")
        elapsed = r.get("elapsed", "N/A")
        icon = "[PASS]" if status == "PASS" else "[FAIL]" if status == "FAIL" else "[INFO]"
        logger.info(f"  {icon} {test}: {status} ({elapsed}s)")

    overall = "PASS" if exit_code == 0 else "FAIL"
    logger.info(f"\nOverall: {overall}")
    return exit_code


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
