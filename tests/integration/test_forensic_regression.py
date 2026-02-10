"""
Forensic Regression Test: Reproduces the 305-Turn SDK Loop Incident.

This test module validates that the SDK Driver architecture ACTUALLY FIXES
the bug discovered during forensic analysis of session a1a0af17:

    ORIGINAL BUG (Feb 7, 2026):
    - Prompt: "Use the bug-hunter agent to check if there are any obvious
      issues in the models.py file."
    - Result: 305 SDK turns, 607 accumulated tool calls, 303 bug-hunter
      agents spawned, 20-minute hang
    - Root cause: No first-turn capture, no session abort, no circuit breaker

    EXPECTED BEHAVIOR (after SDK Driver fix):
    - Same prompt triggers tool calls (delegate + report_intent)
    - SDK Driver captures first turn only
    - Session is aborted immediately after capture
    - ≤3 turns (not 305)
    - ≤10 tool calls (not 607)
    - Completes in <60 seconds (not 20 minutes)

Tests are skipped by default. Run with:
    RUN_LIVE_TESTS=1 python -m pytest tests/integration/test_forensic_regression.py -v -s

Prerequisites:
    - Copilot CLI installed and in PATH (WSL)
    - Valid GitHub Copilot authentication
    - Network access
    - PYTHONPATH includes amplifier-core

Evidence base:
    - Session 497bbab7: Agent loop problem analysis
    - Session a1a0af17: 305-turn forensic analysis
    - _dev_scripts/_forensic_sdk_events.jsonl (305-turn incident data)
    - _dev_scripts/_forensic_amp_events.jsonl (Amplifier-side incident data)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from amplifier_module_provider_github_copilot import CopilotSdkProvider
from amplifier_module_provider_github_copilot._constants import COPILOT_BUILTIN_TOOL_NAMES

logger = logging.getLogger(__name__)

# Skip all tests unless explicitly enabled
pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_LIVE_TESTS"),
    reason="Live tests disabled. Set RUN_LIVE_TESTS=1 to run.",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Forensic Thresholds — derived from the original incident
# ═══════════════════════════════════════════════════════════════════════════════

# Original incident: 305 turns, 607 tool calls, ~1200 seconds
INCIDENT_TURNS = 305
INCIDENT_TOOL_CALLS = 607
INCIDENT_DURATION_S = 1200

# Acceptance thresholds: must be orders of magnitude better
MAX_ACCEPTABLE_TURNS = 5  # Was 305
MAX_ACCEPTABLE_TOOL_CALLS = 10  # Was 607
MAX_ACCEPTABLE_DURATION_S = 60  # Was 1200
MAX_ACCEPTABLE_DELEGATES = 2  # Was 303


# ═══════════════════════════════════════════════════════════════════════════════
# Amplifier-Compatible Tool Specs
# ═══════════════════════════════════════════════════════════════════════════════


def make_amplifier_tools() -> list[Mock]:
    """
    Create mock tools that mirror what Amplifier's foundationbundle provides.

    These are the same 14 tools the original incident session registered.
    We include delegate and report_intent which are the critical ones -
    these triggered the 305-turn loop.
    """
    tool_specs = [
        (
            "delegate",
            "Delegate a task to a specialized agent",
            {
                "type": "object",
                "properties": {
                    "agent": {"type": "string", "description": "Agent identifier"},
                    "instruction": {"type": "string", "description": "Task instruction"},
                    "context_depth": {"type": "string", "description": "Context depth"},
                },
                "required": ["agent", "instruction"],
            },
        ),
        (
            "report_intent",
            "Report your intent before taking action",
            {
                "type": "object",
                "properties": {
                    "intent": {"type": "string", "description": "What you intend to do"},
                },
                "required": ["intent"],
            },
        ),
        (
            "read_file",
            "Read the contents of a file",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "required": ["path"],
            },
        ),
        (
            "write_file",
            "Write content to a file",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        ),
        # NOTE: Do NOT use "bash" or "grep" as tool names here.
        # They collide with SDK built-in names, causing API error:
        # "CAPIError: 400 tools: Tool names must be unique"
        (
            "run_command",
            "Execute a shell command",
            {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                },
                "required": ["command"],
            },
        ),
        (
            "list_directory",
            "List contents of a directory",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "required": ["path"],
            },
        ),
        (
            "search_files",
            "Search for files matching a pattern",
            {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["pattern"],
            },
        ),
        (
            "search_content",
            "Search file contents with regex",
            {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["pattern"],
            },
        ),
        (
            "todo",
            "Track progress on tasks",
            {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "items": {"type": "array"},
                },
                "required": ["action"],
            },
        ),
    ]

    tools: list[Mock] = []
    for name, description, params in tool_specs:
        tool = Mock()
        tool.name = name
        tool.description = description
        tool.parameters = params
        tools.append(tool)
    return tools


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
async def forensic_provider():
    """
    Create a provider configured identically to the incident session.

    Config mirrors what Amplifier used in session a1a0af17:
    - Model: claude-opus-4.5 (same as incident)
    - Streaming: True (required for SDK Driver)
    - Debug: True (for forensic logging)
    - Timeout: 120s (generous but bounded)
    """
    coordinator = Mock()
    coordinator.hooks = Mock()
    coordinator.hooks.emit = AsyncMock()

    provider = CopilotSdkProvider(
        api_key=None,
        config={
            "model": "claude-opus-4.5",
            "timeout": 120,
            "debug": True,
            "use_streaming": True,
            "sdk_max_turns": 5,  # Safety: hard cap at 5 turns
        },
        coordinator=coordinator,
    )
    yield provider
    await provider.close()


@pytest.fixture
def forensic_report_path(tmp_path: Path) -> Path:
    """Path for the forensic comparison report."""
    return tmp_path / "forensic_regression_report.json"


# ═══════════════════════════════════════════════════════════════════════════════
# Test Class: Forensic Regression
# ═══════════════════════════════════════════════════════════════════════════════


class TestForensicRegression:
    """
    Forensic regression tests that reproduce the EXACT scenarios from the
    original 305-turn incident and verify they are now handled correctly.

    These tests compare "before" metrics (from the incident) against
    "after" metrics (from the current code) to prove the fix works.
    """

    @pytest.mark.asyncio
    async def test_bug_hunter_delegation_prompt(
        self,
        forensic_provider: CopilotSdkProvider,
    ) -> None:
        """
        THE EXACT PROMPT that caused the 305-turn incident.

        Original: "Use the bug-hunter agent to check if there are any
        obvious issues in the models.py file."

        This prompt triggers delegate + report_intent tool calls.
        The old code accumulated these across 305 turns.
        The SDK Driver should capture first turn only and abort.
        """
        request = Mock()
        request.messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful AI assistant with access to tools. "
                    "When the user asks you to delegate a task to another agent, "
                    "use the delegate tool. Always report your intent first "
                    "using the report_intent tool."
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
        request.tools = make_amplifier_tools()
        request.stream = None

        # Record start time for comparison
        start = time.time()

        response = await forensic_provider.complete(request)
        elapsed = time.time() - start

        # ═══════════════════════════════════════════════════════════════
        # FORENSIC COMPARISON: Before vs After
        # ═══════════════════════════════════════════════════════════════

        tool_count = len(response.tool_calls) if response.tool_calls else 0

        logger.info(
            f"\n{'=' * 60}\n"
            f"FORENSIC REGRESSION RESULTS\n"
            f"{'=' * 60}\n"
            f"Prompt: 'Use bug-hunter agent to check models.py'\n"
            f"{'─' * 60}\n"
            f"{'Metric':<30} {'BEFORE (incident)':<20} {'AFTER (now)':<20}\n"
            f"{'─' * 60}\n"
            f"{'Duration':<30} {INCIDENT_DURATION_S:>15}s {elapsed:>15.1f}s\n"
            f"{'Tool calls returned':<30} {INCIDENT_TOOL_CALLS:>15} {tool_count:>15}\n"
            f"{'Finish reason':<30} {'tool_use':>15} {response.finish_reason:>15}\n"
            f"{'=' * 60}"
        )

        # ASSERTIONS: Must be dramatically better than incident
        assert elapsed < MAX_ACCEPTABLE_DURATION_S, (
            f"Request took {elapsed:.1f}s — exceeds {MAX_ACCEPTABLE_DURATION_S}s limit. "
            f"Original incident: {INCIDENT_DURATION_S}s. "
            f"Possible regression to pre-SDK-Driver behavior."
        )

        assert tool_count <= MAX_ACCEPTABLE_TOOL_CALLS, (
            f"Got {tool_count} tool calls — exceeds {MAX_ACCEPTABLE_TOOL_CALLS} limit. "
            f"Original incident: {INCIDENT_TOOL_CALLS}. "
            f"Tool call accumulation bug may have regressed."
        )

        # Verify we actually got tool calls (the model should use delegate)
        if tool_count > 0:
            assert response.finish_reason == "tool_use"
            tool_names = [tc.name for tc in response.tool_calls]
            logger.info(f"Captured tools: {tool_names}")

            # Expect delegate or report_intent (the tools from the incident)
            expected_tools = {
                "delegate",
                "report_intent",
                "read_file",
                "run_command",
                "list_directory",
                "search_files",
                "search_content",
                "write_file",
                "todo",
            }
            for name in tool_names:
                assert name in expected_tools, (
                    f"Unexpected tool '{name}' — may indicate SDK built-in bypass"
                )
        else:
            # Model chose not to use tools — acceptable but less interesting
            logger.warning(
                "Model did not request tool calls. "
                "This is valid but doesn't exercise the SDK Driver path."
            )

    @pytest.mark.asyncio
    async def test_file_creation_builtin_bypass(
        self,
        forensic_provider: CopilotSdkProvider,
    ) -> None:
        """
        Reproduce the built-in tool bypass from session 497bbab7.

        Original: "Create a file called /tmp/copilot-test.txt with 'hello world'"
        - Old behavior: SDK's edit built-in executed internally, tool_calls=0
        - Expected: write_file tool captured, no SDK built-in execution

        This validates that ALL 13 built-in tools are excluded.
        """
        request = Mock()
        request.messages = [
            {
                "role": "system",
                "content": (
                    "You have access to file operation tools. "
                    "When asked to create or write a file, use the write_file tool. "
                    "Do NOT use any other method to create files."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Create a file called /tmp/copilot-regression-test.txt "
                    "with the text 'hello world'"
                ),
            },
        ]
        request.tools = make_amplifier_tools()
        request.stream = None

        start = time.time()
        response = await forensic_provider.complete(request)
        elapsed = time.time() - start

        tool_count = len(response.tool_calls) if response.tool_calls else 0

        logger.info(
            f"\n{'=' * 60}\n"
            f"BUILT-IN BYPASS REGRESSION RESULTS\n"
            f"{'=' * 60}\n"
            f"{'Metric':<30} {'BEFORE (497bbab7)':<20} {'AFTER (now)':<20}\n"
            f"{'─' * 60}\n"
            f"{'tool_calls returned':<30} {'0 (bypassed)':>15} {tool_count:>15}\n"
            f"{'File created by SDK?':<30} {'YES (edit)':>15} {'NO (expected)':>15}\n"
            f"{'Duration':<30} {'16.1s':>15} {elapsed:>10.1f}s\n"
            f"{'=' * 60}"
        )

        # CRITICAL: We should see tool_calls > 0 (the model should use write_file)
        # In the old code, tool_calls was 0 because the SDK ran edit internally
        if tool_count > 0:
            # Success: model used our custom tool, not the built-in
            tool_names = [tc.name for tc in response.tool_calls]
            logger.info(f"Captured tools: {tool_names}")

            # Should NOT see any SDK built-in tool names
            for name in tool_names:
                assert name not in COPILOT_BUILTIN_TOOL_NAMES, (
                    f"SDK built-in tool '{name}' appeared in captured tools! "
                    f"Built-in exclusion may not be working."
                )

            # Should see write_file or similar custom tool
            file_tools = [n for n in tool_names if "file" in n or "write" in n]
            if file_tools:
                logger.info(f"File operation captured via custom tool: {file_tools}")
        else:
            # Could mean: (a) model chose to answer without tools, or
            # (b) SDK built-in executed internally (the old bug)
            # Check: was the file actually created?
            import subprocess

            result = subprocess.run(
                ["test", "-f", "/tmp/copilot-regression-test.txt"],
                capture_output=True,
            )
            if result.returncode == 0:
                pytest.fail(
                    "tool_calls=0 but file was created! "
                    "This indicates SDK built-in bypass is still happening. "
                    "Check that ALL built-in tools are in excluded_tools."
                )
            else:
                logger.info(
                    "tool_calls=0 and file not created — model chose text-only response. "
                    "This is valid behavior."
                )

        assert elapsed < MAX_ACCEPTABLE_DURATION_S

    @pytest.mark.asyncio
    async def test_excluded_builtin_tools_verification(
        self,
        forensic_provider: CopilotSdkProvider,
    ) -> None:
        """
        Verify the provider excludes ALL 13 known built-in tools.

        The forensic analysis (497bbab7) showed only 4 tools were excluded:
        bash, glob, grep, web_fetch. The fix expanded this to 13.

        This test instructs hooks.emit to capture the llm:request event
        and inspects the excluded_builtin_tools list.
        """
        emitted_events: list[tuple[str, dict[str, Any]]] = []

        async def capture_emit(event_name: str, data: dict[str, Any]) -> None:
            emitted_events.append((event_name, data))

        # Replace emit with capturing version
        forensic_provider._coordinator.hooks.emit = AsyncMock(side_effect=capture_emit)

        request = Mock()
        request.messages = [
            {"role": "user", "content": "What is 2+2?"},
        ]
        request.tools = make_amplifier_tools()
        request.stream = None

        await forensic_provider.complete(request)

        # Find llm:request event
        llm_requests = [(name, data) for name, data in emitted_events if name == "llm:request"]
        assert llm_requests, "No llm:request event emitted"

        _, request_data = llm_requests[0]
        excluded = request_data.get("excluded_builtin_tools", [])

        logger.info(
            f"Excluded built-in tools ({len(excluded)}): {excluded}\n"
            f"Known built-in tools ({len(COPILOT_BUILTIN_TOOL_NAMES)}): "
            f"{sorted(COPILOT_BUILTIN_TOOL_NAMES)}"
        )

        # Every known built-in should be excluded when user tools are present
        for builtin_tool in COPILOT_BUILTIN_TOOL_NAMES:
            assert builtin_tool in excluded, (
                f"Built-in tool '{builtin_tool}' NOT in excluded list! "
                f"This was the root cause of the 497bbab7 bypass."
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Test Class: SDK Session Forensics
# ═══════════════════════════════════════════════════════════════════════════════


class TestSdkSessionForensics:
    """
    Post-hoc forensic analysis of SDK session data.

    After running a live test, examines the SDK session-state directory
    to verify the SDK saw the correct number of turns and that sessions
    were properly destroyed.
    """

    @pytest.mark.asyncio
    async def test_sdk_session_turn_count(
        self,
        forensic_provider: CopilotSdkProvider,
    ) -> None:
        """
        Run a tool-triggering prompt and correlate with SDK session events.

        After the provider completes, find the SDK session in
        ~/.copilot/session-state/ and count actual turns.
        """
        request = Mock()
        request.messages = [
            {
                "role": "system",
                "content": "You have a read_file tool. Use it when asked to read files.",
            },
            {
                "role": "user",
                "content": "Read the file README.md",
            },
        ]
        request.tools = make_amplifier_tools()[:3]  # delegate, report_intent, read_file
        request.stream = None

        # Record time window
        before = datetime.now(UTC)
        await forensic_provider.complete(request)
        after = datetime.now(UTC)

        # Find the SDK session created during this test
        session_state_dir = Path.home() / ".copilot" / "session-state"
        if not session_state_dir.exists():
            pytest.skip("~/.copilot/session-state/ not found")

        # Find sessions modified in our time window
        candidate_sessions: list[Path] = []
        for session_dir in session_state_dir.iterdir():
            if not session_dir.is_dir():
                continue
            events_file = session_dir / "events.jsonl"
            if events_file.exists():
                mtime = datetime.fromtimestamp(events_file.stat().st_mtime, tz=UTC)
                # Session modified within our test window (with margin)
                if mtime >= before and mtime <= after:
                    candidate_sessions.append(events_file)

        if not candidate_sessions:
            logger.warning("No SDK sessions found in test time window")
            pytest.skip("Could not find SDK session for correlation")

        # Parse the most recent candidate
        events_file = sorted(candidate_sessions, key=lambda p: p.stat().st_mtime)[-1]
        events: list[dict[str, Any]] = []
        with open(events_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))

        # Count turns
        turn_starts = [e for e in events if e.get("type") == "assistant.turn_start"]
        turn_ends = [e for e in events if e.get("type") == "assistant.turn_end"]
        tool_denials = [
            e
            for e in events
            if (
                e.get("type") == "tool.execution_complete"
                and not e.get("data", {}).get("success", True)
            )
        ]

        logger.info(
            f"\n{'=' * 60}\n"
            f"SDK SESSION FORENSIC ANALYSIS\n"
            f"{'=' * 60}\n"
            f"Session: {events_file.parent.name}\n"
            f"Total events: {len(events)}\n"
            f"Turn starts: {len(turn_starts)}\n"
            f"Turn ends: {len(turn_ends)}\n"
            f"Tool denials: {len(tool_denials)}\n"
            f"{'─' * 60}\n"
            f"{'Metric':<30} {'BEFORE (incident)':<20} {'AFTER (now)':<20}\n"
            f"{'─' * 60}\n"
            f"{'SDK turns':<30} {INCIDENT_TURNS:>15} {len(turn_starts):>15}\n"
            f"{'Tool denials':<30} {INCIDENT_TOOL_CALLS:>15} {len(tool_denials):>15}\n"
            f"{'=' * 60}"
        )

        # CRITICAL: Turn count must be dramatically lower
        assert len(turn_starts) <= MAX_ACCEPTABLE_TURNS, (
            f"SDK ran {len(turn_starts)} turns — exceeds {MAX_ACCEPTABLE_TURNS}. "
            f"Original incident had {INCIDENT_TURNS} turns. "
            f"Session abort may not be working."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Test Class: Amplifier End-to-End
# ═══════════════════════════════════════════════════════════════════════════════


def _find_amplifier_cli() -> str | None:
    """Find the amplifier CLI binary on PATH or in known locations."""
    found = shutil.which("amplifier")
    if found:
        return found
    # Check uv tool install location (WSL/Linux)
    home_local = Path.home() / ".local" / "bin" / "amplifier"
    if home_local.exists():
        return str(home_local)
    # Check uv tool install location (Windows)
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA", "")
        if appdata:
            win_path = Path(appdata) / "uv" / "tools" / "amplifier" / "Scripts" / "amplifier.exe"
            if win_path.exists():
                return str(win_path)
    return None


def _get_workspace_cwd() -> str:
    """Return the workspace root as a platform-appropriate path."""
    if sys.platform == "win32":
        return r"E:\amplifier+GHC-CLI-SDK-Experiment"
    return "/mnt/e/amplifier+GHC-CLI-SDK-Experiment"


def _get_provider_cwd() -> str:
    """Return the provider module root as a platform-appropriate path."""
    workspace = _get_workspace_cwd()
    sep = "\\" if sys.platform == "win32" else "/"
    return f"{workspace}{sep}amplifier-module-provider-github-copilot"


class TestAmplifierEndToEnd:
    """
    Full end-to-end tests that invoke Amplifier CLI and verify
    the Copilot provider behaves correctly as an integrated module.

    These tests run 'amplifier' commands via subprocess and parse
    the session events for forensic comparison.

    Prerequisites:
        - Amplifier CLI installed via 'uv tool install' and on PATH
        - Copilot provider configured in Amplifier
    """

    @pytest.mark.asyncio
    async def test_amplifier_simple_math(self) -> None:
        """
        Run a simple math prompt through Amplifier and verify clean completion.

        This is a smoke test that validates Amplifier can use the Copilot
        provider without errors.
        """
        import subprocess

        amplifier_bin = _find_amplifier_cli()
        if not amplifier_bin:
            pytest.skip(
                "Amplifier CLI not found on PATH or in ~/.local/bin/. "
                "Install with: uv tool install amplifier"
            )

        result = subprocess.run(
            [
                amplifier_bin,
                "run",
                "--mode",
                "single",
                "What is 7 * 8? Reply with ONLY the number, no explanation.",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=_get_workspace_cwd(),
        )

        logger.info(
            f"Amplifier exit code: {result.returncode}\n"
            f"stdout: {result.stdout[:500]}\n"
            f"stderr: {result.stderr[:500]}"
        )

        # Should complete without error
        assert result.returncode == 0, (
            f"Amplifier failed with exit code {result.returncode}: {result.stderr}"
        )

        # Should contain "56" somewhere in output
        assert "56" in result.stdout, f"Expected '56' in response, got: {result.stdout[:200]}"

    @pytest.mark.asyncio
    async def test_amplifier_bug_hunter_delegation(self) -> None:
        """
        THE CRITICAL E2E TEST: Run the exact forensic incident prompt
        through Amplifier CLI and verify correct behavior.

        This is the definitive proof that the SDK Driver architecture
        fixes the 305-turn loop.
        """
        import subprocess

        amplifier_bin = _find_amplifier_cli()
        if not amplifier_bin:
            pytest.skip(
                "Amplifier CLI not found on PATH or in ~/.local/bin/. "
                "Install with: uv tool install amplifier"
            )

        start = time.time()
        result = subprocess.run(
            [
                amplifier_bin,
                "run",
                "--mode",
                "single",
                "Use the bug-hunter agent to check if there are any "
                "obvious issues in the models.py file.",
            ],
            capture_output=True,
            text=True,
            timeout=180,  # 3 minutes max (was 20 minutes before)
            cwd=_get_provider_cwd(),
        )
        elapsed = time.time() - start

        logger.info(
            f"\n{'=' * 60}\n"
            f"AMPLIFIER E2E BUG-HUNTER TEST\n"
            f"{'=' * 60}\n"
            f"Exit code: {result.returncode}\n"
            f"Duration: {elapsed:.1f}s\n"
            f"stdout length: {len(result.stdout)} chars\n"
            f"stderr length: {len(result.stderr)} chars\n"
            f"{'=' * 60}"
        )

        # Must complete within time limit
        assert elapsed < MAX_ACCEPTABLE_DURATION_S * 2, (
            f"Amplifier took {elapsed:.1f}s — possible SDK loop regression"
        )

        # Parse the session events for forensic analysis
        await self._analyze_latest_session(elapsed)

    async def _analyze_latest_session(self, elapsed: float) -> None:
        """
        Find and analyze the latest Amplifier session events.

        Counts delegate spawns and LLM request/response metrics.
        """
        # Project name varies by platform: WSL uses the /mnt/e/ path,
        # Windows uses the E:\ path, both slugified by Amplifier.
        if sys.platform == "win32":
            project_slug = "E--amplifier+GHC-CLI-SDK-Experiment"
        else:
            project_slug = "-mnt-e-amplifier+GHC-CLI-SDK-Experiment"
        sessions_dir = Path.home() / ".amplifier" / "projects" / project_slug / "sessions"

        if not sessions_dir.exists():
            logger.warning(f"Sessions dir not found: {sessions_dir}")
            return

        # Find the most recently modified session
        session_dirs = sorted(
            [d for d in sessions_dir.iterdir() if d.is_dir() and "-" not in d.name[36:]],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )

        if not session_dirs:
            logger.warning("No session directories found")
            return

        events_file = session_dirs[0] / "events.jsonl"
        if not events_file.exists():
            logger.warning(f"No events file: {events_file}")
            return

        events: list[dict[str, Any]] = []
        with open(events_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))

        # Count key metrics
        delegate_spawns = sum(1 for e in events if e.get("event") == "delegate:agent_spawned")
        llm_responses = [e for e in events if e.get("event") == "llm:response"]

        total_tool_calls = 0
        for resp in llm_responses:
            tc = resp.get("data", {}).get("tool_calls", 0)
            if isinstance(tc, int):
                total_tool_calls += tc

        logger.info(
            f"\n{'=' * 60}\n"
            f"AMPLIFIER SESSION FORENSIC ANALYSIS\n"
            f"{'=' * 60}\n"
            f"Session: {session_dirs[0].name}\n"
            f"Total events: {len(events)}\n"
            f"{'─' * 60}\n"
            f"{'Metric':<30} {'BEFORE (incident)':<20} {'AFTER (now)':<20}\n"
            f"{'─' * 60}\n"
            f"{'Duration':<30} {INCIDENT_DURATION_S:>15}s {elapsed:>15.1f}s\n"
            f"{'delegate:agent_spawned':<30} {'303':>15} {delegate_spawns:>15}\n"
            f"{'Total tool_calls (llm:resp)':<30} {INCIDENT_TOOL_CALLS:>15} {total_tool_calls:>15}\n"
            f"{'=' * 60}"
        )

        # ASSERTIONS
        assert delegate_spawns <= MAX_ACCEPTABLE_DELEGATES, (
            f"Amplifier spawned {delegate_spawns} delegate agents — "
            f"exceeds limit of {MAX_ACCEPTABLE_DELEGATES}. "
            f"Original incident spawned 303. SDK Driver may not be working."
        )

        assert total_tool_calls <= MAX_ACCEPTABLE_TOOL_CALLS, (
            f"Total tool calls across LLM responses: {total_tool_calls} — "
            f"exceeds limit of {MAX_ACCEPTABLE_TOOL_CALLS}. "
            f"Original incident: {INCIDENT_TOOL_CALLS}."
        )
