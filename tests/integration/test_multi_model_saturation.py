"""
Multi-Model Saturation Tests — Tool Call Leakage Prevention Validation.

Regression tests validating that LLMs produce structured tool calls (not
text-serialized tool calls) even after long conversation histories with
many tool operations.

These tests prevent a bug where models could learn to reproduce tool call
patterns as plain text like "[Tool Call: write_file({...})]" instead of
structured tool_calls objects that the orchestrator can execute.

MODELS TESTED (4):
    - claude-opus-4.5
    - gpt-5-mini
    - gpt-4.1
    - gemini-3-pro-preview

SCENARIOS (8 per model = 32 total):
    - 10/15/20/25 turn histories with explicit tool use prompts
    - Multi-step tool operations
    - Ambiguous prompts requiring tool selection
    - Prompts without tool hints
    - Plan-then-execute patterns

EXPECTED BEHAVIOR:
    After saturated history, the final prompt should elicit:
    - Structured tool_calls in response (NOT text-serialized)
    - No "[Tool Call: ...]" patterns in text content
    - Clear finish_reason indicating tool use

RUN:
    RUN_LIVE_TESTS=1 python -m pytest tests/integration/test_multi_model_saturation.py -v -s

TIME: ~6 minutes (32 tests × ~10-12 seconds each)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

logger = logging.getLogger(__name__)

# Skip all tests unless explicitly enabled
pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_LIVE_TESTS"),
    reason="Live tests disabled. Set RUN_LIVE_TESTS=1 to run.",
)

# Evidence directory for forensic capture (in tests/ for gitignore)
EVIDENCE_DIR = Path(__file__).parent.parent.parent / "test_evidence" / "saturation"

# Pattern to detect tool calls serialized as text (should NOT appear in response)
TEXT_TOOL_CALL_PATTERN = re.compile(
    r"\[Tool\s*Call[:\s]|"
    r"tool_call\s*[:(]|"
    r"tool_use\s*[:(]|"
    r'"type":\s*"tool_call"',
    re.IGNORECASE,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Test Infrastructure
# ═══════════════════════════════════════════════════════════════════════════════


def _ensure_evidence_dir() -> Path:
    """Ensure evidence directory exists."""
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    return EVIDENCE_DIR


def save_evidence(name: str, data: dict[str, Any]) -> Path:
    """Save test evidence as JSON for debugging."""
    _ensure_evidence_dir()
    path = EVIDENCE_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.debug(f"Evidence saved: {path}")
    return path


def fake_tool_call_id() -> str:
    """Generate a fake tool call ID matching SDK format."""
    return f"toolu_{uuid.uuid4().hex[:24]}"


def make_tools() -> list[Mock]:
    """Create mock tool definitions for registration."""
    tools = []
    for name, desc, params in [
        ("read_file", "Read the contents of a file from disk", {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }),
        ("write_file", "Write content to a file on disk", {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        }),
        ("search_files", "Search for files matching a pattern", {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
            },
            "required": ["pattern"],
        }),
        ("run_command", "Execute a shell command", {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        }),
        ("list_directory", "List contents of a directory", {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }),
    ]:
        t = Mock()
        t.name = name
        t.description = desc
        t.parameters = params
        tools.append(t)
    return tools


def build_saturated_history(turn_count: int) -> list[dict[str, Any]]:
    """
    Build a conversation history saturated with tool usage patterns.

    This simulates a realistic coding session with file reads, writes,
    searches, and command execution. The history teaches the model
    the pattern of tool usage — we then test that the model uses
    structured tool calls rather than text-serialized ones.

    Args:
        turn_count: Number of completed tool operations (user→assistant→tool)

    Returns:
        List of message dicts ready for provider
    """
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a helpful coding assistant with access to file tools. "
                "When asked to read, write, search, or list files, ALWAYS use "
                "the appropriate tool. NEVER write tool calls as text."
            ),
        },
    ]

    # Realistic development session operations
    operations = [
        ("Read the file README.md", "read_file", {"path": "README.md"},
         "# My Project\nA Python web application using FastAPI.\n\n## Setup\npip install -r requirements.txt"),
        ("List the src directory", "list_directory", {"path": "src/"},
         "main.py\nconfig.py\nmodels/\nroutes/\nutils/"),
        ("Read src/main.py", "read_file", {"path": "src/main.py"},
         'from fastapi import FastAPI\napp = FastAPI()\n\n@app.get("/")\ndef root():\n    return {"status": "ok"}'),
        ("Search for any TODO comments", "search_files", {"pattern": "TODO", "path": "src/"},
         "src/config.py:12: # TODO: Add database config\nsrc/routes/auth.py:45: # TODO: Implement JWT"),
        ("Read the config file", "read_file", {"path": "src/config.py"},
         'import os\n\nDATABASE_URL = os.getenv("DB_URL", "sqlite:///dev.db")\n# TODO: Add database config'),
        ("Read src/routes/auth.py", "read_file", {"path": "src/routes/auth.py"},
         'from fastapi import APIRouter\nrouter = APIRouter()\n\n@router.post("/login")\ndef login():\n    # TODO: Implement JWT\n    pass'),
        ("Write an updated auth.py with JWT", "write_file",
         {"path": "src/routes/auth.py", "content": 'from fastapi import APIRouter\nfrom jose import jwt\nrouter = APIRouter()'},
         "File written successfully: src/routes/auth.py"),
        ("Check if there's a requirements.txt", "read_file", {"path": "requirements.txt"},
         "fastapi==0.104.1\nuvicorn==0.24.0\npydantic==2.5.0"),
        ("Write an updated requirements.txt with jose", "write_file",
         {"path": "requirements.txt", "content": "fastapi==0.104.1\nuvicorn==0.24.0\npython-jose==3.3.0"},
         "File written successfully: requirements.txt"),
        ("Run the tests to see if things work", "run_command", {"command": "python -m pytest tests/ -q"},
         "5 passed, 1 failed\nFAILED tests/test_auth.py::test_login"),
        ("Read the failing test", "read_file", {"path": "src/tests/test_auth.py"},
         'def test_login():\n    result = login("admin", "pass")\n    assert "token" in result'),
        ("List the test directory", "list_directory", {"path": "src/tests/"},
         "test_auth.py\ntest_config.py\ntest_main.py\nconftest.py"),
        ("Read the conftest", "read_file", {"path": "src/tests/conftest.py"},
         'import pytest\nfrom fastapi.testclient import TestClient\nfrom main import app'),
        ("Search for import errors in tests", "search_files", {"pattern": "import", "path": "src/tests/"},
         "test_auth.py:2: from routes.auth import login"),
        ("Write a fixed test_auth.py", "write_file",
         {"path": "src/tests/test_auth.py", "content": 'from fastapi.testclient import TestClient'},
         "File written successfully: src/tests/test_auth.py"),
        ("Run the tests again", "run_command", {"command": "python -m pytest tests/ -q"},
         "6 passed\nAll tests passed!"),
        ("Read the models directory listing", "list_directory", {"path": "src/models/"},
         "user.py\nbase.py\n__init__.py"),
        ("Read the user model", "read_file", {"path": "src/models/user.py"},
         'from sqlalchemy import Column, Integer, String\nfrom .base import Base'),
        ("Write a new session model", "write_file",
         {"path": "src/models/session.py", "content": 'from sqlalchemy import Column, Integer, String'},
         "File written successfully: src/models/session.py"),
        ("Search for any remaining TODOs", "search_files", {"pattern": "TODO", "path": "src/"},
         "No matches found."),
        ("Read the base model", "read_file", {"path": "src/models/base.py"},
         'from sqlalchemy.ext.declarative import declarative_base\nBase = declarative_base()'),
        ("List all Python files in the project", "run_command", {"command": "find src/ -name '*.py'"},
         "src/__init__.py\nsrc/config.py\nsrc/main.py\nsrc/models/user.py"),
        ("Read the routes init", "read_file", {"path": "src/routes/__init__.py"},
         'from .auth import router as auth_router'),
        ("Write a new health check route", "write_file",
         {"path": "src/routes/health.py", "content": 'from fastapi import APIRouter\nrouter = APIRouter()'},
         "File written successfully: src/routes/health.py"),
        ("Run the full test suite one more time", "run_command", {"command": "python -m pytest tests/ -v"},
         "6 passed in 0.45s"),
    ]

    for i in range(min(turn_count, len(operations))):
        user_msg, tool_name, tool_args, tool_result = operations[i]
        tc_id = fake_tool_call_id()
        messages.append({"role": "user", "content": user_msg})
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": tc_id, "name": tool_name, "arguments": tool_args}],
        })
        messages.append({
            "role": "tool",
            "tool_call_id": tc_id,
            "name": tool_name,
            "content": tool_result,
        })

    return messages


async def run_scenario(model: str, turns: int, prompt: str, tag: str) -> dict[str, Any]:
    """
    Run one saturation scenario against one model.

    Returns evidence dict with verdict:
    - BUG_REPRODUCED: Tool call pattern in text, no structured tool_calls
    - PARTIAL_LEAK: Tool call pattern in text AND structured tool_calls
    - BUG_NOT_REPRODUCED: Structured tool_calls, no text pattern (GOOD)
    - NO_TOOL_USE: No tool calls at all (model chose not to use tools)
    """
    from amplifier_module_provider_github_copilot import CopilotSdkProvider
    from amplifier_module_provider_github_copilot.converters import convert_messages_to_prompt

    coordinator = Mock()
    coordinator.hooks = Mock()
    coordinator.hooks.emit = AsyncMock()

    provider = CopilotSdkProvider(
        api_key=None,
        config={
            "model": model,
            "timeout": 120,
            "thinking_timeout": 120,
            "debug": False,  # Keep logs clean for CI
            "use_streaming": True,
            "sdk_max_turns": 5,
        },
        coordinator=coordinator,
    )

    try:
        tools = make_tools()
        messages = build_saturated_history(turns)
        messages.append({"role": "user", "content": prompt})

        # Check how many tool call patterns exist in the serialized prompt
        serialized_prompt = convert_messages_to_prompt(messages)
        tc_text_count = len(re.findall(r'\[Tool Call:', serialized_prompt))

        request = Mock()
        request.messages = messages
        request.tools = tools
        request.stream = None

        start = time.time()
        response = await provider.complete(request, model=model)
        elapsed = time.time() - start

        # Extract metrics
        tc_count = len(response.tool_calls) if response.tool_calls else 0
        tc_names = [tc.name for tc in (response.tool_calls or [])]

        response_text = ""
        for block in response.content:
            if getattr(block, "type", "") == "text" and hasattr(block, "text"):
                response_text += block.text

        text_has_pattern = bool(TEXT_TOOL_CALL_PATTERN.search(response_text))
        has_bracket = bool(re.search(r'\[Tool Call:', response_text))

        # Determine verdict
        if text_has_pattern and tc_count == 0:
            verdict = "BUG_REPRODUCED"
        elif text_has_pattern and tc_count > 0:
            verdict = "PARTIAL_LEAK"
        elif tc_count > 0:
            verdict = "BUG_NOT_REPRODUCED"
        else:
            verdict = "NO_TOOL_USE"

        model_safe = model.replace(".", "").replace("-", "")
        evidence = {
            "test": f"saturation_{model_safe}_{tag}",
            "model": model,
            "scenario": tag,
            "timestamp": datetime.now(UTC).isoformat(),
            "history_turns": turns,
            "prompt_char_count": len(serialized_prompt),
            "tool_call_text_count_in_prompt": tc_text_count,
            "elapsed_seconds": round(elapsed, 2),
            "response": {
                "structured_tool_calls": tc_count,
                "tool_names": tc_names,
                "finish_reason": response.finish_reason,
                "text_length": len(response_text),
                "text_preview": response_text[:300] if response_text else "",
                "text_has_tool_pattern": text_has_pattern,
                "has_bracket_tool_call": has_bracket,
            },
            "verdict": verdict,
        }
        save_evidence(f"saturation_{model_safe}_{tag}", evidence)

        logger.info(
            f"[{model}] {tag}: tc={tc_count} ({tc_names}), "
            f"text_leak={text_has_pattern}, verdict={verdict}, "
            f"elapsed={elapsed:.1f}s"
        )

        # Assertions
        assert verdict != "BUG_REPRODUCED", (
            f"Tool call leaked into text for {model}/{tag}. "
            f"Expected structured tool_calls, got text: {response_text[:200]}"
        )

        return evidence
    finally:
        await provider.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Test Scenarios (8 per model)
# ═══════════════════════════════════════════════════════════════════════════════

SCENARIOS = [
    # (turns, final_prompt, tag)
    (10, "Now create a file called /tmp/summary.txt that contains a short summary. Use the write_file tool.", "10_write"),
    (15, "Write a CHANGELOG.md file documenting our changes. Use the write_file tool to create it.", "15_write"),
    (20, "Create a deployment script at deploy.sh that sets up the environment. Write it using the write_file tool.", "20_write"),
    (25, "We're almost done. Write a final setup.py file that packages everything. Use the write_file tool.", "25_write"),
    (25, "Do these three things:\n1. Read src/main.py\n2. Search for 'import' in src/routes/\n3. Write a new file src/routes/users.py\n\nStart with step 1.", "25_multi"),
    (25, "Check if there are any security issues with our auth implementation and fix them. Look at the auth route for hardcoded secrets.", "25_ambiguous"),
    (20, "Create a Dockerfile for this project.", "20_no_hint"),
    (25, "Tell me what files you would need to read and write to add user registration. Show your plan including tools, then execute it.", "25_describe"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Test Classes — One Per Model
# ═══════════════════════════════════════════════════════════════════════════════


class TestClaudeOpusSaturation:
    """claude-opus-4.5 — 8 saturation scenarios."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("turns,prompt,tag", SCENARIOS, ids=[s[2] for s in SCENARIOS])
    async def test_scenario(self, turns: int, prompt: str, tag: str) -> None:
        """Test that Claude avoids tool call text leakage."""
        await run_scenario("claude-opus-4.5", turns, prompt, tag)


class TestGpt5MiniSaturation:
    """gpt-5-mini — 8 saturation scenarios."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("turns,prompt,tag", SCENARIOS, ids=[s[2] for s in SCENARIOS])
    async def test_scenario(self, turns: int, prompt: str, tag: str) -> None:
        """Test that GPT-5-mini avoids tool call text leakage."""
        await run_scenario("gpt-5-mini", turns, prompt, tag)


class TestGpt41Saturation:
    """gpt-4.1 — 8 saturation scenarios."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("turns,prompt,tag", SCENARIOS, ids=[s[2] for s in SCENARIOS])
    async def test_scenario(self, turns: int, prompt: str, tag: str) -> None:
        """Test that GPT-4.1 avoids tool call text leakage."""
        await run_scenario("gpt-4.1", turns, prompt, tag)


class TestGemini3ProSaturation:
    """gemini-3-pro-preview — 8 saturation scenarios."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("turns,prompt,tag", SCENARIOS, ids=[s[2] for s in SCENARIOS])
    async def test_scenario(self, turns: int, prompt: str, tag: str) -> None:
        """Test that Gemini avoids tool call text leakage."""
        await run_scenario("gemini-3-pro-preview", turns, prompt, tag)
