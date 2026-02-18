"""
Live integration tests for SDK Driver architecture.

These tests validate the SDK Driver's behavior with real Copilot SDK calls:
- First-turn tool capture with immediate abort
- Circuit breaker protection against runaway loops
- Tool deduplication across turns
- Text-only (no tool) responses

Tests are skipped by default. Run with:
    RUN_LIVE_TESTS=1 python -m pytest tests/integration/test_live_sdk_driver.py -v -s

Prerequisites:
    - Copilot CLI installed and in PATH
    - Valid GitHub Copilot authentication
    - Network access

WARNING: These tests make real API calls.

Evidence base:
    - Session a1a0af17: 305 turns, 607 tool calls
    - Deny+Destroy architecture decision
    - Agent loop problem analysis
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from amplifier_module_provider_github_copilot import CopilotSdkProvider

logger = logging.getLogger(__name__)

# Skip all tests in this module unless explicitly enabled
pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_LIVE_TESTS"),
    reason="Live tests disabled. Set RUN_LIVE_TESTS=1 to run.",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def create_mock_tool(
    name: str,
    description: str,
    parameters: dict[str, Any] | None = None,
) -> Mock:
    """
    Create a mock tool specification for testing.

    Args:
        name: Tool name (e.g., "read_file")
        description: Human-readable tool description
        parameters: JSON Schema parameters dict

    Returns:
        Mock object with name, description, and parameters attributes
    """
    tool = Mock()
    tool.name = name
    tool.description = description
    tool.parameters = parameters or {"type": "object", "properties": {}}
    return tool


def create_mock_request(
    messages: list[dict[str, Any]],
    tools: list[Any] | None = None,
) -> Mock:
    """
    Create a mock ChatRequest for testing.

    Args:
        messages: List of message dicts with role/content
        tools: Optional list of tool specs

    Returns:
        Mock request with messages and tools attributes
    """
    request = Mock()
    request.messages = messages
    request.tools = tools
    request.stream = None
    return request


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
async def live_provider():
    """
    Create a live provider instance with SDK Driver enabled.

    Uses streaming mode (required for SDK Driver event-based tool capture)
    and debug logging for test observability.
    """
    coordinator = Mock()
    coordinator.hooks = Mock()
    coordinator.hooks.emit = AsyncMock()

    provider = CopilotSdkProvider(
        api_key=None,  # Copilot uses GitHub auth, not API key
        config={
            "model": "claude-opus-4.5",
            "timeout": 120,
            "debug": True,
            "use_streaming": True,  # SDK Driver needs streaming
        },
        coordinator=coordinator,
    )
    yield provider
    await provider.close()


@pytest.fixture
async def live_provider_strict():
    """
    Create a live provider with strict circuit breaker (max_turns=1).

    Used for testing that the circuit breaker trips quickly.
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
            "sdk_max_turns": 1,  # Strict: trip after first turn
        },
        coordinator=coordinator,
    )
    yield provider
    await provider.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Test Class
# ═══════════════════════════════════════════════════════════════════════════════


class TestLiveSdkDriver:
    """
    Live integration tests for SDK Driver architecture.

    These tests validate that the SDK Driver correctly controls the
    Copilot SDK's internal agent loop with real API calls. The key
    behaviors tested are:

    1. Text-only responses return cleanly without abort
    2. Tool calls are captured from the first turn
    3. Session is aborted after tool capture (not 305 turns later)
    4. Circuit breaker respects max_turns configuration
    5. Duplicate tool calls are deduplicated
    """

    @pytest.mark.asyncio
    async def test_sdk_driver_text_only_response(self, live_provider: CopilotSdkProvider) -> None:
        """
        Send a simple prompt with no tool triggers.

        Verifies that when the LLM responds with text only:
        - Response has text content
        - No tool calls are returned
        - finish_reason is 'end_turn'
        - Completes in reasonable time (no loop hang)
        """
        request = create_mock_request(
            messages=[
                {
                    "role": "user",
                    "content": "What is 2 + 2? Reply with just the number.",
                },
            ],
            tools=[
                create_mock_tool(
                    name="calculate",
                    description="Perform arithmetic calculations",
                    parameters={
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string"},
                        },
                    },
                ),
            ],
        )

        start = time.time()
        response = await live_provider.complete(request)
        elapsed = time.time() - start

        # Verify response is valid
        assert response is not None
        assert response.content is not None
        assert len(response.content) > 0

        # Model may choose EITHER text-only OR tool_use - both are valid behaviors
        # Check tool_calls FIRST to handle both cases correctly
        if response.tool_calls:
            # Model chose to use the calculate tool - this is valid behavior
            assert response.finish_reason == "tool_use"
            tool_names = [tc.name for tc in response.tool_calls]
            logger.info(f"Model chose to use tools (valid behavior): {tool_names}")
            # When using tools, there may or may not be accompanying text
        else:
            # Model chose text-only response - validate text content
            assert response.finish_reason == "end_turn"
            text_blocks = [
                block for block in response.content if getattr(block, "type", None) == "text"
            ]
            assert len(text_blocks) > 0, "Expected at least one text block for text-only response"
            text = text_blocks[0].text
            logger.info(f"Response text: {text}")
            assert "4" in text, f"Expected '4' in response, got: {text}"

        logger.info(f"Completed in {elapsed:.2f}s")

    @pytest.mark.asyncio
    async def test_sdk_driver_tool_capture(self, live_provider: CopilotSdkProvider) -> None:
        """
        Send a prompt that triggers tool calls.

        Verifies that:
        - Tool calls are captured from the SDK event stream
        - finish_reason is 'tool_use'
        - Each tool call has a valid id, name, and arguments
        - Response completes without the SDK running 305 turns
        """
        request = create_mock_request(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You have access to a read_file tool. "
                        "When asked to read a file, you MUST use the read_file tool. "
                        "Do not simulate the file contents."
                    ),
                },
                {
                    "role": "user",
                    "content": "Please read the file called config.json",
                },
            ],
            tools=[
                create_mock_tool(
                    name="read_file",
                    description="Read the contents of a file at the given path",
                    parameters={
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to the file to read",
                            },
                        },
                        "required": ["path"],
                    },
                ),
            ],
        )

        start = time.time()
        response = await live_provider.complete(request)
        elapsed = time.time() - start

        assert response is not None
        logger.info(
            f"Response: finish_reason={response.finish_reason}, "
            f"tool_calls={len(response.tool_calls) if response.tool_calls else 0}, "
            f"elapsed={elapsed:.2f}s"
        )

        # Verify tool calls captured
        if not response.tool_calls:
            pytest.skip("Model did not request tool call (non-deterministic behavior)")

        assert response.finish_reason == "tool_use"

        for tc in response.tool_calls:
            assert tc.id, f"Tool call missing id: {tc}"
            assert tc.name, f"Tool call missing name: {tc}"
            assert isinstance(tc.arguments, dict), (
                f"Tool call arguments should be dict, got {type(tc.arguments)}"
            )
            logger.info(f"Tool call: {tc.name}({tc.arguments})")

        # Verify we got a reasonable number of tool calls (not 607)
        assert len(response.tool_calls) <= 10, (
            f"Too many tool calls ({len(response.tool_calls)}), "
            f"SDK Driver may not be aborting correctly"
        )

    @pytest.mark.asyncio
    async def test_sdk_driver_abort_after_capture(self, live_provider: CopilotSdkProvider) -> None:
        """
        Verify tool capture completes quickly with abort.

        The key metric: with SDK Driver, a tool-triggering request should
        complete in seconds, NOT the 20 minutes observed in the forensic
        incident (Session a1a0af17).

        Acceptance criteria:
        - Completes in under 60 seconds
        - Turn count is reasonable (≤3, not 305)
        """
        request = create_mock_request(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You have tools available. When asked to create a file, "
                        "use the write_file tool. Do not simulate the action."
                    ),
                },
                {
                    "role": "user",
                    "content": "Create a file called hello.txt with the content 'Hello World'",
                },
            ],
            tools=[
                create_mock_tool(
                    name="write_file",
                    description="Write content to a file",
                    parameters={
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                    },
                ),
            ],
        )

        start = time.time()
        response = await live_provider.complete(request)
        elapsed = time.time() - start

        logger.info(
            f"Abort-after-capture test: elapsed={elapsed:.2f}s, "
            f"finish_reason={response.finish_reason}, "
            f"tool_calls={len(response.tool_calls) if response.tool_calls else 0}"
        )

        # CRITICAL: Must complete quickly (not 20 minutes like the incident)
        assert elapsed < 60, (
            f"Request took {elapsed:.1f}s — SDK Driver may not be aborting. "
            f"Expected <60s. Forensic incident took ~1200s."
        )

        # If model returned tool calls, verify reasonable count
        if response.tool_calls:
            assert len(response.tool_calls) <= 5, (
                f"Got {len(response.tool_calls)} tool calls — possible accumulation (forensic: 607)"
            )

    @pytest.mark.asyncio
    async def test_sdk_driver_respects_max_turns(
        self, live_provider_strict: CopilotSdkProvider
    ) -> None:
        """
        Verify circuit breaker activates with sdk_max_turns=1.

        With max_turns=1, the circuit breaker should trip on the second
        turn (if the SDK retries after denial). The response should still
        return any tools captured from the first turn.
        """
        request = create_mock_request(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You have a search tool. When asked to search, use the search_web tool."
                    ),
                },
                {
                    "role": "user",
                    "content": "Search the web for 'Python asyncio tutorial'",
                },
            ],
            tools=[
                create_mock_tool(
                    name="search_web",
                    description="Search the web for information",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                        },
                        "required": ["query"],
                    },
                ),
            ],
        )

        start = time.time()

        # With max_turns=1, the provider may raise CopilotSdkLoopError
        # if the circuit breaker trips, OR it may return successfully
        # if first-turn capture + abort happens before trip.
        from amplifier_module_provider_github_copilot.exceptions import CopilotSdkLoopError

        try:
            response = await live_provider_strict.complete(request)
            elapsed = time.time() - start

            logger.info(
                f"max_turns=1 test completed in {elapsed:.2f}s: "
                f"finish_reason={response.finish_reason}, "
                f"tool_calls={len(response.tool_calls) if response.tool_calls else 0}"
            )

            # If we got a response, it should be reasonable
            assert response is not None
            if response.tool_calls:
                # First-turn capture succeeded before circuit breaker trip
                assert len(response.tool_calls) <= 5

        except CopilotSdkLoopError as e:
            elapsed = time.time() - start
            logger.info(f"Circuit breaker tripped as expected in {elapsed:.2f}s: {e}")
            # Circuit breaker tripping is valid behavior with max_turns=1
            assert e.max_turns == 1
            assert e.turn_count >= 1

    @pytest.mark.asyncio
    async def test_sdk_driver_deduplication(self, live_provider: CopilotSdkProvider) -> None:
        """
        Verify that tool call deduplication works with live SDK.

        Even if the SDK retries and produces the same tool call multiple
        times, the provider should return only unique tool calls.

        Validated by checking that no two returned tool calls have the
        same (name, arguments) pair.
        """
        request = create_mock_request(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You have tools for file operations. "
                        "When asked to read a file, use the read_file tool."
                    ),
                },
                {
                    "role": "user",
                    "content": "Read the file README.md",
                },
            ],
            tools=[
                create_mock_tool(
                    name="read_file",
                    description="Read file contents",
                    parameters={
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                        },
                        "required": ["path"],
                    },
                ),
            ],
        )

        response = await live_provider.complete(request)

        if not response.tool_calls:
            pytest.skip("Model did not request tool call")

        # Check for duplicates: no two tool calls should have same (name, arguments)
        seen: set[tuple[str, str]] = set()
        for tc in response.tool_calls:
            import json

            key = (tc.name, json.dumps(tc.arguments, sort_keys=True))
            assert key not in seen, (
                f"Duplicate tool call detected: {tc.name}({tc.arguments}). "
                f"Deduplication may not be working."
            )
            seen.add(key)

        logger.info(f"Deduplication check passed: {len(response.tool_calls)} unique tool call(s)")

    @pytest.mark.asyncio
    async def test_sdk_driver_multi_tool_capture(self, live_provider: CopilotSdkProvider) -> None:
        """
        Verify multiple distinct tools are captured in a single turn.

        Sends a prompt that should trigger the model to call multiple
        different tools. Validates that all distinct tool calls are
        captured from the first turn.
        """
        request = create_mock_request(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You have tools for file operations. "
                        "When asked to perform multiple actions, call all "
                        "the necessary tools."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "First read the file config.json, "
                        "then list the contents of the src directory."
                    ),
                },
            ],
            tools=[
                create_mock_tool(
                    name="read_file",
                    description="Read file contents at a given path",
                    parameters={
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                        },
                        "required": ["path"],
                    },
                ),
                create_mock_tool(
                    name="list_directory",
                    description="List the contents of a directory",
                    parameters={
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                        },
                        "required": ["path"],
                    },
                ),
            ],
        )

        start = time.time()
        response = await live_provider.complete(request)
        elapsed = time.time() - start

        logger.info(
            f"Multi-tool test: elapsed={elapsed:.2f}s, "
            f"tool_calls={len(response.tool_calls) if response.tool_calls else 0}"
        )

        if not response.tool_calls:
            pytest.skip("Model did not request tool calls")

        # Should have captured tool calls from first turn
        assert response.finish_reason == "tool_use"
        assert len(response.tool_calls) >= 1

        tool_names = [tc.name for tc in response.tool_calls]
        logger.info(f"Captured tools: {tool_names}")

        # All captured tools should be from our registered set
        registered_names = {"read_file", "list_directory"}
        for name in tool_names:
            assert name in registered_names, (
                f"Unexpected tool '{name}' — not in registered tools {registered_names}"
            )

        # Completes quickly (SDK Driver abort working)
        assert elapsed < 60, f"Took {elapsed:.1f}s — potential SDK loop issue"


# ═══════════════════════════════════════════════════════════════════════════════
# Pytest hooks for --run-live option
# ═══════════════════════════════════════════════════════════════════════════════


def pytest_addoption(parser: Any) -> None:
    """Add --run-live CLI option."""
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="Run live integration tests",
    )


def pytest_configure(config: Any) -> None:
    """Set RUN_LIVE_TESTS env var if --run-live passed."""
    if config.getoption("--run-live", default=False):
        os.environ["RUN_LIVE_TESTS"] = "1"
