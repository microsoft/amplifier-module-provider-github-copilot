"""
Shared test fixtures for Copilot SDK Provider tests.

This module provides common test fixtures, mocks, and utilities
for testing the Copilot SDK provider module.

NOTE: Windows + pytest-asyncio can cause KeyboardInterrupt on cleanup.
The event_loop_policy fixture below addresses this by using
WindowsSelectorEventLoopPolicy when available.
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest


@pytest.fixture(autouse=True)
def mock_cache_home(tmp_path, monkeypatch):
    """
    Auto-use fixture that isolates ALL tests from the real disk cache.

    This prevents tests from reading/writing to ~/.amplifier/cache/ which
    would cause non-deterministic behavior when the provider loads cached
    model data in __init__.

    Each test gets a fresh temp directory as its "home", so:
    - Cache file path becomes: tmp_path/.amplifier/cache/github-copilot-models.json
    - No interference between tests
    - No interference with real user cache
    """
    monkeypatch.setattr(
        "amplifier_module_provider_github_copilot.model_cache.Path.home",
        lambda: tmp_path,
    )
    return tmp_path


@pytest.fixture(autouse=True)
def reset_singleton_state():
    """Reset module-level singleton state before each test.

    Required for test isolation: module-level variables persist across
    tests in the same pytest session. Without this reset, singleton tests
    bleed state into each other.

    Uses hasattr guards because the singleton attributes don't exist yet
    until Task 4 adds them to __init__.py. Guards make this fixture safe
    to land before the implementation.
    """
    import amplifier_module_provider_github_copilot as mod

    # Guard: attributes may not exist until implementation is added (Task 4)
    if hasattr(mod, "_shared_client"):
        mod._shared_client = None  # type: ignore[attr-defined]
    if hasattr(mod, "_shared_client_refcount"):
        mod._shared_client_refcount = 0  # type: ignore[attr-defined]
    if hasattr(mod, "_shared_client_lock"):
        mod._shared_client_lock = None  # type: ignore[attr-defined]
    yield
    if hasattr(mod, "_shared_client"):
        mod._shared_client = None  # type: ignore[attr-defined]
    if hasattr(mod, "_shared_client_refcount"):
        mod._shared_client_refcount = 0  # type: ignore[attr-defined]
    if hasattr(mod, "_shared_client_lock"):
        mod._shared_client_lock = None  # type: ignore[attr-defined]


# Fix for Windows asyncio cleanup issues causing KeyboardInterrupt
# See: https://github.com/pytest-dev/pytest-asyncio/issues/671
if sys.platform == "win32":
    import asyncio

    @pytest.fixture(scope="session")
    def event_loop_policy():
        """Use WindowsSelectorEventLoopPolicy to avoid ProactorEventLoop cleanup issues."""
        return asyncio.WindowsSelectorEventLoopPolicy()


@pytest.fixture
def mock_copilot_client():
    """
    Mock CopilotClient for unit tests.

    Returns a mock that simulates the Copilot SDK CopilotClient class.

    NOTE: By default, models do NOT support reasoning_effort to match
    real-world Copilot SDK behavior (as of 2026-02). The SDK returns
    reasoning_effort=False for Claude Opus 4.5.
    """
    client = AsyncMock()

    # Mock list_models - DEFAULT: reasoning_effort=False (matches real SDK)
    mock_model = Mock()
    mock_model.id = "claude-opus-4.5"
    mock_model.name = "Claude Opus 4.5"
    mock_model.provider = None  # SDK doesn't provide this
    mock_model.vendor = None  # SDK doesn't provide this
    mock_model.capabilities = Mock()
    mock_model.capabilities.supports = Mock()
    mock_model.capabilities.supports.vision = True
    mock_model.capabilities.supports.reasoning_effort = False  # IMPORTANT: matches real SDK
    mock_model.capabilities.limits = Mock()
    mock_model.capabilities.limits.max_context_window_tokens = 200000
    mock_model.capabilities.limits.max_prompt_tokens = 150000
    mock_model.supported_reasoning_efforts = None
    mock_model.default_reasoning_effort = None

    client.list_models = AsyncMock(return_value=[mock_model])

    # Mock get_auth_status
    auth_status = Mock()
    auth_status.isAuthenticated = True
    auth_status.login = "test-user"
    client.get_auth_status = AsyncMock(return_value=auth_status)

    # Mock create_session
    mock_session = AsyncMock()
    mock_session.session_id = "test-session-123"
    mock_session.destroy = AsyncMock()

    # Mock send_and_wait response
    mock_response = Mock()
    mock_response.type = "assistant.message"
    mock_response.data = Mock()
    mock_response.data.content = "Hello! How can I help you?"
    mock_response.data.tool_requests = None
    mock_response.data.input_tokens = 100
    mock_response.data.output_tokens = 50

    mock_session.send_and_wait = AsyncMock(return_value=mock_response)
    client.create_session = AsyncMock(return_value=mock_session)

    # Mock start/stop
    client.start = AsyncMock()
    client.stop = AsyncMock(return_value=[])

    return client


@pytest.fixture
def mock_copilot_client_with_reasoning():
    """
    Mock CopilotClient with a model that DOES support reasoning.

    Use this fixture to test extended thinking behavior WITH supported models.
    """
    client = AsyncMock()

    mock_model = Mock()
    mock_model.id = "o3-reasoning"
    mock_model.name = "O3 Reasoning Model"
    mock_model.provider = "openai"
    mock_model.vendor = None
    mock_model.capabilities = Mock()
    mock_model.capabilities.supports = Mock()
    mock_model.capabilities.supports.vision = True
    mock_model.capabilities.supports.reasoning_effort = True  # This model DOES support it
    mock_model.capabilities.limits = Mock()
    mock_model.capabilities.limits.max_context_window_tokens = 200000
    mock_model.capabilities.limits.max_prompt_tokens = 150000
    mock_model.supported_reasoning_efforts = ["low", "medium", "high"]
    mock_model.default_reasoning_effort = "medium"

    client.list_models = AsyncMock(return_value=[mock_model])

    # Mock get_auth_status
    auth_status = Mock()
    auth_status.isAuthenticated = True
    auth_status.login = "test-user"
    client.get_auth_status = AsyncMock(return_value=auth_status)

    # Mock create_session
    mock_session = AsyncMock()
    mock_session.session_id = "test-session-123"
    mock_session.destroy = AsyncMock()

    mock_response = Mock()
    mock_response.type = "assistant.message"
    mock_response.data = Mock()
    mock_response.data.content = "Hello! How can I help you?"
    mock_response.data.tool_requests = None
    mock_response.data.input_tokens = 100
    mock_response.data.output_tokens = 50

    mock_session.send_and_wait = AsyncMock(return_value=mock_response)
    client.create_session = AsyncMock(return_value=mock_session)

    # Mock start/stop
    client.start = AsyncMock()
    client.stop = AsyncMock(return_value=[])

    return client


@pytest.fixture
def mock_copilot_session():
    """Mock CopilotSession for testing."""
    session = AsyncMock()
    session.session_id = "test-session-123"
    session.destroy = AsyncMock()

    # Default response
    mock_response = Mock()
    mock_response.type = "assistant.message"
    mock_response.data = Mock()
    mock_response.data.content = "Test response"
    mock_response.data.tool_requests = None

    session.send_and_wait = AsyncMock(return_value=mock_response)
    return session


@pytest.fixture
def mock_coordinator():
    """
    Mock ModuleCoordinator for testing.

    Provides a mock coordinator that captures mount calls and events.
    """
    coordinator = Mock()
    coordinator.mounted_providers = {}

    async def mock_mount(category: str, provider: Any, name: str) -> None:
        if category == "providers":
            coordinator.mounted_providers[name] = provider

    coordinator.mount = AsyncMock(side_effect=mock_mount)

    # Mock hooks for event emission
    coordinator.hooks = Mock()
    coordinator.hooks.emit = AsyncMock()

    return coordinator


@pytest.fixture
def sample_messages():
    """Sample conversation messages for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
        {"role": "user", "content": "Can you help me with Python?"},
    ]


@pytest.fixture
def sample_messages_with_tools():
    """Sample messages including tool calls and results."""
    return [
        {"role": "system", "content": "You are a helpful assistant with access to tools."},
        {"role": "user", "content": "Read the file test.py"},
        {
            "role": "assistant",
            "content": "I'll read that file for you.",
            "tool_calls": [
                {
                    "id": "call_123",
                    "name": "read_file",
                    "arguments": {"path": "test.py"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_123",
            "tool_name": "read_file",
            "content": "print('Hello, World!')",
        },
        {"role": "assistant", "content": "The file contains a simple print statement."},
    ]


@pytest.fixture
def mock_tool_response():
    """Mock response with tool calls."""
    response = Mock()
    response.type = "assistant.message"
    response.data = Mock()
    response.data.content = "I'll help you with that."

    # Create tool request mock with explicit attribute assignments
    # (Mock(name="read_file") sets Mock's internal name, not .name attribute)
    tool_request = Mock()
    tool_request.tool_call_id = "call_456"
    tool_request.name = "read_file"
    tool_request.arguments = {"path": "example.py"}
    tool_request.type = "tool"

    response.data.tool_requests = [tool_request]
    response.data.input_tokens = 150
    response.data.output_tokens = 75
    return response


@pytest.fixture
def provider_config():
    """Default provider configuration for testing."""
    return {
        "model": "claude-opus-4.5",
        "timeout": 60.0,
        "debug": True,
        "debug_truncate_length": 100,
        "use_streaming": False,  # Use non-streaming mode for simpler test mocking
        "max_retries": 0,  # Disable retries in tests to avoid real asyncio.sleep delays
    }


class MockCopilotClient:
    """
    Mock implementation of CopilotClient for integration testing.

    This class can be used to simulate various scenarios including
    errors, timeouts, and different response types.
    """

    def __init__(self, responses: list[Any] | None = None):
        self.responses = responses or []
        self.response_index = 0
        self.calls: list[dict[str, Any]] = []
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> list:
        self.stopped = True
        return []

    async def get_auth_status(self):
        status = Mock()
        status.isAuthenticated = True
        status.login = "mock-user"
        return status

    async def list_models(self):
        """Return models WITHOUT reasoning support by default (matches real SDK)."""
        model = Mock()
        model.id = "claude-opus-4.5"
        model.name = "Claude Opus 4.5"
        model.provider = None
        model.vendor = None
        model.capabilities = Mock()
        model.capabilities.supports = Mock(vision=True, reasoning_effort=False)  # Matches real SDK
        model.capabilities.limits = Mock(
            max_context_window_tokens=200000,
            max_prompt_tokens=150000,
        )
        model.supported_reasoning_efforts = None
        model.default_reasoning_effort = None
        return [model]

    async def create_session(self, config: dict[str, Any] | None = None):
        self.calls.append({"method": "create_session", "config": config})
        return MockCopilotSession(self._get_next_response())

    def _get_next_response(self):
        if self.response_index < len(self.responses):
            response = self.responses[self.response_index]
            self.response_index += 1
            return response
        # Default response
        response = Mock()
        response.type = "assistant.message"
        response.data = Mock()
        response.data.content = "Default mock response"
        response.data.tool_requests = None
        return response


class MockCopilotSession:
    """Mock session for integration testing."""

    def __init__(self, response: Any):
        self.session_id = "mock-session-id"
        self.response = response
        self.destroyed = False
        self.messages: list[dict[str, Any]] = []

    async def send_and_wait(self, options: dict[str, Any], timeout: float | None = None):
        self.messages.append(options)
        return self.response

    async def destroy(self) -> None:
        self.destroyed = True


@pytest.fixture
def mock_client_class(monkeypatch):
    """
    Fixture that patches the CopilotClient import.

    Use this to test with MockCopilotClient without actual SDK calls.
    """

    def _create_mock(responses: list[Any] | None = None):
        mock_client = MockCopilotClient(responses)

        def mock_import(*args, **kwargs):
            return mock_client

        return mock_client, mock_import

    return _create_mock


# =============================================================================
# SDK Bundled Binary Mocking Utilities
# =============================================================================


@pytest.fixture
def disable_sdk_bundled_binary():
    """
    Context manager fixture that makes SDK bundled binary discovery fail.

    Use this fixture in tests that want to test the shutil.which fallback path
    of _find_copilot_cli(). Without this, SDK 0.1.28+ bundles the binary and
    the function finds it before checking shutil.which.

    Usage:
        def test_fallback_to_path(disable_sdk_bundled_binary):
            with disable_sdk_bundled_binary():
                # Now _find_copilot_cli will use shutil.which fallback
                ...
    """
    from contextlib import contextmanager
    from unittest.mock import Mock, patch

    @contextmanager
    def _disable():
        # Create a mock copilot module with a __file__ that doesn't have binary
        mock_copilot_mod = Mock()
        mock_copilot_mod.__file__ = "/nonexistent/fake/copilot/__init__.py"

        # Patch sys.modules so import copilot returns our mock
        # AND patch Path.exists to return False for the bin path
        with patch.dict("sys.modules", {"copilot": mock_copilot_mod}):
            # Make the binary path check fail
            original_exists = __import__("pathlib").Path.exists

            def patched_exists(self):
                if "copilot" in str(self) and "bin" in str(self):
                    return False
                return original_exists(self)

            with patch("pathlib.Path.exists", patched_exists):
                yield

    return _disable
