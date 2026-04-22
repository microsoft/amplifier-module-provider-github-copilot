"""Behavioral tests for contracts/behaviors.md contract.

Contract: behaviors.md

Tests verify retry policy behaviors defined in the contract.

Contract Anchors:
- behaviors:Retry:MUST:1 — Respects max_attempts
- behaviors:Retry:MUST:4 — Only retries retryable errors
- behaviors:Retry:MUST:5 — Does NOT retry non-retryable errors
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from amplifier_core import ChatRequest
from amplifier_core.message_models import Message


def _make_standard_request(model: str = "claude-opus-4.5") -> MagicMock:
    """Create a standard mock ChatRequest for provider.complete() calls."""
    request = MagicMock(spec=ChatRequest)
    request.model = model
    request.messages = [MagicMock(spec=Message, role="user", content="test")]
    request.tools = None
    request.max_tokens = None
    request.temperature = None
    request.stop = None
    request.stream = None
    return request


# =============================================================================
# Contract Discovery Tests (verify contract and config exist)
# =============================================================================


# =============================================================================
# Retry Behavior Tests
# =============================================================================


class TestRetryBehavior:
    """Test retry behavior matches behaviors.md contract."""

    @pytest.mark.asyncio
    async def test_retryable_error_is_retried(self) -> None:
        """Contract: behaviors:Retry:MUST:4 — Retryable errors trigger retry.

        When SDK raises a retryable error (e.g., LLMTimeoutError with retryable=True),
        the provider should retry up to max_attempts before propagating the error.
        """
        from contextlib import asynccontextmanager

        from amplifier_core.llm_errors import LLMTimeoutError

        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        call_count = 0

        @asynccontextmanager
        async def mock_session_raises_retryable(**kwargs: Any):  # type: ignore[misc]
            """Mock session context manager that raises retryable error."""
            nonlocal call_count
            call_count += 1
            raise LLMTimeoutError("SDK timeout", retryable=True)
            yield  # Never reached, but required for generator syntax

        # Create provider and patch its _client
        provider = GitHubCopilotProvider()
        mock_client = MagicMock()
        mock_client.session = mock_session_raises_retryable
        provider._client = mock_client  # type: ignore[assignment]

        mock_request = MagicMock(spec=ChatRequest)
        mock_request.messages = [{"role": "user", "content": "test"}]
        mock_request.model = "test-model"

        with pytest.raises(LLMTimeoutError):
            await provider.complete(mock_request)

        # behaviors:Retry:MUST:1 — max_attempts is 3 per config
        # So we expect 3 calls (1 initial + 2 retries)
        assert call_count == 3, f"Expected 3 attempts, got {call_count}"

    @pytest.mark.asyncio
    async def test_non_retryable_error_not_retried(self) -> None:
        """behaviors:Retry:MUST:5 — Non-retryable errors fail immediately.

        When SDK raises a non-retryable error (e.g., AuthenticationError),
        the provider should NOT retry - fail immediately.
        """
        from contextlib import asynccontextmanager

        from amplifier_core.llm_errors import AuthenticationError

        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        call_count = 0

        @asynccontextmanager
        async def mock_session_raises_non_retryable(**kwargs: Any):  # type: ignore[misc]
            """Mock session context manager that raises non-retryable error."""
            nonlocal call_count
            call_count += 1
            raise AuthenticationError("Invalid token", retryable=False)
            yield  # Never reached

        provider = GitHubCopilotProvider()
        mock_client = MagicMock()
        mock_client.session = mock_session_raises_non_retryable
        provider._client = mock_client  # type: ignore[assignment]

        mock_request = MagicMock(spec=ChatRequest)
        mock_request.messages = [{"role": "user", "content": "test"}]
        mock_request.model = "test-model"

        with pytest.raises(AuthenticationError):
            await provider.complete(mock_request)

        # Non-retryable should fail immediately (1 call only)
        assert call_count == 1, f"Expected 1 attempt (no retry), got {call_count}"


# =============================================================================
# Retryable Error Classification Tests
# =============================================================================


class TestRetryableErrorClassification:
    """Verify error retryability matches behaviors.md table."""

    def test_auth_error_not_retryable(self) -> None:
        """behaviors:Retry:MUST:5 — AuthenticationError is not retryable."""
        from amplifier_core.llm_errors import AuthenticationError

        err = AuthenticationError("test")
        assert err.retryable is False

    def test_rate_limit_error_is_retryable(self) -> None:
        """behaviors:Retry:MUST:4 — RateLimitError is retryable."""
        from amplifier_core.llm_errors import RateLimitError

        err = RateLimitError("test")
        assert err.retryable is True

    def test_timeout_error_is_retryable(self) -> None:
        """behaviors:Retry:MUST:4 — LLMTimeoutError is retryable."""
        from amplifier_core.llm_errors import LLMTimeoutError

        err = LLMTimeoutError("test")
        assert err.retryable is True

    def test_content_filter_error_not_retryable(self) -> None:
        """behaviors:Retry:MUST:5 — ContentFilterError is not retryable."""
        from amplifier_core.llm_errors import ContentFilterError

        err = ContentFilterError("test")
        assert err.retryable is False

    def test_network_error_is_retryable(self) -> None:
        """behaviors:Retry:MUST:4 — NetworkError is retryable."""
        from amplifier_core.llm_errors import NetworkError

        err = NetworkError("test")
        assert err.retryable is True

    def test_provider_unavailable_is_retryable(self) -> None:
        """behaviors:Retry:MUST:4 — ProviderUnavailableError is retryable."""
        from amplifier_core.llm_errors import ProviderUnavailableError

        err = ProviderUnavailableError("test")
        assert err.retryable is True

    def test_abort_error_not_retryable(self) -> None:
        """behaviors:Retry:MUST:5 — AbortError is not retryable."""
        from amplifier_core.llm_errors import AbortError

        err = AbortError("test")
        assert err.retryable is False


# =============================================================================
# Contract Tests: behaviors:Streaming:MUST:1, behaviors:Models:MUST:1,2
# Added as part of mock quality audit
# =============================================================================


class TestStreamingBehavior:
    """Tests for behaviors:Streaming:MUST:1."""

    def test_ttft_warning_config_loaded(self) -> None:
        """TTFT warning threshold loads from config.

        Contract: behaviors:Streaming:MUST:1
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_streaming_config,
        )

        config = load_streaming_config()

        # Contract specifies lenient thresholds for SDK latency
        assert config.ttft_warning_ms == 15000
        assert config.event_queue_size == 10000


class TestBoundedQueueBehavior:
    """Tests for behaviors:Streaming:MUST:4 — MUST NOT block on queue full.

    Contract: behaviors:Streaming:MUST:4 — MUST NOT block on queue full (drop oldest)

    These tests verify BOTH code paths (production and test) use bounded queues:
    - provider.py:_execute_sdk_completion() — production path
    - completion.py:complete() — test path

    PR #32 Review: Found divergence where production path had unbounded queue.
    """

    # test_completion_path_uses_bounded_queue removed - completion.py deleted (Issue #6)

    def test_queue_size_matches_config(self) -> None:
        """behaviors:Streaming:MUST:4 — Queue size must come from config.

        Verifies the streaming config's event_queue_size is used (not hardcoded).
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_streaming_config,
        )

        config = load_streaming_config()

        # Config must define event_queue_size
        assert hasattr(config, "event_queue_size"), "Config must have event_queue_size"
        assert config.event_queue_size > 0, "event_queue_size must be positive"
        assert config.event_queue_size == 10000, "Contract specifies 10000 as default"

    @pytest.mark.asyncio
    async def test_queue_full_drops_without_blocking(self) -> None:
        """behaviors:Streaming:MUST:4 — QueueFull must drop, not block.

        This runtime test verifies the contract pattern works correctly:
        1. put_nowait() on full queue raises QueueFull
        2. We catch it and continue (don't block)
        3. Events are dropped, not lost silently

        The production code in provider.py uses this exact pattern.
        """
        import asyncio

        # Create a tiny bounded queue to quickly reach capacity
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=2)

        # Fill the queue to capacity
        queue.put_nowait("event_1")
        queue.put_nowait("event_2")
        assert queue.full(), "Queue should be full after maxsize events"

        # Attempting put_nowait on full queue MUST raise QueueFull
        dropped_count = 0
        for i in range(3):  # Try to add 3 more
            try:
                queue.put_nowait(f"overflow_{i}")
            except asyncio.QueueFull:
                # Contract: drop on full, don't block
                dropped_count += 1

        # All 3 overflow attempts should have been dropped
        assert dropped_count == 3, f"Expected 3 drops, got {dropped_count}"

        # Queue should still have exactly 2 events (original, not overflow)
        assert queue.qsize() == 2, "Queue should still have original 2 events"

        # Verify we can drain the queue (proves we didn't block)
        items: list[str] = []
        while not queue.empty():
            items.append(queue.get_nowait())

        assert items == ["event_1", "event_2"], "Queue should have original events"

    @pytest.mark.asyncio
    async def test_event_handler_logs_on_queue_full(self, caplog: pytest.LogCaptureFixture) -> None:
        """behaviors:Streaming:MUST:4 — EventRouter MUST log when dropping events.

        Runtime behavioral test using the ACTUAL EventRouter class (not a local copy).
        Verifies production logging occurs when bounded queue is full.
        """
        import asyncio
        import logging
        import time

        from amplifier_module_provider_github_copilot.event_router import EventRouter
        from amplifier_module_provider_github_copilot.sdk_adapter import ToolCaptureHandler
        from amplifier_module_provider_github_copilot.streaming import load_event_config
        from tests.fixtures.sdk_mocks import text_delta_event

        caplog.set_level(logging.DEBUG)

        # Tiny bounded queue to easily trigger QueueFull
        queue: asyncio.Queue[object] = asyncio.Queue(maxsize=2)
        idle_event = asyncio.Event()
        error_holder: list[Exception] = []
        usage_holder: list[dict[str, int | None]] = []
        capture_handler = ToolCaptureHandler()
        ttft_state: dict[str, object] = {"checked": False, "start_time": time.time()}
        event_config = load_event_config()

        router = EventRouter(
            queue=queue,
            idle_event=idle_event,
            error_holder=error_holder,
            usage_holder=usage_holder,
            capture_handler=capture_handler,
            ttft_state=ttft_state,
            ttft_threshold_ms=15000,
            event_config=event_config,
            emit_streaming_content=lambda _: None,
        )

        # Fill queue to capacity
        router(text_delta_event("hello"))
        router(text_delta_event("world"))
        assert queue.full(), "Queue should be full after maxsize events"

        # Overflow: 5 more events should be dropped and logged
        for i in range(5):
            router(text_delta_event(f"overflow_{i}"))

        # Verify logging occurred for each drop
        drop_logs = [r for r in caplog.records if "queue full" in r.message.lower()]
        assert len(drop_logs) == 5, (
            f"Expected 5 'queue full' log messages, got {len(drop_logs)}. "
            f"Records: {[r.message for r in caplog.records]}"
        )

        # Queue still has original 2 events (drops didn't corrupt it)
        assert queue.qsize() == 2, "Queue should still have 2 original events"

    @pytest.mark.asyncio
    async def test_production_queue_is_bounded(self) -> None:
        """behaviors:Streaming:MUST:4 — Production path creates bounded queue.

        Verifies provider._execute_sdk_completion() creates asyncio.Queue with
        maxsize > 0 (from streaming config). If reverted to Queue() with no
        maxsize, this test fails.

        Contract: behaviors:Streaming:MUST:4
        """
        import asyncio
        from unittest.mock import patch

        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from tests.fixtures.sdk_mocks import MockCopilotClientWrapper, text_delta_event

        captured_maxsizes: list[int] = []
        original_queue = asyncio.Queue

        def tracking_queue(maxsize: int = 0, **kwargs: object) -> asyncio.Queue[object]:  # type: ignore[type-arg]
            captured_maxsizes.append(maxsize)
            return original_queue(maxsize=maxsize, **kwargs)  # type: ignore[return-value]

        mock_client = MockCopilotClientWrapper(events=[text_delta_event("hello")])
        provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]

        request = _make_standard_request()

        with patch(
            "amplifier_module_provider_github_copilot.provider.asyncio.Queue",
            side_effect=tracking_queue,
        ):
            await provider.complete(request)

        # At least one Queue must have been created with maxsize > 0
        assert any(m > 0 for m in captured_maxsizes), (
            f"No bounded queue found. Queue maxsizes: {captured_maxsizes}. "
            "provider._execute_sdk_completion() MUST create "
            "asyncio.Queue(maxsize=config.event_queue_size)"
        )


class TestModelResolution:
    """Tests for behaviors:Models:MUST:1,4."""

    @pytest.mark.asyncio
    async def test_invalid_model_translates_to_not_found_error(self) -> None:
        """behaviors:Models:MUST:1 — Invalid model request translates to NotFoundError.

        When the SDK rejects a model (via ModelNotFoundError or string-pattern match),
        the error translation layer MUST produce kernel NotFoundError.

        Contract: behaviors:Models:MUST:1
        """
        from amplifier_core.llm_errors import NotFoundError

        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from tests.fixtures.sdk_mocks import MockCopilotClientWrapper

        # ValueError with "model not found" maps to NotFoundError via error_translation
        mock_client = MockCopilotClientWrapper(
            events=[],
            raise_on_session=ValueError("model not found: nonexistent-model-xyz"),
        )
        provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]

        request = _make_standard_request("nonexistent-model-xyz")

        with pytest.raises(NotFoundError):
            await provider.complete(request)


class TestCacheInvalidation:
    """Test cache invalidation function."""

    def test_invalidate_cache_removes_file(self, tmp_path: Path) -> None:
        """invalidate_cache() removes existing cache file.

        Documents the invalidate_cache() function as public API.

        # Contract: behaviors:ModelCache:SHOULD:3 (SHOULD, not MUST)
        """
        from amplifier_module_provider_github_copilot.model_cache import (
            invalidate_cache,
            write_cache,
        )
        from amplifier_module_provider_github_copilot.models import CopilotModelInfo

        cache_file = tmp_path / "test_cache.json"
        models = [
            CopilotModelInfo(
                id="test-model",
                name="Test Model",
                context_window=100000,
                max_output_tokens=8000,
            ),
        ]
        write_cache(models, cache_file=cache_file)
        assert cache_file.exists()

        invalidate_cache(cache_file=cache_file)
        assert not cache_file.exists()


# =============================================================================
# SDK Version Consistency Tests
# =============================================================================


class TestSdkVersionConsistency:
    """Verify SDK version in error message matches pyproject.toml.

    Contract: sdk-boundary.md MUST:5 — fail at import time with clear error
    if github-copilot-sdk is not installed.

    The error message must tell users to install the CORRECT version
    that matches pyproject.toml, not a stale version.
    """

    def test_sdk_version_error_message_matches_metadata(self) -> None:
        """sdk-boundary:Membrane:MUST:5 — error message version matches installed spec.

        Regression prevention: The error message told users to install the wrong
        SDK version. Uses _check_sdk_version() (extracted for testability) and
        importlib.metadata to verify message matches actual declared dependency.
        """
        import importlib.metadata
        import re

        from packaging.requirements import Requirement
        from packaging.specifiers import SpecifierSet

        from amplifier_module_provider_github_copilot import _check_sdk_version

        # Get declared SDK requirement from package metadata
        requires = importlib.metadata.requires("amplifier-module-provider-github-copilot") or []
        sdk_req = next((r for r in requires if r.startswith("github-copilot-sdk")), None)
        assert isinstance(sdk_req, str), "Package must declare github-copilot-sdk dependency"
        metadata_specifier = SpecifierSet(str(Requirement(sdk_req).specifier))

        # Trigger the error with an old SDK version
        with pytest.raises(ImportError) as exc_info:
            _check_sdk_version("0.1.33")

        error_msg = str(exc_info.value)

        # Extract version specifier from pip install instruction in error message
        pip_match = re.search(r"pip install 'github-copilot-sdk([^']+)'", error_msg)
        assert pip_match is not None and isinstance(pip_match.group(1), str), (
            "Error message must contain pip install instruction"
        )
        message_specifier = SpecifierSet(pip_match.group(1))

        # Compare normalized specifiers (order-independent)
        assert message_specifier == metadata_specifier, (
            f"Error message specifier {str(message_specifier)!r} != "
            f"metadata specifier {str(metadata_specifier)!r}"
        )


class TestPackageVersionConsistency:
    """Verify package __version__ matches pyproject.toml version.

    Contract: provider-protocol:public_api:MUST:1 — package identity must be consistent.

    The package __version__ is used by tools like pip show, importlib.metadata,
    and user code that checks provider version. It MUST match pyproject.toml
    to avoid confusion and version drift.
    """

    def test_package_version_matches_installed(self) -> None:
        """Package __version__ matches installed package version.

        # Contract: provider-protocol:public_api:MUST:1
        Regression prevention: __version__ could become stale during releases.
        Uses importlib.metadata as the authoritative source.
        """
        import importlib.metadata

        from amplifier_module_provider_github_copilot import __version__

        # Let PackageNotFoundError propagate — missing package is a test failure
        installed = importlib.metadata.version("amplifier-module-provider-github-copilot")

        assert __version__ == installed, (
            f"__version__ {__version__!r} does not match installed version "
            f"{installed!r}. Fix __version__ in __init__.py to match pyproject.toml."
        )

    def test_package_version_is_valid_semver(self) -> None:
        """Package __version__ must be valid semantic versioning.

        Ensures version can be parsed by packaging tools.
        """
        import re

        from amplifier_module_provider_github_copilot import __version__

        # Semantic versioning pattern (simplified): X.Y.Z with optional pre-release
        semver_pattern = r"^\d+\.\d+\.\d+(-[a-zA-Z0-9.]+)?(\+[a-zA-Z0-9.]+)?$"
        assert re.match(semver_pattern, __version__), (
            f"__version__ '{__version__}' is not valid semantic versioning. "
            f"Expected format: X.Y.Z or X.Y.Z-prerelease"
        )


# =============================================================================
# Production Path Testing with MockCopilotClientWrapper
# =============================================================================


class TestProductionPathWithMockClient:
    """Test that MockCopilotClientWrapper enables production path testing.

    Contract: sdk-boundary:Membrane:MUST:1 — mock at the membrane level

    Architecture Change:
    - OLD: Tests use sdk_create_fn → completion.py (separate code path)
    - NEW: Tests use MockCopilotClientWrapper → production code path

    This ensures tests exercise the SAME code as production.
    """

    @pytest.mark.asyncio
    async def test_mock_client_enables_production_path(self) -> None:
        """MockCopilotClientWrapper allows testing production code path.

        Demonstrates that tests can use provider._execute_sdk_completion()
        (the production path) instead of completion.py (test-only path).
        """
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )
        from tests.fixtures.sdk_mocks import (
            MockCopilotClientWrapper,
            text_delta_event,
        )

        # Create mock client with events
        events = [text_delta_event("Hello, world!")]
        mock_client = MockCopilotClientWrapper(events=events)

        # Inject mock client into provider
        provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]

        # Create request using proper kernel type
        from amplifier_core.message_models import TextBlock

        request = _make_standard_request()

        # Call the production complete() method
        # This uses _execute_sdk_completion(), NOT completion.py
        response = await provider.complete(request)

        # Verify it's a TextBlock (kernel type)
        first_block = response.content[0]
        assert isinstance(first_block, TextBlock)
        assert "Hello, world!" in first_block.text

    @pytest.mark.asyncio
    async def test_mock_client_session_receives_model(self) -> None:
        """MockCopilotClientWrapper.session() receives model parameter."""
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )
        from tests.fixtures.sdk_mocks import MockCopilotClientWrapper

        mock_client = MockCopilotClientWrapper(events=[])
        provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]

        request = _make_standard_request("gpt-4-turbo")

        await provider.complete(request)

        # Verify model was passed to session
        assert mock_client.last_model == "gpt-4-turbo"

    @pytest.mark.asyncio
    async def test_mock_client_session_receives_tools(self) -> None:
        """MockCopilotClientWrapper.session() receives tools parameter."""
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )
        from tests.fixtures.sdk_mocks import MockCopilotClientWrapper

        mock_client = MockCopilotClientWrapper(events=[])
        provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]

        # Create request with tools
        tool: dict[str, Any] = {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {},
        }
        request = _make_standard_request()
        request.tools = [tool]

        await provider.complete(request)

        # Verify tools were passed to session with exact content
        assert mock_client.last_tools == [tool]

    @pytest.mark.asyncio
    async def test_mock_client_session_receives_system_message(self) -> None:
        """MockCopilotClientWrapper.session() receives system_message from ChatRequest.

        This verifies the complete() -> _execute_sdk_completion() -> session() chain
        correctly extracts and forwards system_message from ChatRequest.messages.

        Contract: sdk-boundary:Config:MUST:2
        Bug fix: system_message was missing, causing SDK to use default persona
        instead of Amplifier bundle instructions.
        """
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )
        from tests.fixtures.sdk_mocks import MockCopilotClientWrapper

        mock_client = MockCopilotClientWrapper(events=[])
        provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]

        # Create request with system message in messages
        system_msg = MagicMock(spec=Message)
        system_msg.role = "system"
        system_msg.content = "You are a helpful coding assistant."

        user_msg = MagicMock(spec=Message)
        user_msg.role = "user"
        user_msg.content = "Hello"

        request = _make_standard_request()
        request.messages = [system_msg, user_msg]

        await provider.complete(request)

        # Verify system_message was extracted and forwarded
        assert mock_client.last_system_message == "You are a helpful coding assistant.", (
            "system_message was not forwarded to SDK session. "
            "This causes SDK to use default persona instead of bundle instructions."
        )

    @pytest.mark.asyncio
    async def test_mock_client_propagates_errors_correctly(self) -> None:
        """Errors from MockCopilotClientWrapper are properly translated."""
        from amplifier_core.llm_errors import ProviderUnavailableError

        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )
        from tests.fixtures.sdk_mocks import MockCopilotClientWrapper

        # Create mock that raises on session creation
        mock_client = MockCopilotClientWrapper(
            events=[],
            raise_on_session=ConnectionError("Network failure"),
        )
        provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]

        request = _make_standard_request()

        # Should raise translated error
        with pytest.raises(ProviderUnavailableError):
            await provider.complete(request)

    @pytest.mark.asyncio
    async def test_session_disconnected_after_completion(self) -> None:
        """Session is always disconnected after completion.

        # Contract: deny-destroy:Ephemeral:MUST:2
        # Contract: provider-protocol:Complete:MUST:4

        Contract: deny-destroy.md — Session MUST be destroyed in finally block
        Replaces: test_completion.py::TestSessionLifecycle::
        test_session_created_and_destroyed_on_success
        """
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from tests.fixtures.sdk_mocks import MockCopilotClientWrapper, text_delta_event

        mock_client = MockCopilotClientWrapper(events=[text_delta_event("hello")])
        provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]

        request = _make_standard_request()

        await provider.complete(request)

        # Session should be disconnected
        assert mock_client.session_instance is not None  # narrowed for pyright
        assert mock_client.session_instance.disconnected is True

    @pytest.mark.asyncio
    async def test_session_disconnected_on_error(self) -> None:
        """Session disconnected even when error occurs during send().

        # Contract: deny-destroy:Ephemeral:MUST:2

        Contract: deny-destroy.md — Session MUST be destroyed even on error
        Replaces: test_completion.py::TestSessionLifecycle::test_session_destroyed_on_error
        """
        from amplifier_core.llm_errors import ProviderUnavailableError

        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from tests.fixtures.sdk_mocks import MockCopilotClientWrapper

        mock_client = MockCopilotClientWrapper(
            events=[],
            raise_on_send=ConnectionError("Network failure"),
        )
        provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]

        request = _make_standard_request()

        with pytest.raises(ProviderUnavailableError):
            await provider.complete(request)

        # Session should still be disconnected despite error
        assert mock_client.session_instance is not None  # narrowed for pyright
        assert mock_client.session_instance.disconnected is True

    @pytest.mark.asyncio
    async def test_text_content_accumulated_correctly(self) -> None:
        """Multiple text deltas are accumulated into single response.

        Contract: streaming-contract:Response:MUST:1
        Contract: streaming-contract.md — Text content MUST be accumulated
        Replaces: test_completion.py::TestResponseConstruction::test_text_content_accumulated
        """
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from tests.fixtures.sdk_mocks import MockCopilotClientWrapper, text_delta_event

        # Multiple text delta events
        events = [
            text_delta_event("Hello "),
            text_delta_event("World"),
            text_delta_event("!"),
        ]
        mock_client = MockCopilotClientWrapper(events=events)
        provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]

        request = _make_standard_request()

        response = await provider.complete(request)

        # Should have accumulated text
        from amplifier_core.message_models import TextBlock

        first_block = response.content[0]
        assert isinstance(first_block, TextBlock)
        assert first_block.text == "Hello World!"

    @pytest.mark.asyncio
    async def test_empty_response_handled_gracefully(self) -> None:
        """Empty response (no content events) is handled gracefully.

        Contract: streaming-contract.md — Empty responses MUST not crash
        Replaces: test_completion.py::TestResponseConstruction::test_empty_response_handled
        """
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from tests.fixtures.sdk_mocks import MockCopilotClientWrapper

        # No content events, just idle
        mock_client = MockCopilotClientWrapper(events=[])
        provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]

        request = _make_standard_request()

        response = await provider.complete(request)

        # Empty events MUST produce empty content list
        # Contract: streaming-contract:StreamingResponse:MUST:4 — content_blocks None when empty
        assert response.content == [], (
            f"Expected empty content list for no-event response, got {response.content}"
        )

    @pytest.mark.asyncio
    async def test_error_during_send_translated(self) -> None:
        """SDK errors during send() are translated to kernel errors.

        Contract: sdk-boundary:Translation:MUST:2
        Contract: error-hierarchy.md — SDK errors MUST be translated
        Replaces: test_completion.py::TestErrorHandling::test_sdk_error_translated
        """
        from amplifier_core.llm_errors import ProviderUnavailableError

        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from tests.fixtures.sdk_mocks import MockCopilotClientWrapper

        mock_client = MockCopilotClientWrapper(
            events=[],
            raise_on_send=ConnectionError("Connection refused"),
        )
        provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]

        request = _make_standard_request()

        with pytest.raises(ProviderUnavailableError) as exc_info:
            await provider.complete(request)

        # Should be kernel ProviderUnavailableError with provider info
        # Contract: error-hierarchy:ConnectionError:MUST:1
        assert exc_info.value.provider == "github-copilot"

    @pytest.mark.asyncio
    async def test_original_exception_chained(self) -> None:
        """Original SDK exception is preserved via __cause__.

        Contract: error-hierarchy:Translation:MUST:3
        Replaces: test_completion.py::TestErrorHandling::test_error_preserves_original
        """
        from amplifier_core.llm_errors import LLMError

        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from tests.fixtures.sdk_mocks import MockCopilotClientWrapper

        original_error = ConnectionError("Original connection error")
        mock_client = MockCopilotClientWrapper(
            events=[],
            raise_on_send=original_error,
        )
        provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]

        request = _make_standard_request()

        with pytest.raises(LLMError) as exc_info:
            await provider.complete(request)

        # __cause__ should be the original exception
        assert exc_info.value.__cause__ is original_error

    @pytest.mark.asyncio
    async def test_llm_error_not_double_wrapped(self) -> None:
        """LLMError is not re-translated (double-wrapped).

        Contract: error-hierarchy.md — Pre-translated errors MUST pass through
        Replaces: test_sdk_boundary.py::TestDoubleTranslationGuard::
        test_llm_error_not_double_wrapped
        """
        from amplifier_core.llm_errors import AuthenticationError

        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from tests.fixtures.sdk_mocks import MockCopilotClientWrapper

        # Pre-translated error (already an LLMError subclass)
        pre_translated_error = AuthenticationError("Already translated", provider="test-provider")
        mock_client = MockCopilotClientWrapper(
            events=[],
            raise_on_session=pre_translated_error,
        )
        provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]

        request = _make_standard_request()

        with pytest.raises(AuthenticationError) as exc_info:
            await provider.complete(request)

        # Should be the SAME error, not wrapped
        assert exc_info.value is pre_translated_error
        # Should NOT have a __cause__ (not double-wrapped)
        assert exc_info.value.__cause__ is None
        # Original provider preserved
        assert exc_info.value.provider == "test-provider"


# =============================================================================
# Runtime Config Override (Routing Matrix Support)
# =============================================================================


class TestRuntimeConfigOverride:
    """Test model selection priority with runtime config.

    Contract: behaviors:ModelSelection:MUST:1,2

    Bug Reference: Routing matrix passes config={"default_model": "gpt-5.4"}
    but provider was ignoring it and always using YAML default.
    """

    def test_effective_default_model_respects_runtime_config(self) -> None:
        """config["default_model"] from mount overrides YAML default.

        Contract: behaviors:ModelSelection:MUST:1
        Priority: request.model > config["default_model"] > YAML
        """
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from tests.fixtures.sdk_mocks import MockCopilotClientWrapper

        # Mount with routing matrix config
        routing_model = "gpt-5.4-turbo"
        mock_client = MockCopilotClientWrapper(events=[])

        # Pass runtime config like routing matrix does
        provider = GitHubCopilotProvider(
            config={"default_model": routing_model},
            client=mock_client,  # type: ignore[arg-type]
        )

        # _effective_default_model should use runtime config
        assert provider._effective_default_model == routing_model  # pyright: ignore[reportPrivateUsage]

    def test_effective_default_model_falls_back_to_yaml(self) -> None:
        """Without runtime config, falls back to YAML default.

        Contract: behaviors:ModelSelection:MUST:1
        Priority 3: YAML defaults.model (fallback)
        """
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from tests.fixtures.sdk_mocks import MockCopilotClientWrapper

        mock_client = MockCopilotClientWrapper(events=[])
        provider = GitHubCopilotProvider(
            config={},  # No default_model
            client=mock_client,  # type: ignore[arg-type]
        )

        # Should use YAML default (accessing private members for test verification)
        yaml_default = provider._provider_config.defaults["model"]  # pyright: ignore[reportPrivateUsage]
        assert provider._effective_default_model == yaml_default  # pyright: ignore[reportPrivateUsage]

    @pytest.mark.asyncio
    async def test_complete_uses_runtime_default_model(self) -> None:
        """complete() uses runtime config model when request.model is None.

        Contract: behaviors:ModelSelection:MUST:1
        Priority 2: config["default_model"] when request.model is None
        """
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from tests.fixtures.sdk_mocks import MockCopilotClientWrapper, text_delta_event

        routing_model = "gpt-5.4-turbo"
        mock_client = MockCopilotClientWrapper(events=[text_delta_event("hello")])

        provider = GitHubCopilotProvider(
            config={"default_model": routing_model},
            client=mock_client,  # type: ignore[arg-type]
        )

        # Request WITHOUT model specified — should use runtime default
        request = _make_standard_request()
        request.model = None  # No model in request

        await provider.complete(request)

        # Session should receive the routing matrix model, not YAML default
        assert mock_client.last_model == routing_model

    @pytest.mark.asyncio
    async def test_request_model_overrides_runtime_config(self) -> None:
        """request.model takes precedence over runtime config.

        Contract: behaviors:ModelSelection:MUST:1
        Priority 1: request.model (explicit per-request override)
        """
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from tests.fixtures.sdk_mocks import MockCopilotClientWrapper, text_delta_event

        routing_model = "gpt-5.4-turbo"
        request_model = "claude-opus-4.5"
        mock_client = MockCopilotClientWrapper(events=[text_delta_event("hello")])

        provider = GitHubCopilotProvider(
            config={"default_model": routing_model},
            client=mock_client,  # type: ignore[arg-type]
        )

        request = _make_standard_request(request_model)

        await provider.complete(request)

        # Session should receive request model, not runtime config
        assert mock_client.last_model == request_model

    def test_multiple_providers_dont_share_config(self) -> None:
        """Multiple providers with different configs don't conflict.

        Contract: behaviors:ModelSelection:MUST:2
        Contract: behaviors:Models:MUST:2
        load_models_config() uses @lru_cache — MUST NOT mutate cached config.
        """
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from tests.fixtures.sdk_mocks import MockCopilotClientWrapper

        mock_client1 = MockCopilotClientWrapper(events=[])
        mock_client2 = MockCopilotClientWrapper(events=[])

        # Two providers with different routing matrix configs
        provider1 = GitHubCopilotProvider(
            config={"default_model": "gpt-5.4"},
            client=mock_client1,  # type: ignore[arg-type]
        )
        provider2 = GitHubCopilotProvider(
            config={"default_model": "claude-opus-4.5"},
            client=mock_client2,  # type: ignore[arg-type]
        )

        # Each should have independent effective default
        assert provider1._effective_default_model == "gpt-5.4"  # pyright: ignore[reportPrivateUsage]
        assert provider2._effective_default_model == "claude-opus-4.5"  # pyright: ignore[reportPrivateUsage]

        # The underlying cached config should NOT be mutated
        from amplifier_module_provider_github_copilot.config_loader import load_models_config

        cached_config = load_models_config()
        # Cached config should still have original YAML default
        # (not mutated by either provider)
        assert cached_config.defaults["model"] == "claude-opus-4.5"


# =============================================================================
# C-2: asyncio.CancelledError translation
# =============================================================================


class TestCancelledErrorTranslation:
    """C-2: asyncio.CancelledError MUST translate to AbortError.

    Contract: error-hierarchy:AbortError:MUST:1

    asyncio.CancelledError inherits from BaseException since Python 3.9, NOT Exception.
    A bare ``except Exception`` in provider.complete() silently misses it, causing
    the raw CancelledError to escape to the kernel as an unhandled BaseException —
    violating the error-hierarchy contract and breaking observability (emit_response_error
    never fires).

    Fix: add ``except asyncio.CancelledError`` BEFORE each ``except Exception`` block,
    translate to ``AbortError(retryable=False)``, emit error event, and re-raise the
    translated error.
    """

    @pytest.mark.asyncio
    async def test_cancelled_error_from_session_becomes_abort_error(self) -> None:
        """error-hierarchy:AbortError:MUST:1 — CancelledError from SDK → AbortError.

        This is the FAILING test for C-2.  Before the fix, asyncio.CancelledError
        escapes ``except Exception`` and propagates as raw BaseException.
        After the fix it MUST be caught and re-raised as AbortError.
        """
        import asyncio

        from amplifier_core.llm_errors import AbortError

        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from tests.fixtures.sdk_mocks import MockCopilotClientWrapper

        # Inject CancelledError at session-creation time (before first SDK call)
        mock_client = MockCopilotClientWrapper(
            events=[],
            raise_on_session=asyncio.CancelledError("task cancelled"),  # type: ignore[arg-type]
        )
        provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]

        request = _make_standard_request()

        # MUST raise AbortError — NOT raw asyncio.CancelledError
        with pytest.raises(AbortError) as exc_info:
            await provider.complete(request)

        assert exc_info.value.retryable is False

    @pytest.mark.asyncio
    async def test_cancelled_error_not_retried(self) -> None:
        """error-hierarchy:AbortError:MUST:1 — CancelledError MUST NOT trigger retry loop.

        AbortError has retryable=False, so only one attempt must be made.
        """
        import asyncio

        from amplifier_core.llm_errors import AbortError

        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from tests.fixtures.sdk_mocks import MockCopilotClientWrapper

        call_count = 0

        class CountingCancelMock(MockCopilotClientWrapper):
            @property  # type: ignore[override]
            def _raise_on_session(self) -> asyncio.CancelledError:  # type: ignore[override]
                return asyncio.CancelledError("cancelled")

            @_raise_on_session.setter
            def _raise_on_session(self, value: object) -> None:
                pass  # ignore assignment from __init__

            from contextlib import asynccontextmanager as _acm

            @_acm
            async def session(self, model=None, *, system_message=None, tools=None):  # type: ignore[override]
                nonlocal call_count
                call_count += 1
                raise asyncio.CancelledError("cancelled")
                yield  # noqa: F821

        mock_client = CountingCancelMock(events=[])
        provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]

        request = _make_standard_request()

        with pytest.raises(AbortError):
            await provider.complete(request)

        # AbortError is non-retryable → exactly 1 attempt
        assert call_count == 1, f"Expected 1 attempt (no retry), got {call_count}"
