"""
Tests for observability hook emission.

Contract: provider-protocol.md (Observability Hooks section)

These tests verify that the provider emits llm:request, llm:response,
and PROVIDER_RETRY events for integration with Amplifier's monitoring.

Evidence: Canonical providers (anthropic, openai, azure-openai) emit these hooks.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_coordinator() -> MagicMock:
    """Create mock coordinator with hooks.emit capability."""
    coordinator = MagicMock()
    coordinator.hooks = MagicMock()
    coordinator.hooks.emit = AsyncMock()
    return coordinator


def _make_request(model: str = "claude-opus-4.5") -> MagicMock:
    """Create a standard mock ChatRequest for hooks emission tests."""
    request = MagicMock()
    request.model = model
    request.messages = [MagicMock(role="user", content="test")]
    request.tools = None
    request.max_tokens = None
    request.temperature = None
    request.stop = None
    request.stream = None
    return request


@pytest.fixture
def provider_config() -> dict[str, Any]:
    """Standard provider config for tests."""
    return {
        "model": "claude-opus-4.5",
        "use_streaming": False,
        "debug": False,
    }


@pytest.fixture
def sample_request() -> dict[str, Any]:
    """Sample ChatRequest-like dict for complete() calls."""
    return {
        "messages": [{"role": "user", "content": "Hello"}],
    }


# ────────────────────────────────────────────────────────────────────────────
# emit_event() Helper Tests (observability module)
# ────────────────────────────────────────────────────────────────────────────


class TestEmitEventHelper:
    """Tests for emit_event() helper in observability module.

    Contract: observability:Events:MUST:1, MUST:4, MUST:5
    """

    @pytest.mark.asyncio
    async def test_emit_event_success(
        self, mock_coordinator: MagicMock, provider_config: dict[str, Any]
    ) -> None:
        """Should emit events through coordinator hooks.

        Contract: observability:Events:MUST:1
        """
        from amplifier_module_provider_github_copilot.observability import emit_event

        await emit_event(mock_coordinator, "test:event", {"key": "value"})

        mock_coordinator.hooks.emit.assert_called_once_with("test:event", {"key": "value"})

    @pytest.mark.asyncio
    async def test_emit_event_handles_error_gracefully(
        self, mock_coordinator: MagicMock, provider_config: dict[str, Any]
    ) -> None:
        """Should handle hook emission errors gracefully (no raise).

        Contract: observability:Events:MUST:5
        """
        from amplifier_module_provider_github_copilot.observability import emit_event

        mock_coordinator.hooks.emit = AsyncMock(side_effect=Exception("Hook error"))

        # Should not raise
        await emit_event(mock_coordinator, "test:event", {"key": "value"})

    @pytest.mark.asyncio
    async def test_emit_event_no_coordinator(self, provider_config: dict[str, Any]) -> None:
        """Should handle missing coordinator gracefully (no raise).

        Contract: observability:Events:MUST:4
        """
        from amplifier_module_provider_github_copilot.observability import emit_event

        # Should not raise - coordinator is None
        await emit_event(None, "test:event", {"key": "value"})

    @pytest.mark.asyncio
    async def test_emit_event_coordinator_no_hooks(self, provider_config: dict[str, Any]) -> None:
        """Should handle coordinator without hooks attribute gracefully.

        Contract: observability:Events:MUST:5
        """
        from amplifier_module_provider_github_copilot.observability import emit_event

        coordinator = MagicMock(spec=[])  # No hooks attribute

        # Should not raise
        await emit_event(coordinator, "test:event", {"key": "value"})


# ────────────────────────────────────────────────────────────────────────────
# llm:request Event Tests
# ────────────────────────────────────────────────────────────────────────────


class TestLlmRequestEvent:
    """Tests for llm:request event emission.

    Contract: provider-protocol:hooks:llm_request:MUST:1, MUST:2
    """

    @pytest.mark.asyncio
    async def test_complete_emits_llm_request_event(
        self,
        mock_coordinator: MagicMock,
        sample_request: dict[str, Any],
    ) -> None:
        """complete() should emit llm:request event before SDK call.

        Contract: provider-protocol:hooks:llm_request:MUST:1
        """
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from amplifier_module_provider_github_copilot.streaming import StreamingAccumulator

        provider = GitHubCopilotProvider(
            config={"model": "claude-opus-4.5", "use_streaming": False, "debug": False},
            coordinator=mock_coordinator,
        )

        # Mock _execute_sdk_completion to do nothing
        async def mock_execute(
            *args: Any, accumulator: StreamingAccumulator, **kwargs: Any
        ) -> None:
            """Mock SDK execution that populates accumulator with minimal response."""
            pass  # Just succeed without adding events

        provider._execute_sdk_completion = mock_execute  # type: ignore[method-assign]

        await provider.complete(
            sample_request,  # type: ignore[arg-type]
            model="claude-opus-4.5",
        )

        # Find llm:request call
        request_calls = [
            call
            for call in mock_coordinator.hooks.emit.call_args_list
            if call[0][0] == "llm:request"
        ]
        assert len(request_calls) == 1, "Expected exactly one llm:request event"

    @pytest.mark.asyncio
    async def test_llm_request_contains_required_fields(
        self,
        mock_coordinator: MagicMock,
        sample_request: dict[str, Any],
    ) -> None:
        """llm:request event should contain required metadata fields.

        Contract: provider-protocol:hooks:llm_request:MUST:2
        """
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from amplifier_module_provider_github_copilot.streaming import StreamingAccumulator

        provider = GitHubCopilotProvider(
            config={"model": "claude-opus-4.5", "use_streaming": False, "debug": False},
            coordinator=mock_coordinator,
        )

        async def mock_execute(
            *args: Any, accumulator: StreamingAccumulator, **kwargs: Any
        ) -> None:
            pass

        provider._execute_sdk_completion = mock_execute  # type: ignore[method-assign]

        await provider.complete(
            sample_request,  # type: ignore[arg-type]
            model="claude-opus-4.5",
        )

        # Find llm:request call and validate payload
        request_calls = [
            call
            for call in mock_coordinator.hooks.emit.call_args_list
            if call[0][0] == "llm:request"
        ]
        assert len(request_calls) == 1
        data = request_calls[0][0][1]

        # Required fields per contract
        assert data["provider"] == "github-copilot"
        # Model comes from request or provider config default
        assert "model" in data
        assert "message_count" in data
        assert "streaming" in data
        assert "tool_count" in data
        assert "timeout" in data


# ────────────────────────────────────────────────────────────────────────────
# llm:response Event Tests
# ────────────────────────────────────────────────────────────────────────────


class TestLlmResponseEvent:
    """Tests for llm:response event emission.

    Contract: provider-protocol:hooks:llm_response:MUST:1, MUST:2, MUST:3
    """

    @pytest.mark.asyncio
    async def test_complete_emits_llm_response_event(
        self,
        mock_coordinator: MagicMock,
        sample_request: dict[str, Any],
    ) -> None:
        """complete() should emit llm:response event after SDK call.

        Contract: provider-protocol:hooks:llm_response:MUST:1
        """
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from amplifier_module_provider_github_copilot.streaming import StreamingAccumulator

        provider = GitHubCopilotProvider(
            config={"model": "claude-opus-4.5", "use_streaming": False, "debug": False},
            coordinator=mock_coordinator,
        )

        async def mock_execute(
            *args: Any, accumulator: StreamingAccumulator, **kwargs: Any
        ) -> None:
            pass

        provider._execute_sdk_completion = mock_execute  # type: ignore[method-assign]

        await provider.complete(
            sample_request,  # type: ignore[arg-type]
            model="claude-opus-4.5",
        )

        # Find llm:response call
        response_calls = [
            call
            for call in mock_coordinator.hooks.emit.call_args_list
            if call[0][0] == "llm:response"
        ]
        assert len(response_calls) == 1, "Expected exactly one llm:response event"

    @pytest.mark.asyncio
    async def test_llm_response_contains_duration_ms(
        self,
        mock_coordinator: MagicMock,
        sample_request: dict[str, Any],
    ) -> None:
        """llm:response should include duration_ms timing metric.

        Contract: provider-protocol:hooks:llm_response:MUST:2
        """
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from amplifier_module_provider_github_copilot.streaming import StreamingAccumulator

        provider = GitHubCopilotProvider(
            config={"model": "claude-opus-4.5", "use_streaming": False, "debug": False},
            coordinator=mock_coordinator,
        )

        async def mock_execute(
            *args: Any, accumulator: StreamingAccumulator, **kwargs: Any
        ) -> None:
            pass

        provider._execute_sdk_completion = mock_execute  # type: ignore[method-assign]

        await provider.complete(
            sample_request,  # type: ignore[arg-type]
            model="claude-opus-4.5",
        )

        # Find llm:response call and validate duration_ms
        response_calls = [
            call
            for call in mock_coordinator.hooks.emit.call_args_list
            if call[0][0] == "llm:response"
        ]
        assert len(response_calls) == 1
        data = response_calls[0][0][1]

        assert "duration_ms" in data
        assert isinstance(data["duration_ms"], int)
        assert data["duration_ms"] >= 0

    @pytest.mark.asyncio
    async def test_llm_response_status_ok_on_success(
        self,
        mock_coordinator: MagicMock,
        sample_request: dict[str, Any],
    ) -> None:
        """llm:response should have status "ok" on successful completion.

        Contract: provider-protocol:hooks:llm_response:MUST:3
        """
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from amplifier_module_provider_github_copilot.streaming import StreamingAccumulator

        provider = GitHubCopilotProvider(
            config={"model": "claude-opus-4.5", "use_streaming": False, "debug": False},
            coordinator=mock_coordinator,
        )

        async def mock_execute(
            *args: Any, accumulator: StreamingAccumulator, **kwargs: Any
        ) -> None:
            pass

        provider._execute_sdk_completion = mock_execute  # type: ignore[method-assign]

        await provider.complete(
            sample_request,  # type: ignore[arg-type]
            model="claude-opus-4.5",
        )

        # Find llm:response call and validate status
        response_calls = [
            call
            for call in mock_coordinator.hooks.emit.call_args_list
            if call[0][0] == "llm:response"
        ]
        assert len(response_calls) == 1
        data = response_calls[0][0][1]

        assert data["status"] == "ok"
        assert data["provider"] == "github-copilot"
        assert "usage" in data

    @pytest.mark.asyncio
    async def test_llm_response_contains_sdk_session_id(
        self,
        mock_coordinator: MagicMock,
    ) -> None:
        """llm:response should include sdk_session_id for log correlation.

        Contract: observability:SHOULD:1 — include correlation IDs for tracing

        The sdk_session_id enables foreign-key correlation between Amplifier logs
        and Copilot SDK logs (~/.copilot/logs/) for billing/token forensics.
        """
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from tests.fixtures.sdk_mocks import (
            MockCopilotClientWrapper,
            text_delta_event,
        )

        # Create mock client with events - session will have session_id
        events = [text_delta_event("Hello")]
        mock_client = MockCopilotClientWrapper(events=events)

        # Inject mock client and coordinator
        provider = GitHubCopilotProvider(
            client=mock_client,  # type: ignore[arg-type]
            coordinator=mock_coordinator,
        )

        # Create request
        request = _make_request()

        # Call complete - this uses production path with mock client
        await provider.complete(request)

        # Find llm:response call and validate sdk_session_id
        response_calls = [
            call
            for call in mock_coordinator.hooks.emit.call_args_list
            if call[0][0] == "llm:response"
        ]
        assert len(response_calls) == 1, "Expected llm:response event"
        data = response_calls[0][0][1]

        # Verify sdk_session_id is present
        assert "sdk_session_id" in data, (
            "llm:response must include sdk_session_id for log correlation"
        )
        # Verify it's the mock session ID
        assert data["sdk_session_id"] == "mock-sdk-session-id"

    @pytest.mark.asyncio
    async def test_llm_response_contains_sdk_pid(
        self,
        mock_coordinator: MagicMock,
    ) -> None:
        """llm:response should include sdk_pid for SDK log file correlation.

        Contract: observability:Events:SHOULD:3 — include sdk_pid in llm:response

        The sdk_pid enables direct lookup of SDK log files at:
        ~/.copilot/logs/process-{timestamp}-{pid}.log

        Without sdk_pid, tooling must grep through ALL log files (O(n)).
        With sdk_pid, tooling does O(1) direct file access.
        """
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from tests.fixtures.sdk_mocks import (
            MockCopilotClientWrapper,
            text_delta_event,
        )

        # Create mock client with events and mock PID
        events = [text_delta_event("Hello")]
        mock_client = MockCopilotClientWrapper(events=events)
        # Set mock PID (simulates what real client captures from subprocess)
        mock_client._mock_copilot_pid = "12345"  # pyright: ignore[reportPrivateUsage]

        # Inject mock client and coordinator
        provider = GitHubCopilotProvider(
            client=mock_client,  # type: ignore[arg-type]
            coordinator=mock_coordinator,
        )

        # Create request
        request = _make_request()

        # Call complete - this uses production path with mock client
        await provider.complete(request)

        # Find llm:response call and validate sdk_pid
        response_calls = [
            call
            for call in mock_coordinator.hooks.emit.call_args_list
            if call[0][0] == "llm:response"
        ]
        assert len(response_calls) == 1, "Expected llm:response event"
        data = response_calls[0][0][1]

        # Verify sdk_pid is present
        assert "sdk_pid" in data, "llm:response must include sdk_pid for log correlation"
        # Verify it's the mock PID
        assert data["sdk_pid"] == "12345"

    @pytest.mark.asyncio
    async def test_llm_response_usage_includes_cache_read_tokens_when_present(
        self,
        mock_coordinator: MagicMock,
    ) -> None:
        """llm:response usage dict includes cache_read_tokens when SDK provides them.

        Contract: streaming-contract:usage:MUST:2 — forward cache tokens to kernel Usage
        """
        from amplifier_module_provider_github_copilot.observability import llm_lifecycle

        async with llm_lifecycle(mock_coordinator, "claude-opus-4.5") as ctx:
            await ctx.emit_response_ok(
                usage_input=67000,
                usage_output=23,
                usage_cache_read=62851,
                usage_cache_write=None,
                finish_reason="stop",
                content_blocks=1,
                tool_calls=0,
            )

        response_calls = [
            call
            for call in mock_coordinator.hooks.emit.call_args_list
            if call[0][0] == "llm:response"
        ]
        assert len(response_calls) == 1
        usage = response_calls[0][0][1]["usage"]
        assert usage["input"] == 67000
        assert usage["output"] == 23
        assert usage["cache_read_tokens"] == 62851
        assert "cache_write_tokens" not in usage

    @pytest.mark.asyncio
    async def test_llm_response_usage_omits_cache_tokens_when_none(
        self,
        mock_coordinator: MagicMock,
    ) -> None:
        """llm:response usage dict omits cache_read/write_tokens when SDK sends None.

        Contract: streaming-contract:usage:MUST:2 — None ≠ 0; absent means not sent
        """
        from amplifier_module_provider_github_copilot.observability import llm_lifecycle

        async with llm_lifecycle(mock_coordinator, "claude-opus-4.5") as ctx:
            await ctx.emit_response_ok(
                usage_input=1000,
                usage_output=50,
                usage_cache_read=None,
                usage_cache_write=None,
                finish_reason="stop",
                content_blocks=1,
                tool_calls=0,
            )

        response_calls = [
            call
            for call in mock_coordinator.hooks.emit.call_args_list
            if call[0][0] == "llm:response"
        ]
        assert len(response_calls) == 1
        usage = response_calls[0][0][1]["usage"]
        assert "cache_read_tokens" not in usage
        assert "cache_write_tokens" not in usage

    @pytest.mark.asyncio
    async def test_emit_response_ok_includes_cache_write_tokens_when_present(
        self,
        mock_coordinator: MagicMock,
    ) -> None:
        """usage_cache_write is not None → included in llm:response usage payload.

        Contract: streaming-contract:usage:MUST:2
        Mutation check: remove the cache_write branch in emit_response_ok → key absent.
        """
        from amplifier_module_provider_github_copilot.observability import llm_lifecycle

        async with llm_lifecycle(mock_coordinator, "claude-opus-4.5") as ctx:
            await ctx.emit_response_ok(
                usage_input=1000,
                usage_output=50,
                usage_cache_read=None,
                usage_cache_write=500,  # Non-None: must appear in payload
                finish_reason="stop",
                content_blocks=1,
                tool_calls=0,
            )

        response_calls = [
            call
            for call in mock_coordinator.hooks.emit.call_args_list
            if call[0][0] == "llm:response"
        ]
        assert len(response_calls) == 1
        usage = response_calls[0][0][1]["usage"]
        assert usage["cache_write_tokens"] == 500

    @pytest.mark.asyncio
    async def test_emit_response_ok_uses_tool_use_finish_reason_when_tool_calls_nonzero(
        self,
        mock_coordinator: MagicMock,
    ) -> None:
        """finish_reason=None with tool_calls>0 → config.finish_reasons.tool_use.

        Contract: observability:Events:MUST:3
        Mutation check: remove the `if not finish_reason` branch → finish_reason stays
        None or empty, breaking trace correlation for tool-use turns.
        """
        from amplifier_module_provider_github_copilot.observability import llm_lifecycle

        async with llm_lifecycle(mock_coordinator, "claude-opus-4.5") as ctx:
            await ctx.emit_response_ok(
                usage_input=100,
                usage_output=10,
                usage_cache_read=None,
                usage_cache_write=None,
                finish_reason=None,  # Not provided — must be inferred from tool_calls
                content_blocks=0,
                tool_calls=1,  # tool_calls > 0 → finish_reason = tool_use
            )

        response_calls = [
            call
            for call in mock_coordinator.hooks.emit.call_args_list
            if call[0][0] == "llm:response"
        ]
        assert len(response_calls) == 1
        payload = response_calls[0][0][1]
        assert payload["finish_reason"] == "tool_calls", (
            f"Expected 'tool_calls' (config.finish_reasons.tool_use) when "
            f"finish_reason=None and tool_calls=1, got {payload['finish_reason']!r}"
        )


# ────────────────────────────────────────────────────────────────────────────
# llm_lifecycle Orphan Guard Tests
# ────────────────────────────────────────────────────────────────────────────


class TestLlmLifecycleOrphanGuard:
    """llm_lifecycle auto-emits error response when exception escapes context.

    Contract: observability:Events:MUST:3 — MUST emit llm:response after llm:request.
    """

    @pytest.mark.asyncio
    async def test_auto_emits_error_response_when_exception_escapes(
        self,
        mock_coordinator: MagicMock,
    ) -> None:
        """If request was emitted but response not yet emitted, auto-emits error response.

        Mutation check: remove the try/except in llm_lifecycle that calls
        emit_response_error → llm:response is never emitted, leaving an orphaned
        llm:request event that breaks trace correlation in the telemetry stream.
        """
        from amplifier_module_provider_github_copilot.observability import llm_lifecycle

        with pytest.raises(RuntimeError, match="sdk failure"):
            async with llm_lifecycle(mock_coordinator, "claude-opus-4.5") as ctx:
                await ctx.emit_request(
                    message_count=1,
                    tool_count=0,
                    streaming=False,
                    timeout=30.0,
                )
                raise RuntimeError("sdk failure")

        emitted = [call[0][0] for call in mock_coordinator.hooks.emit.call_args_list]
        assert "llm:request" in emitted, "llm:request must have been emitted"
        assert "llm:response" in emitted, (
            "llm:response MUST be auto-emitted when exception escapes context "
            "to prevent orphaned llm:request events"
        )
        response_call = next(
            call
            for call in mock_coordinator.hooks.emit.call_args_list
            if call[0][0] == "llm:response"
        )
        assert response_call[0][1]["status"] in ("error", "aborted"), (
            f"Auto-emitted llm:response must have error/aborted status, "
            f"got {response_call[0][1]['status']!r}"
        )


# ────────────────────────────────────────────────────────────────────────────
# Event Ordering Tests
# ────────────────────────────────────────────────────────────────────────────


class TestEventOrdering:
    """Tests for event emission ordering.

    Contract: provider-protocol Observability Hooks - Event Ordering Contract
    """

    @pytest.mark.asyncio
    async def test_llm_request_precedes_llm_response(
        self,
        mock_coordinator: MagicMock,
        sample_request: dict[str, Any],
    ) -> None:
        """llm:request must be emitted BEFORE llm:response.

        Contract: provider-protocol Observability Hooks - Event Ordering Contract
        """
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from amplifier_module_provider_github_copilot.streaming import StreamingAccumulator

        provider = GitHubCopilotProvider(
            config={"model": "claude-opus-4.5", "use_streaming": False, "debug": False},
            coordinator=mock_coordinator,
        )

        async def mock_execute(
            *args: Any, accumulator: StreamingAccumulator, **kwargs: Any
        ) -> None:
            pass

        provider._execute_sdk_completion = mock_execute  # type: ignore[method-assign]

        await provider.complete(
            sample_request,  # type: ignore[arg-type]
            model="claude-opus-4.5",
        )

        # Get all emitted event names in order
        emitted_events = [call[0][0] for call in mock_coordinator.hooks.emit.call_args_list]

        # Find indices
        request_idx = next((i for i, e in enumerate(emitted_events) if e == "llm:request"), None)
        response_idx = next((i for i, e in enumerate(emitted_events) if e == "llm:response"), None)

        # Contract: observability:Events:MUST:2,3
        assert "llm:request" in emitted_events, "llm:request event not found"
        assert "llm:response" in emitted_events, "llm:response event not found"
        assert request_idx is not None  # narrowed for pyright
        assert response_idx is not None  # narrowed for pyright
        assert request_idx < response_idx, (
            f"llm:request ({request_idx}) must precede llm:response ({response_idx})"
        )


# ────────────────────────────────────────────────────────────────────────────
# PROVIDER_RETRY Event Tests
# ────────────────────────────────────────────────────────────────────────────


class TestProviderRetryEvent:
    """Tests for PROVIDER_RETRY event emission.

    Contract: provider-protocol:hooks:provider_retry:MUST:1, MUST:2
    """

    @pytest.mark.asyncio
    async def test_emits_provider_retry_on_retryable_error(
        self,
        mock_coordinator: MagicMock,
        sample_request: dict[str, Any],
    ) -> None:
        """Should emit provider:retry event before retry sleep.

        Contract: provider-protocol:hooks:provider_retry:MUST:1
        """
        from unittest.mock import patch

        from amplifier_core import llm_errors

        # PROVIDER_RETRY = "provider:retry" per amplifier_core.events
        PROVIDER_RETRY = "provider:retry"

        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from amplifier_module_provider_github_copilot.streaming import StreamingAccumulator

        provider = GitHubCopilotProvider(
            config={"model": "claude-opus-4.5", "use_streaming": False, "debug": False},
            coordinator=mock_coordinator,
        )

        call_count = 0

        async def mock_execute(
            *args: Any, accumulator: StreamingAccumulator, **kwargs: Any
        ) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Use ProviderUnavailableError which is retryable
                raise llm_errors.ProviderUnavailableError("Service temporarily unavailable")
            # Second call succeeds

        provider._execute_sdk_completion = mock_execute  # type: ignore[method-assign]

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await provider.complete(
                sample_request,  # type: ignore[arg-type]
                model="claude-opus-4.5",
            )

        # Find provider:retry call
        retry_calls = [
            call
            for call in mock_coordinator.hooks.emit.call_args_list
            if call[0][0] == PROVIDER_RETRY
        ]
        assert len(retry_calls) >= 1, f"Expected at least one {PROVIDER_RETRY} event"

    @pytest.mark.asyncio
    async def test_provider_retry_contains_required_fields(
        self,
        mock_coordinator: MagicMock,
        sample_request: dict[str, Any],
    ) -> None:
        """provider:retry event should contain attempt, max_retries, delay.

        Contract: provider-protocol:hooks:provider_retry:MUST:2
        """
        from unittest.mock import patch

        from amplifier_core import llm_errors

        # PROVIDER_RETRY = "provider:retry" per amplifier_core.events
        PROVIDER_RETRY = "provider:retry"

        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from amplifier_module_provider_github_copilot.streaming import StreamingAccumulator

        provider = GitHubCopilotProvider(
            config={"model": "claude-opus-4.5", "use_streaming": False, "debug": False},
            coordinator=mock_coordinator,
        )

        call_count = 0

        async def mock_execute(
            *args: Any, accumulator: StreamingAccumulator, **kwargs: Any
        ) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Use ProviderUnavailableError which is retryable
                raise llm_errors.ProviderUnavailableError("Service temporarily unavailable")

        provider._execute_sdk_completion = mock_execute  # type: ignore[method-assign]

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await provider.complete(
                sample_request,  # type: ignore[arg-type]
                model="claude-opus-4.5",
            )

        # Find provider:retry call and validate payload
        retry_calls = [
            call
            for call in mock_coordinator.hooks.emit.call_args_list
            if call[0][0] == PROVIDER_RETRY
        ]
        assert len(retry_calls) >= 1
        data = retry_calls[0][0][1]

        # Required fields per contract
        assert data["provider"] == "github-copilot"
        assert "attempt" in data
        assert "max_retries" in data
        assert "delay" in data
        assert "error_type" in data

    @pytest.mark.asyncio
    async def test_retry_after_is_none_when_not_present(
        self,
        mock_coordinator: MagicMock,
        sample_request: dict[str, Any],
    ) -> None:
        """retry_after field is None when error carries no Retry-After value.

        Contract: provider-protocol:hooks:provider_retry:MUST:3
        """
        from unittest.mock import patch

        from amplifier_core import llm_errors

        PROVIDER_RETRY = "provider:retry"

        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from amplifier_module_provider_github_copilot.streaming import StreamingAccumulator

        provider = GitHubCopilotProvider(
            config={"model": "claude-opus-4.5", "use_streaming": False, "debug": False},
            coordinator=mock_coordinator,
        )

        call_count = 0

        async def mock_execute(
            *args: Any, accumulator: StreamingAccumulator, **kwargs: Any
        ) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # ProviderUnavailableError has no retry_after attribute
                raise llm_errors.ProviderUnavailableError("Service temporarily unavailable")

        provider._execute_sdk_completion = mock_execute  # type: ignore[method-assign]

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await provider.complete(
                sample_request,  # type: ignore[arg-type]
                model="claude-opus-4.5",
            )

        retry_calls = [
            call
            for call in mock_coordinator.hooks.emit.call_args_list
            if call[0][0] == PROVIDER_RETRY
        ]
        assert len(retry_calls) >= 1
        data = retry_calls[0][0][1]

        # MUST:3 — field is present and its exact value is None (not missing, not 0)
        assert "retry_after" in data, "retry_after must always be present in payload"
        assert data["retry_after"] is None, (
            f"Expected None for non-RateLimitError, got {data['retry_after']!r}"
        )

    @pytest.mark.asyncio
    async def test_retry_after_is_float_when_rate_limit_carries_header(
        self,
        mock_coordinator: MagicMock,
        sample_request: dict[str, Any],
    ) -> None:
        """retry_after field is a float when RateLimitError carries a Retry-After value.

        Contract: provider-protocol:hooks:provider_retry:MUST:3
        """
        from unittest.mock import patch

        from amplifier_core import llm_errors

        PROVIDER_RETRY = "provider:retry"

        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from amplifier_module_provider_github_copilot.streaming import StreamingAccumulator

        provider = GitHubCopilotProvider(
            config={"model": "claude-opus-4.5", "use_streaming": False, "debug": False},
            coordinator=mock_coordinator,
        )

        call_count = 0

        async def mock_execute(
            *args: Any, accumulator: StreamingAccumulator, **kwargs: Any
        ) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # RateLimitError with retry_after=30.0 — simulates Retry-After: 30 header
                raise llm_errors.RateLimitError("rate limited", retry_after=30.0)

        provider._execute_sdk_completion = mock_execute  # type: ignore[method-assign]

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await provider.complete(
                sample_request,  # type: ignore[arg-type]
                model="claude-opus-4.5",
            )

        retry_calls = [
            call
            for call in mock_coordinator.hooks.emit.call_args_list
            if call[0][0] == PROVIDER_RETRY
        ]
        assert len(retry_calls) >= 1
        data = retry_calls[0][0][1]

        # MUST:3 — exact type (float) and exact value (30.0) from the error attribute
        assert "retry_after" in data, "retry_after must always be present in payload"
        assert isinstance(data["retry_after"], float), (
            f"Expected float, got {type(data['retry_after']).__name__}"
        )
        assert data["retry_after"] == 30.0, (
            f"Expected 30.0 from RateLimitError.retry_after, got {data['retry_after']!r}"
        )

    @pytest.mark.asyncio
    async def test_retry_after_propagated_from_translated_exception(
        self,
        mock_coordinator: MagicMock,
        sample_request: dict[str, Any],
    ) -> None:
        """retry_after is extracted and propagated when a raw exception is translated.

        Covers the second emit_retry call site (except Exception → translate_sdk_error).
        A plain Exception whose message matches the RateLimitError mapping and contains
        a Retry-After value must surface that value in the provider:retry payload.

        Contract: provider-protocol:hooks:provider_retry:MUST:3
        """
        from unittest.mock import patch

        PROVIDER_RETRY = "provider:retry"

        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from amplifier_module_provider_github_copilot.streaming import StreamingAccumulator

        provider = GitHubCopilotProvider(
            config={"model": "claude-opus-4.5", "use_streaming": False, "debug": False},
            coordinator=mock_coordinator,
        )

        call_count = 0

        async def mock_execute(
            *args: Any, accumulator: StreamingAccumulator, **kwargs: Any
        ) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Raw (non-kernel) exception — goes through translate_sdk_error.
                # Message matches RateLimitError string_pattern "rate limit" and
                # contains a Retry-After value parseable by _extract_retry_after.
                raise Exception(  # noqa: TRY002
                    "rate limit exceeded. Retry after 45 seconds"
                )

        provider._execute_sdk_completion = mock_execute  # type: ignore[method-assign]

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await provider.complete(
                sample_request,  # type: ignore[arg-type]
                model="claude-opus-4.5",
            )

        retry_calls = [
            call
            for call in mock_coordinator.hooks.emit.call_args_list
            if call[0][0] == PROVIDER_RETRY
        ]
        assert len(retry_calls) >= 1
        data = retry_calls[0][0][1]

        # MUST:3 — retry_after extracted from translated RateLimitError (second call site)
        assert "retry_after" in data, "retry_after must always be present in payload"
        assert isinstance(data["retry_after"], float), (
            f"Expected float from translated RateLimitError, "
            f"got {type(data['retry_after']).__name__}"
        )
        assert data["retry_after"] == 45.0, (
            f"Expected 45.0 extracted from message, got {data['retry_after']!r}"
        )
