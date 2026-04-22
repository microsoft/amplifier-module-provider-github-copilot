"""Coverage tests for provider.py missing branches.

Covers:
- Lines 280-283: list_models() cache hit when SDK fails
- Lines 514-520: CancelledError in fake-tool correction path
- Line 541: for/else exhausted fake-tool corrections
- Lines 751-758: _emit_content_async content without __dict__
- Lines 808-810: _emit_streaming_content with no running event loop

Contract: provider-protocol:complete:MUST:1
Contract: error-hierarchy:AbortError:MUST:1
Contract: streaming-contract:ProgressiveStreaming:SHOULD:1-4
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.fixtures.sdk_mocks import (
    MockCopilotClientWrapper,
    MockSDKSession,
    SessionEvent,
    SessionEventData,
    SessionEventType,
    idle_event,
    usage_event,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_coordinator() -> MagicMock:
    """Create a minimal coordinator with hooks for provider construction."""
    coordinator = MagicMock()
    coordinator.hooks = MagicMock()
    coordinator.hooks.emit = AsyncMock()
    return coordinator


def _make_request(*, tools: list[Any] | None = None) -> MagicMock:
    """Create a minimal ChatRequest mock."""
    request = MagicMock()
    request.messages = [MagicMock(role="user", content="test")]
    request.model = "gpt-4o"
    request.tools = tools
    request.attachments = None
    return request


# Fake-tool text pattern that triggers fake_tool_detection
_FAKE_TOOL_TEXT = "[search_files: query='test']"


class FakeToolTextSession(MockSDKSession):
    """Always returns text that looks like a fake tool call."""

    async def send(
        self,
        prompt: str,
        *,
        attachments: list[dict[str, Any]] | None = None,
    ) -> str:
        self.last_prompt = prompt
        fake_event = SessionEvent(
            type=SessionEventType.ASSISTANT_MESSAGE_DELTA,
            data=SessionEventData(delta_content=_FAKE_TOOL_TEXT),
        )
        for handler in self._handlers:
            handler(fake_event)
        for handler in self._handlers:
            handler(idle_event())
        return "msg-id"


class CancelOnSecondSendSession(MockSDKSession):
    """First send: returns fake tool text. Second send: raises CancelledError."""

    _send_count: int = 0

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._send_count = 0

    async def send(
        self,
        prompt: str,
        *,
        attachments: list[dict[str, Any]] | None = None,
    ) -> str:
        self._send_count += 1
        if self._send_count == 1:
            # First call: return fake tool text to trigger correction
            fake_event = SessionEvent(
                type=SessionEventType.ASSISTANT_MESSAGE_DELTA,
                data=SessionEventData(delta_content=_FAKE_TOOL_TEXT),
            )
            for handler in self._handlers:
                handler(fake_event)
            for handler in self._handlers:
                handler(idle_event())
            return "msg-id"
        else:
            # Second call (correction): cancel
            raise asyncio.CancelledError()


class CountingSessionWrapper:
    """MockCopilotClientWrapper that counts session() calls using a single session."""

    def __init__(self, session_class: type[MockSDKSession]) -> None:
        self._session_class = session_class
        self._closed = False
        self._session_obj: MockSDKSession | None = None

    def is_healthy(self) -> bool:
        return not self._closed

    @asynccontextmanager
    async def session(
        self,
        model: str | None = None,
        *,
        system_message: str | None = None,
        tools: list[Any] | None = None,
    ) -> AsyncIterator[MockSDKSession]:
        """Each time session() is called, the same CancelOnSecondSendSession is used,
        but that session's send_count also increments for each send() call."""
        if self._session_obj is None:
            self._session_obj = self._session_class()
        try:
            yield self._session_obj
        finally:
            await self._session_obj.disconnect()

    async def list_models(self) -> list[Any]:
        return []

    async def close(self) -> None:
        self._closed = True


# ---------------------------------------------------------------------------
# list_models() cache hit when SDK fails (lines 280-283)
# ---------------------------------------------------------------------------


class TestListModelsCacheFallback:
    """list_models() falls back to disk cache when SDK fails."""

    @pytest.mark.asyncio
    async def test_list_models_returns_cached_when_sdk_fails(self) -> None:
        """When SDK fetch fails but disk cache has models, returns cached.

        Contract: behaviors:ModelCache:SHOULD:1
        Lines 280-283 in provider.py — Tier 2 cache fallback
        """
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from amplifier_module_provider_github_copilot.sdk_adapter.model_translation import (
            CopilotModelInfo,
        )

        mock_client = MockCopilotClientWrapper()
        provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]

        # Cached model to return when SDK fails — CopilotModelInfo has required fields
        cached = [
            CopilotModelInfo(
                id="gpt-4o", name="GPT-4o", context_window=128000, max_output_tokens=8192
            )
        ]

        with (
            patch(
                "amplifier_module_provider_github_copilot.provider.fetch_and_map_models",
                side_effect=RuntimeError("SDK unavailable"),
            ),
            patch(
                "amplifier_module_provider_github_copilot.provider.read_cache",
                return_value=cached,
            ),
        ):
            models = await provider.list_models()

        assert len(models) == 1

    @pytest.mark.asyncio
    async def test_list_models_raises_when_sdk_fails_and_no_cache(self) -> None:
        """When SDK fails AND no cache, raises ProviderUnavailableError.

        Contract: behaviors:ModelDiscoveryError:MUST:1
        Lines 280-305 in provider.py — Tier 3 error (no hardcoded fallback)
        """
        from amplifier_module_provider_github_copilot.error_translation import (
            ProviderUnavailableError,
        )
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        mock_client = MockCopilotClientWrapper()
        provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]

        with (
            patch(
                "amplifier_module_provider_github_copilot.provider.fetch_and_map_models",
                side_effect=RuntimeError("SDK unavailable"),
            ),
            patch(
                "amplifier_module_provider_github_copilot.provider.read_cache",
                return_value=[],  # empty cache
            ),
        ):
            with pytest.raises(ProviderUnavailableError):
                await provider.list_models()


# ---------------------------------------------------------------------------
# CancelledError in fake-tool CORRECTION path (lines 514-520)
# ---------------------------------------------------------------------------


class TestCancelledErrorInCorrectionPath:
    """CancelledError during fake-tool correction translates to AbortError."""

    @pytest.mark.asyncio
    async def test_cancelled_error_in_correction_raises_abort_error(self) -> None:
        """asyncio.CancelledError in correction attempt → AbortError raised.

        Contract: error-hierarchy:AbortError:MUST:1
        Lines 514-520 in provider.py
        """
        from amplifier_module_provider_github_copilot.error_translation import AbortError
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        mock_client = MockCopilotClientWrapper()
        coordinator = _make_coordinator()
        provider = GitHubCopilotProvider(
            config={},
            coordinator=coordinator,
            client=mock_client,  # type: ignore[arg-type]
        )

        request = _make_request()

        call_count = 0

        async def fake_execute_sdk_completion(**kwargs: object) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: do nothing (accumulator stays empty — fake tool check follows)
                pass
            elif call_count == 2:
                # Second call (correction): raise CancelledError
                raise asyncio.CancelledError()

        # Patch both the execution and the fake-tool detection to control flow cleanly
        with (
            patch.object(
                provider,
                "_execute_sdk_completion",
                side_effect=fake_execute_sdk_completion,
            ),
            patch(
                "amplifier_module_provider_github_copilot.provider.should_retry_for_fake_tool_calls",
                side_effect=[(True, "[fake_tool:"), (False, None)],
            ),
        ):
            with pytest.raises(AbortError) as exc_info:
                await provider.complete(request)

        assert "cancelled" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Exhausted fake-tool corrections (line 541 — for/else)
# ---------------------------------------------------------------------------


class TestExhaustedFakeToolCorrections:
    """When all correction attempts fail, for/else calls log_exhausted."""

    @pytest.mark.asyncio
    async def test_all_corrections_exhausted_returns_response(self) -> None:
        """When max_correction_attempts exhausted, log_exhausted is called and response returned.

        Contract: provider-protocol:complete:MUST:1 — always returns response
        Line ~541 in provider.py — for...else clause
        """
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        # Client that ALWAYS returns fake-tool text
        mock_client = MockCopilotClientWrapper(
            session_class=FakeToolTextSession,
        )
        coordinator = _make_coordinator()
        provider = GitHubCopilotProvider(
            config={},
            coordinator=coordinator,
            client=mock_client,  # type: ignore[arg-type]
        )

        # Request with tools so fake tool detection is active
        tool_mock = MagicMock()
        tool_mock.name = "search_files"
        tool_mock.description = "Search files"
        tool_mock.parameters = {}
        request = _make_request(tools=[tool_mock])

        # Should complete (not raise) — exhausted corrections still returns response
        response = await provider.complete(request)
        # Contract: provider-protocol:complete:MUST:6
        # (detects fake tools, retries, exhaustion fallback)
        assert response.text == _FAKE_TOOL_TEXT
        assert len(response.content) == 1
        assert response.content[0].text == _FAKE_TOOL_TEXT


# ---------------------------------------------------------------------------
# _emit_content_async with content lacking __dict__ (lines 751-758)
# ---------------------------------------------------------------------------


class TestEmitContentAsyncPrimitive:
    """_emit_content_async handles primitives without __dict__ via else branch."""

    @pytest.mark.asyncio
    async def test_emit_content_async_with_string_uses_value_wrapper(self) -> None:
        """Content without __dict__ is wrapped as {'value': content}.

        Lines 751-758 in provider.py — the else branch of hasattr(content, '__dict__')
        Contract: streaming-contract:ProgressiveStreaming:SHOULD:1
        """
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        coordinator = _make_coordinator()
        provider = GitHubCopilotProvider(config={}, coordinator=coordinator)

        # A plain string has no __dict__ → triggers else branch
        raw_string_content = "hello world"
        assert not hasattr(raw_string_content, "__dict__")  # confirm test premise

        await provider._emit_content_async(raw_string_content)  # pyright: ignore[reportPrivateUsage]

        # hooks.emit should have been called with {'value': 'hello world'}
        coordinator.hooks.emit.assert_called_once()
        call_args = coordinator.hooks.emit.call_args
        content_payload = call_args[0][1]["content"]
        assert content_payload == {"value": "hello world"}

    @pytest.mark.asyncio
    async def test_emit_content_async_with_integer(self) -> None:
        """Integer content also takes the else branch.

        Lines 751-758 in provider.py
        """
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        coordinator = _make_coordinator()
        provider = GitHubCopilotProvider(config={}, coordinator=coordinator)

        await provider._emit_content_async(42)  # pyright: ignore[reportPrivateUsage]

        coordinator.hooks.emit.assert_called_once()
        content_payload = coordinator.hooks.emit.call_args[0][1]["content"]
        # Contract: streaming-contract:ProgressiveStreaming:SHOULD:1
        assert content_payload == {"value": 42}


# ---------------------------------------------------------------------------
# _emit_streaming_content with no running event loop (lines 808-810)
# ---------------------------------------------------------------------------


class TestEmitStreamingContentNoLoop:
    """_emit_streaming_content gracefully handles missing event loop."""

    def test_emit_streaming_content_no_loop_does_not_raise(self) -> None:
        """Calling from synchronous context triggers except RuntimeError silently.

        Lines 808-810 in provider.py — except RuntimeError: skip emission
        Contract: streaming-contract:ProgressiveStreaming:SHOULD:4 — graceful skip
        """
        from amplifier_core import TextContent

        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        coordinator = _make_coordinator()
        provider = GitHubCopilotProvider(config={}, coordinator=coordinator)

        content = TextContent(text="test content")

        # In a synchronous (non-async) function, asyncio.get_running_loop() raises RuntimeError
        # _emit_streaming_content MUST catch it silently
        provider._emit_streaming_content(content)  # pyright: ignore[reportPrivateUsage]

        # No exception raised — emission was silently skipped
        # hooks.emit should NOT have been called (no loop to run the coroutine)
        coordinator.hooks.emit.assert_not_called()

    def test_emit_streaming_content_no_coordinator_does_not_raise(self) -> None:
        """When coordinator is None, _emit_streaming_content skips silently.

        Lines 800-803 in provider.py — SHOULD:4 guard
        """
        from amplifier_core import TextContent

        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider(config={}, coordinator=None)
        content = TextContent(text="test")

        # Should not raise even with no coordinator
        provider._emit_streaming_content(content)  # pyright: ignore[reportPrivateUsage]
        # No assertion needed — must simply not raise


# ---------------------------------------------------------------------------
# Usage injection from usage_holder (lines 751-758)
# ---------------------------------------------------------------------------


class UsageInjectionSession(MockSDKSession):
    """Session that fires usage + idle events during send().

    Simulates SDK usage event arriving before idle (both go to usage_holder and queue).
    Combined with a patch on translate_event, we can force the accumulator to have
    no usage from the queue — exercising the usage_holder injection path.
    """

    async def send(
        self,
        prompt: str,
        *,
        attachments: list[dict[str, Any]] | None = None,
    ) -> str:
        self.last_prompt = prompt
        # Fire assistant.usage → populates usage_holder, goes into queue
        for handler in self._handlers:
            handler(usage_event(input_tokens=10, output_tokens=20))
        # Fire session.idle → unblocks provider
        for handler in self._handlers:
            handler(idle_event())
        return "message-id"


class TestUsageInjectionFromUsageHolder:
    """provider.py injects usage from usage_holder when accumulator has none."""

    @pytest.mark.asyncio
    async def test_usage_injected_when_translate_event_returns_none(self) -> None:
        """Usage from usage_holder is injected when queue draining produces no USAGE_UPDATE.

        Contract: streaming-contract:usage:MUST:1
        Lines 751-758 in provider.py — usage injection fallback

        Scenario: usage event fires before idle, populates usage_holder + queue.
        translate_event is patched to return None for assistant.usage events
        (simulating the race condition where usage arrives outside the translation
        cycle). The accumulator ends up with no usage from the queue, so the
        injection path at lines 751-758 fires.
        """
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from amplifier_module_provider_github_copilot.streaming import (
            translate_event as _real_translate,
        )

        mock_client = MockCopilotClientWrapper(
            session_class=UsageInjectionSession,
        )
        coordinator = _make_coordinator()
        provider = GitHubCopilotProvider(
            config={},
            coordinator=coordinator,
            client=mock_client,  # type: ignore[arg-type]
        )
        request = _make_request()

        def _translate_no_usage(sdk_event: Any, config: Any) -> Any:
            """Delegate all events except assistant.usage to real translate_event."""
            raw_type = sdk_event.get("type", "")
            event_type: str = raw_type.value if hasattr(raw_type, "value") else str(raw_type)
            if event_type == "assistant.usage":
                return None  # Simulate race: usage not translated from queue
            return _real_translate(sdk_event, config)

        with patch(
            "amplifier_module_provider_github_copilot.provider.translate_event",
            side_effect=_translate_no_usage,
        ):
            response = await provider.complete(request)

        # Usage was injected from usage_holder into accumulator at lines 751-758
        assert response.usage is not None  # narrowed for pyright
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 20


# ---------------------------------------------------------------------------
# Canary: captured-tools added to accumulator BEFORE event_queue drain.
# Contract: sdk-protection:ToolCapture:MUST:1,2
# Contract: streaming-contract:Accumulation (is_complete drops post-TURN_COMPLETE)
# ---------------------------------------------------------------------------


class ToolThenIdleSession(MockSDKSession):
    """Fires a tool-request event, then SESSION_IDLE — both enter event_queue.

    This mirrors the real SDK race pattern where tool.execution_complete is
    followed by session.idle. The tool event populates tool_capture_handler
    AND is queued; session.idle also enters the queue and translates to a
    TURN_COMPLETE domain event that would set accumulator.is_complete=True.

    Regression target: provider._execute_sdk_completion MUST add captured_tools
    to the accumulator BEFORE draining the queue. If the drain runs first,
    TURN_COMPLETE sets is_complete=True and the subsequent accumulator.add()
    for TOOL_CALL is silently dropped (see streaming.StreamingAccumulator.add
    is_complete guard). Response would return with tool_calls=[].
    """

    async def send(
        self,
        prompt: str,
        *,
        attachments: list[dict[str, Any]] | None = None,
    ) -> str:
        self.last_prompt = prompt
        # 1. Tool request event (populates tool_capture_handler.captured_tools
        #    via EventRouter, AND queues the event)
        tool_event = SessionEvent(
            type=SessionEventType.ASSISTANT_MESSAGE,
            data=SessionEventData(
                tool_requests=[
                    {
                        "tool_call_id": "canary_tool_id_7",
                        "name": "canary_read",
                        "arguments": {"path": "/tmp/x"},
                    },
                ],
            ),
        )
        for handler in self._handlers:
            handler(tool_event)
        # 2. SESSION_IDLE — unblocks provider; ALSO queued.
        #    Its translation produces TURN_COMPLETE which sets is_complete=True.
        for handler in self._handlers:
            handler(idle_event())
        return "message-id"


class TestCapturedToolsAddedBeforeDrainCanary:
    """Regression canary for the captured-tools-before-drain ordering."""

    @pytest.mark.asyncio
    async def test_captured_tools_survive_turn_complete_in_same_drain(self) -> None:
        """Captured tools MUST land in response even when TURN_COMPLETE is queued.

        Contract: sdk-protection:ToolCapture:MUST:1,2
        Contract: streaming-contract:Accumulation (is_complete guard)

        Mutation check: in provider._execute_sdk_completion, moving the
        `for tool in tool_capture_handler.captured_tools: accumulator.add(...)`
        block to AFTER the `while not event_queue.empty(): ...` drain turns
        this assertion red — TURN_COMPLETE drains first, is_complete=True,
        then accumulator.add(TOOL_CALL) silently drops → tool_calls=[].
        """
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        mock_client = MockCopilotClientWrapper(session_class=ToolThenIdleSession)
        coordinator = _make_coordinator()
        provider = GitHubCopilotProvider(
            config={},
            coordinator=coordinator,
            client=mock_client,  # type: ignore[arg-type]
        )
        request = _make_request(tools=[{"name": "canary_read", "description": "d"}])

        response = await provider.complete(request)

        # Exact assertions — tool must survive the drain, not be silently dropped.
        assert response.tool_calls is not None  # narrowed for pyright
        assert len(response.tool_calls) == 1, (
            "Tool call dropped — captured-tools-before-drain ordering regressed. "
            "See provider._execute_sdk_completion: captured_tools loop MUST run "
            "before the event_queue drain, otherwise TURN_COMPLETE sets "
            "is_complete=True and accumulator.add(TOOL_CALL) is silently discarded."
        )
        assert response.tool_calls[0].id == "canary_tool_id_7"
        assert response.tool_calls[0].name == "canary_read"


# ---------------------------------------------------------------------------
# _parse_raw_flag and _config_int edge cases (provider.py lines 180, 197)
# ---------------------------------------------------------------------------


class TestConfigHelpers:
    """Coverage for _parse_raw_flag and _config_int edge cases.

    Contract: provider-protocol:complete:MUST:1
    """

    def test_parse_raw_flag_non_string_non_bool_uses_bool_coercion(self) -> None:
        """_parse_raw_flag with int 1 → True; int 0 → False.

        Line 180 in provider.py — the `return bool(value)` path.
        Mutation check: remove the final `return bool(value)` → any non-bool, non-str
        truthy value (e.g. int 1) would hit an unhandled code path.
        """
        from amplifier_module_provider_github_copilot.provider import (
            _parse_raw_flag,  # pyright: ignore[reportPrivateUsage]
        )

        assert _parse_raw_flag(1) is True
        assert _parse_raw_flag(0) is False
        assert _parse_raw_flag(42) is True

    def test_config_int_unparseable_value_returns_default(self) -> None:
        """_config_int with non-numeric string → logs warning and returns default.

        Line 197 in provider.py — the `except (TypeError, ValueError)` path.
        Mutation check: remove the except block → unparseable config raises ValueError
        and crashes the provider instead of gracefully falling back.
        """
        from amplifier_module_provider_github_copilot.provider import (
            _config_int,  # pyright: ignore[reportPrivateUsage]
        )

        result = _config_int("not-an-int", default=42)
        assert result == 42, f"Expected default 42, got {result}"

    def test_config_int_none_value_returns_default(self) -> None:
        """_config_int(None, default) returns default immediately.

        Lines 196-197 in provider.py — the `if value is None: return default` guard.
        Mutation check: remove the None guard → int(None) raises TypeError in the
        try block, which is handled by the except, but the None guard is the cheaper
        early exit that documents the contract for callers.
        """
        from amplifier_module_provider_github_copilot.provider import (
            _config_int,  # pyright: ignore[reportPrivateUsage]
        )

        result = _config_int(None, default=7)
        assert result == 7, f"Expected default 7 for None input, got {result}"


# ---------------------------------------------------------------------------
# list_models() cache write failure (provider.py lines 430-433)
# ---------------------------------------------------------------------------


class TestListModelsCacheWriteFailure:
    """write_cache failure during list_models() does not suppress the SDK result."""

    @pytest.mark.asyncio
    async def test_write_cache_failure_still_returns_sdk_models(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """When write_cache raises, list_models still returns the SDK models and logs a warning.

        Contract: provider-protocol:list_models:MUST:1
        Lines 430-433 in provider.py — the except block around write_cache().
        Mutation check: remove the except block → OSError from write_cache propagates,
        list_models raises instead of returning models → callers see error instead of model list.
        """
        import logging
        from unittest.mock import AsyncMock

        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        fake_model = MagicMock()
        fake_copilot_model = MagicMock()
        mock_client = MagicMock()
        mock_client.is_healthy.return_value = True

        with (
            patch(
                "amplifier_module_provider_github_copilot.provider.fetch_and_map_models",
                new=AsyncMock(return_value=([fake_model], [fake_copilot_model])),
            ),
            patch(
                "amplifier_module_provider_github_copilot.provider.write_cache",
                side_effect=OSError("disk full"),
            ),
            caplog.at_level(logging.WARNING),
        ):
            provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]
            models = await provider.list_models()

        assert len(models) == 1, "SDK models must be returned even when cache write fails"
        assert models[0] is fake_model
        assert any("Failed to cache models" in r.getMessage() for r in caplog.records), (
            "write_cache failure must be logged as a warning"
        )


# ---------------------------------------------------------------------------
# Correction loop for/else exhaustion (provider.py line 693)
# ---------------------------------------------------------------------------


class TestCorrectionLoopForElseExhaustion:
    """For/else on the fake-tool correction loop fires when all attempts are used."""

    @pytest.mark.asyncio
    async def test_all_correction_attempts_exhausted_logs_and_returns_response(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """When every correction attempt still detects a fake tool call, the for/else
        fires and log_exhausted is called, then the last response is returned.

        Contract: provider-protocol:complete:MUST:1 — always returns response
        Line 693 in provider.py — for...else clause on correction loop.
        Mutation check: change `for...else` to `for` without else → log_exhausted is
        never called when loop exhausts (only called on exceptions), silently hiding
        the "max corrections exhausted" event from observability.
        """
        import logging

        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        mock_client = MockCopilotClientWrapper(session_class=FakeToolTextSession)
        coordinator = _make_coordinator()
        provider = GitHubCopilotProvider(
            config={},
            coordinator=coordinator,
            client=mock_client,  # type: ignore[arg-type]
        )
        tool_mock = MagicMock()
        tool_mock.name = "search_files"
        tool_mock.description = "Search files"
        tool_mock.parameters = {}
        request = _make_request(tools=[tool_mock])

        # Force should_retry to always return True so the loop exhausts every attempt
        # without breaking. This tests the for/else path, not fake tool detection logic.
        with (
            patch(
                "amplifier_module_provider_github_copilot.provider.should_retry_for_fake_tool_calls",
                return_value=(True, "forced_pattern"),
            ),
            caplog.at_level(logging.WARNING),
        ):
            response = await provider.complete(request)

        assert isinstance(response.content, list), (
            "provider.complete() must return a ChatResponse with content list "
            "even when all correction attempts are exhausted"
        )
        assert any("exhausted" in r.getMessage().lower() for r in caplog.records), (
            "log_exhausted must be called when for/else fires — 'exhausted' not found in logs"
        )
