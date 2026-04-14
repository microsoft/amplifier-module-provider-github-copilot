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
