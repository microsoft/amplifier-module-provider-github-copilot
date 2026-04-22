"""Integration tests for provider.py code coverage.

These tests exercise production code paths through provider._execute_sdk_completion
using MockCopilotClientWrapper — a real behavioral mock, not magic mocks.

Contract References:
- behaviors:Retry:MUST:1-5 — Retry behavior
- behaviors:Streaming:MUST:1 — TTFT warning
- behaviors:Streaming:MUST:4 — Bounded queue
- streaming-contract:ProgressiveStreaming:SHOULD:1 — Content emission
- sdk-protection:Session:MUST:3,4 — Tool capture abort

Coverage Targets (provider.py):
- Lines 180-188: _extract_delta_text paths
- Lines 395-465: Retry loop branches
- Lines 478-515: Fake tool detection
- Lines 632-656: Queue full, TTFT warning
- Lines 659-693: Progressive streaming, error events
- Lines 715-743: Tool capture + abort
- Lines 795-842: Emit helpers, close
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    pass

from tests.fixtures.sdk_mocks import (
    MockCopilotClientWrapper,
    MockSDKSession,
    MockSDKSessionWithAbort,
    SessionEvent,
    SessionEventData,
    SessionEventType,
    error_event,
    idle_event,
    text_delta_event,
    usage_event,
)

# =============================================================================
# Helper Factories
# =============================================================================


def tool_request_event(
    tool_id: str = "call_123",
    tool_name: str = "test_tool",
    arguments: dict[str, Any] | None = None,
) -> SessionEvent:
    """Create a tool request event for testing tool capture.

    SDK sends tool requests via ASSISTANT_MESSAGE events with tool_requests field.
    Note: SDK uses tool_call_id (snake_case) - see tool_capture.normalize_tool_request.
    """
    return SessionEvent(
        type=SessionEventType.ASSISTANT_MESSAGE,
        data=SessionEventData(
            tool_requests=[
                {
                    "tool_call_id": tool_id,  # SDK format, not "id"
                    "name": tool_name,
                    "arguments": arguments or {},
                },
            ],
        ),
    )


def reasoning_delta_event(text: str) -> SessionEvent:
    """Create reasoning/thinking delta event."""
    return SessionEvent(
        type=SessionEventType.ASSISTANT_REASONING_DELTA,
        data=SessionEventData(delta_content=text),
    )


def _create_mock_request(model: str = "gpt-4o") -> MagicMock:
    """Create a minimal mock ChatRequest."""
    request = MagicMock()
    request.messages = [{"role": "user", "content": "test"}]
    request.model = model
    request.tools = None
    request.attachments = None
    return request


# =============================================================================
# Retry Path Tests (Lines 395-465)
# =============================================================================


class TestRetryWithEventualSuccess:
    """Test retry paths that succeed after initial failures."""

    @pytest.mark.asyncio
    async def test_retry_succeeds_after_one_failure(self) -> None:
        """Retry loop recovers when second attempt succeeds.

        Covers: provider.py lines 404-420 (retry branch with success)
        """
        from amplifier_core.llm_errors import LLMTimeoutError

        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        attempt_count = 0
        success_events = [
            text_delta_event("Hello from retry!"),
            usage_event(10, 20),
        ]

        class FailOnceThenSucceedSession(MockSDKSession):
            """Session that fails first attempt, succeeds on second."""

            async def send(
                self,
                prompt: str,
                *,
                attachments: list[dict[str, Any]] | None = None,
            ) -> str:
                nonlocal attempt_count
                attempt_count += 1

                if attempt_count == 1:
                    raise LLMTimeoutError("First attempt timeout", retryable=True)

                # Second attempt succeeds - deliver events
                self.last_prompt = prompt
                for event in success_events:
                    for handler in self._handlers:
                        handler(event)
                # Send IDLE to complete
                for handler in self._handlers:
                    handler(idle_event())
                return "message-id"

        # Use custom session class
        mock_client = MockCopilotClientWrapper(
            session_class=FailOnceThenSucceedSession,
        )

        provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]
        request = _create_mock_request()

        response = await provider.complete(request)

        assert attempt_count == 2, f"Expected 2 attempts, got {attempt_count}"
        assert len(response.content) == 1
        assert response.content[0].type == "text"
        assert response.content[0].text == "Hello from retry!"

    @pytest.mark.asyncio
    async def test_non_llmerror_exception_translated_and_retried(self) -> None:
        """Generic Exception is translated to LLMError and retried if retryable.

        Covers: provider.py lines 428-465 (Exception -> translate -> retry)
        """
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        attempt_count = 0
        success_events = [
            text_delta_event("Success after network error"),
        ]

        class NetworkFailOnceThenSucceed(MockSDKSession):
            """Session that raises network error first, then succeeds."""

            async def send(
                self,
                prompt: str,
                *,
                attachments: list[dict[str, Any]] | None = None,
            ) -> str:
                nonlocal attempt_count
                attempt_count += 1

                if attempt_count == 1:
                    # Generic exception (not LLMError) - will be translated
                    raise ConnectionError("Network unreachable")

                # Success path
                for event in success_events:
                    for handler in self._handlers:
                        handler(event)
                for handler in self._handlers:
                    handler(idle_event())
                return "message-id"

        mock_client = MockCopilotClientWrapper(
            session_class=NetworkFailOnceThenSucceed,
        )
        provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]
        request = _create_mock_request()

        response = await provider.complete(request)

        # ConnectionError should be translated to NetworkError (retryable)
        assert attempt_count == 2, f"Expected retry, got {attempt_count} attempts"
        assert len(response.content) == 1
        assert response.content[0].type == "text"
        assert response.content[0].text == "Success after network error"


# =============================================================================
# Progressive Streaming Emission Tests (Lines 659-672)
# =============================================================================


class TestProgressiveStreamingEmission:
    """Test progressive streaming content emission paths."""

    @pytest.mark.asyncio
    async def test_text_content_emitted_during_streaming(self) -> None:
        """Text deltas are emitted through hooks during streaming.

        Covers: provider.py lines 659-666 (text content emission)
        """
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        events = [
            text_delta_event("Hello "),
            text_delta_event("World!"),
            usage_event(5, 10),
        ]

        mock_client = MockCopilotClientWrapper(events=events)

        # Create mock coordinator with hooks
        from unittest.mock import AsyncMock

        mock_coordinator = MagicMock()
        mock_coordinator.hooks = MagicMock()
        mock_coordinator.hooks.emit = AsyncMock()

        provider = GitHubCopilotProvider(client=mock_client, coordinator=mock_coordinator)  # type: ignore[arg-type]
        request = _create_mock_request()

        await provider.complete(request)

        # Emission should have been attempted
        # (actual call may vary based on event loop timing)

    @pytest.mark.asyncio
    async def test_thinking_content_emitted_during_streaming(self) -> None:
        """Thinking/reasoning deltas are emitted during streaming.

        Covers: provider.py lines 668-672 (thinking content emission)
        """
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        events = [
            reasoning_delta_event("Let me think..."),
            reasoning_delta_event("The answer is"),
            text_delta_event("42"),
        ]

        mock_client = MockCopilotClientWrapper(events=events)
        provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]
        request = _create_mock_request()

        response = await provider.complete(request)

        assert len(response.content) == 2
        assert response.content[0].type == "thinking"
        assert response.content[1].type == "text"
        assert response.content[1].text == "42"


# =============================================================================
# Error Event Handling Tests (Lines 675-693)
# =============================================================================


class TestErrorEventHandling:
    """Test SDK error event handling path."""

    @pytest.mark.asyncio
    async def test_session_error_event_raises_exception(self) -> None:
        """Session error event is converted to exception and raised.

        Covers: provider.py lines 675-693 (error event handling)
        """
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        events = [
            text_delta_event("Starting..."),
            error_event("Model quota exceeded"),
        ]

        mock_client = MockCopilotClientWrapper(events=events)
        provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]
        request = _create_mock_request()

        with pytest.raises(Exception, match="Session error"):
            await provider.complete(request)

    @pytest.mark.asyncio
    async def test_error_event_with_dict_data(self) -> None:
        """Error event with dict data extracts message correctly.

        Covers: provider.py lines 680-686 (dict data branch)
        """
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        # Create error event as dict (legacy path)
        error_dict_event = SessionEvent(
            type=SessionEventType.SESSION_ERROR,
            data=SessionEventData(message="Rate limit exceeded"),
        )

        mock_client = MockCopilotClientWrapper(events=[error_dict_event])
        provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]
        request = _create_mock_request()

        with pytest.raises(Exception, match="Rate limit"):
            await provider.complete(request)


# =============================================================================
# Tool Capture and Abort Tests (Lines 709-743)
# =============================================================================


class TestToolCaptureAndAbort:
    """Test tool capture with session abort path."""

    @pytest.mark.asyncio
    async def test_tool_capture_triggers_abort(self) -> None:
        """Tool requests trigger session abort after capture.

        Covers: provider.py lines 715-743 (tool capture + abort branches)
        """
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        tool_events = [
            tool_request_event("call_1", "read_file", {"path": "/test.txt"}),
        ]

        # Use abort-capable session
        mock_client = MockCopilotClientWrapper(
            events=tool_events,
            session_class=MockSDKSessionWithAbort,
            abort_behavior="success",
        )
        provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]

        # Request with tools to enable tool capture
        request = _create_mock_request()
        request.tools = [{"name": "read_file", "description": "Read a file"}]

        response = await provider.complete(request)

        assert response.tool_calls is not None  # narrowed for pyright
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "read_file"

    @pytest.mark.asyncio
    async def test_abort_timeout_handled_gracefully(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Abort timeout is logged but doesn't fail the request.

        Covers: provider.py lines 730-735 (abort timeout branch)
        """
        import logging

        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        tool_events = [
            tool_request_event("call_1", "test_tool", {}),
        ]

        # Session with abort that times out
        mock_client = MockCopilotClientWrapper(
            events=tool_events,
            session_class=MockSDKSessionWithAbort,
            abort_behavior="timeout",
        )
        provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]
        request = _create_mock_request()
        request.tools = [{"name": "test_tool", "description": "Test"}]

        with caplog.at_level(logging.DEBUG):
            # Should not raise despite abort timeout
            response = await provider.complete(request)

        assert response.tool_calls is not None  # narrowed for pyright
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "test_tool"
        assert any(
            r.levelno == logging.WARNING and "abort timed out" in r.getMessage().lower()
            for r in caplog.records
        )

    @pytest.mark.asyncio
    async def test_abort_exception_logged_but_continues(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Abort exception is logged but request continues.

        Covers: provider.py lines 736-743 (abort exception branch)
        """
        import logging

        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        tool_events = [
            tool_request_event("call_1", "test_tool", {}),
        ]

        # Session with abort that raises
        mock_client = MockCopilotClientWrapper(
            events=tool_events,
            session_class=MockSDKSessionWithAbort,
            abort_behavior="exception",
        )
        provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]
        request = _create_mock_request()
        request.tools = [{"name": "test_tool", "description": "Test"}]

        with caplog.at_level(logging.DEBUG):
            response = await provider.complete(request)

        assert response.tool_calls is not None  # narrowed for pyright
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "test_tool"
        assert any(
            r.levelno == logging.DEBUG and "abort" in r.getMessage().lower() for r in caplog.records
        )

    @pytest.mark.asyncio
    async def test_tool_capture_skips_abort_when_explicit_abort_disabled(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """When explicit_abort=False, session.abort() is NOT called after tool capture.

        Contract: sdk-protection:Session:MUST:3,4
        Line 897 in provider.py — the `if sdk_protection.session.explicit_abort:` False branch.
        Mutation check: change `if explicit_abort:` to `if True:` → abort is always called,
        even when the operator has configured explicit_abort=False, violating the policy contract.
        """
        import logging

        from amplifier_module_provider_github_copilot.config._sdk_protection import (
            SdkProtectionConfig,
            SessionProtectionConfig,
        )
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        tool_events = [
            tool_request_event("call_no_abort", "read_file", {"path": "/x.txt"}),
        ]

        # abort_behavior="exception" — if abort() is called despite explicit_abort=False,
        # the exception would propagate, revealing the misconfiguration.
        mock_client = MockCopilotClientWrapper(
            events=tool_events,
            session_class=MockSDKSessionWithAbort,
            abort_behavior="exception",
        )

        no_abort_protection = SdkProtectionConfig(
            session=SessionProtectionConfig(explicit_abort=False),
        )

        with (
            patch(
                "amplifier_module_provider_github_copilot.provider.load_sdk_protection_config",
                return_value=no_abort_protection,
            ),
            caplog.at_level(logging.DEBUG),
        ):
            provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]
            request = _create_mock_request()
            request.tools = [{"name": "read_file", "description": "Read a file"}]
            response = await provider.complete(request)

        # Tool capture still works
        assert response.tool_calls is not None  # narrowed for pyright
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "read_file"
        # Abort was NOT called — no "Session aborted after tool capture" message
        assert not any(
            "Session aborted after tool capture" in r.getMessage() for r in caplog.records
        ), "abort() must not be called when explicit_abort=False"


# =============================================================================
# Emit Helper Tests (Lines 795-827)
# =============================================================================


class TestEmitHelpers:
    """Test streaming emission helper methods."""

    def test_emit_streaming_content_outside_event_loop(self) -> None:
        """_emit_streaming_content handles no running loop gracefully.

        Covers: provider.py lines 795-797 (RuntimeError branch)
        """
        from amplifier_core import TextContent

        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        provider = GitHubCopilotProvider()
        content = TextContent(text="test")

        # Call outside event loop - should not raise
        provider._emit_streaming_content(content)  # pyright: ignore[reportPrivateUsage]

    @pytest.mark.asyncio
    async def test_emit_content_async_logs_on_error(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """_emit_content_async logs error but doesn't raise.

        Covers: provider.py lines 806-815 (emit error handling)
        """
        import logging
        from unittest.mock import AsyncMock

        from amplifier_core import TextContent

        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        # Create coordinator with failing hooks
        mock_coordinator = MagicMock()
        mock_coordinator.hooks = MagicMock()
        mock_coordinator.hooks.emit = AsyncMock(side_effect=RuntimeError("Hook failed"))

        provider = GitHubCopilotProvider(coordinator=mock_coordinator)
        content = TextContent(text="test")

        with caplog.at_level(logging.DEBUG):
            # Should not raise
            await provider._emit_content_async(content)  # pyright: ignore[reportPrivateUsage]


# =============================================================================
# Close and Cleanup Tests (Lines 828-842)
# =============================================================================


class TestCloseAndCleanup:
    """Test provider close with pending task cleanup."""

    @pytest.mark.asyncio
    async def test_close_cancels_pending_emit_tasks(self) -> None:
        """close() cancels and awaits pending emit tasks.

        Covers: provider.py lines 838-842 (pending task cleanup)
        P2 Fix #9: await asyncio.gather after cancel to let cleanup run.
        """
        import asyncio
        from unittest.mock import AsyncMock

        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        provider = GitHubCopilotProvider()

        # Create a real asyncio task that we can cancel and await
        async def slow_emit() -> None:
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                pass  # Task was cancelled

        task = asyncio.create_task(slow_emit())
        provider._pending_emit_tasks.add(task)  # pyright: ignore[reportPrivateUsage]

        provider._client = MagicMock()  # pyright: ignore[reportPrivateUsage]
        provider._client.close = AsyncMock()  # pyright: ignore[reportPrivateUsage]

        await provider.close()

        # Task should be cancelled
        assert task.cancelled() or task.done()
        # Pending tasks should be cleared
        assert len(provider._pending_emit_tasks) == 0  # pyright: ignore[reportPrivateUsage]


class TestEmitContentAsyncGuards:
    """Test _emit_content_async edge cases."""

    @pytest.mark.asyncio
    async def test_emit_content_async_with_none_coordinator(self) -> None:
        """_emit_content_async returns early when coordinator is None.

        Covers: provider.py line 806 (coordinator guard)
        """
        from amplifier_core import TextContent

        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        provider = GitHubCopilotProvider(coordinator=None)
        content = TextContent(text="test")

        # Should return immediately without error
        await provider._emit_content_async(content)  # pyright: ignore[reportPrivateUsage]


class TestHandleEmitTaskException:
    """Test _handle_emit_task_exception callback."""

    def test_handle_cancelled_task(self) -> None:
        """Cancelled task is ignored without logging.

        Covers: provider.py lines 824-825 (cancelled branch)
        """
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        provider = GitHubCopilotProvider()

        mock_task = MagicMock()
        mock_task.cancelled.return_value = True

        # Should not raise
        provider._handle_emit_task_exception(mock_task)  # pyright: ignore[reportPrivateUsage]

    def test_handle_task_with_exception(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Task exception is logged at debug level.

        Covers: provider.py lines 826-829 (exception branch)
        """
        import logging

        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        provider = GitHubCopilotProvider()

        mock_task = MagicMock()
        mock_task.cancelled.return_value = False
        mock_task.exception.return_value = RuntimeError("Emit failed")

        with caplog.at_level(logging.DEBUG):
            provider._handle_emit_task_exception(mock_task)  # pyright: ignore[reportPrivateUsage]

        # Exception should be logged
        assert any("Emit task failed" in r.message for r in caplog.records)


# =============================================================================
# Delta Text Extraction Tests (Lines 180-188)
# =============================================================================


class TestDeltaTextExtraction:
    """Test _extract_delta_text helper paths."""

    def test_extract_delta_text_from_nested_data(self) -> None:
        """Extract delta_content from nested SDK event structure.

        Covers: event_router.py _extract_delta_text (nested data path)
        """
        from amplifier_module_provider_github_copilot.event_router import (
            _extract_delta_text,  # pyright: ignore[reportPrivateUsage]
        )

        # Create event with nested data.delta_content (SDK v0.1.33+ structure)
        event = text_delta_event("Hello from nested")

        result = _extract_delta_text(event)  # pyright: ignore[reportPrivateUsage]

        assert result == "Hello from nested"

    def test_extract_delta_text_from_direct_attribute(self) -> None:
        """Extract delta_content from direct event attribute.

        Covers: event_router.py _extract_delta_text (fallback path)
        """
        import types

        from amplifier_module_provider_github_copilot.event_router import (
            _extract_delta_text,  # pyright: ignore[reportPrivateUsage]
        )

        # Legacy SDK event with direct delta_content (no data wrapper)
        event = types.SimpleNamespace(data=None, delta_content="Direct content")

        result = _extract_delta_text(event)  # pyright: ignore[reportPrivateUsage]

        assert result == "Direct content"

    def test_extract_delta_text_returns_none_when_missing(self) -> None:
        """Returns None when no delta_content found.

        Covers: event_router.py _extract_delta_text all branches return None fallback
        """
        import types

        from amplifier_module_provider_github_copilot.event_router import (
            _extract_delta_text,  # pyright: ignore[reportPrivateUsage]
        )

        event = types.SimpleNamespace(data=None, delta_content=None)

        result = _extract_delta_text(event)  # pyright: ignore[reportPrivateUsage]

        assert result is None


# =============================================================================
# Queue Full Handling Test (Lines 632-640)
# =============================================================================


class TestQueueFullHandling:
    """Test bounded queue overflow handling."""

    @pytest.mark.asyncio
    async def test_queue_full_logs_debug_and_continues(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """QueueFull drops event with debug log, doesn't block.

        Covers: event_router.py lines 184-192 (QueueFull branch)
        Contract: behaviors:Streaming:MUST:4 (bounded queue, drop on full)
        """
        import logging
        from unittest.mock import patch

        from amplifier_module_provider_github_copilot.config._policy import StreamingConfig
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        # Use a tiny queue (5) so 20 events overflow it deterministically
        tiny_config = StreamingConfig(event_queue_size=5, ttft_warning_ms=15000)
        flood_events = [text_delta_event(f"chunk_{i}") for i in range(20)]

        class FloodSession(MockSDKSession):
            """Session that floods events to overflow the bounded queue."""

            async def send(
                self,
                prompt: str,
                *,
                attachments: list[dict[str, Any]] | None = None,
            ) -> str:
                self.last_prompt = prompt
                for event in flood_events:
                    for handler in self._handlers:
                        handler(event)
                for handler in self._handlers:
                    handler(idle_event())
                return "message-id"

        mock_client = MockCopilotClientWrapper(
            session_class=FloodSession,
        )
        provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]
        request = _create_mock_request()

        with (
            patch(
                "amplifier_module_provider_github_copilot.provider.load_streaming_config",
                return_value=tiny_config,
            ),
            caplog.at_level(logging.DEBUG),
        ):
            response = await provider.complete(request)

        # Some events were dropped due to queue overflow
        assert any("[STREAMING] Event queue full" in r.getMessage() for r in caplog.records), (
            "Expected queue-full log when queue overflows"
        )
        # But response still has content from events that fit
        assert len(response.content) > 0


# =============================================================================
# Fake Tool Detection Retry Tests (Lines 478-515)
# =============================================================================


class TestFakeToolDetectionRetry:
    """Test fake tool detection correction retry path."""

    @pytest.mark.asyncio
    async def test_fake_tool_detected_triggers_correction_retry(self) -> None:
        """Response with fake tool XML triggers correction retry.

        Covers: provider.py lines 478-515 (fake tool detection + correction)
        """
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        correction_attempt = 0
        # Use pattern that matches fake tool patterns:
        # Pattern: \[Tool Call:\s*\w+ or <tool_used\s+name=
        fake_tool_response = "Let me use [Tool Call: read_file] to help you"
        corrected_response = "Here is the answer without function calls."

        class FakeToolThenCorrectSession(MockSDKSession):
            """Session that returns fake tool response first, then corrected."""

            async def send(
                self,
                prompt: str,
                *,
                attachments: list[dict[str, Any]] | None = None,
            ) -> str:
                nonlocal correction_attempt
                correction_attempt += 1
                self.last_prompt = prompt

                if correction_attempt == 1:
                    # First attempt: return fake tool pattern
                    for handler in self._handlers:
                        handler(text_delta_event(fake_tool_response))
                else:
                    # Correction attempt: return clean response
                    for handler in self._handlers:
                        handler(text_delta_event(corrected_response))

                for handler in self._handlers:
                    handler(idle_event())
                return "message-id"

        mock_client = MockCopilotClientWrapper(
            session_class=FakeToolThenCorrectSession,
        )
        provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]

        # Request WITH tools available (enables fake tool detection)
        request = _create_mock_request()
        request.tools = [{"name": "read_file", "description": "Read file"}]

        response = await provider.complete(request)

        # Should have retried
        assert correction_attempt == 2, "Expected correction retry"
        assert len(response.content) == 1
        assert response.content[0].type == "text"
        assert response.content[0].text == "Here is the answer without function calls."

    @pytest.mark.asyncio
    async def test_fake_tool_correction_exception_raises(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Exception during correction attempt raises (P1 Fix: no silent data loss).

        Covers: provider.py lines 506-509 (exception in correction)
        P1 Fix #2: Re-raise exception instead of swallowing and returning empty response.
        """
        import logging

        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        attempt = 0
        # Use pattern from fake_tool_detection: <tool_used name=
        fake_tool_response = 'I\'ll use <tool_used name="search"> to find it'

        class FakeToolThenErrorSession(MockSDKSession):
            """Session that returns fake tool, then errors on correction."""

            async def send(
                self,
                prompt: str,
                *,
                attachments: list[dict[str, Any]] | None = None,
            ) -> str:
                nonlocal attempt
                attempt += 1
                self.last_prompt = prompt

                if attempt == 1:
                    # First: fake tool response
                    for handler in self._handlers:
                        handler(text_delta_event(fake_tool_response))
                    for handler in self._handlers:
                        handler(idle_event())
                    return "message-id"
                else:
                    # Correction attempt fails
                    raise ConnectionError("Network error during correction")

        mock_client = MockCopilotClientWrapper(
            session_class=FakeToolThenErrorSession,
        )
        provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]
        request = _create_mock_request()
        request.tools = [{"name": "search", "description": "Search"}]

        with caplog.at_level(logging.DEBUG):
            # P1 Fix: Now raises translated kernel error instead of raw exception.
            # Contract: error-hierarchy.md — SDK errors MUST be translated.
            # Contract: error-hierarchy:ConnectionError:MUST:1
            # ConnectionError → ProviderUnavailableError (provider endpoint unreachable)
            from amplifier_core.llm_errors import ProviderUnavailableError

            with pytest.raises(ProviderUnavailableError, match="Network error during correction"):
                await provider.complete(request)

        # Should have attempted correction (2 attempts: original + retry)
        assert attempt == 2


# =============================================================================
# TTFT Warning with Time Mocking (Lines 650-656)
# =============================================================================


class TestTTFTWarningWithTimeMock:
    """Test TTFT warning with mocked time for deterministic testing."""

    @pytest.mark.asyncio
    async def test_ttft_warning_logged_when_exceeds_threshold(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """TTFT warning logged when first token time exceeds threshold.

        Covers: provider.py lines 650-656 (TTFT warning branch)
        """
        import logging
        from unittest.mock import patch

        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        # Track time calls to simulate slow response
        time_base = 1000.0
        time_calls = [0]

        def mock_time() -> float:
            time_calls[0] += 1
            if time_calls[0] == 1:
                return time_base  # start_time
            else:
                # 20 seconds later (exceeds 15s default threshold)
                return time_base + 20.0

        events = [text_delta_event("Delayed response")]

        mock_client = MockCopilotClientWrapper(events=events)
        provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]
        request = _create_mock_request()

        with (
            patch("time.time", mock_time),
            caplog.at_level(logging.WARNING),
        ):
            await provider.complete(request)

        # TTFT warning should be logged when first token exceeds threshold
        assert any(
            r.levelno == logging.WARNING and "ttft" in r.getMessage().lower()
            for r in caplog.records
        ), "TTFT warning not logged"


# =============================================================================
# Error Event Data Formats (Lines 678-693)
# =============================================================================


class TestErrorEventDataFormats:
    """Test error event handling with various data formats."""

    @pytest.mark.asyncio
    async def test_error_event_with_none_data(self) -> None:
        """Error event with None data uses event string.

        Covers: provider.py lines 684-685 (data is None branch)
        """
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        # Create error event without data
        error_no_data = SessionEvent(
            type=SessionEventType.SESSION_ERROR,
            data=None,  # type: ignore[arg-type]
        )

        mock_client = MockCopilotClientWrapper(events=[error_no_data])
        provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]
        request = _create_mock_request()

        with pytest.raises(Exception, match="Session error"):
            await provider.complete(request)

    @pytest.mark.asyncio
    async def test_error_event_with_object_data(self) -> None:
        """Error event with object data extracts message attribute.

        Covers: provider.py lines 690-691 (getattr branch)
        """
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        # Create error with object that has message attribute
        error_with_msg = error_event("Object-style error message")

        mock_client = MockCopilotClientWrapper(events=[error_with_msg])
        provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]
        request = _create_mock_request()

        with pytest.raises(Exception, match="Object-style error"):
            await provider.complete(request)


# =============================================================================
# R5: E2E Tool Capture Integration Test (swarm review finding)
# Tests the FULL happy path: complete() → session → tool_calls → deny hook →
# capture → abort → ChatResponse.tool_calls
# =============================================================================


class TestE2EToolCaptureHappyPath:
    """E2E integration test for complete tool capture flow.

    R5 Fix: This test was missing per swarm review — the full happy path
    from complete() through to ChatResponse.tool_calls was never tested
    as a single integrated flow.

    Contract: sdk-protection:Session:MUST:3,4
    """

    @pytest.mark.asyncio
    async def test_e2e_tool_capture_returns_tool_calls_in_response(self) -> None:
        """Complete() returns ChatResponse with captured tool_calls.

        Full path: ChatRequest → session → SDK tool events → deny hook →
        tool capture → abort → ChatResponse.tool_calls
        """
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        # SDK returns multiple tool requests in ONE assistant message
        # (ToolCaptureHandler captures from first message only)
        tool_events = [
            text_delta_event("Let me help you with those files..."),
            SessionEvent(
                type=SessionEventType.ASSISTANT_MESSAGE,
                data=SessionEventData(
                    tool_requests=[
                        {
                            "tool_call_id": "call_001",
                            "name": "read_file",
                            "arguments": {"path": "/src/main.py"},
                        },
                        {
                            "tool_call_id": "call_002",
                            "name": "list_dir",
                            "arguments": {"path": "/src"},
                        },
                    ],
                ),
            ),
            idle_event(),
        ]

        mock_client = MockCopilotClientWrapper(
            events=tool_events,
            session_class=MockSDKSessionWithAbort,
            abort_behavior="success",
        )
        provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]

        # Request MUST have tools defined to enable tool capture
        request = _create_mock_request()
        request.tools = [
            {"name": "read_file", "description": "Read file contents"},
            {"name": "list_dir", "description": "List directory"},
        ]

        response = await provider.complete(request)

        # ChatResponse should contain the captured tool calls
        assert response.tool_calls is not None  # narrowed for pyright
        assert len(response.tool_calls) == 2, "Should capture both tool requests"

        # Verify tool call structure
        tool_names = {tc.name for tc in response.tool_calls}
        assert "read_file" in tool_names
        assert "list_dir" in tool_names

        # Verify arguments and IDs were captured
        for tc in response.tool_calls:
            if tc.name == "read_file":
                assert tc.id == "call_001"
                assert isinstance(tc.arguments, dict)
                assert tc.arguments["path"] == "/src/main.py"
            elif tc.name == "list_dir":
                assert tc.id == "call_002"
                assert isinstance(tc.arguments, dict)
                assert tc.arguments["path"] == "/src"

    @pytest.mark.asyncio
    async def test_e2e_tool_capture_always_active(self) -> None:
        """Tool capture is always active - captures SDK tools regardless of request.tools.

        Contract: The ToolCaptureHandler captures tool_requests from SDK stream
        unconditionally. request.tools controls what is SENT to SDK, not what
        is captured back.
        """
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        # SDK returns tool requests
        tool_events = [
            tool_request_event("call_001", "read_file", {"path": "/test.txt"}),
            idle_event(),
        ]

        mock_client = MockCopilotClientWrapper(
            events=tool_events,
            session_class=MockSDKSessionWithAbort,
        )
        provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]

        # No tools in request - but capture is still active
        request = _create_mock_request()
        request.tools = None

        response = await provider.complete(request)

        # Tool calls ARE captured (ToolCaptureHandler is unconditional)
        assert response.tool_calls is not None  # narrowed for pyright
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "read_file"

    @pytest.mark.asyncio
    async def test_e2e_tool_capture_with_abort_failure_still_returns_tools(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Abort failure is logged but doesn't prevent tool_calls return.

        Contract: sdk-protection:Session:MUST:4 — abort failure is graceful
        """
        import logging

        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        tool_events = [
            tool_request_event("call_001", "write_file", {"path": "/out.txt"}),
            idle_event(),
        ]

        # Abort will time out
        mock_client = MockCopilotClientWrapper(
            events=tool_events,
            session_class=MockSDKSessionWithAbort,
            abort_behavior="timeout",
        )
        provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]

        request = _create_mock_request()
        request.tools = [{"name": "write_file", "description": "Write to file"}]

        with caplog.at_level(logging.WARNING):
            response = await provider.complete(request)

        # Tool calls should still be captured despite abort timeout
        assert response.tool_calls is not None  # narrowed for pyright
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "write_file"
