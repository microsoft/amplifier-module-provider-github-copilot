"""Event Router Module.

Extracted from provider.py per Comprehensive Code Review P1.6.
Handles SDK event routing with 5 concerns:
1. Idle detection
2. Error handling
3. Usage capture
4. Tool capture
5. Streaming emission

Contract: streaming-contract:abort-on-capture:MUST:1
Contract: behaviors:Streaming:MUST:1,4
Contract: streaming-contract:ProgressiveStreaming:SHOULD:1
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from .sdk_adapter import (
    ToolCaptureHandler,
    extract_event_type,
    extract_usage_data,
    is_error_event,
    is_idle_event,
    is_usage_event,
)

if TYPE_CHECKING:
    from .streaming import EventConfig

logger = logging.getLogger(__name__)


def _extract_delta_text(sdk_event: Any) -> str | None:
    """Extract text delta from an SDK streaming event.

    Contract: streaming-contract:ProgressiveStreaming:SHOULD:1

    Handles SDK v0.2.0 nested data structure:
    - event.data.delta_content for text deltas
    - Direct string content as fallback

    Args:
        sdk_event: SDK SessionEvent object

    Returns:
        Extracted text delta or None if not found
    """
    # Try nested data structure first (SDK v0.2.0+)
    sdk_data = getattr(sdk_event, "data", None)
    if sdk_data is not None:
        delta_content = getattr(sdk_data, "delta_content", None)
        if delta_content and isinstance(delta_content, str):
            return delta_content

    # Fallback: check direct delta_content attribute
    delta_content = getattr(sdk_event, "delta_content", None)
    if delta_content and isinstance(delta_content, str):
        return delta_content

    return None


class EventRouter:
    """Routes SDK events to appropriate handlers.

    Extracted from provider.py closure to improve code organization.
    Implements __call__ to be used as sdk_session.on() callback.

    Contract: streaming-contract:abort-on-capture:MUST:1
    Contract: behaviors:Streaming:MUST:1 (TTFT warning)
    Contract: behaviors:Streaming:MUST:4 (bounded queue, drop on full)
    """

    def __init__(
        self,
        *,
        queue: asyncio.Queue[Any],
        idle_event: asyncio.Event,
        error_holder: list[Exception],
        usage_holder: list[dict[str, int]],
        capture_handler: ToolCaptureHandler,
        ttft_state: dict[str, Any],
        ttft_threshold_ms: int,
        event_config: EventConfig,
        emit_streaming_content: Callable[[Any], None],
    ) -> None:
        """Initialize event router with all required dependencies.

        Args:
            queue: Bounded async queue for event processing
            idle_event: Event signaling session idle/completion
            error_holder: List to capture session errors
            usage_holder: List to capture usage data
            capture_handler: Tool capture handler instance
            ttft_state: Mutable dict for TTFT tracking
            ttft_threshold_ms: TTFT warning threshold
            event_config: Event classification config
            emit_streaming_content: Callback for progressive streaming
        """
        self._queue = queue
        self._idle = idle_event
        self._errors = error_holder
        self._usage = usage_holder
        self._capture_handler = capture_handler
        self._ttft = ttft_state
        self._ttft_threshold_ms = ttft_threshold_ms
        self._config = event_config
        self._emit_streaming = emit_streaming_content

    def __call__(self, sdk_event: Any) -> None:
        """Handle incoming SDK event.

        This is the main entry point, called by sdk_session.on().
        Processes events in order of priority:
        1. Usage capture (race condition fix)
        2. Idle detection
        3. Error handling (CRITICAL)
        4. Tool capture (CRITICAL)
        5. Stream queueing (best-effort)
        6. TTFT check
        7. Progressive streaming

        Contract: streaming-contract:abort-on-capture:MUST:1
        Contract: behaviors:Streaming:MUST:4 (bounded queue, drop on full)
        """
        event_type = extract_event_type(sdk_event)

        # Classify event
        is_idle = is_idle_event(event_type, idle_events=self._config.idle_event_types)
        is_err = is_error_event(event_type, error_events=self._config.error_event_types)
        is_usage = is_usage_event(event_type, usage_events=self._config.usage_event_types)

        logger.debug(
            "[EVENT_ROUTER] type=%s, is_idle=%s, is_error=%s, is_usage=%s",
            event_type,
            is_idle,
            is_err,
            is_usage,
        )

        # 1. Capture usage immediately to avoid race condition
        # SDK sends assistant.usage AFTER session.idle in some cases
        # Contract: streaming-contract:usage:MUST:1
        if is_usage:
            usage_data = extract_usage_data(sdk_event)
            if usage_data:
                self._usage.clear()
                self._usage.append(usage_data)

        # 2. Idle detection
        if is_idle:
            logger.debug("[EVENT_ROUTER] IDLE DETECTED - setting idle_event")
            self._idle.set()

        # ═══════════════════════════════════════════════════════════════
        # CRITICAL PATH: Process errors and tool capture BEFORE queue
        # These MUST NOT be lost even if streaming queue overflows.
        # Contract: streaming-contract:abort-on-capture:MUST:1
        # ═══════════════════════════════════════════════════════════════

        # 3. Error handling (CRITICAL - must not be skipped)
        if is_err:
            self._handle_error(sdk_event)
            return  # Error events handled separately - don't queue (prevents orphan)

        # 4. Tool capture (CRITICAL - must not be skipped)
        # P2 Fix: Exclude usage events — they don't contain tool data.
        if not is_idle and not is_err and not is_usage:
            self._capture_handler.on_event(sdk_event)

        # ═══════════════════════════════════════════════════════════════
        # STREAMING PATH: Best-effort queue for progressive UI updates
        # OK to drop deltas if queue full - final response has complete text
        # Contract: behaviors:Streaming:MUST:4 (bounded queue, drop on full)
        # ═══════════════════════════════════════════════════════════════

        # 5. Queue event for processing
        try:
            self._queue.put_nowait(sdk_event)
        except asyncio.QueueFull:
            logger.debug(
                "[STREAMING] Event queue full, dropping delta: %s",
                event_type,
            )
            return  # Skip streaming emission, but critical path already done

        # 6. TTFT check on first content event
        if event_type:
            self._check_ttft(event_type)

        # 7. Progressive streaming
        if event_type:
            self._emit_progressive_content(event_type, sdk_event)

    def _handle_error(self, sdk_event: Any) -> None:
        """Extract and record error from SDK event.

        Note: SDK sends error EVENTS, not exceptions. We extract the message
        and wrap in Exception. The provider.py catch block translates this
        via translate_sdk_error() which pattern-matches the message to determine
        the appropriate kernel error type (e.g., ProviderUnavailableError).

        A-02 Review: Using generic Exception is intentional - we don't have an
        original SDK exception to preserve, only an event payload with a message.
        """
        logger.debug("[EVENT_ROUTER] ERROR DETECTED - setting idle_event")
        sdk_event_str = str(sdk_event)

        data: Any
        if isinstance(sdk_event, dict):
            typed_evt = cast(dict[str, Any], sdk_event)
            data = typed_evt.get("data")
        else:
            data = getattr(sdk_event, "data", None)

        error_msg: str
        if data is None:
            error_msg = sdk_event_str
        elif isinstance(data, dict):
            typed_data = cast(dict[str, Any], data)
            msg_val = typed_data.get("message")
            error_msg = str(msg_val) if msg_val is not None else sdk_event_str
        else:
            error_msg = str(getattr(data, "message", sdk_event_str))

        err = Exception(f"Session error: {error_msg}")
        self._errors.append(err)
        self._idle.set()

    def _check_ttft(self, event_type: str) -> None:
        """Check time to first token and warn if slow.

        Contract: behaviors:Streaming:MUST:1
        """
        if self._ttft["checked"]:
            return

        if event_type in self._config.content_event_types:
            self._ttft["checked"] = True
            elapsed_ms = (time.time() - self._ttft["start_time"]) * 1000
            if elapsed_ms > self._ttft_threshold_ms:
                logger.warning(
                    "[TTFT] Slow time to first token: %.0fms (threshold: %dms)",
                    elapsed_ms,
                    self._ttft_threshold_ms,
                )

    def _emit_progressive_content(self, event_type: str, sdk_event: Any) -> None:
        """Emit content deltas for real-time UI updates.

        Contract: streaming-contract:ProgressiveStreaming:SHOULD:1
        """
        if event_type in self._config.text_content_types:
            delta_text = _extract_delta_text(sdk_event)
            if delta_text:
                from amplifier_core import TextContent

                self._emit_streaming(TextContent(text=delta_text))
        elif event_type in self._config.thinking_content_types:
            delta_text = _extract_delta_text(sdk_event)
            if delta_text:
                from amplifier_core import ThinkingContent

                self._emit_streaming(ThinkingContent(text=delta_text))
