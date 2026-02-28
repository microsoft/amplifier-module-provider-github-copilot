"""
SDK Driver for Copilot CLI SDK.

This module implements the SDK Driver pattern for taming the Copilot SDK's
internal agent loop. It provides:

1. LoopController - Track and control SDK turn iterations
2. ToolCaptureStrategy - Capture tools from first turn only
3. CircuitBreaker - Trip when loop exceeds limits
4. SdkEventHandler - Coordinate all components via SDK events

Evidence Base:
- Session a1a0af17: 305 turns, 607 tools
- Copilot SDK denial_behavior = RETRY
- Solution: Capture first turn, abort immediately

Design Philosophy:
- SDKs are agentic and will stay agentic
- We don't fight the SDK, we CONTROL it
- Early capture + fast abort = no wasted turns
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ._constants import (
    CAPTURE_FIRST_TURN_ONLY,
    DEDUPLICATE_TOOL_CALLS,
    SDK_MAX_TURNS_DEFAULT,
    SDK_MAX_TURNS_HARD_LIMIT,
    LoopExitMethod,
)
from .exceptions import CopilotSdkLoopError, detect_rate_limit_error

logger = logging.getLogger(__name__)


@dataclass
class CapturedToolCall:
    """A tool call captured from SDK events."""

    id: str
    name: str
    arguments: dict[str, Any]
    turn: int  # Which turn this was captured from

    def __hash__(self) -> int:
        """Hash for deduplication (by name + arguments)."""
        args_str = json.dumps(self.arguments, sort_keys=True)
        return hash((self.name, args_str))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CapturedToolCall):
            return NotImplemented
        return self.name == other.name and self.arguments == other.arguments


@dataclass
class LoopState:
    """Current state of SDK's internal loop."""

    turn_count: int = 0
    first_turn_captured: bool = False
    captured_tools: list[CapturedToolCall] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    abort_requested: bool = False
    error: Exception | None = None

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time


class LoopController:
    """
    Controls the SDK's internal agent loop.

    Responsibilities:
    - Track turn count
    - Signal when to abort
    - Enforce circuit breaker limits

    Evidence:
    - SDK fires ASSISTANT_TURN_START before each turn
    - We can count turns and abort before loop runs away
    """

    def __init__(
        self,
        max_turns: int = SDK_MAX_TURNS_DEFAULT,
        exit_method: LoopExitMethod = LoopExitMethod.ABORT,
    ):
        self.max_turns = min(max_turns, SDK_MAX_TURNS_HARD_LIMIT)
        self.exit_method = exit_method
        self.state = LoopState()
        self._abort_callback: Callable[[], None] | None = None
        self._abort_callback_invoked = False

    def set_abort_callback(self, callback: Callable[[], None]) -> None:
        """Set callback to invoke when abort is needed."""
        self._abort_callback = callback

    def on_turn_start(self) -> bool:
        """
        Called when ASSISTANT_TURN_START event fires.

        Returns:
            True if turn should proceed, False if should abort
        """
        self.state.turn_count += 1

        logger.debug(f"[SDK_DRIVER] Turn {self.state.turn_count}/{self.max_turns} started")

        # Check circuit breaker
        if self.state.turn_count > self.max_turns:
            logger.warning(
                f"[SDK_DRIVER] Circuit breaker tripped! "
                f"Turn {self.state.turn_count} exceeds max {self.max_turns}"
            )
            self.state.abort_requested = True

            if self._abort_callback and not self._abort_callback_invoked:
                self._abort_callback_invoked = True
                self._abort_callback()

            return False

        return True

    def should_abort(self) -> bool:
        """Check if we should abort the loop."""
        return self.state.abort_requested

    def request_abort(self, reason: str = "external") -> None:
        """Request loop abort."""
        logger.info(f"[SDK_DRIVER] Abort requested: {reason}")
        self.state.abort_requested = True
        if self._abort_callback and not self._abort_callback_invoked:
            self._abort_callback_invoked = True
            self._abort_callback()


class ToolCaptureStrategy:
    """
    Captures tool calls from SDK events with configurable strategy.

    Strategies:
    - FIRST_TURN_ONLY: Capture from first ASSISTANT_MESSAGE only (recommended)
    - ALL_TURNS: Capture from all turns (with deduplication)

    Evidence:
    - 607 tools captured from 305 turns = accumulation bug
    - First turn had the valid 2 tools (delegate, report_intent)
    - Subsequent turns were retries of the same tools
    """

    def __init__(
        self,
        first_turn_only: bool = CAPTURE_FIRST_TURN_ONLY,
        deduplicate: bool = DEDUPLICATE_TOOL_CALLS,
    ):
        self.first_turn_only = first_turn_only
        self.deduplicate = deduplicate
        self._captured: list[CapturedToolCall] = []
        self._seen_hashes: set[int] = set()
        self._first_capture_done = False
        self._current_turn = 0

    def set_current_turn(self, turn: int) -> None:
        """Update current turn number."""
        self._current_turn = turn

    def capture_from_event(
        self,
        tool_requests: list[Any],
    ) -> list[CapturedToolCall]:
        """
        Process tool requests from ASSISTANT_MESSAGE event.

        Args:
            tool_requests: Raw tool_requests from SDK event

        Returns:
            List of newly captured tools (may be empty if filtered)
        """
        # First-turn-only strategy
        if self.first_turn_only and self._first_capture_done:
            logger.debug(
                f"[SDK_DRIVER] Ignoring tools from turn {self._current_turn} "
                f"(first-turn-only strategy)"
            )
            return []

        newly_captured: list[CapturedToolCall] = []

        for tr in tool_requests:
            tool_id = getattr(tr, "tool_call_id", "") or ""
            name = getattr(tr, "name", "") or ""
            arguments = getattr(tr, "arguments", {}) or {}

            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"raw": arguments}

            captured = CapturedToolCall(
                id=tool_id,
                name=name,
                arguments=arguments,
                turn=self._current_turn,
            )

            # Deduplication
            if self.deduplicate:
                tool_hash = hash(captured)
                if tool_hash in self._seen_hashes:
                    logger.debug(f"[SDK_DRIVER] Skipping duplicate tool: {name}")
                    continue
                self._seen_hashes.add(tool_hash)

            self._captured.append(captured)
            newly_captured.append(captured)

        if newly_captured:
            self._first_capture_done = True
            logger.info(
                f"[SDK_DRIVER] Captured {len(newly_captured)} tool(s) "
                f"from turn {self._current_turn}: "
                f"{[t.name for t in newly_captured]}"
            )

        return newly_captured

    @property
    def captured_tools(self) -> list[CapturedToolCall]:
        """All captured tools."""
        return self._captured.copy()

    @property
    def has_tools(self) -> bool:
        """Whether any tools have been captured."""
        return len(self._captured) > 0


class CircuitBreaker:
    """
    Circuit breaker for SDK loop protection.

    Trips when:
    - Turn count exceeds maximum
    - Request timeout reached
    - Anomalous behavior detected

    Evidence:
    - 305 turns in 20 minutes = ~4 seconds per turn
    - Normal request: 1-2 turns in <10 seconds
    - Circuit breaker at 3 turns catches runaway loops early
    """

    def __init__(
        self,
        max_turns: int = SDK_MAX_TURNS_DEFAULT,
        timeout_seconds: float = 60.0,
    ):
        self.max_turns = min(max_turns, SDK_MAX_TURNS_HARD_LIMIT)
        self.timeout_seconds = timeout_seconds
        self._tripped = False
        self._trip_reason: str | None = None
        self._start_time: float | None = None

    def start(self) -> None:
        """Start the circuit breaker timer."""
        self._start_time = time.time()
        self._tripped = False
        self._trip_reason = None

    def check_turn(self, turn_count: int) -> bool:
        """
        Check if turn count exceeds limit.

        Returns:
            True if OK, False if tripped
        """
        if turn_count > self.max_turns:
            self._trip(f"turn_count={turn_count} > max={self.max_turns}")
            return False
        return True

    def check_timeout(self) -> bool:
        """
        Check if timeout exceeded.

        Returns:
            True if OK, False if tripped
        """
        if self._start_time is None:
            return True

        elapsed = time.time() - self._start_time
        if elapsed > self.timeout_seconds:
            self._trip(f"timeout={elapsed:.1f}s > max={self.timeout_seconds}s")
            return False
        return True

    def _trip(self, reason: str) -> None:
        """Trip the circuit breaker."""
        if not self._tripped:
            logger.warning(f"[SDK_DRIVER] Circuit breaker TRIPPED: {reason}")
            self._tripped = True
            self._trip_reason = reason

    @property
    def is_tripped(self) -> bool:
        """Check if circuit breaker is tripped."""
        return self._tripped

    @property
    def trip_reason(self) -> str | None:
        """Get trip reason if tripped."""
        return self._trip_reason


class SdkEventHandler:
    """
    Unified event handler that coordinates all SDK Driver components.

    Subscribes to SDK session events and routes to:
    - LoopController: Turn tracking, abort decisions
    - ToolCaptureStrategy: Tool extraction from events
    - CircuitBreaker: Limit enforcement

    This is the main integration point with the provider's _complete_streaming().

    Usage:
        handler = SdkEventHandler(max_turns=3)
        unsubscribe = session.on(handler.on_event)

        await session.send({"prompt": prompt})
        await handler.wait_for_capture_or_idle(timeout=60)

        if handler.should_abort:
            await session.abort()

        return handler.captured_tools
    """

    def __init__(
        self,
        max_turns: int = SDK_MAX_TURNS_DEFAULT,
        first_turn_only: bool = CAPTURE_FIRST_TURN_ONLY,
        deduplicate: bool = DEDUPLICATE_TOOL_CALLS,
        emit_event: Callable[[str, dict], None] | None = None,
    ):
        self.loop_controller = LoopController(max_turns=max_turns)
        self.tool_capture = ToolCaptureStrategy(
            first_turn_only=first_turn_only,
            deduplicate=deduplicate,
        )
        self.circuit_breaker = CircuitBreaker(max_turns=max_turns)
        self._emit_event = emit_event

        # Synchronization
        self._capture_event = asyncio.Event()
        self._idle_event = asyncio.Event()
        self._error_event: Exception | None = None
        self._abort_task: asyncio.Task | None = None
        self._session: Any = None

        # Content accumulators
        self.text_content: list[str] = []
        self.thinking_content: list[str] = []

        # Usage tracking
        self.usage_data: dict[str, Any] = {}

    def bind_session(self, session: Any) -> None:
        """Bind to session for abort capability."""
        self._session = session

        def request_abort() -> None:
            if self._session and not self._abort_task:
                self._abort_task = asyncio.create_task(
                    self._do_abort(),
                    name="sdk_driver_abort",
                )

        self.loop_controller.set_abort_callback(request_abort)

    async def _do_abort(self) -> None:
        """Execute abort on session."""
        if self._session:
            try:
                logger.info("[SDK_DRIVER] Executing session.abort()")
                await self._session.abort()
            except Exception as e:
                logger.warning(f"[SDK_DRIVER] Abort failed: {e}")

    def on_event(self, event: Any) -> None:
        """
        Process SDK session event.

        This is the callback passed to session.on().
        """
        from copilot.generated.session_events import SessionEventType

        event_type = event.type
        data = event.data

        # ═══════════════════════════════════════════════════════════
        # Turn tracking
        # ═══════════════════════════════════════════════════════════
        if event_type == SessionEventType.ASSISTANT_TURN_START:
            proceed = self.loop_controller.on_turn_start()
            self.tool_capture.set_current_turn(self.loop_controller.state.turn_count)

            # Check circuit breaker
            if not self.circuit_breaker.check_turn(self.loop_controller.state.turn_count):
                # Emit BEFORE abort so consumers see trip then abort sequence
                if self._emit_event:
                    self._emit_event(
                        "sdk:circuit_breaker_trip",
                        {
                            "turn": self.loop_controller.state.turn_count,
                            "reason": self.circuit_breaker.trip_reason or "turn_limit_exceeded",
                        },
                    )
                self.loop_controller.request_abort("circuit_breaker_turn_limit")
                # Emit abort_requested event
                if self._emit_event:
                    self._emit_event(
                        "sdk:abort_requested",
                        {
                            "turn": self.loop_controller.state.turn_count,
                            "reason": "circuit_breaker_turn_limit",
                        },
                    )

            # Emit observability event
            if self._emit_event:
                self._emit_event(
                    "sdk:turn_start",
                    {
                        "turn": self.loop_controller.state.turn_count,
                        "max_turns": self.loop_controller.max_turns,
                    },
                )

            if not proceed:
                # Circuit breaker tripped
                self._capture_event.set()  # Unblock waiter

        # ═══════════════════════════════════════════════════════════
        # Content streaming
        # ═══════════════════════════════════════════════════════════
        elif event_type == SessionEventType.ASSISTANT_MESSAGE_DELTA:
            delta = getattr(data, "delta_content", None)
            if delta:
                self.text_content.append(delta)

        elif event_type == SessionEventType.ASSISTANT_REASONING_DELTA:
            delta = getattr(data, "delta_content", None)
            if delta:
                self.thinking_content.append(delta)

        elif event_type == SessionEventType.ASSISTANT_REASONING:
            content = getattr(data, "content", None)
            if content:
                self.thinking_content.append(content)

        # ═══════════════════════════════════════════════════════════
        # Usage tracking
        # ═══════════════════════════════════════════════════════════
        elif event_type == SessionEventType.ASSISTANT_USAGE:
            input_tokens = getattr(data, "input_tokens", None)
            output_tokens = getattr(data, "output_tokens", None)
            if input_tokens is not None:
                self.usage_data["input_tokens"] = input_tokens
            if output_tokens is not None:
                self.usage_data["output_tokens"] = output_tokens
            logger.debug(f"[SDK_DRIVER] Usage update: {self.usage_data}")

        # ═══════════════════════════════════════════════════════════
        # Tool capture (CRITICAL PATH)
        # ═══════════════════════════════════════════════════════════
        elif event_type == SessionEventType.ASSISTANT_MESSAGE:
            # Capture content as fallback when no deltas were received
            content = getattr(data, "content", None)
            if content and not self.text_content:
                self.text_content.append(content)

            tool_reqs = getattr(data, "tool_requests", None)

            if tool_reqs:
                newly_captured = self.tool_capture.capture_from_event(tool_reqs)

                if newly_captured and self.tool_capture.first_turn_only:
                    # CRITICAL: We have tools from first turn
                    # Signal capture complete and request abort
                    logger.info(
                        f"[SDK_DRIVER] First-turn capture complete with "
                        f"{len(newly_captured)} tools - requesting abort"
                    )
                    self._capture_event.set()
                    self.loop_controller.request_abort("first_turn_capture_complete")

                    # Emit observability events
                    if self._emit_event:
                        self._emit_event(
                            "sdk:capture_complete",
                            {
                                "turn": self.loop_controller.state.turn_count,
                                "tool_count": len(self.tool_capture.captured_tools),
                                "tools": [t.name for t in self.tool_capture.captured_tools],
                            },
                        )
                        self._emit_event(
                            "sdk:abort_requested",
                            {
                                "turn": self.loop_controller.state.turn_count,
                                "reason": "first_turn_capture_complete",
                            },
                        )

        # ═══════════════════════════════════════════════════════════
        # Session lifecycle
        # ═══════════════════════════════════════════════════════════
        elif event_type == SessionEventType.SESSION_IDLE:
            logger.debug("[SDK_DRIVER] SESSION_IDLE received")
            self._idle_event.set()
            self._capture_event.set()  # Also unblock capture waiter

        elif event_type == SessionEventType.SESSION_ERROR:
            error_msg = getattr(data, "message", str(data))
            logger.error(f"[SDK_DRIVER] SESSION_ERROR: {error_msg}")
            rate_limit_err = detect_rate_limit_error(error_msg)
            if rate_limit_err is not None:
                self._error_event = rate_limit_err
            else:
                self._error_event = Exception(f"Session error: {error_msg}")
            self._idle_event.set()
            self._capture_event.set()

    async def wait_for_capture_or_idle(self, timeout: float) -> None:
        """
        Wait for tool capture or session idle, whichever comes first.

        With first-turn-only strategy:
        - If tools captured: Returns immediately after first turn
        - If no tools: Waits for SESSION_IDLE

        Args:
            timeout: Maximum wait time in seconds

        Raises:
            CopilotSdkLoopError: If circuit breaker tripped
            asyncio.TimeoutError: If timeout exceeded
        """
        logger.debug(
            f"[SDK_DRIVER] Waiting for capture or idle "
            f"(timeout={timeout}s, max_turns={self.loop_controller.max_turns})"
        )
        self.circuit_breaker.start()

        try:
            await asyncio.wait_for(
                self._capture_event.wait(),
                timeout=timeout,
            )
        except TimeoutError:
            self.circuit_breaker.check_timeout()
            if self.circuit_breaker.is_tripped:
                raise CopilotSdkLoopError(
                    f"SDK loop timeout: {self.circuit_breaker.trip_reason}",
                    turn_count=self.loop_controller.state.turn_count,
                    max_turns=self.loop_controller.max_turns,
                    tool_calls_captured=len(self.tool_capture.captured_tools),
                ) from None
            raise

        logger.debug(
            f"[SDK_DRIVER] Wait completed - "
            f"turns={self.loop_controller.state.turn_count}, "
            f"tools={len(self.tool_capture.captured_tools)}, "
            f"abort={'yes' if self.loop_controller.should_abort() else 'no'}, "
            f"elapsed={self.loop_controller.state.elapsed_seconds:.2f}s"
        )

        if self._error_event:
            raise self._error_event

        if self.circuit_breaker.is_tripped:
            raise CopilotSdkLoopError(
                f"SDK loop limit exceeded: {self.circuit_breaker.trip_reason}",
                turn_count=self.loop_controller.state.turn_count,
                max_turns=self.loop_controller.max_turns,
                tool_calls_captured=len(self.tool_capture.captured_tools),
            )

    @property
    def captured_tools(self) -> list[CapturedToolCall]:
        """Get all captured tool calls."""
        return self.tool_capture.captured_tools

    @property
    def should_abort(self) -> bool:
        """Check if abort was requested."""
        return self.loop_controller.should_abort()

    @property
    def turn_count(self) -> int:
        """Get current turn count."""
        return self.loop_controller.state.turn_count
