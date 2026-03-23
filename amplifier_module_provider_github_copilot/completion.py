"""Completion lifecycle orchestration.

Contract: streaming-contract.md, deny-destroy.md

MUST constraints:
- MUST use ephemeral sessions per deny-destroy.md
- MUST yield DomainEvents for streaming
- MUST translate SDK errors to kernel errors
- MUST clean up sessions in finally block
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, cast

from .config_loader import (
    load_models_config,
    load_sdk_protection_config,
)
from .error_translation import (
    ErrorConfig,
    LLMError,
    ProviderUnavailableError,
    load_error_config,
    translate_sdk_error,
)
from .sdk_adapter.event_helpers import (
    extract_event_type,
    is_idle_event,
)
from .sdk_adapter.extract import extract_event_fields
from .sdk_adapter.tool_capture import ToolCaptureHandler
from .sdk_adapter.types import CompletionConfig, SDKSession, SessionConfig
from .streaming import (
    AccumulatedResponse,
    DomainEvent,
    DomainEventType,
    EventConfig,
    StreamingAccumulator,
    load_event_config,
    translate_event,
)

logger = logging.getLogger(__name__)

__all__ = [
    "complete",
    "complete_and_collect",
]


# Type alias for SDK session creation function
SDKCreateFnLocal = Callable[[SessionConfig], Awaitable[SDKSession]]


async def complete(
    request: Any,
    *,
    config: CompletionConfig | None = None,
    sdk_create_fn: SDKCreateFnLocal | None = None,
) -> AsyncIterator[DomainEvent]:
    """Execute completion lifecycle, yielding domain events.

    Contract: streaming-contract.md, deny-destroy.md

    - MUST create ephemeral session with deny hook
    - MUST yield translated domain events
    - MUST destroy session in finally block
    - MUST translate SDK errors to kernel types

    Args:
        request: Completion request with prompt and options.
        config: Optional configuration overrides.
        sdk_create_fn: Optional SDK session factory (for testing).

    Yields:
        DomainEvent for each bridged SDK event.

    Raises:
        LLMError: Translated from SDK errors.

    """
    if config is None:
        config = CompletionConfig()

    # Load configs if not provided
    event_config: EventConfig | None = getattr(config, "event_config", None)
    if event_config is None:
        event_config = load_event_config()

    error_config: ErrorConfig | None = getattr(config, "error_config", None)
    if error_config is None:
        error_config = load_error_config()

    # Get model from request
    request_model = getattr(request, "model", None)
    request_prompt = getattr(request, "prompt", "")
    # Contract: sdk-boundary:ImagePassthrough:MUST:7
    request_attachments: list[dict[str, Any]] | None = getattr(request, "attachments", None)

    # Create session config
    session_config_attr = getattr(config, "session_config", None)
    # Three-Medium: default model comes from YAML config
    models_config = load_models_config()
    default_model = models_config.defaults["model"]
    session_config = session_config_attr or SessionConfig(model=request_model or default_model)

    # Create session
    session: SDKSession | None = None
    try:
        if sdk_create_fn is not None:
            session = await sdk_create_fn(session_config)
            # Use proper error handling (asserts are stripped by -O flag)
            if session is None:
                raise ProviderUnavailableError(
                    "SDK session factory returned None",
                    provider="github-copilot",
                )

            # The SDK uses send() + on() pattern for streaming.
            # The deny hook is passed via session config at creation time.
            if not hasattr(session, "on") or not hasattr(session, "send"):
                raise ProviderUnavailableError(
                    "SDK session lacks on() or send() methods - "
                    "correct SDK API requires these for streaming.",
                    provider="github-copilot",
                )

            # Use send() + on() pattern for streaming
            event_queue: asyncio.Queue[Any] = asyncio.Queue()
            idle_event = asyncio.Event()

            # Load SDK protection config for tool capture and session management
            # Contract: sdk-protection:ToolCapture:MUST:1,2
            # Contract: sdk-protection:Session:MUST:3,4
            sdk_protection = load_sdk_protection_config()

            # Use extracted ToolCaptureHandler for tool capture
            tool_capture_handler = ToolCaptureHandler(
                on_capture_complete=idle_event.set,
                logger_prefix="[completion]",
                config=sdk_protection.tool_capture,
            )

            def event_handler(sdk_event: Any) -> None:
                """Push SDK events to queue for async processing.

                Tool capture delegated to ToolCaptureHandler.
                Contract: streaming-contract:abort-on-capture:MUST:1
                """
                event_queue.put_nowait(sdk_event)
                # Use helper to extract type from dict or object events
                event_type = extract_event_type(sdk_event)
                if is_idle_event(event_type):
                    idle_event.set()
                else:
                    # Delegate tool capture to handler
                    tool_capture_handler.on_event(sdk_event)

            # Register event handler
            unsubscribe = session.on(event_handler)

            try:
                # SDK v0.2.0: send(prompt, attachments=...) replaces send({"prompt": ...})
                # Contract: sdk-boundary:ImagePassthrough:MUST:7
                await session.send(request_prompt, attachments=request_attachments)

                # Wait for session to become idle (with timeout)
                # Three-Medium: timeout comes from YAML config
                timeout = float(models_config.defaults["timeout"])
                try:
                    await asyncio.wait_for(idle_event.wait(), timeout=timeout)
                except TimeoutError:
                    pass  # Process whatever events we have

                # Yield captured tools FIRST, before draining event_queue
                # CRITICAL: Must happen BEFORE TURN_COMPLETE to avoid race condition
                # where accumulator.add() would reject events after is_complete=True
                if tool_capture_handler.captured_tools:
                    for tool in tool_capture_handler.captured_tools:
                        yield DomainEvent(
                            type=DomainEventType.TOOL_CALL,
                            data=tool,
                        )
                    logger.debug(
                        "[completion] Yielded %d captured tools",
                        len(tool_capture_handler.captured_tools),
                    )

                    # Explicit abort after tool capture
                    # Contract: sdk-protection:Session:MUST:3,4
                    if sdk_protection.session.explicit_abort:
                        try:
                            await asyncio.wait_for(
                                session.abort(),
                                timeout=sdk_protection.session.abort_timeout_seconds,
                            )
                            logger.debug("[completion] Session aborted after tool capture")
                        except TimeoutError:
                            logger.warning(
                                "[completion] Session abort timed out after %.1fs",
                                sdk_protection.session.abort_timeout_seconds,
                            )
                        except Exception as e:
                            # Abort failure is non-critical - log and continue
                            logger.debug("[completion] Session abort failed (non-critical): %s", e)

                # Now drain remaining events (including TURN_COMPLETE)
                while not event_queue.empty():
                    sdk_event = event_queue.get_nowait()
                    # Convert SDK event to dict for translate_event
                    event_dict: dict[str, Any]
                    if isinstance(sdk_event, dict):
                        event_dict = cast(dict[str, Any], sdk_event)
                    else:
                        # Use unified extraction (sdk_adapter/extract.py)
                        event_dict = extract_event_fields(sdk_event)
                    domain_event = translate_event(event_dict, event_config)
                    if domain_event is not None:
                        yield domain_event
            finally:
                unsubscribe()

        else:
            raise ProviderUnavailableError(
                "Real SDK path requires CopilotClientWrapper.session() context manager.",
                provider="github-copilot",
            )

    except Exception as e:
        # Don't double-wrap already-translated LLMError
        if isinstance(e, LLMError):
            raise  # Already translated, don't wrap again
        kernel_error = translate_sdk_error(
            e,
            error_config,
            provider="github-copilot",
            model=request_model,
        )
        raise kernel_error from e

    finally:
        if session is not None:
            try:
                if hasattr(session, "disconnect"):
                    await session.disconnect()
            except Exception as disconnect_err:
                logger.warning("Error destroying session: %s", disconnect_err)


async def complete_and_collect(
    request: Any,
    *,
    config: CompletionConfig | None = None,
    sdk_create_fn: SDKCreateFnLocal | None = None,
) -> AccumulatedResponse:
    """Execute completion lifecycle and collect final response.

    Convenience wrapper that accumulates all events into AccumulatedResponse.

    Args:
        request: Completion request with prompt and options.
        config: Optional configuration overrides.
        sdk_create_fn: Optional SDK session factory (for testing).

    Returns:
        AccumulatedResponse with text, tool calls, usage, etc.

    Raises:
        LLMError: Translated from SDK errors.

    """
    accumulator = StreamingAccumulator()

    async for event in complete(
        request,
        config=config,
        sdk_create_fn=sdk_create_fn,
    ):
        accumulator.add(event)

    return accumulator.get_result()
