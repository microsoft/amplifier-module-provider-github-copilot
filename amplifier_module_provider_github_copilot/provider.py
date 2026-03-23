"""Provider Orchestrator Module.

Thin orchestrator implementing Provider Protocol.
Delegates to specialized modules for all logic.

Contract: provider-protocol.md

MUST constraints:
- MUST implement Provider Protocol (4 methods + 1 property)
- MUST delegate tool parsing to tool_parsing module
- MUST delegate request adaptation to request_adapter module
- MUST delegate observability to observability module
- MUST NOT contain SDK imports (delegation only)
- MUST implement mount(), get_info(), list_models(), complete(), parse_tool_calls()

Three-Medium Architecture:
- Provider orchestrates control flow (Python = mechanism)
- Event names and policy values from config/ (YAML = policy)
- Contracts define requirements (Markdown = specification)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, cast

from amplifier_core import (
    ChatRequest,
    ChatResponse,
    ConfigField,
    ModelInfo,
    ModuleCoordinator,
    ProviderInfo,
    ToolCall,
)

# Re-exports for backward compatibility
from .completion import complete, complete_and_collect
from .config_loader import (
    ProviderConfig,
    RetryConfig,
    calculate_backoff_delay,
    get_retry_after,
    is_retryable_error,
    load_models_config,
    load_retry_config,
    load_sdk_protection_config,
    load_streaming_config,
)
from .error_translation import (
    LLMError,
    ProviderUnavailableError,
    load_error_config,
    translate_sdk_error,
)
from .fake_tool_detection import (
    load_fake_tool_detection_config,
    log_detection,
    log_exhausted,
    log_retry,
    log_success,
    should_retry_for_fake_tool_calls,
)

# Model discovery and cache (dynamic SDK fetch)
from .model_cache import read_cache, write_cache
from .models import (
    copilot_model_to_amplifier_model,
    fetch_and_map_models,
    fetch_models,
)

# Observability module for hook event emission (separation of concerns)
from .observability import (
    llm_lifecycle,
    load_observability_config,
)

# Request adapter for ChatRequest conversion (separation of concerns)
# Include private functions for backward compat (tests import from provider.py)
from .request_adapter import (
    _extract_content_block,  # pyright: ignore[reportPrivateUsage]
    _extract_message_content,  # pyright: ignore[reportPrivateUsage]
    convert_chat_request,
)
from .request_adapter import (
    extract_prompt_from_chat_request as _extract_prompt_from_chat_request,
)
from .sdk_adapter.client import CopilotClientWrapper
from .sdk_adapter.event_helpers import (
    extract_event_type,
    is_error_event,
    is_idle_event,
)
from .sdk_adapter.extract import extract_event_fields
from .sdk_adapter.tool_capture import ToolCaptureHandler
from .sdk_adapter.types import (
    CompletionConfig,
    CompletionRequest,
    SDKCreateFn,
    SDKSession,
    SessionConfig,
)
from .streaming import (
    MAX_EXTRACTION_DEPTH,
    AccumulatedResponse,
    DomainEvent,
    DomainEventType,
    EventConfig,
    StreamingAccumulator,
    extract_response_content,
    load_event_config,
    translate_event,
)
from .tool_parsing import parse_tool_calls

# Explicit exports for backward compatibility
__all__ = [
    # Provider class
    "GitHubCopilotProvider",
    # Completion module re-exports
    "complete",
    "complete_and_collect",
    "CompletionRequest",
    "CompletionConfig",
    "SDKCreateFn",
    # Config loader re-exports
    "ProviderConfig",
    "RetryConfig",
    "load_models_config",
    "load_retry_config",
    "calculate_backoff_delay",
    "is_retryable_error",
    "get_retry_after",
    # Streaming re-exports
    "AccumulatedResponse",
    "DomainEvent",
    "StreamingAccumulator",
    "extract_response_content",
    "MAX_EXTRACTION_DEPTH",
    # Error translation re-exports
    "LLMError",
    "ProviderUnavailableError",
    # SDK types re-exports
    "SDKSession",
    "SessionConfig",
    # Private aliases (backward compat)
    "_load_models_config",
    "_is_retryable_error",
    "_get_retry_after",
    # Request adapter re-exports (backward compat)
    "_extract_prompt_from_chat_request",
    "_extract_message_content",
    "_extract_content_block",
]

# Re-export private names for backward compatibility with tests
_load_models_config = load_models_config
_is_retryable_error = is_retryable_error
_get_retry_after = get_retry_after

logger = logging.getLogger(__name__)


class GitHubCopilotProvider:
    """Provider Protocol implementation for GitHub Copilot.

    Contract: provider-protocol.md

    This is a thin orchestrator that delegates to:
    - config_loader module for configuration
    - completion module for LLM calls
    - tool_parsing module for tool extraction

    Implements 4 methods + 1 property Provider Protocol:
    - name (property)
    - get_info()
    - list_models()
    - complete()
    - parse_tool_calls()
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        coordinator: ModuleCoordinator | None = None,
        *,
        client: CopilotClientWrapper | None = None,
    ) -> None:
        """Initialize provider.

        Args:
            config: Optional provider configuration.
            coordinator: Optional Amplifier kernel coordinator.
            client: Optional pre-created client for singleton injection.
                    If None, creates a new CopilotClientWrapper.

        """
        self.config = config or {}
        self.coordinator = coordinator
        self._client = client if client is not None else CopilotClientWrapper()
        self._provider_config = load_models_config()

    @property
    def name(self) -> str:
        """Return provider name.

        Contract: provider-protocol:name:MUST:1
        """
        return "github-copilot"

    def get_info(self) -> ProviderInfo:
        """Return provider metadata.

        Contract: provider-protocol:get_info:MUST:1
        Contract: provider-protocol:get_info:MUST:3 (config_fields)
        """
        cfg = self._provider_config
        return ProviderInfo(
            id=cfg.provider_id,
            display_name=cfg.display_name,
            credential_env_vars=cfg.credential_env_vars,
            capabilities=cfg.capabilities,
            defaults=cfg.defaults,
            config_fields=[
                ConfigField(
                    id="github_token",
                    display_name="GitHub Token",
                    field_type="secret",
                    prompt="Enter your GitHub token (or Copilot agent token)",
                    env_var="GITHUB_TOKEN",
                    required=True,
                ),
            ],
        )

    async def list_models(self) -> list[ModelInfo]:
        """Return available models from GitHub Copilot SDK.

        Contract: sdk-boundary:ModelDiscovery:MUST:1
        - MUST fetch models from SDK list_models() API

        Contract: behaviors:ModelCache:SHOULD:1
        - SHOULD cache SDK models to disk for session persistence

        Contract: behaviors:ModelDiscoveryError:MUST:1
        - MUST raise ProviderUnavailableError when SDK unavailable AND cache empty

        Two-Tier Architecture:
        1. SDK list_models() → Dynamic, authoritative
        2. Disk cache → Fallback when SDK unavailable
        3. ERROR → No hardcoded fallback (fail clearly)
        """
        # Tier 1: Try SDK first (dynamic, authoritative)
        try:
            models = await fetch_and_map_models(self._client)

            # Cache successful result for future use
            try:
                copilot_models = await fetch_models(self._client)
                write_cache(copilot_models)
            except Exception as cache_err:
                logger.warning("Failed to cache models: %s", cache_err)

            return cast(list[ModelInfo], models)
        except Exception as sdk_err:
            logger.warning("SDK list_models failed: %s", sdk_err)

        # Tier 2: Try disk cache (fallback)
        cached_models = read_cache()
        if cached_models:
            logger.info("Using cached models (%d models)", len(cached_models))
            return cast(
                list[ModelInfo],
                [copilot_model_to_amplifier_model(m) for m in cached_models],
            )

        # Tier 3: Error — no hardcoded fallback
        # Contract: behaviors:ModelDiscoveryError:MUST:1
        raise ProviderUnavailableError(
            "Failed to fetch models from SDK and no cached models available. "
            "Check network connectivity and SDK authentication.",
            provider="github-copilot",
        )

    async def complete(
        self,
        request: ChatRequest,
        **kwargs: Any,
    ) -> ChatResponse:
        """Execute completion lifecycle, returning ChatResponse.

        Contract: provider-protocol:complete:MUST:1

        Delegates to:
        - request_adapter module for request conversion
        - observability module for hook emission
        - completion module for SDK execution
        """
        # Convert request using request_adapter module (separation of concerns)
        internal_request = convert_chat_request(
            request,
            default_model=self._provider_config.defaults["model"],
        )

        # Use the SDK client wrapper for real SDK path
        accumulator = StreamingAccumulator()

        # Load configs
        event_config = load_event_config()
        obs_config = load_observability_config()

        # Three-Medium: model from YAML config (validated at load time)
        model = internal_request.model or self._provider_config.defaults["model"]

        # Create lifecycle context for observability (handles timing)
        async with llm_lifecycle(self.coordinator, model, obs_config) as ctx:
            # Check for test injection first
            sdk_create_fn = kwargs.get("sdk_create_fn")
            if sdk_create_fn is not None:
                # Test path: use injected SDK factory via completion module
                # No hooks emitted in test path (sdk_create_fn is for legacy tests)
                async for event in complete(
                    internal_request,
                    config=kwargs.get("config"),
                    sdk_create_fn=sdk_create_fn,
                ):
                    accumulator.add(event)
            else:
                # Real SDK path: use client wrapper with STREAMING
                # Three-Medium: timeout from YAML config
                timeout_seconds: float = kwargs.get(
                    "_timeout_seconds",
                    float(self._provider_config.defaults["timeout"]),
                )

                retry_config = load_retry_config()

                # Emit llm:request event (contract: observability:Events:MUST:2)
                use_streaming = self.config.get("use_streaming", True)
                await ctx.emit_request(
                    message_count=len(getattr(request, "messages", [])),
                    tool_count=len(internal_request.tools) if internal_request.tools else 0,
                    streaming=use_streaming,
                    timeout=timeout_seconds,
                )

                for attempt in range(retry_config.max_attempts):
                    try:
                        await self._execute_sdk_completion(
                            client=self._client,
                            model=model,
                            prompt=internal_request.prompt,
                            timeout=timeout_seconds,
                            event_config=event_config,
                            accumulator=accumulator,
                            tools=internal_request.tools or None,
                            attachments=internal_request.attachments or None,
                        )
                        break  # Success

                    except LLMError as e:
                        if not is_retryable_error(e):
                            await ctx.emit_response_error(
                                error_type=type(e).__name__,
                                error_message=str(e),
                            )
                            raise

                        if attempt < retry_config.max_attempts - 1:
                            delay_ms = self._calculate_retry_delay(e, attempt, retry_config)
                            logger.info(
                                "[RETRY] Attempt %d/%d failed: %s. Retrying in %.0fms",
                                attempt + 1,
                                retry_config.max_attempts,
                                e,
                                delay_ms,
                            )
                            await ctx.emit_retry(
                                attempt=attempt + 1,
                                max_retries=retry_config.max_attempts,
                                delay=delay_ms / 1000,
                                error_type=type(e).__name__,
                                error_message=str(e),
                            )
                            await asyncio.sleep(delay_ms / 1000)
                        else:
                            await ctx.emit_response_error(
                                error_type=type(e).__name__,
                                error_message=str(e),
                            )
                            raise

                    except Exception as e:
                        error_config_for_err = load_error_config()
                        translated = translate_sdk_error(
                            e, error_config_for_err, provider="github-copilot", model=model
                        )

                        if not is_retryable_error(translated):
                            await ctx.emit_response_error(
                                error_type=type(translated).__name__,
                                error_message=str(translated),
                            )
                            raise translated from e

                        if attempt < retry_config.max_attempts - 1:
                            delay_ms = self._calculate_retry_delay(
                                translated, attempt, retry_config
                            )
                            logger.info(
                                "[RETRY] Attempt %d/%d failed: %s. Retrying in %.0fms",
                                attempt + 1,
                                retry_config.max_attempts,
                                translated,
                                delay_ms,
                            )
                            await ctx.emit_retry(
                                attempt=attempt + 1,
                                max_retries=retry_config.max_attempts,
                                delay=delay_ms / 1000,
                                error_type=type(translated).__name__,
                                error_message=str(translated),
                            )
                            await asyncio.sleep(delay_ms / 1000)
                        else:
                            await ctx.emit_response_error(
                                error_type=type(translated).__name__,
                                error_message=str(translated),
                            )
                            raise translated from e

            # Fake tool call detection and retry
            ftd_config = load_fake_tool_detection_config()
            tools_available = bool(internal_request.tools)

            for correction_attempt in range(ftd_config.max_correction_attempts):
                should_retry, matched_pattern = should_retry_for_fake_tool_calls(
                    response_text=accumulator.text_content,
                    tool_calls=accumulator.tool_calls,
                    tools_available=tools_available,
                    config=ftd_config,
                )

                if not should_retry:
                    if correction_attempt > 0:
                        log_success(ftd_config, correction_attempt - 1)
                    break

                log_detection(
                    ftd_config,
                    accumulator.text_content,
                    matched_pattern,
                    accumulator.tool_calls,
                )
                log_retry(ftd_config, correction_attempt, ftd_config.max_correction_attempts)

                corrected_prompt = (
                    internal_request.prompt + "\n\n[User]: " + ftd_config.correction_message
                )
                accumulator = StreamingAccumulator()

                if sdk_create_fn is not None:
                    raw_tools = internal_request.tools or []
                    # Convert tools to dict format for CompletionRequest
                    tools_for_retry: list[dict[str, Any]] = [
                        dict(t)
                        if hasattr(t, "keys")
                        else (t.__dict__.copy() if hasattr(t, "__dict__") else {"name": str(t)})
                        for t in raw_tools
                    ]
                    async for event in complete(
                        CompletionRequest(
                            prompt=corrected_prompt,
                            model=internal_request.model,
                            tools=tools_for_retry,
                        ),
                        config=kwargs.get("config"),
                        sdk_create_fn=sdk_create_fn,
                    ):
                        accumulator.add(event)
                else:
                    # Three-Medium: timeout from YAML config
                    timeout_seconds_retry: float = kwargs.get(
                        "_timeout_seconds",
                        float(self._provider_config.defaults["timeout"]),
                    )
                    try:
                        # Note: attachments=None for correction - the model already saw the image
                        await self._execute_sdk_completion(
                            client=self._client,
                            model=model,
                            prompt=corrected_prompt,
                            timeout=timeout_seconds_retry,
                            event_config=event_config,
                            accumulator=accumulator,
                            tools=internal_request.tools or None,
                            attachments=None,
                        )
                    except Exception:
                        log_exhausted(ftd_config, correction_attempt + 1)
                        break
            else:
                log_exhausted(ftd_config, ftd_config.max_correction_attempts)

            # Build response and emit success event
            response = accumulator.to_chat_response()
            response_tool_calls = len(response.tool_calls) if response.tool_calls else 0

            await ctx.emit_response_ok(
                usage_input=response.usage.input_tokens if response.usage else 0,
                usage_output=response.usage.output_tokens if response.usage else 0,
                finish_reason=response.finish_reason,
                content_blocks=len(response.content) if response.content else 0,
                tool_calls=response_tool_calls,
            )

        return response

    def _calculate_retry_delay(
        self,
        error: Exception,
        attempt: int,
        config: RetryConfig,
    ) -> float:
        """Calculate retry delay in milliseconds.

        Honors retry_after header if present, otherwise uses exponential backoff.
        """
        retry_after = get_retry_after(error)
        if retry_after is not None:
            return retry_after * 1000
        return calculate_backoff_delay(
            attempt=attempt,
            base_delay_ms=config.base_delay_ms,
            max_delay_ms=config.max_delay_ms,
            jitter_factor=config.jitter_factor,
        )

    async def _execute_sdk_completion(
        self,
        client: CopilotClientWrapper,
        model: str,
        prompt: str,
        timeout: float,
        event_config: EventConfig,
        accumulator: StreamingAccumulator,
        tools: list[Any] | None = None,
        attachments: list[dict[str, Any]] | None = None,
    ) -> None:
        """Execute a single SDK completion, draining events to accumulator.

        This is the core SDK interaction pattern extracted for reuse.
        Callers are responsible for error handling (retry vs break).

        Contract: provider-protocol:complete:MUST:1
        Contract: provider-protocol:complete:MUST:2 (tool forwarding)
        Contract: provider-protocol:complete:MUST:8 (attachment forwarding)
        Contract: sdk-protection:ToolCapture:MUST:1,2 (first_turn_only, deduplicate)
        Contract: sdk-protection:Session:MUST:3,4 (explicit_abort, abort_timeout)
        Contract: behaviors:Streaming:MUST:1 (TTFT warning)
        """
        # Load SDK protection config for tool capture and session management
        sdk_protection = load_sdk_protection_config()
        # Load streaming config for TTFT warning
        # Contract: behaviors:Streaming:MUST:1
        streaming_config = load_streaming_config()

        async with asyncio.timeout(timeout):
            async with client.session(model=model, tools=tools) as sdk_session:
                event_queue: asyncio.Queue[Any] = asyncio.Queue()
                idle_event = asyncio.Event()
                error_holder: list[Exception] = []
                # TTFT tracking state (mutable container for closure)
                # Contract: behaviors:Streaming:MUST:1
                ttft_state: dict[str, Any] = {"checked": False, "start_time": 0.0}
                # Use extracted ToolCaptureHandler for tool capture
                # Contract: sdk-protection:ToolCapture:MUST:1,2
                tool_capture_handler = ToolCaptureHandler(
                    on_capture_complete=idle_event.set,
                    logger_prefix="[provider]",
                    config=sdk_protection.tool_capture,
                )

                def event_handler(
                    sdk_event: Any,
                    *,
                    queue: asyncio.Queue[Any] = event_queue,
                    idle: asyncio.Event = idle_event,
                    errors: list[Exception] = error_holder,
                    capture_handler: ToolCaptureHandler = tool_capture_handler,
                    ttft: dict[str, Any] = ttft_state,
                    ttft_threshold_ms: int = streaming_config.ttft_warning_ms,
                ) -> None:
                    """Push SDK events to queue for async processing.

                    Tool capture delegated to ToolCaptureHandler.
                    B023: Loop variables bound via default args.
                    Contract: streaming-contract:abort-on-capture:MUST:1
                    Contract: behaviors:Streaming:MUST:1 (TTFT warning)
                    """
                    try:
                        queue.put_nowait(sdk_event)
                        event_type = extract_event_type(sdk_event)

                        # TTFT check on first content event
                        # Contract: behaviors:Streaming:MUST:1
                        if not ttft["checked"]:
                            content_types = {
                                "assistant.message_delta",
                                "assistant.streaming_delta",
                                "assistant.reasoning_delta",
                            }
                            if event_type in content_types:
                                ttft["checked"] = True
                                elapsed_ms = (time.time() - ttft["start_time"]) * 1000
                                if elapsed_ms > ttft_threshold_ms:
                                    logger.warning(
                                        "[TTFT] Slow time to first token: %.0fms (threshold: %dms)",
                                        elapsed_ms,
                                        ttft_threshold_ms,
                                    )
                        if is_idle_event(event_type):
                            idle.set()
                        elif is_error_event(event_type):
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
                            errors.append(err)
                            idle.set()
                        else:
                            # Delegate tool capture to handler
                            capture_handler.on_event(sdk_event)
                    except Exception as e:
                        logger.warning("Error in event handler: %s", e)

                unsubscribe = sdk_session.on(event_handler)
                try:
                    # Record TTFT start time before send
                    # Contract: behaviors:Streaming:MUST:1
                    ttft_state["start_time"] = time.time()
                    # SDK v0.2.0: send(prompt, attachments=...) replaces send({"prompt": ...})
                    # Contract: sdk-boundary:ImagePassthrough:MUST:7
                    await sdk_session.send(prompt, attachments=attachments)
                    await asyncio.wait_for(idle_event.wait(), timeout=timeout)

                    if error_holder:
                        raise error_holder[0]

                    # Add captured tools to accumulator FIRST, before draining event_queue
                    # CRITICAL: Must happen BEFORE TURN_COMPLETE sets is_complete=True,
                    # otherwise accumulator.add() will silently drop our tool_calls
                    if tool_capture_handler.captured_tools:
                        for tool in tool_capture_handler.captured_tools:
                            accumulator.add(
                                DomainEvent(
                                    type=DomainEventType.TOOL_CALL,
                                    data=tool,
                                )
                            )
                        logger.debug(
                            "[provider] Added %d captured tools to accumulator",
                            len(tool_capture_handler.captured_tools),
                        )

                        # Explicit abort after tool capture
                        # Contract: sdk-protection:Session:MUST:3,4
                        if sdk_protection.session.explicit_abort:
                            try:
                                await asyncio.wait_for(
                                    sdk_session.abort(),
                                    timeout=sdk_protection.session.abort_timeout_seconds,
                                )
                                logger.debug("[provider] Session aborted after tool capture")
                            except TimeoutError:
                                logger.warning(
                                    "[provider] Session abort timed out after %.1fs",
                                    sdk_protection.session.abort_timeout_seconds,
                                )
                            except Exception as e:
                                # Abort failure is non-critical - log and continue
                                logger.debug(
                                    "[provider] Session abort failed (non-critical): %s", e
                                )

                    # Now drain remaining events (including TURN_COMPLETE)
                    while not event_queue.empty():
                        sdk_event = event_queue.get_nowait()
                        event_dict: dict[str, Any]
                        if isinstance(sdk_event, dict):
                            event_dict = cast(dict[str, Any], sdk_event)
                        else:
                            event_dict = extract_event_fields(sdk_event)

                        domain_event = translate_event(event_dict, event_config)
                        if domain_event is not None:
                            accumulator.add(domain_event)
                finally:
                    unsubscribe()

    async def close(self) -> None:
        """Clean up provider resources.

        Contract: provider-protocol:close:MUST:1 — must clean up SDK resources
        Contract: sdk-boundary.md — provider must clean up SDK resources on close

        Delegates to client.close() for SDK resource cleanup.
        Safe to call multiple times (idempotent).
        """
        if hasattr(self, "_client") and self._client:
            await self._client.close()

    def parse_tool_calls(self, response: Any) -> list[ToolCall]:
        """Extract tool calls from response.

        Contract: provider-protocol:parse_tool_calls:MUST:1 through MUST:4

        Delegates to tool_parsing module.
        """
        return parse_tool_calls(response)
