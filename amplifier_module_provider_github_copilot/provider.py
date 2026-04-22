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
    AbortError,
    LLMError,
    ProviderUnavailableError,
    load_error_config,
    translate_sdk_error,
)

# Event routing (moved from inline import per W-02 code review)
from .event_router import EventRouter
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
)

# Observability module for hook event emission (separation of concerns)
from .observability import (
    llm_lifecycle,
)

# Request adapter for ChatRequest conversion (separation of concerns)
# Include private functions for backward compat (tests import from provider.py)
from .request_adapter import (
    _extract_content_block,  # pyright: ignore[reportPrivateUsage]
    _extract_message_content,  # pyright: ignore[reportPrivateUsage]
    build_request_payload_for_observability,
    build_response_payload_for_observability,
    convert_chat_request,
)
from .request_adapter import (
    extract_prompt_from_chat_request as _extract_prompt_from_chat_request,
)

# Contract: sdk-boundary:Membrane:MUST:1 — import from sdk_adapter package, not submodules
from .sdk_adapter import (
    CompletionConfig,
    CompletionRequest,
    CopilotClientWrapper,
    SDKCreateFn,
    SDKSession,
    SessionConfig,
    ToolCaptureHandler,
    extract_event_fields,
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
    # SDK types
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


def _parse_raw_flag(value: Any) -> bool:
    """Strictly parse the raw config flag from provider config dict.

    Handles string inputs so that config={"raw": "false"} is not accidentally
    treated as True (bool("false") == True is a Python footgun).

    Args:
        value: Raw value from provider config dict.

    Returns:
        True only for bool True, string "true"/"1"/"yes", or other truthy non-string.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    return bool(value)


def _config_int(value: Any, default: int) -> int:
    """Safely coerce a config value to int, falling back to default on error.

    None is treated as "not provided" and returns the default silently.
    Any other unparseable value logs a warning and returns the default.

    Args:
        value: Raw value from provider config dict.
        default: Fallback value if parsing fails.

    Returns:
        Parsed int, or default if value is None or unparseable.
    """
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        logger.warning(
            "[PROVIDER] Invalid integer config value %r; using default %s",
            value,
            default,
        )
        return default


def _config_float(value: Any, default: float) -> float:
    """Safely coerce a config value to float, falling back to default on error.

    None is treated as "not provided" and returns the default silently.
    Any other unparseable value logs a warning and returns the default.

    Args:
        value: Raw value from provider config dict.
        default: Fallback value if parsing fails.

    Returns:
        Parsed float, or default if value is None or unparseable.
    """
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning(
            "[PROVIDER] Invalid float config value %r; using default %s",
            value,
            default,
        )
        return default


def _build_retry_config(config: dict[str, Any], defaults: RetryConfig) -> RetryConfig:
    """Build per-instance RetryConfig from provider config dict.

    User-facing keys
    - max_retries:     number of retries (0 = no retry). Stored as max_attempts = retries + 1.
    - min_retry_delay: minimum delay in seconds. Stored internally as base_delay_ms.
    - max_retry_delay: maximum delay in seconds. Stored internally as max_delay_ms.
    - retry_jitter:    jitter factor as float [0.0, 1.0].

    When a key is absent the corresponding field from defaults is used unchanged
    (no unit conversion on the default path avoids float round-trip arithmetic).

    Contract: behaviors:Retry:MUST:7

    Args:
        config: Provider config dict from mount/routing.
        defaults: Frozen RetryConfig with policy defaults from _policy.py.

    Returns:
        New frozen RetryConfig with per-instance overrides applied.
    """
    # max_retries (retries) → max_attempts (total attempts) = retries + 1.
    # max_retries=0 is valid: single attempt, no retry. Clamp negative to 0.
    raw_max_retries = config.get("max_retries")
    if raw_max_retries is not None:
        retries = max(_config_int(raw_max_retries, defaults.max_attempts - 1), 0)
        max_attempts = retries + 1
    else:
        max_attempts = defaults.max_attempts

    # Delay keys: user provides seconds, internal storage is milliseconds.
    # Only convert when the key is present to avoid float round-trip on default path.
    raw_min_delay = config.get("min_retry_delay")
    base_delay_ms = (
        int(_config_float(raw_min_delay, defaults.base_delay_ms / 1000.0) * 1000)
        if raw_min_delay is not None
        else defaults.base_delay_ms
    )

    raw_max_delay = config.get("max_retry_delay")
    max_delay_ms = (
        int(_config_float(raw_max_delay, defaults.max_delay_ms / 1000.0) * 1000)
        if raw_max_delay is not None
        else defaults.max_delay_ms
    )

    # Jitter: float [0.0, 1.0]. calculate_backoff_delay already clamps, no guard needed.
    jitter_factor = _config_float(config.get("retry_jitter"), defaults.jitter_factor)

    # Overloaded error multiplier: scales backoff for rate-limited / overloaded errors.
    # RetryPolicy.__post_init__ enforces >= 1.0; catch ValueError and use minimum-safe
    # value (1.0) so invalid user config degrades gracefully without retry storms.
    raw_odm = config.get("overloaded_delay_multiplier")
    overloaded_delay_multiplier = _config_float(raw_odm, defaults.overloaded_delay_multiplier)

    try:
        return RetryConfig(
            max_attempts=max_attempts,
            base_delay_ms=base_delay_ms,
            max_delay_ms=max_delay_ms,
            jitter_factor=jitter_factor,
            overloaded_delay_multiplier=overloaded_delay_multiplier,
        )
    except ValueError as exc:
        # overloaded_delay_multiplier < 1.0 — clamp to minimum-safe (1.0), NOT the
        # policy default (10.0), to avoid unexpected 10× backoff for bad configs.
        logger.warning("Invalid retry config (%s); using 1.0 for overloaded_delay_multiplier", exc)
        return RetryConfig(
            max_attempts=max_attempts,
            base_delay_ms=base_delay_ms,
            max_delay_ms=max_delay_ms,
            jitter_factor=jitter_factor,
            overloaded_delay_multiplier=1.0,
        )


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
        # Parse raw flag once at init — avoids bool("false")==True footgun at call time
        self._raw: bool = _parse_raw_flag(self.config.get("raw", False))
        # Parse retry config once at init — allows per-instance user overrides
        self._retry_config: RetryConfig = _build_retry_config(self.config, load_retry_config())
        # Track pending streaming emit tasks for cleanup
        # Contract: streaming-contract:ProgressiveStreaming:SHOULD:3
        self._pending_emit_tasks: set[asyncio.Task[Any]] = set()

    @property
    def _effective_default_model(self) -> str:
        """Get effective default model respecting runtime config.

        Priority:
        1. self.config["default_model"] — runtime config from mount/routing matrix
        2. self._provider_config.defaults["model"] — YAML config

        Contract: Three-Medium Architecture — runtime config overrides YAML.
        Note: Does NOT mutate cached ProviderConfig (avoids race conditions
        when multiple sub-agents mount with different configs).
        """
        return self.config.get("default_model") or self._provider_config.defaults["model"]

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
            # Bug fix: fetch_and_map_models returns both mapped models AND raw SDK models
            # for caching, eliminating the redundant fetch_models() call
            models, copilot_models = await fetch_and_map_models(self._client)

            # Cache successful result for future use
            try:
                write_cache(copilot_models)
            except Exception as cache_err:
                from .security_redaction import redact_sensitive_text

                logger.warning("Failed to cache models: %s", redact_sensitive_text(cache_err))

            return cast(list[ModelInfo], models)
        except Exception as sdk_err:
            from .security_redaction import redact_sensitive_text

            logger.warning("SDK list_models failed: %s", redact_sensitive_text(sdk_err))

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
            default_model=self._effective_default_model,
        )

        # Load configs
        event_config = load_event_config()

        # Effective model: request.model > runtime config > YAML default
        model = internal_request.model or self._effective_default_model

        # Create lifecycle context for observability (handles timing)
        # raw=self._raw passes per-instance flag parsed once in __init__
        async with llm_lifecycle(self.coordinator, model, raw=self._raw) as ctx:
            # Real SDK path: use client wrapper with STREAMING
            # Three-Medium: timeout from YAML config
            timeout_seconds: float = kwargs.get(
                "_timeout_seconds",
                float(self._provider_config.defaults["timeout"]),
            )

            retry_config = self._retry_config

            # Emit llm:request event (contract: observability:Events:MUST:2)
            use_streaming = self.config.get("use_streaming", True)
            await ctx.emit_request(
                message_count=len(getattr(request, "messages", [])),
                tool_count=len(internal_request.tools) if internal_request.tools else 0,
                streaming=use_streaming,
                timeout=timeout_seconds,
                raw_request=build_request_payload_for_observability(
                    model=model,
                    request=request,
                    internal_request=internal_request,
                ),
            )

            # Initialize accumulator before loop (for type checker)
            # Reset at start of each iteration to prevent content corruption on retry
            accumulator = StreamingAccumulator()

            for attempt in range(retry_config.max_attempts):
                # Bug fix: Reset accumulator for each attempt to prevent corruption
                # If first attempt partially streams then fails, retry must start fresh
                accumulator = StreamingAccumulator()
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
                        system_message=internal_request.system_message,
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
                            retry_after=get_retry_after(e),
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

                except asyncio.CancelledError:
                    # C-2: asyncio.CancelledError is BaseException (not Exception).
                    # Bare `except Exception` misses it.  Translate to AbortError
                    # so the kernel receives a typed, non-retryable kernel error.
                    # Contract: error-hierarchy:AbortError:MUST:1
                    abort = AbortError("Request cancelled", provider="github-copilot")
                    await ctx.emit_response_error(
                        error_type=type(abort).__name__,
                        error_message=str(abort),
                    )
                    raise abort from None

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
                        delay_ms = self._calculate_retry_delay(translated, attempt, retry_config)
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
                            retry_after=get_retry_after(translated),
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
                        system_message=internal_request.system_message,
                    )
                except asyncio.CancelledError:
                    # C-2: Same guard for the fake-tool correction path.
                    # Contract: error-hierarchy:AbortError:MUST:1
                    abort = AbortError("Request cancelled", provider="github-copilot")
                    log_exhausted(ftd_config, correction_attempt + 1)
                    await ctx.emit_response_error(
                        error_type=type(abort).__name__,
                        error_message=str(abort),
                    )
                    raise abort from None

                except Exception as e:
                    # P1 Fix: Don't silently swallow exception - propagate to caller.
                    # Breaking here would return empty accumulator (silent data loss).
                    # Also emit error event to satisfy observability contract
                    # (llm:response MUST be emitted).
                    #
                    # Contract: error-hierarchy.md — MUST translate SDK errors to kernel errors.
                    # The correction path must use the same error translation as the main path.
                    error_config_for_correction = load_error_config()
                    translated = translate_sdk_error(
                        e, error_config_for_correction, provider="github-copilot", model=model
                    )
                    log_exhausted(ftd_config, correction_attempt + 1)
                    await ctx.emit_response_error(
                        error_type=type(translated).__name__,
                        error_message=str(translated),
                    )
                    raise translated from e
            else:
                log_exhausted(ftd_config, ftd_config.max_correction_attempts)

            # Build response and emit success event
            response = accumulator.to_chat_response()
            response_tool_calls = len(response.tool_calls) if response.tool_calls else 0

            # DEBUG: Log response details before returning to orchestrator
            logger.debug(
                "[COMPLETE] Returning response: finish_reason=%s, tool_calls=%d, "
                "content=%d, text_len=%d",
                response.finish_reason,
                response_tool_calls,
                len(response.content) if response.content else 0,
                len(response.text) if response.text else 0,
            )

            await ctx.emit_response_ok(
                usage_input=response.usage.input_tokens if response.usage else 0,
                usage_output=response.usage.output_tokens if response.usage else 0,
                usage_cache_read=response.usage.cache_read_tokens if response.usage else None,
                usage_cache_write=response.usage.cache_write_tokens if response.usage else None,
                finish_reason=response.finish_reason,
                content_blocks=len(response.content) if response.content else 0,
                tool_calls=response_tool_calls,
                sdk_session_id=accumulator.sdk_session_id,
                sdk_pid=accumulator.sdk_pid,
                raw_response=build_response_payload_for_observability(
                    response=response,
                    tool_calls=response_tool_calls,
                ),
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
        Applies overloaded_delay_multiplier for errors that carry delay_multiplier > 1.0
        (set by error translation for overloaded/rate-limited error types per errors.yaml).
        The multiplied delay is capped at max_delay_ms * overloaded_delay_multiplier.

        Contract: behaviors:Retry:MUST:8
        """
        retry_after = get_retry_after(error)
        if retry_after is not None:
            # Server-provided Retry-After takes precedence over all computed delays.
            return retry_after * 1000
        delay_ms = calculate_backoff_delay(
            attempt=attempt,
            base_delay_ms=config.base_delay_ms,
            max_delay_ms=config.max_delay_ms,
            jitter_factor=config.jitter_factor,
        )
        # Apply policy multiplier when error is marked overloaded (delay_multiplier > 1.0).
        # Sentinel set post-construction by translate_sdk_error for mappings with overloaded=True.
        # Cap at max_delay_ms * multiplier: base is already capped by calculate_backoff_delay,
        # so this is equivalent to delay_ms * multiplier with an explicit upper bound.
        if getattr(error, "delay_multiplier", 1.0) > 1.0:
            delay_ms = min(
                delay_ms * config.overloaded_delay_multiplier,
                config.max_delay_ms * config.overloaded_delay_multiplier,
            )
        return delay_ms

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
        system_message: str | None = None,
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
        # DEBUG: Log entry point with key parameters
        logger.debug(
            "[SDK_COMPLETION] Starting: model=%s, prompt_len=%d, timeout=%.1f, "
            "tools=%d, attachments=%d, idle_events=%s, system_message_len=%d",
            model,
            len(prompt),
            timeout,
            len(tools) if tools else 0,
            len(attachments) if attachments else 0,
            event_config.idle_event_types,
            len(system_message) if system_message else 0,
        )
        # Load SDK protection config for tool capture and session management
        sdk_protection = load_sdk_protection_config()
        # Load streaming config for TTFT warning and bounded queue
        # Contract: behaviors:Streaming:MUST:1, MUST:4
        streaming_config = load_streaming_config()

        async with asyncio.timeout(timeout):
            async with client.session(
                model=model, tools=tools, system_message=system_message
            ) as sdk_session:
                # Capture SDK session ID for observability correlation
                accumulator.sdk_session_id = sdk_session.session_id
                # Capture SDK subprocess PID for log file correlation
                # Contract: observability:Events:SHOULD:3
                accumulator.sdk_pid = client.copilot_pid

                # Contract: behaviors:Streaming:MUST:4 — bounded queue, drop on full
                event_queue: asyncio.Queue[Any] = asyncio.Queue(
                    maxsize=streaming_config.event_queue_size
                )
                idle_event = asyncio.Event()
                error_holder: list[Exception] = []
                # TTFT tracking state (mutable container for closure)
                # Contract: behaviors:Streaming:MUST:1
                ttft_state: dict[str, Any] = {"checked": False, "start_time": 0.0}
                # Usage holder: captures usage directly to avoid race condition
                # SDK may send assistant.usage AFTER session.idle
                # Contract: streaming-contract:usage:MUST:1
                usage_holder: list[dict[str, int | None]] = []
                # Use extracted ToolCaptureHandler for tool capture
                # Contract: sdk-protection:ToolCapture:MUST:1,2
                tool_capture_handler = ToolCaptureHandler(
                    on_capture_complete=idle_event.set,
                    logger_prefix="[provider]",
                    config=sdk_protection.tool_capture,
                )

                # Create EventRouter for SDK event handling
                # Extracted from inline closure per Comprehensive Review P1.6
                # Contract: streaming-contract:abort-on-capture:MUST:1
                # Contract: behaviors:Streaming:MUST:1,4
                event_handler = EventRouter(
                    queue=event_queue,
                    idle_event=idle_event,
                    error_holder=error_holder,
                    usage_holder=usage_holder,
                    capture_handler=tool_capture_handler,
                    ttft_state=ttft_state,
                    ttft_threshold_ms=streaming_config.ttft_warning_ms,
                    event_config=event_config,
                    emit_streaming_content=self._emit_streaming_content,
                )

                unsubscribe = sdk_session.on(event_handler)
                try:
                    # Record TTFT start time before send
                    # Contract: behaviors:Streaming:MUST:1
                    ttft_state["start_time"] = time.time()
                    # SDK v0.2.0: send(prompt, attachments=...) replaces send({"prompt": ...})
                    # Contract: sdk-boundary:ImagePassthrough:MUST:7
                    logger.debug("[SDK_COMPLETION] Sending prompt to SDK session...")
                    await sdk_session.send(prompt, attachments=attachments)
                    logger.debug("[SDK_COMPLETION] Prompt sent, waiting for idle_event...")
                    # Await idle_event directly — deadline is enforced by the enclosing
                    # async with asyncio.timeout(timeout): above.
                    # Contract: error-hierarchy:AbortError:MUST:2
                    # MUST NOT use asyncio.wait_for with the same deadline here.
                    # Duplicating the deadline splits cancel ownership: when both timeouts
                    # fire at the same absolute time, asyncio.timeout.__aexit__ may not
                    # hold sole ownership of the CancelledError and falls back to
                    # re-raising it, causing the C-2 guard to misclassify a server timeout
                    # as AbortError("Request cancelled") instead of LLMTimeoutError.
                    # The outer asyncio.timeout is the sole deadline mechanism for all
                    # awaits within _execute_sdk_completion, including this one.
                    await idle_event.wait()
                    logger.debug("[SDK_COMPLETION] idle_event received, draining queue...")

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
                                from .security_redaction import redact_sensitive_text

                                logger.debug(
                                    "[provider] Session abort failed (non-critical): %s",
                                    redact_sensitive_text(e),
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

                    # Inject captured usage if accumulator doesn't have it
                    # This handles race condition where assistant.usage arrives
                    # after session.idle but before we finish draining the queue
                    # Contract: streaming-contract:usage:MUST:1
                    if not accumulator.usage and usage_holder:
                        usage_data = usage_holder[0]
                        accumulator.add(
                            DomainEvent(
                                type=DomainEventType.USAGE_UPDATE,
                                data=usage_data,
                            )
                        )
                        logger.debug(
                            "[provider] Injected captured usage: %s",
                            usage_data,
                        )

                    # DEBUG: Log completion summary
                    logger.debug(
                        "[SDK_COMPLETION] Complete: text_len=%d, tool_calls=%d, "
                        "usage=%s, finish_reason=%s",
                        len(accumulator.text_content),
                        len(accumulator.tool_calls),
                        accumulator.usage,
                        accumulator.finish_reason,
                    )
                finally:
                    unsubscribe()

    # =========================================================================
    # Progressive Streaming Emission
    # Contract: streaming-contract:ProgressiveStreaming:SHOULD:1-4
    # =========================================================================

    def _emit_streaming_content(
        self,
        content: Any,
    ) -> None:
        """Emit streaming content for real-time UI updates.

        Fire-and-forget pattern: creates async task, doesn't block.
        Contract: streaming-contract:ProgressiveStreaming:SHOULD:1-4

        Args:
            content: Content block to emit (TextContent, ThinkingContent, ToolCallContent)
        """
        # SHOULD:4 — gracefully skip when no coordinator or hooks
        if not self.coordinator or not hasattr(self.coordinator, "hooks"):
            return

        # SHOULD:2 — fire-and-forget async emission
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(
                self._emit_content_async(content),
                name=f"emit_content_{id(content)}",
            )
            # SHOULD:3 — track pending tasks for cleanup
            self._pending_emit_tasks.add(task)
            task.add_done_callback(self._pending_emit_tasks.discard)
            # Handle errors silently to avoid blocking
            task.add_done_callback(self._handle_emit_task_exception)
        except RuntimeError:
            # No running loop - skip emission
            logger.debug("[PROVIDER] No running event loop for streaming emission")

    async def _emit_content_async(self, content: Any) -> None:
        """Async helper to emit content through hooks.

        Contract: streaming-contract:ProgressiveStreaming:SHOULD:1
        """
        # Guard against None coordinator (shouldn't happen due to _emit_streaming_content check)
        if self.coordinator is None:
            return
        try:
            # Serialize content to JSON-compatible dict
            # TextContent/ThinkingContent from amplifier_core have __dict__ with enum fields
            content_data: dict[str, Any]
            if hasattr(content, "__dict__"):
                content_data = {}
                content_vars = cast(dict[str, Any], vars(content))
                for k, v in content_vars.items():
                    # Convert enums to their value for JSON serialization
                    if hasattr(v, "value"):
                        content_data[k] = v.value
                    else:
                        content_data[k] = v
            else:
                content_data = {"value": content}

            await self.coordinator.hooks.emit(
                "llm:content_block",
                {
                    "provider": self.name,
                    "content": content_data,
                },
            )
        except Exception as e:
            from .security_redaction import redact_sensitive_text

            logger.debug("[PROVIDER] Content emit failed: %s", redact_sensitive_text(e))

    def _handle_emit_task_exception(self, task: asyncio.Task[Any]) -> None:
        """Handle exceptions from emit tasks silently.

        Prevents unhandled task exception warnings while still logging.
        """
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            from .security_redaction import redact_sensitive_text

            logger.debug("[PROVIDER] Emit task failed: %s", redact_sensitive_text(exc))

    async def cancel_emit_tasks(self) -> None:
        """Cancel and await all pending background emit tasks.

        Contract: streaming-contract:ProgressiveStreaming:SHOULD:3

        Separates task cancellation from client close so that mount() cleanup
        can cancel tasks without prematurely closing the shared client.
        """
        tasks_to_cancel = [t for t in self._pending_emit_tasks if not t.done()]
        for task in tasks_to_cancel:
            task.cancel()
        if tasks_to_cancel:
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
        self._pending_emit_tasks.clear()

    async def close(self) -> None:
        """Clean up provider resources.

        Contract: provider-protocol:close:MUST:1 — must clean up SDK resources
        Contract: sdk-boundary.md — provider must clean up SDK resources on close
        Contract: streaming-contract:ProgressiveStreaming:SHOULD:3 — clean up emit tasks

        Delegates to client.close() for SDK resource cleanup.
        Safe to call multiple times (idempotent).
        """
        await self.cancel_emit_tasks()

        if hasattr(self, "_client") and self._client:
            await self._client.close()

    def parse_tool_calls(self, response: ChatResponse) -> list[ToolCall]:
        """Extract tool calls from response.

        Contract: provider-protocol:parse_tool_calls:MUST:1 through MUST:4

        M-1 Fix: Type signature now matches kernel contract (ChatResponse).
        The underlying tool_parsing module uses defensive getattr() so it
        works with any response-like object, but the Provider interface
        is contract-compliant.

        Delegates to tool_parsing module.
        """
        return parse_tool_calls(response)
