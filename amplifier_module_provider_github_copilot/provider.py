"""
Copilot SDK Provider for Amplifier.

This module implements the Amplifier Provider protocol using the
GitHub Copilot CLI SDK as the backend.

Pattern A: Stateless Provider (Deny + Destroy)
- Each complete() call creates an ephemeral Copilot session
- Amplifier maintains all conversation state (history, compaction, agents)
- Provider is a "dumb pipe" to the LLM

Tool Calling Architecture (empirically validated):
- Tools are registered with SDK so the LLM sees structured definitions
- A preToolUse deny hook prevents the CLI from executing any tools
- Tool requests are captured from ASSISTANT_MESSAGE events
- Session is destroyed after capture to prevent CLI retry loops
- Amplifier's orchestrator executes tools and calls complete() again

This preserves Amplifier's full potential:
✅ FIC context compaction (handled by Context Manager BEFORE provider call)
✅ Session persistence (handled by Amplifier's hooks)
✅ Agent delegation (handled by Orchestrator)
✅ Tool execution (handled by Orchestrator, NOT the CLI)

Phase 2 Additions:
✅ Streaming via session events (assistant.message_delta, assistant.reasoning_delta)
✅ Extended thinking via reasoning_effort parameter
✅ ThinkingBlock output for reasoning models
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import OrderedDict
from typing import Any

from amplifier_core import (
    ChatResponse,
    ProviderInfo,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolCallContent,
)
from amplifier_core.llm_errors import (
    AbortError as KernelAbortError,
)
from amplifier_core.llm_errors import (
    AuthenticationError as KernelAuthenticationError,
)
from amplifier_core.llm_errors import (
    LLMError as KernelLLMError,
)
from amplifier_core.llm_errors import (
    LLMTimeoutError as KernelLLMTimeoutError,
)
from amplifier_core.llm_errors import (
    NetworkError as KernelNetworkError,
)
from amplifier_core.llm_errors import (
    NotFoundError as KernelNotFoundError,
)
from amplifier_core.llm_errors import (
    ProviderUnavailableError as KernelProviderUnavailableError,
)
from amplifier_core.llm_errors import (
    RateLimitError as KernelRateLimitError,
)
from amplifier_core.utils import truncate_values
from amplifier_core.utils.retry import RetryConfig, retry_with_backoff

from ._constants import (
    COPILOT_BUILTIN_TOOL_NAMES,
    DEFAULT_DEBUG_TRUNCATE_LENGTH,
    DEFAULT_THINKING_TIMEOUT,
    DEFAULT_TIMEOUT,
    MAX_REPAIRED_TOOL_IDS,
    SDK_MAX_TURNS_DEFAULT,
)
from .client import CopilotClientWrapper
from .converters import (
    convert_copilot_response_to_chat_response,
    convert_messages_to_prompt,
    extract_system_message,
)
from .exceptions import (
    CopilotAbortError,
    CopilotAuthenticationError,
    CopilotConnectionError,
    CopilotModelNotFoundError,
    CopilotProviderError,
    CopilotRateLimitError,
    CopilotSdkLoopError,
    CopilotSessionError,
    CopilotTimeoutError,
    detect_rate_limit_error,
)
from .model_cache import (
    CacheEntry,
    get_fallback_limits,
    is_cache_stale,
    load_cache,
    write_cache,
)
from .model_naming import is_thinking_model
from .models import CopilotModelInfo, fetch_and_map_models, get_default_model
from .tool_capture import convert_tools_for_sdk, make_deny_all_hook

logger = logging.getLogger(__name__)


class CopilotSdkProvider:
    """
    Amplifier provider that uses GitHub Copilot CLI SDK.

    This provider implements the 5-method Amplifier Provider protocol:
    1. name (property) - Provider identifier
    2. get_info() - Provider metadata
    3. list_models() - Available models
    4. complete() - Execute LLM completion
    5. parse_tool_calls() - Extract tool calls from response

    Pattern A: Stateless Provider
    - Each complete() call creates an ephemeral Copilot session
    - Session is destroyed after the response is received
    - Amplifier maintains all conversation state externally

    This design ensures Amplifier's full capabilities remain intact:
    - FIC context compaction (Context Manager applies before provider call)
    - Session persistence (Amplifier's hooks manage persistence)
    - Agent delegation (Orchestrator handles agent switching)
    - Tool execution (Orchestrator executes tools, not provider)

    Attributes:
        model: Default model ID for completions
        timeout: Request timeout in seconds
        debug: Enable debug logging

    Example:
        >>> provider = CopilotSdkProvider(
        ...     config={"model": "claude-opus-4.5"},
        ...     coordinator=coordinator,
        ... )
        >>> response = await provider.complete(request)
        >>> tool_calls = provider.parse_tool_calls(response)
    """

    # Class-level provider attributes (required by Amplifier)
    api_label = "GitHub Copilot"

    def __init__(
        self,
        api_key: str | None = None,  # Unused, kept for signature compatibility
        config: dict[str, Any] | None = None,
        coordinator: Any | None = None,  # ModuleCoordinator
        client: CopilotClientWrapper | None = None,  # Shared singleton if provided
    ):
        """
        Initialize the Copilot SDK provider.

        Args:
            api_key: Unused (Copilot uses GitHub auth, not API keys)
            config: Provider configuration dict with optional keys:
                - default_model: Default model ID (default: "claude-opus-4.5")
                - model: Alias for default_model (backward compatibility)
                - timeout: Request timeout in seconds (default: 3600 / 1 hour)
                - thinking_timeout: Timeout for thinking models (default: 3600 / 1 hour)
                - debug: Enable debug logging (default: False)
                - debug_truncate_length: Max length for debug output (default: 180)
                - cli_path: Path to Copilot CLI executable
            coordinator: Amplifier's ModuleCoordinator for hooks and events
            client: Optional pre-created CopilotClientWrapper to reuse. If None,
                a new wrapper is created (backward-compatible default). Pass the
                shared singleton from _acquire_shared_client() to avoid spawning
                multiple copilot subprocesses.

        Note:
            Timeout is automatically selected based on whether extended_thinking
            is enabled in the complete() call. Extended thinking models like
            Claude Opus 4.6 and OpenAI o-series need longer timeouts as they
            "think before responding" (per Anthropic docs: "often > 5 minutes").
        """
        config = config or {}
        self._config = config
        self._coordinator = coordinator
        self.coordinator = coordinator  # Also expose as public attribute

        # Note: We do NOT auto-discover cli_path here.
        # The SDK bundles its own CLI binary which is version-matched.
        # Using a system-installed CLI can cause version mismatches.
        # cli_path is only used if explicitly set in config or COPILOT_CLI_PATH env var.
        # Client wrapper handles the cli_path logic.

        # Core configuration
        # Runtime model (from provider_preferences) > bundle default > code default
        self._model = config.get("model", config.get("default_model", get_default_model()))
        # Dual timeout configuration (like OpenAI provider pattern)
        # - timeout: for regular models (default 5 min)
        # - thinking_timeout: for extended thinking models (default 30 min)
        self._timeout = float(config.get("timeout", DEFAULT_TIMEOUT))
        self._thinking_timeout = float(config.get("thinking_timeout", DEFAULT_THINKING_TIMEOUT))
        self._debug = bool(config.get("debug", False))
        self._raw_debug = bool(config.get("raw_debug", False))
        self._debug_truncate_length = int(
            config.get("debug_truncate_length", DEFAULT_DEBUG_TRUNCATE_LENGTH)
        )

        # Streaming configuration (default: enabled like Anthropic provider)
        self._use_streaming = bool(config.get("use_streaming", True))

        # Initialize client wrapper — use injected singleton if provided,
        # otherwise create a new one (backward-compatible default).
        self._client = (
            client
            if client is not None
            else CopilotClientWrapper(
                config=config,
                timeout=self._timeout,
            )
        )

        # Retry configuration (lighter defaults than HTTP-only providers
        # because each Copilot retry creates a new SDK session + subprocess health check)
        self._retry_config = RetryConfig(
            max_retries=int(config.get("max_retries", 3)),
            min_delay=float(config.get("retry_min_delay", 1.0)),
            max_delay=float(config.get("retry_max_delay", 60.0)),
            jitter=float(config.get("retry_jitter", 0.2)),
        )

        # Track tool call IDs that have been repaired with synthetic results.
        # This prevents infinite loops when the same missing tool results are
        # detected repeatedly across LLM iterations.
        # Uses LRU-bounded dict to prevent unbounded memory growth.
        self._repaired_tool_ids: OrderedDict[str, None] = OrderedDict()
        self._max_repaired_ids = MAX_REPAIRED_TOOL_IDS

        # Cache for model capabilities to avoid repeated API calls
        # Maps model_id -> list of capability strings
        self._model_capabilities_cache: dict[str, list[str]] = {}

        # Cache for full model info to support get_model_info()
        # Maps model_id -> ModelInfo object with context_window, max_output_tokens
        self._model_info_cache: dict[str, Any] = {}

        # Lock for thread-safe cache population (prevents duplicate fetches)
        self._cache_lock = asyncio.Lock()

        # Track pending emit tasks for cleanup on close
        self._pending_emit_tasks: set[asyncio.Task] = set()

        # Session metrics tracking
        # NOTE on semantics:
        # - _request_count counts ATTEMPTS (incremented before API call)
        # - _session_count counts successful session CREATIONS (inside context manager)
        # - _error_count counts failed attempts (in except block)
        # - _total_response_time_ms accumulates only SUCCESSFUL request times
        # Therefore: avg_response_time_ms = total_successful_time / total_attempts
        # This is intentional: it gives a conservative estimate that accounts for
        # failed requests (which contribute 0ms to the numerator but 1 to denominator)
        self._session_count: int = 0
        self._request_count: int = 0
        self._total_response_time_ms: float = 0.0  # Accumulated in milliseconds
        self._error_count: int = 0

        # Load model cache from disk (written by list_models() during amplifier init)
        # This enables get_info() to return accurate context limits without API calls
        self._load_model_cache_from_disk()

        logger.info(
            f"[PROVIDER] CopilotSdkProvider initialized - "
            f"model: {self._model}, timeout: {self._timeout}s, "
            f"thinking_timeout: {self._thinking_timeout}s, streaming: {self._use_streaming}"
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # Provider Protocol: Method 1 - name property
    # ═══════════════════════════════════════════════════════════════════════════

    @property
    def name(self) -> str:
        """
        Return the provider name.

        This is used as the identifier when selecting providers
        in Amplifier configuration.

        Returns:
            Provider name string
        """
        return "github-copilot"

    # ═══════════════════════════════════════════════════════════════════════════
    # Provider Protocol: Method 2 - get_info()
    # ═══════════════════════════════════════════════════════════════════════════

    def get_info(self) -> ProviderInfo:
        """
        Return provider metadata.

        Provides information about the provider's capabilities
        for display and configuration purposes.

        The defaults dict includes context_window and max_output_tokens
        which are read by the context manager to calculate token budgets.
        These values come from cached model info or BUNDLED_MODEL_LIMITS.

        Returns:
            ProviderInfo with provider details
        """
        # Get context limits from model info (cache or fallback)
        # NOTE: get_model_info() already checks cache AND BUNDLED_MODEL_LIMITS fallback
        # If it returns None, the model is truly unknown — don't call get_fallback_limits again
        model_info = self.get_model_info()
        context_window = getattr(model_info, "context_window", None) if model_info else None
        max_output_tokens = getattr(model_info, "max_output_tokens", None) if model_info else None

        # Use hardcoded defaults only if model_info was None (truly unknown model)
        # BUG 3 FIX: Don't call get_fallback_limits() here — get_model_info() already did
        if context_window is None:
            logger.warning(
                f"[PROVIDER] Model '{self._model}' not in cache or BUNDLED_MODEL_LIMITS - "
                f"using default context_window=200000. This may be incorrect."
            )
            context_window = 200000
        if max_output_tokens is None:
            logger.warning(
                f"[PROVIDER] Model '{self._model}' not in cache or BUNDLED_MODEL_LIMITS - "
                f"using default max_output_tokens=32000. This may be incorrect."
            )
            max_output_tokens = 32000

        # Trace log for context manager handshake verification
        logger.debug(
            f"[PROVIDER] get_info() returning context_window={context_window}, "
            f"max_output_tokens={max_output_tokens} for model={self._model}"
        )

        return ProviderInfo(
            id=self.name,
            display_name="GitHub Copilot SDK",
            credential_env_vars=["GITHUB_TOKEN", "GH_TOKEN", "COPILOT_GITHUB_TOKEN"],
            # Provider-level capabilities (model-specific caps like "thinking" are in list_models)
            capabilities=["streaming", "tools", "vision"],
            defaults={
                "model": get_default_model(),
                "max_tokens": 4096,
                "temperature": 0.7,
                "timeout": self._timeout,
                "thinking_timeout": self._thinking_timeout,
                # CRITICAL: Context manager reads these for budget calculation
                # See CONTEXT_CONTRACT.md _calculate_budget()
                "context_window": context_window,
                "max_output_tokens": max_output_tokens,
            },
            config_fields=[],
        )

    def get_model_info(self) -> Any | None:
        """
        Return model info for the currently selected model.

        This method is called by Amplifier's context manager to calculate
        the token budget for compaction. Returns the current model's
        context_window and max_output_tokens from SDK metadata.

        Uses cached model info if available. If cache is empty,
        returns fallback from BUNDLED_MODEL_LIMITS (based on SDK data).

        Returns:
            Object with context_window and max_output_tokens attributes,
            or None if not available.
        """
        # Check cache first (populated by list_models())
        if self._model in self._model_info_cache:
            model_info = self._model_info_cache[self._model]
            logger.debug(
                f"[PROVIDER] get_model_info() - cache hit: {self._model}, "
                f"context_window={model_info.context_window}"
            )
            return model_info

        # Cache miss - use fallback from bundled limits
        fallback = get_fallback_limits(self._model)
        if fallback is not None:
            context_window, max_output = fallback
            fallback = CopilotModelInfo(
                id=self._model,
                name=self._model,
                provider="unknown",
                context_window=context_window,
                max_output_tokens=max_output,
                supports_tools=True,
                supports_vision=False,
                supports_extended_thinking=False,
            )
            logger.info(
                f"[PROVIDER] get_model_info() - using known limits for {self._model}: "
                f"context_window={context_window}, max_output={max_output}"
            )
            return fallback

        # Unknown model - return None, context manager uses bundle config
        logger.debug(
            f"[PROVIDER] get_model_info() - unknown model {self._model}, "
            f"returning None (will use bundle config)"
        )
        return None

    def get_session_metrics(self) -> dict[str, Any]:
        """
        Return session metrics for monitoring and debugging.

        Provides aggregate statistics about provider usage including
        request counts, session counts, average response times, and errors.

        Note on semantics:
            - total_requests: Number of complete() attempts (including failures)
            - total_sessions: Number of successfully created SDK sessions
            - avg_response_time_ms: Average time of successful requests
              divided by total attempts (conservative estimate)
            - error_count: Number of failed complete() calls

        Returns:
            Dict with total_sessions, total_requests, avg_response_time_ms, error_count
        """
        return {
            "total_sessions": self._session_count,
            "total_requests": self._request_count,
            "avg_response_time_ms": self._total_response_time_ms / max(self._request_count, 1),
            "error_count": self._error_count,
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # Provider Protocol: Method 3 - list_models()
    # ═══════════════════════════════════════════════════════════════════════════

    async def list_models(self) -> list[Any]:
        """
        List available models from Copilot SDK.

        Fetches the current list of available models from the
        Copilot SDK and returns them in Amplifier format.

        Returns:
            List of model info objects with attributes:
                - id: Model identifier
                - name: Display name
                - provider: Upstream provider
                - context_window: Max context tokens
                - max_output_tokens: Max output tokens
                - supports_tools: Tool calling support
                - supports_vision: Vision support
        """
        logger.info("[PROVIDER] list_models() - START")
        try:
            result = await fetch_and_map_models(self._client)

            # CLEAR existing cache before populating with fresh SDK data.
            # FIX: Previously merged old cache with new, causing stale models
            # (like test fixtures 'other-model') to persist indefinitely.
            self._model_info_cache.clear()

            # Populate model info cache for get_model_info()
            # This enables context manager to get accurate context_window
            for model in result:
                model_id = getattr(model, "id", None)
                if model_id:
                    self._model_info_cache[model_id] = model
                    logger.debug(
                        f"[PROVIDER] Cached model info: {model_id}, "
                        f"context_window={getattr(model, 'context_window', 'N/A')}"
                    )

            # Persist cache to disk for future sessions
            self._write_model_cache_to_disk()

            logger.info(f"[PROVIDER] list_models() - END, got {len(result)} models")
            return result
        except Exception as e:
            logger.error(f"[PROVIDER] list_models() failed: {e}")
            raise

    # ═══════════════════════════════════════════════════════════════════════════
    # Helper: Check if model supports reasoning/extended_thinking
    # ═══════════════════════════════════════════════════════════════════════════

    async def _model_supports_reasoning(self, model_id: str) -> bool:
        """
        Check if a specific model supports extended thinking/reasoning.

        This method queries the available models and checks if the specified
        model has the reasoning_effort capability enabled. This is used to
        prevent sending reasoning_effort to models that don't support it,
        even if the orchestrator/bundle requests it.

        Uses a cache to avoid repeated API calls - capabilities are fetched
        once per model and cached for the lifetime of the provider instance.

        Thread-safety: Uses double-checked locking pattern to ensure only one
        coroutine fetches the model list when cache is cold, while allowing
        lock-free reads on cache hits.

        Args:
            model_id: The model ID to check (e.g., "claude-opus-4.5")

        Returns:
            True if the model explicitly supports reasoning, False otherwise
        """
        # Fast path: Check cache without lock (common case)
        if model_id in self._model_capabilities_cache:
            caps = self._model_capabilities_cache[model_id]
            return "thinking" in caps or "reasoning" in caps

        # Slow path: Acquire lock and fetch
        async with self._cache_lock:
            # Re-check after acquiring lock (double-checked locking)
            if model_id in self._model_capabilities_cache:
                caps = self._model_capabilities_cache[model_id]
                return "thinking" in caps or "reasoning" in caps

            # Fetch and cache capabilities
            # NOTE: Exceptions from list_models() are NOT caught here.
            # They propagate to _check_model_reasoning_with_fallback()
            # which uses pattern-based fallback when SDK check fails.
            models = await self.list_models()
            for model in models:
                # Cache all models while we have them
                caps = getattr(model, "capabilities", []) or []
                self._model_capabilities_cache[model.id] = list(caps)

            # Model not found in list - cache empty and assume no support
            if model_id not in self._model_capabilities_cache:
                logger.debug(
                    f"[PROVIDER] Model '{model_id}' not in list, assuming no reasoning support"
                )
                self._model_capabilities_cache[model_id] = []

            # Return result while still holding lock
            caps = self._model_capabilities_cache.get(model_id, [])
            return "thinking" in caps or "reasoning" in caps

    async def _check_model_reasoning_with_fallback(self, model_id: str) -> tuple[bool, bool]:
        """
        Check if model supports reasoning, with fallback tracking.

        Returns tuple of (supports_reasoning, sdk_check_succeeded).

        Design: SDK is AUTHORITATIVE. Pattern matching is FALLBACK only
        when SDK check fails (network error, auth expired, etc.).

        This ensures:
        - When SDK works: We trust SDK (e.g., opus-4.5 has NO thinking)
        - When SDK fails: We use pattern as safety net (e.g., "opus" → long timeout)
        """
        try:
            result = await self._model_supports_reasoning(model_id)
            return (result, True)  # SDK succeeded
        except Exception as e:
            # SDK failed - use pattern fallback
            logger.warning(
                f"[PROVIDER] SDK capability check failed for '{model_id}': {e}. "
                f"Using pattern-based fallback."
            )
            return (is_thinking_model(model_id), False)  # SDK failed, using fallback

    # ═══════════════════════════════════════════════════════════════════════════
    # Provider Protocol: Method 4 - complete() [CORE METHOD]
    # ═══════════════════════════════════════════════════════════════════════════

    async def complete(
        self,
        request: Any,  # ChatRequest from Amplifier
        **kwargs: Any,
    ) -> ChatResponse:
        """
        Execute completion using Copilot SDK.

        CRITICAL DESIGN: This method receives the FULL context from Amplifier.

        What Amplifier has ALREADY done before calling this:
        1. Context Manager: Applied FIC compaction if needed
        2. Context Manager: Provided complete message history
        3. Orchestrator: Injected tool definitions
        4. Orchestrator: Set up agent context if delegating

        What this method does:
        1. Extract system message from messages
        2. Convert messages to prompt format
        3. Create ephemeral Copilot session (with streaming if enabled)
        4. Send prompt and collect response (streaming or blocking)
        5. Convert response to ChatResponse (including ThinkingBlock for reasoning)

        What happens AFTER this returns:
        1. Orchestrator: Parses tool calls via parse_tool_calls()
        2. Orchestrator: Executes tools if needed
        3. Context Manager: Stores response
        4. Loop continues if tools were called

        Args:
            request: ChatRequest with messages, tools, and config
            **kwargs: Additional options including:
                - model: Override default model
                - timeout: Override default timeout
                - extended_thinking: Enable extended thinking (reasoning)
                - reasoning_effort: "low", "medium", "high", "xhigh"

        Returns:
            ChatResponse with content blocks, tool calls, and usage

        Raises:
            CopilotProviderError: If completion fails
            CopilotTimeoutError: If request times out
            CopilotSessionError: If session management fails
        """
        # Get model (allow override via kwargs)
        model = kwargs.get("model", self._model)

        # Increment request counter
        self._request_count += 1

        # Extract messages from request
        messages = self._extract_messages(request)

        # VALIDATE AND REPAIR: Check for missing tool results (backup safety net)
        messages = await self._repair_missing_tool_results(messages)

        # Extended thinking / reasoning support
        # NOTE: Amplifier's orchestrator (loop-streaming) may request extended_thinking=True
        # based on bundle config, but we MUST only enable it if the model actually supports it.
        # The SDK will reject reasoning_effort for models that don't support it.
        extended_thinking_requested = bool(kwargs.get("extended_thinking", False))
        reasoning_effort = kwargs.get("reasoning_effort", "medium")

        # Check if model supports reasoning (uses internal cache after first call)
        # SDK is authoritative; pattern fallback only on SDK failure
        (
            model_supports_reasoning,
            sdk_check_succeeded,
        ) = await self._check_model_reasoning_with_fallback(model)

        # Only enable extended thinking if BOTH requested AND supported
        # Named 'enabled' to distinguish from 'requested' (raw user input)
        extended_thinking_enabled = extended_thinking_requested and model_supports_reasoning

        # CRITICAL: Only pass reasoning_effort if model supports it
        effective_reasoning_effort = reasoning_effort if extended_thinking_enabled else None

        # Log what we received vs what we'll actually use
        if extended_thinking_requested and not model_supports_reasoning:
            logger.info(
                f"[PROVIDER] Extended thinking requested but model '{model}' doesn't support it. "
                f"Proceeding without reasoning_effort."
            )
        elif extended_thinking_enabled:
            logger.info(
                f"[PROVIDER] Extended thinking enabled for model '{model}' "
                f"with effort: {effective_reasoning_effort}"
            )

        # Determine if streaming should be used (check request.stream or use default)
        use_streaming = self._use_streaming
        if hasattr(request, "stream") and request.stream is not None:
            use_streaming = bool(request.stream)

        # Deny + Destroy pattern requires event-based tool capture,
        # which needs streaming mode (send + event handler).
        # Force streaming whenever tools are present.
        request_tools_present = (
            getattr(request, "tools", None)
            if not isinstance(request, dict)
            else request.get("tools")
        )
        if request_tools_present and not use_streaming:
            logger.debug("[PROVIDER] Forcing streaming=True for event-based tool capture")
            use_streaming = True

        # Log request info (without sensitive content)
        logger.info(
            f"[PROVIDER] Copilot SDK complete() - "
            f"model: {model}, messages: {len(messages)}, "
            f"streaming: {use_streaming}, thinking: {extended_thinking_enabled}"
        )

        if self._debug:
            logger.debug(
                f"[PROVIDER] extended_thinking_enabled={extended_thinking_enabled}, "
                f"reasoning_effort={reasoning_effort}"
            )

        # Extract system message (handled separately in Copilot)
        system_message = extract_system_message(messages)

        # Convert Amplifier messages to prompt format
        # NOTE: Messages are already compacted by Amplifier if needed
        prompt = convert_messages_to_prompt(messages)

        if self._debug:
            truncated = self._truncate(prompt)
            logger.debug(f"[PROVIDER] Prompt ({len(prompt)} chars): {truncated}")

        # TIMEOUT SELECTION: Based on REQUEST INTENT, not capability detection
        #
        # Design principle: Timeout reflects what the user REQUESTED, not what
        # we DETECTED. Capability detection gates the parameters sent
        # to the API, not the timeout. This prevents premature timeouts when:
        # 1. User explicitly requests extended_thinking but capability check fails
        # 2. Model naturally takes longer (e.g., Claude Opus) even without explicit request
        #
        # If the model doesn't support extended thinking, the API will reject quickly.
        # But if it does support it, we don't want to timeout before getting the response.
        #
        # TIMEOUT SELECTION: SDK is authoritative, pattern is fallback
        #
        # Design:
        # - If SDK check succeeded: trust SDK result exclusively
        # - If SDK check failed: use pattern fallback for safety
        # - If user requested extended thinking: trust user intent
        #
        # This ensures opus-4.5 (SDK says no thinking) gets short timeout,
        # but if network fails and we can't check SDK, pattern saves us.

        if "timeout" in kwargs:
            # User explicitly provided timeout, use it
            timeout = float(kwargs["timeout"])
        elif extended_thinking_requested or model_supports_reasoning:
            # Use thinking timeout if:
            # 1. User explicitly requested extended thinking (trust user intent), OR
            # 2. Model supports reasoning (SDK or fallback says yes)
            timeout = self._thinking_timeout
            logger.info(
                f"[PROVIDER] Using thinking_timeout={timeout}s "
                f"(requested={extended_thinking_requested}, "
                f"detected={model_supports_reasoning}, "
                f"sdk_check_succeeded={sdk_check_succeeded}, "
                f"enabled={extended_thinking_enabled})"
            )
        else:
            # Regular model without extended thinking request: use standard timeout
            timeout = self._timeout

        # Extract tools from request for SDK tool calling (Deny + Destroy pattern)
        # The orchestrator populates request.tools on every iteration of the loop.
        # We convert them to SDK Tool objects with no-op handlers so the LLM
        # sees structured tool definitions and can return tool_requests.
        # A preToolUse deny hook prevents the CLI from executing any tools.
        # Session destroy (via context manager) prevents CLI retry loops.
        sdk_tools = None
        deny_hooks = None
        excluded_builtins: list[str] = []
        request_tools = (
            getattr(request, "tools", None)
            if not isinstance(request, dict)
            else request.get("tools")
        )
        if request_tools:
            # Sort tools alphabetically for consistent ordering
            # Benefits: deterministic logs, reduced model bias, easier debugging
            def get_tool_name(t: Any) -> str:
                if isinstance(t, dict):
                    return t.get("name", "")
                return getattr(t, "name", "")

            sorted_tools = sorted(request_tools, key=get_tool_name)
            sdk_tools = convert_tools_for_sdk(sorted_tools)
            deny_hooks = make_deny_all_hook()

            # Exclude ALL CLI built-ins when user tools are present.
            #
            # EVIDENCE (Session 497bbab7):
            # The CLI SDK has built-in tools (view, edit, bash, grep, etc.)
            # that execute INSIDE the CLI process, invisible to our provider.
            # If not excluded, the model may choose the built-in over our
            # custom tool, bypassing the orchestrator entirely.
            #
            # Additionally, the API returns "Tool names must be unique" if
            # any built-in name overlaps with a custom tool name, even
            # semantically similar ones.
            #
            # Solution: Exclude ALL 13 known built-ins unconditionally
            # when user tools are registered. This ensures:
            # 1. No name collisions at the API level
            # 2. No silent internal execution bypassing the orchestrator
            # 3. The model can ONLY use custom tools we control
            excluded_builtins = sorted(COPILOT_BUILTIN_TOOL_NAMES)

            # Log tool names in sorted order for traceability
            tool_names = [get_tool_name(t) or "?" for t in sorted_tools]
            logger.info(
                f"[PROVIDER] Registering {len(sdk_tools)} tool(s) with SDK session "
                f"(deny + destroy pattern), excluding {len(excluded_builtins)} built-in(s)"
            )
            logger.debug(f"[PROVIDER] Tools (alphabetical): {tool_names}")
            if excluded_builtins:
                logger.debug(f"[PROVIDER] Excluded built-ins: {excluded_builtins}")

        # ── Emit llm:request events (before API call) ──────────────────
        tool_count = len(sdk_tools) if sdk_tools else 0
        request_summary = {
            "provider": self.name,
            "model": model,
            "message_count": len(messages),
            "has_system": bool(system_message),
            "tool_count": tool_count,
            "excluded_builtin_count": len(excluded_builtins),
            "excluded_builtin_tools": excluded_builtins,  # Full list for verification
            "streaming": use_streaming,
            "thinking_enabled": extended_thinking_enabled,
            "reasoning_effort": effective_reasoning_effort,
            "timeout": timeout,
        }
        if deny_hooks:
            request_summary["deny_hooks"] = True

        await self._emit_event("llm:request", request_summary)

        if self._debug:
            debug_payload: dict[str, Any] = {
                "prompt": prompt,
                "system_message": system_message,
            }
            if request_tools:
                debug_payload["tools"] = [
                    {
                        "name": getattr(t, "name", str(t)),
                        "description": getattr(t, "description", ""),
                    }
                    for t in (sdk_tools or [])
                ]
            await self._emit_event(
                "llm:request:debug",
                {
                    "lvl": "DEBUG",
                    "provider": self.name,
                    "request": self._truncate_values(debug_payload),
                },
            )

        if self._debug and self._raw_debug:
            raw_payload: dict[str, Any] = {
                "prompt": prompt,
                "system_message": system_message,
                "model": model,
                "streaming": use_streaming,
                "reasoning_effort": effective_reasoning_effort,
                "timeout": timeout,
            }
            if request_tools:
                raw_payload["tools"] = [
                    {
                        "name": getattr(t, "name", str(t)),
                        "description": getattr(t, "description", ""),
                    }
                    for t in (sdk_tools or [])
                ]
            await self._emit_event(
                "llm:request:raw",
                {
                    "lvl": "DEBUG",
                    "provider": self.name,
                    "params": raw_payload,
                },
            )

        # ── Start timing ───────────────────────────────────────────────
        # Inner function for retry_with_backoff wrapping
        async def _do_complete() -> ChatResponse:
            """Inner function wrapping session creation, send, and error translation.

            retry_with_backoff catches LLMError subtypes and retries those
            marked retryable=True. Non-retryable errors propagate immediately.
            """
            start_time = time.time()

            # Create ephemeral session (Pattern A: stateless, Deny + Destroy)
            try:
                async with self._client.create_session(
                    model=model,
                    system_message=system_message,
                    streaming=use_streaming,
                    reasoning_effort=effective_reasoning_effort,
                    tools=sdk_tools,
                    excluded_tools=excluded_builtins if excluded_builtins else None,
                    hooks=deny_hooks,
                ) as session:
                    # Increment session counter
                    self._session_count += 1

                    if use_streaming:
                        # Streaming mode: collect events and emit content blocks
                        # Event-based tool capture from ASSISTANT_MESSAGE
                        result = await self._complete_streaming(
                            session,
                            prompt,
                            model,
                            timeout,
                            extended_thinking_enabled,
                            has_tools=bool(sdk_tools),
                        )
                    else:
                        # Blocking mode: send and wait for complete response
                        raw_response = await self._client.send_and_wait(
                            session, prompt, timeout=timeout
                        )
                        result = convert_copilot_response_to_chat_response(raw_response, model)

                # Compute timing (inside try, after context manager exits)
                elapsed_ms_inner = int((time.time() - start_time) * 1000)

                # Track response time in milliseconds
                self._total_response_time_ms += elapsed_ms_inner

            except KernelLLMError:
                # Kernel errors pass through — prevent double-wrapping
                self._error_count += 1
                raise
            except CopilotAuthenticationError as e:
                self._error_count += 1
                raise KernelAuthenticationError(str(e), provider=self.name, retryable=False) from e
            except CopilotRateLimitError as e:
                self._error_count += 1
                # Rate-limit fail-fast: if retry_after exceeds max_delay,
                # mark as non-retryable to avoid pointless waits
                retryable = True
                if e.retry_after and e.retry_after > self._retry_config.max_delay:
                    retryable = False
                    logger.info(
                        f"[PROVIDER] Rate limit retry_after={e.retry_after}s "
                        f"exceeds max_delay={self._retry_config.max_delay}s, "
                        f"marking non-retryable"
                    )
                raise KernelRateLimitError(
                    str(e),
                    provider=self.name,
                    retry_after=e.retry_after,
                    retryable=retryable,
                ) from e
            except CopilotTimeoutError as e:
                self._error_count += 1
                raise KernelLLMTimeoutError(str(e), provider=self.name, retryable=True) from e
            except CopilotConnectionError as e:
                self._error_count += 1
                raise KernelNetworkError(str(e), provider=self.name, retryable=True) from e
            except CopilotModelNotFoundError as e:
                self._error_count += 1
                raise KernelNotFoundError(str(e), provider=self.name, retryable=False) from e
            except CopilotSdkLoopError as e:
                self._error_count += 1
                raise KernelProviderUnavailableError(
                    str(e), provider=self.name, retryable=False
                ) from e
            except CopilotSessionError as e:
                self._error_count += 1
                raise KernelProviderUnavailableError(
                    str(e), provider=self.name, retryable=True
                ) from e
            except CopilotAbortError as e:
                self._error_count += 1
                raise KernelAbortError(str(e), provider=self.name, retryable=False) from e
            except CopilotProviderError as e:
                self._error_count += 1
                raise KernelLLMError(str(e), provider=self.name, retryable=True) from e
            except Exception as e:
                self._error_count += 1
                rate_limit_err = detect_rate_limit_error(str(e))
                if rate_limit_err is not None:
                    logger.info(
                        f"[PROVIDER] Catch-all detected rate-limit in unexpected error: {e}"
                    )
                    retryable = True
                    if (
                        rate_limit_err.retry_after
                        and rate_limit_err.retry_after > self._retry_config.max_delay
                    ):
                        retryable = False
                    raise KernelRateLimitError(
                        str(e),
                        provider=self.name,
                        retry_after=rate_limit_err.retry_after,
                        retryable=retryable,
                    ) from e
                raise KernelLLMError(
                    f"Unexpected error: {e}", provider=self.name, retryable=True
                ) from e

            return result

        async def _on_retry(attempt: int, delay: float, error: KernelLLMError) -> None:
            """Emit provider:retry event for observability."""
            logger.info(
                f"[PROVIDER] Retrying complete() - attempt {attempt}, "
                f"delay {delay:.1f}s, error: {error}"
            )
            await self._emit_event(
                "provider:retry",
                {
                    "provider": self.name,
                    "attempt": attempt,
                    "delay": delay,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "model": model,
                },
            )

        # Execute with retry_with_backoff (track wall-clock time across retries)
        outer_start = time.time()
        response = await retry_with_backoff(_do_complete, self._retry_config, on_retry=_on_retry)
        elapsed_ms = int((time.time() - outer_start) * 1000)

        # ── Fix 2: Defensive detection of fake tool calls ──────────────
        # When the LLM writes "[Tool Call: ...]" as plain text instead of
        # issuing structured tool_requests, the orchestrator would display
        # fake results that were never actually executed.  Detect this and
        # retry with a correction message (up to 2 times).
        _FAKE_TOOL_CALL_RE = re.compile(r"\[Tool Call:")
        _MAX_FAKE_TC_RETRIES = 2

        if request_tools and not response.tool_calls:
            # Extract all text from content blocks
            response_text = ""
            for block in response.content or []:
                if hasattr(block, "text"):
                    response_text += block.text

            fake_retry = 0
            while (
                _FAKE_TOOL_CALL_RE.search(response_text)
                and fake_retry < _MAX_FAKE_TC_RETRIES
            ):
                fake_retry += 1
                logger.warning(
                    f"[PROVIDER] Detected fake tool call text in response "
                    f"(retry {fake_retry}/{_MAX_FAKE_TC_RETRIES}). "
                    f"Re-prompting LLM to use structured tool calls."
                )
                await self._emit_event(
                    "provider:fake_tool_retry",
                    {
                        "provider": self.name,
                        "model": model,
                        "retry": fake_retry,
                    },
                )

                # Append a correction hint to the messages and re-complete
                correction_msg = {
                    "role": "user",
                    "content": (
                        "You wrote tool calls as plain text instead of using "
                        "the actual tool calling mechanism. Do NOT write "
                        "'[Tool Call: ...]' as text. Use the structured tool "
                        "calling API to invoke tools."
                    ),
                }
                messages.append(correction_msg)
                prompt = convert_messages_to_prompt(messages)

                response = await retry_with_backoff(
                    _do_complete, self._retry_config, on_retry=_on_retry
                )

                # Re-check text
                if response.tool_calls:
                    break
                response_text = ""
                for block in response.content or []:
                    if hasattr(block, "text"):
                        response_text += block.text

            elapsed_ms = int((time.time() - outer_start) * 1000)

        if self._debug:
            content_preview = self._truncate(str(response.content))
            logger.debug(f"[PROVIDER] Response content: {content_preview}")
            if response.tool_calls:
                tc_count = len(response.tool_calls) if response.tool_calls else 0
                logger.debug(f"[PROVIDER] Tool calls: {tc_count}")

        # Emit llm:response events (after API call, outside retry loop)
        response_tool_calls = len(response.tool_calls) if response.tool_calls else 0
        response_event: dict[str, Any] = {
            "provider": self.name,
            "model": model,
            "usage": {
                "input": response.usage.input_tokens if response.usage else 0,
                "output": response.usage.output_tokens if response.usage else 0,
            },
            "status": "ok",
            "finish_reason": response.finish_reason
            or ("tool_use" if response_tool_calls else "end_turn"),
            "content_blocks": len(response.content) if response.content else 0,
            "tool_calls": response_tool_calls,
            "streaming": use_streaming,
            "duration_ms": elapsed_ms,
        }
        await self._emit_event("llm:response", response_event)

        if self._debug:
            response_debug = {
                "content": [str(block) for block in (response.content or [])],
                "tool_calls": [
                    {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                    for tc in (response.tool_calls or [])
                ],
                "usage": response_event["usage"],
                "finish_reason": response_event["finish_reason"],
            }
            await self._emit_event(
                "llm:response:debug",
                {
                    "lvl": "DEBUG",
                    "provider": self.name,
                    "response": self._truncate_values(response_debug),
                    "status": "ok",
                    "duration_ms": elapsed_ms,
                },
            )

        if self._debug and self._raw_debug:
            raw_response_data = {
                "content": [str(block) for block in (response.content or [])],
                "tool_calls": [
                    {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                    for tc in (response.tool_calls or [])
                ],
                "usage": {
                    "input_tokens": response.usage.input_tokens if response.usage else 0,
                    "output_tokens": response.usage.output_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
                "finish_reason": response_event["finish_reason"],
            }
            await self._emit_event(
                "llm:response:raw",
                {
                    "lvl": "DEBUG",
                    "provider": self.name,
                    "response": raw_response_data,
                },
            )

        return response

    async def _complete_streaming(
        self,
        session: Any,
        prompt: str,
        model: str,
        timeout: float,
        extended_thinking_enabled: bool,
        has_tools: bool = False,
    ) -> ChatResponse:
        """
        Handle streaming completion with SDK Driver protection.

        Uses SdkEventHandler to:
        - Capture tools from first turn only (prevents 607-tool accumulation)
        - Abort SDK loop immediately after capture (prevents 305 retries)
        - Enforce circuit breaker limits (catches runaway loops)

        Evidence for this design:
        - Session a1a0af17: 305 turns from denial loop
        - Solution: First-turn capture + abort = 1 turn instead of 305

        Emits content blocks to hooks for streaming UI support.

        Args:
            session: Active CopilotSession
            prompt: Prompt to send
            model: Model ID for response attribution
            timeout: Request timeout in seconds
            extended_thinking_enabled: Whether thinking blocks should be captured
            has_tools: Whether tools are registered (enables event-based capture)

        Returns:
            ChatResponse with accumulated content and tool calls
        """
        from amplifier_core import TextBlock, ThinkingBlock, ToolCall, ToolCallBlock, Usage

        from .sdk_driver import SdkEventHandler

        # Create event handler with SDK Driver components
        max_turns = int(self._config.get("sdk_max_turns", SDK_MAX_TURNS_DEFAULT))

        handler = SdkEventHandler(
            max_turns=max_turns,
            first_turn_only=True,  # CRITICAL: Prevents 607-tool accumulation
            deduplicate=True,  # Safety net for any duplicates
            emit_event=self._make_emit_callback(),
        )

        # Bind session for abort capability
        handler.bind_session(session)

        # Subscribe to events
        unsubscribe = session.on(handler.on_event)

        try:
            # Send prompt (non-blocking)
            await session.send({"prompt": prompt})

            # Wait for capture or idle.
            # With first-turn-only: returns as soon as first-turn tools are captured
            # OR when session reaches idle (no tools/text-only response)
            try:
                await handler.wait_for_capture_or_idle(timeout=timeout)
            except TimeoutError:
                # Wrap asyncio.TimeoutError in CopilotTimeoutError for API consistency
                if handler.captured_tools:
                    logger.warning(
                        f"[PROVIDER] Timeout during wait but tools captured, "
                        f"returning {len(handler.captured_tools)} tool(s)"
                    )
                else:
                    raise CopilotTimeoutError(
                        f"Streaming request timed out after {timeout}s"
                    ) from None
            except Exception as e:
                # CopilotSdkLoopError or other errors
                # If we captured tools before the error, still return them
                if handler.captured_tools:
                    logger.warning(
                        f"[PROVIDER] Error during wait but tools captured, "
                        f"returning {len(handler.captured_tools)} tool(s): {e}"
                    )
                else:
                    raise

            # If abort was requested (first-turn capture or circuit breaker),
            # try to abort the session to stop the SDK's internal loop.
            # The session context manager will destroy it regardless on exit.
            if handler.should_abort and has_tools:
                try:
                    logger.info("[PROVIDER] Aborting session after tool capture")
                    await session.abort()
                except Exception as e:
                    logger.debug(f"[PROVIDER] Session abort failed (non-critical): {e}")

            # Build response from captured data
            content_blocks: list[Any] = []

            # Thinking content
            if handler.thinking_content and extended_thinking_enabled:
                content_blocks.append(
                    ThinkingBlock(
                        thinking="".join(handler.thinking_content),
                        visibility="internal",
                    )
                )

            # Text content
            if handler.text_content:
                content_blocks.append(TextBlock(type="text", text="".join(handler.text_content)))

            if handler.captured_tools:
                # Build tool call objects from CapturedToolCall
                tool_calls = [
                    ToolCall(
                        id=t.id,
                        name=t.name,
                        arguments=t.arguments,
                    )
                    for t in handler.captured_tools
                ]

                # Add ToolCallBlock content blocks
                for t in handler.captured_tools:
                    content_blocks.append(
                        ToolCallBlock(
                            type="tool_call",
                            id=t.id,
                            name=t.name,
                            input=t.arguments,
                        )
                    )

                logger.info(
                    f"[PROVIDER] Returning {len(tool_calls)} tool call(s) "
                    f"to orchestrator (captured on turn {handler.turn_count})"
                )

                usage_input = int(handler.usage_data.get("input_tokens", 0) or 0)
                usage_output = int(handler.usage_data.get("output_tokens", 0) or 0)

                return ChatResponse(
                    content=content_blocks,
                    tool_calls=tool_calls,
                    usage=Usage(
                        input_tokens=usage_input,
                        output_tokens=usage_output,
                        total_tokens=usage_input + usage_output,
                    ),
                    finish_reason="tool_use",
                )

            # No tools — text-only response
            usage_input = int(handler.usage_data.get("input_tokens", 0) or 0)
            usage_output = int(handler.usage_data.get("output_tokens", 0) or 0)

            return ChatResponse(
                content=content_blocks,
                tool_calls=None,
                usage=Usage(
                    input_tokens=usage_input,
                    output_tokens=usage_output,
                    total_tokens=usage_input + usage_output,
                ),
                finish_reason="end_turn",
            )

        finally:
            unsubscribe()

    def _make_emit_callback(self) -> Any:
        """Create emit callback for SdkEventHandler observability events."""
        if not self._coordinator or not hasattr(self._coordinator, "hooks"):
            return None

        def emit_sync(name: str, data: dict) -> None:
            """Synchronous emit wrapper that creates async tasks."""
            try:
                loop = asyncio.get_running_loop()
                task = loop.create_task(
                    self._emit_event(f"sdk_driver:{name}", data),
                    name=f"sdk_emit_{name}",
                )
                self._pending_emit_tasks.add(task)
                task.add_done_callback(self._pending_emit_tasks.discard)
                task.add_done_callback(self._handle_task_exception)
            except RuntimeError:
                pass  # No running loop

        return emit_sync

    def _emit_streaming_content(
        self,
        content: TextContent | ThinkingContent | ToolCallContent,
    ) -> None:
        """
        Emit streaming content for real-time UI updates.

        This method emits content blocks through the coordinator's hooks
        for streaming UI modules to consume. Fire-and-forget pattern.

        Args:
            content: Content block to emit
        """
        if not self._coordinator or not hasattr(self._coordinator, "hooks"):
            return

        # Fire-and-forget async emit with proper error handling
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(
                self._emit_content_async(content),
                name=f"emit_content_{id(content)}",
            )
            # Track task for cleanup on close
            self._pending_emit_tasks.add(task)
            task.add_done_callback(self._pending_emit_tasks.discard)
            # Add error callback to prevent unhandled task exceptions
            task.add_done_callback(self._handle_task_exception)
        except RuntimeError:
            # No running loop - skip emission
            if self._debug:
                logger.debug("[PROVIDER] No running event loop for streaming emission")
        except Exception as e:
            if self._debug:
                logger.debug(f"[PROVIDER] Failed to emit streaming content: {e}")

    async def _emit_content_async(
        self,
        content: TextContent | ThinkingContent | ToolCallContent,
    ) -> None:
        """Async helper to emit content through hooks."""
        try:
            await self._coordinator.hooks.emit(
                "llm:content_block",
                {
                    "provider": self.name,
                    "content": content,
                },
            )
        except Exception as e:
            if self._debug:
                logger.debug(f"[PROVIDER] Content emit failed: {e}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Provider Protocol: Method 5 - parse_tool_calls()
    # ═══════════════════════════════════════════════════════════════════════════

    def parse_tool_calls(self, response: ChatResponse) -> list[ToolCall]:
        """
        Parse tool calls from response.

        Amplifier's orchestrator will:
        1. Call this to get tool calls
        2. Execute tools via tool modules
        3. Add results to messages
        4. Call complete() again with tool results

        The provider does NOT execute tools - it only parses
        the tool call requests from the LLM response.

        Args:
            response: ChatResponse from complete()

        Returns:
            List of ToolCall objects with id, name, and arguments
        """
        return response.tool_calls or []

    # ═══════════════════════════════════════════════════════════════════════════
    # Lifecycle Methods
    # ═══════════════════════════════════════════════════════════════════════════

    async def close(self) -> None:
        """
        Cleanup provider resources.

        Stops the Copilot client, cancels pending emit tasks, and releases
        any held resources. Safe to call multiple times.
        """
        logger.info("[PROVIDER] Closing CopilotSdkProvider")

        # Cancel any pending emit tasks
        for task in self._pending_emit_tasks:
            task.cancel()
        self._pending_emit_tasks.clear()

        await self._client.close()
        logger.info("[PROVIDER] CopilotSdkProvider closed")

    def invalidate_model_cache(self) -> None:
        """
        Clear all cached model data.

        Clears both the capabilities cache and the model info cache.
        Call this if models may have changed (e.g., after reconfiguration).

        Without clearing _model_info_cache, get_model_info() would return
        stale context_window/max_output_tokens values, causing incorrect
        budget calculation in the context manager.
        """
        self._model_capabilities_cache.clear()
        self._model_info_cache.clear()
        logger.debug("[PROVIDER] Model caches invalidated (capabilities + model info)")

    def _load_model_cache_from_disk(self) -> None:
        """
        Load model info cache from disk if available.

        Called during __init__ to populate _model_info_cache from the
        persistent cache written by list_models() during amplifier init.

        This enables get_info() to return accurate context_window and
        max_output_tokens values without requiring an API call.

        Error Handling:
            - Missing file: OK, cache stays empty (will use BUNDLED_MODEL_LIMITS)
            - Corrupted file: Logged warning, cache stays empty
            - All errors are caught — cache is best-effort
        """
        from types import SimpleNamespace

        cache = load_cache()
        if cache is None:
            # No cache or error loading — will fall back to BUNDLED_MODEL_LIMITS
            return

        # BUG 5 FIX: Check cache staleness and warn user
        if is_cache_stale(cache):
            logger.info(
                f"[PROVIDER] Model cache is stale (cached at {cache.cached_at.isoformat()}). "
                f"Run 'amplifier init' to refresh model metadata."
            )

        # Populate instance cache with SimpleNamespace objects
        # (compatible with get_model_info() which expects .context_window etc.)
        for model_id, entry in cache.models.items():
            self._model_info_cache[model_id] = SimpleNamespace(
                id=model_id,
                context_window=entry.context_window,
                max_output_tokens=entry.max_output_tokens,
            )

        logger.debug(
            f"[PROVIDER] Loaded {len(cache.models)} model(s) from disk cache "
            f"(SDK v{cache.sdk_version})"
        )

    def _write_model_cache_to_disk(self) -> None:
        """
        Write model info cache to disk.

        Called by list_models() after fetching models from the SDK.
        This persists the cache so future sessions can read it without
        API calls.

        Error Handling:
            - All errors are caught and logged — cache write is best-effort
            - Failure doesn't affect list_models() return value
        """
        # Build cache entries from instance cache
        cache_entries: dict[str, CacheEntry] = {}
        for model_id, model in self._model_info_cache.items():
            context_window = getattr(model, "context_window", None)
            max_output_tokens = getattr(model, "max_output_tokens", None)

            if context_window is not None and max_output_tokens is not None:
                cache_entries[model_id] = CacheEntry(
                    context_window=int(context_window),
                    max_output_tokens=int(max_output_tokens),
                )

        if not cache_entries:
            logger.debug("[PROVIDER] No models to write to cache")
            return

        # Get SDK version if available
        sdk_version = self._get_sdk_version()
        write_cache(cache_entries, sdk_version)

    def _get_sdk_version(self) -> str:
        """
        Get the current SDK version.

        Returns:
            SDK version string, or "unknown" if not available.
        """
        try:
            import importlib.metadata

            return importlib.metadata.version("github-copilot-sdk")
        except Exception:
            return "unknown"

    # ═══════════════════════════════════════════════════════════════════════════
    # Internal Methods
    # ═══════════════════════════════════════════════════════════════════════════

    def _extract_messages(self, request: Any) -> list[dict[str, Any]]:
        """
        Extract messages from request object.

        Handles both dict-style and object-style requests.
        Converts Pydantic Message objects to dicts for compatibility
        with the converter functions.

        Args:
            request: ChatRequest (dict or object)

        Returns:
            List of message dicts
        """
        if isinstance(request, dict):
            return request.get("messages", [])
        if hasattr(request, "messages"):
            messages = request.messages
            # Convert Pydantic Message objects to dicts
            return [msg.model_dump() if hasattr(msg, "model_dump") else msg for msg in messages]
        return []

    def _truncate(self, text: str) -> str:
        """
        Truncate text for debug logging.

        Args:
            text: Text to truncate

        Returns:
            Truncated text with ellipsis if needed
        """
        if len(text) <= self._debug_truncate_length:
            return text
        return text[: self._debug_truncate_length] + "..."

    def _truncate_values(self, obj: Any, max_length: int | None = None) -> Any:
        """
        Recursively truncate string values in nested structures.

        Delegates to shared utility from amplifier_core.utils.

        Args:
            obj: Object to truncate (dict, list, or primitive)
            max_length: Max string length (defaults to debug_truncate_length)

        Returns:
            Deep copy with long strings truncated
        """
        if max_length is None:
            max_length = self._debug_truncate_length
        return truncate_values(obj, max_length)

    async def _repair_missing_tool_results(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Detect and repair missing tool results.

        This is a backup safety net for when tool results are missing from the
        conversation history. Injects synthetic error results so the LLM can
        acknowledge the failure and offer to retry.

        Args:
            messages: List of message dicts from the request

        Returns:
            The (possibly modified) messages list with synthetic results injected
        """
        from collections import defaultdict

        missing = self._find_missing_tool_results(messages)

        if not missing:
            return messages

        logger.warning(
            f"[PROVIDER] Copilot: Detected {len(missing)} missing tool result(s). "
            f"Injecting synthetic errors. This indicates a bug in context management. "
            f"Tool IDs: {[call_id for _, call_id, _ in missing]}"
        )

        # Group missing results by source assistant message index
        by_msg_idx: dict[int, list[tuple[str, str]]] = defaultdict(list)
        for msg_idx, call_id, tool_name in missing:
            by_msg_idx[msg_idx].append((call_id, tool_name))

        # Insert synthetic results in reverse order of message index
        # (so earlier insertions don't shift later indices)
        for msg_idx in sorted(by_msg_idx.keys(), reverse=True):
            synthetics = []
            for call_id, tool_name in by_msg_idx[msg_idx]:
                synthetics.append(self._create_synthetic_result(call_id, tool_name))
                # Track this ID so we don't detect it as missing again (LRU bounded)
                self._add_repaired_id(call_id)

            # Insert all synthetic results immediately after the assistant message
            insert_pos = msg_idx + 1
            for i, synthetic in enumerate(synthetics):
                messages.insert(insert_pos + i, synthetic)

        # Emit observability event
        await self._emit_event(
            "provider:tool_sequence_repaired",
            {
                "provider": self.name,
                "repair_count": len(missing),
                "repairs": [
                    {"tool_call_id": call_id, "tool_name": tool_name}
                    for _, call_id, tool_name in missing
                ],
            },
        )

        return messages

    def _find_missing_tool_results(
        self, messages: list[dict[str, Any]]
    ) -> list[tuple[int, str, str]]:
        """
        Find tool calls without matching results.

        Scans conversation for assistant tool calls and validates each has
        a corresponding tool result message. Returns missing pairs WITH their
        source message index so they can be inserted in the correct position.

        Excludes tool call IDs that have already been repaired with synthetic
        results to prevent infinite detection loops.

        Thread-safety note: This method is synchronous with no await points.
        In async/await, coroutines yield only at await. Since this method
        has none, it executes atomically within the event loop. No lock needed.

        Args:
            messages: List of message dicts from the request

        Returns:
            List of (msg_index, call_id, tool_name) tuples for unpaired calls.
            msg_index is the index of the assistant message containing the tool_use block.
        """
        tool_calls: dict[str, tuple[int, str]] = {}  # {call_id: (msg_index, name)}
        tool_results: set[str] = set()  # {call_id}

        for idx, msg in enumerate(messages):
            role = msg.get("role", "")
            content = msg.get("content", [])

            # Check assistant messages for tool_call blocks in content
            if role == "assistant" and isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        block_type = block.get("type", "")
                        if block_type == "tool_call":
                            call_id = block.get("id", "")
                            name = block.get("name", "")
                            if call_id:
                                tool_calls[call_id] = (idx, name)
                    elif hasattr(block, "type") and block.type == "tool_call":
                        # Handle typed content blocks
                        tool_calls[block.id] = (idx, block.name)

            # Check tool messages for tool_call_id
            elif role == "tool":
                tool_call_id = msg.get("tool_call_id", "")
                if tool_call_id:
                    tool_results.add(tool_call_id)

        # Exclude IDs that have already been repaired to prevent infinite loops
        return [
            (msg_idx, call_id, name)
            for call_id, (msg_idx, name) in tool_calls.items()
            if call_id not in tool_results and call_id not in self._repaired_tool_ids
        ]

    def _create_synthetic_result(self, call_id: str, tool_name: str) -> dict[str, Any]:
        """
        Create synthetic error result for missing tool response.

        This is a BACKUP for when tool results go missing AFTER execution.
        The orchestrator should handle tool execution errors at runtime,
        so this should only trigger on context/parsing bugs.

        Args:
            call_id: The tool call ID
            tool_name: Name of the tool

        Returns:
            Message dict in tool role format
        """
        return {
            "role": "tool",
            "content": (
                f"[ERROR: Tool result unavailable]\n\n"
                f"Tool: {tool_name}\n"
                f"Call ID: {call_id}\n\n"
                f"The result for this tool call could not be retrieved.\n"
                f"Please acknowledge this and offer to retry if needed."
            ),
            "tool_call_id": call_id,
            "name": tool_name,
        }

    async def _emit_event(self, event_name: str, data: dict[str, Any]) -> None:
        """
        Emit observability event if coordinator supports hooks.

        Events are used for monitoring, debugging, and integration
        with Amplifier's observability infrastructure.

        Args:
            event_name: Event type (e.g., "llm:complete")
            data: Event data dict
        """
        if self._coordinator and hasattr(self._coordinator, "hooks"):
            try:
                await self._coordinator.hooks.emit(event_name, data)
            except Exception as e:
                logger.warning(f"[PROVIDER] Failed to emit event '{event_name}': {e}")

    def _add_repaired_id(self, call_id: str) -> None:
        """
        Add a tool call ID to the repaired set with LRU eviction.

        Maintains bounded memory by evicting oldest entries when
        the maximum size is reached.

        Thread-safety note: This method is synchronous with no await points.
        In async/await, coroutines yield only at await. Since this method
        has none, it executes atomically within the event loop. No lock needed.

        Args:
            call_id: Tool call ID to track as repaired
        """
        # Move to end if exists (LRU behavior)
        if call_id in self._repaired_tool_ids:
            self._repaired_tool_ids.move_to_end(call_id)
        else:
            self._repaired_tool_ids[call_id] = None
            # Evict oldest if over limit
            while len(self._repaired_tool_ids) > self._max_repaired_ids:
                self._repaired_tool_ids.popitem(last=False)

    def _handle_task_exception(self, task: asyncio.Task) -> None:
        """
        Handle exceptions from fire-and-forget tasks.

        Prevents 'Task exception was never retrieved' warnings by
        explicitly handling and logging any task failures.

        Args:
            task: Completed asyncio.Task to check for exceptions
        """
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            if self._debug:
                logger.debug(f"[PROVIDER] Background task failed: {type(exc).__name__}: {exc}")
