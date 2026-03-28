"""Observability module for hook event emission.

Contract: contracts/observability.md

MUST constraints:
- MUST guard hook calls (observability:Events:MUST:1)
- MUST emit llm:request before SDK call (observability:Events:MUST:2)
- MUST emit llm:response after completion (observability:Events:MUST:3)
- MUST work without coordinator (observability:Events:MUST:4)
- MUST NOT assume coordinator.hooks.emit() exists (observability:Events:MUST:5)

Three-Medium Architecture:
- Event names loaded from config/observability.yaml (YAML = policy)
- This module provides emission helpers (Python = mechanism)
"""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from amplifier_core import ModuleCoordinator

logger = logging.getLogger(__name__)

__all__ = [
    "ObservabilityConfig",
    "load_observability_config",
    "emit_event",
    "llm_lifecycle",
]


# ============================================================================
# Config Loading (Three-Medium: Load policy from YAML)
# ============================================================================


@dataclass
class EventNames:
    """Event name policy from config/observability.yaml."""

    llm_request: str = "llm:request"
    llm_response: str = "llm:response"
    provider_retry: str = "provider:retry"


@dataclass
class StatusValues:
    """Status value policy from config/observability.yaml."""

    ok: str = "ok"
    error: str = "error"


@dataclass
class FinishReasons:
    """Finish reason policy from config/observability.yaml."""

    tool_use: str = "tool_use"
    end_turn: str = "end_turn"
    stop: str = "stop"
    length: str = "length"
    content_filter: str = "content_filter"


@dataclass
class ObservabilityConfig:
    """Observability policy loaded from config/observability.yaml."""

    provider_name: str = "github-copilot"
    event_names: EventNames = field(default_factory=EventNames)
    status: StatusValues = field(default_factory=StatusValues)
    finish_reasons: FinishReasons = field(default_factory=FinishReasons)
    events_enabled: bool = True
    raw_payloads: bool = False


def _default_observability_config() -> ObservabilityConfig:
    """Return default observability config (fallback)."""
    return ObservabilityConfig()


@functools.lru_cache(maxsize=1)
def load_observability_config() -> ObservabilityConfig:
    """Load observability policy from config/observability.yaml.

    Config lives inside the wheel at amplifier_module_provider_github_copilot/config/
    Uses importlib.resources for installed wheel, falls back to filesystem for dev.

    Returns default config on error (graceful degradation per contract).
    """
    yaml_text: str | None = None

    # Try importlib.resources first (installed wheel scenario)
    try:
        from importlib import resources

        config_files = resources.files("amplifier_module_provider_github_copilot.config")
        config_file = config_files.joinpath("observability.yaml")
        yaml_text = config_file.read_text(encoding="utf-8")
    except (ModuleNotFoundError, FileNotFoundError, TypeError):
        # Fall back to filesystem path (dev scenario)
        config_path = Path(__file__).parent / "config" / "observability.yaml"
        if config_path.exists():
            yaml_text = config_path.read_text(encoding="utf-8")
        else:
            logger.warning(
                "[OBSERVABILITY] Config not found via importlib.resources or path. Using defaults."
            )
            return _default_observability_config()

    if not yaml_text:
        return _default_observability_config()

    try:
        data = yaml.safe_load(yaml_text)

        if not data:
            return _default_observability_config()

        # Parse event names
        event_names_data = data.get("event_names", {})
        event_names = EventNames(
            llm_request=event_names_data.get("llm_request", "llm:request"),
            llm_response=event_names_data.get("llm_response", "llm:response"),
            provider_retry=event_names_data.get("provider_retry", "provider:retry"),
        )

        # Parse status values
        status_data = data.get("status", {})
        status = StatusValues(
            ok=status_data.get("ok", "ok"),
            error=status_data.get("error", "error"),
        )

        # Parse finish reasons
        finish_data = data.get("finish_reasons", {})
        finish_reasons = FinishReasons(
            tool_use=finish_data.get("tool_use", "tool_use"),
            end_turn=finish_data.get("end_turn", "end_turn"),
            stop=finish_data.get("stop", "stop"),
            length=finish_data.get("length", "length"),
            content_filter=finish_data.get("content_filter", "content_filter"),
        )

        # Parse events config
        events_data = data.get("events", {})

        return ObservabilityConfig(
            provider_name=data.get("provider_name", "github-copilot"),
            event_names=event_names,
            status=status,
            finish_reasons=finish_reasons,
            events_enabled=events_data.get("enabled", True),
            raw_payloads=events_data.get("raw_payloads", False),
        )

    except Exception as e:
        from .security_redaction import redact_sensitive_text

        # P3 Fix: Re-raise to prevent lru_cache from caching the failure.
        # Returning a fallback here would poison the cache for the process lifetime.
        logger.warning(
            "[OBSERVABILITY] Failed to load config: %s. Re-raising to avoid cache poison.",
            redact_sensitive_text(e),
        )
        raise


# ============================================================================
# Event Emission Helpers
# ============================================================================


async def emit_event(
    coordinator: ModuleCoordinator | None,
    event_name: str,
    data: dict[str, Any],
) -> None:
    """Emit observability event if coordinator supports hooks.

    Contract: observability:Events:MUST:1, MUST:4, MUST:5

    Args:
        coordinator: Amplifier kernel coordinator (may be None).
        event_name: Event type from config (e.g., event_names.llm_request).
        data: Event payload dict.
    """
    if coordinator and hasattr(coordinator, "hooks"):
        try:
            await coordinator.hooks.emit(event_name, data)
        except Exception as e:
            from .security_redaction import redact_sensitive_text

            logger.warning(
                "[OBSERVABILITY] Failed to emit '%s': %s",
                event_name,
                redact_sensitive_text(e),
            )


# ============================================================================
# Lifecycle Context Manager
# ============================================================================


@dataclass
class LlmLifecycleContext:
    """Context for LLM request/response lifecycle."""

    config: ObservabilityConfig
    coordinator: ModuleCoordinator | None
    provider_name: str
    model: str
    start_time: float = field(default_factory=time.time)
    # P2 Fix #8: Track emission state for auto-cleanup on exception
    request_emitted: bool = field(default=False)
    response_emitted: bool = field(default=False)

    async def emit_request(
        self,
        *,
        message_count: int,
        tool_count: int,
        streaming: bool,
        timeout: float,
        raw_request: dict[str, Any] | None = None,
    ) -> None:
        """Emit llm:request event.

        Contract: observability:Events:MUST:2
        Contract: observability:Verbosity:MUST:1 — raw_payloads flag controls inclusion

        Args:
            message_count: Number of messages in request.
            tool_count: Number of tools available.
            streaming: Whether streaming is enabled.
            timeout: Request timeout in seconds.
            raw_request: Optional raw request payload (only included if raw_payloads=True).
        """
        payload: dict[str, Any] = {
            "provider": self.provider_name,
            "model": self.model,
            "message_count": message_count,
            "tool_count": tool_count,
            "streaming": streaming,
            "timeout": timeout,
        }

        # P3-14: Enforce raw_payloads policy
        if self.config.raw_payloads and raw_request:
            from .security_redaction import redact_sensitive_text

            # Redact before including (contract: Verbosity:MUST:2)
            payload["raw_request"] = redact_sensitive_text(str(raw_request))

        await emit_event(
            self.coordinator,
            self.config.event_names.llm_request,
            payload,
        )
        self.request_emitted = True  # P2 Fix: Track for llm_lifecycle cleanup

    async def emit_response_ok(
        self,
        *,
        usage_input: int,
        usage_output: int,
        finish_reason: str | None,
        content_blocks: int,
        tool_calls: int,
        raw_response: dict[str, Any] | None = None,
    ) -> None:
        """Emit llm:response event for successful completion.

        Contract: observability:Events:MUST:3
        Contract: observability:Verbosity:MUST:1 — raw_payloads flag controls inclusion
        """
        elapsed_ms = int((time.time() - self.start_time) * 1000)

        # Normalize finish_reason
        if not finish_reason:
            finish_reason = (
                self.config.finish_reasons.tool_use
                if tool_calls > 0
                else self.config.finish_reasons.end_turn
            )

        payload: dict[str, Any] = {
            "provider": self.provider_name,
            "model": self.model,
            "status": self.config.status.ok,
            "duration_ms": elapsed_ms,
            "usage": {
                "input": usage_input,
                "output": usage_output,
            },
            "finish_reason": finish_reason,
            "content_blocks": content_blocks,
            "tool_calls": tool_calls,
        }

        # P3-14: Enforce raw_payloads policy
        if self.config.raw_payloads and raw_response:
            from .security_redaction import redact_sensitive_text

            # Redact before including (contract: Verbosity:MUST:2)
            payload["raw_response"] = redact_sensitive_text(str(raw_response))

        await emit_event(
            self.coordinator,
            self.config.event_names.llm_response,
            payload,
        )
        self.response_emitted = True  # P2 Fix #8: Track emission for orphan prevention

    async def emit_response_error(
        self,
        *,
        error_type: str,
        error_message: str,
    ) -> None:
        """Emit llm:response event for error.

        Contract: observability:Events:MUST:3
        Contract: behaviors:Logging:MUST:4 - Sanitize error messages
        """
        from .security_redaction import redact_sensitive_text

        elapsed_ms = int((time.time() - self.start_time) * 1000)
        sanitized_message = redact_sensitive_text(error_message)

        await emit_event(
            self.coordinator,
            self.config.event_names.llm_response,
            {
                "provider": self.provider_name,
                "model": self.model,
                "status": self.config.status.error,
                "duration_ms": elapsed_ms,
                "error_type": error_type,
                "error_message": sanitized_message,
            },
        )
        self.response_emitted = True  # P2 Fix #8: Track emission for orphan prevention

    async def emit_retry(
        self,
        *,
        attempt: int,
        max_retries: int,
        delay: float,
        error_type: str,
        error_message: str,
    ) -> None:
        """Emit provider:retry event.

        Contract: provider-protocol:hooks:provider_retry:MUST:1
        Contract: behaviors:Logging:MUST:4 - Sanitize error messages
        """
        from .security_redaction import redact_sensitive_text

        sanitized_message = redact_sensitive_text(error_message)

        await emit_event(
            self.coordinator,
            self.config.event_names.provider_retry,
            {
                "provider": self.provider_name,
                "model": self.model,
                "attempt": attempt,
                "max_retries": max_retries,
                "delay": delay,
                "error_type": error_type,
                "error_message": sanitized_message,
            },
        )


@asynccontextmanager
async def llm_lifecycle(
    coordinator: ModuleCoordinator | None,
    model: str,
    config: ObservabilityConfig | None = None,
) -> AsyncIterator[LlmLifecycleContext]:
    """Context manager for LLM request/response lifecycle.

    Provides timing and event emission helpers.

    Usage:
        async with llm_lifecycle(coordinator, model) as ctx:
            await ctx.emit_request(...)
            # ... do completion ...
            await ctx.emit_response_ok(...)

    Args:
        coordinator: Amplifier kernel coordinator (may be None).
        model: Model identifier.
        config: Observability config (loaded if not provided).

    Yields:
        LlmLifecycleContext with emission helpers.
    """
    if config is None:
        config = load_observability_config()

    ctx = LlmLifecycleContext(
        config=config,
        coordinator=coordinator,
        provider_name=config.provider_name,
        model=model,
    )

    # P2 Fix #8: Wrap yield in try/finally to ensure llm:response is always emitted.
    # Contract: observability:Events:MUST:3 - MUST emit llm:response after llm:request.
    # Without this, exceptions escaping the context manager leave orphaned llm:request events.
    try:
        yield ctx
    except BaseException as exc:
        # Auto-emit error response if request was emitted but response wasn't
        # (prevents orphaned llm:request events that break trace correlation)
        if ctx.request_emitted and not ctx.response_emitted:
            await ctx.emit_response_error(
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
        raise
