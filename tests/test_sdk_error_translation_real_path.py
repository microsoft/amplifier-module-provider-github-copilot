"""Tests for real SDK path error translation.

Contract: contracts/error-hierarchy.md
  "The provider MUST translate SDK errors into kernel error types"

These tests verify that exceptions raised during the real SDK path are properly
translated to kernel error types via translate_sdk_error().
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from amplifier_module_provider_github_copilot.error_translation import (
    AuthenticationError,
    LLMError,
    LLMTimeoutError,
    ProviderUnavailableError,
)
from amplifier_module_provider_github_copilot.provider import (
    CompletionRequest,
    GitHubCopilotProvider,
)
from amplifier_module_provider_github_copilot.sdk_adapter.client import (
    CopilotClientWrapper,
)
from amplifier_module_provider_github_copilot.sdk_adapter.types import SessionHandle


# Stub classes for MagicMock spec= (SDK types not importable at runtime)
class _StubSDKEvent:
    """Stub matching SDK event structure for spec=."""

    type: str
    payload: Any


class _StubSDKSession:
    """Stub matching raw SDK session interface for spec=."""

    def on(self, handler: Callable[[Any], None]) -> Callable[[], None]: ...
    async def send(self, prompt: str, *, attachments: list[Any] | None = None) -> None: ...
    async def disconnect(self) -> None: ...


@pytest.fixture
def provider_with_mock_client() -> tuple[GitHubCopilotProvider, MagicMock]:
    """Create provider with a mocked _client attribute."""
    provider = GitHubCopilotProvider()
    mock_client = MagicMock(spec=CopilotClientWrapper)
    provider._client = mock_client  # type: ignore[reportPrivateUsage]  # Testing internal state
    return provider, mock_client


def create_mock_session_ctx(
    error_to_raise: BaseException | None = None,
    events: list[dict[str, Any]] | None = None,
) -> Any:
    """Create an async context manager that yields a mock session.

    Uses send() + on() pattern, not send_message().

    Args:
        error_to_raise: Exception to raise when send() is called
        events: List of events to deliver via on() handler (if no exception)
    """

    @asynccontextmanager
    async def session_ctx(
        model: str,
        tools: list[dict[str, Any]] | None = None,
        system_message: str | None = None,
    ) -> AsyncIterator[MagicMock]:
        mock_session = MagicMock(spec=SessionHandle)
        mock_session.disconnect = AsyncMock(spec=_StubSDKSession.disconnect)

        # Store handlers registered via on()
        handlers: list[Any] = []

        def mock_on(handler: Any) -> Callable[[], None]:
            handlers.append(handler)
            return lambda: None  # unsubscribe function

        mock_session.on = MagicMock(spec=SessionHandle.on, side_effect=mock_on)

        # send() raises error or delivers events
        async def mock_send(prompt: str, attachments: list[Any] | None = None) -> str:
            if error_to_raise:
                raise error_to_raise
            # Deliver events via handler callback
            for event_data in events or []:
                event = MagicMock(spec=_StubSDKEvent)
                for k, v in event_data.items():
                    setattr(event, k, v)
                event.__dict__.update(event_data)
                for handler in handlers:
                    handler(event)
            # Signal completion
            idle_event = MagicMock(spec=_StubSDKEvent)
            idle_event.type = "SESSION_IDLE"
            idle_event.__dict__["type"] = "SESSION_IDLE"
            for handler in handlers:
                handler(idle_event)
            return "message-id"

        mock_session.send = AsyncMock(spec=SessionHandle.send, side_effect=mock_send)
        yield mock_session

    return session_ctx


class TestRealSDKPathErrorTranslation:
    """Tests for Real SDK path error translation."""

    @pytest.mark.asyncio
    async def test_timeout_error_translated(
        self, provider_with_mock_client: tuple[GitHubCopilotProvider, MagicMock]
    ) -> None:
        """TimeoutError from SDK is translated to LLMTimeoutError.

        Contract: error-hierarchy.md - TimeoutError -> LLMTimeoutError
        """
        provider, mock_client = provider_with_mock_client
        mock_client.session = create_mock_session_ctx(error_to_raise=TimeoutError("SDK timeout"))

        request = CompletionRequest(prompt="test", model="gpt-4o")

        with pytest.raises(LLMTimeoutError):
            await provider.complete(request)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_auth_like_error_translated(
        self, provider_with_mock_client: tuple[GitHubCopilotProvider, MagicMock]
    ) -> None:
        """Auth-like errors from SDK are translated to AuthenticationError.

        Contract: error-hierarchy.md - PermissionError, 401/403 -> AuthenticationError
        """
        provider, mock_client = provider_with_mock_client
        mock_client.session = create_mock_session_ctx(
            error_to_raise=PermissionError("401 Unauthorized")
        )

        request = CompletionRequest(prompt="test", model="gpt-4o")

        with pytest.raises(AuthenticationError):
            await provider.complete(request)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_generic_error_translated_to_provider_unavailable(
        self, provider_with_mock_client: tuple[GitHubCopilotProvider, MagicMock]
    ) -> None:
        """Generic RuntimeError is translated to ProviderUnavailableError (fallback).

        Contract: error-hierarchy.md - Unknown errors -> ProviderUnavailableError
        """
        provider, mock_client = provider_with_mock_client
        mock_client.session = create_mock_session_ctx(
            error_to_raise=RuntimeError("Some SDK failure")
        )

        request = CompletionRequest(prompt="test", model="gpt-4o")

        with pytest.raises(ProviderUnavailableError):
            await provider.complete(request)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_llm_error_passes_through_unchanged(
        self, provider_with_mock_client: tuple[GitHubCopilotProvider, MagicMock]
    ) -> None:
        """Already-translated LLMError subclasses pass through without double-wrapping.

        Contract: LLMError subclasses must not be double-wrapped

        Note: Uses AuthenticationError (non-retryable) to avoid triggering retry loop.
        RateLimitError would cause 30+ second delays due to retry_after handling.
        """
        provider, mock_client = provider_with_mock_client
        # Create an already-translated error (use non-retryable to avoid retry delays)
        original_error = AuthenticationError("Auth failed", provider="github-copilot")
        mock_client.session = create_mock_session_ctx(error_to_raise=original_error)

        request = CompletionRequest(prompt="test", model="gpt-4o")

        with pytest.raises(AuthenticationError) as exc_info:
            await provider.complete(request)  # type: ignore[arg-type]

        # Verify it's the exact same error, not a wrapped one
        assert exc_info.value is original_error
        assert exc_info.value.provider == "github-copilot"

    @pytest.mark.asyncio
    async def test_connection_error_translated(
        self, provider_with_mock_client: tuple[GitHubCopilotProvider, MagicMock]
    ) -> None:
        """ConnectionError from SDK is translated appropriately.

        Contract: error-hierarchy.md - Connection errors -> appropriate mapping
        """
        provider, mock_client = provider_with_mock_client
        mock_client.session = create_mock_session_ctx(
            error_to_raise=ConnectionError("Connection refused")
        )

        request = CompletionRequest(prompt="test", model="gpt-4o")

        # Should translate to some LLMError subclass (likely ProviderUnavailableError)
        with pytest.raises(LLMError):
            await provider.complete(request)  # type: ignore[arg-type]


class TestF078ContextWindowFromYaml:
    """Tests for context_window comes from YAML config.

    Three-Medium: YAML is authoritative source.
    """

    def test_yaml_config_has_context_window(self) -> None:
        """YAML config must include context_window for budget calculation.

        Contract: provider-protocol.md - "MUST include defaults.context_window"
        Three-Medium: YAML is authoritative.
        """
        from amplifier_module_provider_github_copilot.config_loader import load_models_config

        config = load_models_config()

        # Key assertion: context_window must exist
        assert "context_window" in config.defaults
        # claude-opus-4.5 with 200000 context window
        assert config.defaults["context_window"] == 200000

    def test_yaml_config_budget_calculation_succeeds(self) -> None:
        """Budget calculation should work with YAML config values.

        Contract: provider-protocol:get_info:MUST:2
        Budget calculation uses YAML values
        Three-Medium: YAML is authoritative.
        """
        from amplifier_module_provider_github_copilot.config_loader import load_models_config

        config = load_models_config()

        # Budget calculation using YAML values
        context_window = config.defaults["context_window"]
        max_tokens = config.defaults.get("max_tokens", 4096)

        # This should not raise
        remaining_budget = context_window - max_tokens
        assert remaining_budget > 0
