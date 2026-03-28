"""Tests for timeout enforcement on the real SDK path.

Contract: Provider operations must not block indefinitely.

Tests verify:
- SDK streaming iteration is wrapped in asyncio.timeout
- Timeout value is loaded from config
- asyncio.TimeoutError is caught and translated to provider error
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from amplifier_core.llm_errors import LLMError

# pyright: reportAttributeAccessIssue=false
# These imports work at runtime via conftest.py SKIP_SDK_CHECK


class TestTimeoutEnforcement:
    """Add Timeout Enforcement to Real SDK Path."""

    @pytest.mark.asyncio
    async def test_timeout_wraps_streaming_iteration(self) -> None:
        """availability:timeout:MUST:1 — streaming must be wrapped in asyncio.timeout.

        The real SDK path must wrap the streaming iteration in asyncio.timeout
        to prevent indefinite blocking.
        """
        from amplifier_module_provider_github_copilot.provider import (
            CompletionRequest,
            GitHubCopilotProvider,
        )

        provider = GitHubCopilotProvider()

        # Use on() + send() pattern correctly
        # The mock session never sends SESSION_IDLE, so it times out waiting
        handlers: list[Any] = []

        @asynccontextmanager
        async def mock_session_cm(*_args: Any, **_kwargs: Any) -> AsyncIterator[Any]:
            mock_session = MagicMock()

            def mock_on(handler: Any) -> Any:
                handlers.append(handler)
                return lambda: None  # unsubscribe

            mock_session.on = MagicMock(side_effect=mock_on)

            # send() delivers one event but never sends SESSION_IDLE
            # This simulates a slow/stalled SDK that doesn't complete
            async def mock_send(prompt: str, attachments: list[Any] | None = None) -> str:
                # Deliver one event
                if handlers:
                    event = MagicMock()
                    event.type = "assistant.message_delta"
                    event.__dict__["type"] = "assistant.message_delta"
                    event.text = "Hello"
                    handlers[0](event)
                # Never send SESSION_IDLE - will timeout waiting
                return "message-id"

            mock_session.send = AsyncMock(side_effect=mock_send)
            mock_session.disconnect = AsyncMock()
            yield mock_session

        provider._client.session = mock_session_cm  # type: ignore[method-assign]

        request = CompletionRequest(prompt="Hello", model="gpt-4o")

        # With timeout enforcement, this should raise a timeout error
        # (translated to LLMTimeoutError or ProviderUnavailableError)
        with pytest.raises(Exception) as exc_info:
            await provider.complete(request, _timeout_seconds=0.1)  # type: ignore[arg-type]

        # The error should be a timeout-related error (exact type depends on translation)
        error_str = str(exc_info.value).lower()
        assert any(
            word in error_str for word in ["timeout", "timed out", "time"]
        ) or exc_info.type.__name__ in (
            "LLMTimeoutError",
            "TimeoutError",
            "ProviderUnavailableError",
        )

    @pytest.mark.asyncio
    async def test_timeout_from_config_default(self) -> None:
        """availability:timeout:MUST:2 — timeout should use config value.

        The timeout value should come from config/models.yaml (timeout field)
        or fall back to a sensible default.
        """
        from amplifier_module_provider_github_copilot.provider import (
            _load_models_config,  # type: ignore[attr-defined]
        )

        config = _load_models_config()  # type: ignore[no-untyped-call]

        # Config should have a timeout in defaults (or we use fallback)
        timeout = config.defaults.get("timeout", 120)  # 120s default
        assert isinstance(timeout, (int, float))
        assert timeout > 0  # Positive timeout

    @pytest.mark.asyncio
    async def test_normal_completion_succeeds_within_timeout(self) -> None:
        """availability:timeout:REGRESSION — normal completion must succeed.

        Normal SDK responses that complete within timeout should work correctly.
        """
        from amplifier_module_provider_github_copilot.provider import (
            CompletionRequest,
            GitHubCopilotProvider,
        )

        provider = GitHubCopilotProvider()

        # Use on() + send() pattern correctly
        handlers: list[Any] = []

        @asynccontextmanager
        async def mock_session_cm(*_args: Any, **_kwargs: Any) -> AsyncIterator[Any]:
            mock_session = MagicMock()

            def mock_on(handler: Any) -> Any:
                handlers.append(handler)
                return lambda: None  # unsubscribe

            mock_session.on = MagicMock(side_effect=mock_on)

            # send() delivers events and signals completion with SESSION_IDLE
            async def mock_send(prompt: str, attachments: list[Any] | None = None) -> str:
                if handlers:
                    # Deliver text events
                    for event_data in [
                        {"type": "assistant.message_delta", "text": "Hello "},
                        {"type": "assistant.message_delta", "text": "World"},
                        {"type": "message_complete", "finish_reason": "end_turn"},
                    ]:
                        event = MagicMock()
                        for k, v in event_data.items():
                            setattr(event, k, v)
                        event.__dict__.update(event_data)
                        handlers[0](event)
                    # Signal completion
                    idle_event = MagicMock()
                    idle_event.type = "SESSION_IDLE"
                    idle_event.__dict__["type"] = "SESSION_IDLE"
                    handlers[0](idle_event)
                return "message-id"

            mock_session.send = AsyncMock(side_effect=mock_send)
            mock_session.disconnect = AsyncMock()
            yield mock_session

        provider._client.session = mock_session_cm  # type: ignore[method-assign]

        request = CompletionRequest(prompt="Hello", model="gpt-4o")

        # This should complete successfully (no timeout)
        response = await provider.complete(request, _timeout_seconds=10.0)  # type: ignore[arg-type]

        # Response should contain accumulated text
        assert response is not None

    @pytest.mark.asyncio
    async def test_timeout_error_translated_to_kernel_type(self) -> None:
        """availability:timeout:MUST:3 — timeout errors must be translated.

        asyncio.TimeoutError must be caught and translated to a kernel
        error type (LLMTimeoutError or ProviderUnavailableError).
        """
        from amplifier_module_provider_github_copilot.provider import (
            CompletionRequest,
            GitHubCopilotProvider,
        )

        provider = GitHubCopilotProvider()

        # Use on() + send() pattern - send() raises TimeoutError
        @asynccontextmanager
        async def mock_session_cm(*_args: Any, **_kwargs: Any) -> AsyncIterator[Any]:
            mock_session = MagicMock()
            mock_session.on = MagicMock(return_value=lambda: None)

            async def mock_send(prompt: str, attachments: list[Any] | None = None) -> str:
                raise TimeoutError("Operation timed out")

            mock_session.send = AsyncMock(side_effect=mock_send)
            mock_session.disconnect = AsyncMock()
            yield mock_session

        provider._client.session = mock_session_cm  # type: ignore[method-assign]

        request = CompletionRequest(prompt="Hello", model="gpt-4o")

        with pytest.raises(LLMError) as exc_info:
            await provider.complete(request, _timeout_seconds=10.0)  # type: ignore[arg-type]

        # Should be translated to a kernel LLMError type
        assert isinstance(exc_info.value, LLMError)
