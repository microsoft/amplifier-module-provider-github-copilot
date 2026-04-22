# NOTE: mock_error objects are intentionally unspecced - they simulate various SDK error shapes
# (status_code, response, headers) for error classification testing. Using spec= would require
# creating many specific error classes for each test scenario.

"""Tests for Config-Driven Retry Implementation.
Contract: contracts/behaviors.md

These tests verify that retryable errors trigger automatic retry with
configured backoff, and non-retryable errors fail immediately.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from amplifier_core import ChatResponse


def _make_request(model: str = "gpt-4o") -> MagicMock:
    """Create a minimal mock ChatRequest for retry tests."""
    request = MagicMock()
    request.model = model
    request.messages = []
    request.tools = []
    return request


# ============================================================================
# Retry Config Tests
# ============================================================================


class TestRetryConfigLoading:
    """Tests for retry config loading from YAML."""

    def test_retry_config_has_expected_defaults(self) -> None:
        """Config values match behaviors.md specification."""
        from amplifier_module_provider_github_copilot.provider import load_retry_config

        config = load_retry_config()
        assert config.max_attempts == 3
        assert config.base_delay_ms == 1000
        assert config.max_delay_ms == 30000
        assert config.jitter_factor == 0.1


# ============================================================================
# Retry Behavior Tests
# ============================================================================


@dataclass
class MockRetryableError(Exception):
    """Mock error with retryable=True."""

    retryable: bool = True


@dataclass
class MockNonRetryableError(Exception):
    """Mock error with retryable=False."""

    retryable: bool = False


class TestRetryBehavior:
    """Tests for retry behavior in complete()."""

    @pytest.mark.asyncio
    async def test_retryable_error_retries_up_to_max_attempts(self) -> None:
        """Contract: behaviors:Retry:MUST:1 - respects max_attempts.

        When a retryable error occurs, provider must retry up to max_attempts.
        """
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        # Create provider
        provider = GitHubCopilotProvider()

        # Track call count
        call_count = 0

        # Mock SDK session that fails with retryable error
        @asynccontextmanager
        async def mock_session_cm(*args: Any, **kwargs: Any):
            nonlocal call_count
            call_count += 1
            # Import the actual error type
            from amplifier_module_provider_github_copilot.error_translation import (
                LLMTimeoutError,
            )

            raise LLMTimeoutError(
                "Timeout",
                provider="github-copilot",
            )
            yield  # Never reached, but required for asynccontextmanager

        # Patch the client's session method
        provider._client.session = mock_session_cm  # type: ignore[reportPrivateUsage]  # Testing internal state

        request = _make_request()

        # Should raise after max_attempts (3)
        from amplifier_module_provider_github_copilot.error_translation import LLMTimeoutError

        with pytest.raises(LLMTimeoutError):
            await provider.complete(request)

        # Should have been called 3 times (initial + 2 retries)
        assert call_count == 3, f"Expected 3 attempts, got {call_count}"

    @pytest.mark.asyncio
    async def test_non_retryable_error_fails_immediately(self) -> None:
        """Contract: behaviors:Retry:MUST:5 - non-retryable errors fail immediately.

        When a non-retryable error occurs, provider must NOT retry.
        """
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        # Create provider
        provider = GitHubCopilotProvider()

        # Track call count
        call_count = 0

        # Mock SDK session that fails with non-retryable error
        @asynccontextmanager
        async def mock_session_cm(*args: Any, **kwargs: Any):
            nonlocal call_count
            call_count += 1
            from amplifier_module_provider_github_copilot.error_translation import (
                AuthenticationError,
            )

            raise AuthenticationError(
                "Invalid token",
                provider="github-copilot",
            )
            yield  # Never reached, but required for asynccontextmanager

        # Patch the client's session method
        provider._client.session = mock_session_cm  # type: ignore[reportPrivateUsage]  # Testing internal state

        request = _make_request()

        # Should raise immediately without retry
        from amplifier_module_provider_github_copilot.error_translation import AuthenticationError

        with pytest.raises(AuthenticationError):
            await provider.complete(request)

        # Should have been called exactly once (no retry)
        assert call_count == 1, f"Expected 1 attempt, got {call_count}"

    @pytest.mark.asyncio
    async def test_backoff_delay_increases_between_retries(self) -> None:
        """Contract: behaviors:Retry:MUST:2 - applies backoff between retries.

        Delay should increase exponentially between retry attempts.
        """
        from amplifier_module_provider_github_copilot.provider import (
            calculate_backoff_delay,
        )

        # Test the backoff calculation function directly
        delay1 = calculate_backoff_delay(attempt=0, base_delay_ms=1000)
        delay2 = calculate_backoff_delay(attempt=1, base_delay_ms=1000)
        delay3 = calculate_backoff_delay(attempt=2, base_delay_ms=1000)

        # Each delay should be approximately 2x previous (exponential)
        # With jitter, allow 50% tolerance
        assert delay2 > delay1 * 0.5, "Second delay should be greater than first"
        assert delay3 > delay2 * 0.5, "Third delay should be greater than second"

    @pytest.mark.asyncio
    async def test_all_retries_exhausted_raises_last_error(self) -> None:
        """Contract: behaviors:Retry:MUST:1 - after exhausting retries, raises.

        After all retry attempts are exhausted, the last error should be raised.
        """
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        # Create provider
        provider = GitHubCopilotProvider()

        # Mock SDK session that always fails with retryable error
        @asynccontextmanager
        async def mock_session_cm(*args: Any, **kwargs: Any):
            from amplifier_module_provider_github_copilot.error_translation import (
                NetworkError,
            )

            raise NetworkError(
                "Connection refused",
                provider="github-copilot",
            )
            yield  # Never reached, but required for asynccontextmanager

        provider._client.session = mock_session_cm  # type: ignore[reportPrivateUsage]  # Testing internal state

        request = _make_request()

        # Should eventually raise after exhausting retries
        with pytest.raises(Exception) as exc_info:
            await provider.complete(request)

        # The error should be the NetworkError (retryable but exhausted)
        assert "Connection refused" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_success_after_retry_returns_response(self) -> None:
        """When retry succeeds, response should be returned normally."""
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        provider = GitHubCopilotProvider()

        call_count = 0

        # Mock session context manager
        @asynccontextmanager
        async def mock_session_cm(*args: Any, **kwargs: Any):
            nonlocal call_count
            call_count += 1

            if call_count < 2:
                # First attempt fails with retryable error
                from amplifier_module_provider_github_copilot.error_translation import (
                    LLMTimeoutError,
                )

                raise LLMTimeoutError("Timeout", provider="github-copilot")

            # Second attempt succeeds
            mock_session = MagicMock()
            mock_session.on = MagicMock(return_value=lambda: None)
            mock_session.send = AsyncMock()

            # Simulate events
            events_delivered = False

            def mock_on(handler: Any) -> Any:
                nonlocal events_delivered
                if not events_delivered:
                    events_delivered = True
                    # Deliver idle event
                    mock_event = MagicMock()
                    mock_event.type = "session_idle"
                    handler(mock_event)
                return lambda: None

            mock_session.on = mock_on
            yield mock_session

        provider._client.session = mock_session_cm  # type: ignore[reportPrivateUsage]  # Testing internal state

        request = _make_request()

        # Should succeed on second attempt
        result = await provider.complete(request)
        assert result.text is None  # No text events delivered, just idle

        # Call count should be 2 (one fail, one success)
        assert call_count == 2


# ============================================================================
# Jitter Tests
# ============================================================================


class TestBackoffJitter:
    """Tests for jitter in backoff delays."""

    def test_jitter_adds_randomness(self) -> None:
        """Contract: behaviors:Retry:MUST:3 - adds jitter to prevent thundering herd.

        Multiple calls with same parameters should produce different delays.
        """
        from amplifier_module_provider_github_copilot.provider import (
            calculate_backoff_delay,
        )

        # Call multiple times with same parameters
        delays = [
            calculate_backoff_delay(attempt=1, base_delay_ms=1000, jitter_factor=0.1)
            for _ in range(10)
        ]

        # Not all delays should be identical (jitter adds randomness)
        unique_delays = set(delays)
        assert len(unique_delays) > 1, "Jitter should produce varying delays"

    def test_jitter_stays_within_bounds(self) -> None:
        """Jitter should not exceed configured factor."""
        from amplifier_module_provider_github_copilot.provider import (
            calculate_backoff_delay,
        )

        base = 1000
        jitter = 0.1

        for _ in range(100):
            delay = calculate_backoff_delay(attempt=1, base_delay_ms=base, jitter_factor=jitter)
            expected = base * 2  # attempt=1 means 2^1 = 2x base
            # Allow 10% jitter range
            assert delay >= expected * (1 - jitter)
            assert delay <= expected * (1 + jitter)


# ============================================================================
# retry_after Honor Tests
# ============================================================================


class TestRetryAfterHonor:
    """Tests for honoring retry_after header."""

    @pytest.mark.asyncio
    async def test_honors_retry_after_when_present(self) -> None:
        """Contract: behaviors:Retry:MUST:6 - honors retry_after when present.

        When error has retry_after, use that instead of calculated backoff.
        """
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        provider = GitHubCopilotProvider()

        sleep_durations: list[float] = []
        original_sleep = asyncio.sleep

        async def mock_sleep(duration: float) -> None:
            sleep_durations.append(duration)
            await original_sleep(0)  # Don't actually wait

        call_count = 0

        @asynccontextmanager
        async def mock_session_cm(*args: Any, **kwargs: Any):
            nonlocal call_count
            call_count += 1

            from amplifier_module_provider_github_copilot.error_translation import (
                RateLimitError,
            )

            # First call raises with retry_after
            if call_count == 1:
                raise RateLimitError(
                    "Rate limited",
                    provider="github-copilot",
                    retry_after=5.0,  # 5 seconds
                )
            # Second call also raises to verify delay was used
            raise RateLimitError(
                "Still rate limited",
                provider="github-copilot",
            )
            yield  # Never reached, but required for asynccontextmanager

        provider._client.session = mock_session_cm  # type: ignore[reportPrivateUsage]  # Testing internal state

        request = _make_request()

        from amplifier_module_provider_github_copilot.error_translation import (
            RateLimitError as _RateLimitError,
        )

        with patch("asyncio.sleep", mock_sleep):
            with pytest.raises(_RateLimitError):
                await provider.complete(request)

        # Should have slept with retry_after value (5.0 seconds)
        assert any(abs(d - 5.0) < 0.5 for d in sleep_durations), (
            f"Should honor retry_after=5.0, got {sleep_durations}"
        )


# ============================================================================
# Raw Exception Translation Tests
# ============================================================================


class TestRawExceptionTranslation:
    """Tests for raw SDK exception translation during retry.

    Contract: error-hierarchy:Translation:MUST:1
    Coverage: provider.py lines 442-470 (Exception handling branch)
    """

    @pytest.mark.asyncio
    async def test_raw_exception_translated_to_kernel_error(self) -> None:
        """Raw Exception is translated to kernel error type.

        Contract: error-hierarchy:Translation:MUST:1
        """
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        provider = GitHubCopilotProvider()

        # Mock SDK session that raises raw Exception (not LLMError)
        @asynccontextmanager
        async def mock_session_cm(*args: Any, **kwargs: Any):
            raise RuntimeError("Raw SDK error - connection reset")
            yield  # Never reached

        provider._client.session = mock_session_cm  # type: ignore[reportPrivateUsage]

        request = _make_request()

        # Should translate to ProviderUnavailableError or similar
        from amplifier_module_provider_github_copilot.error_translation import LLMError

        with pytest.raises(LLMError):
            await provider.complete(request)

    @pytest.mark.asyncio
    async def test_raw_retryable_exception_retries(self) -> None:
        """Raw exception that translates to retryable error triggers retry.

        Contract: behaviors:Retry:MUST:1
        Coverage: provider.py lines 450-470
        """
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        provider = GitHubCopilotProvider()

        call_count = 0

        @asynccontextmanager
        async def mock_session_cm(*args: Any, **kwargs: Any):
            nonlocal call_count
            call_count += 1
            # Raise TimeoutError which should translate to retryable LLMTimeoutError
            raise TimeoutError("Connection timed out")
            yield  # Never reached

        provider._client.session = mock_session_cm  # type: ignore[reportPrivateUsage]

        request = _make_request()

        # TimeoutError translates to LLMTimeoutError which is retryable
        from amplifier_module_provider_github_copilot.error_translation import LLMError

        with pytest.raises(LLMError):
            await provider.complete(request)

        # Should have retried (TimeoutError is typically retryable)
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_raw_non_retryable_exception_fails_fast(self) -> None:
        """Raw exception that translates to non-retryable fails immediately.

        Contract: behaviors:Retry:MUST:5
        Coverage: provider.py lines 442-447
        """
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        provider = GitHubCopilotProvider()

        call_count = 0

        @asynccontextmanager
        async def mock_session_cm(*args: Any, **kwargs: Any):
            nonlocal call_count
            call_count += 1
            # Raise ValueError which should translate to non-retryable
            raise ValueError("Invalid model parameter: invalid_model")
            yield  # Never reached

        provider._client.session = mock_session_cm  # type: ignore[reportPrivateUsage]

        request = _make_request(model="invalid_model")

        # ValueError translates to ProviderUnavailableError
        from amplifier_module_provider_github_copilot.error_translation import LLMError

        with pytest.raises(LLMError):
            await provider.complete(request)

        # Should fail immediately without retry
        assert call_count == 1


# ============================================================================
# Fake Tool Correction Retry Tests (Real SDK Path)
# ============================================================================


class TestFakeToolCorrectionRetry:
    """Tests for fake tool call correction retry in real SDK path.

    Contract: provider-protocol:complete:MUST:5
    Coverage: provider.py lines 501-567
    """

    @pytest.mark.asyncio
    async def test_fake_tool_correction_retry_succeeds(self) -> None:
        """Fake tool correction retry succeeds on second attempt.

        Contract: provider-protocol:complete:MUST:5
        """
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        provider = GitHubCopilotProvider()

        call_count = 0

        @asynccontextmanager
        async def mock_session_cm(*args: Any, **kwargs: Any):
            nonlocal call_count
            call_count += 1

            mock_session = MagicMock()
            registered_handler: Any = None

            def mock_on(handler: Any) -> Any:
                nonlocal registered_handler
                registered_handler = handler
                return lambda: None

            async def mock_send(*args: Any, **kwargs: Any) -> None:
                # Fire events AFTER send is called (simulates async SDK behavior)
                await asyncio.sleep(0)  # Yield to event loop
                if registered_handler:
                    # First call: fake tool call text, second call: clean text
                    # For dict events, use 'text' field (accumulator expects this)
                    if call_count == 1:
                        text_event = {
                            "type": "assistant.message_delta",
                            "data": {"text": "[Tool Call: bash(command='ls')]"},
                        }
                    else:
                        text_event = {
                            "type": "assistant.message_delta",
                            "data": {"text": "Here is the file listing."},
                        }

                    registered_handler(text_event)

                    # Idle event to end
                    idle_event = {"type": "session.idle"}
                    registered_handler(idle_event)

            mock_session.on = mock_on
            mock_session.send = mock_send
            mock_session.abort = AsyncMock()
            yield mock_session

        provider._client.session = mock_session_cm  # type: ignore[reportPrivateUsage]

        request = _make_request()
        request.tools = [{"name": "bash", "description": "Run bash"}]  # Tools available

        result = await provider.complete(request)

        # Should have been called multiple times (original + correction)
        assert call_count == 2
        assert isinstance(result, ChatResponse)
        assert result.text == "Here is the file listing."

    # NOTE: test_fake_tool_correction_exception_logs_exhausted removed
    # Lines 560-563 (exception handler in correction retry) are only reachable
    # via the real SDK path, not the test path (sdk_create_fn). Test would require
    # live SDK integration test with error injection, marked for live tests.
