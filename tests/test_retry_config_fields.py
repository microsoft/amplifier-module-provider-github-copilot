"""Configurable retry parameter tests.

Contract: contracts/behaviors.md — behaviors:Retry:MUST:7

Ensures provider runtime config keys (max_retries, min_retry_delay, max_retry_delay,
retry_jitter) correctly override the frozen policy defaults from _policy.py, with
proper unit conversion (seconds→ms) and the max_retries→max_attempts (+1) semantics.

Reference: drift-anthropic-ghcp-provider.md — gap-configurable-retry (HIGH)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# =============================================================================
# Default values — frozen policy unchanged when config dict is empty
# =============================================================================


class TestRetryConfigFieldDefaults:
    """behaviors:Retry:MUST:7 — defaults match _policy.py when config is empty."""

    def test_default_max_attempts(self) -> None:
        """max_attempts defaults to 3 (RetryPolicy default)."""
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider(config={})
        assert provider._retry_config.max_attempts == 3

    def test_default_base_delay_ms(self) -> None:
        """base_delay_ms defaults to 1000 ms."""
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider(config={})
        assert provider._retry_config.base_delay_ms == 1000

    def test_default_max_delay_ms(self) -> None:
        """max_delay_ms defaults to 30000 ms."""
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider(config={})
        assert provider._retry_config.max_delay_ms == 30000

    def test_default_jitter_factor(self) -> None:
        """jitter_factor defaults to 0.1."""
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider(config={})
        assert provider._retry_config.jitter_factor == pytest.approx(0.1)


# =============================================================================
# Per-param overrides — each key overrides exactly its corresponding field
# =============================================================================


class TestRetryConfigOverrides:
    """behaviors:Retry:MUST:7 — each config key overrides its corresponding field."""

    def test_max_retries_2_gives_max_attempts_3(self) -> None:
        """max_retries=2 → max_attempts=3 (retries + 1 = total attempts).

        Contract: behaviors:Retry:MUST:7
        """
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider(config={"max_retries": 2})
        assert provider._retry_config.max_attempts == 3

    def test_max_retries_0_gives_max_attempts_1(self) -> None:
        """max_retries=0 → max_attempts=1 (no retry, single attempt, fail fast).

        Contract: behaviors:Retry:MUST:7
        """
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider(config={"max_retries": 0})
        assert provider._retry_config.max_attempts == 1

    def test_max_retries_5_gives_max_attempts_6(self) -> None:
        """max_retries=5 → max_attempts=6.

        Contract: behaviors:Retry:MUST:7
        """
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider(config={"max_retries": 5})
        assert provider._retry_config.max_attempts == 6

    def test_max_retries_string_numeric_parsed(self) -> None:
        """max_retries="3" (string from YAML) → max_attempts=4.

        Contract: behaviors:Retry:MUST:7
        """
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider(config={"max_retries": "3"})
        assert provider._retry_config.max_attempts == 4

    def test_min_retry_delay_seconds_to_ms(self) -> None:
        """min_retry_delay=2.0 (seconds) → base_delay_ms=2000.

        Contract: behaviors:Retry:MUST:7
        """
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider(config={"min_retry_delay": 2.0})
        assert provider._retry_config.base_delay_ms == 2000

    def test_max_retry_delay_seconds_to_ms(self) -> None:
        """max_retry_delay=60.0 (seconds) → max_delay_ms=60000.

        Contract: behaviors:Retry:MUST:7
        """
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider(config={"max_retry_delay": 60.0})
        assert provider._retry_config.max_delay_ms == 60000

    def test_retry_jitter_float_stored(self) -> None:
        """retry_jitter=0.2 → jitter_factor=0.2.

        Contract: behaviors:Retry:MUST:7
        """
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider(config={"retry_jitter": 0.2})
        assert provider._retry_config.jitter_factor == pytest.approx(0.2)

    def test_retry_jitter_zero_disables_jitter(self) -> None:
        """retry_jitter=0.0 → jitter_factor=0.0 (jitter disabled).

        Contract: behaviors:Retry:MUST:7
        """
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider(config={"retry_jitter": 0.0})
        assert provider._retry_config.jitter_factor == 0.0

    def test_all_params_combined(self) -> None:
        """All four retry params set simultaneously are all applied.

        Contract: behaviors:Retry:MUST:7
        """
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider(
            config={
                "max_retries": 1,
                "min_retry_delay": 0.5,
                "max_retry_delay": 10.0,
                "retry_jitter": 0.05,
            }
        )
        assert provider._retry_config.max_attempts == 2
        assert provider._retry_config.base_delay_ms == 500
        assert provider._retry_config.max_delay_ms == 10000
        assert provider._retry_config.jitter_factor == pytest.approx(0.05)


# =============================================================================
# Invalid inputs — fallback to defaults without crash
# =============================================================================


class TestRetryConfigInvalidInputs:
    """Invalid config values fall back to policy defaults gracefully."""

    def test_max_retries_string_bad_falls_back(self) -> None:
        """max_retries="bad" → falls back to default max_attempts=3."""
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider(config={"max_retries": "bad"})
        assert provider._retry_config.max_attempts == 3

    def test_max_retries_negative_clamped_to_zero_retries(self) -> None:
        """max_retries=-1 → clamped to 0 retries → max_attempts=1.

        Negative retry counts are nonsensical; clamp to 0 (single attempt).
        """
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider(config={"max_retries": -1})
        assert provider._retry_config.max_attempts == 1

    def test_min_retry_delay_string_bad_falls_back(self) -> None:
        """min_retry_delay="bad" → falls back to default base_delay_ms=1000."""
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider(config={"min_retry_delay": "bad"})
        assert provider._retry_config.base_delay_ms == 1000


# =============================================================================
# Behavioral tests — complete() uses self._retry_config, not load_retry_config()
# =============================================================================


@pytest.fixture()
def mock_coordinator() -> MagicMock:
    """Coordinator with mocked hooks.emit."""
    coord = MagicMock()
    coord.hooks = MagicMock()
    coord.hooks.emit = AsyncMock()
    return coord


@pytest.fixture()
def sample_request() -> MagicMock:
    """Minimal ChatRequest mock."""
    req = MagicMock()
    req.messages = [MagicMock(role="user", content="Hello")]
    req.model = "gpt-4o"
    return req


class TestRetryConfigBehavioral:
    """behaviors:Retry:MUST:7 — complete() uses self._retry_config.

    Each test would turn red if complete() still called load_retry_config()
    (default max_attempts=3) instead of self._retry_config.
    """

    @pytest.mark.asyncio
    async def test_max_retries_1_makes_exactly_2_attempts(
        self,
        mock_coordinator: MagicMock,
        sample_request: MagicMock,
    ) -> None:
        """max_retries=1 → provider makes exactly 2 total attempts.

        Contract: behaviors:Retry:MUST:7

        Regression: if complete() used load_retry_config() (max_attempts=3),
        this test would observe 3 attempts, not 2.
        """
        from amplifier_module_provider_github_copilot.error_translation import LLMTimeoutError
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from amplifier_module_provider_github_copilot.streaming import StreamingAccumulator

        provider = GitHubCopilotProvider(
            config={"max_retries": 1, "min_retry_delay": 0.0, "use_streaming": False},
            coordinator=mock_coordinator,
        )
        call_count = 0

        async def _mock_execute(
            *args: Any, accumulator: StreamingAccumulator, **kwargs: Any
        ) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise LLMTimeoutError("timeout", provider="github-copilot")
            # Second call succeeds — accumulator left empty (to_chat_response handles it)

        provider._execute_sdk_completion = _mock_execute  # type: ignore[method-assign]

        await provider.complete(sample_request, model="gpt-4o")  # type: ignore[arg-type]

        assert call_count == 2, f"Expected exactly 2 attempts (max_retries=1), got {call_count}"

    @pytest.mark.asyncio
    async def test_max_retries_0_makes_exactly_1_attempt(
        self,
        mock_coordinator: MagicMock,
        sample_request: MagicMock,
    ) -> None:
        """max_retries=0 → provider makes exactly 1 attempt, raises immediately.

        Contract: behaviors:Retry:MUST:7

        Regression: if complete() used load_retry_config() (max_attempts=3),
        this test would observe 3 attempts instead of 1.
        """
        from amplifier_module_provider_github_copilot.error_translation import LLMTimeoutError
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from amplifier_module_provider_github_copilot.streaming import StreamingAccumulator

        provider = GitHubCopilotProvider(
            config={"max_retries": 0, "use_streaming": False},
            coordinator=mock_coordinator,
        )
        call_count = 0

        async def _mock_execute(
            *args: Any, accumulator: StreamingAccumulator, **kwargs: Any
        ) -> None:
            nonlocal call_count
            call_count += 1
            raise LLMTimeoutError("timeout", provider="github-copilot")

        provider._execute_sdk_completion = _mock_execute  # type: ignore[method-assign]

        with pytest.raises(LLMTimeoutError):
            await provider.complete(sample_request, model="gpt-4o")  # type: ignore[arg-type]

        assert call_count == 1, f"Expected exactly 1 attempt (max_retries=0), got {call_count}"
