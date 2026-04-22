"""Overloaded delay multiplier regression tests.

Contract: contracts/behaviors.md — behaviors:Retry:MUST:8

Ensures that errors marked `overloaded: true` in errors.yaml receive a
delay_multiplier sentinel, and that _calculate_retry_delay applies
config.overloaded_delay_multiplier to their computed backoff. Non-overloaded
errors and retry_after-bearing errors are unaffected.

Reference: drift-anthropic-ghcp-provider.md — gap-error-multiplier (MEDIUM)
"""

from __future__ import annotations

from typing import Any

import pytest

# =============================================================================
# Error translation — overloaded flag sets delay_multiplier sentinel
# =============================================================================


class TestOverloadedFlagInTranslation:
    """errors.yaml overloaded:true → translated error gets delay_multiplier > 1.0."""

    def test_rate_limit_error_has_delay_multiplier_sentinel(self) -> None:
        """RateLimitError from SDK translation carries delay_multiplier > 1.0.

        Contract: behaviors:Retry:MUST:8
        """
        from amplifier_module_provider_github_copilot.error_translation import (
            load_error_config,
            translate_sdk_error,
        )

        config = load_error_config()

        class FakeSdkRateLimitError(Exception):
            pass

        exc = FakeSdkRateLimitError("429 rate limit exceeded")
        translated = translate_sdk_error(exc, config, provider="github-copilot", model=None)

        assert getattr(translated, "delay_multiplier", 1.0) > 1.0, (
            f"RateLimitError should have delay_multiplier > 1.0, got: "
            f"{getattr(translated, 'delay_multiplier', 1.0)}"
        )

    def test_timeout_error_has_default_delay_multiplier(self) -> None:
        """LLMTimeoutError from SDK translation keeps delay_multiplier=1.0 (not overloaded).

        Contract: behaviors:Retry:MUST:8
        """
        from amplifier_module_provider_github_copilot.error_translation import (
            load_error_config,
            translate_sdk_error,
        )

        config = load_error_config()

        class FakeSdkTimeoutError(Exception):
            pass

        exc = FakeSdkTimeoutError("timeout waiting for session")
        translated = translate_sdk_error(exc, config, provider="github-copilot", model=None)

        assert getattr(translated, "delay_multiplier", 1.0) == pytest.approx(1.0), (
            f"LLMTimeoutError should have delay_multiplier=1.0, got: "
            f"{getattr(translated, 'delay_multiplier', 1.0)}"
        )


# =============================================================================
# _calculate_retry_delay — multiplier applied for overloaded errors
# =============================================================================


class TestCalculateRetryDelayMultiplier:
    """behaviors:Retry:MUST:8 — _calculate_retry_delay uses policy multiplier for overloaded."""

    def _make_provider(self, config: dict[str, Any] | None = None) -> Any:
        """Create a provider instance for testing."""
        from unittest.mock import AsyncMock, MagicMock  # noqa: F401

        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        coord = MagicMock()
        coord.hooks = MagicMock()
        coord.hooks.emit = AsyncMock()
        return GitHubCopilotProvider(config=config or {}, coordinator=coord)

    def test_overloaded_error_delay_is_multiplied(self) -> None:
        """Rate-limit error with delay_multiplier>1.0 → delay scaled by overloaded_delay_multiplier.

        Contract: behaviors:Retry:MUST:8

        Regression: if _calculate_retry_delay ignored delay_multiplier, the returned
        delay would be ~1000ms (base), not ~10000ms (10x).
        """
        from amplifier_core import RateLimitError

        from amplifier_module_provider_github_copilot.config._policy import RetryPolicy

        provider = self._make_provider()

        # Zero jitter for deterministic assertion
        config = RetryPolicy(
            base_delay_ms=1000,
            max_delay_ms=30000,
            jitter_factor=0.0,
            overloaded_delay_multiplier=10.0,
        )
        error = RateLimitError("429", provider="github-copilot")
        from amplifier_module_provider_github_copilot.error_translation import _OVERLOADED_SENTINEL

        error.delay_multiplier = _OVERLOADED_SENTINEL  # overloaded sentinel

        result = provider._calculate_retry_delay(error, attempt=0, config=config)

        assert result == pytest.approx(10_000.0), f"Expected 10x delay (10000ms), got {result}ms"

    def test_non_overloaded_error_delay_is_not_multiplied(self) -> None:
        """Timeout error with delay_multiplier=1.0 → delay is NOT scaled.

        Contract: behaviors:Retry:MUST:8
        """
        from amplifier_core import LLMTimeoutError

        from amplifier_module_provider_github_copilot.config._policy import RetryPolicy

        provider = self._make_provider()

        config = RetryPolicy(
            base_delay_ms=1000,
            max_delay_ms=30000,
            jitter_factor=0.0,
            overloaded_delay_multiplier=10.0,
        )
        error = LLMTimeoutError("timeout", provider="github-copilot")
        # delay_multiplier defaults to 1.0 — no sentinel set

        result = provider._calculate_retry_delay(error, attempt=0, config=config)

        assert result == pytest.approx(1_000.0), f"Expected base delay (1000ms), got {result}ms"

    def test_retry_after_supersedes_multiplier(self) -> None:
        """RateLimitError with retry_after=30 → 30000ms, ignoring overloaded multiplier.

        Contract: behaviors:Retry:MUST:8

        Regression: if multiplier check ran before retry_after check, this would
        return 300000ms (30000 * 10) instead of 30000ms.
        """
        from amplifier_core import RateLimitError

        from amplifier_module_provider_github_copilot.config._policy import RetryPolicy

        provider = self._make_provider()

        config = RetryPolicy(
            base_delay_ms=1000,
            max_delay_ms=30000,
            jitter_factor=0.0,
            overloaded_delay_multiplier=10.0,
        )
        error = RateLimitError("429", provider="github-copilot", retry_after=30.0)
        from amplifier_module_provider_github_copilot.error_translation import _OVERLOADED_SENTINEL

        error.delay_multiplier = _OVERLOADED_SENTINEL  # overloaded sentinel

        result = provider._calculate_retry_delay(error, attempt=0, config=config)

        assert result == pytest.approx(30_000.0), (
            f"Expected retry_after value (30000ms), got {result}ms — "
            f"retry_after must supersede overloaded multiplier"
        )

    def test_multiplied_delay_does_not_exceed_cap(self) -> None:
        """Multiplied delay is capped at max_delay_ms * overloaded_delay_multiplier.

        Contract: behaviors:Retry:MUST:8

        Regression: if cap was omitted, high attempt numbers could produce unbounded delays.
        """
        from amplifier_core import RateLimitError

        from amplifier_module_provider_github_copilot.config._policy import RetryPolicy

        provider = self._make_provider()

        # base=1000ms, max=5000ms. At attempt=10, backoff would hit max=5000ms.
        # After multiplier: 5000 * 10 = 50000ms. Cap = 5000 * 10 = 50000ms.
        # So result should be exactly 50000ms (cap).
        config = RetryPolicy(
            base_delay_ms=1000,
            max_delay_ms=5000,
            jitter_factor=0.0,
            overloaded_delay_multiplier=10.0,
        )
        error = RateLimitError("429", provider="github-copilot")
        from amplifier_module_provider_github_copilot.error_translation import _OVERLOADED_SENTINEL

        error.delay_multiplier = _OVERLOADED_SENTINEL  # overloaded sentinel

        result = provider._calculate_retry_delay(error, attempt=20, config=config)

        assert result == pytest.approx(50_000.0), (
            f"Expected cap at max_delay_ms * multiplier (50000ms), got {result}ms"
        )

    def test_overloaded_multiplier_user_override(self) -> None:
        """overloaded_delay_multiplier=5.0 in config → delay is 5x, not 10x.

        Contract: behaviors:Retry:MUST:8
        """
        from amplifier_core import RateLimitError

        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider(
            config={"overloaded_delay_multiplier": 5.0, "min_retry_delay": 0.0}
        )

        from amplifier_module_provider_github_copilot.config._policy import RetryPolicy

        config = RetryPolicy(
            base_delay_ms=1000,
            max_delay_ms=30000,
            jitter_factor=0.0,
            overloaded_delay_multiplier=5.0,
        )
        error = RateLimitError("429", provider="github-copilot")
        from amplifier_module_provider_github_copilot.error_translation import _OVERLOADED_SENTINEL

        error.delay_multiplier = _OVERLOADED_SENTINEL  # overloaded sentinel

        result = provider._calculate_retry_delay(error, attempt=0, config=config)

        assert result == pytest.approx(5_000.0), (
            f"Expected 5x delay (5000ms) with overloaded_delay_multiplier=5.0, got {result}ms"
        )

    def test_overloaded_multiplier_stored_in_retry_config(self) -> None:
        """config={'overloaded_delay_multiplier': 3.0} is reflected in _retry_config.

        Contract: behaviors:Retry:MUST:8
        """
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider(config={"overloaded_delay_multiplier": 3.0})
        assert provider._retry_config.overloaded_delay_multiplier == pytest.approx(3.0)

    def test_zero_overloaded_multiplier_clamped_to_one(self) -> None:
        """overloaded_delay_multiplier=0.0 is clamped to 1.0 (retry-storm guard).

        Contract: behaviors:Retry:MUST:8

        Regression: if clamping were absent, a zero multiplier would produce 0ms delays
        under overload — effectively disabling backoff and triggering a retry storm.
        """
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider(config={"overloaded_delay_multiplier": 0.0})
        assert provider._retry_config.overloaded_delay_multiplier == pytest.approx(1.0), (
            "overloaded_delay_multiplier=0.0 must be clamped to 1.0"
        )

    def test_negative_overloaded_multiplier_clamped_to_one(self) -> None:
        """overloaded_delay_multiplier=-5.0 is clamped to 1.0 (negative-delay guard).

        Contract: behaviors:Retry:MUST:8

        Regression: a negative multiplier produces a negative delay_ms value that
        asyncio.sleep() treats as immediate — collapsing all backoff.
        """
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider(config={"overloaded_delay_multiplier": -5.0})
        assert provider._retry_config.overloaded_delay_multiplier == pytest.approx(1.0), (
            "overloaded_delay_multiplier=-5.0 must be clamped to 1.0"
        )

    def test_retry_policy_rejects_invalid_multiplier_directly(self) -> None:
        """RetryPolicy(overloaded_delay_multiplier=0.0) raises ValueError at construction.

        Contract: behaviors:Retry:MUST:8

        Regression: without __post_init__ validation, programmatic construction of
        RetryPolicy bypasses the _build_retry_config guard, silently creating an
        invalid config that would cause 0ms backoff under overload.
        """
        import pytest as _pytest

        from amplifier_module_provider_github_copilot.config._policy import RetryPolicy

        with _pytest.raises(ValueError, match="overloaded_delay_multiplier"):
            RetryPolicy(overloaded_delay_multiplier=0.0)

    def test_retry_policy_rejects_negative_multiplier_directly(self) -> None:
        """RetryPolicy(overloaded_delay_multiplier=-1.0) raises ValueError at construction.

        Contract: behaviors:Retry:MUST:8
        """
        import pytest as _pytest

        from amplifier_module_provider_github_copilot.config._policy import RetryPolicy

        with _pytest.raises(ValueError, match="overloaded_delay_multiplier"):
            RetryPolicy(overloaded_delay_multiplier=-1.0)

    def test_retry_policy_accepts_valid_multiplier_at_boundary(self) -> None:
        """RetryPolicy(overloaded_delay_multiplier=1.0) is accepted (boundary value).

        Contract: behaviors:Retry:MUST:8
        """
        from amplifier_module_provider_github_copilot.config._policy import RetryPolicy

        config = RetryPolicy(overloaded_delay_multiplier=1.0)
        assert config.overloaded_delay_multiplier == pytest.approx(1.0)
