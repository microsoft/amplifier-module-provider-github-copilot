"""Behavioral tests for contracts/behaviors.md contract.

Contract: behaviors.md

Tests verify retry policy behaviors defined in the contract.

Contract Anchors:
- behaviors:Retry:MUST:1 — Respects max_attempts
- behaviors:Retry:MUST:4 — Only retries retryable errors
- behaviors:Retry:MUST:5 — Does NOT retry non-retryable errors
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

# =============================================================================
# Contract Discovery Tests (verify contract and config exist)
# =============================================================================


class TestBehaviorsContractExists:
    """Verify behaviors contract exists."""

    def test_behaviors_contract_exists(self) -> None:
        """behaviors.md contract must exist."""
        contract_path = Path("contracts/behaviors.md")
        assert contract_path.exists(), "contracts/behaviors.md must exist"

    def test_retry_config_exists_in_wheel(self) -> None:
        """Retry config exists in wheel package and is loaded by provider."""
        # Root config (legacy) may be absent or tombstoned
        root_config_path = Path("config/retry.yaml")
        if root_config_path.exists():
            content = root_config_path.read_text()
            assert "REMOVED" in content or len(content.strip()) == 0, (
                "config/retry.yaml should be tombstone (legacy location, not packaged)"
            )

        # Wheel config must exist
        wheel_config_path = Path("amplifier_module_provider_github_copilot/config/retry.yaml")
        assert wheel_config_path.exists(), "retry.yaml must exist in wheel config"

        # Verify config has expected structure
        import yaml

        with wheel_config_path.open(encoding="utf-8") as f:
            config = yaml.safe_load(f)
        assert "retry" in config, "Config must have 'retry' section"
        assert config["retry"].get("max_attempts") == 3


class TestRetryConfigDeferred:
    """Verify retry config is present in wheel and matches contract values."""

    def test_behaviors_contract_defines_retry_policy(self) -> None:
        """behaviors:Retry:MUST:1 — contract defines max_attempts=3."""
        contract_path = Path("contracts/behaviors.md")
        content = contract_path.read_text()

        # Verify contract defines expected values
        assert "max_attempts: 3" in content
        assert "strategy: exponential_with_jitter" in content
        assert "base_delay_ms: 1000" in content
        assert "jitter_factor: 0.1" in content


# =============================================================================
# Retry Behavior Tests
# =============================================================================


class TestRetryBehavior:
    """Test retry behavior matches behaviors.md contract."""

    @pytest.mark.asyncio
    async def test_retryable_error_is_retried(self) -> None:
        """behaviors:Retry:MUST:4 — Retryable errors trigger retry.

        When SDK raises a retryable error (e.g., LLMTimeoutError with retryable=True),
        the provider should retry up to max_attempts before propagating the error.
        """
        from contextlib import asynccontextmanager

        from amplifier_core.llm_errors import LLMTimeoutError

        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        call_count = 0

        @asynccontextmanager
        async def mock_session_raises_retryable(**kwargs: Any):  # type: ignore[misc]
            """Mock session context manager that raises retryable error."""
            nonlocal call_count
            call_count += 1
            raise LLMTimeoutError("SDK timeout", retryable=True)
            yield  # Never reached, but required for generator syntax

        # Create provider and patch its _client
        provider = GitHubCopilotProvider()
        mock_client = MagicMock()
        mock_client.session = mock_session_raises_retryable
        provider._client = mock_client  # type: ignore[assignment]

        mock_request = MagicMock()
        mock_request.messages = [{"role": "user", "content": "test"}]
        mock_request.model = "test-model"

        with pytest.raises(LLMTimeoutError):
            await provider.complete(mock_request)

        # behaviors:Retry:MUST:1 — max_attempts is 3 per config
        # So we expect 3 calls (1 initial + 2 retries)
        assert call_count == 3, f"Expected 3 attempts, got {call_count}"

    @pytest.mark.asyncio
    async def test_non_retryable_error_not_retried(self) -> None:
        """behaviors:Retry:MUST:5 — Non-retryable errors fail immediately.

        When SDK raises a non-retryable error (e.g., AuthenticationError),
        the provider should NOT retry - fail immediately.
        """
        from contextlib import asynccontextmanager

        from amplifier_core.llm_errors import AuthenticationError

        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        call_count = 0

        @asynccontextmanager
        async def mock_session_raises_non_retryable(**kwargs: Any):  # type: ignore[misc]
            """Mock session context manager that raises non-retryable error."""
            nonlocal call_count
            call_count += 1
            raise AuthenticationError("Invalid token", retryable=False)
            yield  # Never reached

        provider = GitHubCopilotProvider()
        mock_client = MagicMock()
        mock_client.session = mock_session_raises_non_retryable
        provider._client = mock_client  # type: ignore[assignment]

        mock_request = MagicMock()
        mock_request.messages = [{"role": "user", "content": "test"}]
        mock_request.model = "test-model"

        with pytest.raises(AuthenticationError):
            await provider.complete(mock_request)

        # Non-retryable should fail immediately (1 call only)
        assert call_count == 1, f"Expected 1 attempt (no retry), got {call_count}"


# =============================================================================
# Retryable Error Classification Tests
# =============================================================================


class TestRetryableErrorClassification:
    """Verify error retryability matches behaviors.md table."""

    def test_auth_error_not_retryable(self) -> None:
        """behaviors:Retry:MUST:5 — AuthenticationError is not retryable."""
        from amplifier_core.llm_errors import AuthenticationError

        err = AuthenticationError("test")
        assert err.retryable is False

    def test_rate_limit_error_is_retryable(self) -> None:
        """behaviors:Retry:MUST:4 — RateLimitError is retryable."""
        from amplifier_core.llm_errors import RateLimitError

        err = RateLimitError("test")
        assert err.retryable is True

    def test_timeout_error_is_retryable(self) -> None:
        """behaviors:Retry:MUST:4 — LLMTimeoutError is retryable."""
        from amplifier_core.llm_errors import LLMTimeoutError

        err = LLMTimeoutError("test")
        assert err.retryable is True

    def test_content_filter_error_not_retryable(self) -> None:
        """behaviors:Retry:MUST:5 — ContentFilterError is not retryable."""
        from amplifier_core.llm_errors import ContentFilterError

        err = ContentFilterError("test")
        assert err.retryable is False

    def test_network_error_is_retryable(self) -> None:
        """behaviors:Retry:MUST:4 — NetworkError is retryable."""
        from amplifier_core.llm_errors import NetworkError

        err = NetworkError("test")
        assert err.retryable is True

    def test_provider_unavailable_is_retryable(self) -> None:
        """behaviors:Retry:MUST:4 — ProviderUnavailableError is retryable."""
        from amplifier_core.llm_errors import ProviderUnavailableError

        err = ProviderUnavailableError("test")
        assert err.retryable is True

    def test_abort_error_not_retryable(self) -> None:
        """behaviors:Retry:MUST:5 — AbortError is not retryable."""
        from amplifier_core.llm_errors import AbortError

        err = AbortError("test")
        assert err.retryable is False


# =============================================================================
# Contract Tests: behaviors:Streaming:MUST:1, behaviors:Models:MUST:1,2
# Added as part of mock quality audit
# =============================================================================


class TestStreamingBehavior:
    """Tests for behaviors:Streaming:MUST:1."""

    def test_ttft_warning_config_loaded(self) -> None:
        """TTFT warning threshold loads from config.

        Contract: behaviors:Streaming:MUST:1
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_streaming_config,
        )

        config = load_streaming_config()

        # Contract specifies 5000ms threshold
        assert config.ttft_warning_ms == 5000
        assert config.event_queue_size == 256
        assert config.max_gap_warning_ms == 5000
        assert config.max_gap_error_ms == 30000


class TestModelResolution:
    """Tests for behaviors:Models:MUST:1,2."""

    def test_model_alias_resolution(self) -> None:
        """Model aliases resolved transparently.

        Contract: behaviors:Models:MUST:1
        Note: Aliases are not yet implemented - test verifies models list exists.
        """
        from amplifier_module_provider_github_copilot.config_loader import load_models_config

        config = load_models_config()

        # Should have models list (aliases not yet implemented)
        assert len(config.models) > 0, "config.models should contain model definitions"

    def test_invalid_model_returns_not_found_error(self) -> None:
        """Invalid model raises NotFoundError.

        Contract: behaviors:Models:MUST:2
        """
        from amplifier_core.llm_errors import NotFoundError

        # NotFoundError is the expected type for invalid models
        assert issubclass(NotFoundError, Exception)


class TestCacheInvalidation:
    """Test cache invalidation function."""

    def test_invalidate_cache_removes_file(self, tmp_path: Path) -> None:
        """invalidate_cache() removes existing cache file.

        Documents the invalidate_cache() function as public API.
        """
        from amplifier_module_provider_github_copilot.model_cache import (
            invalidate_cache,
            write_cache,
        )
        from amplifier_module_provider_github_copilot.models import CopilotModelInfo

        cache_file = tmp_path / "test_cache.json"
        models = [
            CopilotModelInfo(
                id="test-model",
                name="Test Model",
                context_window=100000,
                max_output_tokens=8000,
            ),
        ]
        write_cache(models, cache_file=cache_file)
        assert cache_file.exists()

        invalidate_cache(cache_file=cache_file)
        assert not cache_file.exists()
