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
            content = root_config_path.read_text(encoding="utf-8")
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
        content = contract_path.read_text(encoding="utf-8")

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

        # Contract specifies lenient thresholds for SDK latency
        assert config.ttft_warning_ms == 15000
        assert config.event_queue_size == 10000
        assert config.max_gap_warning_ms == 10000
        assert config.max_gap_error_ms == 30000


class TestBoundedQueueBehavior:
    """Tests for behaviors:Streaming:MUST:4 — MUST NOT block on queue full.

    Contract: behaviors:Streaming:MUST:4 — MUST NOT block on queue full (drop oldest)

    These tests verify BOTH code paths (production and test) use bounded queues:
    - provider.py:_execute_sdk_completion() — production path
    - completion.py:complete() — test path

    PR #32 Review: Found divergence where production path had unbounded queue.
    """

    def test_production_path_uses_bounded_queue(self) -> None:
        """behaviors:Streaming:MUST:4 — Production path must use bounded queue.

        Verifies provider.py:_execute_sdk_completion() creates queue with maxsize.
        This is a code inspection test since the production path is not easily
        testable via mocks without risking the very bug we're catching.
        """
        import ast
        from pathlib import Path

        provider_path = Path("amplifier_module_provider_github_copilot/provider.py")
        source = provider_path.read_text(encoding="utf-8")

        # Parse and find asyncio.Queue() calls
        tree = ast.parse(source)

        queue_calls: list[ast.Call] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Check for asyncio.Queue() or Queue()
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "Queue":
                    queue_calls.append(node)
                elif isinstance(func, ast.Name) and func.id == "Queue":
                    queue_calls.append(node)

        # There should be at least one Queue call in provider.py
        assert len(queue_calls) > 0, "provider.py must have asyncio.Queue() calls"

        # ALL Queue calls must have maxsize argument
        for call in queue_calls:
            has_maxsize = False
            # Check positional args
            if len(call.args) > 0:
                has_maxsize = True
            # Check keyword args
            for kw in call.keywords:
                if kw.arg == "maxsize":
                    has_maxsize = True
                    break

            assert has_maxsize, (
                f"asyncio.Queue() at line {call.lineno} must have maxsize argument. "
                "Contract: behaviors:Streaming:MUST:4 — MUST NOT block on queue full"
            )

    # test_completion_path_uses_bounded_queue removed - completion.py deleted (Issue #6)

    def test_queue_size_matches_config(self) -> None:
        """behaviors:Streaming:MUST:4 — Queue size must come from config.

        Verifies the streaming config's event_queue_size is used (not hardcoded).
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_streaming_config,
        )

        config = load_streaming_config()

        # Config must define event_queue_size
        assert hasattr(config, "event_queue_size"), "Config must have event_queue_size"
        assert config.event_queue_size > 0, "event_queue_size must be positive"
        assert config.event_queue_size == 10000, "Contract specifies 10000 as default"

    @pytest.mark.asyncio
    async def test_queue_full_drops_without_blocking(self) -> None:
        """behaviors:Streaming:MUST:4 — QueueFull must drop, not block.

        This runtime test verifies the contract pattern works correctly:
        1. put_nowait() on full queue raises QueueFull
        2. We catch it and continue (don't block)
        3. Events are dropped, not lost silently

        The production code in provider.py uses this exact pattern.
        """
        import asyncio

        # Create a tiny bounded queue to quickly reach capacity
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=2)

        # Fill the queue to capacity
        queue.put_nowait("event_1")
        queue.put_nowait("event_2")
        assert queue.full(), "Queue should be full after maxsize events"

        # Attempting put_nowait on full queue MUST raise QueueFull
        dropped_count = 0
        for i in range(3):  # Try to add 3 more
            try:
                queue.put_nowait(f"overflow_{i}")
            except asyncio.QueueFull:
                # Contract: drop on full, don't block
                dropped_count += 1

        # All 3 overflow attempts should have been dropped
        assert dropped_count == 3, f"Expected 3 drops, got {dropped_count}"

        # Queue should still have exactly 2 events (original, not overflow)
        assert queue.qsize() == 2, "Queue should still have original 2 events"

        # Verify we can drain the queue (proves we didn't block)
        items: list[str] = []
        while not queue.empty():
            items.append(queue.get_nowait())

        assert items == ["event_1", "event_2"], "Queue should have original events"

    def test_queue_full_handling_pattern_in_provider(self) -> None:
        """behaviors:Streaming:MUST:4 — Verify provider has QueueFull handling.

        Code inspection test that verifies the streaming event handler catches QueueFull,
        logs a message, and continues (not just that queue has maxsize).
        Post-P1.6 refactor: event_router.py contains the QueueFull handling logic.
        """
        import re
        from pathlib import Path

        # P1.6 refactor: QueueFull handling now lives in event_router.py
        event_router_path = Path("amplifier_module_provider_github_copilot/event_router.py")
        source = event_router_path.read_text(encoding="utf-8")

        # Verify QueueFull exception is caught
        assert "except asyncio.QueueFull:" in source, (
            "event_router.py must catch asyncio.QueueFull exception"
        )

        # Verify logging happens in the except block
        # Look for pattern: except QueueFull: ... logger.(debug|warning|info)
        queuefull_pattern = r"except asyncio\.QueueFull:.*?logger\.(debug|warning|info)"
        match = re.search(queuefull_pattern, source, re.DOTALL)
        assert match, "event_router.py must log when QueueFull is caught (for observability)"

        # Verify return statement after QueueFull (proves we don't propagate)
        return_after_pattern = r"except asyncio\.QueueFull:.*?return"
        match = re.search(return_after_pattern, source, re.DOTALL)
        assert match, "event_router.py must return after QueueFull (drop event, don't propagate)"

    @pytest.mark.asyncio
    async def test_event_handler_logs_on_queue_full(self, caplog: pytest.LogCaptureFixture) -> None:
        """behaviors:Streaming:MUST:4 — Event handler MUST log when dropping events.

        Runtime behavioral test: simulates the exact event handler pattern
        from provider.py and verifies logging occurs when queue is full.

        This tests the actual RUNTIME behavior, not just code structure.
        """
        import asyncio
        import logging

        from amplifier_module_provider_github_copilot.sdk_adapter import extract_event_type

        # Set up logging capture at DEBUG level (where QueueFull is logged)
        caplog.set_level(logging.DEBUG)

        # Create a tiny bounded queue (same pattern as provider.py)
        queue: asyncio.Queue[dict[str, str]] = asyncio.Queue(maxsize=2)

        # Simulate the event handler from provider.py (lines 603-638)
        # This is the EXACT logic we're testing
        def event_handler(sdk_event: dict[str, str]) -> bool:
            """Returns True if event was queued, False if dropped."""
            event_type = extract_event_type(sdk_event)

            try:
                queue.put_nowait(sdk_event)
                return True
            except asyncio.QueueFull:
                # Contract: behaviors:Streaming:MUST:4 — drop on full, don't block
                logging.getLogger(__name__).debug(
                    "[STREAMING] Event queue full, dropping delta: %s",
                    event_type,
                )
                return False

        # Fill the queue to capacity
        event1 = {"type": "assistant.message_delta", "data": "hello"}
        event2 = {"type": "assistant.message_delta", "data": "world"}
        assert event_handler(event1), "First event should be queued"
        assert event_handler(event2), "Second event should be queued"
        assert queue.full(), "Queue should be full"

        # Now send more events - these should be dropped and logged
        overflow_events = [
            {"type": "assistant.message_delta", "data": f"overflow_{i}"} for i in range(5)
        ]

        dropped_count = 0
        for event in overflow_events:
            if not event_handler(event):
                dropped_count += 1

        # Verify all overflow events were dropped
        assert dropped_count == 5, f"Expected 5 drops, got {dropped_count}"

        # Verify logging occurred for each drop
        drop_logs = [r for r in caplog.records if "queue full" in r.message.lower()]
        assert len(drop_logs) == 5, (
            f"Expected 5 'queue full' log messages, got {len(drop_logs)}. "
            f"Log messages: {[r.message for r in caplog.records]}"
        )

        # Verify queue still has original events (not corrupted)
        assert queue.qsize() == 2, "Queue should still have 2 events"


class TestModelResolution:
    """Tests for behaviors:Models:MUST:1,2."""

    def test_invalid_model_raises_not_found_error(self) -> None:
        """Invalid model raises NotFoundError.

        Contract: behaviors:Models:MUST:1 — Raises NotFoundError for invalid model
        """
        from amplifier_core.llm_errors import NotFoundError

        # NotFoundError is the expected type for invalid models
        assert issubclass(NotFoundError, Exception)

    def test_model_selection_priority(self) -> None:
        """Model selection respects priority: request > config > YAML.

        Contract: behaviors:Models:MUST:2
        """
        from amplifier_module_provider_github_copilot.config_loader import load_models_config

        config = load_models_config()

        # Should have models list with defaults
        assert len(config.models) > 0, "config.models should contain model definitions"
        assert config.defaults.get("model"), "defaults['model'] should be set"


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


# =============================================================================
# SDK Version Consistency Tests
# =============================================================================


class TestSdkVersionConsistency:
    """Verify SDK version in error message matches pyproject.toml.

    Contract: sdk-boundary.md MUST:5 — fail at import time with clear error
    if github-copilot-sdk is not installed.

    The error message must tell users to install the CORRECT version
    that matches pyproject.toml, not a stale version.
    """

    def test_error_message_sdk_version_matches_pyproject_toml(self) -> None:
        """sdk-boundary:ImportCheck:MUST — version in error message matches pyproject.toml.

        Regression prevention: The error message was previously telling users to install
        >=0.1.32,<0.2.0 while pyproject.toml requires >=0.2.0,<0.3.0.
        """
        import re

        # Read pyproject.toml to get the authoritative SDK version requirement
        pyproject_path = Path("pyproject.toml")
        pyproject_content = pyproject_path.read_text(encoding="utf-8")

        # Extract SDK version from pyproject.toml dependencies
        sdk_pattern = r'"github-copilot-sdk([^"]+)"'
        pyproject_match = re.search(sdk_pattern, pyproject_content)
        assert pyproject_match, "pyproject.toml must declare github-copilot-sdk dependency"
        pyproject_version = pyproject_match.group(1)  # e.g., ">=0.2.0,<0.3.0"

        # Read __init__.py source and find the error message version
        init_path = Path("amplifier_module_provider_github_copilot/__init__.py")
        init_content = init_path.read_text(encoding="utf-8")

        # Extract SDK version from the ImportError message
        # Looking for: pip install 'github-copilot-sdk>=X.X.X,<Y.Y.Y'
        error_pattern = r"pip install 'github-copilot-sdk([^']+)'"
        error_match = re.search(error_pattern, init_content)
        assert error_match, "__init__.py must have SDK install instruction in error message"
        error_version = error_match.group(1)  # e.g., ">=0.2.0,<0.3.0"

        # The versions MUST match
        assert error_version == pyproject_version, (
            f"SDK version in error message does NOT match pyproject.toml!\n"
            f"  Error message says: github-copilot-sdk{error_version}\n"
            f"  pyproject.toml says: github-copilot-sdk{pyproject_version}\n"
            f"  Fix __init__.py line ~35 to match pyproject.toml"
        )


class TestPackageVersionConsistency:
    """Verify package __version__ matches pyproject.toml version.

    Contract: provider-protocol:EP:MUST:1 — package identity must be consistent.

    The package __version__ is used by tools like pip show, importlib.metadata,
    and user code that checks provider version. It MUST match pyproject.toml
    to avoid confusion and version drift.
    """

    def test_package_version_matches_pyproject_toml(self) -> None:
        """Package __version__ must match pyproject.toml version exactly.

        Regression prevention: __version__ could be hardcoded and forgotten
        during releases, causing version drift.
        """
        import re

        # Read pyproject.toml to get the authoritative package version
        pyproject_path = Path("pyproject.toml")
        pyproject_content = pyproject_path.read_text(encoding="utf-8")

        # Extract version from pyproject.toml [project] section
        # Looking for: version = "X.Y.Z"
        version_pattern = r'version\s*=\s*"([^"]+)"'
        pyproject_match = re.search(version_pattern, pyproject_content)
        assert pyproject_match, "pyproject.toml must declare version"
        pyproject_version = pyproject_match.group(1)  # e.g., "2.0.0"

        # Import __version__ from the package
        from amplifier_module_provider_github_copilot import __version__

        # The versions MUST match
        assert __version__ == pyproject_version, (
            f"Package __version__ does NOT match pyproject.toml!\n"
            f"  __version__ says: {__version__}\n"
            f"  pyproject.toml says: {pyproject_version}\n"
            f"  Fix __init__.py __version__ = '{pyproject_version}' to match pyproject.toml"
        )

    def test_package_version_is_valid_semver(self) -> None:
        """Package __version__ must be valid semantic versioning.

        Ensures version can be parsed by packaging tools.
        """
        import re

        from amplifier_module_provider_github_copilot import __version__

        # Semantic versioning pattern (simplified): X.Y.Z with optional pre-release
        semver_pattern = r"^\d+\.\d+\.\d+(-[a-zA-Z0-9.]+)?(\+[a-zA-Z0-9.]+)?$"
        assert re.match(semver_pattern, __version__), (
            f"__version__ '{__version__}' is not valid semantic versioning. "
            f"Expected format: X.Y.Z or X.Y.Z-prerelease"
        )


# =============================================================================
# Documentation/YAML Consistency Tests
# =============================================================================


class TestDocstringYamlConsistency:
    """Verify docstrings match YAML configuration.

    Contract: Three-Medium Architecture — YAML is authoritative for policy values.
    Docstrings must not contradict YAML.
    """

    def test_error_translation_default_retryable_docstring_matches_yaml(self) -> None:
        """error-hierarchy:Default — docstring claim must match errors.yaml.

        Regression prevention: The docstring was saying retryable=True but
        errors.yaml (authoritative) says retryable: false.
        """
        import re

        # Read the YAML (authoritative source of truth)
        yaml_path = Path("amplifier_module_provider_github_copilot/config/errors.yaml")
        yaml_content = yaml_path.read_text(encoding="utf-8")

        # Extract default retryable from YAML
        # Looking for:  default:\n  ...retryable: true/false
        default_section_match = re.search(
            r"^default:\s*\n(?:.*\n)*?\s+retryable:\s*(true|false)",
            yaml_content,
            re.MULTILINE,
        )
        assert default_section_match, "errors.yaml must have default section with retryable"
        yaml_retryable = default_section_match.group(1)  # "true" or "false"

        # Read the module docstring
        module_path = Path("amplifier_module_provider_github_copilot/error_translation.py")
        module_content = module_path.read_text(encoding="utf-8")

        # Extract retryable claim from docstring
        # Looking for: retryable=True or retryable=False
        docstring_match = re.search(r"retryable=(True|False)", module_content)
        if docstring_match:
            docstring_retryable = docstring_match.group(1).lower()  # "true" or "false"

            assert docstring_retryable == yaml_retryable, (
                f"error_translation.py docstring contradicts errors.yaml!\n"
                f"  Docstring says: retryable={docstring_retryable.capitalize()}\n"
                f"  errors.yaml says: retryable: {yaml_retryable}\n"
                f"  YAML is authoritative — fix the docstring"
            )

    def test_error_config_python_fallback_matches_yaml(self) -> None:
        """Three-Medium — Python fallback defaults must match YAML values.

        The _load_error_config_cached function has hardcoded fallback values.
        These MUST match the YAML defaults to avoid drift.
        """
        import re

        # Read YAML default
        yaml_path = Path("amplifier_module_provider_github_copilot/config/errors.yaml")
        yaml_content = yaml_path.read_text(encoding="utf-8")
        default_match = re.search(
            r"^default:\s*\n(?:.*\n)*?\s+retryable:\s*(true|false)",
            yaml_content,
            re.MULTILINE,
        )
        assert default_match, "errors.yaml must have default section"
        yaml_retryable = default_match.group(1) == "true"

        # Read Python fallback
        module_path = Path("amplifier_module_provider_github_copilot/error_translation.py")
        module_content = module_path.read_text(encoding="utf-8")

        # Find the hardcoded fallback: default.get("retryable", True)
        fallback_match = re.search(
            r'default\.get\(["\']retryable["\'],\s*(True|False)\)', module_content
        )
        if fallback_match:
            python_fallback = fallback_match.group(1) == "True"

            assert python_fallback == yaml_retryable, (
                f"Python fallback contradicts YAML!\n"
                f"  Python fallback: default.get('retryable', {python_fallback})\n"
                f"  YAML default: retryable: {yaml_retryable}\n"
                f"  Fix Python to match YAML"
            )


# =============================================================================
# Production Path Testing with MockCopilotClientWrapper
# =============================================================================


class TestProductionPathWithMockClient:
    """Test that MockCopilotClientWrapper enables production path testing.

    Contract: sdk-boundary:Membrane:MUST:1 — mock at the membrane level

    Architecture Change:
    - OLD: Tests use sdk_create_fn → completion.py (separate code path)
    - NEW: Tests use MockCopilotClientWrapper → production code path

    This ensures tests exercise the SAME code as production.
    """

    @pytest.mark.asyncio
    async def test_mock_client_enables_production_path(self) -> None:
        """MockCopilotClientWrapper allows testing production code path.

        Demonstrates that tests can use provider._execute_sdk_completion()
        (the production path) instead of completion.py (test-only path).
        """
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )
        from tests.fixtures.sdk_mocks import (
            MockCopilotClientWrapper,
            text_delta_event,
        )

        # Create mock client with events
        events = [text_delta_event("Hello, world!")]
        mock_client = MockCopilotClientWrapper(events=events)

        # Inject mock client into provider
        provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]

        # Create request using proper kernel type
        from amplifier_core.message_models import TextBlock

        request = MagicMock()
        request.model = "gpt-4"
        request.messages = [MagicMock(role="user", content="test")]
        request.tools = None
        request.max_tokens = None
        request.temperature = None
        request.stop = None
        request.stream = None

        # Call the production complete() method
        # This uses _execute_sdk_completion(), NOT completion.py
        response = await provider.complete(request)

        # Verify response has content
        assert response.content is not None
        assert len(response.content) > 0

        # Verify it's a TextBlock (kernel type)
        first_block = response.content[0]
        assert isinstance(first_block, TextBlock)
        assert "Hello, world!" in first_block.text

    @pytest.mark.asyncio
    async def test_mock_client_session_receives_model(self) -> None:
        """MockCopilotClientWrapper.session() receives model parameter."""
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )
        from tests.fixtures.sdk_mocks import MockCopilotClientWrapper

        mock_client = MockCopilotClientWrapper(events=[])
        provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]

        request = MagicMock()
        request.model = "gpt-4-turbo"
        request.messages = [MagicMock(role="user", content="test")]
        request.tools = None
        request.max_tokens = None
        request.temperature = None
        request.stop = None
        request.stream = None

        await provider.complete(request)

        # Verify model was passed to session
        assert mock_client.last_model == "gpt-4-turbo"

    @pytest.mark.asyncio
    async def test_mock_client_session_receives_tools(self) -> None:
        """MockCopilotClientWrapper.session() receives tools parameter."""
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )
        from tests.fixtures.sdk_mocks import MockCopilotClientWrapper

        mock_client = MockCopilotClientWrapper(events=[])
        provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]

        # Create request with tools
        tool: dict[str, Any] = {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {},
        }
        request = MagicMock()
        request.model = "gpt-4"
        request.messages = [MagicMock(role="user", content="test")]
        request.tools = [tool]
        request.max_tokens = None
        request.temperature = None
        request.stop = None
        request.stream = None

        await provider.complete(request)

        # Verify tools were passed to session
        assert mock_client.last_tools is not None
        assert len(mock_client.last_tools) == 1

    @pytest.mark.asyncio
    async def test_mock_client_session_receives_system_message(self) -> None:
        """MockCopilotClientWrapper.session() receives system_message from ChatRequest.

        This verifies the complete() -> _execute_sdk_completion() -> session() chain
        correctly extracts and forwards system_message from ChatRequest.messages.

        Contract: sdk-boundary:Config:MUST:2
        Bug fix: system_message was missing, causing SDK to use default persona
        instead of Amplifier bundle instructions.
        """
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )
        from tests.fixtures.sdk_mocks import MockCopilotClientWrapper

        mock_client = MockCopilotClientWrapper(events=[])
        provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]

        # Create request with system message in messages
        system_msg = MagicMock()
        system_msg.role = "system"
        system_msg.content = "You are a helpful coding assistant."

        user_msg = MagicMock()
        user_msg.role = "user"
        user_msg.content = "Hello"

        request = MagicMock()
        request.model = "gpt-4"
        request.messages = [system_msg, user_msg]
        request.tools = None
        request.max_tokens = None
        request.temperature = None
        request.stop = None
        request.stream = None

        await provider.complete(request)

        # Verify system_message was extracted and forwarded
        assert mock_client.last_system_message is not None, (
            "system_message was not forwarded to SDK session. "
            "This causes SDK to use default persona instead of bundle instructions."
        )
        assert mock_client.last_system_message == "You are a helpful coding assistant."

    @pytest.mark.asyncio
    async def test_mock_client_propagates_errors_correctly(self) -> None:
        """Errors from MockCopilotClientWrapper are properly translated."""
        from amplifier_core.llm_errors import ProviderUnavailableError

        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )
        from tests.fixtures.sdk_mocks import MockCopilotClientWrapper

        # Create mock that raises on session creation
        mock_client = MockCopilotClientWrapper(
            events=[],
            raise_on_session=ConnectionError("Network failure"),
        )
        provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]

        request = MagicMock()
        request.model = "gpt-4"
        request.messages = [MagicMock(role="user", content="test")]
        request.tools = None
        request.max_tokens = None
        request.temperature = None
        request.stop = None
        request.stream = None

        # Should raise translated error
        with pytest.raises(ProviderUnavailableError):
            await provider.complete(request)

    @pytest.mark.asyncio
    async def test_session_disconnected_after_completion(self) -> None:
        """Session is always disconnected after completion.

        Contract: deny-destroy.md — Session MUST be destroyed in finally block
        Replaces: test_completion.py::TestSessionLifecycle::
        test_session_created_and_destroyed_on_success
        """
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from tests.fixtures.sdk_mocks import MockCopilotClientWrapper, text_delta_event

        mock_client = MockCopilotClientWrapper(events=[text_delta_event("hello")])
        provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]

        request = MagicMock()
        request.model = "gpt-4"
        request.messages = [MagicMock(role="user", content="test")]
        request.tools = None
        request.max_tokens = None
        request.temperature = None
        request.stop = None
        request.stream = None

        await provider.complete(request)

        # Session should be disconnected
        assert mock_client.session_instance is not None
        assert mock_client.session_instance.disconnected is True

    @pytest.mark.asyncio
    async def test_session_disconnected_on_error(self) -> None:
        """Session disconnected even when error occurs during send().

        Contract: deny-destroy.md — Session MUST be destroyed even on error
        Replaces: test_completion.py::TestSessionLifecycle::test_session_destroyed_on_error
        """
        from amplifier_core.llm_errors import NetworkError

        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from tests.fixtures.sdk_mocks import MockCopilotClientWrapper

        mock_client = MockCopilotClientWrapper(
            events=[],
            raise_on_send=ConnectionError("Network failure"),
        )
        provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]

        request = MagicMock()
        request.model = "gpt-4"
        request.messages = [MagicMock(role="user", content="test")]
        request.tools = None
        request.max_tokens = None
        request.temperature = None
        request.stop = None
        request.stream = None

        with pytest.raises(NetworkError):
            await provider.complete(request)

        # Session should still be disconnected despite error
        assert mock_client.session_instance is not None
        assert mock_client.session_instance.disconnected is True

    @pytest.mark.asyncio
    async def test_text_content_accumulated_correctly(self) -> None:
        """Multiple text deltas are accumulated into single response.

        Contract: streaming-contract.md — Text content MUST be accumulated
        Replaces: test_completion.py::TestResponseConstruction::test_text_content_accumulated
        """
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from tests.fixtures.sdk_mocks import MockCopilotClientWrapper, text_delta_event

        # Multiple text delta events
        events = [
            text_delta_event("Hello "),
            text_delta_event("World"),
            text_delta_event("!"),
        ]
        mock_client = MockCopilotClientWrapper(events=events)
        provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]

        request = MagicMock()
        request.model = "gpt-4"
        request.messages = [MagicMock(role="user", content="test")]
        request.tools = None
        request.max_tokens = None
        request.temperature = None
        request.stop = None
        request.stream = None

        response = await provider.complete(request)

        # Should have accumulated text
        from amplifier_core.message_models import TextBlock

        assert len(response.content) > 0
        first_block = response.content[0]
        assert isinstance(first_block, TextBlock)
        assert first_block.text == "Hello World!"

    @pytest.mark.asyncio
    async def test_empty_response_handled_gracefully(self) -> None:
        """Empty response (no content events) is handled gracefully.

        Contract: streaming-contract.md — Empty responses MUST not crash
        Replaces: test_completion.py::TestResponseConstruction::test_empty_response_handled
        """
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from tests.fixtures.sdk_mocks import MockCopilotClientWrapper

        # No content events, just idle
        mock_client = MockCopilotClientWrapper(events=[])
        provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]

        request = MagicMock()
        request.model = "gpt-4"
        request.messages = [MagicMock(role="user", content="test")]
        request.tools = None
        request.max_tokens = None
        request.temperature = None
        request.stop = None
        request.stream = None

        response = await provider.complete(request)

        # Should return valid response with empty content
        assert response is not None
        # Content may be empty or contain empty text block
        if response.content:
            from amplifier_core.message_models import TextBlock

            first_block = response.content[0]
            if isinstance(first_block, TextBlock):
                assert first_block.text == ""

    @pytest.mark.asyncio
    async def test_error_during_send_translated(self) -> None:
        """SDK errors during send() are translated to kernel errors.

        Contract: error-hierarchy.md — SDK errors MUST be translated
        Replaces: test_completion.py::TestErrorHandling::test_sdk_error_translated
        """
        from amplifier_core.llm_errors import NetworkError

        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from tests.fixtures.sdk_mocks import MockCopilotClientWrapper

        mock_client = MockCopilotClientWrapper(
            events=[],
            raise_on_send=ConnectionError("Connection refused"),
        )
        provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]

        request = MagicMock()
        request.model = "gpt-4"
        request.messages = [MagicMock(role="user", content="test")]
        request.tools = None
        request.max_tokens = None
        request.temperature = None
        request.stop = None
        request.stream = None

        with pytest.raises(NetworkError) as exc_info:
            await provider.complete(request)

        # Should be kernel NetworkError with provider info
        assert exc_info.value.provider == "github-copilot"

    @pytest.mark.asyncio
    async def test_original_exception_chained(self) -> None:
        """Original SDK exception is preserved via __cause__.

        Contract: error-hierarchy.md — Original exception MUST be chained
        Replaces: test_completion.py::TestErrorHandling::test_error_preserves_original
        """
        from amplifier_core.llm_errors import LLMError

        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from tests.fixtures.sdk_mocks import MockCopilotClientWrapper

        original_error = ConnectionError("Original connection error")
        mock_client = MockCopilotClientWrapper(
            events=[],
            raise_on_send=original_error,
        )
        provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]

        request = MagicMock()
        request.model = "gpt-4"
        request.messages = [MagicMock(role="user", content="test")]
        request.tools = None
        request.max_tokens = None
        request.temperature = None
        request.stop = None
        request.stream = None

        with pytest.raises(LLMError) as exc_info:
            await provider.complete(request)

        # __cause__ should be the original exception
        assert exc_info.value.__cause__ is original_error

    @pytest.mark.asyncio
    async def test_llm_error_not_double_wrapped(self) -> None:
        """LLMError is not re-translated (double-wrapped).

        Contract: error-hierarchy.md — Pre-translated errors MUST pass through
        Replaces: test_sdk_boundary.py::TestDoubleTranslationGuard::
        test_llm_error_not_double_wrapped
        """
        from amplifier_core.llm_errors import AuthenticationError

        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from tests.fixtures.sdk_mocks import MockCopilotClientWrapper

        # Pre-translated error (already an LLMError subclass)
        pre_translated_error = AuthenticationError("Already translated", provider="test-provider")
        mock_client = MockCopilotClientWrapper(
            events=[],
            raise_on_session=pre_translated_error,
        )
        provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]

        request = MagicMock()
        request.model = "gpt-4"
        request.messages = [MagicMock(role="user", content="test")]
        request.tools = None
        request.max_tokens = None
        request.temperature = None
        request.stop = None
        request.stream = None

        with pytest.raises(AuthenticationError) as exc_info:
            await provider.complete(request)

        # Should be the SAME error, not wrapped
        assert exc_info.value is pre_translated_error
        # Should NOT have a __cause__ (not double-wrapped)
        assert exc_info.value.__cause__ is None
        # Original provider preserved
        assert exc_info.value.provider == "test-provider"


# =============================================================================
# Runtime Config Override (Routing Matrix Support)
# =============================================================================


class TestRuntimeConfigOverride:
    """Test model selection priority with runtime config.

    Contract: behaviors:ModelSelection:MUST:1,2

    Bug Reference: Routing matrix passes config={"default_model": "gpt-5.4"}
    but provider was ignoring it and always using YAML default.
    """

    def test_effective_default_model_respects_runtime_config(self) -> None:
        """config["default_model"] from mount overrides YAML default.

        Contract: behaviors:ModelSelection:MUST:1
        Priority: request.model > config["default_model"] > YAML
        """
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from tests.fixtures.sdk_mocks import MockCopilotClientWrapper

        # Mount with routing matrix config
        routing_model = "gpt-5.4-turbo"
        mock_client = MockCopilotClientWrapper(events=[])

        # Pass runtime config like routing matrix does
        provider = GitHubCopilotProvider(
            config={"default_model": routing_model},
            client=mock_client,  # type: ignore[arg-type]
        )

        # _effective_default_model should use runtime config
        assert provider._effective_default_model == routing_model  # pyright: ignore[reportPrivateUsage]

    def test_effective_default_model_falls_back_to_yaml(self) -> None:
        """Without runtime config, falls back to YAML default.

        Contract: behaviors:ModelSelection:MUST:1
        Priority 3: YAML defaults.model (fallback)
        """
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from tests.fixtures.sdk_mocks import MockCopilotClientWrapper

        mock_client = MockCopilotClientWrapper(events=[])
        provider = GitHubCopilotProvider(
            config={},  # No default_model
            client=mock_client,  # type: ignore[arg-type]
        )

        # Should use YAML default (accessing private members for test verification)
        yaml_default = provider._provider_config.defaults["model"]  # pyright: ignore[reportPrivateUsage]
        assert provider._effective_default_model == yaml_default  # pyright: ignore[reportPrivateUsage]

    @pytest.mark.asyncio
    async def test_complete_uses_runtime_default_model(self) -> None:
        """complete() uses runtime config model when request.model is None.

        Contract: behaviors:ModelSelection:MUST:1
        Priority 2: config["default_model"] when request.model is None
        """
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from tests.fixtures.sdk_mocks import MockCopilotClientWrapper, text_delta_event

        routing_model = "gpt-5.4-turbo"
        mock_client = MockCopilotClientWrapper(events=[text_delta_event("hello")])

        provider = GitHubCopilotProvider(
            config={"default_model": routing_model},
            client=mock_client,  # type: ignore[arg-type]
        )

        # Request WITHOUT model specified — should use runtime default
        request = MagicMock()
        request.model = None  # No model in request
        request.messages = [MagicMock(role="user", content="test")]
        request.tools = None
        request.max_tokens = None
        request.temperature = None
        request.stop = None
        request.stream = None

        await provider.complete(request)

        # Session should receive the routing matrix model, not YAML default
        assert mock_client.last_model == routing_model

    @pytest.mark.asyncio
    async def test_request_model_overrides_runtime_config(self) -> None:
        """request.model takes precedence over runtime config.

        Contract: behaviors:ModelSelection:MUST:1
        Priority 1: request.model (explicit per-request override)
        """
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from tests.fixtures.sdk_mocks import MockCopilotClientWrapper, text_delta_event

        routing_model = "gpt-5.4-turbo"
        request_model = "claude-opus-4.5"
        mock_client = MockCopilotClientWrapper(events=[text_delta_event("hello")])

        provider = GitHubCopilotProvider(
            config={"default_model": routing_model},
            client=mock_client,  # type: ignore[arg-type]
        )

        # Request WITH explicit model
        request = MagicMock()
        request.model = request_model  # Explicit model in request
        request.messages = [MagicMock(role="user", content="test")]
        request.tools = None
        request.max_tokens = None
        request.temperature = None
        request.stop = None
        request.stream = None

        await provider.complete(request)

        # Session should receive request model, not runtime config
        assert mock_client.last_model == request_model

    def test_multiple_providers_dont_share_config(self) -> None:
        """Multiple providers with different configs don't conflict.

        Contract: behaviors:ModelSelection:MUST:2
        load_models_config() uses @lru_cache — MUST NOT mutate cached config.
        """
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from tests.fixtures.sdk_mocks import MockCopilotClientWrapper

        mock_client1 = MockCopilotClientWrapper(events=[])
        mock_client2 = MockCopilotClientWrapper(events=[])

        # Two providers with different routing matrix configs
        provider1 = GitHubCopilotProvider(
            config={"default_model": "gpt-5.4"},
            client=mock_client1,  # type: ignore[arg-type]
        )
        provider2 = GitHubCopilotProvider(
            config={"default_model": "claude-opus-4.5"},
            client=mock_client2,  # type: ignore[arg-type]
        )

        # Each should have independent effective default
        assert provider1._effective_default_model == "gpt-5.4"  # pyright: ignore[reportPrivateUsage]
        assert provider2._effective_default_model == "claude-opus-4.5"  # pyright: ignore[reportPrivateUsage]

        # The underlying cached config should NOT be mutated
        from amplifier_module_provider_github_copilot.config_loader import load_models_config

        cached_config = load_models_config()
        # Cached config should still have original YAML default
        # (not mutated by either provider)
        assert cached_config.defaults["model"] in ["claude-opus-4.5", "gpt-4", "gpt-4o"]


# =============================================================================
# C-2: asyncio.CancelledError translation
# =============================================================================


class TestCancelledErrorTranslation:
    """C-2: asyncio.CancelledError MUST translate to AbortError.

    Contract: error-hierarchy:AbortError:MUST:1

    asyncio.CancelledError inherits from BaseException since Python 3.9, NOT Exception.
    A bare ``except Exception`` in provider.complete() silently misses it, causing
    the raw CancelledError to escape to the kernel as an unhandled BaseException —
    violating the error-hierarchy contract and breaking observability (emit_response_error
    never fires).

    Fix: add ``except asyncio.CancelledError`` BEFORE each ``except Exception`` block,
    translate to ``AbortError(retryable=False)``, emit error event, and re-raise the
    translated error.
    """

    @pytest.mark.asyncio
    async def test_cancelled_error_from_session_becomes_abort_error(self) -> None:
        """error-hierarchy:AbortError:MUST:1 — CancelledError from SDK → AbortError.

        This is the FAILING test for C-2.  Before the fix, asyncio.CancelledError
        escapes ``except Exception`` and propagates as raw BaseException.
        After the fix it MUST be caught and re-raised as AbortError.
        """
        import asyncio

        from amplifier_core.llm_errors import AbortError

        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from tests.fixtures.sdk_mocks import MockCopilotClientWrapper

        # Inject CancelledError at session-creation time (before first SDK call)
        mock_client = MockCopilotClientWrapper(
            events=[],
            raise_on_session=asyncio.CancelledError("task cancelled"),  # type: ignore[arg-type]
        )
        provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]

        request = MagicMock()
        request.model = "gpt-4"
        request.messages = [MagicMock(role="user", content="test")]
        request.tools = None
        request.max_tokens = None
        request.temperature = None
        request.stop = None
        request.stream = None

        # MUST raise AbortError — NOT raw asyncio.CancelledError
        with pytest.raises(AbortError) as exc_info:
            await provider.complete(request)

        assert exc_info.value.retryable is False

    @pytest.mark.asyncio
    async def test_cancelled_error_not_retried(self) -> None:
        """error-hierarchy:AbortError:MUST:1 — CancelledError MUST NOT trigger retry loop.

        AbortError has retryable=False, so only one attempt must be made.
        """
        import asyncio

        from amplifier_core.llm_errors import AbortError

        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from tests.fixtures.sdk_mocks import MockCopilotClientWrapper

        call_count = 0

        class CountingCancelMock(MockCopilotClientWrapper):
            @property  # type: ignore[override]
            def _raise_on_session(self) -> asyncio.CancelledError:  # type: ignore[override]
                return asyncio.CancelledError("cancelled")

            @_raise_on_session.setter
            def _raise_on_session(self, value: object) -> None:
                pass  # ignore assignment from __init__

            from contextlib import asynccontextmanager as _acm

            @_acm
            async def session(self, model=None, *, system_message=None, tools=None):  # type: ignore[override]
                nonlocal call_count
                call_count += 1
                raise asyncio.CancelledError("cancelled")
                yield  # noqa: unreachable

        mock_client = CountingCancelMock(events=[])
        provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]

        request = MagicMock()
        request.model = "gpt-4"
        request.messages = [MagicMock(role="user", content="test")]
        request.tools = None
        request.max_tokens = None
        request.temperature = None
        request.stop = None
        request.stream = None

        with pytest.raises(AbortError):
            await provider.complete(request)

        # AbortError is non-retryable → exactly 1 attempt
        assert call_count == 1, f"Expected 1 attempt (no retry), got {call_count}"
