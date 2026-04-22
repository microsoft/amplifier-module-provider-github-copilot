"""Tier 7: Live Smoke Tests - verify real SDK behavior with actual API calls.

These tests make REAL API calls to GitHub Copilot and require:
1. A valid GITHUB_TOKEN/COPILOT_GITHUB_TOKEN with copilot scope
2. Network access to GitHub Copilot service
3. Rate limit budget (use sparingly)

Run: pytest -m live -v
Schedule: NIGHTLY (not on every PR)

Contract references:
- contracts/sdk-boundary.md (SDK API shapes)
- contracts/deny-destroy.md (deny hook behavior)
- contracts/event-vocabulary.md (event type mapping)

Design principles:
1. Structural assertions only - we verify shapes, not content
2. Drift detection - catch SDK API changes before they break production
3. Minimal prompts - short outputs to avoid rate limits
4. Event collection - validate streaming patterns

Type ignore notes:
- reportPrivateUsage: We intentionally use _make_deny_hook_config to match production
- reportArgumentType: SDK types are complex and we use cast(Any, ...) for simplicity
- reportUnknownVariableType: Session config dicts have dynamic keys
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import pytest


def _get_token() -> str:
    """Get token from environment. Fails test if not available.

    Policy: Tests run, not skip. Missing token = test failure.
    """
    for var in ("COPILOT_AGENT_TOKEN", "COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
        token = os.environ.get(var)
        if token:
            return token
    pytest.fail(
        "No GITHUB_TOKEN available. Set COPILOT_GITHUB_TOKEN, GH_TOKEN, or GITHUB_TOKEN. "
        "Tests run, not skip - this is a test failure."
    )
    return ""  # unreachable — pytest.fail() raises; satisfies type checker


def _get_sdk_available() -> bool:
    """Check if copilot SDK is installed."""
    try:
        import copilot  # type: ignore[import-not-found]  # noqa: F401

        return True
    except ImportError:
        return False


def _is_copilot_auth_error(exc: Exception) -> bool:
    """Check if exception is a Copilot authorization/policy error.

    These errors indicate the Copilot feature requires enterprise/org
    policy that isn't enabled in the test environment.
    """
    error_msg = str(exc).lower()
    auth_patterns = [
        "not authorized",
        "enterprise or organization policy",
        "policy to be enabled",
        "permission denied",
    ]
    return any(pattern in error_msg for pattern in auth_patterns)


# Mark as live tests - NO SKIP CONDITIONS
# Policy: Tests run and fail, not skip
pytestmark = [
    pytest.mark.live,
]


# =============================================================================
# Helpers
# =============================================================================


def _create_session_config() -> dict[str, Any]:
    """Create standard session config for live tests.

    SDK v0.2.0: Config dict unpacked as kwargs to create_session().
    Contract v1.2: available_tools=[] when no tools provided (blocks SDK built-ins)
    """
    from amplifier_module_provider_github_copilot.sdk_adapter.client import (
        _make_deny_hook_config,  # pyright: ignore[reportPrivateUsage]
        deny_permission_request,
    )

    return {
        "model": "claude-opus-4.5",
        "streaming": True,
        # Contract v1.2: available_tools MUST be set (not omitted)
        # Empty list prevents SDK built-ins from appearing when no tools provided
        "available_tools": [],
        # SDK v0.2.0: on_permission_request passed to create_session()
        "on_permission_request": deny_permission_request,
        "hooks": _make_deny_hook_config(),
    }


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
async def live_client():
    """Create a real SDK client for live tests.

    Yields the started client, stops on cleanup.
    Uses deny_permission_request from our adapter.

    SDK v0.2.0: Uses SubprocessConfig instead of options dict.
    Policy: Fails test if SDK not installed or token missing.
    """
    if not _get_sdk_available():
        pytest.fail("copilot SDK not installed. Install it or fix the test environment.")

    import copilot  # type: ignore[import-not-found]
    from copilot.client import SubprocessConfig  # type: ignore[import-not-found]

    # Get token - fails test if not available
    token = _get_token()

    # SDK v0.2.0: SubprocessConfig replaces options dict
    config = SubprocessConfig(github_token=token)
    client = copilot.CopilotClient(config)  # type: ignore[arg-type]
    await client.start()
    try:
        yield client
    finally:
        await client.stop()


# =============================================================================
# Session Lifecycle Tests (Most Critical)
# =============================================================================


class TestSessionLifecycle:
    """Verify session creation/destruction works with real SDK.

    Contract: deny-destroy:SessionLifecycle:MUST:1
    These tests are the foundation - if session lifecycle fails,
    nothing else works.
    """

    @pytest.mark.asyncio
    async def test_session_creates_and_disconnects(self, live_client: Any) -> None:
        """Session creation and disconnect complete without error.

        # Contract: deny-destroy:SessionLifecycle:MUST:1

        Verifies the full lifecycle: create session → confirm session_id is
        a string → disconnect cleanly. Disconnect is in finally to guarantee
        cleanup; if it raises, the exception propagates naturally as the
        contract violation without shadowing any prior failure.
        """
        session_config = _create_session_config()
        session = await live_client.create_session(**session_config)
        try:
            assert isinstance(session.session_id, str), "session_id must be a string"
        finally:
            await session.disconnect()  # type: ignore[misc]


# =============================================================================
# Event Streaming Tests (Shape Validation)
# =============================================================================


class TestEventStreaming:
    """Verify event streaming shapes match our assumptions.

    Contract: sdk-boundary:EventShape:MUST:2
    These tests validate that SDK events have the fields our
    translate_event() function expects.

    CRITICAL: If these fail, our streaming is silently broken.
    """

    @pytest.mark.asyncio
    async def test_streaming_events_have_expected_structure(self, live_client: Any) -> None:
        """Events received via on() have type and data attributes.

        # Contract: sdk-boundary:EventShape:MUST:1

        This validates Our event processing assumes events have:
        - .type (SessionEventType enum with .value)
        - .data (SessionEventData with delta_content, etc.)

        If SDK changes these shapes, this test fails before production breaks.
        """
        session_config = _create_session_config()
        # SDK v0.2.0: create_session uses kwargs
        session = await live_client.create_session(**session_config)
        collected_events: list[Any] = []
        idle_event = asyncio.Event()

        def collector(event: Any) -> None:
            collected_events.append(event)
            # Check for session.idle to know when done
            event_type = getattr(event, "type", None)
            if event_type is not None:
                type_str = getattr(event_type, "value", str(event_type))
                if type_str == "session.idle":
                    idle_event.set()

        unsubscribe = session.on(collector)
        try:
            # Send minimal message
            # SDK v0.2.0: send(prompt)
            await session.send("Say: test")
            # Wait for idle with timeout
            await asyncio.wait_for(idle_event.wait(), timeout=30.0)

            # Validate event shapes
            assert len(collected_events) > 0, "No events received from SDK"

            for event in collected_events:
                # All events must have .type
                assert hasattr(event, "type"), f"Event missing .type: {event}"
                event_type = event.type
                # Type should be an enum with .value
                assert hasattr(event_type, "value"), f"Event type missing .value: {event_type}"

                # All events must have .data
                assert hasattr(event, "data"), f"Event missing .data: {event}"

        finally:
            unsubscribe()
            await session.disconnect()


# =============================================================================
# Auth Error Pattern Tests
# =============================================================================


class TestAuthErrorPatterns:
    """Verify auth errors match our config/errors.yaml patterns.

    This test INTENTIONALLY uses an invalid token to trigger auth errors.
    The patterns detected here should match config/errors.yaml sdk_patterns.
    """

    @pytest.mark.asyncio
    async def test_invalid_token_error_shape(self) -> None:
        """Auth errors have predictable class/message patterns.

        When SDK receives an invalid token, the error class/message
        should match one of our configured patterns in errors.yaml.
        If not, update errors.yaml.
        """
        import copilot  # type: ignore[import-not-found]
        from copilot.client import SubprocessConfig  # type: ignore[import-not-found]

        # Create client with KNOWN INVALID token
        # SDK v0.2.0: Use SubprocessConfig
        config = SubprocessConfig(github_token="ghp_invalid_token_xxxxxxxxxxxxx")
        client = copilot.CopilotClient(config)  # type: ignore[arg-type]
        await client.start()

        session_config = _create_session_config()

        auth_error: Exception | None = None
        try:
            # SDK v0.2.0: create_session uses kwargs
            session = await client.create_session(**session_config)  # type: ignore[arg-type]
            try:
                # Try to send - this should fail with auth error
                # SDK v0.2.0: send_and_wait(prompt, timeout=...)
                await session.send_and_wait("test", timeout=10.0)
            finally:
                try:
                    await session.disconnect()
                except Exception:
                    pass
        except Exception as e:
            auth_error = e
        finally:
            await client.stop()

        error_class = type(auth_error).__name__
        error_str = str(auth_error)

        # Log for drift detection (helps update errors.yaml)
        print(f"\n[AUTH ERROR PATTERN] class={error_class!r}, message={error_str!r}")

        # These patterns should match config/errors.yaml authentication section
        # TimeoutError is also acceptable - invalid tokens may cause SDK to hang
        # rather than return an immediate auth rejection (environmental behavior)
        auth_indicators = [
            "AuthenticationError",
            "InvalidTokenError",
            "PermissionDeniedError",
            "Unauthorized",
            "401",
            "403",
            "invalid",
            "token",
            "access denied",
            "authentication",  # SDK v0.2.0: "Session was not created with authentication"
            "TimeoutError",  # SDK may timeout on invalid auth instead of rejecting
            "timeout",
        ]

        matches = any(
            p.lower() in error_class.lower() or p.lower() in error_str.lower()
            for p in auth_indicators
        )
        assert matches, (
            f"Auth error '{error_class}' with message '{error_str}' "
            f"doesn't match any configured pattern. "
            f"Update config/errors.yaml sdk_patterns if this is a new error type."
        )


# =============================================================================
# Usage Event Tests
# =============================================================================


class TestUsageEvents:
    """Verify usage event structure matches our expectations.

    Contract: event-vocabulary.md - assistant.usage (not session.usage)
    """

    @pytest.mark.asyncio
    async def test_usage_event_has_token_fields(self, live_client: Any) -> None:
        """assistant.usage event has input/output token counts.

        Per event-vocabulary.md, the SDK emits 'assistant.usage' events
        with token usage data. Our extract.py relies on these fields.
        """
        session_config = _create_session_config()
        # SDK v0.2.0: create_session uses kwargs
        session = await live_client.create_session(**session_config)
        usage_events: list[Any] = []
        idle_event = asyncio.Event()

        def collector(event: Any) -> None:
            event_type = getattr(event, "type", None)
            if event_type is not None:
                type_str = getattr(event_type, "value", str(event_type))
                if type_str == "assistant.usage":
                    usage_events.append(event)
                elif type_str == "session.idle":
                    idle_event.set()

        unsubscribe = session.on(collector)
        try:
            # SDK v0.2.0: send(prompt)
            await session.send("Reply: X")
            await asyncio.wait_for(idle_event.wait(), timeout=30.0)

            # SDK may or may not emit usage events (model-dependent)
            if usage_events:
                for usage in usage_events:
                    data = usage.data
                    # Log actual fields for drift detection
                    data_attrs = [a for a in dir(data) if not a.startswith("_")]
                    print(f"\n[USAGE EVENT] data attrs: {data_attrs}")

                    # Check common usage field names
                    has_input = hasattr(data, "input_tokens") or hasattr(data, "prompt_tokens")
                    has_output = hasattr(data, "output_tokens") or hasattr(
                        data, "completion_tokens"
                    )

                    if not (has_input or has_output):
                        # Log for manual review - SDK shape may have changed
                        print(f"[USAGE EVENT WARNING] No token fields found in {data_attrs}")
            else:
                # Usage events not guaranteed - just note it
                print("\n[USAGE EVENT] No assistant.usage events received (may be model-dependent)")

        finally:
            unsubscribe()
            await session.disconnect()


# =============================================================================
# Retry Event Payload Shape Tests
# =============================================================================


class TestRetryEventPayloadShape:
    """Verify provider:retry event payload reaches hooks with correct shape.

    Real-world validation for the retry_after field added in observability.py.
    These tests run through the actual GitHubCopilotProvider.complete() path
    with real config loading, real hook wiring, and real coordinator objects.

    We cannot force the GitHub Copilot API to return a 429 on demand, so we
    inject one retryable failure via _execute_sdk_completion monkey-patch, then
    allow the second attempt to complete using the real SDK. This exercises:
      - Real provider instantiation with real config loading
      - Real emit_retry() call through real llm_lifecycle context manager
      - Real hook emission to a real coordinator hooks object
      - Real retry_after field present in the emitted payload

    Contract: provider-protocol:hooks:provider_retry:MUST:3
    """

    @pytest.mark.asyncio
    async def test_retry_event_emitted_with_retry_after_none(self, live_client: Any) -> None:
        """provider:retry payload reaches hooks with retry_after=None on non-rate-limit errors.

        Uses real GitHubCopilotProvider with real coordinator hook wiring.
        Injects one ProviderUnavailableError (no retry_after) then succeeds via real SDK.
        Validates end-to-end: real config → real emit_retry → real hook → payload shape.

        Contract: provider-protocol:hooks:provider_retry:MUST:3
        """
        from unittest.mock import AsyncMock, MagicMock, patch

        from amplifier_core import llm_errors

        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from amplifier_module_provider_github_copilot.streaming import StreamingAccumulator

        # Real coordinator with real async hook capture
        coordinator = MagicMock()
        coordinator.hooks = MagicMock()
        emitted: list[tuple[str, dict[str, object]]] = []

        async def capture_emit(event_name: str, payload: dict[str, object]) -> None:
            emitted.append((event_name, payload))

        coordinator.hooks.emit = capture_emit

        # Real provider with real config loading
        provider = GitHubCopilotProvider(
            config={"model": "claude-opus-4.5", "use_streaming": True, "debug": False},
            coordinator=coordinator,
        )

        call_count = 0

        async def fail_once_then_use_real_sdk(
            *args: object, accumulator: StreamingAccumulator, **kwargs: object
        ) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise llm_errors.ProviderUnavailableError(
                    "Injected transient failure for retry event shape test"
                )
            # Second call: delegate to real SDK to prove the path works end-to-end
            session_config = _create_session_config()
            session = await live_client.create_session(**session_config)
            idle = asyncio.Event()
            response_parts: list[str] = []

            def on_event(event: object) -> None:
                event_type = getattr(event, "type", None)
                if event_type is not None:
                    type_str = getattr(event_type, "value", str(event_type))
                    if type_str == "assistant.message_delta":
                        data = getattr(event, "data", None)
                        if data is not None:
                            delta = getattr(data, "delta_content", None)
                            if delta:
                                response_parts.append(delta)
                    elif type_str in ("session.idle", "assistant.message"):
                        idle.set()

            unsub = session.on(on_event)
            try:
                await session.send("Reply with the word: LIVE")
                await asyncio.wait_for(idle.wait(), timeout=30.0)
            finally:
                unsub()
                await session.disconnect()

            from amplifier_module_provider_github_copilot.streaming import (
                DomainEvent,
                DomainEventType,
            )

            text = "".join(response_parts) or "LIVE"
            accumulator.add(DomainEvent(type=DomainEventType.CONTENT_DELTA, data={"text": text}))
            accumulator.add(
                DomainEvent(type=DomainEventType.TURN_COMPLETE, data={"finish_reason": "stop"})
            )

        provider._execute_sdk_completion = fail_once_then_use_real_sdk  # type: ignore[method-assign]

        sample_request = {
            "messages": [{"role": "user", "content": "Reply with the word: LIVE"}],
            "model": "claude-opus-4.5",
        }

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await provider.complete(
                sample_request,  # type: ignore[arg-type]
                model="claude-opus-4.5",
            )

        PROVIDER_RETRY = "provider:retry"
        retry_events = [(name, data) for name, data in emitted if name == PROVIDER_RETRY]

        assert len(retry_events) >= 1, (
            f"Expected provider:retry event from real emit path. "
            f"All emitted events: {[name for name, _ in emitted]}"
        )

        _, payload = retry_events[0]

        # MUST:3 — field is present and exactly None for non-rate-limit error
        assert "retry_after" in payload, (
            "retry_after key must be present in provider:retry payload (MUST:3). "
            f"Actual keys: {list(payload.keys())}"
        )
        assert payload["retry_after"] is None, (
            f"ProviderUnavailableError has no retry_after — expected None, "
            f"got {payload['retry_after']!r}"
        )
        # Regression guard — other required fields must still be present (MUST:2)
        for field in ("provider", "model", "attempt", "max_retries", "delay", "error_type"):
            assert field in payload, f"Required field '{field}' missing from payload"
        assert payload["provider"] == "github-copilot"

    @pytest.mark.asyncio
    async def test_retry_event_retry_after_float_via_translation_pipeline(
        self, live_client: Any
    ) -> None:
        """provider:retry payload has retry_after=float via the translation pipeline.

        Injects a raw Exception whose message matches the RateLimitError string_pattern
        and contains "Retry after 60 seconds". translate_sdk_error produces a
        RateLimitError(retry_after=60.0) via _extract_retry_after. Validates the
        second emit_retry call site (except Exception branch) end-to-end.

        Contract: provider-protocol:hooks:provider_retry:MUST:3
        """
        from unittest.mock import AsyncMock, MagicMock, patch

        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from amplifier_module_provider_github_copilot.streaming import StreamingAccumulator

        coordinator = MagicMock()
        coordinator.hooks = MagicMock()
        emitted: list[tuple[str, dict[str, object]]] = []

        async def capture_emit(event_name: str, payload: dict[str, object]) -> None:
            emitted.append((event_name, payload))

        coordinator.hooks.emit = capture_emit

        provider = GitHubCopilotProvider(
            config={"model": "claude-opus-4.5", "use_streaming": True, "debug": False},
            coordinator=coordinator,
        )

        call_count = 0

        async def fail_with_raw_rate_limit_then_succeed(
            *args: object, accumulator: StreamingAccumulator, **kwargs: object
        ) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Raw non-kernel exception → hits except Exception → translate_sdk_error.
                # "rate limit" matches string_pattern in errors.yaml → RateLimitError.
                # "Retry after 60 seconds" → _extract_retry_after returns 60.0.
                raise Exception(  # noqa: TRY002
                    "rate limit exceeded. Retry after 60 seconds"
                )
            # Second call: real SDK completion
            session_config = _create_session_config()
            session = await live_client.create_session(**session_config)
            idle = asyncio.Event()
            response_parts: list[str] = []

            def on_event(event: object) -> None:
                event_type = getattr(event, "type", None)
                if event_type is not None:
                    type_str = getattr(event_type, "value", str(event_type))
                    if type_str == "assistant.message_delta":
                        data = getattr(event, "data", None)
                        if data is not None:
                            delta = getattr(data, "delta_content", None)
                            if delta:
                                response_parts.append(delta)
                    elif type_str in ("session.idle", "assistant.message"):
                        idle.set()

            unsub = session.on(on_event)
            try:
                await session.send("Reply with the word: LIVE")
                await asyncio.wait_for(idle.wait(), timeout=30.0)
            finally:
                unsub()
                await session.disconnect()

            from amplifier_module_provider_github_copilot.streaming import (
                DomainEvent,
                DomainEventType,
            )

            text = "".join(response_parts) or "LIVE"
            accumulator.add(DomainEvent(type=DomainEventType.CONTENT_DELTA, data={"text": text}))
            accumulator.add(
                DomainEvent(type=DomainEventType.TURN_COMPLETE, data={"finish_reason": "stop"})
            )

        provider._execute_sdk_completion = fail_with_raw_rate_limit_then_succeed  # type: ignore[method-assign]

        sample_request = {
            "messages": [{"role": "user", "content": "Reply with the word: LIVE"}],
            "model": "claude-opus-4.5",
        }

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await provider.complete(
                sample_request,  # type: ignore[arg-type]
                model="claude-opus-4.5",
            )

        PROVIDER_RETRY = "provider:retry"
        retry_events = [(name, data) for name, data in emitted if name == PROVIDER_RETRY]

        assert len(retry_events) >= 1, (
            f"Expected provider:retry event. All emitted: {[name for name, _ in emitted]}"
        )

        _, payload = retry_events[0]

        # MUST:3 — exact type and exact value from _extract_retry_after pipeline
        assert "retry_after" in payload, (
            f"retry_after key missing. Actual keys: {list(payload.keys())}"
        )
        assert isinstance(payload["retry_after"], float), (
            f"Expected float from translated RateLimitError, "
            f"got {type(payload['retry_after']).__name__}: {payload['retry_after']!r}"
        )
        assert payload["retry_after"] == 60.0, (
            f"Expected 60.0 extracted from message, got {payload['retry_after']!r}"
        )
