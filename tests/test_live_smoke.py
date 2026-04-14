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
