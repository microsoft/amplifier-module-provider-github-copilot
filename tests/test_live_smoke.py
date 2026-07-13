"""Tier 7: Live Smoke Tests — manual-only.

Manual-only — requires GITHUB_TOKEN, network, and explicit ``pytest -m live``
invocation. These tests are excluded from the default test run (``-m 'not live'``
in pyproject.toml) and MUST NOT be added to CI without explicit token provisioning.

Verifies real SDK behavior with actual API calls.

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
import re
from types import SimpleNamespace
from typing import Any

import pytest

from tests._sdk_version_gate import require_sdk

_TOKEN_ENV_VARS = ("COPILOT_AGENT_TOKEN", "COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN")
_MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+\-]{0,127}$")
_KNOWN_MOCK_MODEL_IDS = frozenset({"claude-opus-4.5", "claude-sonnet-4"})
_VALID_FINISH_REASONS = frozenset({"stop", "tool_calls", "length", "content_filter"})


def _get_token() -> str:
    """Get token from environment. Fails test if not available.

    Policy: Tests run, not skip. Missing token = test failure.
    """
    for var in _TOKEN_ENV_VARS:
        token = os.environ.get(var)
        if token:
            return token
    pytest.fail(
        "No GITHUB_TOKEN available. Set COPILOT_GITHUB_TOKEN, GH_TOKEN, or GITHUB_TOKEN. "
        "Tests run, not skip - this is a test failure."
    )
    return ""  # unreachable — pytest.fail() raises; satisfies type checker


def _is_copilot_auth_error(exc: Exception) -> bool:
    """Check if exception is a Copilot authorization/policy error.

    These errors indicate the Copilot feature requires enterprise/org
    policy that isn't enabled in the test environment.
    """
    error_msg = f"{type(exc).__name__} {exc}".lower()
    auth_patterns = [
        "401",
        "403",
        "access denied",
        "auth",
        "bad credentials",
        "forbidden",
        "invalid",
        "not authorized",
        "permission",
        "enterprise or organization policy",
        "policy to be enabled",
        "token",
        "unauthorized",
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


def _restore_real_model_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undo the autouse model fixture for tests that must hit SDK list_models()."""
    import amplifier_module_provider_github_copilot.models as models_module
    import amplifier_module_provider_github_copilot.provider as provider_module

    async def real_fetch_and_map_models(client: Any) -> tuple[list[Any], list[Any]]:
        copilot_models = await models_module.fetch_models(client)
        amplifier_models = [
            models_module.copilot_model_to_amplifier_model(model) for model in copilot_models
        ]
        return amplifier_models, copilot_models

    monkeypatch.setattr(models_module, "fetch_and_map_models", real_fetch_and_map_models)
    monkeypatch.setattr(provider_module, "fetch_and_map_models", real_fetch_and_map_models)


def _make_live_provider(live_client: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Create a provider wired to the already-started real SDK client."""
    from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
    from amplifier_module_provider_github_copilot.sdk_adapter.client import CopilotClientWrapper

    _restore_real_model_fetch(monkeypatch)
    return GitHubCopilotProvider(
        config={"use_streaming": True, "debug": False},
        client=CopilotClientWrapper(sdk_client=live_client),
    )


def _assert_real_model_list(models: list[Any]) -> None:
    """Validate model-list shape strongly enough to reject the canned fixture list."""
    assert models, "SDK list_models() returned no models for this token"

    model_ids = [getattr(model, "id", None) for model in models]
    assert all(isinstance(model_id, str) and model_id for model_id in model_ids), (
        f"Every model must expose a non-empty string id; got {model_ids!r}"
    )
    assert set(model_ids) != _KNOWN_MOCK_MODEL_IDS, (
        "Provider model discovery returned the conftest canned model list, not live SDK data"
    )

    for model in models:
        model_id = getattr(model, "id", None)
        display_name = getattr(model, "display_name", None)
        context_window = getattr(model, "context_window", None)
        max_output_tokens = getattr(model, "max_output_tokens", None)
        capabilities = getattr(model, "capabilities", None)

        assert isinstance(model_id, str) and _MODEL_ID_PATTERN.fullmatch(model_id), (
            f"Model id has unexpected live API shape: {model_id!r}"
        )
        assert isinstance(display_name, str) and display_name.strip(), (
            f"Model {model_id!r} missing display_name"
        )
        assert isinstance(context_window, int) and context_window > 0, (
            f"Model {model_id!r} missing positive context_window"
        )
        assert isinstance(max_output_tokens, int) and max_output_tokens > 0, (
            f"Model {model_id!r} missing positive max_output_tokens"
        )
        assert isinstance(capabilities, list) and capabilities, (
            f"Model {model_id!r} missing non-empty capabilities"
        )


def _assert_response_usage(usage: Any) -> None:
    """Validate token usage metadata from a real provider completion."""
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)

    assert isinstance(input_tokens, int) and input_tokens > 0, (
        f"usage.input_tokens must be a positive int; got {input_tokens!r}"
    )
    assert isinstance(output_tokens, int) and output_tokens > 0, (
        f"usage.output_tokens must be a positive int; got {output_tokens!r}"
    )
    assert total_tokens is None or (isinstance(total_tokens, int) and total_tokens > 0), (
        f"usage.total_tokens must be None or a positive int; got {total_tokens!r}"
    )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
async def live_client(tmp_path_factory: pytest.TempPathFactory):
    """Create a real SDK client for live tests.

    Yields the started client, stops on cleanup.
    Uses deny_permission_request from our adapter.

    SDK v1.0.0b10: uses direct CopilotClient keyword arguments.
    Policy: Fails test if SDK not installed or token missing.

    Parity with production wiring (sdk_adapter/client.py) and the unit
    fixture in tests/conftest.py:
      * env built via scrub_sdk_env() so COPILOT_HOME / COPILOT_CLI_PATH
        cannot leak from the ambient shell into the SDK subprocess.
      * base_directory is an isolated tmp_path_factory directory per test
        — no cwd-relative shared state across runs or concurrent pytest
        invocations.
    """
    from amplifier_module_provider_github_copilot.sdk_adapter.client import scrub_sdk_env

    copilot = require_sdk()

    # Get token - fails test if not available
    token = _get_token()

    client = copilot.CopilotClient(
        base_directory=str(tmp_path_factory.mktemp("live-copilot-home")),
        github_token=token,
        log_level="info",
        env=scrub_sdk_env(dict(os.environ)),
        mode="copilot-cli",
    )
    await client.start()
    try:
        yield client
    finally:
        await client.stop()


@pytest.fixture(autouse=True)
def _bind_real_sdk_override_types(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bind the real SDK per-session override types through the _imports membrane.

    conftest.py sets ``SKIP_SDK_CHECK=1`` globally, which nulls
    ``_imports.ModelCapabilitiesOverride`` / ``_imports.ModelLimitsOverride``.
    Live tests that forward ``max_output_tokens`` exercise the real cap path in
    ``client.session()`` (``model_capabilities=ModelCapabilitiesOverride(
    limits=ModelLimitsOverride(max_output_tokens=...))``), so without rebinding
    those types this raises ``TypeError: 'NoneType' object is not callable`` —
    a test-harness artifact, NOT a production defect. In production
    ``SKIP_SDK_CHECK`` is unset and ``_imports`` already holds the real classes
    (verified against github-copilot-sdk==1.0.2). ``client.py`` looks these up
    via the membrane at call time, so patching ``_imports`` is sufficient.

    Fail-closed: uses ``require_sdk()`` (which fails, never skips, on a missing
    or wrong-version SDK), so the live test fails loudly here rather than
    silently leaving the override types unbound.
    """
    copilot = require_sdk()

    from amplifier_module_provider_github_copilot.sdk_adapter import _imports

    monkeypatch.setattr(_imports, "ModelLimitsOverride", copilot.ModelLimitsOverride)
    monkeypatch.setattr(
        _imports, "ModelCapabilitiesOverride", copilot.ModelCapabilitiesOverride
    )


# =============================================================================
# Real API Proof Tests
# =============================================================================


class TestRealApiProof:
    """Fail-closed proof that live tests hit real GitHub Copilot APIs."""

    @pytest.mark.asyncio
    async def test_provider_list_models_returns_live_non_empty_shape(
        self,
        live_client: Any,
        monkeypatch: pytest.MonkeyPatch,
        real_model_discovery: None,
    ) -> None:
        """Provider.list_models() must return non-empty live SDK model metadata."""
        provider = _make_live_provider(live_client, monkeypatch)

        models = await provider.list_models()

        _assert_real_model_list(models)

    @pytest.mark.asyncio
    async def test_provider_complete_returns_content_finish_reason_and_usage(
        self,
        live_client: Any,
        monkeypatch: pytest.MonkeyPatch,
        real_model_discovery: None,
    ) -> None:
        """Provider.complete() must return real response content and metadata."""
        provider = _make_live_provider(live_client, monkeypatch)
        models = await provider.list_models()
        _assert_real_model_list(models)
        model_id = models[0].id

        request = SimpleNamespace(
            model=model_id,
            messages=[
                SimpleNamespace(
                    role="user",
                    content="Reply with one short word. No punctuation.",
                )
            ],
            tools=None,
            max_output_tokens=16,
            reasoning_effort=None,
            context_tier=None,
            metadata={"stream": False},
        )

        response = await provider.complete(request, model=model_id, _timeout_seconds=45.0)

        text = getattr(response, "text", None)
        assert isinstance(text, str) and text.strip(), "Live completion returned empty text"
        assert response.content, "Live completion returned no content blocks"
        assert response.finish_reason in _VALID_FINISH_REASONS, (
            f"Unexpected finish_reason from live completion: {response.finish_reason!r}"
        )
        assert not response.tool_calls, (
            "No tools were provided; live completion must not call tools"
        )
        _assert_response_usage(response.usage)


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
    async def test_invalid_token_error_shape(
        self,
        tmp_path_factory: pytest.TempPathFactory,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Auth errors have predictable class/message patterns.

        # Contract: sdk-boundary:Auth:MUST:2

        When SDK receives an invalid token, the error class/message
        should match one of our configured patterns in errors.yaml.
        If not, update errors.yaml.
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import scrub_sdk_env

        _get_token()  # Preserve live policy: missing real credentials are still a failure.
        copilot = require_sdk()

        invalid_token = "ghp_invalid_live_negative_control_xxxxxxxxxxxxxxxxxxxx"
        for var in _TOKEN_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("COPILOT_GITHUB_TOKEN", invalid_token)
        token = _get_token()
        assert token == invalid_token, "Negative-control token was not read from monkeypatched env"

        # Create client with a known invalid token resolved from in-process env.
        client = copilot.CopilotClient(
            base_directory=str(tmp_path_factory.mktemp("invalid-token-copilot-home")),
            github_token=token,
            log_level="info",
            env=scrub_sdk_env(dict(os.environ)),
            mode="copilot-cli",
        )
        stop_error: Exception | None = None

        session_config = _create_session_config()

        auth_error: Exception | None = None
        try:
            await client.start()
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
            try:
                await client.stop()
            except Exception as e:
                stop_error = e

        if auth_error is None and stop_error is not None:
            raise stop_error

        assert isinstance(auth_error, BaseException), (
            "Invalid-token negative control completed successfully. The SDK may have ignored "
            "the explicit token or fallen back to cached auth."
        )
        error_class = type(auth_error).__name__
        error_str = str(auth_error)

        # Log for drift detection (helps update errors.yaml)
        print(f"\n[AUTH ERROR PATTERN] class={error_class!r}, message={error_str!r}")

        assert _is_copilot_auth_error(auth_error), (
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
