"""Tests for ``reasoning_effort`` plumbing and event classification gaps.

Contract: provider-protocol:complete:MUST:11
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from amplifier_module_provider_github_copilot._compat import ConfigurationError
from amplifier_module_provider_github_copilot.request_adapter import (
    convert_chat_request,
    validate_reasoning_effort,
)
from amplifier_module_provider_github_copilot.sdk_adapter import CopilotModelInfo
from tests._sdk_version_gate import require_sdk

# ----------------------------------------------------------------------------
# Test fixtures
# ----------------------------------------------------------------------------


@dataclass
class _Msg:
    role: str
    content: str


@dataclass
class _Req:
    """Minimal duck-typed kernel ChatRequest stand-in."""

    messages: list[_Msg] = field(default_factory=list)
    model: str | None = None
    tools: list[Any] = field(default_factory=list)
    max_output_tokens: int | None = None
    reasoning_effort: str | None = None


def _model_info(
    *,
    supports: bool = True,
    allowlist: tuple[str, ...] = ("low", "medium", "high"),
) -> CopilotModelInfo:
    return CopilotModelInfo(
        id="claude-sonnet-4.6",
        name="Sonnet 4.6",
        context_window=200_000,
        max_output_tokens=8192,
        supports_reasoning_effort=supports,
        supported_reasoning_efforts=allowlist,
    )


# ----------------------------------------------------------------------------
# T5: convert_chat_request copies the field (no validation here)
# ----------------------------------------------------------------------------


class TestConvertChatRequestCarriesReasoningEffort:
    """convert_chat_request preserves reasoning_effort on CompletionRequest.

    Contract: provider-protocol:complete:MUST:11
    """

    def test_none_passes_through_as_none(self) -> None:
        req = _Req(messages=[_Msg("user", "hi")], reasoning_effort=None)
        out = convert_chat_request(req)
        assert out.reasoning_effort is None

    def test_value_preserved_verbatim(self) -> None:
        req = _Req(messages=[_Msg("user", "hi")], reasoning_effort="medium")
        out = convert_chat_request(req)
        assert out.reasoning_effort == "medium"

    def test_empty_string_normalized_to_none(self) -> None:
        """Empty-string is treated as None (no effort requested) per MUST:11."""
        req = _Req(messages=[_Msg("user", "hi")], reasoning_effort="")
        out = convert_chat_request(req)
        assert out.reasoning_effort is None

    def test_missing_attribute_is_none(self) -> None:
        """getattr-with-default tolerates older kernels lacking the field."""

        class _Bare:
            messages = [_Msg("user", "hi")]
            model = None
            tools: list[Any] = []
            max_output_tokens = None
            # No reasoning_effort attribute at all.

        out = convert_chat_request(_Bare())
        assert out.reasoning_effort is None

    def test_non_string_value_raises_configuration_error(self) -> None:
        """Non-str non-None reasoning_effort on ChatRequest must surface
        loudly rather than silently normalize to None. Pinned at the adapter
        boundary in ``convert_chat_request``."""
        bogus_request = SimpleNamespace(
            messages=[SimpleNamespace(role="user", content="hi")],
            model="claude-sonnet-4.6",
            tools=None,
            system=None,
            max_output_tokens=None,
            reasoning_effort=42,  # int, not str
        )
        with pytest.raises(ConfigurationError) as excinfo:
            convert_chat_request(bogus_request)
        msg = str(excinfo.value)
        assert "reasoning_effort" in msg
        assert "int" in msg


# ----------------------------------------------------------------------------
# T6/T7/T8 + edge cases: validate_reasoning_effort gate
# ----------------------------------------------------------------------------


class TestResolveReasoningEffortGate:
    """Layer-1 capability gate (provider-protocol:complete:MUST:11)."""

    def test_returns_value_when_supported_and_in_allowlist(self) -> None:
        info = _model_info()
        result = validate_reasoning_effort("medium", info, model_id="claude-sonnet-4.6")
        assert result == "medium"

    def test_returns_value_when_supported_with_no_allowlist(self) -> None:
        """Empty allowlist means SDK will validate; provider passes through."""
        info = _model_info(allowlist=())
        result = validate_reasoning_effort("medium", info, model_id="claude-sonnet-4.6")
        assert result == "medium"

    def test_returns_none_when_input_none(self) -> None:
        info = _model_info(supports=False)
        # No request → no gate trigger, even on unsupported model.
        assert validate_reasoning_effort(None, info, model_id="m") is None

    def test_returns_none_when_input_empty_string(self) -> None:
        info = _model_info(supports=False)
        assert validate_reasoning_effort("", info, model_id="m") is None

    def test_raises_when_model_does_not_support(self) -> None:
        info = _model_info(supports=False, allowlist=())
        with pytest.raises(ConfigurationError) as excinfo:
            validate_reasoning_effort("medium", info, model_id="claude-haiku-4.5")
        msg = str(excinfo.value)
        assert "claude-haiku-4.5" in msg
        assert "does not support" in msg
        assert "reasoning_effort" in msg

    def test_raises_when_value_not_in_allowlist(self) -> None:
        info = _model_info(allowlist=("low", "medium", "high"))
        with pytest.raises(ConfigurationError) as excinfo:
            validate_reasoning_effort("banana", info, model_id="claude-sonnet-4.6")
        msg = str(excinfo.value)
        assert "claude-sonnet-4.6" in msg
        assert "banana" in msg
        # Allowed values must be enumerated for diagnosability.
        assert "'low'" in msg and "'medium'" in msg and "'high'" in msg

    def test_mixed_case_rejected_no_silent_normalization(self) -> None:
        """SDK Literal is strictly lowercase; reject mixed-case explicitly.

        Defense-in-depth: mixed-case values do not match the well-formed
        token regex ([a-z][a-z_]{0,15}), so the rejected value is rendered
        as ``<redacted; len=N>`` rather than echoed verbatim. This prevents
        an injected secret fragment from leaking via the error text.
        """
        info = _model_info(allowlist=("low", "medium", "high"))
        with pytest.raises(ConfigurationError) as excinfo:
            validate_reasoning_effort("Medium", info, model_id="claude-sonnet-4.6")
        msg = str(excinfo.value)
        # Mixed-case is rejected AND redacted (len=6 placeholder, not the
        # raw "Medium") because uppercase fails the well-formed-token regex.
        assert "<redacted; len=6>" in msg
        assert "Medium" not in msg, (
            "rejected non-token reasoning_effort must NOT be echoed verbatim"
        )
        assert "claude-sonnet-4.6" in msg
        # Mutation guard: pin the exact rejection wording so a regression that
        # silently lower-cases the input or drops the case-sensitivity hint
        # turns this test red instead of green.
        assert "SDK literal allowlist" in msg
        assert "case-sensitive" in msg

    def test_overlong_value_rejected_via_allowlist_with_redaction(self) -> None:
        """Overlong values are rejected by the universal allowlist check; the
        redactor renders them as ``<redacted; len=N>`` so the value never
        appears verbatim in the error."""
        info = _model_info()
        oversize = "x" * 200
        with pytest.raises(ConfigurationError) as excinfo:
            validate_reasoning_effort(oversize, info, model_id="claude-sonnet-4.6")
        msg = str(excinfo.value)
        assert "SDK literal allowlist" in msg
        assert "<redacted; len=200>" in msg
        assert oversize not in msg

    def test_model_info_none_defers_to_layer2_with_info_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Cache miss with a value in the fallback allowlist must NOT raise;
        defer to SDK Layer-2 backstop with an INFO log for traceability."""
        with caplog.at_level(
            logging.INFO,
            logger="amplifier_module_provider_github_copilot.request_adapter",
        ):
            result = validate_reasoning_effort("medium", None, model_id="brand-new-model")
        assert result == "medium"
        assert any(
            "deferring final reasoning_effort validation to SDK backstop" in rec.message
            and "brand-new-model" in rec.message
            for rec in caplog.records
        )

    def test_model_info_none_with_bogus_value_raises(self) -> None:
        """Cache miss + value not in the SDK literal allowlist must raise.

        Without this gate, an arbitrary <=16-char string would silently reach
        the SDK whenever ``CopilotModelInfo`` is unavailable, defeating the
        Layer-1 capability gate.
        """
        with pytest.raises(ConfigurationError) as excinfo:
            validate_reasoning_effort("frobozz", None, model_id="brand-new-model")
        msg = str(excinfo.value)
        assert "frobozz" in msg
        assert "brand-new-model" in msg
        assert "SDK literal allowlist" in msg
        # Must enumerate accepted values so the caller can self-correct.
        for v in ("low", "medium", "high", "xhigh"):
            assert f"'{v}'" in msg

    @pytest.mark.parametrize("bad_value", ["High", "MEDIUM", "Low", "xHigh"])
    def test_mixed_case_rejected_when_supported_efforts_empty(self, bad_value: str) -> None:
        """Contract: provider-protocol:complete:MUST:11

        Universal shape gate must reject mixed-case values even when the
        cached ``CopilotModelInfo`` advertises ``supports_reasoning_effort=True``
        with an empty ``supported_reasoning_efforts`` tuple. Without the
        unconditional shape check, mixed-case strings would short-circuit the
        per-model allowlist guard (``if allowlist and ...``) and reach the SDK,
        producing a remote error instead of a clean ``ConfigurationError``.
        """
        from amplifier_module_provider_github_copilot.sdk_adapter import (
            CopilotModelInfo,
        )

        info = CopilotModelInfo(
            id="brand-x",
            name="Brand X",
            context_window=128_000,
            max_output_tokens=8192,
            supports_vision=False,
            supports_reasoning_effort=True,
            supported_reasoning_efforts=(),
        )
        with pytest.raises(ConfigurationError) as excinfo:
            validate_reasoning_effort(bad_value, info, model_id="brand-x")
        msg = str(excinfo.value)
        # Value is intentionally redacted (short-token leakage mitigation);
        # assert structural signal instead of verbatim echo.
        assert "<redacted" in msg
        assert "brand-x" in msg
        assert "SDK literal allowlist" in msg
        assert "case-sensitive" in msg

    def test_overlong_value_message_does_not_echo_value(self) -> None:
        """Defense in depth: overlong rejected values must NOT be echoed
        verbatim into the error message (could carry a token fragment).
        The redactor renders them as ``<redacted; len=N>`` and the universal
        allowlist gate raises before per-model checks see the raw value."""
        info = _model_info()
        secret_like = "ghp_" + "x" * 36  # mimics a GitHub token shape, len=40
        with pytest.raises(ConfigurationError) as excinfo:
            validate_reasoning_effort(secret_like, info, model_id="claude-sonnet-4.6")
        msg = str(excinfo.value)
        assert "SDK literal allowlist" in msg
        assert "<redacted; len=40>" in msg
        # The raw secret-shaped value MUST NOT appear in the error text.
        assert secret_like not in msg
        assert "ghp_" not in msg


# ----------------------------------------------------------------------------
# T9: client.session forwards reasoning_effort to SDK create_session
# ----------------------------------------------------------------------------


class TestSessionForwardsReasoningEffort:
    """client.session() must pass reasoning_effort to SDK create_session.

    Contract: provider-protocol:complete:MUST:11

    Mutation check: removing the `if reasoning_effort is not None:` block in
    sdk_adapter/client.py makes the value never reach create_session — red.

    Behavioral assertions cover three axes:
      1. Forwarding (value reaches the SDK kwarg).
      2. Omission (None means absent kwarg, not ``reasoning_effort=None``).
      3. Lifecycle (the SDK session is created, used, and torn down via
         ``disconnect`` so the full happy path is exercised — not just the
         create_session call site).
    """

    @pytest.mark.parametrize(
        "effort_value",
        ["low", "medium", "high", "xhigh"],
    )
    @pytest.mark.asyncio
    async def test_value_reaches_sdk_create_session_kwargs(self, effort_value: str) -> None:
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        wrapper = CopilotClientWrapper()
        sdk_client = MagicMock()
        fake_sdk_session = MagicMock()
        fake_sdk_session.session_id = "sid"
        fake_sdk_session.disconnect = AsyncMock()
        sdk_client.create_session = AsyncMock(return_value=fake_sdk_session)

        async def _fake_ensure(caller: str = "session") -> Any:  # noqa: ARG001
            return sdk_client

        wrapper._ensure_client_initialized = _fake_ensure  # type: ignore[assignment]  # noqa: SLF001

        async with wrapper.session(
            model="claude-sonnet-4.6", reasoning_effort=effort_value
        ) as handle:
            # Direct attribute touch — session_id access fails loud (AttributeError
            # on None / non-session) without a weak `is not None` placeholder.
            assert handle.session_id == "sid", (
                f"session context yielded unexpected handle: {handle!r}"
            )

        # Forwarding assertion
        kwargs = sdk_client.create_session.call_args.kwargs
        assert kwargs.get("reasoning_effort") == effort_value, (
            f"Expected reasoning_effort={effort_value!r} on create_session, got kwargs={kwargs!r}"
        )

        # Lifecycle assertion: SDK session was created exactly once and
        # torn down via disconnect. Removing the ``finally: disconnect()``
        # branch from the wrapper would leak — this catches it.
        assert sdk_client.create_session.await_count == 1, (
            f"Expected exactly 1 create_session call, got {sdk_client.create_session.await_count}"
        )
        assert fake_sdk_session.disconnect.await_count == 1, (
            f"Expected exactly 1 disconnect() on session teardown, got "
            f"{fake_sdk_session.disconnect.await_count}"
        )

    @pytest.mark.asyncio
    async def test_none_omits_kwarg_entirely(self) -> None:
        """When reasoning_effort is None, SDK kwarg MUST be absent (not None)."""
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        wrapper = CopilotClientWrapper()
        sdk_client = MagicMock()
        fake_sdk_session = MagicMock()
        fake_sdk_session.session_id = "sid"
        fake_sdk_session.disconnect = AsyncMock()
        sdk_client.create_session = AsyncMock(return_value=fake_sdk_session)

        async def _fake_ensure(caller: str = "session") -> Any:  # noqa: ARG001
            return sdk_client

        wrapper._ensure_client_initialized = _fake_ensure  # type: ignore[assignment]  # noqa: SLF001

        async with wrapper.session(model="gpt-4", reasoning_effort=None):
            pass

        kwargs = sdk_client.create_session.call_args.kwargs
        assert "reasoning_effort" not in kwargs, (
            f"Expected reasoning_effort kwarg absent, got: {kwargs.get('reasoning_effort')!r}"
        )
        # Disconnect lifecycle still exercised on the None path.
        assert fake_sdk_session.disconnect.await_count == 1, (
            f"Expected disconnect() even when reasoning_effort is None, got "
            f"{fake_sdk_session.disconnect.await_count}"
        )


# ----------------------------------------------------------------------------
# T10: provider.complete threads reasoning_effort through BOTH call sites
# ----------------------------------------------------------------------------


def _make_capturing_wrapper(fake_text: str, clean_text: str) -> tuple[Any, list[dict[str, Any]]]:
    """Return (wrapper, session_calls). Call 1 emits fake_text, call 2 clean_text."""
    from tests.fixtures.sdk_mocks import (
        MockSDKSession,
        SessionEvent,
        SessionEventData,
        SessionEventType,
        idle_event,
    )

    call_index_cell: list[int] = [0]
    session_calls: list[dict[str, Any]] = []

    class _CapturingWrapper:
        copilot_pid: str | None = None

        @asynccontextmanager
        async def session(
            self,
            model: str | None = None,
            *,
            system_message: str | None = None,
            tools: list[Any] | None = None,
            max_tokens: int | None = None,
            reasoning_effort: str | None = None,
            context_tier: str | None = None,
        ) -> AsyncIterator[Any]:
            call_index_cell[0] += 1
            idx = call_index_cell[0]
            session_calls.append(
                {
                    "call": idx,
                    "max_tokens": max_tokens,
                    "reasoning_effort": reasoning_effort,
                }
            )
            text = fake_text if idx == 1 else clean_text
            delta = SessionEvent(
                type=SessionEventType.ASSISTANT_MESSAGE_DELTA,
                data=SessionEventData(delta_content=text),
            )
            sess = MockSDKSession(events=[delta, idle_event()])
            try:
                yield sess
            finally:
                await sess.disconnect()

    return _CapturingWrapper(), session_calls


class TestCompleteThreadsReasoningEffortToBothCallSites:
    """provider.complete must forward reasoning_effort identically on the
    main path AND the fake-tool correction retry path.

    Contract: provider-protocol:complete:MUST:11

    Mutation check: dropping the `reasoning_effort=validated_reasoning_effort`
    kwarg from EITHER `_execute_sdk_completion` invocation in provider.py
    makes one of the captured session calls show None — red.
    """

    @staticmethod
    def _make_request(reasoning_effort: str | None) -> MagicMock:
        request = MagicMock()
        request.model = "gpt-4o"
        request.messages = [MagicMock(role="user", content="list files")]
        request.attachments = None
        request.max_output_tokens = None
        request.reasoning_effort = reasoning_effort
        request.context_tier = None
        request.tools = [{"name": "bash", "description": "Run shell commands", "parameters": {}}]
        return request

    @pytest.mark.asyncio
    async def test_correction_retry_sees_same_reasoning_effort_as_main(
        self,
    ) -> None:
        from amplifier_core import ChatResponse

        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        wrapper, session_calls = _make_capturing_wrapper(
            fake_text="[Tool Call: bash(command='ls')]",
            clean_text="ok",
        )
        provider = GitHubCopilotProvider(client=wrapper)  # type: ignore[arg-type]
        result = await provider.complete(self._make_request("medium"))

        assert len(session_calls) == 2, (
            f"Expected main + correction = 2 session calls, got {session_calls!r}"
        )
        assert session_calls[0]["reasoning_effort"] == "medium", (
            f"Main session: reasoning_effort lost — got {session_calls[0]['reasoning_effort']!r}"
        )
        assert session_calls[1]["reasoning_effort"] == "medium", (
            f"Correction session: reasoning_effort dropped on retry — got "
            f"{session_calls[1]['reasoning_effort']!r}; both call sites in "
            f"provider.py MUST forward the validated value identically."
        )
        assert isinstance(result, ChatResponse)

    @pytest.mark.asyncio
    async def test_none_threads_through_as_none_on_both_sites(self) -> None:
        from amplifier_core import ChatResponse

        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        wrapper, session_calls = _make_capturing_wrapper(
            fake_text="[Tool Call: bash(command='ls')]",
            clean_text="ok",
        )
        provider = GitHubCopilotProvider(client=wrapper)  # type: ignore[arg-type]
        result = await provider.complete(self._make_request(None))

        assert len(session_calls) == 2
        assert session_calls[0]["reasoning_effort"] is None
        assert session_calls[1]["reasoning_effort"] is None
        assert isinstance(result, ChatResponse)


# ----------------------------------------------------------------------------
# Provider-level integration test for unsupported cached model
# ----------------------------------------------------------------------------


class TestProviderRaisesOnUnsupportedCachedModel:
    """``provider.complete()`` must raise ``ConfigurationError`` BEFORE any
    SDK ``create_session`` call when the cached ``CopilotModelInfo`` declares
    ``supports_reasoning_effort=False`` and the caller passes a non-empty
    ``reasoning_effort``.

    Pins the wiring contract (gate runs in ``provider.complete()`` before
    ``_execute_sdk_completion``); the unit tests for ``validate_reasoning_effort``
    only cover the function in isolation.

    Mutation check: comment out the ``validate_reasoning_effort(...)`` call in
    ``provider.complete()`` (or move it after ``_execute_sdk_completion``) and
    this test goes red because the SDK wrapper is invoked.

    Contract: provider-protocol:complete:MUST:11
    Contract: observability:Events:MUST:6 (pre-flight emission exemption)
    """

    @pytest.mark.asyncio
    async def test_unsupported_cached_model_raises_before_sdk_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from amplifier_module_provider_github_copilot import provider as provider_mod
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )
        from amplifier_module_provider_github_copilot.sdk_adapter import (
            CopilotModelInfo,
        )

        # Cached capability descriptor: model is known but explicitly does
        # NOT support reasoning_effort. This is the "stale or correct cache"
        # path where Layer-1 must catch the misuse locally.
        unsupported_info = CopilotModelInfo(
            id="claude-haiku-4.5",
            name="Claude Haiku 4.5",
            context_window=200_000,
            max_output_tokens=8192,
            supports_vision=False,
            supports_reasoning_effort=False,
            supported_reasoning_efforts=(),
        )

        # Pin the lookup so the gate sees our hand-crafted descriptor without
        # touching the on-disk cache or the live API.
        monkeypatch.setattr(
            provider_mod.GitHubCopilotProvider,
            "_lookup_copilot_model_info",
            lambda self, model_id: unsupported_info,
        )

        # SDK wrapper must NOT be invoked. Mirror the production interface
        # (``client.session(...)`` async context manager at provider.py:871) so
        # a regression fires our explicit AssertionError, not an incidental
        # AttributeError. ``create_session`` is also instrumented to keep this
        # test useful if the provider switches call shape later.
        sdk_session_calls: list[dict[str, Any]] = []

        class _FailIfCalledClient:
            def session(self, **kwargs: Any) -> Any:
                sdk_session_calls.append(kwargs)
                raise AssertionError(
                    "SDK client.session() was invoked despite Layer-1 gate; "
                    "validate_reasoning_effort must raise BEFORE any SDK "
                    "session is opened. provider.complete() likely lost the "
                    "validate_reasoning_effort(...) call or moved it after "
                    "_execute_sdk_completion."
                )

            async def create_session(self, **kwargs: Any) -> Any:
                # Defense in depth: if the production code path ever switches
                # back to direct create_session(), this still trips.
                sdk_session_calls.append(kwargs)
                raise AssertionError(
                    "SDK client.create_session() was invoked despite "
                    "Layer-1 gate; same contract violation as session()."
                )

            async def stop(self) -> None:
                return None

        request = MagicMock()
        request.model = "claude-haiku-4.5"
        request.messages = [MagicMock(role="user", content="hello")]
        request.attachments = None
        request.max_output_tokens = None
        request.reasoning_effort = "high"
        request.context_tier = None
        request.tools = None

        # Coordinator wired with an AsyncMock hooks.emit so we can pin the
        # observability:Events:MUST:6 exemption: pre-flight ConfigurationError
        # MUST NOT emit llm:request or llm:response. The pair invariant
        # frames the SDK call; the SDK was never reached here.
        from unittest.mock import AsyncMock

        coordinator = MagicMock()
        coordinator.hooks = MagicMock()
        coordinator.hooks.emit = AsyncMock()

        provider = GitHubCopilotProvider(
            client=_FailIfCalledClient(),  # type: ignore[arg-type]
            coordinator=coordinator,
        )

        with pytest.raises(ConfigurationError) as excinfo:
            await provider.complete(request)

        msg = str(excinfo.value)
        assert "claude-haiku-4.5" in msg
        assert "does not support reasoning_effort" in msg
        # Defense-in-depth: SDK MUST NOT have been touched.
        assert sdk_session_calls == [], (
            f"Layer-1 gate bypassed; SDK session was opened with: {sdk_session_calls!r}"
        )
        # Contract: observability:Events:MUST:6. Pre-flight failure emits no
        # llm:request and no llm:response. Operators tracking caller-bug rates
        # consume the [REQUEST_ADAPTER] log channel, not the request/response
        # pair counter. If a future refactor moves validation INSIDE
        # llm_lifecycle, MUST:6 needs to flip and this assertion must follow.
        emitted_event_names = [call.args[0] for call in coordinator.hooks.emit.call_args_list]
        forbidden = {"llm:request", "llm:response"}
        leaked = [name for name in emitted_event_names if name in forbidden]
        assert not leaked, (
            f"observability:Events:MUST:6 violated: pre-flight "
            f"ConfigurationError emitted {leaked!r}. The SDK call never "
            f"happened, so the request/response pair invariant does not apply."
        )


# ----------------------------------------------------------------------------
# Layer-2 SDK-reject live integration test
# ----------------------------------------------------------------------------


class TestLayer2SDKRejectMatchesErrorTranslation:
    """When Layer-1 is bypassed (cache miss with value in fallback allowlist,
    or stale cache for a model whose capability flipped server-side), the live
    SDK rejects ``reasoning_effort`` with a ``JsonRpcError`` whose message
    contains ``"does not support reasoning effort"``. The provider's
    ``error_translation`` (``errors.yaml:P4``) must map this to
    ``ConfigurationError``.

    Drift detection: if the SDK changes the error wording, this test fails
    and forces a synchronized update of the substring rule in ``errors.yaml``.

    Mutation check: change the P4 ``substring_match`` to a different phrase
    and this test goes red — the JsonRpcError falls through to the default
    mapping (``ProviderUnavailableError``) instead of ``ConfigurationError``.

    Contract: provider-protocol:complete:MUST:11 (Layer-2 backstop);
    error-hierarchy.md (translation policy).
    """

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_live_sdk_reject_translates_to_configuration_error(
        self,
    ) -> None:
        import os
        from pathlib import Path

        copilot = require_sdk()

        token = (
            os.environ.get("COPILOT_AGENT_TOKEN")
            or os.environ.get("COPILOT_GITHUB_TOKEN")
            or os.environ.get("GH_TOKEN")
            or os.environ.get("GITHUB_TOKEN")
        )
        if not token:
            pytest.fail(
                "No GITHUB_TOKEN available; live Layer-2 test requires real "
                "SDK auth. Tests run, not skip — set GITHUB_TOKEN."
            )

        from amplifier_core import llm_errors as kernel_errors

        from amplifier_module_provider_github_copilot.error_translation import (
            load_error_config,
            translate_sdk_error,
        )
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            _make_deny_hook_config,  # pyright: ignore[reportPrivateUsage]
            deny_permission_request,
        )

        client = copilot.CopilotClient(
            base_directory=str(Path.cwd() / "logs" / ".pytest-reasoning-effort-home"),
            github_token=token,
            log_level="info",
            env=dict(os.environ),
            mode="copilot-cli",
        )
        await client.start()
        captured_exc: Exception | None = None
        try:
            try:
                # Provoke Layer-2: pass reasoning_effort to a model the
                # backend rejects. claude-haiku-4.5 advertises
                # supports_reasoning_effort=False; this round-trips to the
                # server which raises JsonRpcError.
                session = await client.create_session(
                    model="claude-haiku-4.5",
                    streaming=True,
                    available_tools=[],
                    on_permission_request=deny_permission_request,
                    hooks=_make_deny_hook_config(),
                    reasoning_effort="high",
                )
                # If we got here the contract assumption is broken.
                await session.disconnect()
                pytest.fail(
                    "Live SDK accepted reasoning_effort='high' on "
                    "claude-haiku-4.5; the backend behavior changed and the "
                    "Layer-2 backstop rule may be stale. Re-probe and update "
                    "errors.yaml:P4."
                )
            except Exception as e:
                captured_exc = e
        finally:
            await client.stop()

        # Live SDK raises ``copilot._jsonrpc.JsonRpcError``. The class is not
        # re-exported at ``copilot`` root in b10, so the test imports from the
        # underscored module directly and pins the exact type with isinstance —
        # avoids the fragile-string-compare anti-pattern and makes a future
        # rename or hierarchy change fail loud at this assertion.
        from copilot._jsonrpc import JsonRpcError  # type: ignore[import-untyped]

        assert isinstance(captured_exc, JsonRpcError), (
            f"Live SDK raised {type(captured_exc).__name__} (msg: "
            f"{captured_exc!r}); expected JsonRpcError. Either the SDK error "
            f"hierarchy changed or the backend started rejecting via a "
            f"different transport — investigate before updating this test."
        )
        original_msg = str(captured_exc)
        # Pin the substring our errors.yaml:P4 rule keys on. If the backend
        # rewords this message, this assertion fails BEFORE the translation
        # step, telling us exactly what to update.
        assert "does not support reasoning effort" in original_msg, (
            f"Live SDK error message no longer contains the substring "
            f"errors.yaml:P4 keys on. Current message: {original_msg!r}. "
            f"Update the substring_match rule and this assertion together."
        )

        # End-to-end Layer-2 translation: this is the round-trip the user
        # actually experiences when Layer-1 is bypassed in production.
        translated = translate_sdk_error(
            captured_exc,
            load_error_config(),
            provider="github-copilot",
            model="claude-haiku-4.5",
        )
        assert isinstance(translated, kernel_errors.ConfigurationError), (
            f"errors.yaml:P4 substring rule failed to map live SDK "
            f"JsonRpcError to ConfigurationError; got "
            f"{type(translated).__name__} instead. This breaks the "
            f"Layer-1/Layer-2 same-class contract documented in "
            f"contracts/provider-protocol.md MUST:11."
        )
        assert translated.__cause__ is captured_exc, (
            "ConfigurationError must chain the original SDK exception via "
            "`raise ... from exc` so traces preserve root cause."
        )
