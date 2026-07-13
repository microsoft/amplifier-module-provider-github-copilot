"""Provider-side context_tier forwarding.

Contract: provider-protocol:complete:MUST:12

Mirrors the reasoning_effort data path (MUST:11) but with a single difference:
there is NO per-model capability descriptor for context tier, so validation is a
static membership gate against the SDK literal allowlist only. The value is
forwarded as the verbatim string, never the SDK ``ContextTier`` enum (which is a
plain ``enum.Enum`` and is not JSON-serializable under the SDK's bare
``json.dumps`` JSON-RPC sender).
"""

from __future__ import annotations

import json
import logging
import typing
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from amplifier_module_provider_github_copilot._compat import ConfigurationError
from amplifier_module_provider_github_copilot.request_adapter import (
    _CONTEXT_TIER_ALLOWLIST,  # pyright: ignore[reportPrivateUsage]
    convert_chat_request,
    validate_context_tier,
)
from tests._sdk_version_gate import require_sdk

require_sdk()


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

    messages: list[_Msg] = field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    model: str | None = None
    tools: list[Any] = field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    max_output_tokens: int | None = None
    context_tier: str | None = None


# ----------------------------------------------------------------------------
# convert_chat_request copies the field (no validation here)
# ----------------------------------------------------------------------------


class TestConvertChatRequestCarriesContextTier:
    """convert_chat_request preserves context_tier on CompletionRequest.

    Contract: provider-protocol:complete:MUST:12
    """

    def test_none_passes_through_as_none(self) -> None:
        req = _Req(messages=[_Msg("user", "hi")], context_tier=None)
        out = convert_chat_request(req)
        assert out.context_tier is None

    def test_value_preserved_verbatim(self) -> None:
        req = _Req(messages=[_Msg("user", "hi")], context_tier="long_context")
        out = convert_chat_request(req)
        assert out.context_tier == "long_context"

    def test_empty_string_normalized_to_none(self) -> None:
        """Empty-string is treated as None (no tier requested) per MUST:12."""
        req = _Req(messages=[_Msg("user", "hi")], context_tier="")
        out = convert_chat_request(req)
        assert out.context_tier is None

    def test_missing_attribute_is_none(self) -> None:
        """getattr-with-default tolerates older kernels lacking the field."""

        class _Bare:
            messages = [_Msg("user", "hi")]
            model = None
            tools: list[Any] = []
            max_output_tokens = None
            # No context_tier attribute at all.

        out = convert_chat_request(_Bare())
        assert out.context_tier is None

    def test_non_string_value_raises_configuration_error(self) -> None:
        """Non-str non-None context_tier on ChatRequest must surface loudly
        rather than silently normalize to None. Pinned at the adapter boundary
        in ``convert_chat_request``."""
        bogus_request = SimpleNamespace(
            messages=[SimpleNamespace(role="user", content="hi")],
            model="claude-sonnet-4.6",
            tools=None,
            system=None,
            max_output_tokens=None,
            context_tier=42,  # int, not str
        )
        with pytest.raises(ConfigurationError) as excinfo:
            convert_chat_request(bogus_request)
        msg = str(excinfo.value)
        assert "context_tier" in msg
        assert "int" in msg


# ----------------------------------------------------------------------------
# validate_context_tier static membership gate
# ----------------------------------------------------------------------------


class TestValidateContextTier:
    """Static SDK-literal membership gate.

    Contract: provider-protocol:complete:MUST:12

    There is no per-model capability descriptor, so the gate is membership-only:
    accept ``{"default","long_context"}``, reject everything else with the
    rejected value redacted.
    """

    @pytest.mark.parametrize("value", ["default", "long_context"])
    def test_returns_value_when_in_allowlist(self, value: str) -> None:
        assert validate_context_tier(value, model_id="claude-sonnet-4.6") == value

    def test_returns_none_when_input_none(self) -> None:
        assert validate_context_tier(None, model_id="gpt-5.5") is None

    def test_returns_none_when_input_empty_string(self) -> None:
        assert validate_context_tier("", model_id="gpt-5.5") is None

    def test_raises_when_value_not_in_allowlist(self) -> None:
        with pytest.raises(ConfigurationError) as excinfo:
            validate_context_tier("turbo", model_id="claude-sonnet-4.6")
        msg = str(excinfo.value)
        assert "context_tier" in msg
        assert "default" in msg and "long_context" in msg

    def test_mixed_case_rejected_no_silent_normalization(self) -> None:
        """SDK literal is strictly lowercase; "Long_Context" must be rejected,
        not silently lowercased."""
        with pytest.raises(ConfigurationError):
            validate_context_tier("Long_Context", model_id="claude-sonnet-4.6")

    def test_overlong_value_rejected_with_redaction(self) -> None:
        """A long/secret-shaped value must be rejected AND must not be echoed
        verbatim in the error message."""
        secret = "AKIA" + "X" * 60
        with pytest.raises(ConfigurationError) as excinfo:
            validate_context_tier(secret, model_id="claude-sonnet-4.6")
        msg = str(excinfo.value)
        assert secret not in msg, "rejected value leaked into error text"
        assert "redacted" in msg

    def test_secret_shaped_model_id_redacted_in_error(self) -> None:
        """A secret-shaped model_id must not leak into the error message.

        Mirrors the model_id redaction at the reasoning_effort raise sites:
        ``ChatRequest.model`` is caller-controlled, so a misrouted credential
        reaching this validator must be redacted before it is reflected into
        the ConfigurationError text.
        """
        secret_model = "ghp_" + "x" * 36  # GitHub token shape, len=40
        with pytest.raises(ConfigurationError) as excinfo:
            validate_context_tier("turbo", model_id=secret_model)
        msg = str(excinfo.value)
        assert secret_model not in msg, "caller-controlled model_id leaked into error text"
        assert "[REDACTED]" in msg

    def test_rejected_value_not_logged_verbatim(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        secret = "ZZZ" + "9" * 40
        with caplog.at_level(logging.DEBUG):
            with pytest.raises(ConfigurationError):
                validate_context_tier(secret, model_id="claude-sonnet-4.6")
        assert secret not in caplog.text


# ----------------------------------------------------------------------------
# SDK source-shape pins: allowlist must track the SDK Literal, and the
# rpc enum must remain non-JSON-serializable (the reason we forward a string).
# ----------------------------------------------------------------------------


class TestSDKSourceShape:
    """Pin the design's two load-bearing SDK facts.

    Contract: provider-protocol:complete:MUST:12
    """

    def test_allowlist_matches_sdk_literal(self) -> None:
        """The static allowlist MUST equal the public SDK ContextTier Literal.
        If the SDK grows a tier, this fails and forces an allowlist update."""
        import copilot.session as sdk_session

        tier_args = typing.get_args(sdk_session.ContextTier)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownArgumentType]
        assert set(tier_args) == _CONTEXT_TIER_ALLOWLIST

    def test_rpc_enum_is_not_json_serializable(self) -> None:
        """Documents WHY the membrane forwards the verbatim string, not the SDK
        enum: the rpc ContextTier is a plain Enum and the JSON-RPC sender uses a
        bare json.dumps with no encoder, so forwarding the enum would raise."""
        from copilot.generated.rpc import (
            ContextTier,  # noqa: E501  # pyright: ignore[reportPrivateImportUsage]
        )

        with pytest.raises(TypeError):
            json.dumps(ContextTier.LONG_CONTEXT)
        # The string the provider actually forwards serializes cleanly.
        assert json.dumps("long_context") == '"long_context"'


# ----------------------------------------------------------------------------
# client.session() membrane forwarding
# ----------------------------------------------------------------------------


class TestSessionForwardsContextTier:
    """client.session() must pass context_tier to SDK create_session.

    Contract: provider-protocol:complete:MUST:12

    Mutation check: removing the ``if context_tier is not None:`` block in
    sdk_adapter/client.py makes the value never reach create_session — red.
    """

    @pytest.mark.parametrize("tier_value", ["default", "long_context"])
    @pytest.mark.asyncio
    async def test_value_reaches_sdk_create_session_kwargs(self, tier_value: str) -> None:
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
            model="claude-sonnet-4.6", context_tier=tier_value
        ) as handle:
            assert handle.session_id == "sid", (
                f"session context yielded unexpected handle: {handle!r}"
            )

        kwargs = sdk_client.create_session.call_args.kwargs
        forwarded = kwargs.get("context_tier")
        assert forwarded == tier_value, (
            f"Expected context_tier={tier_value!r} on create_session, got kwargs={kwargs!r}"
        )
        # Must be the verbatim string, never the SDK enum (non-serializable).
        assert isinstance(forwarded, str), (
            f"context_tier forwarded as {type(forwarded).__name__}, expected str"
        )
        assert sdk_client.create_session.await_count == 1
        assert fake_sdk_session.disconnect.await_count == 1

    @pytest.mark.asyncio
    async def test_none_omits_kwarg_entirely(self) -> None:
        """When context_tier is None, SDK kwarg MUST be absent (not None)."""
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

        async with wrapper.session(model="gpt-4", context_tier=None):
            pass

        kwargs = sdk_client.create_session.call_args.kwargs
        assert "context_tier" not in kwargs, (
            f"Expected context_tier kwarg absent, got: {kwargs.get('context_tier')!r}"
        )
        assert fake_sdk_session.disconnect.await_count == 1


# ----------------------------------------------------------------------------
# provider.complete threads context_tier through BOTH call sites
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

        @asynccontextmanager  # pyright: ignore[reportDeprecated]
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
                    "context_tier": context_tier,
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


class TestCompleteThreadsContextTierToBothCallSites:
    """provider.complete must forward context_tier identically on the main path
    AND the fake-tool correction retry path.

    Contract: provider-protocol:complete:MUST:12

    Mutation check: dropping the ``context_tier=validated_context_tier`` kwarg
    from EITHER ``_execute_sdk_completion`` invocation in provider.py makes one
    of the captured session calls show None — red.
    """

    @staticmethod
    def _make_request(context_tier: str | None) -> MagicMock:
        request = MagicMock()
        request.model = "gpt-4o"
        request.messages = [MagicMock(role="user", content="list files")]
        request.attachments = None
        request.max_output_tokens = None
        request.reasoning_effort = None
        request.context_tier = context_tier
        request.tools = [{"name": "bash", "description": "Run shell commands", "parameters": {}}]
        return request

    @pytest.mark.asyncio
    async def test_correction_retry_sees_same_context_tier_as_main(self) -> None:
        from amplifier_core import ChatResponse

        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        wrapper, session_calls = _make_capturing_wrapper(
            fake_text="[Tool Call: bash(command='ls')]",
            clean_text="ok",
        )
        provider = GitHubCopilotProvider(client=wrapper)  # type: ignore[arg-type]
        result = await provider.complete(self._make_request("long_context"))

        assert len(session_calls) == 2, (
            f"Expected main + correction = 2 session calls, got {session_calls!r}"
        )
        assert session_calls[0]["context_tier"] == "long_context", (
            f"Main session: context_tier lost — got {session_calls[0]['context_tier']!r}"
        )
        assert session_calls[1]["context_tier"] == "long_context", (
            f"Correction session: context_tier dropped on retry — got "
            f"{session_calls[1]['context_tier']!r}; both call sites in provider.py "
            f"MUST forward the validated value identically."
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
        assert session_calls[0]["context_tier"] is None
        assert session_calls[1]["context_tier"] is None
        assert isinstance(result, ChatResponse)


# ----------------------------------------------------------------------------
# provider.complete pre-flight rejection: an invalid tier must raise
# BEFORE any SDK session is opened — proves the gate is actually wired in.
# ----------------------------------------------------------------------------


class TestProviderPreflightRejectsInvalidTier:
    """An invalid context_tier raises ConfigurationError before the SDK is touched.

    Contract: provider-protocol:complete:MUST:12
    Contract: observability:Events:MUST:6 (pre-flight ConfigurationError exempt)
    """

    @pytest.mark.asyncio
    async def test_invalid_tier_raises_before_session_opened(self) -> None:
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        wrapper, session_calls = _make_capturing_wrapper(
            fake_text="[Tool Call: bash(command='ls')]",
            clean_text="ok",
        )
        provider = GitHubCopilotProvider(client=wrapper)  # type: ignore[arg-type]

        request = MagicMock()
        request.model = "gpt-4o"
        request.messages = [MagicMock(role="user", content="hi")]
        request.attachments = None
        request.max_output_tokens = None
        request.reasoning_effort = None
        request.context_tier = "ludicrous_speed"
        request.tools = []

        with pytest.raises(ConfigurationError):
            await provider.complete(request)

        assert session_calls == [], (
            "SDK session must NOT be opened when the pre-flight tier gate rejects "
            f"the value; got session_calls={session_calls!r}"
        )


# ----------------------------------------------------------------------------
# provider config `enable_long_context` default (MUST:4 + MUST:13)
# ----------------------------------------------------------------------------


def _elc_request(context_tier: str | None, model: str = "gpt-4o") -> MagicMock:
    """Minimal ChatRequest stand-in for the enable_long_context tests."""
    request = MagicMock()
    request.model = model
    request.messages = [MagicMock(role="user", content="list files")]
    request.attachments = None
    request.max_output_tokens = None
    request.reasoning_effort = None
    request.context_tier = context_tier
    request.tools = [{"name": "bash", "description": "Run shell commands", "parameters": {}}]
    return request


class TestGetInfoExposesEnableLongContextField:
    """get_info() surfaces a boolean enable_long_context ConfigField.

    Contract: provider-protocol:get_info:MUST:4
    """

    def test_enable_long_context_field_present_with_exact_shape(self) -> None:
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        provider = GitHubCopilotProvider(client=MagicMock())
        info = provider.get_info()

        fields = {f.id: f for f in info.config_fields}
        assert "enable_long_context" in fields, (
            f"get_info must expose an enable_long_context ConfigField; "
            f"got ids={sorted(fields)!r}"
        )
        field = fields["enable_long_context"]
        assert field.field_type == "boolean", (
            f"field_type must be 'boolean', got {field.field_type!r}"
        )
        assert field.default == "false", (
            f"default must be 'false' (off), got {field.default!r}"
        )
        assert field.required is False, (
            f"required must be False, got {field.required!r}"
        )
        assert field.requires_model is True, (
            f"requires_model must be True (tier support is per-model), "
            f"got {field.requires_model!r}"
        )


class TestEnableLongContextDefault:
    """provider.complete defaults the tier to long_context when the flag is on
    and the caller omitted context_tier; a caller value always wins.

    Contract: provider-protocol:complete:MUST:13

    Mutation check: deleting the ``if requested is None and
    self._enable_long_context`` default in provider.complete() makes
    test_flag_on_caller_none_defaults_long_context_both_sites show None — red.
    """

    @pytest.mark.asyncio
    async def test_flag_on_caller_none_defaults_long_context_both_sites(self) -> None:
        from amplifier_core import ChatResponse

        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        wrapper, session_calls = _make_capturing_wrapper(
            fake_text="[Tool Call: bash(command='ls')]",
            clean_text="ok",
        )
        provider = GitHubCopilotProvider(
            config={"enable_long_context": True},
            client=wrapper,  # type: ignore[arg-type]
        )
        result = await provider.complete(_elc_request(None))

        assert len(session_calls) == 2, (
            f"Expected main + correction = 2 session calls, got {session_calls!r}"
        )
        assert session_calls[0]["context_tier"] == "long_context", (
            f"Main session: flag-on default lost — got {session_calls[0]['context_tier']!r}"
        )
        assert session_calls[1]["context_tier"] == "long_context", (
            f"Correction session: flag-on default dropped on retry — got "
            f"{session_calls[1]['context_tier']!r}"
        )
        assert isinstance(result, ChatResponse)

    @pytest.mark.asyncio
    async def test_caller_value_wins_over_flag(self) -> None:
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        wrapper, session_calls = _make_capturing_wrapper(
            fake_text="[Tool Call: bash(command='ls')]",
            clean_text="ok",
        )
        provider = GitHubCopilotProvider(
            config={"enable_long_context": True},
            client=wrapper,  # type: ignore[arg-type]
        )
        await provider.complete(_elc_request("default"))

        assert session_calls[0]["context_tier"] == "default", (
            f"Caller-supplied 'default' MUST take precedence over the flag; "
            f"got {session_calls[0]['context_tier']!r}"
        )
        assert session_calls[1]["context_tier"] == "default"

    @pytest.mark.asyncio
    async def test_flag_off_caller_none_stays_none(self) -> None:
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        wrapper, session_calls = _make_capturing_wrapper(
            fake_text="[Tool Call: bash(command='ls')]",
            clean_text="ok",
        )
        provider = GitHubCopilotProvider(client=wrapper)  # type: ignore[arg-type]
        await provider.complete(_elc_request(None))

        assert session_calls[0]["context_tier"] is None, (
            f"Flag off + caller None MUST NOT forward a tier; "
            f"got {session_calls[0]['context_tier']!r}"
        )
        assert session_calls[1]["context_tier"] is None

    @pytest.mark.asyncio
    async def test_empty_string_caller_with_flag_on_defaults_long_context(self) -> None:
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        wrapper, session_calls = _make_capturing_wrapper(
            fake_text="[Tool Call: bash(command='ls')]",
            clean_text="ok",
        )
        provider = GitHubCopilotProvider(
            config={"enable_long_context": True},
            client=wrapper,  # type: ignore[arg-type]
        )
        await provider.complete(_elc_request(""))

        assert session_calls[0]["context_tier"] == "long_context", (
            f"Empty-string caller tier normalizes to None upstream, so the flag "
            f"default MUST apply; got {session_calls[0]['context_tier']!r}"
        )
        assert session_calls[1]["context_tier"] == "long_context", (
            f"Correction session: empty-string default dropped on retry — got "
            f"{session_calls[1]['context_tier']!r}"
        )

    @pytest.mark.asyncio
    async def test_flag_on_does_not_mutate_caller_request(self) -> None:
        """The effective tier is a transient local; the caller's ChatRequest
        MUST be left exactly as supplied (contract MUST:13: MUST NOT mutate).
        """
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        for caller_tier in (None, ""):
            wrapper, _calls = _make_capturing_wrapper(
                fake_text="[Tool Call: bash(command='ls')]",
                clean_text="ok",
            )
            provider = GitHubCopilotProvider(
                config={"enable_long_context": True},
                client=wrapper,  # type: ignore[arg-type]
            )
            request = _elc_request(caller_tier)
            await provider.complete(request)

            assert request.context_tier == caller_tier, (
                f"complete() must not write the long_context default back onto "
                f"the caller's ChatRequest; context_tier mutated from "
                f"{caller_tier!r} to {request.context_tier!r}"
            )

    @pytest.mark.asyncio
    async def test_flag_on_small_window_model_still_forwards_long_context(self) -> None:
        """No per-model window gate: the default forwards even on a 200K model.

        Pins the deliberate decision (the SDK exposes no context-tier capability
        descriptor) so a future contributor cannot add a per-model gate without
        amending MUST:13.
        """
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        wrapper, session_calls = _make_capturing_wrapper(
            fake_text="[Tool Call: bash(command='ls')]",
            clean_text="ok",
        )
        provider = GitHubCopilotProvider(
            config={"enable_long_context": True},
            client=wrapper,  # type: ignore[arg-type]
        )
        await provider.complete(_elc_request(None, model="claude-haiku-4.5"))

        assert session_calls[0]["context_tier"] == "long_context", (
            f"Small-window model MUST still receive the forwarded default "
            f"(no per-model gate); got {session_calls[0]['context_tier']!r}"
        )


class TestEnableLongContextStrictParse:
    """The flag is parsed by _parse_raw_flag (allowlist-truthy); string 'false'
    and friends are OFF — guarding the bool('false')==True footgun.

    Contract: provider-protocol:complete:MUST:13
    """

    @pytest.mark.parametrize(
        ("flag_value", "expected_tier"),
        [
            (True, "long_context"),
            ("true", "long_context"),
            ("True", "long_context"),
            ("1", "long_context"),
            ("yes", "long_context"),
            (False, None),
            ("false", None),
            ("0", None),
            ("no", None),
            ("", None),
            ("maybe", None),
            (" true ", None),
        ],
    )
    @pytest.mark.asyncio
    async def test_strict_parse_drives_forwarded_tier(
        self, flag_value: object, expected_tier: str | None
    ) -> None:
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        wrapper, session_calls = _make_capturing_wrapper(
            fake_text="[Tool Call: bash(command='ls')]",
            clean_text="ok",
        )
        provider = GitHubCopilotProvider(
            config={"enable_long_context": flag_value},
            client=wrapper,  # type: ignore[arg-type]
        )
        await provider.complete(_elc_request(None))

        assert session_calls[0]["context_tier"] == expected_tier, (
            f"enable_long_context={flag_value!r} must parse to forwarded tier "
            f"{expected_tier!r}; got {session_calls[0]['context_tier']!r}"
        )


# ----------------------------------------------------------------------------
# get_info() reports the tier-selected prompt-budget window for the default
# model so Amplifier times compaction correctly (provider-protocol:get_info:MUST:5)
# ----------------------------------------------------------------------------


def _opus48_info() -> Any:
    """CopilotModelInfo mirroring the live opus-4.8 shape (1M ceiling,
    200K/936K prompt budgets)."""
    from amplifier_module_provider_github_copilot.models import CopilotModelInfo

    return CopilotModelInfo(
        id="claude-opus-4.8",
        name="Claude Opus 4.8",
        context_window=1_000_000,
        max_output_tokens=64_000,
        context_window_default=200_000,
        context_window_long=936_000,
    )


def _provider_with_warm_cache(
    *, enable_long_context: bool, default_model: str = "claude-opus-4.8"
) -> Any:
    from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

    provider = GitHubCopilotProvider(
        config={
            "default_model": default_model,
            "enable_long_context": enable_long_context,
        },
        client=MagicMock(),
    )
    # Warm the in-memory capability cache the way list_models() would, so
    # get_info() resolves the default model without disk I/O.
    provider._copilot_models_cache = [_opus48_info()]  # pyright: ignore[reportPrivateUsage]
    return provider


class TestGetInfoTierAwareWindow:
    """get_info().defaults.context_window tracks the active tier's prompt budget
    for the configured default model.

    Contract: provider-protocol:get_info:MUST:5

    Mutation check: reverting get_info() to return cfg.defaults verbatim makes
    the long-tier assertion read 200000 (the static literal) instead of 936000 — red.
    """

    def test_default_tier_reports_default_budget(self) -> None:
        provider = _provider_with_warm_cache(enable_long_context=False)
        info = provider.get_info()
        assert info.defaults["context_window"] == 200_000
        assert info.defaults["max_output_tokens"] == 64_000

    def test_long_tier_reports_long_budget(self) -> None:
        provider = _provider_with_warm_cache(enable_long_context=True)
        info = provider.get_info()
        assert info.defaults["context_window"] == 936_000
        assert info.defaults["max_output_tokens"] == 64_000

    def test_unknown_default_model_falls_back_to_static(self) -> None:
        # Default model absent from the warmed cache => the static cold-cache
        # fallback from config/_models.py is reported unchanged.
        provider = _provider_with_warm_cache(
            enable_long_context=True, default_model="model-not-in-cache"
        )
        info = provider.get_info()
        assert info.defaults["context_window"] == 200_000
        assert info.defaults["max_output_tokens"] == 32_000

    def test_old_cache_zero_sentinel_reports_static_not_ceiling(self) -> None:
        # Regression guard: a pre-tier (old) cache yields 0 tier budgets while the
        # display ceiling is 1M. get_info must report the static cold-cache window,
        # never the ceiling, so compaction is not ~5x over-budget.
        from amplifier_module_provider_github_copilot.models import CopilotModelInfo
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        provider = GitHubCopilotProvider(
            config={
                "default_model": "claude-opus-4.8",
                "enable_long_context": True,
            },
            client=MagicMock(),
        )
        provider._copilot_models_cache = [  # pyright: ignore[reportPrivateUsage]
            CopilotModelInfo(
                id="claude-opus-4.8",
                name="Claude Opus 4.8",
                context_window=1_000_000,
                max_output_tokens=64_000,
                context_window_default=0,
                context_window_long=0,
            )
        ]
        info = provider.get_info()
        assert info.defaults["context_window"] == 200_000
        assert info.defaults["context_window"] != 1_000_000
        # max_output_tokens stays the model's real value (never inflated).
        assert info.defaults["max_output_tokens"] == 64_000

    def test_get_info_does_not_mutate_cached_defaults_singleton(self) -> None:
        from amplifier_module_provider_github_copilot.config_loader import (
            load_models_config,
        )

        provider = _provider_with_warm_cache(enable_long_context=True)
        provider.get_info()
        provider.get_info()

        # The lru_cached singleton defaults must be pristine — get_info copies
        # before injecting the tier window.
        singleton = load_models_config().defaults
        assert singleton["context_window"] == 200_000
        assert singleton["max_output_tokens"] == 32_000

    def test_two_providers_isolated_by_tier(self) -> None:
        from amplifier_module_provider_github_copilot.config_loader import (
            load_models_config,
        )

        short = _provider_with_warm_cache(enable_long_context=False)
        long = _provider_with_warm_cache(enable_long_context=True)

        short_info = short.get_info()
        long_info = long.get_info()

        assert short_info.defaults["context_window"] == 200_000
        assert long_info.defaults["context_window"] == 936_000
        # Neither provider poisoned the shared singleton for the other.
        assert load_models_config().defaults["context_window"] == 200_000
