"""Tests for max_output_tokens forwarding and length truncation visibility.

Contracts:
- provider-protocol:complete:MUST:10 — forward `ChatRequest.max_output_tokens` to the
  SDK as a per-session output cap via `ModelCapabilitiesOverride`
  (relies on `github-copilot-sdk>=0.3.0`)
- streaming-contract:FinishReason:MUST:6 — WARN when finish_reason == "length"
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from amplifier_module_provider_github_copilot.streaming import (
    DomainEvent,
    DomainEventType,
    StreamingAccumulator,
)

# ---------------------------------------------------------------------------
# streaming-contract:FinishReason:MUST:6 — WARN on truncation
# ---------------------------------------------------------------------------


class TestLengthTruncationWarning:
    """Accumulator emits one WARN when finish_reason == 'length'."""

    def test_warns_when_finish_reason_is_length(self, caplog: pytest.LogCaptureFixture) -> None:
        """to_chat_response() MUST log a WARNING when finish_reason == 'length'.

        Contract: streaming-contract:FinishReason:MUST:6
        """
        accumulator = StreamingAccumulator()
        accumulator.add(
            DomainEvent(
                type=DomainEventType.USAGE_UPDATE,
                data={"input_tokens": 100, "output_tokens": 4096},
            )
        )
        accumulator.add(
            DomainEvent(
                type=DomainEventType.TURN_COMPLETE,
                data={"finish_reason": "length"},
            )
        )

        with caplog.at_level(
            logging.WARNING,
            logger="amplifier_module_provider_github_copilot.streaming",
        ):
            response = accumulator.to_chat_response()

        # Behavioral: response is returned unchanged with finish_reason="length"
        assert response.finish_reason == "length"

        # Behavioral: exactly one WARNING about truncation, mentioning the
        # observed output token count for diagnosability.
        truncation_warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "truncat" in r.getMessage().lower()
        ]
        assert len(truncation_warnings) == 1, (
            f"Expected exactly 1 truncation WARNING, got {len(truncation_warnings)}: "
            f"{[r.getMessage() for r in truncation_warnings]}"
        )
        assert "4096" in truncation_warnings[0].getMessage()

    def test_no_warning_when_finish_reason_is_stop(self, caplog: pytest.LogCaptureFixture) -> None:
        """to_chat_response() MUST NOT log truncation WARNING for normal completion.

        Contract: streaming-contract:FinishReason:MUST:6 (negative case)
        """
        accumulator = StreamingAccumulator()
        accumulator.add(
            DomainEvent(
                type=DomainEventType.USAGE_UPDATE,
                data={"input_tokens": 100, "output_tokens": 50},
            )
        )
        accumulator.add(
            DomainEvent(
                type=DomainEventType.TURN_COMPLETE,
                data={"finish_reason": "stop"},
            )
        )

        with caplog.at_level(
            logging.WARNING,
            logger="amplifier_module_provider_github_copilot.streaming",
        ):
            response = accumulator.to_chat_response()

        assert response.finish_reason == "stop"
        truncation_warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "truncat" in r.getMessage().lower()
        ]
        assert truncation_warnings == [], (
            f"Expected 0 truncation WARNINGs for stop, got "
            f"{[r.getMessage() for r in truncation_warnings]}"
        )


# ---------------------------------------------------------------------------
# provider-protocol:complete:MUST:10 — forward max_tokens via model_capabilities
# ---------------------------------------------------------------------------


class TestConvertChatRequestMaxOutputTokens:
    """convert_chat_request() must extract ChatRequest.max_output_tokens onto the
    internal CompletionRequest so downstream session code can forward it.

    The canonical kernel field is max_output_tokens (not max_tokens). Both the
    Anthropic and OpenAI providers read request.max_output_tokens. Evidence:
    - amplifier_core/message_models.py: max_output_tokens: int | None = None
    - proto: ChatRequest field 6 = max_output_tokens
    - Anthropic provider __init__.py:1981: request.max_output_tokens
    - OpenAI provider __init__.py:809: request.max_output_tokens

    Contract: provider-protocol:complete:MUST:10
    """

    def test_extracts_max_output_tokens_when_set(self) -> None:
        from amplifier_module_provider_github_copilot.request_adapter import (
            convert_chat_request,
        )

        request = MagicMock()
        request.model = "claude-opus-4.5"
        request.messages = [MagicMock(role="user", content="hi")]
        request.tools = None
        request.attachments = None
        request.max_output_tokens = 256
        request.reasoning_effort = None
        request.context_tier = None

        result = convert_chat_request(request)

        assert result.max_tokens == 256

    def test_max_tokens_none_when_not_set(self) -> None:
        from amplifier_module_provider_github_copilot.request_adapter import (
            convert_chat_request,
        )

        request = MagicMock()
        request.model = "claude-opus-4.5"
        request.messages = [MagicMock(role="user", content="hi")]
        request.tools = None
        request.attachments = None
        request.max_output_tokens = None
        request.reasoning_effort = None
        request.context_tier = None

        result = convert_chat_request(request)

        assert result.max_tokens is None


class TestSessionForwardsMaxTokens:
    """client.session() must forward max_tokens to SDK create_session as
    model_capabilities=ModelCapabilitiesOverride(limits=ModelLimitsOverride(...)).

    Contract: provider-protocol:complete:MUST:10
    """

    @pytest.fixture
    def fake_sdk_capability_types(self, monkeypatch: pytest.MonkeyPatch) -> tuple[type, type]:
        """Inject minimal stand-in dataclasses for SDK capability types.

        Conftest sets SKIP_SDK_CHECK=1 which leaves the real SDK types as None
        in `_imports`. These stand-ins have the same field surface
        (`limits.max_output_tokens`) and let the test assert the forwarded
        shape without depending on the live SDK.
        """
        from dataclasses import dataclass

        from amplifier_module_provider_github_copilot.sdk_adapter import _imports

        @dataclass
        class FakeLimitsOverride:
            max_output_tokens: int | None = None

        @dataclass
        class FakeCapabilitiesOverride:
            limits: FakeLimitsOverride | None = None

        monkeypatch.setattr(_imports, "ModelLimitsOverride", FakeLimitsOverride)
        monkeypatch.setattr(_imports, "ModelCapabilitiesOverride", FakeCapabilitiesOverride)
        return FakeCapabilitiesOverride, FakeLimitsOverride

    @pytest.mark.asyncio
    async def test_session_passes_model_capabilities_when_max_tokens_set(
        self, fake_sdk_capability_types: tuple[type, type]
    ) -> None:
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        FakeCapabilitiesOverride, FakeLimitsOverride = fake_sdk_capability_types
        wrapper = CopilotClientWrapper()

        # Stub _ensure_client_initialized to return a mock SDK client whose
        # create_session is an AsyncMock we can inspect. Bypass real auth/SDK.
        sdk_client = MagicMock()
        fake_sdk_session = MagicMock()
        fake_sdk_session.session_id = "test-session-id"
        fake_sdk_session.disconnect = AsyncMock()
        sdk_client.create_session = AsyncMock(return_value=fake_sdk_session)

        async def _fake_ensure(caller: str = "session") -> Any:  # noqa: ARG001
            return sdk_client

        wrapper._ensure_client_initialized = _fake_ensure  # type: ignore[assignment]  # noqa: SLF001

        async with wrapper.session(model="gpt-4", max_tokens=512):
            pass

        sdk_client.create_session.assert_called_once()
        kwargs = sdk_client.create_session.call_args.kwargs
        caps = kwargs.get("model_capabilities")
        assert isinstance(caps, FakeCapabilitiesOverride), (
            f"Expected ModelCapabilitiesOverride, got {type(caps).__name__}: {caps!r}"
        )
        assert isinstance(caps.limits, FakeLimitsOverride)
        assert caps.limits.max_output_tokens == 512

    @pytest.mark.asyncio
    async def test_session_forwards_max_tokens_with_real_sdk_capability_types(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pin the max_tokens forward path against the REAL SDK 1.0.2 shapes.

        The sibling test above uses stand-in dataclasses, which would silently
        keep passing if the SDK renamed/removed ``max_output_tokens`` or changed
        ``ModelCapabilitiesOverride(limits=...)``. This test binds the ACTUAL
        ``copilot.ModelCapabilitiesOverride`` / ``copilot.ModelLimitsOverride``
        through the ``_imports`` membrane (conftest sets SKIP_SDK_CHECK=1, which
        nulls them) and asserts ``client.session(max_tokens=...)`` constructs the
        real override object. A future SDK signature change makes
        ``ModelLimitsOverride(max_output_tokens=...)`` raise ``TypeError`` here,
        failing the upgrade loudly in CI rather than only in a manual live run.

        Contract: provider-protocol:complete:MUST:10
        """
        from tests._sdk_version_gate import require_sdk

        copilot = require_sdk()

        from amplifier_module_provider_github_copilot.sdk_adapter import _imports
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        monkeypatch.setattr(_imports, "ModelLimitsOverride", copilot.ModelLimitsOverride)
        monkeypatch.setattr(
            _imports, "ModelCapabilitiesOverride", copilot.ModelCapabilitiesOverride
        )

        wrapper = CopilotClientWrapper()

        sdk_client = MagicMock()
        fake_sdk_session = MagicMock()
        fake_sdk_session.session_id = "test-session-id"
        fake_sdk_session.disconnect = AsyncMock()
        sdk_client.create_session = AsyncMock(return_value=fake_sdk_session)

        async def _fake_ensure(caller: str = "session") -> Any:  # noqa: ARG001
            return sdk_client

        wrapper._ensure_client_initialized = _fake_ensure  # type: ignore[assignment]  # noqa: SLF001

        async with wrapper.session(model="gpt-4", max_tokens=512):
            pass

        sdk_client.create_session.assert_called_once()
        kwargs = sdk_client.create_session.call_args.kwargs
        caps = kwargs.get("model_capabilities")
        assert isinstance(caps, copilot.ModelCapabilitiesOverride), (
            f"Expected real copilot.ModelCapabilitiesOverride, got "
            f"{type(caps).__name__}: {caps!r}"
        )
        assert isinstance(caps.limits, copilot.ModelLimitsOverride)
        assert caps.limits.max_output_tokens == 512

    @pytest.mark.asyncio
    async def test_session_omits_model_capabilities_when_max_tokens_none(
        self,
    ) -> None:
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        wrapper = CopilotClientWrapper()

        sdk_client = MagicMock()
        fake_sdk_session = MagicMock()
        fake_sdk_session.session_id = "test-session-id"
        fake_sdk_session.disconnect = AsyncMock()
        sdk_client.create_session = AsyncMock(return_value=fake_sdk_session)

        async def _fake_ensure(caller: str = "session") -> Any:  # noqa: ARG001
            return sdk_client

        wrapper._ensure_client_initialized = _fake_ensure  # type: ignore[assignment]  # noqa: SLF001

        async with wrapper.session(model="gpt-4", max_tokens=None):
            pass

        sdk_client.create_session.assert_called_once()
        kwargs = sdk_client.create_session.call_args.kwargs
        assert "model_capabilities" not in kwargs, (
            f"Expected no model_capabilities kwarg when max_tokens=None, "
            f"got: {kwargs.get('model_capabilities')!r}"
        )


# ---------------------------------------------------------------------------
# P1-A: Correction path clamps max_tokens to 512
# provider-protocol:complete:MUST:10 — correction must not consume double the
# caller's budget.
# ---------------------------------------------------------------------------


class TestCorrectionPathClampsMaxTokens:
    """complete() → fake-tool detected → correction session(max_tokens=512).

    The correction call targets a bounded token budget (≤ 512) to prevent a
    single complete() call from using 2× the caller's cap.  The clamp is
    `min(512, caller_cap)` — so if the caller's cap is already below 512,
    we respect that lower ceiling.

    Mutation check for the first test: replace `else 512` with `else None`
    in provider.py → second session call gets None instead of 512 → red.
    Mutation check for the second test: replace `min(512, ...)` with `512`
    → second session call gets 512 instead of 256 → red.

    Contract: provider-protocol:complete:MUST:10
    """

    # ------------------------------------------------------------------ #
    # Shared building block                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _make_capturing_wrapper(
        fake_text: str,
        clean_text: str,
    ) -> tuple[Any, list[dict[str, Any]]]:
        """Return (wrapper, session_calls) where wrapper records every session() call.

        Call 1 yields ``fake_text`` (triggers fake-tool correction).
        Call 2 yields ``clean_text`` (correction succeeds).
        """
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
            ) -> AsyncIterator[MockSDKSession]:
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

    @staticmethod
    def _make_request(max_output_tokens: int | None) -> MagicMock:
        request = MagicMock()
        request.model = "gpt-4o"
        request.messages = [MagicMock(role="user", content="list the files")]
        request.attachments = None
        request.max_output_tokens = max_output_tokens
        request.reasoning_effort = None
        request.context_tier = None
        # Non-empty tools → tools_available=True → fake-tool path active
        request.tools = [{"name": "bash", "description": "Run shell commands", "parameters": {}}]
        return request

    # ------------------------------------------------------------------ #

    @pytest.mark.asyncio
    async def test_correction_session_capped_at_512_when_caller_has_no_cap(self) -> None:
        """Correction session max_tokens must be 512 when caller specified no cap.

        Chain under test:
          complete(max_output_tokens=None)
            → main session(max_tokens=None): returns fake-tool text
            → should_retry_for_fake_tool_calls fires
            → correction session(max_tokens=512)   ← assert this

        Contract: provider-protocol:complete:MUST:10
        """
        from amplifier_core import ChatResponse

        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        # "[Tool Call: bash...]" matches the r"\[Tool Call:\s*\w+" detection pattern.
        wrapper, session_calls = self._make_capturing_wrapper(
            fake_text="[Tool Call: bash(command='ls -la')]",
            clean_text="Here is the directory listing.",
        )
        provider = GitHubCopilotProvider(client=wrapper)  # type: ignore[arg-type]
        result = await provider.complete(self._make_request(max_output_tokens=None))

        assert len(session_calls) == 2, (
            f"Expected main + correction = 2 session calls, got {len(session_calls)}: "
            f"{session_calls}"
        )
        assert session_calls[0]["max_tokens"] is None, (
            f"Main session: max_tokens should be None (no caller cap), "
            f"got {session_calls[0]['max_tokens']!r}"
        )
        assert session_calls[1]["max_tokens"] == 512, (
            f"Correction session: max_tokens must be 512 (P1-A clamp), "
            f"got {session_calls[1]['max_tokens']!r}"
        )
        assert isinstance(result, ChatResponse)

    @pytest.mark.asyncio
    async def test_correction_respects_caller_ceiling_below_512(self) -> None:
        """Correction session must respect a caller cap that is already below 512.

        When caller sets max_output_tokens=256, the main session gets max_tokens=256
        and the correction gets min(512, 256)=256 — not 512 — so we never exceed
        the caller's declared budget.

        Contract: provider-protocol:complete:MUST:10
        """
        from amplifier_core import ChatResponse

        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        wrapper, session_calls = self._make_capturing_wrapper(
            fake_text="[Tool Call: bash(command='ls')]",
            clean_text="Done.",
        )
        provider = GitHubCopilotProvider(client=wrapper)  # type: ignore[arg-type]
        result = await provider.complete(self._make_request(max_output_tokens=256))

        assert len(session_calls) == 2, (
            f"Expected 2 session calls, got {len(session_calls)}: {session_calls}"
        )
        assert session_calls[0]["max_tokens"] == 256, (
            f"Main session: max_tokens should match caller cap 256, "
            f"got {session_calls[0]['max_tokens']!r}"
        )
        assert session_calls[1]["max_tokens"] == 256, (
            f"Correction session: must respect caller ceiling (256 < 512), "
            f"got {session_calls[1]['max_tokens']!r}"
        )
        assert isinstance(result, ChatResponse)

    @pytest.mark.asyncio
    async def test_correction_respects_tight_caller_ceiling_below_200(self) -> None:
        """Correction session must respect a tight caller cap (e.g. 100 tokens).

        Boundary case for the min(512, caller_cap) clamp. When a caller sets a
        very small budget (max_output_tokens=100), the correction session must
        propagate that exact cap — never silently widen it to 512 — so a single
        complete() call cannot exceed the caller's declared ceiling.

        Observability of a tight-cap correction is provided by:
          - finish_reason="length" propagated to the response
          - streaming-contract:FinishReason:MUST:6 WARN log

        The provider does not currently skip correction below a minimum window;
        that is a separate (deferred) policy decision that would require a
        contract amendment and a Two-Medium config knob.

        Contract: provider-protocol:complete:MUST:10
        """
        from amplifier_core import ChatResponse

        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        wrapper, session_calls = self._make_capturing_wrapper(
            fake_text="[Tool Call: bash(command='ls')]",
            clean_text="ok",
        )
        provider = GitHubCopilotProvider(client=wrapper)  # type: ignore[arg-type]
        result = await provider.complete(self._make_request(max_output_tokens=100))

        assert len(session_calls) == 2, (
            f"Expected 2 session calls, got {len(session_calls)}: {session_calls}"
        )
        assert session_calls[0]["max_tokens"] == 100, (
            f"Main session: max_tokens should match caller cap 100, "
            f"got {session_calls[0]['max_tokens']!r}"
        )
        assert session_calls[1]["max_tokens"] == 100, (
            f"Correction session: must respect tight caller ceiling (100 < 512), "
            f"got {session_calls[1]['max_tokens']!r}"
        )
        assert isinstance(result, ChatResponse)
