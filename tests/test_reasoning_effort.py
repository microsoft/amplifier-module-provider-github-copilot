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
    resolve_provider_default_effort,
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

    def test_accepts_advertised_effort_outside_static_allowlist(self) -> None:
        """The resolved model's advertised ``supported_reasoning_efforts`` is
        authoritative and WIDENS acceptance beyond the static fallback allowlist.

        A well-formed token the live endpoint advertises — e.g. ``"minimal"`` for
        gemini-3.5-flash, which the backend serves but is absent from the static
        superset ``{"none","low","medium","high","xhigh","max"}`` — MUST be
        accepted verbatim rather than hard-rejected by the shape gate.

        Regression for the review finding that a genuinely-advertised effort was
        rejected before model metadata was consulted.

        Contract: provider-protocol:complete:MUST:11
        """
        info = _model_info(allowlist=("minimal", "low", "medium", "high"))
        result = validate_reasoning_effort(
            "minimal", info, model_id="gemini-3.5-flash"
        )
        assert result == "minimal"

    @pytest.mark.parametrize(
        ("supports", "allowlist"),
        [
            pytest.param(True, ("low",), id="capable-but-not-listed"),
            pytest.param(False, ("minimal",), id="listed-but-incapable"),
            pytest.param(None, None, id="cache-miss"),
        ],
    )
    def test_advertised_widening_requires_capability_and_membership(
        self, supports: bool | None, allowlist: tuple[str, ...] | None
    ) -> None:
        """The advertised-set early-accept needs BOTH ``supports_reasoning_effort``
        AND membership; neither conjunct alone widens acceptance. A capable model
        that omits the token, an incapable model that lists it, and a cache miss
        (no advertised set at all) each fall through to the model-independent shape
        gate — which raises for ``"minimal"`` (outside the static superset). Pins
        that widening only happens when the backend-advertised set is actually
        available AND enabled.

        Contract: provider-protocol:complete:MUST:11
        """
        info = (
            None
            if supports is None
            else _model_info(supports=supports, allowlist=allowlist or ())
        )
        with pytest.raises(ConfigurationError):
            validate_reasoning_effort("minimal", info, model_id="gemini-3.5-flash")

    def test_advertised_shape_invalid_token_is_rejected_not_forwarded(self) -> None:
        """Advertised-set widening relaxes ALLOWLIST membership, never the lexical
        safe-token shape. A malformed or secret-shaped token that appears in the
        resolved model's ``supported_reasoning_efforts`` (e.g. via a poisoned
        on-disk model cache) is NOT early-accepted: it falls through to the shape
        gate and is rejected with the value redacted, so it is never forwarded to
        the SDK nor reflected verbatim in the error text.

        Regression for the review finding that the advertised early-accept
        preceded the shape check, letting a cache-injected token bypass lexical
        validation.

        Contract: provider-protocol:complete:MUST:11
        """
        poisoned = "ghp_" + "a" * 36  # overlong, secret-shaped; fails the token shape
        info = _model_info(supports=True, allowlist=(poisoned, "low", "medium"))
        with pytest.raises(ConfigurationError) as excinfo:
            validate_reasoning_effort(poisoned, info, model_id="claude-sonnet-4.6")
        msg = str(excinfo.value)
        assert poisoned not in msg, "raw poisoned token must not leak into error text"
        assert "redacted" in msg
        # A well-formed advertised token on the SAME model still widens.
        good = _model_info(supports=True, allowlist=("minimal", "low"))
        assert (
            validate_reasoning_effort("minimal", good, model_id="gemini-3.5-flash")
            == "minimal"
        )

    def test_advertised_trailing_newline_token_is_rejected(self) -> None:
        """A trailing-newline token (e.g. ``"low\\n"``) advertised by the resolved
        model must be rejected, not accepted. The shape gate is anchored with
        ``\\A...\\Z`` rather than ``^...$`` because Python's ``$`` also matches
        just before a final newline, which would otherwise let ``"low\\n"`` widen
        through the advertised early-accept.

        Regression guard for the ``$`` -> ``\\Z`` anchoring fix: reverting the
        anchor makes this call return ``"low\\n"`` instead of raising.

        Contract: provider-protocol:complete:MUST:11
        """
        newline_token = "low\n"
        info = _model_info(supports=True, allowlist=(newline_token, "medium"))
        with pytest.raises(ConfigurationError) as excinfo:
            validate_reasoning_effort(newline_token, info, model_id="claude-sonnet-4.6")
        assert "\n" not in str(excinfo.value), "raw newline must not reach the error text"

    def test_returns_none_when_input_none(self) -> None:
        info = _model_info(supports=False)
        # No request → no gate trigger, even on unsupported model.
        assert validate_reasoning_effort(None, info, model_id="m") is None

    def test_returns_none_when_input_empty_string(self) -> None:
        info = _model_info(supports=False)
        assert validate_reasoning_effort("", info, model_id="m") is None

    def test_drops_when_model_does_not_support(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A well-formed caller value the model cannot consume is DROPPED to
        None with a WARNING (server falls back), symmetric with the MUST:14
        operator-default gate. It is NOT a raise.

        Contract: provider-protocol:complete:MUST:11
        """
        info = _model_info(supports=False, allowlist=())
        with caplog.at_level(
            logging.WARNING,
            logger="amplifier_module_provider_github_copilot.request_adapter",
        ):
            result = validate_reasoning_effort("medium", info, model_id="claude-haiku-4.5")
        assert result is None
        assert "claude-haiku-4.5" in caplog.text
        assert "Dropping caller reasoning_effort" in caplog.text
        assert "supports_reasoning_effort=False" in caplog.text
        # Pin the level: the caller drop MUST log at WARNING (the operator
        # default is INFO). record.levelno makes any level drift fail, not just
        # a downgrade below the capture threshold.
        rec = next(
            r for r in caplog.records if "Dropping caller reasoning_effort" in r.getMessage()
        )
        assert rec.levelno == logging.WARNING

    def test_raises_when_value_not_in_allowlist(self) -> None:
        info = _model_info(allowlist=("low", "medium", "high"))
        with pytest.raises(ConfigurationError) as excinfo:
            validate_reasoning_effort("banana", info, model_id="claude-sonnet-4.6")
        msg = str(excinfo.value)
        assert "claude-sonnet-4.6" in msg
        assert "banana" in msg
        # Allowed values must be enumerated for diagnosability.
        assert "'low'" in msg and "'medium'" in msg and "'high'" in msg

    def test_drops_when_valid_shape_value_excluded_by_model_allowlist(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A token that passes the universal shape gate but is absent from the
        model's own ``supported_reasoning_efforts`` is DROPPED to None with a
        WARNING at the per-model gate (provider-protocol:complete:MUST:11), not
        raised. Uses ``"max"`` against a model advertising {low, medium, high} —
        the exact case the MUST:11 superset makes reachable.

        Contract: provider-protocol:complete:MUST:11
        """
        info = _model_info(allowlist=("low", "medium", "high"))
        with caplog.at_level(
            logging.WARNING,
            logger="amplifier_module_provider_github_copilot.request_adapter",
        ):
            result = validate_reasoning_effort("max", info, model_id="claude-sonnet-4.6")
        assert result is None
        assert "claude-sonnet-4.6" in caplog.text
        assert "Dropping caller reasoning_effort" in caplog.text
        assert "not in supported_reasoning_efforts" in caplog.text
        rec = next(
            r for r in caplog.records if "Dropping caller reasoning_effort" in r.getMessage()
        )
        assert rec.levelno == logging.WARNING

    def test_unsupported_drop_log_wording(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The ``supports_reasoning_effort=False`` drop names the model id, the
        rejected effort, and states the server falls back — mirroring MUST:14's
        operator-default drop but at WARNING level for the caller scope.

        Contract: provider-protocol:complete:MUST:11 (wording not pinned by the
        contract; this guards the operator-facing diagnosability of the log).
        """
        info = CopilotModelInfo(
            id="claude-sonnet-4.5",
            name="Claude Sonnet 4.5",
            context_window=168_000,
            max_output_tokens=32_000,
            supports_reasoning_effort=False,
            supported_reasoning_efforts=(),
        )
        with caplog.at_level(
            logging.WARNING,
            logger="amplifier_module_provider_github_copilot.request_adapter",
        ):
            result = validate_reasoning_effort("max", info, model_id="claude-sonnet-4.5")
        assert result is None
        assert "claude-sonnet-4.5" in caplog.text
        assert "reasoning_effort='max'" in caplog.text
        assert "supports_reasoning_effort=False" in caplog.text
        assert "server" in caplog.text and "falls back" in caplog.text

    def test_allowlist_drop_log_wording(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The per-model allowlist drop names the model id, the rejected effort,
        and states the server falls back.

        Contract: provider-protocol:complete:MUST:11
        """
        info = CopilotModelInfo(
            id="claude-sonnet-4.6",
            name="Claude Sonnet 4.6",
            context_window=200_000,
            max_output_tokens=64_000,
            supports_reasoning_effort=True,
            supported_reasoning_efforts=("low", "medium", "high"),
        )
        with caplog.at_level(
            logging.WARNING,
            logger="amplifier_module_provider_github_copilot.request_adapter",
        ):
            result = validate_reasoning_effort("max", info, model_id="claude-sonnet-4.6")
        assert result is None
        assert "claude-sonnet-4.6" in caplog.text
        assert "reasoning_effort='max'" in caplog.text
        assert "not in supported_reasoning_efforts" in caplog.text
        assert "server" in caplog.text and "falls back" in caplog.text

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
        assert "provider fallback allowlist" in msg
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
        assert "provider fallback allowlist" in msg
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
        # Select the specific cache-miss record, then pin BOTH its rendered text
        # and its level. Asserting on record.levelno (not just caplog.text /
        # .message under an INFO capture, which also admits WARNING) is what makes
        # a regression that promotes this caller cache-miss line from INFO to
        # WARNING fail red instead of passing green — mirroring the operator
        # cache-miss guard in test_cache_miss_forwards_verbatim. MUST:11 mandates
        # an INFO log here (parity with the MUST:14 operator cache-miss path).
        rec = next(
            r
            for r in caplog.records
            if "deferring final reasoning_effort validation to SDK backstop" in r.getMessage()
        )
        assert "brand-new-model" in rec.getMessage()
        assert rec.levelno == logging.INFO

    def test_model_info_none_with_bogus_value_raises(self) -> None:
        """Cache miss + value not in the provider fallback allowlist must raise.

        Without this gate, an arbitrary <=16-char string would silently reach
        the SDK whenever ``CopilotModelInfo`` is unavailable, defeating the
        Layer-1 capability gate.
        """
        with pytest.raises(ConfigurationError) as excinfo:
            validate_reasoning_effort("frobozz", None, model_id="brand-new-model")
        msg = str(excinfo.value)
        assert "frobozz" in msg
        assert "brand-new-model" in msg
        assert "provider fallback allowlist" in msg
        # Must enumerate accepted values so the caller can self-correct.
        # "none" and "max" are included because the fallback allowlist extends
        # the v1.0.7 SDK literal with both (advertised by the live list_models
        # endpoint; absent from the v1.0.7 SDK literal).
        for v in ("none", "low", "medium", "high", "xhigh", "max"):
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
        assert "provider fallback allowlist" in msg
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
        assert "provider fallback allowlist" in msg
        assert "<redacted; len=40>" in msg
        # The raw secret-shaped value MUST NOT appear in the error text.
        assert secret_like not in msg
        assert "ghp_" not in msg


class TestResolveProviderDefaultEffort:
    """Unit coverage for ``resolve_provider_default_effort`` (MUST:14).

    The operator-scope default is best-effort per resolved model: applied where
    supported, dropped where not, forwarded-and-deferred on cache-miss, and
    fail-loud only on a malformed value.

    Contract: provider-protocol:complete:MUST:14
    """

    def test_none_default_returns_none(self) -> None:
        assert (
            resolve_provider_default_effort(None, _model_info(), model_id="m")
            is None
        )

    def test_empty_default_returns_none(self) -> None:
        assert (
            resolve_provider_default_effort("", _model_info(), model_id="m") is None
        )

    def test_supported_value_forwarded(self) -> None:
        info = _model_info(supports=True, allowlist=("low", "medium", "high"))
        assert (
            resolve_provider_default_effort("high", info, model_id="m") == "high"
        )

    def test_accepts_advertised_effort_outside_static_allowlist(self) -> None:
        """Symmetric with MUST:11: the resolved model's advertised
        ``supported_reasoning_efforts`` is authoritative and WIDENS acceptance
        beyond the static fallback allowlist. An operator default the model
        advertises (e.g. ``"minimal"`` for gemini-3.5-flash, outside the static
        superset) is applied verbatim rather than rejected by the shape gate.

        Contract: provider-protocol:complete:MUST:14
        """
        info = _model_info(
            supports=True, allowlist=("minimal", "low", "medium", "high")
        )
        assert (
            resolve_provider_default_effort(
                "minimal", info, model_id="gemini-3.5-flash"
            )
            == "minimal"
        )

    def test_advertised_shape_invalid_token_is_rejected_not_forwarded(self) -> None:
        """Symmetric with MUST:11: an operator default that is shape-invalid but
        appears in the resolved model's advertised ``supported_reasoning_efforts``
        (poisoned on-disk cache) is NOT early-accepted — it falls through to the
        shape gate and is rejected with the value redacted, never forwarded.

        Contract: provider-protocol:complete:MUST:14
        """
        poisoned = "ghp_" + "a" * 36
        info = _model_info(supports=True, allowlist=(poisoned, "low", "medium"))
        with pytest.raises(ConfigurationError) as excinfo:
            resolve_provider_default_effort(poisoned, info, model_id="claude-sonnet-4.6")
        msg = str(excinfo.value)
        assert poisoned not in msg, "raw poisoned token must not leak into error text"
        assert "redacted" in msg
        good = _model_info(supports=True, allowlist=("minimal", "low"))
        assert (
            resolve_provider_default_effort("minimal", good, model_id="gemini-3.5-flash")
            == "minimal"
        )

    def test_advertised_trailing_newline_token_is_rejected(self) -> None:
        """Symmetric with MUST:11: a trailing-newline operator default (e.g.
        ``"low\\n"``) advertised by the resolved model is rejected, not accepted.
        The shape gate is anchored with ``\\A...\\Z`` (not ``^...$``) so Python's
        ``$`` trailing-newline match cannot widen ``"low\\n"`` through.

        Regression guard for the ``$`` -> ``\\Z`` anchoring fix.

        Contract: provider-protocol:complete:MUST:14
        """
        newline_token = "low\n"
        info = _model_info(supports=True, allowlist=(newline_token, "medium"))
        with pytest.raises(ConfigurationError) as excinfo:
            resolve_provider_default_effort(
                newline_token, info, model_id="claude-sonnet-4.6"
            )
        assert "\n" not in str(excinfo.value), "raw newline must not reach the error text"

    def test_incapable_model_drops_to_none(self, caplog: pytest.LogCaptureFixture) -> None:
        info = _model_info(supports=False, allowlist=())
        with caplog.at_level(logging.INFO):
            assert (
                resolve_provider_default_effort("high", info, model_id="m") is None
            )
        assert "supports_reasoning_effort=False" in caplog.text
        # Pin the level: the operator-default drop MUST log at INFO (the caller
        # drop is WARNING). Asserting on record.levelno, not just caplog.text,
        # makes a silent INFO->WARNING upgrade fail the test.
        rec = next(
            r for r in caplog.records if "supports_reasoning_effort=False" in r.getMessage()
        )
        assert rec.levelno == logging.INFO

    def test_value_not_in_allowlist_drops_to_none(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        info = _model_info(supports=True, allowlist=("low", "medium"))
        with caplog.at_level(logging.INFO):
            assert (
                resolve_provider_default_effort("high", info, model_id="m") is None
            )
        assert "not in supported_reasoning_efforts" in caplog.text
        rec = next(
            r for r in caplog.records if "not in supported_reasoning_efforts" in r.getMessage()
        )
        assert rec.levelno == logging.INFO

    def test_cache_miss_forwards_verbatim(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO):
            assert (
                resolve_provider_default_effort("high", None, model_id="m") == "high"
            )
        assert "deferring" in caplog.text.lower()
        rec = next(r for r in caplog.records if "deferring" in r.getMessage().lower())
        assert rec.levelno == logging.INFO

    def test_malformed_shape_raises_on_capable_model(self) -> None:
        info = _model_info(supports=True, allowlist=("low", "medium", "high"))
        with pytest.raises(ConfigurationError, match="allowlist"):
            resolve_provider_default_effort("High", info, model_id="m")

    def test_malformed_shape_raises_on_incapable_model(self) -> None:
        info = _model_info(supports=False, allowlist=())
        with pytest.raises(ConfigurationError, match="allowlist"):
            resolve_provider_default_effort("banana", info, model_id="m")

    def test_malformed_shape_raises_on_cache_miss(self) -> None:
        with pytest.raises(ConfigurationError, match="allowlist"):
            resolve_provider_default_effort("High", None, model_id="m")


class TestReasoningEffortNoneLevel:
    """The "none" effort level (advertised by some live models, absent from the
    v1.0.7 SDK ``ReasoningEffort`` Literal) passes the universal shape gate and
    is then governed by the per-model allowlist, exactly like "max".

    Contract: provider-protocol:complete:MUST:11
    """

    def _info(self, efforts: tuple[str, ...]) -> Any:
        from amplifier_module_provider_github_copilot.sdk_adapter import (
            CopilotModelInfo,
        )

        return CopilotModelInfo(
            id="probe-model",
            name="Probe",
            context_window=272_000,
            max_output_tokens=128_000,
            supports_reasoning_effort=True,
            supported_reasoning_efforts=efforts,
        )

    def test_none_accepted_when_model_advertises_it(self) -> None:
        info = self._info(("none", "low", "medium", "high", "xhigh"))
        result = validate_reasoning_effort("none", info, model_id="probe-model")
        assert result == "none"

    def test_none_in_fallback_allowlist(self) -> None:
        from amplifier_module_provider_github_copilot.request_adapter import (
            _REASONING_EFFORT_FALLBACK_ALLOWLIST,
        )

        assert "none" in _REASONING_EFFORT_FALLBACK_ALLOWLIST

    def test_none_passes_universal_gate_on_cache_miss(self) -> None:
        result = validate_reasoning_effort("none", None, model_id="probe-model")
        assert result == "none"

    def test_none_dropped_when_model_excludes_it(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        info = self._info(("low", "medium", "high"))
        with caplog.at_level(
            logging.WARNING,
            logger="amplifier_module_provider_github_copilot.request_adapter",
        ):
            result = validate_reasoning_effort("none", info, model_id="probe-model")
        # Valid shape token excluded by the per-model allowlist → dropped (not
        # raised), same as "max".
        assert result is None
        assert "reasoning_effort='none'" in caplog.text
        assert "probe-model" in caplog.text
        assert "not in supported_reasoning_efforts" in caplog.text


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
        ["none", "low", "medium", "high", "xhigh", "max"],
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


class TestProviderDropsUnsupportedCallerEffort:
    """``provider.complete()`` must DROP a caller ``reasoning_effort`` the cached
    ``CopilotModelInfo`` cannot consume (``supports_reasoning_effort=False``),
    log a WARNING, and proceed with the SDK call carrying ``reasoning_effort=None``
    — the server then falls back. It must NOT raise.

    This is the real delegation-fix behavior: an inherited caller effort routed
    to a ``fast``-tier model that does not support it (e.g. ``claude-haiku-4.5``)
    must not abort the delegated turn. Symmetric with the MUST:14 operator-default
    drop; the value is omitted so the server falls back.

    Pins the wiring contract (the drop runs in ``provider.complete()`` and the
    dropped value reaches ``_execute_sdk_completion`` as ``None``); the unit tests
    for ``validate_reasoning_effort`` only cover the function in isolation.

    Contract: provider-protocol:complete:MUST:11
    """

    @pytest.mark.asyncio
    async def test_unsupported_cached_model_drops_and_calls_sdk_with_none(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        from amplifier_core import ChatResponse

        from amplifier_module_provider_github_copilot import provider as provider_mod
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )
        from amplifier_module_provider_github_copilot.sdk_adapter import (
            CopilotModelInfo,
        )

        # Cached capability descriptor: model is known but explicitly does
        # NOT support reasoning_effort. This is the "stale or correct cache"
        # path where Layer-1 drops the unsupported value locally.
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

        wrapper, session_calls = _make_capturing_wrapper(
            fake_text="ok",
            clean_text="ok",
        )

        request = MagicMock()
        request.model = "claude-haiku-4.5"
        request.messages = [MagicMock(role="user", content="hello")]
        request.attachments = None
        request.max_output_tokens = None
        # Caller (e.g. an inherited session/delegation effort) supplies an
        # effort the target model cannot consume.
        request.reasoning_effort = "high"
        request.context_tier = None
        request.tools = None

        provider = GitHubCopilotProvider(client=wrapper)  # type: ignore[arg-type]

        with caplog.at_level(
            logging.WARNING,
            logger="amplifier_module_provider_github_copilot.request_adapter",
        ):
            result = await provider.complete(request)

        # Delegated turn succeeds — no ConfigurationError.
        assert isinstance(result, ChatResponse)
        # The SDK was called, and the dropped effort reached it as None.
        assert len(session_calls) >= 1
        assert all(call["reasoning_effort"] is None for call in session_calls), (
            f"dropped caller effort must reach the SDK as None; got {session_calls!r}"
        )
        # The drop is observable via the WARNING channel.
        assert "Dropping caller reasoning_effort" in caplog.text
        assert "claude-haiku-4.5" in caplog.text
        assert "supports_reasoning_effort=False" in caplog.text

    @pytest.mark.asyncio
    async def test_dropped_caller_effort_retries_with_none_on_both_sites(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A caller effort DROPPED for an incapable model must reach BOTH
        ``_execute_sdk_completion`` call sites as ``None`` — the main path AND the
        fake-tool correction retry. The isolated-function drop tests and the
        both-sites threading tests each cover only half; this pins that the
        dropped (not merely forwarded) value is what the correction path re-sends.

        Contract: provider-protocol:complete:MUST:11
        """
        from amplifier_core import ChatResponse

        from amplifier_module_provider_github_copilot import provider as provider_mod
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )
        from amplifier_module_provider_github_copilot.sdk_adapter import (
            CopilotModelInfo,
        )

        unsupported_info = CopilotModelInfo(
            id="claude-haiku-4.5",
            name="Claude Haiku 4.5",
            context_window=200_000,
            max_output_tokens=8192,
            supports_vision=False,
            supports_reasoning_effort=False,
            supported_reasoning_efforts=(),
        )
        monkeypatch.setattr(
            provider_mod.GitHubCopilotProvider,
            "_lookup_copilot_model_info",
            lambda self, model_id: unsupported_info,
        )

        # fake-tool text on the first call + tools present → correction retry
        # fires, producing a second _execute_sdk_completion invocation.
        wrapper, session_calls = _make_capturing_wrapper(
            fake_text="[Tool Call: bash(command='ls')]",
            clean_text="ok",
        )

        request = MagicMock()
        request.model = "claude-haiku-4.5"
        request.messages = [MagicMock(role="user", content="list files")]
        request.attachments = None
        request.max_output_tokens = None
        request.reasoning_effort = "high"  # caller effort the model cannot consume
        request.context_tier = None
        request.tools = [
            {"name": "bash", "description": "Run shell commands", "parameters": {}}
        ]

        provider = GitHubCopilotProvider(client=wrapper)  # type: ignore[arg-type]

        with caplog.at_level(
            logging.WARNING,
            logger="amplifier_module_provider_github_copilot.request_adapter",
        ):
            result = await provider.complete(request)

        assert isinstance(result, ChatResponse)
        assert len(session_calls) == 2, (
            f"Expected main + correction = 2 SDK calls, got {session_calls!r}"
        )
        assert all(call["reasoning_effort"] is None for call in session_calls), (
            f"dropped caller effort must reach BOTH call sites as None; "
            f"got {session_calls!r}"
        )
        assert "Dropping caller reasoning_effort" in caplog.text

    @pytest.mark.asyncio
    async def test_dropped_caller_effort_does_not_retrigger_operator_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When a caller effort is DROPPED for an incapable model, the MUST:14
        operator-default path MUST NOT re-apply the stored default on the same
        model. The operator default only fills in when the caller supplied
        nothing (``internal_request.reasoning_effort is None``).

        Guards the provider.py gate: switching the trigger back to
        ``validated_reasoning_effort is None`` would make a dropped caller effort
        spuriously re-enter the operator-default resolve on the same incapable
        model (a redundant second drop, and — worse — an accidental apply on a
        capable-but-different resolve).

        Contract: provider-protocol:complete:MUST:11, MUST:14
        """
        from amplifier_core import ChatResponse

        from amplifier_module_provider_github_copilot import provider as provider_mod
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )
        from amplifier_module_provider_github_copilot.sdk_adapter import (
            CopilotModelInfo,
        )

        unsupported_info = CopilotModelInfo(
            id="claude-haiku-4.5",
            name="Claude Haiku 4.5",
            context_window=200_000,
            max_output_tokens=8192,
            supports_reasoning_effort=False,
            supported_reasoning_efforts=(),
        )
        monkeypatch.setattr(
            provider_mod.GitHubCopilotProvider,
            "_lookup_copilot_model_info",
            lambda self, model_id: unsupported_info,
        )

        wrapper, session_calls = _make_capturing_wrapper(
            fake_text="ok",
            clean_text="ok",
        )
        # Operator default configured AND caller supplies an effort.
        provider = GitHubCopilotProvider(
            config={"reasoning_effort": "high"},
            client=wrapper,  # type: ignore[arg-type]
        )

        request = MagicMock()
        request.model = "claude-haiku-4.5"
        request.messages = [MagicMock(role="user", content="hello")]
        request.attachments = None
        request.max_output_tokens = None
        request.reasoning_effort = "high"  # caller value, unsupported here
        request.context_tier = None
        request.tools = None

        with caplog.at_level(
            logging.INFO,
            logger="amplifier_module_provider_github_copilot.request_adapter",
        ):
            result = await provider.complete(request)

        assert isinstance(result, ChatResponse)
        assert all(call["reasoning_effort"] is None for call in session_calls)
        # Exactly the caller-scope WARNING drop fires; the operator-default INFO
        # drop ("Dropping provider default") MUST NOT appear.
        assert "Dropping caller reasoning_effort" in caplog.text
        assert "Dropping provider default reasoning_effort" not in caplog.text


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


# ----------------------------------------------------------------------------
# MUST:6 — get_info() exposes a reasoning_effort choice ConfigField
# ----------------------------------------------------------------------------


class TestGetInfoExposesReasoningEffortField:
    """get_info() MUST include a choice ConfigField for reasoning_effort.

    Contract: provider-protocol:get_info:MUST:6
    """

    def _get_field(self, field_id: str) -> Any:
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider()
        info = provider.get_info()
        for f in info.config_fields:
            if f.id == field_id:
                return f
        raise AssertionError(
            f"ConfigField '{field_id}' not found in get_info().config_fields; "
            f"ids present: {[f.id for f in info.config_fields]}"
        )

    def test_reasoning_effort_field_present_with_correct_type(self) -> None:
        """Field must have field_type='choice'."""
        f = self._get_field("reasoning_effort")
        assert f.field_type == "choice", (
            f"reasoning_effort ConfigField must be field_type='choice'; got {f.field_type!r}"
        )

    def test_reasoning_effort_field_default_is_model_default(self) -> None:
        """Default MUST be 'model default' so absence is a no-op."""
        f = self._get_field("reasoning_effort")
        assert f.default == "model default", (
            f"reasoning_effort ConfigField must default to 'model default'; got {f.default!r}"
        )

    def test_reasoning_effort_field_not_required(self) -> None:
        """Field is optional — omitting it is valid."""
        f = self._get_field("reasoning_effort")
        assert f.required is False, (
            f"reasoning_effort ConfigField must be required=False; got required={f.required!r}"
        )

    def test_reasoning_effort_field_requires_model(self) -> None:
        """Field only makes sense after a model is selected."""
        f = self._get_field("reasoning_effort")
        assert f.requires_model is True, (
            f"reasoning_effort ConfigField must have requires_model=True; "
            f"got requires_model={f.requires_model!r}"
        )

    def test_reasoning_effort_field_exact_choices(self) -> None:
        """Choices must be the exact ordered list including 'none' and 'max'."""
        f = self._get_field("reasoning_effort")
        expected = ["model default", "none", "low", "medium", "high", "xhigh", "max"]
        assert f.choices == expected, (
            f"reasoning_effort ConfigField choices mismatch.\n"
            f"  Expected: {expected}\n"
            f"  Got:      {f.choices!r}"
        )

    def test_reasoning_effort_choices_match_levels(self) -> None:
        """ConfigField choices MUST stay in sync with REASONING_EFFORT_LEVELS:
        the 'model default' sentinel followed by the shared ordered constant.

        Pins the single-source-of-truth wiring so an edit to the constant
        cannot silently diverge from the wizard choices.
        """
        from amplifier_module_provider_github_copilot.request_adapter import (
            REASONING_EFFORT_LEVELS,
        )

        f = self._get_field("reasoning_effort")
        assert f.choices == ["model default", *REASONING_EFFORT_LEVELS], (
            f"choices must equal ['model default', *REASONING_EFFORT_LEVELS]; "
            f"got {f.choices!r} vs levels {REASONING_EFFORT_LEVELS!r}"
        )

    def test_reasoning_effort_field_positioned_after_enable_long_context(self) -> None:
        """reasoning_effort MUST appear immediately after enable_long_context."""
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider()
        info = provider.get_info()
        ids = [f.id for f in info.config_fields]
        elc_idx = ids.index("enable_long_context")
        re_idx = ids.index("reasoning_effort")
        assert re_idx == elc_idx + 1, (
            f"reasoning_effort must appear immediately after enable_long_context; "
            f"field order: {ids}"
        )

    def test_enable_long_context_field_requires_model(self) -> None:
        """The preceding enable_long_context field is also model-scoped, so the
        wizard groups both model-dependent prompts after model selection."""
        f = self._get_field("enable_long_context")
        assert f.requires_model is True, (
            f"enable_long_context ConfigField must have requires_model=True; "
            f"got requires_model={f.requires_model!r}"
        )


# ----------------------------------------------------------------------------
# MUST:14 — provider-level reasoning_effort default
# ----------------------------------------------------------------------------


class TestReasoningEffortProviderDefault:
    """provider.complete() MUST apply the stored _reasoning_effort default when
    the caller passes None, and the caller value MUST win when present.

    Contract: provider-protocol:complete:MUST:14
    """

    @staticmethod
    def _make_request(
        reasoning_effort: str | None,
        *,
        with_tools: bool = True,
    ) -> MagicMock:
        request = MagicMock()
        request.model = "gpt-4o"
        request.messages = [MagicMock(role="user", content="list files")]
        request.attachments = None
        request.max_output_tokens = None
        request.reasoning_effort = reasoning_effort
        request.context_tier = None
        if with_tools:
            request.tools = [
                {"name": "bash", "description": "Run shell commands", "parameters": {}}
            ]
        else:
            request.tools = []
        return request

    @pytest.mark.asyncio
    async def test_default_applied_on_both_call_sites(self) -> None:
        """When caller passes None and config sets 'medium', both SDK session
        calls (main + correction retry) MUST receive reasoning_effort='medium'.

        Mutation check: removing the MUST:14 block from provider.complete()
        leaves both calls with None.
        """
        from amplifier_core import ChatResponse

        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        wrapper, session_calls = _make_capturing_wrapper(
            fake_text="[Tool Call: bash(command='ls')]",
            clean_text="ok",
        )
        provider = GitHubCopilotProvider(
            config={"reasoning_effort": "medium"},
            client=wrapper,  # type: ignore[arg-type]
        )
        result = await provider.complete(self._make_request(None, with_tools=True))

        assert len(session_calls) == 2, (
            f"Expected main + correction = 2 session calls, got {session_calls!r}"
        )
        assert session_calls[0]["reasoning_effort"] == "medium", (
            f"Main session: provider default not applied — "
            f"got {session_calls[0]['reasoning_effort']!r}"
        )
        assert session_calls[1]["reasoning_effort"] == "medium", (
            f"Correction session: provider default not applied on retry — "
            f"got {session_calls[1]['reasoning_effort']!r}"
        )
        assert isinstance(result, ChatResponse)

    @pytest.mark.asyncio
    async def test_provider_default_does_not_mutate_chat_request(self) -> None:
        """MUST:14: resolving the operator-configured default MUST NOT write the
        effective value back onto the ChatRequest — the resolved effort lives in
        a transient local and is handed to the SDK session directly. This pins
        the "MUST NOT mutate ChatRequest" clause of the MUST:14 row.

        Mutation check: replacing the local with a write-back
        (``request.reasoning_effort = <resolved>``) in the MUST:14 block flips
        the caller-visible attribute from None to 'medium' — this test goes red.

        Contract: provider-protocol:complete:MUST:14
        """
        from amplifier_core import ChatResponse

        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        wrapper, session_calls = _make_capturing_wrapper(
            fake_text="[Tool Call: bash(command='ls')]",
            clean_text="ok",
        )
        provider = GitHubCopilotProvider(
            config={"reasoning_effort": "medium"},
            client=wrapper,  # type: ignore[arg-type]
        )
        request = self._make_request(None, with_tools=True)
        result = await provider.complete(request)

        # The provider default reached the SDK (proves the MUST:14 path ran)...
        assert session_calls[0]["reasoning_effort"] == "medium", (
            f"Provider default not applied — got {session_calls[0]['reasoning_effort']!r}"
        )
        # ...but the ChatRequest attribute is untouched (no write-back).
        assert request.reasoning_effort is None, (
            "MUST:14 forbids mutating ChatRequest.reasoning_effort; the resolved "
            f"effective value must stay a local — got {request.reasoning_effort!r}"
        )
        assert request.context_tier is None, (
            "MUST:14 resolution must not touch unrelated request fields — "
            f"context_tier changed to {request.context_tier!r}"
        )
        assert isinstance(result, ChatResponse)

    @pytest.mark.asyncio
    async def test_caller_wins_over_provider_default(self) -> None:
        """An explicit caller reasoning_effort MUST take precedence over the
        provider-level default on both call sites.

        Mutation check: the MUST:14 guard applies the provider default only when
        the caller supplied nothing (``not internal_request.reasoning_effort``);
        dropping that conjunct so the default always resolves would replace the
        caller's value — this test goes red.
        """
        from amplifier_core import ChatResponse

        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        wrapper, session_calls = _make_capturing_wrapper(
            fake_text="[Tool Call: bash(command='ls')]",
            clean_text="ok",
        )
        provider = GitHubCopilotProvider(
            config={"reasoning_effort": "low"},
            client=wrapper,  # type: ignore[arg-type]
        )
        # Caller passes "high"; provider default is "low". "high" MUST win.
        result = await provider.complete(self._make_request("high", with_tools=True))

        assert len(session_calls) == 2, (
            f"Expected main + correction = 2 session calls, got {session_calls!r}"
        )
        assert session_calls[0]["reasoning_effort"] == "high", (
            f"Main session: caller value lost to provider default — "
            f"got {session_calls[0]['reasoning_effort']!r}"
        )
        assert session_calls[1]["reasoning_effort"] == "high", (
            f"Correction session: caller value lost to provider default — "
            f"got {session_calls[1]['reasoning_effort']!r}"
        )
        assert isinstance(result, ChatResponse)

    @pytest.mark.asyncio
    async def test_provider_default_unsupported_model_skips_and_forwards_none(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """MUST:14 (b): when the caller omits reasoning_effort and the
        operator-configured default targets a model with
        ``supports_reasoning_effort=False`` (e.g. a ``fast`` role delegated to
        haiku), the default is DROPPED — effective ``None`` forwarded on BOTH
        call sites — and an INFO log is emitted. It MUST NOT raise. Fail-loud is
        reserved for malformed operator defaults (the shared shape gate); a
        capability mismatch is dropped, not raised, on both the operator-default
        and caller paths.

        Distinct from TestProviderDropsUnsupportedCallerEffort, which pins the
        explicit-caller path (drop to None + WARNING). This pins the
        provider-default branch of the MUST:14 guard specifically.

        Mutation check: reverting ``resolve_provider_default_effort`` to raise
        (or routing the stored default back through ``validate_reasoning_effort``)
        reintroduces the ConfigurationError and breaks ``fast``->haiku
        delegation — session_calls goes empty and the raise reappears; red.

        Contract: provider-protocol:complete:MUST:14
        """
        from amplifier_core import ChatResponse

        from amplifier_module_provider_github_copilot import provider as provider_mod
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )
        from amplifier_module_provider_github_copilot.sdk_adapter import (
            CopilotModelInfo,
        )

        unsupported_info = CopilotModelInfo(
            id="gpt-4o",
            name="GPT-4o",
            context_window=128_000,
            max_output_tokens=16_384,
            supports_vision=False,
            supports_reasoning_effort=False,
            supported_reasoning_efforts=(),
        )
        monkeypatch.setattr(
            provider_mod.GitHubCopilotProvider,
            "_lookup_copilot_model_info",
            lambda self, model_id: unsupported_info,
        )

        wrapper, session_calls = _make_capturing_wrapper(
            fake_text="[Tool Call: bash(command='ls')]",
            clean_text="ok",
        )
        provider = GitHubCopilotProvider(
            config={"reasoning_effort": "medium"},
            client=wrapper,  # type: ignore[arg-type]
        )

        with caplog.at_level(
            logging.INFO,
            logger="amplifier_module_provider_github_copilot.request_adapter",
        ):
            result = await provider.complete(
                self._make_request(None, with_tools=True)
            )

        assert len(session_calls) == 2, (
            f"Expected main + correction = 2 session calls (default dropped, not "
            f"raised); got {session_calls!r}"
        )
        assert all(c["reasoning_effort"] is None for c in session_calls), (
            f"Provider default MUST drop to None on both call sites for an "
            f"unsupported model; got {[c['reasoning_effort'] for c in session_calls]!r}"
        )
        assert "Dropping provider default reasoning_effort" in caplog.text, (
            "MUST:14 (b) requires one INFO log when the operator default is "
            f"dropped; caplog was: {caplog.text!r}"
        )
        assert isinstance(result, ChatResponse)

    @pytest.mark.asyncio
    async def test_provider_default_not_in_allowlist_skips(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """MUST:14 (c): a reasoning-capable model whose non-empty
        ``supported_reasoning_efforts`` excludes the operator default drops the
        default to ``None`` (both call sites) with an INFO log — symmetric with
        the ``supports_reasoning_effort=False`` branch; NOT a raise.

        Contract: provider-protocol:complete:MUST:14
        """
        from amplifier_core import ChatResponse

        from amplifier_module_provider_github_copilot import provider as provider_mod
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )
        from amplifier_module_provider_github_copilot.sdk_adapter import (
            CopilotModelInfo,
        )

        restricted_info = CopilotModelInfo(
            id="gpt-4o",
            name="GPT-4o",
            context_window=128_000,
            max_output_tokens=16_384,
            supports_reasoning_effort=True,
            supported_reasoning_efforts=("low", "medium"),
        )
        monkeypatch.setattr(
            provider_mod.GitHubCopilotProvider,
            "_lookup_copilot_model_info",
            lambda self, model_id: restricted_info,
        )

        wrapper, session_calls = _make_capturing_wrapper(
            fake_text="[Tool Call: bash(command='ls')]",
            clean_text="ok",
        )
        provider = GitHubCopilotProvider(
            config={"reasoning_effort": "high"},  # not in {low, medium}
            client=wrapper,  # type: ignore[arg-type]
        )

        with caplog.at_level(
            logging.INFO,
            logger="amplifier_module_provider_github_copilot.request_adapter",
        ):
            result = await provider.complete(
                self._make_request(None, with_tools=True)
            )

        assert len(session_calls) == 2
        assert all(c["reasoning_effort"] is None for c in session_calls), (
            f"Allowlist-excluded default MUST drop to None on both sites; got "
            f"{[c['reasoning_effort'] for c in session_calls]!r}"
        )
        assert "not in supported_reasoning_efforts" in caplog.text
        assert isinstance(result, ChatResponse)

    @pytest.mark.asyncio
    async def test_provider_default_cache_miss_forwards_on_both_sites(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """MUST:14 (d): on cache-miss (``CopilotModelInfo`` unavailable) the
        operator default is FORWARDED verbatim on BOTH call sites and an INFO
        log defers final per-model validation to the SDK Layer-2 backstop —
        matching MUST:11 cache-miss semantics (keeps a genuine Layer-2 error
        loud rather than silently masking it).

        Asserting BOTH sites (not just the final call) guards against a
        divergent first call.

        Contract: provider-protocol:complete:MUST:14
        """
        from amplifier_core import ChatResponse

        from amplifier_module_provider_github_copilot import provider as provider_mod
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        monkeypatch.setattr(
            provider_mod.GitHubCopilotProvider,
            "_lookup_copilot_model_info",
            lambda self, model_id: None,
        )

        wrapper, session_calls = _make_capturing_wrapper(
            fake_text="[Tool Call: bash(command='ls')]",
            clean_text="ok",
        )
        provider = GitHubCopilotProvider(
            config={"reasoning_effort": "high"},
            client=wrapper,  # type: ignore[arg-type]
        )

        with caplog.at_level(
            logging.INFO,
            logger="amplifier_module_provider_github_copilot.request_adapter",
        ):
            result = await provider.complete(
                self._make_request(None, with_tools=True)
            )

        assert len(session_calls) == 2, (
            f"Expected main + correction = 2 session calls; got {session_calls!r}"
        )
        assert [c["reasoning_effort"] for c in session_calls] == ["high", "high"], (
            f"Cache-miss default MUST forward verbatim on BOTH call sites; got "
            f"{[c['reasoning_effort'] for c in session_calls]!r}"
        )
        assert "deferring" in caplog.text.lower(), (
            f"MUST:14 (d) requires an INFO log deferring to Layer-2; caplog "
            f"was: {caplog.text!r}"
        )
        assert isinstance(result, ChatResponse)

    @pytest.mark.asyncio
    async def test_provider_default_applied_on_capable_model_both_sites(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MUST:14 (e): a reasoning-capable model that advertises the operator
        default value forwards it verbatim on BOTH call sites — the softening
        of the incapable path MUST NOT drop a legitimately-supported default.

        Contract: provider-protocol:complete:MUST:14
        """
        from amplifier_core import ChatResponse

        from amplifier_module_provider_github_copilot import provider as provider_mod
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )
        from amplifier_module_provider_github_copilot.sdk_adapter import (
            CopilotModelInfo,
        )

        capable_info = CopilotModelInfo(
            id="gpt-4o",
            name="GPT-4o",
            context_window=128_000,
            max_output_tokens=16_384,
            supports_reasoning_effort=True,
            supported_reasoning_efforts=("low", "medium", "high"),
        )
        monkeypatch.setattr(
            provider_mod.GitHubCopilotProvider,
            "_lookup_copilot_model_info",
            lambda self, model_id: capable_info,
        )

        wrapper, session_calls = _make_capturing_wrapper(
            fake_text="[Tool Call: bash(command='ls')]",
            clean_text="ok",
        )
        provider = GitHubCopilotProvider(
            config={"reasoning_effort": "high"},
            client=wrapper,  # type: ignore[arg-type]
        )
        result = await provider.complete(self._make_request(None, with_tools=True))

        assert len(session_calls) == 2
        assert [c["reasoning_effort"] for c in session_calls] == ["high", "high"], (
            f"Capable model MUST receive the supported default on both sites; "
            f"got {[c['reasoning_effort'] for c in session_calls]!r}"
        )
        assert isinstance(result, ChatResponse)

    @pytest.mark.asyncio
    async def test_empty_string_caller_effort_triggers_operator_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MUST:14: an empty-string caller ``reasoning_effort`` is equivalent to
        omission and MUST trigger the operator default.

        The ChatRequest membrane normalizes ``""`` to ``None``, but a
        ``CompletionRequest`` reaches ``complete()`` with ``""`` un-normalized.
        Both MUST:11 and MUST:14 gate on a FALSY ``reasoning_effort`` (not
        strictly ``is None``), so ``""`` behaves like ``None``: the operator
        default is applied on BOTH call sites rather than the request silently
        forwarding ``None`` (no effort) on a capable model.

        Regression for the review finding that ``CompletionRequest("")`` bypassed
        the provider default while ``ChatRequest("")`` received it.

        Mutation check: reverting either gate to ``is None`` leaves both calls
        with ``None`` (default not applied) — red.

        Contract: provider-protocol:complete:MUST:14
        """
        from amplifier_core import ChatResponse

        from amplifier_module_provider_github_copilot import provider as provider_mod
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )
        from amplifier_module_provider_github_copilot.sdk_adapter import (
            CompletionRequest,
            CopilotModelInfo,
        )

        capable_info = CopilotModelInfo(
            id="gpt-4o",
            name="GPT-4o",
            context_window=128_000,
            max_output_tokens=16_384,
            supports_reasoning_effort=True,
            supported_reasoning_efforts=("low", "medium", "high"),
        )
        monkeypatch.setattr(
            provider_mod.GitHubCopilotProvider,
            "_lookup_copilot_model_info",
            lambda self, model_id: capable_info,
        )

        wrapper, session_calls = _make_capturing_wrapper(
            fake_text="[Tool Call: bash(command='ls')]",
            clean_text="ok",
        )
        provider = GitHubCopilotProvider(
            config={"reasoning_effort": "high"},
            client=wrapper,  # type: ignore[arg-type]
        )
        # A real CompletionRequest passes through convert_chat_request UNCHANGED
        # (the isinstance(request, CompletionRequest) passthrough), so "" is NOT
        # normalized to None and reaches the MUST:11/MUST:14 gate verbatim — the
        # exact path the fix addresses. A MagicMock would instead be normalized by
        # convert_chat_request's ChatRequest branch and never exercise the
        # falsy-gate mutation (reverting the gate to `is None` would still pass).
        request = CompletionRequest(
            prompt="list files",
            model="gpt-4o",
            tools=[
                {"name": "bash", "description": "Run shell commands", "parameters": {}}
            ],
            reasoning_effort="",
        )
        # complete() is typed for ChatRequest; convert_chat_request accepts a
        # CompletionRequest via its documented passthrough (request_adapter.py),
        # which is exactly the un-normalized-"" path under test.
        result = await provider.complete(request)  # type: ignore[arg-type]

        assert len(session_calls) == 2
        assert [c["reasoning_effort"] for c in session_calls] == ["high", "high"], (
            f"Empty-string caller effort MUST be treated as omission and receive "
            f"the operator default on both sites; got "
            f"{[c['reasoning_effort'] for c in session_calls]!r}"
        )
        assert isinstance(result, ChatResponse)

    @pytest.mark.asyncio
    async def test_provider_default_malformed_shape_raises_on_capable_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MUST:14 (a): a malformed operator default (mixed-case ``"High"``)
        is an operator misconfiguration and MUST raise ConfigurationError
        before any SDK call, EVEN on a reasoning-capable model. This is the
        critical guard against over-broadening the skip into the shape gate.

        Contract: provider-protocol:complete:MUST:14
        """
        from amplifier_module_provider_github_copilot import provider as provider_mod
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )
        from amplifier_module_provider_github_copilot.sdk_adapter import (
            CopilotModelInfo,
        )

        capable_info = CopilotModelInfo(
            id="gpt-4o",
            name="GPT-4o",
            context_window=128_000,
            max_output_tokens=16_384,
            supports_reasoning_effort=True,
            supported_reasoning_efforts=("low", "medium", "high"),
        )
        monkeypatch.setattr(
            provider_mod.GitHubCopilotProvider,
            "_lookup_copilot_model_info",
            lambda self, model_id: capable_info,
        )

        wrapper, session_calls = _make_capturing_wrapper(
            fake_text="ok",
            clean_text="ok",
        )
        provider = GitHubCopilotProvider(
            config={"reasoning_effort": "High"},  # mixed-case typo
            client=wrapper,  # type: ignore[arg-type]
        )

        with pytest.raises(ConfigurationError, match="allowlist"):
            await provider.complete(self._make_request(None, with_tools=False))

        assert session_calls == [], (
            f"SDK session MUST NOT be called when the operator default fails the "
            f"shape gate; got {session_calls!r}"
        )

    @pytest.mark.asyncio
    async def test_provider_default_malformed_shape_raises_on_incapable_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MUST:14 (a'): the shape gate is checked BEFORE the capability
        branches, so a malformed operator default raises even when the resolved
        model is incapable — a broken config must not be masked by
        incapability.

        Contract: provider-protocol:complete:MUST:14
        """
        from amplifier_module_provider_github_copilot import provider as provider_mod
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )
        from amplifier_module_provider_github_copilot.sdk_adapter import (
            CopilotModelInfo,
        )

        unsupported_info = CopilotModelInfo(
            id="gpt-4o",
            name="GPT-4o",
            context_window=128_000,
            max_output_tokens=16_384,
            supports_reasoning_effort=False,
            supported_reasoning_efforts=(),
        )
        monkeypatch.setattr(
            provider_mod.GitHubCopilotProvider,
            "_lookup_copilot_model_info",
            lambda self, model_id: unsupported_info,
        )

        wrapper, session_calls = _make_capturing_wrapper(
            fake_text="ok",
            clean_text="ok",
        )
        provider = GitHubCopilotProvider(
            config={"reasoning_effort": "banana"},  # not a valid token
            client=wrapper,  # type: ignore[arg-type]
        )

        with pytest.raises(ConfigurationError, match="allowlist"):
            await provider.complete(self._make_request(None, with_tools=False))

        assert session_calls == []

    @pytest.mark.asyncio
    async def test_model_default_config_value_means_no_effort(self) -> None:
        """Config value 'model default' normalises to None — no effort forwarded."""
        from amplifier_core import ChatResponse

        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        wrapper, session_calls = _make_capturing_wrapper(
            fake_text="ok",
            clean_text="ok",
        )
        provider = GitHubCopilotProvider(
            config={"reasoning_effort": "model default"},
            client=wrapper,  # type: ignore[arg-type]
        )
        result = await provider.complete(self._make_request(None, with_tools=False))

        assert len(session_calls) == 1, (
            f"Expected 1 session call (no retry without tools), got {session_calls!r}"
        )
        assert session_calls[0]["reasoning_effort"] is None, (
            f"Config 'model default' must normalise to no effort forwarded; "
            f"got {session_calls[0]['reasoning_effort']!r}"
        )
        assert isinstance(result, ChatResponse)

    @pytest.mark.asyncio
    async def test_model_default_config_value_trimmed_and_case_insensitive(self) -> None:
        """' Model Default ' (surrounding whitespace, mixed case) normalises to
        None — the sentinel is matched after strip().lower(), not literally."""
        from amplifier_core import ChatResponse

        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        wrapper, session_calls = _make_capturing_wrapper(
            fake_text="ok",
            clean_text="ok",
        )
        provider = GitHubCopilotProvider(
            config={"reasoning_effort": "  Model Default  "},
            client=wrapper,  # type: ignore[arg-type]
        )
        result = await provider.complete(self._make_request(None, with_tools=False))

        assert len(session_calls) == 1, (
            f"Expected 1 session call (no retry without tools), got {session_calls!r}"
        )
        assert session_calls[0]["reasoning_effort"] is None, (
            f"' Model Default ' must normalise to no effort forwarded; "
            f"got {session_calls[0]['reasoning_effort']!r}"
        )
        assert isinstance(result, ChatResponse)

    @pytest.mark.asyncio
    async def test_no_config_means_no_effort(self) -> None:
        """Absent reasoning_effort config → no effort forwarded to SDK."""
        from amplifier_core import ChatResponse

        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        wrapper, session_calls = _make_capturing_wrapper(
            fake_text="ok",
            clean_text="ok",
        )
        provider = GitHubCopilotProvider(client=wrapper)  # type: ignore[arg-type]
        result = await provider.complete(self._make_request(None, with_tools=False))

        assert len(session_calls) == 1, (
            f"Expected 1 session call (no retry without tools), got {session_calls!r}"
        )
        assert session_calls[0]["reasoning_effort"] is None, (
            f"No config must mean no effort forwarded; got {session_calls[0]['reasoning_effort']!r}"
        )
        assert isinstance(result, ChatResponse)

    @pytest.mark.asyncio
    async def test_non_str_config_value_normalises_to_none(self) -> None:
        """A non-str reasoning_effort config value MUST normalise to None
        without raising at construction — the isinstance guard prevents the
        ``.strip()`` AttributeError a bare truthy non-str would trigger.

        Mutation check: reverting the init to ``config.get(...) or ""`` makes
        construction raise AttributeError on the int and turns this test red.
        """
        from amplifier_core import ChatResponse

        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        wrapper, session_calls = _make_capturing_wrapper(fake_text="ok", clean_text="ok")
        provider = GitHubCopilotProvider(
            config={"reasoning_effort": 123},  # type: ignore[dict-item]
            client=wrapper,  # type: ignore[arg-type]
        )
        assert provider._reasoning_effort is None  # pyright: ignore[reportPrivateUsage]
        result = await provider.complete(self._make_request(None, with_tools=False))
        assert session_calls[0]["reasoning_effort"] is None
        assert isinstance(result, ChatResponse)

    @pytest.mark.asyncio
    async def test_config_value_stripped_before_store(self) -> None:
        """Surrounding whitespace on a real effort value MUST be stripped at
        store time so the case-sensitive validate gate sees the bare token and
        forwards it; case is preserved (not normalised).

        Mutation check: storing the raw (unstripped) value forwards ' high '
        which the validate gate rejects — this test goes red.
        """
        from amplifier_core import ChatResponse

        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        wrapper, session_calls = _make_capturing_wrapper(
            fake_text="[Tool Call: bash(command='ls')]",
            clean_text="ok",
        )
        provider = GitHubCopilotProvider(
            config={"reasoning_effort": "  high  "},
            client=wrapper,  # type: ignore[arg-type]
        )
        assert provider._reasoning_effort == "high"  # pyright: ignore[reportPrivateUsage]
        result = await provider.complete(self._make_request(None, with_tools=True))
        assert session_calls[0]["reasoning_effort"] == "high", (
            f"stripped provider default not forwarded — "
            f"got {session_calls[0]['reasoning_effort']!r}"
        )
        assert session_calls[1]["reasoning_effort"] == "high"
        assert isinstance(result, ChatResponse)
