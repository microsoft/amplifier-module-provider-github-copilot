"""Tests for SDK event classification gaps.

Contract anchors:
- event-vocabulary:Classification:MUST:1 (each event has exactly one classification)
- event-vocabulary:Drop:MUST:2 (no SDK enum member produces "Unknown SDK event type" warning)

Locks in that every SDK ``SessionEventType`` enum member is explicitly
classified (BRIDGE/CONSUME/DROP) and never falls through to the
"Unknown SDK event type" warning in ``streaming.py``.

Note on SDK availability: ``github-copilot-sdk`` is a hard runtime
dependency of this provider (``pyproject.toml`` install_requires). Tests
in this file import ``copilot.generated.session_events.SessionEventType``
unconditionally; if the import fails, that is a developer-environment
defect (see README "Prerequisites") and the test SHOULD fail loudly
rather than silently skip.
"""

from __future__ import annotations

import logging
from typing import cast

import pytest

from amplifier_module_provider_github_copilot.streaming import (
    EventClassification,
    classify_event,
    load_event_config,
    translate_event,
)

# ----------------------------------------------------------------------------
# T1: meta-test — every live SDK SessionEventType is classified
# ----------------------------------------------------------------------------


class TestEventClassificationCoversLiveSDKEnum:
    """Every member of the live SDK ``SessionEventType`` enum MUST classify
    without producing the "Unknown SDK event type" warning.

    Contract anchors:
    - event-vocabulary:Classification:MUST:1
    - event-vocabulary:Drop:MUST:2 (primary — this test enforces "no unknown
      warning for any SDK enum member")

    Mutation check: removing any explicit entry (e.g., ``commands.changed``)
    from events.yaml makes the corresponding enum member fall through to the
    unknown-event warning — red.

    Drift guard: if the SDK enum shrinks below ten string members, the
    canonical import path likely moved or the SDK was partially installed;
    the test fails loudly rather than silently providing thin coverage.
    """

    @staticmethod
    def _live_sdk_event_type_values() -> list[str]:
        """Return the live SDK ``SessionEventType`` string members.

        SDK 0.3.0 ships ``SessionEventType`` at
        ``copilot.generated.session_events`` (NOT ``copilot.session``); see
        the SDK source at ``copilot/generated/session_events.py:106``. The
        canonical generated path is the single source of truth; an
        ``ImportError`` here means the SDK is not installed and the
        developer needs to fix their environment per README "Prerequisites".
        """
        from copilot.generated.session_events import (  # type: ignore[import-not-found]
            SessionEventType,
        )

        values: list[str] = []
        for member in SessionEventType:
            value = member.value
            if isinstance(value, str) and value != "unknown":
                values.append(value)
        return values

    def test_no_unknown_warnings_for_real_sdk_event_types(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        live_values = self._live_sdk_event_type_values()
        # Drift guard: SDK 0.3.0 ships 70+ string members. If the live enum
        # has fewer than ten, the canonical import path likely moved or the
        # SDK is partially installed — fail loudly, never silently.
        assert len(live_values) >= 10, (
            f"copilot.generated.session_events.SessionEventType yielded only "
            f"{len(live_values)} string members; SDK 0.3.0 ships 70+. "
            f"Investigate: SDK partially installed, or the canonical enum "
            f"location moved."
        )
        config = load_event_config()
        unknown: list[str] = []
        with caplog.at_level(
            logging.WARNING,
            logger="amplifier_module_provider_github_copilot.streaming",
        ):
            for value in live_values:
                caplog.clear()
                classification = classify_event(value, config)
                assert classification in (
                    EventClassification.BRIDGE,
                    EventClassification.CONSUME,
                    EventClassification.DROP,
                )
                if any("Unknown SDK event type" in rec.message for rec in caplog.records):
                    unknown.append(value)

        assert not unknown, (
            f"SDK SessionEventType members fell through to the unknown-event "
            f"warning instead of an explicit classification: {unknown!r}. "
            f"Add each to events.yaml under bridge_mappings/consume/drop. "
            f"This meta-test is the gate that prevents silent classification "
            f"gaps when the SDK enum grows."
        )


# ----------------------------------------------------------------------------
# T2: commands.changed (plural) — explicit DROP, no unknown warning
# ----------------------------------------------------------------------------


class TestExplicitDropEntriesEmitNoUnknownWarning:
    """Explicit DROP entries — ``commands.changed`` plus the SDK 0.3.0 enum
    members added in this PR — MUST classify as DROP and MUST NOT trigger
    the "Unknown SDK event type" warning in ``streaming.py``.

    The plural ``commands.changed`` is NOT matched by the existing
    ``command.*`` (singular) wildcard, so it requires its own literal entry.
    The SDK 0.3.0 entries (``auto_mode_switch.*``, ``mcp.oauth_*``,
    ``sampling.*``, several ``session.*`` lifecycle events) likewise have
    no domain value for amplifier-core and require explicit DROP.

    Contract anchors:
    - event-vocabulary:Classification:MUST:1
    - event-vocabulary:Drop:MUST:2
    """

    def test_commands_changed_drops_without_unknown_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        config = load_event_config()
        with caplog.at_level(
            logging.WARNING,
            logger="amplifier_module_provider_github_copilot.streaming",
        ):
            result = classify_event("commands.changed", config)
        assert result == EventClassification.DROP
        assert not any("Unknown SDK event type" in rec.message for rec in caplog.records), (
            "commands.changed produced the unknown-event warning despite an "
            "explicit DROP entry in events.yaml; check that the entry is "
            "present and not shadowed."
        )

    @pytest.mark.parametrize(
        "event_type",
        [
            # SDK 0.3.0 enum members
            # (copilot.generated.session_events.SessionEventType) that fell
            # through to the unknown-event warning before this PR added
            # explicit DROP entries.
            "auto_mode_switch.requested",
            "auto_mode_switch.completed",
            "mcp.oauth_required",
            "mcp.oauth_completed",
            "sampling.requested",
            "sampling.completed",
            "session.extensions_loaded",
            "session.mcp_servers_loaded",
            "session.mcp_server_status_changed",
            "session.remote_steerable_changed",
            # SDK v1.0.2 enum members added to the events.yaml DROP block.
            # todos_changed is signal-only (no payload);
            # canvas.closed carries only IDs (canvas surface disabled);
            # binary_asset carries canonical bytes but is structurally
            # unreachable under MinimalMode + deny-destroy (see the dedicated
            # tripwire test below).
            "session.todos_changed",
            "session.binary_asset",
            "session.canvas.closed",
            # SDK v1.0.6 (bundled CLI 1.0.69) enum members added to the
            # events.yaml DROP block. All lifecycle/telemetry/MCP/canvas events
            # with no kernel domain mapping; see events.yaml for per-event
            # rationale. assistant.tool_call_delta (partial tool-call input) and
            # assistant.idle (deferred-idle marker) are DROP so they do NOT
            # corrupt the content stream / prematurely emit TURN_COMPLETE.
            "assistant.tool_call_delta",
            "assistant.idle",
            "session.usage_checkpoint",
            "session.session_limits_changed",
            "session.schedule_rearmed",
            "session_limits_exhausted.requested",
            "session_limits_exhausted.completed",
            "mcp.headers_refresh_required",
            "mcp.headers_refresh_completed",
            "session.canvas.unavailable",
            "session.canvas.recorded",
            "session.canvas.removed",
            # SDK v1.0.7 (bundled CLI 1.0.70) enum members added to the
            # events.yaml DROP block. All lack a kernel domain mapping; see
            # events.yaml for per-event rationale. server_tool_progress is
            # partial hosted-server-tool telemetry (result rides
            # assistant.message); auto_mode_resolved / managed_settings_resolved
            # are EXPERIMENTAL analytics/policy snapshots; the mcp.*.list_changed
            # trio are MCP list-change notifications with no subscription surface.
            "assistant.server_tool_progress",
            "session.auto_mode_resolved",
            "session.managed_settings_resolved",
            "mcp.tools.list_changed",
            "mcp.resources.list_changed",
            "mcp.prompts.list_changed",
        ],
    )
    def test_sdk_030_event_types_classified_without_warning(
        self, caplog: pytest.LogCaptureFixture, event_type: str
    ) -> None:
        """SDK 0.3.0 enum members added to DROP must classify as DROP without
        producing the "Unknown SDK event type" warning.

        Each event below was observed to fall through to the
        ``"Unknown SDK event type"`` WARNING path in ``streaming.py`` before
        this PR's events.yaml additions. This test pins the explicit DROP
        classifications so removing any one entry surfaces here as a red test.

        Mutation check (per `testing.instructions.md`): delete any single
        event from the DROP block in events.yaml — the matching parameter
        case here goes red on BOTH the classification equality assertion
        AND the "no unknown warning" assertion.

        Contract anchors:
        - event-vocabulary:Classification:MUST:1
        - event-vocabulary:Drop:MUST:2
        """
        config = load_event_config()
        with caplog.at_level(
            logging.WARNING,
            logger="amplifier_module_provider_github_copilot.streaming",
        ):
            classification = classify_event(event_type, config)
        # Assert exact DROP, not merely "no warning": a regression that
        # classifies the event as BRIDGE/CONSUME but still avoids the
        # unknown-event warning would silently corrupt the streaming contract.
        assert classification == EventClassification.DROP, (
            f"{event_type!r} classified as {classification.name} but should be "
            f"DROP. This event has no domain value for amplifier-core; "
            f"forwarding it as BRIDGE/CONSUME would corrupt the streaming "
            f"contract."
        )
        assert not any("Unknown SDK event type" in rec.message for rec in caplog.records), (
            f"{event_type!r} fell through to unknown-event warning"
        )


class TestBinaryAssetTripwire:
    """``session.binary_asset`` is a classified DROP that ALSO emits a
    metadata-only DEBUG tripwire in ``translate_event``.

    ``session.binary_asset`` is the one DROP-classified event that carries
    actual binary bytes (its ``data`` field — base64). The SDK persists it when
    a TOOL returns binary results for the LLM (``binaryResultsForLlm``). Under
    MinimalMode + deny-destroy the SDK executes no tools (``ToolCaptureHandler``
    aborts at the tool-REQUEST boundary), so this event is structurally
    unreachable. It stays DROP, but if it EVER fires an isolation invariant has
    broken — so the provider records a metadata-only tripwire (asset_id,
    mime_type, byte_length) and NEVER reads or logs the base64 ``data`` bytes.

    Contract anchors:
    - event-vocabulary:Classification:MUST:1
    - event-vocabulary:Drop:MUST:2
    - event-vocabulary:Drop:MUST:3
    """

    def test_binary_asset_drops_and_emits_metadata_only_tripwire(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        config = load_event_config()
        # A distinctive fake base64 payload that MUST NOT appear in any log.
        secret_bytes = "QUJD" + "U0VDUkVUQllURVM" * 64
        sdk_event = {
            "type": "session.binary_asset",
            "data": {
                "asset_id": "asset-deadbeef",
                "mime_type": "image/png",
                "byte_length": 4096,
                "data": secret_bytes,
                "type": "image",
            },
        }
        with caplog.at_level(
            logging.DEBUG,
            logger="amplifier_module_provider_github_copilot.streaming",
        ):
            result = translate_event(sdk_event, config)

        # Classified DROP => no domain event forwarded.
        assert result is None, (
            "session.binary_asset must DROP (translate_event returns None); a "
            "non-None result means it was forwarded as BRIDGE/CONSUME."
        )

        # The metadata-only tripwire fired at DEBUG.
        tripwire = [r for r in caplog.records if "binary-asset tripwire" in r.getMessage()]
        assert tripwire, "binary_asset DEBUG tripwire line was not emitted"
        msg = tripwire[0].getMessage()
        assert "asset-deadbeef" in msg
        assert "image/png" in msg
        assert "4096" in msg

        # SECURITY ASSERTION: the base64 `data` bytes must NEVER reach any log.
        assert secret_bytes not in caplog.text, (
            "binary_asset `data` bytes leaked into the log — the tripwire must "
            "log metadata ONLY."
        )

        # The classified DROP must not produce the unknown-event WARNING either.
        assert not any(
            "Unknown SDK event type" in rec.message for rec in caplog.records
        ), "session.binary_asset fell through to the unknown-event warning path"

    def test_binary_asset_tripwire_never_reads_data_bytes(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Fail-closed proof that the tripwire NEVER touches the ``data`` field.

        The earlier test proves the bytes are never *logged*. This one proves
        they are never *read*: the asset payload raises if its ``data`` member
        is accessed by ANY route (attribute, item, or ``.get``). ``translate_event``
        must still classify the event as DROP, emit the metadata tripwire, and
        return None WITHOUT raising. If the implementation ever reaches for the
        canonical bytes, this test detonates. Contract: event-vocabulary:Drop:MUST:3.
        """

        class _DataTrap:
            """Object payload whose canonical ``data`` is booby-trapped."""

            asset_id = "asset-trap"
            mime_type = "application/octet-stream"
            byte_length = 8192
            type = "resource"

            @property
            def data(self) -> str:  # pragma: no cover - must never be hit
                raise AssertionError(
                    "translate_event read the binary_asset `data` bytes — "
                    "the tripwire must access metadata fields ONLY."
                )

        class _DataTrapDict(dict[str, object]):
            """Dict payload that raises if the ``data`` key is read."""

            def __getitem__(self, key: object) -> object:
                if key == "data":  # pragma: no cover - must never be hit
                    raise AssertionError("read binary_asset `data` via __getitem__")
                return super().__getitem__(cast(str, key))

            def get(self, key: object, default: object = None) -> object:  # type: ignore[override]
                if key == "data":  # pragma: no cover - must never be hit
                    raise AssertionError("read binary_asset `data` via .get")
                return super().get(cast(str, key), default)

            # The production tripwire reads metadata via per-field `.get(field)`
            # ONLY (streaming._binary_asset_meta) and MUST NEVER enumerate the
            # asset payload. Bulk access (dict(payload), .values(), .items(),
            # iteration) is an alternate route to the `data` bytes that the
            # per-key guards above would miss. Detonate on every enumeration
            # entry point so the "never reads data" proof is fail-closed across
            # ALL access shapes, not just keyed reads.
            def keys(self) -> object:  # type: ignore[override]
                raise AssertionError(  # pragma: no cover - must never be hit
                    "enumerated binary_asset payload via .keys()"
                )

            def values(self) -> object:  # type: ignore[override]
                raise AssertionError(  # pragma: no cover - must never be hit
                    "enumerated binary_asset payload via .values()"
                )

            def items(self) -> object:  # type: ignore[override]
                raise AssertionError(  # pragma: no cover - must never be hit
                    "enumerated binary_asset payload via .items()"
                )

            def __iter__(self) -> object:  # type: ignore[override]
                raise AssertionError(  # pragma: no cover - must never be hit
                    "iterated binary_asset payload via __iter__"
                )

        config = load_event_config()

        for payload in (
            _DataTrap(),
            _DataTrapDict(
                {
                    "asset_id": "asset-trap",
                    "mime_type": "application/octet-stream",
                    "byte_length": 8192,
                    "data": "MUSTNOTBEREAD" * 32,
                    "type": "resource",
                }
            ),
        ):
            caplog.clear()
            with caplog.at_level(
                logging.DEBUG,
                logger="amplifier_module_provider_github_copilot.streaming",
            ):
                # Must NOT raise — accessing `data` would detonate the trap.
                result = translate_event(
                    {"type": "session.binary_asset", "data": payload}, config
                )

            assert result is None, "binary_asset must still DROP"
            tripwire = [
                r for r in caplog.records if "binary-asset tripwire" in r.getMessage()
            ]
            assert tripwire, "tripwire did not fire for the data-trap payload"
            msg = tripwire[0].getMessage()
            assert "asset-trap" in msg
            assert "application/octet-stream" in msg
            assert "8192" in msg

    def test_binary_asset_tripwire_sanitizes_hostile_metadata(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The DEBUG tripwire MUST neutralise hostile metadata before logging.

        ``_sanitize_meta_scalar`` (streaming.py) is the only log-injection guard
        on the binary_asset path. The other tripwire tests feed CLEAN metadata,
        so the sanitiser branches are unexercised; this one drives them. It
        proves three properties at once:

        - control characters (NUL/BEL/ESC/CR/LF) are stripped from every emitted
          scalar, so a schema-violating ``asset_id`` cannot inject log lines;
        - a non-string field whose ``__str__`` smuggles an ANSI escape is coerced
          via ``str()`` and THEN stripped (the escape never survives) — a value
          the pre-hardening sanitiser would have passed through verbatim;
        - the base64 ``data`` bytes still never reach any log.

        Contract: event-vocabulary:Drop:MUST:3.
        """
        config = load_event_config()
        secret_bytes = "QUJD" + "U0VDUkVUQllURVM" * 64  # must never appear

        class _HostileStr:
            """A non-string mime_type whose __str__ tries to inject a log line."""

            def __str__(self) -> str:
                return "image/\x1b[31mevil\r\nINJECTED"

        sdk_event = {
            "type": "session.binary_asset",
            "data": {
                "asset_id": "ok-\x00\x07\x1b[2Jclean\r\nid",  # control chars
                "mime_type": _HostileStr(),  # hostile non-str __str__
                "byte_length": 4096,
                "data": secret_bytes,
                "type": "image",
            },
        }
        with caplog.at_level(
            logging.DEBUG,
            logger="amplifier_module_provider_github_copilot.streaming",
        ):
            result = translate_event(sdk_event, config)

        assert result is None, "binary_asset must still DROP"
        tripwire = [r for r in caplog.records if "binary-asset tripwire" in r.getMessage()]
        assert tripwire, "tripwire did not fire"
        line = tripwire[0].getMessage()

        # No control character may survive sanitisation into the emitted line —
        # these are the log-injection vectors.
        for ctrl in ("\x00", "\x07", "\x1b", "\r", "\n"):
            assert ctrl not in line, f"control char {ctrl!r} survived sanitisation"

        # The printable residue of the sanitised scalars is still present...
        assert "cleanid" in line  # asset_id printable chars, CR/LF stripped
        assert "INJECTED" in line  # hostile __str__ was coerced, not executed
        # ...but the raw ANSI escape sequence is gone (only the ESC byte is a
        # control char; stripping it disarms the sequence).
        assert "\x1b[31m" not in line

        # SECURITY ASSERTION: the base64 `data` bytes must NEVER reach any log.
        assert secret_bytes not in caplog.text

    def test_binary_asset_tripwire_truncates_overlong_scalar(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Metadata scalars longer than the cap MUST be truncated, not logged whole.

        A schema-violating emitter could supply a megabyte ``asset_id`` to bloat
        the log; the sanitiser caps it at ``_META_SCALAR_MAX_LEN`` with an
        explicit ``...(truncated)`` marker. Contract: event-vocabulary:Drop:MUST:3.
        """
        config = load_event_config()
        long_id = "Z" * 400  # exceeds _META_SCALAR_MAX_LEN (256)
        sdk_event = {
            "type": "session.binary_asset",
            "data": {
                "asset_id": long_id,
                "mime_type": "image/png",
                "byte_length": 1,
                "data": "QUJD",
                "type": "image",
            },
        }
        with caplog.at_level(
            logging.DEBUG,
            logger="amplifier_module_provider_github_copilot.streaming",
        ):
            translate_event(sdk_event, config)

        line = next(
            r.getMessage()
            for r in caplog.records
            if "binary-asset tripwire" in r.getMessage()
        )
        assert "...(truncated)" in line, "overlong asset_id was not truncated"
        assert long_id not in line, "full overlong asset_id leaked into the log"

