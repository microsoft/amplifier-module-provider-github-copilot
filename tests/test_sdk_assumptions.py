"""Tier 6: SDK Assumption Tests - verify SDK types and shapes without API calls.

These tests import the real SDK, instantiate objects, and verify our structural
assumptions. They require the SDK to be installed but do NOT make API calls.

Contract references:
- contracts/sdk-boundary.md
- contracts/deny-destroy.md

Run: pytest -m sdk_assumption -v
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any, cast

import pytest

from tests._sdk_version_gate import require_sdk


@pytest.mark.sdk_assumption
class TestSDKImportAssumptions:
    """Verify SDK module structure matches our assumptions.

    AC-1: SDK Import Assumptions
    """

    def test_copilot_client_class_exists(self, sdk_module: Any) -> None:
        """We assume copilot.CopilotClient exists and is importable.

        # Contract: sdk-boundary:Lifecycle:MUST:1
        """
        assert isinstance(sdk_module.CopilotClient, type)

    def test_client_has_create_session(self, sdk_module: Any) -> None:
        """We assume CopilotClient has create_session method.

        # Contract: sdk-boundary:Session:MUST:1
        """
        assert inspect.iscoroutinefunction(sdk_module.CopilotClient.create_session)

    def test_create_session_accepts_provider_kwargs(self, sdk_module: Any) -> None:
        """Pin the kwargs the provider forwards to ``CopilotClient.create_session``.

        The provider's ``CopilotClientWrapper.session()`` builds a ``session_config``
        dict (sdk_adapter/client.py) and unpacks it via
        ``await client.create_session(**session_config)``. The receiving
        signature is broad (the SDK accepts ~30 keyword args plus ``**kwargs``),
        and the project-local stub at ``typings/copilot/__init__.pyi`` declares
        a permissive ``**kwargs: Any`` for the long tail. That permissiveness
        means a typo on either side (provider or stub) is type-invisible — only
        the live SDK will reject an unknown keyword at runtime.

        This test pins the live SDK's parameter set against the kwargs the
        provider actually forwards, so an upstream rename (e.g.
        ``infinite_sessions`` → ``continuous_sessions``) fails LOUDLY here
        instead of escaping to live-smoke or production.

        # Contract: sdk-boundary:Session:MUST:1
        # Contract: deny-destroy:DenyHook:MUST:1 (hooks kwarg)
        # Contract: sdk-boundary:MinimalMode:MUST:1-6 (minimal-mode kwargs)
        """
        # Authoritative: every keyword the provider forwards must remain a real
        # parameter of CopilotClient.create_session in the installed SDK.
        # Sources:
        #   - sdk_adapter/client.py session() body (model, tools, available_tools,
        #     streaming, on_permission_request, hooks, model_capabilities,
        #     reasoning_effort)
        #   - sdk_adapter/client.py _minimal_mode_session_config() (infinite_sessions,
        #     enable_config_discovery, mcp_servers, skill_directories, custom_agents,
        #     commands)
        provider_forwarded_kwargs: frozenset[str] = frozenset(
            {
                "model",
                "tools",
                "available_tools",
                "streaming",
                "on_permission_request",
                "hooks",
                "model_capabilities",
                "reasoning_effort",
                "infinite_sessions",
                "enable_config_discovery",
                "mcp_servers",
                "skill_directories",
                "custom_agents",
                "commands",
                # b10 MinimalMode:MUST:7-15 — 9 defense-in-depth pins
                # forwarded by _minimal_mode_session_config(). See
                # contracts/sdk-boundary.md v1.10 History row and
                # config/_sdk_protection.py for the wire shape.
                "enable_session_store",
                "enable_skills",
                "enable_file_hooks",
                "enable_host_git_operations",
                "enable_on_demand_instruction_discovery",
                "skip_embedding_retrieval",
                "embedding_cache_storage",
                "enable_session_telemetry",
                "mcp_oauth_token_storage",
                # v1.0.2 MinimalMode:MUST:16 — the mode-gated `memory` pin,
                # forwarded by _minimal_mode_session_config() alongside the
                # MUST:7-15 pins above. create_session declares it as
                # `memory: MemoryConfiguration | None` (v1.0.2 client.py:1612);
                # pinning it here makes a future SDK rename/removal FAIL at this
                # signature guard instead of silently escaping to live/prod.
                "memory",
            }
        )

        sig = inspect.signature(sdk_module.CopilotClient.create_session)
        sdk_param_names: set[str] = set(sig.parameters.keys())
        accepts_var_keyword: bool = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )

        missing = sorted(provider_forwarded_kwargs - sdk_param_names)
        # If the SDK exposes **kwargs, an unknown name is technically still
        # accepted at the signature level, but that defeats the purpose of the
        # pin: we want a rename to FAIL here, not be silently swallowed. So
        # require every forwarded keyword to be a NAMED parameter on the SDK.
        assert not missing, (
            "github-copilot-sdk drift: CopilotClient.create_session no longer "
            "declares the following keyword(s) the provider forwards: "
            f"{missing}. Either the SDK renamed/removed them (update "
            "sdk_adapter/client.py and typings/copilot/__init__.pyi to match) "
            "or the provider added a keyword that was never on the SDK "
            "(remove it from session_config). The stub's `**kwargs: Any` "
            f"escape hatch hides this at type-check time. "
            f"(SDK accepts **kwargs: {accepts_var_keyword})"
        )

    def test_client_has_start_stop(self, sdk_module: Any) -> None:
        """We assume CopilotClient has start() and stop() lifecycle methods.

        # Contract: sdk-boundary:Lifecycle:MUST:1
        """
        assert inspect.iscoroutinefunction(sdk_module.CopilotClient.start)
        assert inspect.iscoroutinefunction(sdk_module.CopilotClient.stop)

    def test_copilot_client_constructor_accepts_pinned_sdk_kwargs(self, sdk_module: Any) -> None:
        """CopilotClient must expose the b10 keyword constructor surface.

        # Contract: sdk-boundary:Auth:MUST:1

        Verifies the provider's direct-kwarg wiring remains aligned.
        """
        from pathlib import Path

        from copilot import CopilotClient  # type: ignore[import-untyped]

        assert isinstance(CopilotClient, type), (
            "CopilotClient must be a class importable from copilot"
        )

        copilot_ctor: Any = CopilotClient
        instance = copilot_ctor(
            base_directory=str(Path.cwd() / "logs" / ".pytest-sdk-assumptions-home"),
            github_token="test-token",
            log_level="info",
            env={},
            mode="copilot-cli",
        )
        assert isinstance(instance, CopilotClient), (
            "Pinned-sdk kwargs must construct a CopilotClient instance, "
            "not a proxy or exception object (SDKSurface:MUST:8)"
        )
        assert inspect.iscoroutinefunction(instance.start), (
            "Constructed CopilotClient must expose async start() — "
            "proves the pinned kwarg surface returned a live client"
        )

    def test_copilot_client_constructor_is_keyword_only(self, sdk_module: Any) -> None:
        """CopilotClient.__init__ must keep keyword-only constructor semantics.

        If keyword-only enforcement is relaxed, positional config objects can
        silently bypass the expected b10 constructor surface.

        # Contract: sdk-boundary:SDKSurface:MUST:3
        """
        signature = inspect.signature(sdk_module.CopilotClient.__init__)
        positional = [
            p
            for p in signature.parameters.values()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            and p.name != "self"
        ]
        observed = [p.name for p in positional]
        assert positional == [], (
            f"CopilotClient.__init__ must be keyword-only; got positional params {observed}"
        )

    def test_permission_decision_reject_importable(self) -> None:
        """The b10 rejection variant is constructed without a kind argument.

        Contract: sdk-boundary:SDKSurface:MUST:1b
        """
        require_sdk()
        rpc_mod = cast(Any, importlib.import_module("copilot.generated.rpc"))

        rejection = rpc_mod.PermissionDecisionReject()

        assert rejection.kind == "reject"

    def test_constructor_is_keyword_only(self) -> None:
        """The b10 client constructor rejects positional config objects.

        Contract: sdk-boundary:SDKSurface:MUST:8
        """
        copilot = cast(
            Any,
            require_sdk(),
        )

        with pytest.raises(TypeError):
            copilot.CopilotClient("/positional/path")

    def test_constructor_accepts_documented_kwargs(self, tmp_path: Path) -> None:
        """The b10 client constructor accepts the documented process kwargs.

        Contract: sdk-boundary:SDKSurface:MUST:8
        """
        copilot = cast(
            Any,
            require_sdk(),
        )
        client_cls = cast(Any, copilot.CopilotClient)

        client = client_cls(
            base_directory=str(tmp_path),
            github_token="t",
            log_level="info",
            env={"X": "1"},
            mode="copilot-cli",
        )

        assert isinstance(client, client_cls)

    def test_mode_default_is_copilot_cli(self) -> None:
        """The b10 runtime defaults clients to copilot-cli mode after install.

        Contract: sdk-boundary:SDKSurface:MUST:8
        """
        copilot = cast(
            Any,
            require_sdk(),
        )

        signature = inspect.signature(copilot.CopilotClient.__init__)

        assert signature.parameters["mode"].default == "copilot-cli"

    def test_model_info_capabilities_is_required(self, sdk_module: Any) -> None:
        """ModelInfo.capabilities must be a required field with no default.

        The SDK declares ``capabilities: ModelCapabilities`` (no default).
        If a future SDK adds a default of ``None``, code that reads
        ``model.capabilities.limits`` would start NoneType-erroring on
        defaulted instances.

        # Contract: sdk-boundary:SDKSurface:MUST:3
        """
        import dataclasses

        from copilot.client import ModelInfo  # type: ignore[import-untyped]

        fields_by_name = {f.name: f for f in dataclasses.fields(ModelInfo)}  # pyright: ignore[reportArgumentType]
        capabilities = fields_by_name["capabilities"]
        assert capabilities.default is dataclasses.MISSING
        assert capabilities.default_factory is dataclasses.MISSING


@pytest.mark.sdk_assumption
class TestModelBillingServerShapeTolerance:
    """Pin that ``ModelBilling.from_dict`` tolerates the live GitHub server
    shape — a ``billing`` payload that has ``restricted_to`` and
    ``token_prices`` but NO ``multiplier``.

    GitHub no longer emits ``multiplier`` in the upstream ``billing``
    payload. SDK v0.3.0 hard-required it and raised
    ``ValueError("Missing required field 'multiplier' in ModelBilling")``,
    which aborted the entire ``list_models()`` batch and broke
    ``amplifier init`` / ``amplifier provider models`` for fresh
    installs. SDK v1.0.0b4 made the field nullable and the parser
    tolerant.

    This test pins the tolerance at the SDK boundary so any future SDK
    that re-tightens the parser fails loudly here BEFORE shipping to
    users.

    Contract: sdk-boundary:SDKSurface:MUST:3
    """

    def test_model_billing_tolerates_server_shape_without_multiplier(self, sdk_module: Any) -> None:
        """The exact GitHub server-shape payload MUST parse without raising.

        Reproduces the live GitHub server-shape payload that aborted SDK
        v0.3.0's parser: a ``billing`` block with ``restricted_to`` and
        ``token_prices`` but no ``multiplier``. v1.0.0b4+ returns a
        ``ModelBilling`` with ``multiplier`` absent or ``None``.
        """
        from copilot.client import ModelBilling  # type: ignore[import-untyped]

        # Exact shape captured from live `models.list` JSON-RPC response.
        live_server_shape = {
            "restricted_to": [
                "pro",
                "pro_plus",
                "individual_trial",
                "business",
                "enterprise",
                "max",
            ],
            "token_prices": {
                "batch_size": 1_000_000,
                "cache_price": 30_000_000_000,
                "input_price": 300_000_000_000,
                "output_price": 1_500_000_000_000,
            },
        }

        # MUST NOT raise — the whole point of the v1.0.0b4 fix.
        billing = ModelBilling.from_dict(live_server_shape)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportAttributeAccessIssue]

        # Multiplier MUST be absent / None (not invented as 0.0 or 1.0).
        multiplier = getattr(billing, "multiplier", None)  # pyright: ignore[reportUnknownArgumentType]
        assert multiplier is None, (
            f"ModelBilling.multiplier={multiplier!r} expected None "
            "when the server payload omits 'multiplier'. SDK must not "
            "invent a default — that would silently mis-bill consumers."
        )

    def test_model_info_from_dict_survives_billing_without_multiplier(
        self, sdk_module: Any
    ) -> None:
        """End-to-end: a full ``ModelInfo`` payload with the new billing
        shape must parse — proving ``list_models()`` will not abort on
        the first billing-having model in the batch.
        """
        from copilot.client import ModelInfo  # type: ignore[import-untyped]

        live_model_payload = {
            "id": "claude-sonnet-4.6",
            "name": "Claude Sonnet 4.6",
            "capabilities": {
                "family": "claude",
                "type": "chat",
                "tokenizer": "claude",
                "limits": {
                    "max_context_window_tokens": 200_000,
                    "max_output_tokens": 16_384,
                    "max_prompt_tokens": 200_000,
                },
                "supports": {
                    "streaming": True,
                    "tool_calls": True,
                    "parallel_tool_calls": True,
                },
            },
            "billing": {
                "restricted_to": ["pro", "business", "enterprise"],
                "token_prices": {
                    "batch_size": 1_000_000,
                    "cache_price": 30_000_000_000,
                    "input_price": 300_000_000_000,
                    "output_price": 1_500_000_000_000,
                },
            },
        }

        # MUST NOT raise — this is the exact path list_models() walks.
        model = ModelInfo.from_dict(live_model_payload)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportAttributeAccessIssue]
        model_id: str = getattr(model, "id", "")  # pyright: ignore[reportUnknownArgumentType]
        billing = getattr(model, "billing", None)  # pyright: ignore[reportUnknownArgumentType]
        from copilot.client import ModelBilling  # type: ignore[import-untyped]

        assert model_id == "claude-sonnet-4.6"
        # isinstance check IS the existence check (None is not a ModelBilling),
        # while also pinning the exact SDK type returned by from_dict — fails
        # loud if a future SDK swaps the return type.
        assert isinstance(billing, ModelBilling), (
            f"Expected ModelBilling instance; got {type(billing).__name__}: {billing!r}"
        )
        assert getattr(billing, "multiplier", None) is None  # pyright: ignore[reportUnknownArgumentType]


@pytest.mark.sdk_assumption
class TestPermissionRequestResultV030Schema:
    """SDK v0.3.0 introduced a breaking change to PermissionRequestResult.

    Previously (v0.2.x) the result carried multiple fields (kind, rules, feedback,
    message, path) and used kind values 'denied-by-rules' / 'approved'. The v0.3.0
    release reduced the result to kind-only, removed rules/feedback/message/path,
    and renamed the kind values to 'reject' / 'approve-once'.

    These tests pin the v0.3.0 schema we ship against, so any future SDK version
    that mutates this surface (additional fields, renamed kinds, removed kinds)
    fails LOUDLY before reaching production.

    Contract: sdk-boundary:SDKSurface:MUST:1
    """

    def test_kind_literal_contains_exactly_v030_values(self, sdk_module: Any) -> None:
        """PermissionDecisionReject.kind ClassVar MUST equal 'reject' (deny-destroy pin).

        SDK v1.0.0b10 (since b9) removes PermissionRequestResultKind; the discriminator is now
        a ClassVar on each concrete Union member of PermissionRequestResult.
        Pins the kind value the provider's deny-destroy path depends on.

        Contract: sdk-boundary:SDKSurface:MUST:1
        """
        from typing import get_args

        from copilot.generated.rpc import PermissionDecisionReject  # type: ignore[import-untyped]
        from copilot.session import PermissionRequestResult  # type: ignore[import-untyped]

        members = get_args(PermissionRequestResult)
        member_names = {cls.__name__ for cls in members}
        assert "PermissionDecisionReject" in member_names, (
            f"PermissionDecisionReject missing from PermissionRequestResult Union. "
            f"Got: {sorted(member_names)}"
        )
        assert PermissionDecisionReject.kind == "reject", (
            f"PermissionDecisionReject.kind is {PermissionDecisionReject.kind!r}, "
            "expected 'reject'. The deny-destroy flow depends on this discriminator."
        )

    def test_factory_produces_sdk_accepted_reject_result(self, sdk_module: Any) -> None:
        """make_permission_denied() MUST construct a real SDK object with kind='reject'.

        End-to-end check: the factory's output is assignable to a real SDK
        PermissionRequestResult and carries the 'reject' kind. Also verifies
        feedback is None (silent-reject UX — no user-visible message on the wire).

        SKIP_SDK_CHECK is cleared and _imports is reloaded so the factory exercises
        the real SDK code path (conftest.py sets SKIP_SDK_CHECK=1 by default to keep
        the rest of the suite SDK-binary-free).

        Contract: sdk-boundary:SDKSurface:MUST:1
        Contract: deny-destroy:PermissionRequest:MUST:2
        """
        import dataclasses
        import importlib
        import os

        from copilot.session import PermissionRequestResult  # type: ignore[import-untyped]

        from amplifier_module_provider_github_copilot.sdk_adapter import _imports

        # Force the real SDK code path even though conftest set SKIP_SDK_CHECK=1
        original = os.environ.pop("SKIP_SDK_CHECK", None)
        try:
            importlib.reload(_imports)
            result = _imports.make_permission_denied()
        finally:
            if original is not None:
                os.environ["SKIP_SDK_CHECK"] = original
            importlib.reload(_imports)

        # Real SDK type, not the dict fallback (which is test-mode only)
        assert isinstance(result, PermissionRequestResult), (
            f"Factory returned {type(result).__name__}, expected PermissionRequestResult. "
            "Dict fallback should only occur in test mode (SKIP_SDK_CHECK=1)."
        )
        # Exact b10 kind for deny-destroy path
        assert result.kind == "reject", (
            f"Factory returned kind={result.kind!r}, expected 'reject'. "
            "v0.2.x used 'denied-by-rules' — never reintroduce."
        )
        # b10 silent-reject: feedback must be None (no user-visible message on the wire)
        assert result.feedback is None, (  # pyright: ignore[reportAttributeAccessIssue]
            f"Factory returned feedback={result.feedback!r}, expected None. "  # pyright: ignore[reportAttributeAccessIssue]
            "Silent-reject UX requires no feedback payload."
        )
        # b10: PermissionDecisionReject is a @dataclass; 'kind' is ClassVar (not a field),
        # 'feedback' is the only instance field.
        from copilot.generated.rpc import (  # type: ignore[import-untyped]
            PermissionDecisionReject as _RejCls,
        )

        actual_fields = tuple(f.name for f in dataclasses.fields(_RejCls))
        assert actual_fields == ("feedback",), (
            f"PermissionDecisionReject dataclass fields={actual_fields!r}, "
            "expected exactly ('feedback',). "
            "'kind' is a ClassVar in b10 — SDK regressed if this changes."
        )

    def test_permission_decision_reject_to_dict_wire_shape(self, sdk_module: Any) -> None:
        """PermissionDecisionReject().to_dict() MUST emit {"kind": "reject"} — no feedback key.

        The dict-fallback in _imports.make_permission_denied (when SDK is absent)
        constructs exactly this payload; without this assertion an SDK minor-version
        bump that changes to_dict() to include ``"feedback": None`` (or omits
        ``kind``) would silently desynchronise the dict-fallback from the real
        SDK wire shape, regressing deny-destroy without a failing test.

        Contract: sdk-boundary:SDKSurface:MUST:1
        Contract: deny-destroy:PermissionRequest:MUST:2
        """
        from copilot.generated.rpc import (  # type: ignore[import-untyped]
            PermissionDecisionReject,
        )

        result = PermissionDecisionReject().to_dict()

        assert result == {"kind": "reject"}, (
            f"PermissionDecisionReject().to_dict() returned {result!r}, "
            "expected exactly {'kind': 'reject'}. Silent-reject wire shape "
            "MUST omit feedback when None — the dict-fallback in "
            "_imports.make_permission_denied depends on this."
        )

    def test_permission_request_result_is_alias_not_constructor(self) -> None:
        """The b10 permission-request result alias cannot be constructed directly.

        Contract: sdk-boundary:SDKSurface:MUST:1a
        """
        require_sdk()
        session_mod = cast(Any, importlib.import_module("copilot.session"))

        with pytest.raises(TypeError):
            session_mod.PermissionRequestResult()


@pytest.mark.sdk_assumption
class TestReasoningEffortLiteralPin:
    """Pin ``CopilotClient.create_session(reasoning_effort=...)`` against the
    live SDK ``ReasoningEffort`` Literal so SDK additions/renames fail loudly
    here instead of silently weakening the provider's type contract.

    Contract: sdk-boundary:Session:MUST:1

    Mutation check: change the SDK ``create_session(reasoning_effort=...)``
    annotation from ``ReasoningEffort | None`` to ``str | None`` and this
    test goes red because the union member is no longer the Literal.
    """

    def test_create_session_reasoning_effort_annotation_is_literal(self, sdk_module: Any) -> None:
        """``CopilotClient.create_session(reasoning_effort=...)`` MUST be typed
        as ``ReasoningEffort | None`` so the provider's pre-validated value
        is type-compatible at the SDK boundary. If the SDK switches the
        annotation to ``str | None`` (loosening) or removes it entirely, our
        Layer-1 vs Layer-2 split needs to be re-evaluated.
        """
        import inspect
        from types import UnionType
        from typing import Union, get_args, get_origin

        sig = inspect.signature(sdk_module.CopilotClient.create_session)
        # Direct subscript — KeyError fails loud if the parameter is renamed
        # or removed, with a more useful message than an `is not None` check.
        try:
            param = sig.parameters["reasoning_effort"]
        except KeyError as exc:
            pytest.fail(
                "create_session lost the reasoning_effort named parameter. "
                f"Provider forwarding chain is broken. Available params: "
                f"{sorted(sig.parameters.keys())!r} ({exc})"
            )

        ann = param.annotation
        # Resolve string-form annotations from `from __future__ import annotations`
        if isinstance(ann, str):
            from copilot.client import ReasoningEffort as _RE  # noqa: F401

            ann = eval(ann, {"ReasoningEffort": _RE, "None": None})

        origin = get_origin(ann)
        assert origin in (Union, UnionType), (
            f"create_session.reasoning_effort annotation is {ann!r}; "
            f"expected ReasoningEffort | None."
        )
        members = set(get_args(ann)) - {type(None)}
        assert len(members) == 1, (
            f"create_session.reasoning_effort union has unexpected members: "
            f"{members!r}; expected exactly ReasoningEffort | None."
        )
        (literal_type,) = members
        from copilot.client import ReasoningEffort  # type: ignore[import-untyped]

        assert literal_type is ReasoningEffort, (
            f"create_session.reasoning_effort is {literal_type!r}; "
            f"expected copilot.client.ReasoningEffort. Provider Layer-1 "
            f"allowlist may need to follow the SDK's chosen Literal."
        )

    def test_fallback_allowlist_is_superset_of_sdk_reasoning_effort_literal(
        self, sdk_module: Any
    ) -> None:
        """``_REASONING_EFFORT_FALLBACK_ALLOWLIST`` MUST be a strict superset of
        ``frozenset(get_args(ReasoningEffort))``: every SDK Literal member must
        be present, and the only permitted extras are in ``_KNOWN_EXTRAS``.

        The provider allowlist extends the v1.0.2 SDK Literal with ``"max"``
        (advertised by the live list_models endpoint; absent from the v1.0.2
        SDK Literal). Only values documented in
        ``_KNOWN_EXTRAS`` are allowed to differ from the SDK Literal — any
        unreviewed addition is a bug.

        Contract: provider-protocol:complete:MUST:11

        Mutation check:
        - SDK adds ``"ultra"`` but allowlist unchanged → red (missing SDK member).
        - Constant drops ``"xhigh"`` → red (missing SDK member).
        - Constant adds ``"banana"`` without adding to _KNOWN_EXTRAS → red.
        - SDK adds ``"ultra"``; both allowlist and _KNOWN_EXTRAS updated → green.
        """
        from typing import get_args

        from copilot.client import ReasoningEffort  # type: ignore[import-untyped]

        from amplifier_module_provider_github_copilot.request_adapter import (
            _REASONING_EFFORT_FALLBACK_ALLOWLIST,
        )

        # Values the provider forwards beyond the v1.0.2 SDK Literal.
        _KNOWN_EXTRAS: frozenset[str] = frozenset({"none", "max"})

        sdk_members = frozenset(get_args(ReasoningEffort))

        # Every SDK member must be in the provider allowlist.
        missing = sdk_members - _REASONING_EFFORT_FALLBACK_ALLOWLIST
        assert not missing, (
            f"_REASONING_EFFORT_FALLBACK_ALLOWLIST is missing SDK members: "
            f"{sorted(missing)}. Add them to the frozenset in request_adapter.py "
            f"and update contracts/provider-protocol.md MUST:11."
        )

        # No unreviewed extras beyond what is explicitly documented.
        unreviewed = _REASONING_EFFORT_FALLBACK_ALLOWLIST - sdk_members - _KNOWN_EXTRAS
        assert not unreviewed, (
            f"_REASONING_EFFORT_FALLBACK_ALLOWLIST has unreviewed extras vs SDK: "
            f"{sorted(unreviewed)}. Either remove them or add to _KNOWN_EXTRAS "
            f"with a justifying comment."
        )

    def test_reasoning_effort_levels_are_token_safe(self, sdk_module: Any) -> None:
        """Every REASONING_EFFORT_LEVELS member MUST be a bare lowercase token
        (``^[a-z]+$``). models.py joins these with '/' to build the
        ``reasoning=a/b/c`` capability string; a value containing '/' or
        whitespace would make that token ambiguous to parse.

        Contract: sdk-boundary:ModelDiscovery:MUST:3
        """
        import re

        from amplifier_module_provider_github_copilot.request_adapter import (
            REASONING_EFFORT_LEVELS,
        )

        bad = [v for v in REASONING_EFFORT_LEVELS if not re.fullmatch(r"[a-z]+", v)]
        assert not bad, (
            f"REASONING_EFFORT_LEVELS members must match ^[a-z]+$ so the "
            f"'/'-joined reasoning= token is unambiguous; offenders: {bad!r}"
        )

    def test_reasoning_effort_type_is_literal_4_values(self, sdk_module: Any) -> None:
        """SDK ``ReasoningEffort`` MUST be a Literal with exactly 4 members.

        Pins the v1.0.2 SDK surface {low, medium, high, xhigh}. The test goes
        red if the installed SDK's ``ReasoningEffort`` Literal differs from that
        set, signalling that the provider allowlist and Layer-1 gate need
        re-evaluation against the changed SDK surface.

        Contract: sdk-boundary:SDKSurface:MUST:3 (pin SDK Literal surface)
        """
        from typing import Literal, get_args, get_origin

        from copilot.client import ReasoningEffort  # type: ignore[import-untyped]

        expected = frozenset({"low", "medium", "high", "xhigh"})
        actual = frozenset(get_args(ReasoningEffort))
        assert get_origin(ReasoningEffort) is Literal, (
            f"ReasoningEffort is no longer a Literal; got {ReasoningEffort!r}. "
            f"Provider allowlist and Layer-1 gate need re-evaluation."
        )
        assert actual == expected, (
            f"SDK ReasoningEffort Literal changed from expected {sorted(expected)} "
            f"to {sorted(actual)}.\n"
            f"  Added in SDK:   {sorted(actual - expected)}\n"
            f"  Removed in SDK: {sorted(expected - actual)}\n"
            f"Follow the migration steps in this test's docstring."
        )

    def test_max_support_in_live_sdk_but_not_in_literal(self, sdk_module: Any) -> None:
        """Pins that ``"max"`` is advertised by the live GitHub Copilot endpoint
        and is absent from the v1.0.2 SDK ``ReasoningEffort`` Literal.

        The provider's ``_REASONING_EFFORT_FALLBACK_ALLOWLIST`` includes ``"max"``
        so cache-miss paths forward it to the SDK's Layer-2 backstop instead of
        raising a ``ConfigurationError``.

        Contract: provider-protocol:complete:MUST:11 (provider fallback allowlist)
        """
        from typing import get_args

        from copilot.client import ReasoningEffort  # type: ignore[import-untyped]

        from amplifier_module_provider_github_copilot.request_adapter import (
            _REASONING_EFFORT_FALLBACK_ALLOWLIST,
        )

        sdk_members = frozenset(get_args(ReasoningEffort))
        assert "max" not in sdk_members, (
            "SDK ReasoningEffort now includes 'max'. Delete this test, remove "
            "'max' from _KNOWN_EXTRAS in test_fallback_allowlist_is_superset*, "
            "and sync contracts/provider-protocol.md MUST:11."
        )
        assert "max" in _REASONING_EFFORT_FALLBACK_ALLOWLIST, (
            "_REASONING_EFFORT_FALLBACK_ALLOWLIST must contain 'max'; the v1.0.2 "
            "fallback path forwards it to the SDK Layer-2 backstop."
        )

    def test_none_support_in_live_sdk_but_not_in_literal(self, sdk_module: Any) -> None:
        """Pins that ``"none"`` is advertised by the live GitHub Copilot endpoint
        and is absent from the v1.0.2 SDK ``ReasoningEffort`` Literal.

        The provider's ``_REASONING_EFFORT_FALLBACK_ALLOWLIST`` includes ``"none"``
        so cache-miss paths forward it to the SDK's Layer-2 backstop instead of
        raising a ``ConfigurationError``.

        Contract: provider-protocol:complete:MUST:11 (provider fallback allowlist)
        """
        from typing import get_args

        from copilot.client import ReasoningEffort  # type: ignore[import-untyped]

        from amplifier_module_provider_github_copilot.request_adapter import (
            _REASONING_EFFORT_FALLBACK_ALLOWLIST,
        )

        sdk_members = frozenset(get_args(ReasoningEffort))
        assert "none" not in sdk_members, (
            "SDK ReasoningEffort now includes 'none'. Delete this test, remove "
            "'none' from _KNOWN_EXTRAS in test_fallback_allowlist_is_superset*, "
            "and sync contracts/provider-protocol.md MUST:11."
        )
        assert "none" in _REASONING_EFFORT_FALLBACK_ALLOWLIST, (
            "_REASONING_EFFORT_FALLBACK_ALLOWLIST must contain 'none'; the v1.0.2 "
            "fallback path forwards it to the SDK Layer-2 backstop."
        )


@pytest.mark.sdk_assumption
class TestOverrideDataclassShapes:
    """Pin the SDK shapes of ``ModelSupportsOverride`` and
    ``ModelVisionLimitsOverride`` so the corresponding stubs in
    ``typings/copilot/__init__.pyi`` cannot silently drift.

    Contract: sdk-boundary:SDKSurface:MUST:3
    Contract: sdk-boundary:SDKSurface:MUST:4

    Mutation check: rename ``vision`` → ``visions`` on
    ``copilot.ModelSupportsOverride`` (SDK side) or in the stub and this
    test goes red because the field-name tuples no longer match.
    """

    def test_model_supports_override_dataclass_shape(self, sdk_module: Any) -> None:
        import dataclasses

        cls = sdk_module.ModelSupportsOverride
        assert dataclasses.is_dataclass(cls), (
            f"ModelSupportsOverride is no longer a dataclass: {type(cls).__name__}"
        )
        names = tuple(f.name for f in dataclasses.fields(cls))
        assert names == ("vision", "reasoning_effort"), (
            f"ModelSupportsOverride field names drifted: {names!r}; "
            f"expected ('vision', 'reasoning_effort'). Update the stub at "
            f"typings/copilot/__init__.pyi (ModelSupportsOverride) to match."
        )
        for f in dataclasses.fields(cls):
            assert f.default is None, (
                f"ModelSupportsOverride.{f.name} default is {f.default!r}; stub claims None."
            )

    def test_model_vision_limits_override_dataclass_shape(self, sdk_module: Any) -> None:
        import dataclasses

        cls = sdk_module.ModelVisionLimitsOverride
        assert dataclasses.is_dataclass(cls), (
            f"ModelVisionLimitsOverride is no longer a dataclass: {type(cls).__name__}"
        )
        names = tuple(f.name for f in dataclasses.fields(cls))
        assert names == (
            "supported_media_types",
            "max_prompt_images",
            "max_prompt_image_size",
        ), (
            f"ModelVisionLimitsOverride field names drifted: {names!r}; "
            f"expected (supported_media_types, max_prompt_images, "
            f"max_prompt_image_size). Update the stub at "
            f"typings/copilot/__init__.pyi (ModelVisionLimitsOverride)."
        )
        for f in dataclasses.fields(cls):
            assert f.default is None, (
                f"ModelVisionLimitsOverride.{f.name} default is {f.default!r}; stub claims None."
            )


@pytest.mark.sdk_assumption
class TestReasoningEffortReExportedAtSessionPath:
    """``copilot.session.ReasoningEffort`` MUST be the SAME object as
    ``copilot.client.ReasoningEffort``. ``typings/copilot/session.pyi``
    declares the alias under the assumption that the SDK mirrors the symbol
    at both paths. If the SDK ever drops the session-module re-export, the
    stub becomes a fictional surface.

    Contract: sdk-boundary:SDKSurface:MUST:2
    """

    def test_session_module_reasoning_effort_mirrors_client(self) -> None:
        from copilot.client import ReasoningEffort as _ClientRE  # type: ignore[import-untyped]
        from copilot.session import ReasoningEffort as _SessionRE  # type: ignore[import-untyped]

        assert _SessionRE is _ClientRE, (
            "copilot.session.ReasoningEffort and copilot.client.ReasoningEffort "
            f"are not the same object: {_SessionRE!r} vs {_ClientRE!r}. "
            "Either remove the ReasoningEffort declaration from "
            "typings/copilot/session.pyi or update both stubs to reflect the "
            "actual canonical home."
        )


@pytest.mark.sdk_assumption
class TestCopilotSessionWorkspacePathIsCachedProperty:
    """``CopilotSession.workspace_path`` MUST be a ``functools.cached_property``.
    The stub declares it as ``@property`` (the standard stub equivalent
    for read-only computed attributes); if the SDK changes it to a plain
    instance attribute, callers that assign through the stub will silently
    bypass intended caching semantics.

    Contract: sdk-boundary:SDKSurface:MUST:5
    """

    def test_workspace_path_is_cached_property(self) -> None:
        import functools
        import inspect

        from copilot.session import CopilotSession  # type: ignore[import-untyped]

        attr = inspect.getattr_static(CopilotSession, "workspace_path", None)
        assert isinstance(attr, functools.cached_property), (
            f"CopilotSession.workspace_path is no longer functools.cached_property; "
            f"got {type(attr).__name__ if attr is not None else 'missing'}. "
            f"Update the stub at typings/copilot/__init__.pyi (CopilotSession.workspace_path)."
        )


@pytest.mark.sdk_assumption
class TestCopilotSessionSendSignaturePin:
    """``CopilotSession.send`` MUST accept exactly one positional argument
    (``prompt``) and five keyword-only arguments (``attachments``, ``mode``,
    ``agent_mode``, ``request_headers``, ``display_prompt``). The provider's
    ``CopilotClientWrapper`` calls ``session.send(prompt, attachments=...)``;
    the in-tree mock at ``tests/fixtures/sdk_mocks.py`` mirrors the full
    surface so unit tests don't drift from production behaviour. If the SDK
    adds, removes, or renames a kwarg, the mock will silently absorb it
    (because the provider's call site uses named kwargs) and tests would
    stay green against a divergent contract.

    This is an SDK-import-boundary pin, not a behavioural assertion: it
    inspects the live ``CopilotSession.send`` signature and asserts the
    exact parameter set.

    Contract: sdk-boundary:SDKSurface:MUST:6
    Contract: sdk-boundary:Send:MUST:1

    Mutation check: rename ``mode`` → ``execution_mode`` on the SDK side
    and this test goes red because the parameter set no longer matches.
    """

    def test_send_signature_matches_v030_surface(self) -> None:
        from copilot.session import CopilotSession  # type: ignore[import-untyped]

        sig = inspect.signature(CopilotSession.send)
        # Drop ``self`` from the bound view.
        params = [p for name, p in sig.parameters.items() if name != "self"]

        positional_or_keyword = [
            p for p in params if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        ]
        keyword_only = [p for p in params if p.kind == inspect.Parameter.KEYWORD_ONLY]
        var_positional = [p for p in params if p.kind == inspect.Parameter.VAR_POSITIONAL]
        var_keyword = [p for p in params if p.kind == inspect.Parameter.VAR_KEYWORD]

        positional_names = tuple(p.name for p in positional_or_keyword)
        assert positional_names == ("prompt",), (
            f"CopilotSession.send positional parameters drifted: {positional_names!r}; "
            f"expected ('prompt',). The provider call sites in "
            f"sdk_adapter/client.py and the mock in tests/fixtures/sdk_mocks.py "
            f"both rely on this shape."
        )

        keyword_names = frozenset(p.name for p in keyword_only)
        expected = frozenset(
            {"agent_mode", "attachments", "display_prompt", "mode", "request_headers"}
        )
        assert keyword_names == expected, (
            f"CopilotSession.send keyword-only parameters drifted.\n"
            f"  Expected: {sorted(expected)}\n"
            f"  Observed: {sorted(keyword_names)}\n"
            f"  Added:    {sorted(keyword_names - expected)}\n"
            f"  Removed:  {sorted(expected - keyword_names)}\n"
            f"Update tests/fixtures/sdk_mocks.py mock send() and the provider "
            f"call sites in sdk_adapter/client.py together."
        )

        # No *args or **kwargs are permitted on this method — that escape
        # hatch would silently absorb future SDK kwargs and defeat the pin.
        assert not var_positional, (
            f"CopilotSession.send grew a *args parameter: {var_positional!r}. "
            f"This breaks the named-kwarg pin and would hide drift."
        )
        assert not var_keyword, (
            f"CopilotSession.send grew a **kwargs parameter: {var_keyword!r}. "
            f"This breaks the named-kwarg pin and would hide drift."
        )


@pytest.mark.sdk_assumption
class TestSDKToolWrapperCoversSDKToolSurface:
    """``SDKToolWrapper`` MUST structurally cover the real SDK tool object.

    The provider does NOT subclass the SDK's ``copilot.tools.Tool``; it
    duck-types a frozen ``SDKToolWrapper`` and hands instances to
    ``create_session(tools=...)``. The SDK then reads attributes off each
    object when building the tool wire-definitions (``copilot/client.py``)
    and when registering handlers (``copilot/session.py``). If the SDK's
    ``Tool`` dataclass grows a field that those code paths read, a wrapper
    that lacks it raises ``AttributeError`` at runtime on every
    tool-forwarding turn — exactly the v1.0.2 ``tool.defer`` regression, missed
    by the mocked unit and tool-less live smoke suites and surfaced only by an
    E2E that exercises the real tool-definition build path.

    This guard makes the REAL SDK ``Tool`` dataclass the source of truth: the
    field read-set is derived from the live SDK, not hand-written in the test
    body, so a FUTURE SDK tool-field addition fails RED here (before E2E),
    forcing a conscious decision to forward it (add the field to the wrapper)
    or to record a reasoned not-forwarded deferral.

    Contract: sdk-boundary:ToolForwarding:MUST:2
    """

    def test_wrapper_covers_sdk_tool_read_surface(self, sdk_module: Any) -> None:
        import dataclasses

        from amplifier_module_provider_github_copilot.sdk_adapter.types import (
            SDKToolWrapper,
        )

        sdk_tool = sdk_module.tools.Tool
        assert dataclasses.is_dataclass(sdk_tool), (
            "copilot.tools.Tool is no longer a dataclass; this guard introspects "
            "its fields to mirror the SDK's tool read surface. Re-derive the "
            "wrapper coverage check against the new Tool definition."
        )

        sdk_fields = {f.name for f in dataclasses.fields(sdk_tool)}
        wrapper_fields = {f.name for f in dataclasses.fields(SDKToolWrapper)}

        # Explicit, reasoned deferrals: SDK ``Tool`` fields the provider has
        # consciously chosen NOT to mirror on the wrapper. Empty today — the
        # wrapper covers the SDK's entire field surface. Each future entry, if
        # added, MUST cite why the provider's tool-forwarding path never reads
        # that attribute (mirrors the deferral allowlist pattern in
        # test_sdk_boundary_contract.test_no_unpinned_sdk_mode_gated_capability).
        deferred: dict[str, str] = {}

        # Stale-allowlist hygiene: a deferral for a field the SDK has since
        # removed must not rot silently.
        assert set(deferred) <= sdk_fields, (
            f"Stale tool-field deferral allowlist entries no longer on the SDK "
            f"copilot.tools.Tool: {sorted(set(deferred) - sdk_fields)}. Remove "
            f"them from `deferred`."
        )

        missing = sdk_fields - wrapper_fields - set(deferred)
        assert not missing, (
            f"SDKToolWrapper is missing SDK tool field(s): {sorted(missing)}.\n"
            f"  SDK copilot.tools.Tool fields: {sorted(sdk_fields)}\n"
            f"  SDKToolWrapper fields:         {sorted(wrapper_fields)}\n"
            f"The SDK reads these attributes off each tool object when building "
            f"tool definitions (copilot/client.py) and registering handlers "
            f"(copilot/session.py). A wrapper missing any of them raises "
            f"AttributeError on every tool-forwarding turn (this is exactly how "
            f"the v1.0.2 `defer` field broke the provider). Either add the field "
            f"to SDKToolWrapper in sdk_adapter/types.py (default = SDK-parity "
            f"value, i.e. the one that reproduces pre-upgrade wire behavior) and "
            f"update contracts/sdk-boundary.md ToolForwarding, OR add an explicit "
            f"reasoned entry to `deferred` explaining why the forwarding path "
            f"never reads it."
        )

        # Belt-and-suspenders: the SDK could expose a read attribute via a
        # ``property`` rather than a dataclass field, which ``fields()`` would
        # not see. Assert a constructed wrapper instance actually answers every
        # SDK field name via attribute access (the real failure mode is an
        # AttributeError on getattr, not a missing dataclass field).
        instance = SDKToolWrapper(name="probe", description="probe")
        for field_name in sdk_fields - set(deferred):
            assert hasattr(instance, field_name), (
                f"SDKToolWrapper instance has no attribute {field_name!r} that "
                f"the SDK reads off tool objects; getattr would raise "
                f"AttributeError on a tool-forwarding turn."
            )
