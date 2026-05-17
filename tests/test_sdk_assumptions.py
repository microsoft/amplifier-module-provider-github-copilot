"""Tier 6: SDK Assumption Tests - verify SDK types and shapes without API calls.

These tests import the real SDK, instantiate objects, and verify our structural
assumptions. They require the SDK to be installed but do NOT make API calls.

Contract references:
- contracts/sdk-boundary.md
- contracts/deny-destroy.md

Run: pytest -m sdk_assumption -v
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest


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

    def test_subprocess_config_importable(self, sdk_module: Any) -> None:
        """SubprocessConfig must be importable from copilot root (canonical path since v0.2.1).

        # Contract: sdk-boundary:Auth:MUST:1

        copilot.types was deleted in SDK v0.2.1. SubprocessConfig now lives at copilot root.
        Verifies _imports.py's direct import path is correct.
        """
        from copilot import SubprocessConfig  # type: ignore[import-untyped]

        assert isinstance(SubprocessConfig, type), (
            "SubprocessConfig must be a class importable from copilot"
        )
        # Instantiate and verify a known field
        instance = SubprocessConfig(github_token="test-token")
        assert instance.github_token == "test-token"

    def test_subprocess_config_cli_args_is_default_factory_list(
        self, sdk_module: Any
    ) -> None:
        """SubprocessConfig.cli_args must default to an empty list, never None.

        The SDK declares ``cli_args: list[str] = field(default_factory=list)``.
        If a future SDK relaxes this to ``list[str] | None = None``, the
        provider's stub and any caller that assumes a non-None list would
        diverge silently.

        # Contract: sdk-boundary:SDKSurface:MUST:3
        """
        import dataclasses

        from copilot import SubprocessConfig  # type: ignore[import-untyped]

        fields_by_name = {f.name: f for f in dataclasses.fields(SubprocessConfig)}
        cli_args = fields_by_name["cli_args"]
        assert cli_args.default is dataclasses.MISSING
        assert cli_args.default_factory is list
        assert SubprocessConfig(github_token="x").cli_args == []

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

        fields_by_name = {f.name: f for f in dataclasses.fields(ModelInfo)}
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

    def test_model_billing_tolerates_server_shape_without_multiplier(
        self, sdk_module: Any
    ) -> None:
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
                "pro", "pro_plus", "individual_trial",
                "business", "enterprise", "max",
            ],
            "token_prices": {
                "batch_size": 1_000_000,
                "cache_price": 30_000_000_000,
                "input_price": 300_000_000_000,
                "output_price": 1_500_000_000_000,
            },
        }

        # MUST NOT raise — the whole point of the v1.0.0b4 fix.
        billing = ModelBilling.from_dict(live_server_shape)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

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
        model = ModelInfo.from_dict(live_model_payload)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        model_id: str = getattr(model, "id", "")  # pyright: ignore[reportUnknownArgumentType]
        billing = getattr(model, "billing", None)  # pyright: ignore[reportUnknownArgumentType]
        assert model_id == "claude-sonnet-4.6"
        assert billing is not None
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
        """PermissionRequestResultKind Literal MUST equal exactly the v0.3.0 set.

        If the SDK adds, removes, or renames a kind value, our factory and
        deny-destroy flow may silently break. Pin the set explicitly.

        Contract: sdk-boundary:SDKSurface:MUST:1
        """
        from typing import get_args

        from copilot.session import PermissionRequestResultKind  # type: ignore[import-untyped]

        observed = set(get_args(PermissionRequestResultKind))
        expected = {"approve-once", "reject", "user-not-available", "no-result"}
        assert observed == expected, (
            f"SDK PermissionRequestResultKind drifted from v0.3.0 contract.\n"
            f"  Expected: {sorted(expected)}\n"
            f"  Observed: {sorted(observed)}\n"
            f"  Added: {sorted(observed - expected)}\n"
            f"  Removed: {sorted(expected - observed)}\n"
            f"Update _imports.make_permission_denied() and rerun the SDK diff workflow."
        )

    def test_factory_produces_sdk_accepted_reject_result(self, sdk_module: Any) -> None:
        """make_permission_denied() MUST construct a real SDK object with kind='reject'.

        End-to-end check: the factory's output is assignable to a real SDK
        PermissionRequestResult and carries the v0.3.0 'reject' kind. Also
        verifies the result has NO legacy v0.2.x fields (rules/feedback/message/path)
        since the SDK contract now forbids them.

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
        # Exact v0.3.0 kind
        assert result.kind == "reject", (
            f"Factory returned kind={result.kind!r}, expected 'reject'. "
            "v0.2.x used 'denied-by-rules' — never reintroduce."
        )
        # v0.3.0 dataclass fields are exactly ('kind',) — exact tuple, not blocklist
        actual_fields = tuple(f.name for f in dataclasses.fields(PermissionRequestResult))
        assert actual_fields == ("kind",), (
            f"PermissionRequestResult fields={actual_fields!r} expected exactly ('kind',). "
            "v0.3.0 reduced the surface to kind-only — SDK regressed."
        )


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

    def test_create_session_reasoning_effort_annotation_is_literal(
        self, sdk_module: Any
    ) -> None:
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
        param = sig.parameters.get("reasoning_effort")
        assert param is not None, (
            "create_session lost the reasoning_effort named parameter. "
            "Provider forwarding chain is broken."
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

    def test_fallback_allowlist_matches_sdk_reasoning_effort_literal(
        self, sdk_module: Any
    ) -> None:
        """``_REASONING_EFFORT_FALLBACK_ALLOWLIST`` MUST equal
        ``frozenset(get_args(ReasoningEffort))`` so a future SDK Literal
        addition (e.g. ``"ultra"``) is caught here at CI time rather than
        producing a wrong ``ConfigurationError`` in production on a
        cold-cache cache-miss path.

        Contract: provider-protocol:complete:MUST:11 (SDK literal allowlist
        ``{"low","medium","high","xhigh"}`` enumerated verbatim in
        ``contracts/provider-protocol.md``).

        Mutation check:
        - SDK adds ``"ultra"`` but constant unchanged → red (added in SDK).
        - Constant drops ``"xhigh"`` → red (stale in provider).
        - Constant adds ``"banana"`` → red (stale in provider).
        - Both updated to ``{"low","medium","high","xhigh","ultra"}`` → green.
        """
        from typing import get_args

        from copilot.client import ReasoningEffort  # type: ignore[import-untyped]

        from amplifier_module_provider_github_copilot.request_adapter import (
            _REASONING_EFFORT_FALLBACK_ALLOWLIST,
        )

        sdk_members = frozenset(get_args(ReasoningEffort))
        assert sdk_members == _REASONING_EFFORT_FALLBACK_ALLOWLIST, (
            f"Provider _REASONING_EFFORT_FALLBACK_ALLOWLIST drifted from SDK "
            f"ReasoningEffort Literal.\n"
            f"  SDK:               {sorted(sdk_members)}\n"
            f"  Provider:          {sorted(_REASONING_EFFORT_FALLBACK_ALLOWLIST)}\n"
            f"  Added in SDK:      {sorted(sdk_members - _REASONING_EFFORT_FALLBACK_ALLOWLIST)}\n"
            f"  Stale in provider: {sorted(_REASONING_EFFORT_FALLBACK_ALLOWLIST - sdk_members)}\n"
            f"Update _REASONING_EFFORT_FALLBACK_ALLOWLIST in "
            f"amplifier_module_provider_github_copilot/request_adapter.py "
            f"AND the contract enumeration at "
            f"contracts/provider-protocol.md MUST:11."
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
                f"ModelSupportsOverride.{f.name} default is {f.default!r}; "
                f"stub claims None."
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
                f"ModelVisionLimitsOverride.{f.name} default is {f.default!r}; "
                f"stub claims None."
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
    (``prompt``) and three keyword-only arguments (``attachments``, ``mode``,
    ``request_headers``). The provider's ``CopilotClientWrapper`` calls
    ``session.send(prompt, attachments=..., mode=..., request_headers=...)``;
    the in-tree mock at ``tests/fixtures/sdk_mocks.py`` mirrors that surface
    so unit tests don't drift from production behaviour. If the SDK adds,
    removes, or renames a kwarg, the mock will silently absorb it (because
    the provider's call site uses named kwargs) and tests would stay green
    against a divergent contract.

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
            p
            for p in params
            if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        ]
        keyword_only = [
            p for p in params if p.kind == inspect.Parameter.KEYWORD_ONLY
        ]
        var_positional = [
            p for p in params if p.kind == inspect.Parameter.VAR_POSITIONAL
        ]
        var_keyword = [p for p in params if p.kind == inspect.Parameter.VAR_KEYWORD]

        positional_names = tuple(p.name for p in positional_or_keyword)
        assert positional_names == ("prompt",), (
            f"CopilotSession.send positional parameters drifted: {positional_names!r}; "
            f"expected ('prompt',). The provider call sites in "
            f"sdk_adapter/client.py and the mock in tests/fixtures/sdk_mocks.py "
            f"both rely on this shape."
        )

        keyword_names = frozenset(p.name for p in keyword_only)
        expected = frozenset({"attachments", "mode", "request_headers"})
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

