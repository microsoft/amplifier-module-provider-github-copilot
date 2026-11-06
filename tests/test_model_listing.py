"""Controlled catalogs, NOT claims of live account access or vendor availability."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from amplifier_core import ChatRequest, Message

from amplifier_module_provider_github_copilot._model_listing import ModelFilters
from amplifier_module_provider_github_copilot.error_translation import AuthenticationError
from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
from amplifier_module_provider_github_copilot.sdk_adapter import CopilotClientWrapper
from amplifier_module_provider_github_copilot.sdk_adapter import _imports as sdk_imports
from tests.test_models import make_sdk_model_info

# Distinct vendor tiers and routes must survive; versions within each may retire.
INVENTORY = [
    "auto",
    "gpt-5.5-astra",
    "gpt-6-astra",
    "gpt-6-luna",
    "gpt-5.9-sol",
    "gpt-5.10-sol",
    "gpt-6-terra",
    "gpt-6-mini",
    "gpt-6-codex",
    "gpt-6-codex-mini",
    "claude-3.5-sonnet",
    "claude-sonnet-5",
    "claude-sonnet-5.5",
    "claude-opus-5.5",
    "claude-haiku-5.5",
    "claude-fable-5.1",
    "claude-sonnet-5.5-20260901",
    "claude-sonnet-5.6-preview",
    "grok-4.6",
    "grok-4.7",
    "grok-4.7-fast",
    "mai-code-1-flash",
    "mai-code-2-flash",
    "mai-code-2-flash-picker",
    "gemini-3.5-flash",
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    "gemini-3-pro",
    "UnknownVendor-Canary",
    "private/GPt-6-Astra",
]
EXPECTED = sorted(
    set(INVENTORY)
    - {
        "gpt-5.5-astra",
        "gpt-5.9-sol",
        "claude-3.5-sonnet",
        "claude-sonnet-5",
        "claude-sonnet-5.5-20260901",
        "grok-4.6",
        "mai-code-1-flash",
        "gemini-3.5-flash",
    }
)


def test_latest_preserves_distinct_tiers_vendors_routes_unknowns() -> None:
    policy = ModelFilters(latest_per_family=True, hide_dated=True)
    assert policy.select(INVENTORY) == EXPECTED
    assert policy.select(list(reversed(INVENTORY))) == EXPECTED
    assert ModelFilters().select(INVENTORY) == sorted(INVENTORY)
    assert "auto" not in policy.select(["gpt-6-astra"])


@pytest.mark.parametrize(
    ("ids", "expected"),
    [
        (["gpt-6-astra", "gpt-6-astra-2026-11-01"], ["gpt-6-astra"]),
        (["gpt-6-astra-2026-10-01", "gpt-6-astra-2026-11-01"], ["gpt-6-astra-2026-11-01"]),
        (["claude-opus-5-5", "claude-opus-5-4"], ["claude-opus-5-5"]),
        (["claude-3-5-sonnet", "claude-sonnet-5.5"], ["claude-sonnet-5.5"]),
        (["gpt-5.9", "gpt-5.10"], ["gpt-5.10"]),
        (["o3-mini", "o4-mini", "o3"], ["o3", "o4-mini"]),
        (["gpt-6-astra", "gpt-6.1-astra-preview"], ["gpt-6-astra", "gpt-6.1-astra-preview"]),
        (["gpt-4o", "gpt-4.1"], ["gpt-4.1", "gpt-4o"]),
    ],
)
def test_conservative_version_alias_snapshot_rules(ids: list[str], expected: list[str]) -> None:
    assert ModelFilters(latest_per_family=True).select(ids) == expected


# Common conformance vectors: explicit mappings REPLACE, not merge.
@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        ({}, ["GPT-6-Astra", "gpt-5", "gpt-6-astra-2026-11-01", "other"]),
        ({"include": []}, []),
        ({"include": ["GpT-*"], "exclude": ["*-ASTRA*"]}, ["gpt-5"]),
        ({"include": ["gpt-?"], "exclude": ["GPT-6*"]}, ["gpt-5"]),
        ({"hide_dated": True}, ["GPT-6-Astra", "gpt-5", "other"]),
        ({"include": ["*-2026-*"], "hide_dated": True}, ["gpt-6-astra-2026-11-01"]),
    ],
)
def test_common_filter_vectors(filters: dict, expected: list[str]) -> None:
    ids = ["gpt-5", "GPT-6-Astra", "gpt-6-astra-2026-11-01", "other"]
    assert ModelFilters.parse(filters).select(ids) == expected


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        [],
        "gpt-*",
        {"include": "gpt-*"},
        {"include": [None]},
        {"exclude": [" "]},
        {"include": [1]},
        {"hide_dated": 1},
        {"latest_per_family": "true"},
        {"unknown": True},
    ],
)
def test_invalid_configuration_fails_early(invalid: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        GitHubCopilotProvider(config={"model_filters": invalid})


async def test_defaults_overrides_and_mutation(real_model_discovery: None) -> None:
    rpc = AsyncMock(
        return_value=SimpleNamespace(models=[make_sdk_model_info(model_id=id) for id in INVENTORY])
    )
    wrapper = CopilotClientWrapper(
        sdk_client=SimpleNamespace(rpc=SimpleNamespace(models=SimpleNamespace(list=rpc)))
    )
    config = {"model_filters": {"include": ["GPT-*"], "latest_per_family": True}}
    provider = GitHubCopilotProvider(config=config, client=wrapper)
    config["model_filters"]["include"].clear()
    metadata = provider.get_model_list_info()
    assert metadata["filters_keyword"] == "filters"
    assert metadata["override"] == "replace"
    assert metadata["unfiltered"] == {}
    assert metadata["source"] == "provider-api"
    metadata["default_filters"]["include"].clear()
    metadata["filters_schema"]["properties"].clear()
    assert provider.get_model_list_info()["default_filters"]["include"] == ["GPT-*"]
    assert (await provider.list_models()) == (await provider.list_models(filters=None))
    assert [m.id for m in await provider.list_models(filters={})] == sorted(INVENTORY)
    assert [m.id for m in await provider.list_models(filters={"include": ["gpt-5*"]})] == [
        "gpt-5.10-sol",
        "gpt-5.5-astra",
        "gpt-5.9-sol",
    ]
    assert await provider.list_models(filters={"include": []}) == []
    assert rpc.await_count == 5  # Even an empty selection is a fresh complete catalog.
    with pytest.raises(ValueError):
        await provider.list_models(filters={"unknown": True})
    assert rpc.await_count == 5
    assert provider._copilot_models_cache is None


async def test_filters_copied_before_rpc_suspends(real_model_discovery: None) -> None:
    filters = {"include": ["old-*"]}

    async def response(*args: object) -> SimpleNamespace:
        filters["include"].clear()
        return SimpleNamespace(models=[make_sdk_model_info(model_id="old-supported")])

    wrapper = CopilotClientWrapper(
        sdk_client=SimpleNamespace(
            rpc=SimpleNamespace(models=SimpleNamespace(list=AsyncMock(side_effect=response)))
        )
    )
    provider = GitHubCopilotProvider(client=wrapper)
    assert [m.id for m in await provider.list_models(filters=filters)] == ["old-supported"]


async def test_missing_limits_do_not_become_fabricated_catalog_values(
    real_model_discovery: None,
) -> None:
    sdk = SimpleNamespace(
        rpc=SimpleNamespace(
            models=SimpleNamespace(
                list=AsyncMock(
                    return_value=SimpleNamespace(
                        models=[
                            SimpleNamespace(id="auto", name="Auto", capabilities=None),
                        ]
                    )
                )
            )
        )
    )
    provider = GitHubCopilotProvider(client=CopilotClientWrapper(sdk_client=sdk))
    auto = (await provider.list_models())[0]
    assert auto.id == "auto"
    assert auto.context_window == 0
    assert auto.max_output_tokens == 0
    assert "vision" not in auto.capabilities
    assert "thinking" not in auto.capabilities
    assert auto.capabilities == []


async def test_catalog_does_not_infer_missing_output_from_prompt_budget(
    real_model_discovery: None,
) -> None:
    model = make_sdk_model_info(
        model_id="partial-limits",
        context_window=200000,
        max_prompt_tokens=168000,
        billing_context_max=168000,
        billing_long_context_max=936000,
    )
    model.capabilities.limits.max_output_tokens = None
    sdk = SimpleNamespace(
        rpc=SimpleNamespace(
            models=SimpleNamespace(list=AsyncMock(return_value=SimpleNamespace(models=[model])))
        )
    )
    provider = GitHubCopilotProvider(client=CopilotClientWrapper(sdk_client=sdk))
    info = (await provider.list_models())[0]
    assert info.context_window == 168000
    assert info.max_output_tokens == 0
    assert not any(c.startswith("long=") for c in info.capabilities)


async def test_real_sdk_serialization_and_deserialization_without_launch(
    real_model_discovery: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the JSON-RPC transport is controlled; the locked SDK code runs."""
    from copilot.rpc import ModelsListRequest, ServerRpc

    payload = {
        "models": [
            {
                "id": model_id,
                "name": model_id,
                "capabilities": {
                    "limits": {
                        "max_context_window_tokens": 200000,
                        "max_prompt_tokens": 200000,  # Subtraction would lose output.
                        "max_output_tokens": 64000,
                    },
                    "supports": {"vision": True, "reasoningEffort": True, "toolCalls": True},
                },
                "supportedReasoningEfforts": ["low", "high"],
                "defaultReasoningEffort": "low",
            }
            for model_id in INVENTORY
            if model_id != "auto"
        ]
        + [{"id": "auto", "name": "Auto"}]
    }
    request = AsyncMock(side_effect=lambda *a, **k: deepcopy(payload))
    sdk = SimpleNamespace(
        rpc=ServerRpc(SimpleNamespace(request=request)),
        stop=AsyncMock(),
        start=AsyncMock(),
        list_models=AsyncMock(side_effect=AssertionError("cached convenience path")),
    )
    monkeypatch.setattr(sdk_imports, "ModelsListRequest", ModelsListRequest)
    wrapper = CopilotClientWrapper(sdk_client=sdk)
    provider = GitHubCopilotProvider(client=wrapper)
    assert [m.id for m in await provider.list_models()] == EXPECTED
    all_models = await provider.list_models(filters={})
    assert [m.id for m in all_models] == sorted(INVENTORY)
    auto = next(m for m in all_models if m.id == "auto")
    assert (auto.context_window, auto.max_output_tokens) == (0, 0)
    assert auto.capabilities == []
    fixed = next(m for m in all_models if m.id == "gpt-6-astra")
    assert fixed.max_output_tokens == 64000
    assert fixed.defaults["supported_reasoning_efforts"] == ["low", "high"]
    assert "vision" in fixed.capabilities
    assert "tools" in fixed.capabilities
    fixed.defaults["supported_reasoning_efforts"].clear()
    refreshed = await provider.list_models()
    assert next(m for m in refreshed if m.id == fixed.id).defaults[
        "supported_reasoning_efforts"
    ] == ["low", "high"]
    # No filter, account, token or invented pagination params sent to the SDK.
    assert request.await_count == 3
    assert all(c.args == ("models.list", {}) and c.kwargs == {} for c in request.await_args_list)
    await wrapper.close()
    sdk.stop.assert_not_called()  # externally injected client is never owned
    sdk.start.assert_not_called()
    sdk.list_models.assert_not_called()


async def test_ready_rpc_cancellation_and_typed_sanitized_failure_reuse(
    real_model_discovery: None,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocked(*args: object) -> SimpleNamespace:
        started.set()
        await release.wait()
        return SimpleNamespace(models=[])

    rpc = AsyncMock(side_effect=blocked)
    wrapper = CopilotClientWrapper(
        sdk_client=SimpleNamespace(rpc=SimpleNamespace(models=SimpleNamespace(list=rpc)))
    )
    provider = GitHubCopilotProvider(client=wrapper)
    task = asyncio.create_task(provider.list_models())
    await asyncio.wait_for(started.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    secret = "ghp_" + "a" * 36
    rpc.side_effect = PermissionError("401 Unauthorized " + secret)
    with pytest.raises(AuthenticationError) as error:
        await provider.list_models()
    assert secret not in str(error.value)
    rpc.side_effect = None
    rpc.return_value = SimpleNamespace(models=[])
    assert await provider.list_models() == []
    assert rpc.await_count == 3


async def test_hidden_exact_model_inference_parameters_are_unchanged(
    real_model_discovery: None,
) -> None:
    from tests.fixtures.sdk_mocks import MockSDKSession, idle_event

    calls: list[dict[str, Any]] = []

    class Client:
        copilot_pid = None

        async def list_models(self, *, force_refresh: bool = False) -> list[Any]:
            assert force_refresh
            return [make_sdk_model_info(model_id=id) for id in ["gpt-5", "gpt-6"]]

        @asynccontextmanager
        async def session(self, **kwargs: Any) -> AsyncIterator[Any]:
            calls.append(kwargs)
            yield MockSDKSession(events=[idle_event()])

    provider = GitHubCopilotProvider(client=Client())  # type: ignore[arg-type]
    assert [m.id for m in await provider.list_models()] == ["gpt-6"]
    request = ChatRequest(
        model="gpt-5",
        messages=[Message(role="user", content="controlled fixture")],
        reasoning_effort="high",
        max_output_tokens=1234,
    )
    original = request.model_dump()
    await provider.complete(request)
    assert calls[0]["model"] == "gpt-5"
    assert calls[0]["reasoning_effort"] == "high"
    assert calls[0]["max_tokens"] == 1234
    assert request.model_dump() == original
