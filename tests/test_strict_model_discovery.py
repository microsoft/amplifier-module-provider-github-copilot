"""Strict inventory probes exercise the real wrapper at its typed SDK RPC boundary.

Only the SDK RPC is stubbed: provider orchestration, wrapper cache/lock, model
translation and error translation run normally. No SDK launch, credentials or
network are required. Legacy fallback behavior is tested separately below.
"""

from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock, call

import pytest

from amplifier_module_provider_github_copilot import models as model_helpers
from amplifier_module_provider_github_copilot import provider as provider_module
from amplifier_module_provider_github_copilot.error_translation import (
    AuthenticationError,
    LLMError,
    LLMTimeoutError,
    NetworkError,
    ProviderUnavailableError,
)
from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
from amplifier_module_provider_github_copilot.sdk_adapter import CopilotClientWrapper
from amplifier_module_provider_github_copilot.sdk_adapter import _imports as sdk_imports
from tests.test_models import make_sdk_model_info


class ProcessExitedError(Exception):
    """SDK transport error stub, matching the existing translator's type name."""


@pytest.fixture
def inventory_boundary(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Prime every legacy cache while retaining the real wrapper RPC call path."""
    stale = make_sdk_model_info(model_id="other-account-stale-model")
    cached_domain = model_helpers.sdk_model_to_copilot_model(stale)
    read_cache = Mock(return_value=[cached_domain])
    write_cache = Mock()
    monkeypatch.setattr(provider_module, "read_cache", read_cache)
    monkeypatch.setattr(provider_module, "write_cache", write_cache)

    # Keep the API request opaque and pin that exact request at the RPC boundary.
    # The request factory is SDK-owned; there is no model-selection parameter here.
    request = object()
    request_factory = Mock(return_value=request)
    monkeypatch.setattr(sdk_imports, "make_models_list_request", request_factory)
    rpc_list = AsyncMock()
    sdk = SimpleNamespace(
        rpc=SimpleNamespace(models=SimpleNamespace(list=rpc_list)),
        # A plausible cached SDK convenience result must never be consulted.
        list_models=AsyncMock(return_value=[stale]),
        start=AsyncMock(side_effect=AssertionError("Must not launch an SDK process")),
    )
    wrapper = CopilotClientWrapper(sdk_client=sdk)
    wrapper._model_list_cache = [stale]
    config = {
        "model": "selected-not-advertised",
        "default_model": "default-not-advertised",
        "reasoning_effort": "invalid-selection-effort",
        "priority": 99,
    }
    provider = GitHubCopilotProvider(config=config, client=wrapper)
    provider._copilot_models_cache = [cached_domain]
    return SimpleNamespace(
        provider=provider,
        wrapper=wrapper,
        sdk=sdk,
        rpc_list=rpc_list,
        request=request,
        request_factory=request_factory,
        read_cache=read_cache,
        write_cache=write_cache,
        stale=stale,
        cached_domain=cached_domain,
        config=config,
    )


def assert_caches_untouched(boundary: SimpleNamespace) -> None:
    boundary.read_cache.assert_not_called()
    boundary.write_cache.assert_not_called()
    boundary.sdk.list_models.assert_not_called()
    boundary.sdk.start.assert_not_called()
    assert boundary.wrapper._model_list_cache == [boundary.stale]
    assert boundary.provider._copilot_models_cache == [boundary.cached_domain]
    assert boundary.provider.config is boundary.config
    assert boundary.wrapper._sdk_client is boundary.sdk


def test_discovery_is_optional_async_no_argument_extension() -> None:
    method = GitHubCopilotProvider.discover_models
    assert inspect.iscoroutinefunction(method)
    assert list(inspect.signature(method).parameters) == ["self"]
    assert list(inspect.signature(GitHubCopilotProvider.list_models).parameters) == ["self"]


async def test_current_ids_exact_metadata_preserved_selection_ignored(
    inventory_boundary: SimpleNamespace,
) -> None:
    boundary = inventory_boundary
    model = make_sdk_model_info(
        model_id="account-model-vNext",
        name="Current account model",
        supports_reasoning_effort=True,
        supported_reasoning_efforts=["low", "high"],
        default_reasoning_effort="low",
    )
    # Authentic API IDs need no local allowlist; API-advertised "auto" is not invented.
    boundary.rpc_list.return_value = SimpleNamespace(
        models=[model, make_sdk_model_info(model_id="auto")]
    )

    result = await boundary.provider.discover_models()

    assert set(result) == {"version", "source", "models"}
    assert result["version"] == 1
    assert result["source"] == "provider-api"
    assert [item.id for item in result["models"]] == ["account-model-vNext", "auto"]
    assert result["models"][0].display_name == "Current account model"
    assert result["models"][0].defaults["reasoning_effort"] == "low"
    boundary.rpc_list.assert_awaited_once_with(boundary.request)
    assert_caches_untouched(boundary)


async def test_repeated_probes_reach_rpc_and_empty_remains_empty(
    inventory_boundary: SimpleNamespace,
) -> None:
    boundary = inventory_boundary
    boundary.rpc_list.side_effect = [
        SimpleNamespace(models=[make_sdk_model_info(model_id="first-current-model")]),
        SimpleNamespace(models=[]),
        SimpleNamespace(models=[make_sdk_model_info(model_id="new-current-model")]),
    ]

    first = await boundary.provider.discover_models()
    empty = await boundary.provider.discover_models()
    third = await boundary.provider.discover_models()

    assert [model.id for model in first["models"]] == ["first-current-model"]
    assert empty == {"version": 1, "source": "provider-api", "models": []}
    assert [model.id for model in third["models"]] == ["new-current-model"]
    assert boundary.rpc_list.await_args_list == [call(boundary.request)] * 3
    assert boundary.request_factory.call_count == 3
    assert_caches_untouched(boundary)


async def test_concurrent_strict_calls_each_issue_rpc(
    inventory_boundary: SimpleNamespace,
) -> None:
    boundary = inventory_boundary
    boundary.rpc_list.return_value = SimpleNamespace(models=[])

    results = await asyncio.gather(*(boundary.provider.discover_models() for _ in range(3)))

    assert results == [{"version": 1, "source": "provider-api", "models": []}] * 3
    assert boundary.rpc_list.await_count == 3
    assert_caches_untouched(boundary)


@pytest.mark.parametrize(
    ("error", "expected_type"),
    [
        (PermissionError("401 Unauthorized"), AuthenticationError),
        # The existing policy maps endpoint refusal to provider unavailability;
        # process/transport loss maps to NetworkError. Preserve both distinctions.
        (ConnectionError("Connection refused"), ProviderUnavailableError),
        (ProcessExitedError("process exited"), NetworkError),
        (TimeoutError("SDK timeout"), LLMTimeoutError),
        (RuntimeError("unrecognized SDK failure"), ProviderUnavailableError),
    ],
)
async def test_rpc_failures_raise_typed_errors_despite_all_legacy_caches(
    inventory_boundary: SimpleNamespace,
    error: Exception,
    expected_type: type[LLMError],
) -> None:
    boundary = inventory_boundary
    boundary.rpc_list.side_effect = error

    with pytest.raises(expected_type) as raised:
        await boundary.provider.discover_models()

    assert raised.value.__cause__ is error
    assert raised.value.provider == "github-copilot"
    boundary.rpc_list.assert_awaited_once_with(boundary.request)
    assert_caches_untouched(boundary)


@pytest.mark.parametrize("error_type", [AuthenticationError, NetworkError, LLMTimeoutError])
async def test_already_typed_errors_are_not_wrapped(
    inventory_boundary: SimpleNamespace,
    error_type: type[LLMError],
) -> None:
    boundary = inventory_boundary
    error = error_type("Already translated", provider="github-copilot")
    boundary.rpc_list.side_effect = error

    with pytest.raises(error_type) as raised:
        await boundary.provider.discover_models()

    assert raised.value is error
    assert_caches_untouched(boundary)


@pytest.mark.parametrize(
    "response",
    [
        None,
        SimpleNamespace(),
        SimpleNamespace(models=None),
        SimpleNamespace(models={}),
        SimpleNamespace(models=""),
        SimpleNamespace(models=[None]),
        SimpleNamespace(models=[SimpleNamespace(id="missing-required-shape")]),
        SimpleNamespace(models=[make_sdk_model_info(model_id="")]),
        SimpleNamespace(models=[make_sdk_model_info(model_id=" \t")]),
        SimpleNamespace(models=[SimpleNamespace(id=None, name="Invalid", capabilities=None)]),
        SimpleNamespace(models=[SimpleNamespace(id=42, name="Invalid", capabilities=None)]),
        SimpleNamespace(
            models=[
                make_sdk_model_info(model_id="valid-before-malformed"),
                SimpleNamespace(id=None, name="Invalid", capabilities=None),
            ]
        ),
    ],
)
async def test_malformed_inventory_never_becomes_success_or_partial_success(
    inventory_boundary: SimpleNamespace, response: Any
) -> None:
    boundary = inventory_boundary
    boundary.rpc_list.return_value = response

    with pytest.raises(ProviderUnavailableError):
        await boundary.provider.discover_models()

    boundary.rpc_list.assert_awaited_once_with(boundary.request)
    assert_caches_untouched(boundary)


async def test_success_does_not_mask_later_failure_or_prevent_recovery(
    inventory_boundary: SimpleNamespace,
) -> None:
    boundary = inventory_boundary
    boundary.rpc_list.side_effect = [
        SimpleNamespace(models=[make_sdk_model_info(model_id="currently-available")]),
        TimeoutError("SDK timeout"),
        SimpleNamespace(models=[]),
    ]

    assert (await boundary.provider.discover_models())["models"][0].id == "currently-available"
    with pytest.raises(LLMTimeoutError):
        await boundary.provider.discover_models()
    assert await boundary.provider.discover_models() == {
        "version": 1,
        "source": "provider-api",
        "models": [],
    }
    assert boundary.rpc_list.await_count == 3
    assert_caches_untouched(boundary)


async def test_initialization_failure_is_typed_without_cache_fallback(
    inventory_boundary: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    boundary = inventory_boundary
    error = RuntimeError("Copilot client has been stopped")
    initialize = AsyncMock(side_effect=error)
    monkeypatch.setattr(boundary.wrapper, "_ensure_client_initialized", initialize)

    with pytest.raises(ProviderUnavailableError) as raised:
        await boundary.provider.discover_models()

    assert raised.value.__cause__ is error
    boundary.rpc_list.assert_not_called()
    assert_caches_untouched(boundary)


async def test_legacy_wrapper_no_argument_calls_still_cache_and_copy(
    inventory_boundary: SimpleNamespace,
) -> None:
    boundary = inventory_boundary
    boundary.wrapper._model_list_cache = None
    model = make_sdk_model_info(model_id="legacy-current-model")
    boundary.rpc_list.return_value = SimpleNamespace(models=[model])

    first = await boundary.wrapper.list_models()
    second = await boundary.wrapper.list_models()
    first.clear()

    assert second == [model]
    assert await boundary.wrapper.list_models() == [model]
    boundary.rpc_list.assert_awaited_once_with(boundary.request)


async def test_strict_success_does_not_seed_a_cold_legacy_cache(
    inventory_boundary: SimpleNamespace,
) -> None:
    boundary = inventory_boundary
    boundary.wrapper._model_list_cache = None
    boundary.rpc_list.side_effect = [
        SimpleNamespace(models=[]),
        SimpleNamespace(models=[boundary.stale]),
    ]

    assert (await boundary.provider.discover_models())["models"] == []
    assert boundary.wrapper._model_list_cache is None
    assert await boundary.wrapper.list_models() == [boundary.stale]
    assert boundary.rpc_list.await_count == 2


async def test_legacy_fetch_helper_keeps_no_keyword_client_call() -> None:
    class LegacyClient:
        """A client with the old exact no-argument list_models signature."""

        async def list_models(self) -> list[Any]:
            return [make_sdk_model_info(model_id="legacy-no-keyword-model")]

    result = await model_helpers.fetch_models(LegacyClient())
    assert [model.id for model in result] == ["legacy-no-keyword-model"]


async def test_legacy_provider_still_warms_caches(
    inventory_boundary: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    boundary = inventory_boundary
    mapped = model_helpers.copilot_model_to_amplifier_model(boundary.cached_domain)
    fetch = AsyncMock(return_value=([mapped], [boundary.cached_domain]))
    monkeypatch.setattr(provider_module, "fetch_and_map_models", fetch)

    assert await boundary.provider.list_models() == [mapped]

    fetch.assert_awaited_once_with(boundary.wrapper)
    boundary.write_cache.assert_called_once_with([boundary.cached_domain])
    boundary.read_cache.assert_not_called()
    assert boundary.provider._copilot_models_cache == [boundary.cached_domain]


async def test_legacy_provider_still_falls_back_to_disk(
    inventory_boundary: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    boundary = inventory_boundary
    fetch = AsyncMock(side_effect=ProviderUnavailableError("SDK unavailable"))
    monkeypatch.setattr(provider_module, "fetch_and_map_models", fetch)

    result = await boundary.provider.list_models()

    assert [model.id for model in result] == ["other-account-stale-model"]
    fetch.assert_awaited_once_with(boundary.wrapper)
    boundary.read_cache.assert_called_once_with()
    boundary.write_cache.assert_not_called()
    assert boundary.provider._copilot_models_cache == [boundary.cached_domain]
