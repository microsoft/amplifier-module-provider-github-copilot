"""Strict inventory probes exercise the real wrapper at its typed SDK RPC boundary.

The SDK RPC and cold-start SDK construction are stubbed: provider orchestration,
wrapper lifecycle/cache/locks, model translation and error translation run normally.
No SDK launch, credentials or network are required. Legacy fallback behavior is
tested separately below.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock, call

import pytest

from amplifier_module_provider_github_copilot import models as model_helpers
from amplifier_module_provider_github_copilot import provider as provider_module
from amplifier_module_provider_github_copilot.config._paths import ProviderPaths
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
from amplifier_module_provider_github_copilot.sdk_adapter import client as client_module
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


@pytest.fixture
async def cold_inventory_boundary(
    inventory_boundary: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> AsyncIterator[SimpleNamespace]:
    """Real lazy initialization, but RPC does not exist until the start barrier opens."""
    boundary = inventory_boundary
    boundary.wrapper._sdk_client = None
    boundary.wrapper._provider_paths = ProviderPaths(
        provider_home=tmp_path / "cold-provider",
        cache_home=tmp_path / "cold-cache",
    )
    boundary.clients = []
    boundary.tasks = []
    boundary.start_entered = asyncio.Event()
    boundary.start_release = asyncio.Event()
    boundary.stop_entered = asyncio.Event()
    boundary.stop_release = asyncio.Event()
    boundary.stop_release.set()
    boundary.start_error = None
    boundary.stop_error = None
    boundary.rpc_list.return_value = SimpleNamespace(models=[])

    def construct_client(**kwargs: Any) -> SimpleNamespace:
        candidate = SimpleNamespace(
            live=False,
            start_cancelled=asyncio.Event(),
            start_finished=asyncio.Event(),
            stop_finished=asyncio.Event(),
        )
        boundary.clients.append(candidate)

        async def start() -> None:
            candidate.live = True  # Simulate a subprocess before RPC readiness.
            candidate._process = SimpleNamespace(pid=12345)
            boundary.start_entered.set()
            try:
                await boundary.start_release.wait()
                if boundary.start_error is not None:
                    raise boundary.start_error
                candidate.rpc = SimpleNamespace(models=SimpleNamespace(list=boundary.rpc_list))
            except asyncio.CancelledError:
                candidate.start_cancelled.set()
                raise
            finally:
                candidate.start_finished.set()

        async def stop() -> None:
            boundary.stop_entered.set()
            try:
                await boundary.stop_release.wait()
                if boundary.stop_error is not None:
                    raise boundary.stop_error
                candidate.live = False
            finally:
                candidate.stop_finished.set()

        candidate.start = AsyncMock(side_effect=start)
        candidate.stop = AsyncMock(side_effect=stop)
        return candidate

    boundary.construct = Mock(side_effect=construct_client)
    monkeypatch.setattr(client_module, "CopilotClient", boundary.construct)
    monkeypatch.setattr(client_module, "_resolve_token", lambda: None)
    monkeypatch.setattr(client_module, "_resolve_sdk_log_level", lambda: "none")
    monkeypatch.setattr(client_module, "locate_cli_binary", lambda: None)

    yield boundary

    # Failures must not leave blocked probes or shielded mock shutdowns behind.
    # Never make a stalled start succeed merely to tear down a failed close test.
    boundary.stop_release.set()
    for task in boundary.tasks:
        if not task.done():
            task.cancel()
    await asyncio.wait_for(asyncio.gather(*boundary.tasks, return_exceptions=True), timeout=2)
    await boundary.wrapper.close()
    for candidate in boundary.clients:
        if candidate.stop.await_count:
            await asyncio.wait_for(candidate.stop_finished.wait(), timeout=2)


def start_cold_probe(boundary: SimpleNamespace) -> tuple[asyncio.Task[Any], asyncio.Event]:
    """Signal entry without sleeps; the task then runs to its first real suspension."""
    entered = asyncio.Event()

    async def probe() -> dict[str, Any]:
        entered.set()
        return await boundary.provider.discover_models()

    task = asyncio.create_task(probe())
    boundary.tasks.append(task)
    return task, entered


def start_cold_close(boundary: SimpleNamespace) -> tuple[asyncio.Task[None], asyncio.Event]:
    """Track a closer and signal entry before it reaches its first suspension."""
    entered = asyncio.Event()

    async def close() -> None:
        entered.set()
        await boundary.wrapper.close()

    task = asyncio.create_task(close())
    boundary.tasks.append(task)
    return task, entered


async def settled_cold_results(*tasks: asyncio.Task[Any]) -> list[Any]:
    """Observe settlement without cancelling the operations being falsified."""
    _, pending = await asyncio.wait(tasks, timeout=2)
    assert not pending, "Cold-start lifecycle operations did not settle"
    return list(await asyncio.gather(*tasks, return_exceptions=True))


def assert_caches_untouched(boundary: SimpleNamespace, *, injected: bool = True) -> None:
    boundary.read_cache.assert_not_called()
    boundary.write_cache.assert_not_called()
    boundary.sdk.list_models.assert_not_called()
    boundary.sdk.start.assert_not_called()
    assert boundary.wrapper._model_list_cache == [boundary.stale]
    assert boundary.provider._copilot_models_cache == [boundary.cached_domain]
    assert boundary.provider.config is boundary.config
    assert boundary.wrapper._sdk_client is (boundary.sdk if injected else None)


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


async def test_cold_concurrent_strict_probes_wait_for_rpc_readiness(
    cold_inventory_boundary: SimpleNamespace,
) -> None:
    boundary = cold_inventory_boundary
    first, _ = start_cold_probe(boundary)
    await asyncio.wait_for(boundary.start_entered.wait(), timeout=2)
    second, second_entered = start_cold_probe(boundary)
    await asyncio.wait_for(second_entered.wait(), timeout=2)

    assert not first.done()
    assert not second.done()
    assert boundary.construct.call_count == 1
    assert not hasattr(boundary.clients[0], "rpc")
    assert boundary.wrapper._owned_client is None
    assert boundary.wrapper.copilot_pid is None
    boundary.rpc_list.assert_not_called()

    boundary.start_release.set()
    results = await asyncio.wait_for(asyncio.gather(first, second), timeout=2)

    assert results == [{"version": 1, "source": "provider-api", "models": []}] * 2
    assert boundary.rpc_list.await_args_list == [call(boundary.request)] * 2
    assert boundary.wrapper._owned_client is boundary.clients[0]
    assert boundary.wrapper.copilot_pid == "12345"
    boundary.clients[0].start.assert_awaited_once()
    boundary.clients[0].stop.assert_not_called()
    assert_caches_untouched(boundary, injected=False)


async def test_cancelled_cold_start_stops_candidate_before_fresh_probe(
    cold_inventory_boundary: SimpleNamespace,
) -> None:
    boundary = cold_inventory_boundary
    boundary.stop_release.clear()
    owner, _ = start_cold_probe(boundary)
    await asyncio.wait_for(boundary.start_entered.wait(), timeout=2)
    abandoned = boundary.clients[0]

    owner.cancel()
    await asyncio.wait_for(boundary.stop_entered.wait(), timeout=2)
    retry, retry_entered = start_cold_probe(boundary)
    await asyncio.wait_for(retry_entered.wait(), timeout=2)

    assert not owner.done()  # Cancellation waits for bounded cleanup.
    assert not retry.done()
    assert boundary.construct.call_count == 1
    assert boundary.wrapper._owned_client is None
    boundary.rpc_list.assert_not_called()

    boundary.stop_release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(owner, timeout=2)
    assert not abandoned.live
    abandoned.stop.assert_awaited_once()

    boundary.start_release.set()
    assert await asyncio.wait_for(retry, timeout=2) == {
        "version": 1,
        "source": "provider-api",
        "models": [],
    }
    assert boundary.construct.call_count == 2
    assert boundary.wrapper._owned_client is boundary.clients[1]
    assert boundary.clients[1].live
    boundary.clients[1].start.assert_awaited_once()
    boundary.rpc_list.assert_awaited_once_with(boundary.request)
    assert_caches_untouched(boundary, injected=False)


async def test_failed_cold_start_stops_candidate_and_retry_is_fresh(
    cold_inventory_boundary: SimpleNamespace,
) -> None:
    boundary = cold_inventory_boundary
    error = RuntimeError("Cold start failed")
    boundary.start_error = error
    owner, _ = start_cold_probe(boundary)
    await asyncio.wait_for(boundary.start_entered.wait(), timeout=2)
    boundary.start_release.set()

    with pytest.raises(ProviderUnavailableError) as raised:
        await asyncio.wait_for(owner, timeout=2)

    assert raised.value.__cause__ is error
    abandoned = boundary.clients[0]
    abandoned.stop.assert_awaited_once()
    assert not abandoned.live
    assert boundary.wrapper._owned_client is None
    boundary.rpc_list.assert_not_called()
    boundary.start_error = None

    assert (await boundary.provider.discover_models())["models"] == []
    assert boundary.construct.call_count == 2
    assert boundary.wrapper._owned_client is boundary.clients[1]
    boundary.rpc_list.assert_awaited_once_with(boundary.request)
    assert_caches_untouched(boundary, injected=False)


async def test_close_during_cold_start_prevents_publication_and_waiter_restart(
    cold_inventory_boundary: SimpleNamespace,
) -> None:
    boundary = cold_inventory_boundary
    owner, _ = start_cold_probe(boundary)
    await asyncio.wait_for(boundary.start_entered.wait(), timeout=2)
    waiter, waiter_entered = start_cold_probe(boundary)
    await asyncio.wait_for(waiter_entered.wait(), timeout=2)

    await asyncio.wait_for(boundary.wrapper.close(), timeout=2)
    assert not boundary.wrapper.is_healthy()
    # close() must settle both callers without test-driven cancellation or readiness.
    results = await settled_cold_results(owner, waiter)

    assert all(isinstance(result, ProviderUnavailableError) for result in results)
    assert all("stopped" in str(result) for result in results)
    assert not boundary.start_release.is_set()
    assert boundary.construct.call_count == 1
    boundary.clients[0].start.assert_awaited_once()
    assert boundary.clients[0].start_cancelled.is_set()
    assert boundary.clients[0].start_finished.is_set()
    assert not hasattr(boundary.clients[0], "rpc")
    boundary.clients[0].stop.assert_awaited_once()
    assert boundary.clients[0].stop_finished.is_set()
    assert not boundary.clients[0].live
    assert boundary.wrapper._owned_client is None
    assert boundary.wrapper.copilot_pid is None
    boundary.rpc_list.assert_not_called()
    with pytest.raises(ProviderUnavailableError, match="stopped"):
        await boundary.provider.discover_models()
    await boundary.wrapper.close()
    boundary.clients[0].stop.assert_awaited_once()
    # close() intentionally clears the legacy wrapper cache.
    assert boundary.wrapper._model_list_cache is None
    boundary.read_cache.assert_not_called()
    boundary.write_cache.assert_not_called()


async def test_concurrent_and_repeated_close_share_cold_start_cleanup(
    cold_inventory_boundary: SimpleNamespace,
) -> None:
    boundary = cold_inventory_boundary
    boundary.stop_release.clear()
    owner, _ = start_cold_probe(boundary)
    await asyncio.wait_for(boundary.start_entered.wait(), timeout=2)
    waiter, waiter_entered = start_cold_probe(boundary)
    await asyncio.wait_for(waiter_entered.wait(), timeout=2)
    first_close, _ = start_cold_close(boundary)
    await asyncio.wait_for(boundary.stop_entered.wait(), timeout=2)
    second_close, second_entered = start_cold_close(boundary)
    third_close, third_entered = start_cold_close(boundary)
    await asyncio.wait_for(second_entered.wait(), timeout=2)
    await asyncio.wait_for(third_entered.wait(), timeout=2)

    candidate = boundary.clients[0]
    assert candidate.start_cancelled.is_set()
    candidate.stop.assert_awaited_once()
    assert not candidate.stop_finished.is_set()
    assert all(not task.done() for task in (first_close, second_close, third_close))
    assert not owner.done()
    assert not waiter.done()
    assert not boundary.wrapper.is_healthy()
    assert boundary.wrapper._owned_client is None
    assert boundary.wrapper.copilot_pid is None
    boundary.rpc_list.assert_not_called()

    boundary.stop_release.set()
    results = await settled_cold_results(owner, waiter, first_close, second_close, third_close)
    assert all(isinstance(result, ProviderUnavailableError) for result in results[:2])
    assert all("stopped" in str(result) for result in results[:2])
    assert results[2:] == [None, None, None]
    await asyncio.wait_for(boundary.wrapper.close(), timeout=2)
    candidate.stop.assert_awaited_once()
    assert not candidate.live
    assert not hasattr(candidate, "rpc")
    assert not boundary.start_release.is_set()
    assert boundary.construct.call_count == 1
    boundary.rpc_list.assert_not_called()


@pytest.mark.parametrize("cancel_first", [False, True], ids=["close-first", "cancel-first"])
async def test_close_and_external_start_owner_cancellation_share_cleanup(
    cold_inventory_boundary: SimpleNamespace,
    cancel_first: bool,
) -> None:
    boundary = cold_inventory_boundary
    boundary.stop_release.clear()
    owner, _ = start_cold_probe(boundary)
    await asyncio.wait_for(boundary.start_entered.wait(), timeout=2)
    waiter, waiter_entered = start_cold_probe(boundary)
    await asyncio.wait_for(waiter_entered.wait(), timeout=2)

    if cancel_first:
        owner.cancel()
        await asyncio.wait_for(boundary.stop_entered.wait(), timeout=2)
        closer, close_entered = start_cold_close(boundary)
        await asyncio.wait_for(close_entered.wait(), timeout=2)
    else:
        closer, _ = start_cold_close(boundary)
        await asyncio.wait_for(boundary.stop_entered.wait(), timeout=2)
        owner.cancel()
    concurrent_close, concurrent_entered = start_cold_close(boundary)
    await asyncio.wait_for(concurrent_entered.wait(), timeout=2)

    candidate = boundary.clients[0]
    assert candidate.start_cancelled.is_set()
    candidate.stop.assert_awaited_once()
    assert not candidate.stop_finished.is_set()
    assert not boundary.wrapper.is_healthy()
    assert boundary.wrapper._owned_client is None
    assert boundary.wrapper.copilot_pid is None
    boundary.rpc_list.assert_not_called()

    boundary.stop_release.set()
    results = await settled_cold_results(owner, waiter, closer, concurrent_close)
    # An external cancellation may win over close's stopped-error translation.
    assert isinstance(results[0], (asyncio.CancelledError, ProviderUnavailableError))
    if isinstance(results[0], ProviderUnavailableError):
        assert "stopped" in str(results[0])
    assert isinstance(results[1], ProviderUnavailableError)
    assert "stopped" in str(results[1])
    assert results[2:] == [None, None]
    await asyncio.wait_for(boundary.wrapper.close(), timeout=2)
    candidate.stop.assert_awaited_once()
    assert not candidate.live
    assert not hasattr(candidate, "rpc")
    assert not boundary.start_release.is_set()
    assert boundary.construct.call_count == 1
    boundary.rpc_list.assert_not_called()


async def test_close_during_cold_start_bounds_shared_cleanup_and_warns(
    cold_inventory_boundary: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    boundary = cold_inventory_boundary
    config = SimpleNamespace(sdk=SimpleNamespace(close_timeout_seconds=0.01))
    monkeypatch.setattr(client_module, "load_sdk_protection_config", lambda: config)
    caplog.set_level(logging.INFO, logger=client_module.__name__)
    boundary.stop_release.clear()
    owner, _ = start_cold_probe(boundary)
    await asyncio.wait_for(boundary.start_entered.wait(), timeout=2)
    waiter, waiter_entered = start_cold_probe(boundary)
    await asyncio.wait_for(waiter_entered.wait(), timeout=2)
    closer, _ = start_cold_close(boundary)
    candidate = boundary.clients[0]

    try:
        results = await settled_cold_results(owner, waiter, closer)
        assert all(isinstance(result, ProviderUnavailableError) for result in results[:2])
        assert all("stopped" in str(result) for result in results[:2])
        assert results[2] is None
        assert not boundary.start_release.is_set()
        assert candidate.start_cancelled.is_set()
        assert candidate.start_finished.is_set()
        assert not hasattr(candidate, "rpc")
        candidate.stop.assert_awaited_once()
        assert not candidate.stop_finished.is_set()
        assert candidate.live  # Bounded close does not prove subprocess death.
        assert not boundary.wrapper.is_healthy()
        assert boundary.wrapper._owned_client is None
        assert boundary.wrapper.copilot_pid is None
        boundary.rpc_list.assert_not_called()
        with pytest.raises(ProviderUnavailableError, match="stopped"):
            await boundary.provider.discover_models()
        await asyncio.wait_for(boundary.wrapper.close(), timeout=2)
        candidate.stop.assert_awaited_once()
        assert boundary.construct.call_count == 1
        warnings = [
            record.getMessage()
            for record in caplog.records
            if record.name == client_module.__name__ and record.levelno >= logging.WARNING
        ]
        assert sum("stop did not complete within" in message for message in warnings) == 1
        assert any("subprocess may survive" in message for message in warnings)
        assert "Copilot client stopped" not in caplog.text
    finally:
        # Only release shielded shutdown, after observing the bounded-close result.
        # Startup remains blocked even during final test/fixture cleanup.
        boundary.stop_release.set()
        await asyncio.wait_for(candidate.stop_finished.wait(), timeout=2)

    assert not candidate.live
    assert not boundary.start_release.is_set()


@pytest.mark.parametrize("cancel_start", [False, True], ids=["failed-start", "cancelled-start"])
async def test_cold_start_cleanup_error_is_logged_without_masking_original_failure(
    cold_inventory_boundary: SimpleNamespace,
    caplog: pytest.LogCaptureFixture,
    cancel_start: bool,
) -> None:
    boundary = cold_inventory_boundary
    start_error = RuntimeError("Cold start failed")
    boundary.start_error = start_error
    boundary.stop_error = RuntimeError("Mock shutdown failed")
    caplog.set_level(logging.INFO, logger=client_module.__name__)
    owner, _ = start_cold_probe(boundary)
    await asyncio.wait_for(boundary.start_entered.wait(), timeout=2)

    if cancel_start:
        owner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(owner, timeout=2)
    else:
        boundary.start_release.set()
        with pytest.raises(ProviderUnavailableError) as raised:
            await asyncio.wait_for(owner, timeout=2)
        assert raised.value.__cause__ is start_error

    boundary.clients[0].stop.assert_awaited_once()
    assert boundary.clients[0].live  # Failed cleanup must not claim process death.
    assert boundary.wrapper._owned_client is None
    assert "Error stopping client" in caplog.text
    assert "subprocess may survive" in caplog.text
    assert "Copilot client stopped" not in caplog.text
    boundary.rpc_list.assert_not_called()
    assert_caches_untouched(boundary, injected=False)


async def test_cancelled_cold_start_cleanup_uses_close_timeout(
    cold_inventory_boundary: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    boundary = cold_inventory_boundary
    config = SimpleNamespace(sdk=SimpleNamespace(close_timeout_seconds=0.01))
    monkeypatch.setattr(client_module, "load_sdk_protection_config", lambda: config)
    boundary.stop_release.clear()
    owner, _ = start_cold_probe(boundary)
    await asyncio.wait_for(boundary.start_entered.wait(), timeout=2)
    owner.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(owner, timeout=2)

    candidate = boundary.clients[0]
    candidate.stop.assert_awaited_once()
    assert not candidate.stop_finished.is_set()
    assert candidate.live
    assert boundary.wrapper._owned_client is None
    assert "stop did not complete within" in caplog.text
    assert "subprocess may survive" in caplog.text
    boundary.rpc_list.assert_not_called()
    # The existing shield policy lets shutdown finish after the bounded wait.
    boundary.stop_release.set()
    await asyncio.wait_for(candidate.stop_finished.wait(), timeout=2)
    assert not candidate.live
    assert_caches_untouched(boundary, injected=False)


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
