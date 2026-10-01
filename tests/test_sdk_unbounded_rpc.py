"""Run the pinned SDK's actual request wait and cleanup with an in-memory wire."""

from __future__ import annotations

import asyncio
from importlib import import_module
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests._sdk_version_gate import require_sdk


def rpc_client() -> Any:
    require_sdk()
    rpc = import_module("copilot._jsonrpc")
    client = rpc.JsonRpcClient(MagicMock(poll=lambda: 1))
    client._loop = asyncio.get_running_loop()
    client._send_message = AsyncMock()
    return client


async def pending(client: Any) -> str:
    while not client.pending_requests:
        await asyncio.sleep(0)
    return next(iter(client.pending_requests))


@pytest.mark.sdk_assumption
@pytest.mark.asyncio
async def test_actual_sdk_rpc_default_survives_elapsed_time_and_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = rpc_client()
    task = asyncio.create_task(client.request("synthetic.compact"))
    request_id = await pending(client)
    loop = asyncio.get_running_loop()
    original_time = loop.time
    try:
        monkeypatch.setattr(loop, "time", lambda: original_time() + 7200)
        for _ in range(8):
            await asyncio.sleep(0)
        assert not task.done()
    finally:
        monkeypatch.setattr(loop, "time", original_time)
        client._handle_message({"id": request_id, "result": {"success": True}})
    assert await task == {"success": True}
    assert not client.pending_requests


@pytest.mark.sdk_assumption
@pytest.mark.asyncio
async def test_actual_sdk_rpc_cancellation_removes_pending_request() -> None:
    client = rpc_client()
    task = asyncio.create_task(client.request("synthetic.compact"))
    await pending(client)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not client.pending_requests


@pytest.mark.sdk_assumption
@pytest.mark.asyncio
async def test_actual_sdk_rpc_connection_failure_wakes_unlimited_request() -> None:
    client = rpc_client()
    task = asyncio.create_task(client.request("synthetic.compact"))
    await pending(client)
    client._fail_pending_requests()
    error_type = import_module("copilot._jsonrpc").ProcessExitedError
    with pytest.raises(error_type):
        await task
    assert not client.pending_requests


@pytest.mark.sdk_assumption
@pytest.mark.asyncio
async def test_actual_sdk_rpc_explicit_deadline_remains_enforced() -> None:
    client = rpc_client()
    with pytest.raises(TimeoutError):
        await client.request("synthetic.compact", timeout=0.001)
    assert not client.pending_requests
