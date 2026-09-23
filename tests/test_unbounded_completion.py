"""Healthy waits outlive former deadlines; failures and explicit cancellation still end them."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from amplifier_core.llm_errors import AbortError, LLMError, LLMTimeoutError

from amplifier_module_provider_github_copilot.provider import (
    CompletionRequest,
    GitHubCopilotProvider,
)
from amplifier_module_provider_github_copilot.sdk_adapter.types import SessionHandle
from tests.fixtures.sdk_mocks import error_event, idle_event, text_delta_event


class ControlledSession:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.handlers: list[Callable[[Any], None]] = []
        self.session_id = "synthetic-unbounded"
        self.abort = AsyncMock()
        self.disconnect = AsyncMock()

    def on(self, handler: Callable[[Any], None]) -> Callable[[], None]:
        self.handlers.append(handler)
        return lambda: self.handlers.remove(handler)

    async def send(self, prompt: str, **kwargs: Any) -> None:
        self.started.set()

    def emit(self, event: Any) -> None:
        for handler in self.handlers.copy():
            handler(event)


def provider_for(
    raw: ControlledSession, config: dict[str, Any] | None = None, ping: Any = None
) -> GitHubCopilotProvider:
    provider = GitHubCopilotProvider(config={"max_retries": 0, **(config or {})})

    @asynccontextmanager
    async def session(**kwargs: Any) -> AsyncIterator[SessionHandle]:
        try:
            yield SessionHandle(raw, ping=ping, connection_check_interval=0)
        finally:
            await raw.disconnect()

    provider._client.session = session  # type: ignore[method-assign]
    return provider


async def settle() -> None:
    for _ in range(8):
        await asyncio.sleep(0)


@pytest.mark.asyncio
@pytest.mark.parametrize("override_none", [False, True])
async def test_healthy_completion_survives_two_virtual_hours(
    monkeypatch: pytest.MonkeyPatch, override_none: bool
) -> None:
    raw = ControlledSession()
    provider = provider_for(raw, {"timeout": 30} if override_none else {})
    kwargs = {"_timeout_seconds": None} if override_none else {}
    task = asyncio.create_task(
        provider.complete(cast(Any, CompletionRequest(prompt="synthetic")), **kwargs)
    )
    await raw.started.wait()
    loop = asyncio.get_running_loop()
    original_time = loop.time
    # Move the event-loop clock past the old 3600s default without a long test wait.
    try:
        monkeypatch.setattr(loop, "time", lambda: original_time() + 7200)
        await settle()
        assert not task.done(), "A healthy request was terminated by elapsed time"
    finally:
        monkeypatch.setattr(loop, "time", original_time)
        raw.emit(text_delta_event("Finished after the old deadline"))
        raw.emit(idle_event())
    response = await task
    assert response.text == "Finished after the old deadline"
    assert not raw.handlers
    raw.disconnect.assert_awaited_once()
    raw.abort.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("per_call", [True, False])
async def test_explicit_deadline_still_aborts_and_cleans_up(per_call: bool) -> None:
    raw = ControlledSession()
    provider = provider_for(raw, {} if per_call else {"timeout": 0.01})
    kwargs = {"_timeout_seconds": 0.01} if per_call else {}
    with pytest.raises(LLMTimeoutError):
        await provider.complete(cast(Any, CompletionRequest(prompt="synthetic")), **kwargs)
    raw.abort.assert_awaited_once()
    raw.disconnect.assert_awaited_once()
    assert not raw.handlers


@pytest.mark.asyncio
async def test_user_cancellation_aborts_native_work_and_joins_consumer() -> None:
    raw = ControlledSession()
    ping_started = asyncio.Event()
    ping_cancelled = asyncio.Event()

    async def ping() -> None:
        ping_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            ping_cancelled.set()

    provider = provider_for(raw, ping=ping)
    before = set(asyncio.all_tasks())
    task = asyncio.create_task(provider.complete(cast(Any, CompletionRequest(prompt="synthetic"))))
    await raw.started.wait()
    await ping_started.wait()
    task.cancel()
    with pytest.raises(AbortError):
        await task
    assert ping_cancelled.is_set()
    raw.abort.assert_awaited_once()
    raw.disconnect.assert_awaited_once()
    assert not raw.handlers
    assert not [task for task in asyncio.all_tasks() - before if not task.done()]


@pytest.mark.asyncio
async def test_actual_connection_error_after_send_acknowledgment_wakes_waiter() -> None:
    raw = ControlledSession()
    fail = asyncio.Event()

    async def ping() -> None:
        await fail.wait()
        raise ConnectionError("Synthetic SDK pipe closed")

    provider = provider_for(raw, ping=ping)
    task = asyncio.create_task(provider.complete(cast(Any, CompletionRequest(prompt="synthetic"))))
    await raw.started.wait()
    fail.set()
    with pytest.raises(LLMError):
        await task
    raw.disconnect.assert_awaited_once()
    assert not raw.handlers


@pytest.mark.asyncio
async def test_provider_error_event_wakes_unlimited_waiter() -> None:
    raw = ControlledSession()
    task = asyncio.create_task(
        provider_for(raw).complete(cast(Any, CompletionRequest(prompt="synthetic")))
    )
    await raw.started.wait()
    raw.emit(error_event("Synthetic provider failed"))
    with pytest.raises(LLMError):
        await task
    raw.disconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_slow_ping_does_not_become_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = ControlledSession()
    ping_started = asyncio.Event()

    async def ping() -> None:
        ping_started.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(
        provider_for(raw, ping=ping).complete(cast(Any, CompletionRequest(prompt="test")))
    )
    await raw.started.wait()
    await ping_started.wait()
    loop = asyncio.get_running_loop()
    original_time = loop.time
    try:
        monkeypatch.setattr(loop, "time", lambda: original_time() + 7200)
        await settle()
        assert not task.done()
    finally:
        monkeypatch.setattr(loop, "time", original_time)
        raw.emit(text_delta_event("complete"))
        raw.emit(idle_event())
    assert (await task).text == "complete"
