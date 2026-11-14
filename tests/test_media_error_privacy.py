"""Error diagnostics at real SDK/public serializer entry; synthetic stdio only."""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import sys
import traceback
from contextvars import Context
from importlib import import_module
from typing import Any

import pytest
from amplifier_core import ChatRequest, ImageBlock, Message, TextBlock

from amplifier_module_provider_github_copilot.error_translation import LLMError
from amplifier_module_provider_github_copilot.media import (
    IMAGE_ERROR_NOT_RETAINED,
    diagnostic_error_message,
    image_error_scope,
)
from amplifier_module_provider_github_copilot.observability import llm_lifecycle
from tests.test_media_sdk_wire import real_sdk_provider

# Short slash-separated base64 deliberately defeats the unrelated opaque-token
# regex. This is serialization-only input, not a claim of valid image decoding.
PIXELS = base64.b64encode(b"\xff" * 10).decode()


def error_events(peer: Any, message: str, *, partial: bool) -> None:
    def deliver(session_id: str) -> None:
        factory = import_module("copilot.generated.session_events").session_event_from_dict
        rows = []
        if partial:
            rows.append((
                "assistant.message_delta", {"deltaContent": "early text", "messageId": "m1"}
            ))
        rows.append(("session.error", {"message": message, "errorType": "synthetic-rejection"}))
        for index, (kind, data) in enumerate(rows):
            peer.client._sessions[session_id]._dispatch_event(factory({
                "id": f"00000000-0000-0000-0000-{index:012d}",
                "timestamp": "2026-01-01T00:00:00Z", "type": kind, "data": data,
            }))
    # Real SDK notifications can come from a pre-existing reader task/thread,
    # not the completion's context. Error copying must happen in the provider
    # attempt, not rely on scope propagation into notification callbacks.
    peer.deliver_idle = lambda sid: Context().run(deliver, sid)


def request() -> ChatRequest:
    return ChatRequest(model="fixture", messages=[Message(role="user", content=[ImageBlock(source={
        "type": "base64", "media_type": "image/png", "data": PIXELS,
    })])])


def capture_logger_entry(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    entries: list[str] = []
    original = logging.Logger._log

    def entry(self: logging.Logger, level: int, msg: Any, args: Any, **kwargs: Any) -> None:
        # Inspect raw logger arguments, not a handler-filtered or formatted copy.
        info = kwargs.get("exc_info")
        if info:
            if isinstance(info, BaseException):
                info = (type(info), info, info.__traceback__)
            elif not isinstance(info, tuple):
                info = sys.exc_info()
            # Resolve implicit traceback at entry, before a formatter/subscriber
            # can hide it. repr(exc_info=True) alone is a false-green oracle.
            entries.append("".join(traceback.format_exception(info[1])))
        entries.append(repr((msg, args, kwargs)))
        original(self, level, msg, args, **kwargs)

    monkeypatch.setattr(logging.Logger, "_log", entry)
    return entries


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("echo", [PIXELS, f"data:image/png;base64,{PIXELS}"])
@pytest.mark.parametrize("prefix", ["400 invalid image", "429 rate limit retry after 0 seconds"])
async def test_partial_error_has_private_diagnostics_and_never_replays(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    streaming: bool, echo: str, prefix: str,
) -> None:
    caplog.set_level(logging.DEBUG)
    provider, peer, hooks = real_sdk_provider(monkeypatch, streaming, True)
    monkeypatch.setattr(provider, "_calculate_retry_delay", lambda *args: 0)
    entries = capture_logger_entry(monkeypatch)
    error_events(peer, f"{prefix}: {echo}", partial=True)
    canonical = request()
    before = canonical.model_dump()
    try:
        with pytest.raises(LLMError) as raised:
            await provider.complete(canonical)
        assert PIXELS not in json.dumps(hooks.events), "image echo reached hooks.emit entry"
        assert PIXELS not in repr(entries), "image echo reached logger entry"
        sends = [m for m in peer.messages if m["method"] == "session.send"]
        assert len(sends) == 1
        assert sends[0]["params"]["attachments"][0]["data"] == PIXELS
        assert not raised.value.retryable
        assert raised.value.__cause__ is not None
        assert PIXELS in str(raised.value.__cause__)
        assert IMAGE_ERROR_NOT_RETAINED in str(raised.value)
        aborted = [data for name, data in hooks.events if name == "llm:stream_aborted"]
        assert len(aborted) == int(streaming)
        assert any(
            name == "llm:response" and json.loads(data)["status"] == "error"
            for name, data in hooks.events
        )
        assert not any(name == "provider:retry" for name, _ in hooks.events)
        assert canonical.model_dump() == before
        assert not peer.rpc.pending_requests
        assert any(m["method"] == "session.detach" for m in peer.messages)
    finally:
        await provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("rejection", ["400 invalid image", "429 rate limit retry after 0 seconds"])
async def test_create_rejection_echo_is_private_and_retry_policy_preserved(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    streaming: bool, rejection: str,
) -> None:
    caplog.set_level(logging.DEBUG)
    provider, peer, hooks = real_sdk_provider(monkeypatch, streaming, True)
    monkeypatch.setattr(provider, "_calculate_retry_delay", lambda *args: 0)
    entries = capture_logger_entry(monkeypatch)
    creates = 0

    def respond(message: dict[str, Any]) -> Any:
        nonlocal creates
        if message["method"] == "session.create":
            creates += 1
            if creates == 1:
                return {"error": {
                    "code": -32000, "message": f"{rejection}: data:image/png;base64,{PIXELS}"
                }}
        return None

    peer.response_override = respond
    try:
        if rejection.startswith("400"):
            with pytest.raises(LLMError):
                await provider.complete(request())
            assert creates == 1
            assert not any(m["method"] == "session.send" for m in peer.messages)
        else:
            await provider.complete(request())
            assert creates == 2
            assert len([m for m in peer.messages if m["method"] == "session.send"]) == 1
            retry = [json.loads(data) for name, data in hooks.events if name == "provider:retry"]
            assert len(retry) == 1
            assert retry[0]["error_type"] == "RateLimitError"
            assert retry[0]["retry_after"] == 0
        assert PIXELS not in json.dumps(hooks.events)
        assert PIXELS not in repr(entries)
        assert not peer.rpc.pending_requests
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_image_error_scope_is_task_local_restores_and_does_not_scan_text() -> None:
    private = RuntimeError(f"invalid image {PIXELS}")
    gate = asyncio.Event()

    async def image() -> str:
        with image_error_scope(True):
            gate.set()
            await asyncio.sleep(0)
            with image_error_scope(False):
                return diagnostic_error_message(private)

    async def text() -> str:
        await gate.wait()
        return diagnostic_error_message("ordinary opaque text")

    assert await asyncio.gather(image(), text()) == [
        IMAGE_ERROR_NOT_RETAINED, "ordinary opaque text",
    ]
    with pytest.raises(ValueError), image_error_scope(True):
        raise ValueError("unwind")
    assert diagnostic_error_message("ordinary opaque text") == "ordinary opaque text"
    assert str(private) == f"invalid image {PIXELS}"


@pytest.mark.asyncio
async def test_sdk_logger_scope_is_narrow_concurrent_and_resets_after_cancel(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    entries = capture_logger_entry(monkeypatch)
    start = asyncio.Event()
    done = asyncio.Event()
    logger = logging.getLogger("copilot._jsonrpc")
    message = "JsonRpcClient.request JSON-RPC request finished"
    private = ValueError(PIXELS)

    async def image(cancel: bool) -> None:
        with image_error_scope(True):
            start.set()
            # Exercise actual logger-instance.log -> real Logger._log -> real
            # LogRecord and formatter, with malicious extra/stack fields too.
            logger.log(
                logging.WARNING, message, private, exc_info=private,
                extra={"private": private}, stack_info=True,
            )
            if cancel:
                await done.wait()
            else:
                await asyncio.sleep(0)
        assert diagnostic_error_message("restored image task") == "restored image task"

    async def plain() -> None:
        await start.wait()
        logger.log(logging.WARNING, message, exc_info=ValueError("plain-error"))
        # A detached SDK reader thread has no image scope: show this explicitly,
        # rather than claiming globally sanitized SDK/background diagnostics.
        await asyncio.to_thread(Context().run, lambda: logger.log(
            logging.INFO, "plain reader diagnostic %s", "kept-verbatim"
        ))

    first = asyncio.create_task(image(True))
    await asyncio.gather(image(False), plain())
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert diagnostic_error_message("after cancellation") == "after cancellation"
    assert PIXELS not in repr(entries)
    assert PIXELS not in caplog.text
    assert "plain-error" in caplog.text
    assert "kept-verbatim" in caplog.text
    assert caplog.text.count(IMAGE_ERROR_NOT_RETAINED) == 2
    assert all(record.exc_info is None for record in caplog.records
               if record.msg == IMAGE_ERROR_NOT_RETAINED)


@pytest.mark.asyncio
async def test_cleanup_and_hook_errors_do_not_replace_primary_or_leak(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    provider, peer, hooks = real_sdk_provider(monkeypatch, True, True)
    entries = capture_logger_entry(monkeypatch)
    error_events(peer, f"400 invalid image: {PIXELS}", partial=True)
    emit = hooks.emit

    async def failing_hook(name: str, data: Any) -> None:
        await emit(name, data)
        raise RuntimeError(f"observer failure {PIXELS}")

    hooks.emit = failing_hook
    peer.response_override = lambda row: (
        {"error": {"code": -32000, "message": f"detach failure {PIXELS}"}}
        if row["method"] == "session.detach" else None
    )
    try:
        with pytest.raises(LLMError) as caught:
            await provider.complete(request())
        assert PIXELS in str(caught.value.__cause__)
        assert not caught.value.retryable
        assert len([m for m in peer.messages if m["method"] == "session.send"]) == 1
        assert PIXELS not in repr(entries) + caplog.text + json.dumps(hooks.events)
        assert not peer.rpc.pending_requests
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_lifecycle_fallback_omits_private_error_and_restores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, _, hooks = real_sdk_provider(monkeypatch, False, True)
    primary = RuntimeError(f"outside retry body {PIXELS}")
    try:
        with pytest.raises(RuntimeError) as caught, image_error_scope(True):
            async with llm_lifecycle(provider.coordinator, "fixture") as lifecycle:
                await lifecycle.emit_request(
                    message_count=1, tool_count=0, streaming=False, timeout=None,
                )
                raise primary
        assert caught.value is primary
        assert PIXELS not in json.dumps(hooks.events)
        assert IMAGE_ERROR_NOT_RETAINED in hooks.events[-1][1]
        assert diagnostic_error_message("plain after lifecycle") == "plain after lifecycle"
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_error_privacy_oracle_detects_removed_sdk_timing_adapter(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    with image_error_scope(True):
        pass  # Install the real adapter; then remove only its two projections.
    for name in ("copilot._jsonrpc", "copilot.client"):
        logger = logging.getLogger(name)
        wrapped: Any = logger.log
        monkeypatch.setattr(logger, "log", wrapped.__wrapped__)
    monkeypatch.setattr(
        "amplifier_module_provider_github_copilot.media._install_sdk_timing_error_projection",
        lambda: None,
    )
    provider, peer, hooks = real_sdk_provider(monkeypatch, False, True)
    entries = capture_logger_entry(monkeypatch)
    peer.response_override = lambda row: (
        {"error": {"code": -32000, "message": f"400 image_url {PIXELS}"}}
        if row["method"] == "session.create" else None
    )
    try:
        with pytest.raises(LLMError):
            await provider.complete(request())
        # Provider hooks stay safe, but SDK's implicit traceback is a leak.
        assert PIXELS not in json.dumps(hooks.events)
        with pytest.raises(AssertionError):
            assert PIXELS not in repr(entries)
        assert PIXELS in caplog.text
        assert not any(m["method"] == "session.send" for m in peer.messages)
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_concurrent_actual_completions_keep_image_and_plain_errors_isolated(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    provider, peer, hooks = real_sdk_provider(monkeypatch, True, True)
    entries = capture_logger_entry(monkeypatch)
    received: list[str] = []

    def deliver(session_id: str) -> None:
        sent = next(m["params"] for m in peer.messages
                    if m["method"] == "session.send" and m["params"]["sessionId"] == session_id)
        name = "plain-helpful-error" if "plain-one" in sent["prompt"] else PIXELS
        received.append(name)
        # Existing reader context is deliberately detached from all callers.
        factory = import_module("copilot.generated.session_events").session_event_from_dict
        event = factory({
            "id": "00000000-0000-0000-0000-000000000001", "timestamp": "2026-01-01T00:00:00Z",
            "type": "session.error", "data": {
                "message": f"400 invalid image {name}", "errorType": "synthetic-rejection",
            },
        })
        Context().run(peer.client._sessions[session_id]._dispatch_event, event)

    peer.deliver_idle = deliver
    images = [request(), request()]
    for index, image in enumerate(images):
        image.messages[0].content.insert(0, TextBlock(text=f"image-{index}"))
    plain = ChatRequest(model="fixture", messages=[Message(role="user", content="plain-one")])
    try:
        results = await asyncio.gather(
            provider.complete(images[0]), provider.complete(plain), provider.complete(images[1]),
            return_exceptions=True,
        )
        assert all(isinstance(result, LLMError) for result in results)
        assert IMAGE_ERROR_NOT_RETAINED in str(results[0])
        assert "plain-helpful-error" in str(results[1])
        assert IMAGE_ERROR_NOT_RETAINED in str(results[2])
        assert len(received) == 3
        assert PIXELS not in repr(entries) + json.dumps(hooks.events) + caplog.text
        assert len([m for m in peer.messages if m["method"] == "session.send"]) == 3
        assert not peer.rpc.pending_requests
        assert diagnostic_error_message("after actual calls") == "after actual calls"
    finally:
        await provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_partial_image_deadline_aborts_once_without_retry(
    monkeypatch: pytest.MonkeyPatch, streaming: bool,
) -> None:
    provider, peer, hooks = real_sdk_provider(monkeypatch, streaming, True)
    monkeypatch.setattr(provider, "_calculate_retry_delay", lambda *args: 0)
    # Send receives an ACK and a real SDK delta, but no idle/error. The enclosing
    # asyncio.timeout synthesizes TimeoutError outside the inner exception guard.
    def partial(session_id: str) -> None:
        factory = import_module("copilot.generated.session_events").session_event_from_dict
        event = factory({
            "id": "00000000-0000-0000-0000-000000000001", "timestamp": "2026-01-01T00:00:00Z",
            "type": "assistant.message_delta",
            "data": {"deltaContent": "received partial", "messageId": "m1"},
        })
        Context().run(peer.client._sessions[session_id]._dispatch_event, event)
    peer.deliver_idle = partial
    try:
        with pytest.raises(LLMError) as caught:
            await provider.complete(request(), _timeout_seconds=0.05)
        assert type(caught.value).__name__ == "LLMTimeoutError"
        assert caught.value.retryable is False
        assert len([m for m in peer.messages if m["method"] == "session.send"]) == 1
        assert len([m for m in peer.messages if m["method"] == "session.abort"]) == 1
        assert not any(name == "provider:retry" for name, _ in hooks.events)
        assert not peer.rpc.pending_requests
        assert any(m["method"] == "session.detach" for m in peer.messages)
    finally:
        await provider.close()
