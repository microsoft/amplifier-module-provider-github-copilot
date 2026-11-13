"""Offline media projection, exact wire and pre-hook serialization sentinels."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from amplifier_core import ChatRequest, ImageBlock, Message, TextBlock

from amplifier_module_provider_github_copilot._compat import ConfigurationError
from amplifier_module_provider_github_copilot.media import (
    IMAGE_NOT_RETAINED,
    diagnostic_copy,
    project_sdk_images,
)
from amplifier_module_provider_github_copilot.observability import emit_event
from amplifier_module_provider_github_copilot.provider import (
    GitHubCopilotProvider,
    _run_stream_consumer,
    _StreamingContext,
)
from amplifier_module_provider_github_copilot.request_adapter import convert_chat_request
from amplifier_module_provider_github_copilot.streaming import (
    StreamingAccumulator,
    load_event_config,
)
from tests.fixtures.sdk_mocks import MockCopilotClientWrapper, text_delta_event

PIXELS = "c3ludGhldGljLXNhZmUtcGl4ZWxz"


def image() -> ImageBlock:
    return ImageBlock(source={"type": "base64", "media_type": "image/png", "data": PIXELS})


def tracked_client(text: str) -> tuple[Any, list[Any]]:
    client: Any = MockCopilotClientWrapper(events=[text_delta_event(text)])
    original_session = client.session
    sessions: list[Any] = []

    @asynccontextmanager
    async def session(**kwargs: Any) -> AsyncIterator[Any]:
        async with original_session(**kwargs) as current:
            sessions.append(current)
            yield current

    client.session = session
    return client, sessions


@pytest.mark.parametrize("vision", [None, False, True])
def test_original_positions_and_canonical_metadata(vision: bool | None) -> None:
    request = ChatRequest(model="fixture", messages=[
        Message(role="user", content=[TextBlock(text="before"), image(), TextBlock(text="after")]),
        Message(role="assistant", content=[image()]),
        Message(role="tool", tool_call_id="call-1", content=[image()],
                metadata={"opaque": {"signature": "keep"}, "provider_state": "untouched"}),
        Message(role="user", content="current"),
    ])
    original = request.model_dump()
    projected = project_sdk_images(request, supports_vision=vision)
    result = convert_chat_request(request, supports_vision=vision)
    assert not result.attachments
    assert result.prompt.count("Image unavailable") == 3
    assert result.prompt.index("before") < result.prompt.index("Image unavailable")
    assert result.prompt.index("Image unavailable") < result.prompt.index("after")
    assert [m.role for m in projected.messages] == [m.role for m in request.messages]
    assert projected.messages[2].tool_call_id == "call-1"
    assert projected.messages[2].metadata == request.messages[2].metadata
    assert request.model_dump() == original
    assert PIXELS not in result.prompt


@pytest.mark.parametrize("as_dict", [False, True])
@pytest.mark.parametrize("vision", [None, False, True])
def test_current_user_qualified_attachment_only(vision: bool | None, as_dict: bool) -> None:
    msg: Any = Message(role="user", content=[TextBlock(text="inspect"), image()])
    if as_dict:
        msg = msg.model_dump()
    request = SimpleNamespace(messages=[msg])
    result = convert_chat_request(request, supports_vision=vision)
    if vision is True:
        assert result.attachments == [{"type": "blob", "data": PIXELS, "mimeType": "image/png"}]
        assert "supplied via current-user" in result.prompt
    else:
        assert not result.attachments
        assert "Image unavailable" in result.prompt
        assert "no pixels supplied" in result.prompt


def test_last_user_followed_by_tool_is_historical() -> None:
    request = ChatRequest(model="fixture", messages=[
        Message(role="user", content=[image()]),
        Message(role="tool", content="done", tool_call_id="call-1"),
    ])
    result = convert_chat_request(request, supports_vision=True)
    assert not result.attachments
    assert "Image unavailable" in result.prompt


def test_tool_result_structured_images_never_stringify_pixels() -> None:
    request = SimpleNamespace(messages=[SimpleNamespace(
        role="tool", tool_call_id="call-1", content=[
        {"type": "tool_result", "tool_call_id": "call-1", "output": [
            {"type": "text", "text": "before"},
            image().model_dump(),
            {"type": "text", "text": "after"},
        ]}
    ])])
    result = convert_chat_request(request, supports_vision=True)
    assert "call-1" in result.prompt
    assert "Image unavailable" in result.prompt
    assert PIXELS not in result.prompt
    assert not result.attachments


def test_diagnostic_copy_known_shapes_and_core_objects() -> None:
    source = {
        "core": Message(role="user", content=[image()]),
        "anthropic": image().model_dump(),
        "sdk": {"type": "blob", "mimeType": "image/png", "data": PIXELS},
        "openai": {"type": "image_url", "image_url": {"url": "data:image/png;base64," + PIXELS}},
        "gemini": {"inlineData": {"mimeType": "image/png", "data": PIXELS}},
        "opaque": {"signature": "unchanged", "text": '{"type":"image","data":"ordinary text"}'},
    }
    original = deepcopy(source)
    result = diagnostic_copy(source)
    assert PIXELS not in json.dumps(result)
    assert IMAGE_NOT_RETAINED in json.dumps(result)
    assert result["opaque"] == original["opaque"]
    assert source == original
    result["opaque"]["signature"] = "changed"
    assert source["opaque"]["signature"] == "unchanged"


class SerializingHooks:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    async def emit(self, name: str, value: Any) -> None:
        serialized = json.dumps(value)
        assert PIXELS not in serialized  # Entry assertion, before any subscriber.
        self.events.append((name, serialized))


@pytest.mark.asyncio
async def test_all_hook_entry_paths_omit_before_serialization() -> None:
    hooks = SerializingHooks()
    coordinator: Any = SimpleNamespace(hooks=hooks)
    payload = {"raw": {"attachments": [{"type": "blob", "mimeType": "image/png", "data": PIXELS}]}}
    await emit_event(coordinator, "llm:request", payload)
    ctx = _StreamingContext(request_id="fixture")
    ctx._put("llm:stream_delta", payload)
    ctx.signal_done()
    await _run_stream_consumer(ctx, coordinator)
    provider = GitHubCopilotProvider(coordinator=coordinator)
    await provider._emit_content_async(image())
    assert len(hooks.events) == 3
    assert all(IMAGE_NOT_RETAINED in serialized for _, serialized in hooks.events)
    assert payload["raw"]["attachments"][0]["data"] == PIXELS


def test_support_and_real_optional_methods() -> None:
    provider = GitHubCopilotProvider()
    support = provider.get_media_request_support(model="any-model")
    assert support["v"] == 1
    assert support["positions"] == []
    assert support["mime_types"] == ["image/png", "image/jpeg"]
    for name in ("stream", "count_tokens", "request_budget", "compact_context", "refit_request"):
        assert not callable(getattr(provider, name, None))


@pytest.mark.asyncio
async def test_validate_after_session_await_before_send() -> None:
    state = {"revoked": False}
    send = AsyncMock()

    @asynccontextmanager
    async def session(**kwargs: Any) -> AsyncIterator[Any]:
        state["revoked"] = True
        yield SimpleNamespace(
            session_id="offline-fixture", on=lambda _: lambda: None, send=send,
        )

    capability = SimpleNamespace(version=1)
    validations: list[Any] = []
    def validate(request: Any, **kwargs: Any) -> dict[str, bool]:
        validations.append((request, kwargs))
        if state["revoked"]:
            raise ValueError("expired")
        return {"valid": True}
    capability.validate = validate
    coordinator: Any = SimpleNamespace(
        get_capability=lambda _: capability, hooks=SerializingHooks()
    )
    provider = GitHubCopilotProvider(coordinator=coordinator)
    request = ChatRequest(model="fixture", messages=[Message(role="user", content="prepared text")])
    witness = provider._media_validator(request, {"temperature": 0})
    client: Any = SimpleNamespace(session=session, copilot_pid=None)
    with pytest.raises(ConfigurationError, match="refit before transport"):
        await provider._execute_sdk_completion(
            client=client, model="fixture", prompt="prepared text", timeout=1,
            event_config=load_event_config(), accumulator=StreamingAccumulator(),
            validate_media=witness, use_streaming=False,
        )
    assert len(validations) == 2
    assert all(item[0] is request for item in validations)
    assert validations[0][1] == {"purpose": "transport", "request_options": {"temperature": 0}}
    send.assert_not_awaited()


@pytest.mark.asyncio
async def test_retry_revalidates_after_backoff_and_refuses_expiry(monkeypatch: Any) -> None:
    provider = GitHubCopilotProvider()
    request = ChatRequest(model="fixture", messages=[Message(role="user", content="prepared")])
    state = {"expired": False, "attempts": 0}
    def validate(_: Any, **kwargs: Any) -> dict[str, bool]:
        if state["expired"]:
            raise ValueError("expired")
        return {"valid": True}
    capability = SimpleNamespace(version=1, validate=validate)
    coordinator: Any = SimpleNamespace(
        get_capability=lambda _: capability, hooks=SerializingHooks()
    )
    provider.coordinator = coordinator
    async def execute(**kwargs: Any) -> None:
        kwargs["validate_media"]()
        state["attempts"] += 1
        from amplifier_module_provider_github_copilot.error_translation import (
            ProviderUnavailableError,
        )
        raise ProviderUnavailableError("offline failure")
    async def sleep(_: Any) -> None:
        state["expired"] = True
    monkeypatch.setattr(provider, "_execute_sdk_completion", execute)
    monkeypatch.setattr(asyncio, "sleep", sleep)
    with pytest.raises(ConfigurationError, match="refit before transport"):
        await provider.complete(request)
    assert state["attempts"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("vision", [None, False, True])
async def test_complete_actual_sdk_send_attachment_digest(
    streaming: bool, vision: bool | None, monkeypatch: Any
) -> None:
    client, sessions = tracked_client("offline result")
    provider = GitHubCopilotProvider(
        client=client, config={"use_streaming": streaming, "raw": True}
    )
    monkeypatch.setattr(
        provider, "_lookup_copilot_model_info",
        lambda _: None if vision is None else SimpleNamespace(supports_vision=vision),
    )
    request = ChatRequest(model="fixture", messages=[Message(role="user", content=[image()])])
    original = request.model_dump()
    response = await provider.complete(request)
    assert response.text == "offline result"
    assert len(sessions) == 1
    sent = sessions[0]
    if vision is True:
        assert len(sent.last_attachments) == 1
        data = sent.last_attachments[0]["data"]
        assert hashlib.sha256(base64.b64decode(data)).digest() == hashlib.sha256(
            base64.b64decode(PIXELS)
        ).digest()
        assert "supplied via current-user" in sent.last_prompt
    else:
        assert not sent.last_attachments
        assert "Image unavailable" in sent.last_prompt
    assert request.model_dump() == original


@pytest.mark.asyncio
async def test_image_correction_never_claims_new_session_saw_pixels(monkeypatch: Any) -> None:
    client, sessions = tracked_client("offline")
    provider = GitHubCopilotProvider(client=client)
    monkeypatch.setattr(
        provider, "_lookup_copilot_model_info", lambda _: SimpleNamespace(supports_vision=True),
    )
    monkeypatch.setattr(
        "amplifier_module_provider_github_copilot.provider.should_retry_for_fake_tool_calls",
        lambda **_: (True, "offline fixture"),
    )
    request = ChatRequest(model="fixture", messages=[Message(role="user", content=[image()])])
    with pytest.raises(ConfigurationError, match="no automatic text-only correction"):
        await provider.complete(request)
    assert len(sessions) == 1


def test_unprepared_refs_fail_closed_without_host() -> None:
    provider = GitHubCopilotProvider()
    request = ChatRequest(model="fixture", messages=[Message(
        role="tool", content="image ref", tool_call_id="call-1",
        metadata={"experimental.media.v1": {"refs": [{"forged": True}]}},
    )])
    with pytest.raises(ConfigurationError, match="compatible host"):
        provider._media_validator(request, {})()


def test_host_request_options_normalized_without_witness_kwargs() -> None:
    calls: list[dict[str, Any]] = []
    def validate(_: Any, **kwargs: Any) -> dict[str, bool]:
        calls.append(kwargs)
        return {"valid": True}
    coordinator: Any = SimpleNamespace(get_capability=lambda _: SimpleNamespace(
        version=1, validate=validate,
    ))
    provider = GitHubCopilotProvider(coordinator=coordinator)
    request = ChatRequest(model="fixture", messages=[Message(role="user", content="private")])
    provider._media_validator(
        request, {"request_options": {"temperature": 0.5, "model": "fixture"}, "temperature": 0},
    )()
    assert calls == [{
        "purpose": "transport",
        "request_options": {"temperature": 0, "model": "fixture"},
    }]


@pytest.mark.parametrize("role", ["tool", "user", "assistant"])
def test_duck_typed_images_never_claim_attachment(role: str) -> None:
    request = SimpleNamespace(messages=[SimpleNamespace(
        role=role, content=[SimpleNamespace(source="data:image/png;base64," + PIXELS)],
    )])
    result = convert_chat_request(request, supports_vision=True)
    assert not result.attachments
    assert "Image unavailable" in result.prompt
    assert "supplied via current-user" not in result.prompt
    assert PIXELS not in result.prompt


def test_preconverted_attachment_path_cannot_bypass_eligibility() -> None:
    from amplifier_module_provider_github_copilot.sdk_adapter import CompletionRequest

    request = CompletionRequest(
        prompt="unqualified", model="unknown",
        attachments=[{"type": "blob", "mimeType": "image/png", "data": PIXELS}],
    )
    with pytest.raises(ConfigurationError, match="qualified message position"):
        convert_chat_request(request, supports_vision=True)


def test_incompatible_capability_never_silently_bypasses_validation() -> None:
    coordinator: Any = SimpleNamespace(
        get_capability=lambda _: SimpleNamespace(validate=lambda **_: {"valid": True}),
    )
    provider = GitHubCopilotProvider(coordinator=coordinator)
    request = ChatRequest(model="fixture", messages=[Message(role="user", content="private")])
    with pytest.raises(ConfigurationError, match="refit before transport"):
        provider._media_validator(request, {})()
