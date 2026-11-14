"""Real pinned SDK/public create+send+JSON-RPC serialization, synthetic stdio only.

The SDK's process connection is injected, not its create_session/send/request/
serializer methods. No Copilot runtime, socket, credentials or inference is used.
This qualifies encoded transport, not backend acceptance or visual understanding.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from importlib import import_module
from types import SimpleNamespace
from typing import Any

import pytest
from amplifier_core import ChatRequest, ImageBlock, Message, TextBlock

from amplifier_module_provider_github_copilot._compat import ConfigurationError
from amplifier_module_provider_github_copilot.media import IMAGE_NOT_RETAINED
from amplifier_module_provider_github_copilot.observability import emit_event
from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
from amplifier_module_provider_github_copilot.sdk_adapter.client import CopilotClientWrapper
from tests._sdk_version_gate import require_sdk

PIXELS = base64.b64encode(b"offline serialization image sentinel").decode()


class WirePeer:
    """Synthetic process stdin peer; inspect real Content-Length-framed bytes."""

    def __init__(self) -> None:
        self.stdin = self
        self.pending = bytearray()
        self.frames: list[bytes] = []
        self.messages: list[dict[str, Any]] = []
        self.rpc: Any = None
        self.client: Any = None
        self.loop = asyncio.get_running_loop()

    def poll(self) -> None:
        return None

    def write(self, data: bytes) -> int:
        self.pending.extend(data)
        return len(data)

    def flush(self) -> None:
        frame = bytes(self.pending)
        self.pending.clear()
        header, body = frame.split(b"\r\n\r\n", 1)
        assert header == f"Content-Length: {len(body)}".encode()
        message = json.loads(body)
        self.frames.append(frame)
        self.messages.append(message)
        method = message["method"]
        params = message["params"]
        if method == "session.create":
            result = {"sessionId": params["sessionId"], "capabilities": {}}
        elif method == "session.send":
            result = {"messageId": "offline-message"}
        elif method in ("session.options.update", "session.detach"):
            result = {"success": True}
        elif method == "ping":
            result = {"message": "offline", "timestamp": 0, "protocolVersion": 1}
        else:
            raise AssertionError(f"Unexpected RPC method {method}")
        self.rpc._handle_message({"jsonrpc": "2.0", "id": message["id"], "result": result})
        if method == "session.send":
            self.loop.call_soon_threadsafe(self.deliver_idle, params["sessionId"])

    def deliver_idle(self, session_id: str) -> None:
        events = import_module("copilot.generated.session_events")
        event = events.session_event_from_dict({
            "id": "00000000-0000-0000-0000-000000000001",
            "timestamp": "2026-01-01T00:00:00Z", "type": "session.idle", "data": {},
        })
        self.client._sessions[session_id]._dispatch_event(event)


class ImmediateSerializingHooks:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    async def emit(self, name: str, data: Any) -> None:
        serialized = json.dumps(data)
        # Do not assert here: production deliberately catches hook exceptions.
        # The caller inspects the captured entry after the emission returns.
        self.events.append((name, serialized))


def real_sdk_provider(
    monkeypatch: pytest.MonkeyPatch, streaming: bool, vision: bool | None
) -> tuple[GitHubCopilotProvider, WirePeer, ImmediateSerializingHooks]:
    sdk = require_sdk()
    # Refuse any accidental startup: only an injected isolated stdio peer exists.
    async def forbidden_start(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("No real runtime or network is allowed")
    monkeypatch.setattr(sdk.CopilotClient, "start", forbidden_start)
    client: Any = sdk.CopilotClient(use_logged_in_user=False, mode="copilot-cli")
    peer = WirePeer()
    rpc = import_module("copilot._jsonrpc").JsonRpcClient(peer)
    rpc._loop = asyncio.get_running_loop()
    peer.rpc, peer.client = rpc, client
    client._client = rpc
    client._state = "connected"
    hooks = ImmediateSerializingHooks()
    coordinator: Any = SimpleNamespace(get_capability=lambda _: None, hooks=hooks)
    provider = GitHubCopilotProvider(
        client=CopilotClientWrapper(sdk_client=client), coordinator=coordinator,
        config={"use_streaming": streaming, "raw": True, "timeout": 2},
    )
    monkeypatch.setattr(
        provider, "_lookup_copilot_model_info",
        lambda _: None if vision is None else SimpleNamespace(supports_vision=vision),
    )
    return provider, peer, hooks


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("historical", [False, True])
@pytest.mark.parametrize("role", ["user", "tool", "assistant", "system"])
@pytest.mark.parametrize("vision", [None, False, True])
async def test_public_completion_real_sdk_wire_and_pre_hook_privacy(
    monkeypatch: pytest.MonkeyPatch, streaming: bool, historical: bool,
    role: str, vision: bool | None
) -> None:
    provider, peer, hooks = real_sdk_provider(monkeypatch, streaming, vision)
    request = ChatRequest(model="fixture", messages=[Message(
        role=role, tool_call_id="call-1" if role == "tool" else None,
        content=[TextBlock(text="before"), ImageBlock(source={
            "type": "base64", "media_type": "image/png", "data": PIXELS,
        }), TextBlock(text="after")],
    )])
    if historical:
        request.messages.append(Message(role="user", content="current text-only turn"))
    canonical = request.model_dump()
    try:
        await provider.complete(request)
        creates = [m["params"] for m in peer.messages if m["method"] == "session.create"]
        sends = [m["params"] for m in peer.messages if m["method"] == "session.send"]
        assert len(creates) == len(sends) == 1
        assert creates[0]["model"] == "fixture"
        assert creates[0]["availableTools"] == []
        assert creates[0]["hooks"]
        assert creates[0]["streaming"] is True
        sent = sends[0]
        supplied = role == "user" and vision is True and not historical
        if supplied:
            assert len(sent["attachments"]) == 1
            decoded = base64.b64decode(sent["attachments"][0]["data"], validate=True)
            assert hashlib.sha256(decoded).digest() == hashlib.sha256(
                base64.b64decode(PIXELS, validate=True)
            ).digest()
        else:
            assert not sent.get("attachments")
            text = (
                creates[0]["systemMessage"]["content"] if role == "system" else sent["prompt"]
            )
            assert text.index("before") < text.index("Image unavailable") < text.index("after")
            if role == "tool":
                assert "[TOOL]\n[Tool Result (id=call-1): before\n[Image unavailable:" in text
                assert "no pixels supplied]\nafter]" in text
            assert PIXELS not in b"".join(peer.frames).decode()
        # Real SDK output passed back to diagnostics must be sanitized at entry,
        # while the byte-captured transport remains intact.
        before = len(hooks.events)
        await emit_event(provider.coordinator, "test:raw-sdk-transport", {"raw": sent})
        assert len(hooks.events) == before + 1
        name, diagnostic = hooks.events[-1]
        assert name == "test:raw-sdk-transport"
        assert PIXELS not in diagnostic
        if supplied:
            assert IMAGE_NOT_RETAINED in diagnostic
        assert all(PIXELS not in entry for _, entry in hooks.events)
        assert request.model_dump() == canonical
        assert any(m["method"] == "session.detach" for m in peer.messages)
        assert not peer.rpc.pending_requests
    finally:
        await provider.close()



@pytest.mark.asyncio
@pytest.mark.parametrize("nested", [False, True])
async def test_model_override_mismatch_refuses_before_real_sdk_create(
    monkeypatch: pytest.MonkeyPatch, nested: bool
) -> None:
    provider, peer, _ = real_sdk_provider(monkeypatch, False, True)
    request = ChatRequest(model="fitted-model", messages=[Message(role="user", content="text")])
    options = {"request_options": {"model": "other-model"}} if nested else {"model": "other-model"}
    try:
        with pytest.raises(ConfigurationError, match="differs from the fitted request"):
            await provider.complete(request, **options)
        assert not peer.frames
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_privacy_oracle_detects_removed_sanitizer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation sentinel: swallowed hook exceptions must not make this test green."""
    hooks = ImmediateSerializingHooks()
    coordinator: Any = SimpleNamespace(hooks=hooks)
    raw = {"attachments": [{"type": "blob", "mimeType": "image/png", "data": PIXELS}]}

    def assert_private() -> None:
        assert len(hooks.events) == 1
        name, serialized = hooks.events[0]
        assert name == "test:privacy-oracle"
        assert PIXELS not in serialized
        assert IMAGE_NOT_RETAINED in serialized

    await emit_event(coordinator, "test:privacy-oracle", {"raw": raw})
    assert_private()
    hooks.events.clear()
    monkeypatch.setattr(
        "amplifier_module_provider_github_copilot.observability.diagnostic_copy", lambda x: x
    )
    await emit_event(coordinator, "test:privacy-oracle", {"raw": raw})
    with pytest.raises(AssertionError):
        assert_private()
    assert raw["attachments"][0]["data"] == PIXELS

