# NOTE: sdk_session mocks are intentionally unspecced - each test configures custom send()
# behavior to capture attachments and simulate events. MockSDKSession from fixtures is available
# for simpler cases, but these tests need custom callback behavior.

"""
Test image passthrough from ChatRequest to SDK BlobAttachment.

Contract: contracts/sdk-boundary.md § Image/Attachment Passthrough
Contract: contracts/provider-protocol.md § complete()

Anchors verified:
- sdk-boundary:ImagePassthrough:MUST:1 — Extract images from LAST user message only
- sdk-boundary:ImagePassthrough:MUST:2 — Convert ImageBlock to BlobAttachment
- sdk-boundary:ImagePassthrough:MUST:3 — Skip non-base64 images
- sdk-boundary:ImagePassthrough:MUST:4 — Skip empty image data
- sdk-boundary:ImagePassthrough:MUST:5 — No model capability validation
- sdk-boundary:ImagePassthrough:MUST:6 — No image content modification
- sdk-boundary:ImagePassthrough:MUST:7 — Forward attachments via send()
- provider-protocol:complete:MUST:7 — Extracts images from last user message
- provider-protocol:complete:MUST:8 — Forwards images as BlobAttachments to SDK
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

# ---------------------------------------------------------------------------
# Local stubs for test objects (replace bare MagicMock for image/message/request)
# ---------------------------------------------------------------------------


class _ImageBlockStub:
    """Stub for ImageBlock with source dict."""

    def __init__(self, type: str = "image", source: dict[str, Any] | None = None) -> None:
        self.type = type
        self.source = source or {}


class _MessageStub:
    """Stub for ChatMessage with role and content."""

    def __init__(self, role: str, content: Any) -> None:
        self.role = role
        self.content = content


class _ContentBlockStub:
    """Stub for content blocks (text, tool_result, etc.)."""

    def __init__(self, type: str, **kwargs: Any) -> None:
        self.type = type
        for key, value in kwargs.items():
            setattr(self, key, value)


class _RequestStub:
    """Stub for ChatRequest with messages."""

    def __init__(self, messages: list[Any]) -> None:
        self.messages = messages


class _SDKSessionStub:
    """Stub for SDK session with send/disconnect/on methods."""

    def __init__(self) -> None:
        self.send_calls: list[tuple[str, list[dict[str, Any]] | None]] = []
        self._disconnect_called: bool = False

    async def send(self, prompt: str, *, attachments: list[dict[str, Any]] | None = None) -> str:
        self.send_calls.append((prompt, attachments))
        return "msg-123"

    async def disconnect(self) -> None:
        self._disconnect_called = True

    def on(self, handler: object) -> None:
        _ = handler  # SDK session registers event handlers


class TestConvertImageBlockToBlobAttachment:
    """Test ImageBlock → BlobAttachment conversion.

    Contract: sdk-boundary:ImagePassthrough:MUST:2
    """

    def test_valid_base64_image_converts_to_blob_attachment(self) -> None:
        """sdk-boundary:ImagePassthrough:MUST:2 — Convert ImageBlock to BlobAttachment."""
        from amplifier_module_provider_github_copilot.sdk_adapter.types import (
            convert_image_block_to_blob_attachment,
        )

        # ImageBlock from amplifier-core (Pydantic model with source dict)
        image_block = _ImageBlockStub(
            type="image",
            source={
                "type": "base64",
                "media_type": "image/png",
                "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ",
            },
        )

        result = convert_image_block_to_blob_attachment(image_block)

        assert result == {
            "type": "blob",
            "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ",
            "mimeType": "image/png",
        }

    def test_non_base64_image_returns_none(self) -> None:
        """Contract: sdk-boundary:ImagePassthrough:MUST:3 — Skip non-base64 images."""
        from amplifier_module_provider_github_copilot.sdk_adapter.types import (
            convert_image_block_to_blob_attachment,
        )

        # URL-based image (not supported by SDK)
        image_block = _ImageBlockStub(
            type="image",
            source={
                "type": "url",
                "url": "https://example.com/image.png",
            },
        )

        result = convert_image_block_to_blob_attachment(image_block)
        assert result is None

    def test_empty_data_returns_none(self) -> None:
        """Contract: sdk-boundary:ImagePassthrough:MUST:4 — Skip empty image data."""
        from amplifier_module_provider_github_copilot.sdk_adapter.types import (
            convert_image_block_to_blob_attachment,
        )

        image_block = _ImageBlockStub(
            type="image",
            source={
                "type": "base64",
                "media_type": "image/png",
                "data": "",  # Empty data
            },
        )

        result = convert_image_block_to_blob_attachment(image_block)
        assert result is None

    def test_missing_data_returns_none(self) -> None:
        """sdk-boundary:ImagePassthrough:MUST:4 — Skip missing image data."""
        from amplifier_module_provider_github_copilot.sdk_adapter.types import (
            convert_image_block_to_blob_attachment,
        )

        image_block = _ImageBlockStub(
            type="image",
            source={
                "type": "base64",
                "media_type": "image/png",
                # No "data" key
            },
        )

        result = convert_image_block_to_blob_attachment(image_block)
        assert result is None

    def test_default_mime_type_if_missing(self) -> None:
        """Default to image/png if media_type missing."""
        # Contract: sdk-boundary:ImagePassthrough:MUST:2
        from amplifier_module_provider_github_copilot.sdk_adapter.types import (
            convert_image_block_to_blob_attachment,
        )

        image_block = _ImageBlockStub(
            type="image",
            source={
                "type": "base64",
                "data": "somebase64data",
                # No media_type
            },
        )

        result = convert_image_block_to_blob_attachment(image_block)
        assert result == {"type": "blob", "data": "somebase64data", "mimeType": "image/png"}

    def test_data_not_modified(self) -> None:
        """Contract: sdk-boundary:ImagePassthrough:MUST:6 — No image content modification."""
        from amplifier_module_provider_github_copilot.sdk_adapter.types import (
            convert_image_block_to_blob_attachment,
        )

        original_data = "EXACT_BASE64_STRING_NOT_MODIFIED"
        image_block = _ImageBlockStub(
            type="image",
            source={
                "type": "base64",
                "media_type": "image/jpeg",
                "data": original_data,
            },
        )

        result = convert_image_block_to_blob_attachment(image_block)
        # Data must be EXACTLY the same, not modified
        assert result is not None  # narrowed for pyright
        assert result == {"type": "blob", "data": original_data, "mimeType": "image/jpeg"}
        assert result["data"] is original_data


class TestExtractAttachmentsFromChatRequest:
    """Test extraction of images from ChatRequest.

    Contract: sdk-boundary:ImagePassthrough:MUST:1
    Contract: provider-protocol:complete:MUST:7
    """

    def test_extracts_from_last_user_message_only(self) -> None:
        """sdk-boundary:ImagePassthrough:MUST:1 — Extract from LAST user message."""
        from amplifier_module_provider_github_copilot.sdk_adapter.types import (
            extract_attachments_from_chat_request,
        )

        # Build a ChatRequest with multiple user messages
        old_src = {"type": "base64", "data": "OLD", "media_type": "image/png"}
        msg1 = _MessageStub(
            role="user",
            content=[_ImageBlockStub(type="image", source=old_src)],
        )

        msg2 = _MessageStub(role="assistant", content="I see your image")

        new_src = {"type": "base64", "data": "NEW", "media_type": "image/png"}
        msg3 = _MessageStub(
            role="user",
            content=[_ImageBlockStub(type="image", source=new_src)],
        )

        request = _RequestStub(messages=[msg1, msg2, msg3])

        result = extract_attachments_from_chat_request(request)

        # Should only get image from LAST user message (msg3)
        assert len(result) == 1
        assert result[0]["data"] == "NEW"

    def test_no_user_messages_returns_empty(self) -> None:
        """No user messages → empty attachments list."""
        # Contract: provider-protocol:complete:MUST:7
        from amplifier_module_provider_github_copilot.sdk_adapter.types import (
            extract_attachments_from_chat_request,
        )

        request = _RequestStub(messages=[_MessageStub(role="assistant", content="Hello")])

        result = extract_attachments_from_chat_request(request)
        assert result == []

    def test_text_content_returns_empty(self) -> None:
        """User message with plain text → empty attachments."""
        # Contract: provider-protocol:complete:MUST:7
        from amplifier_module_provider_github_copilot.sdk_adapter.types import (
            extract_attachments_from_chat_request,
        )

        request = _RequestStub(
            messages=[_MessageStub(role="user", content="Just text, no images")]  # str, not list
        )

        result = extract_attachments_from_chat_request(request)
        assert result == []

    def test_multiple_images_in_last_message(self) -> None:
        """Multiple images in last user message → all extracted."""
        # Contract: provider-protocol:complete:MUST:7
        from amplifier_module_provider_github_copilot.sdk_adapter.types import (
            extract_attachments_from_chat_request,
        )

        src1 = {"type": "base64", "data": "IMG1", "media_type": "image/png"}
        src2 = {"type": "base64", "data": "IMG2", "media_type": "image/jpeg"}
        request = _RequestStub(
            messages=[
                _MessageStub(
                    role="user",
                    content=[
                        _ContentBlockStub(type="text", text="Here are images"),
                        _ImageBlockStub(type="image", source=src1),
                        _ImageBlockStub(type="image", source=src2),
                    ],
                )
            ]
        )

        result = extract_attachments_from_chat_request(request)

        assert len(result) == 2
        assert result[0]["data"] == "IMG1"
        assert result[1]["data"] == "IMG2"

    def test_mixed_content_filters_images_only(self) -> None:
        """Mixed content blocks → only images extracted."""
        # Contract: provider-protocol:complete:MUST:7
        from amplifier_module_provider_github_copilot.sdk_adapter.types import (
            extract_attachments_from_chat_request,
        )

        img_src = {"type": "base64", "data": "IMG", "media_type": "image/png"}
        request = _RequestStub(
            messages=[
                _MessageStub(
                    role="user",
                    content=[
                        _ContentBlockStub(type="text", text="Some text"),
                        _ImageBlockStub(type="image", source=img_src),
                        _ContentBlockStub(type="tool_result", tool_call_id="123", output="result"),
                    ],
                )
            ]
        )

        result = extract_attachments_from_chat_request(request)

        assert len(result) == 1
        assert result[0]["data"] == "IMG"


class TestNoCapabilityValidation:
    """Test that provider does NOT validate model vision capability.

    Contract: sdk-boundary:ImagePassthrough:MUST:5
    """

    def test_no_model_capability_check(self) -> None:
        """sdk-boundary:ImagePassthrough:MUST:5 — No model capability validation."""
        from amplifier_module_provider_github_copilot.sdk_adapter.types import (
            extract_attachments_from_chat_request,
        )

        # Even if we pass a model that "shouldn't" support vision,
        # we don't check. The SDK handles it.
        img_src = {"type": "base64", "data": "IMG", "media_type": "image/png"}
        request = _RequestStub(
            messages=[
                _MessageStub(
                    role="user",
                    content=[_ImageBlockStub(type="image", source=img_src)],
                )
            ]
        )

        # Function doesn't take model parameter - it's pure passthrough
        result = extract_attachments_from_chat_request(request)
        assert result == [{"type": "blob", "data": "IMG", "mimeType": "image/png"}]


class TestAttachmentForwarding:
    """Test that attachments are forwarded via session.send().

    Contract: sdk-boundary:ImagePassthrough:MUST:7
    Contract: provider-protocol:complete:MUST:8
    """

    @pytest.mark.asyncio
    async def test_attachments_passed_to_session_send(self) -> None:
        """provider-protocol:complete:MUST:8 — Forward images as BlobAttachments."""
        from unittest.mock import AsyncMock

        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        mock_session = _SDKSessionStub()

        mock_client = AsyncMock()
        mock_client.create_session = AsyncMock(return_value=mock_session)

        wrapper = CopilotClientWrapper(sdk_client=mock_client)

        test_attachments: list[dict[str, Any]] = [
            {"type": "blob", "data": "base64data", "mimeType": "image/png"}
        ]

        async with wrapper.session() as session:
            await session.send("prompt", attachments=test_attachments)

        # Verify send was called with attachments
        assert len(mock_session.send_calls) == 1
        call_prompt, call_attachments = mock_session.send_calls[0]
        assert call_prompt == "prompt"
        assert call_attachments == test_attachments


# Real Core entry -> provider -> membrane -> SDK send fixtures. No live client.
PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwM"
    "CAO+jRZkAAAAASUVORK5CYII="
)


@pytest.fixture(autouse=True)
def _sdk_override_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    # conftest's no-client import mode leaves these symbols unset. Exercise
    # the real pinned SDK value objects, without constructing a live client.
    from tests._sdk_version_gate import require_sdk

    sdk = require_sdk()
    from amplifier_module_provider_github_copilot.sdk_adapter import _imports

    monkeypatch.setattr(_imports, "ModelLimitsOverride", sdk.ModelLimitsOverride)
    monkeypatch.setattr(_imports, "ModelCapabilitiesOverride", sdk.ModelCapabilitiesOverride)


def _image(data: str = PNG) -> dict[str, Any]:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}


def _core_request(messages: list[Any], *, stream: bool = True, tools: bool = False) -> Any:
    from amplifier_core import ChatRequest

    # Repository stub still describes the older dataclass; runtime is Core2.x
    # Pydantic and validates these canonical fields (no fake message objects).
    return cast(Any, ChatRequest)(
        model="catalog-vision",
        messages=messages,
        tools=[{"name": "inspect", "description": "Inspect", "parameters": {}}] if tools else [],
        max_output_tokens=None,
        metadata={"stream": stream},
    )


def _provider_with_sessions(texts: list[str], *, raw: bool = False) -> tuple[Any, Any, list[Any]]:
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
    from amplifier_module_provider_github_copilot.sdk_adapter.client import CopilotClientWrapper
    from tests.fixtures.sdk_mocks import MockSDKSession, idle_event, text_delta_event

    sessions = [
        MockSDKSession(events=[text_delta_event(text), idle_event()], session_id=f"fresh-{i}")
        for i, text in enumerate(texts)
    ]
    sdk = SimpleNamespace(create_session=AsyncMock(side_effect=sessions))
    wrapper = CopilotClientWrapper(sdk_client=sdk)
    provider = GitHubCopilotProvider(client=wrapper, config={"raw": raw})
    return provider, sdk, sessions


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.asyncio
async def test_fresh_correction_keeps_original_final_user_pixels(stream: bool) -> None:
    """Regression: a fresh correction session has never seen the first send's pixels."""
    request = _core_request(
        [{"role": "user", "content": [{"type": "text", "text": "Inspect this"}, _image()]}],
        stream=stream,
        tools=True,
    )
    before = request.model_dump()
    provider, sdk, sessions = _provider_with_sessions(["[Tool Call: inspect()]", "done"])
    await provider.complete(request)
    expected = [{"type": "blob", "data": PNG, "mimeType": "image/png"}]
    assert sdk.create_session.await_count == 2
    assert sessions[0].last_attachments == expected
    assert sessions[1].last_attachments == expected
    assert sdk.create_session.call_args_list[0].kwargs["model"] == "catalog-vision"
    assert sdk.create_session.call_args_list[1].kwargs["model"] == "catalog-vision"
    assert request.model_dump() == before
    assert all(session.disconnected for session in sessions)


@pytest.mark.parametrize("last_role", ["assistant", "tool"])
@pytest.mark.asyncio
async def test_old_user_image_never_becomes_current_attachment(last_role: str) -> None:
    messages: list[Any] = [
        {"role": "user", "content": [_image()]},
        {"role": "assistant", "content": "previous answer"},
    ]
    if last_role == "tool":
        messages[1] = {
            "role": "assistant",
            "content": [{"type": "tool_call", "id": "call-1", "name": "inspect", "input": {}}],
        }
        messages.append({"role": "tool", "tool_call_id": "call-1", "content": "result"})
    provider, _, sessions = _provider_with_sessions(["done"])
    await provider.complete(_core_request(messages))
    assert not sessions[0].last_attachments
    prompt = sessions[0].last_prompt
    assert "[USER]\n[Image unavailable:" in prompt
    assert "no pixels supplied" in prompt
    assert PNG not in prompt


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.asyncio
async def test_tool_images_are_unavailable_in_original_correlated_slot(nested: bool) -> None:
    output = [{"type": "text", "text": "before"}, _image(), {"type": "text", "text": "after"}]
    tool_content: Any = output
    if nested:
        tool_content = [{
            "type": "tool_result", "tool_call_id": "call-1",
            "output": {"success": False, "content": output}, "is_error": True,
        }]
    request = _core_request([
        {"role": "user", "content": "inspect"},
        {"role": "assistant", "content": [
            {"type": "tool_call", "id": "call-1", "name": "inspect", "input": {}},
        ]},
        {"role": "tool", "tool_call_id": "call-1", "name": "inspect", "content": tool_content},
        {"role": "user", "content": "continue in plain text"},
    ])
    before = request.model_dump()
    provider, _, sessions = _provider_with_sessions(["done"])
    await provider.complete(request)
    prompt = sessions[0].last_prompt
    assert not sessions[0].last_attachments
    assert PNG not in prompt and "'source'" not in prompt
    assert "[TOOL]\n[Tool Result (id=call-1)" in prompt
    assert prompt.index("before") < prompt.index("[Image unavailable:") < prompt.index("after")
    assert prompt.index("[Image unavailable:") < prompt.index("[USER]\ncontinue")
    if nested:
        assert "False" in prompt or "false" in prompt
        assert "is_error=True" in prompt
    assert request.model_dump() == before


@pytest.mark.parametrize("source", [
    {"type": "url", "url": "https://private.example/pixels?token=private"},
    {"type": "base64", "media_type": "image/png", "data": ""},
])
@pytest.mark.asyncio
async def test_unsupported_current_source_has_explicit_payload_free_slot(
    source: dict[str, Any],
) -> None:
    provider, _, sessions = _provider_with_sessions(["done"])
    await provider.complete(_core_request([{
        "role": "user", "content": [
            {"type": "text", "text": "before"},
            {"type": "image", "source": source},
            {"type": "text", "text": "after"},
        ],
    }]))
    prompt = sessions[0].last_prompt
    assert "before\n[Image unavailable:" in prompt and "no pixels supplied]\nafter" in prompt
    assert "private.example" not in prompt
    assert not sessions[0].last_attachments


def test_dict_and_typed_attachment_selection_parity() -> None:
    from types import SimpleNamespace

    from amplifier_module_provider_github_copilot.sdk_adapter.types import (
        extract_attachments_from_chat_request,
    )

    messages = [{"role": "user", "content": [_image()]}]
    typed = _core_request(messages)
    assert extract_attachments_from_chat_request(SimpleNamespace(messages=messages)) == (
        extract_attachments_from_chat_request(typed)
    )
    assert extract_attachments_from_chat_request(typed)[0]["data"] == PNG


@pytest.mark.parametrize("output", [
    None, 0, False, {}, [], {"business": {"type": "image", "value": 7}},
    {"type": "image", "id": "product-record", "success": False, "count": 0, "output": None},
    {"type": "text", "text": "business", "count": 0},
    [{"type": "image", "id": "business"}, {"type": "text", "text": "business", "count": 0}],
])
def test_ordinary_tool_output_is_not_reinterpreted(output: Any) -> None:
    from types import SimpleNamespace

    from amplifier_module_provider_github_copilot.request_adapter import _extract_content_block

    result = _extract_content_block(SimpleNamespace(
        type="tool_result", output=output, tool_call_id="ordinary",
    ))
    assert result == f"[Tool Result (id=ordinary): {output}]"


@pytest.mark.parametrize("status", [True, False])
def test_core_null_result_keeps_correlation_and_error_status(status: bool) -> None:
    from importlib import import_module

    ToolResultBlock = import_module("amplifier_core.message_models").ToolResultBlock

    from amplifier_module_provider_github_copilot.request_adapter import _extract_content_block

    result = _extract_content_block(ToolResultBlock(
        tool_call_id="null-result", output=None, is_error=status,
    ))
    assert result == f"[Tool Result (id=null-result): is_error={status}; None]"


@pytest.mark.parametrize("as_dict", [True, False])
def test_nested_core_image_extras_stay_unavailable_and_private(as_dict: bool) -> None:
    from importlib import import_module

    from amplifier_module_provider_github_copilot.request_adapter import (
        convert_chat_request,
        request_has_images,
    )

    ImageBlock = import_module("amplifier_core.message_models").ImageBlock
    image = ImageBlock(source=_image()["source"], name="capture.png")
    request = _core_request([{
        "role": "user", "content": [{
            "type": "tool_result", "tool_call_id": "nested",
            "output": {"success": False, "content": [image.model_dump() if as_dict else image]},
        }],
    }])
    assert request_has_images(request)
    prompt = convert_chat_request(request).prompt
    assert "Image unavailable" in prompt and PNG not in prompt


class _WirePeer:
    """In-memory JSON-RPC peer behind real pinned SDK serializers; no process/network."""

    def __init__(self, client: Any, texts: list[str], error: str | None = None) -> None:
        self.client = client
        self.texts = iter(texts)
        self.error = error
        self.messages: list[dict[str, Any]] = []
        self.error_phase = "event"
        self.error_send = 1
        self.send_count = 0

    async def request(self, method: str, params: dict[str, Any], **kwargs: Any) -> Any:
        import asyncio
        import json
        from contextvars import Context
        from importlib import import_module

        self.messages.append({"method": method, "params": json.loads(json.dumps(params))})
        if method == "session.create":
            if self.error and self.error_phase == "create":
                raise RuntimeError(self.error)
            return {"sessionId": params["sessionId"]}
        if method == "session.send":
            self.send_count += 1
            current_error = self.error if self.send_count == self.error_send else None
            if current_error and self.error_phase == "send":
                raise RuntimeError(current_error)
            sid = params["sessionId"]
            kind = "session.error" if current_error else "assistant.message_delta"
            data = (
                {"message": current_error, "errorType": "fixture"} if current_error else
                {"deltaContent": next(self.texts), "messageId": "wire-message"}
            )

            def deliver() -> None:
                factory = import_module("copilot.generated.session_events").session_event_from_dict
                for index, (event_type, event_data) in enumerate([
                    (kind, data), ("session.idle", {}),
                ]):
                    event = factory({
                        "id": f"00000000-0000-0000-0000-{index:012d}",
                        "timestamp": "2026-01-01T00:00:00Z",
                        "type": event_type, "data": event_data,
                    })
                    self.client._sessions[sid]._dispatch_event(event)

            # Actual SDK notification callbacks run on a pre-existing reader.
            asyncio.get_running_loop().call_soon(Context().run, deliver)
            return {"messageId": "wire-message"}
        if method in ("session.detach", "session.options.update"):
            return {"success": True}
        if method == "ping":
            return {"message": "pong", "timestamp": 1, "protocolVersion": 1}
        return {}


def _real_sdk_provider(
    path: Path, texts: list[str], *, error: str | None = None,
) -> tuple[Any, _WirePeer]:
    from tests._sdk_version_gate import require_sdk

    sdk_module = require_sdk()
    from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
    from amplifier_module_provider_github_copilot.sdk_adapter import CopilotClientWrapper

    sdk = sdk_module.CopilotClient(
        mode="empty", base_directory=str(path), github_token="fixture", use_logged_in_user=False,
    )
    peer = _WirePeer(sdk, texts, error)
    sdk._client = peer
    wrapper = CopilotClientWrapper(sdk_client=sdk)
    return GitHubCopilotProvider(client=wrapper, config={"raw": True}), peer


@pytest.mark.asyncio
async def test_pinned_sdk_two_fresh_sessions_send_identical_eligible_blobs(tmp_path: Path) -> None:
    provider, peer = _real_sdk_provider(tmp_path, ["[Tool Call: inspect()]", "done"])
    await provider.complete(_core_request([
        {"role": "user", "content": [{"type": "text", "text": "before"}, _image(),
                                   {"type": "text", "text": "after"}, _image()]},
    ], tools=True))
    sends = [row["params"] for row in peer.messages if row["method"] == "session.send"]
    assert len(sends) == 2 and sends[0]["sessionId"] != sends[1]["sessionId"]
    assert sends[0]["attachments"] == sends[1]["attachments"] == [
        {"type": "blob", "data": PNG, "mimeType": "image/png"},
        {"type": "blob", "data": PNG, "mimeType": "image/png"},
    ]
    assert PNG not in sends[0]["prompt"] and PNG not in sends[1]["prompt"]
    assert len([row for row in peer.messages if row["method"] == "session.detach"]) == 2


@pytest.mark.parametrize("echo", [
    "/////////////w==", "data:image/png;base64,/////////////w==",
    "https://private.example/pixels?secret=yes",
])
@pytest.mark.parametrize("phase", ["event", "send", "create", "correction"])
@pytest.mark.asyncio
async def test_image_error_echo_omitted_before_first_hook_and_logger(
    monkeypatch: pytest.MonkeyPatch, echo: str, caplog: pytest.LogCaptureFixture,
    tmp_path: Path, phase: str,
) -> None:
    import json
    import logging
    import traceback
    from types import SimpleNamespace

    from amplifier_module_provider_github_copilot.error_translation import LLMError
    from amplifier_module_provider_github_copilot.security_redaction import IMAGE_ERROR_OMITTED

    caplog.set_level(logging.DEBUG)
    provider, peer = _real_sdk_provider(
        tmp_path, ["[Tool Call: inspect()]"], error=f"400 invalid image: {echo}",
    )
    peer.error_phase = "event" if phase == "correction" else phase
    peer.error_send = 2 if phase == "correction" else 1
    seen: list[str] = []

    async def emit(name: str, data: Any) -> None:
        seen.append(json.dumps([name, data]))
        assert echo not in seen[-1]  # entry, not a later subscriber/filter

    provider.coordinator = SimpleNamespace(hooks=SimpleNamespace(emit=emit))
    with pytest.raises(LLMError) as raised:
        await provider.complete(_core_request(
            [{"role": "user", "content": [_image()]}], tools=phase == "correction",
        ))
    assert not raised.value.retryable
    assert IMAGE_ERROR_OMITTED in str(raised.value)
    assert echo not in caplog.text
    assert echo not in "".join(traceback.format_exception(raised.value))
    assert raised.value.__cause__ is None and raised.value.__context__ is None
    sends = 0 if phase == "create" else (2 if phase == "correction" else 1)
    assert len([row for row in peer.messages if row["method"] == "session.send"]) == sends
    assert any("llm:response" in event for event in seen)


@pytest.mark.asyncio
async def test_image_unknown_send_is_not_replayed(tmp_path: Path) -> None:
    from amplifier_module_provider_github_copilot.error_translation import LLMError

    provider, peer = _real_sdk_provider(
        tmp_path, [], error="429 rate limit retry after 0 seconds: //////////////w==",
    )
    peer.error_phase = "send"
    with pytest.raises(LLMError) as raised:
        await provider.complete(_core_request([{"role": "user", "content": [_image()]}]))
    assert not raised.value.retryable
    assert type(raised.value).__name__ == "RateLimitError"
    assert len([row for row in peer.messages if row["method"] == "session.send"]) == 1


@pytest.mark.asyncio
async def test_sdk_typed_error_keeps_fields_but_not_raw_args_or_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import traceback
    from types import SimpleNamespace

    from amplifier_core.llm_errors import RateLimitError

    provider, peer = _real_sdk_provider(tmp_path, [])
    original = peer.request
    private = RateLimitError(
        "/////////////w== https://private.example/signed", retryable=True,
        status_code=429, retry_after=3, model="catalog-vision",
    )

    async def request(method: str, params: dict[str, Any], **kwargs: Any) -> Any:
        if method == "session.send":
            raise private
        return await original(method, params, **kwargs)

    seen: list[str] = []

    async def emit(name: str, data: Any) -> None:
        seen.append(repr(data))

    monkeypatch.setattr(peer, "request", request)
    provider.coordinator = SimpleNamespace(hooks=SimpleNamespace(emit=emit))
    with pytest.raises(RateLimitError) as raised:
        await provider.complete(_core_request([{"role": "user", "content": [_image()]}]))
    assert raised.value is not private
    outward = cast(Any, raised.value)  # Runtime Core2.x fields absent from repository stub.
    assert outward.status_code == 429 and outward.retry_after == 3
    assert outward.model == "catalog-vision" and not outward.retryable
    assert "private.example" in str(private)
    assert "private.example" not in "".join(traceback.format_exception(raised.value))
    assert "/////////////w==" not in repr(seen)


@pytest.mark.asyncio
async def test_pre_submission_image_retry_keeps_classification_and_original_blobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, peer = _real_sdk_provider(
        tmp_path, ["done"], error="429 rate limit retry after 0 seconds: //////////////w==",
    )
    peer.error_phase = "create"
    original = peer.request
    creates = 0

    async def request(method: str, params: dict[str, Any], **kwargs: Any) -> Any:
        nonlocal creates
        if method == "session.create":
            creates += 1
            if creates > 1:
                peer.error = None
        return await original(method, params, **kwargs)

    monkeypatch.setattr(peer, "request", request)
    monkeypatch.setattr(provider, "_calculate_retry_delay", lambda *args: 0)
    await provider.complete(_core_request([{"role": "user", "content": [_image()]}]))
    assert creates == 2
    sends = [row for row in peer.messages if row["method"] == "session.send"]
    assert len(sends) == 1 and sends[0]["params"]["attachments"][0]["data"] == PNG


@pytest.mark.asyncio
async def test_image_diagnostic_scope_is_task_local_and_restores() -> None:
    import asyncio

    from amplifier_module_provider_github_copilot.security_redaction import (
        IMAGE_ERROR_OMITTED,
        diagnostic_error_message,
        image_error_scope,
        redact_dict,
    )

    gate = asyncio.Event()

    async def image() -> str:
        with image_error_scope(True):
            gate.set()
            await asyncio.sleep(0)
            assert redact_dict({"schemas": [[{"enum": ["https://private.example"]}]]}) == {
                "schemas": [[{"enum": [IMAGE_ERROR_OMITTED]}]],
            }
            with image_error_scope(False):
                return diagnostic_error_message("/////////////w==")

    async def plain() -> str:
        await gate.wait()
        return diagnostic_error_message("ordinary text")

    assert await asyncio.gather(image(), plain()) == [IMAGE_ERROR_OMITTED, "ordinary text"]
    assert diagnostic_error_message("after image") == "after image"


def test_sdk_timing_errors_omit_traceback_before_logrecord(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import logging
    import sys
    import traceback

    from amplifier_module_provider_github_copilot.security_redaction import image_error_scope

    entries: list[str] = []

    def entry(self: Any, level: int, msg: Any, args: Any, **kwargs: Any) -> None:
        info = kwargs.get("exc_info")
        entries.append(repr((msg, args, kwargs)))
        if info:
            error = info if isinstance(info, BaseException) else sys.exc_info()[1]
            entries.append("".join(traceback.format_exception(error)))

    monkeypatch.setattr(logging.Logger, "_log", entry)
    private = "/////////////w== https://private.example/secret"
    for name, message in [
        ("copilot._jsonrpc", "JsonRpcClient.request JSON-RPC request finished"),
        ("copilot.client", "CopilotClient.create_session failed"),
    ]:
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        try:
            raise RuntimeError(private)
        except RuntimeError:
            with image_error_scope(True):
                logger.log(logging.WARNING, message, exc_info=True, extra={"echo": private})
    assert private not in repr(entries)
