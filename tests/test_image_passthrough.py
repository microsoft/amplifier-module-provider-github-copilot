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

from typing import Any

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
