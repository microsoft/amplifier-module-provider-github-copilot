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
from unittest.mock import MagicMock

import pytest


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
        image_block = MagicMock()
        image_block.source = {
            "type": "base64",
            "media_type": "image/png",
            "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ",
        }

        result = convert_image_block_to_blob_attachment(image_block)

        assert result is not None
        assert result["type"] == "blob"
        assert result["data"] == "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
        assert result["mimeType"] == "image/png"

    def test_non_base64_image_returns_none(self) -> None:
        """sdk-boundary:ImagePassthrough:MUST:3 — Skip non-base64 images."""
        from amplifier_module_provider_github_copilot.sdk_adapter.types import (
            convert_image_block_to_blob_attachment,
        )

        # URL-based image (not supported by SDK)
        image_block = MagicMock()
        image_block.source = {
            "type": "url",
            "url": "https://example.com/image.png",
        }

        result = convert_image_block_to_blob_attachment(image_block)
        assert result is None

    def test_empty_data_returns_none(self) -> None:
        """sdk-boundary:ImagePassthrough:MUST:4 — Skip empty image data."""
        from amplifier_module_provider_github_copilot.sdk_adapter.types import (
            convert_image_block_to_blob_attachment,
        )

        image_block = MagicMock()
        image_block.source = {
            "type": "base64",
            "media_type": "image/png",
            "data": "",  # Empty data
        }

        result = convert_image_block_to_blob_attachment(image_block)
        assert result is None

    def test_missing_data_returns_none(self) -> None:
        """sdk-boundary:ImagePassthrough:MUST:4 — Skip missing image data."""
        from amplifier_module_provider_github_copilot.sdk_adapter.types import (
            convert_image_block_to_blob_attachment,
        )

        image_block = MagicMock()
        image_block.source = {
            "type": "base64",
            "media_type": "image/png",
            # No "data" key
        }

        result = convert_image_block_to_blob_attachment(image_block)
        assert result is None

    def test_default_mime_type_if_missing(self) -> None:
        """Default to image/png if media_type missing."""
        from amplifier_module_provider_github_copilot.sdk_adapter.types import (
            convert_image_block_to_blob_attachment,
        )

        image_block = MagicMock()
        image_block.source = {
            "type": "base64",
            "data": "somebase64data",
            # No media_type
        }

        result = convert_image_block_to_blob_attachment(image_block)
        assert result is not None
        assert result["mimeType"] == "image/png"

    def test_data_not_modified(self) -> None:
        """sdk-boundary:ImagePassthrough:MUST:6 — No image content modification."""
        from amplifier_module_provider_github_copilot.sdk_adapter.types import (
            convert_image_block_to_blob_attachment,
        )

        original_data = "EXACT_BASE64_STRING_NOT_MODIFIED"
        image_block = MagicMock()
        image_block.source = {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": original_data,
        }

        result = convert_image_block_to_blob_attachment(image_block)
        assert result is not None
        # Data must be EXACTLY the same, not modified
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
        request = MagicMock()

        msg1 = MagicMock()
        msg1.role = "user"
        old_src = {"type": "base64", "data": "OLD", "media_type": "image/png"}
        msg1.content = [MagicMock(type="image", source=old_src)]

        msg2 = MagicMock()
        msg2.role = "assistant"
        msg2.content = "I see your image"

        msg3 = MagicMock()
        msg3.role = "user"
        new_src = {"type": "base64", "data": "NEW", "media_type": "image/png"}
        msg3.content = [MagicMock(type="image", source=new_src)]

        request.messages = [msg1, msg2, msg3]

        result = extract_attachments_from_chat_request(request)

        # Should only get image from LAST user message (msg3)
        assert len(result) == 1
        assert result[0]["data"] == "NEW"

    def test_no_user_messages_returns_empty(self) -> None:
        """No user messages → empty attachments list."""
        from amplifier_module_provider_github_copilot.sdk_adapter.types import (
            extract_attachments_from_chat_request,
        )

        request = MagicMock()
        msg = MagicMock()
        msg.role = "assistant"
        msg.content = "Hello"
        request.messages = [msg]

        result = extract_attachments_from_chat_request(request)
        assert result == []

    def test_text_content_returns_empty(self) -> None:
        """User message with plain text → empty attachments."""
        from amplifier_module_provider_github_copilot.sdk_adapter.types import (
            extract_attachments_from_chat_request,
        )

        request = MagicMock()
        msg = MagicMock()
        msg.role = "user"
        msg.content = "Just text, no images"  # str, not list
        request.messages = [msg]

        result = extract_attachments_from_chat_request(request)
        assert result == []

    def test_multiple_images_in_last_message(self) -> None:
        """Multiple images in last user message → all extracted."""
        from amplifier_module_provider_github_copilot.sdk_adapter.types import (
            extract_attachments_from_chat_request,
        )

        request = MagicMock()
        msg = MagicMock()
        msg.role = "user"
        src1 = {"type": "base64", "data": "IMG1", "media_type": "image/png"}
        src2 = {"type": "base64", "data": "IMG2", "media_type": "image/jpeg"}
        msg.content = [
            MagicMock(type="text", text="Here are images"),
            MagicMock(type="image", source=src1),
            MagicMock(type="image", source=src2),
        ]
        request.messages = [msg]

        result = extract_attachments_from_chat_request(request)

        assert len(result) == 2
        assert result[0]["data"] == "IMG1"
        assert result[1]["data"] == "IMG2"

    def test_mixed_content_filters_images_only(self) -> None:
        """Mixed content blocks → only images extracted."""
        from amplifier_module_provider_github_copilot.sdk_adapter.types import (
            extract_attachments_from_chat_request,
        )

        request = MagicMock()
        msg = MagicMock()
        msg.role = "user"
        img_src = {"type": "base64", "data": "IMG", "media_type": "image/png"}
        msg.content = [
            MagicMock(type="text", text="Some text"),
            MagicMock(type="image", source=img_src),
            MagicMock(type="tool_result", tool_call_id="123", output="result"),
        ]
        request.messages = [msg]

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
        request = MagicMock()
        msg = MagicMock()
        msg.role = "user"
        img_src = {"type": "base64", "data": "IMG", "media_type": "image/png"}
        msg.content = [MagicMock(type="image", source=img_src)]
        request.messages = [msg]

        # Function doesn't take model parameter - it's pure passthrough
        result = extract_attachments_from_chat_request(request)
        assert len(result) == 1


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

        mock_session = AsyncMock()
        mock_session.send = AsyncMock(return_value="msg-123")
        mock_session.disconnect = AsyncMock()

        def mock_on(handler: object) -> None:
            _ = handler  # Unused, just to satisfy pyright

        mock_session.on = mock_on

        mock_client = AsyncMock()
        mock_client.create_session = AsyncMock(return_value=mock_session)

        wrapper = CopilotClientWrapper(sdk_client=mock_client)

        test_attachments: list[dict[str, Any]] = [
            {"type": "blob", "data": "base64data", "mimeType": "image/png"}
        ]

        async with wrapper.session() as session:
            await session.send("prompt", attachments=test_attachments)

        # Verify send was called with attachments
        mock_session.send.assert_called_once()
        call_kwargs = mock_session.send.call_args.kwargs
        assert "attachments" in call_kwargs
        assert call_kwargs["attachments"] == test_attachments
