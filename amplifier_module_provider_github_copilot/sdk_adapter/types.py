"""Domain types for SDK adapter.

These types are the ONLY types that cross the SDK boundary.
SDK types MUST NOT leak outside this module.

Contract: contracts/sdk-boundary.md
"""

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, cast

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SessionConfig:
    """Configuration for creating an SDK session.

    Internal convenience type; the SDK's actual session config uses different shapes
    (e.g., system_message: {mode, content}).

    Field Mapping (this dataclass -> SDK session config):
    - model -> session_config["model"]
    - system_prompt -> session_config["system_message"]["content"] (with mode="replace")
    - max_tokens -> (not currently passed to SDK; reserved for future use)

    See client.py:session() for the actual SDK session config construction.

    Attributes:
        model: The model identifier (e.g., "gpt-4", "claude-opus-4.5").
        system_prompt: Optional system prompt. Transformed by client.py into
            SDK's system_message dict with mode="replace".
        max_tokens: Optional maximum tokens for responses.

    Contract: contracts/sdk-boundary.md

    """

    model: str
    system_prompt: str | None = None
    max_tokens: int | None = None


# SDKSession is intentionally an opaque type alias.
# Domain code should not access SDK session internals.
# In the skeleton, we use Any; real implementation will wrap SDK session.
SDKSession = Any


# ============================================================================
# Session Handle (Façade for SDK Session)
# P2-11: SDK membrane - return façade, not raw SDK session
# Contract: sdk-boundary:Membrane:MUST:2, sdk-boundary:TypeTranslation:MUST:4
# ============================================================================


class SessionHandle:
    """Façade for SDK session that hides SDK internals.

    P2-11: This class is the ONLY interface provider.py should use to interact
    with SDK sessions. It ensures SDK types don't leak through the membrane.

    Contract: sdk-boundary:TypeTranslation:MUST:4 — Use opaque handles, not SDK objects

    Attributes:
        _raw_session: Internal SDK session (NEVER expose directly)
        session_id: Opaque string identifier for the session
    """

    __slots__ = ("_raw_session", "session_id")

    def __init__(self, raw_session: Any, session_id: str | None = None) -> None:
        """Create a session handle wrapping a raw SDK session.

        Args:
            raw_session: The raw SDK session object (kept private).
            session_id: Optional session identifier for logging.
        """
        self._raw_session = raw_session
        self.session_id = session_id or getattr(raw_session, "session_id", "unknown")

    def on(self, handler: Callable[[Any], None]) -> Callable[[], None]:
        """Subscribe to session events.

        Args:
            handler: Callback function to receive events.

        Returns:
            Unsubscribe function.
        """
        return self._raw_session.on(handler)  # type: ignore[no-any-return]

    async def send(
        self,
        prompt: str,
        *,
        attachments: list[dict[str, Any]] | None = None,
    ) -> None:
        """Send a message to the session.

        Args:
            prompt: The text prompt to send.
            attachments: Optional list of attachment dicts.
        """
        if attachments:
            await self._raw_session.send(prompt, attachments=attachments)
        else:
            await self._raw_session.send(prompt)

    async def abort(self) -> None:
        """Abort the session."""
        await self._raw_session.abort()

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"SessionHandle(session_id={self.session_id!r})"


SDKSession = Any


# ============================================================================
# Completion Types
# ============================================================================


@dataclass
class CompletionRequest:
    """Request for LLM completion.

    Attributes:
        prompt: The prompt text to send.
        model: Optional model override.
        tools: Tool definitions for the completion (ToolSpec objects or dicts).
        attachments: Image attachments (BlobAttachment dicts) for vision models.
        system_message: Optional system message for SDK session (mode: replace).
        max_tokens: Maximum tokens in response.
        temperature: Sampling temperature.

    """

    prompt: str
    model: str | None = None
    # Note: Using default_factory=list is conventional but pyright strict requires annotation.
    # Runtime type is Any because Amplifier kernel passes ToolSpec objects, not dicts.
    tools: list[Any] = field(default_factory=list)  # type: ignore[misc]
    attachments: list[dict[str, Any]] = field(default_factory=list)  # type: ignore[misc]
    system_message: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None


@dataclass
class CompletionConfig:
    """Configuration for completion lifecycle.

    Attributes:
        session_config: SDK session configuration.
        event_config: Event translation configuration.
        error_config: Error translation configuration.

    """

    session_config: SessionConfig | None = None
    event_config: Any | None = None  # EventConfig from streaming
    error_config: Any | None = None  # ErrorConfig from error_translation


# Type alias for SDK session creation function
SDKCreateFn = Callable[[SessionConfig], Awaitable[SDKSession]]


# ============================================================================
# Tool Conversion
# ============================================================================


@dataclass(frozen=True)
class SDKToolWrapper:
    """Wrapper to provide SDK-compatible attribute access for Amplifier tools.

    The SDK accesses these attributes in TWO code paths:

    1. client.py: Building tool definitions for API
       - tool.name, tool.description, tool.parameters
       - tool.overrides_built_in_tool, tool.skip_permission

    2. session.py: Registering tool handlers locally
       - tool.handler (checked with 'if not tool.handler: continue')

    handler=None is intentional: Amplifier handles tool execution at kernel layer,
    not provider layer. The SDK's 'if not tool.handler: continue' skips registration.

    Contract: sdk-boundary:ToolForwarding:MUST:2
    """

    name: str
    description: str
    parameters: dict[str, Any] | None = None
    overrides_built_in_tool: bool = False
    skip_permission: bool = False
    handler: Any = None  # SDK checks this; None means skip handler registration


def convert_tools_for_sdk(tools: list[Any]) -> list[SDKToolWrapper]:
    """Convert Amplifier tools to SDK-compatible wrapper objects.

    Handles both:
    - ToolSpec objects (Pydantic BaseModel with .name, .description, .parameters attributes)
    - Dict representations (for testing or legacy code)

    Args:
        tools: List of Amplifier tools. Each tool can be:
            - ToolSpec object (from amplifier_core.message_models)
            - Dict with keys: name, description, parameters

    Returns:
        List of SDKToolWrapper objects with SDK-required attributes.

    Contract: sdk-boundary:ToolForwarding:MUST:2

    """
    result: list[SDKToolWrapper] = []
    for tool in tools:
        # Handle both ToolSpec objects (attribute access) and dicts
        if hasattr(tool, "name"):
            # ToolSpec or similar object with attributes
            name = str(tool.name)
            description = str(getattr(tool, "description", "") or "")
            parameters = getattr(tool, "parameters", None)
        else:
            # Dict fallback (for tests or legacy code)
            name = str(tool.get("name", ""))
            description = str(tool.get("description", ""))
            parameters = tool.get("parameters")

        wrapper = SDKToolWrapper(
            name=name,
            description=description,
            parameters=parameters,
            # Contract: deny-destroy:ToolSuppression:MUST:2, sdk-boundary:ToolForwarding:MUST:2
            # Set True so SDK allows user tool to override built-in with same name.
            # Without this, tools like "bash" cause "conflicts with built-in" error.
            overrides_built_in_tool=True,
            skip_permission=False,  # Amplifier handles permissions at kernel layer
        )
        result.append(wrapper)
    return result


# ============================================================================
# Image/Attachment Passthrough
# Contract: contracts/sdk-boundary.md § Image/Attachment Passthrough
# ============================================================================


def convert_image_block_to_blob_attachment(image_block: Any) -> dict[str, Any] | None:
    """Convert amplifier-core ImageBlock to SDK BlobAttachment.

    The provider acts as a pure transport layer for images. No capability
    validation, no filtering, no modification of image data.

    Args:
        image_block: An ImageBlock object with a `source` dict containing:
            - type: "base64" (only supported type)
            - media_type: MIME type (e.g., "image/png")
            - data: Base64-encoded image data

    Returns:
        A BlobAttachment dict for SDK session.send(), or None if:
        - source.type is not "base64" (URL images not supported)
        - data is empty or missing

    Contract Anchors:
        - sdk-boundary:ImagePassthrough:MUST:2 — Convert ImageBlock to BlobAttachment
        - sdk-boundary:ImagePassthrough:MUST:3 — Skip non-base64 images
        - sdk-boundary:ImagePassthrough:MUST:4 — Skip empty image data
        - sdk-boundary:ImagePassthrough:MUST:6 — No image content modification

    """
    # Get source dict from ImageBlock
    source: dict[str, Any] | None = getattr(image_block, "source", None)
    if not isinstance(source, dict):
        return None

    # Only base64 images are supported (not URL)
    if source.get("type") != "base64":
        return None

    # Get data - must be non-empty
    data: Any = source.get("data")
    if not data:
        return None

    # Build BlobAttachment (SDK format)
    # No modification of data - pure passthrough
    media_type: str = str(source.get("media_type", "image/png"))
    return {
        "type": "blob",
        "data": data,  # Pass through unchanged
        "mimeType": media_type,
    }


def extract_attachments_from_chat_request(request: Any) -> list[dict[str, Any]]:
    """Extract image attachments from ChatRequest for SDK.

    SDK constraint: Only images from the LAST user message are extracted.
    Historical images cannot be forwarded to SDK's session.send().

    The provider does NOT validate model vision capability. If a model doesn't
    support vision, the SDK or upstream will handle the error. We are pure
    transport layer.

    Args:
        request: A ChatRequest object with messages attribute.

    Returns:
        List of BlobAttachment dicts ready for SDK session.send(attachments=...).
        Empty list if no images found.

    Contract Anchors:
        - sdk-boundary:ImagePassthrough:MUST:1 — Extract from LAST user message only
        - sdk-boundary:ImagePassthrough:MUST:5 — No model capability validation
        - sdk-boundary:ImagePassthrough:MUST:7 — Forward attachments via send()
        - provider-protocol:complete:MUST:7 — Extracts images from last user message

    """
    attachments: list[dict[str, Any]] = []

    # Get messages from request
    messages: Any = getattr(request, "messages", None)
    if not messages:
        return []

    # Find LAST user message (SDK only supports current turn images)
    last_user_msg: Any = None
    historical_image_count: int = 0
    msg: Any
    for msg in reversed(list(messages)):
        if getattr(msg, "role", "") == "user":
            if last_user_msg is None:
                last_user_msg = msg
            else:
                # L-5: Count images in historical user messages (they will be dropped)
                hist_content = getattr(msg, "content", None)
                if isinstance(hist_content, list):
                    for blk in cast(list[Any], hist_content):
                        if getattr(blk, "type", None) == "image":
                            historical_image_count += 1

    if historical_image_count > 0:
        logger.warning(
            "[IMAGE_PASSTHROUGH] %d image(s) in historical messages silently dropped "
            "— SDK only supports images from the last user message.",
            historical_image_count,
        )

    if not last_user_msg:
        return []

    # Get content - must be a list for multimodal content
    content = getattr(last_user_msg, "content", None)
    if not isinstance(content, list):
        return []  # Plain text content, no images

    # Cast content to list[Any] for safe iteration over heterogeneous content blocks
    # isinstance(content, list) guard above proves content is a list; cast makes it explicit
    content_blocks: list[Any] = cast(list[Any], content)

    # Extract images from content blocks
    # Note: content is list of various block types (text, image, tool_result, etc.)
    for block in content_blocks:
        # Check if this is an image block
        if getattr(block, "type", None) == "image":
            attachment = convert_image_block_to_blob_attachment(block)
            if attachment:
                attachments.append(attachment)

    return attachments
