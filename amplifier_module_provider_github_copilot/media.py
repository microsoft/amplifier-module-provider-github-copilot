"""Provider-owned media placement and diagnostic-copy policy; no storage imports.

The SDK has only a last-user attachment channel, not role-bound tool/history
images. Host media-v1 hydration remains unqualified for this provider.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, is_dataclass
from typing import Any

MEDIA_NAMESPACE = "experimental.media.v1"
IMAGE_NOT_RETAINED = "[Image not retained in diagnostics; pixels omitted]"


def _value(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def _set(value: Any, key: str, replacement: Any) -> None:
    if isinstance(value, dict):
        value[key] = replacement
    else:
        setattr(value, key, replacement)


def unavailable(reason: str) -> dict[str, str]:
    """A text block, not an image/reference that implies successful delivery."""
    return {"type": "text", "text": f"[Image unavailable: {reason}; no pixels supplied]"}


def has_media(request: Any) -> bool:
    """Inspect only canonical structured message carriers."""
    for message in _value(request, "messages", []) or []:
        if MEDIA_NAMESPACE in (_value(message, "metadata", {}) or {}):
            return True
        content = _value(message, "content")
        for block in content if isinstance(content, list) else [content]:
            if _value(block, "type") == "image":
                return True
    return False


def project_sdk_images(request: Any, *, supports_vision: bool | None) -> Any:
    """Copy a Core request, preserving roles, IDs, metadata and block ordering.

    Only an explicitly vision-capable model's last user message can use the
    legacy SDK channel. This is not qualification of durable media hydration.
    """
    def contains_image(content: Any) -> bool:
        if isinstance(content, list):
            return any(contains_image(block) for block in content)
        kind = _value(content, "type")
        return kind == "image" or (
            kind == "tool_result" and contains_image(_value(content, "output"))
        )

    # Text-only requests need no copy or model reconstruction. In particular,
    # never invoke arbitrary caller __deepcopy__ hooks on the ordinary path.
    if not any(
        contains_image(_value(message, "content"))
        for message in _value(request, "messages", []) or []
    ):
        return request
    projected = deepcopy(request)
    messages = _value(projected, "messages", []) or []
    last_user = next(
        (index for index in range(len(messages) - 1, -1, -1)
         if _value(messages[index], "role") == "user"),
        None,
    )

    def blocks(content: Any, *, allowed: bool) -> Any:
        if isinstance(content, list):
            return [blocks(block, allowed=allowed) for block in content]
        kind = _value(content, "type")
        if kind == "image":
            source = _value(content, "source", {}) or {}
            if allowed and _value(source, "type") == "base64" and _value(source, "data"):
                return content
            if not allowed:
                reason = (
                    "selected Copilot model vision capability is unknown"
                    if supports_vision is None
                    else "selected Copilot model does not support vision"
                    if supports_vision is not True
                    else "Copilot SDK does not support this message position"
                )
            else:
                reason = "Copilot SDK requires a nonempty base64 image source"
            return unavailable(reason)
        if kind == "tool_result":
            # Only Core structured output, never JSON-looking text.
            output = _value(content, "output")
            if isinstance(output, (list, dict)):
                _set(content, "output", blocks(output, allowed=False))
        return content

    for index, message in enumerate(messages):
        content = _value(message, "content")
        _set(
            message,
            "content",
            blocks(
                content,
                allowed=(
                    index == last_user == len(messages) - 1 and supports_vision is True
                    and isinstance(content, list)
                ),
            ),
        )
    return projected


def diagnostic_copy(value: Any) -> Any:
    """Remove known image transports from a COPY before hooks serialize data.

    No hydration, ingestion, JSON-string parsing, digest-as-ref or claim of
    retention. Opaque text and unrelated owner metadata are not interpreted.
    """
    if _value(value, "type") in {"image", "image_url", "input_image"}:
        return {"type": "text", "text": IMAGE_NOT_RETAINED}
    if isinstance(value, dict):
        kind = value.get("type")
        mime = value.get("mimeType") or value.get("mime_type") or value.get("media_type")
        if (
            kind in {"image", "image_url", "input_image"}
            or (kind == "blob" and isinstance(mime, str) and mime.startswith("image/"))
            or (kind == "base64" and isinstance(mime, str) and mime.startswith("image/"))
        ):
            return {"type": "text", "text": IMAGE_NOT_RETAINED}
        result: dict[str, Any] = {}
        for key, child in value.items():
            if key in {"inline_data", "inlineData"} and isinstance(child, dict):
                child_mime = child.get("mime_type") or child.get("mimeType")
                if isinstance(child_mime, str) and child_mime.startswith("image/"):
                    result[key] = {"not_retained": IMAGE_NOT_RETAINED}
                    continue
            result[key] = diagnostic_copy(child)
        return result
    if isinstance(value, (list, tuple)):
        return [diagnostic_copy(child) for child in value]
    if isinstance(value, str) and value.lower().startswith("data:image/"):
        return IMAGE_NOT_RETAINED
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return diagnostic_copy({field.name: getattr(value, field.name) for field in fields(value)})
    # Core/Pydantic structured values must not survive as live object aliases.
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump) and isinstance(getattr(type(value), "model_fields", None), dict):
        return diagnostic_copy(model_dump())
    return "[Diagnostic value not retained: unsupported object]"
