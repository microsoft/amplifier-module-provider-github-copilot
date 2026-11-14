"""Provider-owned media placement and diagnostic-copy policy; no storage imports.

The SDK attaches blobs to its current send, without canonical message indices.
This adapter maps only the final user message to that channel; that policy is
not proof the vendor cannot process historical images. Indexed host support and
the actual converter share that same selection; old role-wide hosts stay closed.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import fields, is_dataclass
from functools import wraps
from threading import Lock
from typing import Any

from amplifier_core.message_models import ImageBlock, TextBlock

MEDIA_NAMESPACE = "experimental.media.v1"
IMAGE_NOT_RETAINED = "[Image not retained in diagnostics; pixels omitted]"
SDK_IMAGE_MIME_TYPES = ("image/png", "image/jpeg")
IMAGE_ERROR_NOT_RETAINED = "[Image request error details not retained in diagnostics]"
_image_error_scope: ContextVar[bool] = ContextVar("copilot_image_error_scope", default=False)
_sdk_log_install_lock = Lock()


def _install_sdk_timing_error_projection() -> None:
    """Project the two pinned SDK timing error loggers before Logger._log.

    SDK 1.0.15 logs create/send rejection tracebacks before raising to the
    membrane. A handler/filter is too late. Adapt only these named logger
    instances' timing ``log`` entry, not global logging, SDK/RPC methods,
    verbosity, handlers or exception objects. Outside this task-local image
    scope the original call is forwarded verbatim.
    """
    with _sdk_log_install_lock:
        callers = {
            "copilot._jsonrpc": "JsonRpcClient.request JSON-RPC request finished",
            "copilot.client": "CopilotClient.create_session failed",
        }
        for name, timing_message in callers.items():
            sdk_logger = logging.getLogger(name)
            original = sdk_logger.log
            if getattr(original, "_copilot_image_error_projection", False):
                continue

            def project(original: Any, timing_message: str) -> Any:
                @wraps(original)
                def log(level: int, msg: object, *args: Any, **kwargs: Any) -> None:
                    if (
                        _image_error_scope.get()
                        and type(msg) is str and msg == timing_message
                        and kwargs.get("exc_info")
                    ):
                        # No raw exception, args, extra object or stack can
                        # cross into Logger._log/LogRecord construction.
                        original(level, IMAGE_ERROR_NOT_RETAINED)
                    else:
                        original(level, msg, *args, **kwargs)
                log._copilot_image_error_projection = True  # type: ignore[attr-defined]
                return log

            sdk_logger.log = project(original, timing_message)


@contextmanager
def image_error_scope(image_bearing: bool) -> Iterator[None]:
    """Task-local diagnostic policy, never a transport or retry-state mutation.

    SDK errors can echo arbitrary slices/encodings of known private pixels.
    Do not guess a base64 regex or keep a pixel registry for substring scanning.
    Only error diagnostics within a known image request use a bounded message.
    """
    if image_bearing:
        _install_sdk_timing_error_projection()
    token = _image_error_scope.set(image_bearing or _image_error_scope.get())
    try:
        yield
    finally:
        _image_error_scope.reset(token)


def diagnostic_error_message(error: object) -> str:
    """Project an error COPY before handing it to a queue, hook or logger."""
    if _image_error_scope.get():
        return IMAGE_ERROR_NOT_RETAINED
    from .security_redaction import redact_sensitive_text

    return redact_sensitive_text(error)


def _value(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def _set(value: Any, key: str, replacement: Any) -> None:
    if isinstance(value, dict):
        value[key] = replacement
    else:
        setattr(value, key, replacement)


def unavailable(reason: str, *, original: Any = None) -> Any:
    """Replace the exact image slot with text, preserving its owner visibility."""
    text = f"[Image unavailable: {reason}; no pixels supplied]"
    if isinstance(original, ImageBlock):
        return TextBlock(text=text, visibility=getattr(original, "visibility", None))
    replacement = {"type": "text", "text": text}
    if isinstance(original, dict) and "visibility" in original:
        replacement["visibility"] = original["visibility"]
    return replacement


def has_media(request: Any) -> bool:
    """Inspect only canonical structured message carriers."""
    def contains_image(content: Any) -> bool:
        if isinstance(content, list):
            return any(contains_image(block) for block in content)
        return _value(content, "type") == "image" or (
            _value(content, "type") == "tool_result"
            and contains_image(_value(content, "output"))
        )

    for message in _value(request, "messages", []) or []:
        if MEDIA_NAMESPACE in (_value(message, "metadata", {}) or {}):
            return True
        if contains_image(_value(message, "content")):
            return True
    return False


def final_user_image_position(request: Any, *, supports_vision: bool | None) -> int | None:
    """Select a message, never search backwards for an image-bearing turn.

    Query input may be refs-only while transport input is privately hydrated.
    The selection depends on role/order/list shape, not the presence of pixels.
    """
    messages = _value(request, "messages")
    if supports_vision is not True or not isinstance(messages, list) or not messages:
        return None
    index = len(messages) - 1
    message = messages[index]
    if _value(message, "role") != "user" or not isinstance(_value(message, "content"), list):
        return None
    return index


def project_sdk_images(request: Any, *, supports_vision: bool | None) -> Any:
    """Copy a Core request, preserving roles, IDs, metadata and block ordering.

    Only the shared selected final user message can use the SDK attachment
    channel. No history/tool pixels are relocated into that message.
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
    selected = final_user_image_position(projected, supports_vision=supports_vision)

    def blocks(content: Any, *, reason: str | None) -> Any:
        if isinstance(content, list):
            return [blocks(block, reason=reason) for block in content]
        kind = _value(content, "type")
        if kind == "image":
            source = _value(content, "source", {}) or {}
            if reason is None:
                data = _value(source, "data")
                if _value(source, "media_type", "image/png") not in SDK_IMAGE_MIME_TYPES:
                    reason = "Copilot adapter does not qualify this image MIME type"
                elif _value(source, "type") == "base64" and isinstance(data, str) and data:
                    return content
            return unavailable(
                reason or "Copilot SDK requires a nonempty base64 image source",
                original=content,
            )
        if kind == "tool_result":
            # Only Core structured output, never JSON-looking text.
            output = _value(content, "output")
            if isinstance(output, (list, dict)):
                _set(content, "output", blocks(
                    output, reason="Copilot adapter does not support tool-result images"
                ))
        return content

    for index, message in enumerate(messages):
        content = _value(message, "content")
        role = _value(message, "role")
        if role != "user":
            reason = f"Copilot adapter does not support images at the {role} message position"
        elif index != len(messages) - 1:
            reason = "Copilot adapter does not support historical user images"
        elif supports_vision is None:
            reason = "selected Copilot model vision capability is unknown"
        elif supports_vision is not True:
            reason = "selected Copilot model does not support vision"
        elif not isinstance(content, list):
            reason = "Copilot adapter requires image content in a block list"
        elif index == selected:
            reason = None
        else:
            reason = "Copilot adapter does not qualify this image position"
        _set(message, "content", blocks(content, reason=reason))
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
