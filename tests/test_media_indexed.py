"""Final-user selection conformance shared by query and actual conversion."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

import pytest
from amplifier_core import ChatRequest, ImageBlock, Message, TextBlock

from amplifier_module_provider_github_copilot._compat import ConfigurationError
from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
from amplifier_module_provider_github_copilot.request_adapter import convert_chat_request
from amplifier_module_provider_github_copilot.sdk_adapter import CopilotModelInfo


def cached_provider(vision: bool | None, model: str = "exact-model") -> GitHubCopilotProvider:
    provider = GitHubCopilotProvider()
    provider._copilot_models_cache = [] if vision is None else [CopilotModelInfo(
        id=model, name=model, context_window=10000, max_output_tokens=100,
        supports_vision=vision,
    )]
    return provider


def image(mime: str = "image/png") -> ImageBlock:
    return ImageBlock(source={"type": "base64", "media_type": mime, "data": "eA=="})


@pytest.mark.parametrize("vision", [None, False, True])
@pytest.mark.parametrize("role", ["user", "tool", "assistant", "system", "developer"])
@pytest.mark.parametrize("trailing", [None, "user", "tool", "assistant", "system", "developer"])
def test_query_and_converter_select_same_exact_final_position(
    vision: bool | None, role: str, trailing: str | None
) -> None:
    provider = cached_provider(vision)
    messages = [
        Message(role="user", content=[image()]),
        Message(role=role, tool_call_id="call-image" if role == "tool" else None,
                content=[TextBlock(text="before"), image(), TextBlock(text="after")]),
    ]
    if trailing is not None:
        # A final text-only list still owns its position, not the earlier image.
        messages.append(Message(
            role=trailing, content=[TextBlock(text="current")],
            tool_call_id="call-current" if trailing == "tool" else None,
        ))
    request = ChatRequest(model="exact-model", messages=messages)
    before = deepcopy(request.model_dump())
    support = provider.get_media_request_support_for_request(
        request=request, model="exact-model", request_options={"model": "exact-model"}
    )
    expected_index = len(messages) - 1 if vision is True and messages[-1].role == "user" else None
    assert support == {
        "v": 2,
        "positions": [] if expected_index is None else [
            {"message_index": expected_index, "role": "user"}
        ],
        "mime_types": ["image/png", "image/jpeg"],
    }
    assert json.loads(json.dumps(support)) == support
    converted = convert_chat_request(
        request, supports_vision=provider._selected_media_vision("exact-model")
    )
    should_send = expected_index == 1
    assert len(converted.attachments) == int(should_send)
    assert converted.prompt.count("Image unavailable") == (
        2 - int(should_send) - int(role == "system")
    )
    assert request.model_dump() == before
    assert provider.get_media_request_support(model="exact-model")["positions"] == []


@pytest.mark.parametrize("mime", ["image/png", "image/jpeg", "image/webp", "image/gif"])
def test_indexed_mime_facts_match_actual_converter(mime: str) -> None:
    provider = cached_provider(True)
    request = ChatRequest(model="exact-model", messages=[Message(
        role="user", content=[image(mime)]
    )])
    support = provider.get_media_request_support_for_request(request=request, model="exact-model")
    converted = convert_chat_request(request, supports_vision=True)
    assert bool(converted.attachments) is (mime in support["mime_types"])
    if mime not in support["mime_types"]:
        assert "does not qualify this image MIME type" in converted.prompt


def test_cache_exact_identity_not_model_prefix_and_no_request_mutation() -> None:
    provider = cached_provider(True, model="exact-model-other-route")
    request = ChatRequest(model="exact-model", messages=[Message(role="user", content=[image()])])
    assert provider.get_media_request_support_for_request(
        request=request, model="exact-model"
    )["positions"] == []
    assert provider._selected_media_vision("exact-model") is None


@pytest.mark.parametrize("options,model", [
    ({"model": "other"}, "exact-model"),
    ({"request_options": {"model": "other"}}, "exact-model"),
    ({}, "other"),
])
def test_conflicting_model_selection_refuses_not_empty_fallback(
    options: dict[str, Any], model: str
) -> None:
    request = ChatRequest(model="exact-model", messages=[Message(role="user", content=[image()])])
    with pytest.raises(ConfigurationError, match="differs from the fitted request"):
        cached_provider(True).get_media_request_support_for_request(
            request=request, model=model, request_options=options,
        )


def test_getter_return_has_no_alias_or_opaque_state() -> None:
    provider = cached_provider(True)
    request = ChatRequest(model="exact-model", messages=[Message(
        role="user", content=[image()], metadata={"opaque": "keep"}
    )])
    facts = provider.get_media_request_support_for_request(request=request, model="exact-model")
    facts["positions"].clear()
    facts["mime_types"].clear()
    assert provider.get_media_request_support_for_request(
        request=request, model="exact-model"
    ) == {
        "v": 2, "positions": [{"message_index": 0, "role": "user"}],
        "mime_types": ["image/png", "image/jpeg"],
    }


@pytest.mark.parametrize("untrusted", [1, "true", None])
def test_non_boolean_cached_vision_never_qualifies(untrusted: Any) -> None:
    from types import SimpleNamespace

    provider = cached_provider(None)
    malformed_cache: Any = [SimpleNamespace(id="exact-model", supports_vision=untrusted)]
    provider._copilot_models_cache = malformed_cache
    request = ChatRequest(model="exact-model", messages=[Message(role="user", content=[image()])])
    assert provider.get_media_request_support_for_request(
        request=request, model="exact-model"
    )["positions"] == []

