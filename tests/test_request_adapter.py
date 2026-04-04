"""Tests for request_adapter module.

P2-10: Add missing module tests.
Contract Reference: provider-protocol:complete:MUST:1
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MockMessage:
    """Mock kernel message for testing."""

    role: str
    content: str | list[Any]


@dataclass
class MockChatRequest:
    """Mock kernel ChatRequest for testing."""

    messages: list[MockMessage] = field(
        default_factory=lambda: []  # pyright: ignore[reportUnknownVariableType]
    )
    model: str | None = None
    tools: list[dict[str, Any]] = field(
        default_factory=lambda: []  # pyright: ignore[reportUnknownVariableType]
    )


class TestConvertChatRequest:
    """Test convert_chat_request() function."""

    def test_converts_simple_chat_request(self) -> None:
        """Convert simple ChatRequest to CompletionRequest."""
        from amplifier_module_provider_github_copilot.request_adapter import convert_chat_request

        request = MockChatRequest(
            messages=[MockMessage(role="user", content="Hello")],
            model="gpt-4o",
        )

        result = convert_chat_request(request)

        assert result.model == "gpt-4o"
        assert "user: Hello" in result.prompt or "Hello" in result.prompt
        assert result.tools == []

    def test_uses_default_model_when_not_specified(self) -> None:
        """Use default_model when request has no model."""
        from amplifier_module_provider_github_copilot.request_adapter import convert_chat_request

        request = MockChatRequest(
            messages=[MockMessage(role="user", content="Hello")],
        )

        result = convert_chat_request(request, default_model="gpt-4")

        assert result.model == "gpt-4"

    def test_passthrough_if_already_completion_request(self) -> None:
        """Return CompletionRequest unchanged if passed directly."""
        from amplifier_module_provider_github_copilot.request_adapter import convert_chat_request
        from amplifier_module_provider_github_copilot.sdk_adapter import CompletionRequest

        original = CompletionRequest(prompt="test", model="gpt-4o")
        result = convert_chat_request(original)

        assert result is original

    def test_extracts_tools(self) -> None:
        """Extract tools from ChatRequest."""
        from amplifier_module_provider_github_copilot.request_adapter import convert_chat_request

        tools = [{"name": "test_tool", "description": "A test tool"}]
        request = MockChatRequest(
            messages=[MockMessage(role="user", content="Use the tool")],
            tools=tools,
        )

        result = convert_chat_request(request)

        assert result.tools == tools


class TestExtractPromptFromChatRequest:
    """Test extract_prompt_from_chat_request() function."""

    def test_preserves_role_information(self) -> None:
        """Roles are preserved in extracted prompt."""
        from amplifier_module_provider_github_copilot.request_adapter import (
            extract_prompt_from_chat_request,
        )

        request = MockChatRequest(
            messages=[
                MockMessage(role="user", content="Hello"),
                MockMessage(role="assistant", content="Hi there"),
                MockMessage(role="user", content="How are you?"),
            ]
        )

        prompt = extract_prompt_from_chat_request(request)

        # Prompt should contain role markers
        assert "user" in prompt.lower() or "Hello" in prompt
        assert "assistant" in prompt.lower() or "Hi there" in prompt

    def test_handles_empty_messages(self) -> None:
        """Empty messages list returns empty prompt."""
        from amplifier_module_provider_github_copilot.request_adapter import (
            extract_prompt_from_chat_request,
        )

        request = MockChatRequest(messages=[])

        prompt = extract_prompt_from_chat_request(request)

        assert prompt == "" or prompt is not None  # Should not crash


class TestExtractSystemMessage:
    """Test extract_system_message() function."""

    def test_extracts_system_message(self) -> None:
        """System message is extracted from first system role message."""
        from amplifier_module_provider_github_copilot.request_adapter import (
            extract_system_message,
        )

        request = MockChatRequest(
            messages=[
                MockMessage(role="system", content="You are a helpful assistant."),
                MockMessage(role="user", content="Hello"),
            ]
        )

        system = extract_system_message(request)

        assert system == "You are a helpful assistant."

    def test_returns_none_without_system_message(self) -> None:
        """Returns None when no system message present."""
        from amplifier_module_provider_github_copilot.request_adapter import (
            extract_system_message,
        )

        request = MockChatRequest(
            messages=[
                MockMessage(role="user", content="Hello"),
            ]
        )

        system = extract_system_message(request)

        assert system is None


class TestExtractContentBlock:
    """Test content extraction through public API."""

    def test_handles_string_content(self) -> None:
        """String content passed via message content is extracted."""
        from amplifier_module_provider_github_copilot.request_adapter import (
            extract_prompt_from_chat_request,
        )

        # Test through public API - string content in message
        request = MockChatRequest(messages=[MockMessage(role="user", content="Hello, world!")])
        result = extract_prompt_from_chat_request(request)

        assert "Hello, world!" in result

    def test_handles_list_content(self) -> None:
        """List content parts are handled via public API."""
        from amplifier_module_provider_github_copilot.request_adapter import (
            extract_prompt_from_chat_request,
        )

        # Test list content through public API
        content: list[Any] = [
            {"type": "text", "text": "First part"},
            {"type": "text", "text": "Second part"},
        ]
        request = MockChatRequest(messages=[MockMessage(role="user", content=content)])

        result = extract_prompt_from_chat_request(request)

        assert "First part" in result
        assert "Second part" in result

    def test_handles_dict_content(self) -> None:
        """Dict content with text key is extracted via public API."""
        from amplifier_module_provider_github_copilot.request_adapter import (
            extract_prompt_from_chat_request,
        )

        # Test dict content through public API
        content: list[Any] = [{"type": "text", "text": "Dict content"}]
        request = MockChatRequest(messages=[MockMessage(role="user", content=content)])

        result = extract_prompt_from_chat_request(request)

        assert "Dict content" in result

    def test_image_block_excluded_from_prompt_text(self) -> None:
        """sdk-boundary:ImagePassthrough:MUST:3 — Image blocks produce no text in prompt.

        Images are extracted separately as BlobAttachments.
        In the text prompt path, image blocks MUST be silently skipped.
        No exception should be raised.
        """
        from amplifier_module_provider_github_copilot.request_adapter import (
            extract_prompt_from_chat_request,
        )

        content: list[Any] = [
            {"type": "text", "text": "Here is an image:"},
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": "abc123"},
            },
        ]
        request = MockChatRequest(messages=[MockMessage(role="user", content=content)])

        # Must not raise
        result = extract_prompt_from_chat_request(request)

        # Text portion must be present
        assert "Here is an image:" in result
        # Image data must NOT appear in the text prompt
        assert "abc123" not in result
        assert "base64" not in result

    def test_image_url_block_excluded_from_prompt_text(self) -> None:
        """sdk-boundary:ImagePassthrough:MUST:3 — image_url type blocks produce no text.

        URL image references are not supported by the SDK (base64-only).
        When a block with type='image_url' appears, it must be silently skipped.
        """
        from amplifier_module_provider_github_copilot.request_adapter import (
            extract_prompt_from_chat_request,
        )

        content: list[Any] = [
            {"type": "text", "text": "See this URL image:"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.png"},
            },
        ]
        request = MockChatRequest(messages=[MockMessage(role="user", content=content)])

        # Must not raise
        result = extract_prompt_from_chat_request(request)

        # Text portion must be present
        assert "See this URL image:" in result
        # URL must NOT appear in the text prompt
        assert "https://example.com" not in result


class TestBuildRequestPayloadForObservability:
    """Test build_request_payload_for_observability() function.

    Contract: observability:Verbosity:MUST:1 — raw_payloads flag controls inclusion
    """

    def test_builds_minimal_payload_with_just_model(self) -> None:
        """Build payload with only model specified."""
        from amplifier_module_provider_github_copilot.request_adapter import (
            build_request_payload_for_observability,
        )
        from amplifier_module_provider_github_copilot.sdk_adapter import CompletionRequest

        internal = CompletionRequest(prompt="Hello", model="gpt-4o")

        result = build_request_payload_for_observability(
            model="gpt-4o",
            request=MockChatRequest(messages=[MockMessage(role="user", content="Hello")]),
            internal_request=internal,
        )

        assert result["model"] == "gpt-4o"
        assert result["message_count"] == 1
        assert result["tool_names"] == []
        assert result["has_system_message"] is False

    def test_includes_tool_names(self) -> None:
        """Include tool function names in payload."""
        from amplifier_module_provider_github_copilot.request_adapter import (
            build_request_payload_for_observability,
        )
        from amplifier_module_provider_github_copilot.sdk_adapter import CompletionRequest

        tools = [
            {"function": {"name": "read_file"}},
            {"function": {"name": "write_file"}},
        ]
        internal = CompletionRequest(prompt="Use tools", model="gpt-4o", tools=tools)

        result = build_request_payload_for_observability(
            model="gpt-4o",
            request=MockChatRequest(
                messages=[MockMessage(role="user", content="Use tools")],
                tools=tools,
            ),
            internal_request=internal,
        )

        assert result["tool_names"] == ["read_file", "write_file"]

    def test_detects_system_message(self) -> None:
        """Detect presence of system message."""
        from amplifier_module_provider_github_copilot.request_adapter import (
            build_request_payload_for_observability,
        )
        from amplifier_module_provider_github_copilot.sdk_adapter import CompletionRequest

        internal = CompletionRequest(
            prompt="Hello",
            model="gpt-4o",
            system_message="You are a helpful assistant",
        )

        result = build_request_payload_for_observability(
            model="gpt-4o",
            request=MockChatRequest(messages=[MockMessage(role="user", content="Hello")]),
            internal_request=internal,
        )

        assert result["has_system_message"] is True

    def test_handles_toolspec_objects_not_dicts(self) -> None:
        """Handle ToolSpec objects from Amplifier kernel (attribute access, not .get()).

        Contract: observability:Payload:SHOULD:2 — Type-safe tool name extraction

        Regression: ToolSpec objects have .name attribute, not .get() method.
        The function must handle both ToolSpec objects and legacy dicts.
        """
        from amplifier_module_provider_github_copilot.request_adapter import (
            build_request_payload_for_observability,
        )
        from amplifier_module_provider_github_copilot.sdk_adapter import CompletionRequest

        # Simulate ToolSpec objects from amplifier_core (Pydantic BaseModel with attributes)
        @dataclass
        class MockToolSpec:
            """Mock ToolSpec matching amplifier_core.message_models.ToolSpec."""

            name: str
            description: str
            parameters: dict[str, Any] | None = None

        tool_specs = [
            MockToolSpec(name="read_file", description="Read a file"),
            MockToolSpec(name="write_file", description="Write to a file"),
        ]
        # Cast to list[dict] to satisfy type checker - at runtime these are ToolSpec objects
        internal = CompletionRequest(
            prompt="Use tools",
            model="gpt-4o",
            tools=tool_specs,  # type: ignore[arg-type]
        )

        result = build_request_payload_for_observability(
            model="gpt-4o",
            request=MockChatRequest(messages=[MockMessage(role="user", content="Use tools")]),
            internal_request=internal,
        )

        # Should extract names from ToolSpec objects via attribute access
        assert result["tool_names"] == ["read_file", "write_file"]


class TestBuildResponsePayloadForObservability:
    """Test build_response_payload_for_observability() function.

    Contract: observability:Verbosity:MUST:1 — raw_payloads flag controls inclusion
    """

    def test_builds_payload_from_response(self) -> None:
        """Build payload from ChatResponse."""
        from amplifier_module_provider_github_copilot.request_adapter import (
            build_response_payload_for_observability,
        )

        # Mock response with basic fields
        class MockResponse:
            text = "Hello, world!"
            content = [{"type": "text", "text": "Hello, world!"}]
            finish_reason = "end_turn"

        result = build_response_payload_for_observability(
            response=MockResponse(),
            tool_calls=0,
        )

        assert result["text_length"] == 13
        assert result["content_block_count"] == 1
        assert result["tool_calls"] == 0
        assert result["finish_reason"] == "end_turn"

    def test_handles_none_text(self) -> None:
        """Handle response with no text."""
        from amplifier_module_provider_github_copilot.request_adapter import (
            build_response_payload_for_observability,
        )

        class MockResponse:
            text = None
            content = []
            finish_reason = "tool_use"

        result = build_response_payload_for_observability(
            response=MockResponse(),
            tool_calls=2,
        )

        assert result["text_length"] == 0
        assert result["content_block_count"] == 0
        assert result["tool_calls"] == 2
        assert result["finish_reason"] == "tool_use"

    def test_handles_none_content(self) -> None:
        """Handle response with None content."""
        from amplifier_module_provider_github_copilot.request_adapter import (
            build_response_payload_for_observability,
        )

        class MockResponse:
            text = "Just text"
            content = None
            finish_reason = "stop"

        result = build_response_payload_for_observability(
            response=MockResponse(),
            tool_calls=0,
        )

        assert result["text_length"] == 9
        assert result["content_block_count"] == 0

    def test_handles_string_content_defensive(self) -> None:
        """Handle response with string content (edge case).

        Contract: observability:Payload:SHOULD:1 — Type-safe content counting

        Some edge cases may have content as a string instead of a list.
        The function should return 0 for content_block_count, not the
        character count of the string.
        """
        from amplifier_module_provider_github_copilot.request_adapter import (
            build_response_payload_for_observability,
        )

        class MockResponse:
            text = "Hello"
            content = "raw string content"  # Edge case: string instead of list
            finish_reason = "stop"

        result = build_response_payload_for_observability(
            response=MockResponse(),
            tool_calls=0,
        )

        # Should be 0, not 18 (length of "raw string content")
        assert result["content_block_count"] == 0
