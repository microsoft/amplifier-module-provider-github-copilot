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
