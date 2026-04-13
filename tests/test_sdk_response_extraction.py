"""Tests for SDK response extraction.

Contract: streaming-contract.md, sdk-response.md

The bug: SDK returns Data(content="actual text") but code does str(Data(...))
which produces repr dump instead of extracting .content attribute.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# Import realistic fixtures from the fixtures module
from tests.fixtures.sdk_responses import (
    MockData,
    MockSDKResponse,
)


class TestSDKResponseExtraction:
    """AC-1: Fix SDK response extraction to handle Data.content attribute."""

    def test_extracts_content_from_data_object(self) -> None:
        """MUST extract .content from Data objects, not str() them.

        Contract: sdk-response.md

        This is the critical bug: provider was doing str(Data(...)) which
        produces "Data(content='hello', role='assistant', ...)" instead of "hello".
        """
        from amplifier_module_provider_github_copilot.provider import (
            extract_response_content,
        )

        data = MockData(content="Hello, world!")
        result = extract_response_content(data)

        # MUST be plain text, NOT repr dump
        assert result == "Hello, world!"
        assert "Data(" not in result  # No dataclass repr
        assert "content=" not in result  # No field name in output

    def test_extracts_content_from_response_wrapper(self) -> None:
        """MUST handle response.data -> Data.content path."""
        from amplifier_module_provider_github_copilot.provider import (
            extract_response_content,
        )

        response = MockSDKResponse(data=MockData(content="Nested content"))
        result = extract_response_content(response)

        assert result == "Nested content"

    def test_handles_dict_response(self) -> None:
        """MUST still handle dict responses (backward compat)."""
        from amplifier_module_provider_github_copilot.provider import (
            extract_response_content,
        )

        response: dict[str, Any] = {"content": "Dict content"}
        result = extract_response_content(response)

        assert result == "Dict content"

    def test_handles_nested_dict_in_data(self) -> None:
        """MUST handle response.data as dict."""
        from amplifier_module_provider_github_copilot.provider import (
            extract_response_content,
        )

        response = MockSDKResponse(data={"content": "Nested dict content"})
        result = extract_response_content(response)

        assert result == "Nested dict content"

    def test_handles_empty_content(self) -> None:
        """MUST handle empty content gracefully."""
        from amplifier_module_provider_github_copilot.provider import (
            extract_response_content,
        )

        data = MockData(content="")
        result = extract_response_content(data)

        assert result == ""

    def test_handles_none_response(self) -> None:
        """MUST handle None response gracefully."""
        from amplifier_module_provider_github_copilot.provider import (
            extract_response_content,
        )

        result = extract_response_content(None)

        assert result == ""

    def test_handles_data_none(self) -> None:
        """MUST handle response.data = None gracefully."""
        from amplifier_module_provider_github_copilot.provider import (
            extract_response_content,
        )

        response = MockSDKResponse(data=None)
        result = extract_response_content(response)

        assert result == ""

    def test_handles_content_none(self) -> None:
        """MUST handle .content = None gracefully."""
        from amplifier_module_provider_github_copilot.provider import (
            extract_response_content,
        )

        # Create mock with content=None but NO .data attribute
        # MagicMock auto-creates attributes, causing infinite recursion
        # Use spec to restrict available attributes
        mock_data = MagicMock(spec=["content"])
        mock_data.content = None
        result = extract_response_content(mock_data)

        assert result == ""

    def test_object_with_both_data_and_content_prefers_data(self) -> None:
        """MUST prefer .data over .content when both present (unwrap first)."""
        from amplifier_module_provider_github_copilot.provider import (
            extract_response_content,
        )

        # Object with both .data and .content
        mock_response = MagicMock()
        mock_response.data = MockData(content="from data")
        mock_response.content = "from content"

        result = extract_response_content(mock_response)

        # Should get content from .data path, not direct .content
        assert result == "from data"


class TestE2ECompletionWithRealisticData:
    """AC-2: E2E test with realistic SDK response shapes."""

    @pytest.mark.asyncio
    async def test_complete_returns_text_not_repr(self) -> None:
        """complete() MUST return plain text content, not Data repr.

        This is the E2E test that would have caught the original bug.
        """
        from amplifier_module_provider_github_copilot.provider import (
            CompletionRequest,
            GitHubCopilotProvider,
        )

        # Create async iterator that yields streaming events
        class AsyncEventIterator:  # noqa: B903  # pyright: ignore[reportUnusedClass]
            def __init__(self, events: list[dict[str, Any]]):
                self._events = events
                self._index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self._index >= len(self._events):
                    raise StopAsyncIteration
                event_data = self._events[self._index]
                self._index += 1
                # Create event object with dict-like access
                event = MagicMock()
                for k, v in event_data.items():
                    setattr(event, k, v)
                event.get = lambda k, d=None, e=event_data: e.get(k, d)  # type: ignore[assignment]
                event.__dict__.update(event_data)
                return event

        events = [
            {"type": "assistant.message_delta", "text": "This is the actual response text."},
            {"type": "assistant.turn_end", "finish_reason": "stop"},
        ]

        @asynccontextmanager
        async def session_ctx(
            model: str,
            tools: list[dict[str, Any]] | None = None,
            system_message: str | None = None,
        ):
            # Use correct SDK API pattern (send + on)
            mock_session = MagicMock()
            handlers: list[Any] = []

            def mock_on(handler: Any) -> MagicMock:
                handlers.append(handler)
                return MagicMock()  # unsubscribe

            mock_session.on = MagicMock(side_effect=mock_on)

            async def mock_send(prompt: str, attachments: list[Any] | None = None) -> str:
                # Deliver events via handler callback
                for event_data in events:
                    event = MagicMock()
                    for k, v in event_data.items():
                        setattr(event, k, v)
                    event.__dict__.update(event_data)
                    event.get = lambda key, default=None, e=event_data: e.get(key, default)  # type: ignore[assignment]
                    for handler in handlers:
                        handler(event)
                # Signal completion
                idle_event = MagicMock()
                idle_event.type = "SESSION_IDLE"
                idle_event.__dict__["type"] = "SESSION_IDLE"
                for handler in handlers:
                    handler(idle_event)
                return "message-id"

            mock_session.send = AsyncMock(side_effect=mock_send)
            mock_session.disconnect = AsyncMock()
            yield mock_session

        # Create provider with mock client
        provider = GitHubCopilotProvider()
        provider._client.session = session_ctx  # type: ignore[method-assign]

        # Execute completion
        request = CompletionRequest(prompt="Hello")
        response = await provider.complete(request)  # type: ignore[arg-type]

        # MUST contain plain text, not repr dump
        content_text = ""
        for block in response.content:
            text = getattr(block, "text", None)
            if text is not None:
                content_text += str(text)

        assert content_text == "This is the actual response text."
        assert "Data(" not in content_text
        assert "content=" not in content_text


class TestToolArgumentExtraction:
    """Test that tool arguments are correctly extracted, including empty dicts.

    Contract: streaming-contract.md — tools MUST have arguments field.

    Regression prevention: `if arguments` was dropping empty dict {} because
    `bool({}) == False`. Zero-parameter tools would lose their arguments field.
    """

    def test_extract_event_fields_preserves_empty_arguments(self) -> None:
        """extract_event_fields MUST preserve empty dict arguments.

        Zero-parameter tools (like `list_files()`) have `arguments={}`.
        These MUST NOT be dropped.
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.extract import (
            extract_event_fields,
        )

        # Mock SDK event with empty arguments (zero-parameter tool)
        class MockToolData:
            tool_call_id = "call_123"
            tool_name = "list_files"
            arguments: dict[str, Any] = {}  # Empty dict — zero-parameter tool

        class MockToolEvent:
            type = "tool.call"
            data = MockToolData()

        result = extract_event_fields(MockToolEvent())

        # arguments MUST be present even when empty
        assert "arguments" in result, (
            "Empty arguments dict was dropped! "
            "Fix extract.py: change 'if arguments' to 'if arguments is not None'"
        )
        assert result["arguments"] == {}

    def test_extract_event_fields_preserves_populated_arguments(self) -> None:
        """extract_event_fields MUST preserve populated arguments."""
        from amplifier_module_provider_github_copilot.sdk_adapter.extract import (
            extract_event_fields,
        )

        class MockToolData:
            tool_call_id = "call_456"
            tool_name = "read_file"
            arguments = {"path": "/etc/passwd", "encoding": "utf-8"}

        class MockToolEvent:
            type = "tool.call"
            data = MockToolData()

        result = extract_event_fields(MockToolEvent())

        assert "arguments" in result
        assert result["arguments"] == {"path": "/etc/passwd", "encoding": "utf-8"}

    def test_extract_event_fields_handles_missing_arguments(self) -> None:
        """extract_event_fields handles events with no arguments attribute."""
        from amplifier_module_provider_github_copilot.sdk_adapter.extract import (
            extract_event_fields,
        )

        class MockData:
            delta_content = "hello"  # SDK uses delta_content, not text

        class MockEvent:
            type = "assistant.message_delta"
            data = MockData()

        result = extract_event_fields(MockEvent())

        # Non-tool events shouldn't have arguments but should have text
        assert "arguments" not in result
        assert result.get("text") == "hello"


class TestReasoningOpaqueExtraction:
    """streaming-contract:ThinkingBlock:MUST:1

    Test extraction of reasoning_opaque for ThinkingBlock.signature.
    """

    def test_extract_event_fields_preserves_reasoning_opaque(self) -> None:
        """extract_event_fields MUST preserve reasoning_opaque from SDK events.

        Contract: streaming-contract:ThinkingBlock:MUST:1

        Anthropic models send encrypted extended thinking data in `reasoning_opaque`.
        This MUST be extracted and later mapped to ThinkingBlock.signature.
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.extract import (
            extract_event_fields,
        )

        # Mock SDK event with reasoning_opaque (Claude extended thinking)
        class MockReasoningData:
            reasoning_text = "Let me think about this..."
            reasoning_opaque = "encrypted_signature_abc123xyz"  # Opaque token for multi-turn

        class MockReasoningEvent:
            type = "assistant.reasoning"
            data = MockReasoningData()

        result = extract_event_fields(MockReasoningEvent())

        # reasoning_opaque MUST be extracted
        assert "reasoning_opaque" in result, (
            "reasoning_opaque was not extracted! "
            "This breaks multi-turn extended thinking with Anthropic models."
        )
        assert result["reasoning_opaque"] == "encrypted_signature_abc123xyz"
        # reasoning_text should also be extracted
        assert result.get("reasoning_text") == "Let me think about this..."

    def test_extract_event_fields_handles_missing_reasoning_opaque(self) -> None:
        """extract_event_fields handles events without reasoning_opaque."""
        from amplifier_module_provider_github_copilot.sdk_adapter.extract import (
            extract_event_fields,
        )

        # Non-Claude or non-extended-thinking event
        class MockData:
            reasoning_text = "Simple reasoning"  # No opaque token

        class MockEvent:
            type = "assistant.reasoning"
            data = MockData()

        result = extract_event_fields(MockEvent())

        # Should handle gracefully - no error, just no reasoning_opaque key
        assert "reasoning_opaque" not in result or result.get("reasoning_opaque") is None
        assert result.get("reasoning_text") == "Simple reasoning"


class TestToolCallIdAndNameFallback:
    """Fallback extraction for tool_call_id and tool_name attributes.

    Lines ~117-125 in sdk_adapter/extract.py — SDK events that store
    tool identity in 'tool_call_id'/'tool_name' rather than 'id'/'name'.
    Contract: sdk-boundary:Membrane:MUST:1 — SDK type isolation.
    """

    def test_extract_event_fields_uses_tool_call_id_when_id_absent(self) -> None:
        """If event has no 'id' but has 'tool_call_id', use tool_call_id as id.

        Line ~119-120 in sdk_adapter/extract.py.
        SDK tool result events may use 'tool_call_id' not 'id'.
        Contract: sdk-boundary:Membrane:MUST:1
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.extract import (
            extract_event_fields,
        )

        class MockToolResultEvent:
            """Mimics a SDK tool_result event with tool_call_id, not id."""

            type = "tool_result"
            tool_call_id = "tc-fallback-99"
            # Deliberately no 'id' attribute

        result = extract_event_fields(MockToolResultEvent())

        assert result.get("id") == "tc-fallback-99", (
            f"tool_call_id must populate 'id' when 'id' absent. Got: {result.get('id')!r}"
        )

    def test_extract_event_fields_uses_tool_name_when_name_absent(self) -> None:
        """If event has no 'name' but has 'tool_name', use tool_name as name.

        Line ~124-125 in sdk_adapter/extract.py.
        SDK tool events may use 'tool_name' not 'name'.
        Contract: sdk-boundary:Membrane:MUST:1
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.extract import (
            extract_event_fields,
        )

        class MockToolNameEvent:
            """Mimics a SDK event with tool_name, not name."""

            type = "tool_use_complete"
            tool_name = "read_file_fallback"
            # Deliberately no 'name' attribute

        result = extract_event_fields(MockToolNameEvent())

        assert result.get("name") == "read_file_fallback", (
            f"tool_name must populate 'name' when 'name' absent. Got: {result.get('name')!r}"
        )
