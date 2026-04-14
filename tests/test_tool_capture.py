"""Tests for tool capture handler extraction.

TDD: These tests were written BEFORE the implementation to drive the design.
Contract: streaming-contract:ToolCapture:MUST:1
Contract: sdk-protection:ToolCapture:MUST:1,2
"""

from typing import Any


class TestToolCaptureHandler:
    """Test the extracted ToolCaptureHandler class."""

    def test_normalize_tool_request_from_dict(self) -> None:
        """Extract tool data from dict format (test events).

        Contract: streaming-contract:ToolCapture:MUST:1
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.tool_capture import (
            normalize_tool_request,
        )

        req = {
            "tool_call_id": "call_123",
            "name": "read_file",
            "arguments": {"path": "/tmp/test.txt"},
        }
        result = normalize_tool_request(req)

        assert result["id"] == "call_123"
        assert result["name"] == "read_file"
        assert result["arguments"] == {"path": "/tmp/test.txt"}

    def test_normalize_tool_request_from_dict_camel_case(self) -> None:
        """Extract tool data from dict with camelCase keys.

        Contract: streaming-contract:ToolCapture:MUST:1
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.tool_capture import (
            normalize_tool_request,
        )

        req = {
            "toolCallId": "call_456",
            "name": "write_file",
            "arguments": {"content": "hello"},
        }
        result = normalize_tool_request(req)

        assert result["id"] == "call_456"
        assert result["name"] == "write_file"

    def test_normalize_tool_request_from_object(self) -> None:
        """Extract tool data from SDK object format.

        Contract: streaming-contract:ToolCapture:MUST:1
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.tool_capture import (
            normalize_tool_request,
        )

        class MockToolRequest:
            tool_call_id = "call_789"
            name = "bash"
            arguments = {"command": "ls"}

        result = normalize_tool_request(MockToolRequest())

        assert result["id"] == "call_789"
        assert result["name"] == "bash"
        assert result["arguments"] == {"command": "ls"}

    def test_handler_captures_tools_on_assistant_message(self) -> None:
        """Handler extracts tools from ASSISTANT_MESSAGE event.

        Contract: streaming-contract:ToolCapture:MUST:1
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.tool_capture import (
            ToolCaptureHandler,
        )

        handler = ToolCaptureHandler()

        # Simulate ASSISTANT_MESSAGE with tool_requests
        event: dict[str, Any] = {
            "type": "assistant.message",
            "data": {
                "tool_requests": [
                    {"tool_call_id": "t1", "name": "read_file", "arguments": {}},
                    {"tool_call_id": "t2", "name": "write_file", "arguments": {}},
                ]
            },
        }

        handler.on_event(event)

        assert handler.capture_complete is True
        assert len(handler.captured_tools) == 2
        assert handler.captured_tools[0]["name"] == "read_file"
        assert handler.captured_tools[1]["name"] == "write_file"

    def test_handler_ignores_non_tool_events(self) -> None:
        """Handler ignores events without tool_requests.

        Contract: streaming-contract:ToolCapture:MUST:1
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.tool_capture import (
            ToolCaptureHandler,
        )

        handler = ToolCaptureHandler()

        # Text delta event - no tool_requests
        event = {"type": "content.delta", "data": {"text": "Hello"}}
        handler.on_event(event)

        assert handler.capture_complete is False
        assert len(handler.captured_tools) == 0

    def test_handler_captures_only_first_turn(self) -> None:
        """Handler ignores subsequent tool events after first capture.

        Contract: sdk-protection:ToolCapture:MUST:1
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.tool_capture import (
            ToolCaptureHandler,
        )

        handler = ToolCaptureHandler()

        # First ASSISTANT_MESSAGE
        event1: dict[str, Any] = {
            "type": "assistant.message",
            "data": {
                "tool_requests": [
                    {"tool_call_id": "t1", "name": "first_tool", "arguments": {}},
                ]
            },
        }
        handler.on_event(event1)

        # Second ASSISTANT_MESSAGE (should be ignored)
        event2: dict[str, Any] = {
            "type": "assistant.message",
            "data": {
                "tool_requests": [
                    {"tool_call_id": "t2", "name": "second_tool", "arguments": {}},
                ]
            },
        }
        handler.on_event(event2)

        assert len(handler.captured_tools) == 1
        assert handler.captured_tools[0]["name"] == "first_tool"

    def test_handler_sets_idle_callback(self) -> None:
        """Handler invokes idle callback on capture.

        Contract: sdk-protection:ToolCapture:MUST:1
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.tool_capture import (
            ToolCaptureHandler,
        )

        callback_called = [False]

        def on_capture() -> None:
            callback_called[0] = True

        handler = ToolCaptureHandler(on_capture_complete=on_capture)

        event: dict[str, Any] = {
            "type": "assistant.message",
            "data": {
                "tool_requests": [
                    {"tool_call_id": "t1", "name": "tool", "arguments": {}},
                ]
            },
        }
        handler.on_event(event)

        assert callback_called[0] is True


class TestToolCaptureDeduplication:
    """Test deduplication feature.

    Contract: sdk-protection:ToolCapture:MUST:2
    """

    def test_deduplication_filters_duplicate_tool_ids(self) -> None:
        """Handler deduplicates by tool_call_id.

        Contract: sdk-protection:ToolCapture:MUST:2
        """
        from amplifier_module_provider_github_copilot.config_loader import ToolCaptureConfig
        from amplifier_module_provider_github_copilot.sdk_adapter.tool_capture import (
            ToolCaptureHandler,
        )

        config = ToolCaptureConfig(
            first_turn_only=False,  # Disable first-turn-only to test dedup
            deduplicate=True,
            log_capture_events=False,
        )
        handler = ToolCaptureHandler(config=config)

        # First event with t1
        event1: dict[str, Any] = {
            "type": "assistant.message",
            "data": {
                "tool_requests": [
                    {"tool_call_id": "t1", "name": "read_file", "arguments": {}},
                ]
            },
        }
        handler.on_event(event1)

        # Second event with same t1 (should be filtered)
        event2: dict[str, Any] = {
            "type": "assistant.message",
            "data": {
                "tool_requests": [
                    {"tool_call_id": "t1", "name": "read_file", "arguments": {}},
                    {"tool_call_id": "t2", "name": "write_file", "arguments": {}},
                ]
            },
        }
        handler.on_event(event2)

        assert len(handler.captured_tools) == 2  # t1 and t2
        assert handler.captured_tools[0]["id"] == "t1"
        assert handler.captured_tools[1]["id"] == "t2"
        assert handler.deduplicated_count == 1  # One duplicate filtered

    def test_deduplication_disabled_allows_duplicates(self) -> None:
        """When deduplicate=False, duplicates are allowed.

        Contract: sdk-protection:ToolCapture:MUST:2 (config option)
        """
        from amplifier_module_provider_github_copilot.config_loader import ToolCaptureConfig
        from amplifier_module_provider_github_copilot.sdk_adapter.tool_capture import (
            ToolCaptureHandler,
        )

        config = ToolCaptureConfig(
            first_turn_only=False,
            deduplicate=False,  # Disable deduplication
            log_capture_events=False,
        )
        handler = ToolCaptureHandler(config=config)

        # Same tool_call_id twice
        event1: dict[str, Any] = {
            "type": "assistant.message",
            "data": {
                "tool_requests": [
                    {"tool_call_id": "t1", "name": "read_file", "arguments": {}},
                ]
            },
        }
        handler.on_event(event1)

        event2: dict[str, Any] = {
            "type": "assistant.message",
            "data": {
                "tool_requests": [
                    {"tool_call_id": "t1", "name": "read_file", "arguments": {}},
                ]
            },
        }
        handler.on_event(event2)

        # Both should be captured (no deduplication)
        assert len(handler.captured_tools) == 2

    def test_config_passed_to_handler(self) -> None:
        """Handler respects config settings.

        Contract: sdk-protection:ToolCapture:MUST:1 (first_turn_only config)
        """
        from amplifier_module_provider_github_copilot.config_loader import ToolCaptureConfig
        from amplifier_module_provider_github_copilot.sdk_adapter.tool_capture import (
            ToolCaptureHandler,
        )

        # Disable first_turn_only via config
        config = ToolCaptureConfig(
            first_turn_only=False,
            deduplicate=True,
            log_capture_events=False,
        )
        handler = ToolCaptureHandler(config=config)

        # First event
        event1: dict[str, Any] = {
            "type": "assistant.message",
            "data": {
                "tool_requests": [
                    {"tool_call_id": "t1", "name": "first", "arguments": {}},
                ]
            },
        }
        handler.on_event(event1)

        # Second event (should be captured since first_turn_only=False)
        event2: dict[str, Any] = {
            "type": "assistant.message",
            "data": {
                "tool_requests": [
                    {"tool_call_id": "t2", "name": "second", "arguments": {}},
                ]
            },
        }
        handler.on_event(event2)

        # Both should be captured
        assert len(handler.captured_tools) == 2
        assert handler.captured_tools[0]["name"] == "first"
        assert handler.captured_tools[1]["name"] == "second"


class TestToolCaptureDefaults:
    """Test default config behavior."""

    def test_handler_uses_defaults_when_no_config(self) -> None:
        """Handler uses safe defaults when config is None.

        Contract: sdk-protection:ToolCapture:MUST:1
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.tool_capture import (
            ToolCaptureHandler,
        )

        handler = ToolCaptureHandler()  # No config

        # First event
        event1: dict[str, Any] = {
            "type": "assistant.message",
            "data": {
                "tool_requests": [
                    {"tool_call_id": "t1", "name": "first", "arguments": {}},
                ]
            },
        }
        handler.on_event(event1)

        # Second event (should be ignored - defaults to first_turn_only=True)
        event2: dict[str, Any] = {
            "type": "assistant.message",
            "data": {
                "tool_requests": [
                    {"tool_call_id": "t2", "name": "second", "arguments": {}},
                ]
            },
        }
        handler.on_event(event2)

        # Only first should be captured (first_turn_only default)
        assert len(handler.captured_tools) == 1
        assert handler.captured_tools[0]["name"] == "first"
