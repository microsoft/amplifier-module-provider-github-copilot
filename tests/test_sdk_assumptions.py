"""Tier 6: SDK Assumption Tests - verify SDK types and shapes without API calls.

These tests import the real SDK, instantiate objects, and verify our structural
assumptions. They require the SDK to be installed but do NOT make API calls.

Contract references:
- contracts/sdk-boundary.md
- contracts/deny-destroy.md

Run: pytest -m sdk_assumption -v
"""

from __future__ import annotations

from typing import Any

import pytest


@pytest.mark.sdk_assumption
class TestSDKImportAssumptions:
    """Verify SDK module structure matches our assumptions.

    AC-1: SDK Import Assumptions
    """

    def test_copilot_module_importable(self, sdk_module: Any) -> None:
        """We assume the copilot module is importable."""
        assert sdk_module is not None

    def test_copilot_client_class_exists(self, sdk_module: Any) -> None:
        """We assume copilot.CopilotClient exists and is importable.

        sdk-boundary:Translation:MUST:1
        """
        assert hasattr(sdk_module, "CopilotClient")
        assert sdk_module.CopilotClient is not None

    def test_client_has_create_session(self, sdk_module: Any) -> None:
        """We assume CopilotClient has create_session method.

        sdk-boundary:Session:MUST:1
        """
        assert hasattr(sdk_module.CopilotClient, "create_session")

    def test_client_has_start_stop(self, sdk_module: Any) -> None:
        """We assume CopilotClient has start() and stop() lifecycle methods.

        sdk-boundary:Lifecycle:MUST:1
        """
        assert hasattr(sdk_module.CopilotClient, "start")
        assert hasattr(sdk_module.CopilotClient, "stop")


@pytest.mark.sdk_assumption
class TestSessionInterfaceAssumptions:
    """Verify session-related assumptions without API calls.

    AC-2: Session Lifecycle Assumptions (Tier 6 portion)

    Note: Session OBJECT interface (disconnect, send_message, register_pre_tool_use_hook)
    cannot be verified without creating a real session, which requires credentials.
    Those checks are in test_live_sdk.py (Tier 7).

    Here we verify what CAN be checked without credentials:
    - Client has create_session method
    - Our wrapper has the expected interface
    """

    def test_client_has_create_session_method(self, sdk_module: Any) -> None:
        """CopilotClient must have create_session method.

        sdk-boundary:Session:MUST:1
        """
        assert hasattr(sdk_module.CopilotClient, "create_session")
        # Note: Session object interface (disconnect, send_message) verified in Tier 7


@pytest.mark.sdk_assumption
class TestOurWrapperImports:
    """Verify our wrapper code imports work correctly."""

    def test_copilot_client_wrapper_importable(self) -> None:
        """Our CopilotClientWrapper should be importable."""
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        assert CopilotClientWrapper is not None

    def test_copilot_client_wrapper_has_session_method(self) -> None:
        """CopilotClientWrapper should have session() context manager."""
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        wrapper = CopilotClientWrapper()
        assert hasattr(wrapper, "session")
        assert hasattr(wrapper, "close")


@pytest.mark.sdk_assumption
class TestHelperFunctions:
    """Test our SDK helper functions work correctly."""

    def test_get_event_type_from_dict(self, mock_sdk_event_dict: dict[str, Any]) -> None:
        """get_event_type should extract type from dict."""
        from tests.sdk_helpers import get_event_type

        result = get_event_type(mock_sdk_event_dict)
        assert result == "assistant.message_delta"

    def test_get_event_type_from_object(self, mock_sdk_event_object: Any) -> None:
        """get_event_type should extract type from object."""
        from tests.sdk_helpers import get_event_type

        result = get_event_type(mock_sdk_event_object)
        assert result == "assistant.message_delta"

    def test_get_event_field_from_dict(self, mock_sdk_event_dict: dict[str, Any]) -> None:
        """get_event_field should extract field from nested data dict.

        Contract: sdk-boundary:EventShape:MUST:2
        SDK v0.1.33+ uses event.data.delta_content, not event.text.
        """
        from tests.sdk_helpers import get_event_field

        result = get_event_field(mock_sdk_event_dict, "delta_content")
        assert result == "hello"

    def test_get_event_field_from_object(self, mock_sdk_event_object: Any) -> None:
        """get_event_field should extract field from nested data object.

        Contract: sdk-boundary:EventShape:MUST:2
        SDK v0.1.33+ uses event.data.delta_content, not event.text.
        """
        from tests.sdk_helpers import get_event_field

        result = get_event_field(mock_sdk_event_object, "delta_content")
        assert result == "hello"

    def test_describe_event_dict(self, mock_sdk_event_dict: dict[str, Any]) -> None:
        """describe_event should produce readable string from dict."""
        from tests.sdk_helpers import describe_event

        result = describe_event(mock_sdk_event_dict)
        assert "assistant.message_delta" in result

    def test_describe_event_object(self, mock_sdk_event_object: Any) -> None:
        """describe_event should produce readable string from object."""
        from tests.sdk_helpers import describe_event

        result = describe_event(mock_sdk_event_object)
        assert "MockEvent" in result

    def test_collect_event_types(self) -> None:
        """collect_event_types should return list of type strings."""
        from tests.sdk_helpers import collect_event_types

        events = [
            {"type": "assistant.message_delta"},
            {"type": "assistant.turn_end"},
        ]
        result = collect_event_types(events)
        assert result == ["assistant.message_delta", "assistant.turn_end"]

    def test_has_event_type_true(self) -> None:
        """has_event_type should return True when event type exists."""
        from tests.sdk_helpers import has_event_type

        events = [{"type": "assistant.message_delta"}, {"type": "assistant.turn_end"}]
        assert has_event_type(events, "assistant.message_delta")

    def test_has_event_type_false(self) -> None:
        """has_event_type should return False when event type missing."""
        from tests.sdk_helpers import has_event_type

        events = [{"type": "assistant.message_delta"}]
        assert not has_event_type(events, "tool_result")

    def test_count_event_type(self) -> None:
        """count_event_type should count occurrences."""
        from tests.sdk_helpers import count_event_type

        events = [
            {"type": "assistant.message_delta"},
            {"type": "assistant.message_delta"},
            {"type": "assistant.turn_end"},
        ]
        assert count_event_type(events, "assistant.message_delta") == 2
        assert count_event_type(events, "assistant.turn_end") == 1
        assert count_event_type(events, "tool_result") == 0
