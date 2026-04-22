"""Test fixtures for provider-github-copilot tests.

Exports SDK mock classes and response fixtures.
"""

from .sdk_mocks import (
    MockSDKSession,
    MockSDKSessionWithAbort,
    MockSDKSessionWithError,
    SessionEvent,
    SessionEventData,
    SessionEventType,
    error_event,
    idle_event,
    text_delta_event,
    usage_event,
)
from .sdk_responses import (
    MockData,
    MockSDKResponse,
    MockToolCall,
    MockUsage,
)

__all__ = [
    # SDK mocks
    "MockSDKSession",
    "MockSDKSessionWithAbort",
    "MockSDKSessionWithError",
    "SessionEvent",
    "SessionEventData",
    "SessionEventType",
    "idle_event",
    "text_delta_event",
    "error_event",
    "usage_event",
    # SDK response fixtures
    "MockData",
    "MockSDKResponse",
    "MockToolCall",
    "MockUsage",
]
