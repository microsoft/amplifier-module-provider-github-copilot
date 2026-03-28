"""SDK Adapter Layer - The Membrane.

All SDK imports are quarantined in _imports.py.
Domain code MUST NOT import from SDK directly.

Contract: contracts/sdk-boundary.md

Exports only domain types, not SDK-coupled internals.
Domain types:
- SessionConfig: Configuration for SDK session creation (domain wrapper)
- SDKSession: Opaque session type alias
- SessionHandle: Façade wrapping SDK session (P2-11 membrane boundary)

SDK Utilities (re-exported via membrane):
- get_copilot_spec_origin: Locate SDK package without importing it
"""

from ._spec_utils import get_copilot_spec_origin
from .client import CopilotClientWrapper
from .event_helpers import (
    extract_event_type,
    extract_tool_requests,
    extract_usage_data,
    has_tool_capture_event,
    is_assistant_message,
    is_error_event,
    is_idle_event,
    is_usage_event,
)
from .extract import extract_event_fields
from .model_translation import CopilotModelInfo, sdk_model_to_copilot_model
from .tool_capture import ToolCaptureHandler, normalize_tool_request
from .types import (
    CompletionConfig,
    CompletionRequest,
    SDKCreateFn,
    SDKSession,
    SessionConfig,
    SessionHandle,
    extract_attachments_from_chat_request,
)

# Export domain types only
# CopilotClientWrapper is exposed for provider.py but is an internal detail
# _imports.py contains the actual SDK imports (quarantined)
__all__ = [
    "CopilotClientWrapper",  # Internal: provider.py needs this
    "SessionConfig",  # Domain type
    "SDKSession",  # Domain type alias
    "SessionHandle",  # P2-11: Façade for SDK sessions (membrane boundary)
    "CompletionConfig",  # Domain type for completion settings
    "CompletionRequest",  # Domain type for completion input
    "SDKCreateFn",  # Type alias for SDK session factory
    "extract_event_fields",  # Unified SDK event extraction
    "extract_event_type",  # Event type helpers
    "is_idle_event",  # Event type helpers
    "is_error_event",  # Event type helpers
    "is_usage_event",  # Event type helpers (usage capture)
    "is_assistant_message",  # Event type helpers (tool capture)
    "has_tool_capture_event",  # Abort-on-capture helper
    "extract_tool_requests",  # Tool request extraction
    "extract_usage_data",  # Usage data extraction
    "ToolCaptureHandler",  # Extracted tool capture handler
    "normalize_tool_request",  # Tool request normalization
    "get_copilot_spec_origin",  # SDK utility: find SDK package path
    "extract_attachments_from_chat_request",  # Request attachment extraction
    "CopilotModelInfo",  # Domain type: SDK model → domain model
    "sdk_model_to_copilot_model",  # Translation: SDK ModelInfo → CopilotModelInfo
]
