"""Tool capture handler for abort-on-capture pattern.

This module provides the ToolCaptureHandler class for extracting tool
requests from SDK ASSISTANT_MESSAGE events.

Contract: sdk-protection:ToolCapture:MUST:1,2
Contract: streaming-contract:abort-on-capture:MUST:1

Pattern: First-turn capture only with deduplication
- SDK sends ASSISTANT_MESSAGE with tool_requests when model wants tools
- Capture tools from first message only (prevents accumulation)
- Deduplicate by tool_call_id (prevents duplicate execution)
- Set idle flag to break wait loop and return to Amplifier for execution
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from .event_helpers import extract_tool_requests, has_tool_capture_event

if TYPE_CHECKING:
    from ..config_loader import ToolCaptureConfig

logger = logging.getLogger(__name__)


def normalize_tool_request(req: Any) -> dict[str, Any]:
    """Normalize a tool request from dict or object format.

    SDK tool_requests can be either:
    - Dict with snake_case or camelCase keys (test events)
    - Object with attributes (real SDK events)

    Args:
        req: Tool request in dict or object format

    Returns:
        Normalized dict with id, name, arguments keys

    """
    tool_id: str
    name: str
    arguments: dict[str, Any]

    if isinstance(req, dict):
        # Dict format - handle both snake_case and camelCase
        req_dict = cast(dict[str, Any], req)
        tool_id = str(req_dict.get("tool_call_id") or req_dict.get("toolCallId", ""))
        name = str(req_dict.get("name", ""))
        arguments = req_dict.get("arguments", {}) or {}
    else:
        # Object format - use getattr
        tool_id = str(getattr(req, "tool_call_id", ""))
        name = str(getattr(req, "name", ""))
        arguments = getattr(req, "arguments", {}) or {}

    return {
        "id": tool_id,
        "name": name,
        "arguments": arguments,
    }


class ToolCaptureHandler:
    """Handler for capturing tool requests from SDK events.

    Implements first-turn-only capture strategy with deduplication:
    - Captures tools from first ASSISTANT_MESSAGE with tool_requests
    - Deduplicates by tool_call_id (prevents duplicate execution)
    - Ignores subsequent tool events (prevents accumulation bug)
    - Invokes callback when capture completes (for idle flag)

    Contract: sdk-protection:ToolCapture:MUST:1,2

    Usage:
        config = load_sdk_protection_config().tool_capture
        handler = ToolCaptureHandler(
            config=config,
            on_capture_complete=idle_event.set
        )
        unsubscribe = session.on(handler.on_event)
        ...
        if handler.captured_tools:
            # Return tools to Amplifier
    """

    def __init__(
        self,
        on_capture_complete: Callable[[], None] | None = None,
        logger_prefix: str = "[tool_capture]",
        config: ToolCaptureConfig | None = None,
    ) -> None:
        """Initialize handler.

        Args:
            on_capture_complete: Callback to invoke when tools are captured.
                Typically set to idle_event.set() to break wait loop.
            logger_prefix: Prefix for log messages (e.g., "[provider]")
            config: Tool capture configuration. If None, uses defaults
                (first_turn_only=True, deduplicate=True, log_capture_events=True).

        """
        self._captured_tools: list[dict[str, Any]] = []
        self._seen_ids: set[str] = set()  # For deduplication
        self._duplicates_skipped: int = 0  # Count of duplicates filtered
        self._capture_complete = False
        self._on_capture_complete = on_capture_complete
        self._logger_prefix = logger_prefix

        # Configuration - use defaults if not provided
        # Contract: sdk-protection:ToolCapture:MUST:1 (first_turn_only)
        # Contract: sdk-protection:ToolCapture:MUST:2 (deduplicate)
        if config is not None:
            self._first_turn_only = config.first_turn_only
            self._deduplicate = config.deduplicate
            self._log_capture_events = config.log_capture_events
        else:
            # Defaults match config/sdk_protection.yaml
            self._first_turn_only = True
            self._deduplicate = True
            self._log_capture_events = True

    @property
    def captured_tools(self) -> list[dict[str, Any]]:
        """List of captured tool requests."""
        return self._captured_tools

    @property
    def capture_complete(self) -> bool:
        """Whether tool capture has completed (first turn captured)."""
        return self._capture_complete

    @property
    def deduplicated_count(self) -> int:
        """Number of duplicate tool requests that were filtered out."""
        return self._duplicates_skipped

    def on_event(self, sdk_event: Any) -> None:
        """Process SDK event for tool capture.

        Called by session.on() subscription. Extracts tool_requests
        from ASSISTANT_MESSAGE events.

        Contract: sdk-protection:ToolCapture:MUST:1 (first_turn_only)
        Contract: sdk-protection:ToolCapture:MUST:2 (deduplicate)

        Args:
            sdk_event: SDK event (dict or object)

        """
        # First-turn-only: ignore if already captured
        # Contract: sdk-protection:ToolCapture:MUST:1
        if self._first_turn_only and self._capture_complete:
            return

        # Check if this is an ASSISTANT_MESSAGE with tool_requests
        if not has_tool_capture_event(sdk_event):
            return

        # Extract and normalize tool requests
        tool_reqs = extract_tool_requests(sdk_event)
        captured_count = 0
        duplicates_skipped = 0

        for req in tool_reqs:
            normalized = normalize_tool_request(req)
            tool_id = normalized["id"]

            # Deduplication: skip if we've seen this tool_call_id before
            # Contract: sdk-protection:ToolCapture:MUST:2
            if self._deduplicate and tool_id in self._seen_ids:
                duplicates_skipped += 1
                self._duplicates_skipped += 1  # Track for property
                if self._log_capture_events:
                    logger.debug(
                        "%s Skipping duplicate tool_call_id=%s",
                        self._logger_prefix,
                        tool_id,
                    )
                continue

            # Track seen IDs for deduplication
            self._seen_ids.add(tool_id)
            self._captured_tools.append(normalized)
            captured_count += 1

        # Mark capture complete (first-turn-only)
        # Only mark complete if we actually captured something
        if captured_count > 0:
            self._capture_complete = True

            if self._log_capture_events:
                log_msg = f"{self._logger_prefix} Captured {captured_count} tool(s)"
                if duplicates_skipped > 0:
                    log_msg += f" (skipped {duplicates_skipped} duplicate(s))"
                logger.info(log_msg)

            # Invoke callback to break wait loop
            if self._on_capture_complete:
                self._on_capture_complete()
