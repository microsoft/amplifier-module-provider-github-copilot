"""
Tests for critical security fixes.

Tests for:
- Deny hook on real SDK path
- Double exception translation guard
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from amplifier_module_provider_github_copilot.sdk_adapter.client import (
    CopilotClientWrapper,
)

# =============================================================================
# Stub Classes for Spec-Constrained Mocking
# =============================================================================


class _MockSDKClient:
    """Minimal SDK client stub for spec-constrained mocking.

    Contract: sdk-boundary:Membrane:MUST:1 — Mocks MUST use spec= to prevent
    false-positive tests from silently accepting non-existent attributes.
    """

    async def create_session(self, **kwargs: Any) -> Any:
        """Create SDK session with given config."""
        ...


class _MockSDKSession:
    """Minimal SDK session stub for spec-constrained mocking.

    Reflects attributes used by CopilotClientWrapper.session():
    - session_id: for logging
    - disconnect(): async cleanup method
    """

    session_id: str = "test-session"

    async def disconnect(self) -> None:
        """Disconnect the SDK session."""
        ...


# =============================================================================
# AC-1: Deny Hook on Real SDK Path
# =============================================================================


class TestDenyHookOnRealSDKPath:
    """Verify deny hook is installed on CopilotClientWrapper.session().

    Deny hook is now passed via session config 'hooks' key,
    not via register_pre_tool_use_hook() method call.
    """

    @pytest.mark.asyncio
    async def test_session_registers_deny_hook(self) -> None:
        """AC-1: session() MUST pass deny hook via session config.

        The correct SDK API passes hooks via session_config['hooks'],
        not via a method call on the session object.
        """
        # Arrange: mock SDK client that captures session config
        captured_config: dict[str, Any] = {}

        mock_session = MagicMock(spec=_MockSDKSession)
        mock_session.session_id = "test-session"
        mock_session.disconnect = AsyncMock(spec=_MockSDKSession.disconnect)

        async def capture_config(**config: Any) -> MagicMock:
            captured_config.update(config)
            return mock_session

        mock_client = MagicMock(spec=_MockSDKClient)
        mock_client.create_session = AsyncMock(
            spec=_MockSDKClient.create_session, side_effect=capture_config
        )

        wrapper = CopilotClientWrapper(sdk_client=mock_client)

        # Act: use session context manager
        async with wrapper.session(model="gpt-4"):
            pass

        # Assert: deny hook was passed via session config 'hooks' key
        assert "hooks" in captured_config, "session config must include 'hooks' key"
        hooks = captured_config["hooks"]
        assert "on_pre_tool_use" in hooks, "hooks must include 'on_pre_tool_use'"

        # Verify the hook denies all tools
        deny_hook = hooks["on_pre_tool_use"]
        result = deny_hook({"toolName": "bash"}, {})
        assert result["permissionDecision"] == "deny"


# =============================================================================
# AC-3: Double Exception Translation Guard
# =============================================================================
# TestDoubleExceptionTranslation removed - migrated to test_behaviors.py
# TestProductionPathWithMockClient::test_llm_error_not_double_wrapped (Issue #6)


# =============================================================================
# behaviors:Security:MUST:2 — Mount traceback redacted before DEBUG log
# =============================================================================


class TestMountTracebackRedaction:
    """Mount failure tracebacks MUST be redacted before DEBUG log emission.

    Contract: behaviors:Security:MUST:2

    Raw exc_info=True bypasses security_redaction.py and may emit tokens
    present in exception messages or traceback frame local variables.
    """

    @pytest.mark.asyncio
    async def test_mount_debug_log_redacts_token_in_traceback(self) -> None:
        """behaviors:Security:MUST:2 — A token in the exception chain MUST NOT
        appear in DEBUG log output after mount() failure.

        This test calls the actual mount() function with patched infrastructure
        to inject a fake token into an exception, then verifies the token is
        redacted in the DEBUG log output.
        """
        import logging

        from amplifier_module_provider_github_copilot import mount
        from amplifier_module_provider_github_copilot.security_redaction import REDACTED

        # Capture DEBUG log records
        captured: list[str] = []

        class CapturingHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                captured.append(self.format(record))

        handler = CapturingHandler()
        logger = logging.getLogger("amplifier_module_provider_github_copilot")
        original_level = logger.level
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)

        # Inject a token into the exception that will appear in the traceback
        fake_token = "ghp_" + "A" * 25  # Matches GitHub PAT pattern

        # Create a mock coordinator whose mount() raises with the fake token
        mock_coordinator = MagicMock()
        mock_coordinator.mount = AsyncMock(
            side_effect=RuntimeError(f"SDK failed: token={fake_token}")
        )

        # Mock _acquire_shared_client to return a mock wrapper
        mock_wrapper = MagicMock(spec=CopilotClientWrapper)

        try:
            with (
                patch(
                    "amplifier_module_provider_github_copilot._acquire_shared_client",
                    new=AsyncMock(return_value=mock_wrapper),
                ),
                patch(
                    "amplifier_module_provider_github_copilot._release_shared_client",
                    new=AsyncMock(),
                ),
            ):
                # Call actual mount() - it will fail at coordinator.mount() and
                # trigger the exception handler that logs the redacted traceback
                with pytest.raises(RuntimeError, match="SDK failed"):
                    await mount(mock_coordinator, config=None)
        finally:
            logger.removeHandler(handler)
            logger.setLevel(original_level)

        # The token must NOT appear in any captured log line
        for line in captured:
            assert fake_token not in line, (
                f"Token {fake_token!r} leaked in DEBUG log. "
                f"Contract: behaviors:Security:MUST:2. Line: {line!r}"
            )
        # REDACTED placeholder must appear instead
        assert any(REDACTED in line for line in captured), (
            f"Expected REDACTED in debug log output but found none. Captured: {captured}"
        )
