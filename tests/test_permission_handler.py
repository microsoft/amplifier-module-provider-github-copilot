"""Tests for on_permission_request handler in production path.

Verifies permission handler is present in production CopilotClientWrapper.

Contract: contracts/deny-destroy.md
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
class TestOnPermissionRequestHandler:
    """SDK v0.1.33: CopilotClientWrapper must pass on_permission_request.

    AC: Test that auto-init path includes on_permission_request.
    """

    async def test_permission_handler_denies_all_requests(self) -> None:
        """Permission handler must deny all requests (Deny+Destroy pattern).

        # Contract: deny-destroy:PermissionRequest:MUST:2
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            deny_permission_request,
        )

        # Mock a permission request - use simple spec since SDK PermissionRequest
        # is not available in typings; the handler accepts Any anyway
        mock_request = MagicMock(spec=["tool_name", "arguments"])

        result = deny_permission_request(mock_request)

        # Result can be PermissionRequestResult (with .kind) or dict fallback
        if hasattr(result, "kind"):
            assert result.kind == "denied-by-rules", (
                "Permission request must be denied with kind='denied-by-rules'"
            )
        else:
            assert result["kind"] == "denied-by-rules", (
                "Permission request must be denied with kind='denied-by-rules'"
            )

    async def test_permission_handler_wired_in_session_options(self) -> None:
        """on_permission_request handler is installed in session options.

        # Contract: deny-destroy:PermissionRequest:MUST:1
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
            deny_permission_request,
        )

        captured_kwargs: dict[str, Any] = {}

        # Create a mock SDK client (spec attributes used by wrapper)
        mock_sdk_client = MagicMock(spec=["create_session"])

        # Mock create_session to capture the kwargs passed to it
        async def capture_create_session(**kwargs: Any) -> MagicMock:
            captured_kwargs.update(kwargs)
            mock_session = MagicMock(spec=["session_id", "disconnect"])
            mock_session.session_id = "test-session-id"
            mock_session.disconnect = AsyncMock()
            return mock_session

        mock_sdk_client.create_session = capture_create_session

        # Create wrapper with injected mock client
        wrapper = CopilotClientWrapper(sdk_client=mock_sdk_client)

        # Call session() and verify kwargs
        async with wrapper.session(model="test-model"):
            pass

        # Verify on_permission_request is wired to deny_permission_request
        assert "on_permission_request" in captured_kwargs, (
            "session() must pass on_permission_request to create_session"
        )
        assert captured_kwargs["on_permission_request"] is deny_permission_request, (
            "on_permission_request must be bound to deny_permission_request"
        )


@pytest.mark.sdk_assumption
class TestSDKVersionCompatibility:
    """Detect SDK version drift from v0.1.33 baseline."""

    def test_permission_request_result_has_kind_field(self) -> None:
        """PermissionRequestResult must accept kind parameter (any supported SDK version).

        # Contract: deny-destroy:PermissionRequest:MUST:2

        v0.2.0: copilot.types.PermissionRequestResult
        v0.2.1+: copilot.session.PermissionRequestResult (copilot.types deleted)
        """
        PermissionRequestResult = None
        # Follow the same fallback chain as _imports.py
        try:
            from copilot.types import PermissionRequestResult  # type: ignore[import-untyped]
        except ImportError:
            try:
                from copilot import PermissionRequestResult  # type: ignore[import-untyped,no-redef]
            except ImportError:
                from copilot.session import (  # type: ignore[import-untyped]
                    PermissionRequestResult,  # type: ignore[no-redef]
                )
                # SDK is a hard dependency — ImportError propagates if all fail

        assert isinstance(PermissionRequestResult, type), (  # pragma: no cover
            "SDK installed but PermissionRequestResult not found in any fallback location"
        )

        try:
            result = PermissionRequestResult(  # type: ignore[misc]
                kind="denied-by-rules",
                message="Test denial",
            )
            assert result.kind == "denied-by-rules"  # type: ignore[union-attr]
        except TypeError as e:
            pytest.fail(f"PermissionRequestResult signature changed: {e}")
