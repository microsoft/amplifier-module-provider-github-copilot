"""
Contract Compliance Tests: Permission Request Denial.

Contract: contracts/deny-destroy.md

Test Anchors:
- deny-destroy:PermissionRequest:MUST:1 — on_permission_request handler installed
- deny-destroy:PermissionRequest:MUST:2 — handler returns kind="denied-by-rules"
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


class _MockSDKSession:
    """Minimal stub for the raw SDK session object."""

    session_id: str = "test-session-perm"

    async def disconnect(self) -> None:
        """Disconnect stub."""


class _MockSDKClient:
    """Minimal stub for copilot.CopilotClient."""

    async def create_session(self, **kwargs: Any) -> _MockSDKSession:
        """Create session stub."""
        ...


def _make_mock_sdk_client() -> tuple[AsyncMock, AsyncMock]:
    """Return (mock_sdk_client, mock_sdk_session) for session() tests."""
    mock_sdk_session = AsyncMock(spec=_MockSDKSession)
    mock_sdk_session.session_id = "sess-perm-test"
    mock_sdk_session.disconnect = AsyncMock(spec=_MockSDKSession.disconnect)

    mock_sdk_client = AsyncMock(spec=_MockSDKClient)
    mock_sdk_client.create_session = AsyncMock(
        spec=_MockSDKClient.create_session,
        return_value=mock_sdk_session,
    )

    return mock_sdk_client, mock_sdk_session


class TestPermissionRequestHandlerInstalled:
    """deny-destroy:PermissionRequest:MUST:1 — on_permission_request handler installed."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model", ["gpt-4", None], ids=["with-model", "no-model"])
    async def test_permission_handler_installed_on_session(self, model: str | None) -> None:
        """deny-destroy:PermissionRequest:MUST:1 — on_permission_request in create_session kwargs.

        Every session created by CopilotClientWrapper.session() must include
        the on_permission_request handler in the create_session kwargs,
        regardless of whether a model is specified.
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import CopilotClientWrapper

        mock_sdk_client, _ = _make_mock_sdk_client()
        wrapper = CopilotClientWrapper(sdk_client=mock_sdk_client)

        session_kwargs: dict[str, Any] = {"model": model} if model is not None else {}
        async with wrapper.session(**session_kwargs):
            pass

        kwargs = mock_sdk_client.create_session.call_args.kwargs
        assert "on_permission_request" in kwargs, (
            "PermissionRequest:MUST:1 — create_session must include on_permission_request"
        )
        assert callable(kwargs["on_permission_request"]), (
            "PermissionRequest:MUST:1 — on_permission_request must be callable"
        )

    @pytest.mark.asyncio
    async def test_permission_handler_installed_with_tools(self) -> None:
        """deny-destroy:PermissionRequest:MUST:1 — handler installed when tools provided."""
        from amplifier_module_provider_github_copilot.sdk_adapter.client import CopilotClientWrapper

        mock_sdk_client, _ = _make_mock_sdk_client()
        wrapper = CopilotClientWrapper(sdk_client=mock_sdk_client)

        tools: list[dict[str, Any]] = [
            {"name": "search", "description": "Search the web", "parameters": {}},
        ]
        async with wrapper.session(model="gpt-4", tools=tools):
            pass

        kwargs = mock_sdk_client.create_session.call_args.kwargs
        assert "on_permission_request" in kwargs, (
            "PermissionRequest:MUST:1 — handler must be installed when tools provided"
        )


class TestPermissionRequestDenial:
    """deny-destroy:PermissionRequest:MUST:2 — handler returns kind="denied-by-rules"."""

    def test_deny_permission_request_returns_denied_by_rules(self) -> None:
        """deny-destroy:PermissionRequest:MUST:2 — handler returns denied-by-rules.

        The deny_permission_request function must return a result with
        kind="denied-by-rules" to deny all permission requests at source.
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            deny_permission_request,
        )

        # Call with None to simulate a permission request
        result = deny_permission_request(None)

        # Handle both PermissionRequestResult object and dict fallback
        # (dict fallback used when SDK not installed, e.g., SKIP_SDK_CHECK=true)
        kind = result.kind if hasattr(result, "kind") else result.get("kind")
        assert kind == "denied-by-rules", (
            f"PermissionRequest:MUST:2 — kind must be 'denied-by-rules', got {kind!r}"
        )

    def test_deny_permission_request_has_message(self) -> None:
        """deny-destroy:PermissionRequest:MUST:2 — handler includes denial message."""
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            deny_permission_request,
        )

        result = deny_permission_request(None)

        # Handle both object and dict format
        message = result.message if hasattr(result, "message") else result.get("message")
        assert isinstance(message, str) and "Amplifier" in message, (
            f"PermissionRequest:MUST:2 — denial must include a message mentioning Amplifier, got {message!r}"
        )

    def test_deny_permission_request_with_request_object(self) -> None:
        """deny-destroy:PermissionRequest:MUST:2 — handler works with any request input."""
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            deny_permission_request,
        )

        # Create a mock request object to simulate real usage
        mock_request = MagicMock()
        mock_request.tool_name = "bash"
        mock_request.operation = "execute"

        result = deny_permission_request(mock_request)

        kind = result.kind if hasattr(result, "kind") else result.get("kind")
        assert kind == "denied-by-rules", (
            f"PermissionRequest:MUST:2 — must deny regardless of request content, got {kind!r}"
        )


class TestPermissionHandlerIsDenyPermissionRequest:
    """deny-destroy:PermissionRequest:MUST:1,2 — verify installed handler is the right function."""

    @pytest.mark.asyncio
    async def test_installed_handler_is_deny_permission_request(self) -> None:
        """deny-destroy:PermissionRequest:MUST:1,2 — handler IS deny_permission_request.

        The handler installed in create_session kwargs must be the actual
        deny_permission_request function that returns denied-by-rules.
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
            deny_permission_request,
        )

        mock_sdk_client, _ = _make_mock_sdk_client()
        wrapper = CopilotClientWrapper(sdk_client=mock_sdk_client)

        async with wrapper.session(model="gpt-4"):
            pass

        kwargs = mock_sdk_client.create_session.call_args.kwargs
        installed_handler = kwargs["on_permission_request"]

        # Verify the installed handler is exactly deny_permission_request
        assert installed_handler is deny_permission_request, (
            "PermissionRequest:MUST:1,2 — installed handler must be deny_permission_request"
        )
