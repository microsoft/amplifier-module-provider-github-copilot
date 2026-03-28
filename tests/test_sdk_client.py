"""
Tests for SDK client wrapper.

Contract: contracts/sdk-boundary.md

Acceptance Criteria:
- CopilotClientWrapper class exists
- session() context manager destroys on exit, yields raw session
- SDK import isolated to sdk_adapter/client.py
- AC-5: Proper error translation for auth failures
"""

from unittest.mock import AsyncMock

import pytest


class TestCopilotClientWrapperClass:
    """AC-1: CopilotClientWrapper class has required methods."""

    def test_class_exists(self) -> None:
        from amplifier_module_provider_github_copilot.sdk_adapter.client import CopilotClientWrapper

        assert CopilotClientWrapper is not None

    def test_has_session(self) -> None:
        from amplifier_module_provider_github_copilot.sdk_adapter.client import CopilotClientWrapper

        assert hasattr(CopilotClientWrapper, "session")

    def test_has_close(self) -> None:
        from amplifier_module_provider_github_copilot.sdk_adapter.client import CopilotClientWrapper

        assert hasattr(CopilotClientWrapper, "close")
        assert callable(CopilotClientWrapper.close)


class TestSessionYieldsRawSession:
    """session() yields the raw SDK session, not a wrapper."""

    @pytest.mark.asyncio
    async def test_session_yields_session_handle(self) -> None:
        """P2-11: session() yields SessionHandle façade, not raw SDK session."""
        from amplifier_module_provider_github_copilot.sdk_adapter import SessionHandle
        from amplifier_module_provider_github_copilot.sdk_adapter.client import CopilotClientWrapper

        mock_sdk_session = AsyncMock()
        mock_sdk_session.session_id = "sess-raw"
        mock_sdk_session.disconnect = AsyncMock()

        mock_sdk_client = AsyncMock()
        mock_sdk_client.create_session = AsyncMock(return_value=mock_sdk_session)

        wrapper = CopilotClientWrapper(sdk_client=mock_sdk_client)

        async with wrapper.session(model="gpt-4") as session:
            # Should be SessionHandle, not raw SDK session
            assert isinstance(session, SessionHandle)
            assert session.session_id == "sess-raw"


class TestSessionContextManager:
    """AC-3: session() context manager destroys session on exit."""

    @pytest.mark.asyncio
    async def test_session_destroys_on_normal_exit(self) -> None:
        from amplifier_module_provider_github_copilot.sdk_adapter.client import CopilotClientWrapper

        mock_sdk_session = AsyncMock()
        mock_sdk_session.session_id = "sess-001"
        mock_sdk_session.disconnect = AsyncMock()

        mock_sdk_client = AsyncMock()
        mock_sdk_client.create_session = AsyncMock(return_value=mock_sdk_session)

        wrapper = CopilotClientWrapper(sdk_client=mock_sdk_client)
        async with wrapper.session(model="gpt-4"):
            pass

        mock_sdk_session.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_session_destroys_on_exception(self) -> None:
        from amplifier_module_provider_github_copilot.sdk_adapter.client import CopilotClientWrapper

        mock_sdk_session = AsyncMock()
        mock_sdk_session.session_id = "sess-002"
        mock_sdk_session.disconnect = AsyncMock()

        mock_sdk_client = AsyncMock()
        mock_sdk_client.create_session = AsyncMock(return_value=mock_sdk_session)

        wrapper = CopilotClientWrapper(sdk_client=mock_sdk_client)

        with pytest.raises(ValueError):
            async with wrapper.session(model="gpt-4"):
                raise ValueError("user error")

        mock_sdk_session.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_session_creation_error_translated(self) -> None:
        """AC-5: Session creation errors translate to domain errors."""
        from amplifier_module_provider_github_copilot.error_translation import AuthenticationError
        from amplifier_module_provider_github_copilot.sdk_adapter.client import CopilotClientWrapper

        class FakeAuthError(Exception):
            pass

        FakeAuthError.__name__ = "AuthenticationError"

        mock_sdk_client = AsyncMock()
        mock_sdk_client.create_session = AsyncMock(side_effect=FakeAuthError("token invalid"))

        wrapper = CopilotClientWrapper(sdk_client=mock_sdk_client)

        with pytest.raises(AuthenticationError):
            async with wrapper.session(model="gpt-4"):
                pass  # pragma: no cover


class TestClose:
    """close() cleans up owned client resources."""

    @pytest.mark.asyncio
    async def test_close_owned_client(self) -> None:
        from amplifier_module_provider_github_copilot.sdk_adapter.client import CopilotClientWrapper

        wrapper = CopilotClientWrapper()
        mock_owned = AsyncMock()
        mock_owned.stop = AsyncMock()
        wrapper._owned_client = mock_owned  # type: ignore[attr-defined]

        await wrapper.close()

        mock_owned.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_injected_client_does_not_stop(self) -> None:
        """Injected clients are not owned; close() must not stop them."""
        from amplifier_module_provider_github_copilot.sdk_adapter.client import CopilotClientWrapper

        mock_sdk_client = AsyncMock()
        mock_sdk_client.stop = AsyncMock()

        wrapper = CopilotClientWrapper(sdk_client=mock_sdk_client)
        await wrapper.close()

        mock_sdk_client.stop.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self) -> None:
        from amplifier_module_provider_github_copilot.sdk_adapter.client import CopilotClientWrapper

        wrapper = CopilotClientWrapper()
        await wrapper.close()  # no owned client - should not raise
        await wrapper.close()  # second call - still no error


class TestDenyHookInClient:
    """_make_deny_hook_config() is canonical; create_deny_hook() is deleted (D-006)."""

    def test_create_deny_hook_removed_from_client(self) -> None:
        """D-006: create_deny_hook() must not exist — replaced by _make_deny_hook_config()."""
        import amplifier_module_provider_github_copilot.sdk_adapter.client as client_mod

        assert not hasattr(client_mod, "create_deny_hook"), (
            "create_deny_hook() was not deleted — D-006 requires removal"
        )

    def test_deny_all_constant_exists_in_client(self) -> None:
        from amplifier_module_provider_github_copilot.sdk_adapter.client import DENY_ALL

        assert DENY_ALL["permissionDecision"] == "deny"
        # Minimal reason strategy - don't teach model tools are blocked
        assert DENY_ALL["permissionDecisionReason"] == "Processing"
        assert DENY_ALL["suppressOutput"] is True

    def test_make_deny_hook_config_is_replacement(self) -> None:
        """_make_deny_hook_config() is the canonical replacement for create_deny_hook()."""
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            _make_deny_hook_config,  # type: ignore[reportPrivateUsage]
        )

        config = _make_deny_hook_config()
        assert "on_pre_tool_use" in config
        result = config["on_pre_tool_use"]({"toolName": "bash"}, {})
        assert result["permissionDecision"] == "deny"


class TestSDKIsolation:
    """AC-4: SDK imports are isolated to sdk_adapter/ only."""

    def test_no_copilot_imports_in_domain_modules(self) -> None:
        """Non-adapter Python modules must not import from 'copilot'.

        Uses Python's AST module for accurate import detection that
        ignores docstrings, comments, and string literals.
        """
        import ast
        from pathlib import Path

        src_root = Path("amplifier_module_provider_github_copilot")
        violations: list[str] = []
        files_scanned = 0

        for py_file in src_root.glob("*.py"):
            files_scanned += 1
            source = py_file.read_text(encoding="utf-8")

            try:
                tree = ast.parse(source, filename=str(py_file))
            except SyntaxError:
                # If we can't parse, skip (shouldn't happen in valid code)
                continue

            file_name: str = py_file.name  # Extract once for type clarity
            for node in ast.walk(tree):
                # Check: import copilot or import copilot.something
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "copilot" or alias.name.startswith("copilot."):
                            violations.append(file_name)
                            break
                # Check: from copilot import X or from copilot.something import X
                elif isinstance(node, ast.ImportFrom):
                    if node.module and (
                        node.module == "copilot" or node.module.startswith("copilot.")
                    ):
                        violations.append(file_name)
                        break

        assert files_scanned > 0, "No files found — check path"
        assert violations == [], f"SDK imports found outside sdk_adapter/: {violations}"
