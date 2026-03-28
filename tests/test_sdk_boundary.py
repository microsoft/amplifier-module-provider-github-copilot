"""
SDK boundary tests.

Contract: contracts/sdk-boundary.md, contracts/deny-destroy.md
"""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from amplifier_module_provider_github_copilot.error_translation import (
    LLMError,
)
from amplifier_module_provider_github_copilot.sdk_adapter.client import (
    CopilotClientWrapper,
)
from tests.fixtures.config_capture import ConfigCapturingMock


class TestSDKImportError:
    """AC-2: SDK ImportError raises ProviderUnavailableError."""

    @pytest.mark.asyncio
    async def test_sdk_import_error_raises_provider_unavailable(self) -> None:
        """Missing SDK raises ProviderUnavailableError or other LLMError."""
        # Remove copilot from sys.modules if present to force ImportError
        copilot_module = sys.modules.pop("copilot", None)
        try:
            # Patch sys.modules to make import fail
            with patch.dict(sys.modules, {"copilot": None}):
                wrapper = CopilotClientWrapper()

                # Should raise some form of LLMError (ImportError -> ProviderUnavailableError)
                with pytest.raises(LLMError):
                    async with wrapper.session():
                        pass  # pragma: no cover
        finally:
            # Restore copilot module if it was present
            if copilot_module is not None:
                sys.modules["copilot"] = copilot_module


class TestDenyHookOnWrapper:
    """AC-3: Deny hook registered on CopilotClientWrapper.session() path.

    Deny hook is now passed via session config 'hooks' key,
    not via register_pre_tool_use_hook() method call.
    """

    @pytest.mark.asyncio
    async def test_deny_hook_registered_on_wrapper_session(self) -> None:
        """CopilotClientWrapper.session() passes deny hook via config."""
        from typing import Any

        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        captured_config: dict[str, Any] = {}

        mock_session = MagicMock()
        mock_session.disconnect = AsyncMock()

        async def capture_config(**config: Any) -> MagicMock:
            captured_config.update(config)
            return mock_session

        mock_client = AsyncMock()
        mock_client.create_session = AsyncMock(side_effect=capture_config)

        wrapper = CopilotClientWrapper(sdk_client=mock_client)
        async with wrapper.session():
            pass

        # Verify deny hook was passed via session config
        assert "hooks" in captured_config
        assert "on_pre_tool_use" in captured_config["hooks"]


# TestDoubleTranslationGuard removed - migrated to test_behaviors.py
# TestProductionPathWithMockClient::test_llm_error_not_double_wrapped (Issue #6)


class TestSystemMessageStructure:
    """AC-6: system_message parameter structure.

    Contract: sdk-boundary:Config:MUST:3 (session config is dict)
    Migrated to ConfigCapturingMock per WI-004.
    """

    @pytest.mark.asyncio
    async def test_session_system_message_structure(self) -> None:
        """system_message is passed with correct structure.

        Contract: sdk-boundary:Config:MUST:3
        """
        mock_client = ConfigCapturingMock()
        wrapper = CopilotClientWrapper(sdk_client=mock_client)

        async with wrapper.session(system_message="Be helpful"):
            pass

        config = mock_client.last_config
        # Changed from "append" to "replace" to give Amplifier control over persona
        assert config["system_message"] == {"mode": "replace", "content": "Be helpful"}

    @pytest.mark.asyncio
    async def test_session_without_system_message(self) -> None:
        """Session config omits system_message when not provided.

        Contract: sdk-boundary:Config:MUST:3
        """
        mock_client = ConfigCapturingMock()
        wrapper = CopilotClientWrapper(sdk_client=mock_client)

        async with wrapper.session(model="gpt-4"):
            pass

        config = mock_client.last_config
        assert "system_message" not in config
        assert config["model"] == "gpt-4"
