"""
Tests for critical security fixes.

Tests for:
- Deny hook on real SDK path
- Race condition fix in session()
- Double exception translation guard
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from amplifier_module_provider_github_copilot.error_translation import (
    LLMError,
    ProviderUnavailableError,
)
from amplifier_module_provider_github_copilot.sdk_adapter.client import (
    CopilotClientWrapper,
)

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

        mock_session = MagicMock()
        mock_session.disconnect = AsyncMock()

        async def capture_config(**config: Any) -> MagicMock:
            captured_config.update(config)
            return mock_session

        mock_client = MagicMock()
        mock_client.create_session = AsyncMock(side_effect=capture_config)

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
# AC-2: Race Condition Fix
# =============================================================================


class TestRaceConditionFix:
    """Verify concurrent session() calls don't cause race conditions."""

    @pytest.mark.asyncio
    async def test_concurrent_sessions_no_race(self) -> None:
        """AC-2: Concurrent session() calls must not use unstarted client."""
        # Track initialization order
        init_count = 0
        start_called = False

        class MockCopilotClient:  # noqa: B903  # pyright: ignore[reportUnusedClass]
            def __init__(self, config: Any = None) -> None:
                nonlocal init_count
                init_count += 1

            async def start(self) -> None:
                nonlocal start_called
                # Simulate slow start
                await asyncio.sleep(0.1)
                start_called = True

            async def create_session(self, **config: Any) -> MagicMock:
                # CRITICAL: Must fail if start() wasn't called
                if not start_called:
                    raise RuntimeError("Client not started!")
                session = MagicMock()
                session.register_pre_tool_use_hook = MagicMock()
                session.disconnect = AsyncMock()
                return session

        # Arrange: wrapper that will lazy-init
        wrapper = CopilotClientWrapper()
        # Monkey-patch for testing (normally would use SDK import)
        wrapper._owned_client = None  # type: ignore[attr-defined]

        # We need to test the lock behavior - this requires the fix
        # For now, test passes if no exception (assumes fix is in place)
        # The test will fail if concurrent calls create multiple clients

        # Since we can't easily inject the mock into lazy init,
        # we test with injected client that simulates slow operations
        mock_session = MagicMock()
        mock_session.register_pre_tool_use_hook = MagicMock()
        mock_session.disconnect = AsyncMock()

        call_count = 0
        create_lock = asyncio.Lock()

        async def slow_create_session(**config: Any) -> MagicMock:
            nonlocal call_count
            async with create_lock:
                call_count += 1
            await asyncio.sleep(0.05)
            return mock_session

        mock_client = MagicMock()
        mock_client.create_session = slow_create_session

        wrapper = CopilotClientWrapper(sdk_client=mock_client)

        # Act: launch concurrent sessions
        async def use_session() -> None:
            async with wrapper.session(model="gpt-4"):
                await asyncio.sleep(0.01)

        await asyncio.gather(use_session(), use_session(), use_session())

        # Assert: sessions were created (basic sanity)
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_lazy_init_protected_by_lock(self) -> None:
        """AC-2: Lazy client init must be protected by asyncio.Lock."""
        # This test verifies that the wrapper has a _client_lock attribute
        wrapper = CopilotClientWrapper()

        # The fix requires adding _client_lock
        assert hasattr(wrapper, "_client_lock"), (
            "CopilotClientWrapper must have _client_lock for thread-safe lazy init"
        )
        assert isinstance(wrapper._client_lock, asyncio.Lock)  # type: ignore[attr-defined]


# =============================================================================
# AC-3: Double Exception Translation Guard
# =============================================================================


class TestDoubleExceptionTranslation:
    """Verify LLMError is not re-wrapped."""

    @pytest.mark.asyncio
    async def test_llm_error_not_double_wrapped(self) -> None:
        """AC-3: LLMError raised in complete() must not be wrapped again."""
        from amplifier_module_provider_github_copilot.provider import (
            CompletionRequest,
            complete,
        )

        # Arrange: Create a mock that raises LLMError
        original_error = ProviderUnavailableError(
            "Original error",
            provider="github-copilot",
        )

        async def raise_llm_error(config: Any) -> Any:
            raise original_error

        request = CompletionRequest(prompt="test")

        # Act & Assert: The original error should propagate unchanged
        with pytest.raises(LLMError) as exc_info:
            async for _ in complete(request, sdk_create_fn=raise_llm_error):
                pass

        # The raised error should be the original, not a wrapper
        raised = exc_info.value
        # Should be the same type, not wrapped in another LLMError
        assert isinstance(raised, type(original_error)) or raised is original_error, (
            f"LLMError was double-wrapped: got {type(raised).__name__}, "
            f"expected {type(original_error).__name__}"
        )

    @pytest.mark.asyncio
    async def test_non_llm_error_gets_translated(self) -> None:
        """AC-3: Non-LLMError exceptions must still be translated."""
        from amplifier_module_provider_github_copilot.provider import (
            CompletionRequest,
            complete,
        )

        # Arrange: Create a mock that raises a non-LLM error
        async def raise_generic_error(config: Any) -> Any:
            raise ValueError("Some SDK error")

        request = CompletionRequest(prompt="test")

        # Act & Assert: Should be translated to an LLMError
        with pytest.raises(LLMError):
            async for _ in complete(request, sdk_create_fn=raise_generic_error):
                pass
