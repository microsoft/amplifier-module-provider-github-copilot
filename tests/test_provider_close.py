"""Tests for provider.close() wiring to client.close().

Contract: provider-protocol.md — close() must clean up resources
Contract: sdk-boundary.md — provider must clean up SDK resources on close
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider


class TestProviderCloseWiring:
    """Verify provider.close() delegates to client.close()."""

    @pytest.mark.asyncio
    async def test_close_delegates_to_client_close(self):
        """provider.close() MUST call client.close().

        Contract: provider-protocol:close:MUST:1
        """
        provider = GitHubCopilotProvider()
        # Mock the client
        provider._client = MagicMock()  # type: ignore[reportPrivateUsage]
        provider._client.close = AsyncMock()  # type: ignore[reportPrivateUsage]

        await provider.close()

        provider._client.close.assert_called_once()  # type: ignore[reportPrivateUsage]

    @pytest.mark.asyncio
    async def test_close_safe_when_client_not_initialized(self):
        """provider.close() MUST be safe when _client is None.

        Contract: provider-protocol:close:MUST:2 (defensive)
        """
        provider = GitHubCopilotProvider()
        provider._client = None  # type: ignore[assignment,reportPrivateUsage]  # Testing internal state

        # Should not raise
        await provider.close()

    @pytest.mark.asyncio
    async def test_close_safe_when_client_missing_attribute(self):
        """provider.close() MUST be safe when _client attribute missing.

        Contract: provider-protocol:close:MUST:2 (defensive)
        """
        provider = GitHubCopilotProvider()
        # Delete _client attribute to simulate uninitialized state
        del provider._client  # type: ignore[reportPrivateUsage]  # Testing internal state

        # Should not raise
        await provider.close()

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self):
        """provider.close() MUST be safe to call multiple times.

        Contract: provider-protocol:close:SHOULD:1 (idempotent)
        """
        provider = GitHubCopilotProvider()
        provider._client = MagicMock()  # type: ignore[reportPrivateUsage]  # Testing internal state
        provider._client.close = AsyncMock()  # type: ignore[reportPrivateUsage]  # Testing internal state

        # Call twice
        await provider.close()
        await provider.close()

        # Should not raise, client.close() is idempotent
        assert provider._client.close.call_count == 2  # type: ignore[reportPrivateUsage]  # Testing internal state


class TestCancelEmitTasks:
    """Tests for provider.cancel_emit_tasks() — cleanup gap fix.

    Contract: streaming-contract:ProgressiveStreaming:SHOULD:3
    """

    @pytest.mark.asyncio
    async def test_cancel_emit_tasks_cancels_pending_tasks(self) -> None:
        """cancel_emit_tasks() cancels all non-done pending emit tasks.

        Contract: streaming-contract:ProgressiveStreaming:SHOULD:3
        """
        import asyncio

        provider = GitHubCopilotProvider()

        # Create a never-completing task to simulate a pending emit
        async def never_done() -> None:
            await asyncio.sleep(999)

        task = asyncio.create_task(never_done())
        provider._pending_emit_tasks.add(task)  # type: ignore[reportPrivateUsage]

        await provider.cancel_emit_tasks()

        assert task.cancelled()
        assert len(provider._pending_emit_tasks) == 0  # type: ignore[reportPrivateUsage]

    @pytest.mark.asyncio
    async def test_cancel_emit_tasks_safe_when_no_tasks(self) -> None:
        """cancel_emit_tasks() is safe when no tasks are pending.

        Contract: streaming-contract:ProgressiveStreaming:SHOULD:3
        """
        provider = GitHubCopilotProvider()
        # Should not raise
        await provider.cancel_emit_tasks()

    @pytest.mark.asyncio
    async def test_close_delegates_to_cancel_emit_tasks(self) -> None:
        """close() calls cancel_emit_tasks() to clean up streaming tasks.

        Contract: provider-protocol:close:MUST:1
        Contract: streaming-contract:ProgressiveStreaming:SHOULD:3
        """
        from unittest.mock import patch

        provider = GitHubCopilotProvider()
        provider._client = MagicMock()  # type: ignore[reportPrivateUsage]
        provider._client.close = AsyncMock()  # type: ignore[reportPrivateUsage]

        cancel_called = False

        async def mock_cancel() -> None:
            nonlocal cancel_called
            cancel_called = True

        with patch.object(provider, "cancel_emit_tasks", side_effect=mock_cancel):
            await provider.close()

        assert cancel_called
