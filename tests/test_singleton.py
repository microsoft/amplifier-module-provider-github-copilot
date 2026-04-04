# pyright: reportPrivateUsage=false
"""Tests for process-level singleton for CopilotClientWrapper.

Contract: provider-protocol:complete:MUST:1

These tests verify the singleton lifecycle for CopilotClientWrapper:
- Shared client across mount() calls
- Refcount management
- Health check and client replacement
- Lock timeout protection
- Client injection in GitHubCopilotProvider
"""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from amplifier_module_provider_github_copilot.sdk_adapter.client import CopilotClientWrapper

# ============================================================================
# Test: Singleton Lifecycle
# ============================================================================


class TestSingletonLifecycle:
    """Tests for _acquire_shared_client / _release_shared_client."""

    @pytest.fixture(autouse=True)
    def reset_singleton_state(self) -> Generator[None, None, None]:
        """Reset module-level singleton state before and after each test."""
        import amplifier_module_provider_github_copilot as provider_module

        provider_module._shared_client = None
        provider_module._shared_client_refcount = 0

        yield

        provider_module._shared_client = None
        provider_module._shared_client_refcount = 0

    @pytest.mark.asyncio
    async def test_acquire_creates_client_on_first_call(self) -> None:
        """First _acquire_shared_client() creates a new client.

        Contract: provider-protocol:complete:MUST:1
        """
        import amplifier_module_provider_github_copilot as provider_module

        assert provider_module._shared_client is None
        assert provider_module._shared_client_refcount == 0

        # Mock CopilotClientWrapper using patch with full path (reliable for async)
        mock_client = MagicMock(spec=CopilotClientWrapper)
        mock_wrapper_cls = MagicMock(return_value=mock_client)

        with patch(
            "amplifier_module_provider_github_copilot.CopilotClientWrapper", mock_wrapper_cls
        ):
            client = await provider_module._acquire_shared_client()

            assert client is mock_client
            assert provider_module._shared_client is mock_client
            assert provider_module._shared_client_refcount == 1

    @pytest.mark.asyncio
    async def test_acquire_reuses_client_on_second_call(self) -> None:
        """Second _acquire_shared_client() reuses existing client.

        Contract: provider-protocol:complete:MUST:1
        """
        import amplifier_module_provider_github_copilot as provider_module

        mock_client = MagicMock(spec=CopilotClientWrapper)
        mock_client.is_healthy.return_value = True
        mock_wrapper_cls = MagicMock(return_value=mock_client)

        with patch(
            "amplifier_module_provider_github_copilot.CopilotClientWrapper", mock_wrapper_cls
        ):
            client1 = await provider_module._acquire_shared_client()
            client2 = await provider_module._acquire_shared_client()

            assert client1 is client2
            assert provider_module._shared_client_refcount == 2
            # CopilotClientWrapper should only be created once
            assert mock_wrapper_cls.call_count == 1

    @pytest.mark.asyncio
    async def test_refcount_increments_on_acquire(self) -> None:
        """Each _acquire_shared_client() increments refcount."""
        import amplifier_module_provider_github_copilot as provider_module

        mock_client = MagicMock(spec=CopilotClientWrapper)
        mock_client.is_healthy.return_value = True
        mock_wrapper_cls = MagicMock(return_value=mock_client)

        with patch(
            "amplifier_module_provider_github_copilot.CopilotClientWrapper", mock_wrapper_cls
        ):
            await provider_module._acquire_shared_client()
            assert provider_module._shared_client_refcount == 1

            await provider_module._acquire_shared_client()
            assert provider_module._shared_client_refcount == 2

            await provider_module._acquire_shared_client()
            assert provider_module._shared_client_refcount == 3

    @pytest.mark.asyncio
    async def test_refcount_decrements_on_release(self) -> None:
        """Each _release_shared_client() decrements refcount."""
        import amplifier_module_provider_github_copilot as provider_module

        mock_client = MagicMock(spec=CopilotClientWrapper)
        mock_client.is_healthy.return_value = True
        mock_client.close = AsyncMock()
        mock_wrapper_cls = MagicMock(return_value=mock_client)

        with patch(
            "amplifier_module_provider_github_copilot.CopilotClientWrapper", mock_wrapper_cls
        ):
            await provider_module._acquire_shared_client()
            await provider_module._acquire_shared_client()
            assert provider_module._shared_client_refcount == 2

            await provider_module._release_shared_client()
            assert provider_module._shared_client_refcount == 1

            await provider_module._release_shared_client()
            assert provider_module._shared_client_refcount == 0

    @pytest.mark.asyncio
    async def test_client_closed_only_on_last_release(self) -> None:
        """Client.close() called only when refcount reaches 0.

        Contract: provider-protocol:complete:MUST:1
        """
        import amplifier_module_provider_github_copilot as provider_module

        mock_client = MagicMock(spec=CopilotClientWrapper)
        mock_client.is_healthy.return_value = True
        mock_client.close = AsyncMock()
        mock_wrapper_cls = MagicMock(return_value=mock_client)

        with patch(
            "amplifier_module_provider_github_copilot.CopilotClientWrapper", mock_wrapper_cls
        ):
            await provider_module._acquire_shared_client()
            await provider_module._acquire_shared_client()

            # First release - client should NOT be closed
            await provider_module._release_shared_client()
            mock_client.close.assert_not_called()
            assert provider_module._shared_client is mock_client

            # Second release - client SHOULD be closed
            await provider_module._release_shared_client()
            mock_client.close.assert_called_once()
            assert provider_module._shared_client is None

    @pytest.mark.asyncio
    async def test_negative_refcount_impossible(self) -> None:
        """Extra _release_shared_client() calls don't go negative."""
        import amplifier_module_provider_github_copilot as provider_module

        mock_client = MagicMock(spec=CopilotClientWrapper)
        mock_client.is_healthy.return_value = True
        mock_client.close = AsyncMock()
        mock_wrapper_cls = MagicMock(return_value=mock_client)

        with patch(
            "amplifier_module_provider_github_copilot.CopilotClientWrapper", mock_wrapper_cls
        ):
            await provider_module._acquire_shared_client()
            await provider_module._acquire_shared_client()

            # Release 3 times (more than acquired)
            await provider_module._release_shared_client()
            await provider_module._release_shared_client()
            await provider_module._release_shared_client()

            assert provider_module._shared_client_refcount == 0


# ============================================================================
# Test: Health Check and Client Replacement
# ============================================================================


class TestHealthCheck:
    """Tests for is_healthy() and unhealthy client replacement."""

    @pytest.fixture(autouse=True)
    def reset_singleton_state(self) -> Generator[None, None, None]:
        """Reset module-level singleton state before and after each test."""
        import amplifier_module_provider_github_copilot as provider_module

        provider_module._shared_client = None
        provider_module._shared_client_refcount = 0

        yield

        provider_module._shared_client = None
        provider_module._shared_client_refcount = 0

    def test_is_healthy_true_for_live_client(self) -> None:
        """is_healthy() returns True when client is alive.

        Contract: sdk-boundary:Config:MUST:1
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        wrapper = CopilotClientWrapper()
        # No owned client and no injected client - should be healthy
        # (health check is about owned client state, not existence)
        assert wrapper.is_healthy() is True

    def test_is_healthy_false_for_stopped_client(self) -> None:
        """is_healthy() returns False after close().

        Contract: sdk-boundary:Config:MUST:1
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        wrapper = CopilotClientWrapper()
        mock_client = MagicMock(spec=CopilotClientWrapper)
        wrapper._owned_client = mock_client
        wrapper._stopped = True  # Mark as stopped

        assert wrapper.is_healthy() is False

    @pytest.mark.asyncio
    async def test_unhealthy_client_replaced_on_acquire(self) -> None:
        """_acquire_shared_client() replaces unhealthy client."""
        import amplifier_module_provider_github_copilot as provider_module

        # First client - will become unhealthy
        unhealthy_client = MagicMock(spec=CopilotClientWrapper)
        unhealthy_client.is_healthy.return_value = False
        unhealthy_client.close = AsyncMock()

        # Second client - healthy replacement
        healthy_client = MagicMock(spec=CopilotClientWrapper)
        healthy_client.is_healthy.return_value = True

        mock_wrapper_cls = MagicMock(side_effect=[unhealthy_client, healthy_client])

        with patch(
            "amplifier_module_provider_github_copilot.CopilotClientWrapper", mock_wrapper_cls
        ):
            # First acquire creates client
            client1 = await provider_module._acquire_shared_client()
            assert client1 is unhealthy_client

            # Release to simulate usage cycle
            await provider_module._release_shared_client()

            # Mark as unhealthy (simulating client failure)
            unhealthy_client.is_healthy.return_value = False

            # Next acquire should detect unhealthy and replace
            # Set shared state to trigger replacement logic
            provider_module._shared_client = unhealthy_client
            provider_module._shared_client_refcount = 0

            client2 = await provider_module._acquire_shared_client()

            # Should have closed old and created new
            unhealthy_client.close.assert_called()
            assert client2 is healthy_client


# ============================================================================
# Test: Lock Timeout
# ============================================================================


class TestLockTimeout:
    """Tests for lock timeout protection."""

    @pytest.fixture(autouse=True)
    def reset_singleton_state(self) -> Generator[None, None, None]:
        """Reset module-level singleton state before and after each test."""
        import amplifier_module_provider_github_copilot as provider_module

        provider_module._shared_client = None
        provider_module._shared_client_refcount = 0

        yield

        provider_module._shared_client = None
        provider_module._shared_client_refcount = 0

    @pytest.mark.asyncio
    async def test_lock_timeout_raises_timeout_error(self) -> None:
        """30s lock timeout raises TimeoutError when lock is held.

        Contract: sdk-protection:Singleton:MUST:8
        Timeout value sourced from sdk_protection.yaml singleton.lock_timeout_seconds.
        """
        import threading

        import amplifier_module_provider_github_copilot as provider_module

        # Hold the threading.Lock from a background thread to simulate deadlock
        lock_held = threading.Event()
        release_signal = threading.Event()

        def hold_lock() -> None:
            provider_module._state_lock.acquire()
            lock_held.set()
            release_signal.wait(timeout=5.0)
            provider_module._state_lock.release()

        t = threading.Thread(target=hold_lock, daemon=True)
        t.start()
        try:
            lock_held.wait(timeout=1.0)
            # Patch config to return a tiny timeout so the test runs fast.
            # Contract: sdk-protection:Singleton:MUST:8 — timeout sourced from YAML
            mock_config = MagicMock()
            mock_config.singleton.lock_timeout_seconds = 0.01
            with patch(
                "amplifier_module_provider_github_copilot.load_sdk_protection_config",
                return_value=mock_config,
            ):
                with pytest.raises(TimeoutError):
                    await provider_module._acquire_shared_client()
        finally:
            release_signal.set()
            t.join(timeout=2.0)

    def test_state_lock_is_eager_threading_lock(self) -> None:
        """_state_lock is a threading.Lock, eagerly initialized at import time.

        Contract: sdk-protection:Singleton:MUST:8
        No asyncio.Lock at module level — threading.Lock is event-loop-safe.
        """
        import threading

        import amplifier_module_provider_github_copilot as provider_module

        assert isinstance(provider_module._state_lock, type(threading.Lock()))

    def test_lock_timeout_in_singleton_config(self) -> None:
        """lock_timeout_seconds must be present in SingletonConfig.

        Contract: sdk-protection:Singleton:MUST:8
        Three-Medium: timeout policy lives in YAML, not as Python constant.
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_sdk_protection_config,
        )

        config = load_sdk_protection_config()
        assert hasattr(config.singleton, "lock_timeout_seconds"), (
            "SdkProtectionConfig.singleton must have lock_timeout_seconds field"
        )
        assert isinstance(config.singleton.lock_timeout_seconds, float)
        assert config.singleton.lock_timeout_seconds > 0


# ============================================================================
# Test: Provider Client Injection
# ============================================================================


class TestProviderClientInjection:
    """Tests for GitHubCopilotProvider client injection."""

    def test_provider_uses_injected_client(self) -> None:
        """GitHubCopilotProvider(client=mock) uses injected client.

        Contract: provider-protocol:complete:MUST:1
        """
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        mock_client = MagicMock(spec=CopilotClientWrapper)
        provider = GitHubCopilotProvider(config=None, coordinator=None, client=mock_client)

        assert provider._client is mock_client

    def test_provider_creates_own_client_without_injection(self) -> None:
        """GitHubCopilotProvider() creates CopilotClientWrapper.

        Contract: provider-protocol:complete:MUST:1
        """
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        provider = GitHubCopilotProvider(config=None, coordinator=None)

        assert isinstance(provider._client, CopilotClientWrapper)


# ============================================================================
# Test: Mount Integration
# ============================================================================


class TestMountIntegration:
    """Tests for mount() singleton integration."""

    @pytest.fixture(autouse=True)
    def reset_singleton_state(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Reset module-level singleton state before each test."""
        import amplifier_module_provider_github_copilot as provider_module

        monkeypatch.setattr(provider_module, "_shared_client", None)
        monkeypatch.setattr(provider_module, "_shared_client_refcount", 0)

    @pytest.mark.asyncio
    async def test_mount_creates_shared_client(self) -> None:
        """mount() calls _acquire_shared_client().

        Contract: provider-protocol:complete:MUST:1
        """
        import amplifier_module_provider_github_copilot as provider_module

        mock_coordinator = MagicMock()
        mock_coordinator.mount = AsyncMock()

        with (
            patch.object(
                provider_module, "_acquire_shared_client", new_callable=AsyncMock
            ) as mock_acquire,
            patch.object(provider_module, "CopilotClientWrapper") as mock_wrapper_cls,
        ):
            mock_client = MagicMock(spec=CopilotClientWrapper)
            mock_acquire.return_value = mock_client
            mock_wrapper_cls.return_value = mock_client

            cleanup = await provider_module.mount(mock_coordinator)

            mock_acquire.assert_called_once()
            assert cleanup is not None

    @pytest.mark.asyncio
    async def test_cleanup_calls_release(self) -> None:
        """Cleanup function from mount() calls _release_shared_client().

        Contract: provider-protocol:complete:MUST:1
        """
        import amplifier_module_provider_github_copilot as provider_module

        mock_coordinator = MagicMock()
        mock_coordinator.mount = AsyncMock()

        with (
            patch.object(
                provider_module, "_acquire_shared_client", new_callable=AsyncMock
            ) as mock_acquire,
            patch.object(
                provider_module, "_release_shared_client", new_callable=AsyncMock
            ) as mock_release,
            patch.object(provider_module, "CopilotClientWrapper"),
        ):
            mock_client = MagicMock(spec=CopilotClientWrapper)
            mock_client.close = AsyncMock()
            mock_acquire.return_value = mock_client

            cleanup = await provider_module.mount(mock_coordinator)
            assert cleanup is not None

            await cleanup()
            mock_release.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_cancels_provider_emit_tasks(self) -> None:
        """Cleanup from mount() calls provider.cancel_emit_tasks() before release.

        Contract: streaming-contract:ProgressiveStreaming:SHOULD:3
        """
        import amplifier_module_provider_github_copilot as provider_module

        mock_coordinator = MagicMock()
        mock_coordinator.mount = AsyncMock()

        cancel_tasks_called = False

        async def mock_cancel_emit_tasks() -> None:
            nonlocal cancel_tasks_called
            cancel_tasks_called = True

        # Patch on the class that mount() actually uses (via __init__.py's namespace),
        # not a freshly re-imported copy that may differ after sys.modules manipulation.
        with (
            patch.object(
                provider_module, "_acquire_shared_client", new_callable=AsyncMock
            ) as mock_acquire,
            patch.object(provider_module, "_release_shared_client", new_callable=AsyncMock),
            patch.object(provider_module, "CopilotClientWrapper"),
            patch.object(
                provider_module.GitHubCopilotProvider,
                "cancel_emit_tasks",
                side_effect=mock_cancel_emit_tasks,
            ),
        ):
            mock_client = MagicMock(spec=CopilotClientWrapper)
            mock_client.close = AsyncMock()
            mock_acquire.return_value = mock_client

            cleanup = await provider_module.mount(mock_coordinator)
            assert cleanup is not None

            await cleanup()

        assert cancel_tasks_called

    @pytest.mark.asyncio
    async def test_mount_failure_releases_reference_and_raises(self) -> None:
        """Exception during mount() releases acquired reference, then raises.

        Contract: provider-protocol:complete:MUST:1
        P2 Fix: Raise after releasing reference (not return None).
        """
        import amplifier_module_provider_github_copilot as provider_module

        mock_coordinator = MagicMock()
        mock_coordinator.mount = AsyncMock(side_effect=RuntimeError("Mount failed"))

        with (
            patch.object(
                provider_module, "_acquire_shared_client", new_callable=AsyncMock
            ) as mock_acquire,
            patch.object(
                provider_module, "_release_shared_client", new_callable=AsyncMock
            ) as mock_release,
            patch.object(provider_module, "CopilotClientWrapper"),
        ):
            mock_client = MagicMock(spec=CopilotClientWrapper)
            mock_acquire.return_value = mock_client

            # P2 Fix: mount() now raises instead of returning None
            with pytest.raises(RuntimeError, match="Mount failed"):
                await provider_module.mount(mock_coordinator)

            # Reference should have been released before raise
            mock_release.assert_called_once()

    @pytest.mark.asyncio
    async def test_two_mounts_share_same_client(self) -> None:
        """Two calls to mount() share the same CopilotClientWrapper instance.

        Contract: provider-protocol:complete:MUST:1
        """
        import amplifier_module_provider_github_copilot as provider_module

        mock_coordinator1 = MagicMock()
        mock_coordinator1.mount = AsyncMock()
        mock_coordinator2 = MagicMock()
        mock_coordinator2.mount = AsyncMock()

        mock_client = MagicMock(spec=CopilotClientWrapper)
        mock_client.is_healthy.return_value = True
        mock_client.close = AsyncMock()
        mock_wrapper_cls = MagicMock(return_value=mock_client)

        with patch(
            "amplifier_module_provider_github_copilot.CopilotClientWrapper", mock_wrapper_cls
        ):
            cleanup1 = await provider_module.mount(mock_coordinator1)
            cleanup2 = await provider_module.mount(mock_coordinator2)

            assert cleanup1 is not None
            assert cleanup2 is not None

            # Both should use the same client
            assert provider_module._shared_client is mock_client
            assert provider_module._shared_client_refcount == 2

            # CopilotClientWrapper only created once
            assert mock_wrapper_cls.call_count == 1


# ============================================================================
# Test: Concurrent Access
# ============================================================================


class TestConcurrentAccess:
    """Tests for concurrent mount() calls."""

    @pytest.fixture(autouse=True)
    def reset_singleton_state(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Reset module-level singleton state before each test."""
        import amplifier_module_provider_github_copilot as provider_module

        monkeypatch.setattr(provider_module, "_shared_client", None)
        monkeypatch.setattr(provider_module, "_shared_client_refcount", 0)

    @pytest.mark.asyncio
    async def test_concurrent_mounts_serialized(self) -> None:
        """Two concurrent mount() calls share client via lock serialization."""
        import amplifier_module_provider_github_copilot as provider_module

        mock_coordinator1 = MagicMock()
        mock_coordinator1.mount = AsyncMock()
        mock_coordinator2 = MagicMock()
        mock_coordinator2.mount = AsyncMock()

        mock_client = MagicMock(spec=CopilotClientWrapper)
        mock_client.is_healthy.return_value = True
        mock_client.close = AsyncMock()
        mock_wrapper_cls = MagicMock(return_value=mock_client)

        with patch(
            "amplifier_module_provider_github_copilot.CopilotClientWrapper", mock_wrapper_cls
        ):
            # Run both mounts concurrently
            results = await asyncio.gather(
                provider_module.mount(mock_coordinator1),
                provider_module.mount(mock_coordinator2),
            )

            assert results[0] is not None
            assert results[1] is not None

            # Both should use the same client
            assert provider_module._shared_client is mock_client
            assert provider_module._shared_client_refcount == 2

            # CopilotClientWrapper only created once despite concurrent calls
            assert mock_wrapper_cls.call_count == 1


# ============================================================================
# Merged from test_coverage_gaps_final.py — __init__.py failure paths
# ============================================================================


class TestAcquireSharedClientFailurePaths:
    """Cover __init__.py L144-148: CopilotClientWrapper() raises during acquire."""

    @pytest.fixture(autouse=True)
    def reset_state(self) -> Generator[None, None, None]:
        import amplifier_module_provider_github_copilot as m

        m._shared_client = None
        m._shared_client_refcount = 0
        yield
        m._shared_client = None
        m._shared_client_refcount = 0

    @pytest.mark.asyncio
    async def test_wrapper_creation_failure_clears_state(self) -> None:
        """L144-148: CopilotClientWrapper() raising clears _shared_client to None.

        Contract: provider-protocol:complete:MUST:1
        """
        import amplifier_module_provider_github_copilot as m

        with patch(
            "amplifier_module_provider_github_copilot.CopilotClientWrapper",
            side_effect=RuntimeError("SDK init failed"),
        ):
            with pytest.raises(RuntimeError, match="SDK init failed"):
                await m._acquire_shared_client()

        # After failure, state must be clean
        assert m._shared_client is None
        assert m._shared_client_refcount == 0

    @pytest.mark.asyncio
    async def test_unhealthy_client_close_error_logged_not_raised(self) -> None:
        """L133-134: close() on unhealthy client raises — error is caught and logged.

        Contract: provider-protocol:complete:MUST:1
        """
        import amplifier_module_provider_github_copilot as m

        # Set up unhealthy client whose close() throws
        unhealthy_client = MagicMock(spec=CopilotClientWrapper)
        unhealthy_client.is_healthy.return_value = False
        unhealthy_client.close = AsyncMock(side_effect=RuntimeError("close failed"))

        m._shared_client = unhealthy_client
        m._shared_client_refcount = 0

        # Replacement client
        healthy_client = MagicMock(spec=CopilotClientWrapper)
        healthy_client.is_healthy.return_value = True

        with patch(
            "amplifier_module_provider_github_copilot.CopilotClientWrapper",
            return_value=healthy_client,
        ):
            # Should succeed despite close() raising
            result = await m._acquire_shared_client()

        assert result is healthy_client
        assert m._shared_client is healthy_client


class TestReleaseSharedClientClosureErrors:
    """Cover __init__.py L174-175: close() error on last-ref release."""

    @pytest.fixture(autouse=True)
    def reset_state(self) -> Generator[None, None, None]:
        import amplifier_module_provider_github_copilot as m

        m._shared_client = None
        m._shared_client_refcount = 0
        yield
        m._shared_client = None
        m._shared_client_refcount = 0

    @pytest.mark.asyncio
    async def test_close_error_on_last_release_is_swallowed(self) -> None:
        """L174-175: close() raising on last reference is caught, client set to None.

        Contract: provider-protocol:complete:MUST:1
        """
        import amplifier_module_provider_github_copilot as m

        client = MagicMock(spec=CopilotClientWrapper)
        client.close = AsyncMock(side_effect=OSError("close failed"))
        m._shared_client = client
        m._shared_client_refcount = 1

        # Should not raise despite close() failure
        await m._release_shared_client()

        # Client must be cleared regardless
        assert m._shared_client is None


class TestMountExceptionPaths:
    """Cover __init__.py: mount() exception paths raise (P2 Fix)."""

    @pytest.fixture(autouse=True)
    def reset_state(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import amplifier_module_provider_github_copilot as m

        monkeypatch.setattr(m, "_shared_client", None)
        monkeypatch.setattr(m, "_shared_client_refcount", 0)

    @pytest.mark.asyncio
    async def test_mount_coordinator_exception_raises(self) -> None:
        """coordinator.mount() throwing raises (P2 Fix: not graceful degradation).

        Contract: provider-protocol.md
        P2 Fix: Raise instead of return None.
        """
        import amplifier_module_provider_github_copilot as m

        coordinator = MagicMock()
        coordinator.mount = AsyncMock(side_effect=Exception("coordinator error"))

        mock_client = MagicMock(spec=CopilotClientWrapper)

        with (
            patch.object(
                m, "_acquire_shared_client", new_callable=AsyncMock, return_value=mock_client
            ),
            patch.object(m, "_release_shared_client", new_callable=AsyncMock),
        ):
            with pytest.raises(Exception, match="coordinator error"):
                await m.mount(coordinator)

    @pytest.mark.asyncio
    async def test_mount_acquire_timeout_raises(self) -> None:
        """TimeoutError from _acquire_shared_client raises (P2 Fix).

        Contract: provider-protocol.md
        P2 Fix: Raise instead of return None.
        """
        import amplifier_module_provider_github_copilot as m

        coordinator = MagicMock()

        with patch.object(
            m,
            "_acquire_shared_client",
            new_callable=AsyncMock,
            side_effect=TimeoutError("lock timeout"),
        ):
            with pytest.raises(TimeoutError, match="lock timeout"):
                await m.mount(coordinator)

    @pytest.mark.asyncio
    async def test_mount_acquire_generic_exception_raises(self) -> None:
        """Generic exception from _acquire_shared_client raises (P2 Fix).

        Contract: provider-protocol.md
        P2 Fix: Raise instead of return None.
        """
        import amplifier_module_provider_github_copilot as m

        coordinator = MagicMock()

        with patch.object(
            m,
            "_acquire_shared_client",
            new_callable=AsyncMock,
            side_effect=RuntimeError("sdk error"),
        ):
            with pytest.raises(RuntimeError, match="sdk error"):
                await m.mount(coordinator)


# ============================================================================
# Test: Session Isolation
# P3.19: Verify no cross-session state bleed
# ============================================================================


class TestSessionIsolation:
    """Tests verifying session isolation — no cross-session state bleed.

    Contract: sdk-boundary:Membrane:MUST:1 — Each provider instance isolates state.

    Despite sharing a singleton CopilotClientWrapper, each mount() should:
    - Create an independent GitHubCopilotProvider instance
    - Have independent Session state
    - Not share accumulated response state
    - Cleanup independently
    """

    @pytest.fixture(autouse=True)
    def reset_singleton_state(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Reset module-level singleton state before each test."""
        import amplifier_module_provider_github_copilot as provider_module

        monkeypatch.setattr(provider_module, "_shared_client", None)
        monkeypatch.setattr(provider_module, "_shared_client_refcount", 0)

    @pytest.mark.asyncio
    async def test_provider_instances_are_independent(self) -> None:
        """Two mount() calls create independent GitHubCopilotProvider instances.

        Contract: sdk-boundary:Membrane:MUST:1
        """
        import amplifier_module_provider_github_copilot as provider_module
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        captured_providers: list[GitHubCopilotProvider] = []

        # Capture provider instances during coordinator.mount()
        async def capture_mount(
            namespace: str, provider: GitHubCopilotProvider, *, name: str
        ) -> None:
            captured_providers.append(provider)

        mock_coordinator1 = MagicMock()
        mock_coordinator1.mount = AsyncMock(side_effect=capture_mount)
        mock_coordinator2 = MagicMock()
        mock_coordinator2.mount = AsyncMock(side_effect=capture_mount)

        mock_client = MagicMock(spec=CopilotClientWrapper)
        mock_client.is_healthy.return_value = True
        mock_client.close = AsyncMock()

        with patch(
            "amplifier_module_provider_github_copilot.CopilotClientWrapper",
            return_value=mock_client,
        ):
            cleanup1 = await provider_module.mount(mock_coordinator1)
            cleanup2 = await provider_module.mount(mock_coordinator2)

        # Both cleanups should exist
        assert cleanup1 is not None
        assert cleanup2 is not None

        # Two independent providers should have been created
        assert len(captured_providers) == 2
        provider1, provider2 = captured_providers

        # Different provider instances (key assertion)
        assert provider1 is not provider2

        # Both share the same underlying client (memory efficiency)
        assert provider1._client is provider2._client

        # Cleanup both
        await cleanup1()
        await cleanup2()

    @pytest.mark.asyncio
    async def test_cleanup_for_one_session_does_not_affect_other(self) -> None:
        """Cleanup from first mount() doesn't affect second mount().

        Contract: sdk-boundary:Membrane:MUST:1
        """
        import amplifier_module_provider_github_copilot as provider_module

        mock_coordinator1 = MagicMock()
        mock_coordinator1.mount = AsyncMock()
        mock_coordinator2 = MagicMock()
        mock_coordinator2.mount = AsyncMock()

        mock_client = MagicMock(spec=CopilotClientWrapper)
        mock_client.is_healthy.return_value = True
        mock_client.close = AsyncMock()

        with patch(
            "amplifier_module_provider_github_copilot.CopilotClientWrapper",
            return_value=mock_client,
        ):
            cleanup1 = await provider_module.mount(mock_coordinator1)
            cleanup2 = await provider_module.mount(mock_coordinator2)

            assert provider_module._shared_client_refcount == 2

            # Cleanup first session
            assert cleanup1 is not None
            await cleanup1()

            # Client should still be alive (refcount > 0)
            assert provider_module._shared_client is mock_client
            assert provider_module._shared_client_refcount == 1
            mock_client.close.assert_not_called()

            # Second session cleanup releases last reference
            assert cleanup2 is not None
            await cleanup2()

            # Now client should be closed
            assert provider_module._shared_client_refcount == 0
            mock_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_session_state_not_shared_between_providers(self) -> None:
        """Provider-level state (config, pending_tasks) is not shared.

        Contract: sdk-boundary:Membrane:MUST:1
        """
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )

        mock_client = MagicMock(spec=CopilotClientWrapper)

        # Create two providers with injected client but different configs
        provider1 = GitHubCopilotProvider(
            config={"default_model": "gpt-4o"}, coordinator=None, client=mock_client
        )
        provider2 = GitHubCopilotProvider(
            config={"default_model": "claude-sonnet-4"}, coordinator=None, client=mock_client
        )

        # Same shared client
        assert provider1._client is provider2._client

        # But independent instances
        assert provider1 is not provider2

        # Independent config
        assert provider1.config["default_model"] == "gpt-4o"
        assert provider2.config["default_model"] == "claude-sonnet-4"

        # Independent pending task sets
        assert provider1._pending_emit_tasks is not provider2._pending_emit_tasks

    @pytest.mark.asyncio
    async def test_idempotent_cleanup_calls_safe(self) -> None:
        """Calling cleanup() multiple times on same session is safe.

        Contract: sdk-boundary:Membrane:MUST:1
        """
        import amplifier_module_provider_github_copilot as provider_module

        mock_coordinator = MagicMock()
        mock_coordinator.mount = AsyncMock()

        mock_client = MagicMock(spec=CopilotClientWrapper)
        mock_client.is_healthy.return_value = True
        mock_client.close = AsyncMock()

        with patch(
            "amplifier_module_provider_github_copilot.CopilotClientWrapper",
            return_value=mock_client,
        ):
            cleanup = await provider_module.mount(mock_coordinator)
            assert cleanup is not None

            # First cleanup
            await cleanup()
            assert provider_module._shared_client_refcount == 0

            # Redundant cleanups should be safe (no negative refcount)
            await cleanup()
            await cleanup()

            # Refcount should never go negative
            assert provider_module._shared_client_refcount == 0
