# NOTE: mock_session objects are intentionally unspecced - they simulate SDK session cleanup
# behavior (disconnect() success/failure, raise_on_disconnect). Custom behavior is needed per test.

"""Tests for session disconnect failure handling.

Contract: Observability — failures should be tracked for operational awareness.

These tests verify:
1. Disconnect failures increment a counter
2. Repeated failures (>3) escalate to logger.error
3. Resource leak warning is logged after threshold
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from amplifier_module_provider_github_copilot.sdk_adapter.client import (
    CopilotClientWrapper,
)


class TestDisconnectFailureCounter:
    """Tests for disconnect failure counting.

    Contract: Observability — failures should be tracked.
    """

    @pytest.mark.asyncio
    async def test_disconnect_failure_increments_counter(self) -> None:
        """Disconnect failure should increment _disconnect_failures counter.

        AC: Disconnect failures increment a failure counter.
        """
        # Create mock SDK client
        mock_sdk_client = MagicMock()
        mock_session = MagicMock()
        mock_session.disconnect = AsyncMock(side_effect=Exception("disconnect failed"))
        mock_sdk_client.create_session = AsyncMock(return_value=mock_session)

        wrapper = CopilotClientWrapper(sdk_client=mock_sdk_client)

        # Initial counter should be 0
        assert wrapper._disconnect_failures == 0  # type: ignore[reportPrivateUsage]  # Testing internal state

        # Use session (disconnect will fail)
        async with wrapper.session(model="gpt-4o"):
            pass  # Session body completes

        # Counter should be incremented
        assert wrapper._disconnect_failures == 1  # type: ignore[reportPrivateUsage]  # Testing internal state

    @pytest.mark.asyncio
    async def test_multiple_disconnect_failures_accumulate(self) -> None:
        """Multiple disconnect failures should accumulate in counter."""
        mock_sdk_client = MagicMock()
        mock_session = MagicMock()
        mock_session.disconnect = AsyncMock(side_effect=Exception("disconnect failed"))
        mock_sdk_client.create_session = AsyncMock(return_value=mock_session)

        wrapper = CopilotClientWrapper(sdk_client=mock_sdk_client)

        # Three session uses with failed disconnects
        for _ in range(3):
            async with wrapper.session(model="gpt-4o"):
                pass

        assert wrapper._disconnect_failures == 3  # type: ignore[reportPrivateUsage]  # Testing internal state

    @pytest.mark.asyncio
    async def test_successful_disconnect_does_not_increment(self) -> None:
        """Successful disconnect should not increment counter."""
        mock_sdk_client = MagicMock()
        mock_session = MagicMock()
        mock_session.disconnect = AsyncMock(return_value=None)  # Success
        mock_sdk_client.create_session = AsyncMock(return_value=mock_session)

        wrapper = CopilotClientWrapper(sdk_client=mock_sdk_client)

        async with wrapper.session(model="gpt-4o"):
            pass

        assert wrapper._disconnect_failures == 0  # type: ignore[reportPrivateUsage]  # Testing internal state


class TestDisconnectFailureEscalation:
    """Tests for disconnect failure escalation.

    Contract: Observability — repeated failures escalate to error level.
    """

    @pytest.mark.asyncio
    async def test_escalates_to_error_after_threshold(self) -> None:
        """After >3 disconnect failures, should escalate to logger.error.

        AC: Repeated disconnect failures (>3) escalate to _logger.error.
        """
        mock_sdk_client = MagicMock()
        mock_session = MagicMock()
        mock_session.disconnect = AsyncMock(side_effect=Exception("disconnect failed"))
        mock_sdk_client.create_session = AsyncMock(return_value=mock_session)

        wrapper = CopilotClientWrapper(sdk_client=mock_sdk_client)

        with patch(
            "amplifier_module_provider_github_copilot.sdk_adapter.client.logger"
        ) as mock_logger:
            # First 3 failures - warning only
            for _ in range(3):
                async with wrapper.session(model="gpt-4o"):
                    pass

            # Should have warnings but no error yet
            assert mock_logger.warning.call_count == 3
            assert mock_logger.error.call_count == 0

            # 4th failure - should escalate to error
            async with wrapper.session(model="gpt-4o"):
                pass

            # Should now have error logged
            assert mock_logger.error.call_count == 1
            # Check error message mentions resource leak
            # Contract: sdk-protection:Session:MUST:3
            error_call_args = str(mock_logger.error.call_args)
            assert "potential resource leak" in error_call_args.lower()

    @pytest.mark.asyncio
    async def test_warning_logged_on_every_failure(self) -> None:
        """Every disconnect failure should log a warning."""
        mock_sdk_client = MagicMock()
        mock_session = MagicMock()
        mock_session.disconnect = AsyncMock(side_effect=Exception("disconnect failed"))
        mock_sdk_client.create_session = AsyncMock(return_value=mock_session)

        wrapper = CopilotClientWrapper(sdk_client=mock_sdk_client)

        with patch(
            "amplifier_module_provider_github_copilot.sdk_adapter.client.logger"
        ) as mock_logger:
            for _ in range(5):
                async with wrapper.session(model="gpt-4o"):
                    pass

            # 5 warnings should be logged (one per failure)
            assert mock_logger.warning.call_count == 5


class TestDisconnectFailureInit:
    """Tests for disconnect failure counter initialization."""

    def test_counter_initialized_to_zero(self) -> None:
        """Counter should be initialized to 0 in __init__."""
        wrapper = CopilotClientWrapper()
        assert wrapper._disconnect_failures == 0  # type: ignore[reportPrivateUsage]  # Testing internal state


class TestDisconnectTimeout:
    """Tests that session disconnect has a timeout to prevent indefinite hangs.

    Contract: sdk-boundary:BinaryResolution (implies responsible resource cleanup)
    P2 performance fix: disconnect() without timeout can hang indefinitely.
    """

    @pytest.mark.asyncio
    async def test_disconnect_timeout_raises_internally_not_to_caller(
        self,
    ) -> None:
        """When disconnect() hangs past timeout, error is logged but caller not raised.

        The session context manager MUST NOT propagate disconnect timeout to caller.
        The timeout is sourced from config/_sdk_protection.py session.disconnect_timeout_seconds.
        Contract: sdk-protection:Session:MUST:3
        """
        import asyncio
        from unittest.mock import MagicMock

        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        async def slow_disconnect() -> None:
            await asyncio.sleep(999)  # Simulates indefinite hang

        mock_sdk_client = MagicMock()
        mock_session = MagicMock()
        mock_session.disconnect = slow_disconnect
        mock_session.session_id = "timeout-test"
        mock_sdk_client.create_session = AsyncMock(return_value=mock_session)

        wrapper = CopilotClientWrapper(sdk_client=mock_sdk_client)

        # Patch config to use a tiny timeout so the test runs fast.
        # Contract: sdk-protection:Session:MUST:3 — timeout sourced from YAML
        from amplifier_module_provider_github_copilot.config._sdk_protection import (
            SdkProtectionConfig,
            SessionProtectionConfig,
        )

        mock_config = SdkProtectionConfig(
            session=SessionProtectionConfig(disconnect_timeout_seconds=0.05)
        )
        with patch(
            "amplifier_module_provider_github_copilot.sdk_adapter.client.load_sdk_protection_config",
            return_value=mock_config,
        ):
            async with wrapper.session(model="gpt-4"):
                pass
        # Reaching here means session completed (did not hang)

    def test_disconnect_timeout_in_session_config(
        self,
    ) -> None:
        """disconnect_timeout_seconds must be present in SessionProtectionConfig.

        Contract: sdk-protection:Session:MUST:3
        Three-Medium: timeout policy lives in YAML, not as Python constant.
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_sdk_protection_config,
        )

        config = load_sdk_protection_config()
        assert isinstance(config.session.disconnect_timeout_seconds, float)
        assert config.session.disconnect_timeout_seconds > 0
