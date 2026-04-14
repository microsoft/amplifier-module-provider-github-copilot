"""Coverage tests for sdk_adapter/client.py missing branches.

Covers:
- Lines 123-127: _load_error_config_once() exception handler (graceful fallback)
- Line 272: Double-check inside asyncio.Lock in _ensure_client_initialized

Contract: sdk-boundary:Membrane:MUST:1
Contract: sdk-protection:Subprocess:MUST:6
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# _load_error_config_once exception handler (lines 123-127)
# ---------------------------------------------------------------------------


class TestLoadErrorConfigOnceFallback:
    """_load_error_config_once returns empty ErrorConfig when load_error_config raises."""

    def test_exception_in_load_error_config_returns_default(self) -> None:
        """Exception in load_error_config() → graceful fallback to empty ErrorConfig.

        Contract: behaviors:ErrorConfig:SHOULD:1 — graceful degradation
        Lines 123-127 in sdk_adapter/client.py
        """
        from amplifier_module_provider_github_copilot.error_translation import ErrorConfig
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            _load_error_config_once,  # pyright: ignore[reportPrivateUsage]
        )

        with patch(
            "amplifier_module_provider_github_copilot.error_translation.load_error_config",
            side_effect=Exception("load_error_config failed unexpectedly"),
        ):
            result = _load_error_config_once()

        # Should return empty ErrorConfig, not raise
        assert isinstance(result, ErrorConfig)
        # Empty config has no mappings
        assert result.mappings == []

    def test_load_error_config_success_returns_config(self) -> None:
        """When load_error_config() succeeds, returns its result.

        Happy path — verifies the try branch (line 122).

        # Contract: error-hierarchy:config:MUST:2
        """
        from amplifier_module_provider_github_copilot.error_translation import ErrorConfig
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            _load_error_config_once,  # pyright: ignore[reportPrivateUsage]
        )

        # Call without patching — relies on real file existing in package
        result = _load_error_config_once()

        assert isinstance(result, ErrorConfig)
        # Real config has mappings loaded from errors.yaml
        assert len(result.mappings) > 0


# ---------------------------------------------------------------------------
# Double-check inside asyncio.Lock in _ensure_client_initialized (line 272)
# ---------------------------------------------------------------------------


class TestDoubleCheckInsideLock:
    """Second concurrent coroutine returns client via double-check inside lock."""

    @pytest.mark.asyncio
    async def test_double_check_returns_client_set_by_concurrent_coroutine(self) -> None:
        """When client is set between pre-lock check and inside-lock double-check, returns it.

        This simulates the race condition where two coroutines call _ensure_client_initialized
        concurrently. The second one finds the client already set by the first.

        Contract: sdk-boundary:Config:MUST:1 — singleton-safe lazy init
        Line 272 in sdk_adapter/client.py
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        # Create a subclass that simulates the race:
        # _get_client() returns None on first call (pre-lock check),
        # then returns a mock client on the second call (inside-lock double-check).
        # This mimics another coroutine having initialized between the two checks.

        mock_client = MagicMock(spec=object)  # Minimal spec — SDK client unavailable in test mode

        class RaceSimulatingWrapper(CopilotClientWrapper):
            _call_count: int = 0

            def _get_client(self) -> Any | None:
                self._call_count += 1
                if self._call_count == 1:
                    return None  # Pre-lock check: client not set yet
                else:
                    return mock_client  # Inside-lock double-check: already set!

        wrapper = RaceSimulatingWrapper()

        result = await wrapper._ensure_client_initialized("test")  # pyright: ignore[reportPrivateUsage]

        # Should receive the client found in double-check, not try to init
        assert result is mock_client
        # _call_count == 2 confirms double-check fired
        assert wrapper._call_count == 2  # pyright: ignore[reportPrivateUsage]

    @pytest.mark.asyncio
    async def test_ensure_client_raises_when_stopped(self) -> None:
        """_ensure_client_initialized raises RuntimeError when wrapper is stopped.

        Contract: sdk-protection:Subprocess:MUST:6 — Guard re-init after stop
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        wrapper = CopilotClientWrapper()
        wrapper._stopped = True  # pyright: ignore[reportPrivateUsage]

        with pytest.raises(RuntimeError, match="stopped"):
            await wrapper._ensure_client_initialized("test")  # pyright: ignore[reportPrivateUsage]


# ---------------------------------------------------------------------------
# is_healthy() — returns False when stopped (line 97 area)
# ---------------------------------------------------------------------------


class TestIsHealthyWhenStopped:
    """is_healthy() returns False after client is stopped."""

    def test_is_healthy_returns_false_when_stopped(self) -> None:
        """CopilotClientWrapper.is_healthy() returns False after stop.

        Contract: sdk-boundary:Config:MUST:1 — health check
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        wrapper = CopilotClientWrapper()
        assert wrapper.is_healthy() is True  # Initially healthy

        wrapper._stopped = True  # pyright: ignore[reportPrivateUsage]
        assert wrapper.is_healthy() is False  # Returns False after stop

    @pytest.mark.asyncio
    async def test_close_sets_stopped_flag(self) -> None:
        """close() sets _stopped=True, making is_healthy() return False.

        Contract: sdk-boundary:Lifecycle:MUST:1
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        wrapper = CopilotClientWrapper()
        assert wrapper.is_healthy() is True

        await wrapper.close()

        assert wrapper.is_healthy() is False


# ---------------------------------------------------------------------------
# _get_error_config() lazy init
# ---------------------------------------------------------------------------


class TestGetErrorConfigLazyInit:
    """_get_error_config() initializes on first call, returns cached on subsequent."""

    def test_get_error_config_lazy_init(self) -> None:
        """_get_error_config() creates error config on first call.

        Lines 123-126 (happy path) in sdk_adapter/client.py

        # Contract: error-hierarchy:config:MUST:2
        """
        from amplifier_module_provider_github_copilot.error_translation import ErrorConfig
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        wrapper = CopilotClientWrapper()
        assert wrapper._error_config is None  # pyright: ignore[reportPrivateUsage]  # Not initialized yet

        config = wrapper._get_error_config()  # pyright: ignore[reportPrivateUsage]

        assert isinstance(config, ErrorConfig)
        assert wrapper._error_config is config  # pyright: ignore[reportPrivateUsage]  # Cached

    def test_get_error_config_returns_cached_on_second_call(self) -> None:
        """_get_error_config() returns same cached instance on second call."""
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        wrapper = CopilotClientWrapper()
        config1 = wrapper._get_error_config()  # pyright: ignore[reportPrivateUsage]
        config2 = wrapper._get_error_config()  # pyright: ignore[reportPrivateUsage]

        assert config1 is config2  # Same object returned from cache


# ---------------------------------------------------------------------------
# deny_permission_request happy path — PermissionRequestResult branch (line 97)
# ---------------------------------------------------------------------------


class TestDenyPermissionRequestHappyPath:
    """deny_permission_request() returns PermissionRequestResult when SDK types available."""

    def test_returns_permission_request_result_when_sdk_available(self) -> None:
        """deny_permission_request returns PermissionRequestResult when SDK >= 0.1.28.

        Contract: deny-destroy:PermissionRequest:MUST:2
        Line 97 in sdk_adapter/client.py — PermissionRequestResult branch

        In test mode (SKIP_SDK_CHECK=1), _imports.PermissionRequestResult is None.
        Patching it to a real class exercises the if-branch at line 96-100.
        """
        from unittest.mock import MagicMock, patch

        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            deny_permission_request,
        )

        class _MockPermissionRequestResult:
            """Minimal stand-in for SDK PermissionRequestResult."""

            def __init__(self, *, kind: str, message: str) -> None:
                self.kind = kind
                self.message = message

        mock_request = MagicMock(spec=object)  # Minimal spec — permission request arg is unused

        with patch(
            "amplifier_module_provider_github_copilot.sdk_adapter._imports.PermissionRequestResult",
            _MockPermissionRequestResult,
        ):
            result = deny_permission_request(mock_request)

        assert isinstance(result, _MockPermissionRequestResult)
        assert result.kind == "denied-by-rules"
        assert result.message == "Amplifier orchestrator controls all operations"
