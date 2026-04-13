"""Tests for R7: sdk_adapter/client.py dependency injection.

Verifies that CopilotClientWrapper does not couple domain modules at import
time and that translate_error / safe_log are injectable.

Contract: sdk-boundary:Membrane:MUST:1 (spirit — no domain modules at module level)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from amplifier_module_provider_github_copilot.sdk_adapter.client import CopilotClientWrapper

# ---------------------------------------------------------------------------
# Helpers — minimal fake SDK shapes
# ---------------------------------------------------------------------------


class _FakeSDKSessionNormal:
    """SDK session that succeeds on send and disconnect."""

    session_id = "fake-session-id"

    async def send(self, prompt: str, **kwargs: Any) -> None:
        return None

    async def disconnect(self) -> None:
        return None


class _FakeSDKSessionFailDisconnect:
    """SDK session whose disconnect raises."""

    session_id = "fail-disconnect-session"

    async def send(self, prompt: str, **kwargs: Any) -> None:
        return None

    async def disconnect(self) -> None:
        raise RuntimeError("disconnect-boom")


class _FakeSDKClientRaisesOnSession:
    """SDK client whose create_session always raises."""

    async def create_session(self, **kwargs: Any) -> Any:
        raise ValueError("sdk-session-boom")

    async def list_models(self) -> list[Any]:
        return []


class _FakeSDKClientNormal:
    """SDK client that returns a working session."""

    async def create_session(self, **kwargs: Any) -> _FakeSDKSessionNormal:
        return _FakeSDKSessionNormal()

    async def list_models(self) -> list[Any]:
        return []


class _FakeSDKClientFailDisconnect:
    """SDK client whose session disconnects fail."""

    async def create_session(self, **kwargs: Any) -> _FakeSDKSessionFailDisconnect:
        return _FakeSDKSessionFailDisconnect()

    async def list_models(self) -> list[Any]:
        return []


class _FakeSDKClientRaisesOnListModels:
    """SDK client whose list_models raises."""

    async def create_session(self, **kwargs: Any) -> _FakeSDKSessionNormal:
        return _FakeSDKSessionNormal()

    async def list_models(self) -> list[Any]:
        raise ConnectionError("sdk-models-boom")


# ---------------------------------------------------------------------------
# R7-A: translate_error injection
# ---------------------------------------------------------------------------


class TestTranslateErrorInjection:
    """Injected translate_error is called on SDK session creation failure."""

    @pytest.mark.asyncio
    async def test_injected_translator_called_on_session_error(self) -> None:
        """Custom translator is invoked when create_session raises."""
        calls: list[Exception] = []

        def fake_translator(exc: Exception, config: Any) -> Exception:
            calls.append(exc)
            return RuntimeError(f"translated: {exc}")

        wrapper = CopilotClientWrapper(
            sdk_client=_FakeSDKClientRaisesOnSession(),
            _translate_error=fake_translator,
        )

        with pytest.raises(RuntimeError, match="translated: sdk-session-boom"):
            async with wrapper.session():
                pass  # pragma: no cover

        assert len(calls) == 1
        assert isinstance(calls[0], ValueError)

    @pytest.mark.asyncio
    async def test_injected_translator_called_on_list_models_error(self) -> None:
        """Custom translator is invoked when list_models raises."""
        calls: list[Exception] = []

        def fake_translator(exc: Exception, config: Any) -> Exception:
            calls.append(exc)
            return RuntimeError(f"translated: {exc}")

        wrapper = CopilotClientWrapper(
            sdk_client=_FakeSDKClientRaisesOnListModels(),
            _translate_error=fake_translator,
        )

        with pytest.raises(RuntimeError, match="translated: sdk-models-boom"):
            await wrapper.list_models()

        assert len(calls) == 1
        assert isinstance(calls[0], ConnectionError)

    @pytest.mark.asyncio
    async def test_default_translator_produces_domain_error(self) -> None:
        """Without injection, the real translator maps SDK errors to domain types."""
        from amplifier_core.llm_errors import LLMError

        wrapper = CopilotClientWrapper(sdk_client=_FakeSDKClientRaisesOnSession())

        with pytest.raises(LLMError):
            async with wrapper.session():
                pass  # pragma: no cover


# ---------------------------------------------------------------------------
# R7-B: safe_log injection
# ---------------------------------------------------------------------------


class TestSafeLogInjection:
    """Injected safe_log is called when disconnect or close raises."""

    @pytest.mark.asyncio
    async def test_injected_safe_log_called_on_disconnect_failure(self) -> None:
        """Custom safe_log is called when session disconnect raises."""
        log_calls: list[tuple[str, ...]] = []

        def fake_safe_log(message: str, *args: Any) -> tuple[Any, ...]:
            log_calls.append((message, *args))
            return (message, *args)

        wrapper = CopilotClientWrapper(
            sdk_client=_FakeSDKClientFailDisconnect(),
            _safe_log=fake_safe_log,
        )

        async with wrapper.session():
            pass  # disconnect fires on exit

        assert len(log_calls) >= 1
        # Disconnect error message should have been passed through safe_log
        assert any("disconnect" in str(call).lower() for call in log_calls)

    @pytest.mark.asyncio
    async def test_injected_safe_log_called_on_close_failure(self) -> None:
        """Custom safe_log is called when owned client stop() raises."""
        log_calls: list[tuple[str, ...]] = []

        def fake_safe_log(message: str, *args: Any) -> tuple[Any, ...]:
            log_calls.append((message, *args))
            return (message, *args)

        broken_client = MagicMock()
        broken_client.stop = AsyncMock(side_effect=RuntimeError("stop-boom"))

        wrapper = CopilotClientWrapper(_safe_log=fake_safe_log)
        wrapper._owned_client = broken_client  # simulate owned client

        await wrapper.close()

        assert len(log_calls) >= 1


# ---------------------------------------------------------------------------
# R7-C: kernel error provenance
# ---------------------------------------------------------------------------


class TestKernelErrorProvenance:
    """ConfigurationError and ProviderUnavailableError come from amplifier_core
    directly, not via error_translation re-export."""

    def test_configuration_error_is_kernel_type(self) -> None:
        """ConfigurationError in client.py is from amplifier_core.llm_errors."""
        from amplifier_core.llm_errors import ConfigurationError as KernelConfigError

        import amplifier_module_provider_github_copilot.sdk_adapter.client as client_mod

        # The module-level ConfigurationError must be the same class as the kernel type
        assert client_mod.ConfigurationError is KernelConfigError

    def test_provider_unavailable_error_is_kernel_type(self) -> None:
        """ProviderUnavailableError in client.py is from amplifier_core.llm_errors."""
        from amplifier_core.llm_errors import ProviderUnavailableError as KernelError

        import amplifier_module_provider_github_copilot.sdk_adapter.client as client_mod

        assert client_mod.ProviderUnavailableError is KernelError
