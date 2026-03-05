"""Tests for CP-V Provider Verbosity Collapse (Task 13d).

Verifies:
- `raw: true` config adds `raw` field to base llm:request and llm:response events
- `raw: false` (default) produces no `raw` field in those events
- :debug and :raw tiered events are never emitted
- Config flag `raw` replaces deprecated debug/raw_debug/debug_truncate_length flags
- Copilot-specific: uses underscore-prefixed `_raw` attribute
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock, patch

import pytest

from amplifier_module_provider_github_copilot.client import CopilotClientWrapper
from amplifier_module_provider_github_copilot.provider import CopilotSdkProvider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeHooks:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))

    def emitted_names(self) -> list[str]:
        return [name for name, _ in self.events]

    def payload_for(self, event_name: str) -> dict | None:
        for name, payload in self.events:
            if name == event_name:
                return payload
        return None


class FakeCoordinator:
    def __init__(self):
        self.hooks = FakeHooks()


def _make_provider(*, raw: bool = False) -> CopilotSdkProvider:
    config: dict = {"model": "claude-opus-4.5", "use_streaming": False, "max_retries": 0}
    if raw:
        config["raw"] = True
    return CopilotSdkProvider(
        api_key=None,
        config=config,
        coordinator=FakeCoordinator(),  # type: ignore[arg-type]
    )


def _make_mock_session_cm():
    """Return a mock session async context manager and a send_and_wait mock."""
    mock_session = AsyncMock()
    mock_session.session_id = "test-session"
    mock_session.destroy = AsyncMock()

    mock_response = Mock()
    mock_response.data = Mock()
    mock_response.data.content = "Hello"
    mock_response.data.tool_requests = None
    mock_response.data.input_tokens = 10
    mock_response.data.output_tokens = 5

    @asynccontextmanager
    async def mock_create_session(self, **kwargs):
        yield mock_session

    return mock_create_session, mock_session, mock_response


# ---------------------------------------------------------------------------
# Config flag tests
# ---------------------------------------------------------------------------


class TestRawConfigFlag:
    def test_raw_flag_defaults_to_false(self):
        """Without config, self._raw should be False."""
        provider = CopilotSdkProvider(api_key=None, config={})
        assert provider._raw is False  # type: ignore[attr-defined]

    def test_raw_flag_true_when_configured(self):
        """With raw=True in config, self._raw should be True."""
        provider = CopilotSdkProvider(api_key=None, config={"raw": True})
        assert provider._raw is True  # type: ignore[attr-defined]

    def test_raw_flag_false_when_explicitly_set(self):
        """With raw=False in config, self._raw should be False."""
        provider = CopilotSdkProvider(api_key=None, config={"raw": False})
        assert provider._raw is False  # type: ignore[attr-defined]

    def test_debug_flag_removed(self):
        """The old `_debug` flag should not exist on the provider."""
        provider = CopilotSdkProvider(api_key=None, config={})
        assert not hasattr(provider, "_debug"), (
            "Old `_debug` config flag must be removed; use `_raw` instead"
        )

    def test_raw_debug_flag_removed(self):
        """The old `_raw_debug` flag should not exist on the provider."""
        provider = CopilotSdkProvider(api_key=None, config={})
        assert not hasattr(provider, "_raw_debug"), (
            "Old `_raw_debug` config flag must be removed; use `_raw` instead"
        )

    def test_debug_truncate_length_removed(self):
        """The old `_debug_truncate_length` flag should not exist on the provider."""
        provider = CopilotSdkProvider(api_key=None, config={})
        assert not hasattr(provider, "_debug_truncate_length"), (
            "Old `_debug_truncate_length` config flag must be removed"
        )

    def test_truncate_values_method_removed(self):
        """The `_truncate_values` method should not exist on the provider."""
        provider = CopilotSdkProvider(api_key=None, config={})
        assert not hasattr(provider, "_truncate_values"), (
            "`_truncate_values` method must be removed"
        )


# ---------------------------------------------------------------------------
# llm:request event tests
# ---------------------------------------------------------------------------


class TestLLMRequestEvent:
    @pytest.mark.asyncio
    async def test_base_request_event_emitted_without_raw(self):
        """llm:request is always emitted, even when raw=False."""
        provider = _make_provider(raw=False)
        mock_create_session, mock_session, mock_response = _make_mock_session_cm()

        with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
            with patch.object(
                CopilotClientWrapper, "send_and_wait", new_callable=AsyncMock
            ) as mock_send:
                mock_send.return_value = mock_response
                with patch.object(
                    provider,
                    "_model_supports_reasoning",
                    new_callable=AsyncMock,
                    return_value=False,
                ):
                    await provider.complete(
                        {"messages": [{"role": "user", "content": "Hello"}]},
                        model="gpt-4",
                    )

        hooks = provider._coordinator.hooks  # type: ignore[attr-defined]
        assert "llm:request" in hooks.emitted_names()

    @pytest.mark.asyncio
    async def test_request_event_has_no_raw_field_by_default(self):
        """llm:request payload should NOT have `raw` field when raw=False."""
        provider = _make_provider(raw=False)
        mock_create_session, mock_session, mock_response = _make_mock_session_cm()

        with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
            with patch.object(
                CopilotClientWrapper, "send_and_wait", new_callable=AsyncMock
            ) as mock_send:
                mock_send.return_value = mock_response
                with patch.object(
                    provider,
                    "_model_supports_reasoning",
                    new_callable=AsyncMock,
                    return_value=False,
                ):
                    await provider.complete(
                        {"messages": [{"role": "user", "content": "Hello"}]},
                        model="gpt-4",
                    )

        hooks = provider._coordinator.hooks  # type: ignore[attr-defined]
        payload = hooks.payload_for("llm:request")
        assert payload is not None
        assert "raw" not in payload, (
            "llm:request payload must not contain `raw` field when raw=False"
        )

    @pytest.mark.asyncio
    async def test_request_event_has_raw_field_when_raw_true(self):
        """llm:request payload should have `raw` field when raw=True."""
        provider = _make_provider(raw=True)
        mock_create_session, mock_session, mock_response = _make_mock_session_cm()

        with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
            with patch.object(
                CopilotClientWrapper, "send_and_wait", new_callable=AsyncMock
            ) as mock_send:
                mock_send.return_value = mock_response
                with patch.object(
                    provider,
                    "_model_supports_reasoning",
                    new_callable=AsyncMock,
                    return_value=False,
                ):
                    await provider.complete(
                        {"messages": [{"role": "user", "content": "Hello"}]},
                        model="gpt-4",
                    )

        hooks = provider._coordinator.hooks  # type: ignore[attr-defined]
        payload = hooks.payload_for("llm:request")
        assert payload is not None
        assert "raw" in payload, "llm:request payload must contain `raw` field when raw=True"

    @pytest.mark.asyncio
    async def test_request_raw_field_is_dict(self):
        """`raw` field in llm:request should be a dict (the full params)."""
        provider = _make_provider(raw=True)
        mock_create_session, mock_session, mock_response = _make_mock_session_cm()

        with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
            with patch.object(
                CopilotClientWrapper, "send_and_wait", new_callable=AsyncMock
            ) as mock_send:
                mock_send.return_value = mock_response
                with patch.object(
                    provider,
                    "_model_supports_reasoning",
                    new_callable=AsyncMock,
                    return_value=False,
                ):
                    await provider.complete(
                        {"messages": [{"role": "user", "content": "Hello"}]},
                        model="gpt-4",
                    )

        hooks = provider._coordinator.hooks  # type: ignore[attr-defined]
        payload = hooks.payload_for("llm:request")
        assert payload is not None
        assert isinstance(payload["raw"], dict)

    @pytest.mark.asyncio
    async def test_no_debug_request_event_emitted(self):
        """llm:request:debug must never be emitted."""
        provider = _make_provider(raw=True)
        mock_create_session, mock_session, mock_response = _make_mock_session_cm()

        with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
            with patch.object(
                CopilotClientWrapper, "send_and_wait", new_callable=AsyncMock
            ) as mock_send:
                mock_send.return_value = mock_response
                with patch.object(
                    provider,
                    "_model_supports_reasoning",
                    new_callable=AsyncMock,
                    return_value=False,
                ):
                    await provider.complete(
                        {"messages": [{"role": "user", "content": "Hello"}]},
                        model="gpt-4",
                    )

        hooks = provider._coordinator.hooks  # type: ignore[attr-defined]
        assert "llm:request:debug" not in hooks.emitted_names(), (
            "llm:request:debug must never be emitted after verbosity collapse"
        )

    @pytest.mark.asyncio
    async def test_no_raw_request_event_emitted(self):
        """llm:request:raw must never be emitted."""
        provider = _make_provider(raw=True)
        mock_create_session, mock_session, mock_response = _make_mock_session_cm()

        with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
            with patch.object(
                CopilotClientWrapper, "send_and_wait", new_callable=AsyncMock
            ) as mock_send:
                mock_send.return_value = mock_response
                with patch.object(
                    provider,
                    "_model_supports_reasoning",
                    new_callable=AsyncMock,
                    return_value=False,
                ):
                    await provider.complete(
                        {"messages": [{"role": "user", "content": "Hello"}]},
                        model="gpt-4",
                    )

        hooks = provider._coordinator.hooks  # type: ignore[attr-defined]
        assert "llm:request:raw" not in hooks.emitted_names(), (
            "llm:request:raw must never be emitted after verbosity collapse"
        )

    @pytest.mark.asyncio
    async def test_request_event_base_fields_present(self):
        """llm:request should always have provider, model, message_count fields."""
        provider = _make_provider(raw=False)
        mock_create_session, mock_session, mock_response = _make_mock_session_cm()

        with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
            with patch.object(
                CopilotClientWrapper, "send_and_wait", new_callable=AsyncMock
            ) as mock_send:
                mock_send.return_value = mock_response
                with patch.object(
                    provider,
                    "_model_supports_reasoning",
                    new_callable=AsyncMock,
                    return_value=False,
                ):
                    await provider.complete(
                        {"messages": [{"role": "user", "content": "Hello"}]},
                        model="gpt-4",
                    )

        hooks = provider._coordinator.hooks  # type: ignore[attr-defined]
        payload = hooks.payload_for("llm:request")
        assert payload is not None
        assert payload["provider"] == "github-copilot"
        assert "model" in payload
        assert "message_count" in payload


# ---------------------------------------------------------------------------
# llm:response event tests
# ---------------------------------------------------------------------------


class TestLLMResponseEvent:
    @pytest.mark.asyncio
    async def test_base_response_event_emitted_without_raw(self):
        """llm:response is always emitted, even when raw=False."""
        provider = _make_provider(raw=False)
        mock_create_session, mock_session, mock_response = _make_mock_session_cm()

        with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
            with patch.object(
                CopilotClientWrapper, "send_and_wait", new_callable=AsyncMock
            ) as mock_send:
                mock_send.return_value = mock_response
                with patch.object(
                    provider,
                    "_model_supports_reasoning",
                    new_callable=AsyncMock,
                    return_value=False,
                ):
                    await provider.complete(
                        {"messages": [{"role": "user", "content": "Hello"}]},
                        model="gpt-4",
                    )

        hooks = provider._coordinator.hooks  # type: ignore[attr-defined]
        assert "llm:response" in hooks.emitted_names()

    @pytest.mark.asyncio
    async def test_response_event_has_no_raw_field_by_default(self):
        """llm:response payload should NOT have `raw` field when raw=False."""
        provider = _make_provider(raw=False)
        mock_create_session, mock_session, mock_response = _make_mock_session_cm()

        with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
            with patch.object(
                CopilotClientWrapper, "send_and_wait", new_callable=AsyncMock
            ) as mock_send:
                mock_send.return_value = mock_response
                with patch.object(
                    provider,
                    "_model_supports_reasoning",
                    new_callable=AsyncMock,
                    return_value=False,
                ):
                    await provider.complete(
                        {"messages": [{"role": "user", "content": "Hello"}]},
                        model="gpt-4",
                    )

        hooks = provider._coordinator.hooks  # type: ignore[attr-defined]
        payload = hooks.payload_for("llm:response")
        assert payload is not None
        assert "raw" not in payload, (
            "llm:response payload must not contain `raw` field when raw=False"
        )

    @pytest.mark.asyncio
    async def test_response_event_has_raw_field_when_raw_true(self):
        """llm:response payload should have `raw` field when raw=True."""
        provider = _make_provider(raw=True)
        mock_create_session, mock_session, mock_response = _make_mock_session_cm()

        with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
            with patch.object(
                CopilotClientWrapper, "send_and_wait", new_callable=AsyncMock
            ) as mock_send:
                mock_send.return_value = mock_response
                with patch.object(
                    provider,
                    "_model_supports_reasoning",
                    new_callable=AsyncMock,
                    return_value=False,
                ):
                    await provider.complete(
                        {"messages": [{"role": "user", "content": "Hello"}]},
                        model="gpt-4",
                    )

        hooks = provider._coordinator.hooks  # type: ignore[attr-defined]
        payload = hooks.payload_for("llm:response")
        assert payload is not None
        assert "raw" in payload, "llm:response payload must contain `raw` field when raw=True"

    @pytest.mark.asyncio
    async def test_response_raw_field_is_dict(self):
        """`raw` field in llm:response should be a dict."""
        provider = _make_provider(raw=True)
        mock_create_session, mock_session, mock_response = _make_mock_session_cm()

        with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
            with patch.object(
                CopilotClientWrapper, "send_and_wait", new_callable=AsyncMock
            ) as mock_send:
                mock_send.return_value = mock_response
                with patch.object(
                    provider,
                    "_model_supports_reasoning",
                    new_callable=AsyncMock,
                    return_value=False,
                ):
                    await provider.complete(
                        {"messages": [{"role": "user", "content": "Hello"}]},
                        model="gpt-4",
                    )

        hooks = provider._coordinator.hooks  # type: ignore[attr-defined]
        payload = hooks.payload_for("llm:response")
        assert payload is not None
        assert isinstance(payload["raw"], dict)

    @pytest.mark.asyncio
    async def test_no_debug_response_event_emitted(self):
        """llm:response:debug must never be emitted."""
        provider = _make_provider(raw=True)
        mock_create_session, mock_session, mock_response = _make_mock_session_cm()

        with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
            with patch.object(
                CopilotClientWrapper, "send_and_wait", new_callable=AsyncMock
            ) as mock_send:
                mock_send.return_value = mock_response
                with patch.object(
                    provider,
                    "_model_supports_reasoning",
                    new_callable=AsyncMock,
                    return_value=False,
                ):
                    await provider.complete(
                        {"messages": [{"role": "user", "content": "Hello"}]},
                        model="gpt-4",
                    )

        hooks = provider._coordinator.hooks  # type: ignore[attr-defined]
        assert "llm:response:debug" not in hooks.emitted_names(), (
            "llm:response:debug must never be emitted after verbosity collapse"
        )

    @pytest.mark.asyncio
    async def test_no_raw_response_event_emitted(self):
        """llm:response:raw must never be emitted."""
        provider = _make_provider(raw=True)
        mock_create_session, mock_session, mock_response = _make_mock_session_cm()

        with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
            with patch.object(
                CopilotClientWrapper, "send_and_wait", new_callable=AsyncMock
            ) as mock_send:
                mock_send.return_value = mock_response
                with patch.object(
                    provider,
                    "_model_supports_reasoning",
                    new_callable=AsyncMock,
                    return_value=False,
                ):
                    await provider.complete(
                        {"messages": [{"role": "user", "content": "Hello"}]},
                        model="gpt-4",
                    )

        hooks = provider._coordinator.hooks  # type: ignore[attr-defined]
        assert "llm:response:raw" not in hooks.emitted_names(), (
            "llm:response:raw must never be emitted after verbosity collapse"
        )

    @pytest.mark.asyncio
    async def test_response_event_base_fields_present(self):
        """llm:response should always have provider, model, status fields."""
        provider = _make_provider(raw=False)
        mock_create_session, mock_session, mock_response = _make_mock_session_cm()

        with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
            with patch.object(
                CopilotClientWrapper, "send_and_wait", new_callable=AsyncMock
            ) as mock_send:
                mock_send.return_value = mock_response
                with patch.object(
                    provider,
                    "_model_supports_reasoning",
                    new_callable=AsyncMock,
                    return_value=False,
                ):
                    await provider.complete(
                        {"messages": [{"role": "user", "content": "Hello"}]},
                        model="gpt-4",
                    )

        hooks = provider._coordinator.hooks  # type: ignore[attr-defined]
        payload = hooks.payload_for("llm:response")
        assert payload is not None
        assert payload["provider"] == "github-copilot"
        assert "model" in payload
        assert payload["status"] == "ok"


# ---------------------------------------------------------------------------
# No :debug/:raw events in any path
# ---------------------------------------------------------------------------


class TestNoTieredEvents:
    @pytest.mark.asyncio
    async def test_no_tiered_events_regardless_of_raw_flag(self):
        """No :debug or :raw tiered events should ever be emitted."""
        for raw_flag in [True, False]:
            provider = _make_provider(raw=raw_flag)
            mock_create_session, mock_session, mock_response = _make_mock_session_cm()

            with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
                with patch.object(
                    CopilotClientWrapper, "send_and_wait", new_callable=AsyncMock
                ) as mock_send:
                    mock_send.return_value = mock_response
                    with patch.object(
                        provider,
                        "_model_supports_reasoning",
                        new_callable=AsyncMock,
                        return_value=False,
                    ):
                        await provider.complete(
                            {"messages": [{"role": "user", "content": "Hello"}]},
                            model="gpt-4",
                        )

            hooks = provider._coordinator.hooks  # type: ignore[attr-defined]
            for event_name in hooks.emitted_names():
                assert not event_name.endswith(":debug"), f"Found tiered :debug event: {event_name}"
                assert not event_name.endswith(":raw"), f"Found tiered :raw event: {event_name}"
