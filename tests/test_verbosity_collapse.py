"""Verbosity collapse regression tests.

Contract: contracts/observability.md — observability:Verbosity:MUST:1

Ensures the verbosity flag is a single 'raw' boolean, old names (raw_payloads,
raw_debug, debug) do not exist, and the emitted event payload uses 'raw' as
the key (not 'raw_request' or 'raw_response') when enabled.

Reference: drift-anthropic-ghcp-provider.md — gap-raw-naming (HIGH)
"""

from __future__ import annotations

import dataclasses
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# =============================================================================
# Contract field-name assertions (LISKOV: exact field, exact value, exact type)
# =============================================================================


class TestObservabilityConfigFieldNames:
    """observability:Verbosity:MUST:1 — single 'raw' flag, no tiered aliases."""

    def test_raw_field_exists_with_correct_name(self) -> None:
        """ObservabilityConfig has 'raw' field with default False.

        Contract: observability:Verbosity:MUST:1
        """
        from amplifier_module_provider_github_copilot.observability import ObservabilityConfig

        config = ObservabilityConfig()
        field_names = {f.name for f in dataclasses.fields(config)}
        assert "raw" in field_names
        assert config.raw is False

    def test_raw_payloads_field_does_not_exist(self) -> None:
        """ObservabilityConfig must NOT have 'raw_payloads' (banned old name).

        Contract: observability:Verbosity:MUST:1
        """
        from amplifier_module_provider_github_copilot.observability import ObservabilityConfig

        field_names = {f.name for f in dataclasses.fields(ObservabilityConfig)}
        assert "raw_payloads" not in field_names

    def test_raw_debug_field_does_not_exist(self) -> None:
        """ObservabilityConfig must NOT have 'raw_debug' (tiered alias forbidden).

        Contract: observability:Verbosity:MUST:1
        """
        from amplifier_module_provider_github_copilot.observability import ObservabilityConfig

        field_names = {f.name for f in dataclasses.fields(ObservabilityConfig)}
        assert "raw_debug" not in field_names

    def test_debug_field_does_not_exist(self) -> None:
        """ObservabilityConfig must NOT have 'debug' (tiered alias forbidden).

        Contract: observability:Verbosity:MUST:1
        """
        from amplifier_module_provider_github_copilot.observability import ObservabilityConfig

        field_names = {f.name for f in dataclasses.fields(ObservabilityConfig)}
        assert "debug" not in field_names

    def test_observability_config_is_frozen(self) -> None:
        """ObservabilityConfig must be frozen to prevent singleton mutation."""
        from amplifier_module_provider_github_copilot.observability import ObservabilityConfig

        config = ObservabilityConfig()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            config.raw = True  # type: ignore[misc]


# =============================================================================
# Provider config plumbing — raw flag read at __init__ time
# =============================================================================


class TestProviderRawFlagParsing:
    """_parse_raw_flag() strict boolean parsing (SCHNEIER: string inputs)."""

    def test_provider_stores_raw_false_by_default(self) -> None:
        """Provider._raw defaults to False when config has no 'raw' key."""
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider(config={})
        assert provider._raw is False

    def test_provider_stores_raw_true_when_set(self) -> None:
        """Provider._raw is True when config['raw']=True."""
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider(config={"raw": True})
        assert provider._raw is True

    def test_string_false_not_treated_as_true(self) -> None:
        """config={'raw': 'false'} must NOT enable raw — string 'false' is falsy intent.

        Protects against bool('false') == True Python footgun.
        """
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider(config={"raw": "false"})
        assert provider._raw is False

    def test_string_zero_not_treated_as_true(self) -> None:
        """config={'raw': '0'} must NOT enable raw."""
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider(config={"raw": "0"})
        assert provider._raw is False

    def test_string_true_is_treated_as_true(self) -> None:
        """config={'raw': 'true'} should enable raw (common YAML/env-var pattern)."""
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider(config={"raw": "true"})
        assert provider._raw is True


# =============================================================================
# Emitted event payload shape — via full provider.complete() path
# (BECK: test through complete(), assert old keys absent AND exact value)
# =============================================================================


@pytest.fixture()
def mock_coordinator() -> MagicMock:
    """Mock coordinator that captures emitted events."""
    coordinator = MagicMock()
    coordinator.hooks = MagicMock()
    coordinator.hooks.emit = AsyncMock()
    return coordinator


@pytest.fixture()
def sample_chat_request() -> MagicMock:
    """Minimal ChatRequest mock."""
    req = MagicMock()
    req.messages = [MagicMock(role="user", content="Hello")]
    req.model = "gpt-4o"
    return req


def _extract_event(
    coordinator: MagicMock, event_name: str
) -> dict[str, Any]:
    """Return the payload dict from the first matching emitted event."""
    calls = coordinator.hooks.emit.call_args_list
    for call in calls:
        args = call[0]
        if args and args[0] == event_name:
            return args[1]
    raise AssertionError(f"Event '{event_name}' was not emitted. Calls: {calls}")


class TestRawPayloadKeysInEvents:
    """raw=True emits 'raw' key; default emits nothing; old keys never appear.

    Contract: observability:Verbosity:MUST:1
    """

    @pytest.mark.asyncio
    async def test_raw_true_adds_raw_key_to_llm_request(
        self,
        mock_coordinator: MagicMock,
        sample_chat_request: MagicMock,
    ) -> None:
        """raw=True: llm:request event contains 'raw' key, not 'raw_request'."""
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from amplifier_module_provider_github_copilot.streaming import StreamingAccumulator

        provider = GitHubCopilotProvider(
            config={"raw": True, "use_streaming": False},
            coordinator=mock_coordinator,
        )

        async def _mock_execute(
            *args: Any, accumulator: StreamingAccumulator, **kwargs: Any
        ) -> None:
            pass

        provider._execute_sdk_completion = _mock_execute  # type: ignore[method-assign]

        await provider.complete(sample_chat_request, model="gpt-4o")  # type: ignore[arg-type]

        data = _extract_event(mock_coordinator, "llm:request")
        assert "raw" in data, f"Expected 'raw' key in llm:request payload, got: {list(data)}"
        assert "raw_request" not in data, "Old key 'raw_request' must not appear"

    @pytest.mark.asyncio
    async def test_raw_true_adds_raw_key_to_llm_response(
        self,
        mock_coordinator: MagicMock,
        sample_chat_request: MagicMock,
    ) -> None:
        """raw=True: llm:response event contains 'raw' key, not 'raw_response'."""
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from amplifier_module_provider_github_copilot.streaming import StreamingAccumulator

        provider = GitHubCopilotProvider(
            config={"raw": True, "use_streaming": False},
            coordinator=mock_coordinator,
        )

        async def _mock_execute(
            *args: Any, accumulator: StreamingAccumulator, **kwargs: Any
        ) -> None:
            pass

        provider._execute_sdk_completion = _mock_execute  # type: ignore[method-assign]

        await provider.complete(sample_chat_request, model="gpt-4o")  # type: ignore[arg-type]

        data = _extract_event(mock_coordinator, "llm:response")
        assert "raw" in data, f"Expected 'raw' key in llm:response payload, got: {list(data)}"
        assert "raw_response" not in data, "Old key 'raw_response' must not appear"

    @pytest.mark.asyncio
    async def test_raw_false_default_no_raw_key_in_events(
        self,
        mock_coordinator: MagicMock,
        sample_chat_request: MagicMock,
    ) -> None:
        """Default config (raw not set): neither emitted event contains 'raw' key."""
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from amplifier_module_provider_github_copilot.streaming import StreamingAccumulator

        provider = GitHubCopilotProvider(
            config={"use_streaming": False},
            coordinator=mock_coordinator,
        )

        async def _mock_execute(
            *args: Any, accumulator: StreamingAccumulator, **kwargs: Any
        ) -> None:
            pass

        provider._execute_sdk_completion = _mock_execute  # type: ignore[method-assign]

        await provider.complete(sample_chat_request, model="gpt-4o")  # type: ignore[arg-type]

        req_data = _extract_event(mock_coordinator, "llm:request")
        resp_data = _extract_event(mock_coordinator, "llm:response")
        assert "raw" not in req_data, "Default config must NOT include 'raw' in llm:request"
        assert "raw" not in resp_data, "Default config must NOT include 'raw' in llm:response"

    @pytest.mark.asyncio
    async def test_string_false_config_no_raw_key(
        self,
        mock_coordinator: MagicMock,
        sample_chat_request: MagicMock,
    ) -> None:
        """config={'raw': 'false'} must NOT enable raw payloads.

        SCHNEIER: bool('false')==True footgun prevention.
        """
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from amplifier_module_provider_github_copilot.streaming import StreamingAccumulator

        provider = GitHubCopilotProvider(
            config={"raw": "false", "use_streaming": False},
            coordinator=mock_coordinator,
        )

        async def _mock_execute(
            *args: Any, accumulator: StreamingAccumulator, **kwargs: Any
        ) -> None:
            pass

        provider._execute_sdk_completion = _mock_execute  # type: ignore[method-assign]

        await provider.complete(sample_chat_request, model="gpt-4o")  # type: ignore[arg-type]

        req_data = _extract_event(mock_coordinator, "llm:request")
        resp_data = _extract_event(mock_coordinator, "llm:response")
        assert "raw" not in req_data
        assert "raw" not in resp_data

    @pytest.mark.asyncio
    async def test_raw_flag_isolation_across_provider_instances(
        self,
        sample_chat_request: MagicMock,
    ) -> None:
        """raw=True on one provider instance must not affect a second instance.

        SCHNEIER: no lru_cache singleton bleed from raw=True into raw=False provider.
        """
        from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider
        from amplifier_module_provider_github_copilot.streaming import StreamingAccumulator

        async def _mock_execute(
            *args: Any, accumulator: StreamingAccumulator, **kwargs: Any
        ) -> None:
            pass

        # First provider: raw=True
        coord1: MagicMock = MagicMock()
        coord1.hooks = MagicMock()
        coord1.hooks.emit = AsyncMock()
        p1 = GitHubCopilotProvider(config={"raw": True, "use_streaming": False}, coordinator=coord1)
        p1._execute_sdk_completion = _mock_execute  # type: ignore[method-assign]
        await p1.complete(sample_chat_request, model="gpt-4o")  # type: ignore[arg-type]

        # Second provider: no raw flag (default)
        coord2: MagicMock = MagicMock()
        coord2.hooks = MagicMock()
        coord2.hooks.emit = AsyncMock()
        p2 = GitHubCopilotProvider(config={"use_streaming": False}, coordinator=coord2)
        p2._execute_sdk_completion = _mock_execute  # type: ignore[method-assign]
        await p2.complete(sample_chat_request, model="gpt-4o")  # type: ignore[arg-type]

        # p1 events have 'raw' key
        req1 = _extract_event(coord1, "llm:request")
        assert "raw" in req1, "p1 (raw=True) must emit raw key"

        # p2 events must NOT have 'raw' key (no bleed from p1)
        req2 = _extract_event(coord2, "llm:request")
        assert "raw" not in req2, "p2 (raw=False) must not be contaminated by p1"
