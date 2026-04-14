"""Tests for config-driven timeout in the completion path.

Contract: contracts/behaviors.md (Three-Medium Architecture)

Tests verify that provider.complete() loads timeout from config and passes
it correctly to SDK completion calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


class TestModuleLevelCompleteTimeout:
    """Tests for module-level complete() timeout configuration."""

    @pytest.mark.asyncio
    async def test_complete_uses_config_timeout_not_hardcoded(self) -> None:
        """provider.complete() MUST use timeout from loaded config, not a hardcoded value.

        Contract: behaviors:ConfigLoading:MUST_NOT:5

        Verifies that:
        1. Provider loads timeout from ProviderConfig.defaults["timeout"]
        2. The loaded timeout is passed to _execute_sdk_completion
        3. No hardcoded default like 120.0 is used
        """
        import copy

        from amplifier_module_provider_github_copilot.config_loader import (
            ProviderConfig,
            load_models_config,
        )
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )
        from tests.fixtures.sdk_mocks import (
            MockCopilotClientWrapper,
            text_delta_event,
        )

        # Create mock client with events
        events = [text_delta_event("Hello, world!")]
        mock_client = MockCopilotClientWrapper(events=events)

        # Inject mock client into provider
        provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]

        # Create instance-local copy of config with modified timeout
        # Do NOT mutate the cached config from load_models_config()
        original_config = load_models_config()
        test_config = ProviderConfig(
            provider_id=original_config.provider_id,
            display_name=original_config.display_name,
            credential_env_vars=original_config.credential_env_vars,
            capabilities=original_config.capabilities,
            defaults=copy.deepcopy(original_config.defaults),
        )
        test_config.defaults["timeout"] = 999
        provider._provider_config = test_config

        # Create request using proper kernel type
        from amplifier_core import ChatRequest

        request = MagicMock(spec=ChatRequest)
        request.model = "claude-opus-4.5"
        request.messages = [MagicMock(role="user", content="test")]
        request.tools = None
        request.max_tokens = None
        request.temperature = None
        request.stop = None
        request.stream = None

        # Patch _execute_sdk_completion to capture the timeout argument
        captured_timeout: list[float] = []
        original_execute = provider._execute_sdk_completion

        async def capture_execute(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
            captured_timeout.append(kwargs.get("timeout", 0.0))
            return await original_execute(*args, **kwargs)

        provider._execute_sdk_completion = capture_execute  # type: ignore[method-assign]

        # Call the production complete() method
        await provider.complete(request)

        # Assert timeout from config (999) was passed, not a hardcoded value
        assert len(captured_timeout) == 1, "Expected _execute_sdk_completion to be called once"
        assert captured_timeout[0] == 999.0, (
            f"Expected timeout=999.0 from config, got {captured_timeout[0]}. "
            "Provider may be using hardcoded timeout instead of config."
        )


class TestTimeoutEnforcement:
    """Tests that config timeout is actually enforced at the asyncio boundary."""

    @pytest.mark.asyncio
    async def test_timeout_fires_as_llm_timeout_error(self) -> None:
        """Config timeout must propagate through asyncio.timeout() to LLMTimeoutError.

        Contract: behaviors:Config:MUST:2

        Verifies the FULL chain: config value → _execute_sdk_completion timeout arg →
        asyncio.timeout() → LLMTimeoutError translation. Unlike the wiring test above,
        this test verifies that _execute_sdk_completion actually USES the timeout.
        """
        import asyncio
        import copy

        from amplifier_core import ChatRequest

        from amplifier_module_provider_github_copilot.config_loader import (
            ProviderConfig,
            load_models_config,
        )
        from amplifier_module_provider_github_copilot.error_translation import (
            LLMTimeoutError,
        )
        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )
        from tests.fixtures.sdk_mocks import MockCopilotClientWrapper, MockSDKSession

        class _HangingSession(MockSDKSession):
            """Session that never delivers events — hangs until cancelled."""

            async def send(
                self,
                prompt: str,
                *,
                attachments: list[dict] | None = None,
            ) -> str:
                self.last_prompt = prompt
                await asyncio.sleep(60)  # cancelled by asyncio.timeout()
                return "message-id"  # unreachable

        mock_client = MockCopilotClientWrapper(session_class=_HangingSession)
        provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]

        # Set very short timeout — 100ms is enough to confirm enforcement without test slowness
        original_config = load_models_config()
        test_config = ProviderConfig(
            provider_id=original_config.provider_id,
            display_name=original_config.display_name,
            credential_env_vars=original_config.credential_env_vars,
            capabilities=original_config.capabilities,
            defaults=copy.deepcopy(original_config.defaults),
        )
        test_config.defaults["timeout"] = 0.1  # 100ms
        provider._provider_config = test_config

        request = MagicMock(spec=ChatRequest)
        request.model = "claude-opus-4.5"
        request.messages = [MagicMock(role="user", content="test")]
        request.tools = None
        request.max_tokens = None
        request.temperature = None
        request.stop = None
        request.stream = None

        with pytest.raises(LLMTimeoutError):
            await provider.complete(request)


class TestTimeoutConfigValue:
    """Tests for timeout configuration values."""

    def test_yaml_timeout_is_3600(self) -> None:
        """YAML timeout value is 3600 (1 hour for reasoning models).

        Contract: behaviors:Config:MUST:2
        Three-Medium: YAML is authoritative source.
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_models_config,
        )

        config = load_models_config()
        timeout = config.defaults["timeout"]
        assert timeout == 3600
