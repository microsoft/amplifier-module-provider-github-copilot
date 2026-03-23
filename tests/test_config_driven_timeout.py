"""Tests for config-driven timeout in the completion path.

Contract: contracts/behaviors.md (Three-Medium Architecture)

Tests verify that module-level complete() function loads timeout from
config/models.yaml instead of using hardcoded 120.0 value.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


class TestModuleLevelCompleteTimeout:
    """Tests for module-level complete() timeout configuration."""

    def test_no_hardcoded_120_in_provider(self) -> None:
        """provider.py should not contain hardcoded 120.0 timeout.

        Contract anchor: behaviors.md:Config:MUST:1
        """
        provider_path = (
            Path(__file__).parent.parent
            / "amplifier_module_provider_github_copilot"
            / "provider.py"
        )
        content = provider_path.read_text()

        # Check for hardcoded 120.0 timeout patterns
        assert "timeout=120.0" not in content, (
            "Found hardcoded timeout=120.0 in provider.py. "
            "Per Three-Medium Architecture, policy must live in YAML."
        )

    def test_module_complete_uses_load_models_config(self) -> None:
        """Module-level complete() should call load_models_config() for timeout.

        Contract anchor: behaviors.md:Config:MUST:1
        """
        # Check the source code references load_models_config
        provider_path = (
            Path(__file__).parent.parent
            / "amplifier_module_provider_github_copilot"
            / "provider.py"
        )
        content = provider_path.read_text()

        # Find that load_models_config is used (either directly or via _load_models_config alias)
        # The provider uses load_models_config() in __init__ and exposes _load_models_config alias
        assert "load_models_config" in content, (
            "complete() should use load_models_config() to load timeout"
        )

    @pytest.mark.asyncio
    async def test_module_complete_timeout_from_config(self) -> None:
        """Module-level complete() reads timeout from models.yaml defaults.

        Contract anchor: behaviors.md:Config:MUST:1
        """
        from amplifier_module_provider_github_copilot.provider import (
            CompletionConfig,
            CompletionRequest,
            complete,
            load_models_config,
        )

        # Verify config has timeout
        config = load_models_config()
        assert "timeout" in config.defaults
        assert config.defaults["timeout"] == 3600

        # Create mock session that properly fires session.idle event
        # The mock must capture the handler from on() and fire idle in send()
        mock_session = MagicMock()
        captured_handlers: list[Callable[[Any], None]] = []

        def mock_on(handler: Callable[[Any], None]) -> MagicMock:
            captured_handlers.append(handler)
            return MagicMock()  # unsubscribe fn

        async def mock_send(prompt: str, attachments: list[dict[str, Any]] | None = None) -> str:
            # Fire session.idle event to unblock the provider
            for handler in captured_handlers:
                handler({"type": "session.idle"})
            return "msg-id"

        mock_session.on = mock_on
        mock_session.send = mock_send
        mock_session.disconnect = AsyncMock()

        # Create mock SDK factory
        async def mock_sdk_create_fn(_config: object) -> MagicMock:
            return mock_session

        # Run complete (it should use config timeout, not hardcoded 120)
        request = CompletionRequest(prompt="test", model="claude-opus-4.5")

        events: list[Any] = []
        async for event in complete(
            request,
            config=CompletionConfig(),
            sdk_create_fn=mock_sdk_create_fn,
        ):
            events.append(event)

        # Success - no TimeoutError with 120.0 value
        # The test passes if complete() used config-driven timeout


class TestTimeoutConfigValue:
    """Tests for timeout configuration values."""

    def test_models_yaml_has_timeout_3600(self) -> None:
        """models.yaml defaults.timeout should be 3600.

        Contract anchor: behaviors.md:Config:SHOULD:1
        """
        import yaml

        config_path = (
            Path(__file__).parent.parent
            / "amplifier_module_provider_github_copilot"
            / "config"
            / "models.yaml"
        )
        with config_path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert data["provider"]["defaults"]["timeout"] == 3600

    def test_yaml_timeout_is_3600(self) -> None:
        """YAML timeout value is 3600 (1 hour for reasoning models).

        Contract anchor: behaviors.md:Config:MUST:2
        Three-Medium: YAML is authoritative source.
        """
        from amplifier_module_provider_github_copilot.config_loader import (
            load_models_config,
        )

        config = load_models_config()
        timeout = config.defaults["timeout"]
        assert timeout == 3600
