"""
Tests for token resolution precedence.

Contract: contracts/sdk-boundary.md
"""

from __future__ import annotations

import os
from unittest.mock import patch


class TestResolveTokenPrecedence:
    """AC-1: Test _resolve_token() precedence order."""

    def test_copilot_agent_token_takes_precedence(self) -> None:
        """COPILOT_AGENT_TOKEN takes precedence over GITHUB_TOKEN."""
        # Import inline to avoid module-level import issues
        from amplifier_module_provider_github_copilot.sdk_adapter import client

        with patch.dict(
            os.environ,
            {
                "COPILOT_AGENT_TOKEN": "agent-token",
                "GITHUB_TOKEN": "gh-token",
            },
            clear=True,
        ):
            result = client._resolve_token()  # pyright: ignore[reportPrivateUsage]
            assert result == "agent-token"

    def test_copilot_github_token_second_precedence(self) -> None:
        """COPILOT_GITHUB_TOKEN takes precedence over GH_TOKEN."""
        from amplifier_module_provider_github_copilot.sdk_adapter import client

        with patch.dict(
            os.environ,
            {
                "COPILOT_GITHUB_TOKEN": "copilot-gh-token",
                "GH_TOKEN": "gh-cli-token",
                "GITHUB_TOKEN": "gh-token",
            },
            clear=True,
        ):
            result = client._resolve_token()  # pyright: ignore[reportPrivateUsage]
            assert result == "copilot-gh-token"

    def test_gh_token_third_precedence(self) -> None:
        """GH_TOKEN takes precedence over GITHUB_TOKEN."""
        from amplifier_module_provider_github_copilot.sdk_adapter import client

        with patch.dict(
            os.environ,
            {
                "GH_TOKEN": "gh-cli-token",
                "GITHUB_TOKEN": "gh-token",
            },
            clear=True,
        ):
            result = client._resolve_token()  # pyright: ignore[reportPrivateUsage]
            assert result == "gh-cli-token"

    def test_github_token_fallback(self) -> None:
        """Falls back to GITHUB_TOKEN when others not set."""
        from amplifier_module_provider_github_copilot.sdk_adapter import client

        with patch.dict(os.environ, {"GITHUB_TOKEN": "gh-token"}, clear=True):
            result = client._resolve_token()  # pyright: ignore[reportPrivateUsage]
            assert result == "gh-token"

    def test_returns_none_when_no_token(self) -> None:
        """Returns None when no token environment variable is set."""
        from amplifier_module_provider_github_copilot.sdk_adapter import client

        with patch.dict(os.environ, {}, clear=True):
            result = client._resolve_token()  # pyright: ignore[reportPrivateUsage]
            assert result is None
