"""
Tests for token resolution precedence.

Contract: contracts/sdk-boundary.md
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestResolveTokenPrecedence:
    """AC-1: Test _resolve_token() precedence order."""

    def test_copilot_agent_token_takes_precedence(self) -> None:
        """COPILOT_AGENT_TOKEN takes precedence over GITHUB_TOKEN.

        # Contract: sdk-boundary:Auth:MUST:2
        """
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
        """COPILOT_GITHUB_TOKEN takes precedence over GH_TOKEN.

        # Contract: sdk-boundary:Auth:MUST:2
        """
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
        """GH_TOKEN takes precedence over GITHUB_TOKEN.

        # Contract: sdk-boundary:Auth:MUST:2
        """
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
        """Falls back to GITHUB_TOKEN when others not set.

        # Contract: sdk-boundary:Auth:MUST:2
        """
        from amplifier_module_provider_github_copilot.sdk_adapter import client

        with patch.dict(os.environ, {"GITHUB_TOKEN": "gh-token"}, clear=True):
            result = client._resolve_token()  # pyright: ignore[reportPrivateUsage]
            assert result == "gh-token"

    def test_returns_none_when_no_token(self) -> None:
        """Returns None when no token environment variable is set.

        # Contract: sdk-boundary:Auth:MUST:2
        """
        from amplifier_module_provider_github_copilot.sdk_adapter import client

        with patch.dict(os.environ, {}, clear=True):
            result = client._resolve_token()  # pyright: ignore[reportPrivateUsage]
            assert result is None

    @pytest.mark.parametrize(
        ("env_vars", "expected"),
        [
            pytest.param(
                {"COPILOT_AGENT_TOKEN": "", "COPILOT_GITHUB_TOKEN": "next-token"},
                "next-token",
                id="agent_to_copilot_gh",
            ),
            pytest.param(
                {"COPILOT_GITHUB_TOKEN": "", "GH_TOKEN": "next-token"},
                "next-token",
                id="copilot_gh_to_gh_cli",
            ),
            pytest.param(
                {"GH_TOKEN": "", "GITHUB_TOKEN": "next-token"},
                "next-token",
                id="gh_cli_to_github",
            ),
            pytest.param(
                {"GITHUB_TOKEN": ""},
                None,
                id="github_empty_to_none",
            ),
        ],
    )
    def test_empty_string_token_falls_through(
        self, env_vars: dict[str, str], expected: str | None
    ) -> None:
        """Empty string token falls through to next in precedence order.

        # Contract: sdk-boundary:Auth:MUST:2
        """
        from amplifier_module_provider_github_copilot.sdk_adapter import client

        with patch.dict(os.environ, env_vars, clear=True):
            result = client._resolve_token()  # pyright: ignore[reportPrivateUsage]
            assert result == expected


# ---------------------------------------------------------------------------
# mount() auth-source logging — behavioral check through the real mount path.
# Contract: sdk-boundary:Auth:MUST:2
# ---------------------------------------------------------------------------


class TestLogAuthSource:
    """Verify mount() surfaces the resolver-selected auth path.

    Contract: sdk-boundary:Auth:MUST:2
    """

    _AUTH_VARS: tuple[str, ...] = (
        "COPILOT_AGENT_TOKEN",
        "COPILOT_GITHUB_TOKEN",
        "GH_TOKEN",
        "GITHUB_TOKEN",
    )

    @staticmethod
    def _coordinator() -> MagicMock:
        from amplifier_core import ModuleCoordinator

        coordinator = MagicMock(spec=ModuleCoordinator)
        coordinator.mount = AsyncMock()
        return coordinator

    async def _mount_and_capture_auth_msg(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> str:
        """Run mount() and return the single 'Auth source' log message text.

        Caller MUST configure env vars with monkeypatch BEFORE calling this.
        Asserts exactly one Auth source record was emitted (structural guarantee,
        not a behaviour assertion — behaviour assertions are in each test).
        """
        import logging

        from amplifier_module_provider_github_copilot import mount

        dummy_client = object()
        with caplog.at_level(logging.INFO, logger="amplifier_module_provider_github_copilot"):
            with (
                patch(
                    "amplifier_module_provider_github_copilot._acquire_shared_client",
                    new=AsyncMock(return_value=dummy_client),
                ),
                patch(
                    "amplifier_module_provider_github_copilot._release_shared_client",
                    new=AsyncMock(),
                ),
            ):
                cleanup = await mount(self._coordinator(), config=None)

        assert callable(cleanup), "mount() must return a callable cleanup function"
        await cleanup()

        records = [r for r in caplog.records if "Auth source" in r.getMessage()]
        assert len(records) == 1, (
            f"Expected exactly 1 Auth source record, got {len(records)}: "
            f"{[r.getMessage() for r in caplog.records]}"
        )
        return records[0].getMessage()

    @pytest.mark.asyncio
    async def test_mount_logs_env_var_name_when_copilot_agent_token_set(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Contract: sdk-boundary:Auth:MUST:2

        COPILOT_AGENT_TOKEN set → mount logs it as the active source.
        Mutation check: removing COPILOT_AGENT_TOKEN check in _log_auth_source,
        or removing the call in mount(), turns this red.
        """
        for var in self._AUTH_VARS:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("COPILOT_AGENT_TOKEN", "sentinel-value-never-logged")

        msg = await self._mount_and_capture_auth_msg(caplog)

        assert "COPILOT_AGENT_TOKEN" in msg
        assert "sentinel-value-never-logged" not in msg  # never leak token value
        assert "cached OAuth" not in msg  # env var path must not report OAuth fallback

    @pytest.mark.asyncio
    async def test_mount_logs_cached_oauth_fallback_when_no_env_var(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Contract: sdk-boundary:Auth:MUST:2

        No env var set → mount logs cached-OAuth fallback after resolver miss.
        Guards against silent auth when a user relies on gh auth / VS Code cached OAuth.
        """
        for var in self._AUTH_VARS:
            monkeypatch.delenv(var, raising=False)

        msg = await self._mount_and_capture_auth_msg(caplog)

        assert "cached OAuth" in msg
        for var in self._AUTH_VARS:
            assert var in msg  # all four checked vars must appear as diagnostic signal

    @pytest.mark.asyncio
    async def test_mount_priority_order_copilot_agent_wins_over_github_token(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Contract: sdk-boundary:Auth:MUST:2

        Both env vars set → mount logs the higher-priority source only.
        """
        for var in self._AUTH_VARS:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("COPILOT_AGENT_TOKEN", "v1")
        monkeypatch.setenv("GITHUB_TOKEN", "v2")

        msg = await self._mount_and_capture_auth_msg(caplog)

        assert "COPILOT_AGENT_TOKEN" in msg
        assert "GITHUB_TOKEN (env var)" not in msg  # lower-priority must not appear as active

    @pytest.mark.asyncio
    async def test_mount_succeeds_if_log_auth_source_raises(self) -> None:
        """A logging failure in _log_auth_source never blocks mount.

        Contract: provider-protocol:mount:MUST:2
        Cross-platform guard: broken log handler must not block users on any OS.
        Mutation check: remove try/except around _log_auth_source() in mount() → turns red.
        """
        import amplifier_module_provider_github_copilot as _pkg
        from amplifier_module_provider_github_copilot import mount

        dummy_client = object()
        with (
            patch.object(
                _pkg, "_log_auth_source", side_effect=RuntimeError("simulated log failure")
            ),
            patch(
                "amplifier_module_provider_github_copilot._acquire_shared_client",
                new=AsyncMock(return_value=dummy_client),
            ),
            patch(
                "amplifier_module_provider_github_copilot._release_shared_client",
                new=AsyncMock(),
            ),
        ):
            cleanup = await mount(self._coordinator(), config=None)

        assert callable(cleanup), "mount() must return a callable cleanup function"
        await cleanup()
