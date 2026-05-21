"""Wiring:MUST:1, MUST:2, MUST:5 — SubprocessConfig argument contract.

Contract: contracts/filesystem-layout.md
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from amplifier_module_provider_github_copilot.config._paths import (
    ProviderPaths,
)

ENV_VAR_NAME = "AMPLIFIER_PROVIDER_GITHUB_COPILOT_HOME"


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for k in (
        ENV_VAR_NAME, "XDG_DATA_HOME", "XDG_CACHE_HOME", "LOCALAPPDATA",
        "COPILOT_HOME", "COPILOT_CLI_PATH",
    ):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def _stub_subprocess_config():
    """Return a recorder class that mimics SubprocessConfig.

    Captures kwargs into a list so tests can inspect what the wrapper built.
    """
    calls: list[dict[str, Any]] = []

    class FakeSubprocessConfig:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(kwargs)
            for k, v in kwargs.items():
                setattr(self, k, v)

    return FakeSubprocessConfig, calls


def _stub_copilot_client():
    """Return a stub CopilotClient class for tests."""
    instances: list[Any] = []

    class FakeCopilotClient:
        def __init__(self, config: Any = None) -> None:
            self.config = config
            instances.append(self)

        async def start(self) -> None:  # pragma: no cover - not invoked
            pass

    return FakeCopilotClient, instances


# Contract: filesystem-layout:Wiring:MUST:1
@pytest.mark.asyncio
async def test_token_branch_passes_provider_home_to_sdk(
    clean_env: pytest.MonkeyPatch, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Token branch must construct SubprocessConfig with
    copilot_home=str(load_provider_paths().provider_home).
    """
    target_home = tmp_path / "wiring-home"
    # Override the autouse sandbox so we control the value the wrapper sees.
    sentinel = ProviderPaths(provider_home=target_home, cache_home=tmp_path / "wcache")
    monkeypatch.setattr(
        "amplifier_module_provider_github_copilot.config._paths.load_provider_paths",
        lambda: sentinel,
    )
    # Note: client.py imports load_provider_paths via a function-local
    # `from ..config._paths import ... load_provider_paths` inside
    # _ensure_client_initialized, so the patch above on
    # config._paths.load_provider_paths is the one that takes effect at
    # call time (function-local imports resolve through sys.modules).
    # No second module-level patch is needed.
    clean_env.setenv("GITHUB_TOKEN", "fake-token-for-wiring-test")

    FakeCfg, cfg_calls = _stub_subprocess_config()
    FakeClient, _ = _stub_copilot_client()

    with patch(
        "amplifier_module_provider_github_copilot.sdk_adapter._imports.SubprocessConfig",
        FakeCfg,
    ), patch(
        "amplifier_module_provider_github_copilot.sdk_adapter._imports.CopilotClient",
        FakeClient,
    ):
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )
        w = CopilotClientWrapper()
        try:
            await w._ensure_client_initialized("test")
        except Exception:
            pass

    assert cfg_calls, "Token branch must have constructed SubprocessConfig"
    kwargs = cfg_calls[0]
    assert "copilot_home" in kwargs, "Wiring:MUST:1 — copilot_home kwarg missing"
    assert kwargs["copilot_home"] == str(target_home), (
        f"copilot_home must equal str(provider_home); got {kwargs['copilot_home']!r}"
    )
    # Wiring:MUST:5 — cli_path MUST NOT be pinned (SDK falls back to bundled
    # binary once COPILOT_CLI_PATH is scrubbed). Catches RCE-class regression
    # where a future edit reintroduces explicit cli_path pinning.
    assert "cli_path" not in kwargs or kwargs.get("cli_path") is None, (
        f"cli_path MUST NOT be pinned; got {kwargs.get('cli_path')!r}"
    )


# Contract: filesystem-layout:Wiring:MUST:2
@pytest.mark.asyncio
async def test_token_and_no_token_branches_agree_on_copilot_home(
    clean_env: pytest.MonkeyPatch, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_home = tmp_path / "wiring-home-2"
    sentinel = ProviderPaths(provider_home=target_home, cache_home=tmp_path / "wcache2")
    monkeypatch.setattr(
        "amplifier_module_provider_github_copilot.config._paths.load_provider_paths",
        lambda: sentinel,
    )

    FakeCfg, cfg_calls = _stub_subprocess_config()
    FakeClient, _ = _stub_copilot_client()

    with patch(
        "amplifier_module_provider_github_copilot.sdk_adapter._imports.SubprocessConfig",
        FakeCfg,
    ), patch(
        "amplifier_module_provider_github_copilot.sdk_adapter._imports.CopilotClient",
        FakeClient,
    ):
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )
        w1 = CopilotClientWrapper()
        try:
            await w1._ensure_client_initialized("test-no-token")
        except Exception:
            pass

        clean_env.setenv("GITHUB_TOKEN", "now-with-token")
        w2 = CopilotClientWrapper()
        try:
            await w2._ensure_client_initialized("test-with-token")
        except Exception:
            pass

    assert len(cfg_calls) >= 2
    h1 = cfg_calls[0].get("copilot_home")
    h2 = cfg_calls[-1].get("copilot_home")
    assert h1 == h2 == str(target_home), (
        f"Both branches must pass the same copilot_home; got {h1!r} vs {h2!r}"
    )


# Contract: filesystem-layout:Wiring:MUST:5
@pytest.mark.asyncio
async def test_subprocess_env_omits_copilot_home_and_cli_path(
    clean_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """COPILOT_HOME and COPILOT_CLI_PATH must be scrubbed from
    SubprocessConfig.env so the SDK never inherits them from the parent.
    """
    clean_env.setenv(ENV_VAR_NAME, str(tmp_path / "scrub-home"))
    clean_env.setenv("COPILOT_HOME", "/sentinel/host-copilot-home")
    clean_env.setenv("COPILOT_CLI_PATH", "/sentinel/host-copilot-cli")
    clean_env.setenv("PATH", os.environ.get("PATH", "/usr/bin"))

    FakeCfg, cfg_calls = _stub_subprocess_config()
    FakeClient, _ = _stub_copilot_client()

    with patch(
        "amplifier_module_provider_github_copilot.sdk_adapter._imports.SubprocessConfig",
        FakeCfg,
    ), patch(
        "amplifier_module_provider_github_copilot.sdk_adapter._imports.CopilotClient",
        FakeClient,
    ):
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )
        w = CopilotClientWrapper()
        try:
            await w._ensure_client_initialized("test-env-scrub")
        except Exception:
            pass

    assert cfg_calls, "SubprocessConfig must be constructed"
    kwargs = cfg_calls[0]
    assert "env" in kwargs and isinstance(kwargs["env"], dict), (
        "Wiring:MUST:5 — SubprocessConfig.env must be an explicit dict (not None)"
    )
    env = kwargs["env"]
    assert "COPILOT_HOME" not in env, (
        "Wiring:MUST:5 — COPILOT_HOME must be removed from spawned env"
    )
    assert "COPILOT_CLI_PATH" not in env, (
        "Wiring:MUST:5 — COPILOT_CLI_PATH must be removed from spawned env"
    )
    # PATH should survive — only the two hostile keys are scrubbed
    assert "PATH" in env, "Non-scrubbed env vars must be preserved"
