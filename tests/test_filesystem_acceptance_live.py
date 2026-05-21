"""Acceptance:MUST:1 — live two-call session against real SDK.

Contract: contracts/filesystem-layout.md:Acceptance:MUST:1
Marked @pytest.mark.live; requires a real GitHub/Copilot token and SDK.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def _relative_set(root: Path) -> set[str]:
    if not root.exists():
        return set()
    return {str(p.relative_to(root)) for p in root.rglob("*")}


def _have_token() -> bool:
    return any(
        os.environ.get(k) for k in (
            "COPILOT_AGENT_TOKEN", "COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN",
        )
    )


# Contract: filesystem-layout:Acceptance:MUST:1
@pytest.mark.live
@pytest.mark.asyncio
async def test_live_two_call_session_redirects_sdk_files_and_preserves_legacy_path_sets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not _have_token():
        pytest.skip("Live test requires COPILOT_AGENT_TOKEN / GITHUB_TOKEN.")

    provider_home = tmp_path / "live-provider-home"
    monkeypatch.setenv("AMPLIFIER_PROVIDER_GITHUB_COPILOT_HOME", str(provider_home))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "live-xdg-cache"))

    # Snapshot legacy locations BEFORE any provider activity
    home_dot_copilot = Path.home() / ".copilot"
    home_dot_amplifier = Path.home() / ".amplifier"
    pre_copilot = _relative_set(home_dot_copilot)
    pre_amplifier = _relative_set(home_dot_amplifier)

    from amplifier_module_provider_github_copilot._identity import PROVIDER_ID
    from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider as Provider

    p = Provider()
    info = p.get_info()
    assert info.id == PROVIDER_ID, (
        f"Provider.get_info().id MUST resolve through PROVIDER_ID; got {info.id!r}"
    )

    # Two sequential real `complete()` calls — Acceptance:MUST:1 anchors
    # this on `complete()` specifically (not `list_models()`) so the SDK
    # streaming + token-write paths are exercised, which is what materializes
    # files under provider_home and cache_home.
    async def _one_call(prompt: str) -> None:
        request = {
            "messages": [{"role": "user", "content": prompt}],
            "model": "claude-opus-4.5",
        }
        response = await p.complete(request, model="claude-opus-4.5")  # type: ignore[arg-type]
        assert response is not None, f"live complete() call ({prompt!r}) returned None"

    await _one_call("first")
    post1_provider = _relative_set(provider_home)
    from amplifier_module_provider_github_copilot.config._paths import load_provider_paths
    cache_home = load_provider_paths().cache_home
    post1_cache = _relative_set(cache_home)

    await _one_call("second")
    post2_provider = _relative_set(provider_home)
    post2_cache = _relative_set(cache_home)
    post2_copilot = _relative_set(home_dot_copilot)
    post2_amplifier = _relative_set(home_dot_amplifier)

    # (a) ~/.copilot path-set unchanged
    assert post2_copilot == pre_copilot, (
        f"~/.copilot path-set changed: added {post2_copilot - pre_copilot}, "
        f"removed {pre_copilot - post2_copilot}"
    )
    # (b) ~/.amplifier path-set unchanged
    assert post2_amplifier == pre_amplifier, (
        f"~/.amplifier path-set changed: added {post2_amplifier - pre_amplifier}, "
        f"removed {pre_amplifier - post2_amplifier}"
    )
    # (c) provider_home populated after first call
    assert post1_provider, "provider_home must be populated after first call"
    # (d) provider_home post-first ⊆ post-second
    assert post1_provider.issubset(post2_provider), (
        f"Lifecycle:MUST:4 — provider_home files lost between calls: "
        f"{post1_provider - post2_provider}"
    )
    # (e) cache_home populated, post-first ⊆ post-second, disjoint from provider_home
    assert post1_cache, "cache_home must be populated after first call"
    assert post1_cache.issubset(post2_cache), (
        f"cache_home files lost between calls: {post1_cache - post2_cache}"
    )
    assert post1_provider.isdisjoint(post1_cache), (
        f"provider_home and cache_home must be pairwise disjoint after first call; "
        f"overlap={post1_provider & post1_cache}"
    )
    assert post2_provider.isdisjoint(post2_cache), (
        f"provider_home and cache_home must be pairwise disjoint after second call; "
        f"overlap={post2_provider & post2_cache}"
    )


# Contract: filesystem-layout:Acceptance:MUST:1 (unhappy path)
@pytest.mark.live
@pytest.mark.asyncio
async def test_live_malformed_token_does_not_leak_files_to_default_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failed init with malformed token MUST NOT write files into ~/.copilot
    or ~/.amplifier; failure must stay scoped to provider_home (or nothing).
    """
    provider_home = tmp_path / "live-unhappy-home"
    monkeypatch.setenv("AMPLIFIER_PROVIDER_GITHUB_COPILOT_HOME", str(provider_home))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "live-unhappy-cache"))
    monkeypatch.setenv("GITHUB_TOKEN", "definitely-not-a-real-token-xyz123")
    # Strip stronger token vars so the malformed one is the only credential
    for k in ("COPILOT_AGENT_TOKEN", "COPILOT_GITHUB_TOKEN", "GH_TOKEN"):
        monkeypatch.delenv(k, raising=False)

    pre_copilot = _relative_set(Path.home() / ".copilot")
    pre_amplifier = _relative_set(Path.home() / ".amplifier")

    from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider as Provider

    p = Provider()
    with pytest.raises(Exception):  # noqa: B017 — any provider error class
        await p.list_models()

    post_copilot = _relative_set(Path.home() / ".copilot")
    post_amplifier = _relative_set(Path.home() / ".amplifier")
    assert post_copilot == pre_copilot, (
        "Failed live call must not write to ~/.copilot"
    )
    assert post_amplifier == pre_amplifier, (
        "Failed live call must not write to ~/.amplifier"
    )
