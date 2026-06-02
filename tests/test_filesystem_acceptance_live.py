"""Acceptance:MUST:1 — live two-call session against real SDK.

Contract: contracts/filesystem-layout.md:Acceptance:MUST:1
Marked @pytest.mark.live; requires a real GitHub/Copilot token and SDK.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import pytest

# Capture genuine load_provider_paths before autouse fixtures patch it.
import amplifier_module_provider_github_copilot.config._paths as _paths_mod

_real_load_provider_paths = _paths_mod.load_provider_paths


def _relative_set(root: Path) -> set[str]:
    if not root.exists():
        return set()
    return {str(p.relative_to(root)) for p in root.rglob("*")}


def _have_token() -> bool:
    return any(
        os.environ.get(k)
        for k in (
            "COPILOT_AGENT_TOKEN",
            "COPILOT_GITHUB_TOKEN",
            "GH_TOKEN",
            "GITHUB_TOKEN",
        )
    )


@contextmanager
def _real_sdk_mode() -> Generator[None, None, None]:
    """Clear SKIP_SDK_CHECK and reload SDK modules so live tests hit the real SDK.

    Each ``importlib.reload`` creates a new class object inside the reloaded
    module. Modules that bound ``CopilotClientWrapper`` at their first-import
    time (``provider.py``, ``sdk_adapter/__init__.py``, the top-level package
    ``__init__.py``) keep pointing at the *previous* class object — so after
    a reload, ``isinstance(provider._client, CopilotClientWrapper)`` checks
    against ``sdk_adapter.client.CopilotClientWrapper`` start failing
    elsewhere in the test session (observed: ``test_singleton.py::
    TestProviderClientInjection::test_provider_creates_own_client_without_injection``
    failing only when this live test runs in the same session).

    To preserve class identity for downstream tests we (a) reload once on
    entry to swap in the real-SDK module bodies, (b) reload once on exit to
    restore the SKIP_SDK_CHECK=1 module bodies, and (c) re-bind the
    ``CopilotClientWrapper`` attribute on every dependent module to the
    post-restore class object so all references converge again.
    """
    from amplifier_module_provider_github_copilot.sdk_adapter import _imports
    from amplifier_module_provider_github_copilot.sdk_adapter import client as _client_mod

    original_skip = os.environ.pop("SKIP_SDK_CHECK", None)
    try:
        importlib.reload(_imports)
        importlib.reload(_client_mod)
        yield
    finally:
        if original_skip is not None:
            os.environ["SKIP_SDK_CHECK"] = original_skip
        importlib.reload(_imports)
        importlib.reload(_client_mod)
        # Re-bind dependent modules' ``CopilotClientWrapper`` to the
        # freshly-reloaded class object so subsequent ``isinstance`` checks
        # see one identity across provider.py / sdk_adapter / package init.
        import amplifier_module_provider_github_copilot as _pkg_mod
        import amplifier_module_provider_github_copilot.provider as _provider_mod
        import amplifier_module_provider_github_copilot.sdk_adapter as _sa_mod

        _live_cls = _client_mod.CopilotClientWrapper
        _provider_mod.CopilotClientWrapper = _live_cls
        _sa_mod.CopilotClientWrapper = _live_cls
        _pkg_mod.CopilotClientWrapper = _live_cls


# Contract: filesystem-layout:Acceptance:MUST:1
@pytest.mark.live
@pytest.mark.asyncio
async def test_live_two_call_session_redirects_sdk_files_and_preserves_legacy_path_sets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, real_model_discovery: None
) -> None:
    """Two sequential complete() calls MUST write exclusively to provider_home and
    cache_home (resolved via AMPLIFIER_PROVIDER_GITHUB_COPILOT_HOME / XDG_CACHE_HOME),
    never leaking into ~/.copilot or ~/.amplifier.

    Contract: filesystem-layout:Acceptance:MUST:1
    """
    if not _have_token():
        pytest.fail("Live test requires COPILOT_AGENT_TOKEN / GITHUB_TOKEN.")

    # A2: Set env vars and restore genuine load_provider_paths so the test exercises
    # the contract-named AMPLIFIER_PROVIDER_GITHUB_COPILOT_HOME boundary rather than
    # the autouse sandbox lambda.
    live_provider_home = tmp_path / "live-provider-home"
    live_cache_base = tmp_path / "live-cache-home"
    monkeypatch.setenv("AMPLIFIER_PROVIDER_GITHUB_COPILOT_HOME", str(live_provider_home))
    monkeypatch.setenv("XDG_CACHE_HOME", str(live_cache_base))
    monkeypatch.setattr(
        "amplifier_module_provider_github_copilot.config._paths.load_provider_paths",
        _real_load_provider_paths,
    )
    monkeypatch.setattr(
        "amplifier_module_provider_github_copilot.model_cache.load_provider_paths",
        _real_load_provider_paths,
    )
    _resolved = _real_load_provider_paths()
    provider_home = _resolved.provider_home
    cache_home = _resolved.cache_home

    # real_model_discovery restores fetch_and_map_models; restore the provider
    # module binding too so list_models() routes through the real SDK fetch path.
    import amplifier_module_provider_github_copilot.models as _models_mod

    monkeypatch.setattr(
        "amplifier_module_provider_github_copilot.provider.fetch_and_map_models",
        _models_mod.fetch_and_map_models,
    )

    # model_cache.write_cache() skips writing when PYTEST_CURRENT_TEST is set.
    # Bypass by wrapping with an explicit cache_file path resolved through the
    # genuine load_provider_paths binding established above.
    from amplifier_module_provider_github_copilot.model_cache import (
        get_cache_file_path as _get_cache_file_path,
    )
    from amplifier_module_provider_github_copilot.model_cache import (
        write_cache as _real_write_cache,
    )

    _live_cache_file = _get_cache_file_path()

    def _write_cache_live(models: object, cache_file: object = None) -> None:
        _real_write_cache(models, cache_file=_live_cache_file)  # type: ignore[arg-type]

    monkeypatch.setattr(
        "amplifier_module_provider_github_copilot.provider.write_cache",
        _write_cache_live,
    )

    # Snapshot legacy locations BEFORE any provider activity.
    home_dot_copilot = Path.home() / ".copilot"
    home_dot_amplifier = Path.home() / ".amplifier"
    pre_copilot = _relative_set(home_dot_copilot)
    pre_amplifier = _relative_set(home_dot_amplifier)

    from amplifier_core import ChatResponse
    from amplifier_core.llm_errors import ConfigurationError, ProviderUnavailableError  # noqa: F401

    from amplifier_module_provider_github_copilot._identity import PROVIDER_ID
    from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider as Provider
    from amplifier_module_provider_github_copilot.sdk_adapter import client as _client_mod

    with _real_sdk_mode():
        fresh_wrapper = _client_mod.CopilotClientWrapper()
        p = Provider(client=fresh_wrapper)

        info = p.get_info()
        assert info.id == PROVIDER_ID, (
            f"Provider.get_info().id MUST resolve through PROVIDER_ID; got {info.id!r}"
        )

        # Discover the first model available to this token.
        available = await p.list_models()
        assert available, "SDK returned no models for this token — cannot proceed"
        model_id = available[0].id

        # A3: Guard against mock-model leakage — fail loudly on fixture order drift.
        from tests.conftest import _MOCK_SDK_MODELS

        mock_ids = {m["id"] for m in _MOCK_SDK_MODELS}
        assert model_id not in mock_ids, (
            f"Fixture order regression: real_model_discovery failed to restore provider "
            f"binding; got mock model {model_id}"
        )

        # A1: Baseline BEFORE first complete() so growth is causal to complete(), not
        # to list_models() which may have already written to provider_home.
        baseline_provider = _relative_set(provider_home)
        baseline_cache = _relative_set(cache_home)

        # Two sequential real complete() calls — Acceptance:MUST:1 anchors on
        # complete() specifically so the SDK streaming + token-write paths are
        # exercised, materialising files under provider_home.
        async def _one_call(prompt: str) -> None:
            request = {
                "messages": [{"role": "user", "content": prompt}],
                "model": model_id,
            }
            response = await p.complete(request, model=model_id)  # type: ignore[arg-type]
            # A4: Behavioral assertion — ChatResponse with non-empty content.
            assert isinstance(response, ChatResponse), (
                f"complete() MUST return ChatResponse; got {type(response).__name__}"
            )
            assert response.content, (
                f"live complete() call ({prompt!r}) returned ChatResponse with empty content"
            )

        await _one_call("first")
        post1_provider = _relative_set(provider_home)

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
        # (c) A1: provider_home grew after both complete() calls — strict causal growth.
        assert post2_provider - baseline_provider, (
            "provider_home MUST have new files after both complete() calls; "
            f"baseline={baseline_provider!r}, post2={post2_provider!r}"
        )
        # (d) provider_home post-first ⊆ post-second
        assert post1_provider.issubset(post2_provider), (
            f"Lifecycle:MUST:4 — provider_home files lost between calls: "
            f"{post1_provider - post2_provider}"
        )
        # TODO: Acceptance:MUST:1(e) reword pending — complete() does NOT populate
        # cache_home (only list_models does); see contracts/filesystem-layout.md.
        assert (post2_cache - baseline_cache).isdisjoint(post2_provider - baseline_provider), (
            f"provider_home and cache_home growth must be disjoint; "
            f"overlap={(post2_cache - baseline_cache) & (post2_provider - baseline_provider)}"
        )


# Contract: filesystem-layout:Acceptance:MUST:1 (unhappy path)
@pytest.mark.live
@pytest.mark.asyncio
async def test_live_malformed_token_does_not_leak_files_to_default_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, real_model_discovery: None
) -> None:
    """Failed init with malformed token MUST NOT write files into ~/.copilot
    or ~/.amplifier; failure is rejected before provider activity reaches legacy paths.
    """
    monkeypatch.setenv("GITHUB_TOKEN", "definitely-not-a-real-token-xyz123")
    # Strip stronger token vars so the malformed one is the only credential.
    for k in ("COPILOT_AGENT_TOKEN", "COPILOT_GITHUB_TOKEN", "GH_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    # Isolate provider_home so the failure path is observed against a
    # tmp_path the test owns end-to-end (filesystem-layout:Lifecycle:MUST:4
    # says "MUST NOT delete, rename, or auto-migrate provider_home, cache_home,
    # or their contents on any code path" — the failure path needs the same
    # snapshot discipline as the happy path).
    isolated_home = tmp_path / "fail-provider-home"
    monkeypatch.setenv("AMPLIFIER_PROVIDER_GITHUB_COPILOT_HOME", str(isolated_home))

    pre_copilot = _relative_set(Path.home() / ".copilot")
    pre_amplifier = _relative_set(Path.home() / ".amplifier")
    pre_provider_home = _relative_set(isolated_home) if isolated_home.exists() else set()

    from amplifier_core.llm_errors import (
        AuthenticationError,
        ConfigurationError,
        ProviderUnavailableError,
    )

    from amplifier_module_provider_github_copilot.provider import GitHubCopilotProvider as Provider
    from amplifier_module_provider_github_copilot.sdk_adapter import client as _client_mod

    with _real_sdk_mode():
        fresh_wrapper = _client_mod.CopilotClientWrapper()
        p = Provider(client=fresh_wrapper)

        # A4: complete() drives SDK subprocess startup where token validation occurs.
        # list_models() routes through fetch_and_map_models (patchable) and never
        # reaches the SDK auth layer. The bundled CLI validates the token against
        # GitHub server-side, so a malformed token surfaces as a 401 that the error
        # contract maps to AuthenticationError (errors.yaml: "401"/unauthorized);
        # ProviderUnavailableError/ConfigurationError remain accepted for non-auth
        # subprocess-startup failures.
        with pytest.raises((AuthenticationError, ProviderUnavailableError, ConfigurationError)):
            await p.complete(  # type: ignore[arg-type]
                {"messages": [{"role": "user", "content": "ping"}], "model": "claude-opus-4.5"},  # pyright: ignore[reportArgumentType]
                model="claude-opus-4.5",
            )

    post_copilot = _relative_set(Path.home() / ".copilot")
    post_amplifier = _relative_set(Path.home() / ".amplifier")
    post_provider_home = _relative_set(isolated_home) if isolated_home.exists() else set()
    assert post_copilot == pre_copilot, "Failed live call must not write to ~/.copilot"
    assert post_amplifier == pre_amplifier, "Failed live call must not write to ~/.amplifier"
    # Lifecycle:MUST:4 — failure-path snapshot of the isolated provider_home.
    # Auth-failure teardown MUST NOT delete or rename anything inside the
    # provider's own state root.
    assert post_provider_home == pre_provider_home, (
        f"Failed live call must not delete/rename inside provider_home; "
        f"missing={pre_provider_home - post_provider_home}, "
        f"added={post_provider_home - pre_provider_home}"
    )
