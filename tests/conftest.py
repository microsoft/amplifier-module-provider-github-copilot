"""Pytest configuration and shared fixtures for SDK integration tests.

Provides fixtures for Tier 6 (SDK assumption tests) and Tier 7 (live smoke tests).

All fixtures are typed for pyright strict mode compliance.
Windows event loop policy configured for asyncio subprocess compatibility.

P1-6 Security Fix: SKIP_SDK_CHECK is now guarded by _is_pytest_running() in
__init__.py and _imports.py, making it safe to use in tests while preventing
production misuse.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

import pytest

# P1-6 Security: SKIP_SDK_CHECK is guarded by _is_pytest_running() in production code.
# This allows tests to run without the SDK while preventing production misuse.
os.environ["SKIP_SDK_CHECK"] = "1"

# Store original function before any test patches run
import amplifier_module_provider_github_copilot.models as _models_module  # noqa: E402

_original_fetch_and_map_models = _models_module.fetch_and_map_models

# Windows event loop policy fix
# On Windows, the default ProactorEventLoop has issues with asyncio subprocesses.
# Use WindowsSelectorEventLoopPolicy instead.
if sys.platform == "win32":
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # pyright: ignore[reportAttributeAccessIssue]

# P1-6 Security Fix: Clear GitHub token env vars for non-live tests.
# This prevents the fail-closed security behavior from triggering when
# SubprocessConfig is None (due to SKIP_SDK_CHECK=1) and a token exists.
# Live tests (marked with @pytest.mark.live) should preserve their tokens.
_TOKEN_ENV_VARS = ("COPILOT_AGENT_TOKEN", "COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN")
_saved_tokens: dict[str, str] = {}
for _var in _TOKEN_ENV_VARS:
    _val = os.environ.get(_var)
    if _val:
        _saved_tokens[_var] = _val
        del os.environ[_var]

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


# -- Cache Clearing --


@pytest.fixture(autouse=True)
def clear_config_caches() -> None:
    """Clear LRU caches on config loaders between tests.

    Config loaders use @lru_cache for performance. Tests that manipulate
    config files or paths need fresh loads, not cached values.
    """
    from amplifier_module_provider_github_copilot.config_loader import (
        load_models_config,
        load_retry_config,
        load_sdk_protection_config,
        load_streaming_config,
    )
    from amplifier_module_provider_github_copilot.error_translation import (
        _load_error_config_cached,  # pyright: ignore[reportPrivateUsage]
    )
    from amplifier_module_provider_github_copilot.fake_tool_detection import (
        load_fake_tool_detection_config,
    )
    from amplifier_module_provider_github_copilot.model_cache import load_cache_config
    from amplifier_module_provider_github_copilot.streaming import (
        _load_event_config_cached,  # pyright: ignore[reportPrivateUsage]
    )

    load_models_config.cache_clear()
    load_retry_config.cache_clear()
    load_streaming_config.cache_clear()
    load_sdk_protection_config.cache_clear()
    load_cache_config.cache_clear()
    _load_error_config_cached.cache_clear()
    load_fake_tool_detection_config.cache_clear()
    _load_event_config_cached.cache_clear()


# -- Model Discovery Mock (for tests that don't test SDK integration) --

# Standalone test model list — reflects realistic SDK 0.2.0 models.
# Deliberately NOT sourced from config/_models.py: production list_models()
# uses the SDK, not config. Tests that assert specific model IDs must use this.
# tuple prevents accidental mutation by test code; inner dicts are read-only by convention.
_MOCK_SDK_MODELS: tuple[dict[str, Any], ...] = (
    {
        "id": "claude-opus-4.5",
        "display_name": "Claude Opus 4.5",
        "context_window": 200000,
        "max_output_tokens": 32000,
        "capabilities": ["streaming", "tools", "vision"],
        "defaults": {},
    },
    {
        "id": "claude-sonnet-4",
        "display_name": "Claude Sonnet 4",
        "context_window": 216000,
        "max_output_tokens": 88000,
        "capabilities": ["streaming", "tools", "vision"],
        "defaults": {},
    },
)


@pytest.fixture
def real_model_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore real fetch_and_map_models for SDK integration tests.

    Use this fixture in tests that need to test actual SDK model discovery
    behavior, bypassing the autouse mock_model_discovery fixture.
    """
    monkeypatch.setattr(
        "amplifier_module_provider_github_copilot.models.fetch_and_map_models",
        _original_fetch_and_map_models,
    )


@pytest.fixture(autouse=True)
def mock_model_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto-mock model discovery to return a standalone test model list.

    Production list_models() calls the SDK; this mock replaces that call
    with _MOCK_SDK_MODELS so unit tests run without a live SDK connection.

    Tests that specifically test SDK model discovery should use the
    `real_model_discovery` fixture to disable this mock.
    """
    from amplifier_core import ModelInfo

    from amplifier_module_provider_github_copilot.sdk_adapter.model_translation import (
        CopilotModelInfo,
    )

    async def mock_fetch_and_map_models(
        _client: object,
    ) -> tuple[list[ModelInfo], list[CopilotModelInfo]]:
        """Return standalone test models — not derived from config/_models.py."""
        amplifier_models = [
            ModelInfo(
                id=m["id"],
                display_name=m["display_name"],
                context_window=m["context_window"],
                max_output_tokens=m["max_output_tokens"],
                capabilities=m.get("capabilities", []),
                defaults=m.get("defaults", {}),
            )
            for m in _MOCK_SDK_MODELS
        ]
        copilot_models = [
            CopilotModelInfo(
                id=m["id"],
                name=m["display_name"],
                context_window=m["context_window"],
                max_output_tokens=m["max_output_tokens"],
            )
            for m in _MOCK_SDK_MODELS
        ]
        return amplifier_models, copilot_models

    # Patch both the import location AND the definition
    # This handles cases where the function was already imported
    monkeypatch.setattr(
        "amplifier_module_provider_github_copilot.provider.fetch_and_map_models",
        mock_fetch_and_map_models,
    )
    monkeypatch.setattr(
        "amplifier_module_provider_github_copilot.models.fetch_and_map_models",
        mock_fetch_and_map_models,
    )


# -- Token Helpers --


def _get_github_token() -> str | None:
    """Get the first available GitHub token."""
    for var in ("COPILOT_AGENT_TOKEN", "COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
        token = os.environ.get(var)
        if token:
            return token
    return None


# -- Fixtures --


async def _default_permission_handler(  # pyright: ignore[reportUnusedFunction]
    permission: Any,
) -> dict[str, str]:
    """Default permission handler that denies all permission requests.

    SDK ASSUMPTION: The SDK requires on_permission_request handler when creating
    sessions. Without it, create_session() raises ValueError.
    """
    return {"permissionDecision": "deny", "permissionDecisionReason": "test environment"}


@pytest.fixture(scope="function")
async def sdk_client() -> AsyncIterator[Any]:
    """Function-scoped real SDK client for live tests.

    Fails if SDK not installed or no token available.
    Policy: Tests run and fail, never skip.

    SDK v0.2.0: Uses SubprocessConfig instead of options dict.
    on_permission_request moves to create_session().
    """
    copilot = pytest.importorskip("copilot", reason="github-copilot-sdk not installed")
    from copilot.client import SubprocessConfig  # type: ignore[import-not-found]

    token = _get_github_token()
    if not token:
        pytest.fail(
            "No GITHUB_TOKEN available. Set COPILOT_GITHUB_TOKEN, GH_TOKEN, or GITHUB_TOKEN. "
            "Policy: tests run and fail, never skip."
        )

    # SDK v0.2.0: SubprocessConfig replaces options dict
    config = SubprocessConfig(github_token=token)
    client = copilot.CopilotClient(config)
    await client.start()
    yield client
    await client.stop()


@pytest.fixture(scope="module")
def sdk_module() -> Any:
    """Import the copilot module, skip if not available.

    Use this for Tier 6 tests that need SDK types but not a running client.
    """
    return pytest.importorskip("copilot", reason="github-copilot-sdk not installed")


@pytest.fixture
def restore_github_tokens() -> Generator[None, None, None]:
    """Restore saved GitHub tokens for live tests.

    Use this fixture in @pytest.mark.live tests that need real tokens.
    Tokens were saved and cleared at module load for security (P1-6).
    """
    for var, val in _saved_tokens.items():
        os.environ[var] = val
    yield
    # Restore cleared state after test
    for var in _saved_tokens:
        if var in os.environ:
            del os.environ[var]


@pytest.fixture(autouse=True)
def _auto_restore_tokens_for_live_tests(  # pyright: ignore[reportUnusedFunction]
    request: pytest.FixtureRequest,
) -> Generator[None, None, None]:
    """Automatically restore tokens for @pytest.mark.live tests.

    P1-6 Security Fix: GitHub tokens are cleared at conftest import to prevent
    unit tests from accidentally using real credentials when SubprocessConfig=None.
    Live tests that need real credentials get tokens restored automatically.
    """
    # Check if test is marked as "live"
    if request.node.get_closest_marker("live"):  # pyright: ignore[reportUnknownMemberType]
        for var, val in _saved_tokens.items():
            os.environ[var] = val
        yield
        # Clear after test
        for var in _saved_tokens:
            if var in os.environ:
                del os.environ[var]
    else:
        yield


@pytest.fixture
def mock_sdk_event_dict() -> dict[str, Any]:
    """Sample SDK event as dict for testing helpers.

    Contract: sdk-boundary:EventShape:MUST:1
    Reference: SDK SessionEvent structure from github-copilot-sdk

    SDK v0.1.33+ uses nested data structure:
    - event.data.delta_content for streaming deltas
    - event.data.content for complete messages
    """
    return {
        "type": "assistant.message_delta",
        "data": {
            "delta_content": "hello",
            "message_id": "msg_001",
        },
    }


@pytest.fixture
def mock_sdk_event_object() -> Any:
    """Sample SDK event as object for testing helpers.

    Contract: sdk-boundary:EventShape:MUST:2, sdk-boundary:EventShape:MUST:3
    Reference: SDK SessionEvent structure from github-copilot-sdk

    Matches real SDK SessionEvent structure from generated SessionEvents.
    """

    class MockData:
        delta_content = "hello"
        content = None
        message_id = "msg_001"
        reasoning_id = None

    class MockEvent:
        type = "assistant.message_delta"
        data = MockData()

    return MockEvent()
