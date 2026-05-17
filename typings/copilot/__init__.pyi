"""Type stubs for github-copilot-sdk (imported as ``copilot``).

Mirrors the public API surface of github-copilot-sdk v1.0.0b4. Only the symbols
the provider actually uses are stubbed in detail; everything else is typed as
``Any`` to keep the stub small without losing pyright coverage on imports.

Verified against `inspect.signature(...)` against the live SDK on 2026-05-17.
"""

from collections.abc import Awaitable, Callable
from typing import Any

from .types import SubprocessConfig

class ExternalServerConfig:
    """Configuration for SDK external-server mode (live SDK class)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class ModelVisionLimitsOverride:
    """Per-session vision limit overrides (SDK v0.3.0).

    Real SDK shape (``inspect`` verified 2026-05-12): dataclass with three
    optional fields. Provider does not currently construct one, but the
    type appears as the ``vision`` field of :class:`ModelLimitsOverride`
    and stubbing it explicitly avoids ``Any`` leakage on that boundary.
    """

    supported_media_types: list[str] | None
    max_prompt_images: int | None
    max_prompt_image_size: int | None

    def __init__(
        self,
        *,
        supported_media_types: list[str] | None = None,
        max_prompt_images: int | None = None,
        max_prompt_image_size: int | None = None,
    ) -> None: ...

class ModelSupportsOverride:
    """Per-session capability flag overrides (SDK v0.3.0).

    Real SDK shape (``inspect`` verified 2026-05-12): dataclass with two
    optional bool fields. Used as the ``supports`` field of
    :class:`ModelCapabilitiesOverride`.
    """

    vision: bool | None
    reasoning_effort: bool | None

    def __init__(
        self,
        *,
        vision: bool | None = None,
        reasoning_effort: bool | None = None,
    ) -> None: ...

class ModelLimitsOverride:
    """Per-session token limit overrides (SDK v0.3.0).

    Forwarded by the provider to honor ``ChatRequest.max_tokens`` via
    ``ModelCapabilitiesOverride.limits.max_output_tokens``.
    """

    max_prompt_tokens: int | None
    max_output_tokens: int | None
    max_context_window_tokens: int | None
    vision: ModelVisionLimitsOverride | None

    def __init__(
        self,
        *,
        max_prompt_tokens: int | None = None,
        max_output_tokens: int | None = None,
        max_context_window_tokens: int | None = None,
        vision: ModelVisionLimitsOverride | None = None,
    ) -> None: ...

class ModelCapabilitiesOverride:
    """Per-session capability overrides (SDK v0.3.0)."""

    supports: ModelSupportsOverride | None
    limits: ModelLimitsOverride | None

    def __init__(
        self,
        *,
        supports: ModelSupportsOverride | None = None,
        limits: ModelLimitsOverride | None = None,
    ) -> None: ...

class CopilotSession:
    """Streaming session created by ``CopilotClient.create_session``.

    Public surface limited to what the provider actually invokes; other
    members (``abort``, ``set_model``, ``ui``, ``capabilities``, ...) are
    accessible via ``Any`` attribute lookup at runtime.
    """

    session_id: str

    # Real SDK declares this as ``functools.cached_property`` on
    # ``CopilotSession``; pyright treats a stub ``@property`` as semantically
    # equivalent for type-checking call sites (read-only, computed once).
    # Declaring a plain attribute would invite spurious ``= ...`` assignments.
    @property
    def workspace_path(self) -> Any: ...

    def on(
        self, handler: Callable[[Any], None]
    ) -> Callable[[], None]: ...
    async def send(
        self,
        prompt: str,
        *,
        attachments: list[Any] | None = None,
        mode: Any | None = None,
        request_headers: dict[str, str] | None = None,
    ) -> str: ...
    async def send_and_wait(
        self,
        prompt: str,
        *,
        attachments: list[Any] | None = None,
        mode: Any | None = None,
        request_headers: dict[str, str] | None = None,
        timeout: float = 60.0,
    ) -> Any: ...
    async def disconnect(self) -> None: ...
    async def destroy(self) -> None: ...
    async def __aenter__(self) -> CopilotSession: ...
    async def __aexit__(self, *args: Any) -> None: ...

class CopilotClient:
    """Top-level SDK client (live SDK class).

    Stub captures only the symbols the provider imports / calls. The real
    ``create_session`` accepts ~30 keyword arguments (mcp_servers, hooks,
    custom_agents, ...); they are typed loosely here because the provider
    only forwards a small subset.
    """

    def __init__(
        self,
        config: SubprocessConfig | ExternalServerConfig | None = None,
        *,
        auto_start: bool = True,
        on_list_models: Callable[[], list[Any] | Awaitable[list[Any]]] | None = None,
    ) -> None: ...
    async def create_session(
        self,
        *,
        on_permission_request: Any,
        model: str | None = None,
        reasoning_effort: str | None = None,
        tools: list[Any] | None = None,
        system_message: Any | None = None,
        available_tools: list[str] | None = None,
        excluded_tools: list[str] | None = None,
        model_capabilities: ModelCapabilitiesOverride | None = None,
        streaming: bool | None = None,
        on_event: Callable[[Any], None] | None = None,
        # NOTE: ``**kwargs: Any`` admits the SDK's long tail of optional
        # keywords (hooks, mcp_servers, custom_agents, infinite_sessions,
        # commands, skill_directories, enable_config_discovery, ...). pyright
        # cannot catch typos against this signature; runtime tests in
        # ``tests/test_sdk_assumptions.py`` and live-smoke coverage are the
        # source of truth for keyword spelling. Keep additions here narrow.
        **kwargs: Any,
    ) -> CopilotSession: ...
    async def list_models(self) -> list[Any]: ...
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def __aenter__(self) -> CopilotClient: ...
    async def __aexit__(self, *args: Any) -> None: ...

__all__ = [
    "CopilotClient",
    "CopilotSession",
    "ExternalServerConfig",
    "ModelCapabilitiesOverride",
    "ModelLimitsOverride",
    "ModelSupportsOverride",
    "ModelVisionLimitsOverride",
    "SubprocessConfig",
]
