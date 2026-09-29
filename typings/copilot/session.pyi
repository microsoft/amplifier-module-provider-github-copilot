"""Type stubs for ``copilot.session`` submodule (SDK v1.0.0b10).

The provider imports ``PermissionRequestResult`` from this module via the
quarantined ``sdk_adapter`` membrane. ``ReasoningEffort`` is also re-exported
from ``copilot.session`` by the live SDK (the same Literal object as
``copilot.client.ReasoningEffort``), so the stub mirrors that surface for
downstream consumers that prefer the session-module import path.

Shape verified against the live SDK v1.0.0b10 source:

* ``PermissionRequestResult`` is the type alias ``PermissionDecision |
  PermissionNoResult`` (b10 ``session.py:L275``). It is NOT a constructor;
  declaring it as a dataclass would be a typing lie and would mask call-site
  errors at the boundary. Construct denial via ``PermissionDecisionReject()``
  from ``copilot.generated.rpc``.
* ``PermissionNoResult`` is a dataclass sentinel with a literal ``kind``
  discriminator (b10 ``session.py:L257``).
* ``ReasoningEffort`` is ``Literal["low","medium","high","xhigh"]``
  re-exported from ``copilot.session`` and ``copilot.client``.
"""

from collections.abc import Callable
from typing import Any, Literal, TypeAlias

# from b10 session.py:L43 — PermissionDecision is the codegen discriminated
# union of variant classes (Approve*, Reject, UserNotAvailable). The provider
# never inspects its internals, so the stub treats it as an opaque class.
class PermissionDecision:
    kind: str


class PermissionNoResult:
    """Sentinel returned by a permission handler to leave a request unanswered."""

    kind: Literal["no-result"]

    # b10 ``session.py:L256-L266`` declares PermissionNoResult as @dataclass
    # with a single defaulted field, so the implicit __init__ accepts ``kind``
    # both positionally and as a keyword. The stub mirrors that.
    def __init__(self, kind: Literal["no-result"] = "no-result") -> None: ...


PermissionRequestResult: TypeAlias = PermissionDecision | PermissionNoResult


ReasoningEffort = Literal["low", "medium", "high", "xhigh"]


# from b10 session.py:L1066 — CopilotSession is the streaming session created
# by ``CopilotClient.create_session``. Canonical home is ``copilot.session``;
# ``copilot/__init__.pyi`` re-exports it for the root import path.
class CopilotSession:
    """Streaming session created by ``CopilotClient.create_session``."""

    session_id: str

    # Real SDK declares ``workspace_path`` as ``functools.cached_property``;
    # the stub renders it as a read-only property to forbid spurious assignment.
    @property
    def workspace_path(self) -> Any: ...
    def on(
        self, handler: Callable[[Any], None]
    ) -> Callable[[], None]: ...
    # from v1.0.15 session.py:L1759-L1770: seven keyword-only kwargs
    # (display_prompt added in b10; source and response_schema added in v1.0.15;
    # SDKSurface:MUST:6 pins the full set).
    async def send(
        self,
        prompt: str,
        *,
        attachments: list[Any] | None = None,
        source: Any | None = None,
        mode: Any | None = None,
        agent_mode: Any | None = None,
        request_headers: dict[str, str] | None = None,
        display_prompt: str | None = None,
        response_schema: Any | None = None,
    ) -> str: ...
    async def send_and_wait(
        self,
        prompt: str,
        *,
        attachments: list[Any] | None = None,
        source: Any | None = None,
        mode: Any | None = None,
        agent_mode: Any | None = None,
        request_headers: dict[str, str] | None = None,
        display_prompt: str | None = None,
        response_schema: Any | None = None,
        timeout: float = 60.0,
    ) -> Any: ...
    async def disconnect(self) -> None: ...
    async def destroy(self) -> None: ...
    async def __aenter__(self) -> CopilotSession: ...
    async def __aexit__(self, *args: Any) -> None: ...


__all__ = [
    "CopilotSession",
    "PermissionDecision",
    "PermissionNoResult",
    "PermissionRequestResult",
    "ReasoningEffort",
]
