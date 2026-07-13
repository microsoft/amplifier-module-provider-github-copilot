# Contract: SDK Boundary (The Membrane)

## Version
- **Current:** 1.13 (SDK v1.0.2 — Client Lifecycle annotated with the observed bounded ~12s graceful-shutdown teardown cost from the real-world E2E)
- **Module Reference:** amplifier_module_provider_github_copilot/sdk_adapter/
- **Status:** Non-Negotiable Constraint
- **Update:** 2026-06-21 — Client Lifecycle annotated with the v1.0.2 graceful-shutdown teardown-latency observation. A real-world WSL E2E probe of the provider's `close()`→SDK `stop()` path shows a clean teardown takes ~12s ONCE PER LIFECYCLE: the graceful runtime-shutdown RPC acks in <1s (`client.py:1455` success branch, not `:1462` fail), then the drained CLI process does not self-exit so the bounded `wait(timeout=10)` (`client.py:1491-1496`) elapses and `terminate()` (`:1498`) reaps it — bounded, deterministic, zero orphans, no SIGKILL, NOT per-turn (runtime reused across turns). SDK-owned behavior; provider only calls `stop()`. No source change.
- **Update (v1.12):** 2026-06-21 — ToolForwarding hardened for v1.0.2. The SDK's `copilot.tools.Tool` gained a `defer: Literal["auto","never"] | None` field (installed v1.0.2 `tools.py:65`); the SDK reads it when building tool wire-definitions (`client.py:1810-1811` and `:2391-2392`: `if tool.defer is not None: definition["defer"] = tool.defer`). The provider's duck-typed `SDKToolWrapper` lacked it, so every tool-forwarding turn raised `AttributeError: 'SDKToolWrapper' object has no attribute 'defer'` — a real production blocker not reached by the mocked unit suite or the tool-less live smoke suite (neither exercises the real tool-definition builder). `SDKToolWrapper.defer` added, defaulting to `None` (omits the wire key = byte-identical pre-v1.0.2 payload); Amplifier pre-loads all tools at the kernel layer, so the SDK's lazy tool-search deferral stays off. A new Tier-6 structural guard introspects the real `copilot.tools.Tool` field set against the wrapper so a FUTURE SDK tool-field addition fails RED before E2E.
- **Update (v1.11):** 2026-06-21 — SDK bumped b10 → v1.0.2 (GA, 1.0.1, 1.0.2). MinimalMode extended from MUST:1-15 to MUST:1-16: v1.0.2 adds a mode-gated `memory` session capability whose empty-mode default helper `_memory_default` (installed v1.0.2 `_mode.py:264-276`) returns `{"enabled": False}` only when `mode == "empty"`; our adapter ships `mode="copilot-cli"`, so leaving `memory` unset hands session-memory control to the bundled CLI. Pinned to `{"enabled": False}` (MUST:16). The b10→v1.0.2 span is non-breaking per the SDK diff analysis + freeze-diff battery: the only production wire-shape change is the `memory` emit plus three config-driven event drops (see event-vocabulary 1.4). v1.0.2 also lands a graceful runtime shutdown on the once-per-lifecycle `stop()` path, bounded by `_RUNTIME_SHUTDOWN_TIMEOUT_SECONDS=10` (NOT per-turn).
- **Update (v1.10):** 2026-06-01 — MinimalMode extended from MUST:1-6 to MUST:1-15. b10 added 8 new `create_session` kwargs gating SDK-internal capabilities (session store, skills loader, file hooks, host git, on-demand instruction discovery, embedding retrieval, embedding cache storage, MCP OAuth token storage), plus the pre-existing b9 `enable_session_telemetry` consolidated here. Helpers at b10 `_mode.py:185-258` only collapse `None` to defaults when `mode == "empty"`; our adapter ships `mode="copilot-cli"` (`sdk_adapter/client.py:419-424`), so leaving any kwarg unset hands control to the bundled CLI. Explicit pin is a real wire-shape change, not intent-pinning. v1.10 pins the 9th mode-gated default (`mcp_oauth_token_storage`) and corrects the v1.9 claim that `mcp_servers={}` foreclosed its wire-emit — it does not; the emit is independent at b10 `client.py:1863-1865`. v1.8 SDKSurface clauses verified byte-compatible against b10 — no production source change beyond the 9 MinimalMode emits.
- **History:**
  - **1.13** — SDK v1.0.2: Client Lifecycle annotated (no source change) with the observed graceful-shutdown teardown cost from the real-world WSL E2E. Provider `close()`→SDK `stop()` takes ~12s ONCE PER LIFECYCLE: graceful runtime-shutdown RPC acked <1s (`client.py:1455` success branch, `runtime shutdown complete`; the `:1462` fail branch never fires), then the drained CLI process does not self-exit so the bounded `wait(timeout=_RUNTIME_SHUTDOWN_TIMEOUT_SECONDS=10)` (`client.py:1491-1496`) elapses and `terminate()` (`:1498`) reaps it. Bounded, deterministic, zero orphans, no SIGKILL. Runtime reused across turns so the cost is lifecycle-end only, never per-turn. SDK-owned; the provider only invokes `stop()`.
  - **1.12** — SDK v1.0.2: ToolForwarding:MUST:2 extended — `SDKToolWrapper` gains `defer: Literal["auto","never"] | None = None`, mirroring the SDK's own `copilot.tools.Tool.defer` (installed v1.0.2 `tools.py:65`). The SDK reads `tool.defer` building tool wire-definitions (`client.py:1810-1811`, `:2391-2392`); the duck-typed wrapper lacking it raised `AttributeError` on every tool-forwarding turn — a real blocker not reachable by the mocked unit suite or the tool-less live smoke suite, only by an E2E that exercises the real tool-definition builder. `None` default omits the `"defer"` wire key (byte-identical pre-v1.0.2 payload); Amplifier pre-loads tools at the kernel layer so lazy tool-search deferral has no meaning here. Durable guard added: `tests/test_sdk_assumptions.py::TestSDKToolWrapperCoversSDKToolSurface` introspects the real `Tool` dataclass field set (⊆ wrapper, `require_sdk`-gated, allowlist+hygiene), so a future SDK tool-field addition fails RED before E2E rather than crashing a live turn.
  - **1.11** — SDK v1.0.2: MinimalMode:MUST:16 added — `memory={"enabled": False}`, the 10th SDK mode-gated capability default. Identical in mode-gating pattern to MUST:13/MUST:15: under `mode="copilot-cli"` the helper `_memory_default` (installed v1.0.2 `_mode.py:264-276`) returns its empty-mode value ONLY when `mode == "empty"`, so the bundled-CLI default applies unless pinned. The empty-mode value is a dict `{"enabled": False}` (as in MUST:1's `infinite_sessions`, NOT the `"in-memory"` string of MUST:13/15). Pinned to mirror the SDK's own empty-mode default; Amplifier owns all context/memory and runs ephemeral per-`complete()` sessions (deny-destroy). The b10→v1.0.2 span (GA, 1.0.1, 1.0.2) is non-breaking per the SDK diff analysis + freeze-diff battery; the sole production wire-shape change is the `memory` emit (plus three config-driven event drops, event-vocabulary 1.4). v1.0.2 shifted the `_mode.py` line ranges, so MUST:16's offsets are version-tagged and not contiguous with the b10 block.
  - **1.10** — SDK v1.0.0b10: MinimalMode:MUST:15 added — `mcp_oauth_token_storage="in-memory"`, the 9th and final SDK mode-gated capability default. Corrects the v1.9 rationale that scoped this switch out: the `mcpOAuthTokenStorage` wire-emit (b10 `client.py:1863-1865`) is an INDEPENDENT `if mcp_oauth_token_storage is not None` block, NOT gated by the `mcpServers` emit (b10 `client.py:1860-1861`), so `mcp_servers={}` (MUST:3) does not foreclose it. Under `mode="copilot-cli"` the helper `_mcp_oauth_token_storage_default` (b10 `_mode.py:251-258`) returns `None` — byte-identical in shape to `_embedding_cache_storage_default` (b10 `_mode.py:201-208`, MUST:13) — so the bundled-CLI default applies unless pinned. Pinned `"in-memory"` to mirror the SDK's own empty-mode default and MUST:13. The only reason there was no live exposure at v1.9 is one level up (no MCP servers ⇒ no OAuth flow), a silent coupling now removed. Closure hardened: `test_no_unpinned_sdk_mode_gated_capability` runtime-enumerates the `_mode` mode-gated helpers so a future 10th default fails loudly. `manage_schedule_enabled` / `coauthor_enabled` remain out of scope — see the MinimalMode Out-of-scope subsection.
  - **1.9** — SDK v1.0.0b10: MinimalMode:MUST:7-14 added (`enable_session_store=False`, `enable_skills=False`, `enable_file_hooks=False`, `enable_host_git_operations=False`, `enable_on_demand_instruction_discovery=False`, `skip_embedding_retrieval=True`, `embedding_cache_storage="in-memory"`, `enable_session_telemetry=False`). Mode-mismatch rationale: b10 `_mode.py:175-182` `_empty_mode_bool_default` returns `empty_default` ONLY when `mode == "empty"`; our adapter mode is `copilot-cli`. Wire emit at b10 `client.py:1852-1905` only serializes a kwarg when non-`None`. Sub-clauses recorded in MinimalMode constraint + test-anchor tables. Telemetry is pinned `False` because b10 `client.py:1651-1656` documents telemetry as ON-by-default for GitHub-authenticated sessions (our auth path); leaving the kwarg unset would silently opt the provider in. (`mcp_oauth_token_storage` is addressed separately at v1.10 — see above.)
  - **1.8** — SDK v1.0.0b9: SDKSurface:MUST:6 corrected — `copilot.session.CopilotSession.send` accepts four keyword-only kwargs (`attachments`, `mode`, `agent_mode`, `request_headers`), not three. Anchor: b9 `session.py:L1154-L1160`. Test fixture at `tests/fixtures/sdk_mocks.py:L184-L192` already mirrored this shape; only the contract clause was stale. No production source change.
  - **1.7** — SDK v1.0.0b9: `SubprocessConfig` removed (b7) — confirmed absent from b9 `copilot/__init__.py:L1-L274` and `copilot/client.py:L100-L115`; `CopilotClient.__init__` is keyword-only (b9 `client.py:L1051-L1067`); `PermissionRequestResult` is the type alias `PermissionDecision | PermissionNoResult` (b9 `session.py:L270`); permission-denial uses `PermissionDecisionReject()` imported from `copilot.generated.rpc` (b9 `session.py:L29-L57` imports the variant; b9 root `__init__.py:L94, L203` re-exports the alias but not the variants).
  - **1.6** — SDK v1.0.0b4: `ModelBilling.multiplier` is now `float | None = None`; `ModelBilling.from_dict` no longer raises when `multiplier` is absent. Fresh installs and the model-discovery / configure-wizard paths (`amplifier init`, `amplifier provider models`, `provider edit`) were hard-broken on v0.3.0 once GitHub stopped emitting `multiplier`. The `complete()` runtime path was unaffected. New regression tests under `SDKSurface:MUST:7` pin the tolerance so any future SDK re-tightening fails loudly here.
  - **1.5** — `SessionHandle` reframed as a façade delegating `on`/`send`/`abort` to a private `_raw_session`; lifecycle (`connect`/`disconnect`/`destroy`) owned by `client.session()` (`sdk_adapter/client.py`). Four normative bullets registered as `Types:MUST:4..7` covering `_raw_session` privacy, narrow surface, no lifecycle on handle, and construction-time `session_id` capture. Anchor IDs `:1..:3` unchanged.
  - **1.4** — SDK v0.3.0: `PermissionRequestResult` fields `rules`/`feedback`/`message`/`path` removed; kind literals renamed (`denied-by-rules` → `reject`, etc.). Fallback chains collapsed to direct imports (`copilot` for SubprocessConfig, `copilot.session` for PermissionRequestResult). `make_permission_denied()` factory added to `_imports.py`; `client.py` now expresses intent only.
  - **1.3** — SDK v0.2.1: copilot/types.py deleted. Multi-level fallback required for any type that lived there.
  - **1.2** — SDK v0.2.0 API: SubprocessConfig, create_session kwargs, send(prompt)

---

## Overview

The SDK Adapter is **THE MEMBRANE** — the only place in the codebase where SDK imports are allowed. No SDK type crosses this boundary. Domain code never imports from SDK.

This contract ensures the provider remains testable, maintainable, and isolated from SDK changes.

---

## The Import Quarantine

### MUST Constraints

1. **MUST** confine ALL SDK imports to `sdk_adapter/` package
2. **SHOULD** consolidate SDK imports into ONE file (`_imports.py`)
   > **Note:** Currently SDK imports are spread across multiple files in sdk_adapter/. Target: create _imports.py to quarantine all SDK imports.
3. **MUST NOT** allow SDK imports in ANY module outside `sdk_adapter/`
4. **MUST NOT** export SDK types from `sdk_adapter/__init__.py`
5. **MUST** fail at import time with a clear error if `github-copilot-sdk` is not installed (eager dependency check)
6. **MUST** use direct imports for the currently pinned SDK version. Multi-level fallback chains are only needed when supporting a version range that spans a breaking import reorganisation. With `==1.0.0b10`, the canonical locations are: `CopilotClient` from `copilot` root (defined in `copilot.client`, re-exported at root, keyword-only `__init__` per `SDKSurface:MUST:8`); `ModelLimitsOverride` and `ModelCapabilitiesOverride` from `copilot` root; `PermissionRequestResult` from `copilot.session` (type alias `PermissionDecision | PermissionNoResult` — NOT a constructor); `PermissionDecisionReject` from `copilot.generated.rpc` (carve-out: variants of `PermissionDecision` are not re-exported at `copilot` root, so the membrane imports the variant directly from its canonical generated location per `copilot/generated/rpc.py:10054-10060`). If an import moves in a future version, update the pin AND the import together.
7. **MUST** permit imports from `copilot.generated.*` ONLY inside `sdk_adapter/_imports.py` AND ONLY for variant classes that the SDK does not re-export at the `copilot` package root (currently: `PermissionDecisionReject`). Each such import MUST be tagged with a `# TODO(owner YYYY):` or `# HACK(<url>):` marker citing the upstream re-export gap so the membrane debt is visible.
8. **MUST** encapsulate SDK constructor calls — including field names and Literal values — inside `_imports.py`. Other modules call factories (e.g., `make_permission_denied()`), not SDK constructors directly.

### SDK Version History (Import Changes)

#### SDK v0.3.0 — `PermissionRequestResult` reduced to kind-only; kind literals renamed

| Field | v0.2.2 | v0.3.0 |
|-------|--------|--------|
| `kind` | `"denied-by-rules"`, `"approved"`, … (6 values) | `"approve-once"`, `"reject"`, `"user-not-available"`, `"no-result"` |
| `rules` | present | **removed** |
| `feedback` | present | **removed** |
| `message` | present | **removed** |
| `path` | present | **removed** |

`SubprocessConfig` added `session_idle_timeout_seconds: int | None` (optional, unused by provider).

#### SDK v0.2.1 (2026-03-20) — `copilot.types` deleted (PR #871, brettcannon)

`copilot/types.py` was removed. Types redistributed to semantically owning modules:

| Type | v0.2.0 location | v0.2.1-b6 location | b10 status |
|------|-----------------|---------------------|------------|
| `PermissionRequestResult` | `copilot.types` | `copilot.session` | Type alias in `copilot.session` |
| `PermissionHandler` | `copilot.types` | `copilot.session` | `copilot.session` |
| `SubprocessConfig` | `copilot.types` | `copilot.client` | Removed in b7 |
| Tool types | `copilot.types` | `copilot.tools` | `copilot.tools` |

### Directory Structure

```
sdk_adapter/
├── __init__.py          # Exports ONLY domain types and SDK-independent utilities
├── _imports.py          # THE ONLY FILE with SDK imports (quarantined)
├── _spec_utils.py       # SDK-independent utilities (find_spec, no imports)
├── types.py             # Domain type definitions (SessionHandle, DomainEvent, SessionConfig)
├── client.py            # SDK session lifecycle (create, send, close)
├── extract.py           # SDK event → domain type extraction
├── event_helpers.py     # Event classification and translation helpers
├── tool_capture.py      # Tool capture handler (sdk-protection.md)
└── model_translation.py # SDK ModelInfo → CopilotModelInfo translation
```

---

## Membrane API Pattern

The `sdk_adapter/` package exposes a **public API** via `__init__.py`. All other modules are internal.

### MUST Constraints

1. **MUST** import from `sdk_adapter` (the package), NOT from `sdk_adapter._imports` (internal)
2. **MUST** re-export any utilities needed by domain code via `sdk_adapter/__init__.py`
3. **MUST NOT** allow domain code to reach into private modules (`_imports.py`)

### Example

**WRONG — Bypasses membrane:**
```python
# ❌ Domain code reaching into private module
from amplifier_module_provider_github_copilot.sdk_adapter._imports import get_copilot_spec_origin
```

**RIGHT — Uses membrane API:**
```python
# ✓ Domain code uses public API
from amplifier_module_provider_github_copilot.sdk_adapter import get_copilot_spec_origin
```

### Rationale

- **Encapsulation**: Internal restructuring doesn't break domain code
- **Testability**: Single mock target for SDK utilities
- **Discoverability**: `__init__.py` documents the public API

### Test Anchors

| Anchor | Clause |
|--------|--------|
| `sdk-boundary:Membrane:MUST:1` | Import from sdk_adapter, not _imports |
| `sdk-boundary:Membrane:MUST:2` | Re-export utilities via __init__.py |
| `sdk-boundary:Membrane:MUST:3` | No domain code imports from _imports |

---

## Type Translation Rules

### MUST Constraints

1. **MUST** translate SDK types to domain types at the boundary
2. **MUST** use decomposition, not wrapping
3. **MUST NOT** pass SDK types through the boundary
4. **MUST** expose only opaque domain handles (façade objects whose private state is hidden behind a stable, narrow surface) — never raw SDK object references

### Decomposition Pattern

**WRONG — Wrapping:**
```python
# ❌ SDK type leaks through wrapper
class SessionWrapper:
    def __init__(self, sdk_session: CopilotSession):
        self._session = sdk_session  # SDK type stored
```

**RIGHT — Decomposition (façade + decomposed events):**
```python
# ✓ Façade hides the SDK session; raw reference is private
class SessionHandle:
    __slots__ = ("_raw_session", "session_id")

    def __init__(self, raw_session: Any, session_id: str | None = None) -> None:
        self._raw_session = raw_session  # private — never exposed
        self.session_id = session_id or getattr(raw_session, "session_id", "unknown")

    def on(self, handler: Callable[[Any], None]) -> Callable[[], None]: ...
    async def send(self, prompt: str, *, attachments: list[dict[str, Any]] | None = None) -> None: ...
    async def abort(self) -> None: ...

# ✓ SDK event decomposed to domain primitives
@dataclass
class DomainEvent:
    type: str
    data: dict[str, Any]  # Decomposed, not SDK object
```

---

## Domain Types

### SessionHandle

```python
# Façade class wrapping the raw SDK session.
# NOT a string alias and NOT a registry-keyed entry — both patterns
# leak SDK-session ownership across the membrane (see "Why not a
# string alias?" rationale below).
class SessionHandle:
    __slots__ = ("_raw_session", "session_id")
    # on / send / abort delegate to the private _raw_session
```

- `Types:MUST:4` — **MUST** keep `_raw_session` private: no public attribute, no accessor returning it (`sdk_adapter/types.py:74,83`).
- `Types:MUST:5` — **MUST** expose only `on(handler) → unsubscribe`, `await send(prompt, *, attachments=None)`, `await abort()`, plus the attribute `session_id: str` (`sdk_adapter/types.py:86-120`).
- `Types:MUST:6` — **MUST NOT** expose lifecycle (`connect` / `disconnect` / `destroy` / `close`) on the handle; lifecycle is owned by the `client.session()` async context manager in `sdk_adapter/client.py`.
- `Types:MUST:7` — **MUST** assign `session_id` exactly once in `__init__` from the raw SDK session (`copilot.session.CopilotSession.session_id`); provider code MUST NOT reassign it. Callers **SHOULD** treat it as read-only — `__slots__` constrains attribute names but does not block reassignment of declared slots; a `__setattr__` guard or `frozen` dataclass is tracked as a code-level follow-up.

> **Why not a string alias?** A string-only handle forces an internal SDK-session registry, which leaks lifetime ownership across the membrane and complicates abort/cleanup. The façade pattern colocates the raw reference with its narrow surface, keeps the membrane self-contained, and matches the implementation at `sdk_adapter/types.py:61-120`. See `TypeTranslation:MUST:3` and `TypeTranslation:MUST:4`.

### DomainEvent

```python
@dataclass
class DomainEvent:
    type: str  # "CONTENT_DELTA", "TOOL_CALL", etc.
    data: dict[str, Any]
```

- **MUST** be a pure Python dataclass
- **MUST NOT** contain SDK event objects
- **MUST** use primitive types and dicts

---

## SDK Event Structure (v0.1.33+)

**Reference:** SDK SessionEvent structure from github-copilot-sdk

This section documents the **actual** SDK event structure for use in mocks and tests.
Provider code MUST extract content from these locations:

### SessionEvent Envelope

```python
@dataclass
class SessionEvent:
    data: Data              # Event payload (all fields nested here)
    id: UUID               # Unique event identifier
    timestamp: datetime    # ISO 8601 timestamp
    type: SessionEventType # Enum discriminator
    ephemeral: bool | None # True for transient events
    parent_id: UUID | None # Previous event in chain
```

### Content Location by Event Type

| SDK Event Type | Content Field | Python Accessor |
|----------------|---------------|-----------------|
| `assistant.message_delta` | `deltaContent` | `event.data.delta_content` |
| `assistant.reasoning_delta` | `deltaContent` | `event.data.delta_content` |
| `assistant.message` | `content` | `event.data.content` |
| `assistant.reasoning` | `content` | `event.data.content` |

### CRITICAL: There is NO `event.text` Field

**WRONG (fabricated):**
```python
text = event.text  # ❌ This field does not exist
```

**RIGHT (real SDK):**
```python
text = event.data.delta_content  # ✓ For streaming deltas
text = event.data.content        # ✓ For complete messages
```

### Mock Fixture Pattern

Test fixtures MUST match this structure:

```python
@dataclass
class MockData:
    delta_content: str | None = None
    content: str | None = None
    message_id: str | None = None
    reasoning_id: str | None = None

@dataclass  
class MockSDKEvent:
    type: str
    data: MockData
```

### Test Anchors

| Anchor | Clause |
|--------|--------|
| `sdk-boundary:EventShape:MUST:1` | Mock events have nested data attribute |
| `sdk-boundary:EventShape:MUST:2` | delta_content in data, not event.text |
| `sdk-boundary:EventShape:MUST:3` | Test mocks match SDK structure |

---

### SessionConfig

```python
@dataclass
class SessionConfig:
    """Field names reflect SDK reality, not aspirational contract naming.
    
    SDK uses 'system_prompt' not 'system_message'.
    """
    model: str
    system_prompt: str | None = None  # Becomes SDK system_message.content
    max_tokens: int | None = None
```

- **MUST** use primitives (str, dict, list)
- **MUST NOT** use SDK config types
- **NOTE**: Actual SDK session config uses `system_message: {mode, content}` dict,
  not a simple string. The SessionConfig dataclass captures intent; client.py
  transforms it to SDK format.

---

## Tool Forwarding Contract

The SDK has **three separate parameters** for tool configuration:

| Parameter | Type | Purpose |
|-----------|------|---------|
| `tools` | `list[Tool]` | **Custom tool definitions** — Amplifier's tools for the LLM |
| `available_tools` | `list[str]` | Built-in tool **name allowlist** — empty list disables all SDK built-ins |
| `excluded_tools` | `list[str]` | Built-in tool **name denylist** — ignored if `available_tools` is set |

### MUST Constraints

1. **MUST** forward `ChatRequest.tools` to SDK session via `session_config["tools"]`
2. **MUST** convert Amplifier `ToolSpec` objects to SDK-compatible objects with required attributes
3. **MUST** set `available_tools` to the list of Amplifier tool names (allowlist strategy per deny-destroy:Allowlist:MUST:1)
4. **MUST** set `overrides_built_in_tool=True` on all user tools (per deny-destroy:ToolSuppression:MUST:2)
5. **MUST NOT** confuse `tools` (custom definitions) with `available_tools` (built-in allowlist)

### SDK Tool Format

The SDK iterates tools and accesses **attributes** (not dict keys):

```python
for tool in tools:
    definition = {"name": tool.name, "description": tool.description}
    if tool.parameters:
        definition["parameters"] = tool.parameters
    if tool.overrides_built_in_tool:  # ← MUST exist as attribute
        definition["overridesBuiltInTool"] = True
    if tool.skip_permission:          # ← MUST exist as attribute
        definition["skipPermission"] = True
    if tool.defer is not None:        # ← SDK v1.0.2 reads tool.defer (client.py:1810-1811, :2391-2392)
        definition["defer"] = tool.defer
```

**Required attributes on each tool object:**
- `name: str` — tool name
- `description: str` — tool description
- `parameters: dict | None` — JSON Schema (optional)
- `overrides_built_in_tool: bool` — set to `True` to avoid SDK "conflicts with built-in" error for tools like "bash" (we disable built-ins via available_tools=[] anyway)
- `skip_permission: bool` — set to `False` (Amplifier handles permissions)
- `handler: None` — **MUST exist** (SDK checks handler attribute); set to `None` so SDK skips handler registration (Amplifier handles tools at kernel layer)
- `defer: Literal["auto", "never"] | None` — **MUST exist** (SDK v1.0.2 reads `tool.defer` when building tool definitions); set to `None` so the `defer` key is omitted from the wire payload (exact pre-v1.0.2 behavior). Amplifier pre-loads all tools at the kernel layer, so the SDK's lazy tool-search deferral stays off. Mirrors the SDK's own `copilot.tools.Tool.defer` field.

**Implementation:** Use `SDKToolWrapper` dataclass from `sdk_adapter/types.py`:
```python
# sdk_adapter/types.py — SDKToolWrapper and convert_tools_for_sdk()
@dataclass
class SDKToolWrapper:
    name: str
    description: str
    parameters: dict[str, Any] | None = None
    overrides_built_in_tool: bool = False
    skip_permission: bool = False
    handler: Any = None  # SDK checks this; None skips handler registration
    defer: Literal["auto", "never"] | None = None  # SDK v1.0.2 reads tool.defer; None = not deferred (omitted from wire)

def convert_tools_for_sdk(tools: list[Any]) -> list[SDKToolWrapper]:
    # Handles both ToolSpec objects (attribute access) and dicts
    ...
```

**Why not SDK `Tool` dataclass?** SDK `Tool` requires a `handler: ToolHandler` callable
with actual implementation. Amplifier tools have handlers at the kernel layer, not the
provider layer. `SDKToolWrapper` with `handler=None` provides required attributes
without importing SDK types, and causes SDK to skip handler registration.

### Why This Matters

Without tool definitions in `session_config["tools"]`:
- The LLM has no tool definitions to invoke
- LLM writes fake tool patterns as text (`<function_calls>`, `[Tool Call:]`)
- Provider returns raw text instead of structured `tool_calls`
- Foundation cannot render `🔧 Using tool:` formatting

### Input Tool Formats

The provider accepts tools from ChatRequest in two formats:

1. **Nested format** (OpenAI-style): `{"function": {"name": "...", "description": "...", "parameters": {...}}}`
   - Used when tools originate from OpenAI-compatible schemas
   
2. **Flat format** (Amplifier-native): `{"name": "...", "description": "...", "parameters": {...}}`
   - Used by Amplifier's internal `ToolSpec` Pydantic model (see `message_models.py`)

Both formats are valid and the provider handles them transparently during conversion to SDK format (SimpleNamespace objects).

### Test Anchors

| Anchor | Clause |
|--------|--------|
| `sdk-boundary:ToolForwarding:MUST:1` | tools from ChatRequest forwarded to session_config["tools"] |
| `sdk-boundary:ToolForwarding:MUST:2` | Amplifier tools converted to SDK format |
| `sdk-boundary:ToolForwarding:MUST:3` | available_tools set to tool names allowlist when tools provided; empty list when no tools |
| `sdk-boundary:ToolForwarding:MUST:4` | tools and available_tools not conflated |

---

## Translation Functions

### Event Translation

```python
def translate_sdk_event(sdk_event: Any, config: EventConfig) -> DomainEvent | None:
    """
    Translate SDK event to domain event.
    
    - MUST classify per config (BRIDGE/CONSUME/DROP)
    - MUST return None for DROP events
    - MUST NOT expose SDK event internals
    """
```

### Error Translation

```python
def translate_sdk_error(exc: Exception, config: ErrorConfig) -> LLMError:
    """
    Translate SDK exception to domain exception.
    
    - MUST NOT raise (always returns)
    - MUST preserve original in .original attribute
    - MUST use config patterns (no hardcoded mappings)
    """
```

---

## Session Configuration Contract

The dict passed to `client.create_session()` MUST satisfy these constraints:

### MUST Constraints

1. **MUST** set `available_tools` to the list of Amplifier tool names (allowlist)
2. **MUST** use `system_message.mode: "replace"` when system_message is provided
3. **MUST** set `on_permission_request` handler on every session
4. **MUST** set `streaming: true` for event-based tool capture
5. **MUST** pass deny hook via `session_config["hooks"]` at creation time (NOT registered after)
6. **MUST NOT** include keys that are not in SDK's SessionConfig TypedDict

### Rationale

- **available_tools=<tool_names>**: SDK exposes built-in tools (list_agents, bash, view, edit) by default. Setting `available_tools` to the list of Amplifier tool names creates an allowlist—only those tools are visible to the model. This prevents SDK built-ins from appearing in completions. Note: `available_tools=[]` would disable ALL tools including Amplifier's, so we use a non-empty allowlist.
- **mode="replace"**: With "append", SDK injects "You are GitHub Copilot CLI..." before our system message. With "replace", our bundle persona takes precedence.
- **on_permission_request**: SDK v0.1.33+ requires this handler. We deny all permission requests as the first line of defense.
- **streaming=true**: Required for event-based tool capture. Non-streaming mode cannot capture tool calls.

---

## SDK Minimal Mode Configuration

**Purpose:** Disable SDK features that Amplifier handles, reducing overhead and ensuring Amplifier is the true orchestrator.

**Evidence:** Sessions `7db2b5f7-28e8-49ca-aa6c-562a65331ec4` (baseline) and `2fa58db6-7a30-4d78-8bf8-e9ad3f4c54bf` showed 57% wall-clock improvement (12.5s → 5.4s) and elimination of compaction processing.

### Constraints

| ID | Constraint | Rationale |
|----|------------|--------|
| sdk-boundary:MinimalMode:MUST:1 | `infinite_sessions` MUST be set to `{"enabled": False}` | Disables SDK compaction — Amplifier handles context management |
| sdk-boundary:MinimalMode:MUST:2 | `enable_config_discovery` MUST be set to `False` | Prevents SDK from scanning for .mcp.json and AGENTS.md — Amplifier provides all config |
| sdk-boundary:MinimalMode:MUST:3 | `mcp_servers` MUST be set to `{}` | Explicit empty — Amplifier routes all tools |
| sdk-boundary:MinimalMode:MUST:4 | `skill_directories` MUST be set to `[]` | Explicit empty — Amplifier has its own skills system |
| sdk-boundary:MinimalMode:MUST:5 | `custom_agents` MUST be set to `[]` | Explicit empty — Amplifier orchestrates agents |
| sdk-boundary:MinimalMode:MUST:6 | `commands` MUST be set to `[]` | Explicit empty — Amplifier handles slash commands |
| sdk-boundary:MinimalMode:MUST:7 | `enable_session_store` MUST be set to `False` | Disables SDK cross-session persistent store — ephemeral per-`complete()` sessions (deny-destroy contract). Pinned because `mode="copilot-cli"` does not invoke the empty-mode default helper at b10 `_mode.py`. |
| sdk-boundary:MinimalMode:MUST:8 | `enable_skills` MUST be set to `False` | Disables SDK skills loader — Amplifier has its own skills system. Stronger than `skill_directories=[]` (MUST:4); pinned for the same reason. |
| sdk-boundary:MinimalMode:MUST:9 | `enable_file_hooks` MUST be set to `False` | Disables SDK file-hook discovery (AGENTS.md walkers) — Amplifier registers a single deny-all hook explicitly. |
| sdk-boundary:MinimalMode:MUST:10 | `enable_host_git_operations` MUST be set to `False` | Disables SDK host-git delegation — Amplifier never delegates git to the SDK; deny-hook already blocks every tool. |
| sdk-boundary:MinimalMode:MUST:11 | `enable_on_demand_instruction_discovery` MUST be set to `False` | Disables SDK on-demand instruction scans — additional scan suppression complementing `enable_config_discovery=False` (MUST:2). |
| sdk-boundary:MinimalMode:MUST:12 | `skip_embedding_retrieval` MUST be set to `True` | Disables SDK embedding-based workspace retrieval — Amplifier owns retrieval/context. Pinned because `mode="copilot-cli"` does not invoke the empty-mode default helper at b10 `_mode.py:193-198`. |
| sdk-boundary:MinimalMode:MUST:13 | `embedding_cache_storage` MUST be set to `"in-memory"` | Prevents persistent disk cache of workspace embeddings — aligns with deny-destroy / ephemeral session. Pinned because `mode="copilot-cli"` does not invoke the empty-mode default helper at b10 `_mode.py:201-208`. |
| sdk-boundary:MinimalMode:MUST:14 | `enable_session_telemetry` MUST be set to `False` | Disables SDK-internal session telemetry — Amplifier owns observability. Pinned because b10 `client.py:1651-1656` documents telemetry as ON-by-default for GitHub-authenticated sessions and our `COPILOT_AGENT_TOKEN` path IS that path; without pin, leaving kwarg `None` lets the bundled CLI emit telemetry by default. |
| sdk-boundary:MinimalMode:MUST:15 | `mcp_oauth_token_storage` MUST be set to `"in-memory"` | Keeps MCP OAuth tokens in RAM — no on-disk token residue across the ephemeral session boundary. Pinned because `mode="copilot-cli"` does not invoke the empty-mode default helper at b10 `_mode.py:251-258`; the wire-emit (b10 `client.py:1863-1865`) is INDEPENDENT of the `mcpServers` emit (`client.py:1860-1861`), so `mcp_servers={}` (MUST:3) does NOT foreclose it. |
| sdk-boundary:MinimalMode:MUST:16 | `memory` MUST be set to `{"enabled": False}` | Disables the SDK v1.0.2 mode-gated session-memory capability — Amplifier owns all context/memory; aligns with deny-destroy / ephemeral per-`complete()` sessions. Pinned because `mode="copilot-cli"` does not invoke the empty-mode default helper at installed v1.0.2 `_mode.py:264-276` (`_memory_default` returns `{"enabled": False}` only when `mode == "empty"`). |

### Test Anchors

| Anchor | Clause | Test Location |
|--------|--------|---------------|
| `sdk-boundary:MinimalMode:MUST:1` | infinite_sessions disabled | `tests/test_sdk_boundary_contract.py::TestMinimalModeConfig::test_infinite_sessions_disabled` |
| `sdk-boundary:MinimalMode:MUST:2` | config discovery disabled | `tests/test_sdk_boundary_contract.py::TestMinimalModeConfig::test_config_discovery_disabled` |
| `sdk-boundary:MinimalMode:MUST:3` | mcp_servers empty | `tests/test_sdk_boundary_contract.py::TestMinimalModeConfig::test_mcp_servers_empty` |
| `sdk-boundary:MinimalMode:MUST:4` | skill_directories empty | `tests/test_sdk_boundary_contract.py::TestMinimalModeConfig::test_skill_directories_empty` |
| `sdk-boundary:MinimalMode:MUST:5` | custom_agents empty | `tests/test_sdk_boundary_contract.py::TestMinimalModeConfig::test_custom_agents_empty` |
| `sdk-boundary:MinimalMode:MUST:6` | commands empty | `tests/test_sdk_boundary_contract.py::TestMinimalModeConfig::test_commands_empty` |
| `sdk-boundary:MinimalMode:MUST:7` | enable_session_store disabled | `tests/test_sdk_boundary_contract.py::TestMinimalModeConfig::test_enable_session_store_disabled` |
| `sdk-boundary:MinimalMode:MUST:8` | enable_skills disabled | `tests/test_sdk_boundary_contract.py::TestMinimalModeConfig::test_enable_skills_disabled` |
| `sdk-boundary:MinimalMode:MUST:9` | enable_file_hooks disabled | `tests/test_sdk_boundary_contract.py::TestMinimalModeConfig::test_enable_file_hooks_disabled` |
| `sdk-boundary:MinimalMode:MUST:10` | enable_host_git_operations disabled | `tests/test_sdk_boundary_contract.py::TestMinimalModeConfig::test_enable_host_git_operations_disabled` |
| `sdk-boundary:MinimalMode:MUST:11` | enable_on_demand_instruction_discovery disabled | `tests/test_sdk_boundary_contract.py::TestMinimalModeConfig::test_enable_on_demand_instruction_discovery_disabled` |
| `sdk-boundary:MinimalMode:MUST:12` | embedding retrieval skipped | `tests/test_sdk_boundary_contract.py::TestMinimalModeConfig::test_embedding_retrieval_skipped` |
| `sdk-boundary:MinimalMode:MUST:13` | embedding cache in-memory | `tests/test_sdk_boundary_contract.py::TestMinimalModeConfig::test_embedding_cache_storage_in_memory` |
| `sdk-boundary:MinimalMode:MUST:14` | session telemetry disabled | `tests/test_sdk_boundary_contract.py::TestMinimalModeConfig::test_enable_session_telemetry_disabled` |
| `sdk-boundary:MinimalMode:MUST:15` | mcp oauth token storage in-memory | `tests/test_sdk_boundary_contract.py::TestMinimalModeConfig::test_mcp_oauth_token_storage_in_memory` |
| `sdk-boundary:MinimalMode:MUST:16` | memory disabled | `tests/test_sdk_boundary_contract.py::TestMinimalModeConfig::test_memory_disabled` |

### Out of scope (mode-gated defaults not pinned)

The 10 pins above cover every SDK mode-gated capability default reached through
`create_session` — the `_<kwarg>_default(mode, supplied)` helpers in b10
`_mode.py:185-258`, plus v1.0.2's `_memory_default` (`_mode.py:264-276`; v1.0.2
shifted the `_mode.py` line ranges, so these offsets are version-tagged and not
contiguous with the b10 block). Two related SDK defaults are deliberately NOT pinned:

- **`manage_schedule_enabled` / `coauthor_enabled`** — these ARE
  `create_session` kwargs (b10 `client.py:1576-1577`) but have no
  `_<kwarg>_default(mode, supplied)` mode-gated helper in `_mode.py:185-258`.
  They flow through `_post_create_options_patch` (b10 `_mode.py:261-298`),
  applied internally by `create_session` via `_apply_post_create_options_patch`
  (b10 `client.py:2099`). The provider passes neither, so both default to
  `None`; under `mode="copilot-cli"` the patch builder returns `None`
  (`_mode.py:298`, `return patch or None`) and `session.options.update` is
  never called — neither option reaches the wire. `coauthor_enabled` is
  additionally neutralized by MUST:10 (`enable_host_git_operations=False`).
  Pinning either to `False` would add an `options.update` round-trip the
  provider does not currently make; tracked as a follow-up if the provider
  adopts that path.
- **`mcp_oauth_token_storage` on `resume_session`** — `resume_session`
  (b10 `client.py:2117`) carries the same independent emit at
  `client.py:2429-2431`, but the provider creates sessions only via
  `create_session`; the resume path is unused. If the provider ever resumes
  sessions, MUST:15 must be extended to that call site.

---

## Test Anchors

### Membrane

| Anchor | Clause |
|--------|--------|
| `sdk-boundary:Membrane:MUST:1` | All SDK imports in adapter only |
| `sdk-boundary:Membrane:MUST:2` | Only _imports.py has SDK imports |
| `sdk-boundary:Membrane:MUST:5` | Fail at import time if SDK not installed |

### ImportQuarantine

| Anchor | Clause |
|--------|--------|
| `sdk-boundary:ImportQuarantine:MUST:6` | Direct imports for pinned SDK version — no fallback chains (`CopilotClient` from `copilot` root [defined in `copilot.client`, re-exported at root, keyword-only `__init__` per `SDKSurface:MUST:8`]; `ModelLimitsOverride` and `ModelCapabilitiesOverride` from `copilot` root; `PermissionRequestResult` from `copilot.session` as a type alias; `PermissionDecisionReject` from `copilot.generated.rpc` because `PermissionDecision` variants are not re-exported at the package root). |
| `sdk-boundary:ImportQuarantine:MUST:7` | `copilot.generated.*` imports are restricted to `sdk_adapter/_imports.py`, limited to non-root-reexported SDK variant classes, and marked with upstream re-export debt. |
| `sdk-boundary:ImportQuarantine:MUST:8` | SDK constructor calls encapsulated in _imports.py via factory (make_permission_denied); client.py expresses intent only |

### SDKSurface (v1.0.0b10 shape pins)

These anchors pin specific shapes of the SDK v1.0.0b10 public surface that the
provider's stubs (`typings/copilot/`) and translation code rely on. A test
under each anchor lives in `tests/test_sdk_assumptions.py::TestSDKImportAssumptions`
and turns red if the live SDK surface drifts.

| Anchor | Clause |
|--------|--------|
| `sdk-boundary:SDKSurface:MUST:1` | `copilot.session.PermissionRequestResult` is the type alias `PermissionDecision \| PermissionNoResult` (b10 `session.py:275`) — NOT a constructor; callers MUST NOT invoke `PermissionRequestResult(kind=...)`. Permission-denial intent is expressed via `copilot.generated.rpc.PermissionDecisionReject()` (b10 `generated/rpc.py:10054`); `kind` is a codegen `ClassVar` and MUST NOT be passed by callers. Default `feedback=None` is omitted from `to_dict()` — equivalent to the previous silent-reject behavior. |
| `sdk-boundary:SDKSurface:MUST:2` | `copilot.session.ReasoningEffort` is the same object (identity, not just equal) as `copilot.client.ReasoningEffort` |
| `sdk-boundary:SDKSurface:MUST:3` | `copilot.ModelSupportsOverride` is a dataclass with field tuple `("vision", "reasoning_effort")`, all defaults `None` |
| `sdk-boundary:SDKSurface:MUST:4` | `copilot.ModelVisionLimitsOverride` is a dataclass with field tuple `("supported_media_types", "max_prompt_images", "max_prompt_image_size")`, all defaults `None` |
| `sdk-boundary:SDKSurface:MUST:5` | `copilot.session.CopilotSession.workspace_path` is a `functools.cached_property` |
| `sdk-boundary:SDKSurface:MUST:6` | `copilot.session.CopilotSession.send` accepts exactly the kwargs `{"prompt", "attachments", "mode", "agent_mode", "request_headers", "display_prompt"}` (positional `prompt`, five keyword-only: `attachments`, `mode`, `agent_mode`, `request_headers`, `display_prompt`) per b10 `session.py:L1185-L1194`, so the test mock signature in `tests/fixtures/sdk_mocks.py` and the provider call sites in `sdk_adapter/client.py` stay synchronised with the live SDK. `agent_mode` is the `Literal["interactive", "plan", "autopilot", "shell"] | None` UI-mode kwarg added in b9; `display_prompt` is the `str | None` UI-display-only kwarg added in b10 (lets callers send a separate display string from the actual prompt). |
| `sdk-boundary:SDKSurface:MUST:7` | `copilot.client.ModelBilling.from_dict` tolerates the live GitHub server shape — a payload with `restricted_to` and `token_prices` but no `multiplier` — without raising. `multiplier` MUST remain optional (`float \| None`, no invented default), so `list_models()` does not abort the whole batch on the first billing-carrying model. |
| `sdk-boundary:SDKSurface:MUST:8` | `copilot.CopilotClient.__init__` is keyword-only and accepts at least the keywords `{connection, working_directory, log_level, env, github_token, base_directory, use_logged_in_user, telemetry, session_fs, session_idle_timeout_seconds, enable_remote_sessions, on_list_models, mode}` (b10 `client.py:1073-1089`). An introspection test in `tests/test_sdk_assumptions.py::TestSDKImportAssumptions` MUST fail loudly on drift. |

### Types

| Anchor | Clause |
|--------|--------|
| `sdk-boundary:Types:MUST:1` | No SDK types cross boundary |
| `sdk-boundary:Types:MUST:2` | Domain types are dataclasses/primitives |
| `sdk-boundary:Types:MUST:3` | SessionHandle is a domain façade hiding the raw SDK session; raw session must not leak through `client.session()` |
| `sdk-boundary:Types:MUST:4` | `SessionHandle._raw_session` is private; no public attribute or accessor exposes it |
| `sdk-boundary:Types:MUST:5` | `SessionHandle` public surface is exactly `on`, `send`, `abort`, `session_id` |
| `sdk-boundary:Types:MUST:6` | `SessionHandle` does not expose `connect`/`disconnect`/`destroy`/`close`; lifecycle owned by `client.session()` |
| `sdk-boundary:Types:MUST:7` | `SessionHandle.session_id` assigned once in `__init__` from raw SDK session; provider code does not reassign it (read-only enforcement is a code-level follow-up) |

### Translation

| Anchor | Clause |
|--------|--------|
| `sdk-boundary:Translation:MUST:1` | Events translated to DomainEvent |
| `sdk-boundary:Translation:MUST:2` | Errors translated to domain exceptions |

### TypeTranslation (Test Mocks)

| Anchor | Clause |
|--------|--------|
| `sdk-boundary:TypeTranslation:MUST:1` | SDK types translated to domain types at boundary |
| `sdk-boundary:TypeTranslation:MUST:2` | Mock sessions deliver SessionEvent objects to handlers |
| `sdk-boundary:TypeTranslation:MUST:3` | SessionHandle wraps the raw SDK session; raw session must not be directly exposed to callers |
| `sdk-boundary:TypeTranslation:MUST:4` | SessionHandle delegates `on()`, `send()`, and `abort()` to the raw SDK session without leaking SDK types; session lifecycle (connect / disconnect / destroy) is owned by the `client.session()` async context manager and MUST NOT be exposed on SessionHandle |
| `sdk-boundary:TypeTranslation:SHOULD:1` | Mock sessions accept legacy dict events for backward compat |

### Config

| Anchor | Clause |
|--------|--------|
| `sdk-boundary:Config:MUST:1` | available_tools set to Amplifier tool names allowlist (or empty list if no tools) |
| `sdk-boundary:Config:MUST:2` | system_message mode is replace |
| `sdk-boundary:Config:MUST:3` | on_permission_request always set |
| `sdk-boundary:Config:MUST:4` | streaming is true |
| `sdk-boundary:Config:MUST:5` | deny hook passed via session_config["hooks"] at creation time |
| `sdk-boundary:Config:MUST:6` | no unknown keys in config |

### Model Discovery

| Anchor | Clause |
|--------|--------|
| `sdk-boundary:ModelDiscovery:MUST:1` | Fetches from SDK list_models() API |
| `sdk-boundary:ModelDiscovery:MUST:2` | Translates SDK → CopilotModelInfo |
| `sdk-boundary:ModelDiscovery:MUST:3` | Translates CopilotModelInfo → amplifier_core.ModelInfo |
| `sdk-boundary:ModelDiscovery:MUST_NOT:1` | No hardcoded model lists |

### SDK API Assumptions (SDK v1.0.0b10)

| Anchor | Clause |
|--------|--------|
| `sdk-boundary:Session:MUST:1` | SDK CopilotClient.create_session() accepts kwargs (model=, streaming=, on_permission_request=, hooks=) |
| `sdk-boundary:Lifecycle:MUST:1` | SDK CopilotClient has async start() and stop() lifecycle methods |
| `sdk-boundary:Auth:MUST:1` | SDK CopilotClient accepts `github_token` as a keyword-only constructor argument (b10 `client.py:1073-1089`, wired to `COPILOT_SDK_AUTH_TOKEN` env at `client.py:3189`). |
| `sdk-boundary:Auth:MUST:2` | Token resolution follows SDK priority order; empty string treated as absent |
| `sdk-boundary:Auth:MUST:3` | An explicitly-resolved token MUST be passed as the `github_token` kwarg to CopilotClient. b10 keeps the direct-kwarg surface (no intermediate config object that could silently drop the token); the only residual fail-closed sentinel is the test-mode case where CopilotClient itself is unavailable (`SKIP_SDK_CHECK` + pytest), which raises ConfigurationError to prevent silent fall-through to ambient auth. |
| `sdk-boundary:Events:MUST:1` | Provider uses session.on() + session.send(prompt, attachments=...) pattern |
| `sdk-boundary:Send:MUST:1` | session.send(prompt: str, attachments=...) replaces send({"prompt":...}) |
| `sdk-boundary:Models:MUST:1` | SDK CopilotClient.list_models() returns list[ModelInfo] |

---

## Model Discovery

### Overview

Model discovery MUST fetch models dynamically from the SDK backend. The provider MUST NOT use hardcoded model lists or fallback dictionaries.

### MUST Constraints

1. **MUST** fetch models from SDK `list_models()` API
2. **MUST** translate SDK `ModelInfo` to domain `CopilotModelInfo` (isolation layer)
3. **MUST** translate `CopilotModelInfo` to `amplifier_core.ModelInfo` (kernel contract)
4. **MUST NOT** use hardcoded model lists in production code

### Type Translation Chain

```
SDK ModelInfo          →  CopilotModelInfo       →  amplifier_core.ModelInfo
(copilot.client)          (internal isolation)      (kernel expects this)
```

**Why Three Types?**
- **SDK ModelInfo**: SDK's type structure (may change with SDK versions)
- **CopilotModelInfo**: Our isolation layer — insulates us from SDK changes
- **amplifier_core.ModelInfo**: What the kernel expects from `provider.list_models()`

### Type Translation

```python
# SDK ModelInfo (from copilot.client) — INPUT
@dataclass
class ModelInfo:
    id: str
    name: str
    capabilities: ModelCapabilities  # contains .limits.max_context_window_tokens

# Domain CopilotModelInfo (in models.py) — ISOLATION LAYER
@dataclass(frozen=True)
class CopilotModelInfo:
    id: str
    name: str
    context_window: int
    max_output_tokens: int
    supports_vision: bool = False
    supports_reasoning_effort: bool = False
    context_window_default: int = 0   # prompt budget: billing.token_prices.context_max OR limits.max_prompt_tokens OR static; 0 => use context_window
    context_window_long: int = 0      # long-tier prompt budget: billing.token_prices.long_context.context_max OR context_window_default; 0 => no long tier

# context_window_default / context_window_long are populated by sdk_model_to_copilot_model from the
# SDK billing surface; the 0 defaults preserve positional construction in fixtures and tolerant
# disk-cache reads at _SUPPORTED_CACHE_VERSION = "1.0" (no version bump).

# amplifier_core.ModelInfo — OUTPUT (what kernel expects)
# Imported from amplifier_core, NOT defined by us
from amplifier_core import ModelInfo as AmplifierModelInfo
```

### Limit Derivation

```python
# max_output_tokens = context_window - max_prompt_tokens
context_window = capabilities.limits.max_context_window_tokens
max_prompt = capabilities.limits.max_prompt_tokens
max_output_tokens = context_window - max_prompt
```

### Test Anchors

| Anchor | Clause | Test Reference |
|--------|--------|----------------|
| `sdk-boundary:ModelDiscovery:MUST:1` | Fetches from SDK API | `test_models.py::test_fetch_calls_sdk_list_models` |
| `sdk-boundary:ModelDiscovery:MUST:2` | Translates SDK → CopilotModelInfo | `test_models.py::test_copilot_model_to_internal_extracts_limits` |
| `sdk-boundary:ModelDiscovery:MUST:3` | Translates CopilotModelInfo → amplifier_core.ModelInfo | `test_models.py::test_to_amplifier_model_info_maps_all_fields` |
| `sdk-boundary:ModelDiscovery:MUST_NOT:1` | No hardcoded lists | `test_models.py::test_no_hardcoded_model_list` |

---

## Image/Attachment Passthrough

**Evidence:** SDK `session.send()` signature

The provider acts as a **pure transport layer** for images. No capability validation, no filtering.

### Core Principle

```
ImageBlock (amplifier-core) → BlobAttachment (SDK) → upstream model
```

The provider MUST extract images from ChatRequest and forward them unchanged to the SDK.
The SDK and upstream model handle capability verification.

### Type Mapping

| amplifier-core | SDK | Notes |
|----------------|-----|-------|
| `ImageBlock.source["data"]` | `BlobAttachment["data"]` | Base64 string, no modification |
| `ImageBlock.source["media_type"]` | `BlobAttachment["mimeType"]` | e.g., "image/png" |
| `ImageBlock.source["type"]` | (must be "base64") | URL images not supported |

### BlobAttachment Structure (SDK)

```python
{
    "type": "blob",
    "data": "<base64-encoded-image>",
    "mimeType": "image/png",  # Or image/jpeg, image/webp, image/gif
    "displayName": "image.png"  # Optional
}
```

### MUST Constraints

1. **MUST** extract images from the LAST user message only (SDK limitation)
2. **MUST** convert `ImageBlock` source dict to `BlobAttachment` format
3. **MUST** skip non-base64 images (URL references) — return None
4. **MUST** skip empty/missing image data — return None
5. **MUST NOT** validate model vision capability — pure passthrough
6. **MUST NOT** filter or modify image content
7. **MUST** forward valid attachments to SDK `session.send(prompt, attachments=...)`

### Error Handling

When SDK rejects invalid images (e.g., unsupported MIME type), the SDK returns:
```
CAPIError: 400 invalid request body, failed to validate schema: ...image_url...
```

This MUST be translated to `InvalidRequestError` (non-retryable) per `error-hierarchy.md`.

### Test Anchors

| Anchor | Clause |
|--------|--------|
| `sdk-boundary:ImagePassthrough:MUST:1` | Extract images from LAST user message only |
| `sdk-boundary:ImagePassthrough:MUST:2` | Convert ImageBlock to BlobAttachment |
| `sdk-boundary:ImagePassthrough:MUST:3` | Skip non-base64 images |
| `sdk-boundary:ImagePassthrough:MUST:4` | Skip empty image data |
| `sdk-boundary:ImagePassthrough:MUST:5` | No model capability validation |
| `sdk-boundary:ImagePassthrough:MUST:6` | No image content modification |
| `sdk-boundary:ImagePassthrough:MUST:7` | Forward attachments via send() |

---

### Client Lifecycle

| Anchor | Clause |
|--------|--------|
| `sdk-boundary:client-lifecycle:MUST:1` | Failed start() clears _owned_client to None |
| `sdk-boundary:client-lifecycle:MUST:2` | Retry after failure reinitializes CopilotClient |
| `sdk-boundary:client-lifecycle:MUST:3` | Original exception propagates (not swallowed) |
| `sdk-boundary:client-lifecycle:REGRESSION` | Successful start retains _owned_client for reuse |

**Teardown latency (v1.0.2 graceful shutdown — observed, SDK-owned).** `close()`
calls SDK `stop()`, which (v1.0.2) requests a graceful runtime shutdown before
killing the CLI subprocess. Observed on this exact path: the graceful runtime-shutdown
RPC is acked in **<1s** (`stop()` logs `runtime shutdown complete`, success
branch `client.py:1455` — NOT the `failed` branch `:1462`), but the drained CLI
process does not self-exit, so the bounded post-ack `wait(timeout=_RUNTIME_SHUTDOWN_TIMEOUT_SECONDS=10)`
(`client.py:1491-1496`) elapses and `terminate()` (`:1498`) reaps it. Net: a clean
`close()` takes **~12s, once per lifecycle**, bounded and deterministic, with
**zero orphaned CLI processes** and no `SIGKILL`. The runtime is **reused across
turns**, so this
cost is paid only at lifecycle end, never per turn. This is SDK-internal behavior;
the provider only invokes `stop()`. Consumers wrapping the provider should allow
≥15s for `close()`.

---

## Authentication

### Token Resolution

The provider MUST resolve auth tokens from environment variables in the official SDK priority order
(documented in SDK `docs/auth/index.md`).

### MUST Constraints

1. **MUST** scan environment variables in this exact order: `COPILOT_AGENT_TOKEN`, `COPILOT_GITHUB_TOKEN`, `GH_TOKEN`, `GITHUB_TOKEN`
2. **MUST** treat an empty string token as absent — resolution MUST continue to the next candidate
3. **MUST** return `None` when no non-empty token is found in any variable
4. **MUST** pass an explicitly-resolved token as the `github_token` kwarg to `CopilotClient`
5. **MUST NOT** silently ignore an explicit token under any circumstance, including when `SKIP_SDK_CHECK` is set
6. **MUST NOT** fall through to default/ambient SDK authentication when an explicit token is present but cannot be applied

### Rationale

- **Priority order**: Agent-mode tokens (`COPILOT_AGENT_TOKEN`) take highest precedence; GitHub Actions tokens (`GITHUB_TOKEN`) are lowest. This matches the SDK's documented auth hierarchy.
- **Empty-string fallthrough**: Prevents treating declared-but-empty env vars as valid tokens.
- **Direct token wiring**: b10 passes `github_token` directly to `CopilotClient` and then to `COPILOT_SDK_AUTH_TOKEN` (`client.py:3189`). The security guarantee is structural: there is no intermediate config object that can be absent while token resolution succeeds.

### Test Anchors

| Anchor | Clause |
|--------|--------|
| `sdk-boundary:Auth:MUST:1` | SDK CopilotClient accepts `github_token` as a keyword-only constructor argument (b10 `client.py:1073-1089`, wired to `COPILOT_SDK_AUTH_TOKEN` env at `client.py:3189`). |
| `sdk-boundary:Auth:MUST:2` | Token resolution follows SDK priority order; empty string treated as absent |
| `sdk-boundary:Auth:MUST:3` | An explicitly-resolved token MUST be passed as the `github_token` kwarg to CopilotClient. b10 keeps the direct-kwarg surface (no intermediate config object that could silently drop the token); the only residual fail-closed sentinel is the test-mode case where CopilotClient itself is unavailable (`SKIP_SDK_CHECK` + pytest), which raises ConfigurationError to prevent silent fall-through to ambient auth. |

---

## Why This Matters

1. **Testability** — Domain code testable without SDK installation
2. **Maintainability** — SDK changes isolated to adapter
3. **Clarity** — Clear boundary between "our code" and "SDK code"
4. **Safety** — SDK bugs can't leak through abstraction

---

---

## Binary Resolution

The provider MUST locate and execute the Copilot CLI binary across all supported platforms.

### MUST Constraints

1. **MUST** detect platform (Windows/macOS/Linux) at runtime via `sys.platform`
2. **MUST** locate SDK binary via `importlib.util.find_spec()` — NOT by importing the SDK
3. **MUST** use platform-appropriate binary name (`copilot` vs `copilot.exe`)
4. **MUST** prefer SDK-bundled binary over PATH lookup
5. **MUST** fall back to PATH when SDK binary unavailable
6. **MUST** set execute permission (`S_IXUSR|S_IXGRP`, NOT `S_IXOTH`) on Unix
7. **MUST** be no-op for permissions on Windows
8. **MUST** raise if binary not found (mount() signals failure, not opt-out)
9. **MUST** verify execute permission persisted (post-`chmod` `stat()`) and fail
   with `ProviderUnavailableError` BEFORE subprocess launch when the filesystem
   silently discards mode bits (e.g. NTFS via WSL `/mnt/` without DrvFs
   `metadata`, some FUSE drivers, network-share mounts). Diagnostic message
   MUST be platform-aware (WSL hint vs generic Unix hint vs Windows defensive)
   and MUST include the binary path for actionability.
10. **MUST** preserve typed amplifier-core errors (`ConfigurationError`,
    `ProviderUnavailableError`) raised inside the session-creation try block —
    they MUST NOT be re-translated by the catchall `Exception` handler.
    Rationale: the catchall translator is for raw SDK/OS exceptions; typed
    amplifier-core errors carry intentional fail-closed / routing semantics
    (Security P1-6, error-class-based orchestrator retry) that re-translation
    would silently destroy.

### Test Anchors

| Anchor | Clause |
|--------|--------|
| `sdk-boundary:BinaryResolution:MUST:1` | Platform detection |
| `sdk-boundary:BinaryResolution:MUST:2` | find_spec not import |
| `sdk-boundary:BinaryResolution:MUST:3` | Binary name selection |
| `sdk-boundary:BinaryResolution:MUST:4` | SDK binary preferred |
| `sdk-boundary:BinaryResolution:MUST:5` | PATH fallback |
| `sdk-boundary:BinaryResolution:MUST:6` | Execute permission |
| `sdk-boundary:BinaryResolution:MUST:7` | Windows no-op |
| `sdk-boundary:BinaryResolution:MUST:8` | Raises if binary not found (mount() failure, not opt-out) |
| `sdk-boundary:BinaryResolution:MUST:9` | Verified execute permission + platform-aware diagnostic |
| `sdk-boundary:BinaryResolution:MUST:10` | Typed amplifier-core errors not re-translated |

---

## Verification

To verify this contract:

```bash
# Should find SDK imports ONLY in _imports.py
grep -r "from copilot" amplifier_module_provider_github_copilot/
grep -r "import copilot" amplifier_module_provider_github_copilot/
```

Expected: Only `sdk_adapter/_imports.py` matches.
