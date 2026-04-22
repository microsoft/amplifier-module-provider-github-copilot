# Contract: SDK Boundary (The Membrane)

## Version
- **Current:** 1.3 (SDK v0.2.1 Breaking Changes)
- **Module Reference:** amplifier_module_provider_github_copilot/sdk_adapter/
- **Status:** Non-Negotiable Constraint
- **Update:** 2026-04-04 — SDK v0.2.1: copilot.types deleted, PermissionRequestResult → copilot.session
- **History:**
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
6. **MUST** maintain multi-level fallback chains for any SDK type that has moved between versions: `copilot.types` (v0.2.0) → `copilot` root (potential re-export) → new canonical module (v0.2.1+) → `None`. This pattern is established by `SubprocessConfig` and `PermissionRequestResult` in `_imports.py`.

### SDK Version History (Import Changes)

#### SDK v0.2.1 (2026-03-20) — `copilot.types` deleted (PR #871, brettcannon)

`copilot/types.py` was removed. Types redistributed to semantically owning modules:

| Type | v0.2.0 location | v0.2.1+ location | Re-exported from root? |
|------|-----------------|------------------|------------------------|
| `PermissionRequestResult` | `copilot.types` | `copilot.session` | No |
| `PermissionHandler` | `copilot.types` | `copilot.session` | No |
| `SubprocessConfig` | `copilot.types` | `copilot.client` | **Yes** — `copilot.SubprocessConfig` works |
| Tool types | `copilot.types` | `copilot.tools` | Yes |

`copilot.__init__` now exports only: `CopilotClient`, `CopilotSession`, connection configs, `define_tool`.

**Required fallback pattern** (established in `_imports.py`):
```python
try:
    from copilot.types import X  # v0.2.0
except ImportError:
    try:
        from copilot import X    # potential root re-export
    except ImportError:
        try:
            from copilot.NEW_MODULE import X  # v0.2.1+ canonical location
        except ImportError:
            X = None  # pre-existence stub
```

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
4. **MUST** use opaque handles (strings) instead of SDK object references

### Decomposition Pattern

**WRONG — Wrapping:**
```python
# ❌ SDK type leaks through wrapper
class SessionWrapper:
    def __init__(self, sdk_session: CopilotSession):
        self._session = sdk_session  # SDK type stored
```

**RIGHT — Decomposition:**
```python
# ✓ SDK type decomposed to domain primitives
SessionHandle = str  # Opaque UUID, not SDK reference

@dataclass
class DomainEvent:
    type: str
    data: dict[str, Any]  # Decomposed, not SDK object
```

---

## Domain Types

### SessionHandle

```python
# Opaque handle — UUID string, NOT SDK session reference
SessionHandle = str
```

- **MUST** be generated UUID, not SDK session ID
- **MUST NOT** be the actual SDK session object
- **MUST** map to SDK session via internal registry

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
```

**Required attributes on each tool object:**
- `name: str` — tool name
- `description: str` — tool description
- `parameters: dict | None` — JSON Schema (optional)
- `overrides_built_in_tool: bool` — set to `True` to avoid SDK "conflicts with built-in" error for tools like "bash" (we disable built-ins via available_tools=[] anyway)
- `skip_permission: bool` — set to `False` (Amplifier handles permissions)
- `handler: None` — **MUST exist** (SDK checks handler attribute); set to `None` so SDK skips handler registration (Amplifier handles tools at kernel layer)

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

### Test Anchors

| Anchor | Clause | Test Location |
|--------|--------|---------------|
| `sdk-boundary:MinimalMode:MUST:1` | infinite_sessions disabled | `tests/test_sdk_boundary_contract.py::TestMinimalModeConfig::test_infinite_sessions_disabled` |
| `sdk-boundary:MinimalMode:MUST:2` | config discovery disabled | `tests/test_sdk_boundary_contract.py::TestMinimalModeConfig::test_config_discovery_disabled` |
| `sdk-boundary:MinimalMode:MUST:3` | mcp_servers empty | `tests/test_sdk_boundary_contract.py::TestMinimalModeConfig::test_mcp_servers_empty` |
| `sdk-boundary:MinimalMode:MUST:4` | skill_directories empty | `tests/test_sdk_boundary_contract.py::TestMinimalModeConfig::test_skill_directories_empty` |
| `sdk-boundary:MinimalMode:MUST:5` | custom_agents empty | `tests/test_sdk_boundary_contract.py::TestMinimalModeConfig::test_custom_agents_empty` |
| `sdk-boundary:MinimalMode:MUST:6` | commands empty | `tests/test_sdk_boundary_contract.py::TestMinimalModeConfig::test_commands_empty` |

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
| `sdk-boundary:ImportQuarantine:MUST:6` | Multi-level fallback chains for SDK types that moved between versions |

### Types

| Anchor | Clause |
|--------|--------|
| `sdk-boundary:Types:MUST:1` | No SDK types cross boundary |
| `sdk-boundary:Types:MUST:2` | Domain types are dataclasses/primitives |
| `sdk-boundary:Types:MUST:3` | SessionHandle is opaque string |

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
| `sdk-boundary:TypeTranslation:MUST:4` | SessionHandle delegates on() and close() to raw_session without leaking SDK types |
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

### SDK API Assumptions (SDK v0.2.0)

| Anchor | Clause |
|--------|--------|
| `sdk-boundary:Session:MUST:1` | SDK CopilotClient.create_session() accepts kwargs (model=, streaming=, on_permission_request=, hooks=) |
| `sdk-boundary:Lifecycle:MUST:1` | SDK CopilotClient has async start() and stop() lifecycle methods |
| `sdk-boundary:Auth:MUST:1` | SDK CopilotClient accepts SubprocessConfig(github_token=...) |
| `sdk-boundary:Auth:MUST:2` | Token resolution follows SDK priority order; empty string treated as absent |
| `sdk-boundary:Auth:MUST:3` | Fail closed (ConfigurationError) when explicit token resolved but SubprocessConfig unavailable |
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
(copilot.types)           (internal isolation)      (kernel expects this)
```

**Why Three Types?**
- **SDK ModelInfo**: SDK's type structure (may change with SDK versions)
- **CopilotModelInfo**: Our isolation layer — insulates us from SDK changes
- **amplifier_core.ModelInfo**: What the kernel expects from `provider.list_models()`

### Type Translation

```python
# SDK ModelInfo (from copilot.types) — INPUT
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

---

## Authentication

### Token Resolution

The provider MUST resolve auth tokens from environment variables in the official SDK priority order
(documented in SDK `docs/auth/index.md`).

### MUST Constraints

1. **MUST** scan environment variables in this exact order: `COPILOT_AGENT_TOKEN`, `COPILOT_GITHUB_TOKEN`, `GH_TOKEN`, `GITHUB_TOKEN`
2. **MUST** treat an empty string token as absent — resolution MUST continue to the next candidate
3. **MUST** return `None` when no non-empty token is found in any variable
4. **MUST** raise `ConfigurationError` (fail closed) when a token is resolved but `SubprocessConfig` is unavailable
5. **MUST NOT** silently ignore an explicit token under any circumstance, including when `SKIP_SDK_CHECK` is set
6. **MUST NOT** fall through to default/ambient SDK authentication when an explicit token is present but cannot be applied

### Rationale

- **Priority order**: Agent-mode tokens (`COPILOT_AGENT_TOKEN`) take highest precedence; GitHub Actions tokens (`GITHUB_TOKEN`) are lowest. This matches the SDK's documented auth hierarchy.
- **Empty-string fallthrough**: Prevents treating declared-but-empty env vars as valid tokens.
- **Fail closed**: An explicit token that cannot be applied indicates an SDK version mismatch. Continuing with ambient auth would silently escalate privileges via an unexpected authentication context (OWASP A07: Identification and Authentication Failures).

### Test Anchors

| Anchor | Clause |
|--------|--------|
| `sdk-boundary:Auth:MUST:1` | SDK CopilotClient accepts SubprocessConfig(github_token=...) |
| `sdk-boundary:Auth:MUST:2` | Token resolution follows SDK priority order; empty string treated as absent |
| `sdk-boundary:Auth:MUST:3` | Fail closed (ConfigurationError) when explicit token resolved but SubprocessConfig unavailable |

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

---

## Verification

To verify this contract:

```bash
# Should find SDK imports ONLY in _imports.py
grep -r "from copilot" amplifier_module_provider_github_copilot/
grep -r "import copilot" amplifier_module_provider_github_copilot/
```

Expected: Only `sdk_adapter/_imports.py` matches.
