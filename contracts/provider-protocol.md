# Contract: Provider Protocol

## Version
- **Current:** 1.1 (v2.1 Kernel-Validated)
- **Module Reference:** amplifier_module_provider_github_copilot/provider.py
- **Amplifier Contract:** amplifier-core PROVIDER_CONTRACT.md
- **Status:** Specification
- **Updated:** 2026-03-31 — Clarified mount() failure semantics

---

## Overview

This contract defines the **4 methods + 1 property** Provider Protocol that our provider MUST implement to integrate with Amplifier's orchestrator. The provider is a thin orchestrator that delegates to specialized modules.

---

## Module Entry Point

### mount()

```python
from amplifier_core import ModuleCoordinator

async def mount(
    coordinator: ModuleCoordinator,
    config: dict[str, Any] | None = None,
) -> CleanupFn | None: ...
```

**Behavioral Requirements:**
- **MUST** accept `ModuleCoordinator` as first argument (type-safe)
- **MUST** return cleanup callable on success
- **MUST** raise exception on failure (framework must distinguish failure from opt-out)
- **MUST** register provider with coordinator via `coordinator.mount()`
- **MUST** use process-level singleton for SDK client (memory efficiency)

**Failure Semantics:**
- Returning `None` indicates "provider chose not to load" (opt-out)
- Raising an exception indicates "provider failed to load" (error)
- **RATIONALE:** Framework needs to distinguish between a provider that doesn't apply vs one that's broken

**Type Conformance:**
- **MUST** use `ModuleCoordinator` instead of `Any` for type safety
- **RATIONALE:** Ecosystem providers (anthropic, openai, azure-openai) use typed coordinator

**Test Anchors:**
| Anchor | Clause |
|--------|--------|
| `provider-protocol:mount:MUST:1` | Accepts ModuleCoordinator type |
| `provider-protocol:mount:MUST:2` | Returns cleanup callable |
| `provider-protocol:mount:MUST:3` | Registers provider on coordinator |
| `provider-protocol:mount:MUST:5` | Uses process-level singleton for SDK client |

---

## The Protocol (4 Methods + 1 Property)

### 1. name (property)

```python
@property
def name(self) -> str: ...
```

**Behavioral Requirements:**
- **MUST** return `"github-copilot"` (exact string)
- **MUST** be a property, not a method call
- **MUST NOT** vary based on configuration

**Test Anchors:**
| Anchor | Clause |
|--------|--------|
| `provider-protocol:name:MUST:1` | Returns "github-copilot" |
| `provider-protocol:name:MUST:2` | Is a property |

---

### 2. get_info()

```python
def get_info(self) -> ProviderInfo: ...
```

**Behavioral Requirements:**
- **MUST** return `ProviderInfo` with accurate metadata
- **MUST** include `defaults.context_window` for budget calculation
- **MUST** include `config_fields` for init wizard integration
- **SHOULD** cache model info to avoid repeated API calls
- **MAY** include additional provider-specific metadata

**ConfigField Requirements:**
- **MUST** include ConfigField for GitHub token (`env_var="GITHUB_TOKEN"`)
- **MUST** use `field_type="secret"` for token fields
- **RATIONALE:** Init wizard uses config_fields to prompt user for credentials

**Test Anchors:**
| Anchor | Clause |
|--------|--------|
| `provider-protocol:get_info:MUST:1` | Returns valid ProviderInfo |
| `provider-protocol:get_info:MUST:2` | Includes context_window |
| `provider-protocol:get_info:MUST:3` | Includes config_fields with token field |

---

### 3. list_models()

```python
async def list_models(self) -> list[ModelInfo]: ...
```

**Behavioral Requirements:**
- **MUST** return all available models from SDK
- **MUST** include `context_window` and `max_output_tokens` per model
- **SHOULD** cache results for session lifetime
- **MUST** translate SDK model info to `ModelInfo` domain type

**Test Anchors:**
| Anchor | Clause |
|--------|--------|
| `provider-protocol:list_models:MUST:1` | Returns model list |
| `provider-protocol:list_models:MUST:2` | Includes context_window |

---

### 4. complete()

```python
async def complete(
    self,
    request: ChatRequest,
    **kwargs,
) -> ChatResponse: ...
```

**Note:** The kernel passes `**kwargs` for extensibility. Internal streaming callbacks are provider-internal, not part of the protocol.

**Behavioral Requirements:**
- **MUST** create ephemeral session per call (per deny-destroy.md)
- **MUST** forward `ChatRequest.tools` to SDK session (per sdk-boundary.md Tool Forwarding Contract)
- **MUST** extract and forward images from `ChatRequest.messages` (per sdk-boundary.md Image Passthrough)
- **MUST** capture tool calls (NOT execute them)
- **MUST** destroy session after first turn completes
- **MUST NOT** maintain state between calls
- **MUST** translate SDK errors to kernel errors (per error-hierarchy.md)

**Session Lifecycle:**
```
complete() called
    │
    ├─→ Create ephemeral session (with deny hook + tool definitions)
    │
    ├─→ Extract images from last user message (as BlobAttachments)
    │
    ├─→ Send prompt with attachments, capture response
    │
    ├─→ Capture tool calls (not execute)
    │
    └─→ Destroy session, return response
```

**Test Anchors:**
| Anchor | Clause |
|--------|--------|
| `provider-protocol:complete:MUST:1` | Creates ephemeral session |
| `provider-protocol:complete:MUST:2` | Forwards tools to SDK session |
| `provider-protocol:complete:MUST:3` | Captures tool calls |
| `provider-protocol:complete:MUST:4` | Destroys session after turn |
| `provider-protocol:complete:MUST:5` | No state between calls |
| `provider-protocol:complete:MUST:6` | Detects fake tool calls and retries with correction |
| `provider-protocol:complete:MUST:7` | Extracts images from last user message |
| `provider-protocol:complete:MUST:8` | Forwards images as BlobAttachments to SDK |

---

### 5. parse_tool_calls()

```python
def parse_tool_calls(self, response: ChatResponse) -> list[ToolCall]: ...
```

**Note:** Returns `list[ToolCall]`, NOT `list[ToolCallBlock]`. `ToolCall` has `arguments`, not `input`.

**Behavioral Requirements:**
- **MUST** extract tool calls from response
- **MUST** return empty list if no tool calls
- **MUST NOT** execute tools (orchestrator responsibility)
- **MUST** preserve tool call IDs for result correlation
- **MUST** be synchronous — callers MUST NOT await the return value

**Synchronous Contract:**

`parse_tool_calls` is defined as a plain function (not a coroutine) at every
layer of the Amplifier stack. This is not a convention — it is a hard requirement
imposed by the kernel bridges that call this method without `await`:

- **Python Protocol** (`amplifier_core/interfaces.py`, class `Provider(Protocol)`):
  `def parse_tool_calls(self, response: ChatResponse) -> list[ToolCall]`
- **Rust trait** (`amplifier-core/crates/amplifier-core/src/traits.rs:188`):
  `fn parse_tool_calls(&self, response: &ChatResponse) -> Vec<ToolCall>`
- **WASM bridge** (`amplifier-core/crates/amplifier-core/src/bridges/wasm_provider.rs:255`):
  inline comment — *"Call WASM synchronously. parse_tool_calls is not async in the trait"*
- **gRPC bridge** (`amplifier-core/crates/amplifier-core/src/bridges/grpc_provider.rs:174`):
  `fn parse_tool_calls(&self, response: &ChatResponse) -> Vec<ToolCall>`

**Consequence of violation:** An `async def` implementation returns a coroutine
object. Python coroutines are truthy — the kernel's tool-dispatch loop would
interpret every response as having tool calls regardless of actual content. This
failure is silent: no exception is raised, no type error, no warning. The kernel
enters an infinite dispatch loop on ordinary text responses.

**ToolCall Structure:**
```python
@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]  # NOT "input"
```

**Test Anchors:**
| Anchor | Clause |
|--------|--------|
| `provider-protocol:parse_tool_calls:MUST:1` | Extracts tool calls |
| `provider-protocol:parse_tool_calls:MUST:2` | Returns empty list when none |
| `provider-protocol:parse_tool_calls:MUST:3` | Preserves tool call IDs |
| `provider-protocol:parse_tool_calls:MUST:4` | Uses arguments, not input |
| `provider-protocol:parse_tool_calls:MUST:5` | Is synchronous — callers must not await |

---

## Observability Hooks

### Hook Emission Requirements

Providers **MUST** emit observability events to `coordinator.hooks.emit()` for integration with Amplifier's monitoring infrastructure.

**Evidence:** All canonical providers (anthropic, openai, azure-openai) emit these hooks.

### Required Events

#### llm:request

**MUST** emit before SDK API call with request metadata.

```python
await self._emit_event("llm:request", {
    "provider": self.name,
    "model": model,
    "message_count": len(messages),
    "tool_count": len(tools) if tools else 0,
    "streaming": use_streaming,
    "timeout": timeout,
    # Optional: "raw": redact_secrets(raw_payload) for debug
})
```

**Test Anchors:**
| Anchor | Clause |
|--------|--------|
| `provider-protocol:hooks:llm_request:MUST:1` | Emits before SDK call |
| `provider-protocol:hooks:llm_request:MUST:2` | Includes provider, model, message_count |

#### llm:response

**MUST** emit after SDK response with status and timing.

```python
# Success
await self._emit_event("llm:response", {
    "provider": self.name,
    "model": model,
    "status": "ok",
    "duration_ms": elapsed_ms,
    "usage": {
        "input": response.usage.input_tokens,
        "output": response.usage.output_tokens,
    },
    # Per amplifier-core proto: "stop", "tool_calls", "length", "content_filter"
    # Not "end_turn" which is an SDK-specific input value
    "finish_reason": response.finish_reason or "stop",
    "tool_calls": len(response.tool_calls) if response.tool_calls else 0,
})

# Error
await self._emit_event("llm:response", {
    "provider": self.name,
    "model": model,
    "status": "error",
    "duration_ms": elapsed_ms,
    "error_type": type(error).__name__,
    "error_message": str(error),
})
```

**Test Anchors:**
| Anchor | Clause |
|--------|--------|
| `provider-protocol:hooks:llm_response:MUST:1` | Emits after SDK response |
| `provider-protocol:hooks:llm_response:MUST:2` | Includes duration_ms timing |
| `provider-protocol:hooks:llm_response:MUST:3` | Uses status "ok" or "error" |

#### PROVIDER_RETRY (provider:retry)

**MUST** emit before retry sleep when retrying failed requests.

```python
from amplifier_core.events import PROVIDER_RETRY

await self._emit_event(PROVIDER_RETRY, {
    "provider": self.name,
    "model": model,
    "attempt": attempt,
    "max_retries": max_retries,
    "delay": delay_seconds,
    "error_type": type(error).__name__,
    "error_message": str(error),
})
```

**Test Anchors:**
| Anchor | Clause |
|--------|--------|
| `provider-protocol:hooks:provider_retry:MUST:1` | Emits before retry sleep |
| `provider-protocol:hooks:provider_retry:MUST:2` | Includes attempt, max_retries, delay |

### Event Ordering Contract

- **MUST** emit `llm:request` BEFORE `llm:response`
- **MUST** emit `PROVIDER_RETRY` between `llm:request` and next retry attempt

### Graceful Degradation

- **MUST** handle missing coordinator gracefully (no raise)
- **MUST** catch and log hook emission errors (no raise)

The real hook emission mechanism is the `llm_lifecycle` async context manager in `observability.py`,
not a `_emit_event()` method. The context manager handles request, response, and retry hooks
as a unit, ensuring the response hook always fires even on error paths.

```python
# Real pattern — observability.py llm_lifecycle context manager
from .observability import llm_lifecycle

async with llm_lifecycle(coordinator, model) as ctx:
    await ctx.emit_request(request_payload)
    response = await sdk_call(...)
    await ctx.emit_response(response)
# llm_lifecycle emits llm:response on exit even if exception raised
```

**Test Anchors:**
| Anchor | Clause |
|--------|--------|
| `provider-protocol:hooks:emit:MUST:1` | No raise on missing coordinator |
| `provider-protocol:hooks:emit:MUST:2` | No raise on hook errors |

---

## Cross-References

- **deny-destroy.md** — Session ephemerality and deny hook requirements
- **error-hierarchy.md** — Exception translation requirements (kernel types)
- **amplifier-core PROVIDER_CONTRACT.md** — Kernel interface specification

---

## Quality Gates

**MUST** constraints for release readiness:

| Gate | Command | Requirement |
|------|---------|-------------|
| Main package lint | `ruff check amplifier_module_provider_github_copilot/` | 0 errors |
| Main package types | `pyright amplifier_module_provider_github_copilot/` | 0 errors |
| **Test file types** | `pyright tests/` | **0 errors** |
| Full repo types | `pyright .` | 0 errors |
| All tests pass | `pytest tests/ -v` | 0 failures |

**Test Anchors:**
| Anchor | Clause |
|--------|--------|
| `provider-protocol:QualityGates:MUST:1` | Test files must be type-clean (zero pyright errors) |
| `provider-protocol:QualityGates:MUST:2` | Full-repo `pyright .` must pass before release |

**Rationale:** Running pyright only on the main package allowed test file errors to accumulate undetected. Test files are part of the deliverable and must be type-clean for Microsoft OSS release.

---

## Public API Surface

### __all__ Export List

The module's `__all__` defines the stable public API.

**Behavioral Requirements:**
- **MUST** export only `mount` and the provider class
- **MUST NOT** re-export kernel types (`ModelInfo`, `ProviderInfo`)
- **MUST** match ecosystem convention (anthropic, openai, azure-openai)

**Rationale:** Kernel types belong to `amplifier_core`. Re-exporting them couples the provider's API to kernel internals and creates version skew risks.

**Canonical Pattern:**
```python
__all__ = ["mount", "GitHubCopilotProvider"]
```

**Test Anchors:**
| Anchor | Clause |
|--------|--------|
| `provider-protocol:public_api:MUST:1` | Exports only mount and provider class |
| `provider-protocol:public_api:MUST:2` | Does not re-export kernel types |

---

## Implementation Checklist

- [ ] `mount()` accepts `ModuleCoordinator` type (not `Any`)
- [ ] `mount()` returns cleanup callable
- [ ] `name` property returns "github-copilot"
- [ ] `get_info()` returns valid ProviderInfo
- [ ] `get_info()` includes config_fields with GitHub token field
- [ ] `list_models()` queries SDK and caches
- [ ] `complete()` accepts `**kwargs` (not named callback)
- [ ] `complete()` creates ephemeral session with deny hook
- [ ] `complete()` forwards tools to SDK session (per sdk-boundary.md ToolForwarding:MUST:1)
- [ ] `complete()` captures and returns tool calls
- [ ] `parse_tool_calls()` returns `list[ToolCall]`
- [ ] `parse_tool_calls()` uses `arguments` field
- [ ] All SDK errors translated to kernel types
- [ ] **Test files pass `pyright tests/` with 0 errors**
- [ ] `complete()` emits `llm:request` before SDK call
- [ ] `complete()` emits `llm:response` after completion (success or error)
- [ ] Retry loop emits `PROVIDER_RETRY` before sleep
- [ ] `llm_lifecycle` context manager handles missing coordinator gracefully
