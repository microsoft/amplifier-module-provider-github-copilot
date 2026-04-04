# Contract: Observability

## Version
- **Current:** 1.2 (Dead Code Removed)
- **Module Reference:** provider, completion
- **Config:** amplifier_module_provider_github_copilot/config/observability.yaml
- **Status:** IMPLEMENTED (event emission)
- **Note:** Event emission implemented. OTEL spans deferred to future version.

---

## Overview

This contract defines observability policy for provider instrumentation. Observability is BOTH a provider and kernel concern with clear boundaries:
- **Kernel** owns generic `gen_ai.chat` spans (request/response boundary)
- **Provider** owns internal spans (SDK session, retry, vendor-specific)

---

## Event Emission Policy

### MUST Constraints

1. **MUST** guard hook calls: `if self.coordinator and hasattr(self.coordinator, 'hooks')`
2. **MUST** emit `llm:request` event BEFORE SDK session.send()
3. **MUST** emit `llm:response` event AFTER accumulator completes
4. **MUST** work without coordinator (standalone mode)
5. **MUST NOT** assume coordinator.hooks.emit() exists

### SHOULD Constraints

1. **SHOULD** include correlation IDs for tracing
2. **SHOULD** include model ID in all events
3. **SHOULD** include `sdk_pid` in `llm:response` for SDK log file correlation (`~/.copilot/logs/process-*-{pid}.log`)

---

## OTEL Policy

### Status: DEFERRED

OpenTelemetry span support is deferred to a future version. The `otel:` config section was removed from observability.yaml as dead code.

When implementing OTEL support:

1. **MUST** be opt-in via config (NOT auto-detect from package presence)
2. **MUST** gracefully degrade when opentelemetry not installed
3. **MUST** use `gen_ai.*` semantic conventions for spans
4. **MUST** redact sensitive data in span events
5. **MUST NOT** create `gen_ai.chat` boundary spans (kernel responsibility)

### Provider-Internal Spans (Future)

Provider MAY create internal spans for:
- SDK session lifecycle
- Retry attempts
- Event translation timing
- Vendor-specific metrics

---

## Verbosity Policy

Following the verbosity collapse principle:

1. **MUST** use single `raw_payloads: false` flag (not tiered debug modes)
2. **MUST** apply security redaction before including raw payloads
3. **MUST NOT** create separate `:debug` or `:raw` event suffixes

---

## Config Schema

```yaml
# config/observability.yaml
version: "1.0"

events:
  raw_payloads: false
```

**Note:** `events.enabled` was removed as dead config. Event emission is always on when observability is loaded. To disable events, don't subscribe to hooks.

**Note:** `otel:` and `logging:` sections were removed as dead code — the Python implementation never parsed these values.

---

## Test Anchors

| Anchor | Clause |
|--------|--------|
| `observability:Events:MUST:1` | Guard hook calls |
| `observability:Events:MUST:2` | Emit llm:request before send |
| `observability:Events:MUST:3` | Emit llm:response after complete |
| `observability:Events:SHOULD:3` | Include sdk_pid in llm:response |
| `observability:Verbosity:MUST:1` | Single raw_payloads flag |
| `observability:Payload:SHOULD:1` | Type-safe content counting |
| `observability:Payload:SHOULD:2` | Type-safe tool name extraction |
| `observability:Redaction:SHOULD:1` | Redaction audit trail |

**Deferred anchors (for future OTEL implementation):**
- `observability:OTEL:MUST:1` — Opt-in via config
- `observability:OTEL:MUST:2` — Graceful degradation

---

## Payload Building Policy

### SHOULD Constraints

1. **SHOULD** use isinstance guards when counting content blocks to handle edge cases where `response.content` may be a string instead of a sequence. This prevents returning character count instead of block count.

```python
# CORRECT: Defensive content counting
from collections.abc import Sequence

if content is not None and isinstance(content, Sequence) and not isinstance(content, str):
    content_block_count = len(content)
```

2. **SHOULD** use hasattr/isinstance guards when extracting tool names. Amplifier kernel passes `ToolSpec` objects (Pydantic models with `.name` attribute), not dicts. Never call `.get()` without first checking `isinstance(tool, dict)`.

```python
# CORRECT: Defensive tool name extraction
for tool in tools:
    if hasattr(tool, "name") and not isinstance(tool, dict):
        name = getattr(tool, "name", None)  # ToolSpec object
    elif isinstance(tool, dict):
        name = tool.get("name")  # Dict fallback
```

---

## Redaction Audit Trail (E2)

### Status: CONTRACT DEFINED, IMPLEMENTATION DEFERRED

### SHOULD Constraints

1. **SHOULD** provide audit trail when redaction occurs, enabling security auditing without exposing secrets.

### Proposed API (Not Yet Implemented)

The `security_redaction.py` module SHOULD support optional audit trail:

```python
@dataclass
class RedactionResult:
    """Result of redaction with optional audit trail."""
    text: str
    redaction_count: int = 0
    secret_types_found: frozenset[str] = field(default_factory=frozenset)
    
    @property
    def redacted(self) -> bool:
        """True if any redaction occurred."""
        return self.redaction_count > 0
```

### Usage Example

```python
# Current (no audit trail)
safe_text = redact_sensitive_text(raw_text)

# Future (with audit trail)
result = redact_sensitive_text_with_audit(raw_text)
if result.redacted:
    logger.debug(
        "Redacted %d secrets, types: %s",
        result.redaction_count,
        result.secret_types_found,
    )
safe_text = result.text
```

### Deferred Rationale

This change requires:
1. API design coordination with amplifier-core (event schema)
2. New test infrastructure for audit trail verification
3. Careful balance between audit detail and information leakage

The contract is defined; implementation is deferred to a dedicated session.
