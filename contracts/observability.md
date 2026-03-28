# Contract: Observability

## Version
- **Current:** 1.1 (Status-Corrected)
- **Module Reference:** provider, completion
- **Config:** amplifier_module_provider_github_copilot/config/observability.yaml
- **Status:** PARTIAL IMPLEMENTATION
- **Note:** Config exists; OTEL spans not implemented; event emission partial

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

---

## OTEL Policy

### MUST Constraints

1. **MUST** be opt-in via config (NOT auto-detect from package presence)
2. **MUST** gracefully degrade when opentelemetry not installed
3. **MUST** use `gen_ai.*` semantic conventions for spans
4. **MUST** redact sensitive data in span events
5. **MUST NOT** create `gen_ai.chat` boundary spans (kernel responsibility)

### Provider-Internal Spans

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
  enabled: true
  raw_payloads: false

otel:
  enabled: false
  span_events_redacted: true

logging:
  level: INFO
```

---

## Test Anchors

| Anchor | Clause |
|--------|--------|
| `observability:Events:MUST:1` | Guard hook calls |
| `observability:Events:MUST:2` | Emit llm:request before send |
| `observability:Events:MUST:3` | Emit llm:response after complete |
| `observability:OTEL:MUST:1` | Opt-in via config |
| `observability:OTEL:MUST:2` | Graceful degradation |
| `observability:Verbosity:MUST:1` | Single raw_payloads flag |
