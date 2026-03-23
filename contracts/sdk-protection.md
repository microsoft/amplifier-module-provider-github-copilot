# Contract: SDK Protection

## Version
- **Current:** 1.0
- **Module Reference:** amplifier_module_provider_github_copilot/sdk_adapter/tool_capture.py
- **Config Reference:** amplifier_module_provider_github_copilot/config/sdk_protection.yaml
- **Status:** Defensive Enhancement
- **Created:** 2026-03-21 — Defense-in-depth layer for SDK interaction

---

## Overview

SDK Protection defines invariants for safe interaction with the GitHub Copilot SDK. While the core Deny+Destroy pattern prevents the SDK from executing tools, this contract adds defensive measures for tool capture, deduplication, and session cleanup.

This contract complements `deny-destroy.md` — Deny+Destroy handles sovereignty (no SDK tool execution), SDK Protection handles robustness (clean capture and shutdown).

---

## Tool Capture Invariants

### MUST-1: First Turn Only

The provider MUST capture tools from the first `ASSISTANT_MESSAGE` event only. Subsequent tool events within the same session MUST be ignored.

**Rationale:** Without first-turn-only, the SDK's internal agent loop can retry after each tool denial, causing runaway turn accumulation. First-turn-only capture breaks the cycle.

**Implementation:** `ToolCaptureHandler._capture_complete` flag

### MUST-2: Deduplicate by ID

The provider MUST deduplicate tool requests by `tool_call_id`. If the SDK emits the same tool_call_id twice, only the first occurrence MUST be captured.

**Rationale:** The SDK may emit duplicate events during network reconnection, retry loops, or event replay. Duplicate tool calls would cause Amplifier to execute the same operation twice.

**Implementation:** `ToolCaptureHandler._seen_ids` set

### SHOULD-1: Log Capture Events

The provider SHOULD log tool capture events when `tool_capture.log_capture_events` is true in config. This aids debugging and forensic analysis without impacting production performance.

**Implementation:** `ToolCaptureHandler` logs at INFO level

---

## Session Management Invariants

### MUST-3: Explicit Abort

When `session.explicit_abort` is true in config, the provider MUST call `session.abort()` before context manager cleanup when tools have been captured.

**Rationale:** Explicit abort signals to the SDK that we're done, rather than relying on timeout-based cleanup. This provides cleaner shutdown semantics.

**Implementation:** Call in `_stream_with_session()` after tool capture detected

### MUST-4: Abort Timeout

The provider MUST bound the abort call by `session.abort_timeout_seconds`. If abort hangs, log warning and proceed with cleanup.

**Rationale:** A hung abort call would block the session from returning to Amplifier. The timeout ensures forward progress.

**Implementation:** `asyncio.wait_for(session.abort(), timeout=abort_timeout)`

### SHOULD-2: Idle Timeout

The provider SHOULD use `session.idle_timeout_seconds` as a safety bound when waiting for the idle event. This prevents indefinite waits if the SDK malfunctions.

**Implementation:** `asyncio.wait_for(idle_event.wait(), timeout=idle_timeout)`

---

## Architectural Notes

### Circuit Breaker NOT Required

Some providers implement a turn-count circuit breaker to prevent runaway loops. This provider's architecture makes that unnecessary:

| Architecture | Pattern | Loop Risk |
|--------------|---------|----------|
| Multi-turn loop | `while not done: send_message(); process_events()` | High |
| This provider | `send() once; queue.drain(); done` | None |

Runaway loops are structurally impossible in the single-turn-then-destroy pattern. We handle `CircuitBreakerError` from the SDK (errors.yaml) in case the SDK internally trips a circuit breaker.

### Performance Consideration

Deduplication uses O(n) set membership check where n = number of captured tools. Since tool capture is first-turn-only and typical tool calls are < 10, this is negligible overhead.

---

## Test Anchors

| Anchor | Clause | Test Location |
|--------|--------|---------------|
| `sdk-protection:ToolCapture:MUST:1` | First turn only | `tests/test_tool_capture.py` |
| `sdk-protection:ToolCapture:MUST:2` | Deduplicate by ID | `tests/test_tool_capture.py` |
| `sdk-protection:ToolCapture:SHOULD:1` | Log capture events | (observational) |
| `sdk-protection:Session:MUST:3` | Explicit abort | `tests/test_sdk_protection.py` |
| `sdk-protection:Session:MUST:4` | Abort timeout | `tests/test_sdk_protection.py` |
| `sdk-protection:Session:SHOULD:2` | Idle timeout | `tests/test_sdk_protection.py` |

---

## Configuration

Policy values are defined in `config/sdk_protection.yaml`. The Python code loads these at runtime and uses them to configure `ToolCaptureHandler` and session cleanup.

| Key | Type | Default | Purpose |
|-----|------|---------|---------|
| `tool_capture.first_turn_only` | bool | true | First-turn capture |
| `tool_capture.deduplicate` | bool | true | Deduplicate by tool_call_id |
| `tool_capture.log_capture_events` | bool | true | Log captures at INFO |
| `session.explicit_abort` | bool | true | Call session.abort() |
| `session.abort_timeout_seconds` | float | 5.0 | Abort call timeout |
| `session.idle_timeout_seconds` | float | 30.0 | Idle wait safety bound |

---

## Related Contracts

- `deny-destroy.md` — Sovereignty: SDK never executes tools
- `streaming-contract.md` — Event streaming and abort-on-capture pattern
- `behaviors.md` — General retry and resilience policy
