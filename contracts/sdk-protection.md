# Contract: SDK Protection

## Version
- **Current:** 1.1
- **Module Reference:** amplifier_module_provider_github_copilot/sdk_adapter/tool_capture.py, __init__.py, client.py
- **Config Reference:** amplifier_module_provider_github_copilot/config/sdk_protection.yaml
- **Status:** Defensive Enhancement
- **Created:** 2026-03-21 — Defense-in-depth layer for SDK interaction
- **Updated:** 2026-03-31 — Added Subprocess Management Invariants (MUST-5,6,7)

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

**Implementation:** Call in `provider.py` `complete()` after tool capture detected, inside the `create_session()` context manager

### MUST-4: Abort Timeout

The provider MUST bound the abort call by `session.abort_timeout_seconds`. If abort hangs, log warning and proceed with cleanup.

**Rationale:** A hung abort call would block the session from returning to Amplifier. The timeout ensures forward progress.

**Implementation:** `asyncio.wait_for(session.abort(), timeout=abort_timeout)`

### SHOULD-2: Idle Timeout (Safety Bound Only)

The provider SHOULD NOT use `session.idle_timeout_seconds` for the main idle wait. SDK API calls can take 60+ seconds for complex operations (e.g., agent delegation). Use the caller's request timeout instead.

The `idle_timeout_seconds` config is retained for abort operations only.

**Implementation:** `asyncio.wait_for(idle_event.wait(), timeout=timeout)` — uses caller's timeout

---

## Subprocess Management Invariants

### MUST-5: Prewarm Task Tracking

When `sdk.prewarm_subprocess` is true, the provider MUST track the prewarm asyncio.Task and cancel it during cleanup. Untracked fire-and-forget tasks can orphan SDK subprocesses during rapid mount/unmount cycles.

**Rationale:** Without task tracking, repeated mount/unmount (e.g., during testing or hot-reload) leaks Copilot subprocesses. Each subprocess consumes ~100MB memory and holds authentication state.

**Implementation:** Store task reference in module-level variable, cancel in cleanup function.

### MUST-6: Guard Re-initialization After Stop

The provider MUST check `_stopped` flag in `_ensure_client_initialized()` and raise `RuntimeError` if the client has been stopped. This prevents prewarm or lazy-init from resurrecting a stopped client.

**Rationale:** After `stop()` is called, the shared client is in a terminal state. Allowing re-init would violate the singleton lifecycle and could cause resource leaks.

**Implementation:** `if self._stopped: raise RuntimeError("Copilot client has been stopped")`

### MUST-7: Validate SDK Config Values

The provider MUST validate `sdk.log_level` against the allowlist defined in config. This applies to both YAML values AND runtime environment overrides.

- YAML validation: Invalid values MUST raise `ConfigurationError` at load time
- ENV validation: Invalid env values MUST fall back to YAML default with warning

**Rationale:** Three-Medium Architecture requires fail-fast on invalid policy values. Silent acceptance of invalid log levels could enable verbose logging of sensitive conversation data.

**Valid log levels:** `none`, `error`, `warning`, `info`, `debug`, `all`

**Implementation:** 
- YAML: Validation in `load_sdk_protection_config()`
- ENV: Validation in `_resolve_sdk_log_level()`

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
| `sdk-protection:Subprocess:MUST:5` | Prewarm task tracking | `tests/test_client_lifecycle.py` |
| `sdk-protection:Subprocess:MUST:6` | Guard re-init after stop | `tests/test_client_lifecycle.py` |
| `sdk-protection:Subprocess:MUST:7` | Validate SDK config | `tests/test_sdk_protection.py` |

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
| `sdk.log_level` | str | "info" | SDK subprocess log level |
| `sdk.log_level_env_var` | str | "COPILOT_SDK_LOG_LEVEL" | Env var override |
| `sdk.prewarm_subprocess` | bool | false | Spawn subprocess at mount() |
| `sdk.valid_log_levels` | list | see below | Allowlist for validation |

**Valid log levels:** `["none", "error", "warning", "info", "debug", "all"]`

---

## Related Contracts

- `deny-destroy.md` — Sovereignty: SDK never executes tools
- `streaming-contract.md` — Event streaming and abort-on-capture pattern
- `behaviors.md` — General retry and resilience policy
