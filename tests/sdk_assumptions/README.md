# SDK Behavioral Assumptions

> **Last Validated SDK Version**: `github-copilot-sdk >= 0.1.0`
>
> **Last Validation Date**: 2026-02-07

This directory contains tests that validate behavioral assumptions about
the GitHub Copilot CLI SDK. These assumptions are NOT part of the SDK's
public API contract but are critical for our "Deny + Destroy" external
orchestration pattern.

## Why These Tests Exist

Our provider uses the SDK in a non-standard way: we capture tool calls
and return them to an external orchestrator (Amplifier) instead of letting
the CLI execute tools internally. This requires relying on SDK behaviors
that aren't explicitly documented or guaranteed.

**When SDK behavior changes, these tests fail BEFORE production breaks.**

## Assumptions We Depend On

### 1. Event Ordering (Critical)

**File**: [test_event_ordering.py](test_event_ordering.py)

| Assumption | Why Critical |
| ---------- | ------------ |
| `ASSISTANT_MESSAGE` events with `tool_requests` fire BEFORE `preToolUse` hook | We capture tool_requests from the message event. If hooks fire first, we have no data to capture. |
| Events fire in deterministic order within a single turn | Streaming deltas must precede final message for proper content assembly. |
| `SESSION_IDLE` fires after all turn processing completes | We wait for idle to know the response is complete. |

**Breaking Change Impact**: Tool calls silently captured as `None`, provider
returns empty responses, external orchestration fails completely.

### 2. preToolUse Hook Deny Behavior (Critical)

**File**: [test_deny_hook.py](test_deny_hook.py)

| Assumption | Why Critical |
| ---------- | ------------ |
| Returning `{"permissionDecision": "deny"}` prevents tool handler invocation | Our no-op handlers must never execute; deny must block execution. |
| Deny hook fires for BOTH user-defined AND built-in tools | We need to block built-in tools like `edit`, `create` from executing. |
| Deny reason is logged/observable (not swallowed) | Debugging requires seeing why tools were denied. |

**Breaking Change Impact**: Tools execute despite deny (side effects occur),
or only user tools are blocked while built-ins execute.

### 3. Session Lifecycle (Critical)

**File**: [test_session_lifecycle.py](test_session_lifecycle.py)

| Assumption | Why Critical |
| ---------- | ------------ |
| `session.destroy()` terminates the CLI's internal agent loop | Without reliable destroy, CLI retries with built-in tools. |
| No further events fire after destroy is called | Event handlers must not receive stale events after cleanup. |
| Destroy is idempotent (safe to call multiple times) | Cleanup code may call destroy defensively. |

**Breaking Change Impact**: CLI bypass behavior returns (built-in tools
execute internally), resource leaks, race conditions.

### 4. Tool Registration (Important)

**File**: [test_tool_registration.py](test_tool_registration.py)

| Assumption | Why Critical |
| ---------- | ------------ |
| Tools passed to `create_session()` become visible to the LLM | LLM must see our tool definitions to emit tool calls. |
| Duplicate tool names cause predictable error (400) | We need to deduplicate before registration. |
| `excluded_tools` parameter excludes specified tools | Future use for excluding built-in tools if supported. |

**Breaking Change Impact**: LLM ignores our tools, unexpected errors,
tool conflicts.

## Upgrade Workflow

When upgrading `github-copilot-sdk`:

### Step 1: Run Assumption Tests FIRST

```bash
cd amplifier-module-provider-github-copilot
python -m pytest tests/sdk_assumptions/ -v --tb=long
```

### Step 2: If Tests Pass

- Proceed with upgrade
- Update "Last Validated SDK Version" in this README
- Update "Last Validation Date"

### Step 3: If Tests Fail

**DO NOT proceed with upgrade without investigation.**

1. **Read the failing test docstring** — it explains the assumption
2. **Check SDK changelog/release notes** for intentional changes
3. **If behavior changed intentionally**:
   - Update our implementation to match new behavior
   - Update the test to validate new behavior
   - Document the change in this README
4. **If behavior changed accidentally**:
   - File an issue with the SDK team
   - Pin to the previous working version until fixed

### Step 4: Document Changes

Add an entry to the Change History section below.

## Change History

| Date | SDK Version | Change | Migration Action |
| ---- | ----------- | ------ | ---------------- |
| 2026-02-07 | 0.1.0 | Initial validation | Baseline established |

## Test Categories Explained

### Unit Tests (Mocked SDK)

Tests in this directory use mocks to validate assumptions WITHOUT requiring
a live SDK connection. This allows:

- Fast execution (< 1 second)
- CI/CD integration without credentials
- Isolated testing of specific behaviors

### Integration Tests (Live SDK)

For live validation against the real SDK, use the integration tests:

```bash
RUN_LIVE_TESTS=1 python -m pytest tests/integration/ -v -s
```

Live tests are skipped by default and require authentication.

## Architecture Reference

```text
┌────────────────────────────────────────────────────────────────────────┐
│                         SDK Event Flow                                  │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  1. session.send(prompt)                                               │
│           │                                                            │
│           ▼                                                            │
│  2. ASSISTANT_MESSAGE_DELTA (streaming content)                        │
│           │                                                            │
│           ▼                                                            │
│  3. ASSISTANT_MESSAGE [with tool_requests if tools called]    ◄─────┐ │
│           │                                                          │ │
│           ▼                                                          │ │
│  4. preToolUse hook fires (we return "deny")                 ◄─────┐│ │
│           │                                                        ││ │
│           ▼                                                        ││ │
│  5. Tool handler NOT invoked (deny blocks it)                      ││ │
│           │                                                        ││ │
│           ▼                                                        ││ │
│  6. session.destroy() called (terminates loop)  ──────────────────┘│ │
│                                                                     │ │
│  ❌ Without destroy: CLI would retry with built-in tools ──────────┘ │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘

Critical Timing:
  - We capture tool_requests at step 3
  - We prevent execution at step 4
  - We stop retry loop at step 6
```

## Related Documentation

- [tool_capture.py](../../amplifier_module_provider_github_copilot/tool_capture.py) — Architecture note
- [Copilot SDK types.py](../../../copilot-sdk/python/copilot/types.py) — Hook type definitions
- [Copilot SDK session.py](../../../copilot-sdk/python/copilot/session.py) — Session implementation
