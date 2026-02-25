# Process-Level Singleton Fix for GitHub Copilot Provider

## Goal

Fix the GitHub Copilot provider so that all sub-agents spawned by a recipe within a single Amplifier session share one `CopilotClientWrapper` instance (and therefore one copilot CLI subprocess), rather than each spawning their own.

Tracked as: [Issue #7](https://github.com/microsoft/amplifier-module-provider-github-copilot/issues/7)

## Background

The copilot CLI binary is Electron-bundled at ~500 MB per process. With the current implementation, N sub-agents spawned by a recipe creates N subprocesses consuming N × ~500 MB RAM — the root cause of the OOM being investigated.

Two architectural facts constrain the solution:

- Sub-agents spawned by the `task` tool run as async coroutines in the **same Python process** and same asyncio event loop as the parent session (kernel-guaranteed)
- Each sub-agent gets its own fresh `ModuleCoordinator` — coordinators are not shared across sessions

There is no kernel facility for cross-session resource sharing; resource pooling is module-level policy, consistent with Amplifier's "mechanism not policy" principle.

## Approach

A process-level singleton held as module-level state in `__init__.py`, with reference counting to ensure the subprocess lives exactly as long as at least one session is mounted. The change is entirely contained to `__init__.py` — no other files change.

Two other approaches were considered and rejected:

- **Process-level cap (semaphore)**: Limits blast radius but doesn't eliminate waste; more complex without being more correct
- **Status quo + polish only**: Valid if the OOM were purely a runaway delegation bug, but the singleton is the right architecture independently — even legitimate heavy recipes waste N × 500 MB today

## Architecture

The change is surgical. `CopilotClientWrapper` itself doesn't change. The only change is in when and how many times it gets created.

**Before:**
```
mount() called → new CopilotClientWrapper() → new copilot subprocess
mount() called → new CopilotClientWrapper() → new copilot subprocess
mount() called → new CopilotClientWrapper() → new copilot subprocess
```

**After:**
```
mount() called → _acquire_shared_client() → new copilot subprocess   (ref=1)
mount() called → _acquire_shared_client() → reuse existing            (ref=2)
mount() called → _acquire_shared_client() → reuse existing            (ref=3)
cleanup called → _release_shared_client() → ref=2
cleanup called → _release_shared_client() → ref=1
cleanup called → _release_shared_client() → ref=0 → subprocess shuts down
```

**Blast radius:** `__init__.py` only. `CopilotClientWrapper`, `provider.py`, `tool_capture.py`, `_constants.py`, and `converters.py` are untouched.

## Components

### Module-Level Singleton State

Three module-level variables at the top of `__init__.py`:

```python
_shared_client: CopilotClientWrapper | None = None
_shared_client_refcount: int = 0
_shared_client_lock: asyncio.Lock | None = None
```

The lock is initialized lazily rather than at module load time. `asyncio.Lock()` must be created on the running event loop — creating it at import time can cause issues if the module is imported before an event loop exists (common in test environments). A `_get_lock()` helper handles this:

```python
def _get_lock() -> asyncio.Lock:
    global _shared_client_lock
    if _shared_client_lock is None:
        _shared_client_lock = asyncio.Lock()
    return _shared_client_lock
```

### Acquire and Release Functions

```python
async def _acquire_shared_client(config, timeout) -> CopilotClientWrapper:
    global _shared_client, _shared_client_refcount
    async with _get_lock():
        if _shared_client is None:
            _shared_client = CopilotClientWrapper(config, timeout)
        _shared_client_refcount += 1
        return _shared_client

async def _release_shared_client() -> None:
    global _shared_client, _shared_client_refcount
    async with _get_lock():
        _shared_client_refcount -= 1
        if _shared_client_refcount <= 0:
            if _shared_client is not None:
                await _shared_client.close()
                _shared_client = None
            _shared_client_refcount = 0  # safety floor against accidental negatives
```

The safety floor at zero matters: if cleanup is ever called more times than mount (e.g., in test teardown), a negative count must not prevent the next acquisition from creating a fresh client.

### Updated `mount()`

`mount()` changes in one place only — acquiring the shared client instead of constructing a new one directly:

```python
async def mount(coordinator, config):
    timeout = config.get("timeout", DEFAULT_TIMEOUT)

    client = await _acquire_shared_client(config, timeout)

    provider = CopilotSdkProvider(client=client, config=config)
    coordinator.providers.register("github-copilot", provider)

    async def cleanup():
        await _release_shared_client()

    return cleanup
```

Everything else — provider construction, coordinator registration, returning a cleanup callable — is unchanged.

## Data Flow

`_acquire_shared_client()` only uses `config` and `timeout` on the **first** call when it constructs the wrapper. Subsequent calls reuse the existing client regardless of the values passed. Sub-agents inherit bundle config from the parent, so in practice they all pass identical values.

If a sub-agent passes a different `timeout` than the first mount, `_acquire_shared_client()` emits a `DEBUG` log warning and returns the existing client unchanged. No exception is raised — the caller receives a working client and the discrepancy is visible in logs.

## Error Handling

**Config mismatch**: `DEBUG` warning logged, existing client returned, no exception raised.

**Concurrent mounts at startup**: Two sub-agents mounting simultaneously before the client exists are serialized by the `asyncio.Lock`. The second waiter finds `_shared_client` already populated, increments the refcount, and returns. Double-creation is not possible.

**Unclean shutdown** (SIGKILL / process crash): The refcount never reaches zero, so `close()` never fires and the copilot subprocess becomes an orphan. This is acceptable — the OS reclaims the subprocess when the parent Python process dies. A code comment will document this reasoning explicitly so future maintainers don't treat it as an oversight.

## Observability

Three events are emitted via `coordinator.hooks.emit()`, consistent with existing module patterns (e.g., `provider:tool_sequence_repaired`):

| Event | When | Payload |
|---|---|---|
| `github-copilot:subprocess_created` | First acquisition | `session_id` |
| `github-copilot:subprocess_reused` | Subsequent acquisitions | `session_id`, `refcount` |
| `github-copilot:subprocess_shutdown` | Refcount reaches zero | `session_id` |

These are visible in session logs without any extra configuration.

## Testing Strategy

New tests live in `tests/test_mount.py`. `CopilotClientWrapper.__init__` is mocked to count instantiations. A pytest fixture resets `_shared_client`, `_shared_client_refcount`, and `_shared_client_lock` to `None`/`0`/`None` before each test — required for isolation with module-level state.

| Test | What it verifies |
|---|---|
| **Singleton creation** | `mount()` once → one `CopilotClientWrapper` created, refcount=1 |
| **Reuse** | `mount()` three times → one `CopilotClientWrapper` created, refcount=3 |
| **Cleanup lifecycle** | Three mounts, three cleanups → `close()` called only on the third (refcount=0) |
| **Concurrent mounts** | `asyncio.gather()` fires multiple `mount()` calls simultaneously → still one client (exercises the lock) |
| **Config mismatch warning** | Mount with `timeout=300`, mount again with `timeout=600` → `DEBUG` warning emitted, no exception |

No integration tests are needed — the copilot subprocess is already mocked in the existing suite. These tests only exercise the reference counting logic, which is pure Python with no external dependencies. Existing tests for `CopilotClientWrapper` and `CopilotSdkProvider` are unaffected since neither class changes.

## Open Questions

None — design is fully validated and ready for implementation.
