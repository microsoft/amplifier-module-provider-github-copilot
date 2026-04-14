# Migration Guide: v1.0.x → v2.0.0

## Overview

v2.0.0 is a **breaking change release** driven by the upgrade of the underlying
`github-copilot-sdk` dependency from `0.1.x` to `0.2.x`. The SDK's `0.2.0`
release introduced breaking changes to its public API, which required renaming
the provider class, simplifying the exception hierarchy, and removing several
symbols that were previously exposed as implementation details.

**Summary of changes:**

- Provider class renamed: `CopilotSdkProvider` → `GitHubCopilotProvider`
- Exception names simplified (dropped `Copilot` prefix)
- Internal classes removed from public API (they were never intended for external use)
- SDK authentication types removed (SDK v0.2.0 changed authentication patterns)
- `ModelIdPattern` removed — use model name strings directly
- SDK dependency: `github-copilot-sdk>=0.1.32,<0.2.0` → `>=0.2.0,<0.3.0`

---

## Dependency Update

Update your `pyproject.toml` or `requirements.txt`:

```toml
# v1.x
github-copilot-sdk>=0.1.32,<0.2.0

# v2.0.0
github-copilot-sdk>=0.2.0,<0.3.0
```

---

## Renamed Symbols

### Provider Class

| v1.x | v2.0.0 |
|------|--------|
| `CopilotSdkProvider` | `GitHubCopilotProvider` |

### Exceptions

| v1.x | v2.0.0 |
|------|--------|
| `CopilotProviderError` | `ProviderError` |
| `CopilotAuthenticationError` | `AuthenticationError` |
| `CopilotConnectionError` | `ConnectionError` |
| `CopilotRateLimitError` | `RateLimitError` |
| `CopilotModelNotFoundError` | `ModelNotFoundError` |
| `CopilotSessionError` | `SessionError` |
| `CopilotSdkLoopError` | `SdkLoopError` |
| `CopilotAbortError` | `AbortError` |
| `CopilotTimeoutError` | `TimeoutError` |

---

## Removed Symbols

The following symbols have been **removed entirely** and have no replacement.
Importing them will raise `ImportError` with a descriptive message.

### Internal Implementation Details

These were never part of the public API contract. Remove any imports of these:

| Symbol | Reason |
|--------|--------|
| `SdkEventHandler` | Internal implementation detail |
| `LoopController` | Internal implementation detail |
| `ToolCaptureStrategy` | Internal implementation detail |
| `CircuitBreaker` | Internal implementation detail |
| `CapturedToolCall` | Internal implementation detail |

### SDK Authentication Types

These types were tied to `github-copilot-sdk` v0.1.x authentication patterns,
which changed in v0.2.0:

| Symbol | Reason |
|--------|--------|
| `AuthStatus` | SDK v0.2.0 changed authentication patterns |
| `SessionInfo` | SDK v0.2.0 changed authentication patterns |
| `SessionListResult` | SDK v0.2.0 changed authentication patterns |

### Model Utilities

| Symbol | Replacement |
|--------|-------------|
| `ModelIdPattern` | Use model name strings directly (e.g., `"gpt-4o"`) |

---

## Import Changes

```python
# v1.x
from amplifier_module_provider_github_copilot import (
    CopilotSdkProvider,
    CopilotProviderError,
    CopilotAuthenticationError,
    CopilotConnectionError,
    CopilotRateLimitError,
    CopilotModelNotFoundError,
    CopilotSessionError,
    CopilotSdkLoopError,
    CopilotAbortError,
    CopilotTimeoutError,
)

# v2.0.0
from amplifier_module_provider_github_copilot import (
    GitHubCopilotProvider,
    ProviderError,
    AuthenticationError,
    ConnectionError,
    RateLimitError,
    ModelNotFoundError,
    SessionError,
    SdkLoopError,
    AbortError,
    TimeoutError,
)
```

---

## Configuration Changes

### Breaking: `ObservabilityConfig.raw_payloads` renamed to `.raw`

The verbosity flag on `ObservabilityConfig` has been renamed from `raw_payloads` to `raw`.
Any code that read `config.raw_payloads` directly will get an `AttributeError`. There is
no user-facing YAML file for this setting — `load_observability_config()` returns the
dataclass with defaults; the `raw` flag is set via the provider `config:` block (see
additive keys table below).

```python
# Before
if config.raw_payloads:  # AttributeError from this release onward
    ...

# After
if config.raw:
    ...
```

The default remains `False`. Most users are unaffected (raw payloads are off by default
and the flag is an internal implementation detail, not part of the stable public API).

### New (additive): Runtime overrides in provider config

The following keys can now be set in the provider `config` block of your bundle YAML.
All keys are optional; absent keys fall back to policy defaults.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `raw` | `bool` | `false` | Include raw request/response payloads in `llm:request`/`llm:response` events. Accepts `true`/`false` or `"true"`/`"false"` strings. |
| `max_retries` | `int` | `2` | Number of retries (0 = no retry). Total attempts = max_retries + 1. |
| `min_retry_delay` | `float` | `1.0` | Minimum retry back-off in **seconds**. |
| `max_retry_delay` | `float` | `30.0` | Maximum retry back-off cap in **seconds**. |
| `retry_jitter` | `float` | `0.1` | Jitter factor `[0.0, 1.0]` applied to computed delay. |
| `overloaded_delay_multiplier` | `float` | `10.0` | Multiplier applied to back-off for errors marked `overloaded: true` (e.g., rate-limit). Must be ≥ 1.0. |

Example bundle YAML:

```yaml
providers:
  - module: provider-github-copilot
    name: github-copilot
    config:
      raw: false
      max_retries: 3
      min_retry_delay: 2.0
      max_retry_delay: 60.0
      retry_jitter: 0.2
      overloaded_delay_multiplier: 5.0
```

---

## Public API Surface (v2.0.0)

The stable public API is:

```python
from amplifier_module_provider_github_copilot import (
    mount,                  # Amplifier module entrypoint
    GitHubCopilotProvider,  # Provider class
)

# Kernel types — import directly from amplifier_core
from amplifier_core import ProviderInfo, ModelInfo
```

All other symbols are internal implementation details and may change without notice.
