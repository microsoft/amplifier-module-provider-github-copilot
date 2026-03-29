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

No YAML configuration key changes. Existing `amplifier_settings.yaml` and
provider config files are compatible with v2.0.0 without modification.

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
