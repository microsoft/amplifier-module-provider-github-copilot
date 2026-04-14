# Contract: Error Hierarchy

## Version
- **Current:** 1.2
- **Module Reference:** amplifier_module_provider_github_copilot/error_translation.py, provider.py
- **History:**
  - 1.1 (2026-03-15) — Removed erroneous `src/` prefix
  - 1.2 (2026-04-04) — Added AbortError:MUST:2 (asyncio.timeout expiry must not produce AbortError)
- **Config:** amplifier_module_provider_github_copilot/config/errors.yaml
- **Kernel Source:** `amplifier_core.llm_errors`
- **Status:** Specification

---

## Overview

This contract defines the error translation requirements for the provider. The provider MUST translate SDK errors into kernel error types from `amplifier_core.llm_errors`. Custom error types are NOT allowed — they break cross-provider error handling.

---

## Kernel Error Hierarchy

All errors come from `amplifier_core.llm_errors`:

```
LLMError (base)
├── AuthenticationError (HTTP 401/403)
│   └── AccessDeniedError (HTTP 403 - permission denied)
├── RateLimitError (HTTP 429, retryable=True)
│   └── QuotaExceededError (billing limit, retryable=False)
├── LLMTimeoutError (retryable=True)
├── ContentFilterError (safety filter)
├── ProviderUnavailableError (HTTP 5xx, retryable=True)
│   └── NetworkError (connection failure, retryable=True)
├── NotFoundError (HTTP 404 - model not found)
├── ContextLengthError (HTTP 413 - context exceeded)
├── InvalidRequestError (HTTP 400/422)
├── StreamError (mid-stream connection failure, retryable=True)
├── AbortError (caller cancellation)
├── InvalidToolCallError (malformed tool call)
└── ConfigurationError (setup problems)
```

---

## Base Error Attributes

All `LLMError` subclasses have:

```python
class LLMError(Exception):
    provider: str | None       # Provider name (e.g., "github-copilot")
    model: str | None          # Model identifier
    status_code: int | None    # HTTP status code
    retryable: bool            # Whether retry is appropriate
    retry_after: float | None  # Seconds to wait before retry
    delay_multiplier: float    # Backoff delay multiplier (default 1.0)
```

---

## SDK → Kernel Error Mapping

| SDK Error Pattern | Kernel Error | Retryable |
|-------------------|--------------|-----------|
| `AuthenticationError`, 401, 403 | `AuthenticationError` | No |
| `RateLimitError`, 429 | `RateLimitError` | Yes |
| `QuotaExceededError` | `QuotaExceededError` | No |
| `TimeoutError` | `LLMTimeoutError` | Yes |
| `ContentFilterError`, safety | `ContentFilterError` | No |
| `ConnectionError`, 5xx | `ProviderUnavailableError` | Yes |
| `ProcessExitedError`, network | `NetworkError` | Yes |

**error-hierarchy:ConnectionError:MUST:1** — SDK `ConnectionError` MUST map to
`ProviderUnavailableError` (not `NetworkError`). Connection refused indicates the
SDK subprocess or remote endpoint is unavailable — a provider-level failure, not a
raw network transport failure. `ProcessExitedError` maps to `NetworkError`.
| `ModelNotFoundError`, 404 | `NotFoundError` | No |
| Session errors | `ProviderUnavailableError` | Yes |
| Circuit breaker | `ProviderUnavailableError` | No |
| Stream interruption | `StreamError` | Yes |
| Abort signal | `AbortError` | No |
| `CAPIError: 400`, image_url validation | `InvalidRequestError` | No |

### Image Validation Error

**Evidence:** SDK returns `CAPIError: 400 invalid request body, failed to validate schema: ...image_url...`
when an unsupported MIME type is sent (e.g., `image/tiff`).

| Test Anchor | Clause |
|-------------|--------|
| `error-hierarchy:ImageValidation:MUST:1` | CAPIError 400 with image_url → InvalidRequestError |
| `error-hierarchy:ImageValidation:MUST:2` | Image validation errors are non-retryable |
| `error-hierarchy:ImageValidation:MUST:3` | Original exception preserved via chaining |

---

## Config Schema (errors.yaml)

```yaml
version: "1.0"

error_mappings:
  - sdk_patterns: ["AuthenticationError", "InvalidTokenError", "PermissionDeniedError"]
    string_patterns: ["401", "403", "unauthorized", "permission denied"]
    kernel_error: AuthenticationError
    retryable: false

  - sdk_patterns: ["RateLimitError"]
    string_patterns: ["429", "rate limit"]
    kernel_error: RateLimitError
    retryable: true
    extract_retry_after: true

  - sdk_patterns: ["QuotaExceededError"]
    string_patterns: ["quota exceeded", "billing"]
    kernel_error: QuotaExceededError
    retryable: false

  - sdk_patterns: ["TimeoutError", "RequestTimeoutError"]
    string_patterns: ["timeout", "timed out"]
    kernel_error: LLMTimeoutError
    retryable: true

  - sdk_patterns: ["ContentFilterError", "SafetyError"]
    string_patterns: ["content filter", "safety", "blocked"]
    kernel_error: ContentFilterError
    retryable: false

  - sdk_patterns: ["ConnectionError", "ProcessExitedError"]
    string_patterns: ["connection refused", "process exited"]
    kernel_error: NetworkError
    retryable: true

  - sdk_patterns: ["ModelNotFoundError"]
    string_patterns: ["model not found", "404"]
    kernel_error: NotFoundError
    retryable: false

default:
  kernel_error: ProviderUnavailableError
  retryable: false    # Conservative: unknown errors do not retry. Prevents spurious retries on genuinely broken requests.
```

---

## Translation Function

```python
from amplifier_core.llm_errors import (
    LLMError,
    AuthenticationError,
    RateLimitError,
    LLMTimeoutError,
    ContentFilterError,
    ProviderUnavailableError,
    NetworkError,
    NotFoundError,
)

def translate_sdk_error(
    exc: Exception,
    config: ErrorConfig,
    *,
    provider: str = "github-copilot",
    model: str | None = None,
) -> LLMError:
    """
    Translate SDK exception to kernel LLMError.
    
    Contract: error-hierarchy.md
    
    - MUST NOT raise (always returns)
    - MUST use config patterns (no hardcoded mappings)
    - MUST chain original via `raise X from exc`
    - MUST set provider attribute
    """
```

---

## MUST Constraints

1. **MUST** use kernel error types from `amplifier_core.llm_errors`
2. **MUST NOT** create custom error classes
3. **MUST** set `provider="github-copilot"` on all errors
4. **MUST** preserve original exception via chaining (`raise X from original`)
5. **MUST** use config-driven pattern matching
6. **MUST** fall through to `ProviderUnavailableError(retryable=False)` for unknown errors (conservative non-retry per Golden Vision V2)

---

## AbortError Translation

**MUST:1** — SDK `AbortError`, `CancelledError`, or exceptions with message patterns matching `abort`, `cancelled`, `canceled`, `user interrupt`, or `keyboard interrupt` MUST translate to kernel `AbortError(retryable=False)`.

**MUST:2** — `CancelledError` arising from `asyncio.timeout` deadline expiry inside `_execute_sdk_completion` MUST NOT translate to `AbortError`. The `asyncio.timeout` context manager MUST hold sole cancellation ownership during `idle_event.wait()`; no nested `asyncio.wait_for` with the same deadline MAY be present. Duplicating the deadline splits cancel ownership and can prevent `asyncio.timeout.__aexit__` from converting `CancelledError` to `TimeoutError`, causing the C-2 guard to misclassify an internal timeout as `AbortError(retryable=False)` instead of `LLMTimeoutError(retryable=True)`. The outer `asyncio.timeout` is the sole deadline enforcement mechanism for the idle wait.

---

## SessionLifecycle Errors

SDK `SessionCreateError`, `SessionInitError`, `SessionDestroyError`, `SessionCloseError`, or exceptions with message patterns matching session lifecycle operations SHOULD translate to `ProviderUnavailableError(retryable=True)` as these represent transient infrastructure failures.

---

## Context Extraction

When `context_extraction` patterns are defined for an error mapping in `errors.yaml`, the translated error message SHOULD include extracted context fields appended as `[context: field=value, ...]` suffix. This enables debugging without requiring full stack traces.

---

## Config Loading

Error config loading supports two paths:
1. **File path loading** — Direct path to `errors.yaml`
2. **Client loading** — Via `importlib.resources` from package

Both paths MUST:
- Parse `context_extraction` patterns
- Produce identical `ErrorConfig` objects
- Fall back gracefully when config is missing (default to `ProviderUnavailableError`, `retryable=False` — conservative non-retry per Golden Vision V2)

---

## Test Anchors

### Kernel

| Anchor | Clause |
|--------|--------|
| `error-hierarchy:Kernel:MUST:1` | Uses kernel types only |
| `error-hierarchy:Kernel:MUST:2` | Sets provider attribute |

### Translation

| Anchor | Clause |
|--------|--------|
| `error-hierarchy:Translation:MUST:1` | Never raises |
| `error-hierarchy:Translation:MUST:2` | Uses config patterns |
| `error-hierarchy:Translation:MUST:3` | Chains original exception |

### RateLimit

| Anchor | Clause |
|--------|--------|
| `error-hierarchy:RateLimit:MUST:1` | Extracts retry_after |

### Default

| Anchor | Clause |
|--------|--------|
| `error-hierarchy:Default:MUST:1` | Falls through to ProviderUnavailableError |

### AbortError

| Anchor | Clause |
|--------|--------|
| `error-hierarchy:AbortError:MUST:1` | User abort produces non-retryable AbortError |
| `error-hierarchy:AbortError:MUST:2` | asyncio.timeout expiry must not produce AbortError — outer timeout holds sole ownership |}

### SessionLifecycle

| Anchor | Clause |
|--------|--------|
| `error-hierarchy:SessionLifecycle:SHOULD:1` | Session lifecycle errors are retryable |

### Context

| Anchor | Clause |
|--------|--------|
| `error-hierarchy:Context:SHOULD:1` | Error messages include actionable context |

### ConnectionError

| Anchor | Clause |
|--------|--------|
| `error-hierarchy:ConnectionError:MUST:1` | SDK ConnectionError maps to ProviderUnavailableError (not NetworkError) |

### Config

| Anchor | Clause |
|--------|--------|
| `error-hierarchy:config:MUST:1` | File path loading includes context_extraction |
| `error-hierarchy:config:MUST:2` | Client loading includes context_extraction |
| `error-hierarchy:config:MUST:3` | Both loading paths produce identical results |
| `error-hierarchy:config:MUST:4` | Client config supports context extraction |
| `error-hierarchy:config:SHOULD:1` | Missing config falls back gracefully |

---

## Implementation Checklist

- [ ] Import all error types from `amplifier_core.llm_errors`
- [ ] Config file has all pattern mappings
- [ ] Translation function uses config patterns
- [ ] All errors set `provider="github-copilot"`
- [ ] Original exception chained via `raise ... from`
- [ ] Rate limit extracts retry_after from message
- [ ] Unknown errors default to ProviderUnavailableError
