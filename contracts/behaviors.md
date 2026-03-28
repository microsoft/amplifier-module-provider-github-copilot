# Contract: Behaviors

## Version
- **Current:** 1.0 (v2.1 Kernel-Validated)
- **Module Reference:** Multiple modules
- **Config:** amplifier_module_provider_github_copilot/config/retry.yaml
- **Kernel Errors:** `amplifier_core.llm_errors`
- **Status:** Specification

---

## Overview

This contract defines behavioral policies that are config-driven. Unlike mechanism (Python), these behaviors can be tuned via YAML configuration.

---

## Retry Policy

### Config Schema

```yaml
retry:
  max_attempts: 3
  backoff:
    strategy: exponential_with_jitter
    base_delay_ms: 1000
    max_delay_ms: 30000
    jitter_factor: 0.1
```

### MUST Constraints

1. **MUST** respect max_attempts
2. **MUST** apply backoff between retries
3. **MUST** add jitter to prevent thundering herd
4. **MUST** only retry errors with `retryable=True`
5. **MUST NOT** retry errors with `retryable=False`
6. **MUST** honor `retry_after` when present

### Retryable Errors (Kernel Types)

| Kernel Error Type | Retryable | Notes |
|-------------------|-----------|-------|
| `AuthenticationError` | No | Invalid credentials |
| `AccessDeniedError` | No | Permission denied |
| `RateLimitError` | Yes | Has retry_after |
| `QuotaExceededError` | No | Billing limit |
| `LLMTimeoutError` | Yes | Transient |
| `ContentFilterError` | No | Content blocked |
| `ProviderUnavailableError` | Yes | Service down |
| `NetworkError` | Yes | Connection failure |
| `NotFoundError` | No | Model doesn't exist |
| `ContextLengthError` | No | Request too large |
| `InvalidRequestError` | No | Malformed request |
| `StreamError` | Yes | Mid-stream failure |
| `AbortError` | No | User cancellation |
| `InvalidToolCallError` | No | Malformed tool call |
| `ConfigurationError` | No | Setup problems |

---

## Streaming Timing Policy

### Config Schema

```yaml
streaming:
  event_queue_size: 1024
  ttft_warning_ms: 15000     # Time to first token (lenient for SDK latency)
  max_gap_warning_ms: 10000  # Max gap between tokens (lenient for tool work)
  max_gap_error_ms: 30000    # Error threshold
```

### MUST Constraints

1. **MUST** warn if TTFT exceeds threshold
2. **MUST** warn if inter-token gap exceeds warning threshold
3. **MAY** raise `LLMTimeoutError` if gap exceeds error threshold
4. **MUST NOT** block on queue full (drop oldest)

---

## Model Selection Policy

### Config Schema (models.yaml)

```yaml
provider:
  defaults:
    model: "claude-opus-4.5"   # Default model

models:
  - id: claude-opus-4.5
    display_name: "Claude Opus 4.5"
    context_window: 200000
    max_output_tokens: 32000
    capabilities:
      - streaming
      - tool_use
      - vision
```

### Model Selection Priority

When determining which model to use for a completion request:

```
Priority 1: request.model         (explicit per-request override)
Priority 2: config["default_model"]  (runtime config from mount/routing matrix)
Priority 3: YAML defaults.model    (static configuration fallback)
```

This priority order enables:
- **Routing Matrix Integration**: Amplifier kernel passes `config={"default_model": "model-id"}` at `mount()` time. Each sub-agent/delegate gets its assigned model.
- **Request Override**: Individual requests can still specify a different model.
- **Fallback Safety**: If mount config is missing, YAML provides sensible defaults.

### MUST Constraints

1. **MUST** respect model selection priority: request.model > config["default_model"] > YAML
2. **MUST NOT** mutate cached ProviderConfig (multiple providers share via @lru_cache)
3. **SHOULD** validate model exists before request
4. **MUST** raise `NotFoundError` for invalid models

### Test Anchors

| Anchor | Clause |
|--------|--------|
| `behaviors:ModelSelection:MUST:1` | Respects priority order |
| `behaviors:ModelSelection:MUST:2` | Does not mutate cached config |

---

## Logging Policy

### MUST Constraints

1. **MUST** log errors at ERROR level
2. **SHOULD** log warnings at WARN level
3. **MAY** log debug info at DEBUG level
4. **MUST NOT** log sensitive data (tokens, prompts in production)
5. **MUST** include correlation IDs for tracing

---

## Config Loading Policy

### MUST Constraints

1. **MUST** load config from `amplifier_module_provider_github_copilot/config/` (package directory)
2. **MUST** fail-fast with `ConfigurationError` if required config keys are missing (YAML is authoritative per Three-Medium Architecture)
3. **MUST NOT** load config from root `config/` directory (legacy, not packaged)
4. **MUST NOT** duplicate policy values in Python code (no hardcoded fallbacks that shadow YAML)

---

---

## Test File Organization Policy

### MUST Constraints

1. **MUST** add tests to existing domain-appropriate test files when the domain is already covered
2. **MUST** inventory existing test files before creating new ones (`ls tests/test_*.py`)
3. **MUST** map features to source modules, then to corresponding test files
4. **MUST NOT** create "gap" files (e.g., `test_coverage_gaps_*.py`, `test_*_final.py`)
5. **MUST NOT** use feature numbers in test file names (e.g., `test_f106_*.py`)
6. **MAY** create new test file ONLY when feature creates new source module

### Domain Mapping

| Source Module | Test Files |
|---------------|------------|
| `provider.py` | `test_provider.py`, `test_completion.py`, `test_retry.py` |
| `streaming.py` | `test_streaming.py` |
| `error_translation.py` | `test_error_translation.py`, `test_error_context.py` |
| Config loading | `test_config_loading.py`, `test_config_driven_timeout.py` |
| Tool parsing | `test_tool_parsing.py` |
| Observability | `test_observability.py`, `test_log_redaction.py` |
| SDK boundary | `test_sdk_boundary.py`, `test_sdk_boundary_quarantine.py` |

### Rationale

This is an open-source community contribution. Test files must be properly scoped to their domain, not lazy dump files that require future cleanup and merge work.

---

## Model Cache

### Overview

The provider SHOULD cache SDK model information to disk for persistence across sessions. This enables instant startup without API calls when the SDK is temporarily unavailable.

### Config Schema (model_cache.yaml)

```yaml
# config/model_cache.yaml
version: "1.0"

cache:
  disk_ttl_seconds: 86400     # 24 hours
  max_stale_seconds: 604800   # 7 days
  background_refresh: true
```

### SHOULD Constraints

1. **SHOULD** cache SDK models to disk for session persistence
2. **SHOULD** respect TTL from `config/model_cache.yaml`
3. **SHOULD** invalidate cache when stale (exceeds max_stale_seconds)

### Cross-Platform Cache Directory

| Platform | Directory |
|----------|-----------|
| Windows | `%LOCALAPPDATA%/amplifier/provider-github-copilot/` |
| macOS | `~/Library/Caches/amplifier/provider-github-copilot/` |
| Linux | `$XDG_CACHE_HOME/amplifier/provider-github-copilot/` |

### Test Anchors

| Anchor | Clause | Test Reference |
|--------|--------|----------------|
| `behaviors:ModelCache:SHOULD:1` | Persists to disk | `test_model_cache.py::test_write_cache_creates_file` |
| `behaviors:ModelCache:SHOULD:2` | Respects TTL | `test_model_cache.py::test_read_cache_returns_none_when_stale` |
| `behaviors:ModelCache:SHOULD:3` | Invalidates when stale | `test_model_cache.py::test_read_cache_respects_max_stale` |

---

## Model Discovery Errors

### Overview

When the SDK is unavailable AND the disk cache is empty/invalid, the provider MUST fail clearly with `ProviderUnavailableError`. The provider MUST NOT return hardcoded fallback values.

### Philosophy

**Fail clearly rather than fail silently with stale data.**

### MUST Constraints

1. **MUST** raise `ProviderUnavailableError` when SDK unavailable AND disk cache empty/invalid
2. **MUST** include reason in error message ("SDK connection failed, no cached models available")

### MUST NOT Constraints

1. **MUST NOT** return hardcoded fallback values — fail clearly instead
2. **MUST NOT** have `BUNDLED_MODEL_LIMITS` or similar hardcoded dictionaries

### Test Anchors

| Anchor | Clause | Test Reference |
|--------|--------|----------------|
| `behaviors:ModelDiscoveryError:MUST:1` | Raises ProviderUnavailableError | `test_models.py::test_list_models_raises_when_sdk_fails_and_no_cache` |
| `behaviors:ModelDiscoveryError:MUST:2` | Error includes reason | `test_models.py::test_error_message_includes_reason` |
| `behaviors:ModelDiscoveryError:MUST_NOT:1` | No hardcoded fallback | `test_model_cache.py::test_no_bundled_model_limits_dict` |

---

## Test Anchors

### Retry

| Anchor | Clause |
|--------|--------|
| `behaviors:Retry:MUST:1` | Respects max_attempts |
| `behaviors:Retry:MUST:2` | Applies backoff |
| `behaviors:Retry:MUST:3` | Adds jitter to prevent thundering herd |
| `behaviors:Retry:MUST:4` | Only retries errors with retryable=True |
| `behaviors:Retry:MUST:5` | Does NOT retry errors with retryable=False |
| `behaviors:Retry:MUST:6` | Honors retry_after when present |

### Streaming

| Anchor | Clause |
|--------|--------|
| `behaviors:Streaming:MUST:1` | Warns on slow TTFT |

### Models

| Anchor | Clause |
|--------|--------|
| `behaviors:Models:MUST:1` | Raises NotFoundError for invalid model |
| `behaviors:Models:MUST:2` | Respects model selection priority (request > config > YAML) |

### Config

| Anchor | Clause |
|--------|--------|
| `behaviors:Config:MUST:1` | Loads config from package directory |
| `behaviors:Config:MUST:2` | Fallback defaults match YAML |

### Logging

| Anchor | Clause |
|--------|--------|
| `behaviors:Logging:MUST:1` | Logs errors at ERROR level |
| `behaviors:Logging:MUST:4` | Does NOT log sensitive data |
| `behaviors:Logging:MUST:5` | Includes correlation IDs |

### TestFiles

| Anchor | Clause |
|--------|--------|
| `behaviors:TestFiles:MUST:1` | Add to existing domain files |
| `behaviors:TestFiles:MUST:2` | No gap files |
| `behaviors:TestFiles:MUST:3` | No feature numbers in names |

---

## Implementation Checklist

- [ ] Retry policy reads from config
- [ ] Backoff with jitter implemented
- [ ] Only retry errors with `retryable=True`
- [ ] Streaming timing warnings work
- [ ] NotFoundError for invalid models
- [ ] Model selection priority respected
- [ ] Logging follows policy
