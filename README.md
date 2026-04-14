# Amplifier GitHub Copilot Provider Module

GitHub Copilot SDK integration for Amplifier — provides access to Anthropic and OpenAI models via your GitHub Copilot plan.

## Prerequisites

- **Python 3.11+**
- **[GitHub Copilot plan](https://github.com/features/copilot/plans)** — Free, Pro, Pro+, Business, or Enterprise
- **[UV](https://github.com/astral-sh/uv)** (optional) — Fast Python package manager (pip works too)

### Installing UV

```bash
# macOS/Linux/WSL
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

> **No Node.js required.** The Copilot SDK binary is bundled with the Python package
> and discovered automatically.

## Purpose

Provides access to Anthropic Claude and OpenAI GPT models as an LLM provider for Amplifier, using the GitHub Copilot SDK. Model availability reflects your GitHub Copilot plan — models are discovered dynamically at runtime.

## Authentication

Set a GitHub token as an environment variable. The provider checks these in order (first non-empty wins):

| Priority | Variable | Use case |
| --- | --- | --- |
| 1 | `COPILOT_AGENT_TOKEN` | Copilot agent mode |
| 2 | `COPILOT_GITHUB_TOKEN` | Recommended for direct use |
| 3 | `GH_TOKEN` | GitHub CLI compatible |
| 4 | `GITHUB_TOKEN` | GitHub Actions compatible |

### Option 1: `gh` CLI bridge (recommended)

**Linux/macOS:**
```bash
export GITHUB_TOKEN=$(gh auth token)
```

**Windows PowerShell:**
```powershell
$env:GITHUB_TOKEN = (gh auth token)
```

One command to bridge your existing `gh` CLI authentication into Amplifier.

> **Tip:** Many developers already have `gh` CLI authenticated —
> if so, this is the fastest path to get started.

### Option 2: Direct token

**Linux/macOS:**
```bash
export GITHUB_TOKEN="<YOUR_TOKEN_HERE>"
```

**Windows PowerShell:**
```powershell
$env:GITHUB_TOKEN = "<YOUR_TOKEN_HERE>"
```

Use a GitHub Personal Access Token directly.

## Installation

### Quick Start (Recommended Order)

**Linux/macOS:**
```bash
# 1. Set token (if using gh CLI)
export GITHUB_TOKEN=$(gh auth token)

# 2. Install provider (includes SDK)
amplifier provider install github-copilot

# 3. Configure
amplifier init
```

**Windows PowerShell:**
```powershell
# 1. Set token (if using gh CLI)
$env:GITHUB_TOKEN = (gh auth token)

# 2. Install provider (includes SDK)
amplifier provider install github-copilot

# 3. Configure
amplifier init
```

> **Tip:** For permanent token setup:
> - **Linux:** Add `export GITHUB_TOKEN=$(gh auth token)` to `~/.bashrc`
> - **macOS:** Add `export GITHUB_TOKEN=$(gh auth token)` to `~/.zshrc`
> - **Windows:** Add `$env:GITHUB_TOKEN = (gh auth token)` to your PowerShell profile (`$PROFILE`)

### Bundle reference

Reference the provider directly from a bundle YAML using a branch or commit SHA:

```yaml
providers:
  - module: provider-github-copilot
    source: git+https://github.com/microsoft/amplifier-module-provider-github-copilot@main
    config:
      default_model: claude-opus-4.5
```

## Usage

```bash
# Interactive session
amplifier run -p github-copilot

# One-shot prompt
amplifier run -p github-copilot -m claude-sonnet-4 "Explain this codebase"

# List available models
amplifier provider models github-copilot
```

## Supported Models

Models are discovered dynamically from the SDK at runtime — the list reflects your GitHub Copilot plan. The tables below show the current set as of SDK 0.2.2; run `amplifier provider models github-copilot` for the live list.

**Anthropic:**

| Model ID | Context | Max Output | Capabilities |
| --- | --- | --- | --- |
| `claude-sonnet-4.6` | 200k | 32k | streaming, tools, vision, thinking |
| `claude-sonnet-4.5` | 200k | 32k | streaming, tools, vision |
| `claude-haiku-4.5` | 200k | 64k | streaming, tools, vision |
| `claude-opus-4.6` | 200k | 32k | streaming, tools, vision, thinking |
| `claude-opus-4.6-1m` | 1M | 64k | streaming, tools, vision, thinking |
| `claude-opus-4.5` | 200k | 32k | streaming, tools, vision |
| `claude-sonnet-4` | 216k | 88k | streaming, tools, vision |

**OpenAI:**

| Model ID | Context | Max Output | Capabilities |
| --- | --- | --- | --- |
| `gpt-5.4` | 400k | 128k | streaming, tools, vision, thinking |
| `gpt-5.3-codex` | 400k | 128k | streaming, tools, vision, thinking |
| `gpt-5.2-codex` | 400k | 128k | streaming, tools, vision, thinking |
| `gpt-5.2` | 400k | 128k | streaming, tools, vision, thinking |
| `gpt-5.1` | 264k | 136k | streaming, tools, vision, thinking |
| `gpt-5.4-mini` | 400k | 128k | streaming, tools, vision, thinking |
| `gpt-5-mini` | 264k | 136k | streaming, tools, vision, thinking |
| `gpt-4.1` | 128k | 64k | streaming, tools, vision |

> **Tip:** Want intelligent model selection? Use the [Routing Matrix bundle](https://github.com/microsoft/amplifier-bundle-routing-matrix) to select models by semantic role (`coding`, `reasoning`, `fast`) rather than hardcoding a model ID.

## Configuration

The provider runs with sensible defaults. Set values in the `config` block of your bundle YAML:

```yaml
providers:
  - module: provider-github-copilot
    name: github-copilot
    config:
      default_model: claude-opus-4.5
```

| Key | Default | Description |
| --- | --- | --- |
| `default_model` | `"claude-opus-4.5"` | Model used when the caller does not specify one. Any ID from `list_models()` is valid. |
| `raw` | `false` | Include raw SDK payloads as a `"raw"` field in `llm:request` / `llm:response` events. See [Raw Payload Logging](#raw-payload-logging). |

### Raw Payload Logging

Set `raw: true` to capture the exact data exchanged with the Copilot SDK before any processing:

```yaml
providers:
  - module: provider-github-copilot
    config:
      raw: true
```

When enabled, the standard `llm:request` and `llm:response` events include an additional `"raw"` field containing the complete, redacted payload:

| Event | `"raw"` field contains |
| --- | --- |
| `llm:request` | Complete request payload sent to the SDK (model, prompt, tools, system message) |
| `llm:response` | Complete response object returned by the SDK |

Raw payloads pass through `redact_dict()` — tokens and credentials are scrubbed before the field is added to the event.

> **Warning:** Raw events contain the full conversation content including tool definitions
> and system messages. Use only for deep provider integration debugging. Disable in
> production to avoid high log volume and potential data exposure.

> **Note:** Accepts `true`/`false` (bool) or strings `"true"`, `"1"`, `"yes"` (truthy) /
> anything else (falsy). The string `"false"` is correctly treated as disabled —
> `bool("false") == True` is a Python footgun that `_parse_raw_flag` guards against.

### Retry and Error Handling

The provider manages its own retry loop, giving full control over backoff timing, per-error-class behaviour, and `Retry-After` header honoring.

#### Error Translation

All errors are translated to typed kernel error types before the retry loop evaluates them. Every translated error preserves the original as `__cause__`.

| Trigger | Kernel Error | Retryable |
| --- | --- | --- |
| Circuit breaker open | `ProviderUnavailableError` | No |
| Authentication or permission failure | `AuthenticationError` | No |
| Rate limit (429) | `RateLimitError` | Yes |
| Quota or billing limit exceeded | `QuotaExceededError` | No |
| Request timed out | `LLMTimeoutError` | Yes |
| Content policy violation | `ContentFilterError` | No |
| Connection refused or unreachable | `ProviderUnavailableError` | Yes |
| SDK process exited unexpectedly | `NetworkError` | Yes |
| Model not found | `NotFoundError` | No |
| Context window exceeded | `ContextLengthError` | No |
| Stream interrupted | `StreamError` | Yes |
| Malformed or conflicting tool call | `InvalidToolCallError` | No |
| Provider configuration error | `ConfigurationError` | No |
| Request aborted or cancelled | `AbortError` | No |
| Session lifecycle failure | `ProviderUnavailableError` | Yes |
| Invalid request body (e.g. unsupported image format) | `InvalidRequestError` | No |
| Any other error | `ProviderUnavailableError` | No |

`RateLimitError` responses carry a `Retry-After` value; if present, it is used directly as the next retry delay (overriding the backoff formula).

#### Backoff Formula

Each retry delay is computed as follows. `attempt` is **0-indexed** — `0` is the first retry:

```
delay  = min(base_delay × 2^attempt, max_delay)   # attempt = 0, 1, 2, …
jitter = delay × jitter_factor × random(−1, 1)
sleep  = max(0, delay + jitter)
```

| Attempt (0-indexed) | Base | Capped | With jitter (±10%) |
| --- | --- | --- | --- |
| 0 (first retry) | 1 s | 1 s | 0.9 – 1.1 s |
| 1 | 2 s | 2 s | 1.8 – 2.2 s |
| 2 | 4 s | 4 s | 3.6 – 4.4 s |

**Example: Overloaded signal (10× multiplier, defaults)**

When a `RateLimitError` carries `delay_multiplier > 1.0` (set by the provider on overloaded responses), the base delay (after capping, with jitter) is multiplied. Default `overloaded_delay_multiplier` is `10.0`:

| Attempt | base_delay | capped | ×10 | Sleep range (±10%) |
| --- | --- | --- | --- | --- |
| 0 (first retry) | 1 s | 1 s | 10 s | 9 – 11 s |
| 1 | 2 s | 2 s | 20 s | 18 – 22 s |

With default `max_retries: 2`, total wait is ≈ 30 s before the request is abandoned.
`Retry-After` from the server header always takes precedence over the multiplied delay.

#### Retry Configuration

Retry parameters can be overridden via bundle config. All keys are optional; omitted keys use the defaults shown.

| Config Key | Default | Description |
| --- | --- | --- |
| `max_retries` | `2` | Number of retries after the first attempt (`0` = fail fast, single attempt) |
| `min_retry_delay` | `1.0` | Minimum base delay in seconds (doubles each attempt) |
| `max_retry_delay` | `30.0` | Maximum delay cap in seconds before jitter is applied |
| `retry_jitter` | `0.1` | Jitter fraction applied as ± of the capped delay (`0.0`–`1.0`) |
| `overloaded_delay_multiplier` | `10.0` | Multiplier applied to backoff when an overloaded signal is present (e.g. `RateLimitError` with `delay_multiplier > 1.0`); `Retry-After` still takes precedence. Must be `>= 1.0` — values below `1.0` are rejected at construction and fall back to `1.0`. |

#### Retry Events

A `provider:retry` event is emitted before each retry sleep:

| Field | Description |
| --- | --- |
| `provider` | Provider name (`"github-copilot"`) |
| `model` | Model being called |
| `attempt` | Current attempt number (1-based in event payload) |
| `max_retries` | Total attempt count including the initial call (`max_retries + 1`); e.g. with `max_retries: 2` configured, the event emits `3` |
| `delay` | Computed sleep duration in seconds |
| `retry_after` | Server `Retry-After` value in seconds, or `null` |
| `error_type` | Kernel error class name (e.g. `RateLimitError`) |
| `error_message` | Sanitized error description |

> `error_message` is passed through `redact_sensitive_text()` before emission — tokens and credentials are never leaked into events.

## Observability

The provider emits three event types via the Amplifier hook system:

### `llm:request`

Emitted immediately before the SDK call.

| Field | Type | Description |
| --- | --- | --- |
| `provider` | string | `"github-copilot"` |
| `model` | string | Model ID used for this request |
| `message_count` | int | Number of messages in the conversation |
| `tool_count` | int | Number of tools available |
| `streaming` | bool | Whether streaming is enabled (default: `true`) |
| `timeout` | float | Request timeout in seconds |

### `llm:response`

Emitted after the SDK call completes (success or error).

| Field | Type | Description |
| --- | --- | --- |
| `provider` | string | `"github-copilot"` |
| `model` | string | Model ID |
| `status` | string | `"ok"` or `"error"` |
| `duration_ms` | int | Wall-clock time in milliseconds |
| `usage` | object | `{"input": int, "output": int}` token counts |
| `finish_reason` | string | `"stop"`, `"tool_calls"`, `"length"`, `"content_filter"`, `"end_turn"` |
| `content_blocks` | int | Number of content blocks in the response |
| `tool_calls` | int | Number of structured tool calls returned |
| `sdk_session_id` | string (optional) | Copilot SDK session ID for log correlation |
| `sdk_pid` | string (optional) | SDK process identifier for log correlation |

On error: `status`, `error_type`, `error_message` (redacted), `duration_ms`.

### `provider:retry`

See [Retry Events](#retry-events) above.

## Features

- Streaming support (always on; `llm:request` event reflects this)
- Tool use (function calling)
- Extended thinking (on supported models)
- Vision capabilities (on supported models)
- Token counting and management
- Prompt injection prevention — role-marker sequences (`[USER]`, `[SYSTEM]`, etc.) in user content and tool call IDs are escaped before the request reaches the SDK
- Tool sequence repair — orphaned tool calls are automatically repaired with synthetic results before LLM submission (see [Tool Sequence Repair](#tool-sequence-repair))
- All log output and observability events pass through secret redaction (tokens, Bearer headers, GitHub token formats, API keys, JWTs, PEM blocks)
- Raw payload logging — full SDK request/response capture for deep debugging (see [Raw Payload Logging](#raw-payload-logging))

## Contract

| Field | Value |
| --- | --- |
| **Module Type** | Provider |
| **Module ID** | `provider-github-copilot` |
| **Provider Name** | `github-copilot` |
| **Mount Point** | `providers` |
| **Entry Point** | `amplifier_module_provider_github_copilot:mount` |
| **Source URI** | `git+https://github.com/microsoft/amplifier-module-provider-github-copilot@main` |

## Architecture

The provider uses a singleton SDK client shared across all instances, with ephemeral sessions created per `complete()` call and destroyed after each request. Tool execution remains the orchestrator's responsibility — the provider never executes tools directly.

For module structure, design decisions, and contract index see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Graceful Error Recovery

The provider translates all SDK errors to typed kernel errors before they reach the caller. Each `complete()` call uses an independent session — no state accumulates between requests. The shared client and disk model cache persist across requests by design.

On `list_models()` failure, the provider falls back to a disk cache (24-hour TTL) before raising `ProviderUnavailableError`.

## Tool Sequence Repair

The provider automatically detects and repairs incomplete tool call sequences before sending the request to the LLM.

**The Problem:** If a conversation history contains a tool call from the assistant that has no corresponding tool result (due to context compaction bugs, parsing errors, or state corruption), the LLM receives an incoherent message history and may produce confused or repetitive responses. The missing result is invisible to the caller.

**The Solution:** Before prompt extraction, the provider scans assistant messages for tool call blocks without matching tool results. For each unmatched call, a synthetic tool-result message is inserted immediately after the offending assistant message. The LLM receives a coherent history and can acknowledge the gap and continue.

**What happens:**

1. Orphaned tool calls are detected (by `tool_call_id` set-difference)
2. A synthetic user message containing a `tool_result` block is inserted after each offending assistant message
3. One `WARNING` is logged per repair event with the count of repaired calls
4. Prompt extraction proceeds on the repaired message list; the original request is not mutated

**Synthetic result content:**
```
Tool result unavailable — the result for this tool call was lost. Please acknowledge this and continue.
```

**Example:**
```python
# Incoming messages (tool result missing)
messages = [
    {"role": "user",      "content": "Search for Python"},
    {"role": "assistant", "content": [{"type": "tool_call", "tool_call_id": "call-abc", "tool_name": "search"}]},
    # MISSING: tool_result for call-abc
    {"role": "user",      "content": "What did you find?"}
]

# After repair, the assistant message is followed by a synthetic result:
# {"role": "user", "content": [{"type": "tool_result", "tool_call_id": "call-abc",
#                                "output": "Tool result unavailable — ..."}]}
```

**Observability:** Repairs are logged as `WARNING` via the module logger. Monitor for `"Malformed tool sequence repaired"` log lines to detect upstream context management issues.

**Security:** `tool_call_id` values are sanitized through the same injection-prevention pipeline as user content before they are interpolated into the prompt. Role-marker sequences such as `[SYSTEM]` in a crafted ID are escaped automatically.

## Fake Tool Call Detection

LLMs occasionally emit tool calls as plain text instead of using the structured calling mechanism. The provider detects and automatically corrects this before returning a response.

Detection only fires when the request included tools and the response contains no structured tool calls. Up to 2 correction attempts are made; if the model still does not use structured tool calls, the last response is returned as-is.

## Environment Variables

| Variable | Description |
| --- | --- |
| `COPILOT_AGENT_TOKEN` | GitHub token — Copilot agent mode (highest priority) |
| `COPILOT_GITHUB_TOKEN` | GitHub token — recommended for direct use |
| `GH_TOKEN` | GitHub token — GitHub CLI compatible |
| `GITHUB_TOKEN` | GitHub token — GitHub Actions compatible |
| `COPILOT_SDK_LOG_LEVEL` | SDK log verbosity: `none`, `error`, `warning`, `info` (default), `debug`, `all` |

> **Warning:** `debug` and `all` produce high-volume output including sensitive conversation data. Use only for targeted SDK debugging.

## Development

### Setup

```bash
cd amplifier-module-provider-github-copilot

# Install dependencies (using UV)
uv sync --extra dev

# Or using pip
pip install -e ".[dev]"
```

### Testing

```bash
make test          # Run unit tests (excludes live API calls)
make live          # Run live integration tests (requires GITHUB_TOKEN)
make coverage      # Run with branch coverage report
make check         # Full check (lint + test)
make smoke         # Quick E2E smoke test (seconds)
```

### Live Integration Tests

Live tests make real API calls and require valid GitHub Copilot authentication:

```bash
export GITHUB_TOKEN=$(gh auth token)
make live
```

Or run directly:

```bash
python -m pytest tests/ -m live -v --tb=short
```

On Windows PowerShell:

```powershell
$env:GITHUB_TOKEN = (gh auth token)
python -m pytest tests/ -m live -v --tb=short
```

## Project Status

**Experimental.** Breaking changes may occur without deprecation notice. For questions open a [Discussion](https://github.com/microsoft/amplifier-module-provider-github-copilot/discussions); for bugs open an [Issue](https://github.com/microsoft/amplifier-module-provider-github-copilot/issues).

## Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| `Copilot SDK not installed` | Provider module not installed | Run `amplifier provider install github-copilot` |
| `Not authenticated to GitHub Copilot` | Token not set | **Linux/macOS:** `export GITHUB_TOKEN=$(gh auth token)` **Windows:** `$env:GITHUB_TOKEN = (gh auth token)` |
| `gh: command not found` | GitHub CLI missing | [Install gh CLI](https://cli.github.com/) |
| Stale or wrong model list | Cached models | Delete `%LOCALAPPDATA%\amplifier\provider-github-copilot\models_cache.json` (Windows), `~/Library/Caches/amplifier/provider-github-copilot/models_cache.json` (macOS), or `~/.cache/amplifier/provider-github-copilot/models_cache.json` (Linux) |
| `Permission denied` on SDK binary | `uv` stripped execute bits | Provider auto-repairs on startup; if it fails, run `chmod +x <path-to-copilot-binary>` (Linux/macOS only) |

### Common Mistake

Running `amplifier init` before authentication:

**Linux/macOS:**
```bash
❌ amplifier init                         # Fails with auth error
✅ export GITHUB_TOKEN=$(gh auth token)   # Set token first
✅ amplifier provider install github-copilot
✅ amplifier init                         # Now works
```

**Windows PowerShell:**
```powershell
❌ amplifier init                              # Fails with auth error
✅ $env:GITHUB_TOKEN = (gh auth token)        # Set token first
✅ amplifier provider install github-copilot
✅ amplifier init                              # Now works
```

## Dependencies

- `amplifier-core` (provided by Amplifier runtime, not installed separately)
- `github-copilot-sdk>=0.2.0,<0.3.0`
- `pyyaml>=6.0`

> **Note:** `github-copilot-sdk` is installed automatically when you install or initialize
> the provider via Amplifier (`amplifier provider install github-copilot` or `amplifier init`).
> It is not bundled with the main `amplifier` package.

## Contributing

> [!NOTE]
> This project is not currently accepting external contributions, but we're actively working toward opening this up. We value community input and look forward to collaborating in the future. For now, feel free to fork and experiment!

Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [Contributor License Agreements](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
