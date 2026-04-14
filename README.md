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

### Alternative: Non-interactive

```bash
# Requires: GITHUB_TOKEN set AND provider installed
amplifier init --yes
```

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

> **Note:** Retry parameters, session timeouts, and event queue sizes use fixed defaults
> and cannot be overridden via bundle config. The request timeout defaults to 3600 s.

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

#### Retry Defaults

Retry parameters are fixed and cannot be changed via bundle config.

| Parameter | Default | Description |
| --- | --- | --- |
| `max_attempts` | `3` | Maximum retry attempts before surfacing the error to the caller |
| `base_delay_ms` | `1000` | Base delay in milliseconds (doubles each attempt) |
| `max_delay_ms` | `30000` | Delay cap in milliseconds before jitter is applied |
| `jitter_factor` | `0.1` | Jitter fraction applied as ± of the capped delay (0.0 – 1.0) |

#### Retry Events

A `provider:retry` event is emitted before each retry sleep:

| Field | Description |
| --- | --- |
| `provider` | Provider name (`"github-copilot"`) |
| `model` | Model being called |
| `attempt` | Current attempt number (1-based in event payload) |
| `max_retries` | Configured maximum attempts |
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
- Prompt injection prevention — role-marker sequences (`[USER]`, `[SYSTEM]`, etc.) in user content are escaped before the request reaches the SDK
- All log output and observability events pass through secret redaction (tokens, Bearer headers, GitHub token formats, API keys, JWTs, PEM blocks)

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

### SDK Client Singleton

All provider instances share a single SDK client. The singleton is created on first `mount()` and released when the last mounted instance is cleaned up, ensuring efficient shared resource management across concurrent sub-agents.

### Session Lifecycle

A fresh SDK session is created for each `complete()` call and torn down when the call returns. This provides clean, independent context for each request. The shared client and disk model cache persist across requests intentionally — these are not session state.

### Tool Isolation

Only tools explicitly passed by Amplifier's orchestrator in the `ChatRequest` are available within each session. Tool execution is the orchestrator's responsibility — the provider does not execute tools directly.

## Graceful Error Recovery

The provider translates all SDK errors to typed kernel errors before they reach the caller. Each `complete()` call uses an independent session — no state accumulates between requests. The shared client and disk model cache persist across requests by design.

On `list_models()` failure, the provider falls back to a disk cache (24-hour TTL) before raising `ProviderUnavailableError`.

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
