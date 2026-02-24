# Amplifier GitHub Copilot Provider Module

> This module is created by HDMowri

GitHub Copilot SDK integration for Amplifier via Copilot CLI.

## Prerequisites

- **Python 3.11+**
- **GitHub Copilot subscription** — Active Business or Enterprise subscription
- **[UV](https://github.com/astral-sh/uv)** (optional) — Fast Python package manager (pip works too)

> **No Node.js required.** The Copilot SDK binary is bundled with the Python package
> and discovered automatically.

## Authentication

Set a GitHub token as an environment variable. The provider checks these in order:
`COPILOT_GITHUB_TOKEN`, `GH_TOKEN`, `GITHUB_TOKEN`.

### Option 1: Environment variable (recommended)

```bash
export GITHUB_TOKEN="ghp_your_token_here"
```

> **Tip:** Many developers already have `GITHUB_TOKEN` set from `gh` CLI usage —
> if so, you're already authenticated. No extra setup needed.

### Option 2: `amplifier init` setup wizard

```bash
amplifier init
# Select "GitHub Copilot" from the provider list
# Launches browser OAuth flow if no token is set
```

### Option 3: `gh` CLI bridge

```bash
export GITHUB_TOKEN=$(gh auth token)
```

One command to bridge your existing `gh` CLI authentication into Amplifier.

## Installation

GitHub Copilot is a well-known provider — `amplifier init` handles everything:

```bash
# Interactive setup — select Copilot from the provider list
amplifier init

# Non-interactive — auto-detects GITHUB_TOKEN
amplifier init --yes
```

Or install manually:

```bash
amplifier provider install github-copilot
amplifier provider use github-copilot
```

Or reference it directly in a bundle:

```yaml
providers:
  - module: provider-github-copilot
    source: git+https://github.com/microsoft/amplifier-module-provider-github-copilot@main
    config:
      default_model: claude-sonnet-4
```

## Usage

```bash
# Interactive session
amplifier run -p github-copilot

# One-shot prompt
amplifier run -p github-copilot -m claude-sonnet-4 "Explain this codebase"

# Set as default provider locally (just you)
amplifier provider use github-copilot --local

# Set as default provider for the project (team)
amplifier provider use github-copilot --model claude-sonnet-4 --project
```

## Supported Models (18)

All 18 models available through your Copilot subscription are exposed at runtime:

**Anthropic:** `claude-haiku-4.5`, `claude-opus-4.5`, `claude-opus-4.6`, `claude-opus-4.6-1m`, `claude-opus-4.6-fast`, `claude-sonnet-4`, `claude-sonnet-4.5`

**OpenAI:** `gpt-4.1`, `gpt-5`, `gpt-5-mini`, `gpt-5.1`, `gpt-5.1-codex`, `gpt-5.1-codex-max`, `gpt-5.1-codex-mini`, `gpt-5.2`, `gpt-5.2-codex`, `gpt-5.3-codex`

**Google:** `gemini-3-pro-preview`

## Configuration

Works with sensible defaults out of the box. Default model is `claude-opus-4.5` with streaming enabled and a 1-hour request timeout.

All options can be set via provider config in your bundle or amplifier configuration. See the source code for the full list of configurable parameters.

Set `debug: true` for request/response event logging, or `debug: true, raw_debug: true` for full API I/O capture.

## Features

- Streaming support
- Tool use (function calling)
- Extended thinking (on supported models)
- Vision capabilities (on supported models)
- Token counting and management
- Message validation before API calls (defense in depth)

## Contract

| Field | Value |
| --- | --- |
| **Module Type** | Provider |
| **Module ID** | `provider-github-copilot` |
| **Provider Name** | `github-copilot` |
| **Entry Point** | `amplifier_module_provider_github_copilot:mount` |
| **Source URI** | `git+https://github.com/microsoft/amplifier-module-provider-github-copilot@main` |

## Graceful Error Recovery

The provider automatically detects and repairs incomplete tool call sequences in conversation history. If tool results are missing (due to context compaction, parsing errors, or state corruption), synthetic results are injected so the API accepts the request and the session continues.

Repairs are logged as warnings and emit `provider:tool_sequence_repaired` events for monitoring.

## Development

### Setup

```bash
cd amplifier-module-provider-github-copilot

# Install dependencies (using UV)
uv sync --dev

# Or using pip
pip install -e ".[dev]"
```

### Testing

```bash
make test          # Run tests
make coverage      # Run with coverage report
make sdk-assumptions  # Before upgrading SDK
make check         # Full check (lint + test)
```

### Live Integration Tests

Live tests require `RUN_LIVE_TESTS=1` and valid GitHub Copilot authentication:

```bash
RUN_LIVE_TESTS=1 python -m pytest tests/integration/ -v
```

On Windows PowerShell:

```powershell
$env:RUN_LIVE_TESTS="1"; python -m pytest tests/integration/ -v
```

## Dependencies

- `amplifier-core` (provided by Amplifier runtime, not installed separately)
- `github-copilot-sdk>=0.1.0,<0.2.0`
