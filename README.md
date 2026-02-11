# Amplifier GitHub Copilot Provider Module

GitHub Copilot SDK integration for Amplifier via Copilot CLI.

## Prerequisites

- **Python 3.11+**
- **Node.js 18+** — Required to install the Copilot CLI
- **GitHub Copilot subscription** — Active Business or Enterprise subscription
- **[UV](https://github.com/astral-sh/uv)** (optional) — Fast Python package manager (pip works too)

### Installing Copilot CLI

The Copilot CLI is a Node.js binary that the Python SDK controls via JSON-RPC.
Both the CLI and the SDK are required.

```bash
# Install Copilot CLI (requires Node.js/npm)
npm install -g @github/copilot

# Verify installation
copilot --version
```

### Authentication

You must be authenticated to GitHub Copilot:

```bash
copilot auth login
```

## Installation

Register the module, install its dependencies, and set it as your active provider:

```bash
amplifier module add provider-github-copilot \
  --source git+https://github.com/microsoft/amplifier-module-provider-github-copilot@main

amplifier provider install github-copilot

amplifier provider use github-copilot
```

> **Note:** The `provider install` step is required to install the module's Python
> dependencies (including the GitHub Copilot SDK) into the Amplifier environment.
> The built-in providers skip this step because they are pre-installed during
> `amplifier init`.

Or reference it directly in a bundle (no separate install needed):

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

### Provider Preferences in Bundles

Use provider preferences for ordered model fallback:

```yaml
provider_preferences:
  - provider: github-copilot
    model: claude-sonnet-4
  - provider: github-copilot
    model: gpt-*
```

## Supported Models

All models available through your Copilot subscription are exposed at runtime. Examples:

- `claude-opus-4.5`, `claude-sonnet-4`, `claude-haiku-4.5`
- `gpt-5`, `gpt-5.1`, `gpt-5.1-codex`
- `gemini-3-pro-preview`

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

| | |
|---|---|
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

## Dependencies

- `amplifier-core` (provided by Amplifier runtime, not installed separately)
- `github-copilot-sdk>=0.1.0,<0.2.0`


