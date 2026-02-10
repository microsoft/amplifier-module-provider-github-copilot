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

## Purpose

Provides access to LLM models (Claude, GPT, Gemini) through your GitHub Copilot subscription as an LLM provider for Amplifier.

## Contract

**Module Type:** Provider
**Mount Point:** `providers`
**Entry Point:** `amplifier_module_provider_github_copilot:mount`

## Supported Models

The provider exposes all models available through your Copilot subscription. Model availability and context windows are fetched from the SDK at runtime. Examples:

- `claude-opus-4.5`, `claude-sonnet-4`, `claude-haiku-4.5`
- `gpt-5`, `gpt-5.1`, `gpt-5.1-codex`
- `gemini-3-pro-preview`

## Configuration

The provider works with sensible defaults and requires no configuration for basic use. The default model is `claude-opus-4.5` with streaming enabled and a 1-hour request timeout.

All options can be set via the provider config in your bundle or amplifier configuration. See the source code for the full list of configurable parameters.

Set `debug: true` for request/response event logging, or `debug: true, raw_debug: true` for full API I/O capture.

## Usage

```bash
# Start chat with Copilot SDK provider
amplifier chat --provider github-copilot

# Use with specific model
amplifier chat --provider github-copilot --model claude-sonnet-4
```

## Features

- Streaming support
- Tool use (function calling)
- Extended thinking (on supported models)
- Vision capabilities (on supported models)
- Token counting and management
- **Message validation** before API calls (defense in depth)

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

- `amplifier-core>=1.0.0`
- `github-copilot-sdk>=0.1.0,<0.2.0`


