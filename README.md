# Amplifier GitHub Copilot Provider Module

Enables GitHub Enterprise Subscription users to leverage the Copilot SDK as a provider and access available LLM models through Amplifier

## Prerequisites

- **Python 3.11+**
- **GitHub Copilot subscription** — Individual, Business, or Enterprise
- **[UV](https://github.com/astral-sh/uv)** (optional) — Fast Python package manager (pip works too)

> **No Node.js required.** The Copilot SDK binary is bundled with the Python package
> and discovered automatically.

## Authentication

Set a GitHub token as an environment variable. The provider checks these in order:
`COPILOT_GITHUB_TOKEN`, `GH_TOKEN`, `GITHUB_TOKEN`.

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

Reference the provider directly in a bundle:

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

## Supported Models (18)

All 18 models available through your Copilot subscription are exposed at runtime:

**Anthropic:** `claude-haiku-4.5`, `claude-opus-4.5`, `claude-opus-4.6`, `claude-opus-4.6-1m`, `claude-sonnet-4`, `claude-sonnet-4.5`, `claude-sonnet-4.6`

**OpenAI:** `gpt-4.1`, `gpt-5-mini`, `gpt-5.1`, `gpt-5.1-codex`, `gpt-5.1-codex-max`, `gpt-5.1-codex-mini`, `gpt-5.2`, `gpt-5.2-codex`, `gpt-5.3-codex`, `gpt-5.4`

**Google:** `gemini-3-pro-preview`

> **Tip:** Want intelligent routing across models? Use the [Routing Matrix bundle](https://github.com/microsoft/amplifier-bundle-routing-matrix) to route prompts based on task type, cost, or latency.

## Configuration

Works with sensible defaults out of the box. Default model is `claude-opus-4.5` with streaming enabled and a 1-hour request timeout.

All options can be set via provider config in your bundle or amplifier configuration. See the source code for the full list of configurable parameters.

Set `raw: true` to include raw API request/response payloads in `llm:request` and `llm:response` events.

> **Security Warning:** The `raw: true` setting exposes full API payloads including potentially sensitive data. Never enable in production environments. Use only for local debugging with non-sensitive test data.

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
uv sync --extra dev

# Or using pip
pip install -e ".[dev]"
```

### Testing

```bash
make test          # Run unit tests (excludes live API calls)
make live          # Run live integration tests (requires GITHUB_TOKEN)
make coverage      # Run with branch coverage report
make check         # Full check (lint + type-check + test)
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

This is an **experimental project** exploring integration between the Amplifier framework and the GitHub Copilot CLI SDK. We are sharing it openly to enable community learning and experimentation.

As an experimental project:

- Response times may vary
- Breaking changes may occur without deprecation periods
- Features may be added, changed, or removed based on learnings

For questions, feel free to open a [GitHub Discussion](../../discussions). For bug reports, open a [GitHub Issue](../../issues) with reproduction steps.

## Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| `Copilot SDK not installed` | Provider module not installed | Run `amplifier provider install github-copilot` |
| `Not authenticated to GitHub Copilot` | Token not set | **Linux/macOS:** `export GITHUB_TOKEN=$(gh auth token)` **Windows:** `$env:GITHUB_TOKEN = (gh auth token)` |
| `gh: command not found` | GitHub CLI missing | [Install gh CLI](https://cli.github.com/) |

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

> **Note:** The `github-copilot-sdk` is installed automatically when you run 
> `amplifier provider install github-copilot`. It is NOT bundled with the main 
> `amplifier` package.

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
