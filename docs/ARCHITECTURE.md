# Architecture

GitHub Copilot provider for [Amplifier](https://github.com/microsoft/amplifier).

## Module Structure

```
amplifier_module_provider_github_copilot/
├── provider.py          # Entry point: mount(), get_info(), list_models(), complete()
├── completion.py        # LLM call lifecycle, streaming
├── streaming.py         # SDK event → domain event translation
├── error_translation.py # SDK error → kernel error mapping
├── config_loader.py     # YAML config loading
├── models.py            # Data structures
├── config/              # YAML configuration files
│   ├── models.yaml      # Default model, provider metadata
│   ├── errors.yaml      # Error translation rules
│   ├── events.yaml      # Event classification
│   └── retry.yaml       # Retry policy
└── sdk_adapter/         # SDK isolation layer
    ├── _imports.py      # Only file with SDK imports
    ├── client.py        # Session lifecycle
    └── types.py         # Domain types for boundary crossing
```

## Key Design Decisions

### SDK Isolation

All `github-copilot-sdk` imports are quarantined in `sdk_adapter/_imports.py`. Domain code never imports SDK types directly. This isolates the codebase from SDK version changes.

### Config-Driven Behavior

Policy values live in YAML, not Python:
- Error mappings: `config/errors.yaml`
- Model defaults: `config/models.yaml`
- Retry policy: `config/retry.yaml`

### Ephemeral Sessions

Sessions are created per `complete()` call and destroyed after the first turn. No session reuse.

### Error Translation

SDK errors are translated to `amplifier_core.llm_errors.*` types via config-driven pattern matching. No custom exception classes.

## Contracts

The `contracts/` directory contains behavioral specifications:

| Contract | Purpose |
|----------|---------|
| `provider-protocol.md` | Provider interface requirements |
| `sdk-boundary.md` | SDK isolation rules |
| `error-hierarchy.md` | Error translation spec |
| `streaming-contract.md` | Streaming behavior |
