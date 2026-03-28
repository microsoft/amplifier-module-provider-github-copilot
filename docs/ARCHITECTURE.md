# Architecture

GitHub Copilot provider for [Amplifier](https://github.com/microsoft/amplifier).

## Module Structure

```
amplifier_module_provider_github_copilot/
├── __init__.py           # Entry point: mount(), get_info()
├── _compat.py            # Compatibility layer: ConfigurationError fallback
├── _permissions.py       # Permission constants for SDK hooks
├── _platform.py          # Platform detection utilities
├── provider.py           # Provider class: list_models(), complete()
├── event_router.py       # SDK event routing (extracted from provider.py)
├── streaming.py          # SDK event → domain event translation
├── error_translation.py  # SDK error → kernel error mapping
├── config_loader.py      # YAML config loading + model fallbacks
├── models.py             # Data structures, Amplifier ↔ SDK translation
├── request_adapter.py    # ChatRequest → internal request conversion
├── observability.py      # Hook event emission, timing
├── tool_parsing.py       # Tool call extraction from response
├── fake_tool_detection.py # Detect/correct fake tool calls
├── model_cache.py        # SDK model list caching
├── security_redaction.py # Sensitive data redaction
├── config/               # YAML configuration files
│   ├── models.yaml       # Default model, provider metadata
│   ├── errors.yaml       # Error translation rules
│   ├── events.yaml       # Event classification
│   ├── retry.yaml        # Retry policy
│   ├── observability.yaml # Hook event names
│   ├── model_cache.yaml  # Model cache TTL settings
│   ├── sdk_protection.yaml # Tool capture, session management
│   └── fake-tool-detection.yaml # Fake tool detection patterns
└── sdk_adapter/          # SDK isolation layer (THE MEMBRANE)
    ├── __init__.py       # Public API exports (membrane boundary)
    ├── _imports.py       # SDK imports quarantine (ONLY SDK imports here)
    ├── _spec_utils.py    # SDK-independent spec utilities
    ├── client.py         # Session lifecycle (create, send, close)
    ├── types.py          # Domain types for boundary crossing
    ├── extract.py        # SDK event → domain type extraction
    ├── event_helpers.py  # Event classification and translation
    ├── tool_capture.py   # Tool capture handler (sdk-protection.md)
    └── model_translation.py # SDK ModelInfo → CopilotModelInfo
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

SDK errors are translated to `amplifier_core.llm_errors.*` types via config-driven pattern matching. `ConfigurationError` is the only custom exception (for config loading failures).

## Contracts

The `contracts/` directory contains behavioral specifications:

| Contract | Purpose |
|----------|---------|
| `provider-protocol.md` | Provider interface requirements (4 methods + 1 property) |
| `sdk-boundary.md` | SDK isolation rules (THE MEMBRANE) |
| `deny-destroy.md` | Deny + Destroy pattern (no SDK tool execution) |
| `sdk-protection.md` | Tool capture, deduplication, session cleanup |
| `sdk-response.md` | SDK response extraction rules |
| `error-hierarchy.md` | Error translation spec |
| `streaming-contract.md` | Streaming behavior + EventRouter |
| `event-vocabulary.md` | Domain events (BRIDGE, CONSUME, DROP) |
| `behaviors.md` | Retry policy, streaming timing policy |
| `observability.md` | Hook events, OTEL policy |
