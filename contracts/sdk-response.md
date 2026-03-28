# SDK Response Contract

**Status**: Active
**Version**: 1.1 (Anchors-Added)

## Overview

This contract defines the expected shapes of SDK responses from `github-copilot-sdk` and the extraction requirements for the provider.

## Test Anchors

| Anchor | Clause |
|--------|--------|
| `sdk-response:extraction:MUST:1` | Handle None response |
| `sdk-response:extraction:MUST:2` | Unwrap `.data` wrapper |
| `sdk-response:extraction:MUST:3` | Extract `.content` attribute |
| `sdk-response:extraction:MUST:4` | Handle dict responses |
| `sdk-response:extraction:MUST:5` | Return empty string fallback |
| `sdk-response:extraction:MUST:6` | Never call str() on Data objects |
| `sdk-response:extraction:MUST:7` | Follow extraction priority order |
| `sdk-response:extraction:SHOULD:1` | Graceful handling edge cases |

## SDK Response Structure

### `send_and_wait()` Response

The SDK's `session.send_and_wait()` method returns a response object with the following structure:

```
Response
└── data: Data
    ├── content: str     # The actual text content (MUST extract this)
    ├── role: str        # "assistant"
    └── model: str       # e.g., "gpt-4o"
```

**Key Type**: `copilot.generated.session_events.Data`

### MUST Constraints

1. **MUST extract `.content` from Data objects** — NOT `str(Data(...))` which produces repr dump
2. **MUST handle nested `.data` wrapper** — SDK may return `response.data` containing the `Data` object
3. **MUST handle dict responses** — For backward compatibility with older SDK versions
4. **MUST return empty string for None** — Graceful handling of missing responses

### Extraction Priority

The `extract_response_content()` function MUST check in this order:

1. `response is None` → return empty string
2. `hasattr(response, "data")` → recurse to unwrap wrapper
3. `hasattr(response, "content")` → extract `.content` attribute (the Data object case)
4. `isinstance(response, dict)` → get `content` key
5. Fallback → return empty string

### Anti-Pattern: The Bug This Contract Prevents

```python
# WRONG - produces "Data(content='hello', role='assistant', ...)"
content = str(response_data) if response_data else ""

# CORRECT - produces "hello"
if hasattr(response_data, "content"):
    content = response_data.content
```

## Test Requirements

Tests MUST use realistic SDK response shapes (not bare strings):

```python
@dataclass
class MockData:
    content: str
    role: str = "assistant"
    model: str = "gpt-4o"

# Use in tests:
sdk_response.data = MockData(content="Actual response text")
```

## Version Compatibility

| SDK Version | Response Shape |
|-------------|----------------|
| 0.1.32+ | `Data` dataclass with `.content` |
| Future | May change — test fixtures document expected shape |

## Related Contracts

- `streaming-contract.md` — Event translation
- `provider-protocol.md` — `complete()` return type
