# Contract: Streaming

## Version
- **Current:** 1.2 (Anchor-Prefix-Corrected)
- **Module Reference:** amplifier_module_provider_github_copilot/streaming.py
- **Kernel Types:** `amplifier_core.message_models` (Pydantic, NOT content_models dataclass)
- **Status:** Specification
- **History:**
  - **1.2** — Fixed anchor prefix from `streaming:` to `streaming-contract:`, added missing anchors
  - **1.1** — Expert panel verified: ChatResponse.content uses message_models types

---

## Overview

This contract defines how streaming events are accumulated into a complete response. The streaming callback is provider-internal — the kernel protocol uses `**kwargs`, not a named callback parameter.

---

## Kernel Content Types

Use types from `amplifier_core.message_models` (Pydantic models, NOT content_models dataclasses):

```python
from amplifier_core import (
    TextBlock,
    ThinkingBlock,
    ToolCall,
)
```

> **IMPORTANT:** The kernel has TWO content type systems:
> - `content_models`: TextContent, ThinkingContent (dataclass) — for internal kernel use
> - `message_models`: TextBlock, ThinkingBlock (Pydantic) — for ChatResponse.content
>
> Providers MUST use message_models types because ChatResponse.content is typed as `list[TextBlock | ThinkingBlock | RedactedThinkingBlock | ToolCall]`.

### TextBlock (Pydantic)
```python
class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str
```

### ThinkingBlock (Pydantic)
```python
class ThinkingBlock(BaseModel):
    type: Literal["thinking"] = "thinking"
    thinking: str
```

### ToolCall (Pydantic)
```python
class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]
```

---

## Streaming Flow

```
SDK Event Stream
    │
    ├─→ Event Handler
    │   ├─→ BRIDGE: Translate → Accumulate → (internal callback)
    │   ├─→ CONSUME: Process internally
    │   └─→ DROP: Ignore
    │
    └─→ Final ChatResponse
        ├─→ content: [TextBlock, ThinkingBlock, ToolCall...]
        ├─→ tool_calls: [ToolCall...]
        └─→ usage: {token counts}
```

---

## Content Accumulation

### MUST Constraints

1. **MUST** accumulate text deltas in order
2. **MUST** use kernel message types (`TextBlock`, `ThinkingBlock`, `ToolCall`)
3. **MUST** maintain block boundaries
4. **MUST** handle out-of-order deltas gracefully
5. **MUST NOT** lose deltas during accumulation
6. **MUST NOT** define custom content types

### Accumulator State

```python
@dataclass
class StreamAccumulator:
    text_blocks: list[str]              # Accumulated text per block
    thinking_blocks: list[str]          # Accumulated thinking per block
    tool_calls: list[ToolCall]           # Captured tool calls
    usage: Usage | None                 # Token usage
    finish_reason: str                  # Final finish reason
```

---

## Internal Streaming (Provider Implementation)

The provider MAY implement internal streaming callbacks for real-time UI updates. This is NOT part of the kernel protocol.

```python
# Internal implementation detail — not protocol
async def _stream_completion(
    self,
    request: ChatRequest,
    on_event: Callable[[ContentBlock], None] | None = None,
) -> ChatResponse:
    """Provider-internal streaming implementation."""
```

---

## Tool Call Handling

### MUST Constraints

1. **MUST** capture tool calls from SDK events
2. **MUST NOT** execute tool calls (deny-destroy.md)
3. **MUST** return tool calls as `ToolCall` in response
4. **MUST** preserve tool call IDs for correlation

### Tool Call Accumulation

```python
def handle_tool_call_event(self, event: DomainEvent) -> None:
    tool_call = ToolCall(
        id=event.data["id"],
        name=event.data["name"],
        arguments=event.data["arguments"],
    )
    self.tool_calls.append(tool_call)
```

---

## Final Response Assembly

```python
from amplifier_core import TextBlock, ThinkingBlock, ToolCall

def assemble_response(accumulator: StreamAccumulator) -> ChatResponse:
    """
    Assemble final response from accumulated state.
    
    Uses kernel message types (Pydantic), not content_models dataclasses.
    """
    content_blocks: list[TextBlock | ThinkingBlock | ToolCall] = []
    
    for text in accumulator.text_blocks:
        if text:
            content_blocks.append(TextBlock(text=text))
    
    for thinking in accumulator.thinking_blocks:
        if thinking:
            content_blocks.append(ThinkingBlock(thinking=thinking))
    
    for tc in accumulator.tool_calls:
        content_blocks.append(tc)  # Already ToolCall
    
    return ChatResponse(
        content=content_blocks,
        tool_calls=[...],  # ToolCall list
        usage=accumulator.usage,
        finish_reason=accumulator.finish_reason,
    )
```

---

## Test Anchors

| Anchor | Clause |
|--------|--------|
| `streaming-contract:ContentTypes:MUST:1` | Uses kernel content types |
| `streaming-contract:Accumulation:MUST:1` | Deltas accumulated in order |
| `streaming-contract:Accumulation:MUST:2` | Block boundaries maintained |
| `streaming-contract:ToolCapture:MUST:1` | Tool calls captured |
| `streaming-contract:ToolCapture:MUST:2` | Tool calls in final response |
| `streaming-contract:Response:MUST:1` | Final response uses kernel types |
| `streaming-contract:completion:MUST:1` | First stream yields before completion |
| `streaming-contract:completion:MUST:2` | Completion includes all content |

---

## SDK Response Extraction

### MUST Constraints

1. **MUST** extract `.content` from SDK `Data` objects — NOT `str(Data(...))`
2. **MUST** unwrap `.data` wrapper first before checking `.content`
3. **MUST** handle dict responses for backward compatibility
4. **MUST** return empty string for None responses

See `contracts/sdk-response.md` for full extraction specification.

---

## Implementation Checklist

- [ ] Import message types from `amplifier_core` (TextBlock, ThinkingBlock, ToolCall)
- [ ] StreamAccumulator tracks all state
- [ ] Text deltas accumulated per block
- [ ] Thinking deltas accumulated per block (ThinkingBlock.thinking, not .text)
- [ ] Tool calls captured as ToolCall
- [ ] Final response uses kernel message types (Pydantic)
- [ ] No custom content types defined
- [ ] SDK response extraction uses `extract_response_content()`
