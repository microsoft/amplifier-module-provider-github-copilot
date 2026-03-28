# Contract: Streaming

## Version
- **Current:** 1.4 (EventRouter-Extracted)
- **Module Reference:** 
  - `amplifier_module_provider_github_copilot/streaming.py` (accumulator, event translation)
  - `amplifier_module_provider_github_copilot/event_router.py` (SDK event routing)
- **Kernel Types:** `amplifier_core.message_models` (Pydantic, NOT content_models dataclass)
- **Status:** Specification
- **History:**
  - **1.4** — Extracted EventRouter from provider.py (separation of concerns)
  - **1.3** — Added completion guard, usage capture, session lifecycle anchors (bug fix documentation)
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
    ├─→ EventRouter (event_router.py)
    │   ├─→ BRIDGE: Translate → Accumulate → (internal callback)
    │   ├─→ CONSUME: Process internally (idle, usage, tool capture)
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

## Completion Guard

The accumulator MUST guard against late-arriving events after completion signals.

### MUST Constraints

**streaming-contract:completion:MUST:1**: Events after TURN_COMPLETE MUST be ignored (except USAGE_UPDATE).

**streaming-contract:completion:MUST:2**: Events after ERROR MUST be ignored (except USAGE_UPDATE).

**Rationale:** SDK may send spurious events after session completion (e.g., trailing deltas, duplicate turn_end). These MUST NOT corrupt accumulated state. However, USAGE_UPDATE events are special — see Usage Capture below.

---

## Usage Capture

The SDK sends `assistant.usage` events AFTER `session.idle` (turn completion). This creates a race condition where usage data arrives after we've marked the session complete.

### MUST Constraints

**streaming-contract:usage:MUST:1**: Provider MUST capture USAGE_UPDATE events even after completion.

**Bug Discovery:** Session 65131f78 showed `usage={input:0, output:0}` because the completion guard blocked the usage event that arrived after TURN_COMPLETE.

**Implementation:**
```python
def add(self, event: DomainEvent) -> None:
    # USAGE_UPDATE bypasses completion guard
    if event.type == DomainEventType.USAGE_UPDATE:
        self.usage = event.data
        return
    
    # All other events blocked after completion
    if self.is_complete:
        return
    # ... process event
```

---

## Session Lifecycle Configuration

Provider MUST validate session lifecycle configuration at load time to prevent silent runtime failures.

### MUST Constraints

**streaming-contract:SessionLifecycle:MUST:1**: Provider MUST raise `ConfigurationError` if `session_lifecycle.idle_events` is empty or missing.

**Rationale:** If `idle_events` is empty, `is_idle_event()` always returns False → session never completes → infinite hang with no error message. Developers need loud failures at startup, not 4-minute debugging sessions.

**Bug Discovery:** Session hung for 4+ minutes because `load_event_config` returned empty sets, and `is_idle_event()` always returned False.

**Implementation:**
```python
# In load_event_config()
if not idle_event_types:
    raise ConfigurationError(
        "session_lifecycle.idle_events is empty or missing in events.yaml. "
        "Provider cannot detect session completion without this configuration.",
        provider="github-copilot",
    )
```

---

## Internal Streaming (Provider Implementation)

The provider MAY implement internal streaming callbacks for real-time UI updates. This is NOT part of the kernel protocol.

### Progressive Streaming Events

For real-time UI updates, the provider SHOULD emit `llm:content_block` events as content arrives:

**SHOULD-1:** Provider SHOULD emit `llm:content_block` events for each content delta when coordinator has hooks.

**SHOULD-2:** Provider SHOULD emit events asynchronously (fire-and-forget) to avoid blocking the SDK event handler.

**SHOULD-3:** Provider SHOULD track pending emit tasks and clean up on provider close.

**SHOULD-4:** Provider SHOULD gracefully skip emission when no coordinator or no hooks available.

### Event Schema

```python
await coordinator.hooks.emit(
    "llm:content_block",
    {
        "provider": "github-copilot",
        "content": TextContent | ThinkingContent | ToolCallContent,
    },
)
```

### Implementation Pattern

```python
def _emit_streaming_content(
    self,
    content: TextContent | ThinkingContent | ToolCallContent,
) -> None:
    """Emit streaming content for real-time UI updates.
    
    Fire-and-forget pattern: creates async task, doesn't block.
    Contract: streaming-contract:ProgressiveStreaming:SHOULD:1-4
    """
    if not self._coordinator or not hasattr(self._coordinator, "hooks"):
        return
    
    try:
        loop = asyncio.get_running_loop()
        task = loop.create_task(
            self._emit_content_async(content),
            name=f"emit_content_{id(content)}",
        )
        self._pending_emit_tasks.add(task)
        task.add_done_callback(self._pending_emit_tasks.discard)
    except RuntimeError:
        pass  # No running loop - skip emission
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

### Finish Reason Normalization

The `finish_reason` field MUST be normalized before returning to the orchestrator:

| Condition | finish_reason | Rationale |
|-----------|---------------|-----------|
| Tool calls present | `"tool_calls"` | Orchestrator must execute tools (ALWAYS overrides SDK) |
| No tool calls, SDK sent finish_reason | preserve SDK value | Normal completion with SDK-provided reason |
| No tool calls, no SDK finish_reason | `"end_turn"` | Default for text-only responses |
| Error occurred | `"error"` | Error path |

**MUST-5:** Provider MUST set `finish_reason="tool_calls"` when `tool_calls` is non-empty, **regardless of what the SDK sent**.

**Rationale:** The SDK may send `TURN_COMPLETE` with `finish_reason="stop"` even when there are tool_calls (e.g., in the deny+capture flow where the SDK completes normally after tool denial). The orchestrator relies on `finish_reason="tool_calls"` (per amplifier-core proto) to know that it should execute the captured tools and continue the agent loop. Returning "stop" causes premature exit to interactive mode.

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
    
    tool_calls = [...]  # Convert to ToolCall list
    for tc in accumulator.tool_calls:
        content_blocks.append(tc)  # Already ToolCall
    
    # CRITICAL: finish_reason normalization
    # tool_calls ALWAYS override SDK finish_reason
    if tool_calls:
        finish_reason = "tool_calls"  # Override any SDK value (amplifier-core canonical)
    elif not accumulator.finish_reason:
        finish_reason = "end_turn"  # Default for text-only
    else:
        finish_reason = accumulator.finish_reason  # Preserve SDK value
    
    return ChatResponse(
        content=content_blocks,
        tool_calls=tool_calls,
        usage=accumulator.usage,
        finish_reason=finish_reason,
    )
```

---

## StreamingChatResponse

The `StreamingChatResponse` extends `ChatResponse` to support real-time streaming UI.

### Dual-Type System

| Field | Purpose | Types |
|-------|---------|-------|
| `content` | Context persistence | `TextBlock`, `ThinkingBlock`, `ToolCall` (message_models, Pydantic) |
| `content_blocks` | Streaming UI events | `TextContent`, `ThinkingContent`, `ToolCallContent` (content_models, dataclass) |

Both fields represent the same content but serve different consumers:
- `content` → stored in conversation context by orchestrator
- `content_blocks` → consumed by streaming orchestrator for UI events, then discarded

### MUST Constraints

1. **MUST** extend `ChatResponse` (kernel compatibility)
2. **MUST** populate `content_blocks` with content_models types
3. **MUST** populate `content` with message_models types (per existing contract)
4. **MUST** set `content_blocks` to `None` when no content (not empty list)

### Definition

```python
from amplifier_core import ChatResponse, TextContent, ThinkingContent, ToolCallContent


class StreamingChatResponse(ChatResponse):
    """ChatResponse with content_blocks for streaming UI compatibility."""
    
    content_blocks: list[TextContent | ThinkingContent | ToolCallContent] | None = None
    text: str | None = None
```

---

## Test Anchors

| Anchor | Clause |
|--------|--------|
| `streaming-contract:StreamingResponse:MUST:1` | Extends ChatResponse |
| `streaming-contract:StreamingResponse:MUST:2` | content_blocks uses content_models types |
| `streaming-contract:StreamingResponse:MUST:3` | content uses message_models types |
| `streaming-contract:StreamingResponse:MUST:4` | content_blocks is None when empty |
| `streaming-contract:ContentTypes:MUST:1` | Uses kernel content types |
| `streaming-contract:Accumulation:MUST:1` | Deltas accumulated in order |
| `streaming-contract:Accumulation:MUST:2` | Block boundaries maintained |
| `streaming-contract:ToolCapture:MUST:1` | Tool calls captured |
| `streaming-contract:ToolCapture:MUST:2` | Tool calls in final response |
| `streaming-contract:FinishReason:MUST:5` | finish_reason="tool_calls" when tool_calls present |
| `streaming-contract:Response:MUST:1` | Final response uses kernel types |
| `streaming-contract:completion:MUST:1` | Events after TURN_COMPLETE are ignored (except usage) |
| `streaming-contract:completion:MUST:2` | Events after ERROR are ignored (except usage) |
| `streaming-contract:usage:MUST:1` | Usage events captured even after completion |
| `streaming-contract:SessionLifecycle:MUST:1` | Provider validates idle_events config at load time |
| `streaming-contract:ProgressiveStreaming:SHOULD:1` | Emit llm:content_block events |
| `streaming-contract:ProgressiveStreaming:SHOULD:2` | Async fire-and-forget emission |
| `streaming-contract:ProgressiveStreaming:SHOULD:3` | Track and clean up emit tasks |
| `streaming-contract:ProgressiveStreaming:SHOULD:4` | Graceful skip when no coordinator |

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
