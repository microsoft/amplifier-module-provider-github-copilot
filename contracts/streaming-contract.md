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
    ToolCallBlock,  # For ChatResponse.content
)
```

> **IMPORTANT:** The kernel has TWO content type systems:
> - `content_models`: TextContent, ThinkingContent (dataclass) — for internal kernel use
> - `message_models`: TextBlock, ThinkingBlock, ToolCallBlock (Pydantic) — for ChatResponse.content
>
> Providers MUST use message_models types for `ChatResponse.content`, which is typed as
> `list[ContentBlockUnion]` — a discriminated union on the `type` field (message_models.py:266).
>
> **CRITICAL:** `ToolCallBlock` (field: `input: dict`, `type: Literal["tool_call"]`) is the
> correct type for `content`. `ToolCall` (field: `arguments: dict`) is used ONLY in
> `ChatResponse.tool_calls: list[ToolCall]`. These are separate fields with different purposes:
> - `content` — context persistence: what the model said (including tool requests as ToolCallBlock)
> - `tool_calls` — execution signal: which tools the orchestrator should execute (ToolCall)

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
    signature: str | None = None  # Opaque extended thinking signature
```

**streaming-contract:ThinkingBlock:MUST:1**: Provider MUST preserve `reasoning_opaque` from SDK events as `ThinkingBlock.signature`.

**Rationale:** Anthropic models send encrypted extended thinking data in `reasoning_opaque` which must be returned verbatim in subsequent turns for multi-turn extended thinking to work. The kernel's `ThinkingBlock.signature` field maps directly to the SDK's `reasoning_opaque`. GPT models do not use this field.

### ToolCallBlock (Pydantic) — for ChatResponse.content
```python
class ToolCallBlock(BaseModel):
    type: Literal["tool_call"] = "tool_call"  # Discriminator for ContentBlockUnion
    id: str
    name: str
    input: dict[str, Any]  # NOTE: field is 'input', not 'arguments'
```

**streaming-contract:ToolCallBlock:MUST:1**: Provider MUST append a `ToolCallBlock` to
`ChatResponse.content` for each tool call the model requests. This enables multi-turn
context reconstruction — the assistant's tool call requests must be visible in `content`
for downstream consumers that rebuild conversation history from content blocks.

### ToolCall (Pydantic) — for ChatResponse.tool_calls
```python
class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]  # NOTE: field is 'arguments', not 'input'
```

**Note:** `ToolCall` is NOT in `ContentBlockUnion`. It belongs ONLY in
`ChatResponse.tool_calls: list[ToolCall]` — the orchestrator's execution signal.

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
        ├─→ content: [TextBlock, ThinkingBlock, ToolCallBlock...]  ← ContentBlockUnion
        ├─→ tool_calls: [ToolCall...]  ← execution signal
        └─→ usage: {token counts}
```

---

## Content Accumulation

### MUST Constraints

1. **MUST** accumulate text deltas in order
2. **MUST** use kernel message types (`TextBlock`, `ThinkingBlock`, `ToolCallBlock`) in `content`;
   `ToolCall` goes ONLY in `tool_calls` — NOT in `content`
3. **MUST** maintain block boundaries
4. **MUST** handle out-of-order deltas gracefully
5. **MUST NOT** lose deltas during accumulation
6. **MUST NOT** define custom content types

**streaming-contract:Accumulation:MUST:3**: Empty text deltas (text == "") MUST NOT be added to `_ordered_blocks`. The GitHub Copilot SDK interleaves empty `assistant.streaming_delta` events between consecutive `assistant.reasoning_delta` (thinking) events. Without this guard, each empty text creates a text-type block entry that fragments thinking consolidation — producing one `ThinkingContent` in `content_blocks` per thinking token (up to 47 fragmented thinking boxes in production).

**Rationale:** The loop-streaming orchestrator emits one `content_block:start` + `content_block:end` pair per entry in `response.content_blocks`. If `content_blocks` has 47 `ThinkingContent` entries (one per reasoning delta token), the CLI renders 47 `🧠 Thinking...` boxes. The fix: only append to `_ordered_blocks` when `text` is non-empty, so consecutive thinking deltas consolidate into one block.

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

**streaming-contract:usage:MUST:2**: When the SDK `assistant.usage` event includes `cache_read_tokens` or `cache_write_tokens`, the provider MUST extract and forward them to the kernel `Usage` object as `cache_read_tokens: int | None` and `cache_write_tokens: int | None` respectively.

**Rationale:** The kernel `Usage` model includes optional `cache_read_tokens` and `cache_write_tokens` fields (Pydantic `int | None`, default `None`). The SDK populates `cacheReadTokens` in the `assistant.usage` event when the upstream LLM provider's (e.g., Anthropic/Claude) prompt cache is hit. If these fields are dropped, Amplifier session analytics cannot distinguish cached from uncached token cost — masking significant cost differences (cache-read tokens bill at 0.10× the base rate). The SDK schema (`session_events.Data`) defines both `cache_read_tokens` and `cache_write_tokens` as `float | None`; `cache_write_tokens` is not currently populated by the SDK even when a cache write occurs, but the provider MUST extract it when present so the implementation is correct as soon as the SDK populates it. `None` is semantically distinct from `0`: `None` means the field was not reported; `0` means the SDK reported a confirmed zero.

**streaming-contract:usage:MUST:3**: The provider MUST set `Usage.input_tokens` to the **fresh (uncacheable) token count only** — not the SDK's billing total. When cache fields are present, `input_tokens = sdk_input_tokens - cache_read_tokens - cache_write_tokens`. `total_tokens` MUST equal `input_tokens + output_tokens`.

**Rationale:** The Copilot SDK's `assistant.usage` event reports `input_tokens` as the billing total (`fresh + cache_read + cache_write`). The kernel `Usage.input_tokens` convention — established by the Anthropic provider and consumed by the Amplifier streaming UI — defines `input_tokens` as the fresh/uncacheable portion only. The streaming UI computes the display total as `input_tokens + cache_read + cache_write`, so if the provider passes the SDK's billing total, the UI double-counts the cache buckets. Subtracting both `cache_read` and `cache_write` means the streaming UI display exactly recovers the SDK billing total: `fresh + cache_read + cache_write = sdk_input_tokens`.

**Evidence:** Streaming UI source (`amplifier-module-hooks-streaming-ui/__init__.py`):
```python
# When caching is active, input_tokens is just the uncacheable portion
total_input = input_tokens + cache_read + cache_create
```

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

**SHOULD-5:** Provider SHOULD NOT emit thinking deltas progressively per-token. Per-token `ThinkingContent` emission causes the CLI to render one thinking box per streaming chunk (hundreds of micro-boxes). Thinking emission granularity is controlled by `streaming_emission.thinking_content_types` in `events.yaml`. Setting this to an empty list suppresses per-delta thinking emission; the full consolidated `ThinkingContent` block is still present in the final `StreamingChatResponse.content_blocks`.

**Rationale for SHOULD-5:** The Amplifier CLI renders each `llm:content_block` hook event as a discrete UI element. Text deltas are appended inline; thinking blocks are rendered as separate collapsible boxes. Emitting one `ThinkingContent` per token produces hundreds of identical empty boxes during a thinking phase.

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
| No tool calls, no SDK finish_reason | `"stop"` | Default for text-only responses (per amplifier-core proto, not "end_turn") |
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
        finish_reason = "stop"  # Default for text-only (per amplifier-core proto: "stop", not "end_turn")
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
| `streaming-contract:ThinkingBlock:MUST:1` | Provider MUST preserve reasoning_opaque as ThinkingBlock.signature |
| `streaming-contract:ToolCallBlock:MUST:1` | Provider MUST append ToolCallBlock to content for each tool call |
| `streaming-contract:Accumulation:MUST:1` | Deltas accumulated in order |
| `streaming-contract:Accumulation:MUST:2` | Block boundaries maintained |
| `streaming-contract:ToolCapture:MUST:1` | Tool calls captured |
| `streaming-contract:ToolCapture:MUST:2` | Tool calls in final response |
| `streaming-contract:FinishReason:MUST:5` | finish_reason="tool_calls" when tool_calls present |
| `streaming-contract:Response:MUST:1` | Final response uses kernel types |
| `streaming-contract:completion:MUST:1` | Events after TURN_COMPLETE are ignored (except usage) |
| `streaming-contract:completion:MUST:2` | Events after ERROR are ignored (except usage) |
| `streaming-contract:usage:MUST:1` | Usage events captured even after completion |
| `streaming-contract:usage:MUST:2` | Cache token fields forwarded to kernel Usage when present |
| `streaming-contract:usage:MUST:3` | `Usage.input_tokens` = fresh only (`sdk_input - cache_read - cache_write`); `total_tokens = input + output` |
| `streaming-contract:SessionLifecycle:MUST:1` | Provider validates idle_events config at load time |
| `streaming-contract:ProgressiveStreaming:SHOULD:1` | Emit llm:content_block events |
| `streaming-contract:ProgressiveStreaming:SHOULD:2` | Async fire-and-forget emission |
| `streaming-contract:ProgressiveStreaming:SHOULD:3` | Track and clean up emit tasks |
| `streaming-contract:ProgressiveStreaming:SHOULD:4` | Graceful skip when no coordinator |
| `streaming-contract:ProgressiveStreaming:SHOULD:5` | Do not emit thinking deltas per-token; control via events.yaml |
| `streaming-contract:Accumulation:MUST:3` | Empty text deltas MUST NOT disrupt thinking block consolidation |

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
