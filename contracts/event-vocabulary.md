# Contract: Event Vocabulary

## Version
- **Current:** 1.3
- **Module Reference:** amplifier_module_provider_github_copilot/streaming.py
- **Correction:** 2026-03-15 — Removed erroneous `src/` prefix
- **Amendment:** 2026-04-13 — Added Bridge:MUST:3 (BRIDGE event data flattening requirement)
- **Config:** amplifier_module_provider_github_copilot/config/events.yaml
- **Status:** Specification

---

## Overview

This contract defines the 5 stable domain events and how SDK events are classified into BRIDGE, CONSUME, or DROP categories. Event classification is config-driven.

---

## The Five Domain Events

| Event | Description | Source |
|-------|-------------|--------|
| `CONTENT_DELTA` | Text/thinking chunk from assistant | SDK assistant.message_delta, assistant.reasoning_delta |
| `TOOL_CALL` | Tool invocation request | SDK tool.call (captured, not executed) |
| `USAGE_UPDATE` | Token usage statistics | SDK assistant.usage |
| `TURN_COMPLETE` | Assistant turn finished | SDK assistant.message / session.idle |
| `SESSION_IDLE` | Reserved for future use | Not currently emitted |
| `ERROR` | Session error event | SDK error / exception |

**NOTE:** Thinking/reasoning content uses `CONTENT_DELTA` with `block_type="THINKING"` per events.yaml bridge mappings. This simplifies the domain model by treating thinking as a content type rather than a separate event category.

**Note:** `session.idle` triggers TURN_COMPLETE, not SESSION_IDLE.
The SESSION_IDLE enum value exists for potential future extensibility.

---

## Event Classification

### BRIDGE Events
Events translated to domain events and passed to Amplifier.

| SDK Event | Domain Event | Notes |
|-----------|--------------|-------|
| `assistant.message_delta` | `CONTENT_DELTA` | Text streaming (block_type=TEXT) |
| `assistant.reasoning_delta` | `CONTENT_DELTA` | Thinking streaming (block_type=THINKING) |
| `assistant.message` | `TURN_COMPLETE` | Final message |
| `session.idle` | `TURN_COMPLETE` | Turn finished (not SESSION_IDLE) |
| `assistant.usage` | `USAGE_UPDATE` | Token counts |

### CONSUME Events
Events processed internally but not forwarded.

| SDK Event | Action | Notes |
|-----------|--------|-------|
| `tool.call` | Capture tool request | Tool calls accumulated |
| `tool.result` | Internal state update | Not forwarded |
| `session.start` | Internal state update | Session lifecycle |
| `session.resume` | Internal state update | Session lifecycle |
| `system.notification` | Log internally | SDK v0.1.33 notification |
| `system_notification` | Log internally | Legacy underscore alias |

### DROP Events
Events ignored entirely. The authoritative list is `config/events.yaml`; this table shows representative examples.

**SDK version-skew rule:** The Copilot CLI binary may emit new `session.*` events before the Python SDK documents them. Any such event with no domain value MUST be added to the DROP list in `events.yaml`. It MUST NOT produce a warning log in production.

| SDK Event | Reason |
|-----------|--------|
| `debug.*` | Development only |
| `heartbeat` | Connection keepalive |
| `session.compaction.*` | Internal optimization |
| `session.custom_agents_updated` | SDK v0.2.1+ session-state notification; no domain value |
| `session.skills_loaded` | SDK SessionSkillsLoadedEvent; Amplifier does not host skills; no domain value |
| `system.message` | SDK SystemMessageEvent; carries system/developer prompt text; not forwarded |

---

## Domain Event Structure

```python
@dataclass
class DomainEvent:
    type: str  # One of the 6 domain events
    data: dict[str, Any]
    block_type: str | None = None  # e.g. "THINKING" for reasoning content_deltas
```

### CONTENT_DELTA

```python
# Normal text chunk
DomainEvent(
    type="CONTENT_DELTA",
    data={"text": "partial text...", "index": 0},
    block_type=None,
)

# Reasoning/thinking chunk
DomainEvent(
    type="CONTENT_DELTA",
    data={"text": "reasoning text...", "index": 0},
    block_type="THINKING",
)
```

### SESSION_IDLE

```python
DomainEvent(
    type="SESSION_IDLE",
    data={},
)
```

### ERROR

```python
DomainEvent(
    type="ERROR",
    data={"message": "error description"},
)
```

### TOOL_CALL

```python
DomainEvent(
    type="TOOL_CALL",
    data={
        "id": "call_123",
        "name": "read_file",
        "arguments": {"path": "file.py"},
    }
)
```

### USAGE_UPDATE

```python
DomainEvent(
    type="USAGE_UPDATE",
    data={
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
    }
)
```

### TURN_COMPLETE

```python
DomainEvent(
    type="TURN_COMPLETE",
    data={
        # Per amplifier-core proto: "stop", "tool_calls", "length", "content_filter"
        "finish_reason": "stop",  # or "tool_calls" for tool invocations
        "message_id": "msg_123",
    }
)
```

---

## BRIDGE Event Data Flattening

**event-vocabulary:Bridge:MUST:3**

For all BRIDGE events, `translate_event` MUST promote all SDK event data fields to the top level
of `DomainEvent.data`. The result MUST be a flat `dict[str, Any]` with no nested SDK objects under
the `"data"` key and no residual `"data"` sub-key. Non-`data` fields from the envelope (e.g.
`message_id`, `finish_reason`) are passed through as-is and are not subject to the serialisability
guarantee.

**Mechanism:** `_extract_event_data()` handles two shapes:

1. **Dict path** — `sdk_event["data"]` is a plain dict (legacy, test, or pre-extracted events):
   all sub-keys are merged into the result at the top level.

2. **Object path** — `sdk_event["data"]` is a `SessionEventData` object (real SDK path, after
   `extract_event_fields()` runs): `extract_event_fields()` is called recursively on the object
   and its fields are merged at the top level.

In both cases the raw `"data"` key is consumed and does not appear in `DomainEvent.data`.

**Rationale:**

- `StreamingAccumulator.add()` reads all fields via `event.data.get(key)` directly
  (e.g. `event.data.get("text", "")`, `event.data.get("finish_reason", "stop")`).
- `StreamingAccumulator.add()` stores `event.data` as `self.usage` for `USAGE_UPDATE` events;
  a non-flat dict would produce incorrect token counts.
- Any code that logs or serialises `DomainEvent` requires `DomainEvent.data` to be a plain
  `dict[str, Any]`. A retained `SessionEventData` object in the dict silently breaks
  `json.dumps()` with `TypeError`.

**Upstream evidence:** `EventRouter.__call__()` calls `self._queue.put_nowait(sdk_event)` with
the raw `SessionEvent` object. `extract_event_fields()` at `provider.py:L752` returns a dict
that includes the `"data": <SessionEventData>` key (not excluded by `_ENVELOPE_FIELDS`).
`_extract_event_data()` is the only barrier preventing that object from reaching
`DomainEvent.data`.

---

## Finish Reason Mapping

| SDK Reason | Domain Reason |
|------------|---------------|
| `end_turn` | `"stop"` |
| `stop` | `"stop"` |
| `tool_use` | `"tool_calls"` |
| `max_tokens` | `"length"` |
| `content_filter` | `"content_filter"` |
| (default) | `"stop"` |

**Note:** The SDK sends `tool_use` as the finish_reason when tool calls are made. The provider
normalizes this to `"tool_calls"` (amplifier-core proto value). See `events.yaml` finish_reason_map
and streaming-contract.md for details.

---

## Config Schema (events.yaml)

```yaml
version: "1.0"

event_classifications:
  bridge:
    - sdk_type: "assistant.message_delta"
      domain_type: "CONTENT_DELTA"
      block_type: "TEXT"
    
    - sdk_type: "assistant.reasoning_delta"
      domain_type: "CONTENT_DELTA"
      block_type: "THINKING"
    
    - sdk_type: "assistant.message"
      domain_type: "TURN_COMPLETE"
    
    - sdk_type: "session.idle"
      domain_type: "TURN_COMPLETE"
    
    - sdk_type: "assistant.usage"
      domain_type: "USAGE_UPDATE"
  
  consume:
    - "tool.call"
    - "tool.result"
    - "session.start"
    # NOTE: session.resume is DROP in events.yaml (not CONSUME)
    # See events.yaml for the complete authoritative list
  
  drop:
    - "debug.*"
    - "heartbeat"
    - "session.resume"
    - "session.compaction.*"

finish_reason_map:
  # Values MUST be lowercase to match amplifier-core proto:
  #   "stop", "tool_calls", "length", "content_filter"
  # SDK sends "tool_use" — mapped to domain "tool_calls"
  end_turn: stop
  stop: stop
  tool_use: tool_calls  # SDK sends tool_use; domain value is tool_calls
  max_tokens: length
  content_filter: content_filter
  "": stop  # SDK sends empty string for normal completion
  _default: stop  # Normal completion when unspecified
```

> **Note:** The Config Schema above shows the minimal structure. See `config/events.yaml` for the
> complete authoritative classification list including all legacy aliases and DROP patterns.

---

## Test Anchors

### Events

| Anchor | Clause |
|--------|--------|
| `event-vocabulary:Events:MUST:1` | 6 domain events defined |

### Bridge

| Anchor | Clause |
|--------|--------|
| `event-vocabulary:Bridge:MUST:1` | BRIDGE events translated |
| `event-vocabulary:Bridge:MUST:2` | Uses config classification |
| `event-vocabulary:Bridge:MUST:3` | BRIDGE event data promoted to top-level serialisable dict; no nested SDK objects |

### Consume

| Anchor | Clause |
|--------|--------|
| `event-vocabulary:Consume:MUST:1` | CONSUME events processed internally |
| `event-vocabulary:classification:system_notification:MUST:1` | system_notification classified as CONSUME |

### Drop

| Anchor | Clause |
|--------|--------|
| `event-vocabulary:Drop:MUST:1` | DROP events ignored |
| `event-vocabulary:Drop:MUST:2` | New session.* events from CLI binary version-skew added to DROP in events.yaml; MUST NOT produce warning log |

### FinishReason

| Anchor | Clause |
|--------|--------|
| `event-vocabulary:FinishReason:MUST:1` | SDK reasons mapped correctly |

### Classification

| Anchor | Clause |
|--------|--------|
| `event-vocabulary:Classification:MUST:1` | Each event type has exactly one classification |

---

## Implementation Checklist

- [ ] 6 domain events defined
- [ ] Config file has all classifications
- [ ] BRIDGE events produce DomainEvent
- [ ] CONSUME events update internal state
- [ ] DROP events return None
- [ ] Finish reason mapping works
