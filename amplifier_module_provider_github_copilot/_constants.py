"""Constants for GitHub Copilot SDK provider.

This module defines constants used across the Copilot SDK provider implementation,
following the principle of single source of truth.

Timeout Philosophy:
- Unified 1-hour default for ALL models (both regular and thinking)
- Users with unlimited Copilot tokens shouldn't worry about premature timeouts
- Override via config if you need faster failure detection:
    {"timeout": 300, "thinking_timeout": 600}
- The SDK/model will reject quickly if parameters are invalid; we wait generously
  for valid requests that naturally take time (complex reasoning, long outputs)

═══════════════════════════════════════════════════════════════════════════════
MODEL ID NAMING CONVENTION (CRITICAL)
═══════════════════════════════════════════════════════════════════════════════

Copilot SDK uses PERIODS for version numbers, not dashes:
  - CORRECT: claude-opus-4.5, gpt-5.1
  - WRONG:   claude-opus-4-5, gpt-5-1

See `model_naming.py` for:
  - Full evidence-based documentation (from live SDK data)
  - Pattern parsing and validation utilities
  - is_thinking_model() for timeout selection

WHY THIS MATTERS:
  - Model ID format affects capability detection
  - Capability detection determines if model supports extended thinking
  - Extended thinking controls whether reasoning_effort is sent to API
  - API rejects reasoning_effort for non-thinking models → understanding format prevents errors

"""

from enum import Enum, auto

# Default configuration values
# Model IDs use PERIODS per Copilot SDK (not dashes like Anthropic API)
DEFAULT_MODEL = "claude-opus-4.5"
DEFAULT_DEBUG_TRUNCATE_LENGTH = 180

# Timeout configuration
# Unified 1-hour default for all models. Users with unlimited tokens shouldn't
# need to worry about premature timeouts. Override via config if needed:
#   {"timeout": 300, "thinking_timeout": 600}
DEFAULT_TIMEOUT = 3600.0  # 1 hour - generous default for all models
DEFAULT_THINKING_TIMEOUT = 3600.0  # 1 hour - same as regular (can differentiate later)

# Valid reasoning effort levels per Copilot SDK
VALID_REASONING_EFFORTS = frozenset({"low", "medium", "high", "xhigh"})

# Maximum repaired tool IDs to track (LRU eviction)
MAX_REPAIRED_TOOL_IDS = 1000

# Buffer added to SDK-level timeout so our asyncio.timeout wins the race
# and provides consistent error handling. The SDK timeout acts as a fallback.
SDK_TIMEOUT_BUFFER_SECONDS = 5.0

# ═══════════════════════════════════════════════════════════════════════════════
# Copilot SDK Built-in Tool Names
# ═══════════════════════════════════════════════════════════════════════════════
#
# The Copilot CLI binary registers its own built-in tools with each session.
# The CLI server runs its OWN internal agent loop — built-in tool calls are
# executed silently inside the CLI process and are INVISIBLE to the SDK caller.
# The SDK only sees the final response text after all built-in tool execution.
#
# This creates TWO distinct problems:
#
# 1. SHADOWING: If a user-defined tool has the same name as a built-in,
#    the built-in handler shadows it — the user handler is never called.
#
# 2. BYPASS: Even when names DON'T collide, the model may choose a built-in
#    tool (e.g., CLI's "edit") over the equivalent user tool (e.g.,
#    Amplifier's "write_file"), causing the CLI to execute internally and
#    bypass the orchestrator's tool execution pipeline entirely.
#
# Evidence (forensic analysis 2026-02-07):
#   - Session 497bbab7: Model used CLI's "edit" built-in to create a file
#     instead of Amplifier's "write_file" tool → tool_calls=0, output=2 tokens,
#     but file was created (CLI executed internally, invisible to provider)
#   - SHADOWING: grep, glob, web_fetch cause session hangs
#   - BYPASS: edit, view execute internally without triggering tool capture
#
# SOLUTION: When user tools are registered, ALL known built-in tools MUST be
# excluded to ensure the Amplifier orchestrator has complete control over tool execution.
#
# NOTE: This set should be updated if the CLI adds new built-in tools.
# Source: SDK e2e tests, agentic-workflow.json schema, SDK hook docs.

COPILOT_BUILTIN_TOOL_NAMES: frozenset[str] = frozenset(
    {
        # File operations
        "view",  # Read/view file contents
        "edit",  # Edit/create/modify files (THE file writing tool)
        "grep",  # Search text patterns in files
        "glob",  # Find files matching glob patterns
        # Shell execution — Linux/macOS
        "bash",  # Execute bash commands
        "read_bash",  # Read-only bash commands
        "write_bash",  # Write bash commands
        # Shell execution — Windows
        "powershell",  # Execute PowerShell commands
        "read_powershell",  # Read-only PowerShell commands
        "write_powershell",  # Write PowerShell commands
        # Web
        "web_fetch",  # Fetch web content
        "web_search",  # Web search (internet searches)
        # User interaction
        "ask_user",  # Request user input (conditional on handler)
        # Hidden built-ins (discovered 2026-02-07 via exploratory testing)
        # These cause session hangs if a user tool has the same name.
        # They are NOT documented but conflict with custom tools.
        "report_intent",  # Hidden: Causes hang if custom tool uses same name
        "task",  # Hidden: Causes hang if custom tool uses same name
    }
)

# ═══════════════════════════════════════════════════════════════════════════════
# Built-in → Amplifier Capability Mapping
# ═══════════════════════════════════════════════════════════════════════════════
#
# Maps each CLI built-in tool to Amplifier tool names whose capabilities
# overlap. A built-in is excluded if ANY of its mapped Amplifier tools
# are registered, preventing the model from choosing the built-in over
# the orchestrator-controlled version.
#
# IMPORTANT: Excluding ALL built-ins at once hangs the CLI (tested 2026-02-07,
# session 2a1fe04a). Only exclude built-ins that have a corresponding
# user-registered tool.
#
# Evidence:
#   - Session 497bbab7: "edit" not excluded → CLI bypassed orchestrator
#   - Session 2a1fe04a: ALL 13 excluded → CLI hangs indefinitely
#   - Working: Exclude only overlapping built-ins

# ═══════════════════════════════════════════════════════════════════════════════
# SDK DRIVER BEHAVIOR CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
#
# These constants encode the empirically-observed behavior of the Copilot SDK.
# Evidence sources:
#   - Session a1a0af17: 305 turns from denial loop
#   - test_dumb_pipe_strategies.py: Validated denial causes retry
#   - SDK session.py source: abort() API exists
#
# When adding support for new SDKs (if this pattern is generalized), these
# constants would move to a per-SDK configuration system.


class DenialBehavior(Enum):
    """What happens when preToolUse hook denies tool execution."""

    RETRY = auto()  # SDK feeds error to LLM, LLM retries (Copilot!)
    FAIL = auto()  # SDK stops with error
    ESCALATE = auto()  # SDK asks user for decision
    IGNORE = auto()  # SDK continues without tool


class LoopExitMethod(Enum):
    """How to exit SDK's internal agent loop early."""

    ABORT = auto()  # Call session.abort() - interrupts processing
    DESTROY = auto()  # Call session.destroy() - terminates session
    TIMEOUT = auto()  # Wait for SDK timeout (slow, not recommended)


# Copilot SDK behavioral profile
COPILOT_DENIAL_BEHAVIOR = DenialBehavior.RETRY  # Causes 305-turn loops!
COPILOT_RECOMMENDED_EXIT = LoopExitMethod.ABORT  # Fastest clean exit

# Circuit breaker settings
# Evidence: 305 turns observed in incident. 3 is generous for legitimate retries.
SDK_MAX_TURNS_DEFAULT = 3
SDK_MAX_TURNS_HARD_LIMIT = 10  # Absolute maximum, even if configured higher

# Capture strategy
# Evidence: 607 tools captured from all turns. Only first turn is valid.
CAPTURE_FIRST_TURN_ONLY = True

# Deduplication
# Evidence: Same (delegate, report_intent) pair repeated 303 times
DEDUPLICATE_TOOL_CALLS = True

BUILTIN_TO_AMPLIFIER_CAPABILITY: dict[str, frozenset[str]] = {
    # File operation overlaps
    "view": frozenset({"read_file"}),
    "edit": frozenset({"write_file", "edit_file"}),
    "grep": frozenset({"grep"}),
    "glob": frozenset({"glob"}),
    # Shell overlaps
    "bash": frozenset({"bash"}),
    "read_bash": frozenset({"bash"}),
    "write_bash": frozenset({"bash"}),
    "powershell": frozenset({"bash"}),  # Amplifier uses "bash" on all platforms
    "read_powershell": frozenset({"bash"}),
    "write_powershell": frozenset({"bash"}),
    # Web overlaps
    "web_fetch": frozenset({"web_fetch"}),
    "web_search": frozenset({"web_search"}),
    # No Amplifier equivalent — don't exclude unless name collision
    "ask_user": frozenset(),
    # Hidden built-ins — exclude unconditionally to prevent hangs
    # These are not documented but cause session hangs if user tool has same name.
    "report_intent": frozenset({"report_intent"}),  # Hidden: Always exclude
    "task": frozenset({"task"}),  # Hidden: Always exclude
}
