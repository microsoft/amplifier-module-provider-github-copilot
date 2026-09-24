"""SDK protection policy — dataclasses with hardcoded defaults.

Python policy module — replaces the former config/sdk_protection.yaml YAML file.
Contract: contracts/sdk-protection.md

SoC: This module contains DATA (dataclasses + defaults) only.
     No loading logic. No file I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = [
    "MinimalModeConfig",
    "ToolCaptureConfig",
    "SessionProtectionConfig",
    "SingletonConfig",
    "SdkConfig",
    "SdkProtectionConfig",
]


@dataclass(frozen=True)
class MinimalModeConfig:
    """Minimal mode session configuration policy.

    Contract: sdk-boundary:MinimalMode:MUST:1-16

    Disables SDK features that Amplifier handles, ensuring Amplifier is the sole
    orchestrator. Evidence: 57% wall-clock improvement (12.5s → 5.4s) confirmed
    in sessions 7db2b5f7 and 2fa58db6 (MUST:1-6 baseline, b9).

    MUST:7-15 are pinned as part of the b10 bump. MUST:7-13 and MUST:15 are new
    create_session kwargs introduced at b10 (verified against b10
    `client.py:1582-1605` and absent from b9); MUST:14
    (`enable_session_telemetry`) was already a b9 kwarg (b9 `client.py:1550`)
    and is consolidated under MinimalMode now so all SDK-internal session
    capabilities live in one place. Each of the 9 kwargs has an empty-mode
    default helper at b10 `_mode.py:185-258` that fires ONLY when
    `mode == "empty"`. Our adapter ships `mode="copilot-cli"`, so explicit pins
    ARE the wire shape — leaving any of these `None` lets the bundled CLI
    defaults apply.
    """

    # MUST:1 — Disable SDK compaction; Amplifier manages context.
    infinite_sessions_enabled: bool = False

    # MUST:2 — Prevent SDK from scanning for .mcp.json and AGENTS.md.
    enable_config_discovery: bool = False

    # MUST:3 — Explicit empty; Amplifier routes all tools.
    mcp_servers: dict[str, Any] = field(default_factory=dict)

    # MUST:4 — Explicit empty; Amplifier has its own skills system.
    skill_directories: list[str] = field(default_factory=list)

    # MUST:5 — Explicit empty; Amplifier orchestrates agents.
    custom_agents: list[str] = field(default_factory=list)

    # MUST:6 — Explicit empty; Amplifier handles slash commands.
    commands: list[str] = field(default_factory=list)

    # MUST:7 — Disable SDK cross-session persistent store; sessions are
    # ephemeral per `complete()` call (deny-destroy contract).
    enable_session_store: bool = False

    # MUST:8 — Disable SDK skills loader; stronger than MUST:4 (which only
    # empties the directory list).
    enable_skills: bool = False

    # MUST:9 — Disable SDK file-hook discovery (AGENTS.md walkers etc.);
    # Amplifier registers a single deny-all hook explicitly.
    enable_file_hooks: bool = False

    # MUST:10 — Disable SDK host-git delegation; Amplifier never delegates
    # git to the SDK and the deny-all hook would block it anyway.
    enable_host_git_operations: bool = False

    # MUST:11 — Disable SDK on-demand instruction scans; complements MUST:2.
    enable_on_demand_instruction_discovery: bool = False

    # MUST:12 — Disable SDK embedding-based workspace retrieval; Amplifier
    # owns context construction.
    skip_embedding_retrieval: bool = True

    # MUST:13 — Keep embedding cache in RAM; no on-disk residue across the
    # ephemeral session boundary.
    embedding_cache_storage: Literal["persistent", "in-memory"] = "in-memory"

    # MUST:14 — Disable SDK-internal session telemetry; Amplifier owns
    # observability. b10 `client.py:1651-1656` documents this as ON-by-default
    # for GitHub-authenticated sessions, which is our COPILOT_AGENT_TOKEN path.
    enable_session_telemetry: bool = False

    # MUST:15 — Keep MCP OAuth token storage in RAM; no on-disk token residue
    # across the ephemeral session boundary. Mirrors MUST:13. The wire emit at
    # b10 `client.py:1863-1865` is INDEPENDENT of `mcp_servers` (MUST:3); under
    # `mode="copilot-cli"` the helper `_mcp_oauth_token_storage_default`
    # (b10 `_mode.py:251-258`) returns None, so without this pin the bundled-CLI
    # default applies.
    mcp_oauth_token_storage: Literal["persistent", "in-memory"] = "in-memory"

    # MUST:16 — Disable the SDK memory feature; Amplifier owns context and
    # persistence, and sessions are ephemeral per `complete()` call. Added at
    # v1.0.2 as a new mode-gated create_session kwarg with empty-mode default
    # helper `_memory_default` (v1.0.2 `_mode.py:264-276`) that returns
    # `{"enabled": False}` ONLY when `mode == "empty"`. Our adapter ships
    # `mode="copilot-cli"`, where the helper returns None and the bundled-CLI
    # default would otherwise apply — so this explicit pin IS the wire shape.
    # Value mirrors the SDK empty-mode default exactly (contract:
    # sdk-boundary:MinimalMode SDK-introspection closure).
    memory: dict[str, Any] = field(default_factory=lambda: {"enabled": False})

    # MUST:17 — Confine custom-agent resolution to the local session; never let
    # the SDK resolve agents from shared/remote sources. Added at v1.0.14 as a
    # new mode-gated create_session kwarg with empty-mode default helper
    # `_custom_agents_local_only_default` returning True ONLY when
    # `mode == "empty"`. Our adapter ships `mode="copilot-cli"`, where the
    # helper returns None and the bundled-CLI default would otherwise apply —
    # so this explicit pin IS the wire shape. Complements MUST:5 (which only
    # empties the custom_agents list). Value mirrors the SDK empty-mode default
    # exactly (contract: sdk-boundary:MinimalMode SDK-introspection closure).
    custom_agents_local_only: bool = True

    # MUST:18 — Keep SDK experimental features off. Added at v1.0.14 as a new
    # mode-gated create_session kwarg with empty-mode default helper
    # `_enable_experimental_mode_default` returning False ONLY when
    # `mode == "empty"`. Under `mode="copilot-cli"` the helper returns None and
    # the bundled-CLI default applies, so without this pin a CLI upgrade could
    # silently switch experimental behavior on underneath the provider — the
    # wire shape must be owned here, not inherited. Value mirrors the SDK
    # empty-mode default exactly.
    enable_experimental_mode: bool = False


@dataclass(frozen=True)
class ToolCaptureConfig:
    """Tool capture policy.

    Contract: sdk-protection:ToolCapture:MUST:1,2

    The SDK emits ASSISTANT_MESSAGE events with tool_requests when the model
    wants to call tools. These must be captured and returned to Amplifier's
    orchestrator for execution.

    CRITICAL: Do NOT let the SDK execute tools — only Amplifier does execution.
    """

    # First-turn-only prevents accumulation across multiple SDK turns.
    # When the model requests tools, capture them once and abort — don't loop.
    first_turn_only: bool = True

    # Deduplicate by tool_call_id to prevent duplicate execution.
    # SDK may emit duplicate events during reconnection or retry scenarios.
    deduplicate: bool = True

    # Log capture events for debugging and forensics.
    # Set to false in production for reduced log volume.
    log_capture_events: bool = True


@dataclass(frozen=True)
class SessionProtectionConfig:
    """Session protection policy.

    Contract: sdk-protection:Session:MUST:3,4

    Note: Named SessionProtectionConfig (not SessionConfig) to avoid collision
    with sdk_adapter.types.SessionConfig which configures SDK session creation.
    This class configures session lifecycle protection (abort, idle timeouts).

    Sessions are ephemeral — created per complete() call and destroyed after.
    This prevents state accumulation and ensures clean abort on capture.
    """

    # Call session.abort() explicitly after tool capture.
    explicit_abort: bool = True

    # Timeout for abort call to prevent hang on shutdown.
    # If abort takes longer than this, log warning and proceed with cleanup.
    abort_timeout_seconds: float = 5.0

    # Maximum wait time for session.idle event (abort/cleanup operations only).
    # NOTE: This is NOT used for main SDK wait. SDK API calls can take 60+ seconds
    # for complex operations like delegation. The provider uses caller's timeout.
    # Contract: streaming-contract:abort-on-capture:MUST:1
    idle_timeout_seconds: float = 30.0

    # Timeout for session.disconnect() call.
    # Prevents indefinite hang on stubborn sessions during cleanup.
    # Contract: sdk-protection:Session:MUST:3
    disconnect_timeout_seconds: float = 30.0


@dataclass
class SingletonConfig:
    """Singleton lifecycle policy.

    Contract: sdk-protection:Singleton:MUST:8

    The provider uses a process-level singleton for CopilotClientWrapper.
    This prevents N sub-agents from each spawning a ~500MB Electron subprocess.
    """

    # Timeout for acquiring the singleton threading.Lock.
    # If this fires, another mount() is deadlocked — indicates a bug.
    lock_timeout_seconds: float = 30.0


@dataclass
class SdkConfig:
    """SDK subprocess configuration.

    Contract: sdk-protection:Subprocess:MUST:7
    """

    # Log level for SDK subprocess. Writes to ~/.copilot/logs/.
    # WARNING: debug/all logs contain full conversation data (prompts, responses).
    log_level: str = "info"

    # Allowlist for log_level validation.
    # Contract: sdk-protection:Subprocess:MUST:7
    valid_log_levels: list[str] = field(
        default_factory=lambda: ["none", "error", "warning", "info", "debug", "all"]
    )

    # Environment variable to override log_level at runtime.
    # Allows operators to enable debug without code changes.
    log_level_env_var: str = "COPILOT_SDK_LOG_LEVEL"

    # Timeout for CopilotClient.stop() inside CopilotClientWrapper.close().
    # stop() tears down the ~500MB Electron subprocess; an unresponsive or
    # wedged subprocess leaves that await pending forever, and close() is on
    # the mount()-cleanup path -- so an unbounded stop hangs Amplifier's
    # session cleanup for the whole process. Bound it, warn, and abandon.
    # Contract: sdk-protection:Subprocess:MUST:8
    close_timeout_seconds: float = 5.0

    # Pre-warm SDK subprocess at mount() time.
    # When true, subprocess spawn (~2s) happens in background during mount(),
    # so first complete() has ~200ms latency instead of ~2000ms.
    # Trade-off: Subprocess memory (~100MB) allocated even if unused.
    prewarm_subprocess: bool = False


@dataclass
class SdkProtectionConfig:
    """SDK protection policy.

    Contract: contracts/sdk-protection.md

    Instantiate with no arguments to get hardcoded defaults:
        config = SdkProtectionConfig()
    """

    minimal_mode: MinimalModeConfig = field(default_factory=MinimalModeConfig)
    tool_capture: ToolCaptureConfig = field(default_factory=ToolCaptureConfig)
    session: SessionProtectionConfig = field(default_factory=SessionProtectionConfig)
    sdk: SdkConfig = field(default_factory=SdkConfig)
    singleton: SingletonConfig = field(default_factory=SingletonConfig)
