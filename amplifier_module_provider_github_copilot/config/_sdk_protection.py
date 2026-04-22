"""SDK protection policy — dataclasses with hardcoded defaults.

Python policy module — replaces the former config/sdk_protection.yaml YAML file.
Contract: contracts/sdk-protection.md

SoC: This module contains DATA (dataclasses + defaults) only.
     No loading logic. No file I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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

    Contract: sdk-boundary:MinimalMode:MUST:1-6

    Disables SDK features that Amplifier handles, ensuring Amplifier is the sole
    orchestrator. Evidence: 57% wall-clock improvement (12.5s → 5.4s) confirmed
    in sessions 7db2b5f7 and 2fa58db6.
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
