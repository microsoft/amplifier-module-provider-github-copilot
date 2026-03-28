"""Type stubs for copilot.types module."""

from typing import Any, Literal
from dataclasses import dataclass


LogLevel = Literal["debug", "info", "warn", "error"]


@dataclass
class TelemetryConfig:
    """Telemetry configuration."""
    enabled: bool = True


@dataclass
class SubprocessConfig:
    """Configuration for SDK subprocess mode.
    
    Matches real SDK signature from help(SubprocessConfig).
    """
    cli_path: str | None = None
    cli_args: list[str] | None = None
    cwd: str | None = None
    use_stdio: bool = True
    port: int = 0
    log_level: LogLevel = "info"
    env: dict[str, str] | None = None
    github_token: str | None = None
    use_logged_in_user: bool | None = None
    telemetry: TelemetryConfig | None = None


@dataclass
class BlobAttachment:
    """Image attachment for vision models."""
    data: bytes
    media_type: str


@dataclass
class PermissionRequestResult:
    """Result of a permission request."""
    kind: str
    allowed: bool = True
    message: str | None = None


@dataclass
class ModelInfo:
    """Information about an available model."""
    id: str
    name: str
    family: str | None = None
    vendor: str | None = None
    capabilities: list[str] | None = None
    context_window: int | None = None
    max_output_tokens: int | None = None
    preview: bool = False
    is_default: bool = False
    policy: "ModelPolicy | None" = None


@dataclass
class ModelPolicy:
    """Policy settings for a model."""
    state: Literal["available", "limited", "unavailable"] = "available"
    terms: str | None = None


# SDK event types (generic - actual structure varies by event)
class SDKEvent:
    """Base type for SDK streaming events."""
    type: str
    payload: Any


__all__ = [
    "SubprocessConfig",
    "BlobAttachment", 
    "PermissionRequestResult",
    "ModelInfo",
    "ModelPolicy",
    "SDKEvent",
]
