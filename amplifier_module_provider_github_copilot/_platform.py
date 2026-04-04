"""Cross-Platform Binary Discovery.

Platform detection and CLI binary location for GitHub Copilot SDK.

Contract: sdk-boundary:BinaryResolution:MUST:1-8

This module handles:
- Platform detection (Windows/macOS/Linux)
- Binary name resolution (copilot vs copilot.exe)
- SDK-bundled binary location via importlib.util.find_spec
- PATH fallback for system-installed CLI

MUST constraints:
- MUST detect platform once and cache via @lru_cache
- MUST use importlib.util.find_spec, NOT import copilot (sdk-boundary membrane)
- MUST prefer SDK-bundled binary over PATH (security)
"""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

__all__ = [
    "PlatformInfo",
    "get_platform_info",
    "get_cli_binary_name",
    "get_sdk_binary_path",
    "find_cli_in_path",
    "locate_cli_binary",
    "is_pytest_running",
]

# Binary name constants
CLI_BINARY_NAME_UNIX = "copilot"
CLI_BINARY_NAME_WINDOWS = "copilot.exe"
CLI_BINARY_SUBDIR = "bin"


@dataclass(frozen=True)
class PlatformInfo:
    """Immutable platform facts. Cached — platform doesn't change at runtime.

    Attributes:
        name: Human-readable platform name ("Windows", "macOS", "Unix")
        is_windows: True if running on Windows
        is_wsl: True if running inside Windows Subsystem for Linux
        cli_binary_name: Platform-appropriate binary name

    """

    name: str
    is_windows: bool
    cli_binary_name: str
    is_wsl: bool = False


@lru_cache(maxsize=1)
def get_platform_info() -> PlatformInfo:
    """Single source of truth for platform detection.

    MUST be the ONLY place sys.platform is checked for binary naming.

    Contract: sdk-boundary:BinaryResolution:MUST:1

    Returns:
        PlatformInfo with cached platform details.

    """
    platform = sys.platform.lower()

    if platform == "win32":
        return PlatformInfo(
            name="Windows",
            is_windows=True,
            cli_binary_name=CLI_BINARY_NAME_WINDOWS,
        )
    elif platform == "darwin":
        return PlatformInfo(
            name="macOS",
            is_windows=False,
            cli_binary_name=CLI_BINARY_NAME_UNIX,
        )
    else:
        # Linux, cygwin, other Unix-like systems
        # Detect WSL via /proc/version containing "microsoft"
        is_wsl = False
        try:
            with open("/proc/version", encoding="utf-8") as f:
                is_wsl = "microsoft" in f.read().lower()
        except OSError:
            pass  # /proc/version not present on non-Linux or in some containers
        return PlatformInfo(
            name="Unix",
            is_windows=False,
            cli_binary_name=CLI_BINARY_NAME_UNIX,
            is_wsl=is_wsl,
        )


def get_cli_binary_name() -> str:
    """Return platform-appropriate binary name.

    Contract: sdk-boundary:BinaryResolution:MUST:3

    Returns:
        "copilot.exe" on Windows, "copilot" elsewhere.

    """
    return get_platform_info().cli_binary_name


def get_sdk_binary_path() -> Path | None:
    """Locate SDK-bundled binary via sdk_adapter membrane.

    Uses the membrane's public API to locate copilot package path,
    keeping all SDK references properly quarantined.

    Contract: sdk-boundary:BinaryResolution:MUST:2
    Contract: sdk-boundary:Membrane:MUST:1

    Returns:
        Path to copilot/bin/copilot[.exe] if found, None otherwise.

    """
    from amplifier_module_provider_github_copilot.sdk_adapter import (
        get_copilot_spec_origin,
    )

    origin = get_copilot_spec_origin()
    if origin is None:
        return None

    # Get package directory from origin (which is the __init__.py path)
    try:
        package_dir = Path(origin).parent
    except (TypeError, ValueError):
        return None

    # Look for binary in bin/ subdirectory
    binary_name = get_cli_binary_name()
    binary_path = package_dir / CLI_BINARY_SUBDIR / binary_name

    if binary_path.is_file():
        return binary_path

    return None


def find_cli_in_path() -> Path | None:
    """Fallback: find binary in system PATH via shutil.which.

    Tries platform-appropriate name first, then alternate.
    Handles WSL edge case (Windows PATH entries visible in WSL).

    Contract: sdk-boundary:BinaryResolution:MUST:5
    Contract: sdk-boundary:BinaryResolution:SHOULD:1

    Returns:
        Path to CLI binary in PATH, or None if not found.

    """
    platform_info = get_platform_info()

    # Try platform-appropriate name first
    primary = shutil.which(platform_info.cli_binary_name)
    if primary:
        return Path(primary)

    # Try alternate name (handles WSL with Windows PATH)
    alternate = CLI_BINARY_NAME_UNIX if platform_info.is_windows else CLI_BINARY_NAME_WINDOWS
    fallback = shutil.which(alternate)
    if fallback:
        return Path(fallback)

    return None


def locate_cli_binary() -> Path | None:
    """Locate the CLI binary.

    Resolution order:

    1. SDK bundled binary (preferred — version-matched, tamper-resistant)
    2. System PATH (fallback — less secure)

    Contract: sdk-boundary:BinaryResolution:MUST:4
    Contract: sdk-boundary:BinaryResolution:MUST:5

    Returns:
        Path to CLI binary, or None if not found.

    """
    # Try SDK binary first (preferred)
    sdk_path = get_sdk_binary_path()
    if sdk_path is not None:
        return sdk_path

    # Fallback to PATH
    return find_cli_in_path()


def is_pytest_running() -> bool:
    """Check if pytest is currently executing as the test runner.

    Used as a guard in test-only SDK bypass logic. Centralised here
    (single source of truth) so both __init__.py and sdk_adapter/_imports.py
    share the same implementation.

    Note: Test-only convenience bypass — NOT a security boundary.
    An adversary can bypass this by importing pytest before setting
    SKIP_SDK_CHECK. Guards against accidental misuse only.
    """
    return "pytest" in sys.modules
