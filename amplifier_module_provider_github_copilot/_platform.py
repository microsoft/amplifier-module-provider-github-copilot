"""Cross-platform utilities for the GitHub Copilot provider module.

This module centralizes ALL platform-specific logic following the principle
of Single Source of Truth. Platform detection and binary naming are done
in ONE place to prevent the scattered `sys.platform` checks that led to
the Windows .exe bug (Hotfix 2026-03-07).

Design Patterns Used:
- Strategy Pattern: Platform-specific behavior via PlatformInfo
- Factory Function: get_platform_info() returns appropriate implementation
- Single Source of Truth: Binary names defined once, used everywhere

Cross-Platform Considerations:
- Windows: Uses .exe extension for executables
- Linux/macOS/WSL: No extension for executables
- WSL: Reports as 'linux' (sys.platform), uses Unix conventions
- Cygwin: Reports as 'cygwin', uses Unix conventions

Usage:
    from ._platform import get_cli_binary_name, get_sdk_binary_path

    # Get just the binary name for current platform
    name = get_cli_binary_name()  # "copilot.exe" on Windows, "copilot" elsewhere

    # Get full path from SDK module
    path = get_sdk_binary_path()  # Full path or None
"""

from __future__ import annotations

import logging
import shutil
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Constants - Single Source of Truth for binary names
# ═══════════════════════════════════════════════════════════════════════════════

CLI_BINARY_NAME_UNIX = "copilot"
CLI_BINARY_NAME_WINDOWS = "copilot.exe"
CLI_BINARY_SUBDIR = "bin"


# ═══════════════════════════════════════════════════════════════════════════════
# Platform Detection
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PlatformInfo:
    """Immutable platform information.

    This dataclass encapsulates all platform-specific details needed
    for CLI binary location. Using frozen=True makes it immutable,
    preventing accidental modification.

    Attributes:
        name: Human-readable platform name (for logging)
        is_windows: True if running on Windows
        cli_binary_name: Name of the CLI binary ("copilot" or "copilot.exe")
        uses_exe_extension: True if platform uses .exe for executables
    """

    name: str
    is_windows: bool
    cli_binary_name: str
    uses_exe_extension: bool


@lru_cache(maxsize=1)
def get_platform_info() -> PlatformInfo:
    """Get platform information for the current system.

    This is the ONLY place where sys.platform is checked for binary naming.
    All other code should use this function.

    The result is cached since platform doesn't change during runtime.
    Use get_platform_info.cache_clear() to reset if needed (e.g., testing).

    Returns:
        PlatformInfo with current platform details
    """
    is_windows = sys.platform == "win32"

    if is_windows:
        return PlatformInfo(
            name="Windows",
            is_windows=True,
            cli_binary_name=CLI_BINARY_NAME_WINDOWS,
            uses_exe_extension=True,
        )
    elif sys.platform == "darwin":
        return PlatformInfo(
            name="macOS",
            is_windows=False,
            cli_binary_name=CLI_BINARY_NAME_UNIX,
            uses_exe_extension=False,
        )
    else:
        # Linux, WSL, Cygwin, FreeBSD, etc.
        return PlatformInfo(
            name="Unix",
            is_windows=False,
            cli_binary_name=CLI_BINARY_NAME_UNIX,
            uses_exe_extension=False,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Binary Location Functions
# ═══════════════════════════════════════════════════════════════════════════════


def get_cli_binary_name() -> str:
    """Get the CLI binary name for the current platform.

    Returns:
        "copilot.exe" on Windows, "copilot" elsewhere
    """
    return get_platform_info().cli_binary_name


def get_sdk_binary_path() -> Path | None:
    """Get the path to the SDK-bundled CLI binary.

    Attempts to locate the CLI binary bundled with the copilot SDK package.
    This is the preferred binary as it's version-matched with the SDK.

    Returns:
        Path to the binary if found, None otherwise
    """
    try:
        import copilot as _copilot_mod  # type: ignore[import-untyped]

        mod_file = _copilot_mod.__file__
        if mod_file is None:
            logger.debug("[PLATFORM] copilot module has no __file__")
            return None

        platform = get_platform_info()
        bin_dir = Path(mod_file).parent / CLI_BINARY_SUBDIR
        cli_bin = bin_dir / platform.cli_binary_name

        if cli_bin.exists():
            logger.debug(f"[PLATFORM] Found SDK binary at: {cli_bin}")
            return cli_bin
        else:
            logger.debug(f"[PLATFORM] SDK binary not found at: {cli_bin}")
            return None

    except ImportError:
        logger.debug("[PLATFORM] copilot SDK not installed")
        return None
    except Exception as e:
        logger.debug(f"[PLATFORM] Error locating SDK binary: {e}")
        return None


def find_cli_in_path() -> Path | None:
    """Find the CLI binary in system PATH.

    Searches for both "copilot" and "copilot.exe" to handle cases where
    a Windows binary might be found on a Unix system (e.g., Wine) or
    vice versa.

    Security Note (PATH Hijack Risk):
        This function uses shutil.which() which searches PATH directories
        in order. A malicious binary placed earlier in PATH could be
        executed instead of the legitimate CLI.

        Mitigation:
        - This is a FALLBACK only — SDK bundled binary is preferred
        - locate_cli_binary() checks get_sdk_binary_path() first
        - PATH lookup only occurs when SDK binary is unavailable
        - Standard security practice: don't add untrusted dirs to PATH

    Returns:
        Path to the binary if found, None otherwise
    """
    # Try platform-appropriate name first
    platform = get_platform_info()
    found = shutil.which(platform.cli_binary_name)

    if found:
        logger.debug(f"[PLATFORM] Found CLI in PATH: {found}")
        return Path(found)

    # Fallback: try the other variant
    # This handles edge cases like Windows Subsystem for Linux with Windows PATH
    alternate_name = (
        CLI_BINARY_NAME_UNIX if platform.uses_exe_extension else CLI_BINARY_NAME_WINDOWS
    )
    found = shutil.which(alternate_name)

    if found:
        logger.debug(f"[PLATFORM] Found CLI in PATH (alternate): {found}")
        return Path(found)

    logger.debug("[PLATFORM] CLI not found in PATH")
    return None


def locate_cli_binary() -> Path | None:
    """Locate the CLI binary using the standard resolution order.

    Resolution order (security-conscious):
    1. SDK bundled binary (PREFERRED, version-matched, tamper-resistant)
    2. System PATH fallback (only when SDK binary unavailable)

    This is the main entry point for CLI location. It combines all
    discovery strategies and returns the first successful result.

    Security Design:
        The SDK binary is ALWAYS preferred over PATH because:
        - It's bundled with the pip package (integrity verified)
        - It's version-matched with the SDK API
        - It's not susceptible to PATH hijacking attacks

        PATH is only used when:
        - SDK is not installed via pip (e.g., development setup)
        - SDK binary path resolution fails (import error)

    Returns:
        Path to the CLI binary, or None if not found
    """
    # Strategy 1: SDK bundled binary (preferred - secure)
    sdk_path = get_sdk_binary_path()
    if sdk_path:
        return sdk_path

    # Strategy 2: System PATH (fallback - less secure, see docstring)
    path_binary = find_cli_in_path()
    if path_binary:
        return path_binary

    logger.debug("[PLATFORM] CLI binary not found via any strategy")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Testing Utilities
# ═══════════════════════════════════════════════════════════════════════════════


def _make_test_platform_info(
    *,
    is_windows: bool = False,
    name: str | None = None,
) -> PlatformInfo:
    """Create a PlatformInfo for testing purposes.

    This function is prefixed with underscore to indicate it's for testing.
    Use this instead of mocking sys.platform directly.

    Args:
        is_windows: Whether to simulate Windows
        name: Optional custom name

    Returns:
        PlatformInfo configured for testing
    """
    if is_windows:
        return PlatformInfo(
            name=name or "Windows (test)",
            is_windows=True,
            cli_binary_name=CLI_BINARY_NAME_WINDOWS,
            uses_exe_extension=True,
        )
    else:
        return PlatformInfo(
            name=name or "Unix (test)",
            is_windows=False,
            cli_binary_name=CLI_BINARY_NAME_UNIX,
            uses_exe_extension=False,
        )
