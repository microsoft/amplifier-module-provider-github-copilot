"""Execute Permission Handling.

Repairs execute permissions for SDK-bundled CLI binary stripped by package managers.

Contract: sdk-boundary:BinaryResolution:MUST:6-7

The `uv` package manager strips execute permissions from bundled binaries.
This module repairs the permissions before mount() proceeds.

MUST constraints:
- MUST set S_IXUSR | S_IXGRP only (no world-execute per security review)
- MUST be no-op on Windows (os.access unreliable)
- MUST be idempotent (no chmod if already executable)
"""

from __future__ import annotations

import logging
import stat
from pathlib import Path

from .security_redaction import safe_log_message

logger = logging.getLogger(__name__)

# Permission bits for user and group execute
_EXECUTE_BITS = stat.S_IXUSR | stat.S_IXGRP


def ensure_executable(path: Path) -> bool:
    """Ensure file has user+group execute permission (S_IXUSR|S_IXGRP).

    MUST NOT set world-execute (S_IXOTH) — least privilege.
    Idempotent: returns True immediately if already executable.
    No-op on Windows (always returns True).

    Contract: sdk-boundary:BinaryResolution:MUST:6
    Contract: sdk-boundary:BinaryResolution:MUST:7

    Args:
        path: Path to the file to make executable.

    Returns:
        True if executable (or made so), False on failure.

    """
    from ._platform import get_platform_info

    # No-op on Windows — os.access(X_OK) is unreliable
    if get_platform_info().is_windows:
        return True

    # Check if file exists
    if not path.is_file():
        logger.warning("[PERMISSIONS] File not found: %s", path)
        return False

    try:
        # Get current mode
        current_mode = path.stat().st_mode

        # Check if already executable (user execute bit is sufficient)
        if current_mode & stat.S_IXUSR:
            return True

        # Add execute bits (user + group only, no world)
        new_mode = current_mode | _EXECUTE_BITS
        path.chmod(new_mode)

        logger.debug("[PERMISSIONS] Added execute permission: %s", path)
        return True

    except PermissionError as e:
        logger.warning(*safe_log_message("[PERMISSIONS] Permission denied fixing %s: %s", path, e))
        return False
    except OSError as e:
        logger.warning(*safe_log_message("[PERMISSIONS] OS error fixing %s: %s", path, e))
        return False
