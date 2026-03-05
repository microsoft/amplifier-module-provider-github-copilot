"""File permission utilities for the GitHub Copilot provider module.

This module handles file permission operations, specifically ensuring
binaries have execute permissions following least-privilege principles.
"""

import logging
import os
import stat
from pathlib import Path

logger = logging.getLogger(__name__)


def ensure_executable(path: Path) -> bool:
    """Ensure a file has user and group execute permissions.

    Sets S_IXUSR | S_IXGRP only — does NOT set S_IXOTH (world execute)
    to follow the principle of least privilege.

    Args:
        path: Path to the file to make executable

    Returns:
        True if file is executable (or was made so), False on failure

    Note:
        Security requirement from architecture review (PR #6):
        "Both executable-permission fixes add world-execute (S_IXOTH).
        Should be S_IXUSR | S_IXGRP only."
    """
    # Edge case: file doesn't exist
    if not path.exists():
        logger.debug(f"File does not exist: {path}")
        return False

    # Idempotent: already executable by user
    if os.access(path, os.X_OK):
        return True

    try:
        current = path.stat().st_mode
        # Security: S_IXUSR | S_IXGRP only (no world-execute S_IXOTH)
        path.chmod(current | stat.S_IXUSR | stat.S_IXGRP)
        logger.debug(f"Added execute permission to {path}")
        return True
    except PermissionError:
        logger.debug(f"Permission denied setting execute on {path}")
        return False
    except OSError as e:
        logger.debug(f"Failed to set execute on {path}: {e}")
        return False
