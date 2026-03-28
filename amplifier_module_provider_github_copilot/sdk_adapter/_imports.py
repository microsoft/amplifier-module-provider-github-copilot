"""SDK Import Quarantine.

All SDK imports are isolated here per sdk-boundary.md contract.

This enables:
- Easy SDK version tracking (all imports in one place)
- Single point for SDK compatibility shims
- Clear boundary for membrane violations
- Import-time failure if SDK not installed

Contract: contracts/sdk-boundary.md
"""

from __future__ import annotations

import os
import sys
from typing import Any

# Re-export SDK-independent utilities for backward compatibility.
# New code should import directly from sdk_adapter (the membrane).
from ._spec_utils import get_copilot_spec_origin

# =============================================================================
# SDK imports - THE ONLY PLACE IN THE CODEBASE where SDK is imported
# =============================================================================


def _is_pytest_running() -> bool:
    """Check if pytest is running (for test-only SDK bypass).

    P1-6 Security Fix: SDK bypass only allowed when pytest is actually running.
    This prevents production misuse while preserving test functionality.
    """
    return "pytest" in sys.modules


# Only skip SDK imports if BOTH conditions are met:
# 1. SKIP_SDK_CHECK env var is set
# 2. pytest is actually running (prevents production misuse)
_SKIP_SDK_CHECK = os.environ.get("SKIP_SDK_CHECK") and _is_pytest_running()

# Guard against import failures - fail fast with clear error
# Unless SKIP_SDK_CHECK is set (for testing without SDK)
CopilotClient: Any
PermissionRequestResult: Any
SubprocessConfig: Any

if _SKIP_SDK_CHECK:
    # Test mode: provide None stubs that tests can mock
    CopilotClient = None  # type: ignore[misc,assignment]
    PermissionRequestResult = None  # type: ignore[misc,assignment]
    SubprocessConfig = None  # type: ignore[misc,assignment]
else:
    try:
        from copilot import CopilotClient  # type: ignore[import-untyped,no-redef]
    except ImportError as e:
        raise ImportError(
            "github-copilot-sdk not installed. Install with: pip install github-copilot-sdk"
        ) from e

    # SDK v0.2.0: SubprocessConfig replaces options dict
    try:
        from copilot.types import SubprocessConfig  # type: ignore[import-untyped,no-redef]
    except ImportError:
        SubprocessConfig = None  # type: ignore[misc,assignment]

    # Optional SDK types for backward compatibility
    # SDK < 0.1.28 doesn't have PermissionRequestResult
    try:
        from copilot.types import PermissionRequestResult  # type: ignore[import-untyped,no-redef]
    except ImportError:
        # Provide a stub type for older SDK versions
        PermissionRequestResult = None  # type: ignore[misc,assignment]

# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "CopilotClient",
    "PermissionRequestResult",
    "SubprocessConfig",
    "get_copilot_spec_origin",  # Re-export from _spec_utils
]
