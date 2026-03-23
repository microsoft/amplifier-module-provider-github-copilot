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
from typing import Any

# Re-export SDK-independent utilities for backward compatibility.
# New code should import directly from sdk_adapter (the membrane).
from ._spec_utils import get_copilot_spec_origin

# =============================================================================
# SDK imports - THE ONLY PLACE IN THE CODEBASE where SDK is imported
# =============================================================================

# TESTING: Set SKIP_SDK_CHECK=1 to allow imports without SDK installed.
# Tests use pytest.importorskip() and skip markers to handle SDK availability.
# This matches the pattern in __init__.py for consistent test behavior.
_SKIP_SDK_CHECK = os.environ.get("SKIP_SDK_CHECK")

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
    "get_copilot_spec_origin",  # Re-export from _spec_utils
]
