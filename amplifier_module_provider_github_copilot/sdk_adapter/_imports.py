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
from types import SimpleNamespace
from typing import Any

# Single source of truth for pytest detection (defined in _platform.py).
from .._platform import is_pytest_running

# Re-export SDK-independent utilities for backward compatibility.
# New code should import directly from sdk_adapter (the membrane).
from ._spec_utils import get_copilot_spec_origin

# =============================================================================
# SDK imports - THE ONLY PLACE IN THE CODEBASE where SDK is imported
# =============================================================================

# Only skip SDK imports if BOTH conditions are met:
# 1. SKIP_SDK_CHECK env var is set
# 2. pytest is actually running (convenience guard, NOT a security boundary)
_SKIP_SDK_CHECK = os.environ.get("SKIP_SDK_CHECK") and is_pytest_running()

CopilotClient: Any
PermissionDecisionReject: Any
PermissionRequestResult: Any
ModelCapabilitiesOverride: Any
ModelLimitsOverride: Any
ModelsListRequest: Any

if _SKIP_SDK_CHECK:
    # Test mode: provide None stubs that tests can mock
    CopilotClient = None  # type: ignore[misc,assignment]
    PermissionDecisionReject = None  # type: ignore[misc,assignment]
    PermissionRequestResult = None  # type: ignore[misc,assignment]
    ModelCapabilitiesOverride = None  # type: ignore[misc,assignment]
    ModelLimitsOverride = None  # type: ignore[misc,assignment]
    ModelsListRequest = None  # type: ignore[misc,assignment]
else:
    try:
        from copilot import CopilotClient  # type: ignore[import-untyped,no-redef]
    except ImportError as e:
        raise ImportError(
            "github-copilot-sdk not installed. Install with: pip install github-copilot-sdk"
        ) from e

    # Per-session capability overrides (e.g. max_output_tokens cap) used by
    # client.session() to honor ChatRequest.max_tokens.
    # Contract: provider-protocol:complete:MUST:10
    from copilot import (  # type: ignore[import-untyped,no-redef]
        ModelCapabilitiesOverride,
        ModelLimitsOverride,
    )

    # TODO(maintainer 2026): generated.rpc is part of the SDK boundary surface per ImportQuarantine:MUST:7 (carve-out).  # noqa: E501
    from copilot.generated.rpc import (  # type: ignore[import-untyped,no-redef]
        PermissionDecisionReject,
    )
    from copilot.rpc import ModelsListRequest  # type: ignore[import-untyped,no-redef]

    # b10 type alias: PermissionDecision | PermissionNoResult. Not a constructor.
    # Contract: sdk-boundary:SDKSurface:MUST:1a
    from copilot.session import (  # type: ignore[import-untyped]
        PermissionRequestResult,  # type: ignore[no-redef]
    )

# =============================================================================
# Exports
# =============================================================================


def make_permission_denied() -> Any:
    """Return a permission decision that rejects the SDK's request.

    Contract: sdk-boundary:SDKSurface:MUST:1b, ImportQuarantine:MUST:8.
    The b10 variant `PermissionDecisionReject` has `kind` as a codegen ClassVar
    and a default `feedback=None` that `to_dict()` omits — preserving the
    silent-reject wire shape used by earlier SDK releases.

    Test-mode fallback (SKIP_SDK_CHECK=1 sets `PermissionDecisionReject = None`)
    returns a structural stand-in matching the attributes the SDK serializer
    reads on the production object.
    """
    if PermissionDecisionReject is not None:
        return PermissionDecisionReject()  # type: ignore[return-value]
    return SimpleNamespace(kind="reject", feedback=None)


def make_models_list_request() -> Any:
    """Build the SDK's public typed model-discovery request inside the membrane."""
    if ModelsListRequest is not None:
        return ModelsListRequest()
    return SimpleNamespace()  # Test-mode stand-in; mocked RPC ignores the request.


# SDK v1.0.0b10 (and b9 before it) keeps SubprocessConfig out of the public
# surface (was removed in b7; confirmed absent from b10 copilot/__init__.py
# and copilot/client.py). Quarantined as None — intentional fail-closed shape:
# caller truthiness gates (``if SubprocessConfig is not None:``) no-op safely,
# and any caller that attempts ``SubprocessConfig(...)`` raises TypeError at
# the membrane rather than ImportError deep in user code. Removing this
# symbol would require a corresponding MIGRATION.md entry (per the project's
# public-surface change policy: removed/renamed public symbols ship with a
# what / when / behavior / replacement / rollback note in MIGRATION.md in the
# same diff) and an sdk-boundary contract amendment.
# Contract: sdk-boundary:ImportQuarantine:MUST:6 + History §1.7.
# Pinned by: tests/test_sdk_boundary_quarantine.py::TestSDKImportsRealPath
#   (test_subprocess_config_is_quarantined_from_sdk_surface,
#    test_subprocess_config_absent_from_sdk).
SubprocessConfig: Any = None


__all__ = [
    "CopilotClient",
    "ModelCapabilitiesOverride",
    "ModelLimitsOverride",
    "PermissionDecisionReject",
    "PermissionRequestResult",
    "SubprocessConfig",
    "make_permission_denied",
    "get_copilot_spec_origin",  # Re-export from _spec_utils
]
