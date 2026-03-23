"""SDK-independent utilities.

These functions work WITHOUT importing the SDK, using only
importlib metadata inspection.

Contract: sdk-boundary:BinaryResolution:MUST:2 — find_spec not import
"""

from __future__ import annotations

import importlib.util

__all__ = ["get_copilot_spec_origin"]


def get_copilot_spec_origin() -> str | None:
    """Locate copilot package origin path without importing it.

    Uses importlib.util.find_spec() to locate the package path
    without triggering SDK imports. This enables binary discovery
    even when SDK types haven't been loaded.

    Contract: sdk-boundary:BinaryResolution:MUST:2
    Contract: sdk-boundary:Membrane:MUST:1

    Returns:
        Path to copilot/__init__.py if found, None otherwise.
    """
    spec = importlib.util.find_spec("copilot")
    if spec is None or spec.origin is None:
        return None
    return spec.origin
