# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Microsoft Corporation
"""Compatibility layer for amplifier_core runtime imports.

Three-Medium Architecture:
- This module provides fallback implementations for types from amplifier_core
- amplifier_core is provided by Amplifier runtime, not installed as a dependency
- All modules in this package should import from here, not directly from amplifier_core

Contract: sdk-boundary:Membrane:MUST:1 — Single import point for runtime dependencies.
"""

from __future__ import annotations

__all__ = ["ConfigurationError"]


# =============================================================================
# ConfigurationError
# =============================================================================
# Import from amplifier_core at runtime (provided by Amplifier runtime)
# Fallback for standalone testing without amplifier_core

try:
    from amplifier_core.llm_errors import (
        ConfigurationError,  # pyright: ignore[reportAssignmentType]
    )
except ImportError:  # pragma: no cover
    # Fallback for standalone testing without amplifier_core
    class ConfigurationError(Exception):  # type: ignore[no-redef]
        """Fallback ConfigurationError when amplifier_core unavailable.

        This matches the amplifier_core.llm_errors.ConfigurationError signature.
        Contract: error-hierarchy:ConfigurationError:MUST:1 — Fail-fast on config issues.
        """

        def __init__(
            self,
            message: str,
            *,
            provider: str = "github-copilot",
            **kwargs: object,
        ) -> None:
            super().__init__(message)
            self.provider = provider
