"""Tier 6: SDK Assumption Tests - verify SDK types and shapes without API calls.

These tests import the real SDK, instantiate objects, and verify our structural
assumptions. They require the SDK to be installed but do NOT make API calls.

Contract references:
- contracts/sdk-boundary.md
- contracts/deny-destroy.md

Run: pytest -m sdk_assumption -v
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest


@pytest.mark.sdk_assumption
class TestSDKImportAssumptions:
    """Verify SDK module structure matches our assumptions.

    AC-1: SDK Import Assumptions
    """

    def test_copilot_client_class_exists(self, sdk_module: Any) -> None:
        """We assume copilot.CopilotClient exists and is importable.

        # Contract: sdk-boundary:Lifecycle:MUST:1
        """
        assert isinstance(sdk_module.CopilotClient, type)

    def test_client_has_create_session(self, sdk_module: Any) -> None:
        """We assume CopilotClient has create_session method.

        # Contract: sdk-boundary:Session:MUST:1
        """
        assert inspect.iscoroutinefunction(sdk_module.CopilotClient.create_session)

    def test_client_has_start_stop(self, sdk_module: Any) -> None:
        """We assume CopilotClient has start() and stop() lifecycle methods.

        # Contract: sdk-boundary:Lifecycle:MUST:1
        """
        assert inspect.iscoroutinefunction(sdk_module.CopilotClient.start)
        assert inspect.iscoroutinefunction(sdk_module.CopilotClient.stop)

    def test_subprocess_config_importable(self, sdk_module: Any) -> None:
        """SubprocessConfig must be importable (multi-version fallback chain).

        # Contract: sdk-boundary:Auth:MUST:1

        The _imports.py module tries multiple import paths for SubprocessConfig.
        If ALL fallback paths fail, authentication silently breaks at runtime.
        """
        # Try the same import paths as _imports.py to verify SDK exposes SubprocessConfig
        subprocess_config_cls = None
        try:
            from copilot.types import SubprocessConfig as SC  # type: ignore[import-untyped]

            subprocess_config_cls = SC
        except ImportError:
            try:
                from copilot import SubprocessConfig as SC  # type: ignore[import-untyped]

                subprocess_config_cls = SC
            except ImportError:
                pass

        assert isinstance(subprocess_config_cls, type), (
            "SubprocessConfig must be importable from copilot.types or copilot root"
        )
        # Instantiate and verify a known field
        instance = subprocess_config_cls(github_token="test-token")
        assert instance.github_token == "test-token"



