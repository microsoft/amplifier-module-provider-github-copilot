"""Tests for on_permission_request handler in production path.

Verifies permission handler is present in production CopilotClientWrapper.
Includes SDK version drift detection tests.

Contract: contracts/deny-destroy.md
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest


@pytest.mark.asyncio
class TestOnPermissionRequestHandler:
    """SDK v0.1.33: CopilotClientWrapper must pass on_permission_request.

    AC: Test that auto-init path includes on_permission_request.
    """

    async def test_deny_permission_request_is_used_in_options(self) -> None:
        """Production code must set on_permission_request to deny_permission_request.

        deny-destroy:PermissionRequest:MUST:1

        This test verifies by inspecting the source code that on_permission_request
        is added to the options dict in the session() method.
        """
        import inspect

        from amplifier_module_provider_github_copilot.sdk_adapter import client

        # Get the source code of the session method
        source = inspect.getsource(client.CopilotClientWrapper.session)

        # Verify on_permission_request is added to options
        assert "on_permission_request" in source, (
            "session() method must add on_permission_request to options"
        )
        assert "deny_permission_request" in source, (
            "on_permission_request must be set to deny_permission_request"
        )

    async def test_permission_handler_denies_all_requests(self) -> None:
        """Permission handler must deny all requests (Deny+Destroy pattern).

        deny-destroy:PermissionRequest:MUST:2
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            deny_permission_request,
        )

        # Mock a permission request
        mock_request = MagicMock()

        result = deny_permission_request(mock_request)

        # Result can be PermissionRequestResult (with .kind) or dict fallback
        if hasattr(result, "kind"):
            assert result.kind == "denied-by-rules", (
                "Permission request must be denied with kind='denied-by-rules'"
            )
        else:
            assert result["kind"] == "denied-by-rules", (
                "Permission request must be denied with kind='denied-by-rules'"
            )

    async def test_deny_permission_request_function_exists(self) -> None:
        """A deny_permission_request function must exist.

        deny-destroy:PermissionRequest:MUST:1
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            deny_permission_request,
        )

        assert callable(deny_permission_request)


class TestSystemNotificationClassification:
    """system.notification must be explicitly classified."""

    def test_system_notification_classified_as_consume(self) -> None:
        """system_notification must be classified as CONSUME (not unknown).

        event-vocabulary:classification:system_notification:MUST:1
        """
        from amplifier_module_provider_github_copilot.streaming import (
            EventClassification,
            classify_event,
            load_event_config,
        )

        config = load_event_config()
        result = classify_event("system_notification", config)
        assert result == EventClassification.CONSUME, (
            "system_notification should be CONSUME, not unknown"
        )

    def test_system_notification_no_warning_logged(self, caplog: Any) -> None:
        """system_notification should not produce 'Unknown SDK event type' warning."""
        import logging

        from amplifier_module_provider_github_copilot.streaming import (
            classify_event,
            load_event_config,
        )

        config = load_event_config()
        with caplog.at_level(logging.WARNING):
            classify_event("system_notification", config)

        assert "Unknown SDK event type: system_notification" not in caplog.text


@pytest.mark.sdk_assumption
class TestSDKVersionCompatibility:
    """Detect SDK version drift from v0.1.33 baseline."""

    def test_sdk_version_is_known(self) -> None:
        """SDK package version must be accessible and within expected range."""
        import importlib.metadata

        try:
            version = importlib.metadata.version("github-copilot-sdk")
        except importlib.metadata.PackageNotFoundError:
            pytest.skip("SDK package metadata not available")
            return  # Make pyright happy about version being bound

        # Document baseline — update when upgrading SDK
        parts = version.split(".")
        major, minor = int(parts[0]), int(parts[1])
        assert (major, minor) >= (0, 1), f"SDK version {version} below baseline 0.1"
        # Print for drift visibility in CI
        print(f"SDK version: {version}")

    def test_sdk_has_permission_request_result_type(self) -> None:
        """SDK v0.1.33: PermissionRequestResult type must exist."""
        try:
            from copilot.types import PermissionRequestResult  # type: ignore[import-untyped]

            assert PermissionRequestResult is not None
        except ImportError:
            pytest.skip("SDK not installed or PermissionRequestResult not available")

    def test_permission_request_result_has_kind_field(self) -> None:
        """SDK v0.1.33: PermissionRequestResult must accept kind parameter."""
        try:
            from copilot.types import PermissionRequestResult  # type: ignore[import-untyped]

            # Try creating with kind parameter
            result = PermissionRequestResult(
                kind="denied-by-rules",
                message="Test denial",
            )
            assert result.kind == "denied-by-rules"
        except ImportError:
            pytest.skip("SDK not installed")
        except TypeError as e:
            pytest.fail(f"PermissionRequestResult signature changed: {e}")
