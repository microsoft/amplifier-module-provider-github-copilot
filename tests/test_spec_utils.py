"""Tests for sdk_adapter/_spec_utils module.

P2-10: Add missing module tests.
Contract Reference: sdk-boundary:BinaryResolution:MUST:2
"""

from __future__ import annotations

import importlib.machinery  # noqa: F401  # Used in MagicMock spec=
from unittest.mock import MagicMock, patch


class TestGetCopilotSpecOrigin:
    """Test get_copilot_spec_origin() function."""

    def test_returns_origin_when_copilot_installed(self) -> None:
        """Returns origin path when copilot package is installed.

        Contract: sdk-boundary:BinaryResolution:MUST:2 — find_spec not import
        """
        from amplifier_module_provider_github_copilot.sdk_adapter._spec_utils import (
            get_copilot_spec_origin,
        )

        # If copilot SDK is installed, we should get a path
        # If not installed, we should get None
        result = get_copilot_spec_origin()

        # Contract: sdk-boundary:BinaryResolution:MUST:2
        # Result is a valid path string (copilot SDK is installed in this env)
        assert isinstance(result, str)
        assert "copilot" in result.lower()
        assert result.lower().endswith("__init__.py")

    def test_returns_none_when_package_not_found(self) -> None:
        """Returns None when package doesn't exist.

        Contract: sdk-boundary:BinaryResolution:MUST:2
        """
        import importlib.util

        # Mock find_spec to return None (package not found)
        with patch.object(importlib.util, "find_spec", return_value=None):
            # Clear any cached result by reimporting
            import importlib

            from amplifier_module_provider_github_copilot.sdk_adapter import (
                _spec_utils,  # type: ignore[reportPrivateUsage]  # noqa: PLC2701
            )

            importlib.reload(_spec_utils)

            result = _spec_utils.get_copilot_spec_origin()

            assert result is None

    def test_uses_find_spec_not_import(self) -> None:
        """Uses find_spec() which doesn't trigger import.

        Contract: sdk-boundary:BinaryResolution:MUST:2 — find_spec not import
        """
        import importlib.util

        # Contract: sdk-boundary:BinaryResolution:MUST:2
        # Create a mock spec with origin
        mock_spec = MagicMock(spec=importlib.machinery.ModuleSpec)  # type: ignore[reportAttributeAccessIssue]
        mock_spec.origin = "/path/to/copilot/__init__.py"

        with patch.object(importlib.util, "find_spec", return_value=mock_spec) as mock_find:
            import importlib

            from amplifier_module_provider_github_copilot.sdk_adapter import (
                _spec_utils,  # type: ignore[reportPrivateUsage]  # noqa: PLC2701
            )

            importlib.reload(_spec_utils)

            result = _spec_utils.get_copilot_spec_origin()

            # Verify find_spec was called with 'copilot'
            mock_find.assert_called_with("copilot")
            assert result == "/path/to/copilot/__init__.py"

    def test_handles_spec_without_origin(self) -> None:
        """Returns None when spec has no origin attribute.

        Contract: sdk-boundary:BinaryResolution:MUST:2
        """
        import importlib.util

        # Contract: sdk-boundary:BinaryResolution:MUST:2
        # Mock spec exists but origin is None
        mock_spec = MagicMock(spec=importlib.machinery.ModuleSpec)  # type: ignore[reportAttributeAccessIssue]
        mock_spec.origin = None

        with patch.object(importlib.util, "find_spec", return_value=mock_spec):
            import importlib

            from amplifier_module_provider_github_copilot.sdk_adapter import (
                _spec_utils,  # type: ignore[reportPrivateUsage]  # noqa: PLC2701
            )

            importlib.reload(_spec_utils)

            result = _spec_utils.get_copilot_spec_origin()

            assert result is None
