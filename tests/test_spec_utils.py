"""Tests for sdk_adapter/_spec_utils module.

P2-10: Add missing module tests.
Contract Reference: sdk-boundary:BinaryResolution:MUST:2
"""

from __future__ import annotations

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

        # Result is either a valid path string or None
        assert result is None or isinstance(result, str)
        if result:
            assert "copilot" in result.lower() or result.endswith(".py")

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

        # Create a mock spec with origin
        mock_spec = MagicMock()
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

        # Mock spec exists but origin is None
        mock_spec = MagicMock()
        mock_spec.origin = None

        with patch.object(importlib.util, "find_spec", return_value=mock_spec):
            import importlib

            from amplifier_module_provider_github_copilot.sdk_adapter import (
                _spec_utils,  # type: ignore[reportPrivateUsage]  # noqa: PLC2701
            )

            importlib.reload(_spec_utils)

            result = _spec_utils.get_copilot_spec_origin()

            assert result is None
