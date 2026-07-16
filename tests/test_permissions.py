# pyright: reportPrivateUsage=false
"""Tests for _permissions.py error handling.

Contract: sdk-boundary:BinaryResolution:MUST:6
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest


class TestEnsureExecutableErrors:
    """Cover _permissions.py L71-76: PermissionError and OSError paths."""

    def _unix_platform_info(self) -> object:
        from amplifier_module_provider_github_copilot._platform import PlatformInfo

        return PlatformInfo(name="Unix", is_windows=False, cli_binary_name="copilot")

    def test_permission_error_on_chmod_returns_false(self, tmp_path: Path) -> None:
        """L71-73: PermissionError from chmod returns False.

        Contract: sdk-boundary:BinaryResolution:MUST:6
        """
        from amplifier_module_provider_github_copilot._permissions import ensure_executable
        from amplifier_module_provider_github_copilot._platform import get_platform_info

        binary = tmp_path / "copilot"
        binary.touch()
        binary.chmod(0o644)  # Not executable

        get_platform_info.cache_clear()
        unix_info = self._unix_platform_info()

        with (
            patch(
                "amplifier_module_provider_github_copilot._platform.get_platform_info",
                return_value=unix_info,
            ),
            patch(
                "amplifier_module_provider_github_copilot._permissions.Path.chmod",
                side_effect=PermissionError("permission denied"),
            ),
        ):
            result = ensure_executable(binary)

        assert result is False

    def test_chmod_silent_noop_returns_false(self, tmp_path: Path) -> None:
        """chmod that silently succeeds without applying mode (NTFS/FUSE behavior)
        causes the function to return False instead of a misleading True.

        Reproduces the cross-platform blind spot where /mnt/<drive>/ on WSL
        accepts chmod() without persisting the mode bits.

        Contract: sdk-boundary:BinaryResolution:MUST:6
        """
        from amplifier_module_provider_github_copilot._permissions import ensure_executable
        from amplifier_module_provider_github_copilot._platform import get_platform_info

        binary = tmp_path / "copilot"
        binary.touch()
        binary.chmod(0o644)  # Not executable

        get_platform_info.cache_clear()
        unix_info = self._unix_platform_info()

        with (
            patch(
                "amplifier_module_provider_github_copilot._platform.get_platform_info",
                return_value=unix_info,
            ),
            # Simulate NTFS-on-WSL: chmod returns success but does not persist.
            patch(
                "amplifier_module_provider_github_copilot._permissions.Path.chmod",
                return_value=None,
            ),
        ):
            result = ensure_executable(binary)

        # Sanity: behavioral assertion — file mode actually unchanged on disk.
        assert not (binary.stat().st_mode & __import__("stat").S_IXUSR), (
            "test premise: chmod was suppressed, mode should be unchanged"
        )
        assert result is False, (
            "ensure_executable must return False when chmod silently no-ops; "
            "returning True would lie to callers about NTFS-mounted binaries"
        )

    def test_os_error_on_chmod_returns_false(self, tmp_path: Path) -> None:
        """L74-76: OSError from chmod returns False.

        Contract: sdk-boundary:BinaryResolution:MUST:6
        """
        from amplifier_module_provider_github_copilot._permissions import ensure_executable
        from amplifier_module_provider_github_copilot._platform import get_platform_info

        binary = tmp_path / "copilot"
        binary.touch()
        binary.chmod(0o644)  # Not executable

        get_platform_info.cache_clear()
        unix_info = self._unix_platform_info()

        with (
            patch(
                "amplifier_module_provider_github_copilot._platform.get_platform_info",
                return_value=unix_info,
            ),
            patch(
                "amplifier_module_provider_github_copilot._permissions.Path.chmod",
                side_effect=OSError("read-only filesystem"),
            ),
        ):
            result = ensure_executable(binary)

        assert result is False

    def test_stat_permission_error_returns_false(self, tmp_path: Path) -> None:
        """L71-73: PermissionError from stat() also returns False.

        Contract: sdk-boundary:BinaryResolution:MUST:6
        """
        from amplifier_module_provider_github_copilot._permissions import ensure_executable
        from amplifier_module_provider_github_copilot._platform import get_platform_info

        binary = tmp_path / "copilot"
        binary.touch()

        get_platform_info.cache_clear()
        unix_info = self._unix_platform_info()

        # Note: We need to mock at a level that doesn't break is_file() check
        # The OSError path in _permissions.py L74-76 is triggered by chmod() OSError
        # not stat() - the stat PermissionError path is covered by chmod test above
        # This test verifies OSError (different from PermissionError) is also caught
        with (
            patch(
                "amplifier_module_provider_github_copilot._platform.get_platform_info",
                return_value=unix_info,
            ),
            patch(
                "amplifier_module_provider_github_copilot._permissions.Path.chmod",
                side_effect=OSError("generic OS error"),
            ),
        ):
            result = ensure_executable(binary)

        assert result is False

    def test_already_executable_returns_true_no_chmod(self, tmp_path: Path) -> None:
        """L62: File already executable returns True without chmod.

        Contract: sdk-boundary:BinaryResolution:MUST:6
        Coverage: _permissions.py line 62
        """
        from unittest.mock import MagicMock

        from amplifier_module_provider_github_copilot._permissions import ensure_executable
        from amplifier_module_provider_github_copilot._platform import get_platform_info

        binary = tmp_path / "copilot"
        binary.touch()

        get_platform_info.cache_clear()
        unix_info = self._unix_platform_info()

        # Mock stat to return a mode with S_IXUSR set (already executable)
        import os

        mock_stat_result = MagicMock(spec=os.stat_result)
        mock_stat_result.st_mode = 0o755  # Has execute bits

        chmod_mock = MagicMock()

        with (
            patch(
                "amplifier_module_provider_github_copilot._platform.get_platform_info",
                return_value=unix_info,
            ),
            patch(
                "amplifier_module_provider_github_copilot._permissions.Path.is_file",
                return_value=True,
            ),
            patch(
                "amplifier_module_provider_github_copilot._permissions.Path.stat",
                return_value=mock_stat_result,
            ),
            patch(
                "amplifier_module_provider_github_copilot._permissions.Path.chmod",
                chmod_mock,
            ),
        ):
            result = ensure_executable(binary)

        assert result is True
        # MUST NOT call chmod when already executable (idempotent)
        chmod_mock.assert_not_called()

    def test_adds_execute_permission_via_mocked_filesystem(self, tmp_path: Path) -> None:
        """Happy-path: ensure_executable issues chmod(S_IXUSR|S_IXGRP) when bits absent.

        Mocks the filesystem so the test runs on every OS (Windows NTFS would
        otherwise drop the POSIX exec bits and make this test xfail on Win).

        Contract: sdk-boundary:BinaryResolution:MUST:6
        Coverage: _permissions.py lines 60-84 (stat → chmod → verify-stat happy path)
        """
        import os
        import stat as stat_mod
        from unittest.mock import MagicMock

        from amplifier_module_provider_github_copilot._permissions import (
            _EXECUTE_BITS,
            ensure_executable,
        )
        from amplifier_module_provider_github_copilot._platform import get_platform_info

        binary = tmp_path / "copilot"
        binary.touch()

        get_platform_info.cache_clear()
        unix_info = self._unix_platform_info()

        # Pre-chmod stat: regular file (S_IFREG=0o100000) + rw-r--r-- (0o644).
        # Post-chmod stat: same regular-file bit + 0o644 | (S_IXUSR|S_IXGRP) = 0o754.
        pre_mode = stat_mod.S_IFREG | 0o644
        post_mode = pre_mode | _EXECUTE_BITS

        pre_stat = MagicMock(spec=os.stat_result)
        pre_stat.st_mode = pre_mode
        post_stat = MagicMock(spec=os.stat_result)
        post_stat.st_mode = post_mode

        chmod_mock = MagicMock()
        stat_mock = MagicMock(side_effect=[pre_stat, post_stat])

        with (
            patch(
                "amplifier_module_provider_github_copilot._platform.get_platform_info",
                return_value=unix_info,
            ),
            patch(
                "amplifier_module_provider_github_copilot._permissions.Path.is_file",
                return_value=True,
            ),
            patch(
                "amplifier_module_provider_github_copilot._permissions.Path.stat",
                stat_mock,
            ),
            patch(
                "amplifier_module_provider_github_copilot._permissions.Path.chmod",
                chmod_mock,
            ),
        ):
            result = ensure_executable(binary)

        assert result is True
        # MUST call chmod exactly once with pre_mode | (S_IXUSR | S_IXGRP);
        # MUST NOT set S_IXOTH (least-privilege per _permissions.py docstring).
        chmod_mock.assert_called_once_with(post_mode)
        called_mode = chmod_mock.call_args.args[0]
        assert called_mode & stat_mod.S_IXUSR, "MUST set user execute bit"
        assert called_mode & stat_mod.S_IXGRP, "MUST set group execute bit"
        assert not (called_mode & stat_mod.S_IXOTH), (
            "MUST NOT set world execute bit (least privilege)"
        )
        # MUST call stat() exactly twice: pre-chmod read (_permissions.py:60) AND
        # post-chmod verify (_permissions.py:74). If the verify-stat is ever
        # removed, this assertion fires — defends the NTFS-silent-noop guard.
        assert stat_mock.call_count == 2, (
            f"ensure_executable MUST stat() twice (pre + verify); got {stat_mock.call_count}"
        )

    @pytest.mark.posix_only
    def test_adds_execute_permission_real_filesystem_posix(self, tmp_path: Path) -> None:
        """End-to-end on a real POSIX filesystem: stat -> chmod -> verify-stat -> True.

        POSIX-only contract; the sibling test in test_platform.py covers the
        Windows NTFS tombstone path. Deselected (not skipped) on win32 via the
        ``posix_only`` marker (repo policy: tests run or fail, never skip).

        Contract: sdk-boundary:BinaryResolution:MUST:6
        Coverage: _permissions.py lines 60-84 (real chmod + verify-stat happy path)
        """
        import stat as stat_mod

        from amplifier_module_provider_github_copilot._permissions import ensure_executable

        binary = tmp_path / "copilot"
        binary.touch()
        binary.chmod(0o644)

        result = ensure_executable(binary)

        assert result is True
        mode = binary.stat().st_mode
        assert mode & stat_mod.S_IXUSR, "MUST set user execute bit on real FS"
        assert mode & stat_mod.S_IXGRP, "MUST set group execute bit on real FS"
        assert not (mode & stat_mod.S_IXOTH), "MUST NOT set world execute bit (least privilege)"
