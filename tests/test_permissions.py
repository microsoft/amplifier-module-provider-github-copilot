"""Tests for _permissions.py - file permission utilities.

TDD: These tests are written FIRST, before implementation.
Batch 2 - Security Hardening (v1.0.2)
"""

import os
import stat
import sys
from pathlib import Path

import pytest

# Skip tests that require Unix permissions on Windows
UNIX_ONLY = pytest.mark.skipif(
    sys.platform == "win32", reason="Unix permission bits (chmod, S_IXUSR) not supported on Windows"
)


class TestEnsureExecutable:
    """Tests for ensure_executable() function."""

    @UNIX_ONLY
    def test_ensure_executable_sets_user_execute(self, tmp_path: Path) -> None:
        """Should set S_IXUSR permission."""
        from amplifier_module_provider_github_copilot._permissions import ensure_executable

        script = tmp_path / "script"
        script.write_text("#!/bin/sh\necho hello")
        script.chmod(0o644)  # rw-r--r-- (no execute)

        result = ensure_executable(script)

        assert result is True
        mode = script.stat().st_mode
        assert mode & stat.S_IXUSR, "User execute bit should be set"

    @UNIX_ONLY
    def test_ensure_executable_sets_group_execute(self, tmp_path: Path) -> None:
        """Should set S_IXGRP permission."""
        from amplifier_module_provider_github_copilot._permissions import ensure_executable

        script = tmp_path / "script"
        script.write_text("#!/bin/sh\necho hello")
        script.chmod(0o644)

        result = ensure_executable(script)

        assert result is True
        mode = script.stat().st_mode
        assert mode & stat.S_IXGRP, "Group execute bit should be set"

    def test_ensure_executable_no_world_execute(self, tmp_path: Path) -> None:
        """Should NOT set S_IXOTH - least privilege principle."""
        from amplifier_module_provider_github_copilot._permissions import ensure_executable

        script = tmp_path / "script"
        script.write_text("#!/bin/sh\necho hello")
        script.chmod(0o644)

        ensure_executable(script)

        mode = script.stat().st_mode
        assert not (mode & stat.S_IXOTH), "World execute bit should NOT be set"

    def test_ensure_executable_idempotent(self, tmp_path: Path) -> None:
        """Calling twice should not change permissions."""
        from amplifier_module_provider_github_copilot._permissions import ensure_executable

        script = tmp_path / "script"
        script.write_text("#!/bin/sh\necho hello")
        script.chmod(0o755)  # Already executable

        # Call twice
        result1 = ensure_executable(script)
        mode1 = script.stat().st_mode
        result2 = ensure_executable(script)
        mode2 = script.stat().st_mode

        assert result1 is True
        assert result2 is True
        assert mode1 == mode2, "Permissions should not change on second call"

    def test_ensure_executable_returns_true_on_success(self, tmp_path: Path) -> None:
        """Should return True when chmod succeeds."""
        from amplifier_module_provider_github_copilot._permissions import ensure_executable

        script = tmp_path / "script"
        script.write_text("#!/bin/sh\necho hello")
        script.chmod(0o644)

        result = ensure_executable(script)

        assert result is True

    def test_ensure_executable_returns_true_when_already_executable(self, tmp_path: Path) -> None:
        """Should return True when file is already executable (no-op)."""
        from amplifier_module_provider_github_copilot._permissions import ensure_executable

        script = tmp_path / "script"
        script.write_text("#!/bin/sh\necho hello")
        script.chmod(0o755)  # Already executable

        result = ensure_executable(script)

        assert result is True

    def test_ensure_executable_returns_false_for_nonexistent_file(self) -> None:
        """Should return False when file doesn't exist."""
        from amplifier_module_provider_github_copilot._permissions import ensure_executable

        nonexistent = Path("/nonexistent/file/that/does/not/exist")

        result = ensure_executable(nonexistent)

        assert result is False

    def test_ensure_executable_preserves_existing_permissions(self, tmp_path: Path) -> None:
        """Should preserve existing read/write permissions."""
        from amplifier_module_provider_github_copilot._permissions import ensure_executable

        script = tmp_path / "script"
        script.write_text("#!/bin/sh\necho hello")
        script.chmod(0o644)  # rw-r--r--

        ensure_executable(script)

        mode = script.stat().st_mode
        # Original permissions should be preserved
        assert mode & stat.S_IRUSR, "User read should be preserved"
        assert mode & stat.S_IWUSR, "User write should be preserved"
        assert mode & stat.S_IRGRP, "Group read should be preserved"

    def test_ensure_executable_handles_path_as_string(self, tmp_path: Path) -> None:
        """Should work when passed a string path (via Path conversion)."""
        from amplifier_module_provider_github_copilot._permissions import ensure_executable

        script = tmp_path / "script"
        script.write_text("#!/bin/sh\necho hello")
        script.chmod(0o644)

        # Function signature takes Path, but callers might use strings
        result = ensure_executable(Path(str(script)))

        assert result is True
        assert os.access(script, os.X_OK)

    @UNIX_ONLY
    def test_ensure_executable_verifies_with_os_access(self, tmp_path: Path) -> None:
        """os.access(path, X_OK) should return True after ensure_executable."""
        from amplifier_module_provider_github_copilot._permissions import ensure_executable

        script = tmp_path / "script"
        script.write_text("#!/bin/sh\necho hello")
        script.chmod(0o644)
        assert not os.access(script, os.X_OK), "Precondition: not executable"

        ensure_executable(script)

        assert os.access(script, os.X_OK), "Should be executable after call"


class TestEdgeCases:
    """Edge case tests for error handling."""

    @UNIX_ONLY
    def test_ensure_executable_handles_oserror(self, tmp_path: Path, monkeypatch) -> None:
        """Should return False and log DEBUG on generic OSError."""
        from amplifier_module_provider_github_copilot._permissions import ensure_executable

        script = tmp_path / "script"
        script.write_text("#!/bin/sh")
        script.chmod(0o644)

        # Mock chmod to raise OSError (not PermissionError)
        def mock_chmod(*args, **kwargs):
            raise OSError("Mocked OSError")

        monkeypatch.setattr(Path, "chmod", mock_chmod)

        result = ensure_executable(script)

        assert result is False

    @UNIX_ONLY
    def test_ensure_executable_handles_permission_error(self, tmp_path: Path, monkeypatch) -> None:
        """Should return False on PermissionError."""
        from amplifier_module_provider_github_copilot._permissions import ensure_executable

        script = tmp_path / "script"
        script.write_text("#!/bin/sh")
        script.chmod(0o644)

        # Mock chmod to raise PermissionError
        def mock_chmod(*args, **kwargs):
            raise PermissionError("Mocked PermissionError")

        monkeypatch.setattr(Path, "chmod", mock_chmod)

        result = ensure_executable(script)

        assert result is False


class TestSecurityCompliance:
    """Security-focused tests for permission handling."""

    def test_s_ixoth_never_set(self, tmp_path: Path) -> None:
        """S_IXOTH (world execute) must NEVER be set - security requirement."""
        from amplifier_module_provider_github_copilot._permissions import ensure_executable

        # Test with various starting permissions
        test_modes = [0o600, 0o644, 0o640, 0o400, 0o444]

        for start_mode in test_modes:
            script = tmp_path / f"script_{start_mode:o}"
            script.write_text("#!/bin/sh")
            script.chmod(start_mode)

            ensure_executable(script)

            mode = script.stat().st_mode
            assert not (mode & stat.S_IXOTH), (
                f"S_IXOTH should not be set for start mode {start_mode:o}"
            )

    def test_world_execute_not_added_to_existing_permissions(self, tmp_path: Path) -> None:
        """Even if file has other world permissions, don't add execute."""
        from amplifier_module_provider_github_copilot._permissions import ensure_executable

        script = tmp_path / "script"
        script.write_text("#!/bin/sh")
        script.chmod(0o646)  # rw-r--rw- (world read+write, no execute)

        ensure_executable(script)

        mode = script.stat().st_mode
        assert not (mode & stat.S_IXOTH), (
            "World execute should not be added even if world has other perms"
        )
