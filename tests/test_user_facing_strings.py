"""
Tests for user-facing string constants.

This test file validates that user-facing strings in error messages are accurate.
These tests exist because prior to v1.0.2, the codebase had incorrect strings:
- `pip install copilot-sdk` should be `pip install github-copilot-sdk`
- `copilot auth login` should be `gh auth login` or GITHUB_TOKEN

TDD Gap Fixed: This is a class of bug that TDD would have caught IF tests
validated message content, not just exception types.

See /memories/repo/tdd-gap-user-facing-strings.md for the full analysis.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess

import pytest

# =============================================================================
# CONSTANTS VALIDATION TESTS
# =============================================================================


class TestUserFacingConstants:
    """Tests that validate the accuracy of user-facing constants."""

    def test_sdk_package_name_is_correct(self) -> None:
        """Verify SDK_PACKAGE_NAME matches the official PyPI package name."""
        from amplifier_module_provider_github_copilot._constants import SDK_PACKAGE_NAME

        # The official package name (verified from copilot-sdk/python/pyproject.toml)
        assert SDK_PACKAGE_NAME == "github-copilot-sdk"

    def test_sdk_package_name_follows_pypi_naming_convention(self) -> None:
        """Verify SDK_PACKAGE_NAME follows PyPI naming conventions."""
        from amplifier_module_provider_github_copilot._constants import SDK_PACKAGE_NAME

        # PyPI package names: lowercase, numbers, hyphens, underscores
        # Must start with alphanumeric
        assert re.match(r"^[a-z0-9][a-z0-9_-]*$", SDK_PACKAGE_NAME), (
            f"SDK_PACKAGE_NAME '{SDK_PACKAGE_NAME}' does not follow PyPI naming conventions"
        )

    def test_sdk_install_command_uses_package_name(self) -> None:
        """Verify SDK_INSTALL_COMMAND includes the correct package name."""
        from amplifier_module_provider_github_copilot._constants import (
            SDK_INSTALL_COMMAND,
            SDK_PACKAGE_NAME,
        )

        assert SDK_PACKAGE_NAME in SDK_INSTALL_COMMAND
        assert SDK_INSTALL_COMMAND == f"pip install {SDK_PACKAGE_NAME}"

    def test_auth_command_is_gh_auth_login(self) -> None:
        """Verify AUTH_COMMAND points to the GitHub CLI auth command."""
        from amplifier_module_provider_github_copilot._constants import AUTH_COMMAND

        # Must be the GitHub CLI auth command (not the bundled copilot CLI)
        assert AUTH_COMMAND == "gh auth login"

    def test_auth_env_var_is_github_token(self) -> None:
        """Verify AUTH_ENV_VAR is GITHUB_TOKEN."""
        from amplifier_module_provider_github_copilot._constants import AUTH_ENV_VAR

        # GITHUB_TOKEN is the standard env var for GitHub authentication
        assert AUTH_ENV_VAR == "GITHUB_TOKEN"

    def test_auth_instructions_contains_both_methods(self) -> None:
        """Verify AUTH_INSTRUCTIONS mentions both env var and command."""
        from amplifier_module_provider_github_copilot._constants import (
            AUTH_COMMAND,
            AUTH_ENV_VAR,
            AUTH_INSTRUCTIONS,
        )

        assert AUTH_ENV_VAR in AUTH_INSTRUCTIONS
        assert AUTH_COMMAND in AUTH_INSTRUCTIONS


# =============================================================================
# ERROR MESSAGE INTEGRATION TESTS
# =============================================================================


class TestErrorMessagesUseConstants:
    """Tests that error messages use the defined constants, not hardcoded strings.

    These tests verify:
    1. Source code uses the constants (not hardcoded strings)
    2. The constants have the correct values
    """

    def test_import_error_handler_uses_sdk_install_command_constant(self) -> None:
        """Verify client.py ImportError handler references SDK_INSTALL_COMMAND."""
        from pathlib import Path

        from amplifier_module_provider_github_copilot._constants import SDK_INSTALL_COMMAND

        # Read client.py source directly to avoid mock interference
        client_path = (
            Path(__file__).parent.parent / "amplifier_module_provider_github_copilot" / "client.py"
        )
        source = client_path.read_text(encoding="utf-8")

        # Verify it uses the constant, not a hardcoded string
        assert "SDK_INSTALL_COMMAND" in source, (
            "client.py should use SDK_INSTALL_COMMAND constant, not hardcoded string"
        )

        # Verify the constant has the correct package name
        assert "github-copilot-sdk" in SDK_INSTALL_COMMAND, (
            f"SDK_INSTALL_COMMAND should reference 'github-copilot-sdk', got: {SDK_INSTALL_COMMAND}"
        )

        # Verify it does NOT contain the old incorrect package name
        assert (
            "copilot-sdk" not in SDK_INSTALL_COMMAND or "github-copilot-sdk" in SDK_INSTALL_COMMAND
        ), "SDK_INSTALL_COMMAND should not use old 'pip install copilot-sdk' syntax"

    def test_auth_error_handler_uses_auth_constants(self) -> None:
        """Verify client.py auth error handler references AUTH_INSTRUCTIONS constant."""
        from pathlib import Path

        from amplifier_module_provider_github_copilot._constants import (
            AUTH_COMMAND,
            AUTH_ENV_VAR,
            AUTH_INSTRUCTIONS,
        )

        # Read client.py source directly to avoid mock interference
        client_path = (
            Path(__file__).parent.parent / "amplifier_module_provider_github_copilot" / "client.py"
        )
        source = client_path.read_text(encoding="utf-8")

        # Verify it uses the constant, not a hardcoded string
        assert "AUTH_INSTRUCTIONS" in source, (
            "client.py should use AUTH_INSTRUCTIONS constant for auth errors"
        )

        # Verify AUTH_INSTRUCTIONS contains the correct env var and command
        assert AUTH_ENV_VAR in AUTH_INSTRUCTIONS, f"AUTH_INSTRUCTIONS should contain {AUTH_ENV_VAR}"
        assert AUTH_COMMAND in AUTH_INSTRUCTIONS, f"AUTH_INSTRUCTIONS should contain {AUTH_COMMAND}"

        # Verify it does NOT contain old incorrect command
        assert "copilot auth login" not in AUTH_INSTRUCTIONS, (
            "AUTH_INSTRUCTIONS should not reference old 'copilot auth login' command"
        )


# =============================================================================
# LIVE VALIDATION TESTS (Optional, require network/CLI)
# =============================================================================


class TestLiveValidation:
    """Live validation tests that verify constants against real systems.

    These tests require network access or installed CLIs to run.
    They're marked with appropriate markers to skip in offline environments.
    """

    @pytest.mark.skipif(
        shutil.which("gh") is None,
        reason="GitHub CLI (gh) not installed",
    )
    def test_gh_auth_command_exists(self) -> None:
        """Verify that 'gh auth' is a valid subcommand."""
        from amplifier_module_provider_github_copilot._constants import AUTH_COMMAND

        # Extract the CLI and subcommand
        parts = AUTH_COMMAND.split()
        assert parts[0] == "gh"
        assert parts[1] == "auth"

        # Verify the command structure is valid (--help should work)
        result = subprocess.run(
            ["gh", "auth", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"'gh auth --help' failed: {result.stderr}"
        assert "login" in result.stdout.lower(), "'login' subcommand not found in gh auth help"

    @pytest.mark.live
    @pytest.mark.skipif(
        not os.environ.get("RUN_LIVE_TESTS"),
        reason="Requires RUN_LIVE_TESTS=1 (makes HTTP request to PyPI)",
    )
    def test_sdk_package_exists_on_pypi(self) -> None:
        """Verify SDK_PACKAGE_NAME exists on PyPI.

        This test makes a real HTTP request to PyPI.
        Skipped by default; set RUN_LIVE_TESTS=1 to enable.
        """
        import urllib.error
        import urllib.request

        from amplifier_module_provider_github_copilot._constants import SDK_PACKAGE_NAME

        url = f"https://pypi.org/pypi/{SDK_PACKAGE_NAME}/json"
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                assert response.status == 200
        except urllib.error.HTTPError as e:
            if e.code == 404:
                pytest.fail(f"Package '{SDK_PACKAGE_NAME}' not found on PyPI")
            raise


# =============================================================================
# REGRESSION TESTS
# =============================================================================


class TestNoHardcodedStrings:
    """Tests that ensure hardcoded strings are not reintroduced."""

    def test_no_hardcoded_copilot_sdk_package_name(self) -> None:
        """Verify 'copilot-sdk' (wrong) is not hardcoded in error messages."""
        import inspect

        from amplifier_module_provider_github_copilot import client

        source = inspect.getsource(client)

        # The OLD incorrect package name should NOT appear
        # (except in test-related comments, which is why we check specific patterns)
        assert 'pip install copilot-sdk"' not in source
        assert "pip install copilot-sdk'" not in source

    def test_no_hardcoded_copilot_auth_login(self) -> None:
        """Verify 'copilot auth login' (wrong) is not hardcoded in error messages."""
        import inspect

        from amplifier_module_provider_github_copilot import client, exceptions

        client_source = inspect.getsource(client)
        exceptions_source = inspect.getsource(exceptions)

        # The OLD incorrect command should NOT appear in code
        # (may appear in comments/docstrings explaining the history)
        for source in [client_source, exceptions_source]:
            # Check for the pattern in string literals
            assert 'copilot auth login"' not in source.replace("# ", "")
            assert "copilot auth login'" not in source.replace("# ", "")
