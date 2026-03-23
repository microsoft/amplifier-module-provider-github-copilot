"""Structural tests inherited from amplifier-core validation suite.

Contract: provider-protocol.md

These tests inherit from ProviderStructuralTests which provides:
1. test_structural_validation - Runs ProviderValidator on module_path

The amplifier-core pytest plugin auto-detects our module from directory name
(amplifier_module_provider_github_copilot) and provides the `module_path` fixture.
"""

from __future__ import annotations

from amplifier_core.validation.structural import ProviderStructuralTests


class TestGitHubCopilotStructural(ProviderStructuralTests):
    """Automatic structural validation tests.

    The test method is inherited from ProviderStructuralTests.
    The plugin provides the module_path fixture by detecting
    the module directory from naming conventions.
    """

    pass
