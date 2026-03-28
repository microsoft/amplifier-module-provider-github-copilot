"""Behavioral tests inherited from amplifier-core validation suite.

Contract: provider-protocol.md

These tests inherit from ProviderBehaviorTests which provides 5 standard tests:
1. test_mount_succeeds - mount() works with MockCoordinator
2. test_get_info_returns_valid_provider_info - ProviderInfo has required fields
3. test_list_models_returns_list - list_models() returns a list
4. test_provider_has_name_attribute - name property exists
5. test_parse_tool_calls_returns_list - parse_tool_calls() returns list

The amplifier-core pytest plugin auto-detects our module from directory name
(amplifier_module_provider_github_copilot) and provides the `provider_module` fixture.
"""

from __future__ import annotations

from amplifier_core.validation.behavioral import ProviderBehaviorTests


class TestGitHubCopilotBehavior(ProviderBehaviorTests):
    """Automatic protocol behavior tests.

    All 5 test methods are inherited from ProviderBehaviorTests.
    The plugin provides the provider_module fixture by:
    1. Detecting module type from directory name pattern
    2. Creating a MockCoordinator
    3. Calling mount(coordinator, config=None)
    4. Extracting the mounted provider instance
    """

    pass
