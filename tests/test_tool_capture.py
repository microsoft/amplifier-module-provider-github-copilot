"""
Tests for tool_capture module (Deny + Destroy pattern).

This module tests the tool bridge functions that enable external orchestration:
- convert_tools_for_sdk: Convert Amplifier ToolSpec to SDK Tool objects
- make_deny_all_hook: Create preToolUse hook that denies all tool execution

CRITICAL: Tests for overrides_built_in_tool flag (SDK 0.1.30+)
When registering tools that share names with SDK built-ins, the SDK requires
setting overrides_built_in_tool=True. Without this, SDK rejects with:
  "External tool 'glob' conflicts with a built-in tool of the same name."

See: mydocs/releases/hotfix-2026-03-07-windows/HOTFIX-PROVIDER-GITHUB-COPILOT.md
"""

import pytest

from unittest.mock import Mock
from unittest.mock import patch as unittest_mock_patch


# ═══════════════════════════════════════════════════════════════════════════════
# SDK 0.1.30+ OVERRIDE FLAG TESTS — HIGHEST PRIORITY
# ═══════════════════════════════════════════════════════════════════════════════
# These tests ensure that tools with built-in names have overrides_built_in_tool=True.
# This is the fix for the "glob conflicts with built-in" error discovered 2026-03-07.
# ═══════════════════════════════════════════════════════════════════════════════


class TestOverridesBuiltInToolFlag:
    """Tests for SDK 0.1.30+ overrides_built_in_tool flag.

    CRITICAL: These tests are the heartbeat of our tool registration.
    Without overrides_built_in_tool=True on conflicting tools, the SDK
    rejects registration with a runtime error.

    Background:
    - SDK 0.1.30 (PR #636) introduced enforcement of overridesBuiltInTool
    - Before: duplicate names caused "Tool names must be unique" error
    - After: SDK allows overrides but REQUIRES explicit flag

    Evidence: Incident 2026-03-07 Windows testing
    """

    # Amplifier tools that DIRECTLY conflict with SDK built-ins
    AMPLIFIER_CONFLICTING_TOOLS = [
        "glob",  # tool-filesystem exposes glob
        "bash",  # tool-bash exposes bash
        "grep",  # tool-search exposes grep
        "web_fetch",  # tool-web exposes web_fetch
        "web_search",  # tool-web exposes web_search
    ]

    @pytest.fixture
    def builtin_tool_names(self):
        """Load COPILOT_BUILTIN_TOOL_NAMES without full package import."""
        # Direct import from _constants to avoid provider.py import chain
        # which requires amplifier_core (not always available in test env)
        import importlib.util
        import os
        
        constants_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "amplifier_module_provider_github_copilot",
            "_constants.py"
        )
        spec = importlib.util.spec_from_file_location("_constants", constants_path)
        constants = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(constants)
        return constants.COPILOT_BUILTIN_TOOL_NAMES

    @pytest.mark.parametrize("tool_name", AMPLIFIER_CONFLICTING_TOOLS)
    def test_amplifier_conflicting_tools_have_override_flag_true(self, tool_name: str):
        """Amplifier tools that conflict with SDK built-ins MUST have override=True.

        This is the exact scenario that caused the 2026-03-07 incident.
        When Amplifier registers 'glob' (from tool-filesystem), the SDK must
        see overrides_built_in_tool=True or it rejects with a runtime error.
        """
        from amplifier_module_provider_github_copilot.tool_capture import convert_tools_for_sdk

        tool_spec = {"name": tool_name, "description": f"Amplifier {tool_name} tool"}
        result = convert_tools_for_sdk([tool_spec])

        assert len(result) == 1, f"Should produce exactly 1 tool for {tool_name}"
        tool = result[0]
        assert tool.name == tool_name
        assert tool.overrides_built_in_tool is True, (
            f"Tool '{tool_name}' MUST have overrides_built_in_tool=True. "
            f"This tool conflicts with an SDK built-in and without the flag, "
            f"SDK 0.1.30+ will reject it with: "
            f"'External tool \"{tool_name}\" conflicts with a built-in tool.'"
        )

    def test_all_28_builtins_have_override_flag_when_registered(self, builtin_tool_names):
        """ALL 28 known SDK built-ins should get override=True if registered.

        This ensures future tools using any built-in name are covered.
        The list is maintained in COPILOT_BUILTIN_TOOL_NAMES.
        """
        from amplifier_module_provider_github_copilot.tool_capture import convert_tools_for_sdk

        # Create tool specs for ALL built-ins
        tool_specs = [
            {"name": name, "description": f"Test {name}"}
            for name in sorted(builtin_tool_names)
        ]

        result = convert_tools_for_sdk(tool_specs)

        assert len(result) == len(builtin_tool_names), (
            f"Should produce {len(builtin_tool_names)} tools, got {len(result)}"
        )

        for tool in result:
            assert tool.overrides_built_in_tool is True, (
                f"Tool '{tool.name}' is in COPILOT_BUILTIN_TOOL_NAMES but "
                f"does not have overrides_built_in_tool=True"
            )

    def test_non_builtin_tools_have_override_flag_false(self):
        """Custom tools NOT in built-in list should have override=False.

        Setting overrides_built_in_tool=True for custom tools is harmless
        but indicates incorrect logic. We should only set it for conflicts.
        """
        from amplifier_module_provider_github_copilot.tool_capture import convert_tools_for_sdk

        custom_tools = [
            {"name": "delegate", "description": "Delegate to sub-agent"},
            {"name": "read_file", "description": "Read file contents"},
            {"name": "write_file", "description": "Write file contents"},
            {"name": "custom_tool_xyz", "description": "Custom tool"},
            {"name": "my_analyzer", "description": "Analysis tool"},
        ]

        result = convert_tools_for_sdk(custom_tools)

        assert len(result) == 5
        for tool in result:
            assert tool.overrides_built_in_tool is False, (
                f"Tool '{tool.name}' is NOT a built-in but has "
                f"overrides_built_in_tool=True. This indicates the logic "
                f"incorrectly identifies it as a built-in."
            )

    def test_mixed_builtin_and_custom_tools(self):
        """Mix of built-in and custom tool names should have correct flags."""
        from amplifier_module_provider_github_copilot.tool_capture import convert_tools_for_sdk

        tool_specs = [
            {"name": "glob", "description": "Built-in conflict"},
            {"name": "delegate", "description": "Custom tool"},
            {"name": "bash", "description": "Built-in conflict"},
            {"name": "read_file", "description": "Custom tool"},
            {"name": "grep", "description": "Built-in conflict"},
        ]

        result = convert_tools_for_sdk(tool_specs)

        # Build a dict for easier assertion
        override_flags = {t.name: t.overrides_built_in_tool for t in result}

        assert override_flags["glob"] is True
        assert override_flags["bash"] is True
        assert override_flags["grep"] is True
        assert override_flags["delegate"] is False
        assert override_flags["read_file"] is False

    def test_override_flag_with_object_style_spec(self):
        """overrides_built_in_tool should work with Mock object specs too."""
        from amplifier_module_provider_github_copilot.tool_capture import convert_tools_for_sdk

        spec = Mock()
        spec.name = "web_fetch"  # Built-in conflict
        spec.description = "Fetch web content"
        spec.parameters = {"type": "object"}

        result = convert_tools_for_sdk([spec])

        assert len(result) == 1
        assert result[0].overrides_built_in_tool is True

    def test_override_flag_logged_for_builtins(self):
        """A debug log should be emitted when override flag is set."""
        from amplifier_module_provider_github_copilot.tool_capture import convert_tools_for_sdk

        with unittest_mock_patch(
            "amplifier_module_provider_github_copilot.tool_capture.logger"
        ) as mock_logger:
            convert_tools_for_sdk([{"name": "glob", "description": "Test"}])

        # Should have logged about the override
        debug_calls = [str(c) for c in mock_logger.debug.call_args_list]
        assert any("glob" in c and "overrides" in c.lower() for c in debug_calls), (
            "Should log when a tool overrides a built-in"
        )


class TestConvertToolsForSdk:
    """Tests for convert_tools_for_sdk function.

    This function bridges Amplifier ToolSpec to SDK Tool objects.
    Key responsibility: deduplicate tool names to prevent Copilot API 400 errors.
    """

    def test_tool_deduplication_keeps_first(self):
        """Duplicate tool names should be deduplicated (first wins)."""
        from amplifier_module_provider_github_copilot.tool_capture import convert_tools_for_sdk

        # Mock ToolSpec objects with duplicate names
        tool1 = Mock()
        tool1.name = "write_file"
        tool1.description = "Write to a file (first)"
        tool1.parameters = {"type": "object"}

        tool2 = Mock()
        tool2.name = "read_file"
        tool2.description = "Read a file"
        tool2.parameters = {}

        tool3 = Mock()
        tool3.name = "write_file"  # Duplicate!
        tool3.description = "Write to a file (second)"
        tool3.parameters = {"type": "object", "extra": True}

        result = convert_tools_for_sdk([tool1, tool2, tool3])

        # Should only have 2 tools (duplicate removed)
        assert len(result) == 2
        names = [t.name for t in result]
        assert names == ["write_file", "read_file"]
        # First write_file should be kept
        assert result[0].description == "Write to a file (first)"

    def test_tool_deduplication_with_dict_specs(self):
        """Deduplication should work with dict-style tool specs."""
        from amplifier_module_provider_github_copilot.tool_capture import convert_tools_for_sdk

        tool_specs = [
            {"name": "search", "description": "Search v1"},
            {"name": "list", "description": "List items"},
            {"name": "search", "description": "Search v2"},  # Duplicate
            {"name": "list", "description": "List v2"},  # Duplicate
        ]

        result = convert_tools_for_sdk(tool_specs)

        assert len(result) == 2
        names = [t.name for t in result]
        assert "search" in names
        assert "list" in names
        # First occurrences kept
        assert result[0].description == "Search v1"
        assert result[1].description == "List items"

    def test_tool_deduplication_empty_input(self):
        """Empty tool list should return empty list."""
        from amplifier_module_provider_github_copilot.tool_capture import convert_tools_for_sdk

        result = convert_tools_for_sdk([])
        assert result == []

    def test_tool_deduplication_all_unique(self):
        """All unique tools should be preserved."""
        from amplifier_module_provider_github_copilot.tool_capture import convert_tools_for_sdk

        tool_specs = [
            {"name": "tool_a", "description": "A"},
            {"name": "tool_b", "description": "B"},
            {"name": "tool_c", "description": "C"},
        ]

        result = convert_tools_for_sdk(tool_specs)

        assert len(result) == 3
        assert [t.name for t in result] == ["tool_a", "tool_b", "tool_c"]

    def test_tool_no_name_skipped(self):
        """Tools without names should be skipped."""
        from amplifier_module_provider_github_copilot.tool_capture import convert_tools_for_sdk

        tool_specs = [
            {"name": "", "description": "No name"},
            {"name": "valid", "description": "Valid tool"},
        ]

        result = convert_tools_for_sdk(tool_specs)

        assert len(result) == 1
        assert result[0].name == "valid"


class TestMakeDenyAllHook:
    """Tests for make_deny_all_hook function.

    This function creates a preToolUse hook that denies all tool execution,
    which is essential for the Deny + Destroy pattern.
    """

    def test_returns_dict_with_on_pre_tool_use_key(self):
        """Should return a dict with 'on_pre_tool_use' key."""
        from amplifier_module_provider_github_copilot.tool_capture import make_deny_all_hook

        result = make_deny_all_hook()

        assert isinstance(result, dict)
        assert "on_pre_tool_use" in result
        assert callable(result["on_pre_tool_use"])

    def test_hook_returns_deny_decision(self):
        """Hook should return permissionDecision='deny'."""
        from amplifier_module_provider_github_copilot.tool_capture import make_deny_all_hook

        hooks = make_deny_all_hook()
        deny_hook = hooks["on_pre_tool_use"]

        # Call the hook with a mock input
        result = deny_hook({"toolName": "write_file"}, context=None)

        assert result["permissionDecision"] == "deny"
        assert "permissionDecisionReason" in result

    def test_hook_includes_tool_name_in_reason(self):
        """Denial reason should be minimal to avoid model learning."""
        from amplifier_module_provider_github_copilot.tool_capture import make_deny_all_hook

        hooks = make_deny_all_hook()
        deny_hook = hooks["on_pre_tool_use"]

        result = deny_hook({"toolName": "execute_code"}, context=None)

        reason = result["permissionDecisionReason"]
        # Reason is minimal "Processing" - we intentionally don't include
        # tool name because model learns from denial reasons and would
        # stop trying tools if it sees explanatory text
        assert reason == "Processing"

    def test_hook_handles_missing_tool_name(self):
        """Hook should gracefully handle missing toolName."""
        from amplifier_module_provider_github_copilot.tool_capture import make_deny_all_hook

        hooks = make_deny_all_hook()
        deny_hook = hooks["on_pre_tool_use"]

        # Call with empty dict (missing toolName)
        result = deny_hook({}, context=None)

        assert result["permissionDecision"] == "deny"
        # Reason is minimal "Processing" regardless of tool name
        assert result["permissionDecisionReason"] == "Processing"


# =========================================================================
# Additional coverage tests
# =========================================================================


class TestNoopToolHandler:
    """Tests for _noop_tool_handler (line 94)."""

    def test_returns_tool_result_dict(self):
        """Handler should return a ToolResult dict with textResultForLlm.

        SDK contract: ToolHandler = Callable[[dict], ToolResult]
        ToolResult is a TypedDict with textResultForLlm key.
        """
        from amplifier_module_provider_github_copilot.tool_capture import _noop_tool_handler

        result = _noop_tool_handler({"some": "args"})

        assert isinstance(result, dict)
        assert "textResultForLlm" in result
        assert "denied" in result["textResultForLlm"].lower()

    def test_returns_dict_for_any_input(self):
        """Handler result should always be a dict (ToolResult)."""
        from amplifier_module_provider_github_copilot.tool_capture import _noop_tool_handler

        result = _noop_tool_handler(None)
        assert isinstance(result, dict)
        assert "textResultForLlm" in result

    def test_ignores_arguments(self):
        """Handler should return the same result regardless of args."""
        from amplifier_module_provider_github_copilot.tool_capture import _noop_tool_handler

        result1 = _noop_tool_handler(None)
        result2 = _noop_tool_handler({"key": "value"})
        result3 = _noop_tool_handler("string_arg")

        assert result1 == result2 == result3


class TestConvertToolsForSdkExtended:
    """Additional tests for convert_tools_for_sdk covering uncovered branches."""

    def test_unknown_spec_type_skipped_with_warning(self):
        """Non-object, non-dict specs should be skipped with a warning."""
        from amplifier_module_provider_github_copilot.tool_capture import convert_tools_for_sdk

        with unittest_mock_patch(
            "amplifier_module_provider_github_copilot.tool_capture.logger"
        ) as mock_logger:
            result = convert_tools_for_sdk([42, "not_a_tool", 3.14])

        assert result == []
        # Should have warned 3 times (once per unknown spec)
        assert mock_logger.warning.call_count == 3

    def test_object_spec_with_none_description_defaults_to_empty(self):
        """Object spec with description=None should default to empty string."""
        from amplifier_module_provider_github_copilot.tool_capture import convert_tools_for_sdk

        spec = Mock()
        spec.name = "my_tool"
        spec.description = None
        spec.parameters = None

        result = convert_tools_for_sdk([spec])

        assert len(result) == 1
        assert result[0].name == "my_tool"
        assert result[0].description == ""

    def test_object_spec_without_description_attribute(self):
        """Object spec with no description attribute should default to empty."""
        from amplifier_module_provider_github_copilot.tool_capture import convert_tools_for_sdk

        spec = Mock(spec=["name"])  # Only has 'name' attribute
        spec.name = "minimal_tool"

        result = convert_tools_for_sdk([spec])

        assert len(result) == 1
        assert result[0].name == "minimal_tool"
        assert result[0].description == ""

    def test_object_spec_without_parameters_attribute(self):
        """Object spec with no parameters attribute should set parameters=None."""
        from amplifier_module_provider_github_copilot.tool_capture import convert_tools_for_sdk

        spec = Mock(spec=["name", "description"])
        spec.name = "no_params_tool"
        spec.description = "A tool"

        result = convert_tools_for_sdk([spec])

        assert len(result) == 1
        assert result[0].name == "no_params_tool"
        assert result[0].parameters is None

    def test_dict_spec_with_parameters(self):
        """Dict spec with parameters should pass them through."""
        from amplifier_module_provider_github_copilot.tool_capture import convert_tools_for_sdk

        params = {"type": "object", "properties": {"path": {"type": "string"}}}
        tool_specs = [{"name": "read_file", "description": "Read", "parameters": params}]

        result = convert_tools_for_sdk(tool_specs)

        assert len(result) == 1
        assert result[0].parameters == params

    def test_dict_spec_without_parameters_key(self):
        """Dict spec missing 'parameters' key should set parameters=None."""
        from amplifier_module_provider_github_copilot.tool_capture import convert_tools_for_sdk

        tool_specs = [{"name": "simple_tool", "description": "Simple"}]

        result = convert_tools_for_sdk(tool_specs)

        assert len(result) == 1
        assert result[0].parameters is None

    def test_dict_spec_missing_name_key(self):
        """Dict spec without 'name' key should be skipped (empty name)."""
        from amplifier_module_provider_github_copilot.tool_capture import convert_tools_for_sdk

        tool_specs = [{"description": "orphan"}]

        result = convert_tools_for_sdk(tool_specs)
        assert result == []

    def test_produced_tools_use_noop_handler(self):
        """All produced SDK Tool objects should use _noop_tool_handler."""
        from amplifier_module_provider_github_copilot.tool_capture import convert_tools_for_sdk

        tool_specs = [{"name": "test_tool", "description": "Test"}]
        result = convert_tools_for_sdk(tool_specs)

        assert len(result) == 1
        # Invoke the handler to verify it's the noop handler
        handler_result = result[0].handler({"arg": "val"})
        # Handler returns ToolResult dict per SDK contract
        assert isinstance(handler_result, dict)
        assert "textResultForLlm" in handler_result
        assert "denied" in handler_result["textResultForLlm"].lower()

    def test_mixed_object_and_dict_specs(self):
        """Should handle a mix of object specs, dict specs, and unknowns."""
        from amplifier_module_provider_github_copilot.tool_capture import convert_tools_for_sdk

        obj_spec = Mock()
        obj_spec.name = "obj_tool"
        obj_spec.description = "Object tool"
        obj_spec.parameters = {"type": "object"}

        dict_spec = {"name": "dict_tool", "description": "Dict tool"}

        result = convert_tools_for_sdk([obj_spec, 999, dict_spec, None])

        # obj_spec + dict_spec = 2 (999 and None skipped)
        assert len(result) == 2
        assert [t.name for t in result] == ["obj_tool", "dict_tool"]

    def test_debug_logging_on_conversion(self):
        """Should log the converted tool names at debug level."""
        from amplifier_module_provider_github_copilot.tool_capture import convert_tools_for_sdk

        with unittest_mock_patch(
            "amplifier_module_provider_github_copilot.tool_capture.logger"
        ) as mock_logger:
            convert_tools_for_sdk([{"name": "alpha", "description": "A"}])

        # Final debug log should mention converted tools
        debug_calls = [str(c) for c in mock_logger.debug.call_args_list]
        assert any("Converted 1 tool(s)" in c for c in debug_calls)

    def test_duplicate_debug_logging(self):
        """Should log debug message when skipping duplicates."""
        from amplifier_module_provider_github_copilot.tool_capture import convert_tools_for_sdk

        with unittest_mock_patch(
            "amplifier_module_provider_github_copilot.tool_capture.logger"
        ) as mock_logger:
            convert_tools_for_sdk(
                [
                    {"name": "dup", "description": "first"},
                    {"name": "dup", "description": "second"},
                ]
            )

        debug_calls = [str(c) for c in mock_logger.debug.call_args_list]
        assert any("Skipping duplicate tool: dup" in c for c in debug_calls)


class TestMakeDenyAllHookExtended:
    """Additional tests for make_deny_all_hook covering uncovered lines."""

    def test_hook_suppress_output_is_true(self):
        """Hook should set suppressOutput=True to prevent denial pollution."""
        from amplifier_module_provider_github_copilot.tool_capture import make_deny_all_hook

        hooks = make_deny_all_hook()
        result = hooks["on_pre_tool_use"]({"toolName": "write_file"}, context=None)

        assert result["suppressOutput"] is True

    def test_hook_returns_all_three_keys(self):
        """Hook result should contain exactly the 3 required keys."""
        from amplifier_module_provider_github_copilot.tool_capture import make_deny_all_hook

        hooks = make_deny_all_hook()
        result = hooks["on_pre_tool_use"]({"toolName": "test"}, context=None)

        assert set(result.keys()) == {
            "permissionDecision",
            "permissionDecisionReason",
            "suppressOutput",
        }

    def test_hook_debug_logs_tool_name(self):
        """Hook should log the denied tool name at debug level."""
        from amplifier_module_provider_github_copilot.tool_capture import make_deny_all_hook

        with unittest_mock_patch(
            "amplifier_module_provider_github_copilot.tool_capture.logger"
        ) as mock_logger:
            hooks = make_deny_all_hook()
            hooks["on_pre_tool_use"]({"toolName": "execute_cmd"}, context=None)

        mock_logger.debug.assert_called_once()
        assert "execute_cmd" in str(mock_logger.debug.call_args)

    def test_hook_uses_unknown_for_missing_tool_name(self):
        """Hook should default toolName to 'unknown' when missing."""
        from amplifier_module_provider_github_copilot.tool_capture import make_deny_all_hook

        with unittest_mock_patch(
            "amplifier_module_provider_github_copilot.tool_capture.logger"
        ) as mock_logger:
            hooks = make_deny_all_hook()
            hooks["on_pre_tool_use"]({}, context=None)

        assert "unknown" in str(mock_logger.debug.call_args)

    def test_hook_only_key_is_on_pre_tool_use(self):
        """make_deny_all_hook should return dict with only 'on_pre_tool_use'."""
        from amplifier_module_provider_github_copilot.tool_capture import make_deny_all_hook

        result = make_deny_all_hook()
        assert list(result.keys()) == ["on_pre_tool_use"]

    def test_hook_context_is_ignored(self):
        """Hook should work regardless of context value."""
        from amplifier_module_provider_github_copilot.tool_capture import make_deny_all_hook

        hooks = make_deny_all_hook()
        deny_hook = hooks["on_pre_tool_use"]

        # Various context values should all produce same result
        r1 = deny_hook({"toolName": "t"}, context=None)
        r2 = deny_hook({"toolName": "t"}, context={"session": "abc"})
        r3 = deny_hook({"toolName": "t"}, context=42)

        assert r1 == r2 == r3
