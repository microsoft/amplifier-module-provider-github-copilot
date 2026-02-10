"""
Tests for tool_capture module (Deny + Destroy pattern).

This module tests the tool bridge functions that enable external orchestration:
- convert_tools_for_sdk: Convert Amplifier ToolSpec to SDK Tool objects
- make_deny_all_hook: Create preToolUse hook that denies all tool execution
"""

from unittest.mock import Mock
from unittest.mock import patch as unittest_mock_patch


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

    def test_returns_json_error_string(self):
        """Handler should return a JSON string with an error message."""
        import json

        from amplifier_module_provider_github_copilot.tool_capture import _noop_tool_handler

        result = _noop_tool_handler({"some": "args"})

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert "error" in parsed
        assert parsed["error"] == "Tool execution denied by provider policy"

    def test_returns_valid_json(self):
        """Handler result should always be valid JSON."""
        import json

        from amplifier_module_provider_github_copilot.tool_capture import _noop_tool_handler

        result = _noop_tool_handler(None)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

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
        import json

        from amplifier_module_provider_github_copilot.tool_capture import convert_tools_for_sdk

        tool_specs = [{"name": "test_tool", "description": "Test"}]
        result = convert_tools_for_sdk(tool_specs)

        assert len(result) == 1
        # Invoke the handler to verify it's the noop handler
        handler_result = result[0].handler({"arg": "val"})
        parsed = json.loads(handler_result)
        assert parsed["error"] == "Tool execution denied by provider policy"

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
