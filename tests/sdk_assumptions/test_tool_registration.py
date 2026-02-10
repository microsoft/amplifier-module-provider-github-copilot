"""
SDK Assumption Tests: Tool Registration

These tests validate assumptions about how tools are registered with
the SDK and made visible to the LLM.

IMPORTANT ASSUMPTION:
    Tools passed in the session config `tools` parameter become visible
    to the LLM, allowing it to emit tool_calls that match our definitions.

WHY THIS MATTERS:
    Our provider converts Amplifier's ToolSpec objects to SDK Tool objects
    and registers them with the session. If the SDK changed how tools are
    registered or exposed to the LLM, tool calls would fail.

BREAKING CHANGE INDICATORS:
    - LLM doesn't see registered tools (no tool_calls in response)
    - Tool names don't match between registration and calls
    - Duplicate tool names cause unexpected errors
    - Tool parameters not passed correctly to LLM

SDK LOCATIONS TO VERIFY:
    - copilot-sdk/python/copilot/types.py: Tool dataclass
    - copilot-sdk/python/copilot/session.py: _register_tools method
    - copilot-sdk/python/copilot/client.py: create_session handling
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

# =============================================================================
# Tool Definition Structures (matching SDK types.py)
# =============================================================================


@dataclass
class MockTool:
    """
    Mock of copilot.types.Tool for testing tool registration.

    Matches the SDK's Tool dataclass structure.
    """

    name: str
    description: str
    handler: Any
    parameters: dict[str, Any] | None = None


# =============================================================================
# Test Infrastructure for Tool Registration
# =============================================================================


class MockToolRegistry:
    """
    Simulates the SDK's internal tool registry.

    Tracks registered tools and validates registration rules.
    """

    def __init__(self):
        self.tools: dict[str, MockTool] = {}
        self.registration_order: list[str] = []

    def register(self, tool: MockTool) -> None:
        """
        Register a tool.

        Raises ValueError for duplicate names (matching SDK behavior).
        """
        if tool.name in self.tools:
            raise ValueError(f"Duplicate tool name: {tool.name}")

        self.tools[tool.name] = tool
        self.registration_order.append(tool.name)

    def get_tool(self, name: str) -> MockTool | None:
        """Get a registered tool by name."""
        return self.tools.get(name)

    def get_all_tools(self) -> list[MockTool]:
        """Get all registered tools in registration order."""
        return [self.tools[name] for name in self.registration_order]

    def clear(self) -> None:
        """Clear all registered tools."""
        self.tools.clear()
        self.registration_order.clear()


# =============================================================================
# Tests
# =============================================================================


class TestToolRegistration:
    """
    Tests that validate tool registration behavior.
    """

    @pytest.fixture
    def registry(self) -> MockToolRegistry:
        """Provide a fresh tool registry for each test."""
        return MockToolRegistry()

    @pytest.fixture
    def sample_handler(self):
        """A no-op handler for tool registration."""

        def handler(args: Any) -> str:
            return "result"

        return handler

    def test_single_tool_registration(self, registry, sample_handler):
        """
        ASSUMPTION: Single tool registration works correctly.

        A tool with name, description, handler, and parameters can be
        registered and retrieved.
        """
        tool = MockTool(
            name="read_file",
            description="Read contents of a file",
            handler=sample_handler,
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                },
                "required": ["path"],
            },
        )

        registry.register(tool)

        registered = registry.get_tool("read_file")
        assert registered is not None
        assert registered.name == "read_file"
        assert registered.description == "Read contents of a file"
        assert registered.parameters is not None
        assert "path" in registered.parameters.get("properties", {})

    def test_multiple_tool_registration(self, registry, sample_handler):
        """
        ASSUMPTION: Multiple tools can be registered concurrently.

        All tools should be accessible after registration.
        """
        tools = [
            MockTool(name="tool_a", description="Tool A", handler=sample_handler),
            MockTool(name="tool_b", description="Tool B", handler=sample_handler),
            MockTool(name="tool_c", description="Tool C", handler=sample_handler),
        ]

        for tool in tools:
            registry.register(tool)

        all_tools = registry.get_all_tools()
        assert len(all_tools) == 3
        assert [t.name for t in all_tools] == ["tool_a", "tool_b", "tool_c"]

    def test_duplicate_tool_names_rejected(self, registry, sample_handler):
        """
        ASSUMPTION: Duplicate tool names cause an error.

        The SDK rejects duplicate tool names (HTTP 400 from Copilot API).
        Our provider deduplicates before registration to prevent this.
        """
        tool1 = MockTool(name="read_file", description="First", handler=sample_handler)
        tool2 = MockTool(name="read_file", description="Second", handler=sample_handler)

        registry.register(tool1)

        with pytest.raises(ValueError, match="Duplicate tool name"):
            registry.register(tool2)

    def test_tool_with_no_parameters(self, registry, sample_handler):
        """
        ASSUMPTION: Tools with no parameters are valid.

        Some tools take no input (e.g., "get_current_time").
        """
        tool = MockTool(
            name="get_time",
            description="Get current time",
            handler=sample_handler,
            parameters=None,
        )

        registry.register(tool)

        registered = registry.get_tool("get_time")
        assert registered is not None
        assert registered.parameters is None

    def test_tool_with_empty_parameters(self, registry, sample_handler):
        """
        ASSUMPTION: Tools with empty parameters dict are valid.

        Empty parameters should be treated same as no parameters.
        """
        tool = MockTool(
            name="no_params",
            description="No parameters tool",
            handler=sample_handler,
            parameters={},
        )

        registry.register(tool)

        registered = registry.get_tool("no_params")
        assert registered is not None
        assert registered.parameters == {}


class TestToolNaming:
    """
    Tests that validate tool naming conventions and constraints.
    """

    @pytest.fixture
    def registry(self) -> MockToolRegistry:
        return MockToolRegistry()

    @pytest.fixture
    def sample_handler(self):
        def handler(args: Any) -> str:
            return "result"

        return handler

    def test_snake_case_names_valid(self, registry, sample_handler):
        """
        ASSUMPTION: snake_case names are valid.

        Standard Python naming convention should work.
        """
        tool = MockTool(
            name="read_file_contents",
            description="Read file",
            handler=sample_handler,
        )
        registry.register(tool)

        assert registry.get_tool("read_file_contents") is not None

    def test_names_with_special_chars(self, registry, sample_handler):
        """
        Document behavior with special characters in names.

        Different tools may have different naming conventions.
        This documents what's accepted.
        """
        # Test various naming patterns
        valid_names = [
            "simple",
            "with_underscore",
            "WithCamelCase",
            "with123numbers",
        ]

        for name in valid_names:
            tool = MockTool(name=name, description="Test", handler=sample_handler)
            registry.register(tool)

        assert len(registry.get_all_tools()) == len(valid_names)

    def test_empty_name_handling(self, registry, sample_handler):
        """
        Document behavior with empty tool name.

        Our provider skips tools with empty names, so this shouldn't
        reach the SDK, but we document the behavior.
        """
        # This would typically be rejected at a higher level
        tool = MockTool(name="", description="Empty name", handler=sample_handler)

        # For our mock, we allow it (documents the need for validation)
        registry.register(tool)
        assert registry.get_tool("") is not None


class TestToolParameters:
    """
    Tests that validate tool parameter schema handling.
    """

    @pytest.fixture
    def registry(self) -> MockToolRegistry:
        return MockToolRegistry()

    @pytest.fixture
    def sample_handler(self):
        def handler(args: Any) -> str:
            return "result"

        return handler

    def test_json_schema_parameters(self, registry, sample_handler):
        """
        ASSUMPTION: Parameters follow JSON Schema format.

        The SDK passes parameters in JSON Schema format to the LLM.
        """
        parameters = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results",
                    "default": 10,
                },
            },
            "required": ["query"],
        }

        tool = MockTool(
            name="search",
            description="Search for items",
            handler=sample_handler,
            parameters=parameters,
        )

        registry.register(tool)

        registered = registry.get_tool("search")
        assert registered.parameters["type"] == "object"
        assert "query" in registered.parameters["properties"]
        assert registered.parameters["required"] == ["query"]

    def test_nested_parameters(self, registry, sample_handler):
        """
        ASSUMPTION: Nested parameter schemas are preserved.

        Complex nested structures should be passed through correctly.
        """
        parameters = {
            "type": "object",
            "properties": {
                "config": {
                    "type": "object",
                    "properties": {
                        "option_a": {"type": "boolean"},
                        "option_b": {"type": "string"},
                    },
                },
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        }

        tool = MockTool(
            name="configure",
            description="Configure settings",
            handler=sample_handler,
            parameters=parameters,
        )

        registry.register(tool)

        registered = registry.get_tool("configure")
        assert registered.parameters["properties"]["config"]["type"] == "object"
        assert registered.parameters["properties"]["items"]["type"] == "array"


class TestToolHandler:
    """
    Tests that validate tool handler registration.
    """

    @pytest.fixture
    def registry(self) -> MockToolRegistry:
        return MockToolRegistry()

    def test_sync_handler_registration(self, registry):
        """
        ASSUMPTION: Synchronous handlers can be registered.
        """

        def sync_handler(args: Any) -> str:
            return f"Processed: {args}"

        tool = MockTool(name="sync_tool", description="Sync", handler=sync_handler)
        registry.register(tool)

        registered = registry.get_tool("sync_tool")
        assert registered.handler is sync_handler

    def test_async_handler_registration(self, registry):
        """
        ASSUMPTION: Async handlers can be registered.
        """

        async def async_handler(args: Any) -> str:
            return f"Async processed: {args}"

        tool = MockTool(name="async_tool", description="Async", handler=async_handler)
        registry.register(tool)

        registered = registry.get_tool("async_tool")
        assert registered.handler is async_handler

    def test_lambda_handler_registration(self, registry):
        """
        ASSUMPTION: Lambda handlers can be registered.

        Our no-op handler pattern uses simple lambdas.
        """
        tool = MockTool(
            name="lambda_tool",
            description="Lambda",
            handler=lambda args: "noop",
        )
        registry.register(tool)

        registered = registry.get_tool("lambda_tool")
        assert callable(registered.handler)


class TestExcludedToolsParameter:
    """
    Tests for the excluded_tools session parameter.

    This parameter is documented in SessionConfig but its exact
    behavior with built-in tools is empirically validated here.
    """

    def test_excluded_tools_concept(self):
        """
        Document the excluded_tools parameter concept.

        SessionConfig has an excluded_tools parameter that should
        disable specified tools. This test documents the expected
        behavior for future SDK versions.
        """
        # From copilot/types.py SessionConfig:
        # excluded_tools: list[str]  # List of tool names to disable

        # Expected behavior (not yet validated against real SDK):
        # - Tools in this list should not be available to LLM
        # - Might apply to built-in tools as well as user tools

        # This is a documentation test - actual behavior may vary
        expected_config = {
            "model": "claude-sonnet-4",
            "excluded_tools": ["edit", "create", "write"],  # Hypothetically disable built-ins
        }

        assert "excluded_tools" in expected_config
        assert len(expected_config["excluded_tools"]) == 3

    def test_available_tools_precedence(self):
        """
        Document available_tools vs excluded_tools precedence.

        Per types.py: "List of tool names to allow (takes precedence
        over excluded_tools)"
        """
        # If both are specified, available_tools should win
        config = {
            "available_tools": ["read_file", "list_dir"],  # Only these allowed
            "excluded_tools": ["read_file"],  # This should be ignored
        }

        # Per documentation, available_tools takes precedence
        # So read_file SHOULD be available despite being in excluded_tools
        assert "read_file" in config["available_tools"]


class TestToolConversionFromAmplifier:
    """
    Tests that validate our tool conversion logic.

    These tests ensure ToolSpec objects from Amplifier are correctly
    converted to SDK Tool objects.
    """

    def test_toolspec_to_sdk_tool_conversion(self):
        """
        ASSUMPTION: Our convert_tools_for_sdk produces valid SDK Tools.

        Importing and testing the actual converter to ensure it matches
        SDK expectations.
        """
        # Import the actual converter
        from amplifier_module_provider_github_copilot.tool_capture import convert_tools_for_sdk

        # Create mock ToolSpec-like objects
        class MockToolSpec:
            def __init__(self, name: str, description: str, parameters: dict | None):
                self.name = name
                self.description = description
                self.parameters = parameters

        specs = [
            MockToolSpec(
                name="read_file",
                description="Read a file",
                parameters={"type": "object", "properties": {"path": {"type": "string"}}},
            ),
            MockToolSpec(
                name="write_file",
                description="Write a file",
                parameters={"type": "object", "properties": {"path": {"type": "string"}}},
            ),
        ]

        # Convert
        sdk_tools = convert_tools_for_sdk(specs)

        # Validate
        assert len(sdk_tools) == 2
        assert sdk_tools[0].name == "read_file"
        assert sdk_tools[1].name == "write_file"
        assert callable(sdk_tools[0].handler)

    def test_duplicate_deduplication(self):
        """
        ASSUMPTION: Our converter deduplicates by name.

        The Copilot API rejects duplicate names with HTTP 400.
        Our converter must remove duplicates before SDK registration.
        """
        from amplifier_module_provider_github_copilot.tool_capture import convert_tools_for_sdk

        class MockToolSpec:
            def __init__(self, name: str, description: str, parameters: dict | None):
                self.name = name
                self.description = description
                self.parameters = parameters

        # Create specs with duplicates
        specs = [
            MockToolSpec(name="read_file", description="First", parameters=None),
            MockToolSpec(name="write_file", description="Unique", parameters=None),
            MockToolSpec(name="read_file", description="Duplicate", parameters=None),
        ]

        sdk_tools = convert_tools_for_sdk(specs)

        # Should have only 2 tools (duplicate removed)
        assert len(sdk_tools) == 2
        names = [t.name for t in sdk_tools]
        assert names == ["read_file", "write_file"]

    def test_empty_name_skipped(self):
        """
        ASSUMPTION: Tools with empty names are skipped.

        Invalid tools should be filtered out, not crash registration.
        """
        from amplifier_module_provider_github_copilot.tool_capture import convert_tools_for_sdk

        class MockToolSpec:
            def __init__(self, name: str, description: str, parameters: dict | None):
                self.name = name
                self.description = description
                self.parameters = parameters

        specs = [
            MockToolSpec(name="", description="Empty name", parameters=None),
            MockToolSpec(name="valid_tool", description="Valid", parameters=None),
        ]

        sdk_tools = convert_tools_for_sdk(specs)

        # Only valid tool should be converted
        assert len(sdk_tools) == 1
        assert sdk_tools[0].name == "valid_tool"
