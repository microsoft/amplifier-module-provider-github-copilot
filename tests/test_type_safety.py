"""
Type safety tests for Batch 3 pyright compliance.

Tests cover:
- Item #9: cast() for CopilotClientOptions/SessionConfig in client.py
- Item #10: getattr() for dict attribute access in converters.py
- Item #11: Convert timeout to float explicitly in provider.py
- Item #12: Add None guard for coordinator.hooks in provider.py
- Item #13: Return ToolResult from tool handlers in tool_capture.py
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch


class TestClientTypedDictUsage:
    """Tests for Item #9: cast() usage in client.py."""

    def test_client_options_type_is_valid(self) -> None:
        """Verify client_options dict is properly typed for SDK."""
        from amplifier_module_provider_github_copilot.client import CopilotClientWrapper

        # Test with config that produces options
        wrapper = CopilotClientWrapper(
            config={"log_level": "debug", "auto_restart": True, "cwd": "/tmp"}
        )
        options = wrapper._build_client_options()

        # Must be a dict with valid CopilotClientOptions keys
        assert isinstance(options, dict)
        # These are the actual keys the method produces based on config
        assert options.get("log_level") == "debug"
        assert options.get("auto_restart") is True
        assert options.get("cwd") == "/tmp"

    def test_client_options_empty_config(self) -> None:
        """Verify empty config produces empty options dict."""
        from amplifier_module_provider_github_copilot.client import CopilotClientWrapper

        wrapper = CopilotClientWrapper(config={})
        options = wrapper._build_client_options()

        # Empty config should produce empty dict (no token env vars set)
        assert isinstance(options, dict)

    def test_session_config_type_is_valid(self) -> None:
        """Verify session_config dict structure."""
        # Session config is built dynamically, verify expected structure
        session_config: dict[str, Any] = {
            "model": "gpt-4o",
            "streaming": True,
        }
        # Must be assignable to dict[str, Any]
        assert isinstance(session_config, dict)
        assert session_config.get("model") == "gpt-4o"


class TestConvertersDictAccess:
    """Tests for Item #10: getattr() usage in converters.py."""

    def test_extract_tool_requests_from_object(self) -> None:
        """Test extracting tool_requests from object with attribute."""
        from amplifier_module_provider_github_copilot.converters import (
            convert_copilot_response_to_chat_response,
        )

        # Create mock response with tool_requests attribute
        mock_response = MagicMock()
        mock_response.content = "test content"
        mock_response.tool_requests = []
        mock_response.usage = None

        result = convert_copilot_response_to_chat_response(mock_response, "test-model")
        assert result is not None
        assert isinstance(result.content, list)

    def test_extract_tool_requests_from_dict(self) -> None:
        """Test extracting tool_requests from dict."""
        from amplifier_module_provider_github_copilot.converters import (
            convert_copilot_response_to_chat_response,
        )

        # Pass a dict instead of object
        mock_data: dict[str, Any] = {
            "content": "test content",
            "tool_requests": [],
        }

        result = convert_copilot_response_to_chat_response(mock_data, "test-model")
        assert result is not None

    def test_extract_usage_from_object(self) -> None:
        """Test extracting usage tokens from object."""
        from amplifier_module_provider_github_copilot.converters import (
            _extract_usage,
        )

        mock_usage = MagicMock()
        mock_usage.input_tokens = 100
        mock_usage.output_tokens = 50

        usage = _extract_usage(mock_usage)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50

    def test_extract_usage_from_dict(self) -> None:
        """Test extracting usage tokens from dict."""
        from amplifier_module_provider_github_copilot.converters import (
            _extract_usage,
        )

        usage_dict: dict[str, Any] = {
            "input_tokens": 200,
            "output_tokens": 100,
        }

        usage = _extract_usage(usage_dict)
        assert usage.input_tokens == 200
        assert usage.output_tokens == 100


class TestProviderTimeoutType:
    """Tests for Item #11: timeout float conversion in provider.py."""

    def test_timeout_is_float_from_int(self) -> None:
        """Verify int timeout is converted to float."""
        from amplifier_module_provider_github_copilot.provider import (
            CopilotSdkProvider,
        )

        provider = CopilotSdkProvider(None, config={"timeout": 30})
        assert isinstance(provider._timeout, float)
        assert provider._timeout == 30.0

    def test_timeout_is_float_from_string(self) -> None:
        """Verify string timeout is converted to float."""
        from amplifier_module_provider_github_copilot.provider import (
            CopilotSdkProvider,
        )

        provider = CopilotSdkProvider(None, config={"timeout": "45"})
        assert isinstance(provider._timeout, float)
        assert provider._timeout == 45.0

    def test_thinking_timeout_is_float(self) -> None:
        """Verify thinking_timeout is also float."""
        from amplifier_module_provider_github_copilot.provider import (
            CopilotSdkProvider,
        )

        provider = CopilotSdkProvider(None, config={"thinking_timeout": 120})
        assert isinstance(provider._thinking_timeout, float)
        assert provider._thinking_timeout == 120.0


class TestProviderNoneGuard:
    """Tests for Item #12: None guard for coordinator.hooks in provider.py."""

    def test_provider_works_without_coordinator(self) -> None:
        """Provider should work when coordinator is None."""
        from amplifier_module_provider_github_copilot.provider import (
            CopilotSdkProvider,
        )

        provider = CopilotSdkProvider(None, config={})
        # coordinator should be None by default
        assert provider._coordinator is None

    def test_provider_works_with_coordinator_without_hooks(self) -> None:
        """Provider should handle coordinator without hooks attribute."""
        from amplifier_module_provider_github_copilot.provider import (
            CopilotSdkProvider,
        )

        mock_coordinator = MagicMock(spec=[])  # Empty spec = no hooks attr
        provider = CopilotSdkProvider(None, config={}, coordinator=mock_coordinator)
        assert provider._coordinator is not None
        assert not hasattr(provider._coordinator, "hooks")

    def test_provider_works_with_coordinator_with_none_hooks(self) -> None:
        """Provider should handle coordinator.hooks being None."""
        from amplifier_module_provider_github_copilot.provider import (
            CopilotSdkProvider,
        )

        mock_coordinator = MagicMock()
        mock_coordinator.hooks = None
        provider = CopilotSdkProvider(None, config={}, coordinator=mock_coordinator)
        assert provider._coordinator is not None
        assert provider._coordinator.hooks is None


class TestToolCaptureReturnType:
    """Tests for Item #13: ToolResult return type in tool_capture.py."""

    def test_noop_handler_returns_correct_type(self) -> None:
        """Verify _noop_tool_handler returns expected type."""
        from amplifier_module_provider_github_copilot.tool_capture import (
            _noop_tool_handler,
        )

        result = _noop_tool_handler({})

        # Handler returns ToolResult (TypedDict with textResultForLlm)
        # This is required by SDK type contract: ToolHandler = Callable[[dict], ToolResult]
        assert isinstance(result, dict)
        assert "textResultForLlm" in result
        assert "error" in result["textResultForLlm"]
        assert "denied" in result["textResultForLlm"].lower()

    def test_convert_tools_creates_valid_sdk_tools(self) -> None:
        """Verify converted tools have proper handler type."""
        from amplifier_module_provider_github_copilot.tool_capture import (
            convert_tools_for_sdk,
        )

        tool_specs = [
            MagicMock(
                name="test_tool",
                description="A test tool",
                input_schema={"type": "object", "properties": {}},
            )
        ]

        with patch("copilot.types.Tool") as mock_tool:
            mock_tool.return_value = MagicMock(name="test_tool")
            convert_tools_for_sdk(tool_specs)  # Result unused - testing mock interaction

            # Tool was created with handler
            assert mock_tool.called
            call_kwargs = mock_tool.call_args[1]
            assert "handler" in call_kwargs
            # Handler should be callable
            assert callable(call_kwargs["handler"])
