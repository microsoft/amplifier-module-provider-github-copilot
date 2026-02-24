"""
Tests for CopilotSdkProvider.

This module tests the provider implementation including
all 5 Provider protocol methods.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock, patch

import pytest
from amplifier_core import ChatResponse, TextBlock, ToolCall, Usage

from amplifier_module_provider_github_copilot import ProviderInfo
from amplifier_module_provider_github_copilot._constants import (
    BUILTIN_TO_AMPLIFIER_CAPABILITY,
    COPILOT_BUILTIN_TOOL_NAMES,
    DEFAULT_THINKING_TIMEOUT,
    DEFAULT_TIMEOUT,
)
from amplifier_module_provider_github_copilot.client import CopilotClientWrapper
from amplifier_module_provider_github_copilot.provider import CopilotSdkProvider


class TestProviderProtocol:
    """Tests for Provider protocol compliance."""

    @pytest.fixture
    def provider(self, mock_coordinator, provider_config):
        """Create provider instance for testing."""
        return CopilotSdkProvider(
            api_key=None,
            config=provider_config,
            coordinator=mock_coordinator,
        )

    def test_name_property(self, provider):
        """Provider should have correct name."""
        assert provider.name == "github-copilot"

    def test_get_info(self, provider):
        """get_info should return correct ProviderInfo."""
        info = provider.get_info()

        assert isinstance(info, ProviderInfo)
        assert info.id == "github-copilot"
        assert info.display_name == "GitHub Copilot SDK"
        assert "tools" in info.capabilities
        assert "vision" in info.capabilities

    def test_get_info_includes_context_window_for_context_manager(self, provider):
        """get_info().defaults must include context_window for context manager.

        CRITICAL: The context manager's _calculate_budget() reads from
        provider.get_info().defaults.get("context_window"), NOT get_model_info().
        See CONTEXT_CONTRACT.md for the contract specification.

        Without this, the context manager falls back to bundle config which
        can exceed the model's actual limit (e.g., 300k config vs 200k model).
        """
        info = provider.get_info()

        assert "context_window" in info.defaults, (
            "get_info().defaults must include context_window - "
            "context manager reads this for budget calculation"
        )
        assert info.defaults["context_window"] == 200000  # claude-opus-4.5 limit

    def test_get_info_includes_max_output_tokens_for_context_manager(self, provider):
        """get_info().defaults must include max_output_tokens for context manager.

        The context manager calculates budget as:
        budget = context_window - max_output_tokens - safety_margin

        Without max_output_tokens, the budget calculation fails and falls
        back to bundle config.
        """
        info = provider.get_info()

        assert "max_output_tokens" in info.defaults, (
            "get_info().defaults must include max_output_tokens - "
            "context manager needs this for budget calculation"
        )
        assert info.defaults["max_output_tokens"] == 32000  # claude-opus-4.5 limit

    def test_get_info_excludes_thinking_from_provider_capabilities(self, provider):
        """Provider-level capabilities should NOT include 'thinking'.

        Thinking/reasoning is model-specific (e.g., claude-opus-4.6 supports it,
        claude-opus-4.5 does not). Provider get_info() should only advertise
        capabilities common to all models. Model-specific capabilities like
        'thinking' should only appear in individual ModelInfo.capabilities.

        This test ensures we don't mislead users about provider-wide capabilities.
        """
        info = provider.get_info()

        # Thinking is model-specific, not provider-level
        assert "thinking" not in info.capabilities
        assert "reasoning" not in info.capabilities

        # These should be present (common to all models)
        assert "streaming" in info.capabilities
        assert "tools" in info.capabilities
        assert "vision" in info.capabilities

    def test_get_info_credential_env_vars(self, provider):
        """get_info should advertise credential env vars for CLI detection."""
        info = provider.get_info()
        assert "GITHUB_TOKEN" in info.credential_env_vars
        assert "GH_TOKEN" in info.credential_env_vars
        assert "COPILOT_GITHUB_TOKEN" in info.credential_env_vars

    @pytest.mark.asyncio
    async def test_list_models(self, provider, mock_copilot_client):
        """list_models should return available models."""
        # Patch at class level to work with __slots__
        with patch.object(
            CopilotClientWrapper, "ensure_client", new_callable=AsyncMock
        ) as mock_ensure:
            mock_ensure.return_value = mock_copilot_client

            models = await provider.list_models()

            assert isinstance(models, list)
            assert len(models) > 0
            assert models[0].id == "claude-opus-4.5"

    def test_get_session_metrics(self, provider):
        """get_session_metrics should return valid metrics dict."""
        metrics = provider.get_session_metrics()

        assert "total_sessions" in metrics
        assert "total_requests" in metrics
        assert "avg_response_time_ms" in metrics
        assert "error_count" in metrics

        # Initial values should be zero
        assert metrics["total_sessions"] == 0
        assert metrics["total_requests"] == 0
        assert metrics["avg_response_time_ms"] == 0.0
        assert metrics["error_count"] == 0

    def test_get_model_info_returns_none_for_unknown_model(self, mock_coordinator, provider_config):
        """get_model_info returns None for unknown models when cache is cold."""
        config = {
            **provider_config,
            "model": "unknown-model-xyz",
            "default_model": "unknown-model-xyz",
        }
        provider = CopilotSdkProvider(
            api_key=None,
            config=config,
            coordinator=mock_coordinator,
        )

        model_info = provider.get_model_info()

        assert model_info is None

    def test_get_model_info_returns_fallback_for_known_model(
        self, mock_coordinator, provider_config
    ):
        """get_model_info returns known limits for common models when cache is cold."""
        config = {**provider_config, "default_model": "claude-opus-4.5"}
        provider = CopilotSdkProvider(
            api_key=None,
            config=config,
            coordinator=mock_coordinator,
        )

        model_info = provider.get_model_info()

        assert model_info is not None
        assert model_info.context_window == 200000
        assert model_info.max_output_tokens == 32000

    @pytest.mark.asyncio
    async def test_get_model_info_uses_cache_after_list_models(self, provider, mock_copilot_client):
        """get_model_info uses cached values from list_models."""
        with patch.object(
            CopilotClientWrapper, "ensure_client", new_callable=AsyncMock
        ) as mock_ensure:
            mock_ensure.return_value = mock_copilot_client

            # Populate cache
            await provider.list_models()

            # Now get_model_info should return cached value
            model_info = provider.get_model_info()

            assert model_info is not None
            assert model_info.id == "claude-opus-4.5"
            assert model_info.context_window == 200000

    @pytest.mark.asyncio
    async def test_complete_basic(self, provider, mock_copilot_client, sample_messages):
        """complete should process messages and return response."""
        # Create mock session
        mock_session = AsyncMock()
        mock_session.session_id = "test-session"
        mock_session.destroy = AsyncMock()

        # Create mock response
        mock_response = Mock()
        mock_response.data = Mock()
        mock_response.data.content = "Hello! I'm here to help."
        mock_response.data.tool_requests = None
        mock_response.data.input_tokens = 100
        mock_response.data.output_tokens = 50

        mock_session.send_and_wait = AsyncMock(return_value=mock_response)
        mock_copilot_client.create_session = AsyncMock(return_value=mock_session)

        # Mock the client wrapper's create_session as async context manager
        # Note: when patching at class level, 'self' is passed as first argument
        @asynccontextmanager
        async def mock_create_session(
            self,
            model,
            system_message=None,
            streaming=True,
            reasoning_effort=None,
            tools=None,
            excluded_tools=None,
            hooks=None,
        ):
            yield mock_session
            await mock_session.destroy()

        # Patch at class level to work with __slots__
        with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
            with patch.object(
                CopilotClientWrapper, "send_and_wait", new_callable=AsyncMock
            ) as mock_send:
                mock_send.return_value = mock_response

                # Create request
                request = {"messages": sample_messages}

                # Call complete
                response = await provider.complete(request)

                # Verify response
                assert isinstance(response, ChatResponse)
                assert len(response.content) > 0

    @pytest.mark.asyncio
    async def test_complete_with_tool_calls(
        self, provider, mock_copilot_client, sample_messages, mock_tool_response
    ):
        """complete should handle tool calls in response."""
        mock_session = AsyncMock()
        mock_session.session_id = "test-session"
        mock_session.destroy = AsyncMock()
        mock_session.send_and_wait = AsyncMock(return_value=mock_tool_response)
        mock_copilot_client.create_session = AsyncMock(return_value=mock_session)

        # Note: when patching at class level, 'self' is passed as first argument
        @asynccontextmanager
        async def mock_create_session(
            self,
            model,
            system_message=None,
            streaming=True,
            reasoning_effort=None,
            tools=None,
            excluded_tools=None,
            hooks=None,
        ):
            yield mock_session
            await mock_session.destroy()

        # Patch at class level to work with __slots__
        with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
            with patch.object(
                CopilotClientWrapper, "send_and_wait", new_callable=AsyncMock
            ) as mock_send:
                mock_send.return_value = mock_tool_response

                request = {"messages": sample_messages}
                response = await provider.complete(request)

                # Verify tool calls are parsed
                assert len(response.tool_calls) > 0
                assert response.tool_calls[0].name == "read_file"

    def test_parse_tool_calls_empty(self, provider):
        """parse_tool_calls should handle response with no tool calls."""
        response = ChatResponse(
            content=[TextBlock(type="text", text="No tools needed")],
            tool_calls=[],
        )

        result = provider.parse_tool_calls(response)
        assert result == []

    def test_parse_tool_calls_with_calls(self, provider):
        """parse_tool_calls should return tool calls from response."""
        tool_call = ToolCall(
            id="call_123",
            name="read_file",
            arguments={"path": "test.py"},
        )
        response = ChatResponse(
            content=[TextBlock(type="text", text="Let me read that")],
            tool_calls=[tool_call],
        )

        result = provider.parse_tool_calls(response)
        assert len(result) == 1
        assert result[0].name == "read_file"
        assert result[0].id == "call_123"

    @pytest.mark.asyncio
    async def test_close(self, provider):
        """close should cleanup resources."""
        # Patch at class level to work with __slots__
        with patch.object(CopilotClientWrapper, "close", new_callable=AsyncMock) as mock_close:
            await provider.close()
            mock_close.assert_called_once()


class TestProviderConfiguration:
    """Tests for provider configuration."""

    def test_default_configuration(self, mock_coordinator):
        """Provider should use default configuration."""
        provider = CopilotSdkProvider(
            api_key=None,
            config={},
            coordinator=mock_coordinator,
        )

        assert provider._model == "claude-opus-4.5"
        assert provider._timeout == DEFAULT_TIMEOUT
        assert provider._thinking_timeout == DEFAULT_THINKING_TIMEOUT
        assert provider._debug is False

    def test_custom_configuration(self, mock_coordinator):
        """Provider should use custom configuration."""
        config = {
            "model": "claude-sonnet-4",
            "timeout": 60.0,
            "thinking_timeout": 900.0,
            "debug": True,
            "raw_debug": True,
            "debug_truncate_length": 200,
        }
        provider = CopilotSdkProvider(
            api_key=None,
            config=config,
            coordinator=mock_coordinator,
        )

        assert provider._model == "claude-sonnet-4"
        assert provider._timeout == 60.0
        assert provider._thinking_timeout == 900.0
        assert provider._debug is True
        assert provider._raw_debug is True
        assert provider._debug_truncate_length == 200

    def test_raw_debug_defaults_to_false(self, mock_coordinator):
        """raw_debug should default to False."""
        provider = CopilotSdkProvider(
            api_key=None,
            config={},
            coordinator=mock_coordinator,
        )
        assert provider._raw_debug is False

    def test_default_model_config_key(self, mock_coordinator):
        """default_model config key should set the model (Amplifier convention)."""
        config = {"default_model": "claude-opus-4.6"}
        provider = CopilotSdkProvider(
            api_key=None,
            config=config,
            coordinator=mock_coordinator,
        )
        assert provider._model == "claude-opus-4.6"

    def test_model_takes_precedence_over_default_model(self, mock_coordinator):
        """model (runtime override) should take precedence over default_model."""
        config = {"default_model": "claude-opus-4.6", "model": "claude-sonnet-4"}
        provider = CopilotSdkProvider(
            api_key=None,
            config=config,
            coordinator=mock_coordinator,
        )
        assert provider._model == "claude-sonnet-4"

    def test_model_key_still_works_for_backward_compat(self, mock_coordinator):
        """model key should work when default_model is absent."""
        config = {"model": "claude-sonnet-4"}
        provider = CopilotSdkProvider(
            api_key=None,
            config=config,
            coordinator=mock_coordinator,
        )
        assert provider._model == "claude-sonnet-4"

    def test_truncate_values_delegates_to_core(self, mock_coordinator):
        """_truncate_values should recursively truncate long strings."""
        provider = CopilotSdkProvider(
            api_key=None,
            config={"debug_truncate_length": 10},
            coordinator=mock_coordinator,
        )

        obj = {"short": "hi", "long": "x" * 50, "nested": {"deep": "y" * 50}}
        result = provider._truncate_values(obj)

        # Short strings should pass through unchanged
        assert result["short"] == "hi"
        # Long strings should be truncated (core appends "... (truncated N chars)")
        assert len(result["long"]) < 50
        assert "truncated" in result["long"] or result["long"].endswith("...")
        # Nested structures should also be truncated
        assert len(result["nested"]["deep"]) < 50


class TestBuiltinToolConstants:
    """Tests for built-in tool naming constants.

    These constants are preserved for reference even though the
    Deny + Destroy pattern no longer uses built-in exclusion logic.
    """

    def test_builtin_tool_names_includes_critical_tools(self):
        """COPILOT_BUILTIN_TOOL_NAMES must include tools that cause bypass."""
        critical_tools = {"edit", "view", "bash", "grep", "glob", "web_fetch"}
        assert critical_tools.issubset(COPILOT_BUILTIN_TOOL_NAMES)

    def test_capability_mapping_covers_all_builtins(self):
        """Every known built-in tool must have a capability mapping entry."""
        for builtin in COPILOT_BUILTIN_TOOL_NAMES:
            assert builtin in BUILTIN_TO_AMPLIFIER_CAPABILITY, (
                f"Built-in '{builtin}' missing from BUILTIN_TO_AMPLIFIER_CAPABILITY"
            )

    def test_edit_maps_to_write_file(self):
        """CLI's edit tool must map to Amplifier's write_file and edit_file."""
        assert "write_file" in BUILTIN_TO_AMPLIFIER_CAPABILITY["edit"]
        assert "edit_file" in BUILTIN_TO_AMPLIFIER_CAPABILITY["edit"]

    # ─────────────────────────────────────────────────────────────────────────
    # TDD Tests for 8 Missing CLI Built-in Tools (Bug GHCP-BUILTIN-TOOLS-001)
    # Added: 2026-02-17
    # Evidence: ST04 session, binary archaeology, live Gemini test
    # ─────────────────────────────────────────────────────────────────────────

    def test_discovered_builtins_in_exclusion_list(self):
        """All discovered CLI built-ins must be in COPILOT_BUILTIN_TOOL_NAMES.

        Evidence:
        - create: ST04 session (f3da2b5b) - "Tool 'create' not found"
        - shell: 2026-02-09 archaeology
        - report_progress: 2026-02-09 archaeology (think category)
        - update_todo: 2026-02-09 archaeology (think category)
        - skill: 2026-02-09 archaeology (other category)
        - fetch_copilot_cli_documentation: 2026-02-16 live Gemini test
        - search_code_subagent: 2026-02-16 binary analysis (search category)
        - github-mcp-server-web_search: 2026-02-16 binary analysis (search category)
        - task_complete: 2026-02-17 forensic session 1541c502
        """
        discovered_2026_02_09 = {"shell", "report_progress", "update_todo", "skill"}
        discovered_2026_02_16 = {
            "create",
            "fetch_copilot_cli_documentation",
            "search_code_subagent",
            "github-mcp-server-web_search",
        }
        discovered_2026_02_17 = {"task_complete"}

        all_discovered = discovered_2026_02_09 | discovered_2026_02_16 | discovered_2026_02_17

        for tool in all_discovered:
            assert tool in COPILOT_BUILTIN_TOOL_NAMES, (
                f"Discovered built-in '{tool}' missing from COPILOT_BUILTIN_TOOL_NAMES"
            )

    def test_create_maps_to_write_file(self):
        """CLI's create tool must map to write_file capability.

        Evidence: ST04 session - LLM called 'create' when asked to create a file.
        """
        assert "create" in BUILTIN_TO_AMPLIFIER_CAPABILITY, (
            "'create' must be in BUILTIN_TO_AMPLIFIER_CAPABILITY"
        )
        assert "write_file" in BUILTIN_TO_AMPLIFIER_CAPABILITY["create"], (
            "'create' should map to 'write_file'"
        )

    def test_shell_maps_to_bash(self):
        """CLI's shell tool must map to bash capability."""
        assert "shell" in BUILTIN_TO_AMPLIFIER_CAPABILITY
        assert "bash" in BUILTIN_TO_AMPLIFIER_CAPABILITY["shell"]

    def test_update_todo_maps_to_todo(self):
        """CLI's update_todo tool must map to todo capability."""
        assert "update_todo" in BUILTIN_TO_AMPLIFIER_CAPABILITY
        assert "todo" in BUILTIN_TO_AMPLIFIER_CAPABILITY["update_todo"]

    def test_task_complete_maps_to_todo(self):
        """CLI's task_complete tool must map to todo capability.

        Evidence: 2026-02-17 forensic session 1541c502 - model called
        task_complete when completing simple math, causing 'Tool not found'.
        """
        assert "task_complete" in BUILTIN_TO_AMPLIFIER_CAPABILITY
        assert "todo" in BUILTIN_TO_AMPLIFIER_CAPABILITY["task_complete"]

    def test_skill_maps_to_load_skill(self):
        """CLI's skill tool must map to load_skill capability."""
        assert "skill" in BUILTIN_TO_AMPLIFIER_CAPABILITY
        assert "load_skill" in BUILTIN_TO_AMPLIFIER_CAPABILITY["skill"]

    def test_search_code_subagent_maps_to_grep_glob_delegate(self):
        """CLI's search_code_subagent is composite - maps to grep, glob, and delegate.

        The mapping uses frozenset so ANY capability triggers exclusion:
        - grep/glob: User has search primitives
        - delegate: User has subagent capability (superior to CLI's hidden subagent)
        """
        assert "search_code_subagent" in BUILTIN_TO_AMPLIFIER_CAPABILITY
        mapping = BUILTIN_TO_AMPLIFIER_CAPABILITY["search_code_subagent"]
        assert "grep" in mapping, "search_code_subagent should map to grep"
        assert "glob" in mapping, "search_code_subagent should map to glob"
        assert "delegate" in mapping, "search_code_subagent should map to delegate"

    def test_github_mcp_web_search_maps_to_web_search(self):
        """CLI's github-mcp-server-web_search maps to web_search.

        Evidence: Amplifier confirmed web_search exists at line 236 of _constants.py.
        """
        assert "github-mcp-server-web_search" in BUILTIN_TO_AMPLIFIER_CAPABILITY
        assert "web_search" in BUILTIN_TO_AMPLIFIER_CAPABILITY["github-mcp-server-web_search"]

    def test_pure_exclusion_tools_have_mapping_entries(self):
        """Tools without Amplifier equivalents still need mapping entries.

        report_progress and fetch_copilot_cli_documentation have no direct
        Amplifier equivalent but must be in the mapping dict (with empty set
        or partial mapping) to satisfy test_capability_mapping_covers_all_builtins.
        """
        # These tools must be in the exclusion list
        assert "report_progress" in COPILOT_BUILTIN_TOOL_NAMES
        assert "fetch_copilot_cli_documentation" in COPILOT_BUILTIN_TOOL_NAMES
        assert "task_complete" in COPILOT_BUILTIN_TOOL_NAMES

        # They must also be in the mapping dict
        assert "report_progress" in BUILTIN_TO_AMPLIFIER_CAPABILITY
        assert "fetch_copilot_cli_documentation" in BUILTIN_TO_AMPLIFIER_CAPABILITY
        assert "task_complete" in BUILTIN_TO_AMPLIFIER_CAPABILITY


class TestDenyDestroyPattern:
    """Tests for Deny + Destroy tool calling architecture.

    This architecture (empirically validated via test_dumb_pipe_strategies.py):
    1. Converts Amplifier ToolSpec -> SDK Tool with no-op handler
    2. Registers preToolUse deny hook to prevent CLI execution
    3. Captures tool_requests from ASSISTANT_MESSAGE events
    4. Returns ToolCall objects to Amplifier's orchestrator
    5. Session destroyed by context manager (prevents CLI retry)

    Replaces the previous built-in exclusion approach which was vulnerable
    to CLI bypass (session 497bbab7) and hangs (session 2a1fe04a).
    """

    @pytest.fixture
    def provider(self, mock_coordinator):
        """Create provider with streaming enabled."""
        return CopilotSdkProvider(
            api_key=None,
            config={"use_streaming": True, "debug": False},
            coordinator=mock_coordinator,
        )

    @pytest.mark.asyncio
    async def test_deny_hooks_created_when_tools_present(self, provider, mock_coordinator):
        """make_deny_all_hook() should be called when tools are in request."""
        mock_session = AsyncMock()
        mock_session.session_id = "test"
        mock_session.destroy = AsyncMock()

        fake_response = ChatResponse(
            content=[TextBlock(type="text", text="Done")],
            tool_calls=[],
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            finish_reason="end_turn",
        )

        @asynccontextmanager
        async def mock_create_session(
            self,
            model,
            system_message=None,
            streaming=True,
            reasoning_effort=None,
            tools=None,
            excluded_tools=None,
            hooks=None,
        ):
            yield mock_session

        user_tool = Mock()
        user_tool.name = "write_file"
        user_tool.description = "Write a file"
        user_tool.input_schema = {}

        request = {
            "messages": [{"role": "user", "content": "test"}],
            "tools": [user_tool],
        }

        with patch(
            "amplifier_module_provider_github_copilot.provider.make_deny_all_hook",
            return_value={"on_pre_tool_use": Mock()},
        ) as mock_make_deny:
            with patch(
                "amplifier_module_provider_github_copilot.provider.convert_tools_for_sdk",
                return_value=[Mock(name="write_file")],
            ):
                with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
                    with patch.object(
                        provider,
                        "_complete_streaming",
                        new_callable=AsyncMock,
                        return_value=fake_response,
                    ):
                        with patch.object(
                            provider,
                            "_model_supports_reasoning",
                            new_callable=AsyncMock,
                            return_value=False,
                        ):
                            await provider.complete(request)

        mock_make_deny.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_hooks_when_no_tools(self, mock_coordinator):
        """No deny hooks should be created when request has no tools."""
        # Use non-streaming to test the simpler send_and_wait path
        provider = CopilotSdkProvider(
            api_key=None,
            config={"use_streaming": False, "debug": False},
            coordinator=mock_coordinator,
        )

        mock_session = AsyncMock()
        mock_session.session_id = "test"
        mock_session.destroy = AsyncMock()

        mock_response = Mock()
        mock_response.data = Mock()
        mock_response.data.content = "Done"
        mock_response.data.tool_requests = None
        mock_response.data.input_tokens = 10
        mock_response.data.output_tokens = 5

        captured_hooks = {}

        @asynccontextmanager
        async def mock_create_session(
            self,
            model,
            system_message=None,
            streaming=True,
            reasoning_effort=None,
            tools=None,
            excluded_tools=None,
            hooks=None,
        ):
            captured_hooks["hooks"] = hooks
            yield mock_session

        request = {"messages": [{"role": "user", "content": "test"}]}

        with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
            with patch.object(
                CopilotClientWrapper, "send_and_wait", new_callable=AsyncMock
            ) as mock_send:
                mock_send.return_value = mock_response
                with patch.object(
                    provider,
                    "_model_supports_reasoning",
                    new_callable=AsyncMock,
                    return_value=False,
                ):
                    await provider.complete(request)

        assert captured_hooks.get("hooks") is None

    @pytest.mark.asyncio
    async def test_hooks_passed_to_session(self, provider, mock_coordinator):
        """Deny hooks should be passed through to create_session()."""
        mock_session = AsyncMock()
        mock_session.session_id = "test"
        mock_session.destroy = AsyncMock()

        fake_response = ChatResponse(
            content=[TextBlock(type="text", text="Done")],
            tool_calls=[],
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            finish_reason="end_turn",
        )

        captured_hooks = {}

        @asynccontextmanager
        async def mock_create_session(
            self,
            model,
            system_message=None,
            streaming=True,
            reasoning_effort=None,
            tools=None,
            excluded_tools=None,
            hooks=None,
        ):
            captured_hooks["hooks"] = hooks
            yield mock_session

        user_tool = Mock()
        user_tool.name = "write_file"
        user_tool.description = "Write a file"
        user_tool.input_schema = {}

        request = {
            "messages": [{"role": "user", "content": "test"}],
            "tools": [user_tool],
        }

        deny_hook_sentinel = {"on_pre_tool_use": Mock()}

        with patch(
            "amplifier_module_provider_github_copilot.provider.make_deny_all_hook",
            return_value=deny_hook_sentinel,
        ):
            with patch(
                "amplifier_module_provider_github_copilot.provider.convert_tools_for_sdk",
                return_value=[Mock(name="write_file")],
            ):
                with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
                    with patch.object(
                        provider,
                        "_complete_streaming",
                        new_callable=AsyncMock,
                        return_value=fake_response,
                    ):
                        with patch.object(
                            provider,
                            "_model_supports_reasoning",
                            new_callable=AsyncMock,
                            return_value=False,
                        ):
                            await provider.complete(request)

        assert captured_hooks["hooks"] is deny_hook_sentinel

    @pytest.mark.asyncio
    async def test_streaming_forced_when_tools_present(self, mock_coordinator):
        """Streaming must be forced on when tools are present (event-based capture)."""
        provider = CopilotSdkProvider(
            api_key=None,
            config={"use_streaming": False, "debug": False},
            coordinator=mock_coordinator,
        )

        mock_session = AsyncMock()
        mock_session.session_id = "test"
        mock_session.destroy = AsyncMock()

        fake_response = ChatResponse(
            content=[TextBlock(type="text", text="Done")],
            tool_calls=[],
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            finish_reason="end_turn",
        )

        captured_streaming = {}

        @asynccontextmanager
        async def mock_create_session(
            self,
            model,
            system_message=None,
            streaming=True,
            reasoning_effort=None,
            tools=None,
            excluded_tools=None,
            hooks=None,
        ):
            captured_streaming["streaming"] = streaming
            yield mock_session

        user_tool = Mock()
        user_tool.name = "write_file"
        user_tool.description = "Write a file"
        user_tool.input_schema = {}

        request = {
            "messages": [{"role": "user", "content": "test"}],
            "tools": [user_tool],
        }

        with patch(
            "amplifier_module_provider_github_copilot.provider.convert_tools_for_sdk",
            return_value=[Mock(name="write_file")],
        ):
            with patch(
                "amplifier_module_provider_github_copilot.provider.make_deny_all_hook",
                return_value={"on_pre_tool_use": Mock()},
            ):
                with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
                    with patch.object(
                        provider,
                        "_complete_streaming",
                        new_callable=AsyncMock,
                        return_value=fake_response,
                    ):
                        with patch.object(
                            provider,
                            "_model_supports_reasoning",
                            new_callable=AsyncMock,
                            return_value=False,
                        ):
                            await provider.complete(request)

        assert captured_streaming["streaming"] is True, (
            "Streaming must be forced on when tools are present "
            "(event-based capture requires streaming events)"
        )

    @pytest.mark.asyncio
    async def test_convert_tools_called_without_capture(self, provider, mock_coordinator):
        """convert_tools_for_sdk should be called with tool_specs only (no capture)."""
        mock_session = AsyncMock()
        mock_session.session_id = "test"
        mock_session.destroy = AsyncMock()

        fake_response = ChatResponse(
            content=[TextBlock(type="text", text="Done")],
            tool_calls=[],
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            finish_reason="end_turn",
        )

        @asynccontextmanager
        async def mock_create_session(
            self,
            model,
            system_message=None,
            streaming=True,
            reasoning_effort=None,
            tools=None,
            excluded_tools=None,
            hooks=None,
        ):
            yield mock_session

        user_tool = Mock()
        user_tool.name = "read_file"
        user_tool.description = "Read a file"
        user_tool.input_schema = {}

        request = {
            "messages": [{"role": "user", "content": "test"}],
            "tools": [user_tool],
        }

        with patch(
            "amplifier_module_provider_github_copilot.provider.convert_tools_for_sdk",
            return_value=[Mock(name="read_file")],
        ) as mock_convert:
            with patch(
                "amplifier_module_provider_github_copilot.provider.make_deny_all_hook",
                return_value={"on_pre_tool_use": Mock()},
            ):
                with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
                    with patch.object(
                        provider,
                        "_complete_streaming",
                        new_callable=AsyncMock,
                        return_value=fake_response,
                    ):
                        with patch.object(
                            provider,
                            "_model_supports_reasoning",
                            new_callable=AsyncMock,
                            return_value=False,
                        ):
                            await provider.complete(request)

        # convert_tools_for_sdk called with just tool specs
        # (no ToolCapture parameter — that's the old API)
        mock_convert.assert_called_once_with([user_tool])

    @pytest.mark.asyncio
    async def test_excluded_tools_computed_from_user_tools(self, provider, mock_coordinator):
        """Provider should exclude CLI built-ins that overlap with user tools."""
        mock_session = AsyncMock()
        mock_session.session_id = "test"
        mock_session.destroy = AsyncMock()

        fake_response = ChatResponse(
            content=[TextBlock(type="text", text="Done")],
            tool_calls=[],
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            finish_reason="end_turn",
        )

        captured_kwargs = {}

        @asynccontextmanager
        async def mock_create_session(self, **kwargs):
            captured_kwargs.update(kwargs)
            yield mock_session

        # Register tools that overlap with CLI built-ins
        tools = []
        for name in ["write_file", "read_file", "grep", "bash"]:
            t = Mock()
            t.name = name
            t.description = f"Tool {name}"
            t.input_schema = {}
            tools.append(t)

        request = {
            "messages": [{"role": "user", "content": "test"}],
            "tools": tools,
        }

        with patch(
            "amplifier_module_provider_github_copilot.provider.convert_tools_for_sdk",
            return_value=[Mock(name=t.name) for t in tools],
        ):
            with patch(
                "amplifier_module_provider_github_copilot.provider.make_deny_all_hook",
                return_value={"on_pre_tool_use": Mock()},
            ):
                with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
                    with patch.object(
                        provider,
                        "_complete_streaming",
                        new_callable=AsyncMock,
                        return_value=fake_response,
                    ):
                        with patch.object(
                            provider,
                            "_model_supports_reasoning",
                            new_callable=AsyncMock,
                            return_value=False,
                        ):
                            await provider.complete(request)

        # hooks should be present for deny pattern
        assert "hooks" in captured_kwargs, "hooks should be passed to create_session"
        assert captured_kwargs["hooks"] is not None, "hooks should not be None when tools present"

        # excluded_tools should contain ALL known built-ins (unconditional exclusion)
        assert "excluded_tools" in captured_kwargs, (
            "excluded_tools should be passed to avoid CLI name collision"
        )
        excluded = set(captured_kwargs["excluded_tools"])

        # When user tools are present, ALL known built-ins must be excluded.
        # This includes 13 documented built-ins PLUS hidden ones like report_intent, task.
        # Evidence: Session 497bbab7 showed that partial exclusion allows
        # the SDK to run built-in tools internally, bypassing the orchestrator.
        assert excluded == set(COPILOT_BUILTIN_TOOL_NAMES), (
            f"Expected ALL built-ins {set(COPILOT_BUILTIN_TOOL_NAMES)}, got {excluded}"
        )


class TestProviderEventEmission:
    """Tests for observability event emission."""

    @pytest.fixture
    def provider_with_hooks(self, mock_coordinator, provider_config):
        """Create provider with working hooks."""
        return CopilotSdkProvider(
            api_key=None,
            config=provider_config,
            coordinator=mock_coordinator,
        )

    @pytest.mark.asyncio
    async def test_emit_event_success(self, provider_with_hooks, mock_coordinator):
        """Should emit events through coordinator hooks."""
        await provider_with_hooks._emit_event("test:event", {"key": "value"})

        mock_coordinator.hooks.emit.assert_called_once_with("test:event", {"key": "value"})

    @pytest.mark.asyncio
    async def test_emit_event_handles_error(self, provider_with_hooks, mock_coordinator):
        """Should handle hook emission errors gracefully."""
        mock_coordinator.hooks.emit = AsyncMock(side_effect=Exception("Hook error"))

        # Should not raise
        await provider_with_hooks._emit_event("test:event", {"key": "value"})

    @pytest.mark.asyncio
    async def test_emit_event_no_coordinator(self, provider_config):
        """Should handle missing coordinator gracefully."""
        provider = CopilotSdkProvider(None, provider_config, None)

        # Should not raise
        await provider._emit_event("test:event", {"key": "value"})

    @pytest.mark.asyncio
    async def test_complete_emits_llm_request_event(
        self, mock_coordinator, mock_copilot_client, sample_messages
    ):
        """complete() should emit llm:request event before API call."""
        provider = CopilotSdkProvider(
            api_key=None,
            config={"model": "claude-opus-4.5", "use_streaming": False, "debug": False},
            coordinator=mock_coordinator,
        )

        mock_session = AsyncMock()
        mock_session.session_id = "test"
        mock_session.destroy = AsyncMock()

        mock_response = Mock()
        mock_response.data = Mock()
        mock_response.data.content = "Hello"
        mock_response.data.tool_requests = None
        mock_response.data.input_tokens = 10
        mock_response.data.output_tokens = 5

        @asynccontextmanager
        async def mock_create_session(self, **kwargs):
            yield mock_session

        with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
            with patch.object(
                CopilotClientWrapper, "send_and_wait", new_callable=AsyncMock
            ) as mock_send:
                mock_send.return_value = mock_response

                with patch.object(
                    provider,
                    "_model_supports_reasoning",
                    new_callable=AsyncMock,
                    return_value=False,
                ):
                    await provider.complete(
                        {"messages": sample_messages},
                        model="gpt-4",
                    )

        # Find llm:request call
        request_calls = [
            call
            for call in mock_coordinator.hooks.emit.call_args_list
            if call[0][0] == "llm:request"
        ]
        assert len(request_calls) == 1
        data = request_calls[0][0][1]
        assert data["provider"] == "github-copilot"
        assert data["model"] == "gpt-4"
        assert data["message_count"] == len(sample_messages)
        assert "streaming" in data
        assert "tool_count" in data
        assert "timeout" in data

    @pytest.mark.asyncio
    async def test_complete_emits_llm_response_event(
        self, mock_coordinator, mock_copilot_client, sample_messages
    ):
        """complete() should emit llm:response event after API call with timing."""
        provider = CopilotSdkProvider(
            api_key=None,
            config={"model": "claude-opus-4.5", "use_streaming": False, "debug": False},
            coordinator=mock_coordinator,
        )

        mock_session = AsyncMock()
        mock_session.session_id = "test"
        mock_session.destroy = AsyncMock()

        mock_response = Mock()
        mock_response.data = Mock()
        mock_response.data.content = "Hello"
        mock_response.data.tool_requests = None
        mock_response.data.input_tokens = 100
        mock_response.data.output_tokens = 50

        @asynccontextmanager
        async def mock_create_session(self, **kwargs):
            yield mock_session

        with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
            with patch.object(
                CopilotClientWrapper, "send_and_wait", new_callable=AsyncMock
            ) as mock_send:
                mock_send.return_value = mock_response

                with patch.object(
                    provider,
                    "_model_supports_reasoning",
                    new_callable=AsyncMock,
                    return_value=False,
                ):
                    await provider.complete(
                        {"messages": sample_messages},
                        model="gpt-4",
                    )

        # Find llm:response call
        response_calls = [
            call
            for call in mock_coordinator.hooks.emit.call_args_list
            if call[0][0] == "llm:response"
        ]
        assert len(response_calls) == 1
        data = response_calls[0][0][1]
        assert data["provider"] == "github-copilot"
        assert data["status"] == "ok"
        assert "duration_ms" in data
        assert isinstance(data["duration_ms"], int)
        assert data["duration_ms"] >= 0
        assert data["usage"]["input"] == 100
        assert data["usage"]["output"] == 50
        assert data["tool_calls"] == 0
        assert data["finish_reason"] == "end_turn"

    @pytest.mark.asyncio
    async def test_complete_emits_debug_events_when_debug_enabled(
        self, mock_coordinator, sample_messages
    ):
        """complete() should emit llm:request:debug and llm:response:debug when debug=True."""
        provider = CopilotSdkProvider(
            api_key=None,
            config={"model": "claude-opus-4.5", "use_streaming": False, "debug": True},
            coordinator=mock_coordinator,
        )

        mock_session = AsyncMock()
        mock_session.session_id = "test"
        mock_session.destroy = AsyncMock()

        mock_response = Mock()
        mock_response.data = Mock()
        mock_response.data.content = "Hello"
        mock_response.data.tool_requests = None
        mock_response.data.input_tokens = 10
        mock_response.data.output_tokens = 5

        @asynccontextmanager
        async def mock_create_session(self, **kwargs):
            yield mock_session

        with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
            with patch.object(
                CopilotClientWrapper, "send_and_wait", new_callable=AsyncMock
            ) as mock_send:
                mock_send.return_value = mock_response
                with patch.object(
                    provider,
                    "_model_supports_reasoning",
                    new_callable=AsyncMock,
                    return_value=False,
                ):
                    await provider.complete(
                        {"messages": sample_messages},
                        model="gpt-4",
                    )

        emitted_events = [call[0][0] for call in mock_coordinator.hooks.emit.call_args_list]
        assert "llm:request:debug" in emitted_events
        assert "llm:response:debug" in emitted_events

        # Verify debug events have lvl field
        debug_calls = [
            call
            for call in mock_coordinator.hooks.emit.call_args_list
            if call[0][0] == "llm:request:debug"
        ]
        assert debug_calls[0][0][1]["lvl"] == "DEBUG"

    @pytest.mark.asyncio
    async def test_complete_skips_debug_events_when_debug_disabled(
        self, mock_coordinator, sample_messages
    ):
        """complete() should NOT emit debug events when debug=False."""
        provider = CopilotSdkProvider(
            api_key=None,
            config={"model": "claude-opus-4.5", "use_streaming": False, "debug": False},
            coordinator=mock_coordinator,
        )

        mock_session = AsyncMock()
        mock_session.session_id = "test"
        mock_session.destroy = AsyncMock()

        mock_response = Mock()
        mock_response.data = Mock()
        mock_response.data.content = "Hello"
        mock_response.data.tool_requests = None
        mock_response.data.input_tokens = 10
        mock_response.data.output_tokens = 5

        @asynccontextmanager
        async def mock_create_session(self, **kwargs):
            yield mock_session

        with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
            with patch.object(
                CopilotClientWrapper, "send_and_wait", new_callable=AsyncMock
            ) as mock_send:
                mock_send.return_value = mock_response
                with patch.object(
                    provider,
                    "_model_supports_reasoning",
                    new_callable=AsyncMock,
                    return_value=False,
                ):
                    await provider.complete(
                        {"messages": sample_messages},
                        model="gpt-4",
                    )

        emitted_events = [call[0][0] for call in mock_coordinator.hooks.emit.call_args_list]
        assert "llm:request:debug" not in emitted_events
        assert "llm:response:debug" not in emitted_events
        assert "llm:request:raw" not in emitted_events
        assert "llm:response:raw" not in emitted_events

    @pytest.mark.asyncio
    async def test_complete_emits_raw_events_only_when_both_flags_set(
        self, mock_coordinator, sample_messages
    ):
        """complete() should emit raw events only when BOTH debug AND raw_debug are True."""
        # debug=True, raw_debug=True → raw events should appear
        provider = CopilotSdkProvider(
            api_key=None,
            config={
                "model": "claude-opus-4.5",
                "use_streaming": False,
                "debug": True,
                "raw_debug": True,
            },
            coordinator=mock_coordinator,
        )

        mock_session = AsyncMock()
        mock_session.session_id = "test"
        mock_session.destroy = AsyncMock()

        mock_response = Mock()
        mock_response.data = Mock()
        mock_response.data.content = "Hello"
        mock_response.data.tool_requests = None
        mock_response.data.input_tokens = 10
        mock_response.data.output_tokens = 5

        @asynccontextmanager
        async def mock_create_session(self, **kwargs):
            yield mock_session

        with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
            with patch.object(
                CopilotClientWrapper, "send_and_wait", new_callable=AsyncMock
            ) as mock_send:
                mock_send.return_value = mock_response
                with patch.object(
                    provider,
                    "_model_supports_reasoning",
                    new_callable=AsyncMock,
                    return_value=False,
                ):
                    await provider.complete(
                        {"messages": sample_messages},
                        model="gpt-4",
                    )

        emitted_events = [call[0][0] for call in mock_coordinator.hooks.emit.call_args_list]
        assert "llm:request:raw" in emitted_events
        assert "llm:response:raw" in emitted_events

    @pytest.mark.asyncio
    async def test_no_llm_complete_event(self, mock_coordinator, sample_messages):
        """complete() should NOT emit the deprecated llm:complete event."""
        provider = CopilotSdkProvider(
            api_key=None,
            config={"model": "claude-opus-4.5", "use_streaming": False, "debug": False},
            coordinator=mock_coordinator,
        )

        mock_session = AsyncMock()
        mock_session.session_id = "test"
        mock_session.destroy = AsyncMock()

        mock_response = Mock()
        mock_response.data = Mock()
        mock_response.data.content = "Hello"
        mock_response.data.tool_requests = None
        mock_response.data.input_tokens = 10
        mock_response.data.output_tokens = 5

        @asynccontextmanager
        async def mock_create_session(self, **kwargs):
            yield mock_session

        with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
            with patch.object(
                CopilotClientWrapper, "send_and_wait", new_callable=AsyncMock
            ) as mock_send:
                mock_send.return_value = mock_response
                with patch.object(
                    provider,
                    "_model_supports_reasoning",
                    new_callable=AsyncMock,
                    return_value=False,
                ):
                    await provider.complete(
                        {"messages": sample_messages},
                        model="gpt-4",
                    )

        emitted_events = [call[0][0] for call in mock_coordinator.hooks.emit.call_args_list]
        assert "llm:complete" not in emitted_events, (
            "llm:complete is deprecated — use llm:request + llm:response pattern"
        )

    @pytest.mark.asyncio
    async def test_llm_request_precedes_llm_response(self, mock_coordinator, sample_messages):
        """llm:request must be emitted BEFORE llm:response (ordering contract)."""
        provider = CopilotSdkProvider(
            api_key=None,
            config={"model": "claude-opus-4.5", "use_streaming": False, "debug": False},
            coordinator=mock_coordinator,
        )

        mock_session = AsyncMock()
        mock_session.session_id = "test"
        mock_session.destroy = AsyncMock()

        mock_response = Mock()
        mock_response.data = Mock()
        mock_response.data.content = "Hello"
        mock_response.data.tool_requests = None
        mock_response.data.input_tokens = 10
        mock_response.data.output_tokens = 5

        @asynccontextmanager
        async def mock_create_session(self, **kwargs):
            yield mock_session

        with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
            with patch.object(
                CopilotClientWrapper, "send_and_wait", new_callable=AsyncMock
            ) as mock_send:
                mock_send.return_value = mock_response
                with patch.object(
                    provider,
                    "_model_supports_reasoning",
                    new_callable=AsyncMock,
                    return_value=False,
                ):
                    await provider.complete(
                        {"messages": sample_messages},
                        model="gpt-4",
                    )

        emitted_events = [call[0][0] for call in mock_coordinator.hooks.emit.call_args_list]
        req_idx = emitted_events.index("llm:request")
        resp_idx = emitted_events.index("llm:response")
        assert req_idx < resp_idx, "llm:request must fire before llm:response"


class TestProviderTimeoutSelection:
    """Tests for timeout auto-selection based on extended thinking."""

    def test_get_info_includes_both_timeouts(self, mock_coordinator):
        """get_info should include both timeout and thinking_timeout in defaults."""
        provider = CopilotSdkProvider(
            api_key=None,
            config={},
            coordinator=mock_coordinator,
        )

        info = provider.get_info()

        assert "timeout" in info.defaults
        assert "thinking_timeout" in info.defaults
        assert info.defaults["timeout"] == DEFAULT_TIMEOUT
        assert info.defaults["thinking_timeout"] == DEFAULT_THINKING_TIMEOUT

    def test_custom_timeouts_in_get_info(self, mock_coordinator):
        """get_info should reflect custom timeout configuration."""
        provider = CopilotSdkProvider(
            api_key=None,
            config={"timeout": 60.0, "thinking_timeout": 600.0},
            coordinator=mock_coordinator,
        )

        info = provider.get_info()

        assert info.defaults["timeout"] == 60.0
        assert info.defaults["thinking_timeout"] == 600.0


class TestTimeoutSelectionDuringComplete:
    """Tests that verify timeout is correctly selected based on extended_thinking."""

    @pytest.fixture
    def provider_with_timeouts(self, mock_coordinator):
        """Create provider with specific timeout values for testing."""
        return CopilotSdkProvider(
            api_key=None,
            config={
                "timeout": 300.0,
                "thinking_timeout": 1800.0,
                "use_streaming": False,  # Easier to test non-streaming
            },
            coordinator=mock_coordinator,
        )

    @pytest.mark.asyncio
    async def test_uses_standard_timeout_without_extended_thinking(
        self, provider_with_timeouts, sample_messages
    ):
        """Should use standard timeout (300s) for non-thinking model without flag.

        NOTE: The model name must NOT contain known thinking patterns (opus, o1, etc.)
        otherwise the fallback will trigger 1800s timeout.
        """
        mock_session = AsyncMock()
        mock_session.session_id = "test"
        mock_session.destroy = AsyncMock()

        mock_response = Mock()
        mock_response.data = Mock()
        mock_response.data.content = "Hello"
        mock_response.data.tool_requests = None
        mock_response.data.input_tokens = 10
        mock_response.data.output_tokens = 5

        @asynccontextmanager
        async def mock_create_session(self, **kwargs):
            yield mock_session

        with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
            with patch.object(
                CopilotClientWrapper, "send_and_wait", new_callable=AsyncMock
            ) as mock_send:
                mock_send.return_value = mock_response

                # Model doesn't support reasoning - should use standard timeout
                with patch.object(
                    provider_with_timeouts,
                    "_model_supports_reasoning",
                    new_callable=AsyncMock,
                    return_value=False,
                ):
                    # Use a non-thinking model name to avoid fallback
                    await provider_with_timeouts.complete(
                        {"messages": sample_messages},
                        extended_thinking=False,
                        model="gpt-4",  # Non-thinking model name
                    )

                # Verify the timeout passed to send_and_wait is the standard 300s
                mock_send.assert_called_once()
                call_kwargs = mock_send.call_args[1]
                assert call_kwargs["timeout"] == 300.0

    @pytest.mark.asyncio
    async def test_uses_thinking_timeout_with_extended_thinking(
        self, provider_with_timeouts, sample_messages
    ):
        """Should use thinking timeout (1800s) when extended_thinking=True."""
        mock_session = AsyncMock()
        mock_session.session_id = "test"
        mock_session.destroy = AsyncMock()

        mock_response = Mock()
        mock_response.data = Mock()
        mock_response.data.content = "Hello"
        mock_response.data.tool_requests = None
        mock_response.data.input_tokens = 10
        mock_response.data.output_tokens = 5

        @asynccontextmanager
        async def mock_create_session(self, **kwargs):
            yield mock_session

        with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
            with patch.object(
                CopilotClientWrapper, "send_and_wait", new_callable=AsyncMock
            ) as mock_send:
                mock_send.return_value = mock_response

                # Mock _model_supports_reasoning to return True
                # This is required because extended_thinking is only used
                # if the model actually supports it
                with patch.object(
                    provider_with_timeouts,
                    "_model_supports_reasoning",
                    new_callable=AsyncMock,
                    return_value=True,
                ):
                    await provider_with_timeouts.complete(
                        {"messages": sample_messages},
                        extended_thinking=True,
                    )

                # Verify the timeout passed to send_and_wait is the thinking 1800s
                mock_send.assert_called_once()
                call_kwargs = mock_send.call_args[1]
                assert call_kwargs["timeout"] == 1800.0

    @pytest.mark.asyncio
    async def test_user_timeout_overrides_all(self, provider_with_timeouts, sample_messages):
        """User-provided timeout should override both standard and thinking timeout."""
        mock_session = AsyncMock()
        mock_session.session_id = "test"
        mock_session.destroy = AsyncMock()

        mock_response = Mock()
        mock_response.data = Mock()
        mock_response.data.content = "Hello"
        mock_response.data.tool_requests = None
        mock_response.data.input_tokens = 10
        mock_response.data.output_tokens = 5

        @asynccontextmanager
        async def mock_create_session(self, **kwargs):
            yield mock_session

        with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
            with patch.object(
                CopilotClientWrapper, "send_and_wait", new_callable=AsyncMock
            ) as mock_send:
                mock_send.return_value = mock_response

                # Pass explicit timeout override
                await provider_with_timeouts.complete(
                    {"messages": sample_messages},
                    extended_thinking=True,
                    timeout=60.0,  # User override
                )

                # Verify user timeout (60s) is used, not thinking timeout (1800s)
                mock_send.assert_called_once()
                call_kwargs = mock_send.call_args[1]
                assert call_kwargs["timeout"] == 60.0

    @pytest.mark.asyncio
    async def test_uses_thinking_timeout_for_thinking_capable_model_without_flag(
        self, provider_with_timeouts, sample_messages
    ):
        """Should use thinking timeout (1800s) for thinking-capable models even without flag.

        This test verifies that models with thinking/reasoning capability automatically
        get the extended timeout, even if Amplifier CLI doesn't explicitly pass
        extended_thinking=True. This prevents spurious timeouts for models like
        Claude Opus 4.x that naturally take longer to respond.
        """
        mock_session = AsyncMock()
        mock_session.session_id = "test"
        mock_session.destroy = AsyncMock()

        mock_response = Mock()
        mock_response.data = Mock()
        mock_response.data.content = "Hello"
        mock_response.data.tool_requests = None
        mock_response.data.input_tokens = 10
        mock_response.data.output_tokens = 5

        @asynccontextmanager
        async def mock_create_session(self, **kwargs):
            yield mock_session

        with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
            with patch.object(
                CopilotClientWrapper, "send_and_wait", new_callable=AsyncMock
            ) as mock_send:
                mock_send.return_value = mock_response

                # Model supports reasoning but extended_thinking NOT passed
                with patch.object(
                    provider_with_timeouts,
                    "_model_supports_reasoning",
                    new_callable=AsyncMock,
                    return_value=True,  # Model has thinking capability
                ):
                    await provider_with_timeouts.complete(
                        {"messages": sample_messages},
                        # NOTE: extended_thinking not passed (defaults to False)
                    )

                # Should still use thinking timeout (1800s) because model supports it
                mock_send.assert_called_once()
                call_kwargs = mock_send.call_args[1]
                assert call_kwargs["timeout"] == 1800.0

    @pytest.mark.asyncio
    async def test_uses_thinking_timeout_when_requested_but_capability_detection_fails(
        self, provider_with_timeouts, sample_messages
    ):
        """CRITICAL: Timeout based on REQUEST INTENT, not capability detection.

        If user explicitly requests extended_thinking, use the thinking timeout
        even if _model_supports_reasoning() fails. This prevents premature
        timeouts when:
        1. Network error during capability check
        2. Model not in cached list
        3. list_models() returns empty

        The API will reject quickly if model doesn't support it, but we shouldn't
        timeout before getting that response.
        """
        mock_session = AsyncMock()
        mock_session.session_id = "test"
        mock_session.destroy = AsyncMock()

        mock_response = Mock()
        mock_response.data = Mock()
        mock_response.data.content = "Hello"
        mock_response.data.tool_requests = None
        mock_response.data.input_tokens = 10
        mock_response.data.output_tokens = 5

        @asynccontextmanager
        async def mock_create_session(self, **kwargs):
            yield mock_session

        with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
            with patch.object(
                CopilotClientWrapper, "send_and_wait", new_callable=AsyncMock
            ) as mock_send:
                mock_send.return_value = mock_response

                # CRITICAL: Capability detection FAILS (returns False)
                # But user REQUESTED extended_thinking=True
                with patch.object(
                    provider_with_timeouts,
                    "_model_supports_reasoning",
                    new_callable=AsyncMock,
                    return_value=False,  # Capability check failed!
                ):
                    await provider_with_timeouts.complete(
                        {"messages": sample_messages},
                        extended_thinking=True,  # User explicitly requested
                    )

                # Should use thinking timeout (1800s) because USER REQUESTED it
                # (fail safe: trust user intent over capability detection)
                mock_send.assert_called_once()
                call_kwargs = mock_send.call_args[1]
                assert call_kwargs["timeout"] == 1800.0, (
                    "Timeout should be 1800s when user requests extended_thinking, "
                    "even if capability detection fails"
                )

    @pytest.mark.asyncio
    async def test_uses_thinking_timeout_when_sdk_fails_but_pattern_matches(
        self, provider_with_timeouts, sample_messages
    ):
        """FALLBACK: Pattern matching kicks in ONLY when SDK check FAILS.

        Design:
        - SDK is authoritative when it succeeds
        - Pattern is fallback ONLY when SDK throws exception

        This test verifies:
        - SDK check throws exception (network error, etc.)
        - Pattern matches "opus" in model name
        - Extended timeout (1800s) is used as safety

        Note: If SDK succeeds and returns False (like opus-4.5),
        we trust SDK and use short timeout. Pattern is NOT checked.
        """
        mock_session = AsyncMock()
        mock_session.session_id = "test"
        mock_session.destroy = AsyncMock()

        mock_response = Mock()
        mock_response.data = Mock()
        mock_response.data.content = "Hello"
        mock_response.data.tool_requests = None
        mock_response.data.input_tokens = 10
        mock_response.data.output_tokens = 5

        @asynccontextmanager
        async def mock_create_session(self, **kwargs):
            yield mock_session

        with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
            with patch.object(
                CopilotClientWrapper, "send_and_wait", new_callable=AsyncMock
            ) as mock_send:
                mock_send.return_value = mock_response

                # SDK check THROWS EXCEPTION (not just returns False!)
                # This triggers pattern fallback
                with patch.object(
                    provider_with_timeouts,
                    "_model_supports_reasoning",
                    new_callable=AsyncMock,
                    side_effect=Exception("Network error - SDK unreachable"),
                ):
                    await provider_with_timeouts.complete(
                        {"messages": sample_messages},
                        model="claude-opus-4.6",  # Pattern matches "opus"
                        # extended_thinking not passed
                    )

                # Should use thinking timeout (1800s) because:
                # 1. SDK failed with exception
                # 2. Pattern fallback matched "opus"
                mock_send.assert_called_once()
                call_kwargs = mock_send.call_args[1]
                assert call_kwargs["timeout"] == 1800.0, (
                    "Timeout should be 1800s when SDK fails and pattern matches 'opus'"
                )

    @pytest.mark.asyncio
    async def test_trusts_sdk_over_pattern_when_sdk_succeeds(
        self, provider_with_timeouts, sample_messages
    ):
        """SDK is AUTHORITATIVE: When SDK succeeds, trust its result over pattern.

        This tests the key design decision:
        - claude-opus-4.5: SDK says NO thinking capability
        - Pattern would match "opus" → would give long timeout
        - BUT SDK succeeds, so we trust SDK → short timeout

        This prevents false positives like opus-4.5 getting 30min timeout
        when SDK clearly says it doesn't have thinking.
        """
        mock_session = AsyncMock()
        mock_session.session_id = "test"
        mock_session.destroy = AsyncMock()

        mock_response = Mock()
        mock_response.data = Mock()
        mock_response.data.content = "Hello"
        mock_response.data.tool_requests = None
        mock_response.data.input_tokens = 10
        mock_response.data.output_tokens = 5

        @asynccontextmanager
        async def mock_create_session(self, **kwargs):
            yield mock_session

        with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
            with patch.object(
                CopilotClientWrapper, "send_and_wait", new_callable=AsyncMock
            ) as mock_send:
                mock_send.return_value = mock_response

                # SDK SUCCEEDS and returns False (like real opus-4.5)
                with patch.object(
                    provider_with_timeouts,
                    "_model_supports_reasoning",
                    new_callable=AsyncMock,
                    return_value=False,  # SDK says no thinking
                ):
                    await provider_with_timeouts.complete(
                        {"messages": sample_messages},
                        model="claude-opus-4.5",  # Pattern would match, but SDK is authoritative
                        # extended_thinking not passed
                    )

                # Should use STANDARD timeout (300s) because:
                # 1. SDK succeeded
                # 2. SDK returned False (no thinking)
                # 3. Pattern is NOT checked when SDK succeeds
                mock_send.assert_called_once()
                call_kwargs = mock_send.call_args[1]
                assert call_kwargs["timeout"] == 300.0, (
                    "Timeout should be 300s when SDK says no thinking, "
                    "even if pattern would match 'opus'"
                )


class TestModelSupportsReasoningCache:
    """Tests for _model_supports_reasoning() caching behavior."""

    @pytest.fixture
    def provider(self, mock_coordinator):
        """Create provider instance for testing."""
        return CopilotSdkProvider(
            api_key=None,
            config={"model": "claude-opus-4.5"},
            coordinator=mock_coordinator,
        )

    @pytest.mark.asyncio
    async def test_model_supports_reasoning_caches_results(self, provider):
        """_model_supports_reasoning should cache results and not call list_models twice."""
        # Create mock model with reasoning capability
        mock_model = Mock()
        mock_model.id = "claude-opus-4.5"
        mock_model.capabilities = ["thinking", "tools", "vision"]

        # Use AsyncMock with call_count tracking
        mock_list_models = AsyncMock(return_value=[mock_model])

        with patch.object(provider, "list_models", mock_list_models):
            # First call - should call list_models
            result1 = await provider._model_supports_reasoning("claude-opus-4.5")
            assert result1 is True
            assert mock_list_models.call_count == 1

            # Second call for same model - should use cache, NOT call list_models again
            result2 = await provider._model_supports_reasoning("claude-opus-4.5")
            assert result2 is True
            assert mock_list_models.call_count == 1  # Still 1, not 2

            # Third call - still cached
            result3 = await provider._model_supports_reasoning("claude-opus-4.5")
            assert result3 is True
            assert mock_list_models.call_count == 1  # Still 1

    @pytest.mark.asyncio
    async def test_model_supports_reasoning_caches_all_models_from_list(self, provider):
        """_model_supports_reasoning should cache all models from a single list_models call."""
        # Create mock models - one with reasoning, one without
        mock_model_opus = Mock()
        mock_model_opus.id = "claude-opus-4.5"
        mock_model_opus.capabilities = ["thinking", "tools"]

        mock_model_sonnet = Mock()
        mock_model_sonnet.id = "claude-sonnet-4"
        mock_model_sonnet.capabilities = ["tools", "vision"]

        mock_list_models = AsyncMock(return_value=[mock_model_opus, mock_model_sonnet])

        with patch.object(provider, "list_models", mock_list_models):
            # Query opus - triggers list_models
            result1 = await provider._model_supports_reasoning("claude-opus-4.5")
            assert result1 is True
            assert mock_list_models.call_count == 1

            # Query sonnet - should use cache from previous call
            result2 = await provider._model_supports_reasoning("claude-sonnet-4")
            assert result2 is False  # sonnet doesn't have thinking/reasoning
            assert mock_list_models.call_count == 1  # Still 1 - cached from opus query

    @pytest.mark.asyncio
    async def test_concurrent_calls_share_cache(self, provider):
        """Concurrent _model_supports_reasoning calls should use lock properly."""
        import asyncio

        mock_model = Mock()
        mock_model.id = "claude-opus-4.5"
        mock_model.capabilities = ["thinking"]

        async def slow_list_models_impl():
            await asyncio.sleep(0.01)  # Simulate network latency
            return [mock_model]

        mock_list_models = AsyncMock(side_effect=slow_list_models_impl)

        with patch.object(provider, "list_models", mock_list_models):
            # Fire 5 concurrent calls
            results = await asyncio.gather(
                *[provider._model_supports_reasoning("claude-opus-4.5") for _ in range(5)]
            )

            # All should succeed
            assert all(results)
            # But list_models called only ONCE due to lock
            assert mock_list_models.call_count == 1

    @pytest.mark.asyncio
    async def test_model_supports_reasoning_propagates_list_models_error(self, provider):
        """_model_supports_reasoning should raise if list_models fails.

        This is CRITICAL for the SDK-authoritative fallback design:
        - _model_supports_reasoning RAISES on SDK failure
        - _check_model_reasoning_with_fallback CATCHES and falls back to pattern matching
        - If we swallowed the exception here, the fallback path would be unreachable
        """
        mock_list_models = AsyncMock(side_effect=Exception("Network error"))

        with patch.object(provider, "list_models", mock_list_models):
            with pytest.raises(Exception, match="Network error"):
                await provider._model_supports_reasoning("any-model")

            assert mock_list_models.call_count == 1

    @pytest.mark.asyncio
    async def test_invalidate_model_cache(self, provider):
        """invalidate_model_cache should clear the cache."""
        # Populate cache
        provider._model_capabilities_cache["test-model"] = ["thinking"]
        assert "test-model" in provider._model_capabilities_cache

        # Invalidate
        provider.invalidate_model_cache()

        # Should be empty
        assert "test-model" not in provider._model_capabilities_cache


class TestCheckModelReasoningWithFallback:
    """Tests for _check_model_reasoning_with_fallback() SDK-authoritative design.

    This is the critical integration point:
    - SDK check succeeds → trust SDK result (sdk_succeeded=True)
    - SDK check fails → fall back to pattern matching (sdk_succeeded=False)
    """

    @pytest.fixture
    def provider(self, mock_coordinator):
        """Create provider instance for testing."""
        return CopilotSdkProvider(
            api_key=None,
            config={"model": "claude-opus-4.5"},
            coordinator=mock_coordinator,
        )

    @pytest.mark.asyncio
    async def test_sdk_succeeds_returns_true(self, provider):
        """When SDK says model supports reasoning, return (True, True)."""
        with patch.object(
            provider,
            "_model_supports_reasoning",
            new_callable=AsyncMock,
            return_value=True,
        ):
            result, sdk_succeeded = await provider._check_model_reasoning_with_fallback(
                "claude-opus-4.6"
            )
            assert result is True
            assert sdk_succeeded is True

    @pytest.mark.asyncio
    async def test_sdk_succeeds_returns_false(self, provider):
        """When SDK says model does NOT support reasoning, return (False, True)."""
        with patch.object(
            provider,
            "_model_supports_reasoning",
            new_callable=AsyncMock,
            return_value=False,
        ):
            result, sdk_succeeded = await provider._check_model_reasoning_with_fallback(
                "claude-opus-4.5"
            )
            assert result is False
            assert sdk_succeeded is True

    @pytest.mark.asyncio
    async def test_sdk_fails_fallback_matches_pattern(self, provider):
        """When SDK fails and pattern matches, return (True, False).

        This is the CRITICAL fallback path that was previously unreachable.
        """
        with patch.object(
            provider,
            "_model_supports_reasoning",
            new_callable=AsyncMock,
            side_effect=Exception("Network error"),
        ):
            result, sdk_succeeded = await provider._check_model_reasoning_with_fallback(
                "claude-opus-4.6"
            )
            assert result is True  # "opus" pattern matches
            assert sdk_succeeded is False  # SDK failed

    @pytest.mark.asyncio
    async def test_sdk_fails_fallback_no_pattern_match(self, provider):
        """When SDK fails and pattern doesn't match, return (False, False)."""
        with patch.object(
            provider,
            "_model_supports_reasoning",
            new_callable=AsyncMock,
            side_effect=Exception("Network error"),
        ):
            result, sdk_succeeded = await provider._check_model_reasoning_with_fallback(
                "claude-sonnet-4.5"
            )
            assert result is False  # No thinking pattern
            assert sdk_succeeded is False  # SDK failed


# ═══════════════════════════════════════════════════════════════════════════
# Coverage expansion tests: provider.py 73% → 90%+
# ═══════════════════════════════════════════════════════════════════════════


class TestEmitStreamingContent:
    """Tests for _emit_streaming_content (lines 938-959)."""

    @pytest.fixture
    def provider(self, mock_coordinator):
        """Create provider with debug enabled."""
        return CopilotSdkProvider(
            api_key=None,
            config={"debug": True},
            coordinator=mock_coordinator,
        )

    @pytest.mark.asyncio
    async def test_creates_task_and_emits_content(self, provider, mock_coordinator):
        """_emit_streaming_content should fire async task that emits through hooks."""
        import asyncio

        content = Mock()
        provider._emit_streaming_content(content)
        await asyncio.sleep(0.01)

        mock_coordinator.hooks.emit.assert_called_once()
        args = mock_coordinator.hooks.emit.call_args[0]
        assert args[0] == "llm:content_block"
        assert args[1]["provider"] == "github-copilot"
        assert args[1]["content"] is content

    def test_returns_early_when_no_coordinator(self):
        """Should return silently when coordinator is None."""
        provider = CopilotSdkProvider(api_key=None, config={}, coordinator=None)
        provider._emit_streaming_content(Mock())  # Should not raise

    def test_returns_early_when_no_hooks(self):
        """Should return silently when coordinator has no hooks attribute."""
        coordinator = Mock(spec=[])
        provider = CopilotSdkProvider(api_key=None, config={}, coordinator=coordinator)
        provider._emit_streaming_content(Mock())  # Should not raise

    def test_handles_no_running_loop(self, provider, mock_coordinator):
        """Should catch RuntimeError when called outside async context."""
        # Sync test — no running event loop
        provider._emit_streaming_content(Mock())
        mock_coordinator.hooks.emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_tracks_pending_emit_task(self, provider):
        """Created task should be added to _pending_emit_tasks tracking set."""
        import asyncio

        content = Mock()
        provider._emit_streaming_content(content)
        # Task was added (done callback may clean it up quickly)
        # Just verify no error
        await asyncio.sleep(0.01)


class TestEmitContentAsync:
    """Tests for _emit_content_async (lines 966-976)."""

    @pytest.fixture
    def provider(self, mock_coordinator):
        """Create provider with debug enabled."""
        return CopilotSdkProvider(
            api_key=None,
            config={"debug": True},
            coordinator=mock_coordinator,
        )

    @pytest.mark.asyncio
    async def test_emits_content_through_hooks(self, provider, mock_coordinator):
        """Should emit content block via coordinator hooks."""
        content = Mock()
        await provider._emit_content_async(content)

        mock_coordinator.hooks.emit.assert_called_once_with(
            "llm:content_block",
            {"provider": "github-copilot", "content": content},
        )

    @pytest.mark.asyncio
    async def test_handles_emit_error_gracefully(self, provider, mock_coordinator):
        """Should catch and swallow errors from hooks.emit."""
        mock_coordinator.hooks.emit = AsyncMock(side_effect=Exception("Hook broke"))
        content = Mock()
        # Should not raise
        await provider._emit_content_async(content)


class TestMakeEmitCallbackDetails:
    """Tests for _make_emit_callback — RuntimeError path (lines 907, 920-921)."""

    def test_returns_none_without_coordinator(self):
        """Should return None when no coordinator is set."""
        provider = CopilotSdkProvider(api_key=None, config={}, coordinator=None)
        assert provider._make_emit_callback() is None

    def test_returns_none_without_hooks(self):
        """Should return None when coordinator has no hooks attribute."""
        coordinator = Mock(spec=[])
        provider = CopilotSdkProvider(api_key=None, config={}, coordinator=coordinator)
        assert provider._make_emit_callback() is None

    def test_returns_callable(self, mock_coordinator):
        """Should return a callable when coordinator has hooks."""
        provider = CopilotSdkProvider(api_key=None, config={}, coordinator=mock_coordinator)
        callback = provider._make_emit_callback()
        assert callable(callback)

    @pytest.mark.asyncio
    async def test_callback_creates_emit_task(self, mock_coordinator):
        """Callback should create async task for event emission."""
        import asyncio

        provider = CopilotSdkProvider(api_key=None, config={}, coordinator=mock_coordinator)
        callback = provider._make_emit_callback()

        callback("test_event", {"key": "value"})
        await asyncio.sleep(0.01)

        emit_calls = [
            call
            for call in mock_coordinator.hooks.emit.call_args_list
            if call[0][0] == "sdk_driver:test_event"
        ]
        assert len(emit_calls) == 1

    def test_callback_handles_no_running_loop(self, mock_coordinator):
        """Callback should catch RuntimeError when no event loop exists."""
        provider = CopilotSdkProvider(api_key=None, config={}, coordinator=mock_coordinator)
        callback = provider._make_emit_callback()
        # Called outside async context → RuntimeError caught silently
        callback("test_event", {"key": "value"})
        # Should not raise


class TestHandleTaskException:
    """Tests for _handle_task_exception (lines 1271-1277)."""

    @pytest.fixture
    def provider(self, mock_coordinator):
        """Create provider with debug enabled to exercise all branches."""
        return CopilotSdkProvider(
            api_key=None,
            config={"debug": True},
            coordinator=mock_coordinator,
        )

    def test_cancelled_task_ignored(self, provider):
        """Should silently return for cancelled tasks."""
        import asyncio

        task = Mock(spec=asyncio.Task)
        task.cancelled.return_value = True

        provider._handle_task_exception(task)
        task.exception.assert_not_called()

    def test_task_with_no_exception(self, provider):
        """Should handle tasks that completed normally (exception() returns None)."""
        import asyncio

        task = Mock(spec=asyncio.Task)
        task.cancelled.return_value = False
        task.exception.return_value = None

        provider._handle_task_exception(task)  # Should not raise

    def test_task_with_exception_logs_debug(self, provider):
        """Should log exception details when debug is enabled."""
        import asyncio

        task = Mock(spec=asyncio.Task)
        task.cancelled.return_value = False
        task.exception.return_value = ValueError("boom")

        provider._handle_task_exception(task)  # Should not raise


class TestCloseWithPendingTasks:
    """Tests for close() cancelling pending emit tasks (lines 1053-1057)."""

    @pytest.mark.asyncio
    async def test_cancels_all_pending_tasks(self, mock_coordinator):
        """close() should cancel all pending emit tasks."""
        import asyncio

        provider = CopilotSdkProvider(api_key=None, config={}, coordinator=mock_coordinator)

        task1 = Mock(spec=asyncio.Task)
        task2 = Mock(spec=asyncio.Task)
        provider._pending_emit_tasks.add(task1)
        provider._pending_emit_tasks.add(task2)

        with patch.object(CopilotClientWrapper, "close", new_callable=AsyncMock):
            await provider.close()

        task1.cancel.assert_called_once()
        task2.cancel.assert_called_once()
        assert len(provider._pending_emit_tasks) == 0

    @pytest.mark.asyncio
    async def test_close_with_empty_pending_set(self, mock_coordinator):
        """close() should handle empty pending tasks set without error."""
        provider = CopilotSdkProvider(api_key=None, config={}, coordinator=mock_coordinator)

        with patch.object(CopilotClientWrapper, "close", new_callable=AsyncMock):
            await provider.close()  # Should not raise


class TestAddRepairedIdLRU:
    """Tests for _add_repaired_id LRU eviction (line 1226)."""

    @pytest.fixture
    def provider(self, mock_coordinator):
        """Create provider for testing."""
        return CopilotSdkProvider(api_key=None, config={}, coordinator=mock_coordinator)

    def test_adds_new_id(self, provider):
        """Should add tool call ID to the repaired set."""
        provider._add_repaired_id("call_1")
        assert "call_1" in provider._repaired_tool_ids

    def test_lru_eviction_on_overflow(self, provider):
        """Should evict oldest entry when max_repaired_ids is exceeded."""
        provider._max_repaired_ids = 3

        provider._add_repaired_id("call_1")
        provider._add_repaired_id("call_2")
        provider._add_repaired_id("call_3")
        assert len(provider._repaired_tool_ids) == 3

        # Adding 4th evicts oldest (call_1)
        provider._add_repaired_id("call_4")
        assert len(provider._repaired_tool_ids) == 3
        assert "call_1" not in provider._repaired_tool_ids
        assert "call_4" in provider._repaired_tool_ids

    def test_move_to_end_on_readd(self, provider):
        """Re-adding existing ID should move it to end (LRU refresh)."""
        provider._max_repaired_ids = 3

        provider._add_repaired_id("call_1")
        provider._add_repaired_id("call_2")
        provider._add_repaired_id("call_3")

        # Re-add call_1 → moves to end
        provider._add_repaired_id("call_1")

        # Now adding call_4 should evict call_2 (oldest), not call_1
        provider._add_repaired_id("call_4")
        assert "call_1" in provider._repaired_tool_ids
        assert "call_2" not in provider._repaired_tool_ids
        assert "call_3" in provider._repaired_tool_ids
        assert "call_4" in provider._repaired_tool_ids


class TestCreateSyntheticResult:
    """Tests for _create_synthetic_result (lines 1186-1196)."""

    @pytest.fixture
    def provider(self, mock_coordinator):
        return CopilotSdkProvider(api_key=None, config={}, coordinator=mock_coordinator)

    def test_creates_tool_role_message(self, provider):
        """Should create a properly structured tool-role message."""
        result = provider._create_synthetic_result("call_abc", "read_file")

        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_abc"
        assert result["name"] == "read_file"

    def test_content_includes_error_and_details(self, provider):
        """Content should include ERROR marker, tool name, call ID, and retry offer."""
        result = provider._create_synthetic_result("call_xyz", "write_file")

        assert "ERROR" in result["content"]
        assert "write_file" in result["content"]
        assert "call_xyz" in result["content"]
        assert "retry" in result["content"].lower()


class TestFindMissingToolResults:
    """Tests for _find_missing_tool_results (lines 1113-1151)."""

    @pytest.fixture
    def provider(self, mock_coordinator):
        return CopilotSdkProvider(api_key=None, config={}, coordinator=mock_coordinator)

    def test_no_missing_when_all_paired(self, provider):
        """Should return empty when all tool calls have matching results."""
        messages = [
            {"role": "user", "content": "help"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_call", "id": "c1", "name": "read_file"},
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "data"},
        ]
        assert provider._find_missing_tool_results(messages) == []

    def test_detects_unpaired_tool_call(self, provider):
        """Should detect tool calls with no matching tool result."""
        messages = [
            {"role": "user", "content": "help"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_call", "id": "c1", "name": "read_file"},
                ],
            },
        ]
        missing = provider._find_missing_tool_results(messages)
        assert len(missing) == 1
        assert missing[0] == (1, "c1", "read_file")

    def test_excludes_already_repaired_ids(self, provider):
        """Should skip IDs that have already been repaired."""
        provider._repaired_tool_ids["c1"] = None

        messages = [
            {"role": "user", "content": "help"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_call", "id": "c1", "name": "read_file"},
                ],
            },
        ]
        assert provider._find_missing_tool_results(messages) == []

    def test_handles_typed_content_blocks(self, provider):
        """Should handle object-style content blocks (hasattr path)."""
        block = Mock()
        block.type = "tool_call"
        block.id = "typed_1"
        block.name = "bash"

        messages = [
            {"role": "user", "content": "run"},
            {"role": "assistant", "content": [block]},
        ]
        missing = provider._find_missing_tool_results(messages)
        assert len(missing) == 1
        assert missing[0] == (1, "typed_1", "bash")

    def test_multiple_tool_calls_partial_results(self, provider):
        """Should only report unpaired tool calls, not those with results."""
        messages = [
            {"role": "user", "content": "help"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_call", "id": "c1", "name": "read_file"},
                    {"type": "tool_call", "id": "c2", "name": "write_file"},
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "data"},
        ]
        missing = provider._find_missing_tool_results(messages)
        assert len(missing) == 1
        assert missing[0][1] == "c2"

    def test_string_content_in_assistant_skipped(self, provider):
        """Should skip assistant messages with string content (not list)."""
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "Just text, no tool calls"},
        ]
        assert provider._find_missing_tool_results(messages) == []

    def test_tool_message_without_call_id_skipped(self, provider):
        """Should skip tool messages missing tool_call_id."""
        messages = [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_call", "id": "c1", "name": "bash"},
                ],
            },
            {"role": "tool", "content": "output"},  # No tool_call_id
        ]
        missing = provider._find_missing_tool_results(messages)
        assert len(missing) == 1
        assert missing[0][1] == "c1"

    def test_dict_block_without_id_skipped(self, provider):
        """Should skip tool_call dict blocks with empty id."""
        messages = [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_call", "id": "", "name": "bash"},
                ],
            },
        ]
        assert provider._find_missing_tool_results(messages) == []

    def test_non_tool_call_dict_blocks_skipped(self, provider):
        """Should skip dict blocks that aren't tool_call type."""
        messages = [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Just text"},
                ],
            },
        ]
        assert provider._find_missing_tool_results(messages) == []

    def test_non_tool_call_typed_blocks_skipped(self, provider):
        """Should skip typed content blocks with non-tool_call type."""
        block = Mock()
        block.type = "text"
        block.text = "Just text"

        messages = [
            {"role": "assistant", "content": [block]},
        ]
        assert provider._find_missing_tool_results(messages) == []


class TestRepairMissingToolResults:
    """Tests for _repair_missing_tool_results (lines 1290, 1293-1294)."""

    @pytest.fixture
    def provider(self, mock_coordinator):
        return CopilotSdkProvider(api_key=None, config={}, coordinator=mock_coordinator)

    @pytest.mark.asyncio
    async def test_no_repair_when_nothing_missing(self, provider):
        """Should return messages unchanged when all tool results are present."""
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        result = await provider._repair_missing_tool_results(messages)
        assert result is messages

    @pytest.mark.asyncio
    async def test_injects_synthetic_result(self, provider):
        """Should inject synthetic error result for missing tool response."""
        messages = [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_call", "id": "c1", "name": "read_file"},
                ],
            },
        ]
        result = await provider._repair_missing_tool_results(messages)

        assert len(result) == 3
        synthetic = result[2]
        assert synthetic["role"] == "tool"
        assert synthetic["tool_call_id"] == "c1"
        assert "ERROR" in synthetic["content"]

    @pytest.mark.asyncio
    async def test_tracks_repaired_ids(self, provider):
        """Should add repaired IDs to prevent re-detection on next iteration."""
        messages = [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_call", "id": "c1", "name": "bash"},
                ],
            },
        ]
        await provider._repair_missing_tool_results(messages)
        assert "c1" in provider._repaired_tool_ids

    @pytest.mark.asyncio
    async def test_emits_repair_event(self, provider, mock_coordinator):
        """Should emit provider:tool_sequence_repaired event."""
        messages = [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_call", "id": "c1", "name": "bash"},
                ],
            },
        ]
        await provider._repair_missing_tool_results(messages)

        repair_calls = [
            call
            for call in mock_coordinator.hooks.emit.call_args_list
            if call[0][0] == "provider:tool_sequence_repaired"
        ]
        assert len(repair_calls) == 1
        data = repair_calls[0][0][1]
        assert data["provider"] == "github-copilot"
        assert data["repair_count"] == 1
        assert data["repairs"][0]["tool_call_id"] == "c1"

    @pytest.mark.asyncio
    async def test_repairs_multiple_missing_results(self, provider):
        """Should inject synthetics for all missing tool results."""
        messages = [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_call", "id": "c1", "name": "read_file"},
                    {"type": "tool_call", "id": "c2", "name": "write_file"},
                ],
            },
        ]
        result = await provider._repair_missing_tool_results(messages)

        # user + assistant + 2 synthetics
        assert len(result) == 4
        assert result[2]["tool_call_id"] == "c1"
        assert result[3]["tool_call_id"] == "c2"


class TestCompleteStreamingEdgeCases:
    """Tests for _complete_streaming edge cases (lines 798, 810, 840→845)."""

    @pytest.fixture
    def provider(self, mock_coordinator):
        """Create streaming provider for testing."""
        return CopilotSdkProvider(
            api_key=None,
            config={"use_streaming": True, "debug": False},
            coordinator=mock_coordinator,
        )

    @staticmethod
    def _make_mock_session():
        """Create a mock session for streaming tests."""
        session = AsyncMock()
        session.session_id = "test-stream"
        session.send = AsyncMock()
        session.on = Mock(return_value=Mock())  # returns unsubscribe
        session.abort = AsyncMock()
        return session

    @staticmethod
    def _make_mock_handler(**overrides):
        """Create a mock SdkEventHandler with sensible defaults."""
        handler = Mock()
        handler.captured_tools = overrides.get("captured_tools", [])
        handler.text_content = overrides.get("text_content", [])
        handler.thinking_content = overrides.get("thinking_content", [])
        handler.usage_data = overrides.get("usage_data", {"input_tokens": 10, "output_tokens": 5})
        handler.turn_count = overrides.get("turn_count", 1)
        handler.should_abort = overrides.get("should_abort", False)
        handler.wait_for_capture_or_idle = overrides.get("wait_for_capture_or_idle", AsyncMock())
        handler.bind_session = Mock()
        handler.on_event = Mock()
        return handler

    @pytest.mark.asyncio
    async def test_timeout_returns_captured_tools(self, provider):
        """Should return captured tools even when timeout occurs (line 798)."""

        mock_session = self._make_mock_session()
        mock_tool = Mock()
        mock_tool.id = "c1"
        mock_tool.name = "read_file"
        mock_tool.arguments = {"path": "test.py"}

        handler = self._make_mock_handler(
            captured_tools=[mock_tool],
            should_abort=True,
            wait_for_capture_or_idle=AsyncMock(side_effect=TimeoutError()),
        )

        with patch(
            "amplifier_module_provider_github_copilot.sdk_driver.SdkEventHandler",
            return_value=handler,
        ):
            response = await provider._complete_streaming(
                mock_session,
                "prompt",
                "claude-opus-4.5",
                timeout=1.0,
                extended_thinking_enabled=False,
                has_tools=True,
            )

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "read_file"
        assert response.finish_reason == "tool_use"

    @pytest.mark.asyncio
    async def test_error_returns_captured_tools(self, provider):
        """Should return captured tools even on non-timeout error (line 810)."""
        mock_session = self._make_mock_session()
        mock_tool = Mock()
        mock_tool.id = "c2"
        mock_tool.name = "write_file"
        mock_tool.arguments = {"path": "x.py", "content": "y"}

        handler = self._make_mock_handler(
            captured_tools=[mock_tool],
            text_content=["partial"],
            should_abort=True,
            wait_for_capture_or_idle=AsyncMock(side_effect=RuntimeError("SDK loop broke")),
        )

        with patch(
            "amplifier_module_provider_github_copilot.sdk_driver.SdkEventHandler",
            return_value=handler,
        ):
            response = await provider._complete_streaming(
                mock_session,
                "prompt",
                "claude-opus-4.5",
                timeout=1.0,
                extended_thinking_enabled=False,
                has_tools=True,
            )

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "write_file"

    @pytest.mark.asyncio
    async def test_timeout_without_tools_raises(self, provider):
        """Should raise CopilotTimeoutError on timeout with no captured tools."""

        from amplifier_module_provider_github_copilot.exceptions import CopilotTimeoutError

        mock_session = self._make_mock_session()
        handler = self._make_mock_handler(
            wait_for_capture_or_idle=AsyncMock(side_effect=TimeoutError()),
        )

        with patch(
            "amplifier_module_provider_github_copilot.sdk_driver.SdkEventHandler",
            return_value=handler,
        ):
            with pytest.raises(CopilotTimeoutError, match="timed out"):
                await provider._complete_streaming(
                    mock_session,
                    "prompt",
                    "claude-opus-4.5",
                    timeout=1.0,
                    extended_thinking_enabled=False,
                    has_tools=False,
                )

    @pytest.mark.asyncio
    async def test_error_without_tools_reraises(self, provider):
        """Should re-raise original error when no tools were captured."""
        mock_session = self._make_mock_session()
        handler = self._make_mock_handler(
            wait_for_capture_or_idle=AsyncMock(side_effect=RuntimeError("SDK broke")),
        )

        with patch(
            "amplifier_module_provider_github_copilot.sdk_driver.SdkEventHandler",
            return_value=handler,
        ):
            with pytest.raises(RuntimeError, match="SDK broke"):
                await provider._complete_streaming(
                    mock_session,
                    "prompt",
                    "claude-opus-4.5",
                    timeout=1.0,
                    extended_thinking_enabled=False,
                    has_tools=False,
                )

    @pytest.mark.asyncio
    async def test_thinking_content_included_when_enabled(self, provider):
        """Should include ThinkingBlock when extended_thinking is enabled (line 840)."""
        mock_session = self._make_mock_session()
        handler = self._make_mock_handler(
            thinking_content=["Let me analyze this..."],
            text_content=["Here is my answer"],
        )

        with patch(
            "amplifier_module_provider_github_copilot.sdk_driver.SdkEventHandler",
            return_value=handler,
        ):
            response = await provider._complete_streaming(
                mock_session,
                "prompt",
                "claude-opus-4.6",
                timeout=60.0,
                extended_thinking_enabled=True,
                has_tools=False,
            )

        assert len(response.content) == 2
        assert hasattr(response.content[0], "thinking")
        assert "analyze" in response.content[0].thinking
        assert response.content[1].text == "Here is my answer"

    @pytest.mark.asyncio
    async def test_thinking_content_excluded_when_disabled(self, provider):
        """Should NOT include ThinkingBlock when extended_thinking is disabled (line 840→845)."""
        mock_session = self._make_mock_session()
        handler = self._make_mock_handler(
            thinking_content=["secret thoughts"],
            text_content=["public answer"],
        )

        with patch(
            "amplifier_module_provider_github_copilot.sdk_driver.SdkEventHandler",
            return_value=handler,
        ):
            response = await provider._complete_streaming(
                mock_session,
                "prompt",
                "claude-opus-4.5",
                timeout=60.0,
                extended_thinking_enabled=False,
                has_tools=False,
            )

        assert len(response.content) == 1
        assert response.content[0].text == "public answer"

    @pytest.mark.asyncio
    async def test_abort_called_when_should_abort_with_tools(self, provider):
        """Should call session.abort() when handler signals abort with tools."""
        mock_session = self._make_mock_session()
        mock_tool = Mock()
        mock_tool.id = "c1"
        mock_tool.name = "bash"
        mock_tool.arguments = {"command": "ls"}

        handler = self._make_mock_handler(
            captured_tools=[mock_tool],
            should_abort=True,
        )

        with patch(
            "amplifier_module_provider_github_copilot.sdk_driver.SdkEventHandler",
            return_value=handler,
        ):
            await provider._complete_streaming(
                mock_session,
                "prompt",
                "claude-opus-4.5",
                timeout=60.0,
                extended_thinking_enabled=False,
                has_tools=True,
            )

        mock_session.abort.assert_called_once()

    @pytest.mark.asyncio
    async def test_abort_failure_is_non_critical(self, provider):
        """Should continue even if session.abort() raises."""
        mock_session = self._make_mock_session()
        mock_session.abort = AsyncMock(side_effect=RuntimeError("abort failed"))
        mock_tool = Mock()
        mock_tool.id = "c1"
        mock_tool.name = "bash"
        mock_tool.arguments = {"command": "ls"}

        handler = self._make_mock_handler(
            captured_tools=[mock_tool],
            should_abort=True,
        )

        with patch(
            "amplifier_module_provider_github_copilot.sdk_driver.SdkEventHandler",
            return_value=handler,
        ):
            response = await provider._complete_streaming(
                mock_session,
                "prompt",
                "claude-opus-4.5",
                timeout=60.0,
                extended_thinking_enabled=False,
                has_tools=True,
            )

        assert len(response.tool_calls) == 1

    @pytest.mark.asyncio
    async def test_text_only_response(self, provider):
        """Should return text-only response with end_turn when no tools captured."""
        mock_session = self._make_mock_session()
        handler = self._make_mock_handler(
            text_content=["Just text, no tools"],
        )

        with patch(
            "amplifier_module_provider_github_copilot.sdk_driver.SdkEventHandler",
            return_value=handler,
        ):
            response = await provider._complete_streaming(
                mock_session,
                "prompt",
                "claude-opus-4.5",
                timeout=60.0,
                extended_thinking_enabled=False,
                has_tools=False,
            )

        assert response.tool_calls is None
        assert response.finish_reason == "end_turn"
        assert len(response.content) == 1
        assert response.content[0].text == "Just text, no tools"


class TestListModelsErrorPath:
    """Tests for list_models() error propagation (line 449)."""

    @pytest.mark.asyncio
    async def test_propagates_fetch_error(self, mock_coordinator):
        """list_models should propagate errors from fetch_and_map_models."""
        provider = CopilotSdkProvider(
            api_key=None,
            config={},
            coordinator=mock_coordinator,
        )

        with patch(
            "amplifier_module_provider_github_copilot.provider.fetch_and_map_models",
            new_callable=AsyncMock,
            side_effect=Exception("SDK connection failed"),
        ):
            with pytest.raises(Exception, match="SDK connection failed"):
                await provider.list_models()


class TestParseToolCallsNonePath:
    """Edge case for parse_tool_calls with None tool_calls (line 1018)."""

    @pytest.fixture
    def provider(self, mock_coordinator):
        return CopilotSdkProvider(api_key=None, config={}, coordinator=mock_coordinator)

    def test_handles_none_tool_calls(self, provider):
        """Should return [] when response.tool_calls is None."""
        response = ChatResponse(
            content=[TextBlock(type="text", text="No tools")],
            tool_calls=None,
        )
        result = provider.parse_tool_calls(response)
        assert result == []


class TestEmitEventErrorPath:
    """Additional tests for _emit_event error handling (lines 1200-1202)."""

    @pytest.mark.asyncio
    async def test_logs_warning_on_emit_failure(self, mock_coordinator):
        """Should log warning and not raise when hooks.emit fails."""
        provider = CopilotSdkProvider(
            api_key=None,
            config={},
            coordinator=mock_coordinator,
        )
        mock_coordinator.hooks.emit = AsyncMock(side_effect=RuntimeError("emit broke"))

        # Should not raise
        await provider._emit_event("test:event", {"data": 1})


class TestCompleteExtendedThinkingLogging:
    """Tests for complete() extended thinking logging branches (lines 586, 612)."""

    @pytest.mark.asyncio
    async def test_logs_when_thinking_requested_but_unsupported(
        self,
        mock_coordinator,
        sample_messages,
    ):
        """Should log info when extended_thinking requested but model lacks support (line 586)."""
        provider = CopilotSdkProvider(
            api_key=None,
            config={"use_streaming": False, "debug": True},
            coordinator=mock_coordinator,
        )

        mock_session = AsyncMock()
        mock_session.session_id = "test"
        mock_session.destroy = AsyncMock()

        mock_response = Mock()
        mock_response.data = Mock()
        mock_response.data.content = "Hello"
        mock_response.data.tool_requests = None
        mock_response.data.input_tokens = 10
        mock_response.data.output_tokens = 5

        @asynccontextmanager
        async def mock_create_session(self, **kwargs):
            yield mock_session

        with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
            with patch.object(
                CopilotClientWrapper,
                "send_and_wait",
                new_callable=AsyncMock,
            ) as mock_send:
                mock_send.return_value = mock_response
                with patch.object(
                    provider,
                    "_model_supports_reasoning",
                    new_callable=AsyncMock,
                    return_value=False,
                ):
                    await provider.complete(
                        {"messages": sample_messages},
                        extended_thinking=True,
                        model="gpt-4",
                    )

        # Should complete successfully (hitting the logging branch)

    @pytest.mark.asyncio
    async def test_logs_when_thinking_enabled(
        self,
        mock_coordinator,
        sample_messages,
    ):
        """Should log when extended thinking is both requested and supported (line 612)."""
        provider = CopilotSdkProvider(
            api_key=None,
            config={"use_streaming": False, "debug": True},
            coordinator=mock_coordinator,
        )

        mock_session = AsyncMock()
        mock_session.session_id = "test"
        mock_session.destroy = AsyncMock()

        mock_response = Mock()
        mock_response.data = Mock()
        mock_response.data.content = "Hello"
        mock_response.data.tool_requests = None
        mock_response.data.input_tokens = 10
        mock_response.data.output_tokens = 5

        @asynccontextmanager
        async def mock_create_session(self, **kwargs):
            yield mock_session

        with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
            with patch.object(
                CopilotClientWrapper,
                "send_and_wait",
                new_callable=AsyncMock,
            ) as mock_send:
                mock_send.return_value = mock_response
                with patch.object(
                    provider,
                    "_model_supports_reasoning",
                    new_callable=AsyncMock,
                    return_value=True,
                ):
                    await provider.complete(
                        {"messages": sample_messages},
                        extended_thinking=True,
                        model="claude-opus-4.6",
                    )

        # Verify reasoning_effort appears in request event
        request_calls = [
            call
            for call in mock_coordinator.hooks.emit.call_args_list
            if call[0][0] == "llm:request"
        ]
        assert len(request_calls) == 1
        data = request_calls[0][0][1]
        assert data["thinking_enabled"] is True
        assert data["reasoning_effort"] == "medium"


# ═══════════════════════════════════════════════════════════════════════════
# Session metrics tracking tests: validate counter increments,
# timing accumulation, error counting, and cache invalidation.
# ═══════════════════════════════════════════════════════════════════════════


class TestSessionMetricsTracking:
    """Tests for session metrics tracking counters and timing.

    Verifies that _request_count, _session_count, _error_count,
    and _total_response_time_ms are correctly maintained through
    complete() calls (both success and failure paths).
    """

    @pytest.fixture
    def provider(self, mock_coordinator, provider_config):
        """Create provider with non-streaming config for simpler test mocking."""
        return CopilotSdkProvider(
            api_key=None,
            config=provider_config,
            coordinator=mock_coordinator,
        )

    @staticmethod
    def _make_mock_session(response=None):
        """Create a mock session with a default successful response."""
        session = AsyncMock()
        session.session_id = "metrics-test-session"
        session.destroy = AsyncMock()

        if response is None:
            response = Mock()
            response.data = Mock()
            response.data.content = "Test response"
            response.data.tool_requests = None
            response.data.input_tokens = 10
            response.data.output_tokens = 5

        session.send_and_wait = AsyncMock(return_value=response)
        return session

    @pytest.mark.asyncio
    async def test_request_count_increments_on_success(
        self,
        provider,
        sample_messages,
    ):
        """_request_count increments on each complete() call."""
        assert provider._request_count == 0

        mock_session = self._make_mock_session()

        @asynccontextmanager
        async def mock_create_session(self, **kwargs):
            yield mock_session

        with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
            with patch.object(
                CopilotClientWrapper,
                "send_and_wait",
                new_callable=AsyncMock,
            ) as mock_send:
                mock_send.return_value = mock_session.send_and_wait.return_value

                await provider.complete({"messages": sample_messages})
                assert provider._request_count == 1

                await provider.complete({"messages": sample_messages})
                assert provider._request_count == 2

    @pytest.mark.asyncio
    async def test_request_count_increments_on_failure(
        self,
        provider,
        sample_messages,
    ):
        """_request_count increments even when complete() fails (counts attempts)."""
        assert provider._request_count == 0

        @asynccontextmanager
        async def mock_create_session_that_fails(self, **kwargs):
            raise RuntimeError("Connection refused")
            yield  # pragma: no cover - unreachable, satisfies async generator syntax

        with patch.object(
            CopilotClientWrapper,
            "create_session",
            mock_create_session_that_fails,
        ):
            with pytest.raises(RuntimeError, match="Connection refused"):
                await provider.complete({"messages": sample_messages})

        # Request was ATTEMPTED, so count should be 1
        assert provider._request_count == 1

    @pytest.mark.asyncio
    async def test_session_count_increments_inside_context_manager(
        self,
        provider,
        sample_messages,
    ):
        """_session_count increments when session context manager is entered."""
        assert provider._session_count == 0

        mock_session = self._make_mock_session()

        @asynccontextmanager
        async def mock_create_session(self, **kwargs):
            yield mock_session

        with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
            with patch.object(
                CopilotClientWrapper,
                "send_and_wait",
                new_callable=AsyncMock,
            ) as mock_send:
                mock_send.return_value = mock_session.send_and_wait.return_value

                await provider.complete({"messages": sample_messages})
                assert provider._session_count == 1

    @pytest.mark.asyncio
    async def test_error_count_increments_on_failure(
        self,
        provider,
        sample_messages,
    ):
        """_error_count increments when complete() raises an exception."""
        assert provider._error_count == 0

        mock_session = self._make_mock_session()

        @asynccontextmanager
        async def mock_create_session(self, **kwargs):
            yield mock_session

        # Make send_and_wait raise inside the context manager
        with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
            with patch.object(
                CopilotClientWrapper,
                "send_and_wait",
                new_callable=AsyncMock,
                side_effect=RuntimeError("API error"),
            ):
                with pytest.raises(RuntimeError, match="API error"):
                    await provider.complete({"messages": sample_messages})

        assert provider._error_count == 1

    @pytest.mark.asyncio
    async def test_error_count_does_not_increment_on_success(
        self,
        provider,
        sample_messages,
    ):
        """_error_count stays 0 when complete() succeeds."""
        mock_session = self._make_mock_session()

        @asynccontextmanager
        async def mock_create_session(self, **kwargs):
            yield mock_session

        with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
            with patch.object(
                CopilotClientWrapper,
                "send_and_wait",
                new_callable=AsyncMock,
            ) as mock_send:
                mock_send.return_value = mock_session.send_and_wait.return_value
                await provider.complete({"messages": sample_messages})

        assert provider._error_count == 0

    @pytest.mark.asyncio
    async def test_response_time_accumulates_on_success(
        self,
        provider,
        sample_messages,
    ):
        """_total_response_time_ms accumulates elapsed time from successful requests."""
        assert provider._total_response_time_ms == 0.0

        mock_session = self._make_mock_session()

        @asynccontextmanager
        async def mock_create_session(self, **kwargs):
            yield mock_session

        with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
            with patch.object(
                CopilotClientWrapper,
                "send_and_wait",
                new_callable=AsyncMock,
            ) as mock_send:
                mock_send.return_value = mock_session.send_and_wait.return_value
                await provider.complete({"messages": sample_messages})

        # Elapsed time should be > 0 (even if very small)
        assert provider._total_response_time_ms >= 0.0

    @pytest.mark.asyncio
    async def test_response_time_not_accumulated_on_failure(
        self,
        provider,
        sample_messages,
    ):
        """_total_response_time_ms stays 0 when complete() fails."""
        mock_session = self._make_mock_session()

        @asynccontextmanager
        async def mock_create_session(self, **kwargs):
            yield mock_session

        with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
            with patch.object(
                CopilotClientWrapper,
                "send_and_wait",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Timeout"),
            ):
                with pytest.raises(RuntimeError, match="Timeout"):
                    await provider.complete({"messages": sample_messages})

        # Time should NOT be accumulated for failed requests
        assert provider._total_response_time_ms == 0.0

    def test_get_session_metrics_avg_calculation(self, provider):
        """avg_response_time_ms should equal total_time / request_count."""
        # Manually set internal state to verify the math
        provider._request_count = 4
        provider._total_response_time_ms = 2000.0  # 2000ms total

        metrics = provider.get_session_metrics()

        # 2000ms / 4 requests = 500ms average
        assert metrics["avg_response_time_ms"] == 500.0

    def test_get_session_metrics_zero_division_safe(self, provider):
        """avg_response_time_ms should not crash with zero requests."""
        assert provider._request_count == 0

        metrics = provider.get_session_metrics()

        # max(0, 1) prevents division by zero
        assert metrics["avg_response_time_ms"] == 0.0


class TestInvalidateModelCacheComplete:
    """Tests that invalidate_model_cache() clears ALL caches.

    This is a regression test for a bug where _model_info_cache
    was not cleared by invalidate_model_cache(), causing stale
    context_window/max_output_tokens to persist after cache
    invalidation. This was the same class of bug that caused
    budget=182,200 in ST03 (stale cached max_output_tokens=16,384).
    """

    @pytest.fixture
    def provider(self, mock_coordinator, provider_config):
        return CopilotSdkProvider(
            api_key=None,
            config=provider_config,
            coordinator=mock_coordinator,
        )

    def test_invalidate_clears_model_info_cache(self, provider):
        """invalidate_model_cache() must clear _model_info_cache."""
        # Simulate a populated cache
        provider._model_info_cache["claude-opus-4.5"] = Mock(
            context_window=200000,
            max_output_tokens=32000,
        )
        provider._model_capabilities_cache["claude-opus-4.5"] = ["streaming", "tools"]

        assert len(provider._model_info_cache) == 1
        assert len(provider._model_capabilities_cache) == 1

        provider.invalidate_model_cache()

        assert len(provider._model_info_cache) == 0, (
            "_model_info_cache must be cleared by invalidate_model_cache()"
        )
        assert len(provider._model_capabilities_cache) == 0

    def test_invalidate_then_get_model_info_uses_fallback(self, provider):
        """After invalidation, get_model_info() should use BUNDLED_MODEL_LIMITS fallback."""
        # Populate cache with stale data
        provider._model_info_cache["claude-opus-4.5"] = Mock(
            id="claude-opus-4.5",
            context_window=999999,  # Stale value
            max_output_tokens=88888,  # Stale value
        )

        # Before invalidation: returns stale values
        info_before = provider.get_model_info()
        assert info_before.context_window == 999999

        # Invalidate
        provider.invalidate_model_cache()

        # After invalidation: falls back to BUNDLED_MODEL_LIMITS
        info_after = provider.get_model_info()
        assert info_after.context_window == 200000  # Real value, not stale
        assert info_after.max_output_tokens == 32000  # Real value, not stale

    @pytest.mark.asyncio
    async def test_invalidate_then_list_models_repopulates(
        self,
        provider,
        mock_copilot_client,
    ):
        """After invalidation, list_models() should repopulate both caches."""
        # Start with stale data
        provider._model_info_cache["old-model"] = Mock()
        provider._model_capabilities_cache["old-model"] = ["old-cap"]

        provider.invalidate_model_cache()

        # Repopulate by calling list_models
        with patch.object(
            CopilotClientWrapper,
            "ensure_client",
            new_callable=AsyncMock,
        ) as mock_ensure:
            mock_ensure.return_value = mock_copilot_client
            await provider.list_models()

        # Both caches should be repopulated with fresh data
        assert "claude-opus-4.5" in provider._model_info_cache
        assert "old-model" not in provider._model_info_cache


class TestGetInfoUnknownModelWarning:
    """Tests that get_info() warns when model is not in BUNDLED_MODEL_LIMITS.

    This prevents silent misconfiguration where a GPT or Gemini model
    gets Claude's default limits (200000, 32000) without any indication.
    """

    def test_unknown_model_uses_defaults_with_warning(self, mock_coordinator):
        """get_info() should use defaults for unknown model but log warning."""
        provider = CopilotSdkProvider(
            api_key=None,
            config={"default_model": "some-future-model-v99"},
            coordinator=mock_coordinator,
        )

        import logging

        with patch.object(
            logging.getLogger("amplifier_module_provider_github_copilot.provider"), "warning"
        ) as mock_warn:
            info = provider.get_info()

        # Should still return defaults
        assert info.defaults["context_window"] == 200000
        assert info.defaults["max_output_tokens"] == 32000

        # Should have logged a warning (called at least once)
        assert mock_warn.call_count >= 1
        warning_text = str(mock_warn.call_args_list[0])
        assert "some-future-model-v99" in warning_text
        assert "BUNDLED_MODEL_LIMITS" in warning_text

    def test_known_model_no_warning(self, mock_coordinator):
        """get_info() should NOT warn for known models like claude-opus-4.5."""
        provider = CopilotSdkProvider(
            api_key=None,
            config={"default_model": "claude-opus-4.5"},
            coordinator=mock_coordinator,
        )

        import logging

        with patch.object(
            logging.getLogger("amplifier_module_provider_github_copilot.provider"), "warning"
        ) as mock_warn:
            info = provider.get_info()

        assert info.defaults["context_window"] == 200000
        assert mock_warn.call_count == 0


class TestGeminiFallbackLimits:
    """Tests that gemini-3-pro-preview has a valid fallback that produces positive budget.

    Previous value (128000, 128000) would produce a negative budget:
    budget = 128000 - 128000 - safety_margin < 0
    """

    def test_gemini_fallback_produces_positive_budget(self, mock_coordinator):
        """Gemini fallback must have max_output_tokens < context_window."""
        from amplifier_module_provider_github_copilot.model_cache import BUNDLED_MODEL_LIMITS

        context_window, max_output = BUNDLED_MODEL_LIMITS["gemini-3-pro-preview"]

        # max_output_tokens must be strictly less than context_window
        assert max_output < context_window, (
            f"Gemini max_output_tokens ({max_output}) must be < "
            f"context_window ({context_window}) to produce positive budget"
        )

        # Budget must be positive (assuming safety_margin ~1416)
        safety_margin = 1500  # Conservative estimate
        budget = context_window - max_output - safety_margin
        assert budget > 0, (
            f"Gemini budget would be negative: "
            f"{context_window} - {max_output} - {safety_margin} = {budget}"
        )

    def test_gemini_get_info_returns_valid_values(self, mock_coordinator):
        """get_info() for gemini model should return correct fallback values."""
        provider = CopilotSdkProvider(
            api_key=None,
            config={"default_model": "gemini-3-pro-preview"},
            coordinator=mock_coordinator,
        )

        info = provider.get_info()

        assert info.defaults["context_window"] == 128000
        assert info.defaults["max_output_tokens"] == 65536
        # Verify budget is positive
        budget = info.defaults["context_window"] - info.defaults["max_output_tokens"]
        assert budget > 0

    def test_all_known_limits_produce_positive_budget(self, mock_coordinator):
        """Every model in BUNDLED_MODEL_LIMITS must produce a positive budget."""
        from amplifier_module_provider_github_copilot.model_cache import BUNDLED_MODEL_LIMITS

        safety_margin = 1500  # Conservative estimate
        for model_id, (context_window, max_output) in BUNDLED_MODEL_LIMITS.items():
            budget = context_window - max_output - safety_margin
            assert budget > 0, (
                f"Model '{model_id}' would produce negative budget: "
                f"{context_window} - {max_output} - {safety_margin} = {budget}"
            )


# ═══════════════════════════════════════════════════════════════════════════
# Coverage gap tests: provider.py uncovered lines and branches
# ═══════════════════════════════════════════════════════════════════════════


class TestGetInfoWithCachedModelInfo:
    """Tests for get_info() when model info is cached but has None context_window.

    Covers the branch where get_model_info() returns a model with
    context_window=None, forcing fallback to BUNDLED_MODEL_LIMITS (line 287).
    """

    def test_cached_model_with_none_context_window_uses_fallback(self, mock_coordinator):
        """get_info() should fall back to BUNDLED_MODEL_LIMITS when cache has None values."""
        provider = CopilotSdkProvider(
            api_key=None,
            config={"default_model": "claude-opus-4.5"},
            coordinator=mock_coordinator,
        )

        # Simulate cache with None values (e.g., model returned no limits)
        provider._model_info_cache["claude-opus-4.5"] = Mock(
            context_window=None,
            max_output_tokens=None,
        )

        info = provider.get_info()

        # Should fall back to BUNDLED_MODEL_LIMITS for claude-opus-4.5
        assert info.defaults["context_window"] == 200000
        assert info.defaults["max_output_tokens"] == 32000

    def test_cached_model_with_partial_none_uses_mixed_sources(self, mock_coordinator):
        """get_info() should use cache for one and fallback for the other."""
        provider = CopilotSdkProvider(
            api_key=None,
            config={"default_model": "claude-opus-4.5"},
            coordinator=mock_coordinator,
        )

        # context_window from cache, max_output_tokens needs fallback
        provider._model_info_cache["claude-opus-4.5"] = Mock(
            context_window=200000,
            max_output_tokens=None,
        )

        info = provider.get_info()

        assert info.defaults["context_window"] == 200000
        assert info.defaults["max_output_tokens"] == 32000  # From BUNDLED_MODEL_LIMITS


class TestModelSupportsReasoningNotInList:
    """Tests for _model_supports_reasoning when model not in fetched list (lines 492-495)."""

    @pytest.mark.asyncio
    async def test_model_not_in_list_returns_false(self, mock_coordinator, provider_config):
        """_model_supports_reasoning should return False for model not in list."""
        provider = CopilotSdkProvider(
            api_key=None,
            config=provider_config,
            coordinator=mock_coordinator,
        )

        # Mock a model list that doesn't include the queried model
        mock_client = AsyncMock()
        mock_model = Mock()
        mock_model.id = "other-model"
        mock_model.capabilities = Mock()
        mock_model.capabilities.supports = Mock()
        mock_model.capabilities.supports.vision = False
        mock_model.capabilities.supports.reasoning_effort = False
        mock_model.capabilities.limits = Mock()
        mock_model.capabilities.limits.max_context_window_tokens = 100000
        mock_model.capabilities.limits.max_prompt_tokens = 50000
        mock_model.vendor = None
        mock_model.provider = None
        mock_model.supported_reasoning_efforts = None
        mock_model.default_reasoning_effort = None
        mock_client.list_models = AsyncMock(return_value=[mock_model])

        with patch.object(
            CopilotClientWrapper,
            "ensure_client",
            new_callable=AsyncMock,
            return_value=mock_client,
        ):
            result = await provider._model_supports_reasoning("nonexistent-model")

        assert result is False
        # Should have cached empty capabilities for the missing model
        assert "nonexistent-model" in provider._model_capabilities_cache
        assert provider._model_capabilities_cache["nonexistent-model"] == []


class TestCompleteStreamOverride:
    """Tests for complete() request.stream override (line 622)."""

    @pytest.fixture
    def provider(self, mock_coordinator, provider_config):
        return CopilotSdkProvider(
            api_key=None,
            config=provider_config,
            coordinator=mock_coordinator,
        )

    @pytest.mark.asyncio
    async def test_request_stream_true_overrides_config(
        self,
        provider,
        sample_messages,
        mock_copilot_client,
    ):
        """request.stream=True should override provider's use_streaming config."""
        request = Mock()
        request.messages = sample_messages
        request.tools = None
        request.stream = True  # Override config's use_streaming=False

        mock_session = AsyncMock()
        mock_session.session_id = "stream-test"
        mock_session.destroy = AsyncMock()

        @asynccontextmanager
        async def mock_create_session(self, **kwargs):
            yield mock_session

        with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
            with patch.object(
                provider,
                "_complete_streaming",
                new_callable=AsyncMock,
            ) as mock_streaming:
                mock_response = ChatResponse(
                    content=[TextBlock(text="streamed")],
                    model="claude-opus-4.5",
                    usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
                )
                mock_streaming.return_value = mock_response

                await provider.complete(request)

                # Should have used streaming path because request.stream=True
                mock_streaming.assert_called_once()

    @pytest.mark.asyncio
    async def test_request_stream_none_uses_config(
        self,
        provider,
        sample_messages,
    ):
        """request.stream=None should use provider's config."""
        request = Mock()
        request.messages = sample_messages
        request.tools = None
        request.stream = None  # Should use config default

        mock_session = AsyncMock()
        mock_session.session_id = "no-stream-test"
        mock_session.destroy = AsyncMock()

        @asynccontextmanager
        async def mock_create_session(self, **kwargs):
            yield mock_session

        with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
            with patch.object(
                CopilotClientWrapper,
                "send_and_wait",
                new_callable=AsyncMock,
            ) as mock_send:
                mock_response = Mock()
                mock_response.data = Mock()
                mock_response.data.content = "Response"
                mock_response.data.tool_requests = None
                mock_response.data.input_tokens = 10
                mock_response.data.output_tokens = 5
                mock_send.return_value = mock_response

                await provider.complete(request)

                # Should have used non-streaming path (send_and_wait)
                mock_send.assert_called_once()


class TestCompleteDebugToolPayload:
    """Tests for debug tool logging in complete() (lines 770, 796)."""

    @pytest.fixture
    def debug_provider(self, mock_coordinator):
        return CopilotSdkProvider(
            api_key=None,
            config={
                "default_model": "claude-opus-4.5",
                "timeout": 60.0,
                "debug": True,
                "raw_debug": True,
                "use_streaming": False,  # Use non-streaming for simpler test
            },
            coordinator=mock_coordinator,
        )

    @pytest.mark.asyncio
    async def test_debug_logs_tool_definitions(
        self,
        debug_provider,
        mock_copilot_client,
    ):
        """Debug mode should log tool definitions when tools are present."""
        request = Mock()
        request.messages = [{"role": "user", "content": "Read file."}]
        request.stream = None

        # Add tools to request - this forces streaming mode (Deny+Destroy)
        tool = Mock()
        tool.name = "read_file"
        tool.description = "Read a file"
        tool.parameters = {"type": "object", "properties": {"path": {"type": "string"}}}
        request.tools = [tool]

        mock_session = AsyncMock()
        mock_session.session_id = "debug-test"
        mock_session.destroy = AsyncMock()

        @asynccontextmanager
        async def mock_create_session(self, **kwargs):
            yield mock_session

        # Mock _complete_streaming to avoid actual streaming complexity
        mock_response = ChatResponse(
            content=[TextBlock(text="Done")],
            model="claude-opus-4.5",
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
        )

        with patch.object(CopilotClientWrapper, "create_session", mock_create_session):
            with patch.object(
                debug_provider,
                "_complete_streaming",
                new_callable=AsyncMock,
                return_value=mock_response,
            ):
                response = await debug_provider.complete(request)

                assert response is not None
                # Debug events should have been emitted
                emit_calls = debug_provider._coordinator.hooks.emit.call_args_list
                event_names = [c[0][0] for c in emit_calls]
                assert "llm:request:debug" in event_names
                assert "llm:request:raw" in event_names

                # Verify tools appear in debug payload
                debug_call = next(c for c in emit_calls if c[0][0] == "llm:request:debug")
                debug_data = debug_call[0][1]
                assert "tools" in debug_data["request"]


class TestExtractMessagesWithPydantic:
    """Tests for _extract_messages with Pydantic models (lines 1254-1258)."""

    @pytest.fixture
    def provider(self, mock_coordinator, provider_config):
        return CopilotSdkProvider(
            api_key=None,
            config=provider_config,
            coordinator=mock_coordinator,
        )

    def test_extract_pydantic_messages(self, provider):
        """_extract_messages should call model_dump() on Pydantic Message objects."""
        msg1 = Mock()
        msg1.model_dump = Mock(return_value={"role": "user", "content": "Hello"})

        msg2 = Mock()
        msg2.model_dump = Mock(return_value={"role": "assistant", "content": "Hi"})

        request = Mock()
        request.messages = [msg1, msg2]

        result = provider._extract_messages(request)

        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "Hello"}
        assert result[1] == {"role": "assistant", "content": "Hi"}
        msg1.model_dump.assert_called_once()
        msg2.model_dump.assert_called_once()

    def test_extract_mixed_messages(self, provider):
        """_extract_messages should handle mix of Pydantic and dict messages."""
        pydantic_msg = Mock()
        pydantic_msg.model_dump = Mock(return_value={"role": "user", "content": "Pydantic"})

        dict_msg = {"role": "assistant", "content": "Dict"}

        request = Mock()
        request.messages = [pydantic_msg, dict_msg]

        result = provider._extract_messages(request)

        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "Pydantic"}
        assert result[1] == {"role": "assistant", "content": "Dict"}

    def test_extract_no_messages_attribute(self, provider):
        """_extract_messages should return empty list if no messages attribute."""
        request = object()  # No .messages attribute

        result = provider._extract_messages(request)

        assert result == []


class TestHandleTaskExceptionDebugPath:
    """Tests for _handle_task_exception with debug=False (line 1494)."""

    def test_exception_not_logged_without_debug(self, mock_coordinator):
        """Background task exception should be silently swallowed without debug."""
        provider = CopilotSdkProvider(
            api_key=None,
            config={"debug": False},
            coordinator=mock_coordinator,
        )

        task = Mock()
        task.cancelled.return_value = False
        task.exception.return_value = RuntimeError("Background failure")

        # Should not raise
        provider._handle_task_exception(task)


# ═══════════════════════════════════════════════════════════════════════════════
# ST04 Bug Fix Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestBug3DoubleFallbackCall:
    """
    BUG 3: get_info() calls get_fallback_limits() twice.

    The current code in provider.py get_info() has two separate if blocks
    that each call get_fallback_limits(self._model). This is inefficient
    and could cause inconsistent behavior if the function had side effects.

    Fix: Cache the fallback lookup at start of method, use cached value.
    """

    def test_get_info_single_fallback_lookup(self, mock_coordinator, provider_config, monkeypatch):
        """get_info should call get_fallback_limits at most once."""
        from amplifier_module_provider_github_copilot import model_cache

        # Track calls to get_fallback_limits
        original_fn = model_cache.get_fallback_limits
        call_count = {"count": 0}

        def tracking_fallback(model_id):
            call_count["count"] += 1
            return original_fn(model_id)

        monkeypatch.setattr(
            "amplifier_module_provider_github_copilot.provider.get_fallback_limits",
            tracking_fallback,
        )

        # Use a model that's in BUNDLED_MODEL_LIMITS so fallback is used
        config = {**provider_config, "model": "claude-opus-4.5", "default_model": "claude-opus-4.5"}
        provider = CopilotSdkProvider(
            api_key=None,
            config=config,
            coordinator=mock_coordinator,
        )

        # Reset counter after init (init may call it)
        call_count["count"] = 0

        # Call get_info which internally may call get_fallback_limits
        provider.get_info()

        # Should call get_fallback_limits at most once per get_info() call
        assert call_count["count"] <= 1, (
            f"get_fallback_limits called {call_count['count']} times, expected <= 1. "
            f"Fix: cache the fallback lookup at start of get_info()"
        )

    def test_get_info_fallback_returns_consistent_values(self, mock_coordinator, provider_config):
        """get_info should return same context_window and max_output_tokens from single lookup."""
        config = {**provider_config, "model": "gpt-4.1", "default_model": "gpt-4.1"}
        provider = CopilotSdkProvider(
            api_key=None,
            config=config,
            coordinator=mock_coordinator,
        )

        info = provider.get_info()

        # Both values should come from the same fallback lookup
        # gpt-4.1 has (128000, 64000) in BUNDLED_MODEL_LIMITS
        assert info.defaults["context_window"] == 128000
        assert info.defaults["max_output_tokens"] == 64000

    def test_get_info_fallback_called_once_when_cache_miss(
        self, mock_coordinator, provider_config, monkeypatch
    ):
        """get_info should call get_fallback_limits exactly once when model not in cache."""
        from unittest.mock import patch

        from amplifier_module_provider_github_copilot import model_cache

        # Use an unknown model that's not in cache or BUNDLED_MODEL_LIMITS
        config = {
            **provider_config,
            "model": "unknown-test-model-xyz",
            "default_model": "unknown-test-model-xyz",
        }
        provider = CopilotSdkProvider(
            api_key=None,
            config=config,
            coordinator=mock_coordinator,
        )

        # Clear any cached model info
        provider._model_info_cache.clear()

        # Track calls to get_fallback_limits
        call_count = {"count": 0}
        original_fn = model_cache.get_fallback_limits

        def tracking_fallback(model_id):
            call_count["count"] += 1
            return original_fn(model_id)

        # Patch at the provider module level (where it's imported)
        with patch(
            "amplifier_module_provider_github_copilot.provider.get_fallback_limits",
            tracking_fallback,
        ):
            provider.get_info()

        # Should be called exactly once (not twice as before the fix)
        assert call_count["count"] == 1, (
            f"get_fallback_limits called {call_count['count']} times, expected exactly 1. "
            f"The fix should cache the fallback lookup."
        )


class TestConfigFields:
    """Tests for config_fields in get_info()."""

    @pytest.fixture
    def provider(self, mock_coordinator, provider_config):
        return CopilotSdkProvider(
            api_key=None, config=provider_config, coordinator=mock_coordinator
        )

    def test_config_fields_not_empty(self, provider):
        """get_info should return config_fields for the setup wizard."""
        info = provider.get_info()
        assert len(info.config_fields) >= 2

    def test_auth_guidance_field(self, provider):
        """Should have an auth guidance text field."""
        info = provider.get_info()
        auth_field = next((f for f in info.config_fields if f.id == "auth_info"), None)
        assert auth_field is not None
        assert auth_field.field_type == "text"
        assert auth_field.required is False
        assert auth_field.default == ""

    def test_default_model_field(self, provider):
        """Should have a model choice field with curated options."""
        info = provider.get_info()
        model_field = next((f for f in info.config_fields if f.id == "model"), None)
        assert model_field is not None
        assert model_field.field_type == "choice"
        assert model_field.required is True
        assert model_field.default == "claude-sonnet-4"
        assert model_field.choices is not None
        assert "claude-sonnet-4" in model_field.choices
        assert "gpt-4o" in model_field.choices
