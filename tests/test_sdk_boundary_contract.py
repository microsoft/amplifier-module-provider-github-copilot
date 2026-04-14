"""Boundary contract tests for SDK session configuration.

These tests verify the exact configuration dict that CopilotClientWrapper.session()
sends to client.create_session(). They use ConfigCapturingMock instead of MagicMock
to ensure we test what is SENT, not just that something was called.

Contract: contracts/sdk-boundary.md
"""

from __future__ import annotations

from typing import Any

import pytest

from amplifier_module_provider_github_copilot.sdk_adapter.client import (
    CopilotClientWrapper,
)
from tests.fixtures.config_capture import ConfigCapturingMock


class TestSessionConfigContract:
    """Verify the session_config dict sent to SDK matches our contract."""

    @pytest.mark.asyncio
    async def test_available_tools_empty_when_no_tools_provided(self) -> None:
        """available_tools MUST be set to [] when no Amplifier tools are provided.

        This prevents SDK built-in tools (list_agents, bash, edit, etc.) from
        appearing to the model. Setting available_tools=[] blocks all tools,
        which is correct when no Amplifier tools exist.

        Contract: deny-destroy:ToolSuppression:MUST:1 — MUST NOT omit available_tools
        Contract: sdk-boundary:ToolForwarding:MUST:3 — available_tools always set
        """
        mock_client = ConfigCapturingMock()
        wrapper = CopilotClientWrapper(sdk_client=mock_client)

        async with wrapper.session(model="gpt-4o"):
            pass

        config = mock_client.last_config
        # Contract v1.2: available_tools MUST be set (not omitted)
        # When no tools provided, available_tools=[] prevents SDK built-ins from appearing
        assert "available_tools" in config, "available_tools MUST be set"
        assert config["available_tools"] == [], "available_tools MUST be [] when no tools"

    @pytest.mark.asyncio
    async def test_available_tools_allowlist_when_tools_provided(self) -> None:
        """available_tools MUST be set to Amplifier tool names when tools are provided.

        This creates an allowlist - only the provided tools are visible to the model.
        SDK built-ins are blocked because they're not in the allowlist.

        Contract: deny-destroy:Allowlist:MUST:1 — available_tools = tool names
        Contract: deny-destroy:Allowlist:MUST:2 — allowlist blocks SDK built-ins
        Contract: sdk-boundary:ToolForwarding:MUST:1 — tools forwarded to session
        """
        mock_client = ConfigCapturingMock()
        wrapper = CopilotClientWrapper(sdk_client=mock_client)

        # Provide some tools as dicts (simulating Amplifier's ToolSpec objects)
        empty_params: dict[str, Any] = {}
        tools: list[dict[str, Any]] = [
            {"name": "search", "description": "Search the web", "parameters": empty_params},
            {"name": "write_file", "description": "Write a file", "parameters": empty_params},
        ]

        async with wrapper.session(model="gpt-4o", tools=tools):
            pass

        config = mock_client.last_config
        # available_tools should be set to the allowlist of tool names
        assert "available_tools" in config, "available_tools MUST be set"
        assert config["available_tools"] == ["search", "write_file"], (
            "available_tools must be the list of tool names"
        )

    @pytest.mark.asyncio
    async def test_system_message_uses_replace_mode(self) -> None:
        """System message MUST use replace mode.

        SDK ref: copilot/types.py SystemMessageConfig
        SDK ref: copilot/client.py line 522-524 (system_message handling)
        """
        # Contract: sdk-boundary:Config:MUST:2
        mock_client = ConfigCapturingMock()
        wrapper = CopilotClientWrapper(sdk_client=mock_client)

        async with wrapper.session(
            model="gpt-4o", system_message="You are the Amplifier assistant."
        ):
            pass

        config = mock_client.last_config
        assert config["system_message"]["mode"] == "replace"
        assert config["system_message"]["content"] == "You are the Amplifier assistant."

    @pytest.mark.asyncio
    async def test_system_message_absent_when_not_provided(self) -> None:
        """No system_message key when caller doesn't provide one."""
        # Contract: sdk-boundary:Config:MUST:2
        mock_client = ConfigCapturingMock()
        wrapper = CopilotClientWrapper(sdk_client=mock_client)

        async with wrapper.session(model="gpt-4o"):
            pass

        config = mock_client.last_config
        assert "system_message" not in config

    @pytest.mark.asyncio
    async def test_permission_handler_always_set(self) -> None:
        """Permission handler MUST be set on every session.

        Contract: deny-destroy:DenyHook:MUST:1
        """
        mock_client = ConfigCapturingMock()
        wrapper = CopilotClientWrapper(sdk_client=mock_client)

        async with wrapper.session(model="gpt-4o"):
            pass

        config = mock_client.last_config
        assert "on_permission_request" in config
        assert callable(config["on_permission_request"])

    @pytest.mark.asyncio
    async def test_streaming_always_enabled(self) -> None:
        """Streaming MUST be enabled for event-based tool capture."""
        # Contract: sdk-boundary:Config:MUST:4
        mock_client = ConfigCapturingMock()
        wrapper = CopilotClientWrapper(sdk_client=mock_client)

        async with wrapper.session(model="gpt-4o"):
            pass

        config = mock_client.last_config
        assert config["streaming"] is True

    @pytest.mark.asyncio
    async def test_model_passed_through(self) -> None:
        """Model parameter forwarded to SDK session config."""
        # Contract: sdk-boundary:Session:MUST:1
        mock_client = ConfigCapturingMock()
        wrapper = CopilotClientWrapper(sdk_client=mock_client)

        async with wrapper.session(model="claude-sonnet-4"):
            pass

        config = mock_client.last_config
        assert config["model"] == "claude-sonnet-4"

    @pytest.mark.asyncio
    async def test_deny_hook_registered_on_session(self) -> None:
        """Deny hook MUST be passed via session config 'hooks' key.

        Contract: deny-destroy:DenyHook:MUST:1
        Hooks are passed via session config, not method calls.
        """
        mock_client = ConfigCapturingMock()

        wrapper = CopilotClientWrapper(sdk_client=mock_client)

        async with wrapper.session(model="gpt-4o"):
            pass

        # Verify deny hook was passed via session config
        config = mock_client.last_config
        assert "hooks" in config, "session config must include 'hooks' key"
        assert "on_pre_tool_use" in config["hooks"], "hooks must include 'on_pre_tool_use'"


class TestToolForwardingContract:
    """Verify tools from ChatRequest are forwarded to SDK session.

    Contract: sdk-boundary.md § Tool Forwarding Contract
    """

    @pytest.mark.asyncio
    async def test_tools_forwarded_to_session_config(self) -> None:
        """Tools MUST be forwarded to session_config["tools"].

        Contract: sdk-boundary:ToolForwarding:MUST:1
        Layer: CopilotClientWrapper.session() → SDK create_session()
        """
        mock_client = ConfigCapturingMock()
        wrapper = CopilotClientWrapper(sdk_client=mock_client)

        # Amplifier tool format (simplified)
        tools: list[dict[str, object]] = [
            {"name": "bash", "description": "Run shell commands", "parameters": {}},
            {"name": "read_file", "description": "Read a file", "parameters": {}},
        ]

        async with wrapper.session(model="gpt-4o", tools=tools):
            pass

        config = mock_client.last_config
        assert "tools" in config, "tools MUST be in session_config"
        # Verify tools are forwarded (count matches)
        assert len(config["tools"]) == len(tools), "All tools MUST be forwarded"

    @pytest.mark.asyncio
    async def test_tools_converted_to_sdk_format(self) -> None:
        """Tools MUST be converted to objects with SDK-required attributes.

        Contract: sdk-boundary:ToolForwarding:MUST:2

        SDK iterates tools and accesses these attributes:
        - tool.name
        - tool.description
        - tool.parameters
        - tool.overrides_built_in_tool
        - tool.skip_permission

        Without these attributes, SDK raises AttributeError.
        """
        mock_client = ConfigCapturingMock()
        wrapper = CopilotClientWrapper(sdk_client=mock_client)

        # Amplifier tool format (what kernel sends)
        tools: list[dict[str, object]] = [
            {"name": "bash", "description": "Run shell commands", "parameters": {"type": "object"}},
        ]

        async with wrapper.session(model="gpt-4o", tools=tools):
            pass

        config = mock_client.last_config
        assert "tools" in config, "tools MUST be in session_config"

        # Verify each tool has SDK-required attributes
        for tool in config["tools"]:
            # Verify values are correct
            assert tool.name == "bash"
            assert tool.description == "Run shell commands"
            assert tool.parameters == {"type": "object"}
            # Contract: deny-destroy:ToolSuppression:MUST:2 (see sdk-boundary:ToolForwarding:MUST:2)
            assert tool.overrides_built_in_tool is True, (
                "overrides_built_in_tool MUST be True to avoid SDK 'conflicts with built-in' error"
            )
            assert tool.skip_permission is False
            assert tool.handler is None, "handler MUST be None (Amplifier handles tools)"

    @pytest.mark.asyncio
    async def test_toolspec_objects_converted_correctly(self) -> None:
        """ToolSpec objects (Pydantic BaseModel) MUST be handled correctly.

        Contract: sdk-boundary:ToolForwarding:MUST:2
        Evidence: amplifier_core.message_models.ToolSpec

        Amplifier kernel sends ToolSpec objects, not dicts. The converter
        MUST handle attribute access (tool.name) not dict access (tool["name"]).

        This test uses SimpleNamespace to simulate ToolSpec without importing
        amplifier_core, keeping tests isolated from external dependencies.
        """
        from types import SimpleNamespace

        mock_client = ConfigCapturingMock()
        wrapper = CopilotClientWrapper(sdk_client=mock_client)

        # Simulate ToolSpec objects (Pydantic BaseModel with attributes)
        # ToolSpec has: name: str, parameters: dict, description: str | None
        toolspec_like = [
            SimpleNamespace(
                name="bash",
                description="Run shell commands",
                parameters={"type": "object", "properties": {}},
            ),
            SimpleNamespace(
                name="read_file",
                description="Read a file",
                parameters={"type": "object"},
            ),
        ]

        async with wrapper.session(model="gpt-4o", tools=toolspec_like):
            pass

        config = mock_client.last_config
        assert "tools" in config, "tools MUST be in session_config"
        assert len(config["tools"]) == 2, "All ToolSpec objects MUST be converted"

        # Verify SDK-required attributes are present
        tool = config["tools"][0]
        assert tool.name == "bash"
        assert tool.description == "Run shell commands"
        assert tool.parameters == {"type": "object", "properties": {}}
        assert tool.overrides_built_in_tool is True, (
            "overrides_built_in_tool MUST be True to avoid SDK 'conflicts with built-in' error"
        )
        assert tool.skip_permission is False
        assert tool.handler is None, "handler MUST be None for SDK to skip registration"

    @pytest.mark.asyncio
    async def test_execute_sdk_completion_accepts_tools_param(self) -> None:
        """_execute_sdk_completion MUST forward tools to SDK session.

        Contract: provider-protocol:complete:MUST:2
        Layer: Provider._execute_sdk_completion() → CopilotClientWrapper.session()

        This test behaviorally verifies tools are forwarded through the provider.
        """
        from types import SimpleNamespace

        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )
        from tests.fixtures.sdk_mocks import MockCopilotClientWrapper, text_delta_event

        # Contract: sdk-boundary:Session:MUST:1 (all config forwarded through boundary)
        mock_client = MockCopilotClientWrapper(events=[text_delta_event("ok")])
        provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]

        # Simulate tool passed via ChatRequest
        tool = SimpleNamespace(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object"},
        )
        request = SimpleNamespace(
            messages=[SimpleNamespace(role="user", content="test")],
            model="gpt-4o",
            tools=[tool],
            attachments=None,
            system_message=None,
        )

        await provider.complete(request)  # type: ignore[arg-type]

        # Behavioral proof: MockCopilotClientWrapper.session() recorded tools
        assert mock_client.last_tools is not None  # narrowed for pyright
        assert len(mock_client.last_tools) == 1, "Exactly one tool MUST be forwarded"

    @pytest.mark.asyncio
    async def test_tools_none_when_not_provided(self) -> None:
        """No tools key when caller doesn't provide tools."""
        # Contract: sdk-boundary:Session:MUST:1
        mock_client = ConfigCapturingMock()
        wrapper = CopilotClientWrapper(sdk_client=mock_client)

        async with wrapper.session(model="gpt-4o"):
            pass

        config = mock_client.last_config
        assert "tools" not in config, "tools key should be absent when not provided"

    @pytest.mark.asyncio
    async def test_tools_and_available_tools_not_conflated(self) -> None:
        """tools (custom) and available_tools (built-in allowlist) are separate.

        Contract: sdk-boundary:ToolForwarding:MUST:4

        available_tools IS set to Amplifier tool names to create an allowlist of
        those tools, preventing SDK built-in tools (like list_agents) from being
        visible to the model.
        """
        mock_client = ConfigCapturingMock()
        wrapper = CopilotClientWrapper(sdk_client=mock_client)

        tools: list[dict[str, object]] = [
            {"name": "bash", "description": "Run shell commands", "parameters": {}}
        ]

        async with wrapper.session(model="gpt-4o", tools=tools):
            pass

        config = mock_client.last_config
        # available_tools SHOULD be set to just the Amplifier tool names
        assert "available_tools" in config, "available_tools MUST be set to allowlist"
        assert config["available_tools"] == ["bash"], (
            "available_tools must equal Amplifier tool names"
        )
        # tools should be custom tools (converted to SDK-compatible format)
        sdk_tools = config["tools"]
        assert len(sdk_tools) == len(tools), "All custom tools MUST be forwarded"
        assert sdk_tools[0].name == "bash", "Tool name MUST be preserved"


class TestConfigInvariants:
    """Configuration invariants that must ALWAYS hold."""

    # available_tools removed - Bug #1 fix: setting it to [] was disabling all tools
    INVARIANTS: dict[str, object] = {
        "streaming": True,  # Required for event capture
        "available_tools": [],  # Contract v1.2: MUST be set (empty when no tools)
    }

    # Fields that must NOT be present (were for Bug #1 fix, now obsolete)
    # Contract v1.2 corrected: available_tools MUST be set, not absent
    ABSENT_INVARIANTS: list[str] = []

    CALLABLE_INVARIANTS = [
        "on_permission_request",  # Always set
    ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model", ["gpt-4", "gpt-4o", "claude-sonnet-4", None])
    async def test_invariants_hold_for_any_model(self, model: str | None) -> None:
        """Config invariants hold regardless of model selection."""
        # Contract: sdk-boundary:Config:MUST:1,3,4
        mock_client = ConfigCapturingMock()
        wrapper = CopilotClientWrapper(sdk_client=mock_client)

        kwargs: dict[str, object] = {}
        if model:
            kwargs["model"] = model

        async with wrapper.session(**kwargs):  # type: ignore[arg-type]
            pass

        config = mock_client.last_config
        for key, expected in self.INVARIANTS.items():
            assert config.get(key) == expected, (
                f"Invariant violated: {key} should be {expected!r}, got {config.get(key)!r}"
            )
        for key in self.ABSENT_INVARIANTS:
            assert key not in config, f"Absent invariant violated: {key} must NOT be in config"
        for key in self.CALLABLE_INVARIANTS:
            assert key in config and callable(config[key]), (
                f"Callable invariant violated: {key} must be present and callable"
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "system_message",
        [
            "You are helpful",
            "Custom persona",
            None,
        ],
    )
    async def test_invariants_hold_with_system_message_variations(
        self, system_message: str | None
    ) -> None:
        """Config invariants hold regardless of system message."""
        # Contract: sdk-boundary:Config:MUST:1,2,4
        mock_client = ConfigCapturingMock()
        wrapper = CopilotClientWrapper(sdk_client=mock_client)

        kwargs: dict[str, object] = {"model": "gpt-4o"}
        if system_message:
            kwargs["system_message"] = system_message

        async with wrapper.session(**kwargs):  # type: ignore[arg-type]
            pass

        config = mock_client.last_config
        # Contract v1.2: available_tools MUST be set (not omitted)
        # When no tools provided, available_tools=[] prevents SDK built-ins
        assert "available_tools" in config
        assert config["available_tools"] == []
        assert config.get("streaming") is True

    @pytest.mark.asyncio
    async def test_no_unexpected_keys_in_config(self) -> None:
        """Session config should only contain known SDK keys.

        Guards against typos or wrong key names that SDK silently ignores.
        SDK ref: copilot/types.py SessionConfig fields
        """
        # Contract: sdk-boundary:Config:MUST:6
        KNOWN_SDK_KEYS = {
            "session_id",
            "client_name",
            "model",
            "reasoning_effort",
            "tools",
            "system_message",
            "available_tools",
            "excluded_tools",
            "on_permission_request",
            "on_user_input_request",
            "hooks",
            "working_directory",
            "provider",
            "streaming",
            "mcp_servers",
            "custom_agents",
            "agent",
            "config_dir",
            "skill_directories",
            "disabled_skills",
            "infinite_sessions",
            "on_event",
        }

        mock_client = ConfigCapturingMock()
        wrapper = CopilotClientWrapper(sdk_client=mock_client)

        async with wrapper.session(model="gpt-4o", system_message="test"):
            pass

        config = mock_client.last_config
        unknown_keys = set(config.keys()) - KNOWN_SDK_KEYS
        assert unknown_keys == set(), (
            f"Unknown keys in session config: {unknown_keys}. "
            f"These may be typos that the SDK silently ignores."
        )


class TestRuntimeSDKTypeLeak:
    """sdk-boundary:TypeTranslation:MUST:1 — Runtime SDK type leak assertion.

    Verifies at RUNTIME (not AST) that SDK types do not leak through the membrane.
    This complements test_sdk_boundary_quarantine.py (static AST checks).
    """

    def test_session_handle_is_domain_type_not_sdk(self) -> None:
        """sdk-boundary:TypeTranslation:MUST:1 — SessionHandle is a domain class.

        The module attribute for SessionHandle must point to our sdk_adapter,
        not to the Copilot SDK package.
        """
        from amplifier_module_provider_github_copilot.sdk_adapter import SessionHandle

        # Domain type: __module__ must NOT point to the SDK (copilot.*)
        module = getattr(SessionHandle, "__module__", "")
        assert module.startswith("amplifier_module_provider_github_copilot"), (
            f"SessionHandle must be defined in amplifier_module_provider_github_copilot, "
            f"got __module__={module!r} — possible SDK type leak"
        )

    def test_all_exported_classes_are_domain_types(self) -> None:
        """Contract: sdk-boundary:Types:MUST:1 — Public class exports are domain types.

        All class objects in sdk_adapter.__all__ must be defined in our package,
        not imported directly from the Copilot SDK.
        """
        import amplifier_module_provider_github_copilot.sdk_adapter as sdk_adapter_mod

        sdk_package_prefix = "copilot"
        our_package_prefix = "amplifier_module_provider_github_copilot"

        leaked: list[str] = []
        for name in sdk_adapter_mod.__all__:
            obj = getattr(sdk_adapter_mod, name, None)
            if obj is None or not isinstance(obj, type):
                continue  # Skip non-class exports (functions, type aliases)
            mod = getattr(obj, "__module__", "") or ""
            if mod.startswith(sdk_package_prefix) and not mod.startswith(our_package_prefix):
                leaked.append(f"{name} (__module__={mod!r})")

        assert not leaked, "SDK types leaked into sdk_adapter public API at runtime:\n" + "\n".join(
            leaked
        )

    @pytest.mark.asyncio
    async def test_session_context_manager_yields_domain_type(self) -> None:
        """Contract: sdk-boundary:Types:MUST:3 — session() yields domain type, not SDK object.

        At runtime, the yielded value must be SessionHandle (domain), not the raw SDK session.
        """
        from unittest.mock import AsyncMock

        from amplifier_module_provider_github_copilot.sdk_adapter import SessionHandle

        mock_sdk_session = AsyncMock()
        mock_sdk_session.session_id = "sess-type-test"
        mock_sdk_session.disconnect = AsyncMock()

        mock_sdk_client = AsyncMock()
        mock_sdk_client.create_session = AsyncMock(return_value=mock_sdk_session)

        wrapper = CopilotClientWrapper(sdk_client=mock_sdk_client)

        async with wrapper.session(model="gpt-4") as handle:
            # Must be SessionHandle (domain type), not the raw AsyncMock
            assert isinstance(handle, SessionHandle), (
                f"session() must yield SessionHandle (domain type), "
                f"got {type(handle).__name__!r} — SDK type leak"
            )
            # The raw SDK session must NOT be directly accessible (encapsulated)
            assert handle is not mock_sdk_session, (
                "session() must not yield the raw SDK session object directly"
            )


class TestSessionEventPattern:
    """Verify provider uses session.on() + send() + unsubscribe() pattern.

    Contract: sdk-boundary:Events:MUST:1

    The provider MUST register an event handler via on() BEFORE calling send(),
    then call the returned unsubscribe function in the finally block.

    Replaces the source-scan tests deleted from test_sdk_api_conformance.py.
    Behavioral proof: MockSDKSession.send() delivers events only to registered
    handlers — if on() is skipped, no text arrives and response.text_content == "".
    """

    @pytest.mark.asyncio
    async def test_provider_registers_handler_before_send_and_unsubscribes(
        self,
    ) -> None:
        """Provider MUST call on() before send(), then unsubscribe() in finally.

        Contract: sdk-boundary:Events:MUST:1
        Contract: deny-destroy:NoExecution:MUST:1
        Contract: deny-destroy:NoExecution:MUST:2

        Behavioral proof via MockSDKSession semantics:
        - send() delivers events ONLY to handlers registered via on().
          If on() is skipped: _handlers empty → no events → text_content == "".
        - After completion, _handlers is empty → unsubscribe() was called.
        - last_prompt is str → send(prompt=str, ...) was called.
        """
        from types import SimpleNamespace

        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )
        from tests.fixtures.sdk_mocks import (
            MockCopilotClientWrapper,
            MockSDKSession,
            text_delta_event,
        )

        expected_text = "response from sdk"
        mock_client = MockCopilotClientWrapper(events=[text_delta_event(expected_text)])
        provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]

        # SimpleNamespace avoids MagicMock-without-spec; provider uses getattr() access
        request = SimpleNamespace(
            messages=[SimpleNamespace(role="user", content="test event pattern")],
            model="gpt-4o",
            tools=None,
            attachments=None,
            system_message=None,
        )

        response = await provider.complete(request)  # type: ignore[arg-type]

        session = mock_client._session  # type: ignore[attr-defined]
        assert isinstance(session, MockSDKSession), (
            "provider.complete() MUST create a session via client.session()"
        )

        # Proves send(prompt: str) was called — last_prompt set only during send()
        assert isinstance(session.last_prompt, str), (
            f"session.send() MUST be called with a str prompt. "
            f"Got {type(session.last_prompt).__name__!r}."
        )

        # Proves unsubscribe() was called in finally — handlers removed after completion
        assert len(session._handlers) == 0, (
            f"unsubscribe() MUST be called in finally block. "
            f"{len(session._handlers)} handler(s) remain — possible resource leak."
        )

        # Proves on() was called BEFORE send():
        # text_delta_event delivered → handler was registered when send() ran.
        # If on() were skipped: _handlers empty → no events → content is empty list.
        from amplifier_core import TextBlock

        assert len(response.content) == 1, (
            f"on() MUST be called before send() so handler receives text_delta events. "
            f"content list must have exactly 1 TextBlock, got {len(response.content)}."
        )
        block = response.content[0]
        assert isinstance(block, TextBlock), (
            f"Expected TextBlock from text_delta_event, got {type(block).__name__!r}."
        )
        assert block.text == expected_text, (
            f"TextBlock.text must equal delivered event text. "
            f"Expected {expected_text!r}, got {block.text!r}."
        )

    @pytest.mark.asyncio
    async def test_send_uses_string_prompt_not_dict(self) -> None:
        """session.send() MUST be called with (prompt: str), not send({"prompt": ...}).

        Contract: sdk-boundary:Send:MUST:1

        SDK v0.2.0 changed from send({"prompt": ...}) to send(prompt, attachments=...).
        This test catches reversion to the dict-based API.
        """
        from types import SimpleNamespace

        from amplifier_module_provider_github_copilot.provider import (
            GitHubCopilotProvider,
        )
        from tests.fixtures.sdk_mocks import MockCopilotClientWrapper, MockSDKSession

        mock_client = MockCopilotClientWrapper(events=[])
        provider = GitHubCopilotProvider(client=mock_client)  # type: ignore[arg-type]

        request = SimpleNamespace(
            messages=[SimpleNamespace(role="user", content="check prompt type")],
            model="gpt-4o",
            tools=None,
            attachments=None,
            system_message=None,
        )

        await provider.complete(request)  # type: ignore[arg-type]

        session = mock_client._session  # type: ignore[attr-defined]
        assert isinstance(session, MockSDKSession)

        # SDK v0.2.0: send(prompt: str, attachments=...) — prompt is a plain str.
        # Old API: send({"prompt": ...}) — a dict. last_prompt being str catches reversion.
        assert isinstance(session.last_prompt, str), (
            f"session.send() MUST receive a str prompt (sdk-boundary:Send:MUST:1). "
            f"Got {type(session.last_prompt).__name__!r}. "
            "Reversion to send(dict) would produce a non-str last_prompt."
        )
