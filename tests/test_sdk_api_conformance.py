"""
Tests for SDK API conformance.

Verifies that the provider uses correct SDK methods:
- send() + on() for streaming, NOT send_message()
- hooks via session config dict, NOT register_pre_tool_use_hook()

Contract anchors:
- sdk-boundary:Translation:MUST:1 - correct SDK method usage
- deny-destroy:DenyHook:MUST:1 - deny hook installed correctly
"""

import pytest


class TestSdkMethodUsage:
    """Verify correct SDK method names are used."""

    @pytest.mark.asyncio
    async def test_no_send_message_method_calls(self) -> None:
        """AC-1: No code should call session.send_message().

        The SDK has send() and send_and_wait(), not send_message().
        This test verifies by scanning source code for the forbidden pattern.
        """
        from pathlib import Path

        src_dir = Path("amplifier_module_provider_github_copilot")
        forbidden_pattern = ".send_message("
        violations: list[str] = []

        for py_file in src_dir.rglob("*.py"):
            content = py_file.read_text(encoding="utf-8")
            if forbidden_pattern in content:
                for i, line in enumerate(content.split("\n"), 1):
                    stripped = line.strip()
                    if forbidden_pattern in line and not stripped.startswith("#"):
                        violations.append(f"{py_file}:{i}: {stripped}")

        assert not violations, (
            f"Found {len(violations)} references to send_message():\n"
            + "\n".join(violations)
            + "\n\nSDK uses send() + on() pattern, not send_message()."
        )

    @pytest.mark.asyncio
    async def test_no_register_pre_tool_use_hook_calls(self) -> None:
        """AC-2: No code should call session.register_pre_tool_use_hook().

        Hooks are passed via session config, not method calls.
        """
        # This test verifies by scanning source code for the forbidden pattern
        from pathlib import Path

        src_dir = Path("amplifier_module_provider_github_copilot")
        forbidden_pattern = "register_pre_tool_use_hook"
        violations: list[str] = []

        for py_file in src_dir.rglob("*.py"):
            content = py_file.read_text(encoding="utf-8")
            if forbidden_pattern in content:
                # Find line numbers, excluding comments and docstrings
                in_docstring = False
                for i, line in enumerate(content.split("\n"), 1):
                    stripped = line.strip()
                    # Track docstring boundaries
                    if '"""' in stripped or "'''" in stripped:
                        # Toggle docstring state (simplified: assumes balanced quotes per line)
                        in_docstring = not in_docstring
                        continue
                    if (
                        forbidden_pattern in line
                        and not stripped.startswith("#")
                        and not in_docstring
                    ):
                        violations.append(f"{py_file}:{i}")

        assert not violations, (
            f"Found {len(violations)} references to register_pre_tool_use_hook:\n"
            + "\n".join(violations)
            + "\n\nHooks must be passed via session config 'hooks' key, "
            "not via method call."
        )


class TestDenyHookViaConfig:
    """Verify deny hook is passed via session config."""

    def test_deny_hook_in_session_config(self) -> None:
        """AC-3: Deny hook passed via session config 'hooks' key.

        Contract: deny-destroy:DenyHook:MUST:1
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            CopilotClientWrapper,
        )

        # The session config should include hooks dict with on_pre_tool_use
        # when session() is called

        # Create wrapper
        wrapper = CopilotClientWrapper()

        # Inspect the session() method to verify it builds correct config
        # The implementation should pass hooks={'on_pre_tool_use': deny_fn}
        import inspect

        source = inspect.getsource(wrapper.session)

        # Check for hooks pattern in session config
        assert "'hooks'" in source or '"hooks"' in source, (
            "session() must include 'hooks' key in session config for deny hook"
        )


class TestStreamingPatternUsage:
    """Verify correct send() + on() streaming pattern."""

    @pytest.mark.asyncio
    async def test_streaming_uses_send_and_on_pattern(self) -> None:
        """AC-4: Streaming uses session.on() + session.send() pattern.

        Contract: sdk-boundary:Events:MUST:1

        The correct SDK pattern is:
        1. Register event handler with session.on(handler)
        2. Send message with session.send(options)
        3. Events arrive via handler callback
        """
        # This test verifies the pattern is used in provider.py
        from pathlib import Path

        provider_py = Path("amplifier_module_provider_github_copilot/provider.py")
        content = provider_py.read_text(encoding="utf-8")

        # Check for send() method (not send_message)
        # The implementation should use sdk_session.send() or similar
        has_send_pattern = ".send(" in content and ".send_message(" not in content

        assert has_send_pattern, (
            "provider.py should use .send() for message sending, "
            "not .send_message() which doesn't exist in SDK"
        )


class TestSessionConfigShape:
    """Verify session config includes required keys."""

    def test_session_config_includes_hooks_key(self) -> None:
        """AC-4: Session config dict has 'hooks' key.

        Contract: sdk-boundary:Config:MUST:1
        """
        from pathlib import Path

        client_py = Path("amplifier_module_provider_github_copilot/sdk_adapter/client.py")
        content = client_py.read_text(encoding="utf-8")

        # Session config should include hooks for deny hook
        # Look for session_config["hooks"] or session_config.update({"hooks": ...})
        has_hooks_config = (
            'session_config["hooks"]' in content
            or "session_config['hooks']" in content
            or "'hooks':" in content
            or '"hooks":' in content
        )

        assert has_hooks_config, (
            "client.py must set 'hooks' key in session_config for deny hook installation. "
            "The correct pattern is: session_config['hooks'] = {'on_pre_tool_use': deny_fn}"
        )


class TestNoHallucinatedMethods:
    """Comprehensive check for hallucinated SDK methods."""

    def test_no_send_message_in_codebase(self) -> None:
        """Verify send_message() is not used anywhere.

        This method was hallucinated - it doesn't exist in the SDK.
        """
        from pathlib import Path

        src_dir = Path("amplifier_module_provider_github_copilot")
        violations: list[str] = []

        for py_file in src_dir.rglob("*.py"):
            content = py_file.read_text(encoding="utf-8")
            if ".send_message(" in content:
                for i, line in enumerate(content.split("\n"), 1):
                    if ".send_message(" in line and not line.strip().startswith("#"):
                        violations.append(f"{py_file}:{i}: {line.strip()}")

        assert not violations, (
            "Found send_message() calls (hallucinated method):\n"
            + "\n".join(violations)
            + "\n\nSDK provides send() + on() pattern, not send_message()."
        )
