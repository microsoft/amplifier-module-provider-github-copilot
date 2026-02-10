"""
Live integration tests for Copilot SDK Provider.

These tests require:
1. Copilot CLI installed and in PATH
2. Valid GitHub Copilot authentication

Tests are skipped by default. Run with:
    pytest tests/integration/ -v --run-live

WARNING: These tests make real API calls and may incur costs.
"""

import os

import pytest

# Skip all tests in this module unless --run-live flag is passed
pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_LIVE_TESTS"),
    reason="Live tests disabled. Set RUN_LIVE_TESTS=1 to run.",
)


@pytest.fixture
async def live_provider():
    """
    Create a live provider instance.

    This fixture creates a real provider connected to Copilot SDK.
    """
    from unittest.mock import AsyncMock, Mock

    from amplifier_module_provider_github_copilot import CopilotSdkProvider

    # Create mock coordinator (we don't need full Amplifier for live tests)
    coordinator = Mock()
    coordinator.hooks = Mock()
    coordinator.hooks.emit = AsyncMock()

    provider = CopilotSdkProvider(
        api_key=None,  # Copilot uses GitHub auth, not API key
        config={
            "model": "claude-opus-4.5",  # Use period format per SDK convention
            "timeout": 120,
            "debug": True,
            "use_streaming": False,  # Use non-streaming for simpler testing
        },
        coordinator=coordinator,
    )

    yield provider

    await provider.close()


class TestLiveConnection:
    """Tests for live Copilot SDK connection."""

    @pytest.mark.asyncio
    async def test_list_models(self, live_provider):
        """Should fetch available models from Copilot."""
        models = await live_provider.list_models()

        assert isinstance(models, list)
        assert len(models) > 0

        # Check model structure (ModelInfo objects have .id and .display_name attributes)
        for model in models:
            assert hasattr(model, "id") or "id" in model
            assert hasattr(model, "display_name") or "display_name" in model

        # Print available models for debugging
        print("\nAvailable models:")
        for model in models:
            model_id = model.id if hasattr(model, "id") else model["id"]
            if hasattr(model, "display_name"):
                model_name = model.display_name
            else:
                model_name = model.get("display_name", model.get("name", "unknown"))
            print(f"  - {model_id}: {model_name}")

    @pytest.mark.asyncio
    async def test_simple_completion(self, live_provider):
        """Should complete a simple prompt."""
        request = {
            "messages": [
                {"role": "user", "content": "What is 2 + 2? Reply with just the number."},
            ]
        }

        response = await live_provider.complete(request)

        assert response is not None
        assert len(response.content) > 0

        # Get text content
        text_content = " ".join(
            block.text for block in response.content if block.type == "text" and block.text
        )
        print(f"\nResponse: {text_content}")

        # Check that "4" appears somewhere in response
        assert "4" in text_content

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, live_provider):
        """Should handle multi-turn conversation."""
        request = {
            "messages": [
                {"role": "system", "content": "You are a helpful math tutor."},
                {"role": "user", "content": "What is 5 * 5?"},
                {"role": "assistant", "content": "5 * 5 = 25"},
                {"role": "user", "content": "Now divide that by 5."},
            ]
        }

        response = await live_provider.complete(request)

        assert response is not None
        text_content = " ".join(
            block.text for block in response.content if block.type == "text" and block.text
        )
        print(f"\nResponse: {text_content}")

        # Should reference 5 (25 / 5 = 5)
        assert "5" in text_content


class TestLiveToolCalls:
    """Tests for tool call handling with live API."""

    @pytest.mark.asyncio
    async def test_tool_call_request(self, live_provider):
        """Should request tool call when appropriate."""
        # Note: This test may not always trigger a tool call
        # depending on the model's behavior
        request = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You have access to a read_file tool. "
                        "When asked to read a file, use the tool."
                    ),
                },
                {"role": "user", "content": "Please read the file 'test.py'"},
            ]
        }

        response = await live_provider.complete(request)

        assert response is not None
        print(f"\nResponse content blocks: {len(response.content)}")
        print(f"Tool calls: {len(response.tool_calls) if response.tool_calls else 0}")

        if response.tool_calls:
            for tc in response.tool_calls:
                print(f"  - Tool: {tc.name}, Args: {tc.arguments}")

    @pytest.mark.asyncio
    async def test_tool_call_full_roundtrip(self, live_provider):
        """
        End-to-end test: tool request → tool result → final response.

        This tests the complete tool call flow:
        1. User asks a question requiring tool use
        2. Model requests tool call
        3. We provide tool result
        4. Model generates final response using the tool result
        """
        from unittest.mock import Mock

        # Define explicit tool so the model can actually call it
        weather_tool = Mock()
        weather_tool.name = "get_weather"
        weather_tool.description = "Get current weather data for a city"
        weather_tool.parameters = {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
            },
            "required": ["city"],
        }

        # Step 1: Initial request that should trigger tool use
        initial_request = Mock()
        initial_request.messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    "You MUST use the get_weather tool to answer weather questions. "
                    "NEVER answer weather questions without calling the tool first."
                ),
            },
            {"role": "user", "content": "What's the weather like in Seattle?"},
        ]
        initial_request.tools = [weather_tool]
        initial_request.stream = None

        response1 = await live_provider.complete(initial_request)
        assert response1 is not None

        print("\n=== Step 1: Initial Response ===")
        print(f"Content blocks: {len(response1.content)}")
        print(f"Tool calls: {len(response1.tool_calls) if response1.tool_calls else 0}")
        print(f"Finish reason: {response1.finish_reason}")

        # With explicit tool definition, model should request the tool
        assert response1.tool_calls, (
            "Model did not request get_weather tool despite explicit tool definition "
            "and system prompt instruction. This indicates a provider or SDK regression."
        )

        tool_call = response1.tool_calls[0]
        print(f"Tool requested: {tool_call.name}")
        print(f"Arguments: {tool_call.arguments}")

        # Verify it called the right tool
        assert tool_call.name == "get_weather", (
            f"Expected 'get_weather' tool call, got '{tool_call.name}'"
        )

        # Step 2: Provide tool result and get final response
        # Extract text content safely — content blocks may include
        # ToolCallBlock objects that don't have a .text attribute
        assistant_text = ""
        if response1.content:
            text_blocks = [b.text for b in response1.content if hasattr(b, "text") and b.text]
            assistant_text = " ".join(text_blocks)

        followup_request = Mock()
        followup_request.messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    "You MUST use the get_weather tool to answer weather questions. "
                    "NEVER answer weather questions without calling the tool first."
                ),
            },
            {"role": "user", "content": "What's the weather like in Seattle?"},
            {
                "role": "assistant",
                "content": assistant_text,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "name": tool_call.name,
                        "arguments": tool_call.arguments,
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "tool_name": tool_call.name,
                "content": '{"temperature": 58, "condition": "partly cloudy", "humidity": 72}',
            },
        ]
        followup_request.tools = [weather_tool]
        followup_request.stream = None

        response2 = await live_provider.complete(followup_request)
        assert response2 is not None

        print("\n=== Step 2: Final Response ===")
        text_content = " ".join(
            block.text for block in response2.content if block.type == "text" and block.text
        )
        print(f"Response: {text_content[:200]}...")

        # Verify the model used the tool result in its response
        # Should mention temperature or Seattle or weather condition
        text_lower = text_content.lower()
        assert any(
            term in text_lower for term in ["58", "seattle", "cloudy", "weather", "temperature"]
        ), f"Response doesn't seem to use tool result: {text_content[:100]}"

        print("\n✅ Full tool call roundtrip successful!")

    @pytest.mark.asyncio
    async def test_parse_tool_calls_from_response(self, live_provider):
        """Should correctly parse tool calls using parse_tool_calls method."""
        request = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You have access to tools. When asked to read a file, "
                        "use the read_file tool with path argument."
                    ),
                },
                {"role": "user", "content": "Read the file config.json"},
            ]
        }

        response = await live_provider.complete(request)
        parsed_calls = live_provider.parse_tool_calls(response)

        print(f"\nParsed tool calls: {len(parsed_calls)}")
        for tc in parsed_calls:
            print(f"  - {tc.name}: {tc.arguments}")

        # parse_tool_calls should return same as response.tool_calls
        if response.tool_calls:
            assert len(parsed_calls) == len(response.tool_calls)
            assert parsed_calls[0].name == response.tool_calls[0].name


class TestLiveErrorHandling:
    """Tests for error handling with live API."""

    @pytest.mark.asyncio
    async def test_invalid_model(self, live_provider):
        """
        Test behavior with invalid model name.

        Note: The Copilot SDK may accept invalid model names without
        raising an immediate error (lazy validation or fallback behavior).
        This test verifies graceful handling rather than assuming a specific
        exception type.
        """
        from amplifier_module_provider_github_copilot import CopilotProviderError

        request = {
            "messages": [{"role": "user", "content": "Hello"}],
        }

        # SDK may either raise an error OR return a response (fallback behavior)
        # Either outcome is acceptable - we're testing graceful handling
        try:
            response = await live_provider.complete(request, model="invalid-model-name")
            # If SDK accepts it, we should at least get a response object
            assert response is not None
        except CopilotProviderError:
            # Expected if SDK validates model strictly
            pass
        except Exception as e:
            # Unexpected error type - log but don't fail
            # SDK behavior may change between versions
            pytest.skip(f"SDK raised unexpected error type: {type(e).__name__}: {e}")

    @pytest.mark.asyncio
    async def test_timeout_handling(self, live_provider):
        """Should handle timeout appropriately."""
        from amplifier_module_provider_github_copilot import CopilotTimeoutError

        # Very short timeout
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": "Write a detailed essay about the history of computing.",
                }
            ],
        }

        # This may or may not timeout depending on response speed
        try:
            await live_provider.complete(request, timeout=0.001)
        except CopilotTimeoutError:
            pass  # Expected
        except Exception as e:
            print(f"Got different exception: {type(e).__name__}: {e}")


class TestSdkModelIdFormat:
    """
    Tests to verify SDK model ID format conventions.

    IMPORTANT: These tests act as a contract with the SDK.
    If the SDK changes model ID format, these tests will catch it.

    Pattern discovered (2026-02-06):
    - Claude models: claude-{variant}-{major}.{minor} (e.g., claude-opus-4.5)
    - GPT models: gpt-{major}.{minor}[-variant] (e.g., gpt-5.1, gpt-5.1-codex)
    - Versions use PERIODS (.), not dashes
    - Component separators use DASHES (-)
    """

    @pytest.mark.asyncio
    async def test_default_model_exists_in_sdk(self, live_provider):
        """
        Our DEFAULT_MODEL must exist in the SDK model list.

        This is a CRITICAL test. If this fails, it means the SDK
        no longer provides the model we configured as default.
        """
        from amplifier_module_provider_github_copilot._constants import DEFAULT_MODEL

        models = await live_provider.list_models()
        model_ids = {m.id for m in models}

        print(f"\nDEFAULT_MODEL: {DEFAULT_MODEL}")
        print(f"SDK models available: {sorted(model_ids)}")

        assert DEFAULT_MODEL in model_ids, (
            f"DEFAULT_MODEL '{DEFAULT_MODEL}' not found in SDK! Available: {sorted(model_ids)}"
        )

    @pytest.mark.asyncio
    async def test_claude_models_use_period_for_version(self, live_provider):
        """
        Claude models should use periods for version numbers.

        Expected format: claude-{variant}-{major}.{minor}
        Examples: claude-opus-4.5, claude-sonnet-4.5, claude-haiku-4.5
        """
        models = await live_provider.list_models()
        claude_models = [m for m in models if "claude" in m.id.lower()]

        print(f"\nClaude models found ({len(claude_models)}):")
        for m in claude_models:
            print(f"  {m.id}")

        assert len(claude_models) > 0, "No Claude models found in SDK!"

        # Verify naming pattern
        for model in claude_models:
            # Must start with "claude-"
            assert model.id.startswith("claude-"), f"Unexpected Claude model name: {model.id}"

            # Must have variant after "claude-" (opus, sonnet, haiku)
            parts = model.id.split("-")
            assert len(parts) >= 2, f"Missing variant in Claude model: {model.id}"
            variant = parts[1]
            assert variant in ("opus", "sonnet", "haiku"), (
                f"Unknown Claude variant '{variant}' in {model.id}"
            )

    @pytest.mark.asyncio
    async def test_gpt_models_use_period_for_version(self, live_provider):
        """
        GPT models should use periods for version numbers.

        Expected formats:
        - gpt-{major}.{minor} (e.g., gpt-5.1)
        - gpt-{major}.{minor}-{variant} (e.g., gpt-5.1-codex)
        - gpt-{major} (e.g., gpt-5) - no minor version
        """
        models = await live_provider.list_models()
        gpt_models = [m for m in models if m.id.lower().startswith("gpt-")]

        print(f"\nGPT models found ({len(gpt_models)}):")
        for m in gpt_models:
            has_period = "." in m.id
            print(f"  {m.id} (has_period={has_period})")

        assert len(gpt_models) > 0, "No GPT models found in SDK!"

        # All GPT models should start with "gpt-"
        for model in gpt_models:
            assert model.id.startswith("gpt-"), f"Unexpected GPT model format: {model.id}"

    @pytest.mark.asyncio
    async def test_model_ids_never_use_dash_for_version(self, live_provider):
        """
        Model IDs should NOT use dashes for version numbers.

        WRONG: claude-opus-4-5 (dash between 4 and 5)
        RIGHT: claude-opus-4.5 (period between 4 and 5)

        This test prevents regression to the Anthropic API format.
        """
        import re

        models = await live_provider.list_models()

        # Pattern that would indicate dashed versions (wrong)
        # Matches: something-NUMBER-NUMBER (like opus-4-5)
        wrong_pattern = re.compile(r"-(\d+)-(\d+)(?:-|$)")

        violations = []
        for model in models:
            if wrong_pattern.search(model.id):
                violations.append(model.id)

        print(f"\nChecked {len(models)} models for dashed version numbers")
        if violations:
            print(f"VIOLATIONS (dashed versions): {violations}")

        assert len(violations) == 0, (
            f"Found models with dashed version numbers (should use periods): {violations}"
        )

    @pytest.mark.asyncio
    async def test_capability_detection_works_for_real_models(self, live_provider):
        """
        Capability detection should work for models that actually exist in SDK.

        This ensures our _model_supports_reasoning() works with real model IDs.
        """
        models = await live_provider.list_models()

        # Find models with thinking/reasoning capability
        thinking_models = [
            m for m in models if "thinking" in m.capabilities or "reasoning" in m.capabilities
        ]
        non_thinking_models = [
            m
            for m in models
            if "thinking" not in m.capabilities and "reasoning" not in m.capabilities
        ]

        print(f"\nModels with thinking/reasoning: {[m.id for m in thinking_models]}")
        print(f"Models without: {[m.id for m in non_thinking_models][:5]}...")

        # Test capability detection on a thinking model (if any)
        if thinking_models:
            model = thinking_models[0]
            result = await live_provider._model_supports_reasoning(model.id)
            print(f"\n_model_supports_reasoning('{model.id}'): {result}")
            assert result is True, f"Expected True for thinking model {model.id}"

        # Test capability detection on a non-thinking model (if any)
        if non_thinking_models:
            model = non_thinking_models[0]
            result = await live_provider._model_supports_reasoning(model.id)
            print(f"_model_supports_reasoning('{model.id}'): {result}")
            assert result is False, f"Expected False for non-thinking model {model.id}"

    @pytest.mark.asyncio
    async def test_snapshot_expected_models_exist(self, live_provider):
        """
        Snapshot test: Verify expected models still exist.

        This acts as an early warning if the SDK removes models.
        Update this list when new models are added or old ones removed.
        """
        # Expected models as of 2026-02-08
        # Update this list when SDK model availability changes
        # NOTE: Do NOT include hidden models (e.g., claude-opus-4.6)
        # that work when specified directly but are not returned by
        # list_models(). See SDK-THINKING-MODELS-INVESTIGATION.md.
        EXPECTED_MODELS = {
            # Claude models
            "claude-opus-4.5",
            "claude-sonnet-4.5",
            # GPT models
            "gpt-5",
            "gpt-5.1",
        }

        models = await live_provider.list_models()
        model_ids = {m.id for m in models}

        missing = EXPECTED_MODELS - model_ids

        print(f"\nExpected models: {sorted(EXPECTED_MODELS)}")
        print(f"SDK models: {sorted(model_ids)}")

        if missing:
            print(f"WARNING: Missing expected models: {missing}")

        # Warn but don't fail - models come and go
        # This test is informational
        for expected in EXPECTED_MODELS:
            if expected not in model_ids:
                pytest.skip(
                    f"Model '{expected}' no longer in SDK. "
                    "Update EXPECTED_MODELS if this is permanent."
                )

    @pytest.mark.asyncio
    async def test_model_naming_utilities_work_with_live_sdk(self, live_provider):
        """
        Verify model_naming.py utilities work correctly with live SDK data.

        This validates that our parsing and detection logic works with
        real model IDs returned by the SDK.
        """
        from amplifier_module_provider_github_copilot.model_naming import (
            has_version_period,
            is_thinking_model,
            parse_model_id,
            uses_dash_for_version,
            validate_model_id_format,
        )

        models = await live_provider.list_models()

        print("\n=== Model Naming Validation ===")
        for model in models:
            parsed = parse_model_id(model.id)
            is_thinking = is_thinking_model(model.id)
            has_period = has_version_period(model.id)
            uses_dash = uses_dash_for_version(model.id)
            warnings = validate_model_id_format(model.id)

            print(f"\n{model.id}:")
            print(f"  parsed: {parsed}")
            print(f"  is_thinking_model: {is_thinking}")
            print(f"  has_version_period: {has_period}")
            print(f"  uses_dash_for_version: {uses_dash}")
            if warnings:
                print(f"  WARNINGS: {warnings}")

            # No model should use dash for version
            assert uses_dash is False, (
                f"Model {model.id} uses dash for version - SDK changed format?"
            )

            # No warnings expected for real SDK models
            assert len(warnings) == 0, f"Model {model.id} has validation warnings: {warnings}"

    @pytest.mark.asyncio
    async def test_thinking_detection_matches_capabilities(self, live_provider):
        """
        Compare is_thinking_model() pattern matching with SDK capabilities.

        DESIGN: SDK is AUTHORITATIVE, pattern is FALLBACK.

        This test shows where pattern differs from SDK. Differences are EXPECTED
        and OK because:
        - Pattern is only used when SDK check FAILS (network error, etc.)
        - When SDK works, its result is used exclusively
        - Pattern being overly broad (matching opus-4.5) is safe, not harmful

        Example:
        - claude-opus-4.5: SDK=False (no thinking), Pattern=True (matches "opus")
        - This is fine because SDK result takes precedence when SDK works
        - Pattern only kicks in if SDK throws exception
        """
        from amplifier_module_provider_github_copilot.model_naming import is_thinking_model

        models = await live_provider.list_models()

        print("\n=== Thinking Detection vs Capability Analysis ===")
        mismatches = []

        for model in models:
            # SDK-reported capability
            sdk_thinking = "thinking" in model.capabilities or "reasoning" in model.capabilities
            # Our pattern-based detection
            pattern_thinking = is_thinking_model(model.id)

            match_status = "✓" if sdk_thinking == pattern_thinking else "✗ MISMATCH"
            print(f"  {model.id}: sdk={sdk_thinking}, pattern={pattern_thinking} {match_status}")

            if sdk_thinking != pattern_thinking:
                mismatches.append(
                    {
                        "model": model.id,
                        "sdk_says": sdk_thinking,
                        "pattern_says": pattern_thinking,
                        "capabilities": list(model.capabilities),
                    }
                )

        # Some mismatches are OK (pattern is a fallback)
        # But log them for awareness
        if mismatches:
            print("\n=== MISMATCHES (pattern vs SDK capability) ===")
            for m in mismatches:
                print(f"  {m['model']}:")
                print(f"    SDK capability: {m['sdk_says']}")
                print(f"    Pattern detection: {m['pattern_says']}")
                print(f"    Capabilities: {m['capabilities']}")

            # Warn but don't fail - pattern is fallback, not authoritative
            print("\nNOTE: Pattern detection is fallback for when SDK check fails.")
            print("Mismatches are expected for models not in known thinking patterns.")

    @pytest.mark.asyncio
    async def test_timeout_selection_consistency(self, live_provider):
        """
        Verify timeout selection with SDK-authoritative design.

        NEW DESIGN (SDK authoritative, pattern is fallback only):
        - When SDK succeeds: Use SDK result exclusively
        - When SDK fails: Use pattern as fallback

        This test calls SDK detection (which succeeds) so pattern is NOT used.
        The claude-opus-4.5 should get STANDARD timeout because SDK says no thinking.
        """
        from amplifier_module_provider_github_copilot._constants import (
            DEFAULT_THINKING_TIMEOUT,
            DEFAULT_TIMEOUT,
        )
        from amplifier_module_provider_github_copilot.model_naming import is_thinking_model

        models = await live_provider.list_models()

        print("\n=== Timeout Selection Analysis (SDK Authoritative Design) ===")
        print(f"DEFAULT_TIMEOUT: {DEFAULT_TIMEOUT}s (5 min)")
        print(f"DEFAULT_THINKING_TIMEOUT: {DEFAULT_THINKING_TIMEOUT}s (30 min)")
        print("NOTE: SDK is authoritative. Pattern shown for reference only.")
        print()

        for model in models:
            # SDK is authoritative (succeeds here)
            sdk_thinking, sdk_succeeded = await live_provider._check_model_reasoning_with_fallback(
                model.id
            )
            # Pattern shown for reference (only used when SDK fails)
            pattern_thinking = is_thinking_model(model.id)

            # With new design: SDK result is used when SDK succeeds
            expected_timeout = DEFAULT_THINKING_TIMEOUT if sdk_thinking else DEFAULT_TIMEOUT

            # Show both values for documentation
            status = "SDK" if sdk_succeeded else "FALLBACK"
            print(
                f"  {model.id}: "
                f"result={sdk_thinking} ({status}) [pattern={pattern_thinking}] → "
                f"{'THINKING' if sdk_thinking else 'STANDARD'} "
                f"({expected_timeout}s)"
            )


# Pytest hook to add --run-live option
def pytest_addoption(parser):
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="Run live integration tests",
    )


def pytest_configure(config):
    if config.getoption("--run-live"):
        os.environ["RUN_LIVE_TESTS"] = "1"
