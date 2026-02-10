"""
Coverage gap tests for modules with small remaining gaps.

Targets uncovered lines across:
- exceptions.py: lines 204-207 (CopilotAbortError)
- converters.py: lines 168, 332-334 (unknown block type, failed tool request)
- model_naming.py: lines 322, 357 (year-like pattern, parse failure)
- sdk_driver.py: lines 308, 590, 610 (circuit breaker timeout before start,
  loop error + error event propagation)
"""

from unittest.mock import Mock

import pytest

from amplifier_module_provider_github_copilot.exceptions import (
    CopilotAbortError,
    CopilotProviderError,
    CopilotSdkLoopError,
)

# ═══════════════════════════════════════════════════════════════════════════
# exceptions.py — CopilotAbortError (lines 204-207)
# ═══════════════════════════════════════════════════════════════════════════


class TestCopilotAbortError:
    """Tests for CopilotAbortError exception class."""

    def test_is_subclass_of_provider_error(self):
        """CopilotAbortError should inherit from CopilotProviderError."""
        assert issubclass(CopilotAbortError, CopilotProviderError)

    def test_can_be_instantiated_with_message(self):
        """Should accept a message string."""
        err = CopilotAbortError("abort() timed out after 5s")
        assert str(err) == "abort() timed out after 5s"

    def test_can_be_caught_as_provider_error(self):
        """Should be catchable as CopilotProviderError."""
        with pytest.raises(CopilotProviderError):
            raise CopilotAbortError("test")

    def test_can_be_raised_and_caught(self):
        """Should work as a standard exception."""
        with pytest.raises(CopilotAbortError, match="session abort failed"):
            raise CopilotAbortError("session abort failed")


# ═══════════════════════════════════════════════════════════════════════════
# converters.py — _extract_content unknown block type (line 168)
# ═══════════════════════════════════════════════════════════════════════════


class TestExtractContentUnknownBlockType:
    """Tests for _extract_content handling of unknown content block types."""

    def test_unknown_block_type_with_text_key(self):
        """Should extract 'text' key from unknown block types."""
        from amplifier_module_provider_github_copilot.converters import _extract_content

        msg = {
            "content": [
                {"type": "custom_widget", "text": "Widget output"},
            ]
        }
        result = _extract_content(msg)
        assert result == "Widget output"

    def test_unknown_block_type_with_content_key(self):
        """Should fall back to 'content' key when 'text' is missing."""
        from amplifier_module_provider_github_copilot.converters import _extract_content

        msg = {
            "content": [
                {"type": "rich_embed", "content": "Embedded content"},
            ]
        }
        result = _extract_content(msg)
        assert result == "Embedded content"

    def test_unknown_block_type_neither_key(self):
        """Should return empty string when neither 'text' nor 'content' key exists."""
        from amplifier_module_provider_github_copilot.converters import _extract_content

        msg = {
            "content": [
                {"type": "binary_data", "data": b"bytes"},
            ]
        }
        result = _extract_content(msg)
        assert result == ""

    def test_mixed_known_and_unknown_blocks(self):
        """Should handle mix of known and unknown block types."""
        from amplifier_module_provider_github_copilot.converters import _extract_content

        msg = {
            "content": [
                {"type": "text", "text": "Known text"},
                {"type": "custom", "text": "Custom text"},
                {"type": "image_url", "url": "http://img.png"},
            ]
        }
        result = _extract_content(msg)
        assert "Known text" in result
        assert "Custom text" in result
        assert "[Image]" in result


# ═══════════════════════════════════════════════════════════════════════════
# converters.py — _extract_tool_request failure path (lines 332-334)
# ═══════════════════════════════════════════════════════════════════════════


class TestConvertToolRequestFailure:
    """Tests for _convert_tool_request exception handling path."""

    def test_unknown_tool_request_format_returns_none(self):
        """Should return None and log warning for unknown format."""
        from amplifier_module_provider_github_copilot.converters import _convert_tool_request

        result = _convert_tool_request(42)
        assert result is None

    def test_tool_request_string_type_returns_none(self):
        """Should return None for string input (not dict or object)."""
        from amplifier_module_provider_github_copilot.converters import _convert_tool_request

        result = _convert_tool_request("not a tool request")
        assert result is None

    def test_tool_request_exception_returns_none(self):
        """Should return None when conversion raises an exception."""
        from amplifier_module_provider_github_copilot.converters import _convert_tool_request

        # Create object that raises on attribute access
        bad_request = Mock()
        bad_request.tool_call_id = "id"
        bad_request.name = "test"
        # Arguments property that raises
        type(bad_request).arguments = property(
            lambda self: (_ for _ in ()).throw(ValueError("bad args"))
        )

        result = _convert_tool_request(bad_request)
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════
# model_naming.py — _find_dash_version_pair year-like skip (line 322)
# ═══════════════════════════════════════════════════════════════════════════


class TestFindDashVersionPairYearSkip:
    """Tests for _find_dash_version_pair year-like pattern skipping."""

    def test_year_like_pattern_not_treated_as_version(self):
        """Model IDs like 'model-2024-01' should not flag 01 as version."""
        from amplifier_module_provider_github_copilot.model_naming import (
            _find_dash_version_pair,
        )

        result = _find_dash_version_pair("model-2024-01")
        assert result is None

    def test_year_like_pattern_with_suffix(self):
        """Model IDs like 'gpt-2024-01-preview' should skip year-month pair."""
        from amplifier_module_provider_github_copilot.model_naming import (
            _find_dash_version_pair,
        )

        result = _find_dash_version_pair("gpt-2024-01-preview")
        assert result is None

    def test_real_version_still_detected(self):
        """Non-year patterns like 'claude-opus-4-5' should still be detected."""
        from amplifier_module_provider_github_copilot.model_naming import (
            _find_dash_version_pair,
        )

        result = _find_dash_version_pair("claude-opus-4-5")
        assert result == ("4-5", "4.5")


# ═══════════════════════════════════════════════════════════════════════════
# model_naming.py — validate_model_id_format with unparseable ID (line 357)
# ═══════════════════════════════════════════════════════════════════════════


class TestValidateModelIdFormatUnparseable:
    """Tests for validate_model_id_format with unparseable model IDs."""

    def test_unparseable_model_id_returns_warning(self):
        """Should return warning when model ID doesn't match any pattern."""
        from amplifier_module_provider_github_copilot.model_naming import (
            validate_model_id_format,
        )

        # "123" starts with digits, not matching [a-z]+ family pattern
        warnings = validate_model_id_format("123")
        assert any("does not match expected pattern" in w for w in warnings)

    def test_empty_string_returns_warning(self):
        """Empty string should produce a warning."""
        from amplifier_module_provider_github_copilot.model_naming import (
            validate_model_id_format,
        )

        warnings = validate_model_id_format("")
        assert len(warnings) > 0

    def test_valid_model_id_no_warnings(self):
        """Valid model ID should produce no warnings."""
        from amplifier_module_provider_github_copilot.model_naming import (
            validate_model_id_format,
        )

        warnings = validate_model_id_format("claude-opus-4.5")
        assert warnings == []


# ═══════════════════════════════════════════════════════════════════════════
# sdk_driver.py — CircuitBreaker check_timeout before start (line 308)
# ═══════════════════════════════════════════════════════════════════════════


class TestCircuitBreakerTimeoutBeforeStart:
    """Tests for CircuitBreaker.check_timeout() when start() not called."""

    def test_check_timeout_returns_true_before_start(self):
        """check_timeout should return True (OK) when start_time is None."""
        from amplifier_module_provider_github_copilot.sdk_driver import CircuitBreaker

        cb = CircuitBreaker(max_turns=3, timeout_seconds=10.0)
        # Don't call start() — _start_time is None
        assert cb.check_timeout() is True

    def test_check_timeout_trips_after_start(self):
        """check_timeout should trip after timeout exceeds."""
        from amplifier_module_provider_github_copilot.sdk_driver import CircuitBreaker

        cb = CircuitBreaker(max_turns=3, timeout_seconds=0.0)
        cb.start()
        # Timeout of 0 seconds should trip immediately
        result = cb.check_timeout()
        # Elapsed > 0 > timeout=0 → trips
        assert result is False
        assert cb.is_tripped


# ═══════════════════════════════════════════════════════════════════════════
# sdk_driver.py — SdkEventHandler wait paths (lines 590, 610)
# ═══════════════════════════════════════════════════════════════════════════


class TestSdkEventHandlerWaitPaths:
    """Tests for SdkEventHandler.wait_for_capture_or_idle error paths."""

    @pytest.mark.asyncio
    async def test_timeout_with_circuit_breaker_trip_raises_loop_error(self):
        """Should raise CopilotSdkLoopError when timeout AND circuit breaker trips (line 590)."""
        from amplifier_module_provider_github_copilot.sdk_driver import SdkEventHandler

        handler = SdkEventHandler(max_turns=3, first_turn_only=True)

        # Set CB timeout_seconds to 0 so check_timeout() trips after any elapsed time
        handler.circuit_breaker.timeout_seconds = 0.0

        with pytest.raises(CopilotSdkLoopError, match="timeout"):
            await handler.wait_for_capture_or_idle(timeout=0.01)

    @pytest.mark.asyncio
    async def test_error_event_propagated_after_wait(self):
        """Should raise stored error after capture event fires (line 610)."""
        from amplifier_module_provider_github_copilot.sdk_driver import SdkEventHandler

        handler = SdkEventHandler(max_turns=3, first_turn_only=True)

        # Store an error event
        handler._error_event = RuntimeError("SDK internal error")

        # Set the capture event so wait completes
        handler._capture_event.set()

        with pytest.raises(RuntimeError, match="SDK internal error"):
            await handler.wait_for_capture_or_idle(timeout=5.0)

    @pytest.mark.asyncio
    async def test_circuit_breaker_tripped_after_normal_wait_raises(self):
        """Should raise CopilotSdkLoopError when circuit breaker was tripped during event handling."""
        from amplifier_module_provider_github_copilot.sdk_driver import SdkEventHandler

        handler = SdkEventHandler(max_turns=3, first_turn_only=True)

        # Set capture event so wait completes immediately
        handler._capture_event.set()

        # Patch start() to also trip the CB after normal start behavior.
        # start() is called at the top of wait_for_capture_or_idle, resetting _tripped.
        # We need the CB to be tripped AFTER start() runs but BEFORE the post-wait check.
        original_start = handler.circuit_breaker.start

        def start_then_trip():
            original_start()
            handler.circuit_breaker._tripped = True
            handler.circuit_breaker._trip_reason = "turn_count=4 > max=3"

        handler.circuit_breaker.start = start_then_trip

        with pytest.raises(CopilotSdkLoopError, match="limit exceeded"):
            await handler.wait_for_capture_or_idle(timeout=5.0)


# ═══════════════════════════════════════════════════════════════════════════
# Additional CopilotSdkLoopError tests (exercise all attributes)
# ═══════════════════════════════════════════════════════════════════════════


class TestCopilotSdkLoopError:
    """Tests for CopilotSdkLoopError exception attributes."""

    def test_all_attributes_set(self):
        """Should store all attributes correctly."""
        err = CopilotSdkLoopError(
            message="Circuit breaker tripped",
            turn_count=4,
            max_turns=3,
            tool_calls_captured=2,
        )
        assert str(err) == "Circuit breaker tripped"
        assert err.turn_count == 4
        assert err.max_turns == 3
        assert err.tool_calls_captured == 2

    def test_default_tool_calls_captured(self):
        """tool_calls_captured should default to 0."""
        err = CopilotSdkLoopError(
            message="test",
            turn_count=1,
            max_turns=3,
        )
        assert err.tool_calls_captured == 0

    def test_is_provider_error(self):
        """Should be a subclass of CopilotProviderError."""
        assert issubclass(CopilotSdkLoopError, CopilotProviderError)
