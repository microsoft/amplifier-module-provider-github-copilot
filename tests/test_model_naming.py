"""
Unit tests for model naming conventions.

These tests verify that our model ID parsing and validation functions
work correctly with MOCK data. They are complemented by integration
tests in tests/integration/test_live_copilot.py that verify against
LIVE SDK data.

The combination ensures:
1. Unit tests: Fast, deterministic, run in CI without SDK access
2. Integration tests: Validate against real SDK, catch naming changes

WHY THIS MATTERS:
- Model ID parsing affects timeout selection (5 min vs 30 min)
- Wrong parsing = premature timeouts for thinking models
- SDK naming changes could break our capability detection
"""

from __future__ import annotations

from amplifier_module_provider_github_copilot.model_naming import (
    KNOWN_THINKING_PATTERNS,
    ModelIdPattern,
    has_version_period,
    is_thinking_model,
    parse_model_id,
    uses_dash_for_version,
    validate_model_id_format,
)


class TestParseModelId:
    """Tests for parse_model_id() function."""

    # ═══════════════════════════════════════════════════════════════════════
    # Claude Models (verified from live SDK 2026-02-06)
    # ═══════════════════════════════════════════════════════════════════════

    def test_claude_opus_with_minor_version(self) -> None:
        """Claude Opus with period-based version number."""
        result = parse_model_id("claude-opus-4.5")
        assert result == ModelIdPattern(
            family="claude",
            variant="opus",
            major_version=4,
            minor_version=5,
            suffix="",
        )

    def test_claude_opus_higher_version(self) -> None:
        """Claude Opus 4.6 - newer version from SDK."""
        result = parse_model_id("claude-opus-4.6")
        assert result is not None
        assert result.family == "claude"
        assert result.variant == "opus"
        assert result.major_version == 4
        assert result.minor_version == 6

    def test_claude_sonnet_no_minor_version(self) -> None:
        """Claude Sonnet 4 - no minor version."""
        result = parse_model_id("claude-sonnet-4")
        assert result is not None
        assert result.family == "claude"
        assert result.variant == "sonnet"
        assert result.major_version == 4
        assert result.minor_version is None

    def test_claude_sonnet_with_minor_version(self) -> None:
        """Claude Sonnet 4.5 - with minor version."""
        result = parse_model_id("claude-sonnet-4.5")
        assert result is not None
        assert result.minor_version == 5

    def test_claude_haiku(self) -> None:
        """Claude Haiku model."""
        result = parse_model_id("claude-haiku-4.5")
        assert result is not None
        assert result.variant == "haiku"

    # ═══════════════════════════════════════════════════════════════════════
    # GPT Models (verified from live SDK 2026-02-06)
    # ═══════════════════════════════════════════════════════════════════════

    def test_gpt_simple_no_minor(self) -> None:
        """GPT-5 without minor version."""
        result = parse_model_id("gpt-5")
        assert result is not None
        assert result.family == "gpt"
        assert result.variant == ""  # No variant for simple gpt-N
        assert result.major_version == 5
        assert result.minor_version is None

    def test_gpt_with_minor_version(self) -> None:
        """GPT-5.1 with period-based minor version."""
        result = parse_model_id("gpt-5.1")
        assert result is not None
        assert result.major_version == 5
        assert result.minor_version == 1

    def test_gpt_with_variant_suffix(self) -> None:
        """GPT-5.1-codex with variant suffix."""
        result = parse_model_id("gpt-5.1-codex")
        assert result is not None
        assert result.major_version == 5
        assert result.minor_version == 1
        assert result.suffix == "codex"

    def test_gpt_with_compound_suffix(self) -> None:
        """GPT-5.1-codex-max with compound suffix."""
        result = parse_model_id("gpt-5.1-codex-max")
        assert result is not None
        assert result.suffix == "codex-max"

    def test_gpt_mini_variant(self) -> None:
        """GPT-5-mini variant without minor version."""
        result = parse_model_id("gpt-5-mini")
        assert result is not None
        assert result.major_version == 5
        assert result.minor_version is None
        assert result.suffix == "mini"

    def test_gpt_4_1(self) -> None:
        """GPT-4.1 - older model in SDK."""
        result = parse_model_id("gpt-4.1")
        assert result is not None
        assert result.major_version == 4
        assert result.minor_version == 1

    # ═══════════════════════════════════════════════════════════════════════
    # Other Models
    # ═══════════════════════════════════════════════════════════════════════

    def test_gemini_different_scheme(self) -> None:
        """Gemini uses different naming (dashes only)."""
        # gemini-3-pro-preview doesn't follow the period convention
        result = parse_model_id("gemini-3-pro-preview")
        # May or may not parse - that's OK, we don't require all models to parse
        if result:
            assert result.family == "gemini"

    # ═══════════════════════════════════════════════════════════════════════
    # O-Series Models (Finding #16)
    # ═══════════════════════════════════════════════════════════════════════

    def test_o1_mini_parses(self) -> None:
        """o1-mini should parse with family=o1, variant=mini.

        Finding #16: The original regex required family=[a-z]+ which
        excluded the digit in 'o1'. Updated to [a-z]+\\d* to allow.
        """
        result = parse_model_id("o1-mini")
        assert result is not None, "o1-mini should be parseable"
        assert result.family == "o1"
        assert result.variant == "mini"

    def test_o3_mini_parses(self) -> None:
        """o3-mini should parse with family=o3, variant=mini."""
        result = parse_model_id("o3-mini")
        assert result is not None, "o3-mini should be parseable"
        assert result.family == "o3"
        assert result.variant == "mini"

    def test_o1_preview_parses(self) -> None:
        """o1-preview should parse with family=o1, variant=preview."""
        result = parse_model_id("o1-preview")
        assert result is not None, "o1-preview should be parseable"
        assert result.family == "o1"
        assert result.variant == "preview"

    def test_o4_standalone_parses(self) -> None:
        """o4 alone should parse as family=o4."""
        result = parse_model_id("o4")
        assert result is not None, "o4 should be parseable"
        assert result.family == "o4"

    # ═══════════════════════════════════════════════════════════════════════
    # Edge Cases and Invalid Inputs
    # ═══════════════════════════════════════════════════════════════════════

    def test_empty_string(self) -> None:
        """Empty string should not parse."""
        result = parse_model_id("")
        assert result is None

    def test_single_word(self) -> None:
        """Single word may not parse as expected."""
        result = parse_model_id("gpt")
        # May parse as just family
        if result:
            assert result.family == "gpt"


class TestIsThinkingModel:
    """Tests for is_thinking_model() function - CRITICAL for timeout selection."""

    # ═══════════════════════════════════════════════════════════════════════
    # Known Thinking Patterns (must return True)
    # ═══════════════════════════════════════════════════════════════════════

    def test_claude_opus_is_thinking(self) -> None:
        """Claude Opus models detected by pattern matching.

        NOTE: Pattern matching returns True for ALL Opus models because
        it's a FALLBACK for when SDK capability detection fails.
        The SDK itself only reports 'thinking' for opus-4.6, not 4.5.

        This is intentional: better to use long timeout unnecessarily
        than to timeout a thinking model prematurely.
        """
        # Pattern-based detection (fallback) - matches all Opus
        assert is_thinking_model("claude-opus-4.5") is True
        assert is_thinking_model("claude-opus-4.6") is True
        assert is_thinking_model("claude-opus-5.0") is True

    def test_o1_models_are_thinking(self) -> None:
        """OpenAI o1 models MUST be detected as thinking models."""
        assert is_thinking_model("gpt-o1-preview") is True
        assert is_thinking_model("o1-mini") is True
        assert is_thinking_model("o1-2024-12-17") is True

    def test_o3_models_are_thinking(self) -> None:
        """OpenAI o3 models MUST be detected as thinking models."""
        assert is_thinking_model("o3-mini") is True
        assert is_thinking_model("gpt-o3") is True

    def test_o4_models_are_thinking(self) -> None:
        """OpenAI o4 models MUST be detected as thinking models (future-proofing)."""
        assert is_thinking_model("o4-preview") is True
        assert is_thinking_model("gpt-o4") is True

    def test_explicit_thinking_suffix(self) -> None:
        """Models with -thinking suffix MUST be detected."""
        assert is_thinking_model("gpt-5-thinking") is True
        assert is_thinking_model("claude-sonnet-4.5-thinking") is True

    def test_explicit_reasoning_suffix(self) -> None:
        """Models with -reasoning suffix MUST be detected."""
        assert is_thinking_model("gpt-5-reasoning") is True
        assert is_thinking_model("claude-4-reasoning") is True

    # ═══════════════════════════════════════════════════════════════════════
    # Non-Thinking Models (must return False)
    # ═══════════════════════════════════════════════════════════════════════

    def test_claude_sonnet_is_not_thinking(self) -> None:
        """Claude Sonnet should NOT be detected as thinking model."""
        assert is_thinking_model("claude-sonnet-4") is False
        assert is_thinking_model("claude-sonnet-4.5") is False

    def test_claude_haiku_is_not_thinking(self) -> None:
        """Claude Haiku should NOT be detected as thinking model."""
        assert is_thinking_model("claude-haiku-4.5") is False

    def test_gpt5_is_thinking_per_sdk_evidence(self) -> None:
        """GPT-5 models have 'reasoning' capability per live SDK (2026-02-06).

        Pattern matches 'gpt-5' substring for fallback detection.
        SDK reports 'reasoning' for: gpt-5, gpt-5.1, gpt-5.2, gpt-5.1-codex, etc.
        SDK reports NO reasoning for: gpt-4.1
        """
        # GPT-5.x models - SDK has 'reasoning', pattern matches
        assert is_thinking_model("gpt-5") is True
        assert is_thinking_model("gpt-5.1") is True
        assert is_thinking_model("gpt-5.1-codex") is True
        assert is_thinking_model("gpt-5.2") is True
        assert is_thinking_model("gpt-5-mini") is True

        # GPT-4.x - SDK has NO reasoning, pattern doesn't match
        assert is_thinking_model("gpt-4.1") is False

    def test_gemini_is_not_thinking(self) -> None:
        """Gemini models should NOT be detected as thinking."""
        assert is_thinking_model("gemini-3-pro-preview") is False

    # ═══════════════════════════════════════════════════════════════════════
    # Case Insensitivity
    # ═══════════════════════════════════════════════════════════════════════

    def test_case_insensitive(self) -> None:
        """Detection should be case-insensitive."""
        assert is_thinking_model("CLAUDE-OPUS-4.5") is True
        assert is_thinking_model("Claude-Opus-4.5") is True
        assert is_thinking_model("O1-Mini") is True


class TestKnownThinkingPatterns:
    """Tests for the KNOWN_THINKING_PATTERNS constant."""

    def test_patterns_are_lowercase(self) -> None:
        """All patterns should be lowercase for case-insensitive matching."""
        for pattern in KNOWN_THINKING_PATTERNS:
            assert pattern == pattern.lower(), f"Pattern '{pattern}' is not lowercase"

    def test_includes_opus(self) -> None:
        """Must include 'opus' for Claude Opus models."""
        assert "opus" in KNOWN_THINKING_PATTERNS

    def test_includes_gpt5(self) -> None:
        """Must include 'gpt-5' for GPT-5 reasoning models per live SDK."""
        assert "gpt-5" in KNOWN_THINKING_PATTERNS

    def test_includes_o1(self) -> None:
        """Must include 'o1' for OpenAI o1 models."""
        assert "o1" in KNOWN_THINKING_PATTERNS

    def test_includes_explicit_suffixes(self) -> None:
        """Must include explicit thinking/reasoning suffixes."""
        assert "-thinking" in KNOWN_THINKING_PATTERNS
        assert "-reasoning" in KNOWN_THINKING_PATTERNS


class TestHasVersionPeriod:
    """Tests for has_version_period() function."""

    def test_with_period(self) -> None:
        """Models with minor version have period."""
        assert has_version_period("claude-opus-4.5") is True
        assert has_version_period("gpt-5.1") is True
        assert has_version_period("gpt-5.1-codex") is True

    def test_without_period(self) -> None:
        """Models without minor version have no period."""
        assert has_version_period("gpt-5") is False
        assert has_version_period("claude-sonnet-4") is False
        assert has_version_period("gemini-3-pro-preview") is False


class TestUsesDashForVersion:
    """Tests for uses_dash_for_version() - anti-pattern detector.

    This function detects when someone writes "claude-opus-4-5" instead
    of "claude-opus-4.5". Verified: NO real Copilot SDK model uses this
    anti-pattern (checked live 2026-02-07).
    """

    # ── Live SDK model IDs (captured 2026-02-07) ──
    # Every single one must return False (none use dash-for-version).
    LIVE_SDK_MODEL_IDS = [
        "claude-sonnet-4.5",
        "claude-haiku-4.5",
        "claude-opus-4.6",
        "claude-opus-4.5",
        "claude-sonnet-4",
        "gemini-3-pro-preview",
        "gpt-5.2-codex",
        "gpt-5.2",
        "gpt-5.1-codex-max",
        "gpt-5.1-codex",
        "gpt-5.1",
        "gpt-5",
        "gpt-5.1-codex-mini",
        "gpt-5-mini",
        "gpt-4.1",
    ]

    def test_no_live_sdk_model_triggers(self) -> None:
        """Every real SDK model ID must return False."""
        for model_id in self.LIVE_SDK_MODEL_IDS:
            assert uses_dash_for_version(model_id) is False, (
                f"LIVE model '{model_id}' incorrectly flagged as dash-for-version"
            )

    def test_detects_wrong_dash_format(self) -> None:
        """Deliberately wrong IDs (dash instead of period) should be caught."""
        assert uses_dash_for_version("claude-opus-4-5") is True
        assert uses_dash_for_version("gpt-5-1") is True
        assert uses_dash_for_version("gpt-5-1-codex") is True

    def test_date_suffix_not_flagged(self) -> None:
        """Date suffixes like 2024-12-17 should NOT trigger false positive."""
        assert uses_dash_for_version("o1-2024-12-17") is False
        assert uses_dash_for_version("gpt-4-2024-01-15") is False
        assert uses_dash_for_version("model-2025-06") is False


class TestValidateModelIdFormat:
    """Tests for validate_model_id_format() function."""

    def test_valid_model_ids(self) -> None:
        """Valid model IDs should return empty list."""
        # All these are from live SDK
        assert validate_model_id_format("claude-opus-4.5") == []
        assert validate_model_id_format("claude-opus-4.6") == []
        assert validate_model_id_format("claude-sonnet-4") == []
        assert validate_model_id_format("gpt-5") == []
        assert validate_model_id_format("gpt-5.1") == []
        assert validate_model_id_format("gpt-5.1-codex") == []

    def test_invalid_dash_version(self) -> None:
        """Model with dash-based version should get warning."""
        warnings = validate_model_id_format("claude-opus-4-5")
        assert len(warnings) >= 1
        assert any("dash for version" in w for w in warnings)
        assert any("4-5" in w and "4.5" in w for w in warnings)


class TestTimeoutImplications:
    """
    Tests that verify the connection between model naming and timeout selection.

    This is CRITICAL documentation-as-tests showing why naming patterns matter.
    """

    def test_thinking_models_get_long_timeout(self) -> None:
        """
        Pattern-based detection is a FALLBACK for timeout selection.

        IMPORTANT DISTINCTION:
        - SDK capability check is authoritative (only opus-4.6 has 'thinking')
        - Pattern matching is fallback when SDK check fails
        - Pattern matching is INTENTIONALLY broad (all opus) for safety

        If SDK check fails but pattern matches, we use long timeout.
        This is safer than timing out a thinking model prematurely.
        """
        # These match the PATTERN (fallback detection)
        # SDK may report differently (e.g., opus-4.5 has no 'thinking' cap)
        pattern_thinking_models = [
            "claude-opus-4.5",  # Pattern matches, but SDK says no thinking
            "claude-opus-4.6",  # Pattern matches, SDK confirms thinking
            "o1-preview",
            "o1-mini",
            "o3-mini",
            "gpt-5-thinking",
        ]
        for model in pattern_thinking_models:
            assert is_thinking_model(model) is True, (
                f"CRITICAL: {model} not detected by pattern! "
                f"This affects fallback timeout selection."
            )

    def test_non_thinking_models_get_short_timeout(self) -> None:
        """
        Non-thinking models should NOT be detected as thinking.

        If is_thinking_model() returns True incorrectly:
        - Timeout will be 30 minutes unnecessarily
        - Users wait too long on errors
        - Server resources held longer than needed

        Based on live SDK 2026-02-06:
        - claude-sonnet, claude-haiku: NO thinking/reasoning
        - gpt-4.1: NO reasoning
        - gemini-3-pro-preview: NO thinking/reasoning
        """
        non_thinking_models = [
            "claude-sonnet-4.5",
            "claude-haiku-4.5",
            "gpt-4.1",  # Only GPT-4.x, not GPT-5.x
            "gemini-3-pro-preview",
        ]
        for model in non_thinking_models:
            assert is_thinking_model(model) is False, (
                f"ERROR: {model} incorrectly detected as thinking model! "
                f"This will cause unnecessarily long timeouts."
            )


class TestRealWorldScenarios:
    """
    Tests based on real-world model IDs from the SDK.

    These mirror the live integration tests but run without SDK access.
    The data is based on SDK output captured 2026-02-06.
    """

    # Live SDK model IDs (captured 2026-02-06)
    LIVE_CLAUDE_MODELS = [
        "claude-sonnet-4",
        "claude-sonnet-4.5",
        "claude-haiku-4.5",
        "claude-opus-4.5",
        "claude-opus-4.6",
    ]

    LIVE_GPT_MODELS = [
        "gpt-4.1",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5.1",
        "gpt-5.1-codex",
        "gpt-5.1-codex-max",
        "gpt-5.1-codex-mini",
        "gpt-5.2",
        "gpt-5.2-codex",
    ]

    LIVE_OTHER_MODELS = [
        "gemini-3-pro-preview",
    ]

    def test_all_live_claude_models_parseable(self) -> None:
        """All Claude models from live SDK should be parseable."""
        for model_id in self.LIVE_CLAUDE_MODELS:
            result = parse_model_id(model_id)
            assert result is not None, f"Failed to parse: {model_id}"
            assert result.family == "claude"

    def test_all_live_gpt_models_parseable(self) -> None:
        """All GPT models from live SDK should be parseable."""
        for model_id in self.LIVE_GPT_MODELS:
            result = parse_model_id(model_id)
            assert result is not None, f"Failed to parse: {model_id}"
            assert result.family == "gpt"

    def test_no_live_model_uses_dash_for_version(self) -> None:
        """No live SDK model should use dash for version numbers."""
        all_models = self.LIVE_CLAUDE_MODELS + self.LIVE_GPT_MODELS + self.LIVE_OTHER_MODELS
        for model_id in all_models:
            assert uses_dash_for_version(model_id) is False, (
                f"Model {model_id} uses dash for version - our code may be wrong!"
            )

    def test_only_opus_claude_models_are_thinking(self) -> None:
        """Pattern matcher detects Opus by name (fallback behavior).

        NOTE: This tests PATTERN detection, not SDK capability.
        - Pattern matcher: All 'opus' models → True (safety fallback)
        - SDK capability: Only opus-4.6 has 'thinking' (as of 2026-02-06)

        The pattern is intentionally broad to prevent timeout failures
        when SDK capability check fails.
        """
        for model_id in self.LIVE_CLAUDE_MODELS:
            # Pattern matches 'opus' substring
            expected_thinking = "opus" in model_id.lower()
            actual = is_thinking_model(model_id)
            assert actual == expected_thinking, (
                f"{model_id}: pattern expected={expected_thinking}, got {actual}"
            )
