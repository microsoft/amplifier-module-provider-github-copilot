"""
Model ID Naming Conventions for GitHub Copilot SDK.

This module documents and implements the model ID naming patterns used by the
GitHub Copilot CLI SDK. Understanding these patterns is CRITICAL because:

1. **Timeout Selection**: Model ID parsing determines if a model supports
   extended thinking, which controls timeout (5 min vs 30 min).

2. **Capability Detection**: The `_model_supports_reasoning()` function uses
   pattern matching on model IDs as a fallback when API detection fails.

3. **Breaking Changes**: If SDK changes naming conventions, our timeout logic
   could break, causing premature timeouts for thinking models.

═══════════════════════════════════════════════════════════════════════════════
EVIDENCE-BASED NAMING PATTERNS (Captured 2026-02-06 from live SDK)
═══════════════════════════════════════════════════════════════════════════════

Live SDK returned these model IDs (grouped by family):

Claude Models:
  - claude-sonnet-4       # No minor version
  - claude-sonnet-4.5     # Period for minor version
  - claude-haiku-4.5      # Period for minor version
  - claude-opus-4.5       # Period for minor version (NO thinking per SDK)
  - claude-opus-4.6       # Period for minor version (HAS thinking per SDK)

GPT Models:
  - gpt-4.1               # Period for minor version
  - gpt-5                 # No minor version
  - gpt-5-mini            # Dash for variant, no minor version
  - gpt-5.1               # Period for minor version
  - gpt-5.1-codex         # Period for version, dash for variant
  - gpt-5.1-codex-max     # Multiple dash-separated variants
  - gpt-5.1-codex-mini    # Multiple dash-separated variants
  - gpt-5.2               # Period for minor version
  - gpt-5.2-codex         # Period for version, dash for variant

Other Models:
  - gemini-3-pro-preview  # Dashes only, no period (different scheme)

═══════════════════════════════════════════════════════════════════════════════
PATTERN RULES (Derived from Evidence)
═══════════════════════════════════════════════════════════════════════════════

1. PERIODS (.) = Version numbers
   - Format: `{major}.{minor}` (e.g., 4.5, 5.1, 5.2)
   - Examples: claude-opus-4.5, gpt-5.1, gpt-4.1

2. DASHES (-) = Component separators
   - Family separator: claude-opus, gpt-5
   - Variant suffix: gpt-5.1-codex, gpt-5-mini
   - Multiple variants: gpt-5.1-codex-max

3. OPTIONAL Minor Version
   - Some models omit minor version: gpt-5, claude-sonnet-4
   - These still use dashes for separators

4. EXCEPTIONS
   - gemini-3-pro-preview uses dashes for what might be a version (3)
   - Google may use different conventions

═══════════════════════════════════════════════════════════════════════════════
SIGNIFICANCE FOR TIMEOUT SELECTION
═══════════════════════════════════════════════════════════════════════════════

The timeout selection logic uses these patterns for fallback detection:

```python
is_known_thinking_model = any(
    pattern in model_lower
    for pattern in ("opus", "o1", "o3", "o4", "-thinking", "-reasoning")
)
```

If we use wrong model IDs (e.g., "claude-opus-4-5" instead of "claude-opus-4.5"):
1. API call would fail (model not found)
2. Capability detection would fail (no match in model list)
3. BUT fallback would still work (contains "opus")

However, if naming patterns change (e.g., SDK renames to "claude-o4-5"):
1. DEFAULT_MODEL would be wrong
2. Pattern "opus" wouldn't match "o4-5"
3. Timeout would be 5 min instead of 30 min
4. Extended thinking requests would timeout prematurely

This is why we have:
- Integration tests that verify SDK model IDs against expected patterns
- Snapshot tests that alert when expected models disappear
- Pattern validation tests that ensure no model uses dashes for versions

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import re
from typing import NamedTuple


class ModelIdPattern(NamedTuple):
    """Parsed components of a Copilot SDK model ID."""

    family: str  # e.g., "claude", "gpt", "gemini"
    variant: str  # e.g., "opus", "sonnet", "" for gpt-5
    major_version: int | None  # e.g., 4, 5
    minor_version: int | None  # e.g., 5, 1, None if not present
    suffix: str  # e.g., "codex", "mini", "codex-max"


# Known thinking model patterns (for FALLBACK detection only)
# These patterns are matched against lowercase model IDs
#
# IMPORTANT DISTINCTION:
# - SDK capability detection is AUTHORITATIVE (check model.capabilities)
# - Pattern matching is FALLBACK when SDK detection fails (network error, etc.)
#
# As of 2026-02-06, SDK reports 'thinking' or 'reasoning' capability for:
# - claude-opus-4.6: YES (has 'thinking' in capabilities)
# - claude-opus-4.5: NO (no 'thinking' in capabilities)
# - gpt-5, gpt-5.1, gpt-5.2, etc.: YES (have 'reasoning' in capabilities)
# - gpt-4.1: NO (no 'reasoning')
#
# Patterns include "opus" and "gpt-5" because:
# 1. Future versions should inherit capability
# 2. If SDK check fails, safer to use long timeout than short
# 3. Pattern is last-resort fallback, not primary detection
KNOWN_THINKING_PATTERNS: frozenset[str] = frozenset(
    {
        "opus",  # Claude Opus (safety: catch future versions)
        "gpt-5",  # GPT-5.x models have 'reasoning' per SDK
        "o1",  # OpenAI o1 reasoning models
        "o3",  # OpenAI o3 reasoning models
        "o4",  # OpenAI o4 reasoning models (future-proofing)
        "-thinking",  # Explicit thinking suffix
        "-reasoning",  # Explicit reasoning suffix
    }
)

# Regex for parsing Copilot SDK model IDs
# Pattern: {family}-{variant}-{major}.{minor}[-suffix]
# Examples: claude-opus-4.5, gpt-5.1-codex, gpt-5, gemini-3-pro-preview
_MODEL_ID_PATTERN = re.compile(
    r"^"
    r"(?P<family>[a-z]+\d*)"  # Family: claude, gpt, gemini, o1, o3
    r"(?:-(?P<variant>[a-z]+))?"  # Optional variant: opus, sonnet, mini
    r"(?:-(?P<major>\d+))?"  # Optional major version: 4, 5
    r"(?:\.(?P<minor>\d+))?"  # Optional minor version: .5, .1
    r"(?:-(?P<suffix>[a-z0-9-]+))?"  # Optional suffix: codex, mini
    r"$",
    re.IGNORECASE,
)


def parse_model_id(model_id: str) -> ModelIdPattern | None:
    """
    Parse a Copilot SDK model ID into its components.

    Args:
        model_id: The model ID to parse (e.g., "claude-opus-4.5")

    Returns:
        ModelIdPattern with parsed components, or None if parsing fails

    Examples:
        >>> parse_model_id("claude-opus-4.5")
        ModelIdPattern(family='claude', variant='opus', major_version=4,
                       minor_version=5, suffix='')

        >>> parse_model_id("gpt-5.1-codex")
        ModelIdPattern(family='gpt', variant='', major_version=5,
                       minor_version=1, suffix='codex')

        >>> parse_model_id("gpt-5-mini")
        ModelIdPattern(family='gpt', variant='', major_version=5,
                       minor_version=None, suffix='mini')
    """
    match = _MODEL_ID_PATTERN.match(model_id)
    if not match:
        return None

    groups = match.groupdict()
    return ModelIdPattern(
        family=groups.get("family", "").lower(),
        variant=groups.get("variant", "") or "",
        major_version=int(groups["major"]) if groups.get("major") else None,
        minor_version=int(groups["minor"]) if groups.get("minor") else None,
        suffix=groups.get("suffix", "") or "",
    )


def is_thinking_model(model_id: str) -> bool:
    """
    FALLBACK check if a model ID suggests it's a thinking/reasoning model.

    This is a FALLBACK check using pattern matching, used when SDK
    capability detection fails (network error, model not in list, etc.).

    IMPORTANT: This is NOT the authoritative source for thinking capability!
    - AUTHORITATIVE: SDK model.capabilities contains 'thinking' or 'reasoning'
    - FALLBACK: This function, using name pattern matching

    The patterns are INTENTIONALLY BROAD for safety:
    - "opus" matches ALL opus models (even opus-4.5 which SDK says has no thinking)
    - This is safer than missing a thinking model and timing out at 5 min

    As of 2026-02-06, SDK reports thinking capability ONLY for:
    - claude-opus-4.6: YES (has 'thinking' in capabilities)
    - claude-opus-4.5: NO (no 'thinking' in capabilities)

    But this function returns True for BOTH because it's a safety fallback.

    Args:
        model_id: The model ID to check

    Returns:
        True if the model ID matches known thinking patterns (fallback)

    Examples:
        >>> is_thinking_model("claude-opus-4.5")
        True  # Contains "opus"

        >>> is_thinking_model("gpt-o1-preview")
        True  # Contains "o1"

        >>> is_thinking_model("claude-sonnet-4.5")
        False  # No thinking pattern
    """
    model_lower = model_id.lower()
    return any(pattern in model_lower for pattern in KNOWN_THINKING_PATTERNS)


def has_version_period(model_id: str) -> bool:
    """
    Check if a model ID uses a period for version number.

    Per Copilot SDK convention:
    - Periods indicate version numbers (e.g., 4.5, 5.1)
    - Dashes are component separators (e.g., claude-opus, gpt-5-codex)

    Args:
        model_id: The model ID to check

    Returns:
        True if the model ID contains a period (likely has minor version)

    Examples:
        >>> has_version_period("claude-opus-4.5")
        True

        >>> has_version_period("gpt-5")
        False

        >>> has_version_period("gemini-3-pro-preview")
        False
    """
    return "." in model_id


def uses_dash_for_version(model_id: str) -> bool:
    """
    Check if a model ID INCORRECTLY uses a dash for version number.

    This is an anti-pattern detector. Copilot SDK uses:
    - PERIODS for versions: claude-opus-4.5, gpt-5.1
    - DASHES for separators: claude-opus, gpt-5-codex

    A model ID like "claude-opus-4-5" would be WRONG (should be 4.5).

    Logic (no regex — readable):
    Split by dash. If two consecutive segments are both short numbers
    (1-2 digits), that looks like "major-minor" instead of "major.minor".
    Exclude date-suffix patterns where a 4-digit year precedes the pair.

    Verified against live SDK output (2026-02-07): No real Copilot model
    uses this anti-pattern. All versions use periods.

    Args:
        model_id: The model ID to check

    Returns:
        True if the model ID appears to use dashes for version numbers

    Examples:
        >>> uses_dash_for_version("claude-opus-4-5")
        True  # WRONG: should be claude-opus-4.5

        >>> uses_dash_for_version("claude-opus-4.5")
        False  # CORRECT

        >>> uses_dash_for_version("gpt-5-codex")
        False  # CORRECT: dash is for variant "codex"

        >>> uses_dash_for_version("o1-2024-12-17")
        False  # Date suffix, not a version separator
    """
    parts = model_id.split("-")
    for i in range(len(parts) - 1):
        a, b = parts[i], parts[i + 1]
        # Both segments must be short digit-only strings (1-2 digits)
        if not (a.isdigit() and b.isdigit() and len(a) <= 2 and len(b) <= 2):
            continue
        # Skip date patterns: preceded by a 4-digit year segment
        if i > 0 and parts[i - 1].isdigit() and len(parts[i - 1]) == 4:
            continue
        return True
    return False


def _find_dash_version_pair(model_id: str) -> tuple[str, str] | None:
    """Find the first digit-digit pair that looks like a wrong version separator.

    Returns (wrong_part, correct_part) e.g. ("4-5", "4.5"), or None.
    """
    parts = model_id.split("-")
    for i in range(len(parts) - 1):
        a, b = parts[i], parts[i + 1]
        if not (a.isdigit() and b.isdigit() and len(a) <= 2 and len(b) <= 2):
            continue
        if i > 0 and parts[i - 1].isdigit() and len(parts[i - 1]) == 4:
            continue
        return f"{a}-{b}", f"{a}.{b}"
    return None


def validate_model_id_format(model_id: str) -> list[str]:
    """
    Validate a model ID against Copilot SDK naming conventions.

    Returns a list of validation warnings/errors. Empty list means valid.

    Args:
        model_id: The model ID to validate

    Returns:
        List of validation messages (empty if valid)

    Examples:
        >>> validate_model_id_format("claude-opus-4.5")
        []  # Valid

        >>> validate_model_id_format("claude-opus-4-5")
        ["Model ID uses dash for version (4-5); should use period (4.5)"]
    """
    warnings = []

    # Check for dash-based version numbers
    pair = _find_dash_version_pair(model_id)
    if pair is not None:
        wrong, correct = pair
        warnings.append(f"Model ID uses dash for version ({wrong}); should use period ({correct})")

    # Check if parseable
    parsed = parse_model_id(model_id)
    if parsed is None:
        warnings.append(f"Model ID '{model_id}' does not match expected pattern")

    return warnings
