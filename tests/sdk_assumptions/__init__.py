# =============================================================================
# SDK Behavioral Assumption Tests
# =============================================================================
#
# This package contains tests that validate our assumptions about GitHub
# Copilot CLI SDK internal behaviors. These tests are critical for detecting
# breaking changes when upgrading the SDK.
#
# See README.md in this directory for:
# - Full list of assumptions we depend on
# - Why each assumption is critical to our architecture
# - Upgrade workflow when tests fail
#
# =============================================================================
"""
SDK behavioral assumption tests.

These tests validate that the GitHub Copilot CLI SDK behaves as expected
for the "Deny + Destroy" external orchestration pattern to work correctly.

Test Categories:
    test_event_ordering.py  - Event sequence assumptions
    test_deny_hook.py       - preToolUse hook deny behavior
    test_session_lifecycle.py - Session destroy semantics

Run these tests when:
    1. Upgrading github-copilot-sdk version
    2. Investigating tool-calling regressions
    3. After SDK breaking change announcements
"""
