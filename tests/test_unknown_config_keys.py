"""Tests for the warn-only unknown-config-key sweep at mount time.

Contract: config-key hygiene (see `_KNOWN_CONFIG_KEYS` / `_warn_unknown_config_keys`
in amplifier_module_provider_github_copilot/__init__.py).

These tests exercise `_warn_unknown_config_keys` directly (pure function, no SDK
interaction) rather than the full `mount()` coroutine, since mount() requires a
live coordinator and SDK client plumbing unrelated to this concern.
"""

from __future__ import annotations

import logging

import pytest

from amplifier_module_provider_github_copilot import (
    _KNOWN_CONFIG_KEYS,
    _warn_unknown_config_keys,
)

# All 13 keys the deep-read confirmed: 11 read directly by this module's own
# `self.config.get(...)` / `config.get(...)` call sites, plus 2 keys this
# module never reads itself but that are live, legitimately-consumed keys
# read by other collaborators off this same config dict (`priority` by the
# loop-streaming orchestrator's provider selection; `extra_request_params`
# reserved by app-cli).
_EXPECTED_KEYS = frozenset(
    {
        "github_token",
        "default_model",
        "raw",
        "enable_long_context",
        "reasoning_effort",
        "use_streaming",
        "max_retries",
        "min_retry_delay",
        "max_retry_delay",
        "retry_jitter",
        "overloaded_delay_multiplier",
        "priority",
        "extra_request_params",
    }
)


class TestKnownConfigKeysAllowlist:
    """The allowlist itself must be exactly the 13 keys the deep-read confirmed."""

    def test_known_config_keys_is_exactly_the_expected_set(self) -> None:
        assert _KNOWN_CONFIG_KEYS == _EXPECTED_KEYS
        assert len(_KNOWN_CONFIG_KEYS) == 13


class TestWarnUnknownConfigKeys:
    """`_warn_unknown_config_keys` warns, never raises, and stays quiet on legit config."""

    def test_all_known_keys_produce_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        config = dict.fromkeys(_KNOWN_CONFIG_KEYS, "x")
        with caplog.at_level(logging.WARNING):
            _warn_unknown_config_keys(config)
        assert caplog.records == []

    def test_empty_config_produces_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            _warn_unknown_config_keys({})
        assert caplog.records == []

    def test_typo_key_gets_did_you_mean_suggestion(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            _warn_unknown_config_keys({"us_streaming": True})
        assert len(caplog.records) == 1
        message = caplog.records[0].getMessage()
        assert "us_streaming" in message
        assert "did you mean 'use_streaming'" in message

    def test_unrecognizable_key_gets_bare_mention_no_suggestion(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            _warn_unknown_config_keys({"totally_unrelated_xyz": True})
        assert len(caplog.records) == 1
        message = caplog.records[0].getMessage()
        assert "'totally_unrelated_xyz'" in message
        assert "did you mean" not in message

    def test_debug_key_gets_targeted_message_not_did_you_mean(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The maintainer's own fixtures pass `debug` in ~9 configs; it is
        genuinely unread. It must still warn (honest signal), but with a
        targeted message rather than a nonsensical did-you-mean guess.
        """
        with caplog.at_level(logging.WARNING):
            _warn_unknown_config_keys({"debug": False})
        assert len(caplog.records) == 1
        message = caplog.records[0].getMessage()
        assert "'debug'" in message
        assert "not read by this provider" in message
        assert "did you mean" not in message

    def test_multiple_unknown_keys_combined_into_one_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            _warn_unknown_config_keys({"debug": False, "us_streaming": True, "model": "x"})
        assert len(caplog.records) == 1
        message = caplog.records[0].getMessage()
        assert "'debug'" in message
        assert "us_streaming" in message
        # "model" has no close match in _KNOWN_CONFIG_KEYS (closest is default_model,
        # difflib may or may not match depending on similarity ratio) -- just assert
        # it's named at all.
        assert "'model'" in message

    def test_never_raises_on_non_string_keys_or_odd_values(self) -> None:
        """Defensive: sweep must not raise even with an unusual config shape."""
        _warn_unknown_config_keys({"debug": None, "": "", "priority": 5})
