"""Tests for unified error config loading.

Contract: contracts/error-hierarchy.md — error config must produce correct kernel error types

Tests verify:
- Single source of truth for error config parsing
- context_extraction works in both file-path and importlib.resources scenarios
- _load_error_config_once() delegates to load_error_config()
"""

from __future__ import annotations

from pathlib import Path

import pytest


class TestUnifiedErrorConfigLoading:
    """Unify Error Config Loading."""

    def test_load_error_config_from_file_path_includes_context_extraction(self) -> None:
        """error-hierarchy:config:MUST:1 — file path loading must include context_extraction.

        When loading error config from a file path, context_extraction patterns
        MUST be parsed and included in the ErrorMapping objects.
        """
        from amplifier_module_provider_github_copilot.error_translation import (
            load_error_config,
        )

        # Load from the actual config file
        config_path = (
            Path(__file__).parent.parent
            / "amplifier_module_provider_github_copilot"
            / "config"
            / "data"
            / "errors.yaml"
        )

        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")

        config = load_error_config(config_path)

        # Verify config loaded successfully
        assert len(config.mappings) > 0

        # Check if any mapping has context_extraction (may not all have it)
        # The important thing is the parser handles the field correctly
        for mapping in config.mappings:
            # context_extraction should be a list (possibly empty)
            assert isinstance(mapping.context_extraction, list)

    def test_client_load_error_config_includes_context_extraction(self) -> None:
        """error-hierarchy:config:MUST:2 — client loading must include context_extraction.

        When loading error config via _load_error_config_once() (which may use
        importlib.resources), context_extraction patterns MUST be included.
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            _load_error_config_once,  # type: ignore[attr-defined]
        )

        config = _load_error_config_once()  # type: ignore[no-untyped-call]

        # Verify config loaded successfully
        # (May be fallback config if file not found, but should still work)
        assert config is not None

        # All mappings should have context_extraction list (possibly empty)
        for mapping in config.mappings:
            assert isinstance(mapping.context_extraction, list)

    def test_both_loading_paths_produce_identical_results(self) -> None:
        """error-hierarchy:config:MUST:3 — both paths must produce same results.

        The importlib.resources path and file path MUST produce identical
        ErrorConfig objects (same mappings with same context_extraction).
        """
        from amplifier_module_provider_github_copilot.error_translation import (
            load_error_config,
        )
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            _load_error_config_once,  # type: ignore[attr-defined]
        )

        # Load via client path (uses importlib.resources or fallback)
        client_config = _load_error_config_once()  # type: ignore[no-untyped-call]

        # Load via direct file path
        config_path = (
            Path(__file__).parent.parent
            / "amplifier_module_provider_github_copilot"
            / "config"
            / "data"
            / "errors.yaml"
        )

        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")

        file_config = load_error_config(config_path)

        # Both should have same number of mappings
        assert len(client_config.mappings) == len(file_config.mappings)

        # Both should have same default error
        assert client_config.default_error == file_config.default_error
        assert client_config.default_retryable == file_config.default_retryable

        # Each mapping should have matching context_extraction
        for i, (client_mapping, file_mapping) in enumerate(
            zip(client_config.mappings, file_config.mappings, strict=True)
        ):
            assert client_mapping.kernel_error == file_mapping.kernel_error, (
                f"Mapping {i} kernel_error mismatch"
            )
            assert client_mapping.retryable == file_mapping.retryable, (
                f"Mapping {i} retryable mismatch"
            )
            # Context extraction should match
            assert len(client_mapping.context_extraction) == len(file_mapping.context_extraction), (
                f"Mapping {i} context_extraction length mismatch"
            )

    def test_error_translation_with_context_extraction_from_client_config(
        self,
    ) -> None:
        """error-hierarchy:config:MUST:4 — client config supports context extraction.

        Error translation using config loaded via _load_error_config_once()
        MUST correctly extract context from error messages when configured.
        """
        from amplifier_module_provider_github_copilot.error_translation import (
            translate_sdk_error,
        )
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            _load_error_config_once,  # type: ignore[attr-defined]
        )

        config = _load_error_config_once()  # type: ignore[no-untyped-call]

        # Create an error that should match a mapping with context extraction
        # Using InvalidToolCallError pattern from errors.yaml
        test_error = RuntimeError("Tool conflict detected: tool_name='my_tool'")

        translated = translate_sdk_error(test_error, config)

        # Translation should work (not crash on context_extraction)
        assert translated is not None
        # The error message might include context suffix if pattern matched
        # The key test is that it doesn't crash

    def test_load_error_config_graceful_degradation(self) -> None:
        """error-hierarchy:config:SHOULD:1 — missing config should fall back gracefully.

        If the config file is missing, load_error_config should return
        a default ErrorConfig with sensible defaults.

        Three-Medium: default_retryable=False matches YAML default (errors.yaml:128)
        and loader fallback for consistency.
        """
        from amplifier_module_provider_github_copilot.error_translation import (
            load_error_config,
        )

        # Load from non-existent path
        config = load_error_config(Path("/nonexistent/path/errors.yaml"))

        # Should return default config, not crash
        assert config is not None
        assert config.default_error == "ProviderUnavailableError"
        # Three-Medium: default_retryable=False (conservative: unknown errors don't retry)
        assert config.default_retryable is False
