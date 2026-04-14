"""Tests for unified error config loading.

Contract: contracts/error-hierarchy.md — error config must produce correct kernel error types

Tests verify:
- Single source of truth for error config parsing
- context_extraction works in both file-path and importlib.resources scenarios
- _load_error_config_once() delegates to load_error_config()
"""

from __future__ import annotations

from pathlib import Path


class TestUnifiedErrorConfigLoading:
    """Unify Error Config Loading."""

    def test_load_error_config_from_file_path_includes_context_extraction(self) -> None:
        """Contract: error-hierarchy:Config:MUST:1

        File path loading must include context_extraction.

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

        assert config_path.exists(), f"Required config file missing: {config_path}"

        config = load_error_config(config_path)

        # Exact count from errors.yaml (17 mappings)
        assert len(config.mappings) == 17

        # At least one mapping must have non-empty context_extraction (parsing verified)
        mappings_with_ctx = [m for m in config.mappings if m.context_extraction]
        assert len(mappings_with_ctx) >= 1, (
            "No mappings have context_extraction — YAML parsing may have failed"
        )

        # Verify InvalidToolCallError mapping has expected extraction patterns
        invalid_tool_mappings = [
            m for m in config.mappings if m.kernel_error == "InvalidToolCallError"
        ]
        assert len(invalid_tool_mappings) == 1
        ctx = invalid_tool_mappings[0].context_extraction
        assert len(ctx) == 2
        assert ctx[0].field == "tool_name"
        assert ctx[0].pattern == "tool '([^']+)'"

    def test_client_load_error_config_includes_context_extraction(self) -> None:
        """error-hierarchy:config:MUST:2 — client loading must include context_extraction.

        When loading error config via _load_error_config_once() (which may use
        importlib.resources), context_extraction patterns MUST be included.
        """
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            _load_error_config_once,  # type: ignore[attr-defined]
        )

        config = _load_error_config_once()  # type: ignore[no-untyped-call]

        # Exact count from errors.yaml (17 mappings)
        assert len(config.mappings) == 17

        # Verify InvalidToolCallError mapping has expected extraction patterns
        invalid_tool_mappings = [
            m for m in config.mappings if m.kernel_error == "InvalidToolCallError"
        ]
        assert len(invalid_tool_mappings) == 1
        ctx = invalid_tool_mappings[0].context_extraction
        assert len(ctx) == 2
        assert ctx[0].field == "tool_name"
        assert ctx[0].pattern == "tool '([^']+)'"

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

        assert config_path.exists(), f"Required config file missing: {config_path}"

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
        """Contract: error-hierarchy:Config:MUST:4 — client config supports context extraction.

        Error translation using config loaded via _load_error_config_once()
        MUST correctly extract context from error messages when configured.
        """
        from amplifier_module_provider_github_copilot.error_translation import (
            InvalidToolCallError,
            translate_sdk_error,
        )
        from amplifier_module_provider_github_copilot.sdk_adapter.client import (
            _load_error_config_once,  # type: ignore[attr-defined]
        )

        config = _load_error_config_once()  # type: ignore[no-untyped-call]

        # Message matches string_pattern "tool conflict" AND context_extraction regex
        # Pattern: "tool '([^']+)'" captures tool_name from "tool 'my_tool'"
        test_error = RuntimeError("tool conflict: tool 'my_tool' conflicts with a built-in")

        translated = translate_sdk_error(test_error, config)

        assert isinstance(translated, InvalidToolCallError)
        assert translated.retryable is False
        # Context extraction appends [context: tool_name=my_tool, conflict_type=built-in]
        assert "tool_name=my_tool" in str(translated)
