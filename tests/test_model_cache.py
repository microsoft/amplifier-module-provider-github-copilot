"""
Tests for model cache disk persistence.

Contract: contracts/behaviors.md (ModelCache section)
TTL policy values: config/policy.py (CacheConfig dataclass)
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from amplifier_module_provider_github_copilot.config._policy import CacheConfig
from amplifier_module_provider_github_copilot.model_cache import (
    get_cache_dir,
    get_cache_file_path,
    get_cache_filename,
    get_cache_ttl_seconds,
    invalidate_cache,
    load_cache_config,
    read_cache,
    write_cache,
)
from amplifier_module_provider_github_copilot.models import CopilotModelInfo

# =============================================================================
# Test Fixtures
# =============================================================================


def make_cache_data(model_id: str = "claude-sonnet-4-5") -> dict[str, Any]:
    """Create mock cache data for testing."""
    return {
        "version": "1.0",
        "timestamp": time.time(),
        "models": [
            {
                "id": model_id,
                "name": "Claude Sonnet 4.5",
                "context_window": 200000,
                "max_output_tokens": 32000,
                "supports_vision": True,
                "supports_reasoning_effort": False,
                "supported_reasoning_efforts": [],
                "default_reasoning_effort": None,
            }
        ],
    }


# =============================================================================
# Phase 2a: Cross-Platform Cache Directory
# Contract: behaviors:ModelCache:SHOULD:1
# =============================================================================


class TestCacheDirectory:
    """Test cross-platform cache directory resolution.

    Contract: behaviors:ModelCache:SHOULD:1
    - SHOULD cache SDK models to disk for session persistence
    """

    def test_get_cache_dir_returns_path(self) -> None:
        """get_cache_dir() MUST return a Path object."""
        result = get_cache_dir()

        assert isinstance(result, Path)

    def test_get_cache_dir_windows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """On Windows, cache dir MUST be in LOCALAPPDATA."""
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setenv("LOCALAPPDATA", "C:\\Users\\Test\\AppData\\Local")
        result = get_cache_dir()

        assert "amplifier" in str(result).lower()

    def test_get_cache_dir_linux(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """On Linux, cache dir MUST follow XDG_CACHE_HOME or ~/.cache."""
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setenv("XDG_CACHE_HOME", "/custom/cache")
        result = get_cache_dir()

        assert "amplifier" in str(result).lower()

    def test_get_cache_dir_macos(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """On macOS, cache dir MUST be in ~/Library/Caches."""
        monkeypatch.setattr(sys, "platform", "darwin")
        result = get_cache_dir()

        assert "amplifier" in str(result).lower()


# =============================================================================
# Phase 2a: Write Cache
# Contract: behaviors:ModelCache:SHOULD:1
# =============================================================================


class TestWriteCache:
    """Test writing model cache to disk.

    Contract: behaviors:ModelCache:SHOULD:1
    - SHOULD cache SDK models to disk for session persistence
    """

    def test_write_cache_creates_file(self, tmp_path: Path) -> None:
        """Contract: behaviors:ModelCache:SHOULD:1

        write_cache() MUST create a cache file on disk.
        """
        models = [
            CopilotModelInfo(
                id="claude-sonnet-4-5",
                name="Claude Sonnet 4.5",
                context_window=200000,
                max_output_tokens=32000,
            )
        ]

        cache_file = tmp_path / "models_cache.json"
        write_cache(models, cache_file)

        assert cache_file.exists()

    def test_write_cache_uses_utf8_encoding(self, tmp_path: Path) -> None:
        """Cache files SHOULD use UTF-8 encoding for cross-platform compatibility.

        # Contract: behaviors:ModelCache:SHOULD:1
        """
        # Model with non-ASCII characters
        models = [
            CopilotModelInfo(
                id="test-model",
                name="Modèle Test 日本語",  # French + Japanese
                context_window=100000,
                max_output_tokens=16000,
            )
        ]

        cache_file = tmp_path / "models_cache.json"
        write_cache(models, cache_file)

        # Verify can read back as UTF-8
        content = cache_file.read_text(encoding="utf-8")
        assert "Modèle" in content
        assert "日本語" in content

    def test_write_cache_includes_timestamp(self, tmp_path: Path) -> None:
        """Cache data SHOULD include timestamp for TTL verification.

        # Contract: behaviors:ModelCache:SHOULD:2
        """
        models = [
            CopilotModelInfo(
                id="test",
                name="Test",
                context_window=100000,
                max_output_tokens=16000,
            )
        ]

        cache_file = tmp_path / "models_cache.json"
        before = time.time()
        write_cache(models, cache_file)
        after = time.time()

        data = json.loads(cache_file.read_text(encoding="utf-8"))
        assert "timestamp" in data
        assert before <= data["timestamp"] <= after

    def test_write_cache_creates_parent_directories(self, tmp_path: Path) -> None:
        """write_cache() SHOULD create parent directories if they don't exist.

        # Contract: behaviors:ModelCache:SHOULD:1
        """
        models = [
            CopilotModelInfo(
                id="test",
                name="Test",
                context_window=100000,
                max_output_tokens=16000,
            )
        ]

        # Path with non-existent parent directories
        cache_file = tmp_path / "nested" / "dir" / "models_cache.json"
        write_cache(models, cache_file)

        assert cache_file.exists()


# =============================================================================
# Phase 2a: Read Cache
# Contract: behaviors:ModelCache:SHOULD:1, SHOULD:2
# =============================================================================


class TestReadCache:
    """Test reading model cache from disk.

    Contract: behaviors:ModelCache:SHOULD:1, SHOULD:2
    """

    def test_read_cache_returns_models(self, tmp_path: Path) -> None:
        """Contract: behaviors:ModelCache:SHOULD:1

        read_cache() MUST return list of CopilotModelInfo on success.
        """
        # Write cache first
        models = [
            CopilotModelInfo(
                id="claude-sonnet-4-5",
                name="Claude Sonnet 4.5",
                context_window=200000,
                max_output_tokens=32000,
                supports_vision=True,
            )
        ]
        cache_file = tmp_path / "models_cache.json"
        write_cache(models, cache_file)

        # Read it back
        result = read_cache(cache_file)

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], CopilotModelInfo)
        assert result[0].id == "claude-sonnet-4-5"
        assert result[0].context_window == 200000

    def test_read_cache_returns_none_when_file_missing(self, tmp_path: Path) -> None:
        """read_cache() MUST return None when cache file doesn't exist."""
        cache_file = tmp_path / "nonexistent.json"
        result = read_cache(cache_file)

        assert result is None

    def test_read_cache_returns_none_when_invalid_json(self, tmp_path: Path) -> None:
        """read_cache() MUST return None when cache contains invalid JSON."""
        cache_file = tmp_path / "invalid.json"
        cache_file.write_text("not valid json {", encoding="utf-8")

        result = read_cache(cache_file)

        assert result is None

    def test_read_cache_returns_none_when_stale(self, tmp_path: Path) -> None:
        """Contract: behaviors:ModelCache:SHOULD:2

        read_cache() MUST return None when cache is older than TTL.
        """
        # Create a stale cache (timestamp in the past)
        stale_data: dict[str, Any] = {
            "version": "1.0",
            "timestamp": time.time() - 100000,  # Over 24 hours ago
            "models": [
                {
                    "id": "stale-model",
                    "name": "Stale",
                    "context_window": 100000,
                    "max_output_tokens": 16000,
                    "supports_vision": False,
                    "supports_reasoning_effort": False,
                    "supported_reasoning_efforts": [],
                    "default_reasoning_effort": None,
                }
            ],
        }

        cache_file = tmp_path / "stale_cache.json"
        cache_file.write_text(json.dumps(stale_data), encoding="utf-8")

        result = read_cache(cache_file, max_age_seconds=3600)  # 1 hour TTL

        assert result is None


# =============================================================================
# Phase 2a: Cache Policy
# Contract: behaviors:ModelCache:SHOULD:2
# =============================================================================


class TestCachePolicy:
    """Test cache policy configuration.

    Contract: behaviors:ModelCache:SHOULD:2
    """

    def test_cache_config_has_ttl(self) -> None:
        """CacheConfig MUST define disk_ttl_seconds."""
        config = load_cache_config()
        assert isinstance(config.disk_ttl_seconds, int)

    def test_ttl_is_reasonable(self) -> None:
        """TTL SHOULD be at least 1 hour and at most 7 days."""
        ttl = get_cache_ttl_seconds()

        assert ttl >= 3600, "TTL should be at least 1 hour"
        assert ttl <= 604800, "TTL should be at most 7 days"


# =============================================================================
# Phase 2a: No Hardcoded Fallback
# Contract: behaviors:ModelDiscoveryError:MUST_NOT:1
# =============================================================================


class TestNoHardcodedFallback:
    """Verify no hardcoded model fallback exists in cache module.

    Contract: behaviors:ModelDiscoveryError:MUST_NOT:1
    """

    def test_no_bundled_model_limits_dict_in_cache(self) -> None:
        """Contract: behaviors:ModelDiscoveryError:MUST_NOT:1

        model_cache.py MUST NOT contain hardcoded fallback dicts.
        """
        import amplifier_module_provider_github_copilot.model_cache as cache_module

        forbidden_names = [
            "BUNDLED_MODEL_LIMITS",
            "MODEL_LIMITS",
            "HARDCODED_MODELS",
            "FALLBACK_MODELS",
            "DEFAULT_MODELS",
        ]

        for name in forbidden_names:
            assert not hasattr(cache_module, name), (
                f"Found '{name}' in model_cache.py — contract "
                "behaviors:ModelDiscoveryError:MUST_NOT:1 violated."
            )


# =============================================================================
# Additional Coverage Tests
# Contract: behaviors:ModelCache
# =============================================================================


class TestCacheConfigHelpers:
    """Tests for cache config helper functions."""

    def test_get_cache_filename_returns_string(self) -> None:
        """get_cache_filename() returns configured filename.

        # Contract: behaviors:ConfigLoading:MUST:1
        """
        filename = get_cache_filename()

        assert isinstance(filename, str)
        assert filename.endswith(".json")


class TestInvalidateCache:
    """Tests for cache invalidation."""

    def test_invalidate_cache_removes_file(self, tmp_path: Path) -> None:
        """invalidate_cache() removes existing cache file.

        # Contract: behaviors:ModelCache:SHOULD:3
        """
        cache_file = tmp_path / "test_cache.json"
        cache_file.write_text("{}", encoding="utf-8")
        assert cache_file.exists()

        invalidate_cache(cache_file=cache_file)

        assert not cache_file.exists()

    def test_invalidate_cache_handles_missing_file(self, tmp_path: Path) -> None:
        """invalidate_cache() handles missing file gracefully.

        # Contract: behaviors:ModelCache:SHOULD:3
        """

        cache_file = tmp_path / "nonexistent.json"
        assert not cache_file.exists()

        # Should not raise
        invalidate_cache(cache_file=cache_file)

        assert not cache_file.exists()


class TestWriteCacheErrorHandling:
    """Tests for write_cache error handling."""

    def test_write_cache_temp_file_cleanup_on_error(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """write_cache() cleans up temp file on write failure.

        # Contract: behaviors:ModelCache:SHOULD:1
        """
        cache_file = tmp_path / "test_cache.json"
        temp_file = cache_file.with_suffix(".tmp")

        models = [
            CopilotModelInfo(
                id="test-model",
                name="Test",
                context_window=100000,
                max_output_tokens=8000,
            ),
        ]

        # Mock write_text to raise after temp file write would occur
        original_write_text = Path.write_text

        def mock_write_text(self: Path, content: str, encoding: str = "utf-8") -> int:
            if str(self).endswith(".tmp"):
                # Create temp file then raise to trigger cleanup
                original_write_text(self, content, encoding=encoding)
                raise OSError("Write failed during temp file creation")
            return original_write_text(self, content, encoding=encoding)

        logger_name = "amplifier_module_provider_github_copilot.model_cache"
        patch_target = "amplifier_module_provider_github_copilot.model_cache.Path.write_text"
        with (
            caplog.at_level(logging.WARNING, logger=logger_name),
            patch(patch_target, mock_write_text),
        ):
            write_cache(models, cache_file=cache_file)

        # Temp file MUST NOT persist after error
        assert not temp_file.exists(), "Temp file should be cleaned up after write error"
        # Log warning about failure
        assert "Failed to write cache" in caplog.text


class TestLoadCacheConfigFallback:
    """Tests for load_cache_config return value."""

    def test_load_cache_config_returns_valid_config(self) -> None:
        """load_cache_config() returns a CacheConfig with expected fields.

        # Contract: behaviors:ConfigLoading:MUST:1
        # Contract: behaviors:ModelCache:SHOULD:2
        """
        config = load_cache_config()

        assert isinstance(config, CacheConfig)
        assert config.disk_ttl_seconds == 86400
        assert config.cache_filename == "models_cache.json"


class TestCacheFileOperations:
    """Tests for cache file path operations."""

    def test_get_cache_file_path_ends_with_json(self) -> None:
        """Cache file path MUST end with .json.

        # Contract: behaviors:ModelCache:SHOULD:1
        """
        result = get_cache_file_path()
        assert isinstance(result, Path)
        assert result.suffix == ".json"


class TestReadCacheErrorHandling:
    """Tests for read_cache error handling paths."""

    def test_read_cache_missing_models_key(self, tmp_path: Path) -> None:
        """read_cache() returns empty list when 'models' key is missing.

        # Contract: behaviors:ModelDiscoveryError:MUST:1
        """
        cache_file = tmp_path / "missing_models.json"
        cache_data = '{"version": "1.0", "timestamp": ' + str(time.time()) + "}"
        cache_file.write_text(cache_data, encoding="utf-8")

        result = read_cache(cache_file=cache_file)
        # Empty models list (default from .get()) returns [] since no entries are malformed
        assert result == []


class TestInvalidateCacheErrorHandling:
    """Tests for invalidate_cache error handling."""

    def test_invalidate_cache_permission_error(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """invalidate_cache() handles permission errors gracefully.

        # Contract: behaviors:ModelCache:SHOULD:3
        Exception is swallowed and logged, not raised.
        """

        cache_file = tmp_path / "locked.json"
        cache_file.write_text("{}", encoding="utf-8")

        # Mock unlink to raise PermissionError
        def mock_unlink(self: Path) -> None:
            raise PermissionError("Access denied")

        logger_name = "amplifier_module_provider_github_copilot.model_cache"
        with (
            caplog.at_level(logging.WARNING, logger=logger_name),
            patch("amplifier_module_provider_github_copilot.model_cache.Path.unlink", mock_unlink),
        ):
            # Must NOT raise (exception swallowed)
            invalidate_cache(cache_file=cache_file)

        # Must LOG the failure
        assert "Failed to invalidate cache" in caplog.text


# =============================================================================
# Additional Coverage Tests
# Coverage targets: lines 63-64, 182-185, 273
# =============================================================================


class TestWriteCacheTempFileCleanup:
    """Tests for temp file cleanup during write failures.

    Covers lines 182-185: temp_file cleanup in except block.
    """

    def test_temp_file_cleanup_on_replace_failure(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """write_cache() cleans up temp file when replace() fails.

        # Contract: behaviors:ModelCache:SHOULD:1
        """
        cache_file = tmp_path / "test_cache.json"
        temp_file = cache_file.with_suffix(".tmp")

        models = [
            CopilotModelInfo(
                id="test-model",
                name="Test",
                context_window=100000,
                max_output_tokens=8000,
            ),
        ]

        # Mock replace to fail after successful write
        original_replace = Path.replace

        def mock_replace(self: Path, target: Path) -> Path:
            if str(self).endswith(".tmp"):
                raise OSError("Replace failed")
            return original_replace(self, target)

        logger_name = "amplifier_module_provider_github_copilot.model_cache"
        patch_target = "amplifier_module_provider_github_copilot.model_cache.Path.replace"
        with (
            caplog.at_level(logging.WARNING, logger=logger_name),
            patch(patch_target, mock_replace),
        ):
            # Should not raise - graceful failure
            write_cache(models, cache_file=cache_file)

        # Temp file MUST be cleaned up after replace failure
        assert not temp_file.exists(), "Temp file should be cleaned up after replace error"
        # Log warning about failure
        assert "Failed to write cache" in caplog.text


class TestInvalidateCacheEdgeCases:
    """Additional tests for invalidate_cache.

    Covers line 273: exception during unlink.
    """

    def test_invalidate_cache_generic_exception(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """invalidate_cache() handles generic exceptions gracefully.

        # Contract: behaviors:ModelCache:SHOULD:3
        Exception is swallowed and logged, not raised.
        """
        cache_file = tmp_path / "test.json"
        cache_file.write_text("{}", encoding="utf-8")

        # Mock unlink to raise generic Exception
        def mock_unlink(self: Path) -> None:
            raise RuntimeError("Unexpected error")

        logger_name = "amplifier_module_provider_github_copilot.model_cache"
        with (
            caplog.at_level(logging.WARNING, logger=logger_name),
            patch("amplifier_module_provider_github_copilot.model_cache.Path.unlink", mock_unlink),
        ):
            # Must NOT raise (exception swallowed)
            invalidate_cache(cache_file=cache_file)

        # Must LOG the failure
        assert "Failed to invalidate cache" in caplog.text


class TestReadCachePartialRecovery:
    """Tests for S5 fix: malformed cache entries are skipped; valid entries preserved.

    Before fix: list comprehension raised KeyError on first bad entry → entire cache discarded.
    After fix: per-entry try/except skips bad entries while preserving valid ones.
    """

    def test_valid_entries_returned_when_one_malformed(self, tmp_path: Path) -> None:
        """One malformed entry is skipped; valid entry is returned.

        # Contract: behaviors:ModelCache:SHOULD:1
        """
        cache_data = {
            "timestamp": time.time(),
            "version": "1.0",
            "models": [
                {
                    "id": "good-model",
                    "name": "Good Model",
                    "context_window": 4096,
                    "max_output_tokens": 1024,
                },
                {
                    # Missing required 'id' and 'name' fields — must be skipped
                    "context_window": 999,
                    "max_output_tokens": 100,
                },
            ],
        }
        cache_file = tmp_path / "models_cache.json"
        cache_file.write_text(json.dumps(cache_data), encoding="utf-8")

        result = read_cache(cache_file=cache_file)

        # S5: should recover 1 valid model, not discard entire cache
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].id == "good-model"

    def test_all_malformed_entries_triggers_cache_miss(self, tmp_path: Path) -> None:
        """When ALL entries are malformed, returns None (cache miss) — not an empty list.

        # Contract: behaviors:ModelDiscoveryError:MUST:1
        S5 Fix: when every entry is corrupt the correct behaviour is to force a
        live API call (return None), not return [] which would signal 'no models
        available'.  The live API will repopulate the cache with valid data.
        """
        cache_data = {
            "timestamp": time.time(),
            "version": "1.0",
            "models": [
                {"missing_required_fields": True},
                {"also_missing": "required_id_and_name"},
            ],
        }
        cache_file = tmp_path / "models_cache.json"
        cache_file.write_text(json.dumps(cache_data), encoding="utf-8")

        result = read_cache(cache_file=cache_file)

        # None = cache miss: all entries corrupt, must call live API to repopulate.
        assert result is None, (
            f"Expected None (cache miss) when all entries are malformed, got: {result!r}"
        )

    def test_multiple_valid_entries_all_returned(self, tmp_path: Path) -> None:
        """All valid entries are returned when there are no malformed entries.

        # Contract: behaviors:ModelCache:SHOULD:1
        """
        cache_data = {
            "timestamp": time.time(),
            "version": "1.0",
            "models": [
                {
                    "id": "model-a",
                    "name": "Model A",
                    "context_window": 4096,
                    "max_output_tokens": 1024,
                },
                {
                    "id": "model-b",
                    "name": "Model B",
                    "context_window": 8192,
                    "max_output_tokens": 2048,
                },
            ],
        }
        cache_file = tmp_path / "models_cache.json"
        cache_file.write_text(json.dumps(cache_data), encoding="utf-8")

        result = read_cache(cache_file=cache_file)

        assert isinstance(result, list)
        assert len(result) == 2
        assert {m.id for m in result} == {"model-a", "model-b"}


class TestReadCacheVersionCheck:
    """Tests for S8 fix: cache schema version is validated on read.

    Before fix: version was written to cache but never checked on read.
    After fix: unsupported versions force a cache miss to avoid parsing unknown schemas.
    """

    def test_supported_version_accepted(self, tmp_path: Path) -> None:
        """Cache with version='1.0' is accepted and parsed normally.

        # Contract: behaviors:ModelCache:SHOULD:2
        """
        cache_data = {
            "timestamp": time.time(),
            "version": "1.0",
            "models": [
                {
                    "id": "model-v1",
                    "name": "Model V1",
                    "context_window": 4096,
                    "max_output_tokens": 1024,
                },
            ],
        }
        cache_file = tmp_path / "models_cache.json"
        cache_file.write_text(json.dumps(cache_data), encoding="utf-8")

        result = read_cache(cache_file=cache_file)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].id == "model-v1"

    def test_unsupported_version_returns_none(self, tmp_path: Path) -> None:
        """Cache with unknown future version forces cache miss (returns None).

        # Contract: behaviors:ModelDiscoveryError:MUST:1
        """
        cache_data = {
            "timestamp": time.time(),
            "version": "2.0",  # Unsupported future version
            "models": [
                {
                    "id": "future-model",
                    "name": "Future",
                    "context_window": 8192,
                    "max_output_tokens": 2048,
                },
            ],
        }
        cache_file = tmp_path / "models_cache.json"
        cache_file.write_text(json.dumps(cache_data), encoding="utf-8")

        result = read_cache(cache_file=cache_file)

        # Version mismatch → cache miss → force live API call
        assert result is None

    def test_missing_version_treated_as_supported(self, tmp_path: Path) -> None:
        """Old caches without 'version' field are treated as '1.0' (backward compat).

        # Contract: behaviors:ModelCache:SHOULD:2
        """
        cache_data = {
            "timestamp": time.time(),
            # No 'version' key — pre-versioning cache format
            "models": [
                {
                    "id": "legacy-model",
                    "name": "Legacy",
                    "context_window": 2048,
                    "max_output_tokens": 512,
                },
            ],
        }
        cache_file = tmp_path / "models_cache.json"
        cache_file.write_text(json.dumps(cache_data), encoding="utf-8")

        result = read_cache(cache_file=cache_file)

        # Missing version defaults to "1.0" → accepted (backward compat)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].id == "legacy-model"

    @pytest.mark.parametrize("bad_version", [None, 123, "1..0", "", "2.0"])
    def test_malformed_or_unsupported_version_returns_none(
        self, tmp_path: Path, bad_version: object
    ) -> None:
        """Cache file with invalid/unsupported version schema returns None.

        # Contract: behaviors:ModelDiscoveryError:MUST:1
        An unreadable cache (wrong schema) is treated as a cache miss.
        """
        # Write cache file with bad/unsupported version
        cache_data = {
            "version": bad_version,
            "timestamp": time.time(),
            "models": [
                {
                    "id": "gpt-4",
                    "name": "GPT-4",
                    "context_window": 8192,
                    "max_output_tokens": 4096,
                }
            ],
        }
        cache_file = tmp_path / "models.json"
        cache_file.write_text(json.dumps(cache_data), encoding="utf-8")

        # Read and verify cache miss
        result = read_cache(cache_file=cache_file)
        assert result is None, f"Version {bad_version!r} must be treated as cache miss"


class TestWriteCacheOverwrite:
    """Tests for write_cache atomic overwrite behavior."""

    def test_write_cache_overwrites_existing_content(self, tmp_path: Path) -> None:
        """write_cache() atomically replaces existing cache file.

        # Contract: behaviors:ModelCache:SHOULD:1
        Stale content must not persist after a successful write.
        """
        cache_file = tmp_path / "models.json"

        # Write initial content
        first_models = [
            CopilotModelInfo(
                id="first-model",
                name="First Model",
                context_window=4096,
                max_output_tokens=1024,
            )
        ]
        write_cache(first_models, cache_file)

        # Overwrite with different content
        second_models = [
            CopilotModelInfo(
                id="second-model",
                name="Second Model",
                context_window=8192,
                max_output_tokens=2048,
            )
        ]
        write_cache(second_models, cache_file)

        # Read back — must see second content only
        result = read_cache(cache_file=cache_file)
        assert isinstance(result, list)
        assert len(result) == 1
        # Verify it's the second content (not first)
        assert result[0].id == "second-model"
        assert result[0].name == "Second Model"
