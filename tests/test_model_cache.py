"""
Unit tests for model_cache module.

Tests cache I/O operations in isolation from the provider.
Uses tmp_path for all file operations to avoid side effects.

TDD Phase: These tests should pass after implementing the cache module.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from amplifier_module_provider_github_copilot.model_cache import (
    CACHE_FORMAT_VERSION,
    CacheEntry,
    ModelCache,
    get_cache_path,
    is_cache_stale,
    load_cache,
    write_cache,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Test Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


def setup_cache_file(
    tmp_path: Path,
    models: dict[str, dict[str, int]],
    sdk_version: str = "0.1.24",
    cached_at: str | None = None,
    format_version: int = 1,
) -> Path:
    """
    Create a cache file for testing.

    Args:
        tmp_path: pytest tmp_path (treated as HOME)
        models: Dict mapping model_id -> {"context_window": int, "max_output_tokens": int}
        sdk_version: SDK version string
        cached_at: ISO 8601 timestamp (defaults to now)
        format_version: Cache format version

    Returns:
        Path to created cache file
    """
    cache_path = tmp_path / ".amplifier" / "cache" / "github-copilot-models.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cached_at is None:
        cached_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    data = {
        "format_version": format_version,
        "cached_at": cached_at,
        "sdk_version": sdk_version,
        "models": models,
    }
    cache_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return cache_path


@pytest.fixture
def mock_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """
    Fixture that patches Path.home() to use tmp_path.

    All tests should use this fixture to avoid polluting the real home directory.
    """
    monkeypatch.setattr(
        "amplifier_module_provider_github_copilot.model_cache.Path.home",
        lambda: tmp_path,
    )
    return tmp_path


# ═══════════════════════════════════════════════════════════════════════════════
# Category 1.1: Cache Path Resolution
# ═══════════════════════════════════════════════════════════════════════════════


class TestGetCachePath:
    """Tests for get_cache_path()"""

    def test_cache_path_is_absolute(self) -> None:
        """Cache path should be absolute, not relative"""
        path = get_cache_path()
        assert path.is_absolute()

    def test_cache_path_is_deterministic(self) -> None:
        """Multiple calls should return same path"""
        p1 = get_cache_path()
        p2 = get_cache_path()
        assert p1 == p2

    def test_cache_path_contains_provider_identifier(self) -> None:
        """Path should identify this as github-copilot provider's cache"""
        path = get_cache_path()
        assert "github-copilot" in str(path)

    def test_cache_path_contains_amplifier_dir(self) -> None:
        """Path should be under .amplifier directory"""
        path = get_cache_path()
        assert ".amplifier" in str(path)

    def test_cache_path_is_in_cache_subdir(self) -> None:
        """Path should be in cache subdirectory"""
        path = get_cache_path()
        assert "cache" in str(path)

    def test_cache_path_ends_with_json(self) -> None:
        """Path should end with .json extension"""
        path = get_cache_path()
        assert path.suffix == ".json"


# ═══════════════════════════════════════════════════════════════════════════════
# Category 1.2: Cache Write Operations
# ═══════════════════════════════════════════════════════════════════════════════


class TestWriteCache:
    """Tests for write_cache()"""

    def test_write_creates_cache_directory_if_missing(
        self, mock_home: Path
    ) -> None:
        """Should create ~/.amplifier/cache/ if it doesn't exist"""
        cache_dir = mock_home / ".amplifier" / "cache"
        assert not cache_dir.exists()

        models = {"test-model": CacheEntry(context_window=100000, max_output_tokens=10000)}
        result = write_cache(models, sdk_version="0.1.24")

        assert result is True
        assert cache_dir.exists()
        assert (cache_dir / "github-copilot-models.json").exists()

    def test_write_serializes_all_models(self, mock_home: Path) -> None:
        """All models should be written to cache"""
        models = {
            "claude-opus-4.6-1m": CacheEntry(context_window=1000000, max_output_tokens=64000),
            "gpt-5.3-codex": CacheEntry(context_window=400000, max_output_tokens=128000),
        }
        write_cache(models, sdk_version="0.1.24")

        cache_path = mock_home / ".amplifier" / "cache" / "github-copilot-models.json"
        data = json.loads(cache_path.read_text(encoding="utf-8"))

        assert "claude-opus-4.6-1m" in data["models"]
        assert "gpt-5.3-codex" in data["models"]
        assert data["models"]["claude-opus-4.6-1m"]["context_window"] == 1000000
        assert data["models"]["claude-opus-4.6-1m"]["max_output_tokens"] == 64000

    def test_write_includes_format_version(self, mock_home: Path) -> None:
        """Cache should include format_version for forward compatibility"""
        models = {"test": CacheEntry(context_window=100000, max_output_tokens=10000)}
        write_cache(models, sdk_version="0.1.24")

        cache_path = mock_home / ".amplifier" / "cache" / "github-copilot-models.json"
        data = json.loads(cache_path.read_text(encoding="utf-8"))

        assert "format_version" in data
        assert data["format_version"] == CACHE_FORMAT_VERSION

    def test_write_includes_timestamp(self, mock_home: Path) -> None:
        """Cache should include cached_at timestamp"""
        models = {"test": CacheEntry(context_window=100000, max_output_tokens=10000)}
        write_cache(models, sdk_version="0.1.24")

        cache_path = mock_home / ".amplifier" / "cache" / "github-copilot-models.json"
        data = json.loads(cache_path.read_text(encoding="utf-8"))

        assert "cached_at" in data
        # Should be valid ISO 8601
        ts_str = data["cached_at"]
        if ts_str.endswith("Z"):
            ts_str = ts_str[:-1] + "+00:00"
        datetime.fromisoformat(ts_str)  # Should not raise

    def test_write_includes_sdk_version(self, mock_home: Path) -> None:
        """Cache should include sdk_version"""
        models = {"test": CacheEntry(context_window=100000, max_output_tokens=10000)}
        write_cache(models, sdk_version="0.1.24")

        cache_path = mock_home / ".amplifier" / "cache" / "github-copilot-models.json"
        data = json.loads(cache_path.read_text(encoding="utf-8"))

        assert data["sdk_version"] == "0.1.24"

    def test_write_overwrites_existing_cache(self, mock_home: Path) -> None:
        """Writing should replace existing cache file"""
        cache_path = mock_home / ".amplifier" / "cache" / "github-copilot-models.json"
        cache_path.parent.mkdir(parents=True)
        cache_path.write_text('{"old": "data", "models": {}}', encoding="utf-8")

        models = {"new-model": CacheEntry(context_window=100000, max_output_tokens=10000)}
        write_cache(models, sdk_version="0.1.24")

        data = json.loads(cache_path.read_text(encoding="utf-8"))
        assert "old" not in data
        assert "new-model" in data["models"]

    def test_write_handles_empty_models(self, mock_home: Path) -> None:
        """Writing empty models dict should create valid JSON"""
        result = write_cache({}, sdk_version="0.1.24")

        cache_path = mock_home / ".amplifier" / "cache" / "github-copilot-models.json"
        assert result is True
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        assert data["models"] == {}

    def test_write_uses_utf8_encoding(self, mock_home: Path) -> None:
        """Cache should use UTF-8 encoding"""
        # Unicode model names should work
        models = {"模型-测试": CacheEntry(context_window=100000, max_output_tokens=10000)}
        result = write_cache(models, sdk_version="0.1.24")

        assert result is True
        cache_path = mock_home / ".amplifier" / "cache" / "github-copilot-models.json"
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        assert "模型-测试" in data["models"]


# ═══════════════════════════════════════════════════════════════════════════════
# Category 1.3: Cache Read Operations
# ═══════════════════════════════════════════════════════════════════════════════


class TestLoadCache:
    """Tests for load_cache()"""

    def test_load_returns_model_cache_object(self, mock_home: Path) -> None:
        """Reading valid cache should return ModelCache"""
        setup_cache_file(
            mock_home,
            {"claude-opus-4.6-1m": {"context_window": 1000000, "max_output_tokens": 64000}},
        )

        cache = load_cache()

        assert cache is not None
        assert isinstance(cache, ModelCache)
        assert "claude-opus-4.6-1m" in cache.models
        assert cache.models["claude-opus-4.6-1m"].context_window == 1000000
        assert cache.models["claude-opus-4.6-1m"].max_output_tokens == 64000

    def test_load_returns_none_for_missing_file(self, mock_home: Path) -> None:
        """Missing cache file should return None, not raise"""
        cache = load_cache()
        assert cache is None

    def test_load_returns_none_for_corrupted_json(self, mock_home: Path) -> None:
        """Corrupted JSON should return None, not raise"""
        cache_path = mock_home / ".amplifier" / "cache" / "github-copilot-models.json"
        cache_path.parent.mkdir(parents=True)
        cache_path.write_text("not valid json {{{", encoding="utf-8")

        cache = load_cache()
        assert cache is None

    def test_load_returns_none_for_missing_models_key(self, mock_home: Path) -> None:
        """JSON without 'models' key should return None"""
        cache_path = mock_home / ".amplifier" / "cache" / "github-copilot-models.json"
        cache_path.parent.mkdir(parents=True)
        cache_path.write_text('{"cached_at": "2026-02-16", "format_version": 1}', encoding="utf-8")

        cache = load_cache()
        assert cache is None

    def test_load_returns_none_for_missing_format_version(self, mock_home: Path) -> None:
        """JSON without 'format_version' key should return None"""
        cache_path = mock_home / ".amplifier" / "cache" / "github-copilot-models.json"
        cache_path.parent.mkdir(parents=True)
        cache_path.write_text('{"models": {}}', encoding="utf-8")

        cache = load_cache()
        assert cache is None

    def test_load_parses_sdk_version(self, mock_home: Path) -> None:
        """Cache should include sdk_version in returned object"""
        setup_cache_file(
            mock_home,
            {"test": {"context_window": 100000, "max_output_tokens": 10000}},
            sdk_version="0.1.24",
        )

        cache = load_cache()

        assert cache is not None
        assert cache.sdk_version == "0.1.24"

    def test_load_parses_cached_at(self, mock_home: Path) -> None:
        """Cache should include cached_at in returned object"""
        setup_cache_file(
            mock_home,
            {"test": {"context_window": 100000, "max_output_tokens": 10000}},
        )

        cache = load_cache()

        assert cache is not None
        assert cache.cached_at is not None
        assert isinstance(cache.cached_at, datetime)

    def test_load_parses_z_suffix_timestamp(self, mock_home: Path) -> None:
        """Should parse timestamp with Z suffix"""
        setup_cache_file(
            mock_home,
            {"test": {"context_window": 100000, "max_output_tokens": 10000}},
            cached_at="2026-02-16T10:30:00Z",
        )

        cache = load_cache()

        assert cache is not None
        assert cache.cached_at.year == 2026
        assert cache.cached_at.month == 2
        assert cache.cached_at.day == 16

    def test_load_skips_invalid_model_entries(
        self, mock_home: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Should skip models with invalid values but load valid ones"""
        cache_path = mock_home / ".amplifier" / "cache" / "github-copilot-models.json"
        cache_path.parent.mkdir(parents=True)
        cache_path.write_text(
            json.dumps({
                "format_version": 1,
                "cached_at": "2026-02-16T10:00:00Z",
                "sdk_version": "0.1.24",
                "models": {
                    "valid-model": {"context_window": 100000, "max_output_tokens": 10000},
                    "invalid-context": {"context_window": -1, "max_output_tokens": 10000},
                    "invalid-output": {"context_window": 100000, "max_output_tokens": 0},
                    "missing-field": {"context_window": 100000},
                },
            }),
            encoding="utf-8",
        )

        with caplog.at_level(logging.DEBUG):
            cache = load_cache()

        assert cache is not None
        assert len(cache.models) == 1
        assert "valid-model" in cache.models


# ═══════════════════════════════════════════════════════════════════════════════
# Category 1.4: Cache Staleness Check
# ═══════════════════════════════════════════════════════════════════════════════


class TestIsCacheStale:
    """Tests for is_cache_stale()"""

    def test_fresh_cache_is_not_stale(self) -> None:
        """Cache from today should not be stale"""
        cache = ModelCache(
            format_version=1,
            cached_at=datetime.now(UTC),
            sdk_version="0.1.24",
            models={},
        )

        assert is_cache_stale(cache) is False

    def test_old_cache_is_stale(self) -> None:
        """Cache from 60 days ago should be stale (default 30 days)"""
        cache = ModelCache(
            format_version=1,
            cached_at=datetime.now(UTC) - timedelta(days=60),
            sdk_version="0.1.24",
            models={},
        )

        assert is_cache_stale(cache) is True

    def test_custom_staleness_threshold(self) -> None:
        """Should respect custom days parameter"""
        cache = ModelCache(
            format_version=1,
            cached_at=datetime.now(UTC) - timedelta(days=10),
            sdk_version="0.1.24",
            models={},
        )

        assert is_cache_stale(cache, days=7) is True
        assert is_cache_stale(cache, days=14) is False

    def test_handles_naive_datetime(self) -> None:
        """Should handle timezone-naive datetime in cache"""
        # Simulate a cache with naive datetime
        cache = ModelCache(
            format_version=1,
            cached_at=datetime.now(),  # Naive datetime
            sdk_version="0.1.24",
            models={},
        )

        # Should not raise, should treat as UTC
        result = is_cache_stale(cache)
        assert isinstance(result, bool)


# ═══════════════════════════════════════════════════════════════════════════════
# Category 1.5: Round-Trip Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestCacheRoundTrip:
    """Tests that verify write → read cycle preserves data"""

    def test_write_then_read_preserves_model_data(self, mock_home: Path) -> None:
        """Data written should be readable with identical values"""
        original_models = {
            "claude-opus-4.6-1m": CacheEntry(context_window=1000000, max_output_tokens=64000),
            "gpt-5.3-codex": CacheEntry(context_window=400000, max_output_tokens=128000),
            "gemini-3-pro-preview": CacheEntry(context_window=128000, max_output_tokens=65536),
        }

        write_cache(original_models, sdk_version="0.1.24")
        loaded = load_cache()

        assert loaded is not None
        assert len(loaded.models) == 3

        for model_id, original in original_models.items():
            assert model_id in loaded.models
            loaded_entry = loaded.models[model_id]
            assert loaded_entry.context_window == original.context_window
            assert loaded_entry.max_output_tokens == original.max_output_tokens

    def test_write_then_read_preserves_sdk_version(self, mock_home: Path) -> None:
        """SDK version should survive round-trip"""
        models = {"test": CacheEntry(context_window=100000, max_output_tokens=10000)}
        write_cache(models, sdk_version="0.1.25-beta")

        loaded = load_cache()

        assert loaded is not None
        assert loaded.sdk_version == "0.1.25-beta"

    def test_multiple_write_cycles(self, mock_home: Path) -> None:
        """Multiple writes should work, with latest data winning"""
        # First write
        models_v1 = {"model-a": CacheEntry(context_window=100000, max_output_tokens=10000)}
        write_cache(models_v1, sdk_version="0.1.23")

        # Second write
        models_v2 = {
            "model-a": CacheEntry(context_window=200000, max_output_tokens=20000),
            "model-b": CacheEntry(context_window=300000, max_output_tokens=30000),
        }
        write_cache(models_v2, sdk_version="0.1.24")

        loaded = load_cache()

        assert loaded is not None
        assert loaded.sdk_version == "0.1.24"
        assert len(loaded.models) == 2
        assert loaded.models["model-a"].context_window == 200000


# ═══════════════════════════════════════════════════════════════════════════════
# Category 1.6: Edge Cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestCacheEdgeCases:
    """Edge cases and error conditions"""

    def test_cache_file_with_unicode_model_names(self, mock_home: Path) -> None:
        """Model names with unicode should work"""
        models = {
            "模型-1": CacheEntry(context_window=100000, max_output_tokens=10000),
            "mödel-ñ": CacheEntry(context_window=200000, max_output_tokens=20000),
        }

        result = write_cache(models, sdk_version="0.1.24")
        assert result is True

        loaded = load_cache()
        assert loaded is not None
        assert "模型-1" in loaded.models
        assert "mödel-ñ" in loaded.models

    def test_cache_with_large_values(self, mock_home: Path) -> None:
        """Very large context windows should work"""
        models = {
            "mega-model": CacheEntry(
                context_window=10_000_000,  # 10M tokens
                max_output_tokens=1_000_000,  # 1M tokens
            ),
        }

        write_cache(models, sdk_version="0.1.24")
        loaded = load_cache()

        assert loaded is not None
        assert loaded.models["mega-model"].context_window == 10_000_000

    def test_cache_dir_exists_as_file(self, mock_home: Path) -> None:
        """If .amplifier/cache is a file (not dir), write should fail gracefully"""
        # Create .amplifier/cache as a FILE
        amplifier_dir = mock_home / ".amplifier"
        amplifier_dir.mkdir(parents=True)
        cache_as_file = amplifier_dir / "cache"
        cache_as_file.write_text("i am a file", encoding="utf-8")

        models = {"test": CacheEntry(context_window=100000, max_output_tokens=10000)}
        result = write_cache(models, sdk_version="0.1.24")

        assert result is False  # Should fail gracefully

    def test_empty_json_file(self, mock_home: Path) -> None:
        """Empty file should return None"""
        cache_path = mock_home / ".amplifier" / "cache" / "github-copilot-models.json"
        cache_path.parent.mkdir(parents=True)
        cache_path.write_text("", encoding="utf-8")

        cache = load_cache()
        assert cache is None

    def test_json_null_file(self, mock_home: Path) -> None:
        """JSON null should return None"""
        cache_path = mock_home / ".amplifier" / "cache" / "github-copilot-models.json"
        cache_path.parent.mkdir(parents=True)
        cache_path.write_text("null", encoding="utf-8")

        cache = load_cache()
        assert cache is None

    def test_json_array_file(self, mock_home: Path) -> None:
        """JSON array (not object) should return None"""
        cache_path = mock_home / ".amplifier" / "cache" / "github-copilot-models.json"
        cache_path.parent.mkdir(parents=True)
        cache_path.write_text("[1, 2, 3]", encoding="utf-8")

        cache = load_cache()
        assert cache is None


# ═══════════════════════════════════════════════════════════════════════════════
# Category 1.7: Logging Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestCacheLogging:
    """Tests for proper logging output"""

    def test_load_logs_model_count(
        self, mock_home: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Cache load should log number of models loaded"""
        setup_cache_file(
            mock_home,
            {
                "model-1": {"context_window": 100000, "max_output_tokens": 10000},
                "model-2": {"context_window": 200000, "max_output_tokens": 20000},
                "model-3": {"context_window": 300000, "max_output_tokens": 30000},
            },
        )

        with caplog.at_level(logging.INFO):
            load_cache()

        assert "3 model" in caplog.text.lower()

    def test_load_logs_missing_file_at_debug(
        self, mock_home: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Missing cache file should log at DEBUG level"""
        with caplog.at_level(logging.DEBUG):
            load_cache()

        assert "no cache file" in caplog.text.lower() or "not found" in caplog.text.lower()

    def test_load_logs_corruption_at_warning(
        self, mock_home: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Corrupted cache should log at WARNING level"""
        cache_path = mock_home / ".amplifier" / "cache" / "github-copilot-models.json"
        cache_path.parent.mkdir(parents=True)
        cache_path.write_text("not valid json {{{", encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            load_cache()

        assert "corrupt" in caplog.text.lower() or "json" in caplog.text.lower()

    def test_write_logs_success_at_info(
        self, mock_home: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Cache write should log success at INFO level"""
        models = {"test": CacheEntry(context_window=100000, max_output_tokens=10000)}

        with caplog.at_level(logging.INFO):
            write_cache(models, sdk_version="0.1.24")

        assert "wrote" in caplog.text.lower() or "model" in caplog.text.lower()

    def test_stale_cache_logs_at_debug(
        self, mock_home: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Stale cache check should log at DEBUG level"""
        cache = ModelCache(
            format_version=1,
            cached_at=datetime.now(UTC) - timedelta(days=60),
            sdk_version="0.1.24",
            models={},
        )

        with caplog.at_level(logging.DEBUG):
            is_cache_stale(cache)

        assert "stale" in caplog.text.lower() or "days" in caplog.text.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# Category 1.8: Cross-Platform Path Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestCachePathMacOS:
    """macOS-specific cache path tests (mocked)"""

    def test_macos_cache_path_structure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """macOS uses /Users/<user>/.amplifier/cache/"""
        mock_home = Path("/Users/developer")
        monkeypatch.setattr(
            "amplifier_module_provider_github_copilot.model_cache.Path.home",
            lambda: mock_home,
        )

        cache_path = get_cache_path()

        assert cache_path == Path("/Users/developer/.amplifier/cache/github-copilot-models.json")
        assert ".amplifier" in str(cache_path)
        assert "cache" in str(cache_path)

    def test_macos_home_with_spaces(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """macOS home directories can have spaces (e.g., /Users/John Doe)"""
        mock_home = Path("/Users/John Doe")
        monkeypatch.setattr(
            "amplifier_module_provider_github_copilot.model_cache.Path.home",
            lambda: mock_home,
        )

        cache_path = get_cache_path()

        assert cache_path == Path("/Users/John Doe/.amplifier/cache/github-copilot-models.json")
        # Spaces are handled correctly by Path
        assert "John Doe" in str(cache_path)

    def test_macos_cache_write_read_roundtrip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Full write/read cycle works with macOS-style paths"""
        # Simulate macOS home structure using tmp_path
        mock_home = tmp_path / "Users" / "developer"
        mock_home.mkdir(parents=True)
        monkeypatch.setattr(
            "amplifier_module_provider_github_copilot.model_cache.Path.home",
            lambda: mock_home,
        )

        models = {
            "claude-opus-4.5": CacheEntry(context_window=200000, max_output_tokens=32000),
        }

        # Write should succeed
        result = write_cache(models, sdk_version="0.1.24")
        assert result is True

        # File should exist at correct location
        expected_path = mock_home / ".amplifier" / "cache" / "github-copilot-models.json"
        assert expected_path.exists()

        # Read should succeed
        loaded = load_cache()
        assert loaded is not None
        assert "claude-opus-4.5" in loaded.models


class TestCachePathLinux:
    """Linux-specific cache path tests (mocked)"""

    def test_linux_cache_path_structure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Linux uses /home/<user>/.amplifier/cache/"""
        mock_home = Path("/home/developer")
        monkeypatch.setattr(
            "amplifier_module_provider_github_copilot.model_cache.Path.home",
            lambda: mock_home,
        )

        cache_path = get_cache_path()

        assert cache_path == Path("/home/developer/.amplifier/cache/github-copilot-models.json")
        assert ".amplifier" in str(cache_path)

    def test_linux_wsl_path_structure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """WSL uses /home/<user>/ just like regular Linux"""
        mock_home = Path("/home/mowrim")
        monkeypatch.setattr(
            "amplifier_module_provider_github_copilot.model_cache.Path.home",
            lambda: mock_home,
        )

        cache_path = get_cache_path()

        assert cache_path == Path("/home/mowrim/.amplifier/cache/github-copilot-models.json")

    def test_linux_root_user_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Root user home is /root"""
        mock_home = Path("/root")
        monkeypatch.setattr(
            "amplifier_module_provider_github_copilot.model_cache.Path.home",
            lambda: mock_home,
        )

        cache_path = get_cache_path()

        assert cache_path == Path("/root/.amplifier/cache/github-copilot-models.json")

    def test_linux_cache_write_read_roundtrip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Full write/read cycle works with Linux-style paths"""
        mock_home = tmp_path / "home" / "developer"
        mock_home.mkdir(parents=True)
        monkeypatch.setattr(
            "amplifier_module_provider_github_copilot.model_cache.Path.home",
            lambda: mock_home,
        )

        models = {
            "gpt-4.1": CacheEntry(context_window=128000, max_output_tokens=32768),
        }

        result = write_cache(models, sdk_version="0.1.24")
        assert result is True

        loaded = load_cache()
        assert loaded is not None
        assert loaded.models["gpt-4.1"].context_window == 128000


class TestCachePathWindows:
    """Windows-specific cache path tests (mocked)"""

    def test_windows_cache_path_structure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Windows uses C:\\Users\\<user>\\.amplifier\\cache\\"""
        mock_home = Path("C:/Users/Developer")
        monkeypatch.setattr(
            "amplifier_module_provider_github_copilot.model_cache.Path.home",
            lambda: mock_home,
        )

        cache_path = get_cache_path()

        # Path normalizes to forward slashes internally, but comparison works
        assert ".amplifier" in str(cache_path)
        assert "cache" in str(cache_path)
        assert "github-copilot-models.json" in str(cache_path)

    def test_windows_path_with_spaces(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Windows usernames can have spaces"""
        mock_home = Path("C:/Users/John Smith")
        monkeypatch.setattr(
            "amplifier_module_provider_github_copilot.model_cache.Path.home",
            lambda: mock_home,
        )

        cache_path = get_cache_path()

        assert "John Smith" in str(cache_path)
        assert ".amplifier" in str(cache_path)

    def test_windows_cache_write_read_roundtrip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Full write/read cycle works with Windows-style paths"""
        # Use tmp_path which works on all platforms
        mock_home = tmp_path / "Users" / "Developer"
        mock_home.mkdir(parents=True)
        monkeypatch.setattr(
            "amplifier_module_provider_github_copilot.model_cache.Path.home",
            lambda: mock_home,
        )

        models = {
            "o3-mini": CacheEntry(context_window=200000, max_output_tokens=100000),
        }

        result = write_cache(models, sdk_version="0.1.24")
        assert result is True

        loaded = load_cache()
        assert loaded is not None
        assert loaded.models["o3-mini"].max_output_tokens == 100000


class TestCachePathNegative:
    """Negative tests for path edge cases across platforms"""

    def test_home_returns_none_handled(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """If Path.home() raises an error, operations should fail gracefully"""

        def raise_runtime_error():
            raise RuntimeError("Could not determine home directory")

        monkeypatch.setattr(
            "amplifier_module_provider_github_copilot.model_cache.Path.home",
            raise_runtime_error,
        )

        # get_cache_path will raise, but write_cache should catch it
        models = {"test": CacheEntry(context_window=100000, max_output_tokens=10000)}

        with caplog.at_level(logging.WARNING):
            result = write_cache(models, sdk_version="0.1.24")

        # Should fail gracefully, not crash
        assert result is False
        assert (
            "failed" in caplog.text.lower()
            or "error" in caplog.text.lower()
            or "could not" in caplog.text.lower()
        )

    def test_load_when_home_raises_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """load_cache returns None if home directory cannot be determined"""

        def raise_runtime_error():
            raise RuntimeError("Could not determine home directory")

        monkeypatch.setattr(
            "amplifier_module_provider_github_copilot.model_cache.Path.home",
            raise_runtime_error,
        )

        result = load_cache()
        assert result is None

    def test_readonly_home_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Write should fail gracefully if .amplifier cannot be created"""
        # Create a read-only directory structure
        mock_home = tmp_path / "readonly_home"
        mock_home.mkdir()

        # Make it read-only (platform-dependent, may not work on all systems)
        import stat

        original_mode = mock_home.stat().st_mode
        mock_home.chmod(stat.S_IRUSR | stat.S_IXUSR)  # r-x------

        monkeypatch.setattr(
            "amplifier_module_provider_github_copilot.model_cache.Path.home",
            lambda: mock_home,
        )

        try:
            models = {"test": CacheEntry(context_window=100000, max_output_tokens=10000)}
            result = write_cache(models, sdk_version="0.1.24")

            # Should fail gracefully (may succeed on some Windows systems)
            # The important thing is it doesn't crash
            assert result in (True, False)
        finally:
            # Restore permissions for cleanup
            mock_home.chmod(original_mode)

    def test_symlink_home_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cache should work when home is a symlink (common on Linux)"""
        # Create actual home and symlink
        actual_home = tmp_path / "actual_home"
        actual_home.mkdir()
        symlink_home = tmp_path / "symlink_home"

        try:
            symlink_home.symlink_to(actual_home)
        except OSError:
            pytest.skip("Symlinks not supported on this platform")

        monkeypatch.setattr(
            "amplifier_module_provider_github_copilot.model_cache.Path.home",
            lambda: symlink_home,
        )

        models = {"test": CacheEntry(context_window=100000, max_output_tokens=10000)}

        result = write_cache(models, sdk_version="0.1.24")
        assert result is True

        # File should exist in actual location (resolved through symlink)
        actual_cache = actual_home / ".amplifier" / "cache" / "github-copilot-models.json"
        assert actual_cache.exists()

        # Read should work through symlink
        loaded = load_cache()
        assert loaded is not None
        assert "test" in loaded.models

    def test_nonexistent_home_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Write should create .amplifier/cache even if home exists but is empty"""
        mock_home = tmp_path / "empty_home"
        mock_home.mkdir()

        monkeypatch.setattr(
            "amplifier_module_provider_github_copilot.model_cache.Path.home",
            lambda: mock_home,
        )

        models = {"test": CacheEntry(context_window=100000, max_output_tokens=10000)}

        result = write_cache(models, sdk_version="0.1.24")
        assert result is True

        # Directory structure should be created
        assert (mock_home / ".amplifier").exists()
        assert (mock_home / ".amplifier" / "cache").exists()
        assert (mock_home / ".amplifier" / "cache" / "github-copilot-models.json").exists()

    def test_path_with_special_characters(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Paths with unicode/special characters should work"""
        # Common on non-English systems
        mock_home = tmp_path / "用户" / "开发者"  # Chinese: "user/developer"
        mock_home.mkdir(parents=True)

        monkeypatch.setattr(
            "amplifier_module_provider_github_copilot.model_cache.Path.home",
            lambda: mock_home,
        )

        models = {"test": CacheEntry(context_window=100000, max_output_tokens=10000)}

        result = write_cache(models, sdk_version="0.1.24")
        assert result is True

        loaded = load_cache()
        assert loaded is not None

    def test_extremely_long_home_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Very long paths should be handled (Windows MAX_PATH is 260)"""
        # Create a moderately long path (not exceeding limits)
        long_name = "a" * 50
        mock_home = tmp_path / long_name / long_name / long_name
        mock_home.mkdir(parents=True)

        monkeypatch.setattr(
            "amplifier_module_provider_github_copilot.model_cache.Path.home",
            lambda: mock_home,
        )

        models = {"test": CacheEntry(context_window=100000, max_output_tokens=10000)}

        result = write_cache(models, sdk_version="0.1.24")
        assert result is True

        loaded = load_cache()
        assert loaded is not None


# ═══════════════════════════════════════════════════════════════════════════════
# Category 1.9: ST04 Bug Fix Tests (TDD)
# ═══════════════════════════════════════════════════════════════════════════════


class TestBug1RaceConditionFix:
    """
    BUG 1: Race condition in write_cache() — tempfile.mkstemp fix.

    The old code used cache_path.with_suffix(".tmp") which:
    1. Creates predictable temp file path (security concern)
    2. If interrupted, .tmp file may cause issues on retry
    3. No cleanup on failure

    Fix: Use tempfile.mkstemp(dir=same_directory) + cleanup on failure.
    """

    def test_write_cache_uses_unique_temp_file(
        self, mock_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """write_cache should use tempfile.mkstemp, not predictable .tmp suffix."""
        import tempfile as tempfile_module

        mkstemp_calls: list[dict] = []
        original_mkstemp = tempfile_module.mkstemp

        def tracking_mkstemp(*args, **kwargs):
            mkstemp_calls.append({"args": args, "kwargs": kwargs})
            return original_mkstemp(*args, **kwargs)

        monkeypatch.setattr(tempfile_module, "mkstemp", tracking_mkstemp)

        models = {"test": CacheEntry(context_window=100000, max_output_tokens=10000)}
        result = write_cache(models, sdk_version="0.1.24")

        assert result is True
        # Should have called mkstemp at least once
        assert len(mkstemp_calls) >= 1, "write_cache should use tempfile.mkstemp"

        # mkstemp should be called with dir= pointing to cache directory
        call = mkstemp_calls[0]
        assert "dir" in call["kwargs"], "mkstemp should specify dir= for atomic rename"

    def test_write_cache_cleanup_on_failure(
        self, mock_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Temp file should be cleaned up if write fails."""

        cache_dir = mock_home / ".amplifier" / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Make the replace operation fail
        def failing_replace(self, target):
            raise OSError("Simulated replace failure")

        monkeypatch.setattr(Path, "replace", failing_replace)

        models = {"test": CacheEntry(context_window=100000, max_output_tokens=10000)}
        result = write_cache(models, sdk_version="0.1.24")

        assert result is False

        # No .tmp files should remain in cache directory
        tmp_files = list(cache_dir.glob("*.tmp"))
        assert len(tmp_files) == 0, f"Temp files should be cleaned up on failure: {tmp_files}"

    def test_write_cache_temp_in_same_directory(
        self, mock_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Temp file must be in same directory for atomic rename."""
        import tempfile as tempfile_module

        captured_dir: list[Path] = []
        original_mkstemp = tempfile_module.mkstemp

        def tracking_mkstemp(*args, **kwargs):
            if "dir" in kwargs:
                captured_dir.append(Path(kwargs["dir"]))
            return original_mkstemp(*args, **kwargs)

        monkeypatch.setattr(tempfile_module, "mkstemp", tracking_mkstemp)

        models = {"test": CacheEntry(context_window=100000, max_output_tokens=10000)}
        write_cache(models, sdk_version="0.1.24")

        # Verify mkstemp dir is the cache directory
        expected_dir = mock_home / ".amplifier" / "cache"
        assert len(captured_dir) >= 1, "mkstemp should be called with dir="
        assert captured_dir[0] == expected_dir, (
            f"Temp file dir should be cache dir for atomic rename: "
            f"got {captured_dir[0]}, expected {expected_dir}"
        )


class TestBug2IsFileCheck:
    """
    BUG 2: No is_file() check before read_text().

    If cache_path exists but is a directory or symlink to directory,
    read_text() raises IsADirectoryError.

    Fix: Use is_file() instead of exists().
    """

    def test_load_cache_when_path_is_directory(
        self, mock_home: Path
    ) -> None:
        """load_cache should return None if cache path is a directory."""
        cache_path = mock_home / ".amplifier" / "cache" / "github-copilot-models.json"
        # Create as directory instead of file
        cache_path.mkdir(parents=True, exist_ok=True)

        result = load_cache()

        assert result is None, "load_cache should handle directory at cache path"

    def test_load_cache_when_path_is_symlink_to_directory(
        self, mock_home: Path
    ) -> None:
        """load_cache should return None if cache path is symlink to directory."""
        cache_dir = mock_home / ".amplifier" / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Create a directory and symlink to it
        target_dir = mock_home / "some_directory"
        target_dir.mkdir()

        cache_path = cache_dir / "github-copilot-models.json"
        try:
            cache_path.symlink_to(target_dir)
        except OSError:
            pytest.skip("Symlinks not supported on this platform")

        result = load_cache()

        assert result is None, "load_cache should handle symlink to directory"

    def test_load_cache_when_path_is_symlink_to_file(
        self, mock_home: Path
    ) -> None:
        """load_cache should work when cache path is symlink to valid file."""
        cache_dir = mock_home / ".amplifier" / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Create actual cache file elsewhere
        actual_file = mock_home / "actual_cache.json"
        cache_data = {
            "format_version": 1,
            "cached_at": datetime.now(UTC).isoformat(),
            "sdk_version": "0.1.24",
            "models": {
                "test-model": {
                    "context_window": 100000,
                    "max_output_tokens": 10000,
                }
            },
        }
        actual_file.write_text(json.dumps(cache_data), encoding="utf-8")

        # Symlink cache path to actual file
        cache_path = cache_dir / "github-copilot-models.json"
        try:
            cache_path.symlink_to(actual_file)
        except OSError:
            pytest.skip("Symlinks not supported on this platform")

        result = load_cache()

        assert result is not None, "load_cache should work with symlink to file"
        assert "test-model" in result.models


class TestBug5StalenessWarning:
    """
    BUG 5: is_cache_stale() exists but is never called.

    Provider should check staleness after loading cache and log a warning
    if cache is older than CACHE_STALE_DAYS.

    Note: This test validates the model_cache module behavior directly.
    Provider integration tests are in test_model_cache_integration.py.
    """

    def test_stale_cache_detection_45_days(self, mock_home: Path) -> None:
        """Cache 45 days old should be detected as stale (threshold=30)."""
        cache = ModelCache(
            format_version=1,
            cached_at=datetime.now(UTC) - timedelta(days=45),
            sdk_version="0.1.24",
            models={},
        )

        result = is_cache_stale(cache, days=30)

        assert result is True, "45-day-old cache should be stale (threshold=30)"

    def test_fresh_cache_not_stale(self, mock_home: Path) -> None:
        """Cache 5 days old should not be stale."""
        cache = ModelCache(
            format_version=1,
            cached_at=datetime.now(UTC) - timedelta(days=5),
            sdk_version="0.1.24",
            models={},
        )

        result = is_cache_stale(cache, days=30)

        assert result is False, "5-day-old cache should not be stale"

    def test_stale_cache_logs_warning(
        self, mock_home: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Stale cache should log a debug message with 'amplifier init' suggestion."""
        cache = ModelCache(
            format_version=1,
            cached_at=datetime.now(UTC) - timedelta(days=45),
            sdk_version="0.1.24",
            models={},
        )

        with caplog.at_level(logging.DEBUG):
            is_cache_stale(cache)

        assert "amplifier init" in caplog.text.lower() or "stale" in caplog.text.lower()
