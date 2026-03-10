"""Unit tests for L2 Experience Store (PR 2)."""

from __future__ import annotations

import pytest

from src.memory.store import DictStore, ExperienceStore


class TestDictStore:
    """Tests for the in-memory DictStore implementation."""

    def setup_method(self):
        self.store = DictStore()

    # ── Basic CRUD ─────────────────────────────────────────────────────

    def test_write_and_read(self):
        self.store.write("idx_001", "Hello world")
        assert self.store.read("idx_001") == "Hello world"

    def test_read_nonexistent_returns_none(self):
        assert self.store.read("nonexistent") is None

    def test_exists(self):
        assert not self.store.exists("idx_001")
        self.store.write("idx_001", "content")
        assert self.store.exists("idx_001")

    def test_contains_dunder(self):
        self.store.write("idx_001", "content")
        assert "idx_001" in self.store
        assert "missing" not in self.store

    def test_delete(self):
        self.store.write("idx_001", "content")
        assert self.store.delete("idx_001") is True
        assert self.store.read("idx_001") is None
        assert self.store.delete("idx_001") is False  # Already gone

    def test_list_indices(self):
        self.store.write("a", "1")
        self.store.write("b", "2")
        self.store.write("c", "3")
        indices = self.store.list_indices()
        assert set(indices) == {"a", "b", "c"}

    def test_size_and_len(self):
        assert self.store.size() == 0
        assert len(self.store) == 0
        self.store.write("x", "data")
        assert self.store.size() == 1
        assert len(self.store) == 1

    def test_clear(self):
        self.store.write("a", "1")
        self.store.write("b", "2")
        self.store.clear()
        assert self.store.size() == 0
        assert self.store.read("a") is None

    # ── SHA-256 Deduplication ──────────────────────────────────────────

    def test_write_returns_true_for_new_content(self):
        assert self.store.write("idx", "content") is True

    def test_write_returns_false_for_duplicate(self):
        self.store.write("idx", "content")
        assert self.store.write("idx", "content") is False

    def test_write_returns_true_for_updated_content(self):
        self.store.write("idx", "version 1")
        assert self.store.write("idx", "version 2") is True
        assert self.store.read("idx") == "version 2"

    def test_dedup_uses_sha256(self):
        content = "test content for hashing"
        self.store.write("idx", content)
        expected_hash = ExperienceStore.content_hash(content)
        actual_hash = self.store.get_hash("idx")
        assert actual_hash == expected_hash

    def test_content_hash_is_deterministic(self):
        h1 = ExperienceStore.content_hash("hello")
        h2 = ExperienceStore.content_hash("hello")
        assert h1 == h2

    def test_content_hash_differs_for_different_content(self):
        h1 = ExperienceStore.content_hash("hello")
        h2 = ExperienceStore.content_hash("world")
        assert h1 != h2

    # ── Overwrite Behavior ─────────────────────────────────────────────

    def test_overwrite_updates_content_and_hash(self):
        self.store.write("idx", "old content")
        old_hash = self.store.get_hash("idx")
        self.store.write("idx", "new content")
        new_hash = self.store.get_hash("idx")
        assert self.store.read("idx") == "new content"
        assert old_hash != new_hash

    # ── Edge Cases ─────────────────────────────────────────────────────

    def test_empty_string_content(self):
        self.store.write("idx", "")
        assert self.store.read("idx") == ""

    def test_large_content(self):
        large = "x" * 1_000_000  # 1MB
        self.store.write("big", large)
        assert self.store.read("big") == large

    def test_special_characters_in_index(self):
        self.store.write("ctx/code/001", "data")
        assert self.store.read("ctx/code/001") == "data"

    def test_unicode_content(self):
        self.store.write("idx", "日本語テスト 🎉")
        assert self.store.read("idx") == "日本語テスト 🎉"

    # ── Per-Episode Isolation ──────────────────────────────────────────

    def test_separate_stores_are_isolated(self):
        store_a = DictStore()
        store_b = DictStore()
        store_a.write("idx", "episode A data")
        assert store_b.read("idx") is None

    def test_clear_simulates_episode_reset(self):
        self.store.write("loc", "room IDs")
        self.store.write("obj", "found objects")
        self.store.clear()
        # Fresh episode
        self.store.write("loc", "new room IDs")
        assert self.store.read("loc") == "new room IDs"
        assert self.store.read("obj") is None
        assert self.store.size() == 1


class TestRedisStoreImport:
    """Test that RedisStore raises a helpful error without redis installed."""

    def test_import_error_message(self):
        # This test will either pass because redis is installed,
        # or verify the error message is helpful
        try:
            from src.memory.store import RedisStore
            store = RedisStore(url="redis://localhost:6379")
            # If redis is installed and server is running, this is fine
        except ImportError as e:
            assert "pip install memex-rl[redis]" in str(e)
        except Exception:
            # Connection refused is fine — means redis package exists but server isn't running
            pass
