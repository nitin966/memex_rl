"""L2 Experience Store for Memex(RL).

The external key-value store D: index → content that holds full-fidelity
archived artifacts. This is the "L2" in the tiered memory architecture.

Three backends behind a common interface:
  - DictStore: In-memory dict (paper's default, per-episode)
  - RedisStore: Redis-backed (distributed/persistent) — stub
  - RocksDBStore: RocksDB-backed (local high-throughput) — stub

Design principles from the paper:
  - O(1) lookup for ReadExperience
  - SHA-256 content hashing for semantic deduplication
  - Per-episode isolation: each episode gets a fresh store
  - Content blocks are raw strings for maximum fidelity
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Iterator


class ExperienceStore(ABC):
    """Abstract L2 experience store interface.

    The external KV database D from Definition 2:
      D: index → content
    Accessed only via explicit index dereferencing (ReadExperience).
    """

    @abstractmethod
    def write(self, index: str, content: str) -> bool:
        """Write content to the store under the given index.

        Uses SHA-256 deduplication: if the same content (by hash) already
        exists under this index, the write is a no-op (idempotent).

        Args:
            index: Stable index key (e.g. "ctx_code_001").
            content: Raw content string to archive.

        Returns:
            True if new content was written, False if deduplicated (no-op).
        """
        ...

    @abstractmethod
    def read(self, index: str) -> str | None:
        """Read content from the store by index.

        O(1) lookup. Returns None if the index does not exist.

        Args:
            index: The index to dereference.

        Returns:
            The archived content string, or None if not found.
        """
        ...

    @abstractmethod
    def exists(self, index: str) -> bool:
        """Check whether an index exists in the store."""
        ...

    @abstractmethod
    def delete(self, index: str) -> bool:
        """Delete an index from the store. Returns True if it existed."""
        ...

    @abstractmethod
    def list_indices(self) -> list[str]:
        """List all indices currently in the store."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Remove all entries. Used for per-episode reset."""
        ...

    @abstractmethod
    def size(self) -> int:
        """Number of entries in the store."""
        ...

    @staticmethod
    def content_hash(content: str) -> str:
        """SHA-256 hash of content for deduplication."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def __contains__(self, index: str) -> bool:
        return self.exists(index)

    def __len__(self) -> int:
        return self.size()


class DictStore(ExperienceStore):
    """In-memory dict-backed experience store.

    This is the default backend matching the paper's implementation.
    Each episode creates a fresh DictStore instance. O(1) read/write.
    SHA-256 hashing prevents duplicate content storage.
    """

    def __init__(self) -> None:
        self._data: dict[str, str] = {}
        self._hashes: dict[str, str] = {}  # index → content hash

    def write(self, index: str, content: str) -> bool:
        new_hash = self.content_hash(content)
        if index in self._hashes and self._hashes[index] == new_hash:
            return False  # Deduplicated — identical content already stored
        self._data[index] = content
        self._hashes[index] = new_hash
        return True

    def read(self, index: str) -> str | None:
        return self._data.get(index)

    def exists(self, index: str) -> bool:
        return index in self._data

    def delete(self, index: str) -> bool:
        if index in self._data:
            del self._data[index]
            del self._hashes[index]
            return True
        return False

    def list_indices(self) -> list[str]:
        return list(self._data.keys())

    def clear(self) -> None:
        self._data.clear()
        self._hashes.clear()

    def size(self) -> int:
        return len(self._data)

    def get_hash(self, index: str) -> str | None:
        """Get the content hash for a given index (useful for testing)."""
        return self._hashes.get(index)


class RedisStore(ExperienceStore):
    """Redis-backed experience store (stub for future implementation).

    Intended for distributed/persistent deployments where multiple
    agent instances share the same experience store.
    """

    def __init__(self, url: str = "redis://localhost:6379", prefix: str = "memex:") -> None:
        self._url = url
        self._prefix = prefix
        # Lazy import — redis is an optional dependency
        try:
            import redis
            self._client = redis.from_url(url, decode_responses=True)
        except ImportError:
            raise ImportError(
                "RedisStore requires the 'redis' package. "
                "Install with: pip install memex-rl[redis]"
            )

    def _key(self, index: str) -> str:
        return f"{self._prefix}{index}"

    def _hash_key(self, index: str) -> str:
        return f"{self._prefix}hash:{index}"

    def write(self, index: str, content: str) -> bool:
        new_hash = self.content_hash(content)
        existing_hash = self._client.get(self._hash_key(index))
        if existing_hash == new_hash:
            return False
        pipe = self._client.pipeline()
        pipe.set(self._key(index), content)
        pipe.set(self._hash_key(index), new_hash)
        pipe.execute()
        return True

    def read(self, index: str) -> str | None:
        return self._client.get(self._key(index))

    def exists(self, index: str) -> bool:
        return bool(self._client.exists(self._key(index)))

    def delete(self, index: str) -> bool:
        deleted = self._client.delete(self._key(index), self._hash_key(index))
        return deleted > 0

    def list_indices(self) -> list[str]:
        keys = self._client.keys(f"{self._prefix}[!h]*")  # Exclude hash: keys
        prefix_len = len(self._prefix)
        return [k[prefix_len:] for k in keys if not k.startswith(f"{self._prefix}hash:")]

    def clear(self) -> None:
        keys = self._client.keys(f"{self._prefix}*")
        if keys:
            self._client.delete(*keys)

    def size(self) -> int:
        return len(self.list_indices())
