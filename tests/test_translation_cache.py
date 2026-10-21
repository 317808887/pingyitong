"""推理缓存持久化（PersistentCacheStore）回归测试。"""

from __future__ import annotations

import sqlite3
import time
import unittest
from pathlib import Path

from pingyitong.components.llm.cache_store import PersistentCacheStore
from pingyitong.components.llm.translation_cache import TranslationCache


class TestPersistentCacheStore(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "translation_cache.db"

    def _store(self, **kwargs) -> PersistentCacheStore:
        store = PersistentCacheStore(self.path, **kwargs)
        self.addCleanup(store.close)
        return store

    def test_entries_survive_reopen(self) -> None:
        store = self._store()
        store.put("k1", "译文一", 1000.0)
        store.put("k2", "译文二", 2000.0)
        store.close()

        reopened = PersistentCacheStore(self.path)
        self.addCleanup(reopened.close)
        rows = reopened.load(100)
        self.assertEqual([(k, v) for k, v, _ in rows], [("k2", "译文二"), ("k1", "译文一")])

    def test_load_respects_limit_and_order(self) -> None:
        store = self._store()
        for i in range(5):
            store.put(f"k{i}", f"v{i}", float(i))
        store.close()
        reopened = PersistentCacheStore(self.path)
        self.addCleanup(reopened.close)
        rows = reopened.load(2)
        self.assertEqual([k for k, _, _ in rows], ["k4", "k3"])

    def test_prune_keeps_newest_entries(self) -> None:
        store = self._store(max_disk_entries=3)
        for i in range(5):
            store.put(f"k{i}", f"v{i}", float(i))
        store.close()
        # 同样容量上限的实例在启动时裁剪最旧条目
        reopened = PersistentCacheStore(self.path, max_disk_entries=3)
        self.addCleanup(reopened.close)
        rows = reopened.load(100)
        self.assertEqual([k for k, _, _ in rows], ["k4", "k3", "k2"])

    def test_clear_wipes_pending_and_disk(self) -> None:
        store = self._store()
        store.put("k1", "v1", 1.0)
        store.clear()
        store.put("k2", "v2", 2.0)
        store.close()
        reopened = PersistentCacheStore(self.path)
        self.addCleanup(reopened.close)
        self.assertEqual([k for k, _, _ in reopened.load(100)], ["k2"])

    def test_corrupt_db_degrades_gracefully(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(b"not a sqlite database at all")
        store = PersistentCacheStore(self.path)
        self.addCleanup(store.close)
        self.assertEqual(store.load(10), [])
        store.put("k1", "v1", 1.0)  # 不应抛出

    def test_close_is_idempotent(self) -> None:
        store = self._store()
        store.put("k1", "v1", 1.0)
        store.close()
        store.close()  # 第二次不应抛出
        self.assertFalse(store.load(10))


class TestTranslationCachePersistence(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "translation_cache.db"

    def _cache(self, ttl: float = 1800.0, **kwargs) -> TranslationCache:
        store = PersistentCacheStore(self.path)
        self.addCleanup(store.close)
        return TranslationCache(ttl_seconds=ttl, store=store, **kwargs)

    def test_seeded_entries_hit_without_inference(self) -> None:
        first = self._cache()
        first.put("a|en>zh|g1", "你好")
        first.put("b|en>zh|g1", "世界")
        first.close()

        second = self._cache()
        self.assertEqual(second.get("a|en>zh|g1"), "你好")
        self.assertEqual(second.get("b|en>zh|g1"), "世界")
        self.assertEqual(second.stats().hits, 2)

    def test_reloaded_entries_get_fresh_ttl(self) -> None:
        short = self._cache(ttl=0.05)
        short.put("k|en>zh|g1", "旧译文")
        time.sleep(0.1)
        self.assertIsNone(short.get("k|en>zh|g1"))  # 会话内已过期
        short.close()

        long = self._cache(ttl=1800.0)
        self.assertEqual(long.get("k|en>zh|g1"), "旧译文")  # 重启后 TTL 重新起算

    def test_clear_removes_disk_entries_too(self) -> None:
        first = self._cache()
        first.put("k|en>zh|g1", "你好")
        first.close()

        second = self._cache()
        second.put("k|en>zh|g1", "你好")
        self.assertEqual(second.clear(), 1)
        second.close()

        third = self._cache()
        self.assertIsNone(third.get("k|en>zh|g1"))

    def test_close_flushes_pending_writes(self) -> None:
        cache = self._cache()
        for i in range(8):
            cache.put(f"k{i}|en>zh|g1", f"v{i}")
        cache.close()  # 立即关闭：排队写入必须刷尽

        reopened = self._cache()
        for i in range(8):
            self.assertEqual(reopened.get(f"k{i}|en>zh|g1"), f"v{i}")

    def test_disabled_cache_never_touches_disk(self) -> None:
        store = PersistentCacheStore(self.path)
        self.addCleanup(store.close)
        cache = TranslationCache(enabled=False, store=store)
        cache.put("k", "v")
        cache.clear()
        cache.close()
        reopened = PersistentCacheStore(self.path)
        self.addCleanup(reopened.close)
        self.assertEqual(reopened.load(10), [])

    def test_seeding_respects_max_entries(self) -> None:
        writer = PersistentCacheStore(self.path)
        for i in range(6):
            writer.put(f"k{i}|en>zh|g1", f"v{i}", float(i))
        writer.close()

        store = PersistentCacheStore(self.path)
        self.addCleanup(store.close)
        cache = TranslationCache(max_entries=3, store=store)
        # 只装入最新的 3 条；最旧的 k0/k1/k2 不应出现
        self.assertIsNone(cache.get("k0|en>zh|g1"))
        self.assertEqual(cache.get("k3|en>zh|g1"), "v3")
        self.assertEqual(cache.get("k5|en>zh|g1"), "v5")

    def test_corrupt_store_falls_back_to_memory(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(b"garbage")
        store = PersistentCacheStore(self.path)
        self.addCleanup(store.close)
        cache = TranslationCache(store=store)
        cache.put("k", "v")
        self.assertEqual(cache.get("k"), "v")  # 内存路径照常工作

    def test_existing_schema_table_readable(self) -> None:
        store = PersistentCacheStore(self.path)
        store.put("k", "v", 42.0)
        store.close()
        conn = sqlite3.connect(str(self.path))
        try:
            row = conn.execute(
                "SELECT value, created_at FROM inference_cache WHERE key = 'k'"
            ).fetchone()
        finally:
            conn.close()
        self.assertEqual(row, ("v", 42.0))

    def test_attach_store_after_startup_enabling(self) -> None:
        # 启动时缓存关闭（无持久层）→ 运行时开启 → 补接后既有条目可用
        seed = self._cache()
        seed.put("k|en>zh|g1", "你好")
        seed.close()

        cache = TranslationCache(enabled=False)  # 启动时未接持久层
        self.assertIsNone(cache.get("k|en>zh|g1"))

        store = PersistentCacheStore(self.path)
        self.addCleanup(store.close)
        self.assertTrue(cache.attach_store(store))
        second = PersistentCacheStore(self.path)
        try:
            self.assertFalse(cache.attach_store(second))  # 已有持久层时不再补接
        finally:
            second.close()
        self.assertIs(cache.persistent_store, store)
        cache.configure(enabled=True)
        self.assertEqual(cache.get("k|en>zh|g1"), "你好")

    def test_attach_store_persists_new_entries(self) -> None:
        cache = TranslationCache(enabled=False)
        store = PersistentCacheStore(self.path)
        self.addCleanup(store.close)
        cache.attach_store(store)
        cache.configure(enabled=True)
        cache.put("k|en>zh|g1", "你好")
        cache.close()  # 刷尽排队写入

        reopened = PersistentCacheStore(self.path)
        self.addCleanup(reopened.close)
        self.assertEqual([k for k, _, _ in reopened.load(10)], ["k|en>zh|g1"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
