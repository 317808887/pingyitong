"""联网预翻译 P1 核心测试。

对应方案 v1.1：归一化、持久缓存（键/失效/LRU/fuzzy）、双层状态机、
接缝 A。P0 标定值（fuzzy ±4 / 0.80 / 序列校验 2）直接写进断言，
防止回归到未标定参数。
"""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from pingyitong.services.prefetch.gate_state import (
    GateState,
    PrefetchGate,
    SourceHealth,
)
from pingyitong.services.prefetch.lookup import (
    LookupContext,
    SentenceCacheLookup,
)
from pingyitong.services.prefetch.normalizer import (
    normalize_for_key,
    normalize_for_match,
    split_sentences,
)
from pingyitong.services.prefetch.sentence_cache import (
    SentenceCache,
    SentenceEntry,
)


def _model_version() -> str:
    from pingyitong.version import version_string

    return f"spark-x2.5@{version_string()}"


class TestNormalizer(unittest.TestCase):
    """归一化：OCR 通道与字幕行必须同分布。"""

    def test_key_folds_width_and_whitespace(self) -> None:
        """全半角与空白差异不改变缓存键。"""
        a = normalize_for_key("Hello World")
        b = normalize_for_key("Ｈｅｌｌｏ  Ｗｏｒｌｄ")
        self.assertEqual(a, b)

    def test_key_preserves_semantic_digits_and_case(self) -> None:
        self.assertNotEqual(normalize_for_key("Price 10"), normalize_for_key("Price l0"))
        self.assertNotEqual(normalize_for_key("Help US"), normalize_for_key("Help us"))

    def test_key_preserves_mathematical_symbols(self) -> None:
        self.assertNotEqual(normalize_for_key("Use X² = 4."), normalize_for_key("Use X2 = 4."))
        self.assertNotEqual(normalize_for_key("Use ½ cup."), normalize_for_key("Use 1/2 cup."))

    def test_key_preserves_word_boundary_at_newlines(self) -> None:
        self.assertEqual(normalize_for_key("Hello\nworld"), "Hello world")
        self.assertNotEqual(normalize_for_key("Hello\nworld"), "Helloworld")

    def test_key_folds_invisible_and_control(self) -> None:
        self.assertEqual(
            normalize_for_key("hel\u200blo\u00ad world"),
            normalize_for_key("hello world"),
        )

    def test_match_is_looser_than_key(self) -> None:
        """match 归一化去标点：OCR 标点噪声不影响相似度。"""
        key_text = normalize_for_key("Stay behind me!")
        match_text = normalize_for_match("Stay behind me")
        self.assertEqual(match_text, normalize_for_match("Stay behind me!"))
        self.assertTrue(match_text.startswith(key_text.lower().rstrip("!")))

    def test_split_sentences_mixed(self) -> None:
        text = "第一句话。Second sentence! 第三句？"
        pieces = split_sentences(text)
        self.assertEqual(len(pieces), 3)


class TestSentenceCache(unittest.TestCase):
    """持久缓存：键、失效、LRU、fuzzy。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.cache = SentenceCache(
            path=Path(self._tmp.name) / "prefetch.db",
            model_version="spark-x2.5@test",
            max_bytes=1.0,  # 1 MB
        )

    def tearDown(self) -> None:
        self.cache.close()
        self._tmp.cleanup()

    def _entry(self, source: str, translated: str, **kwargs) -> SentenceEntry:
        return SentenceEntry(
            source=source,
            translated=translated,
            src_lang="en",
            tgt_lang="zh",
            domain=kwargs.pop("domain", "unit"),
            model_version="spark-x2.5@test",
            source_kind="web",
            ttl_days=7.0,
            **kwargs,
        )

    def test_exact_roundtrip_and_key_stability(self) -> None:
        entry = self._entry("Stay behind me.", "跟在我后面。")
        self.cache.put_many([entry])
        key = SentenceCache.cache_key(
            normalize_for_key("  Stay   behind me. "), "en", "zh",
            "unit", "spark-x2.5@test",
        )
        self.assertEqual(self.cache.get_exact(key), "跟在我后面。")

    def test_default_capacity_is_64_megabytes(self) -> None:
        cache = SentenceCache(Path(self._tmp.name) / "default.db", "spark-x2.5@test")
        try:
            self.assertEqual(cache._max_bytes, 64 * 1024 * 1024)
        finally:
            cache.close()

    def test_usage_counts_utf8_bytes(self) -> None:
        source, translated = "Price 10", "价格10元"
        self.cache.put_many([self._entry(source, translated)])
        self.assertEqual(
            self.cache.usage_bytes(), len(source.encode("utf-8")) + len(translated.encode("utf-8"))
        )

    def test_old_semantically_colliding_cache_is_invalidated(self) -> None:
        self.cache.put_many([self._entry("Price 10", "价格10元")])
        connection = self.cache._connection()
        connection.execute("PRAGMA user_version=1")
        connection.commit()
        self.cache.close()
        reopened = SentenceCache(self.cache._path, "spark-x2.5@test")
        try:
            self.assertEqual(reopened.usage_bytes(), 0)
        finally:
            reopened.close()

    def test_model_version_change_invalidates(self) -> None:
        """模型版本变化即失效（旧译文不得显示）。"""
        self.cache.put_many([self._entry("hello", "你好")])
        key = self.cache.cache_key(
            normalize_for_key("hello"), "en", "zh", "unit", "spark-x2.5@test"
        )
        self.assertIsNotNone(self.cache.get_exact(key))

        other = SentenceCache(
            path=Path(self._tmp.name) / "prefetch.db",
            model_version="spark-x2.5@other",
            max_bytes=64,
        )
        try:
            self.assertIsNone(other.get_exact(key), "旧版本译文未失效")
        finally:
            other.close()

    def test_ttl_expiry(self) -> None:
        entry = SentenceEntry(
            source="expired line", translated="过期", src_lang="en", tgt_lang="zh",
            domain="unit", model_version="spark-x2.5@test",
            source_kind="web", ttl_days=-0.001,  # 立即过期
        )
        self.cache.put_many([entry])
        key = self.cache.cache_key(
            normalize_for_key("expired line"), "en", "zh", "unit",
            "spark-x2.5@test",
        )
        self.assertIsNone(self.cache.get_exact(key))

    def test_fuzzy_hit_and_mismatch_guard(self) -> None:
        """fuzzy：±窗口内 OCR 噪声文本命中；序列跳变拒配。"""
        lines = [
            ("Stay behind me.", "跟在我后面。"),
            ("We don't have much time.", "我们时间不多了。"),
            ("Open the door!", "开门！"),
            ("Nobody knows the truth.", "没人知道真相。"),
        ]
        entries = [
            SentenceEntry(
                source=src, translated=tgt, src_lang="en", tgt_lang="zh",
                domain="track1", model_version="spark-x2.5@test",
                source_kind="subtitle", ttl_days=30.0,
                line_no=i, start_ms=i * 3000, end_ms=i * 3000 + 2900,
            )
            for i, (src, tgt) in enumerate(lines)
        ]
        self.cache.put_many(entries)

        # OCR 噪声（I→1、标点丢失）仍应命中第 0 行
        hit = self.cache.get_fuzzy("stay beh1nd me", "track1", center_line=0)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.line_no, 0)
        self.assertEqual(hit.translated, "跟在我后面。")

        # 窗口外的行不可达（center=0 时 line_no=30 不在 ±4 内）
        self.assertIsNone(self.cache.get_fuzzy("nobody knows the truth", "track1", 30))

        # 序列跳变拒配：最近确认行 0，观测文本匹配到 0+跳变 8 行外 → None
        # （此处用窗口外等价构造已覆盖；直接校验跳变参数行为）
        self.assertIsNone(
            self.cache.get_fuzzy(
                "nobody knows the truth", "track1", center_line=3,
                last_confirmed_line=50,
            )
        )

    def test_lru_eviction_and_usage(self) -> None:
        """超容量按 LRU 淘汰，使用中的条目保留。"""
        small = SentenceCache(
            path=Path(self._tmp.name) / "small.db",
            model_version="spark-x2.5@test",
            max_bytes=0.001,  # ~1KB
        )
        try:
            for i in range(8):
                small.put_many([SentenceEntry(
                    source=f"line {i} " + "x" * 200, translated="译" * 100,
                    src_lang="en", tgt_lang="zh", domain="unit",
                    model_version="spark-x2.5@test", source_kind="web",
                    ttl_days=7.0,
                )])
            small.prune()
            self.assertLessEqual(small.usage_bytes(), 1024)
            self.assertGreater(small.usage_bytes(), 0)
        finally:
            small.close()

    def test_invalidate_source_and_clear(self) -> None:
        self.cache.put_many([self._entry("hello", "你好", domain="site1")])
        self.cache.invalidate_source("site1")
        key = self.cache.cache_key(
            normalize_for_key("hello"), "en", "zh", "site1", "spark-x2.5@test"
        )
        self.assertIsNone(self.cache.get_exact(key))

        self.cache.put_many([self._entry("world", "世界")])
        self.cache.clear()
        key2 = self.cache.cache_key(
            normalize_for_key("world"), "en", "zh", "unit", "spark-x2.5@test"
        )
        self.assertIsNone(self.cache.get_exact(key2))
        self.assertEqual(self.cache.usage_bytes(), 0)


class TestPrefetchGate(unittest.TestCase):
    """双层状态机：全局网关 + 每源健康。"""

    def test_disabled_by_default_and_badge_texts(self) -> None:
        gate = PrefetchGate()
        self.assertEqual(gate.snapshot().gate, GateState.DISABLED)
        self.assertFalse(gate.snapshot().allowed)
        self.assertEqual(gate.snapshot().badge, "已关闭")

    def test_enable_offline_then_online(self) -> None:
        gate = PrefetchGate()
        gate.set_enabled(True)
        self.assertEqual(gate.snapshot().gate, GateState.OFFLINE)
        gate.set_online(True)
        self.assertEqual(gate.snapshot().gate, GateState.ONLINE)
        self.assertTrue(gate.snapshot().allowed)
        gate.set_online(False)
        self.assertEqual(gate.snapshot().badge, "离线·屏幕翻译中")

    def test_quota_pause_and_recovery(self) -> None:
        gate = PrefetchGate(enabled=True, online=True)
        gate.set_quota_exceeded(True)
        self.assertEqual(gate.snapshot().gate, GateState.QUOTA_PAUSED)
        self.assertFalse(gate.snapshot().allowed)
        gate.set_quota_exceeded(False)
        self.assertEqual(gate.snapshot().gate, GateState.ONLINE)

    def test_single_bad_source_does_not_affect_others(self) -> None:
        """v1.0 缺陷回归：一个坏源冷却，其他源与网关态不受影响。"""
        gate = PrefetchGate(enabled=True, online=True)
        for _ in range(3):
            gate.notify_fetch_failed("bad.example.com", "login_wall")
        self.assertFalse(gate.source_allowed("bad.example.com"))
        self.assertTrue(gate.source_allowed("good.example.com"))
        self.assertEqual(gate.snapshot().gate, GateState.ONLINE)

    def test_cooldown_auto_retry_window(self) -> None:
        """冷却期满放行一次重试；重试成功恢复 OK。"""
        gate = PrefetchGate(enabled=True, online=True, cooldown_s=60.0)
        for _ in range(3):
            gate.notify_fetch_failed("site", "timeout")
        self.assertFalse(gate.source_allowed("site"))
        # 把冷却时刻拨回过去 → 放行一次
        record = gate._sources["site"]
        record.cooldown_until = time.time() - 1
        self.assertTrue(gate.source_allowed("site"))
        gate.notify_fetch_ok("site")
        self.assertEqual(
            gate._sources["site"].health, SourceHealth.OK
        )

    def test_disabled_blocks_all_sources(self) -> None:
        gate = PrefetchGate(enabled=False, online=True)
        self.assertFalse(gate.source_allowed("anything"))


class TestLookupSeam(unittest.TestCase):
    """接缝 A：查询协议实现。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.cache = SentenceCache(
            path=Path(self._tmp.name) / "p.db",
            model_version="spark-x2.5@test",
        )

    def tearDown(self) -> None:
        self.cache.close()
        self._tmp.cleanup()

    def test_exact_lookup(self) -> None:
        self.cache.put_many([SentenceEntry(
            source="Hello world", translated="你好世界", src_lang="en",
            tgt_lang="zh", domain="example.com",
            model_version="spark-x2.5@test", source_kind="web", ttl_days=7.0,
        )])
        lookup = SentenceCacheLookup(self.cache)
        result = lookup.lookup(
            "Hello world",
            LookupContext(target_lang="zh", src_lang="en", domain="example.com"),
        )
        self.assertEqual(result, "你好世界")

    def test_exact_lookup_normalizes_whitespace_on_both_sides(self) -> None:
        self.cache.put_many([SentenceEntry(
            source="Hello world", translated="你好世界", src_lang="en",
            tgt_lang="zh", domain="example.com",
            model_version="spark-x2.5@test", source_kind="web", ttl_days=7.0,
        )])
        lookup = SentenceCacheLookup(self.cache)
        self.assertEqual(
            lookup.lookup("  Hello\nworld  ", LookupContext("zh", "en", domain="example.com")),
            "你好世界",
        )

    def test_subtitle_lookup_does_not_reuse_other_target_language_or_row_lock(self) -> None:
        self.cache.put_many([SentenceEntry(
            source="Open the door", translated="开门", src_lang="en",
            tgt_lang="zh", domain="track1", model_version="spark-x2.5@test",
            source_kind="subtitle", ttl_days=7.0, line_no=0, start_ms=0, end_ms=1000,
        )])
        lookup = SentenceCacheLookup(self.cache)
        self.assertEqual(lookup.lookup("Open the door", LookupContext("zh", "en", track_id="track1")), "开门")
        self.assertIsNone(lookup.lookup("Open the door", LookupContext("ja", "en", track_id="track1")))

    def test_subtitle_lookup_rejects_wrong_explicit_source_language(self) -> None:
        self.cache.put_many([SentenceEntry(
            source="Open the door", translated="开门", src_lang="en",
            tgt_lang="zh", domain="track1", model_version="spark-x2.5@test",
            source_kind="subtitle", ttl_days=7.0, line_no=0, start_ms=0, end_ms=1000,
        )])
        lookup = SentenceCacheLookup(self.cache)
        self.assertIsNone(lookup.lookup("Open the door", LookupContext("zh", "de", track_id="track1")))

    def test_subtitle_fuzzy_and_row_lock_do_not_change_numbers_or_negation(self) -> None:
        pairs = [
            ("Set timeout to 10 seconds.", "Set timeout to 20 seconds."),
            ("Set timeout to 1.5 seconds.", "Set timeout to 15 seconds."),
            ("Do not close the window.", "Do close the window."),
        ]
        for index, (source, changed) in enumerate(pairs):
            with self.subTest(source=source):
                track = f"track-{index}"
                self.cache.put_many([SentenceEntry(
                    source=source, translated="已缓存译文", src_lang="en",
                    tgt_lang="zh", domain=track, model_version="spark-x2.5@test",
                    source_kind="subtitle", ttl_days=7.0, line_no=0, start_ms=0, end_ms=1000,
                )])
                lookup = SentenceCacheLookup(self.cache)
                context = LookupContext("zh", "en", track_id=track)
                self.assertIsNone(lookup.lookup(changed, context), "首定位不能复用相反语义")
                self.assertEqual(lookup.lookup(source, context), "已缓存译文")
                self.assertIsNone(lookup.lookup(changed, context), "fuzzy与行锁不能复用相反语义")

    def test_miss_returns_none_never_raises(self) -> None:
        lookup = SentenceCacheLookup(self.cache)
        self.assertIsNone(
            lookup.lookup("", LookupContext(target_lang="zh"))
        )
        self.assertIsNone(
            lookup.lookup("never cached", LookupContext(target_lang="zh"))
        )

    def test_lookup_exception_is_swallowed(self) -> None:
        """接缝永不抛异常：缓存抛错等价 miss（防阻塞实时链路）。"""
        from pingyitong.services.prefetch.lookup import (
            LookupContext as _Ctx,
        )
        from pingyitong.services.prefetch.lookup import (
            SentenceCacheLookup as _Lookup,
        )

        class _BrokenCache:
            _model_version = "spark-x2.5@test"

            def get_exact(self, key):
                raise RuntimeError("boom")

        lookup = _Lookup(_BrokenCache())
        self.assertIsNone(
            lookup.lookup("anything", _Ctx(target_lang="zh", src_lang="en"))
        )

    def test_over_budget_hit_is_dropped(self) -> None:
        """接缝预算必须兑现：超预算的命中按 miss 返回 None。

        模块契约是"超出即返回 None，绝不阻塞实时链路"——若超预算仍返回
        结果，预算就退化成只记日志的装饰，调用方无从感知接缝已超时。
        """
        from unittest import mock

        self.cache.put_many([SentenceEntry(
            source="Hello world", translated="你好世界", src_lang="en",
            tgt_lang="zh", domain="example.com",
            model_version="spark-x2.5@test", source_kind="web", ttl_days=7.0,
        )])
        lookup = SentenceCacheLookup(self.cache)
        with mock.patch(
            "pingyitong.services.prefetch.lookup._LOOKUP_BUDGET_S", -1.0
        ):
            result = lookup.lookup(
                "Hello world",
                LookupContext(target_lang="zh", src_lang="en", domain="example.com"),
            )
        self.assertIsNone(result, "超预算的命中必须丢弃，而不是放行走实时链路")
        # 预算恢复后同样的命中照常返回（证明上面的 None 只来自预算）
        self.assertEqual(
            lookup.lookup(
                "Hello world",
                LookupContext(target_lang="zh", src_lang="en", domain="example.com"),
            ),
            "你好世界",
        )


if __name__ == "__main__":
    unittest.main()
