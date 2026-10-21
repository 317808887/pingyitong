"""P2 尾余项测试：全轨首定位、接缝 C 行锁、字幕行往返。

对应方案 v1.2 §3.3（首定位）/ §6.3（防闪烁）/ §3.4（行索引）。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pingyitong.services.prefetch.lookup import (
    LookupContext,
    SentenceCacheLookup,
)
from pingyitong.services.prefetch.sentence_cache import (
    SentenceCache,
    SentenceEntry,
)

NL = chr(10)


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.cache = SentenceCache(
            path=Path(self._tmp.name) / "p.db",
            model_version="mv",
        )
        lines = [
            ("Stay behind me.", "跟在我后面。"),
            ("We don't have much time.", "我们时间不多了。"),
            ("Open the door!", "开门！"),
            ("Nobody knows the truth.", "没人知道真相。"),
        ]
        self.cache.put_many([
            SentenceEntry(
                source=src, translated=tgt, src_lang="en", tgt_lang="zh",
                domain="track1", model_version="mv", source_kind="subtitle",
                ttl_days=30.0, line_no=i, start_ms=i * 3000,
                end_ms=i * 3000 + 2900,
            )
            for i, (src, tgt) in enumerate(lines)
        ])

    def tearDown(self) -> None:
        self.cache.close()
        self._tmp.cleanup()


class TestFirstLocate(_Base):
    """全轨首定位：无锚点时的第一帧。"""

    def test_locates_from_full_track(self) -> None:
        """无中心锚点时也能定位（P0 标定的低置信阈值 0.60）。"""
        lookup = SentenceCacheLookup(self.cache)
        lookup.set_active_track("track1")
        result = lookup.lookup("Stay behind me.", LookupContext(target_lang="zh"))
        self.assertEqual(result, "跟在我后面。")
        # 首定位成功后回写锚点，后续走 ±4 窗口
        self.assertEqual(lookup._last_line.get("track1"), 0)

    def test_first_locate_low_confidence_returns_none(self) -> None:
        """与全轨任何行都不相似的文本不得错配。"""
        lookup = SentenceCacheLookup(self.cache)
        lookup.set_active_track("track1")
        self.assertIsNone(
            lookup.lookup("完全无关的一句话", LookupContext(target_lang="zh"))
        )


class TestRowLock(_Base):
    """接缝 C 行锁：miss 时保持已确认译文，防闪烁。"""

    def _locked_lookup(self) -> SentenceCacheLookup:
        lookup = SentenceCacheLookup(self.cache, row_lock_misses=2)
        lookup.set_active_track("track1")
        # 第一帧：正常命中，上行锁
        first = lookup.lookup("Stay behind me.", LookupContext(target_lang="zh"))
        self.assertEqual(first, "跟在我后面。")
        return lookup

    def test_lock_holds_through_brief_misses(self) -> None:
        lookup = self._locked_lookup()
        # 相似长度、无数值/肯否变化的 OCR 乱码可短暂保持已确认译文。
        self.assertEqual(
            lookup.lookup("stay bxxxxd me", LookupContext(target_lang="zh")),
            "跟在我后面。",
        )
        self.assertEqual(
            lookup.lookup("stay byyyyd me", LookupContext(target_lang="zh")),
            "跟在我后面。",
        )

    def test_lock_releases_after_tolerance(self) -> None:
        """连续 miss 超过容限后释放，交还实时推理。"""
        lookup = self._locked_lookup()
        for _ in range(2):
            lookup.lookup("stay bxxxxd me", LookupContext(target_lang="zh"))
        self.assertIsNone(
            lookup.lookup("stay byyyyd me", LookupContext(target_lang="zh")),
            "行锁容限用尽后应释放",
        )

    def test_lock_does_not_show_prior_full_sentence_for_new_short_text(self) -> None:
        lookup = self._locked_lookup()
        self.assertIsNone(lookup.lookup("Yes", LookupContext(target_lang="zh")))

    def test_lock_replaced_on_next_hit(self) -> None:
        """字幕切到下一行：锁跟随新确认行。"""
        lookup = self._locked_lookup()
        result = lookup.lookup("Open the door!", LookupContext(target_lang="zh"))
        self.assertEqual(result, "开门！")
        self.assertEqual(lookup._last_line.get("track1"), 2)

    def test_release_on_track_change(self) -> None:
        lookup = self._locked_lookup()
        lookup.set_active_track("track2")
        self.assertNotIn("track1", lookup._row_locks)


class TestSubtitleLineRoundtrip(_Base):
    """字幕行索引：调度器 pull 需要原文与时间轴。"""

    def test_get_subtitle_line(self) -> None:
        line = self.cache.get_subtitle_line("track1", 1)
        self.assertIsNotNone(line)
        self.assertEqual(line["source_raw"], "We don't have much time.")
        self.assertEqual(line["start_ms"], 3000)

    def test_missing_line_returns_none(self) -> None:
        self.assertIsNone(self.cache.get_subtitle_line("track1", 999))


if __name__ == "__main__":
    unittest.main()


class TestSubtitleDirectoryWatcher(unittest.TestCase):
    """字幕目录监视：发现新字幕文件并回调。

    QFileSystemWatcher 需要事件循环，因此用 QTest.qWait 驱动；
    .txt 等非字幕文件不得触发；同一文件只回调一次。
    """

    @classmethod
    def setUpClass(cls) -> None:
        from PyQt6.QtWidgets import QApplication

        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication([])

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self._tmp.name)

    def tearDown(self) -> None:
        # 关闭前处理掉挂起回调，避免句柄竞争
        from PyQt6.QtTest import QTest

        QTest.qWait(700)
        self._tmp.cleanup()

    def _watch(self, directories):
        from pingyitong.services.prefetch.subtitle_source import (
            SubtitleDirectoryWatcher,
        )

        found: list[Path] = []
        watcher = SubtitleDirectoryWatcher(
            [str(d) for d in directories], lambda p: found.append(p)
        )
        return watcher, found

    def test_existing_files_are_picked_up(self) -> None:
        """监视启动时目录里**已存在**的字幕也要处理。"""
        from PyQt6.QtTest import QTest

        srt = NL.join((
            "1", "00:00:01,000 --> 00:00:02,000", "Hello", "",
        ))
        (self.directory / "movie.srt").write_text(srt, encoding="utf-8")
        watcher, found = self._watch([self.directory])
        QTest.qWait(1500)  # 加固：重负载下 900ms 偶发不足
        self.assertEqual([p.name for p in found], ["movie.srt"])

    def test_new_file_triggers_callback(self) -> None:
        """监视启动后新写入的字幕文件触发回调。"""
        from PyQt6.QtTest import QTest

        watcher, found = self._watch([self.directory])
        QTest.qWait(100)
        vtt = NL.join((
            "WEBVTT", "", "00:00:01.000 --> 00:00:02.000", "Hi", "",
        ))
        (self.directory / "new.vtt").write_text(vtt, encoding="utf-8")
        QTest.qWait(1200)
        self.assertTrue(
            any(p.name == "new.vtt" for p in found),
            f"新字幕未被发现：{found}",
        )

    def test_non_subtitle_ignored(self) -> None:
        """.txt 等非字幕文件不得触发回调。"""
        from PyQt6.QtTest import QTest

        watcher, found = self._watch([self.directory])
        QTest.qWait(100)
        (self.directory / "notes.txt").write_text("忽略我", encoding="utf-8")
        QTest.qWait(1200)
        self.assertEqual(found, [])


if __name__ == "__main__":
    unittest.main()
