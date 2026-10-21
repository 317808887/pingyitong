"""字幕解析器与指标测试（P2）。

解析器对应方案 v1.2 §3.2"解析器现实清单"：GBK 编码、ASS 标签与
Comment 行、VTT NOTE/STYLE、双语轨、时间轴容错。这些是中文社区
真实字幕文件的高频形态，逐条有测试。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pingyitong.services.prefetch.metrics import PrefetchMetrics
from pingyitong.services.prefetch.normalizer import normalize_for_key
from pingyitong.services.prefetch.subtitle_source import (
    detect_bilingual,
    parse_ass,
    parse_srt,
    parse_subtitle_file,
    parse_vtt,
)


def _normalizer(text: str) -> str:
    return normalize_for_key(text)


SRT_EN = """1
00:00:01,000 --> 00:00:03,500
Stay behind me.

2
00:00:04,000 --> 00:00:06,000
We don't have much time.
"""

SRT_GBK = SRT_EN.encode("gbk")
ASS_SAMPLE = """[Script Info]
Title: test

[V4+ Styles]

[Events]
Comment: 0,0:00:00.00,0:00:01.00,,字幕注释行，不应出现
Dialogue: 0,0:00:10.00,0:00:12.50,Default,,0,0,0,,{\\i1}Open the door!{\\i0}
Dialogue: 0,0:00:13.00,0:00:15.00,Default,,0,0,0,,Line one\\NLine two
"""

VTT_SAMPLE = """WEBVTT

NOTE 这是注释块，不应出现

STYLE
::cue { color: red }

1
00:00:02.000 --> 00:00:04.000
<i>Stay behind me.</i>

00:00:05.000 --> 00:00:07.000
No index line here.
"""


class TestSubtitleParsers(unittest.TestCase):
    def _file(self, content: bytes | str, suffix: str) -> Path:
        path = Path(self._tmp.name) / f"sample.{suffix}"
        if isinstance(content, str):
            content = content.encode("utf-8")
        path.write_bytes(content)
        return path

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_srt_basic(self) -> None:
        lines = parse_srt(self._file(SRT_EN, "srt"), _normalizer)
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0].text_raw, "Stay behind me.")
        self.assertEqual(lines[0].start_ms, 1000)
        self.assertEqual(lines[0].end_ms, 3500)
        self.assertEqual(lines[0].text_norm, normalize_for_key("Stay behind me."))

    def test_srt_gbk_encoding(self) -> None:
        """GBK 字幕必须能读——中文社区最常见的导入失败原因。"""
        content = "1\n00:00:01,000 --> 00:00:02,000\n跟我来。\n"
        lines = parse_srt(self._file(content.encode("gbk"), "srt"), _normalizer)
        self.assertEqual(len(lines), 1)
        self.assertIn("跟我来", lines[0].text_raw)

    def test_ass_strips_tags_and_comments(self) -> None:
        lines = parse_ass(self._file(ASS_SAMPLE, "ass"), _normalizer)
        self.assertEqual(len(lines), 2, "Comment 行不得计入")
        self.assertEqual(lines[0].text_raw, "Open the door!")
        self.assertEqual(lines[1].text_raw, "Line one\nLine two")
        self.assertEqual(lines[0].start_ms, 10000)

    def test_vtt_skips_note_and_style(self) -> None:
        lines = parse_vtt(self._file(VTT_SAMPLE, "vtt"), _normalizer)
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0].text_raw, "Stay behind me.")
        self.assertEqual(lines[1].text_raw, "No index line here.")
        self.assertEqual(lines[1].start_ms, 5000)

    def test_dispatch_by_suffix(self) -> None:
        lines = parse_subtitle_file(self._file(SRT_EN, "srt"), _normalizer)
        self.assertEqual(len(lines), 2)

    def test_bilingual_filter_keeps_source_language(self) -> None:
        """双语轨：源语言为主语言，副语言行不进匹配。"""
        lines = parse_srt(self._file(
            "1\n00:00:01,000 --> 00:00:02,000\nStay behind me.\n"
            "2\n00:00:02,000 --> 00:00:03,000\n跟在我后面。\n",
            "srt",
        ), _normalizer)
        zh = detect_bilingual(lines, "zh")
        self.assertEqual(len(zh), 1)
        self.assertIn("跟在我后面", zh[0].text_raw)
        en = detect_bilingual(lines, "en")
        self.assertEqual(len(en), 1)
        self.assertIn("Stay behind", en[0].text_raw)


class TestPrefetchMetrics(unittest.TestCase):
    """指标：真实计数、可聚合、无模拟填充。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.metrics = PrefetchMetrics(Path(self._tmp.name) / "metrics.db")

    def tearDown(self) -> None:
        self.metrics.close()   # 先释放句柄，否则 Windows 下临时目录清不掉
        self._tmp.cleanup()

    def test_hit_rate_and_latency(self) -> None:
        for _ in range(7):
            self.metrics.on_lookup(True, "exact")
        for _ in range(3):
            self.metrics.on_lookup(False, "miss")
        for ms in (10, 20, 30, 100):
            self.metrics.on_latency(ms, "cache_exact")

        snap = self.metrics.snapshot()
        self.assertEqual(snap.lookups, 10)
        self.assertAlmostEqual(snap.hit_rate, 0.7)
        self.assertEqual(snap.latency_p50_ms, 30)
        self.assertEqual(snap.latency_p95_ms, 100)

    def test_prefetch_accuracy_and_waste(self) -> None:
        for i in range(10):
            self.metrics.on_prefetch(f"k{i}", "example.com", 100)
        for i in range(6):
            self.metrics.on_prefetch_used(f"k{i}")
        snap = self.metrics.snapshot()
        self.assertAlmostEqual(snap.prefetch_accuracy, 0.6)
        self.assertEqual(snap.bytes_used, 1000)

    def test_fallback_reasons_bucketed(self) -> None:
        self.metrics.on_fallback("login_wall", "a.com")
        self.metrics.on_fallback("login_wall", "a.com")
        self.metrics.on_fallback("timeout", "b.com")
        snap = self.metrics.snapshot()
        self.assertEqual(snap.fallbacks, 3)
        self.assertEqual(snap.fallback_reasons.get("a.com"), 2)

    def test_text_summary_is_human_readable(self) -> None:
        for _ in range(3):
            self.metrics.on_lookup(True, "exact")
        self.metrics.on_lookup(False, "miss")
        snap = self.metrics.snapshot()
        self.assertIn("75.0%", snap.as_text())


if __name__ == "__main__":
    unittest.main()
