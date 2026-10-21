"""OCR 引擎测试。

覆盖两类内容：

1. **线程数解析**——这是实测发现的性能关键点。默认线程配置在本负载下
   慢 6 倍（整屏 5.89s vs 0.97s），因此"自动值解析正确"必须被固定住。
2. **后端降级与预热**——依赖缺失时不能抛异常，且要能给出可操作的原因。
"""

from __future__ import annotations

import unittest
from unittest import mock

from pingyitong.components.ocr import (
    NullOcrBackend,
    OcrEngine,
    RapidOcrBackend,
    create_ocr_engine,
    resolve_threads,
)
from pingyitong.components.ocr.ocr_engine import (
    _MAX_AUTO_THREADS,
    _MAX_REC_RATIO,
    _join_segment_results,
    _pick_cut_column,
    _Segment,
    _split_wide_crop,
)
from pingyitong.config.schema import OcrConfig


class TestResolveThreads(unittest.TestCase):
    """线程数解析。"""

    def test_explicit_value_preserved(self) -> None:
        """显式配置的线程数原样使用。"""
        for value in (1, 2, 4, 8, 16, 32):
            with self.subTest(threads=value):
                self.assertEqual(resolve_threads(value), value)

    def test_auto_capped_at_optimal(self) -> None:
        """自动值不得超过实测最优上限。

        实测（24 核、整屏 2560×1600）：默认 5.89s、1 线程 8.08s、
        4 线程 3.00s、**8 线程 2.25s**、16 线程 2.62s。
        超过 8 之后同步开销反超收益，因此上限必须卡在 8。
        """
        with mock.patch("psutil.cpu_count", return_value=64):
            self.assertEqual(resolve_threads(0), _MAX_AUTO_THREADS)

    def test_auto_never_exceeds_physical_cores(self) -> None:
        """核心少的机器上不应过度订阅线程。"""
        for cores, expected in ((2, 2), (4, 4), (6, 6), (8, 8)):
            with (
                self.subTest(cores=cores),
                mock.patch("psutil.cpu_count", return_value=cores),
            ):
                self.assertEqual(resolve_threads(0), expected)

    def test_auto_handles_single_core(self) -> None:
        """单核机器不应解析出 0 线程。"""
        with mock.patch("psutil.cpu_count", return_value=1):
            self.assertGreaterEqual(resolve_threads(0), 1)

    def test_auto_survives_psutil_failure(self) -> None:
        """psutil 异常时仍应给出可用值（不抛异常）。"""
        with mock.patch("psutil.cpu_count", side_effect=RuntimeError("boom")):
            value = resolve_threads(0)
        self.assertGreaterEqual(value, 1)
        self.assertLessEqual(value, _MAX_AUTO_THREADS)

    def test_negative_treated_as_auto(self) -> None:
        """负数按自动处理（配置被手工改坏时的兜底）。"""
        self.assertEqual(resolve_threads(-1), resolve_threads(0))


class TestOcrConfigDefaults(unittest.TestCase):
    """OCR 配置默认值。"""

    def test_threads_default_is_auto(self) -> None:
        """默认应为自动（0），而不是写死某个数字。

        写死会在不同核心数的机器上表现不同：24 核机器适合 8，
        4 核机器适合 4。
        """
        self.assertEqual(OcrConfig().threads, 0)

    def test_factory_passes_threads(self) -> None:
        """工厂函数应把线程配置传给引擎。"""
        config = OcrConfig()
        config.threads = 6
        engine = create_ocr_engine(config)
        self.assertEqual(engine.configured_threads, 6)
        self.assertEqual(engine.threads, 6)

    def test_factory_auto_resolves(self) -> None:
        """自动配置在引擎上应解析为具体值。"""
        engine = create_ocr_engine(OcrConfig())
        self.assertEqual(engine.configured_threads, 0)
        self.assertGreaterEqual(engine.threads, 1)


class TestEngineConfiguration(unittest.TestCase):
    """引擎参数更新。"""

    def test_configure_updates_threads(self) -> None:
        """运行时改线程数应生效。"""
        engine = OcrEngine(threads=4)
        self.assertEqual(engine.threads, 4)
        engine.configure(threads=2)
        self.assertEqual(engine.threads, 2)

    def test_configure_threads_resets_backend(self) -> None:
        """改线程数必须丢弃已建后端，否则新参数不生效。

        onnxruntime 的线程数在创建推理会话时就固定了，不重建会话改了也没用。
        """
        engine = OcrEngine(threads=4)
        engine._backend = NullOcrBackend("占位")  # 模拟已创建后端
        engine.configure(threads=8)
        self.assertIsNone(engine._backend, "改线程数后应丢弃后端以便重建")

    def test_configure_same_threads_keeps_backend(self) -> None:
        """线程数没变时不应无谓地重建后端（重建约 0.3 秒）。"""
        engine = OcrEngine(threads=4)
        backend = NullOcrBackend("占位")
        engine._backend = backend
        engine.configure(threads=4)
        self.assertIs(engine._backend, backend)

    def test_configure_other_params(self) -> None:
        """其他参数更新不受影响。"""
        engine = OcrEngine()
        engine.configure(min_confidence=0.9, min_box_height=10, box_padding=5)
        self.assertAlmostEqual(engine.min_confidence, 0.9)
        self.assertEqual(engine.min_box_height, 10)
        self.assertEqual(engine.box_padding, 5)


class TestBackendBehaviour(unittest.TestCase):
    """后端行为。"""

    def test_null_backend_is_not_ready(self) -> None:
        """占位后端必须报告不可用，避免上层误以为能识别。"""
        backend = NullOcrBackend("测试原因")
        self.assertFalse(backend.is_ready())
        self.assertEqual(backend.recognize(None), [])
        self.assertIn("测试原因", backend.error)

    def test_rapid_backend_reports_threads(self) -> None:
        """RapidOCR 后端应报告解析后的线程数。"""
        backend = RapidOcrBackend(threads=0)
        self.assertGreaterEqual(backend.threads, 1)
        backend = RapidOcrBackend(threads=3)
        self.assertEqual(backend.threads, 3)

    def test_engine_warm_up_does_not_raise(self) -> None:
        """预热在依赖缺失时也不应抛异常（只返回 False）。"""
        engine = OcrEngine()
        engine._backend = NullOcrBackend("依赖缺失")
        self.assertFalse(engine.warm_up())

    def test_engine_ready_delegates_to_backend(self) -> None:
        """引擎的可用性应来自后端。"""
        engine = OcrEngine()
        engine._backend = NullOcrBackend("依赖缺失")
        self.assertFalse(engine.is_ready())

    def test_engine_returns_empty_when_backend_unavailable(self) -> None:
        """后端不可用时识别应返回空列表而不是抛异常。"""
        engine = OcrEngine()
        engine._backend = NullOcrBackend("依赖缺失")
        self.assertEqual(engine.recognize_image(None), [])

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("numpy") is not None, "需要 numpy"
    )
    def test_engine_handles_empty_image(self) -> None:
        """空图像应安全返回空列表。"""
        import numpy as np

        engine = OcrEngine()
        engine._backend = NullOcrBackend("依赖缺失")
        self.assertEqual(engine.recognize_image(np.zeros((0, 0, 3), dtype=np.uint8)), [])


class TestThreadConfigSanitize(unittest.TestCase):
    """线程配置的清洗。"""

    def _config(self, threads: int):
        import json
        import tempfile
        from pathlib import Path

        from pingyitong.config import SettingsManager

        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "settings.json"
            config_path.write_text(json.dumps({"ocr": {"threads": threads}}), encoding="utf-8")
            return SettingsManager(config_path).load()

    def test_excessive_threads_clamped(self) -> None:
        """异常大的线程数应被夹住。

        写错成 9999 会创建海量线程，识别反而更慢甚至卡死。
        """
        config = self._config(9999)
        self.assertLessEqual(config.ocr.threads, 64)

    def test_negative_threads_clamped(self) -> None:
        """负数应被夹为 0（自动）。"""
        config = self._config(-5)
        self.assertGreaterEqual(config.ocr.threads, 0)


class TestEdgePadAndCropExpand(unittest.TestCase):
    """贴边漏检与裁剪缺尾的几何修复（两个真实缺陷的回归锁）。"""

    def test_pad_edge_grows_hw_only(self) -> None:
        """补边只扩 H/W 两轴，通道轴与数据保持不变。"""
        import numpy

        from pingyitong.components.ocr.ocr_engine import _pad_edge

        image = numpy.arange(10 * 12 * 3, dtype=numpy.uint8).reshape(10, 12, 3)
        padded, pad = _pad_edge(image, 4)
        self.assertEqual(pad, 4)
        self.assertEqual(padded.shape, (18, 20, 3))
        # 中心区域原样保留
        self.assertTrue((padded[4:14, 4:16] == image).all())
        # 边缘复制而非填黑
        self.assertTrue((padded[0, 0] == image[0, 0]).all())

    def test_pad_edge_skips_tiny_images(self) -> None:
        """图像比边距还小时原样返回，避免补出噪声。"""
        import numpy

        from pingyitong.components.ocr.ocr_engine import _pad_edge

        image = numpy.zeros((3, 5, 3), dtype=numpy.uint8)
        padded, pad = _pad_edge(image, 24)
        self.assertEqual(pad, 0)
        self.assertIs(padded, image)

    def test_expand_boxes_grows_each_side(self) -> None:
        """扩框后四边各外推指定像素，中心不变。"""
        import numpy

        from pingyitong.components.ocr.ocr_engine import _expand_boxes

        boxes = numpy.array([[[10.0, 20.0], [110.0, 20.0], [110.0, 60.0], [10.0, 60.0]]])
        grown = _expand_boxes(boxes, 4)
        xs, ys = grown[0][:, 0], grown[0][:, 1]
        self.assertAlmostEqual(xs.min(), 6.0, places=3)
        self.assertAlmostEqual(xs.max(), 114.0, places=3)
        self.assertAlmostEqual(ys.min(), 16.0, places=3)
        self.assertAlmostEqual(ys.max(), 64.0, places=3)

    def test_clamp_rect_keeps_box_inside_frame(self) -> None:
        """越界的框必须夹回图像范围（补边坐标减除后可能出现负值）。"""
        from pingyitong.components.capture.geometry import Rect
        from pingyitong.components.ocr.ocr_engine import _clamp_rect

        self.assertEqual(_clamp_rect(Rect(-8, -4, 30, 20), 100, 50), Rect(0, 0, 22, 16))
        self.assertEqual(_clamp_rect(Rect(90, 40, 30, 30), 100, 50), Rect(90, 40, 10, 10))
        self.assertEqual(_clamp_rect(Rect(10, 10, 20, 20), 100, 50), Rect(10, 10, 20, 20))


class TestWideCropSplitting(unittest.TestCase):
    """超宽裁剪块切段（实机整屏 OCR 4.3s 的修复）。

    背景：``text_rec`` 把整批识别输入补边到批内最大长宽比，混入一个
    ratio≈60 的整行代码块会把同批 5 个普通块从 36ms 拖到 887ms。
    修复是把超宽块按空白列切段后再识别、识别结果拼接回整行。
    """

    def _make_crop(self, text: str, width: int, height: int = 40):
        """渲染一张指定宽度的白字黑底行图（BGR）。"""
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont

        try:
            font = ImageFont.truetype("arial.ttf", 28)
        except OSError:  # pragma: no cover - 测试环境缺字体时降级
            font = ImageFont.load_default()
        image = Image.new("RGB", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.text((4, 4), text, fill=(255, 255, 255), font=font)
        return np.array(image)[:, :, ::-1].copy()

    def test_normal_crop_not_split(self) -> None:
        """常规比例的裁剪块原样返回，不切段。"""
        crop = self._make_crop("Some normal text here", 300)
        segments = _split_wide_crop(crop)
        self.assertEqual(len(segments), 1)
        self.assertIs(segments[0].image, crop)
        self.assertTrue(segments[0].from_space)

    def test_wide_crop_split_into_segments(self) -> None:
        """超宽块被切成多段，段比例不回到毒化区间。"""
        crop = self._make_crop(
            "A very long line of code that stretches across "
            "the whole editor width for testing purposes indeed",
            2400,
        )
        segments = _split_wide_crop(crop)
        self.assertGreater(len(segments), 1)
        for segment in segments:
            ratio = segment.image.shape[1] / float(segment.image.shape[0])
            # 切点邻域搜索允许少量越界，但不能回到毒化区间
            self.assertLessEqual(ratio, _MAX_REC_RATIO + 2)
        # 英文句子的词间空格足够密，切点应至少有一个落在真空白上
        self.assertTrue(any(s.from_space for s in segments[1:]))

    def test_join_segments_reconstructs_words(self) -> None:
        """空白切点拼接补空格、硬切点直接相连。"""
        segs = [_Segment(None, True), _Segment(None, True)]
        joined = _join_segment_results(
            [("A very long line", 0.99), ("of code", 0.98)], segs
        )
        self.assertEqual(joined[0], "A very long line of code")
        self.assertAlmostEqual(joined[1], 0.98)

        # 硬切点（切穿单词）→ 直接相连，不得插入空格
        joined = _join_segment_results(
            [("frag", 0.9), ("ment", 0.9)], [_Segment(None, True), _Segment(None, False)]
        )
        self.assertEqual(joined[0], "fragment")

    def test_split_prefers_word_space_over_nearby_letter_gap(self) -> None:
        """切点应选词间空白，避免把相邻字母误识别为其他字形。"""
        import numpy as np

        ink = np.ones(100, dtype=np.float32)
        ink[27:38] = 0.0  # 距目标较远的词间空格
        ink[49:51] = 0.0  # 距目标最近的字母间隙
        self.assertEqual(_pick_cut_column(ink, 50, 8), 32)

    def test_join_segments_chinese_no_space(self) -> None:
        """中文段拼接不插空格。"""
        segs = [_Segment(None, True), _Segment(None, True)]
        joined = _join_segment_results([("屏幕", 0.99), ("翻译", 0.99)], segs)
        self.assertEqual(joined[0], "屏幕翻译")

    def test_join_confidence_is_minimum(self) -> None:
        """整行置信度取各段最小值。"""
        segs = [_Segment(None, True), _Segment(None, True), _Segment(None, True)]
        joined = _join_segment_results([("a", 0.99), ("b", 0.60), ("c", 0.95)], segs)
        self.assertAlmostEqual(joined[1], 0.60)

    def test_small_batch_uses_fixed_recognition_width(self) -> None:
        """少量字幕行固定识别宽度，多行恢复原配置。"""
        import numpy as np

        class FakeRecognizer:
            def __init__(self) -> None:
                self.rec_image_shape = [3, 48, 320]
                self.widths: list[int] = []

            def __call__(self, crops, _return_word_box):
                self.widths.append(self.rec_image_shape[2])
                return [("text", 0.99)] * len(crops), 0.0

        class FakeEngine:
            text_rec = FakeRecognizer()

        backend = RapidOcrBackend()
        engine = FakeEngine()
        crop = np.zeros((40, 200, 3), dtype=np.uint8)

        backend._recognize_crops(engine, [crop, crop.copy()])
        self.assertEqual(engine.text_rec.widths, [960])
        self.assertEqual(engine.text_rec.rec_image_shape[2], 320)

        backend._crop_cache.clear()
        backend._recognize_crops(engine, [crop.copy() for _ in range(5)])
        self.assertEqual(engine.text_rec.widths[-1], 320)


if __name__ == "__main__":
    unittest.main(verbosity=2)
