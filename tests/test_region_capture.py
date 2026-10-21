"""框选区域模式的捕获与提示。

覆盖两个真实缺陷：

**一、框选区域用了逻辑坐标去捕获物理屏幕。**
``RegionSelector`` 给出并存入配置的是逻辑坐标，而
``ScreenCapturer.capture_region`` 要的是物理坐标。两者之间必须换算，
否则 150% 缩放屏上"框选一块、实际翻译的是另一块"——
实测框选 1068×852（逻辑）时只捕获了同样数字的物理区域，
而正确范围是 1602×1278 物理，少截了一大圈。

**二、整帧无需翻译时静默无反应。**
屏幕文字本身就是目标语言时，程序按设计跳过全部段落并原样画回，
外观与"程序卡死"无法区分。用户看到的是"点了开始翻译却什么都没变"。
必须由界面说明原因。
"""

from __future__ import annotations

import unittest
from unittest import mock

try:
    from PyQt6.QtWidgets import QApplication

    PYQT_AVAILABLE = True
except ImportError:  # pragma: no cover
    PYQT_AVAILABLE = False

from pingyitong.components.capture.geometry import Rect, ScreenInfo
from pingyitong.core.coordinate import CoordinateMapper


def _ensure_app():
    """Qt 定时器需要 QApplication 实例。"""
    if not PYQT_AVAILABLE:
        return None
    app = QApplication.instance()
    if app is None:
        try:
            app = QApplication([])
        except Exception:  # noqa: BLE001
            return None
    return app


class _StubMapper:
    """记录调用参数的坐标映射替身。"""

    def __init__(self, result: Rect, monitor_index: int | None = 1) -> None:
        self.result = result
        self.monitor_index = monitor_index
        self.calls: list[Rect] = []
        self.refreshed = 0

    def capture_region_physical(self, region: Rect) -> Rect:
        self.calls.append(region)
        return self.result

    def monitor_index_for_physical(self, _rect: Rect) -> int | None:
        return self.monitor_index

    def refresh(self) -> list:
        self.refreshed += 1
        return []


class _StubCapturer:
    """记录被要求捕获的区域与显示器序号。"""

    def __init__(self) -> None:
        self.regions: list[Rect] = []
        self.monitor_indices: list[int | None] = []

    def capture_region(self, region: Rect, monitor_index: int | None = None):
        self.regions.append(region)
        self.monitor_indices.append(monitor_index)
        raise _StopCaptureError  # 拿到参数即可，不必真的截图

    def capture_monitor(self, index: int):
        raise _StopCaptureError

    def capture_all(self):
        raise _StopCaptureError


class _StopCaptureError(Exception):
    """用于在捕获处中断，只取参数。"""


def _make_pipeline(config, capturer):
    """构造一个只用于调用 ``_grab_frame`` 的流水线。"""
    from pingyitong.core.pipeline import Pipeline

    return Pipeline(
        config=config,
        capturer=capturer,
        differ=object(),
        ocr_engine=object(),
        translator=object(),
        overlay=object(),
    )


class _CaptureConfig:
    """最小的捕获配置替身。"""

    def __init__(self, mode: str, region=None, monitors=None) -> None:
        self.mode = mode
        self.region = region
        self.monitors = monitors or []


class _Config:
    """最小的总配置替身。"""

    def __init__(self, capture) -> None:
        self.capture = capture


@unittest.skipUnless(PYQT_AVAILABLE, "需要 PyQt6")
class TestRegionCaptureConversion(unittest.TestCase):
    """框选区域必须换算为物理坐标再捕获。"""

    @classmethod
    def setUpClass(cls) -> None:
        if _ensure_app() is None:
            raise unittest.SkipTest("无法创建 QApplication")

    def test_region_goes_through_coordinate_mapper(self) -> None:
        """区域必须先交给坐标映射，且用映射结果去捕获。

        这条断言锁定"少了换算"这个缺陷：修复前是把配置里的逻辑坐标
        原样传给了 ``capture_region``。
        """
        config = _Config(_CaptureConfig("region", region=[269, 9, 1068, 852]))
        capturer = _StubCapturer()
        pipeline = _make_pipeline(config, capturer)

        physical = Rect(404, 14, 1602, 1278)
        stub = _StubMapper(physical)
        pipeline._mapper = stub

        with self.assertRaises(_StopCaptureError):
            pipeline._grab_frame()

        self.assertEqual(len(stub.calls), 1, "必须调用坐标映射进行换算")
        self.assertEqual(
            stub.calls[0],
            Rect(269, 9, 1068, 852),
            "传给映射的应当是配置里存的逻辑坐标",
        )
        self.assertEqual(
            capturer.regions,
            [physical],
            "实际捕获的必须是换算后的物理区域，而不是逻辑坐标原样",
        )

    def test_region_is_not_passed_through_unconverted(self) -> None:
        """逻辑坐标与物理坐标不同时，捕获参数不得等于逻辑坐标。"""
        logical = Rect(269, 9, 1068, 852)
        config = _Config(_CaptureConfig("region", region=logical.to_sequence()))
        capturer = _StubCapturer()
        pipeline = _make_pipeline(config, capturer)
        stub = _StubMapper(Rect(404, 14, 1602, 1278))
        pipeline._mapper = stub

        with self.assertRaises(_StopCaptureError):
            pipeline._grab_frame()

        self.assertNotEqual(
            capturer.regions[0],
            logical,
            "捕获区域与逻辑坐标相同，说明换算被跳过了",
        )

    def test_region_capture_records_monitor_index(self) -> None:
        """框选区域捕获必须带上所属显示器序号。

        这是"翻译在跑、翻译层却一片空白"的根因：区域捕获原先写死
        ``monitor_index=None``，序号一路传到渲染阶段，而渲染阶段按显示器
        分发条目——序号为 ``None`` 的条目被判为"不属于任何被捕获的显示器"，
        于是被**整体丢弃**。状态栏的帧率与延迟看起来都正常，屏幕上却什么都
        没有，极难定位。
        """
        config = _Config(_CaptureConfig("region", region=[269, 9, 1068, 852]))
        capturer = _StubCapturer()
        pipeline = _make_pipeline(config, capturer)
        pipeline._mapper = _StubMapper(Rect(404, 14, 1602, 1278), monitor_index=1)

        with self.assertRaises(_StopCaptureError):
            pipeline._grab_frame()

        self.assertEqual(
            capturer.monitor_indices,
            [1],
            "捕获时必须把所属显示器序号传下去，否则译文会被整体丢弃",
        )

    def test_fullscreen_mode_does_not_touch_region_path(self) -> None:
        """全屏模式不受影响，仍走 capture_all。"""
        config = _Config(_CaptureConfig("fullscreen"))
        capturer = _StubCapturer()
        pipeline = _make_pipeline(config, capturer)
        pipeline._mapper = _StubMapper(Rect(0, 0, 1, 1))

        with self.assertRaises(_StopCaptureError):
            pipeline._grab_frame()

        self.assertEqual(capturer.regions, [], "全屏模式不应调用 capture_region")

    def test_refresh_screens_refreshes_mapper(self) -> None:
        """显示器变化后必须刷新映射缓存，否则仍按旧缩放比例换算。"""
        config = _Config(_CaptureConfig("fullscreen"))
        pipeline = _make_pipeline(config, _StubCapturer())
        stub = _StubMapper(Rect(0, 0, 1, 1))
        pipeline._mapper = stub

        pipeline.refresh_screens()
        self.assertEqual(stub.refreshed, 1)


class _StubMapperWithDesktop(_StubMapper):
    """带虚拟桌面的映射替身，用于"区域落在屏幕外"场景。"""

    def __init__(
        self,
        result: Rect,
        monitor_index: int | None = 1,
        desktop: Rect | None = None,
    ) -> None:
        super().__init__(result, monitor_index)
        self.desktop = desktop if desktop is not None else Rect(0, 0, 1920, 1080)

    def virtual_desktop_physical(self) -> Rect:
        return self.desktop


@unittest.skipUnless(PYQT_AVAILABLE, "需要 PyQt6")
class TestRegionOffscreenFallback(unittest.TestCase):
    """框选区域所在的显示器断开后，不能静默停摆。

    实机复现（2026-10-10）：副屏上的框选区域 ``[174, -720, 1280, 720]``
    在副屏被拔掉后仍被继续使用。mss 对虚拟桌面之外的区域不报错，
    而是"成功"返回全黑位图——差分永远无变化，流水线日志戛然而止，
    界面状态却仍显示"翻译中"。必须回退整屏捕获并告知用户。
    """

    @classmethod
    def setUpClass(cls) -> None:
        if _ensure_app() is None:
            raise unittest.SkipTest("无法创建 QApplication")

    def test_offscreen_region_falls_back_to_full_capture(self) -> None:
        """区域整体在虚拟桌面之外时必须回退 capture_all。"""
        config = _Config(_CaptureConfig("region", region=[174, -720, 1280, 720]))
        capturer = _StubCapturer()
        pipeline = _make_pipeline(config, capturer)
        # 换算结果仍是副屏坐标，但副屏已不存在——虚拟桌面只剩主屏
        pipeline._mapper = _StubMapperWithDesktop(
            Rect(174, -720, 1280, 720), desktop=Rect(0, 0, 1920, 1080)
        )

        with self.assertRaises(_StopCaptureError):
            pipeline._grab_frame()

        self.assertEqual(
            capturer.regions, [], "屏幕外区域不得再走 capture_region"
        )

    def test_offscreen_region_notice_emitted_once(self) -> None:
        """状态翻转时发一次提示，恢复后不再提示。"""
        config = _Config(_CaptureConfig("region", region=[174, -720, 1280, 720]))
        capturer = _StubCapturer()
        pipeline = _make_pipeline(config, capturer)
        stub = _StubMapperWithDesktop(
            Rect(174, -720, 1280, 720), desktop=Rect(0, 0, 1920, 1080)
        )
        pipeline._mapper = stub

        notices: list[tuple[str, str]] = []
        pipeline.skip_notice.connect(lambda r, t: notices.append((r, t)))

        for _ in range(3):
            with self.assertRaises(_StopCaptureError):
                pipeline._grab_frame()

        self.assertEqual(
            [reason for reason, _ in notices],
            ["region_offscreen"],
            "连续的屏幕外帧只应提示一次",
        )

        # 假装显示器回来了：区域重新落回虚拟桌面
        stub.result = Rect(174, 100, 1280, 720)
        with self.assertRaises(_StopCaptureError):
            pipeline._grab_frame()
        self.assertEqual(capturer.regions, [Rect(174, 100, 1280, 720)])

        # 再次拔掉：状态重新翻转，允许再提示一次
        stub.result = Rect(174, -720, 1280, 720)
        for _ in range(2):
            with self.assertRaises(_StopCaptureError):
                pipeline._grab_frame()
        self.assertEqual(
            [reason for reason, _ in notices],
            ["region_offscreen", "region_offscreen"],
        )

    def test_onscreen_region_unaffected(self) -> None:
        """区域在虚拟桌面内时行为不变，仍走 capture_region。"""
        config = _Config(_CaptureConfig("region", region=[100, 100, 800, 600]))
        capturer = _StubCapturer()
        pipeline = _make_pipeline(config, capturer)
        physical = Rect(100, 100, 800, 600)
        pipeline._mapper = _StubMapperWithDesktop(physical)

        with self.assertRaises(_StopCaptureError):
            pipeline._grab_frame()

        self.assertEqual(capturer.regions, [physical])

    def test_mapper_without_desktop_method_keeps_region_path(self) -> None:
        """映射器没有虚拟桌面接口时（测试替身/旧接口）不改变行为。"""
        config = _Config(_CaptureConfig("region", region=[100, 100, 800, 600]))
        capturer = _StubCapturer()
        pipeline = _make_pipeline(config, capturer)
        physical = Rect(174, -720, 1280, 720)
        pipeline._mapper = _StubMapper(physical)

        with self.assertRaises(_StopCaptureError):
            pipeline._grab_frame()

        self.assertEqual(capturer.regions, [physical])


@unittest.skipUnless(PYQT_AVAILABLE, "需要 PyQt6")
class TestDegenerateRegionSkip(unittest.TestCase):
    """退化为 0×0 的框选区域必须整帧跳过（不抓不译）。

    悬空的框选配置（宽或高为 0，或映射结果被裁剪成空）会让 mss
    对 0×0 区域直接报错，即便截到空图 OCR 与翻译也要白跑一整轮。
    正确行为是返回 None 让捕获循环跳过本帧，且日志只在状态翻转时
    记一次（按帧率刷屏的 debug 日志等于没有日志）。
    """

    @classmethod
    def setUpClass(cls) -> None:
        if _ensure_app() is None:
            raise unittest.SkipTest("无法创建 QApplication")

    def test_zero_size_logical_region_skips_capture(self) -> None:
        """配置里宽高为 0：不调用捕获，返回 None。"""
        config = _Config(_CaptureConfig("region", region=[100, 100, 0, 0]))
        capturer = _StubCapturer()
        pipeline = _make_pipeline(config, capturer)
        pipeline._mapper = _StubMapper(Rect(404, 14, 1602, 1278))

        self.assertIsNone(pipeline._grab_frame())
        self.assertEqual(capturer.regions, [], "0×0 区域不得发起捕获")

    def test_empty_physical_mapping_skips_capture(self) -> None:
        """逻辑区域非空但映射结果被裁成 0×0：同样跳过。"""
        config = _Config(_CaptureConfig("region", region=[100, 100, 800, 600]))
        capturer = _StubCapturer()
        pipeline = _make_pipeline(config, capturer)
        pipeline._mapper = _StubMapper(Rect(0, 0, 0, 0))

        self.assertIsNone(pipeline._grab_frame())
        self.assertEqual(capturer.regions, [])

    def test_degenerate_logs_once_until_recovery(self) -> None:
        """连续退化帧只记一次 debug 日志；恢复后可再次记录。"""
        config = _Config(_CaptureConfig("region", region=[100, 100, 800, 600]))
        capturer = _StubCapturer()
        pipeline = _make_pipeline(config, capturer)
        stub = _StubMapper(Rect(0, 0, 0, 0))
        pipeline._mapper = stub

        with mock.patch("pingyitong.core.pipeline.logger") as logger:
            for _ in range(5):
                self.assertIsNone(pipeline._grab_frame())
            self.assertEqual(
                logger.debug.call_count, 1, "连续退化帧只应记一次日志"
            )

            # 区域恢复有效：捕获继续，恢复也只记一次
            stub.result = Rect(404, 14, 1602, 1278)
            with self.assertRaises(_StopCaptureError):
                pipeline._grab_frame()
            self.assertEqual(capturer.regions, [Rect(404, 14, 1602, 1278)])
            self.assertEqual(logger.debug.call_count, 2)

            with self.assertRaises(_StopCaptureError):
                pipeline._grab_frame()
            self.assertEqual(logger.debug.call_count, 2, "恢复后不应重复记录")

    def test_fullscreen_mode_unaffected(self) -> None:
        """全屏模式没有区域路径，行为保持不变。"""
        config = _Config(_CaptureConfig("fullscreen"))
        capturer = _StubCapturer()
        pipeline = _make_pipeline(config, capturer)
        pipeline._mapper = _StubMapper(Rect(0, 0, 0, 0))

        with self.assertRaises(_StopCaptureError):
            pipeline._grab_frame()


class _FakeOverlayWindow:
    """替身翻译层窗口，只记录收到的条目。"""

    def __init__(self) -> None:
        self.items: list = []
        self.replace_flags: list[bool] = []

    def set_items(self, items, replace: bool = True) -> None:
        """与真实窗口保持同一签名（含 ``replace``）。"""
        self.items = list(items)
        self.replace_flags.append(replace)


class TestMonitorFiltering(unittest.TestCase):
    """渲染阶段的显示器过滤：不能把"序号未知"当成"不要画"。"""

    @classmethod
    def setUpClass(cls) -> None:
        if _ensure_app() is None:
            raise unittest.SkipTest("无法创建 QApplication")

    def _overlay_with_one_screen(self, name: str = "PRIMARY"):
        """构造一个只挂了一块屏的翻译层管理器。"""
        from pingyitong.components.render.overlay_window import OverlayManager

        overlay = OverlayManager.__new__(OverlayManager)
        overlay._visible = True
        overlay._screens = [
            ScreenInfo(
                name=name,
                logical=Rect(0, 0, 1707, 1067),
                physical=Rect(0, 0, 2560, 1600),
                device_pixel_ratio=1.5,
                is_primary=True,
                mss_index=1,
            )
        ]
        window = _FakeOverlayWindow()
        overlay._windows = {name: window}
        overlay._style = object()
        return overlay, window

    @staticmethod
    def _item(box: Rect, monitor_index: int | None, text: str = "translated"):
        from pingyitong.components.render.render_item import RenderItem

        return RenderItem(box=box, text=text, source_text="原文", monitor_index=monitor_index)

    def test_unknown_monitor_index_is_kept(self) -> None:
        """序号未知的条目必须保留，不能被静默丢弃。

        锁定的缺陷：框选区域捕获曾不记录显示器序号，条目全是 ``None``，
        渲染时被当成"不属于任何被捕获的显示器"而**整体丢弃**——
        翻译一直在跑，翻译层却始终空白。
        """
        overlay, window = self._overlay_with_one_screen()
        item = self._item(Rect(100, 100, 300, 40), monitor_index=None)

        rendered = overlay.apply([item], {1})

        self.assertEqual(rendered, 1, "序号未知的条目被丢弃了，翻译层会整片空白")
        self.assertEqual(len(window.items), 1)

    def test_known_mismatched_index_is_filtered(self) -> None:
        """序号明确且不属于被捕获的显示器时，应当过滤（原意要保留）。"""
        overlay, window = self._overlay_with_one_screen()
        item = self._item(Rect(100, 100, 300, 40), monitor_index=9)

        rendered = overlay.apply([item], {1})

        self.assertEqual(rendered, 0)
        self.assertEqual(window.items, [])

    def test_matching_index_is_kept(self) -> None:
        """序号匹配时正常绘制。"""
        overlay, window = self._overlay_with_one_screen()
        item = self._item(Rect(100, 100, 300, 40), monitor_index=1)

        rendered = overlay.apply([item], {1})

        self.assertEqual(rendered, 1)
        self.assertEqual(len(window.items), 1)

    def test_no_filter_keeps_everything(self) -> None:
        """未指定过滤集合时全部保留。"""
        overlay, window = self._overlay_with_one_screen()
        items = [
            self._item(Rect(100, 100, 300, 40), monitor_index=None),
            self._item(Rect(100, 200, 300, 40), monitor_index=9),
        ]

        rendered = overlay.apply(items, None)

        self.assertEqual(rendered, 2)


class TestIgnoreRegions(unittest.TestCase):
    """OCR 结果必须排除本程序自身界面（悬浮栏、文本框）。

    这些窗口显示在被捕获的屏幕上，不排除的话它们的文字会被
    识别 → 翻译 → 再显示，形成自我反馈的乱码循环。
    """

    def test_blocks_inside_ignore_regions_are_dropped(self) -> None:
        from pingyitong.components.capture.geometry import Rect
        from pingyitong.components.ocr.text_block import TextBlock

        config = _Config(_CaptureConfig("fullscreen"))
        pipeline = _make_pipeline(config, _StubCapturer())
        pipeline._ignore_regions = [Rect(900, 1300, 560, 100)]

        blocks = [
            TextBlock(text="已暂停", box=Rect(910, 1310, 80, 24), confidence=0.9),
            TextBlock(text="极速", box=Rect(1000, 1312, 60, 22), confidence=0.9),
            TextBlock(text="正文内容", box=Rect(100, 200, 400, 30), confidence=0.9),
        ]

        kept = [
            block for block in blocks
            if not any(block.box.intersects(r) for r in pipeline._ignore_regions)
        ]
        self.assertEqual([b.text for b in kept], ["正文内容"])

    def test_set_ignore_regions_replaces(self) -> None:
        from pingyitong.components.capture.geometry import Rect

        config = _Config(_CaptureConfig("fullscreen"))
        pipeline = _make_pipeline(config, _StubCapturer())
        pipeline.set_ignore_regions([Rect(0, 0, 10, 10), Rect(50, 50, 10, 10)])
        self.assertEqual(len(pipeline._ignore_regions), 2)
        pipeline.set_ignore_regions([])
        self.assertEqual(pipeline._ignore_regions, [])


class TestRenderTickTranscript(unittest.TestCase):
    """渲染分发与文本框数据路径。

    **锁定的缺陷**：``_render_tick`` 曾在翻译层隐藏时直接返回——而文本框
    模式恰恰要隐藏翻译层，结果译文永远到不了文本框，面板停在
    "（等待识别…）"。翻译线程其实一直在产出，全部在结果队列里被当作
    过期帧丢弃。
    """

    @classmethod
    def setUpClass(cls) -> None:
        if _ensure_app() is None:
            raise unittest.SkipTest("无法创建 QApplication")

    def _pipeline_with_overlay(self, overlay_visible: bool):
        """构造带可控翻译层可见性的流水线。"""

        class _Overlay:
            def __init__(self) -> None:
                self.visible = overlay_visible
                self.applied: list = []

            @property
            def is_visible(self) -> bool:
                return self.visible

            def apply(self, items, active_monitors=None, replace: bool = True) -> int:
                self.applied.append((list(items), replace))
                return len(items)

        config = _Config(_CaptureConfig("fullscreen"))
        config.translation = _TranslationConfig("en")
        capturer = _StubCapturer()
        pipeline = _make_pipeline(config, capturer)
        overlay = _Overlay()
        pipeline._overlay = overlay
        pipeline.set_transcript_enabled(True)
        return pipeline, overlay

    def _frame(self):
        from pingyitong.components.capture.geometry import Rect
        from pingyitong.components.render.render_item import RenderFrame, RenderItem

        item = RenderItem(
            box=Rect(100, 100, 300, 40),
            text="Hello",
            source_text="你好",
            monitor_index=1,
        )
        return RenderFrame(seq=1, origin=Rect(0, 0, 1920, 1080), items=[item])

    def test_hidden_overlay_still_feeds_transcript(self) -> None:
        """翻译层隐藏时，译文必须继续流入文本框。

        这是"开启文本框翻译模式后没有进行翻译"的直接回归测试。
        """
        pipeline, overlay = self._pipeline_with_overlay(overlay_visible=False)
        received: list = []
        pipeline.transcript_updated.connect(received.append)

        pipeline._result_queue.put(self._frame(), seq=1)
        pipeline._render_tick()

        self.assertEqual(len(received), 1, "翻译层隐藏时文本框没有收到译文")
        self.assertEqual(overlay.applied, [], "翻译层隐藏时不应向其分发")

    def test_hidden_overlay_and_disabled_transcript_short_circuits(self) -> None:
        """翻译层与文本框都不可见时，保持原有的快速返回（省电）。"""
        pipeline, overlay = self._pipeline_with_overlay(overlay_visible=False)
        pipeline.set_transcript_enabled(False)
        received: list = []
        pipeline.transcript_updated.connect(received.append)

        pipeline._result_queue.put(self._frame(), seq=1)
        pipeline._render_tick()

        self.assertEqual(received, [])
        self.assertEqual(overlay.applied, [])

    def test_visible_overlay_still_receives_items(self) -> None:
        """翻译层可见时行为不变（防过度修正）。"""
        pipeline, overlay = self._pipeline_with_overlay(overlay_visible=True)
        received: list = []
        pipeline.transcript_updated.connect(received.append)

        pipeline._result_queue.put(self._frame(), seq=1)
        pipeline._render_tick()

        self.assertEqual(len(overlay.applied), 1)
        self.assertEqual(len(received), 1)


class TestParagraphMonitorIndex(unittest.TestCase):
    """预处理必须把来源显示器序号写到段落上。"""

    def test_paragraph_carries_monitor_index(self) -> None:
        """``process`` 传入的显示器序号应当出现在每个段落上。"""
        from pingyitong.components.ocr.text_block import TextBlock
        from pingyitong.services.text_preprocess import TextPreprocessor

        blocks = [
            TextBlock(text="Hello world, this is a sentence.", box=Rect(100, 100, 400, 42), confidence=0.9),
            TextBlock(text="Second line of text here.", box=Rect(100, 160, 380, 40), confidence=0.9),
        ]
        paragraphs, _ = TextPreprocessor().process(blocks, monitor_index=2)

        self.assertTrue(paragraphs, "应当产出段落")
        for paragraph in paragraphs:
            self.assertEqual(paragraph.monitor_index, 2)

    def test_monitor_index_defaults_to_none(self) -> None:
        """未提供时保持 ``None``，不臆造序号。"""
        from pingyitong.components.ocr.text_block import TextBlock
        from pingyitong.services.text_preprocess import TextPreprocessor

        blocks = [
            TextBlock(text="Hello world, this is a sentence.", box=Rect(100, 100, 400, 42), confidence=0.9),
        ]
        paragraphs, _ = TextPreprocessor().process(blocks)

        self.assertTrue(paragraphs)
        self.assertIsNone(paragraphs[0].monitor_index)


class TestRegionScalingRule(unittest.TestCase):
    """区域换算必须与全局坐标换算遵循同一条规则。"""

    @classmethod
    def setUpClass(cls) -> None:
        # CoordinateMapper 构造时会枚举真实显示器，需要 QApplication；
        # 枚举结果随后被替换为合成屏幕，保证测试与环境无关。
        if _ensure_app() is None:
            raise unittest.SkipTest("无法创建 QApplication")

    @staticmethod
    def _mapper_with_secondary_screen() -> CoordinateMapper:
        """构造一块"原点不为零 + 150% 缩放"的副屏。

        原点不为零是关键：错误实现 ``x * ratio`` 与正确实现
        "原点不缩放、只缩放尺寸"只在原点为 0 时才碰巧相同。
        """
        mapper = CoordinateMapper()
        mapper._screens = [
            ScreenInfo(
                name="PRIMARY",
                logical=Rect(0, 0, 1707, 1067),
                physical=Rect(0, 0, 2560, 1600),
                device_pixel_ratio=1.5,
                is_primary=True,
                mss_index=1,
            ),
            ScreenInfo(
                name="SECOND",
                logical=Rect(2560, 0, 1280, 720),
                physical=Rect(2560, 0, 1920, 1080),
                device_pixel_ratio=1.5,
                is_primary=False,
                mss_index=2,
            ),
        ]
        return mapper

    def test_scales_size_but_not_screen_origin(self) -> None:
        """副屏上换算：原点按物理原点对齐，尺寸按比例放大。"""
        mapper = self._mapper_with_secondary_screen()
        logical = Rect(2600, 100, 200, 100)
        physical = mapper.capture_region_physical(logical)

        # 物理 = 屏幕物理原点 + (逻辑 - 屏幕物理原点) * 比例
        self.assertEqual(physical.x, 2560 + round((2600 - 2560) * 1.5))
        self.assertEqual(physical.y, round(100 * 1.5))
        self.assertEqual(physical.width, 300)
        self.assertEqual(physical.height, 150)

    def test_matches_global_converter(self) -> None:
        """区域换算与 :meth:`logical_to_physical` 结果一致。

        两份实现曾经各写一遍并因此漂移，这里钉住"只有一份算法"。
        """
        mapper = self._mapper_with_secondary_screen()
        for logical in (
            Rect(2600, 100, 200, 100),
            Rect(3000, 300, 400, 200),
            Rect(100, 100, 300, 300),
        ):
            with self.subTest(rect=logical):
                expected = mapper.logical_to_physical(logical)
                actual = mapper.capture_region_physical(logical)
                # 完全落在单屏内时裁切不应改变结果
                self.assertEqual(actual, expected)

    def test_region_is_clamped_to_single_screen(self) -> None:
        """跨屏框选必须裁剪到单块显示器内，否则位图会被屏幕间隙破坏。

        捕获区域若横跨两块屏，位图里会包含显示器之间的空隙，
        后续坐标换算全部失准。
        """
        mapper = self._mapper_with_secondary_screen()
        # 从副屏左侧开始、向右伸出副屏且跨回主屏的矩形
        spanning = Rect(2400, 0, 800, 600)
        physical = mapper.capture_region_physical(spanning)

        chosen = mapper.screen_for_logical(spanning)
        self.assertIsNotNone(chosen, "应当能判定该区域属于哪块显示器")
        bounds = chosen.physical
        self.assertGreaterEqual(physical.left, bounds.left)
        self.assertGreaterEqual(physical.top, bounds.top)
        self.assertLessEqual(physical.right, bounds.right)
        self.assertLessEqual(physical.bottom, bounds.bottom)

        # 且确实被裁小了（说明裁切生效，而非原样返回）
        self.assertLess(physical.width, round(spanning.width * 1.5))


class _TranslationConfig:
    """最小的翻译配置替身。"""

    def __init__(self, target_language: str) -> None:
        self.target_language = target_language


class _StubItem:
    """渲染条目替身。"""

    def __init__(self, text: str, source: str) -> None:
        self.text = text
        self.source_text = source
        self.is_renderable = bool(text.strip())

    @property
    def box(self):  # pragma: no cover - 仅满足接口
        return Rect(0, 0, 10, 10)


class _StubFrame:
    """渲染帧替身。"""

    def __init__(self, items) -> None:
        self.items = list(items)


@unittest.skipUnless(PYQT_AVAILABLE, "需要 PyQt6")
class TestSkipNotice(unittest.TestCase):
    """整帧内容没有可见变化时必须给出一次说明。

    用户报过"选框选取后开启翻译，并没有进行翻译"——实际原因是屏幕文字
    本身就是目标语言，程序因此按设计原样输出，外观与卡死无法区分。
    这里锁定"要说明原因"且"不要反复刷屏"。

    判据是"译文与原文是否相同"而不是统计数字：实测出现过
    17 段里"译了 4 段"、但那 4 段译出来与原文一模一样的情况
    （路径、代码片段这类内容会被模型原样保留），
    统计上不算"全部跳过"，用户看到的却依然是"什么都没变"。
    """

    @classmethod
    def setUpClass(cls) -> None:
        if _ensure_app() is None:
            raise unittest.SkipTest("无法创建 QApplication")

    def _pipeline(self, target: str = "zh"):
        config = _Config(_CaptureConfig("fullscreen"))
        config.translation = _TranslationConfig(target)
        return _make_pipeline(config, _StubCapturer())

    def test_notice_fires_when_output_equals_input(self) -> None:
        """译出来与原文一样时应提示，并带上目标语言。"""
        pipeline = self._pipeline()
        received: list[tuple[str, str]] = []
        pipeline.skip_notice.connect(lambda r, t: received.append((r, t)))

        frame = _StubFrame(
            [
                _StubItem("思考·持续了2秒", "思考·持续了2秒"),
                _StubItem("终端 cd F:/x", "终端 cd F:/x"),
            ]
        )
        pipeline._check_skip_notice(frame)

        self.assertEqual(received, [("already_target", "zh")])

    def test_notice_suppressed_when_any_item_translated(self) -> None:
        """混合帧：只要有一条内容真的变了，就不该提示。"""
        pipeline = self._pipeline()
        received: list = []
        pipeline.skip_notice.connect(lambda r, t: received.append((r, t)))

        frame = _StubFrame(
            [
                _StubItem("思考·持续了2秒", "思考·持续了2秒"),
                _StubItem("Thinking·Lasted 2 seconds", "思考·持续了2秒"),
            ]
        )
        pipeline._check_skip_notice(frame)

        self.assertEqual(received, [])

    def test_whitespace_only_difference_counts_as_no_change(self) -> None:
        """仅空白差异不算翻译生效（OCR 与模型都可能增减空格）。"""
        pipeline = self._pipeline()
        received: list = []
        pipeline.skip_notice.connect(lambda r, t: received.append((r, t)))

        frame = _StubFrame([_StubItem("  hello   world ", "hello world")])
        pipeline._check_skip_notice(frame)

        self.assertEqual(len(received), 1)

    def test_notice_is_not_repeated_for_same_conclusion(self) -> None:
        """同样的结论只提示一次，否则每帧一条通知会淹没用户。"""
        pipeline = self._pipeline()
        received: list = []
        pipeline.skip_notice.connect(lambda r, t: received.append((r, t)))

        frame = _StubFrame([_StubItem("同文", "同文")])
        for _ in range(10):
            pipeline._check_skip_notice(frame)

        self.assertEqual(len(received), 1, "重复结论不应反复提示")

    def test_notice_is_not_repeated_when_screen_flaps(self) -> None:
        """屏幕在"有变化/无变化"之间翻转时，冷却期内不得反复提示。

        这是实测出现过的问题：光标闪烁、时钟走动都会让判断结果翻转，
        只按"结论变化"去重会导致 9 秒内弹出两次提示。
        """
        pipeline = self._pipeline()
        received: list = []
        pipeline.skip_notice.connect(lambda r, t: received.append((r, t)))

        same = _StubFrame([_StubItem("同文", "同文")])
        changed = _StubFrame([_StubItem("Translated", "同文")])

        pipeline._check_skip_notice(same)
        for _ in range(5):
            # 一小块内容变化 -> 再变回无变化，结论来回翻转
            pipeline._check_skip_notice(changed)
            pipeline._check_skip_notice(same)

        self.assertEqual(len(received), 1, "冷却期内不应重复提示")

    def test_notice_fires_again_after_target_language_changes(self) -> None:
        """换了目标语言后应重新提示（这是新的结论，且属用户主动操作）。"""
        pipeline = self._pipeline()
        received: list = []
        pipeline.skip_notice.connect(lambda r, t: received.append((r, t)))

        frame = _StubFrame([_StubItem("同文", "同文")])
        pipeline._check_skip_notice(frame)
        pipeline._config.translation = _TranslationConfig("ja")
        pipeline._check_skip_notice(frame)

        self.assertEqual([item[1] for item in received], ["zh", "ja"])

    def test_notice_fires_again_after_cooldown(self) -> None:
        """冷却时间过后，同样的结论可以再提示一次（长时间无内容可译时提醒用户）。"""
        from pingyitong.constants import SKIP_NOTICE_COOLDOWN

        pipeline = self._pipeline()
        received: list = []
        pipeline.skip_notice.connect(lambda r, t: received.append((r, t)))

        frame = _StubFrame([_StubItem("同文", "同文")])
        pipeline._check_skip_notice(frame)

        # 把上次提示时刻往前推，模拟冷却结束
        pipeline._skip_notice_at -= SKIP_NOTICE_COOLDOWN + 1.0
        pipeline._check_skip_notice(frame)

        self.assertEqual(len(received), 2)

    def test_notice_suppressed_when_translation_changed_text(self) -> None:
        """只要有一条内容真的变了，就说明翻译生效，不应提示。"""
        pipeline = self._pipeline()
        received: list = []
        pipeline.skip_notice.connect(lambda r, t: received.append((r, t)))

        frame = _StubFrame([_StubItem("Translated", "同文")])
        pipeline._check_skip_notice(frame)
        # 冷却期内再判一次，同样不应提示
        pipeline._check_skip_notice(frame)

        self.assertEqual(received, [])

    def test_real_translation_does_not_bypass_cooldown(self) -> None:
        """中途出现过真译文，也不解除冷却。

        这条曾经按"出现过译文就清空记录"实现，结果是屏幕上的一小块变化
        足以让提示重新弹出。冷却必须只看时间，不看中间发生过什么。
        """
        pipeline = self._pipeline()
        received: list = []
        pipeline.skip_notice.connect(lambda r, t: received.append((r, t)))

        same = _StubFrame([_StubItem("同文", "同文")])
        changed = _StubFrame([_StubItem("Translated", "同文")])

        pipeline._check_skip_notice(same)
        pipeline._check_skip_notice(changed)
        pipeline._check_skip_notice(same)

        self.assertEqual(len(received), 1, "冷却期内即使中间有译文也不应重新提示")

        # 冷却结束后可以再提示一次
        from pingyitong.constants import SKIP_NOTICE_COOLDOWN

        pipeline._skip_notice_at -= SKIP_NOTICE_COOLDOWN + 1.0
        pipeline._check_skip_notice(same)
        self.assertEqual(len(received), 2)

    def test_empty_frame_is_ignored(self) -> None:
        """没有可渲染内容时不提示（可能只是没识别到文字）。"""
        pipeline = self._pipeline()
        received: list = []
        pipeline.skip_notice.connect(lambda r, t: received.append((r, t)))

        pipeline._check_skip_notice(_StubFrame([]))
        pipeline._check_skip_notice(_StubFrame([_StubItem("   ", "同文")]))

        self.assertEqual(received, [])


if __name__ == "__main__":
    unittest.main()
