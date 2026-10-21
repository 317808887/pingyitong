"""屏幕几何与坐标映射测试。

坐标对齐是原位翻译的成败关键：差一个缩放比例，译文就会整体偏移。
本测试覆盖 :class:`Rect` 的几何运算与 :class:`CoordinateMapper` 的
物理/逻辑坐标换算，后者通过注入模拟显示器配置来避免依赖真实硬件。
"""

from __future__ import annotations

import unittest

from pingyitong.components.capture.geometry import (
    Rect,
    ScreenInfo,
    normalise_region,
    physical_to_logical,
    screen_at,
    virtual_desktop_rect,
)


def make_screen(
    name: str,
    physical: tuple[int, int, int, int],
    ratio: float,
    primary: bool = False,
    mss_index: int | None = None,
) -> ScreenInfo:
    """构造一个模拟显示器。

    参数按 **物理坐标** 给出。逻辑坐标按 Qt ``QHighDpiScaling`` 的规则推导::

        逻辑原点 = 物理原点        （原点不缩放）
        逻辑尺寸 = 物理尺寸 / 缩放  （只缩放尺寸）

    这一点是高 DPI 多屏场景下译文错位的头号原因：如果把原点也除以缩放
    比例，150% 副屏上的译文会整体偏移数百像素甚至落到屏幕之外。
    本文件中的 :class:`TestCoordinateMapper` 专门对该规则做了回归测试。
    """
    x, y, width, height = physical
    return ScreenInfo(
        name=name,
        logical=Rect(x, y, round(width / ratio), round(height / ratio)),
        physical=Rect(x, y, width, height),
        device_pixel_ratio=ratio,
        is_primary=primary,
        mss_index=mss_index,
    )


class TestRect(unittest.TestCase):
    """矩形几何运算。"""

    def test_from_ltrb(self) -> None:
        """由左上右下构造。"""
        rect = Rect.from_ltrb(10, 20, 110, 70)
        self.assertEqual((rect.x, rect.y, rect.width, rect.height), (10, 20, 100, 50))

    def test_from_sequence(self) -> None:
        """由序列构造。"""
        rect = Rect.from_sequence([5, 6, 7, 8])
        self.assertEqual(rect.x, 5)
        self.assertEqual(rect.height, 8)

    def test_from_sequence_rejects_wrong_length(self) -> None:
        """长度不足时应当报错，避免静默产生错误坐标。"""
        with self.assertRaises(ValueError):
            Rect.from_sequence([1, 2, 3])

    def test_to_ltrb_roundtrip(self) -> None:
        """构造与还原应当一致。"""
        original = Rect(3, 4, 20, 30)
        self.assertEqual(Rect.from_ltrb(*original.to_ltrb()), original)

    def test_to_sequence_roundtrip(self) -> None:
        """序列化应当可逆（用于配置持久化）。"""
        original = Rect(3, 4, 20, 30)
        self.assertEqual(Rect.from_sequence(original.to_sequence()), original)

    def test_intersects(self) -> None:
        """相交判定。"""
        first = Rect(0, 0, 100, 100)
        self.assertTrue(first.intersects(Rect(50, 50, 100, 100)))
        self.assertFalse(first.intersects(Rect(200, 200, 10, 10)))

    def test_touching_edges_do_not_intersect(self) -> None:
        """仅边界相接不算相交（半开区间语义）。"""
        self.assertFalse(Rect(0, 0, 10, 10).intersects(Rect(10, 0, 10, 10)))

    def test_intersection(self) -> None:
        """求交集。"""
        overlap = Rect(0, 0, 100, 100).intersection(Rect(50, 50, 100, 100))
        self.assertEqual(overlap.to_sequence(), [50, 50, 50, 50])

    def test_intersection_disjoint_is_empty(self) -> None:
        """无交集时返回空矩形。"""
        self.assertTrue(Rect(0, 0, 10, 10).intersection(Rect(50, 50, 10, 10)).is_empty)

    def test_union(self) -> None:
        """求并集。"""
        merged = Rect(0, 0, 10, 10).union(Rect(20, 20, 10, 10))
        self.assertEqual(merged.to_sequence(), [0, 0, 30, 30])

    def test_expand(self) -> None:
        """向外扩展。"""
        expanded = Rect(10, 10, 20, 20).expand(2)
        self.assertEqual(expanded.to_sequence(), [8, 8, 24, 24])

    def test_clamp(self) -> None:
        """裁剪到边界内。"""
        bounds = Rect(0, 0, 50, 50)
        clipped = Rect(-10, -10, 100, 100).clamp(bounds)
        self.assertEqual(clipped.to_sequence(), [0, 0, 50, 50])

    def test_translate(self) -> None:
        """平移不改变尺寸。"""
        moved = Rect(10, 10, 20, 20).translate(5, -5)
        self.assertEqual(moved.to_sequence(), [15, 5, 20, 20])

    def test_local_global_roundtrip(self) -> None:
        """局部坐标与全局坐标应当可逆。"""
        origin = Rect(100, 200, 0, 0)
        global_rect = Rect(150, 250, 20, 10)
        local = global_rect.to_local(origin)
        self.assertEqual(local.to_sequence(), [50, 50, 20, 10])
        self.assertEqual(local.to_global(origin), global_rect)

    def test_area(self) -> None:
        """面积计算。"""
        self.assertEqual(Rect(0, 0, 10, 20).area, 200)

    def test_negative_size_is_empty(self) -> None:
        """负尺寸视为空矩形。"""
        self.assertTrue(Rect(0, 0, -1, 10).is_empty)


class TestNormaliseRegion(unittest.TestCase):
    """拖拽框选的方向规范化。"""

    def test_left_top_to_right_bottom(self) -> None:
        """右下方向拖拽。"""
        self.assertEqual(
            normalise_region((10, 10), (100, 60)).to_sequence(), [10, 10, 90, 50]
        )

    def test_right_bottom_to_left_top(self) -> None:
        """左上方向拖拽应得到同样的矩形。"""
        self.assertEqual(
            normalise_region((100, 60), (10, 10)).to_sequence(), [10, 10, 90, 50]
        )

    def test_mixed_direction(self) -> None:
        """混合方向拖拽。"""
        self.assertEqual(
            normalise_region((100, 10), (10, 60)).to_sequence(), [10, 10, 90, 50]
        )


class TestScreenMapping(unittest.TestCase):
    """显示器查找与坐标换算。"""

    def setUp(self) -> None:
        # 模拟一个典型的高 DPI 多显示器环境：
        # 主屏 1920×1080 物理 @100%（物理原点 0）
        # 副屏 2560×1440 物理 @150%（物理原点 x=1920，即主屏右侧）
        #   → 逻辑尺寸 1707×960，逻辑原点 x = 1920（原点不缩放）
        self.primary = make_screen(
            "PRIMARY", (0, 0, 1920, 1080), 1.0, primary=True, mss_index=1
        )
        self.secondary = make_screen(
            "SECONDARY", (1920, 0, 2560, 1440), 1.5, mss_index=2
        )
        self.screens = [self.primary, self.secondary]

    def test_logical_origin_of_scaled_secondary(self) -> None:
        """副屏逻辑原点等于物理原点，只有尺寸被缩放。"""
        self.assertEqual(self.secondary.logical.x, 1920)
        self.assertEqual(self.secondary.logical.width, round(2560 / 1.5))

    def test_logical_geometry_does_not_overlap(self) -> None:
        """逻辑几何不得重叠，否则按点查屏会落到错误的显示器上。"""
        self.assertGreaterEqual(
            self.secondary.logical.left, self.primary.logical.right
        )

    def test_screen_at_primary(self) -> None:
        """命中主屏。"""
        found = screen_at((500, 500), self.screens, logical=True)
        self.assertIsNotNone(found)
        self.assertEqual(found.name, "PRIMARY")

    def test_screen_at_secondary(self) -> None:
        """命中副屏。"""
        found = screen_at((2000, 500), self.screens, logical=True)
        self.assertIsNotNone(found)
        self.assertEqual(found.name, "SECONDARY")

    def test_screen_at_outside_returns_nearest(self) -> None:
        """落在显示器之外时返回最近的显示器而不是 None。

        多屏之间存在无法点到的"缝隙"，此处回退可避免上游到处判空。
        """
        found = screen_at((99999, 99999), self.screens, logical=True)
        self.assertIsNotNone(found)

    def test_scale_percent(self) -> None:
        """缩放百分比展示。"""
        self.assertEqual(self.primary.scale_percent, 100)
        self.assertEqual(self.secondary.scale_percent, 150)

    def test_physical_to_logical_on_scaled_screen(self) -> None:
        """150% 缩放下物理坐标换算为逻辑坐标。"""
        physical = Rect(1920 + 300, 150, 300, 150)
        logical = physical_to_logical(physical, self.screens)
        # 原点保持 1920，偏移量 300 折算为 200
        self.assertEqual(logical.x, 1920 + 200)
        self.assertEqual(logical.width, 200)

    def test_virtual_desktop_union(self) -> None:
        """虚拟桌面应覆盖全部显示器的物理范围。"""
        desktop = virtual_desktop_rect(self.screens)
        self.assertEqual(desktop.width, 1920 + 2560)
        self.assertEqual(desktop.height, 1440)

    def test_negative_origin_desktop(self) -> None:
        """副屏位于主屏左侧时虚拟桌面原点为负。

        这是 Windows 上很常见的排布，若忽略负原点会导致译文整体错位。
        """
        left_screen = make_screen("LEFT", (-1920, 0, 1920, 1080), 1.0, mss_index=2)
        desktop = virtual_desktop_rect([self.primary, left_screen])
        self.assertEqual(desktop.x, -1920)
        self.assertEqual(desktop.width, 3840)


class TestListScreens(unittest.TestCase):
    """list_screens 的物理坐标换算（高 DPI 多屏回归）。

    需要真实 PyQt6（导入 :class:`QGuiApplication`），无显示环境时跳过——
    这里不创建 QApplication，只对类方法打桩。
    """

    @classmethod
    def setUpClass(cls) -> None:
        try:
            from PyQt6.QtCore import QRect
            from PyQt6.QtGui import QGuiApplication
        except ImportError as exc:  # pragma: no cover - 无 PyQt6 环境
            raise unittest.SkipTest("需要 PyQt6") from exc
        cls.QRect = QRect
        cls.QGuiApplication = QGuiApplication

    def test_scaled_secondary_origin_is_not_multiplied(self) -> None:
        """150% 副屏的物理原点必须等于 Qt 逻辑原点（原点不缩放）。

        Qt 6 在逐屏缩放模式下把每块屏幕的逻辑原点放在它的物理原点上，
        mss 报告的也是同一个原点，因此 ``list_screens`` 求物理矩形时
        **只缩放尺寸**。若把原点也乘上缩放比例，副屏会整体偏移
        ``origin × (ratio-1)`` 像素：mss 索引匹配失败、译文被推出窗口。
        """
        from unittest.mock import patch

        from pingyitong.components.capture import geometry as geometry_module

        QRect = self.QRect  # noqa: N806 - 测试内沿用 Qt 类型名

        class FakeScreen:
            def __init__(self, name, x, y, w, h, ratio):
                self._name = name
                self._geo = QRect(x, y, w, h)
                self._ratio = ratio

            def name(self):
                return self._name

            def geometry(self):
                return self._geo

            def devicePixelRatio(self):
                return self._ratio

        # 主屏 1920×1080 @100%（物理原点 0）；
        # 副屏位于主屏右侧，Qt 逻辑几何 1920×1080 @150%、逻辑原点 x=1920，
        # 对应物理 2880×1620、物理原点 x=1920（mss 报告值一致）。
        screens = [
            FakeScreen("\\\\.\\DISPLAY1", 0, 0, 1920, 1080, 1.0),
            FakeScreen("\\\\.\\DISPLAY2", 1920, 0, 1920, 1080, 1.5),
        ]
        mss_monitors = [
            {"left": 0, "top": 0, "width": 1920, "height": 1080},
            {"left": 1920, "top": 0, "width": 2880, "height": 1620},
        ]

        class FakeMssSession:
            monitors = [None] + mss_monitors

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        class FakeMssModule:
            @staticmethod
            def MSS():
                return FakeMssSession()

        with (
            patch.object(self.QGuiApplication, "screens", return_value=screens),
            patch.object(
                self.QGuiApplication, "primaryScreen", return_value=screens[0]
            ),
            patch.dict("sys.modules", {"mss": FakeMssModule()}),
        ):
            result = geometry_module.list_screens()

        self.assertEqual(len(result), 2)
        primary, secondary = result

        # 主屏原点为 0，缩放与否结果相同（旧实现在单屏上碰巧正确）
        self.assertEqual(primary.physical.to_sequence(), [0, 0, 1920, 1080])
        self.assertEqual(primary.mss_index, 1)

        # 关键断言：副屏物理原点保持 1920，只有尺寸被放大
        self.assertEqual(secondary.physical.to_sequence(), [1920, 0, 2880, 1620])
        self.assertEqual(secondary.logical.to_sequence(), [1920, 0, 1920, 1080])
        # 原点与尺寸都对得上，mss 索引应能精确匹配（不再依赖尺寸兜底）
        self.assertEqual(secondary.mss_index, 2)


class TestCoordinateMapper(unittest.TestCase):
    """坐标映射器（注入模拟显示器，不依赖真实硬件）。"""

    def setUp(self) -> None:
        from pingyitong.core.coordinate import CoordinateMapper

        self.mapper = CoordinateMapper()
        # 主屏 1920×1080 物理 @100%；副屏 2560×1440 物理 @150%（位于主屏右侧）
        self.primary = make_screen("P1", (0, 0, 1920, 1080), 1.0, primary=True, mss_index=1)
        self.hidpi = make_screen("P2", (1920, 0, 2560, 1440), 1.5, mss_index=2)
        self.mapper.set_screens([self.primary, self.hidpi])

    def test_identity_on_100_percent(self) -> None:
        """100% 缩放下换算结果不变。"""
        rect = Rect(100, 200, 300, 40)
        self.assertEqual(self.mapper.map_box(rect).logical, rect)

    def test_map_box_on_hidpi_screen(self) -> None:
        """150% 缩放下逻辑坐标应为"原点不变、偏移量按比例折算"。"""
        physical = Rect(1920, 150, 300, 60)
        mapped = self.mapper.map_box(physical)
        self.assertEqual(mapped.screen_name, "P2")
        self.assertAlmostEqual(mapped.device_pixel_ratio, 1.5)
        # 物理原点 1920 恰为副屏左边界 → 逻辑原点仍为 1920
        self.assertEqual(mapped.logical.x, 1920)
        self.assertEqual(mapped.logical.width, 200)

    def test_map_box_offset_is_scaled(self) -> None:
        """屏内偏移量需要按比例折算，但原点不折算。"""
        mapped = self.mapper.map_box(Rect(1920 + 300, 0, 150, 30))
        self.assertEqual(mapped.logical.x, 1920 + 200)
        self.assertEqual(mapped.logical.width, 100)

    def test_to_window_coords_is_within_window(self) -> None:
        """窗口内坐标必须落在窗口尺寸范围内。

        这是关键回归测试：若把逻辑原点重复扣减一次，副屏上的译文会得到
        负坐标被推出窗口，表现为"翻译层什么都不显示"。
        """
        physical = Rect(1920 + 300, 150, 150, 30)
        window_rect = self.mapper.to_window_coords(physical, self.hidpi)

        self.assertEqual(window_rect.x, 200)
        self.assertEqual(window_rect.y, 100)
        self.assertEqual(window_rect.width, 100)
        # 窗口逻辑尺寸为 1707×960，坐标必须落在其中
        self.assertGreaterEqual(window_rect.x, 0)
        self.assertLess(window_rect.x + window_rect.width, self.hidpi.logical.width)

    def test_to_window_coords_at_screen_origin(self) -> None:
        """屏幕原点处应映射为窗口左上角 (0, 0)。"""
        window_rect = self.mapper.to_window_coords(
            Rect(self.hidpi.physical.x, self.hidpi.physical.y, 90, 30),
            self.hidpi,
        )
        self.assertEqual((window_rect.x, window_rect.y), (0, 0))
        self.assertEqual(window_rect.width, 60)

    def test_roundtrip_physical_logical(self) -> None:
        """物理 ↔ 逻辑换算应当近似可逆（允许取整误差）。"""
        physical = Rect(1920 + 300, 150, 300, 60)
        logical = self.mapper.physical_to_logical(physical)
        back = self.mapper.logical_to_physical(logical)
        self.assertLessEqual(abs(back.x - physical.x), 2)
        self.assertLessEqual(abs(back.width - physical.width), 2)

    def test_capture_region_is_clamped_to_single_screen(self) -> None:
        """跨屏框选时捕获区域应被裁剪到单块显示器内。

        跨屏矩形的位图会被显示器间隙破坏，导致坐标换算出错，
        因此必须裁剪到归属的那块屏幕范围内。
        """
        spanning = Rect(1600, 500, 800, 300)
        clamped = self.mapper.capture_region_physical(spanning)
        self.assertGreaterEqual(clamped.left, self.hidpi.physical.left)
        self.assertLessEqual(clamped.right, self.hidpi.physical.right)
        self.assertLessEqual(clamped.bottom, self.hidpi.physical.bottom)

    def test_monitor_indices(self) -> None:
        """应能列出各显示器的 mss 序号。"""
        self.assertEqual(self.mapper.monitor_indices(), [1, 2])

    def test_is_single_screen(self) -> None:
        """单屏判定（用于跳过部分换算开销）。"""
        self.assertFalse(self.mapper.is_single_screen())
        self.mapper.set_screens([self.primary])
        self.assertTrue(self.mapper.is_single_screen())


if __name__ == "__main__":
    unittest.main(verbosity=2)
