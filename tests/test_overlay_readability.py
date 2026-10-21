"""Real Qt measurement and pixels for dense in-place translations."""

from __future__ import annotations

import os
import random
import unittest
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt6.QtCore import QRectF
from PyQt6.QtGui import QColor, QFont, QFontDatabase, QFontMetricsF, QImage, QPainter
from PyQt6.QtWidgets import QApplication

from pingyitong.components.capture.geometry import Rect, ScreenInfo
from pingyitong.components.render.layout import build_font, layout_text, wrap_text
from pingyitong.components.render.overlay_window import OverlayWindow
from pingyitong.components.render.render_item import RenderItem
from pingyitong.config.schema import StyleConfig

SCREEN = ScreenInfo("DENSE", Rect(0, 0, 520, 380), Rect(0, 0, 520, 380), 1.0, True)


def pixels(image: QImage) -> np.ndarray:
    converted = image.convertToFormat(QImage.Format.Format_RGBA8888)
    buffer = converted.bits()
    buffer.setsize(converted.sizeInBytes())
    return np.frombuffer(buffer, dtype=np.uint8).reshape(
        converted.height(), converted.bytesPerLine() // 4, 4
    )[:, :converted.width()].copy()


class TestOverlayReadability(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])
        cls.original_font = cls.app.font()
        # Windows' offscreen Qt backend has no native font discovery. Load real
        # installed fonts so widths/ink are measured rather than tofu boxes.
        for filename in ("msyh.ttc", "segoeui.ttf", "seguiemj.ttf"):
            path = Path("C:/Windows/Fonts") / filename
            if path.exists():
                QFontDatabase.addApplicationFont(str(path))
        if "Microsoft YaHei" in QFontDatabase.families():
            cls.app.setFont(QFont("Microsoft YaHei"))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.app.setFont(cls.original_font)

    def setUp(self) -> None:
        self.window = OverlayWindow(SCREEN, StyleConfig())

    def tearDown(self) -> None:
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def _draw(self, item: RenderItem, background: str | None = None) -> QImage:
        image = QImage(520, 380, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(QColor(background) if background else QColor(0, 0, 0, 0))
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        try:
            self.window._paint_item(painter, item)
        finally:
            painter.end()
        return image

    def test_long_translation_fits_padded_height_and_signals_omitted_content(self) -> None:
        item = RenderItem(Rect(20, 20, 110, 20), "这是明显比原文更长的译文，用于验证不会遮住下一行。" * 2)
        cached = self.window._get_layout(item)
        self.assertIsNotNone(cached)
        self.assertLessEqual(cached.layout.total_height, 18.0)
        self.assertLessEqual(cached.layout.content_width, 104.0)
        self.assertTrue(cached.truncated)
        shown = cached.elided if cached.elided is not None else cached.lines
        self.assertTrue(shown[-1].endswith("…"), "空间不足必须明确标记省略")
        self.assertEqual(item.text.count("验证"), 2, "原始完整译文不得被布局改写")

    def test_fixed_font_is_preserved_but_visible_lines_fit_height(self) -> None:
        self.window.apply_style(StyleConfig(font_size_mode="fixed", font_size_fixed=14))
        cached = self.window._get_layout(RenderItem(Rect(20, 20, 120, 30), "译文较长，需要换行显示。" * 4))
        self.assertEqual(cached.font.pixelSize(), 14)
        self.assertLessEqual(cached.layout.total_height, 28.0)
        self.assertTrue(cached.lines[-1].endswith("…"))

    def test_font_override_is_measured_with_the_font_actually_drawn(self) -> None:
        item = RenderItem(Rect(20, 20, 300, 90), "根据指定字号重新断行" * 4, font_px=28)
        cached = self.window._get_layout(item)
        self.assertEqual(cached.font.pixelSize(), 28)
        metrics = QFontMetricsF(cached.font)
        self.assertGreaterEqual(cached.line_height, metrics.height())
        self.assertLessEqual(cached.layout.total_height, 88.0)
        self.assertTrue(all(metrics.horizontalAdvance(line) <= 294.0 for line in cached.lines))

    def test_emoji_wrap_preserves_every_unicode_character(self) -> None:
        text = "😀abc😀def🌍中文🙂English" * 3
        lines = wrap_text(text, build_font("", 15), 75)
        self.assertEqual("".join(lines), text)
        metrics = QFontMetricsF(build_font("", 15))
        self.assertTrue(all(metrics.horizontalAdvance(line) <= 75.0 for line in lines))

    def test_explicit_newlines_make_distinct_rows(self) -> None:
        self.assertEqual(wrap_text("第一行\n第二行\n\n第四行", build_font("", 12), 300), [
            "第一行", "第二行", "", "第四行",
        ])

    def test_low_line_spacing_never_stacks_glyphs_within_one_translation(self) -> None:
        style = StyleConfig(font_size_mode="fixed", font_size_fixed=18, line_spacing=0.8)
        result = layout_text("第一行\n第二行\n第三行", 220, 120, style)
        self.assertGreaterEqual(result.line_height, QFontMetricsF(build_font("", 18)).height())

    def test_dense_rows_paint_no_glyph_or_shadow_into_neighbors(self) -> None:
        self.window.apply_style(StyleConfig(mode="overlay", bg_opacity=0.0, shadow=True))
        for index in range(10):
            item = RenderItem(Rect(20, 15 + index * 24, 100, 20), "密集页面较长的译文内容，不能画到下一行。" * 2)
            drawn = pixels(self._draw(item))[:, :, 3]
            allowed = np.zeros(drawn.shape, dtype=bool)
            allowed[item.box.y:item.box.bottom, item.box.x:item.box.right] = True
            self.assertEqual(int(np.count_nonzero(drawn[~allowed])), 0)
            self.assertGreater(int(np.count_nonzero(drawn[allowed])), 0)

    def test_cover_hides_original_tail_beyond_shorter_translation(self) -> None:
        item = RenderItem(Rect(20, 20, 280, 28), "短译文")
        image = self._draw(item, "white")
        self.assertEqual(image.pixelColor(280, 34).name(), "#101418")
        for x, y in ((20, 20), (299, 20), (20, 47), (299, 47)):
            self.assertEqual(image.pixelColor(x, y).name(), "#101418")

    def test_overlapping_source_boxes_do_not_paint_two_translations_on_same_pixels(self) -> None:
        self.window.apply_style(StyleConfig(
            mode="overlay", bg_opacity=0.0, shadow=False,
            font_size_mode="fixed", font_size_fixed=18,
        ))
        items = [
            RenderItem(Rect(20, 20, 160, 26), "WWWWWW"),
            RenderItem(Rect(20, 27, 160, 26), "WWWWWW"),
        ]
        self.window.set_items(items)
        first, second = self.window._items
        a = pixels(self._draw(first))[:, :, 3] > 0
        b = pixels(self._draw(second))[:, :, 3] > 0
        self.assertGreater(int(np.count_nonzero(a)), 0)
        self.assertGreater(int(np.count_nonzero(b)), 0)
        self.assertEqual(int(np.count_nonzero(a & b)), 0)

    def test_collision_assignment_is_stable_under_input_order_and_shrinks_only(self) -> None:
        items = [
            RenderItem(Rect(20, 20, 150, 28), "上行"),
            RenderItem(Rect(22, 38, 150, 28), "下行"),
            RenderItem(Rect(155, 20, 120, 28), "另一列"),
        ]
        assigned = []
        for order in (items, list(reversed(items))):
            self.window.set_items(order)
            mapping = {}
            rects = []
            for item in self.window._items:
                rect = self.window._paint_rect_for(item)
                self.assertTrue(QRectF(item.box.x, item.box.y, item.box.width, item.box.height).contains(rect))
                mapping[item.text] = (rect.x(), rect.y(), rect.width(), rect.height())
                rects.append(rect)
            for index, rect in enumerate(rects):
                for other in rects[index + 1:]:
                    self.assertFalse(rect.intersects(other))
            assigned.append(mapping)
        self.assertEqual(assigned[0], assigned[1])

    def test_scaled_monitor_collision_assignment_uses_logical_coordinates(self) -> None:
        screen = replace(SCREEN, logical=Rect(-600, -100, 520, 380), physical=Rect(-600, -100, 780, 570), device_pixel_ratio=1.5)
        self.window.refresh_screen(screen)
        self.window.set_items([
            RenderItem(Rect(-570, -70, 240, 39), "第一行"),
            RenderItem(Rect(-570, -59, 240, 39), "第二行"),
        ])
        rects = [self.window._paint_rect_for(item) for item in self.window._items]
        self.assertFalse(rects[0].intersects(rects[1]))
        self.assertEqual(min(rect.x() for rect in rects), 20.0)

    def test_cover_masks_overlaps_before_drawing_any_translated_glyph(self) -> None:
        self.window.apply_style(StyleConfig(
            mode="cover", shadow=False, font_size_mode="auto", min_font_size=9,
        ))
        self.window.set_items([
            RenderItem(Rect(20, 20, 160, 46), "WWWWWW"),
            RenderItem(Rect(20, 47, 160, 46), "WWWWWW"),
        ])
        first = self.window._items[0]
        single = pixels(self._draw(first))
        white = np.all(single[:, :, :3] > 200, axis=2)
        self.assertGreater(int(np.count_nonzero(white)), 0)
        combined = pixels(self.window.grab().toImage())
        self.assertTrue(np.all(np.all(combined[:, :, :3] > 200, axis=2)[white]))

    def test_padding_and_large_shadow_cannot_cross_a_touching_column(self) -> None:
        self.window.apply_style(StyleConfig(
            mode="overlay", bg_opacity=0.62, padding_x=8, padding_y=3,
            shadow=True, shadow_offset=6, font_size_mode="fixed", font_size_fixed=16,
        ))
        self.window.set_items([
            RenderItem(Rect(20, 20, 110, 30), "左侧较长的译文" * 3),
            RenderItem(Rect(130, 20, 110, 30), "右侧较长的译文" * 3),
        ])
        a, b = [pixels(self._draw(item))[:, :, 3] > 0 for item in self.window._items]
        self.assertEqual(int(np.count_nonzero(a & b)), 0)
        self.assertEqual(int(np.count_nonzero(a[:, 130:])), 0)
        self.assertEqual(int(np.count_nonzero(b[:, :130])), 0)

    def test_varied_dense_box_collisions_keep_slots_disjoint(self) -> None:
        rng = random.Random(20261012)
        for scene in range(40):
            items = [RenderItem(
                Rect(rng.randrange(0, 300), rng.randrange(0, 120), rng.randrange(10, 160), rng.randrange(8, 90)),
                f"场景{scene}条目{index}",
            ) for index in range(25)]
            self.window.set_items(items)
            rects = [self.window._paint_rect_for(item) for item in self.window._items]
            for index, rect in enumerate(rects):
                for other in rects[index + 1:]:
                    self.assertFalse(rect.intersects(other))

    def test_source_copies_keep_geometry_but_draw_no_repeated_original(self) -> None:
        for text, source in (
            ("文件管理 程序", "文件管理\n程序"),
            ("C:\\Windows\\Fonts", "C:\\Windows\\Fonts"),
            ("已经是中文", "  已经是中文  "),
        ):
            with self.subTest(source=source):
                item = RenderItem(Rect(20, 20, 160, 28), text, source_text=source)
                self.window.set_items([item])
                self.assertEqual(len(self.window._items), 1)
                self.assertEqual(self.window._items[0].source_text, source)
                self.assertEqual(int(np.count_nonzero(pixels(self._draw(self.window._items[0]))[:, :, 3])), 0)
                self.assertEqual(int(np.count_nonzero(pixels(self.window.grab().toImage())[:, :, 3])), 0)

    def test_dirty_source_copy_replaces_old_translation_without_erasing_other_regions(self) -> None:
        changed = Rect(20, 20, 160, 28)
        kept = Rect(20, 80, 160, 28)
        self.window.set_items([
            RenderItem(changed, "打开菜单", source_text="Open menu"),
            RenderItem(kept, "关闭窗口", source_text="Close window"),
        ])
        self.window.show()
        self.app.processEvents()
        before = pixels(self.window.grab().toImage())[:, :, 3]
        self.assertGreater(int(np.count_nonzero(before[20:48, 20:180])), 0)
        unchanged_source = RenderItem(changed, "文件菜单", source_text="文件菜单")
        self.window.set_items([unchanged_source], replace=False, dirty_regions=[changed])
        self.app.processEvents()
        after = pixels(self.window.grab().toImage())[:, :, 3]
        self.assertEqual(int(np.count_nonzero(after[20:48, 20:180])), 0)
        self.assertGreater(int(np.count_nonzero(after[80:108, 20:180])), 0)
        self.assertEqual({item.text for item in self.window._items}, {"文件菜单", "关闭窗口"})

    def test_high_dpi_backing_image_keeps_glyph_and_shadow_inside_source_box(self) -> None:
        self.window.apply_style(StyleConfig(mode="overlay", bg_opacity=0.0, shadow_offset=3))
        item = RenderItem(Rect(20, 20, 100, 20), "这段译文很长，不能侵入相邻的原位译文。" * 2)
        for ratio in (1.5, 2.0):
            with self.subTest(ratio=ratio):
                image = QImage(round(520 * ratio), round(380 * ratio), QImage.Format.Format_ARGB32_Premultiplied)
                image.setDevicePixelRatio(ratio)
                image.fill(QColor(0, 0, 0, 0))
                painter = QPainter(image)
                try:
                    self.window._paint_item(painter, item)
                finally:
                    painter.end()
                drawn = pixels(image)[:, :, 3]
                allowed = np.zeros(drawn.shape, dtype=bool)
                allowed[round(item.box.top * ratio):round(item.box.bottom * ratio), round(item.box.left * ratio):round(item.box.right * ratio)] = True
                self.assertEqual(int(np.count_nonzero(drawn[~allowed])), 0)
                self.assertGreater(int(np.count_nonzero(drawn[allowed])), 0)


if __name__ == "__main__":
    unittest.main()
