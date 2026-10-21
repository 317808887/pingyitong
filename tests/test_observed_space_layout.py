"""Observed whitespace, real fonts and actual pixels of readable inline layout."""

from __future__ import annotations

import os
import random
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt6.QtCore import QRectF
from PyQt6.QtGui import QColor, QFont, QFontDatabase, QFontMetricsF, QImage, QPainter
from PyQt6.QtWidgets import QApplication

from pingyitong.components.capture.geometry import Rect, ScreenInfo
from pingyitong.components.render.overlay_window import OverlayManager, OverlayWindow
from pingyitong.components.render.render_item import RenderFrame, RenderItem
from pingyitong.components.render.slot_layout import right_to_left
from pingyitong.config.schema import StyleConfig
from pingyitong.services.reading_order import reading_text

SCREEN = ScreenInfo("OBSERVED", Rect(0, 0, 1280, 800), Rect(0, 0, 1280, 800), 1.0, True)


def image_pixels(image: QImage) -> np.ndarray:
    converted = image.convertToFormat(QImage.Format.Format_RGBA8888)
    buffer = converted.bits()
    buffer.setsize(converted.sizeInBytes())
    return np.frombuffer(buffer, dtype=np.uint8).reshape(
        converted.height(), converted.bytesPerLine() // 4, 4,
    )[:, :converted.width()].copy()


class TestObservedSpaceLayout(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])
        cls.original_font = cls.app.font()
        for filename in ("msyh.ttc", "segoeui.ttf", "arial.ttf", "seguiemj.ttf"):
            path = Path("C:/Windows/Fonts") / filename
            if path.exists():
                QFontDatabase.addApplicationFont(str(path))
        if "Microsoft YaHei" in QFontDatabase.families():
            cls.app.setFont(QFont("Microsoft YaHei"))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.app.setFont(cls.original_font)

    def setUp(self) -> None:
        self.window = OverlayWindow(SCREEN, StyleConfig(shadow=False))

    def tearDown(self) -> None:
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def draw(self, items: list[RenderItem] | None = None, ratio: float = 1.0) -> QImage:
        image = QImage(
            round(SCREEN.logical.width * ratio), round(SCREEN.logical.height * ratio),
            QImage.Format.Format_ARGB32_Premultiplied,
        )
        image.setDevicePixelRatio(ratio)
        image.fill(QColor(0, 0, 0, 0))
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        try:
            for item in self.window._items if items is None else items:
                self.window._paint_item(painter, item)
        finally:
            painter.end()
        return image

    def observe(self, bounds: Rect, items: list[RenderItem], sources=None) -> list[dict]:
        self.window.set_layout_context(bounds, items if sources is None else sources)
        self.window.set_items(items)
        return self.window.paint_evidence()

    def assert_disjoint_and_bounded(self) -> None:
        rects = [self.window._paint_rect_for(item) for item in self.window._items]
        for index, rect in enumerate(rects):
            if not rect.isEmpty():
                self.assertTrue(self.window._layout_bounds.contains(rect))
            for other in rects[index + 1:]:
                self.assertFalse(rect.intersects(other))

    def test_five_units_from_failed_small_faint_scene_get_complete_readable_slots(self) -> None:
        # Values preserve the actual r10 five OCR units (one contains two
        # source lines). This is an offline layout regression, not new OCR,
        # semantic, model or installed-software acceptance evidence.
        texts = [
            "レポートには19ページと8つの表が含まれています。",
            "テストが失敗した場合は、更新をインストールしないでください。",
            "管理者のみがこれらのファイルを削除できます。"
            "ウィンドウを閉じる前にファイルを保存してください。",
            "転送が完了するまで、ケーブルを切断しないでください。",
            "青いボタンは赤いボタンの上にあります。",
        ]
        boxes = [
            Rect(146, 400, 221, 16), Rect(146, 435, 211, 14),
            Rect(145, 468, 217, 50), Rect(143, 536, 307, 16),
            Rect(147, 571, 211, 13),
        ]
        items = [RenderItem(box, text) for box, text in zip(boxes, texts, strict=True)]
        details = self.observe(Rect(120, 380, 1120, 330), items)
        self.assertEqual({item["font_px"] for item in details}, {14})
        self.assertTrue(all(item["disposition"] == "translated" for item in details))
        self.assertTrue(all(not item["truncated"] for item in details))
        self.assertEqual(["".join(item["rendered_lines"]) for item in details], texts)
        self.assertTrue(all(item["inline_delivery"] == "full" for item in details))
        self.assert_disjoint_and_bounded()
        masks = []
        for item in self.window._items:
            pixels = image_pixels(self.draw([item]))
            ink = np.all(pixels[:, :, :3] > 200, axis=2) & (pixels[:, :, 3] > 0)
            self.assertGreater(int(ink.sum()), 50, "Actual text ink must exist, not only outlines")
            masks.append(pixels[:, :, 3] > 0)
        for index, mask in enumerate(masks):
            self.assertFalse(any(np.any(mask & other) for other in masks[index + 1:]))

    def test_untranslated_numeric_source_blocks_safe_expansion(self) -> None:
        item = RenderItem(Rect(30, 30, 60, 12), "A translated instruction that needs more space.")
        number = RenderItem(Rect(180, 22, 28, 34), "404", source_text="404")
        details = self.observe(Rect(20, 20, 300, 90), [item], [item, number])
        rect = self.window._paint_rect_for(self.window._items[0])
        self.assertFalse(rect.intersects(QRectF(180, 22, 28, 34)))
        pixels = image_pixels(self.draw())
        self.assertEqual(int(pixels[22:56, 180:208, 3].sum()), 0)
        self.assertIn(details[0]["layout_reason"], (
            "expanded_into_observed_space", "no_readable_space",
        ))

    def test_neighbor_column_is_reserved_including_different_text_heights(self) -> None:
        items = [
            RenderItem(Rect(20, 25, 110, 12), "左側の詳しい説明は隣の列を隠しません。"),
            RenderItem(Rect(250, 20, 160, 35), "Right column remains readable."),
            RenderItem(Rect(20, 80, 100, 12), "別の行も元の内容に対応します。"),
            RenderItem(Rect(250, 85, 170, 16), "A different row in the right column."),
        ]
        self.observe(Rect(10, 10, 440, 145), items)
        self.assert_disjoint_and_bounded()
        masks = [image_pixels(self.draw([item]))[:, :, 3] > 0 for item in self.window._items]
        self.assertFalse(any(np.any(a & b) for index, a in enumerate(masks) for b in masks[index + 1:]))
        self.assertTrue(all(self.window._get_layout(item).font.pixelSize() >= 14
                            for item in self.window._items))

    def test_no_space_keeps_source_and_full_panel_text_without_painted_ellipsis(self) -> None:
        full = "Only the supervisor may remove the encrypted archive after backup succeeds. " * 5
        item = RenderItem(Rect(20, 20, 100, 12), full, source_text="Original text")
        details = self.observe(Rect(20, 20, 100, 15), [item])
        self.assertEqual(details[0]["layout_reason"], "no_readable_space")
        self.assertEqual(details[0]["inline_delivery"], "none")
        self.assertEqual(details[0]["rendered_lines"], [])
        self.assertIn("insufficient_height", details[0]["layout_overflow_reasons"])
        self.assertIn("no_complete_line", details[0]["layout_overflow_reasons"])
        self.assertGreater(details[0]["layout_needed_lines"], 1)
        self.assertEqual(details[0]["full_text"], full)
        self.assertEqual(reading_text(self.window._items), full.strip())
        image = self.draw()
        self.assertEqual(image.pixelColor(60, 25).alpha(), 0)
        pixels = image_pixels(image)
        white = np.all(pixels[:, :, :3] > 230, axis=2) & (pixels[:, :, 3] > 0)
        self.assertEqual(int(white.sum()), 0)

    def test_reviewed_compact_and_full_panel_are_distinct_deliveries(self) -> None:
        full = "在安全检查与备份均完成以前，请不要关闭窗口，也不要移除设备。"
        compact = "检查及备份完成前，勿关窗或拔设备。"
        item = RenderItem(Rect(20, 20, 160, 15), full, display_text=compact)
        details = self.observe(Rect(10, 10, 380, 80), [item])
        self.assertEqual(details[0]["inline_delivery"], "compact")
        self.assertEqual("".join(details[0]["rendered_lines"]), compact)
        self.assertEqual(reading_text(self.window._items), full)
        self.assertEqual(details[0]["full_text"], full)
        self.assertTrue(details[0]["compact_generated"])
        # Render accepts a reviewed input field; this test does not assert
        # that the supplied two Chinese sentences are semantically equivalent.

    def test_rtl_expands_left_and_shapes_nonempty_right_aligned_ink(self) -> None:
        self.window.apply_style(StyleConfig(font_family="Arial", shadow=False))
        for text in ("لا تغلق النافذة قبل اكتمال النسخ.", "אין לסגור לפני סיום הגיבוי."):
            with self.subTest(text=text):
                item = RenderItem(Rect(500, 35, 30, 14), text)
                details = self.observe(Rect(20, 20, 530, 80), [item])
                self.assertTrue(right_to_left(text))
                self.assertEqual(details[0]["disposition"], "translated")
                rect = self.window._paint_rect_for(self.window._items[0])
                self.assertLess(rect.left(), 500)
                self.assertEqual(rect.right(), 530)
                metrics = QFontMetricsF(self.window._get_layout(self.window._items[0]).font)
                self.assertGreater(metrics.horizontalAdvance(text), 50)
                pixels = image_pixels(self.draw())
                white = np.all(pixels[:, :, :3] > 220, axis=2) & (pixels[:, :, 3] > 0)
                self.assertGreater(int(white.sum()), 60)
                self.assertEqual(int(pixels[:, 531:, 3].sum()), 0)

    def test_window_exclusion_over_anchor_reports_hidden_delivery(self) -> None:
        item = RenderItem(Rect(40, 40, 120, 25), "Readable translation")
        self.observe(Rect(20, 20, 400, 100), [item])
        self.window.set_excluded_regions([Rect(40, 40, 120, 25)])
        details = self.window.paint_evidence()
        self.assertEqual(details[0]["layout_reason"], "source_or_window_obstructs_anchor")
        self.assertEqual(details[0]["inline_delivery"], "none")
        self.window.set_excluded_regions([])
        self.assertEqual(self.window.paint_evidence()[0]["disposition"], "translated")

    def test_scaled_nonzero_monitor_uses_exact_capture_boundary(self) -> None:
        screen = replace(
            SCREEN, physical=Rect(-1920, -240, 1920, 1200),
            logical=Rect(-1920, -240, 1280, 800), device_pixel_ratio=1.5,
        )
        self.window.refresh_screen(screen)
        item = RenderItem(Rect(-1750, -120, 80, 14), "A line with meaningful content.")
        self.observe(Rect(-1801, -180, 301, 151), [item])
        bounds = self.window._layout_bounds
        self.assertAlmostEqual(bounds.left(), 119 / 1.5)
        self.assertAlmostEqual(bounds.right(), 420 / 1.5)
        self.assert_disjoint_and_bounded()
        for ratio in (1.5, 2.0):
            pixels = image_pixels(self.draw(ratio=ratio))
            ys, xs = np.nonzero(pixels[:, :, 3])
            self.assertGreater(len(xs), 0)
            self.assertGreaterEqual(xs.min() + 1, bounds.left() * ratio)
            self.assertLessEqual(xs.max(), bounds.right() * ratio)
            self.assertGreaterEqual(ys.min() + 1, bounds.top() * ratio)
            self.assertLessEqual(ys.max(), bounds.bottom() * ratio)

    def test_raw_source_lines_reach_allocator_even_if_filtered_from_render_sources(self) -> None:
        item = RenderItem(Rect(20, 30, 80, 12), "A much wider translated instruction.")
        frame = RenderFrame(1, Rect(10, 10, 400, 100), [item], source_items=[item])
        frame._source_lines = [
            {"text": "Original instruction", "rect": [20, 30, 80, 12]},
            {"text": "7", "rect": [150, 25, 15, 35]},
        ]
        manager = SimpleNamespace(
            _windows={SCREEN.name: self.window},
            apply=lambda items, *args: self.window.set_items(items) or len(items),
        )
        self.assertEqual(OverlayManager.apply_frame(manager, frame), 1)
        rect = self.window._paint_rect_for(self.window._items[0])
        self.assertFalse(rect.intersects(QRectF(150, 25, 15, 35)))
        self.assertEqual(len(self.window._source_context), 2)

    def test_partial_observations_preserve_unchanged_untranslated_sources(self) -> None:
        source_a = RenderItem(Rect(20, 20, 100, 15), "Source A")
        source_b = RenderItem(Rect(180, 20, 20, 25), "8")
        bounds = Rect(10, 10, 400, 100)
        self.window.set_layout_context(bounds, [source_a, source_b])
        self.window.set_layout_context(bounds, [source_a], False, [source_a.box])
        self.assertEqual({item.text for item in self.window._source_context}, {"Source A", "8"})
        self.window.set_layout_context(bounds, [], False, [source_b.box])
        self.assertEqual([item.text for item in self.window._source_context], ["Source A"])

    def test_paragraph_owns_consumed_source_lines_but_not_filtered_interior_number(self) -> None:
        item = RenderItem(
            Rect(20, 30, 100, 45), "A complete translated paragraph.",
            source_text="First source row second source row",
        )
        lines = [
            {"text": "First source row", "rect": [20, 30, 100, 12]},
            {"text": "second source row", "rect": [20, 60, 100, 15]},
        ]
        frame = RenderFrame(1, Rect(10, 10, 500, 130), [item])
        frame._source_lines = lines
        frame._paragraph_provenance = [{
            "text": item.source_text, "rect": item.box.to_sequence(), "source_lines": lines,
        }]
        manager = SimpleNamespace(
            _windows={SCREEN.name: self.window},
            apply=lambda items, *args: self.window.set_items(items) or len(items),
        )
        OverlayManager.apply_frame(manager, frame)
        self.assertEqual(self.window.paint_evidence()[0]["disposition"], "translated")
        # This extra source is geometrically inside the paragraph but was not
        # consumed by preprocessing. It must not disappear under its cover.
        frame._source_lines = [*lines, {"text": "404", "rect": [55, 47, 20, 10]}]
        OverlayManager.apply_frame(manager, frame)
        self.assertEqual(self.window.paint_evidence()[0]["inline_delivery"], "none")
        self.assertEqual(self.window.paint_evidence()[0]["layout_reason"],
                         "source_or_window_obstructs_anchor")
        self.assertEqual(int(image_pixels(self.draw())[47:57, 55:75, 3].sum()), 0)

    def test_source_line_ownership_survives_unrelated_partial_observation(self) -> None:
        source = RenderItem(Rect(20, 20, 80, 12), "First row")
        bounds = Rect(10, 10, 400, 130)
        provenance = [{
            "text": "First row second row", "rect": [20, 20, 100, 40],
            "source_lines": [{"text": "First row", "rect": source.box.to_sequence()}],
        }]
        self.window.set_layout_context(bounds, [source], provenance=provenance)
        owners = dict(self.window._source_owners)
        self.window.set_layout_context(bounds, [], False, [Rect(250, 90, 20, 20)])
        self.assertEqual(self.window._source_owners, owners)

    def test_allocation_is_deterministic_and_never_paints_outside_observed_roi(self) -> None:
        rng = random.Random(1240857193)
        for scene in range(20):
            items = [RenderItem(
                Rect(rng.randrange(20, 260), rng.randrange(20, 170),
                     rng.randrange(20, 120), rng.randrange(8, 30)),
                f"Scene {scene}, instruction {index}: retain the complete meaning.",
            ) for index in range(9)]
            assigned = []
            for order in (items, list(reversed(items))):
                self.observe(Rect(10, 10, 400, 220), order)
                self.assert_disjoint_and_bounded()
                assigned.append({item["full_text"]: (item["rect"], item["layout_reason"])
                                 for item in self.window.paint_evidence()})
            self.assertEqual(assigned[0], assigned[1])

    def test_cached_geometry_tracks_new_object_ids_source_obstacles_style_and_capture(self) -> None:
        item = RenderItem(Rect(20, 30, 90, 12), "A translated line with all its original meaning.")
        bounds = Rect(10, 10, 500, 120)
        initial = self.observe(bounds, [item])
        self.window.set_items([replace(item)])
        self.assertEqual(self.window.paint_evidence(), initial)
        blocker = RenderItem(Rect(130, 20, 30, 45), "404")
        self.observe(bounds, [item], [item, blocker])
        pixels = image_pixels(self.draw())
        self.assertEqual(int(pixels[20:65, 130:160, 3].sum()), 0)
        self.window.apply_style(StyleConfig(font_size_mode="fixed", font_size_fixed=20))
        self.assertEqual(self.window.paint_evidence()[0]["font_px"], 20)
        tiny = self.observe(Rect(20, 30, 90, 12), [item])
        self.assertEqual(tiny[0]["inline_delivery"], "none")
        self.assertEqual(tiny[0]["layout_reason"], "no_readable_space")


if __name__ == "__main__":
    unittest.main()
