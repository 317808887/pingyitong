"""Full reading text, reviewed compact overlay, and conservative fallbacks."""
import unittest
from dataclasses import replace

from PyQt6.QtGui import QColor, QImage, QPainter

from pingyitong.components.capture.geometry import Rect
from pingyitong.components.render.render_item import RenderItem
from pingyitong.services.reading_order import reading_text
from tests.test_overlay_readability import TestOverlayReadability as _OverlayFixture
from tests.test_overlay_readability import pixels


class TestReviewedDisplay(unittest.TestCase):
    setUpClass = classmethod(_OverlayFixture.setUpClass.__func__)
    tearDownClass = classmethod(_OverlayFixture.tearDownClass.__func__)
    setUp = _OverlayFixture.setUp
    tearDown = _OverlayFixture.tearDown
    _draw = _OverlayFixture._draw
    def test_compact_never_replaces_full_reading_copy(self):
        item = RenderItem(Rect(20, 20, 240, 40), "在备份完成以前，请不要关闭窗口。",
                          "Do not close the window before backup completes.",
                          display_text="备份完成前勿关窗口")
        self.window.set_items([item])
        actual = self.window._items[0]
        assert reading_text([actual]) == item.text
        assert "".join(self.window._get_layout(actual).lines) == item.display_text

    def test_uncertain_keeps_source_pixels_and_explains_in_reading_panel(self):
        item = RenderItem(Rect(20, 20, 240, 40), "Original source", "Original source",
                          quality_status="uncertain", quality_reason="condition")
        image = self._draw(item, "#abcdef")
        assert image.pixelColor(60, 40) == QColor("#abcdef")
        assert "不确定" in reading_text([item])
        assert reading_text([item]).endswith("Original source")

    def test_overflow_does_not_cover_source_with_incomplete_translation(self):
        item = RenderItem(Rect(20, 20, 80, 20), "只有验证身份且备份完成后才允许删除这份报告。" * 3)
        image = self._draw(item, "#abcdef")
        assert self.window._get_layout(item).truncated
        assert image.pixelColor(40, 30) == QColor("#abcdef")

    def test_overlapping_backgrounds_are_drawn_once_even_when_translucent(self):
        self.window.apply_style(replace(self.window.style_config, mode="overlay", shadow=False))
        self.window.set_items([
            RenderItem(Rect(20, 20, 240, 40), "译文甲"),
            RenderItem(Rect(20, 44, 240, 40), "译文乙"),
        ])
        image = QImage(520, 380, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(QColor(0, 0, 0, 0))
        painter = QPainter(image)
        for item in self.window._items:
            self.window._paint_item(painter, item)
        painter.end()
        # Background alpha alone never exceeds one application; glyph alpha
        # may be opaque, so compare only the far edge away from glyphs.
        assert pixels(image)[20:84, 245, 3].max() <= int(.62 * 255)
