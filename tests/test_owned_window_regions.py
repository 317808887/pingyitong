"""Interactive translator windows must not be covered by underlying translations."""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QEvent
from PyQt6.QtWidgets import QApplication, QWidget

from pingyitong.components.capture.geometry import Rect, ScreenInfo
from pingyitong.components.render.overlay_window import OverlayWindow
from pingyitong.config.schema import StyleConfig
from pingyitong.ui.app import Application
from pingyitong.utils.capture_exclusion import CaptureExclusionFilter


class _SettingsWindow(QWidget):
    __module__ = "pingyitong.ui.synthetic_settings"


class TestOwnedWindowRegions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_settings_geometry_clips_existing_overlay_and_exposed_pixels_are_refreshed(self):
        settings = _SettingsWindow()
        unrelated = QWidget()
        bar = QWidget()
        screen = ScreenInfo("test", Rect(0, 0, 800, 600), Rect(0, 0, 800, 600), 1.0, True)
        translated = OverlayWindow(screen, StyleConfig())
        settings.setGeometry(80, 100, 300, 220)
        unrelated.setGeometry(400, 100, 100, 100)
        settings.show()
        unrelated.show()
        translated.show()
        pipeline, overlay = Mock(), Mock()
        controller = SimpleNamespace(
            _qt_app=self.app, floating_bar=bar,
            transcript_panel=SimpleNamespace(ignorable_rect=None), pipeline=pipeline,
            overlay=overlay, coordinate=SimpleNamespace(logical_to_physical=lambda rect: rect),
        )
        try:
            Application._push_ignore_regions(controller)
            rects = pipeline.set_ignore_regions.call_args.args[0]
            outer = settings.frameGeometry()
            expected = Rect(outer.x(), outer.y(), outer.width(), outer.height())
            self.assertIn(expected, rects)
            self.assertNotIn(Rect(400, 100, 100, 100), rects)
            self.assertNotIn(screen.physical, rects)
            overlay.set_excluded_regions.assert_called_with(rects)
            pipeline.invalidate.assert_called_once()
            Application._push_ignore_regions(controller)
            pipeline.invalidate.assert_called_once()
            settings.hide()
            Application._push_ignore_regions(controller)
            self.assertNotIn(expected, pipeline.set_ignore_regions.call_args.args[0])
            self.assertEqual(pipeline.invalidate.call_count, 2)
        finally:
            for window in (settings, unrelated, bar, translated):
                window.close()

    def test_show_hide_move_resize_notify_geometry_but_only_show_excludes_capture(self):
        filter_ = CaptureExclusionFilter(self.app)
        window = _SettingsWindow()
        changed = Mock()
        filter_.windows_changed.connect(changed)
        try:
            int(window.winId())
            changed.reset_mock()
            with patch("pingyitong.utils.capture_exclusion.exclude_window_from_capture") as exclude:
                for kind in (QEvent.Type.Show, QEvent.Type.Hide, QEvent.Type.Move, QEvent.Type.Resize):
                    filter_.eventFilter(window, QEvent(kind))
                self.assertEqual(changed.call_count, 4)
                exclude.assert_called_once()
        finally:
            self.app.removeEventFilter(filter_)
            window.close()
            filter_.deleteLater()


if __name__ == "__main__":
    unittest.main()
