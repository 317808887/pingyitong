"""Capture exclusion and dirty-region rendering must prevent OCR feedback."""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt6.QtCore import QEvent, QRect
from PyQt6.QtGui import QColor, QImage
from PyQt6.QtWidgets import QApplication, QWidget

from pingyitong.components.capture.geometry import Rect, ScreenInfo
from pingyitong.components.render import RenderItem
from pingyitong.components.render.overlay_window import OverlayManager, OverlayWindow
from pingyitong.config.schema import StyleConfig
from pingyitong.utils import capture_exclusion


class _AffinityApi:
    def __init__(self, initial=0, *, set_success=True, readback_success=True):
        self.affinity = initial
        self.set_success = set_success
        self.readback_success = readback_success
        self.set_calls = []
        self.get_calls = 0

    def IsWindow(self, _hwnd):  # noqa: N802
        return True

    def GetWindowDisplayAffinity(self, _hwnd, pointer):  # noqa: N802
        self.get_calls += 1
        pointer._obj.value = self.affinity
        return self.readback_success or self.get_calls == 1

    def SetWindowDisplayAffinity(self, hwnd, affinity):  # noqa: N802
        self.set_calls.append((hwnd, affinity))
        if self.set_success:
            self.affinity = affinity
        return self.set_success


class TestWindowCaptureAffinity(unittest.TestCase):
    def _exclude(self, api, *, platform="win32", build=19041, hwnd=123):
        with (
            patch.object(capture_exclusion.sys, "platform", platform),
            patch.object(capture_exclusion.sys, "getwindowsversion", return_value=SimpleNamespace(build=build), create=True),
            patch.object(capture_exclusion, "_window_api", return_value=api),
        ):
            return capture_exclusion.exclude_window_from_capture(hwnd)

    def test_success_requires_confirmed_capture_exclusion(self):
        api = _AffinityApi()
        self.assertTrue(self._exclude(api))
        self.assertEqual(api.set_calls, [(123, 0x11)])
        self.assertEqual(api.get_calls, 2)

    def test_already_excluded_does_not_reset_affinity(self):
        api = _AffinityApi(initial=0x11)
        self.assertTrue(self._exclude(api))
        self.assertEqual(api.set_calls, [])

    def test_successful_set_with_failed_readback_is_not_reported_success(self):
        self.assertFalse(self._exclude(_AffinityApi(readback_success=False)))

    def test_failed_set_is_not_reported_success(self):
        self.assertFalse(self._exclude(_AffinityApi(set_success=False)))

    def test_unsupported_platform_build_and_invalid_handle_do_not_call_windows(self):
        for kwargs in ({"platform": "linux"}, {"build": 18362}, {"hwnd": 0}):
            with self.subTest(kwargs=kwargs):
                api = _AffinityApi()
                self.assertFalse(self._exclude(api, **kwargs))
                self.assertEqual(api.set_calls, [])
                self.assertEqual(api.get_calls, 0)

    def test_missing_windows_capture_api_is_supported_fallback(self):
        with (
            patch.object(capture_exclusion.sys, "platform", "win32"),
            patch.object(capture_exclusion.sys, "getwindowsversion", return_value=SimpleNamespace(build=19041), create=True),
            patch.object(capture_exclusion, "_window_api", side_effect=OSError("unsupported")),
        ):
            self.assertFalse(capture_exclusion.exclude_window_from_capture(123))


class _OwnWindow(QWidget):
    __module__ = "pingyitong.ui.test_window"


class TestCaptureExclusionFilter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.filter = capture_exclusion.CaptureExclusionFilter(self.app)
        self.top = _OwnWindow()

    def tearDown(self):
        self.app.removeEventFilter(self.filter)
        self.top.close()
        self.filter.deleteLater()

    def test_own_top_level_show_and_handle_recreation_are_excluded(self):
        for event_type in (QEvent.Type.Show, QEvent.Type.WinIdChange):
            with self.subTest(event_type=event_type):
                # Create the handle before patching, isolating the dispatched event.
                hwnd = int(self.top.winId())
                with patch.object(capture_exclusion, "exclude_window_from_capture") as exclude:
                    self.assertFalse(self.filter.eventFilter(self.top, QEvent(event_type)))
                    exclude.assert_called_once_with(hwnd)

    def test_qt_children_and_external_widget_classes_are_ignored(self):
        child = _OwnWindow(self.top)
        external = QWidget()
        try:
            with patch.object(capture_exclusion, "exclude_window_from_capture") as exclude:
                self.filter.eventFilter(child, QEvent(QEvent.Type.Show))
                self.filter.eventFilter(external, QEvent(QEvent.Type.Show))
                self.filter.eventFilter(self.app, QEvent(QEvent.Type.Show))
                self.filter.eventFilter(self.top, QEvent(QEvent.Type.Resize))
                exclude.assert_not_called()
        finally:
            child.close()
            external.close()

    def test_handle_change_during_exclusion_cannot_reenter(self):
        int(self.top.winId())
        with patch.object(capture_exclusion, "exclude_window_from_capture") as exclude:
            exclude.side_effect = lambda _hwnd: self.filter.eventFilter(self.top, QEvent(QEvent.Type.WinIdChange))
            self.filter.eventFilter(self.top, QEvent(QEvent.Type.Show))
            exclude.assert_called_once()


SCREEN = ScreenInfo(
    name="scaled-secondary",
    logical=Rect(-1200, -200, 800, 600),
    physical=Rect(-1200, -200, 1200, 900),
    device_pixel_ratio=1.5, is_primary=False, mss_index=2,
)


class TestOverlayDirtyRegions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = OverlayWindow(SCREEN, StyleConfig())

    def tearDown(self):
        self.window.close()

    def test_blank_increment_deletes_old_translation_at_nonzero_scaled_origin(self):
        deleted = Rect(-1050, -50, 150, 30)
        kept = Rect(-600, -50, 150, 30)
        self.window.set_items([RenderItem(deleted, "old"), RenderItem(kept, "keep")])
        self.assertEqual(self.window._items[0].box, Rect(100, 100, 100, 20))
        self.window.set_items([], replace=False, dirty_regions=[deleted])
        self.assertEqual([item.text for item in self.window._items], ["keep"])

    def test_dirty_conversion_clips_other_monitor_and_rounds_outwards(self):
        regions = [Rect(-1051, -51, 152, 32), Rect(20, 20, 10, 10)]
        self.assertEqual(self.window._to_window_regions(regions), [Rect(99, 99, 102, 22)])

    def test_empty_increment_without_dirty_metadata_keeps_other_translations(self):
        self.window.set_items([RenderItem(Rect(-1050, -50, 150, 30), "keep")])
        self.window.set_items([], replace=False)
        self.assertEqual([item.text for item in self.window._items], ["keep"])

    def test_overlay_manager_dispatches_dirty_region_even_without_new_text(self):
        window = Mock()
        manager = SimpleNamespace(_visible=True, _screens=[SCREEN], _windows={SCREEN.name: window})
        deleted = Rect(-1050, -50, 150, 30)
        self.assertEqual(OverlayManager.apply(manager, [], replace=False, dirty_regions=[deleted]), 0)
        window.set_items.assert_called_once_with([], replace=False, dirty_regions=[deleted])

    def _paint_pixels(self):
        image = self.window.grab().toImage().convertToFormat(QImage.Format.Format_RGBA8888)
        buffer = image.bits()
        buffer.setsize(image.sizeInBytes())
        return np.frombuffer(buffer, dtype=np.uint8).reshape(image.height(), image.bytesPerLine() // 4, 4)[:, :image.width()].copy()

    def test_interactive_window_region_is_transparent_and_other_text_remains(self):
        blocked = Rect(-1050, -50, 150, 30)
        visible = Rect(-600, -50, 150, 30)
        self.window.set_items([RenderItem(blocked, "BLOCKED"), RenderItem(visible, "VISIBLE")])
        before = self._paint_pixels()
        self.assertGreater(int(before[100:120, 100:200, 3].sum()), 0)
        self.window.set_excluded_regions([blocked])
        after = self._paint_pixels()
        self.assertEqual(int(after[100:120, 100:200, 3].sum()), 0)
        self.assertGreater(int(after[100:120, 400:500, 3].sum()), 0)

    def test_paint_clip_blocks_text_or_shadow_extending_outside_source_box(self):
        blocked = Rect(-1050, -50, 150, 30)
        item = RenderItem(Rect(-1125, -50, 60, 30), "extends")
        self.window.set_items([item])
        self.window.set_excluded_regions([blocked])

        def extended_paint(painter, logical_item):
            box = logical_item.box
            painter.fillRect(QRect(box.x, box.y, box.width + 100, box.height), QColor("blue"))

        with patch.object(self.window, "_paint_item", side_effect=extended_paint):
            pixels = self._paint_pixels()
        self.assertGreater(int(pixels[100:120, 50:90, 3].sum()), 0)
        self.assertEqual(int(pixels[100:120, 100:190, 3].sum()), 0)

    def test_clearing_excluded_regions_restores_the_existing_translation(self):
        blocked = Rect(-1050, -50, 150, 30)
        self.window.set_items([RenderItem(blocked, "RESTORED")])
        self.window.set_excluded_regions([blocked])
        self.assertEqual(int(self._paint_pixels()[100:120, 100:200, 3].sum()), 0)
        self.window.set_excluded_regions([])
        self.assertGreater(int(self._paint_pixels()[100:120, 100:200, 3].sum()), 0)

    def test_manager_preserves_exclusions_for_hotplugged_screen_windows(self):
        blocked = Rect(-1050, -50, 150, 30)
        with patch("pingyitong.components.render.overlay_window.list_screens", return_value=[]):
            manager = OverlayManager(StyleConfig())
        manager.set_excluded_regions([blocked])
        with patch("pingyitong.components.render.overlay_window.list_screens", return_value=[SCREEN]):
            manager.sync_screens()
        try:
            self.assertEqual(manager._windows[SCREEN.name]._excluded_regions, [blocked])
        finally:
            manager.shutdown()


if __name__ == "__main__":
    unittest.main()
