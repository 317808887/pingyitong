"""Real Qt entry, window switching and close behavior for the reading window."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QObject, Qt
from PyQt6.QtTest import QSignalSpy, QTest
from PyQt6.QtWidgets import QApplication

from pingyitong.config.settings import SettingsManager
from pingyitong.ui.app import Application
from pingyitong.ui.design import DURATION, SPACE
from pingyitong.ui.design.theme import build_theme
from pingyitong.ui.floating_bar import FloatingBar
from pingyitong.ui.main_panel import MainPanel
from pingyitong.ui.widgets.button import Button
from pingyitong.ui.widgets.transcript_panel import TranscriptPanel


class _Overlay:
    def __init__(self) -> None:
        self.is_visible = True
        self.visibility_changes: list[bool] = []

    def show_all(self, visible: bool) -> None:
        self.is_visible = visible
        self.visibility_changes.append(visible)


class TestTranscriptEntry(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.qt_app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.controller = Application.__new__(Application)
        QObject.__init__(self.controller)
        self.controller.theme = build_theme("dark")
        self.controller.settings = SettingsManager(Path(self.directory.name) / "config.json")
        self.controller.pipeline = mock.Mock(is_running=True)
        self.controller.overlay = _Overlay()
        self.controller._push_ignore_regions = mock.Mock()
        self.controller.floating_bar = FloatingBar(self.controller.theme)
        self.controller.floating_bar.move(40, 40)
        self.controller.floating_bar.show_bar(animate=False)
        self.controller.transcript_panel = TranscriptPanel(self.controller.theme)
        self.controller.panel = MainPanel(self.controller)
        # Same production signal route as Application._connect_signals.
        self.controller.transcript_panel.expanded_changed.connect(
            self.controller._on_transcript_toggled
        )
        self.controller.panel.show()
        self.qt_app.processEvents()

    def tearDown(self) -> None:
        self.controller.transcript_panel.expanded_changed.disconnect()
        self.controller.transcript_panel.hide_panel()
        for widget in (
            self.controller.panel,
            self.controller.transcript_panel,
            self.controller.floating_bar,
        ):
            widget.hide()
            widget.deleteLater()
        self.controller.deleteLater()
        self.qt_app.processEvents()
        self.directory.cleanup()

    def _entry(self) -> Button:
        matches = [
            button for button in self.controller.panel.findChildren(Button)
            if button.text() in ("文字栏", "收起文字栏")
        ]
        self.assertEqual(len(matches), 1, "主设置窗口必须有清楚的文字栏入口")
        return matches[0]

    def test_reading_window_can_be_found_by_native_window_switcher(self) -> None:
        panel = self.controller.transcript_panel
        flags = panel.windowFlags()
        self.assertEqual(flags & Qt.WindowType.WindowType_Mask, Qt.WindowType.Window)
        self.assertTrue(flags & Qt.WindowType.FramelessWindowHint)
        self.assertTrue(flags & Qt.WindowType.WindowStaysOnTopHint)
        self.assertEqual(panel.windowTitle(), "屏译通 · 实时译文")

    def test_main_panel_entry_clicks_toggle_reading_and_preserve_alignment(self) -> None:
        entry = self._entry()
        panel = self.controller.panel
        transcript = self.controller.transcript_panel
        bar = self.controller.floating_bar
        header_layout = entry.parentWidget().layout()
        self.assertEqual(
            header_layout.indexOf(entry), header_layout.indexOf(panel._toggle_button) - 1
        )
        self.assertEqual(entry.accessibleName(), "展开文字栏")

        QTest.mouseClick(entry, Qt.MouseButton.LeftButton)
        self.assertTrue(transcript.is_expanded)
        self.assertEqual(entry.text(), "收起文字栏")
        self.assertEqual(entry.accessibleName(), "收起文字栏")
        self.assertEqual(transcript.width(), bar.width())
        self.assertEqual(
            transcript.y(), bar.mapToGlobal(bar.rect().bottomLeft()).y() + SPACE["xs"]
        )
        self.assertEqual(self.controller.overlay.visibility_changes, [False])

        QTest.mouseClick(entry, Qt.MouseButton.LeftButton)
        self.assertFalse(transcript.is_expanded)
        self.assertEqual(entry.text(), "文字栏")
        self.assertEqual(entry.accessibleName(), "展开文字栏")
        self.assertEqual(self.controller.overlay.visibility_changes, [False, True])
        self.assertEqual(
            self.controller.pipeline.set_transcript_enabled.call_args_list,
            [mock.call(True), mock.call(False)],
        )

    def test_external_toggle_and_native_close_sync_entry_and_restore_overlay(self) -> None:
        entry = self._entry()
        transcript = self.controller.transcript_panel
        transcript.set_text("第一行译文。\n第二行译文。")
        changes = QSignalSpy(transcript.expanded_changed)

        self.controller.toggle_transcript_panel()
        self.assertEqual(entry.text(), "收起文字栏")
        QTest.qWait(DURATION["accordion"] + 30)
        # QWidget.close sends the same QCloseEvent as the window manager/Alt+F4.
        self.assertFalse(transcript.close(), "关闭阅读窗口应转为收起，保留应用运行")
        self.assertFalse(transcript.is_expanded)
        self.assertEqual(entry.text(), "文字栏")
        self.assertEqual(list(changes), [[True], [False]])
        self.assertTrue(self.controller.overlay.is_visible)
        self.assertTrue(self.controller.panel.isVisible())
        QTest.qWait(DURATION["accordion"] + 30)
        self.assertFalse(transcript.isVisible())

        QTest.mouseClick(entry, Qt.MouseButton.LeftButton)
        self.assertTrue(transcript.is_expanded)
        self.assertEqual(transcript.text(), "第一行译文。\n第二行译文。")
        self.assertEqual(entry.text(), "收起文字栏")

    def test_close_during_expand_restores_previously_hidden_overlay(self) -> None:
        self.controller.overlay.is_visible = False
        transcript = self.controller.transcript_panel
        changes = QSignalSpy(transcript.expanded_changed)
        self.controller.toggle_transcript_panel()

        self.assertFalse(transcript.close())
        self.assertFalse(transcript.is_expanded)
        self.assertFalse(self.controller.overlay.is_visible)
        self.assertEqual(list(changes), [[True], [False]])
        QTest.qWait(DURATION["accordion"] + 30)
        self.assertFalse(transcript.isVisible())
        # A repeated close must not emit another state transition.
        self.assertFalse(transcript.close())
        self.assertEqual(list(changes), [[True], [False]])


if __name__ == "__main__":
    unittest.main()
