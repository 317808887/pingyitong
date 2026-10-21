"""Regression coverage for readable, stable live translation output."""

from __future__ import annotations

import itertools
import unittest
from types import SimpleNamespace
from unittest import mock

from pingyitong.components.capture.geometry import Rect
from pingyitong.components.render.render_item import RenderItem
from pingyitong.services.reading_order import reading_text

try:
    from PyQt6.QtGui import QTextCursor
    from PyQt6.QtWidgets import QApplication

    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False


class TestTallBlockReadingOrder(unittest.TestCase):
    def test_tall_heading_does_not_reverse_two_separate_lines(self) -> None:
        items = [
            RenderItem(Rect(0, 0, 100, 60), "标题"),
            RenderItem(Rect(130, 5, 300, 20), "第一行"),
            RenderItem(Rect(110, 35, 300, 20), "第二行"),
        ]
        for permutation in itertools.permutations(items):
            with self.subTest(order=[item.text for item in permutation]):
                self.assertEqual(reading_text(list(permutation)), "标题\n第一行\n第二行")

    def test_overlapping_row_chain_does_not_move_bottom_line_above_top(self) -> None:
        items = [
            RenderItem(Rect(200, 0, 100, 20), "上行"),
            RenderItem(Rect(100, 10, 220, 20), "同行偏移"),
            RenderItem(Rect(0, 20, 310, 20), "下行"),
        ]
        for permutation in itertools.permutations(items):
            with self.subTest(order=[item.text for item in permutation]):
                actual = reading_text(list(permutation)).splitlines()
                self.assertLess(actual.index("上行"), actual.index("下行"))


@unittest.skipUnless(PYQT_AVAILABLE, "需要真实 PyQt6")
class TestLiveTranscriptReadability(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        from pingyitong.ui.design.theme import build_theme
        from pingyitong.ui.widgets.transcript_panel import TranscriptPanel

        self.panel = TranscriptPanel(build_theme("dark"))
        self.panel.resize(428, 260)
        self.panel.show()
        self.app.processEvents()
        self.lines = [f"第 {i:03d} 行：这是当前屏幕的译文。" for i in range(200)]
        self.original = "\n".join(self.lines)
        self.panel.set_text(self.original)
        self.app.processEvents()

    def tearDown(self) -> None:
        self.panel.hide()
        self.panel.deleteLater()
        self.app.processEvents()

    def _select(self, selected: str) -> None:
        # QTextCursor offsets count UTF-16 code units, including emoji pairs.
        text = self.panel.text()
        start = len(text[:text.index(selected)].encode("utf-16-le")) // 2
        cursor = self.panel._text.textCursor()
        cursor.setPosition(start)
        cursor.setPosition(
            start + len(selected.encode("utf-16-le")) // 2,
            QTextCursor.MoveMode.KeepAnchor,
        )
        self.panel._text.setTextCursor(cursor)

    def test_update_keeps_reader_scroll_position(self) -> None:
        scrollbar = self.panel._text.verticalScrollBar()
        scrollbar.setValue(40)
        self.panel.set_text(self.original.replace(self.lines[-1], "最后一行的新译文。"))
        self.app.processEvents()
        self.assertEqual(scrollbar.value(), 40)

    def test_update_retains_selection_in_unchanged_translation(self) -> None:
        selected = self.lines[50]
        self._select(selected)
        self.panel.set_text(self.original.replace(self.lines[-1], "最后一行的新译文。"))
        self.assertEqual(self.panel._text.textCursor().selectedText(), selected)

    def test_update_before_selection_preserves_selected_text(self) -> None:
        selected = self.lines[50]
        self._select(selected)
        updated = self.original.replace(self.lines[0], "首行的更新译文，比原文长一些。")
        self.panel.set_text(updated)
        self.assertEqual(self.panel.text(), updated)
        self.assertEqual(self.panel._text.textCursor().selectedText(), selected)

    def test_two_updated_lines_keep_selection_in_unchanged_middle(self) -> None:
        selected = self.lines[50]
        self._select(selected)
        updated = self.original.replace(self.lines[0], "首行更新。")
        updated = updated.replace(self.lines[-1], "末行也更新。")
        self.panel.set_text(updated)
        self.assertEqual(self.panel.text(), updated)
        self.assertEqual(self.panel._text.textCursor().selectedText(), selected)

    def test_new_top_row_and_removed_bottom_row_keep_shifted_selection(self) -> None:
        selected = self.lines[50]
        self._select(selected)
        updated = "新增标题行\n" + "\n".join(self.lines[:-1])
        self.panel.set_text(updated)
        self.assertEqual(self.panel.text(), updated)
        self.assertEqual(self.panel._text.textCursor().selectedText(), selected)

    def test_emoji_update_does_not_split_surrogate_or_remove_following_text(self) -> None:
        original = "🌏 第一行译文。\n第二行译文。\n第三行译文。"
        self.panel.set_text(original)
        self._select("第三行译文。")
        updated = "🌏 第一行更新后的译文。\n第二行译文。\n第三行译文。"
        self.panel.set_text(updated)
        self.assertEqual(self.panel.text(), updated)
        self.assertEqual(self.panel._text.textCursor().selectedText(), "第三行译文。")

    def test_repeated_middle_lines_keep_selection_when_both_ends_change(self) -> None:
        original = "首行\n" + "重复的译文行\n" * 1000 + "末行"
        self.panel.set_text(original)
        self._select("重复的译文行")
        updated = original.replace("首行", "更新后的首行").replace("末行", "更新后的末行")
        self.panel.set_text(updated)
        self.assertEqual(self.panel.text(), updated)
        self.assertEqual(self.panel._text.textCursor().selectedText(), "重复的译文行")

    def test_append_keeps_following_new_translation_at_bottom(self) -> None:
        scrollbar = self.panel._text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        self.panel.set_text(self.original + "\n新增的一行译文。")
        self.app.processEvents()
        self.assertEqual(scrollbar.value(), scrollbar.maximum())

    def test_identical_update_keeps_selection_and_emits_no_document_change(self) -> None:
        self._select(self.lines[50])
        changes = []
        self.panel._text.document().contentsChanged.connect(lambda: changes.append(True))
        self.panel.set_text(self.original)
        self.assertFalse(changes)
        self.assertEqual(self.panel._text.textCursor().selectedText(), self.lines[50])


@unittest.skipUnless(PYQT_AVAILABLE, "需要真实 PyQt6")
class TestFloatingStatusReadability(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_long_latency_fits_and_complete_value_remains_available(self) -> None:
        from pingyitong.ui.design.theme import build_theme
        from pingyitong.ui.floating_bar import FloatingBar

        bar = FloatingBar(build_theme("dark"))
        try:
            status = "12500ms · 准确模式"
            bar.set_status(status)
            label = bar._status_label
            self.assertLessEqual(
                label.fontMetrics().horizontalAdvance(label.text()),
                label.contentsRect().width(),
            )
            self.assertEqual(label.toolTip(), status)
            self.assertEqual(label.accessibleName(), status)
            bar.set_running(False)
            self.assertEqual(label.text(), "已暂停")
            self.assertEqual(label.toolTip(), "已暂停")
        finally:
            bar.deleteLater()


@unittest.skipUnless(PYQT_AVAILABLE, "需要真实 PyQt6")
class TestCopyCurrentScreenTranslations(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_copy_contains_unchanged_rows_and_current_row_in_reading_order(self) -> None:
        from pingyitong.ui.app import Application

        items = [
            RenderItem(Rect(100, 80, 300, 20), "增量更新后的末行"),
            RenderItem(Rect(100, 0, 300, 20), "未变化的首行"),
            RenderItem(Rect(100, 40, 300, 20), "未变化的中间行"),
            RenderItem(Rect(100, 100, 300, 20), " "),
        ]
        controller = SimpleNamespace(
            _qt_app=self.app,
            pipeline=SimpleNamespace(snapshot_items=lambda: list(items)),
            translator=SimpleNamespace(last_output_text=lambda: "仅最新的增量行"),
            notify=mock.Mock(),
        )
        Application.copy_result(controller)
        self.assertEqual(
            self.app.clipboard().text(), "未变化的首行\n未变化的中间行\n增量更新后的末行"
        )
        controller.notify.assert_called_once_with("已复制翻译结果", "共 3 条", "success")

    def test_cleared_snapshot_cannot_copy_previous_translation(self) -> None:
        from pingyitong.ui.app import Application

        self.app.clipboard().setText("用户已有的剪贴板内容")
        controller = SimpleNamespace(
            _qt_app=self.app,
            pipeline=SimpleNamespace(snapshot_items=lambda: []),
            translator=SimpleNamespace(last_output_text=lambda: "过期译文"),
            notify=mock.Mock(),
        )
        Application.copy_result(controller)
        self.assertEqual(self.app.clipboard().text(), "用户已有的剪贴板内容")
        controller.notify.assert_called_once_with(
            "暂无可复制的译文", "请先开始翻译并等待识别结果", "warning"
        )


if __name__ == "__main__":
    unittest.main()
