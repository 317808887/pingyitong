"""热键全部注册失败时，面板与通知必须明确告知（不得静默）。

``hotkey_service.start()`` 全部失败时会把监听线程一并停掉，
``bindings()`` 随之返回空表——UI 若只看 bindings，会把"全被占用"
误展示成"未启用"，用户永远不知道"按了没反应"的原因。
部分失败（部分键被占用）已有面板+通知路径，全部失败的体验必须一致。
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from pingyitong.config.settings import SettingsManager
from pingyitong.services.hotkey_service import (
    ACTION_COPY_RESULT,
    ACTION_SWITCH_MODEL,
    ACTION_TOGGLE_OVERLAY,
    HotkeyBinding,
)
from pingyitong.ui.app import Application
from pingyitong.ui.design.theme import build_theme
from pingyitong.ui.main_panel import MainPanel
from pingyitong.ui.tray_icon import TrayIcon


def _controller(bindings: list) -> SimpleNamespace:
    """构造 ``_refresh_hotkey_status`` 所需的最小宿主。"""
    return SimpleNamespace(
        hotkeys=SimpleNamespace(bindings=lambda: bindings),
        tray=mock.Mock(),
        panel=mock.Mock(),
        notify=mock.Mock(),
    )


class TestHotkeyAllFailedNotice(unittest.TestCase):
    """start() 全失败必须写面板并发出通知。"""

    def test_total_failure_marks_panel_and_notifies(self) -> None:
        controller = _controller([])
        with mock.patch("pingyitong.ui.app.logger"):
            Application._refresh_hotkey_status(controller, False)
        controller.panel.set_hotkey_status.assert_called_once_with(
            [], all_failed=True
        )
        controller.notify.assert_called_once()
        self.assertEqual(
            controller.notify.call_args.kwargs.get("key"), "hotkey-all-failed"
        )
        self.assertEqual(controller.notify.call_args.args[2], "warning")

    def test_before_registration_keeps_quiet_placeholder(self) -> None:
        # 界面初次构建时还没调用 start()，空 bindings 是正常状态，
        # 此时不能误报"注册失败"打扰用户。
        controller = _controller([])
        with mock.patch("pingyitong.ui.app.logger"):
            Application._refresh_hotkey_status(controller)
        controller.panel.set_hotkey_status.assert_called_once_with(
            [], all_failed=False
        )
        controller.notify.assert_not_called()

    def test_success_never_notifies(self) -> None:
        bindings = [
            HotkeyBinding(
                action=ACTION_TOGGLE_OVERLAY, sequence="Ctrl+Alt+T", registered=True
            )
        ]
        controller = _controller(bindings)
        with mock.patch("pingyitong.ui.app.logger"):
            Application._refresh_hotkey_status(controller, True)
        controller.panel.set_hotkey_status.assert_called_once_with(
            bindings, all_failed=False
        )
        controller.notify.assert_not_called()

    def test_partial_conflict_keeps_existing_notice(self) -> None:
        bindings = [
            HotkeyBinding(
                action=ACTION_TOGGLE_OVERLAY, sequence="Ctrl+Alt+T", registered=True
            ),
            HotkeyBinding(
                action=ACTION_COPY_RESULT,
                sequence="Ctrl+Alt+C",
                registered=False,
                error="已被占用",
            ),
        ]
        controller = _controller(bindings)
        with mock.patch("pingyitong.ui.app.logger"):
            Application._refresh_hotkey_status(controller, True)
        controller.panel.set_hotkey_status.assert_called_once_with(
            bindings, all_failed=False
        )
        controller.notify.assert_called_once()
        self.assertEqual(
            controller.notify.call_args.kwargs.get("key"), "hotkey-conflict"
        )

    def test_fallback_is_reported_and_tray_receives_effective_binding(self) -> None:
        binding = HotkeyBinding(
            action=ACTION_SWITCH_MODEL, sequence="Ctrl+Alt+F1",
            registered=True, fallback_from="Ctrl+Alt+M",
        )
        controller = _controller([binding])
        with mock.patch("pingyitong.ui.app.logger"):
            Application._refresh_hotkey_status(controller, True)
        controller.tray.set_hotkey_status.assert_called_once_with([binding])
        self.assertEqual(controller.notify.call_args.kwargs.get("key"), "hotkey-fallback")


class TestHotkeyStatusPanelText(unittest.TestCase):
    """面板状态区文案：全失败是危险态，不能沿用"未启用"占位。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_all_failed_shows_danger_text_not_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = SettingsManager(Path(directory) / "config.json")
            controller = SimpleNamespace(theme=build_theme(), settings=settings)
            panel = MainPanel(controller)
            try:
                panel.set_hotkey_status([], all_failed=True)
                self.assertEqual(
                    panel._fast_path_hotkeys.text(),
                    "热键全部注册失败（可能被其他程序占用）",
                )
                self.assertEqual(panel._fast_path_hotkeys.objectName(), "Danger")
                self.assertEqual(panel._hotkey_hint.objectName(), "Danger")

                # 未尝试注册时仍展示原有占位文案，两条路径不互相污染。
                panel.set_hotkey_status([])
                self.assertEqual(
                    panel._fast_path_hotkeys.text(),
                    "未启用（或当前平台不支持全局热键）",
                )
            finally:
                panel.close()

    def test_fallback_shows_actual_shortcut(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = SettingsManager(Path(directory) / "config.json")
            panel = MainPanel(SimpleNamespace(theme=build_theme(), settings=settings))
            try:
                panel.set_hotkey_status([HotkeyBinding(
                    action=ACTION_SWITCH_MODEL, sequence="Ctrl+Alt+F1",
                    registered=True, fallback_from="Ctrl+Alt+M",
                )])
                self.assertIn("Ctrl+Alt+F1", panel._fast_path_hotkeys.text())
                self.assertIn("原 Ctrl+Alt+M 被占用", panel._fast_path_hotkeys.text())
            finally:
                panel.close()

    def test_tray_uses_registered_shortcut_only(self) -> None:
        tray = TrayIcon(build_theme())
        try:
            tray.set_hotkey_status([
                HotkeyBinding(
                    action=ACTION_SWITCH_MODEL, sequence="Ctrl+Alt+F1",
                    registered=True, fallback_from="Ctrl+Alt+M",
                ),
                HotkeyBinding(
                    action=ACTION_COPY_RESULT, sequence="Ctrl+Alt+C",
                    registered=False, error="被占用",
                ),
            ])
            self.assertEqual(tray._actions["model"].shortcut().toString(), "Ctrl+Alt+F1")
            self.assertTrue(tray._actions["copy"].shortcut().isEmpty())
        finally:
            tray.hide()


if __name__ == "__main__":
    unittest.main()
