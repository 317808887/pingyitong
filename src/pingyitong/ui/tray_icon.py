"""系统托盘。

对应开发文档一·（五）"UI 交互层：系统托盘图标、主设置面板、
悬浮控制条、原位翻译透明层"。

托盘是程序**唯一的常驻入口**：主面板可以关闭，翻译层可以隐藏，
但只要程序在运行，托盘图标就一定在。因此：

- 图标状态反映运行状态（运行中 / 已暂停 / 有错误），
  让用户不必打开面板就能判断程序是否在工作；
- 菜单项与面板、热键使用同一套动作，任何一处操作都会同步其他入口的状态；
- 关闭主面板时向用户说明"程序仍在后台运行"，
  避免用户误以为已退出而反复启动（这也正是单实例守卫要处理的场景）。
"""

from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QAction, QIcon, QKeySequence
from PyQt6.QtWidgets import QMenu, QSystemTrayIcon

from ..constants import (
    ICON_APP_ICO,
    ICON_APP_PNG,
    ICON_TRAY_ACTIVE,
    ICON_TRAY_IDLE,
    ICON_TRAY_PAUSED,
    MODEL_LABELS,
)
from ..utils import icon_path
from ..utils.logger import get_logger
from ..version import version_string
from .design.theme import Theme

logger = get_logger(__name__)


class TrayIcon(QObject):
    """系统托盘图标与菜单。"""

    #: 请求开启 / 暂停翻译
    toggle_translation = pyqtSignal()
    #: 请求切换模型
    switch_model = pyqtSignal()
    #: 请求框选区域
    pick_region = pyqtSignal()
    #: 请求显示 / 隐藏翻译层
    toggle_overlay = pyqtSignal()
    #: 请求复制当前译文
    copy_result = pyqtSignal()
    #: 请求打开设置面板
    open_settings = pyqtSignal()
    #: 请求打开历史记录
    open_history = pyqtSignal()
    open_web_reader = pyqtSignal()
    #: 请求退出程序
    quit_requested = pyqtSignal()
    #: 双击托盘图标（用于快速唤起面板）
    activated = pyqtSignal()

    def __init__(self, theme: Theme, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._theme = theme
        self._running = False
        self._overlay_visible = True
        self._model_id = ""
        self._status_text = ""

        self._tray = QSystemTrayIcon(self._load_icon(ICON_TRAY_PAUSED), parent)
        self._tray.setToolTip("屏译通")
        self._menu = QMenu()
        self._actions: dict[str, QAction] = {}
        self._build_menu()
        self._tray.setContextMenu(self._menu)
        self._tray.activated.connect(self._on_activated)

    # ------------------------------------------------------------------
    # 图标
    # ------------------------------------------------------------------

    def _load_icon(self, file_name: str) -> QIcon:
        """加载图标，缺失时按回退链降级，最终返回空图标。

        图标缺失不应导致程序无法启动——托盘功能可以降级，但程序必须能跑。
        """
        for candidate in (file_name, ICON_APP_ICO, ICON_APP_PNG):
            path = icon_path(candidate)
            if path.exists():
                icon = QIcon(str(path))
                if not icon.isNull():
                    return icon

        logger.warning("未找到托盘图标资源，托盘将显示为空白图标")
        return QIcon()

    def _refresh_icon(self) -> None:
        """按运行状态切换托盘图标。

        三种图标各自承载明确含义，用户无需打开面板即可判断程序状态：

        - 运行中：实心图标；
        - 已暂停：灰色图标；
        - 异常（如 OCR 组件缺失）：警告图标。
        """
        if self._status_text:
            file_name = ICON_TRAY_IDLE
        elif self._running:
            file_name = ICON_TRAY_ACTIVE
        else:
            file_name = ICON_TRAY_PAUSED
        self._tray.setIcon(self._load_icon(file_name))
        self._tray.setToolTip(self._build_tooltip())

    def _build_tooltip(self) -> str:
        """构造托盘悬停提示。"""
        lines = ["屏译通", version_string()]
        lines.append("状态：翻译中" if self._running else "状态：已暂停")
        if self._model_id:
            lines.append(f"模型：{MODEL_LABELS.get(self._model_id, self._model_id)}")
        if self._status_text:
            lines.append(self._status_text)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 菜单
    # ------------------------------------------------------------------

    def _build_menu(self) -> None:
        """构建右键菜单。

        菜单项顺序按使用频率排列：最常用的"开启/暂停"在最上，
        "退出"在最下并与其他项用分隔线隔开，避免误点。
        """
        self._actions["toggle"] = self._add_action(
            "开始翻译", self.toggle_translation.emit, shortcut="Ctrl+Alt+T"
        )
        self._actions["model"] = self._add_action(
            "切换模型", self.switch_model.emit, shortcut="Ctrl+Alt+M"
        )

        self._menu.addSeparator()

        self._actions["region"] = self._add_action("框选翻译区域…", self.pick_region.emit)
        self._actions["web"] = self._add_action("网页翻译…", self.open_web_reader.emit)
        self._actions["overlay"] = self._add_action(
            "隐藏翻译层", self.toggle_overlay.emit, shortcut="Ctrl+Alt+H"
        )
        self._actions["copy"] = self._add_action(
            "复制翻译结果", self.copy_result.emit, shortcut="Ctrl+Alt+C"
        )

        self._menu.addSeparator()

        self._actions["history"] = self._add_action("翻译历史…", self.open_history.emit)
        self._actions["settings"] = self._add_action("设置…", self.open_settings.emit)

        self._menu.addSeparator()

        self._actions["quit"] = self._add_action("退出屏译通", self.quit_requested.emit)

    def _add_action(
        self,
        text: str,
        callback,
        shortcut: str = "",
    ) -> QAction:
        """添加一个菜单项。"""
        action = QAction(text, self._menu)
        action.triggered.connect(callback)
        if shortcut:
            action.setShortcut(shortcut)
        self._menu.addAction(action)
        return action

    # ------------------------------------------------------------------
    # 状态同步
    # ------------------------------------------------------------------

    def set_running(self, running: bool) -> None:
        """同步运行状态到菜单文案与图标。"""
        self._running = bool(running)
        toggle = self._actions.get("toggle")
        if toggle is not None:
            # 文案描述"点击后会发生什么"，而不是当前状态——
            # 菜单项是动作而不是状态显示，用动作命名更不易误读。
            toggle.setText("暂停翻译" if running else "开始翻译")
        self._refresh_icon()

    def set_hotkey_status(self, bindings: list) -> None:
        """Show only shortcuts that actually registered with Windows."""
        from ..services.hotkey_service import (
            ACTION_COPY_RESULT,
            ACTION_SWITCH_MODEL,
            ACTION_TOGGLE_OVERLAY,
            ACTION_TOGGLE_TRANSLATION,
            format_sequence,
        )

        actions = {
            ACTION_TOGGLE_TRANSLATION: "toggle",
            ACTION_SWITCH_MODEL: "model",
            ACTION_TOGGLE_OVERLAY: "overlay",
            ACTION_COPY_RESULT: "copy",
        }
        registered = {binding.action: binding for binding in bindings if binding.registered}
        for action, key in actions.items():
            binding = registered.get(action)
            sequence = format_sequence(binding.sequence) if binding else ""
            self._actions[key].setShortcut(QKeySequence(sequence))

    def set_model(self, model_id: str) -> None:
        """同步模型选择。"""
        self._model_id = model_id
        action = self._actions.get("model")
        if action is not None:
            action.setText(f"切换模型（当前：{MODEL_LABELS.get(model_id, model_id)}）")
        self._tray.setToolTip(self._build_tooltip())

    def set_overlay_visible(self, visible: bool) -> None:
        """同步翻译层显示状态。"""
        self._overlay_visible = bool(visible)
        action = self._actions.get("overlay")
        if action is not None:
            action.setText("隐藏翻译层" if visible else "显示翻译层")

    def set_status_text(self, text: str) -> None:
        """同步运行信息（如错误提示）。

        图标逻辑绑定在状态文本上（有状态文本时显示未运行图标），
        因此文本变化时必须同步刷新图标——否则"预加载失败 → 用户开始
        翻译 → 预加载恢复"这条链上图标会一直停在未运行态，
        直到下一次暂停 / 恢复才纠正。
        """
        self._status_text = text
        self._refresh_icon()

    def set_theme(self, theme: Theme) -> None:
        """应用主题（托盘菜单随系统主题，此处仅记录）。"""
        self._theme = theme

    # ------------------------------------------------------------------
    # 显示
    # ------------------------------------------------------------------

    def show(self) -> bool:
        """显示托盘图标。

        :return: 成功返回 ``True``；系统不支持托盘时返回 ``False``。
        """
        if not QSystemTrayIcon.isSystemTrayAvailable():
            logger.warning("当前系统不支持系统托盘")
            return False
        self._tray.show()
        return True

    def hide(self) -> None:
        """隐藏托盘图标。"""
        self._tray.hide()

    def show_message(self, title: str, message: str) -> None:
        """通过系统气泡提示用户（用于"程序仍在后台运行"等场景）。"""
        if self._tray.isVisible():
            self._tray.showMessage(title, message, self._tray.icon(), 4000)

    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        """托盘图标交互。"""
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.activated.emit()
        elif reason == QSystemTrayIcon.ActivationReason.Trigger:
            # 单击：切换翻译开关，这是最高频的操作
            self.toggle_translation.emit()
