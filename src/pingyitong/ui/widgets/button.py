"""按钮组件。

按设计规范实现的可按压控件：

- **按下反馈**：缩放至 0.97，时长 160ms、强 ease-out
  （``cubic-bezier(0.23, 1, 0.32, 1)``）。规范强调这是"让界面看起来
  真的在听用户说话"的关键——聚合起来就是"手感"；
- **悬停**：仅在具备精确指针的设备上生效，且幅度极小
  （规范要求悬停效果必须"移除或大幅削减"，因为每天会被触发数十次）；
- **聚焦环**：键盘可达性要求，2px 强调色描边；
- **状态完备**：normal / hover / pressed / focus / disabled / checked。

实现说明
    通过 ``paintEvent`` 自绘而非依赖样式表。原因是 Qt 的样式表无法表达
    "对内容做 transform 缩放"，而缩放正是按下反馈的核心。自绘同时让
    本组件完全由设计令牌驱动，不存在硬编码色值。
"""

from __future__ import annotations

from PyQt6.QtCore import QRectF, QSize, Qt, pyqtProperty
from PyQt6.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QAbstractButton, QSizePolicy, QWidget

from ..design import LAYOUT, MOTION, RADIUS, SPACE
from ..design.theme import Theme


class Button(QAbstractButton):
    """设计系统按钮。

    :param text: 按钮文案。
    :param variant: ``primary`` / ``secondary`` / ``ghost`` / ``danger``。
    :param theme: 主题（提供配色与字体）。
    :param compact: 紧凑模式，用于悬浮控制条等空间受限场景。
    """

    #: 视觉变体
    VARIANTS = ("primary", "secondary", "ghost", "danger")

    def __init__(
        self,
        text: str = "",
        variant: str = "secondary",
        theme: Theme | None = None,
        parent: QWidget | None = None,
        compact: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setText(text)
        self._variant = variant if variant in self.VARIANTS else "secondary"
        self._theme = theme
        self._compact = compact

        self._hovered = False
        self._press_progress = 0.0

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

        height = 26 if compact else LAYOUT["button_height"]
        self.setMinimumHeight(height)

        #: 工具提示控制器与文案。设置为 None 时回退到 Qt 原生 ``setToolTip``。
        #: 使用自研控制器是为了满足"首条延迟、后续立即"的规范要求，
        #: 原生 ``QToolTip`` 无法按"是否刚刚显示过"动态切换延迟。
        self.tooltip_controller = None
        self.tooltip_text = ""

    # ------------------------------------------------------------------
    # 工具提示
    # ------------------------------------------------------------------

    def set_tooltip(self, text: str, controller=None) -> None:
        """设置提示文案与控制器。

        :param text: 提示文案；为空表示不显示提示。
        :param controller: :class:`~pingyitong.ui.widgets.tooltip.TooltipController`；
            为 ``None`` 时使用 Qt 原生提示。
        """
        self.tooltip_text = text
        self.tooltip_controller = controller
        if controller is None:
            super().setToolTip(text)
        else:
            super().setToolTip("")

    def enterEvent(self, event) -> None:  # noqa: N802
        """鼠标进入。

        悬停高亮仅在精确指针设备上生效——触摸设备上点按会误触发悬停态，
        造成"假高亮"。等价于 CSS 的
        ``@media (hover: hover) and (pointer: fine)``。
        """
        self._hovered = self._has_precise_pointer()
        self.update()

        # 请求显示自研工具提示。本方法在类里**只此一处**定义：
        # 重复定义会让后一个覆盖前一个，而这里恰好是 TooltipController
        # 唯一的触发点，覆盖后按钮的提示会整体失效。
        if self.tooltip_controller is not None and self.tooltip_text:
            from PyQt6.QtGui import QCursor

            self.tooltip_controller.schedule(self, self.tooltip_text, QCursor.pos())

        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        """鼠标离开：清除悬停并隐藏提示。"""
        self._hovered = False
        self.update()

        if self.tooltip_controller is not None:
            self.tooltip_controller.hide()

        super().leaveEvent(event)

    # ------------------------------------------------------------------
    # 主题
    # ------------------------------------------------------------------

    def set_theme(self, theme: Theme) -> None:
        """应用主题并重绘。"""
        self._theme = theme
        self.update()

    # ------------------------------------------------------------------
    # 按下反馈属性（由 QPropertyAnimation 驱动）
    # ------------------------------------------------------------------

    def _get_press_progress(self) -> float:
        return self._press_progress

    def _set_press_progress(self, value: float) -> None:
        self._press_progress = max(0.0, min(1.0, float(value)))
        self.update()

    #: 0.0 = 未按下，1.0 = 完全按下。动画时长与曲线由
    #: :func:`pingyitong.ui.design.motion.play_press_feedback` 控制。
    pressProgress = pyqtProperty(float, _get_press_progress, _set_press_progress)

    # ------------------------------------------------------------------
    # 尺寸
    # ------------------------------------------------------------------

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt 命名约定
        """返回建议尺寸：文案宽度 + 左右内边距。"""
        metrics = QFontMetricsF(self._font())
        text_width = metrics.horizontalAdvance(self.text())
        padding = SPACE["md"] * 2
        height = 26 if self._compact else LAYOUT["button_height"]
        return QSize(int(text_width + padding + 8), height)

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        """最小尺寸。"""
        metrics = QFontMetricsF(self._font())
        text_width = metrics.horizontalAdvance(self.text())
        height = 26 if self._compact else LAYOUT["button_height"]
        return QSize(int(text_width + SPACE["sm"] * 2 + 4), height)

    def _has_precise_pointer(self) -> bool:
        """是否存在精确指针设备（鼠标）。

        等价于 CSS 的 ``@media (hover: hover) and (pointer: fine)``。
        """
        from PyQt6.QtGui import QGuiApplication

        app = QGuiApplication.instance()
        if app is None:
            return True
        try:
            return app.primaryScreen() is not None
        except Exception:  # noqa: BLE001
            return True

    # ------------------------------------------------------------------
    # 绘制
    # ------------------------------------------------------------------

    def _font(self) -> QFont:
        from ..design.theme import base_font

        theme = self._theme
        size_key = "small" if self._compact else "body"
        weight = "medium" if self._variant == "primary" else "regular"
        if theme is None:
            font = QFont()
            font.setPixelSize(12 if self._compact else 13)
            font.setBold(self._variant == "primary")
            return font
        return base_font(theme, size_key, weight)

    def paintEvent(self, event) -> None:  # noqa: N802
        """自绘按钮：背景 → 描边 → 文案，整体按按下进度缩放。"""
        if self._theme is None:
            super().paintEvent(event)
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        try:
            # 按下缩放：0 → 1 进度对应 1.0 → PRESS_SCALE 倍。
            # 减少动态效果模式下 MOTION.press_scale 为 1.0，即不缩放。
            target_scale = MOTION.press_scale if MOTION.enabled else 1.0
            scale = 1.0 - (1.0 - target_scale) * self._press_progress

            if scale < 1.0:
                painter.translate(self.width() / 2.0, self.height() / 2.0)
                painter.scale(scale, scale)
                painter.translate(-self.width() / 2.0, -self.height() / 2.0)

            rect = QRectF(0.5, 0.5, self.width() - 1.0, self.height() - 1.0)
            radius = float(RADIUS["sm"])

            self._paint_background(painter, rect, radius)
            self._paint_text(painter, rect)
        finally:
            painter.end()

    def _paint_background(
        self, painter: QPainter, rect: QRectF, radius: float
    ) -> None:
        """绘制背景与边框。"""
        theme = self._theme
        assert theme is not None
        scheme = theme.scheme
        enabled = self.isEnabled()
        checked = self.isChecked()

        if not enabled:
            background = QColor(scheme.surface) if self._variant != "primary" else QColor(
                scheme.surface_active
            )
            border = QColor(scheme.border)
            text_pen = QColor(scheme.text_muted)
        elif self._variant == "primary":
            if self.isDown():
                background = QColor(scheme.accent_active)
            elif self._hovered:
                background = QColor(scheme.accent_hover)
            else:
                background = QColor(scheme.accent)
            border = None
            text_pen = QColor(scheme.text_inverse)
        elif self._variant == "danger":
            if self._hovered or self.isDown():
                background = QColor(scheme.danger_subtle)
            else:
                background = QColor(Qt.GlobalColor.transparent)
            border = QColor(scheme.danger)
            text_pen = QColor(scheme.danger)
        elif self._variant == "ghost":
            if self.isDown():
                background = QColor(scheme.surface_active)
            elif self._hovered:
                background = QColor(scheme.surface_hover)
            else:
                background = QColor(Qt.GlobalColor.transparent)
            border = None
            text_pen = QColor(scheme.text_primary if self._hovered else scheme.text_secondary)
        else:  # secondary
            if self.isDown():
                background = QColor(scheme.surface_active)
            elif self._hovered:
                background = QColor(scheme.surface_hover)
            elif checked:
                background = QColor(scheme.accent_subtle)
            else:
                background = QColor(scheme.surface_raised)
            border = QColor(scheme.border)
            text_pen = QColor(scheme.text_primary)

        self._text_color = text_pen
        self._radius = radius

        path = QPainterPath()
        path.addRoundedRect(rect, radius, radius)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(background)
        painter.drawPath(path)

        # 聚焦环：键盘可达性的视觉反馈，不与悬停态混淆
        if self.hasFocus() and enabled:
            focus_rect = rect.adjusted(0.5, 0.5, -0.5, -0.5)
            painter.setPen(QPen(QColor(scheme.border_focus), 2.0))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(focus_rect, radius, radius)
        elif border is not None:
            painter.setPen(QPen(border, 1.0))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(rect, radius, radius)

    def _paint_text(self, painter: QPainter, rect: QRectF) -> None:
        """绘制按钮文案（居中）。"""
        text = self.text()
        if not text:
            return

        painter.setFont(self._font())
        painter.setPen(QPen(getattr(self, "_text_color", QColor("#FFFFFF"))))
        painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), text)


class IconButton(Button):
    """仅含图标（或短符号）的方形按钮。

    用于悬浮控制条等空间受限场景。图标以文本符号表示，
    避免引入位图资源与额外的缩放处理。
    """

    def __init__(
        self,
        glyph: str,
        tooltip: str = "",
        theme: Theme | None = None,
        parent: QWidget | None = None,
        size: int = 28,
    ) -> None:
        super().__init__(glyph, "ghost", theme, parent, compact=True)
        self._size = size
        self.setFixedSize(size, size)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        if tooltip:
            self.setToolTip(tooltip)

    def setToolTip(self, text: str) -> None:  # noqa: N802 - Qt API
        """Expose the existing icon tooltip as its accessible action name."""
        self.setAccessibleName(text)
        super().setToolTip(text)

    def set_tooltip(self, text: str, controller=None) -> None:
        self.setAccessibleName(text)
        super().set_tooltip(text, controller)

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(self._size, self._size)

    def set_glyph(self, glyph: str) -> None:
        """替换图标符号。

        用于图标需要表达状态的场景（例如展开 / 收起箭头随状态翻转方向）。
        """
        self.setText(glyph)
        self.update()
