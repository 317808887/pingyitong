"""工具提示（Tooltip）。

实现依据：设计规范中关于工具提示的三条要求。

1. **首次延迟**——防止鼠标划过时误触发，导致提示框乱闪；
2. **后续立即**——一旦有一条提示已经显示，相邻控件的提示必须
   ``transition-duration: 0ms`` 立即出现、且不做动画。
   规范明确指出这是让"整个工具栏感觉更快"的关键，
   并把"每个提示都延迟 + 动画"列为需要主动排查的反模式；
3. **自触发器位置展开**——提示的缩放原点应指向触发它的控件，
   而不是从自身中心展开（模态是唯一例外）。

Qt 内置的 ``QToolTip`` 无法满足第 2 条（它只有全局的
``showText``，且延迟由样式提示控制、无法按"是否已有提示在显示"动态切换），
因此本模块自行实现。
"""

from __future__ import annotations

from PyQt6.QtCore import QPoint, QRect, Qt, QTimer
from PyQt6.QtGui import QColor, QFontMetricsF, QGuiApplication, QPainter, QPainterPath
from PyQt6.QtWidgets import QWidget

from ..design import RADIUS, SPACE, TOOLTIP_INITIAL_DELAY_MS
from ..design.motion import enter, exit_widget
from ..design.theme import Theme, base_font


class TooltipController:
    """全局工具提示控制器。

    同一时刻只有一个提示可见，且共享"是否刚刚显示过"的状态，
    从而实现"后续提示立即出现"。
    """

    def __init__(self, theme: Theme) -> None:
        self._theme = theme
        self._widget: _TooltipWidget | None = None
        self._timer = QTimer()
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._show_pending)
        self._pending: tuple[QWidget, str, QPoint] | None = None
        #: 最近一次显示提示的时刻，用于判断"是否处于连续浏览状态"
        self._last_shown_at = 0.0

    def set_theme(self, theme: Theme) -> None:
        """应用主题。"""
        self._theme = theme
        if self._widget is not None:
            self._widget.apply_theme(theme)

    def schedule(self, anchor: QWidget, text: str, global_pos: QPoint) -> None:
        """请求显示提示。

        若上一条提示刚刚显示过（800ms 内），则跳过延迟与动画立即显示——
        这正是规范要求的"跳过延迟 + 跳过动画"。
        """
        if not text:
            return

        import time

        recently_active = (time.perf_counter() - self._last_shown_at) < 0.8

        if recently_active:
            self._hide()
            self._present(anchor, text, global_pos, instant=True)
            return

        self._pending = (anchor, text, global_pos)
        self._timer.start(TOOLTIP_INITIAL_DELAY_MS)

    def hide(self) -> None:
        """隐藏提示并取消待显示请求。"""
        self._timer.stop()
        self._pending = None
        self._hide()

    def _hide(self) -> None:
        if self._widget is not None:
            widget = self._widget
            self._widget = None
            exit_widget(widget, "tooltip", True)

    def _show_pending(self) -> None:
        if self._pending is None:
            return
        anchor, text, position = self._pending
        self._pending = None
        self._hide()
        self._present(anchor, text, position, instant=False)

    def _present(
        self,
        anchor: QWidget,
        text: str,
        global_pos: QPoint,
        instant: bool,
    ) -> None:
        """创建并显示提示。"""
        widget = _TooltipWidget(text, self._theme)

        # 位置：默认在控件下方居中；下方空间不足则放到上方
        screen = QGuiApplication.screenAt(global_pos) or QGuiApplication.primaryScreen()
        anchor_rect = QRect(anchor.mapToGlobal(QPoint(0, 0)), anchor.size())

        x = anchor_rect.center().x() - widget.width() // 2
        y = anchor_rect.bottom() + 6
        if screen is not None:
            area = screen.availableGeometry()
            if y + widget.height() > area.bottom():
                y = anchor_rect.top() - widget.height() - 6
            x = max(area.left() + 4, min(x, area.right() - widget.width() - 4))

        self._widget = widget
        if instant:
            # 连续浏览：瞬时出现，不做入场动效
            widget.setGeometry(QRect(QPoint(x, y), widget.size()))
            widget.show()
            widget.raise_()
        else:
            enter(widget, QRect(QPoint(x, y), widget.size()), "tooltip")

        import time

        self._last_shown_at = time.perf_counter()


class _TooltipWidget(QWidget):
    """提示框本体。"""

    def __init__(self, text: str, theme: Theme, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._text = text
        self._theme = theme

        self.setWindowFlags(
            Qt.WindowType.ToolTip
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowTransparentForInput
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        font = base_font(theme, "small")
        metrics = QFontMetricsF(font)
        width = int(metrics.horizontalAdvance(text)) + SPACE["sm"] * 2 + 4
        height = int(metrics.height()) + SPACE["xs"] * 2 + 2
        self.setFixedSize(max(40, width), max(24, height))

    def apply_theme(self, theme: Theme) -> None:
        """应用主题。"""
        self._theme = theme
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        """绘制提示背景与文字。"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        try:
            scheme = self._theme.scheme
            rect = self.rect().adjusted(0, 0, -1, -1)
            radius = float(RADIUS["sm"])

            path = QPainterPath()
            path.addRoundedRect(
                float(rect.x()), float(rect.y()),
                float(rect.width()), float(rect.height()), radius, radius
            )
            painter.setPen(QColor(scheme.border))
            painter.setBrush(QColor(scheme.surface_raised))
            painter.drawPath(path)

            painter.setFont(base_font(self._theme, "small"))
            painter.setPen(QColor(scheme.text_primary))
            painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), self._text)
        finally:
            painter.end()


class TooltipMixin:
    """为控件提供统一的提示接入。

    控件在 ``enterEvent`` 中调用 :meth:`show_tooltip`，
    在 ``leaveEvent`` 中调用 :meth:`hide_tooltip`。
    """

    _tooltip_controller: TooltipController | None = None

    @classmethod
    def set_controller(cls, controller: TooltipController) -> None:
        """注入全局控制器。"""
        TooltipMixin._tooltip_controller = controller

    def show_tooltip(self, text: str) -> None:
        """请求显示提示。"""
        controller = TooltipMixin._tooltip_controller
        if controller is None or not text:
            return
        widget = self if isinstance(self, QWidget) else None
        if widget is None:
            return
        from PyQt6.QtGui import QCursor

        controller.schedule(widget, text, QCursor.pos())

    def hide_tooltip(self) -> None:
        """隐藏提示。"""
        controller = TooltipMixin._tooltip_controller
        if controller is not None:
            controller.hide()
