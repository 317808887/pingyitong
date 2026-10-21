"""开关与分段控件。

两个组件都遵循同一条规范：**状态切换必须"看得见过程"**。

- 开关：滑块位置用 200ms 强 ease-out 补间，让用户确认"状态确实变了"；
- 分段控件：指示器用 250ms 强 ease-in-out 在选项间滑动
  （规范中"标签页指示器"使用 ``clip-path`` 250ms ``ease-in-out``，
  Qt 中等价为几何补间），而不是瞬间跳变。

这两处都属于"偶尔触发"的控件（不是每天上百次的键盘操作），
因此适用标准动效；但时长都严格控制在 300ms 以内。
"""

from __future__ import annotations

from PyQt6.QtCore import QRectF, Qt, pyqtProperty, pyqtSignal
from PyQt6.QtGui import QColor, QFontMetricsF, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QAbstractButton, QSizePolicy, QWidget

from ..design import LAYOUT, MOTION, RADIUS, SPACE
from ..design.motion import animate_property
from ..design.theme import Theme, base_font


class Switch(QAbstractButton):
    """拨动开关。

    规范要点：开关属于"状态指示"类动效——它的存在意义就是让用户
    确认状态已改变，因此必须动，且动得要快（200ms）。
    """

    def __init__(
        self,
        checked: bool = False,
        theme: Theme | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self.setCheckable(True)
        self.setChecked(checked)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFixedSize(38, 22)

        self._progress = 1.0 if checked else 0.0
        self.toggled.connect(self._animate)

    def set_theme(self, theme: Theme) -> None:
        """应用主题。"""
        self._theme = theme
        self.update()

    # ------------------------------------------------------------------
    # 动画进度
    # ------------------------------------------------------------------

    def _get_progress(self) -> float:
        return self._progress

    def _set_progress(self, value: float) -> None:
        self._progress = max(0.0, min(1.0, float(value)))
        self.update()

    #: 0.0 = 关闭，1.0 = 开启
    progress = pyqtProperty(float, _get_progress, _set_progress)

    def _animate(self, checked: bool) -> None:
        """切换时驱动滑块位移。"""
        animate_property(
            self,
            b"progress",
            self._progress,
            1.0 if checked else 0.0,
            MOTION.duration_of("indicator"),
            "out",
        )

    # ------------------------------------------------------------------
    # 绘制
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        """绘制轨道与滑块。"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        try:
            scheme = self._theme.scheme if self._theme else None
            track_off = QColor(scheme.surface_active) if scheme else QColor("#282D38")
            track_on = QColor(scheme.accent) if scheme else QColor("#2E6BE6")
            knob_color = QColor(scheme.text_inverse) if scheme else QColor("#FFFFFF")

            if not self.isEnabled():
                track_on = track_off = QColor(
                    scheme.surface if scheme else QColor("#171A21")
                )
                knob_color = QColor(scheme.text_muted) if scheme else QColor("#7C8598")

            # 轨道颜色不做补间：颜色过渡在"减少动态效果"下仍需保留，
            # 但这里位置补间已经足够传达状态，二者叠加会显得嘈杂。
            track_color = _blend(track_off, track_on, self._progress)

            height = self.height()
            radius = height / 2.0
            track = QRectF(0.0, 0.0, self.width(), height)

            path = QPainterPath()
            path.addRoundedRect(track, radius, radius)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(track_color)
            painter.drawPath(path)

            # 滑块位置由 progress 决定，两端各留 2px 内边距
            knob_diameter = height - 4.0
            travel = self.width() - knob_diameter - 4.0
            knob_x = 2.0 + travel * self._progress
            knob = QRectF(knob_x, 2.0, knob_diameter, knob_diameter)

            painter.setBrush(knob_color)
            painter.drawEllipse(knob)

            if self.hasFocus():
                painter.setPen(QPen(QColor(scheme.border_focus) if scheme else QColor("#3D6FD8"), 2.0))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRoundedRect(track.adjusted(0.5, 0.5, -0.5, -0.5), radius, radius)
        finally:
            painter.end()


class SegmentedControl(QAbstractButton):
    """分段控件（用于"极速 / 高精度"模型切换）。

    规范中标签页指示器的做法是"复制一份高亮层并用 clip-path 裁切"，
    在 Qt 中等价于用一个可移动的指示器矩形——它带来的是
    "颜色平滑过渡"这一 CSS 单独补间颜色无法达到的效果：
    指示器作为一个整体滑动，视觉上是一个物体在移动，而不是两处颜色各自变化。
    """

    #: 选中项变更，参数为索引
    selected_changed = pyqtSignal(int)

    def __init__(
        self,
        options: list[str],
        current: int = 0,
        theme: Theme | None = None,
        parent: QWidget | None = None,
        min_segment_width: int = 96,
    ) -> None:
        super().__init__(parent)
        self._options = list(options)
        self._theme = theme
        self._current = max(0, min(len(self._options) - 1, int(current))) if self._options else 0
        self._hover_index = -1
        self._indicator = float(self._current)
        self._min_segment_width = min_segment_width

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(LAYOUT["control_height"])

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------

    @property
    def current_index(self) -> int:
        """当前选中项索引。"""
        return self._current

    @property
    def current_text(self) -> str:
        """当前选中项文本。"""
        if 0 <= self._current < len(self._options):
            return self._options[self._current]
        return ""

    def set_current_index(self, index: int, animate: bool = True) -> None:
        """设置选中项。

        :param animate: 是否播放指示器滑动。键盘操作（左右方向键）
            属于高频动作，规范要求"永不加动画"，因此键盘路径传入 ``False``。
        """
        if not self._options:
            return
        index = max(0, min(len(self._options) - 1, int(index)))
        if index == self._current:
            return

        self._current = index
        if animate:
            animate_property(
                self,
                b"indicator",
                self._indicator,
                float(index),
                MOTION.duration_of("indicator"),
                "in_out",
            )
        else:
            self._set_indicator(float(index))
        self.update()
        self.selected_changed.emit(index)

    def set_theme(self, theme: Theme) -> None:
        """应用主题。"""
        self._theme = theme
        self.update()

    def _get_indicator(self) -> float:
        return self._indicator

    def _set_indicator(self, value: float) -> None:
        self._indicator = float(value)
        self.update()

    #: 指示器位置（以"选项索引"为单位的浮点数）
    indicator = pyqtProperty(float, _get_indicator, _set_indicator)

    # ------------------------------------------------------------------
    # 尺寸
    # ------------------------------------------------------------------

    def sizeHint(self):  # noqa: N802
        from PyQt6.QtCore import QSize

        width = self._min_segment_width * max(1, len(self._options)) + 8
        return QSize(width, LAYOUT["control_height"])

    def minimumSizeHint(self):  # noqa: N802
        from PyQt6.QtCore import QSize

        metrics = QFontMetricsF(base_font(self._theme) if self._theme else self.font())
        widest = max(
            (metrics.horizontalAdvance(text) for text in self._options), default=60
        )
        width = int(widest + SPACE["lg"] * 2) * max(1, len(self._options)) + 8
        return QSize(width, LAYOUT["control_height"])

    # ------------------------------------------------------------------
    # 交互
    # ------------------------------------------------------------------

    def _segment_width(self) -> float:
        if not self._options:
            return 0.0
        return (self.width() - 8.0) / len(self._options)

    def _index_at(self, x: float) -> int:
        if not self._options:
            return -1
        width = self._segment_width()
        if width <= 0:
            return -1
        index = int((x - 4.0) // width)
        if 0 <= index < len(self._options):
            return index
        return -1

    def mousePressEvent(self, event) -> None:  # noqa: N802
        """点击选中。"""
        index = self._index_at(event.position().x())
        if index >= 0:
            self.set_current_index(index, animate=True)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        """跟踪悬停项。"""
        index = self._index_at(event.position().x())
        if index != self._hover_index:
            self._hover_index = index
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        """清除悬停。"""
        self._hover_index = -1
        self.update()
        super().leaveEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        """键盘切换。

        规范把键盘触发的操作列为"永不加动画"——用户可能连续按十几次，
        每次 250ms 的滑动会让界面显得迟钝。因此键盘路径瞬时切换。
        """
        key = event.key()
        if key in (Qt.Key.Key_Left, Qt.Key.Key_Up):
            self.set_current_index(self._current - 1, animate=False)
            event.accept()
            return
        if key in (Qt.Key.Key_Right, Qt.Key.Key_Down):
            self.set_current_index(self._current + 1, animate=False)
            event.accept()
            return
        super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # 绘制
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        """绘制轨道、滑动指示器与各段文字。"""
        if not self._options or self._theme is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        try:
            scheme = self._theme.scheme
            outer = QRectF(0.0, 0.0, self.width(), self.height())
            radius = float(RADIUS["sm"])

            painter.setPen(QPen(QColor(scheme.border), 1.0))
            painter.setBrush(QColor(scheme.surface))
            painter.drawRoundedRect(outer.adjusted(0.5, 0.5, -0.5, -0.5), radius, radius)

            segment_width = self._segment_width()
            # 指示器：一个整体滑动的矩形，而不是两处颜色各自改变
            indicator_rect = QRectF(
                4.0 + self._indicator * segment_width + 2.0,
                3.0,
                segment_width - 4.0,
                self.height() - 6.0,
            )
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(scheme.accent))
            painter.drawRoundedRect(
                indicator_rect, radius - 2.0, radius - 2.0
            )

            painter.setFont(base_font(self._theme, "small", "medium"))
            for index, text in enumerate(self._options):
                cell = QRectF(
                    4.0 + index * segment_width, 0.0, segment_width, float(self.height())
                )
                if index == self._current:
                    painter.setPen(QPen(QColor(scheme.text_inverse)))
                elif index == self._hover_index:
                    painter.setPen(QPen(QColor(scheme.text_primary)))
                else:
                    painter.setPen(QPen(QColor(scheme.text_secondary)))
                painter.drawText(cell, int(Qt.AlignmentFlag.AlignCenter), text)
        finally:
            painter.end()


def _blend(start: QColor, end: QColor, ratio: float) -> QColor:
    """在两色之间做线性插值。"""
    ratio = max(0.0, min(1.0, ratio))
    return QColor(
        int(start.red() + (end.red() - start.red()) * ratio),
        int(start.green() + (end.green() - start.green()) * ratio),
        int(start.blue() + (end.blue() - start.blue()) * ratio),
        int(start.alpha() + (end.alpha() - start.alpha()) * ratio),
    )
