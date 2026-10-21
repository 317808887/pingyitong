"""自定义翻译区域框选。

对应产品功能规划三·（一）：用户框选屏幕上的一块矩形区域作为翻译范围，
区域外内容不处理，以降低资源占用并避免误翻译无关窗口。

交互设计
    覆盖层铺满**整个虚拟桌面**（跨多显示器一块整体），这样用户可以从
    一块屏幕拖到另一块，符合直觉；但真正的捕获区域会被裁剪到单块显示器内，
    因为跨屏矩形的位图会被显示器间隙破坏，导致坐标换算失准。

键盘可用性
    框选这类纯鼠标操作必须提供键盘出口，否则无法完成——
    因此支持 ``Esc`` 取消、``Enter`` 确认，并在界面上直接写出这两个提示。
"""

from __future__ import annotations

from PyQt6.QtCore import QPoint, QRect, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QGuiApplication, QKeyEvent, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from ..components.capture.geometry import (
    Rect,
    list_screens,
    region_to_monitor,
    virtual_desktop_rect,
)
from ..utils.logger import get_logger
from .design import RADIUS, SPACE
from .design.theme import Theme, base_font

logger = get_logger(__name__)


class RegionSelector(QWidget):
    """区域框选覆盖层。

    :param theme: 主题，用于提示文字与选框配色。
    """

    #: 用户完成框选，参数为逻辑坐标矩形与所属显示器序号
    region_selected = pyqtSignal(object, object)
    #: 用户取消框选
    cancelled = pyqtSignal()

    #: 框选被判定为有效所需的最小边长（逻辑像素）
    MIN_SIZE = 24

    def __init__(self, theme: Theme, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme = theme
        self._origin: QPoint | None = None
        self._current: QPoint | None = None
        self._dragging = False

        self._setup_window()

    # ------------------------------------------------------------------
    # 窗口
    # ------------------------------------------------------------------

    def _setup_window(self) -> None:
        """配置覆盖整个虚拟桌面的框选窗口。"""
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setWindowTitle("选择翻译区域")

        # 覆盖虚拟桌面：多屏时窗口跨越所有显示器
        desktop = virtual_desktop_rect(list_screens())
        if desktop.is_empty:
            screen = QGuiApplication.primaryScreen()
            geometry = screen.geometry() if screen else QRect(0, 0, 1920, 1080)
        else:
            geometry = QRect(desktop.x, desktop.y, desktop.width, desktop.height)

        self.setGeometry(geometry)

    def start(self) -> None:
        """显示覆盖层并接管焦点。"""
        self._origin = None
        self._current = None
        self._dragging = False
        self.show()
        self.raise_()
        self.activateWindow()
        self.setFocus(Qt.FocusReason.OtherFocusReason)
        logger.debug("区域框选覆盖层已显示")

    def cancel(self) -> None:
        """取消框选。"""
        self._origin = None
        self._current = None
        self._dragging = False
        self.hide()
        self.cancelled.emit()

    # ------------------------------------------------------------------
    # 鼠标交互
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:  # noqa: N802
        """开始拖拽。"""
        if event.button() == Qt.MouseButton.LeftButton:
            self._origin = event.position().toPoint()
            self._current = self._origin
            self._dragging = True
            self.update()
        elif event.button() == Qt.MouseButton.RightButton:
            self.cancel()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        """更新选框。"""
        self._current = event.position().toPoint()
        self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        """完成框选。"""
        if event.button() != Qt.MouseButton.LeftButton or not self._dragging:
            super().mouseReleaseEvent(event)
            return

        self._dragging = False
        logical = self._confirmed_selection()
        if logical is None:
            # 过小的框选多半是误触，保持覆盖层打开让用户重新框选
            self._origin = None
            self._current = None
            self.update()
            super().mouseReleaseEvent(event)
            return

        monitor = self._monitor_index_for(logical)
        self.hide()
        logger.info(
            "已选择翻译区域：%sx%s @(%s, %s)",
            logical.width, logical.height, logical.x, logical.y
        )
        self.region_selected.emit(logical, monitor)
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        """键盘出口：Esc 取消，Enter 确认当前选框。"""
        if event.key() == Qt.Key.Key_Escape:
            self.cancel()
            event.accept()
            return
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            # 与鼠标松开走同一套校验，避免键盘确认提交出误触级别的选区
            logical = self._confirmed_selection()
            if logical is not None:
                self.hide()
                self.region_selected.emit(logical, self._monitor_index_for(logical))
            event.accept()
            return
        super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    def _selection_rect(self) -> QRect | None:
        """返回当前选框（窗口局部坐标，Qt 的 :class:`QRect`）。

        **返回的是 Qt 的 :class:`QRect`，不是本项目自己的**
        :class:`~pingyitong.components.capture.geometry.Rect`。两者的
        宽高访问方式**不同**，这是本文件最容易出错的地方：

        ============  ==================  ==================
        含义           Qt ``QRect``        本项目 ``Rect``
        ============  ==================  ==================
        左上角 x       ``rect.x()``         ``rect.x``
        宽             ``rect.width()``     ``rect.width``
        是否为空       ``rect.isEmpty()``   ``rect.is_empty``
        右下角         ``rect.right()``     ``rect.right``
        ============  ==================  ==================

        把 Qt 的 ``rect.width``（方法对象）当成宽度传进本项目的
        :class:`Rect` 不会立即报错——``dataclass`` 不做类型检查，
        要等到后面真正做算术时才炸：``int + builtin_function_or_method``。
        而一旦在 Qt 事件处理器里抛出未捕获异常，PyQt6 会调用
        ``qFatal()`` **直接终止进程**（退出码 0xC0000409），
        连堆栈都不打印——用户看到的就是"一点框选程序就没了"。

        因此请**不要**在这里混用两套 API：需要全局逻辑坐标时一律走
        :meth:`_selection_logical`，它会一次性完成换算。
        """
        if self._origin is None or self._current is None:
            return None
        return QRect(self._origin, self._current).normalized()

    def _selection_logical(self) -> Rect | None:
        """返回选框的全局逻辑坐标（本项目的 :class:`Rect`）。

        这是**唯一**把 Qt 选区转成本项目矩形的出口：换算只写在这里，
        其余代码拿到的都是本项目的 :class:`Rect`，属性和方法不会再混。

        注意 Qt 用**两个点**构造的矩形是"含首含尾"的：
        ``QRect(QPoint(0,0), QPoint(9,9))`` 宽高都是 10 而不是 9。
        因此原地按下再松开得到的是 1×1 矩形而非空矩形——它会被
        :meth:`_confirmed_selection` 的尺寸下限挡掉。
        完全没有拖拽过（``_origin``/``_current`` 为空）时返回 ``None``。
        """
        selection = self._selection_rect()
        if selection is None or selection.isEmpty():
            return None

        origin = self.geometry()
        return Rect(
            origin.x() + selection.x(),
            origin.y() + selection.y(),
            selection.width(),
            selection.height(),
        )

    def _confirmed_selection(self) -> Rect | None:
        """返回**通过尺寸下限校验**的选区，不合格时返回 ``None``。

        鼠标松开与回车确认都走这里，保证两条提交路径的判定完全一致——
        此前只有鼠标那条检查了 :attr:`MIN_SIZE`，键盘确认会把
        一像素的误触也当成有效选区提交。
        """
        logical = self._selection_logical()
        if logical is None:
            return None
        if logical.width < self.MIN_SIZE or logical.height < self.MIN_SIZE:
            logger.debug("框选区域过小（%sx%s），已忽略", logical.width, logical.height)
            return None
        return logical

    def _monitor_index_for(self, logical: Rect) -> int | None:
        """返回选框主要落在哪块显示器上。"""
        screen = region_to_monitor(logical, list_screens())
        return screen.mss_index if screen else None

    # ------------------------------------------------------------------
    # 绘制
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        """绘制遮罩、选框与操作提示。"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        try:
            scheme = self._theme.scheme
            full = self.rect()

            # 全屏半透明遮罩，让用户看清"未被选中的区域会被忽略"
            painter.fillRect(full, QColor(0, 0, 0, 110))

            selection = self._selection_rect()
            if selection is not None and not selection.isEmpty():
                # 选中区域挖空：先清除遮罩，再描边框
                painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
                painter.fillRect(selection, Qt.GlobalColor.transparent)
                painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

                painter.setPen(QPen(QColor(scheme.accent), 2.0))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(selection.adjusted(1, 1, -1, -1))

                self._paint_dimension_badge(painter, selection, scheme)

            self._paint_hints(painter, scheme)
        finally:
            painter.end()

    def _paint_dimension_badge(self, painter: QPainter, selection: QRect, scheme) -> None:
        """在选框旁显示尺寸，让用户确认区域大小。

        ``selection`` 是 Qt 的 :class:`QRect`，因此宽高必须**调用**：
        写成 ``selection.width`` 会把方法对象本身插进文本，界面上显示的
        就是 ``<built-in method width of QRect object at 0x...>``。
        """
        text = f"{selection.width()} × {selection.height()}"
        painter.setFont(base_font(self._theme, "small", "medium"))

        metrics = painter.fontMetrics()
        width = metrics.horizontalAdvance(text) + SPACE["sm"] * 2
        height = metrics.height() + SPACE["xs"]

        # 默认放在选框上方；空间不足则放到选框内部顶端
        x = selection.left()
        y = selection.top() - height - SPACE["xs"]
        if y < 0:
            y = selection.top() + SPACE["xs"]

        badge = QRect(x, y, width, height)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(scheme.accent))
        painter.drawRoundedRect(badge, RADIUS["sm"], RADIUS["sm"])

        painter.setPen(QColor(scheme.text_inverse))
        painter.drawText(badge, int(Qt.AlignmentFlag.AlignCenter), text)

    def _paint_hints(self, painter: QPainter, scheme) -> None:
        """绘制操作提示。

        提示必须常驻显示而不是藏在提示框里——用户此刻正处在"不知道该
        怎么退出"的状态，任何需要额外探索的设计都会让人卡住。
        """
        lines = [
            "拖动鼠标框选要翻译的屏幕区域",
            "Enter 确认 · Esc 或右键取消",
        ]

        painter.setFont(base_font(self._theme, "subtitle", "medium"))
        metrics = painter.fontMetrics()

        max_width = max(metrics.horizontalAdvance(line) for line in lines)
        box_width = max_width + SPACE["xxl"] * 2
        box_height = len(lines) * (metrics.height() + SPACE["xs"]) + SPACE["lg"] * 2

        desktop_center = self.rect().center()
        box = QRect(
            desktop_center.x() - box_width // 2,
            desktop_center.y() - box_height // 2,
            box_width,
            box_height,
        )

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 200))
        painter.drawRoundedRect(box, RADIUS["lg"], RADIUS["lg"])

        painter.setPen(QColor("#FFFFFF"))
        for index, line in enumerate(lines):
            font = (
                base_font(self._theme, "subtitle", "medium")
                if index == 0
                else base_font(self._theme, "small")
            )
            painter.setFont(font)
            painter.setPen(QColor("#FFFFFF") if index == 0 else QColor("#A6AEBD"))
            line_rect = QRect(
                box.left(),
                box.top() + SPACE["lg"] + index * (metrics.height() + SPACE["xs"]),
                box.width(),
                metrics.height(),
            )
            painter.drawText(line_rect, int(Qt.AlignmentFlag.AlignCenter), line)


def create_region_overlay(theme: Theme) -> RegionSelector:
    """工厂函数：创建区域框选覆盖层。"""
    return RegionSelector(theme)
