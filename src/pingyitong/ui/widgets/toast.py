"""通知（Toast）系统。

实现依据：``ask-sonner`` 规范（Sonner 的设计原则与 API 约定）。

规范中与本实现直接对应的要点：

- **单一实例**：整程序只挂载一个通知管理器，绝不按页面/条件创建，
  否则同一条通知会被渲染多次；
- **默认停留 4000ms**，同时最多显示 **3** 条，条间距 **14px**，
  距屏幕边缘 **20px**（桌面取 32px 会显得过于靠内，此处按窗口场景收紧）；
- **动效 400ms ``ease``**，从 ``translateY(100%)`` 进入、**沿同一边缘退出**——
  规范强调"对称路径才是滑动关闭之所以直观的原因"；
- **可中断**：通知会被快速连续触发，因此必须用补间而非关键帧。
  规范明确指出"关键帧从中断处重新开始，补间从当前位置继续"；
  这里用 ``QPropertyAnimation`` 驱动位移/不透明度属性，天然可中断；
- **速度判定关闭**：拖动速度超过 ``0.11 px/ms`` 即关闭，
  不必拖过固定距离——"一次轻扫就够"。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from PyQt6.QtCore import (
    QObject,
    QPoint,
    QPointF,
    QRect,
    QRectF,
    Qt,
    QTimer,
    pyqtProperty,
    pyqtSignal,
)
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QWidget

from ...utils.logger import get_logger
from ..design import (
    LAYOUT,
    MOTION,
    RADIUS,
    SPACE,
    SWIPE_DISMISS_VELOCITY,
    TOAST_DURATION_MS,
    TOAST_GAP,
    TOAST_OFFSET,
    TOAST_VISIBLE_MAX,
)
from ..design.theme import Theme, base_font

logger = get_logger(__name__)


class ToastKind(Enum):
    """通知类型。"""

    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ToastRequest:
    """一条待显示的通知。"""

    title: str
    description: str = ""
    kind: ToastKind = ToastKind.INFO
    #: 停留时长（毫秒）；``0`` 表示不自动消失
    duration_ms: int = TOAST_DURATION_MS
    #: 操作按钮文案与回调
    action_text: str = ""
    action: Callable[[], None] | None = None
    #: 去重标识：相同 ``key`` 的通知会替换而不是叠加
    key: str = ""
    #: 附加数据，供调用方自行关联
    payload: dict = field(default_factory=dict)


class ToastWidget(QWidget):
    """单条通知的呈现。

    自绘实现，配色与圆角全部来自设计令牌。
    """

    dismissed = pyqtSignal(object)

    def __init__(
        self,
        request: ToastRequest,
        theme: Theme,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.request = request
        self._theme = theme
        self._offset = 0.0
        self._drag_origin: QPoint | None = None
        self._drag_started_at = 0.0
        self._drag_distance = 0
        self._timer: QTimer | None = None
        #: 堆叠位置对应的基准 Y 坐标（其他通知关闭后由管理器更新）
        self.stack_y = 0
        #: 操作按钮的点击热区（在 paintEvent 中计算）
        self._action_rect: QRectF | None = None

        self.setWindowFlags(
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setFixedWidth(self._preferred_width())

        # ``offset`` 用于堆叠时的位移动画，由管理器驱动
        self.setFixedHeight(self._preferred_height())

    # ------------------------------------------------------------------
    # 动画属性
    # ------------------------------------------------------------------

    def _get_offset(self) -> float:
        return self._offset

    def _set_offset(self, value: float) -> None:
        self._offset = float(value)
        self.update()

    #: 0.0 = 完全显示，1.0 = 完全移出屏幕（用于入场/退场补间）
    offset = pyqtProperty(float, _get_offset, _set_offset)

    # ------------------------------------------------------------------
    # 尺寸
    # ------------------------------------------------------------------

    def _preferred_width(self) -> int:
        minimum = LAYOUT["toast_min_width"]
        maximum = LAYOUT["toast_max_width"]
        # 依据标题长度粗略估算，长标题更宽，但不超过上限
        estimated = len(self.request.title) * 14 + LAYOUT["toast_padding"] * 2 + 40
        return max(minimum, min(maximum, estimated))

    def _preferred_height(self) -> int:
        base = 44
        if self.request.description:
            base += 20
        if self.request.action_text:
            base += 24
        return base

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def show_at(self, position: QPoint, animate: bool = True) -> None:
        """在指定位置显示（带入场动效）。

        入场自下方 100% 高度处滑入——规范要求 Toast 从同一边缘进出。
        """
        final = QRect(position, self.size())
        start_y = position.y() + self.height()

        self.setGeometry(QRect(position.x(), start_y, self.width(), self.height()))
        self.show()
        self.raise_()

        duration = MOTION.duration_of("toast")
        if not animate or duration <= 0:
            self.setGeometry(final)
            self._schedule_auto_close()
            return

        # 补间作用于窗口位置：从下方滑入到最终位置
        from ..design.motion import animate_pos

        animate_pos(
            self,
            QPoint(position.x(), start_y),
            QPoint(position.x(), position.y()),
            duration,
            "out",
            self._schedule_auto_close,
        )

    def dismiss(self, animate: bool = True) -> None:
        """关闭通知（沿进入的同一边缘退出）。"""
        self._cancel_timer()
        duration = MOTION.duration_of("toast")

        if not animate or duration <= 0:
            self._finish()
            return

        from ..design.motion import animate_pos

        current = self.pos()
        animate_pos(
            self,
            current,
            QPoint(current.x(), current.y() + self.height()),
            duration,
            "out",
            self._finish,
        )

    def _finish(self) -> None:
        """移除自身并通知管理器。"""
        self.dismissed.emit(self)
        self.hide()
        self.deleteLater()

    def _schedule_auto_close(self) -> None:
        """安排自动关闭。"""
        if self.request.duration_ms <= 0:
            return
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(lambda: self.dismiss(True))
        self._timer.start(self.request.duration_ms)

    def _cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def enterEvent(self, event) -> None:  # noqa: N802
        """鼠标悬停时暂停自动关闭。

        规范把这类处理称为"不被察觉地处理边界情况"——用户正在读通知
        时它不该消失。不这样做用户会感到"来不及看清"。
        """
        self._cancel_timer()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        """离开后重新计时。"""
        self._schedule_auto_close()
        super().leaveEvent(event)

    # ------------------------------------------------------------------
    # 拖动关闭
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:  # noqa: N802
        """开始拖动。"""
        if event.button() == Qt.MouseButton.LeftButton:
            # 规范要求拖动开始后捕获指针，即使移出控件范围也要继续响应
            self.grabMouse()
            self._drag_origin = event.globalPosition().toPoint()
            self._drag_distance = 0
            self._drag_base_y = self.pos().y()
            import time

            self._drag_started_at = time.perf_counter()
            self._cancel_timer()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        """跟随拖动，并对超出边界做阻尼。

        规范指出"现实中物体不会突然停住，而是逐渐减速"，
        因此向下拖过头时位移会被压缩，而不是硬性禁止。
        """
        if self._drag_origin is None:
            return
        current = event.globalPosition().toPoint()
        delta = current - self._drag_origin
        self._drag_distance = delta.y()

        # 向上拖动（超出自然边界）施加阻尼：位移按 0.3 倍衰减
        effective = self._drag_distance if self._drag_distance > 0 else int(
            self._drag_distance * 0.3
        )
        base_y = getattr(self, "_drag_base_y", self.pos().y())
        self.move(self.pos().x(), base_y + effective)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        """结束拖动：先判定是否点击了操作按钮，再判定是否关闭。"""
        self.releaseMouse()
        import time

        elapsed_ms = max(1.0, (time.perf_counter() - self._drag_started_at) * 1000.0)
        velocity = abs(self._drag_distance) / elapsed_ms
        drag_distance = self._drag_distance
        self._drag_origin = None

        # 位移很小的按压视为点击：命中操作按钮热区时执行回调。
        # 规范提到"点击操作按钮会关闭通知，除非回调显式阻止"，
        # 这里遵循该默认行为。
        if abs(drag_distance) <= 4 and self._action_rect is not None:
            local = self.mapFromGlobal(event.globalPosition().toPoint())
            if self._action_rect.contains(QPointF(local)):
                action = self.request.action
                self.dismiss(animate=True)
                if action is not None:
                    try:
                        action()
                    except Exception:  # noqa: BLE001 - 回调异常不应影响通知系统
                        logger.exception("通知操作回调执行失败")
                super().mouseReleaseEvent(event)
                return

        threshold = self.height() * 0.4
        if drag_distance > threshold or velocity > SWIPE_DISMISS_VELOCITY:
            self.dismiss(True)
        else:
            # 回弹到原位，使用强 ease-out 让"取消"感觉干脆
            from ..design.motion import animate_pos

            animate_pos(
                self,
                self.pos(),
                QPoint(self.pos().x(), self.stack_y),
                MOTION.duration_of("popover"),
                "out",
                self._schedule_auto_close,
            )
        super().mouseReleaseEvent(event)

    def set_stack_position(self, position: QPoint, animate: bool = True) -> None:
        """由管理器设置的堆叠位置（其他通知关闭后重新排布）。"""
        self.stack_y = position.y()
        if self._drag_origin is not None:
            return  # 拖动中不打断用户操作
        duration = MOTION.duration_of("popover")
        if animate and duration > 0:
            from ..design.motion import animate_pos

            animate_pos(self, self.pos(), position, duration, "out")
        else:
            self.move(position)

    # ------------------------------------------------------------------
    # 绘制
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        """绘制通知卡片。"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        try:
            scheme = self._theme.scheme
            rect = self.rect().adjusted(0, 0, -1, -1)
            radius = float(RADIUS["md"])

            path = QPainterPath()
            path.addRoundedRect(float(rect.x()), float(rect.y()),
                                float(rect.width()), float(rect.height()), radius, radius)

            painter.setPen(QPen(QColor(scheme.border), 1.0))
            painter.setBrush(QColor(scheme.surface_raised))
            painter.drawPath(path)

            # 左侧色条标识类型。规范强调"状态不能只靠颜色传达"，
            # 因此同时使用了不同符号（见下方 _symbol）。
            accent = {
                ToastKind.SUCCESS: scheme.success,
                ToastKind.WARNING: scheme.warning,
                ToastKind.ERROR: scheme.danger,
                ToastKind.INFO: scheme.accent,
            }.get(self.request.kind, scheme.accent)

            bar_width = 3.0
            bar = QPainterPath()
            bar.addRoundedRect(
                float(rect.x() + 1), float(rect.y() + 8),
                bar_width, float(rect.height() - 16), 1.5, 1.5
            )
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(accent))
            painter.drawPath(bar)

            padding = float(LAYOUT["toast_padding"])
            text_left = float(rect.x()) + padding + SPACE["sm"]
            text_width = float(rect.width()) - padding * 2 - SPACE["sm"]

            # 文本绘制矩形统一使用 QRectF：内边距是浮点数（令牌里可能出现
            # 非整数），而 QRect 只接受整型参数，混用会在绘制时抛 TypeError。
            # 图标位置是固定的整数布局，仍用 QRect 表达更直观。
            painter.setFont(base_font(self._theme, "body", "medium"))
            painter.setPen(QPen(QColor(accent)))
            painter.drawText(
                QRect(rect.x() + int(padding), rect.y(), 16, rect.height()),
                int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                _symbol(self.request.kind),
            )

            cursor_y = float(rect.y()) + padding
            painter.setFont(base_font(self._theme, "body", "medium"))
            painter.setPen(QPen(QColor(scheme.text_primary)))
            painter.drawText(
                QRectF(text_left, cursor_y, text_width, 18.0),
                int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                self.request.title,
            )
            cursor_y += 18.0

            if self.request.description:
                painter.setFont(base_font(self._theme, "small"))
                painter.setPen(QPen(QColor(scheme.text_secondary)))
                painter.drawText(
                    QRectF(text_left, cursor_y, text_width, 18.0),
                    int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                    self.request.description,
                )
                cursor_y += 20.0

            if self.request.action_text:
                painter.setFont(base_font(self._theme, "small", "medium"))
                painter.setPen(QPen(QColor(scheme.accent_text)))
                metrics_rect = QRectF(text_left, cursor_y, text_width, 20.0)
                painter.drawText(
                    metrics_rect,
                    int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                    self.request.action_text,
                )
                # 记录热区，供点击判定使用
                self._action_rect = metrics_rect
            else:
                self._action_rect = None
        finally:
            painter.end()


def _symbol(kind: ToastKind) -> str:
    """返回类型的文字符号。

    使用文字符号而非图形，避免引入图标资源；同时保证状态不依赖颜色单独传达。
    """
    return {
        ToastKind.SUCCESS: "✓",
        ToastKind.WARNING: "!",
        ToastKind.ERROR: "✕",
        ToastKind.INFO: "i",
    }.get(kind, "i")


class ToastManager(QObject):
    """通知管理器：全程序唯一实例。

    规范中"绝不在多处挂载通知容器"的具体落实——本类由
    :class:`~pingyitong.ui.app.Application` 持有唯一实例，
    所有模块通过它发通知。
    """

    def __init__(self, theme: Theme, enabled: bool = True, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._theme = theme
        self._enabled = bool(enabled)
        self._toasts: list[ToastWidget] = []
        self._anchor_widget: QWidget | None = None

    def set_theme(self, theme: Theme) -> None:
        """应用主题到后续新建的通知。"""
        self._theme = theme
        for toast in self._toasts:
            toast._theme = theme
            toast.update()

    def set_enabled(self, enabled: bool) -> None:
        """启用或禁用通知。"""
        self._enabled = bool(enabled)
        if not self._enabled:
            self.dismiss_all()

    def set_anchor(self, widget: QWidget | None) -> None:
        """设置定位参照控件（通常是主面板）。

        通知会出现在参照控件所在显示器的右下角；未设置时使用主显示器。
        """
        self._anchor_widget = widget

    # ------------------------------------------------------------------
    # 发送
    # ------------------------------------------------------------------

    def show(
        self,
        title: str,
        description: str = "",
        kind: ToastKind = ToastKind.INFO,
        duration_ms: int = TOAST_DURATION_MS,
        action_text: str = "",
        action: Callable[[], None] | None = None,
        key: str = "",
    ) -> ToastWidget | None:
        """显示一条通知。

        :param key: 去重标识。相同 ``key`` 的通知会替换已有那条，
            而不是堆积多条——这样"翻译已暂停"之类的高频提示
            不会把屏幕占满。
        """
        if not self._enabled:
            logger.debug("通知已禁用，忽略：%s", title)
            return None

        if key:
            # 相同 key 的通知替换而非叠加。这里只负责"关闭"，
            # 从列表中移除由 dismissed 信号统一处理——否则会出现双重移除
            # （dismiss 内部同步发信号完成移除，随后再 remove 就抛
            # ValueError: list.remove(x): x not in list）。
            for existing in list(self._toasts):
                if existing.request.key == key:
                    existing.dismiss(animate=False)

        request = ToastRequest(
            title=title,
            description=description,
            kind=kind,
            duration_ms=duration_ms,
            action_text=action_text,
            action=action,
            key=key,
        )

        widget = ToastWidget(request, self._theme)
        widget.dismissed.connect(self._on_dismissed)
        self._toasts.append(widget)

        # 超出可见上限时挤掉最早的一条（规范默认 visibleToasts = 3）。
        #
        # 必须**立即从跟踪列表移除**：dismiss() 是异步的（先播退场动画再回收），
        # 若只调用 dismiss 而不移除，计数不会下降，后续每次新增都会重复挤同一条，
        # 上限形同虚设——连续来十条通知就会十条全都留在屏幕上。
        # 控件自身的退场动画不受影响；稍后 _on_dismissed 再收到该控件时
        # 会发现它已不在列表里，那里已做了容错。
        while len(self._toasts) > TOAST_VISIBLE_MAX:
            departing = self._toasts.pop(0)
            departing.dismiss(animate=True)

        position = self._position_for(len(self._toasts) - 1)
        widget.stack_y = position.y()
        widget.show_at(position)
        return widget

    def info(self, title: str, description: str = "", **kwargs) -> ToastWidget | None:
        """信息通知。"""
        return self.show(title, description, ToastKind.INFO, **kwargs)

    def success(self, title: str, description: str = "", **kwargs) -> ToastWidget | None:
        """成功通知。"""
        return self.show(title, description, ToastKind.SUCCESS, **kwargs)

    def warning(self, title: str, description: str = "", **kwargs) -> ToastWidget | None:
        """警告通知。"""
        return self.show(title, description, ToastKind.WARNING, **kwargs)

    def error(self, title: str, description: str = "", **kwargs) -> ToastWidget | None:
        """错误通知。"""
        return self.show(title, description, ToastKind.ERROR, **kwargs)

    def dismiss_all(self) -> None:
        """关闭全部通知。"""
        for toast in list(self._toasts):
            toast.dismiss(animate=False)
        self._toasts.clear()

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _on_dismissed(self, widget: ToastWidget) -> None:
        """一条通知关闭后重排其余通知。"""
        if widget in self._toasts:
            self._toasts.remove(widget)
        self._relayout()

    def _relayout(self) -> None:
        """重新计算堆叠位置。

        规范提到"不透明度与高度的配合没有公式，需要调到手感对为止"，
        这里的做法是：位置用 200ms 强 ease-out 补间（可中断），
        不随动画改变高度，避免布局抖动。
        """
        for index, toast in enumerate(self._toasts):
            toast.set_stack_position(self._position_for(index), animate=True)

    def _position_for(self, index: int) -> QPoint:
        """计算第 ``index`` 条通知（从底部数起）的屏幕位置。"""
        area = self._available_geometry()
        heights = sum(toast.height() for toast in self._toasts[:index])
        bottom_offset = TOAST_OFFSET + heights + index * TOAST_GAP
        return QPoint(area.right() - LAYOUT["toast_max_width"] + 40 - TOAST_OFFSET,
                      area.bottom() - bottom_offset)

    def _available_geometry(self) -> QRect:
        """返回通知可用的屏幕区域。"""
        from PyQt6.QtGui import QGuiApplication

        if self._anchor_widget is not None:
            handle = self._anchor_widget.windowHandle()
            if handle is not None and handle.screen() is not None:
                return handle.screen().availableGeometry()

        app = QGuiApplication.instance()
        if app is not None:
            screen = app.primaryScreen()
            if screen is not None:
                return screen.availableGeometry()
        return QRect(0, 0, 1920, 1080)
