"""悬浮控制条。

对应产品功能规划四："用户在前台正常操作电脑……如需暂停、切换模型或
调整样式，通过悬浮控制条或全局热键即可完成。"

设计取舍
    控制条是**贴在别人界面上的浮层**，它的存在本身就是一种干扰。
    因此设计目标不是"信息丰富"，而是"尽量不存在"：

    - 高度压缩到 36px，仅保留状态点与 5 个图标按钮；
    - **不做任何持续性动画**——没有呼吸灯、没有旋转指示器。
      规范把"高频出现、纯装饰的动效"列为应当删除的对象，
      而这条控制条会全程显示在用户屏幕上；持续动效在此处是有害的，
      它会持续抢走用户对真正工作内容的注意力；
    - 状态同时用**颜色 + 文字**表达（"翻译中 / 已暂停"），
      满足"信息不依赖颜色单独传达"的无障碍要求；
    - 整条可拖动，位置持久化，用户可把它挪到不碍事的地方。

动效
    显示/隐藏属于"偶尔触发"，使用 200ms 强 ease-out 标准入场动效；
    按钮的按下反馈由 :class:`~pingyitong.ui.widgets.button.Button` 处理。
"""

from __future__ import annotations

from PyQt6.QtCore import QPoint, QRect, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QWidget

from ..constants import MODEL_ACCURATE, MODEL_FAST, MODEL_SHORT_LABELS
from ..utils.logger import get_logger
from .design import LAYOUT, RADIUS, SPACE
from .design.motion import animate_if_occasional
from .design.theme import Theme, base_font
from .widgets import IconButton, SegmentedControl
from .widgets.tooltip import TooltipController

logger = get_logger(__name__)


class FloatingBar(QWidget):
    """桌面悬浮控制条。"""

    #: 请求开启 / 暂停翻译
    toggle_translation = pyqtSignal()
    #: 请求切换模型，参数为模型标识
    model_changed = pyqtSignal(str)
    #: 请求重新框选区域
    pick_region = pyqtSignal()
    #: 请求打开设置面板
    open_settings = pyqtSignal()
    #: 请求打开历史记录
    open_history = pyqtSignal()
    #: 请求展开 / 收起实时译文文本框
    toggle_transcript = pyqtSignal()
    #: 位置变化（用于持久化），参数为左上角坐标
    position_changed = pyqtSignal(int, int)

    def __init__(
        self,
        theme: Theme,
        tooltips: TooltipController | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self._tooltips = tooltips
        self._running = False
        self._drag_offset: QPoint | None = None
        self._status_text = "已暂停"

        self._setup_window()
        self._build_ui()
        self._apply_tooltips()

    # ------------------------------------------------------------------
    # 窗口
    # ------------------------------------------------------------------

    def _setup_window(self) -> None:
        """配置无边框、置顶、不占任务栏的浮层窗口。"""
        self.setWindowTitle("屏译通 · 悬浮控制条")
        self.setAccessibleName("屏译通 · 悬浮控制条")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        self.setFixedHeight(
            LAYOUT["floating_bar_height"] + LAYOUT["floating_bar_padding"] * 2
        )

    def _build_ui(self) -> None:
        """构建控制条内容。"""
        layout = QHBoxLayout(self)
        padding = LAYOUT["floating_bar_padding"]
        layout.setContentsMargins(padding + SPACE["xs"], padding, padding + SPACE["xs"], padding)
        layout.setSpacing(SPACE["xs"])

        # 状态指示点 + 文字：颜色与文字并行表达状态
        self._status_dot = _StatusDot(self._theme, self)
        layout.addWidget(self._status_dot)

        self._status_label = QLabel(self._status_text)
        self._status_label.setObjectName("Secondary")
        self._status_label.setFixedWidth(50)
        self._status_label.setFont(base_font(self._theme, "small"))
        self.set_status(self._status_text)
        layout.addWidget(self._status_label)

        # 模型切换：紧凑分段控件，滑动指示器 250ms 强 ease-in-out
        self._model_switch = SegmentedControl(
            [MODEL_SHORT_LABELS[MODEL_FAST], MODEL_SHORT_LABELS[MODEL_ACCURATE]],
            current=1,
            theme=self._theme,
            min_segment_width=64,
        )
        self._model_switch.setFixedHeight(24)
        self._model_switch.selected_changed.connect(self._on_model_selected)
        layout.addWidget(self._model_switch)

        layout.addSpacing(SPACE["xs"])

        # 操作按钮
        self._toggle_button = self._make_button("▶", "开启 / 暂停翻译")
        self._toggle_button.clicked.connect(self.toggle_translation.emit)
        layout.addWidget(self._toggle_button)

        self._region_button = self._make_button("▣", "框选翻译区域")
        self._region_button.clicked.connect(self.pick_region.emit)
        layout.addWidget(self._region_button)

        self._history_button = self._make_button("☰", "翻译历史")
        self._history_button.clicked.connect(self.open_history.emit)
        layout.addWidget(self._history_button)

        self._settings_button = self._make_button("⚙", "设置")
        self._settings_button.clicked.connect(self.open_settings.emit)
        layout.addWidget(self._settings_button)

        # 实时译文文本框的展开 / 收起。放在最右侧：它控制的是"向下长出内容"，
        # 与前面几个"打开某个窗口"的按钮性质不同，独立在末尾不易误点。
        self._transcript_button = self._make_button("▾", "展开实时译文文本框")
        self._transcript_button.clicked.connect(self.toggle_transcript.emit)
        layout.addWidget(self._transcript_button)

        self.adjustSize()

    def _make_button(self, glyph: str, tooltip: str) -> IconButton:
        """创建图标按钮。"""
        return IconButton(
            glyph,
            tooltip,
            self._theme,
            self,
            size=LAYOUT["floating_icon_size"] + SPACE["xs"],
        )

    def _apply_tooltips(self) -> None:
        """把提示控制器接入所有按钮。

        自研控制器用于满足"首条提示需延迟、后续提示立即出现"的规范要求。
        """
        buttons = (
            (self._toggle_button, "开启 / 暂停翻译"),
            (self._region_button, "框选翻译区域"),
            (self._history_button, "翻译历史"),
            (self._settings_button, "设置"),
            (self._transcript_button, "展开实时译文文本框"),
        )
        for button, text in buttons:
            button.set_tooltip(text if self._tooltips is not None else text, self._tooltips)

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------

    def set_running(self, running: bool) -> None:
        """更新运行状态。"""
        self._running = bool(running)
        self.set_status("翻译中" if running else "已暂停")
        self._toggle_button.setText("⏸" if running else "▶")
        self._status_dot.set_active(self._running)

    def set_model(self, model_id: str) -> None:
        """更新模型选择。

        这里禁用动画：模型可能因自动调度而切换，若每次都播放 250ms 滑动，
        控制条会反复晃动，反而成为干扰源。
        """
        index = 0 if model_id == MODEL_FAST else 1
        self._model_switch.set_current_index(index, animate=False)

    def set_status(self, text: str) -> None:
        """显示简短运行信息（如端到端延迟）。

        按实际像素宽度省略，完整内容可悬停查看或由屏幕阅读器读取。
        保持固定宽度，避免实时延迟变化时控制条反复移动。
        """
        self._status_text = text
        self._status_label.setText(self._status_label.fontMetrics().elidedText(
            text, Qt.TextElideMode.ElideRight, self._status_label.contentsRect().width()
        ))
        self._status_label.setToolTip(text)
        self._status_label.setAccessibleName(text)

    def set_theme(self, theme: Theme) -> None:
        """应用主题。"""
        self._theme = theme
        self._status_label.setFont(base_font(theme, "small"))
        self.set_status(self._status_text)
        self._status_dot.set_theme(theme)
        self._model_switch.set_theme(theme)
        for button in (
            self._toggle_button,
            self._region_button,
            self._history_button,
            self._settings_button,
            self._transcript_button,
        ):
            button.set_theme(theme)
        self.update()

    def set_transcript_open(self, open_: bool) -> None:
        """同步文本框展开状态：箭头方向随之翻转。

        箭头要如实反映"点下去会发生什么"——展开时朝上（点击收起），
        收起时朝下（点击展开）。只靠 tooltip 说明会让人犹豫。
        """
        self._transcript_button.set_glyph("▴" if open_ else "▾")
        self._transcript_button.set_tooltip(
            "收起实时译文文本框" if open_ else "展开实时译文文本框",
            self._tooltips,
        )

    def _on_model_selected(self, index: int) -> None:
        """分段控件选择模型。"""
        self.model_changed.emit(MODEL_FAST if index == 0 else MODEL_ACCURATE)

    # ------------------------------------------------------------------
    # 显示 / 隐藏
    # ------------------------------------------------------------------

    def show_bar(self, geometry: QRect | None = None, animate: bool = True) -> None:
        """显示控制条。

        :param geometry: 目标位置与尺寸；``None`` 表示使用当前位置。
        """
        target = geometry or self.geometry()
        if animate:
            # 显示/隐藏属于"偶尔触发"，适用标准入场动效
            animate_if_occasional(self, target, "popover", "occasional")
        else:
            self.setGeometry(target)
            self.show()
            self.raise_()

    def hide_bar(self) -> None:
        """隐藏控制条（瞬时，不做退场动画）。

        隐藏通常由热键触发，属于高频操作，规范要求不加动画。
        """
        self.hide()

    def move_to_default_position(self) -> None:
        """移动到默认位置：主显示器底部居中偏上。"""
        from PyQt6.QtGui import QGuiApplication

        app = QGuiApplication.instance()
        screen = app.primaryScreen() if app is not None else None
        if screen is None:
            return

        area = screen.availableGeometry()
        self.adjustSize()
        x = area.center().x() - self.width() // 2
        y = area.bottom() - self.height() - 96
        self.move(x, y)

    def _clamped_position(self, x: int, y: int) -> QPoint | None:
        """把给定坐标夹进所在屏幕的可用区域，返回夹取后的位置。

        以 ``(x+20, y+10)`` 探测所属屏幕——与 :meth:`restore_position` 的
        判定保持一致。没有任何屏幕包含该点时返回 ``None``，由调用方决定
        回退策略（restore 回退默认位置，拖动释放保留原值）。
        """
        from PyQt6.QtGui import QGuiApplication

        app = QGuiApplication.instance()
        if app is None:
            return None
        self.adjustSize()
        probe = QPoint(x + 20, y + 10)
        for screen in app.screens():
            area = screen.availableGeometry()
            if area.contains(probe):
                return QPoint(
                    max(area.left(), min(x, area.right() - self.width() + 1)),
                    max(area.top(), min(y, area.bottom() - self.height() + 1)),
                )
        return None

    def restore_position(self, x: int, y: int) -> None:
        """恢复到上次保存的位置，并确保**整个控制条**都落在屏幕内。

        两类偏差都要处理：

        - 显示器拔掉后旧坐标可能完全位于屏幕外，此时回退到默认位置，
          否则控制条会"消失"且用户无法找回；
        - 控制条**加宽**后（例如新增按钮），只检查左上角在屏幕内是不够的——
          右半截会被推出屏幕边缘，右侧的按钮（包括最后的展开箭头）就点不到了。
          因此还要把位置**夹回**所在屏幕的可用区域内。
        """
        clamped = self._clamped_position(x, y)
        if clamped is not None:
            if (clamped.x(), clamped.y()) != (x, y):
                logger.debug(
                    "控制条位置已按屏幕范围收紧：(%s, %s) -> (%s, %s)",
                    x, y, clamped.x(), clamped.y(),
                )
            self.move(clamped)
            return

        logger.debug("保存的控制条位置已不在任何屏幕内，回退到默认位置")
        self.move_to_default_position()

    # ------------------------------------------------------------------
    # 拖动
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:  # noqa: N802
        """开始拖动。"""
        if event.button() == Qt.MouseButton.LeftButton:
            # 拖动期间捕获指针，即使拖出窗口范围也能继续跟随
            self.grabMouse()
            self._drag_offset = event.globalPosition().toPoint() - self.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        """跟随拖动。"""
        if self._drag_offset is None:
            return
        self.move(event.globalPosition().toPoint() - self._drag_offset)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        """结束拖动并持久化位置。

        拖动途中允许跟手越界，但**松手时必须把整条夹回屏幕内**再持久化：
        否则松手时停在屏外的控制条会挂到本次会话结束——``restore_position``
        只在显示与屏幕布局变化时执行，期间右侧的按钮（包括最后的展开箭头）
        一直点不到（实测 0.6.21 曾因此把 ``floating_bar_pos`` 存成屏外坐标）。
        """
        if self._drag_offset is not None:
            self.releaseMouse()
            self._drag_offset = None
            self.setCursor(Qt.CursorShape.SizeAllCursor)
            clamped = self._clamped_position(self.x(), self.y())
            if clamped is not None and clamped != self.pos():
                self.move(clamped)
            self.position_changed.emit(self.x(), self.y())
        super().mouseReleaseEvent(event)

    # ------------------------------------------------------------------
    # 绘制
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        """绘制控制条背景。"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        try:
            scheme = self._theme.scheme
            rect = self.rect().adjusted(0, 0, -1, -1)
            radius = float(RADIUS["md"])
            pill = self.height() / 2.0 - 1.0

            path = QPainterPath()
            path.addRoundedRect(
                float(rect.x()), float(rect.y()),
                float(rect.width()), float(rect.height()),
                min(radius, pill), min(radius, pill),
            )

            # 浮层需要与下层内容区分开，因此使用抬升表面色 + 描边
            painter.setPen(QPen(QColor(scheme.border), 1.0))
            painter.setBrush(QColor(scheme.surface_raised))
            painter.drawPath(path)
        finally:
            painter.end()


class _StatusDot(QWidget):
    """状态指示点。

    颜色 + 位置不承载唯一信息（状态文字始终并排显示），
    因此色觉障碍用户也能准确判断当前状态。
    """

    def __init__(self, theme: Theme, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme = theme
        self._active = False
        self.setFixedSize(8, 8)

    def set_active(self, active: bool) -> None:
        """更新激活状态。"""
        self._active = bool(active)
        self.update()

    def set_theme(self, theme: Theme) -> None:
        """应用主题。"""
        self._theme = theme
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        """绘制圆点。"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        try:
            scheme = self._theme.scheme
            color = QColor(scheme.success) if self._active else QColor(scheme.text_muted)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawEllipse(0, 0, self.width(), self.height())
        finally:
            painter.end()
