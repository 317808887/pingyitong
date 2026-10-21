"""实时译文文本框（自研）。

对应需求：点悬浮栏右侧的向下箭头后，在下方展开一个文本框，
**译文实时显示在里面**，文字按界面上"从上到下、从左到右"的阅读顺序排列，
以便连起来读依然通顺。

**为什么不用现成的折叠面板组件**：调研过 GitHub 上的方案，都不合适——

- ``bec-project/bec_widgets`` 的 ``CollapsiblePanelManager`` 依赖 ``qtpy``
  与它自己一整套控件框架，视觉上也与本项目的设计令牌对不上；
- Stack Overflow 上流传的 ``Spoiler`` 写法基于 PyQt4/5 的旧 API；
- Wireshark 的 ``AccordionFrame`` 是 C++ 实现，无法直接用。

而"折叠 + 淡入"在本项目里只需要几十行：项目已有
:mod:`pingyitong.ui.design.motion` 的动画工具与设计令牌。故自行实现，
**因为不是取自 GitHub 的组件，项目文档无需新增来源标注。**

动画实现要点（这几条都是踩过的坑）
    1. 直接动画 ``height``/``width`` **无效**——QWidget 的几何不是可动画属性，
       必须动 ``maximumHeight``/``minimumHeight``（或自定义 Property）；
    2. 展开前要先让布局算出"自然高度"，否则动画没有目标值，
       表现为"点一下闪一下就没了"；
    3. 动画结束后必须**复位高度约束**，否则窗口被永久锁定在动画终值上，
       内容变化时无法自适应。
"""

from __future__ import annotations

from difflib import SequenceMatcher

from PyQt6.QtCore import (
    QEasingCurve,
    QObject,
    QPropertyAnimation,
    Qt,
    pyqtProperty,
    pyqtSignal,
)
from PyQt6.QtGui import QCloseEvent, QPainter, QTextCursor
from PyQt6.QtWidgets import (
    QFrame,
    QPlainTextEdit,
    QStyle,
    QStyleOption,
    QVBoxLayout,
    QWidget,
)

from ...utils.logger import get_logger
from ..design import DURATION, SPACE
from ..design.theme import Theme, base_font

logger = get_logger(__name__)

#: 文本框默认尺寸（逻辑像素）。
#:
#: 宽度**必须与悬浮栏一致**（用户要求两者对齐）——悬浮栏宽度由内容
#: 自适应决定，展开时直接取锚点宽度；只有完全拿不到锚点（程序化展开）
#: 才用这里的回退值。高度取 1080p 屏幕下约一半高度——比这更高会盖住
#: 大半屏幕，读起来反而累。
DEFAULT_PANEL_WIDTH = 428
DEFAULT_PANEL_HEIGHT = 560


class _HeightAnimator(QObject):
    """把窗口高度包装成**可动画**的 Qt 属性。

    为什么需要这一层：QWidget 的 ``height`` 不是 Qt 属性，没法直接做动画；
    而只动画 ``maximumHeight`` 又不够——它只是"允许"长到这么高，
    窗口的实际高度仍取决于布局的 sizeHint（文本框的 sizeHint 很小），
    实测表现为"点了箭头，面板纹丝不动"。必须真正 setFixedHeight 才会长出来。
    """

    def __init__(self, widget: QWidget) -> None:
        super().__init__(widget)
        self._widget = widget

    def _get_height(self) -> int:
        return self._widget.height()

    def _set_height(self, value: int) -> None:
        self._widget.setFixedHeight(max(0, int(value)))

    #: 动画用的高度属性
    height_value = pyqtProperty(int, fget=_get_height, fset=_set_height)


class TranscriptPanel(QFrame):
    """跟随悬浮栏展开 / 收起的实时译文文本框。

    :param theme: 主题。
    :param parent: 父控件；通常传悬浮栏，以便跟随其移动。
    """

    #: 展开状态变化，参数为是否展开
    expanded_changed = pyqtSignal(bool)

    def __init__(self, theme: Theme, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme = theme
        self._expanded = False
        self._target_height = DEFAULT_PANEL_HEIGHT
        self._animation: QPropertyAnimation | None = None
        self._height_animator = _HeightAnimator(self)

        self._setup_window()
        self._build_ui()

    # ------------------------------------------------------------------
    # 窗口与内容
    # ------------------------------------------------------------------

    def _setup_window(self) -> None:
        """配置为可从任务栏和窗口切换找到的无边框置顶阅读窗口。"""
        self.setWindowTitle("屏译通 · 实时译文")
        self.setAccessibleName("屏译通 · 实时译文")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Window
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        # 半透明无框窗口上，样式表的 background-color 需要**显式开启**
        # WA_StyledBackground 才会绘制。漏掉这一行的实测表现是：
        # 面板完全透明，只有文字浮在桌面上，看不出这是一个面板。
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setObjectName("TranscriptPanel")
        self.setFixedWidth(DEFAULT_PANEL_WIDTH)

    def _build_ui(self) -> None:
        """构建文本框。"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            SPACE["sm"], SPACE["sm"], SPACE["sm"], SPACE["sm"]
        )
        layout.setSpacing(0)

        self._text = QPlainTextEdit(self)
        self._text.setObjectName("TranscriptText")
        self._text.setAccessibleName("实时译文内容")
        self._text.setReadOnly(True)
        self._text.setUndoRedoEnabled(False)
        # 只读文本框仍要能选中复制，因此保留文本交互但不接受编辑
        self._text.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        self._text.setFrameShape(QFrame.Shape.NoFrame)
        self._text.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self._text.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._text.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._text.setFont(base_font(self._theme, "body"))
        layout.addWidget(self._text)

    def paintEvent(self, event) -> None:  # noqa: N802 - 覆盖 Qt 接口
        """绘制面板样式表背景。

        ``WA_TranslucentBackground`` 会禁用顶层窗口的自动背景填充，
        而 :class:`QFrame` 的默认 ``paintEvent`` 在 Qt 6.11 下**不会**代绘
        样式表背景——实测表现为面板完全透明，只剩文字浮在桌面上。
        标准做法是主动走一次 ``PE_Widget``，让样式表的
        ``#TranscriptPanel`` 规则真正落到屏幕上。
        """
        option = QStyleOption()
        option.initFrom(self)
        painter = QPainter(self)
        try:
            self.style().drawPrimitive(
                QStyle.PrimitiveElement.PE_Widget, option, painter, self
            )
        finally:
            painter.end()
        super().paintEvent(event)

    # ------------------------------------------------------------------
    # 展开 / 收起
    # ------------------------------------------------------------------

    @property
    def is_expanded(self) -> bool:
        """文本框当前是否展开。"""
        return self._expanded

    @property
    def ignorable_rect(self):
        """返回用于 OCR 忽略的完整矩形（全局逻辑坐标）。

        **不要用 ``geometry()`` 配合 ``mapToGlobal`` 现算**：展开动画期间
        ``height()`` 在从 0 长到目标值的路上，此刻取到的矩形只有几十像素高
        ——动画结束后面板大部分区域不在忽略区内，其上的文字会被 OCR 读走
        （实测文本框全文 748px 高被识别成一段乱码并入翻译）。这里直接
        返回**目标高度**，与动画进度无关。
        """
        if not self.isVisible():
            return None
        from ...components.capture.geometry import Rect

        top_left = self.mapToGlobal(self.rect().topLeft())
        height = self._target_height if self._expanded else self.height()
        return Rect(top_left.x(), top_left.y(), self.width(), max(1, height))

    def toggle(self, anchor: QWidget | None = None) -> bool:
        """切换展开状态并返回新状态。"""
        if self._expanded:
            self.collapse()
        else:
            self.expand(anchor)
        return self._expanded

    def expand(self, anchor: QWidget | None = None) -> None:
        """展开文本框，并贴到 ``anchor`` **下方**。

        **只向下，永不向上**（用户明确要求）：悬浮栏下方空间不足时，
        缩短文本框高度而不是把它翻到悬浮栏上方——向上展开会盖住
        屏幕上大块内容，与"在悬浮栏下方生成文本框"的直觉相悖。
        """
        if anchor is not None:
            self._place_below(anchor)

        if self._expanded:
            # 已展开时的再次调用来自"悬浮栏被拖动后的跟随"：只需更新位置
            # 与高度约束。重发 expanded_changed 会把"翻译层原本可见"的恢复
            # 状态覆盖成 False（此刻翻译层正被文本框功能隐藏着），收起文本
            # 框后翻译层就不再恢复；重播动画则会让面板先缩到 0 再长回来，
            # 肉眼可见地闪一下。
            target = self._target_height or DEFAULT_PANEL_HEIGHT
            if self._animation is None:
                self.setMinimumHeight(target)
                self.setMaximumHeight(target)
            return

        target = self._target_height or DEFAULT_PANEL_HEIGHT
        self._expanded = True
        self.show()
        self.raise_()
        self._animate_height(0, target)
        self.expanded_changed.emit(True)

    def collapse(self) -> None:
        """收起文本框。"""
        if not self._expanded:
            return
        self._expanded = False
        current = max(1, self.height())
        self._animate_height(current, 0)
        self.expanded_changed.emit(False)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt 接口
        """窗口管理器关闭与收起按钮共用状态切换，恢复原位译文。"""
        event.ignore()
        if self._expanded:
            self.collapse()
        elif self._animation is None:
            self.hide()

    def _place_below(self, anchor: QWidget) -> None:
        """把面板放在锚点控件**正下方**，高度收缩到剩余空间内。

        悬浮栏可以被拖到屏幕任意位置。曾被实现为"下方放不下就翻到上方"，
        用户明确否决：面板出现在悬浮栏上方会盖住屏幕上大块内容。
        因此现在**永远向下**：顶部贴住悬浮栏底边，高度取
        ``min(期望高度, 屏幕底部剩余空间)``。用户想要更高的文本框，
        把悬浮栏往上拖即可——行为可预测。
        """
        from PyQt6.QtGui import QGuiApplication

        screen = anchor.screen() or QGuiApplication.primaryScreen()
        if screen is None:
            return

        area = screen.availableGeometry()
        bar_bottom_left = anchor.mapToGlobal(anchor.rect().bottomLeft())
        # 宽度与悬浮栏完全一致（用户要求）：悬浮栏变宽变窄，文本框跟着变
        width = anchor.width()
        x = bar_bottom_left.x()
        x = max(area.left(), min(x, area.right() - width))

        y = bar_bottom_left.y() + SPACE["xs"]
        available = area.bottom() - y + 1
        self._target_height = max(1, min(DEFAULT_PANEL_HEIGHT, available))

        self.setFixedWidth(width)
        self.move(x, y)

    # ------------------------------------------------------------------
    # 内容
    # ------------------------------------------------------------------

    def set_text(self, text: str) -> None:
        """替换文本框内容。

        滚动位置的处理：用户往下翻看历史时不要把他拽回顶部，
        只有原本就在底部（或内容为空）才自动跟随最新译文。
        """
        previous = self._text.toPlainText()
        if previous == text:
            return

        scrollbar = self._text.verticalScrollBar()
        scroll_position = scrollbar.value()
        at_bottom = scroll_position >= scrollbar.maximum() - 4

        # Live OCR updates often change one line. Replacing the entire document
        # resets scrolling and removes the user's selection, even elsewhere.
        # Keep the unchanged prefix/suffix in the same Qt document instead.
        prefix = 0
        common_length = min(len(previous), len(text))
        while prefix < common_length and previous[prefix] == text[prefix]:
            prefix += 1
        suffix = 0
        while (
            suffix < common_length - prefix
            and previous[-suffix - 1] == text[-suffix - 1]
        ):
            suffix += 1

        previous_end = len(previous) - suffix
        replacement_end = len(text) - suffix
        # Python counts Unicode characters, while QTextCursor uses UTF-16 units.
        # This conversion prevents an emoji before the edit from shifting it.
        reader_cursor = self._text.textCursor()
        old_lines = previous[prefix:previous_end].splitlines(keepends=True)
        new_lines = text[prefix:replacement_end].splitlines(keepends=True)
        offsets = [prefix]
        for line in old_lines:
            offsets.append(offsets[-1] + len(line))
        aligned = list(zip(old_lines, new_lines, strict=False))
        if sum(old == new for old, new in aligned) >= max(len(old_lines), len(new_lines)) / 2:
            # Stable screen rows are usually aligned already. This linear path
            # also avoids quadratic matching of hundreds of repeated lines.
            changes = [
                ("replace", index, index + 1, index, index + 1)
                for index, (old, new) in enumerate(aligned) if old != new
            ]
            if len(old_lines) != len(new_lines):
                changes.append((
                    "replace", len(aligned), len(old_lines), len(aligned), len(new_lines)
                ))
        else:
            changes = SequenceMatcher(None, old_lines, new_lines).get_opcodes()
        # Apply from the end so each old position stays valid. Matching lines
        # in the middle retain their selection even when both ends change.
        for action, old_start, old_end, new_start, new_end in reversed(changes):
            if action == "equal":
                continue
            start_position = len(previous[:offsets[old_start]].encode("utf-16-le")) // 2
            end_position = len(previous[:offsets[old_end]].encode("utf-16-le")) // 2
            edit_cursor = QTextCursor(self._text.document())
            edit_cursor.setPosition(start_position)
            edit_cursor.setPosition(end_position, QTextCursor.MoveMode.KeepAnchor)
            edit_cursor.insertText("".join(new_lines[new_start:new_end]))
        self._text.setTextCursor(reader_cursor)
        if at_bottom:
            scrollbar.setValue(scrollbar.maximum())
        else:
            scrollbar.setValue(min(scroll_position, scrollbar.maximum()))

    def text(self) -> str:
        """返回当前文本框内容。"""
        return self._text.toPlainText()

    def clear(self) -> None:
        """清空内容。"""
        self._text.clear()

    def set_theme(self, theme: Theme) -> None:
        """应用主题。"""
        self._theme = theme
        self._text.setFont(base_font(theme, "body"))

    # ------------------------------------------------------------------
    # 动画
    # ------------------------------------------------------------------

    def _animate_height(self, start: int, end: int) -> None:
        """在给定高度之间做动画。

        动的是 ``maximumHeight``（见模块文档说明为何不能用 ``height``），
        并在结束后复位约束，使内容变化时仍能自适应。
        """
        if self._animation is not None:
            self._animation.stop()

        self._height_animator = _HeightAnimator(self)
        self.setFixedHeight(start)

        animation = QPropertyAnimation(self._height_animator, b"height_value", self)
        animation.setDuration(DURATION["accordion"])
        animation.setStartValue(start)
        animation.setEndValue(max(0, int(end)))
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        animation.finished.connect(lambda: self._on_animation_finished(end))
        self._animation = animation
        animation.start()

    def _on_animation_finished(self, end: int) -> None:
        """动画结束后复位高度约束与可见性。

        必须解除 ``setFixedHeight`` 施加的双向约束，否则窗口会被永久锁死在
        动画终值上，之后内容变多也不会自适应。
        """
        if end <= 0:
            self.setMinimumHeight(0)
            self.setMaximumHeight(0)
            self.hide()
        else:
            target = self._target_height or DEFAULT_PANEL_HEIGHT
            self.setMinimumHeight(target)
            self.setMaximumHeight(target)
        self._animation = None

    def hide_panel(self) -> None:
        """立即收起（不播动画），用于程序退出或界面重置。"""
        if self._animation is not None:
            self._animation.stop()
            self._animation = None
        self._expanded = False
        self.setMinimumHeight(0)
        self.setMaximumHeight(16777215)
        self.hide()
