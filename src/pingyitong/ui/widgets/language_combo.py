"""语言选择控件。

对应产品功能规划二·（四）"多语言互译支持"。界面要求：

- **自由选择**——下拉里列出全部候选语言，不做人为限制；
- **可选择切换**——点开就是列表，能按名字快速跳转；
- **标注可支持性**——每项用符号标明实测结论（✓ 确定支持 / ~ 疑似支持 /
  ✗ 实测不支持），用户一眼就能看出哪些语言是经过验证的。

**为什么要标注而不敢全都平铺**：模型的官方说法是"支持 200+ 种语言"，
但未公布具体清单。项目用固定探针句逐语言实测（见
:mod:`pingyitong.language_test_results`），实测发现并非每种都可用——
部分语言输出为空或纯属同脚本胡言乱语。把实测结论直接标在选项上，
用户就不必自己去试错。

**为什么不用可编辑下拉框**：一开始做成了"可编辑 + 补全器"的搜索框，
结果连出三个问题——点文本框不展开列表（只有右侧那条细箭头能展开，
用户点了没有反应，看到的是一个空格子）、键入的关键字接在现有标签后面
搜不到东西、失焦后框里的文字与实际生效的语言对不上。
语言选择是个**从固定清单里挑一个**的操作，用普通下拉框最直接：
点哪儿都能展开，框里显示的永远等于实际生效的语言，
快速查找改由 :meth:`LanguageCombo.keyboardSearch` 承担。
"""

from __future__ import annotations

from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QComboBox,
    QListView,
    QVBoxLayout,
    QWidget,
)

from ...languages import LANGUAGES, verification_status
from ..design import LAYOUT, SPACE
from ..design.theme import Theme, base_font

#: 实测档位对应的标记符号。
#:
#: 用符号而非仅颜色：色觉障碍用户同样能分辨（见 UI 设计规范的无障碍要求）。
STATUS_MARKS: dict[str, str] = {
    "verified": "✓",
    "likely": "~",
    "unsupported": "✗",
    "untested": "?",
}

#: 档位的中文说明，用于鼠标悬停提示。
STATUS_TOOLTIPS: dict[str, str] = {
    "verified": "实测确定支持",
    "likely": "实测疑似支持（脚本正确但无法确证语义）",
    "unsupported": "实测不支持（输出为空、未翻译或脚本不符）",
    "untested": "尚未实测",
}


def language_label(code: str, with_status: bool = True) -> str:
    """构造下拉项显示文本。

    :param code: 语言代码。
    :param with_status: 是否附加实测状态标记。
    """
    info = LANGUAGES.get(code)
    if info is None:
        return code

    if not with_status:
        return info.name

    mark = STATUS_MARKS.get(verification_status(code), "?")
    # 同时给出中文名与母语名：母语名能帮用户确认是不是自己想找的那门语言
    # （例如"荷兰语 Nederlands"、"斯瓦希里语 Kiswahili"）。
    if info.native_name and info.native_name != info.name:
        return f"{mark} {info.name} · {info.native_name}"
    return f"{mark} {info.name}"


def search_key(code: str) -> str:
    """返回一项语言的可搜索文本（供关键字跳转使用）。

    包含中文名、英文名、母语名与代码，因此键入 ``冰``、``ice``、``isl``
    都能定位到冰岛语。显示文本以状态符号开头，直接拿它做前缀匹配等于失效，
    所以另建一份用于查找的文本。
    """
    info = LANGUAGES.get(code)
    if info is None:
        return code
    return f"{info.name} {info.english_name} {info.native_name} {code}".lower()


class LanguageCombo(QComboBox):
    """语言下拉框（含实测状态标注与关键字跳转）。

    :param current: 当前选中的语言代码。
    :param with_auto: 是否在首位提供"自动检测"选项（仅源语言需要）。
    :param include_unsupported: 是否列出实测不支持的语言。
        默认列出但标注出来——完全隐藏会让用户以为软件不支持那门语言，
        而实际上可能只是这一次实测未通过。
    """

    #: 选中语言变化，参数为语言代码
    language_changed = pyqtSignal(str)

    #: 箭头指示器占用的宽度（逻辑像素），与样式表里 ``::drop-down`` 的宽度一致
    _INDICATOR_WIDTH = 22

    def __init__(
        self,
        current: str = "en",
        theme: Theme | None = None,
        with_auto: bool = False,
        include_unsupported: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self._codes: list[str] = []
        self._search_keys: list[str] = []

        # 非可编辑：点控件任意位置都能展开列表（Qt 原生行为），
        # 且显示内容永远等于当前选择，不会出现"框里是空的、配置里却有值"。
        self.setEditable(False)
        self.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.setMaxVisibleItems(20)
        self.setFixedHeight(LAYOUT["control_height"])
        self.setMinimumWidth(240)

        # 让下拉列表支持较长内容
        view = QListView()
        view.setUniformItemSizes(True)
        self.setView(view)

        self._populate(with_auto, include_unsupported)
        self.set_language(current)

        if theme is not None:
            self.setFont(base_font(theme, "body"))

        self.currentIndexChanged.connect(self._on_index_changed)

    # ------------------------------------------------------------------
    # 内容
    # ------------------------------------------------------------------

    def _populate(self, with_auto: bool, include_unsupported: bool) -> None:
        """填充语言列表。

        排序策略：确定支持的语言排在前面（用户最可能选到可用的），
        疑似与不支持的排在后面但仍可见。
        """
        self._codes = []
        self._search_keys = []

        if with_auto:
            self.addItem("自动检测", "auto")
            self._codes.append("auto")
            self._search_keys.append("自动检测 auto 自动 auto detect")
            self.setItemData(
                self.count() - 1,
                "由程序判断屏幕文字的语言",
                Qt.ItemDataRole.ToolTipRole,
            )

        ordered: list[tuple[int, str]] = []
        for code in LANGUAGES:
            status = verification_status(code)
            if status == "unsupported" and not include_unsupported:
                continue
            # 排序权重：确定 0、疑似 1、不支持 2
            rank = {"verified": 0, "likely": 1}.get(status, 2)
            ordered.append((rank, code))

        # 稳定排序：同档内保持语言表原有顺序（常用语在前）
        ordered.sort(key=lambda item: item[0])

        for _rank, code in ordered:
            status = verification_status(code)
            self.addItem(language_label(code), code)
            index = self.count() - 1
            self.setItemData(
                index,
                STATUS_TOOLTIPS.get(status, ""),
                Qt.ItemDataRole.ToolTipRole,
            )
            self._codes.append(code)
            self._search_keys.append(search_key(code))

    # ------------------------------------------------------------------
    # 选中
    # ------------------------------------------------------------------

    @property
    def language(self) -> str:
        """当前选中的语言代码。"""
        data = self.currentData()
        if isinstance(data, str) and data:
            return data
        # 理论不可达（非可编辑下拉框的选中项必来自列表），保守回退
        return self._codes[0] if self._codes else "en"

    def set_language(self, code: str) -> None:
        """设置选中语言（不触发信号）。"""
        index = self.findData(code)
        if index < 0:
            index = 0
        blocked = self.blockSignals(True)
        self.setCurrentIndex(index)
        self.blockSignals(blocked)

    def _on_index_changed(self, _index: int) -> None:
        """选中项变化时发出语言代码。"""
        self.language_changed.emit(self.language)

    # ------------------------------------------------------------------
    # 查找与交互
    # ------------------------------------------------------------------

    def keyboardSearch(self, search: str) -> None:  # noqa: N802 - 覆盖 Qt 接口
        """按关键字跳转到匹配的语言。

        非可编辑下拉框的默认行为是按**显示文本前缀**查找，而显示文本以
        状态符号（``✓``）开头，所有项都一样，等于完全失效。
        这里改为在中文名、英文名、母语名与代码中查找，因此键入
        ``冰`` / ``ice`` / ``isl`` 都能跳到冰岛语。

        Qt 会把连续键入的字符累积成一个字符串传进来，并在超时后清空，
        因此这里每次都用完整的 ``search`` 去匹配，从当前项之后开始找，
        反复键入同一个关键字可以在多个匹配项之间循环。
        """
        needle = (search or "").strip().lower()
        if not needle or not self._codes:
            return

        total = len(self._codes)
        start = self.currentIndex() + 1

        for offset in range(total):
            index = (start + offset) % total
            if needle in self._search_keys[index]:
                self.setCurrentIndex(index)
                self._scroll_to_current()
                return

    def showPopup(self) -> None:  # noqa: N802 - 覆盖 Qt 接口
        """展开列表，并把当前选中项滚动到可见位置。

        语言有上百项，展开后若停在列表顶部，用户看不到自己当前选的是哪个。
        """
        super().showPopup()
        self._scroll_to_current()

    def _scroll_to_current(self) -> None:
        """把当前项滚动到列表可视区域内。"""
        view = self.view()
        index = self.currentIndex()
        if index < 0 or not view.isVisible():
            return
        model_index = view.model().index(index, 0)
        if model_index.isValid():
            view.scrollTo(model_index, view.ScrollHint.PositionAtCenter)

    def paintEvent(self, event: object) -> None:  # noqa: N802 - 覆盖 Qt 接口
        """在右侧绘制指示箭头。

        样式表把 ``::drop-down`` 的边框去掉后 Qt 就不再画默认箭头，
        界面因此看不出这里是个可选列表。这里自己画一个下折角，
        颜色取自主题的次要文字色。
        """
        super().paintEvent(event)

        if self._theme is None:
            return

        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            color = QColor(self._theme.scheme.text_secondary)
            if not self.isEnabled():
                color.setAlpha(120)
            pen = QPen(color)
            pen.setWidthF(1.6)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)

            center_x = self.width() - self._INDICATOR_WIDTH / 2.0
            center_y = self.height() / 2.0
            half = 3.5
            drop = 2.0
            painter.drawLine(
                QPointF(center_x - half, center_y - drop),
                QPointF(center_x, center_y + drop),
            )
            painter.drawLine(
                QPointF(center_x, center_y + drop),
                QPointF(center_x + half, center_y - drop),
            )
        finally:
            painter.end()

    def set_theme(self, theme: Theme) -> None:
        """应用主题。"""
        self._theme = theme
        self.setFont(base_font(theme, "body"))
        self.update()


class LanguageRow(QWidget):
    """源语言 + 目标语言的成对选择行。

    :param source: 初始源语言代码（``auto`` 表示自动检测）。
    :param target: 初始目标语言代码。
    """

    #: 语言对变化：(源语言, 目标语言)
    languages_changed = pyqtSignal(str, str)

    def __init__(
        self,
        source: str = "auto",
        target: str = "zh",
        theme: Theme | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACE["sm"])

        self.source_combo = LanguageCombo(source, theme, with_auto=True)
        self.target_combo = LanguageCombo(target, theme, with_auto=False)
        layout.addWidget(self.source_combo)
        layout.addWidget(self.target_combo)

        self.source_combo.language_changed.connect(self._emit)
        self.target_combo.language_changed.connect(self._emit)

    def _emit(self, _code: str = "") -> None:
        """发出语言对变化。"""
        self.languages_changed.emit(self.source_combo.language, self.target_combo.language)

    @property
    def source_language(self) -> str:
        """当前源语言代码。"""
        return self.source_combo.language

    @property
    def target_language(self) -> str:
        """当前目标语言代码。"""
        return self.target_combo.language

    def set_languages(self, source: str, target: str) -> None:
        """设置语言对（不触发信号）。"""
        self.source_combo.set_language(source)
        self.target_combo.set_language(target)

    def set_theme(self, theme: Theme) -> None:
        """应用主题。"""
        self._theme = theme
        self.source_combo.set_theme(theme)
        self.target_combo.set_theme(theme)
