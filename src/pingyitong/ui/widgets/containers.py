"""容器与设置行组件。

设置面板的视觉结构由三个组件构成：

- :class:`Card` —— 成组的设置项容器；
- :class:`SettingRow` —— 单条"标签 + 说明 + 控件"的设置行；
- :class:`SectionHeader` —— 分组标题。

设计规范中与本模块相关的两点：

1. **一致性来自聚合**——所有设置行共用同一套间距、行高与对齐规则，
   用户不会被"这一行比上一行高 2px"这类细节干扰；
2. **长文说明必须可见但克制**——说明文字使用次要色与更小字号，
   不抢主标签的注意力，但始终可见（不使用悬停才显示的提示，
   因为那会让用户漏读关键约束）。
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..design import LAYOUT, SPACE
from ..design.theme import Theme


class Divider(QFrame):
    """水平分隔线。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Divider")
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFixedHeight(1)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)


class Card(QWidget):
    """设置分组容器。

    :param title: 分组标题；为空时不渲染标题区域。
    :param description: 分组说明，展示在标题下方。
    """

    def __init__(
        self,
        title: str = "",
        description: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("Surface")

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(
            SPACE["lg"], SPACE["lg"], SPACE["lg"], SPACE["lg"]
        )
        self._layout.setSpacing(SPACE["sm"])

        if title:
            title_label = QLabel(title)
            title_label.setObjectName("Subtitle")
            self._layout.addWidget(title_label)

        if description:
            description_label = QLabel(description)
            description_label.setObjectName("Secondary")
            description_label.setWordWrap(True)
            self._layout.addWidget(description_label)

        if title or description:
            self._layout.addSpacing(SPACE["xs"])

        #: 供外部添加设置行
        self.body = QVBoxLayout()
        self.body.setContentsMargins(0, 0, 0, 0)
        self.body.setSpacing(SPACE["md"])
        self._layout.addLayout(self.body)
        self._layout.addStretch(1)

    def add_row(self, row: SettingRow) -> SettingRow:
        """添加一条设置行。"""
        self.body.addWidget(row)
        return row

    def add_widget(self, widget: QWidget) -> QWidget:
        """添加自定义控件。"""
        self.body.addWidget(widget)
        return widget


class SettingRow(QWidget):
    """单条设置行：左侧标签与说明，右侧控件。

    :param label: 设置名称。
    :param description: 补充说明（可选），用于解释该设置的取舍。
    :param control: 右侧控件。
    :param stack: 为 ``True`` 时控件另起一行占满宽度，
        适用于需要较宽空间的自定义控件（如样式预览、区域选择）。
    """

    def __init__(
        self,
        label: str,
        control: QWidget,
        description: str = "",
        theme: Theme | None = None,
        stack: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self._label_text = label

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(SPACE["xs"])

        if stack:
            row = QVBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(SPACE["sm"])
            row.addWidget(self._make_label_block(label, description))
            row.addWidget(control)
        else:
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(SPACE["lg"])
            row.addWidget(self._make_label_block(label, description), 1)

            control.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            row.addWidget(control, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        outer.addLayout(row)
        self.setMinimumHeight(LAYOUT["row_min_height"])

    def _make_label_block(self, label: str, description: str) -> QWidget:
        """构造左侧标签区域。"""
        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)

        name = QLabel(label)
        name.setWordWrap(True)
        layout.addWidget(name)

        if description:
            hint = QLabel(description)
            hint.setObjectName("Secondary")
            hint.setWordWrap(True)
            layout.addWidget(hint)

        return container

    @property
    def label_text(self) -> str:
        """设置名称。"""
        return self._label_text


class SectionHeader(QLabel):
    """分组标题。"""

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setObjectName("SectionHeader")


class StatusPill(QLabel):
    """状态徽标（如"已加载"、"未安装"）。

    无障碍说明：状态文字使用明确的词（"已加载"）而不是仅靠颜色区分，
    满足"信息不依赖颜色单独传达"的要求。
    """

    def __init__(
        self,
        text: str = "",
        tone: str = "info",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(text, parent)
        self.set_tone(tone)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def set_tone(self, tone: str) -> None:
        """设置色调：``info`` / ``success`` / ``warning``。"""
        object_names = {
            "info": "Badge",
            "success": "BadgeSuccess",
            "warning": "BadgeWarning",
        }
        self.setObjectName(object_names.get(tone, "Badge"))
        # 更换 objectName 后需重新应用样式表
        style = self.style()
        if style is not None:
            style.unpolish(self)
            style.polish(self)
        self.update()

    def set_status(self, text: str, tone: str = "info") -> None:
        """同时更新文字与色调。"""
        self.setText(text)
        self.set_tone(tone)
