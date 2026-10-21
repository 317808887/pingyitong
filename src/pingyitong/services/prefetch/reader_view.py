"""纯文本双栏对照阅读器（联网预翻译 P3，自研）。

对应方案 v1.2 §3.7 reader_view / D2：网页翻译 = 粘贴 URL 抓静态正文后
在**内置纯文本双栏视图**里对照阅读——左栏原文先行，右栏译文逐块替换。

**明确不引入 QWebEngine/Chromium**（§9）：所有抽取文本按纯文本渲染
（QLabel 默认不解析 HTML），天然不执行任何脚本。

行锁语义（§6.3）同样适用：一个块的译文一旦写入，在其 key 的 TTL 内
不被后续重译替换（由缓存键决定，视图层无状态）。
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ...utils.logger import get_logger

logger = get_logger(__name__)


class _BlockRow(QWidget):
    """一对（原文 | 译文）行。"""

    def __init__(self, index: int, source: str, theme, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._index = index
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.source_label = QLabel(source)
        self.source_label.setTextFormat(Qt.TextFormat.PlainText)
        self.source_label.setWordWrap(True)
        self.source_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.source_label.setObjectName("ReaderSource")

        self.translated_label = QLabel("…")
        self.translated_label.setTextFormat(Qt.TextFormat.PlainText)
        self.translated_label.setWordWrap(True)
        self.translated_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.translated_label.setObjectName("ReaderTranslated")

        layout.addWidget(self.source_label, 1)
        layout.addWidget(self.translated_label, 1)

    def set_translation(self, text: str) -> None:
        self.translated_label.setText(text)


class ReaderView(QWidget):
    """双栏对照视图：set_blocks 先显示原文，update_translation 逐块替换。"""

    def __init__(self, theme, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rows: list[_BlockRow] = []

        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget()
        self._layout = QVBoxLayout(container)
        self._layout.setContentsMargins(8, 8, 8, 8)
        self._layout.setSpacing(6)
        self._layout.addStretch(1)
        self._scroll.setWidget(container)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._scroll)

    def set_blocks(self, blocks: list[str]) -> None:
        """替换全部原文块；译文先显示占位，等预翻译/实时翻译回填。"""
        for row in self._rows:
            row.setParent(None)
            row.deleteLater()
        self._rows.clear()
        for index, block in enumerate(blocks):
            row = _BlockRow(index, block, None)
            self._layout.insertWidget(self._layout.count() - 1, row)
            self._rows.append(row)

    def update_translation(self, index: int, text: str) -> None:
        """句/块级替换译文（越界静默忽略——视图与数据异步）。"""
        if 0 <= index < len(self._rows):
            self._rows[index].set_translation(text)

    def block_count(self) -> int:
        return len(self._rows)
