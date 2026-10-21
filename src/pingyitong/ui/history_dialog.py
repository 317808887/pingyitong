"""翻译历史记录浏览器。

对应产品功能规划三·（五）：按时间、关键词检索，支持批量导出为 TXT/CSV。

隐私提示
    历史记录里可能包含屏幕上的敏感文字。因此本对话框：

    - 在界面上明确写出"记录仅保存在本机"；
    - 提供"清空历史"入口，且该操作需要二次确认；
    - 当用户在设置中关闭了"记录原文"时，原文列显示为占位提示
      而不是空白——让用户明确知道这是自己的选择，而不是程序出了故障。
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..services import HistoryService
from ..utils.logger import get_logger
from .design import SPACE
from .design.theme import Theme
from .widgets import Button, ToastManager

logger = get_logger(__name__)


class HistoryDialog(QDialog):
    """翻译历史浏览与导出。"""

    #: 单次加载的最大条数。超过这个数量界面会变得难以滚动，
    #: 用户应当改用关键词缩小范围。
    PAGE_SIZE = 300

    def __init__(
        self,
        history: HistoryService,
        theme: Theme,
        toasts: ToastManager | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._history = history
        self._theme = theme
        self._toasts = toasts
        self._keyword = ""

        self.setWindowTitle("翻译历史")
        self.setMinimumSize(760, 480)
        self.setModal(False)

        self._build_ui()
        self.reload()

    # ------------------------------------------------------------------
    # 界面
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """构建界面。"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACE["xl"], SPACE["xl"], SPACE["xl"], SPACE["xl"])
        layout.setSpacing(SPACE["md"])

        # 检索行
        search_row = QHBoxLayout()
        search_row.setSpacing(SPACE["sm"])

        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("输入关键词检索原文或译文…")
        self._search_input.setClearButtonEnabled(True)
        self._search_input.returnPressed.connect(self._on_search)
        search_row.addWidget(self._search_input, 1)

        search_button = Button("检索", "primary", self._theme)
        search_button.clicked.connect(self._on_search)
        search_row.addWidget(search_button)

        layout.addLayout(search_row)

        # 结果统计
        self._summary = QLabel()
        self._summary.setObjectName("Secondary")
        layout.addWidget(self._summary)

        # 结果表格
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["时间", "原文", "译文", "模型 / 缓存"])
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setAlternatingRowColors(False)
        self._table.verticalHeader().setVisible(False)
        self._table.setWordWrap(False)

        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)

        layout.addWidget(self._table, 1)

        # 隐私说明：常驻显示，不藏在提示里
        privacy = QLabel("历史记录仅保存在本机数据库中，不会上传或同步。")
        privacy.setObjectName("Muted")
        layout.addWidget(privacy)

        # 操作按钮
        buttons = QHBoxLayout()
        buttons.setSpacing(SPACE["sm"])

        export_csv = Button("导出 CSV…", "secondary", self._theme)
        export_csv.clicked.connect(lambda: self._export("csv"))
        buttons.addWidget(export_csv)

        export_txt = Button("导出 TXT…", "secondary", self._theme)
        export_txt.clicked.connect(lambda: self._export("txt"))
        buttons.addWidget(export_txt)

        buttons.addStretch(1)

        clear_button = Button("清空历史", "danger", self._theme)
        clear_button.clicked.connect(self._on_clear)
        buttons.addWidget(clear_button)

        close_button = Button("关闭", "secondary", self._theme)
        close_button.clicked.connect(self.close)
        buttons.addWidget(close_button)

        layout.addLayout(buttons)

    # ------------------------------------------------------------------
    # 数据
    # ------------------------------------------------------------------

    def reload(self) -> None:
        """重新加载记录。"""
        records = self._history.search(
            keyword=self._keyword, limit=self.PAGE_SIZE
        )
        total = self._history.count(self._keyword)

        self._table.setRowCount(0)
        for row_index, record in enumerate(records):
            self._table.insertRow(row_index)
            self._table.setItem(row_index, 0, QTableWidgetItem(record.created_at_display))

            source_item = QTableWidgetItem(
                record.source_text if record.source_text else "（未记录原文）"
            )
            source_item.setToolTip(record.source_text)
            self._table.setItem(row_index, 1, source_item)

            target_item = QTableWidgetItem(record.target_text)
            target_item.setToolTip(record.target_text)
            self._table.setItem(row_index, 2, target_item)

            model_text = record.model_id or "—"
            if record.from_cache:
                model_text += " · 缓存命中"
            self._table.setItem(row_index, 3, QTableWidgetItem(model_text))

        if total > len(records):
            self._summary.setText(
                f"显示最近 {len(records)} 条，共 {total} 条。"
                f"请使用关键词缩小范围以查看更多。"
            )
        elif total == 0:
            self._summary.setText(
                "暂无匹配记录。" if self._keyword else "暂无翻译记录。"
            )
        else:
            self._summary.setText(f"共 {total} 条记录。")

    def _on_search(self) -> None:
        """执行检索。"""
        self._keyword = self._search_input.text().strip()
        self.reload()

    # ------------------------------------------------------------------
    # 导出与清空
    # ------------------------------------------------------------------

    def _export(self, fmt: str) -> None:
        """导出历史记录。"""
        if self._history.count(self._keyword) == 0:
            if self._toasts is not None:
                self._toasts.info("没有可导出的记录")
            return

        suffix = "csv" if fmt == "csv" else "txt"
        filters = "CSV 文件 (*.csv)" if fmt == "csv" else "文本文件 (*.txt)"
        path_text, _ = QFileDialog.getSaveFileName(
            self,
            "导出翻译历史",
            f"屏译通-翻译历史.{suffix}",
            filters,
        )
        if not path_text:
            return

        from pathlib import Path

        target = Path(path_text)
        if fmt == "csv":
            count = self._history.export_csv(target, keyword=self._keyword)
        else:
            count = self._history.export_txt(target, keyword=self._keyword)

        if count > 0:
            logger.info("已导出 %s 条历史到 %s", count, target)
            if self._toasts is not None:
                self._toasts.success("导出完成", f"已写入 {count} 条记录")
        elif self._toasts is not None:
            self._toasts.error("导出失败", "请检查目标文件是否被占用")

    def _on_clear(self) -> None:
        """清空历史（需二次确认）。"""
        total = self._history.count()
        if total == 0:
            if self._toasts is not None:
                self._toasts.info("历史记录已为空")
            return

        # 破坏性操作必须二次确认，且默认焦点在"取消"上
        answer = QMessageBox.warning(
            self,
            "清空翻译历史",
            f"将永久删除全部 {total} 条历史记录，此操作无法撤销。\n\n是否继续？",
            QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Yes,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        deleted = self._history.clear()
        self.reload()
        if self._toasts is not None:
            self._toasts.success("已清空历史", f"共删除 {deleted} 条记录")
