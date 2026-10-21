"""设置项输入组件：滑块行、颜色行、字体行。

这三个组件是"翻译样式自定义"（产品功能规划三·（三））的界面载体，
共同点是把"数值/颜色/字体"这类抽象参数变成**可即时预览**的操作。

设计说明
    规范强调"良好的默认值比大量选项更重要"。因此：

    - 每个滑块都给出默认值，并在拖动时实时显示当前数值；
    - 颜色选择提供"跟随原文背景"的推荐值，而不是让用户从色轮里挑；
    - 字体列表只列出系统中确实存在的中文字体，避免选到缺字字体后
      译文变成方框——这是中文界面最容易踩的坑。
"""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFontDatabase
from PyQt6.QtWidgets import (
    QColorDialog,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..design import LAYOUT, SPACE
from ..design.theme import Theme


class SliderRow(QWidget):
    """带数值显示的滑块。

    :param label: 名称。
    :param minimum: 最小值。
    :param maximum: 最大值。
    :param value: 当前值。
    :param suffix: 数值后缀（如 ``"%"``、``"px"``）。
    :param scale: 显示时的缩放系数（例如内部存 0.0-1.0，界面显示 0-100）。
    :param on_change: 数值变化回调。
    """

    value_changed = pyqtSignal(float)

    def __init__(
        self,
        label: str,
        minimum: float,
        maximum: float,
        value: float,
        theme: Theme | None = None,
        suffix: str = "",
        scale: float = 1.0,
        decimals: int = 0,
        on_change: Callable[[float], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self._scale = scale
        self._decimals = decimals
        self._suffix = suffix
        self._on_change = on_change
        self._label_text = label

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACE["xxs"])

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(SPACE["sm"])

        self._label = QLabel(label)
        header.addWidget(self._label)
        header.addStretch(1)

        self._value_label = QLabel()
        self._value_label.setObjectName("Secondary")
        header.addWidget(self._value_label)
        layout.addLayout(header)

        # 滑块内部使用整数刻度，避免 Qt 的整型限制导致精度丢失
        self._factor = 10 ** decimals
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(int(minimum * self._factor))
        self._slider.setMaximum(int(maximum * self._factor))
        self._slider.setValue(int(value * self._factor))
        self._slider.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._slider.valueChanged.connect(self._handle_change)
        layout.addWidget(self._slider)

        self._update_value_label(value)

    def _handle_change(self, raw: int) -> None:
        """滑块数值变化。"""
        value = raw / self._factor
        self._update_value_label(value)
        self.value_changed.emit(value)
        if self._on_change is not None:
            self._on_change(value)

    def _update_value_label(self, value: float) -> None:
        """刷新数值显示。"""
        displayed = value * self._scale
        text = f"{displayed:.{self._decimals}f}" if self._decimals else f"{displayed:.0f}"
        self._value_label.setText(f"{text}{self._suffix}")

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------

    @property
    def value(self) -> float:
        """当前值。"""
        return self._slider.value() / self._factor

    def set_value(self, value: float) -> None:
        """设置当前值（不触发回调）。"""
        blocked = self._slider.blockSignals(True)
        self._slider.setValue(int(value * self._factor))
        self._slider.blockSignals(blocked)
        self._update_value_label(value)

    def set_theme(self, theme: Theme) -> None:
        """应用主题。"""
        self._theme = theme


class ColorRow(QWidget):
    """颜色选择行：色块 + 十六进制值 + 选择按钮。"""

    color_changed = pyqtSignal(str)

    def __init__(
        self,
        label: str,
        color: str,
        theme: Theme | None = None,
        alpha_enabled: bool = False,
        on_change: Callable[[str], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self._color = QColor(color)
        self._on_change = on_change
        self._alpha_enabled = alpha_enabled

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACE["sm"])

        self._label = QLabel(label)
        layout.addWidget(self._label)
        layout.addStretch(1)

        self._swatch = QLabel()
        self._swatch.setFixedSize(LAYOUT["control_height"], 20)
        self._swatch.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._swatch)

        self._value_label = QLabel()
        self._value_label.setObjectName("Secondary")
        layout.addWidget(self._value_label)

        self._button = QPushButton("选择…")
        self._button.setFixedHeight(26)
        self._button.clicked.connect(self._pick_color)
        layout.addWidget(self._button)

        self._refresh()

    def _refresh(self) -> None:
        """刷新色块与色值显示。"""
        self._swatch.setStyleSheet(
            f"background-color: {self._color.name()};"
            f" border: 1px solid rgba(128,128,128,0.4); border-radius: 4px;"
        )
        self._value_label.setText(self._color.name().upper())

    def _pick_color(self) -> None:
        """弹出系统取色器。"""
        options = QColorDialog.ColorDialogOption.ShowAlphaChannel if self._alpha_enabled else QColorDialog.ColorDialogOption(0)
        chosen = QColorDialog.getColor(
            self._color,
            self,
            f"选择{self._label.text()}",
            options,
        )
        if chosen.isValid():
            self.set_color(chosen.name())

    @property
    def color(self) -> str:
        """当前颜色（``#RRGGBB``）。"""
        return self._color.name()

    def set_color(self, color: str, notify: bool = True) -> None:
        """设置颜色。"""
        new_color = QColor(color)
        if not new_color.isValid():
            return
        self._color = new_color
        self._refresh()
        if notify:
            self.color_changed.emit(self._color.name())
            if self._on_change is not None:
                self._on_change(self._color.name())

    def set_theme(self, theme: Theme) -> None:
        """应用主题。"""
        self._theme = theme


class FontRow(QWidget):
    """字体选择行。

    只列出系统中真实存在的中文字体，避免用户选中缺字字体后
    译文显示为方框。列表第一项为"跟随系统默认"。
    """

    font_changed = pyqtSignal(str)

    #: 中文界面可用的字体白名单关键字。
    #: 不直接罗列全部系统字体——其中绝大多数不含中文字形。
    CJK_KEYWORDS = (
        "YaHei", "SimHei", "SimSun", "Microsoft YaHei", "PingFang",
        "Source Han", "Noto Sans CJK", "Noto Serif CJK", "Heiti", "Songti",
        "KaiTi", "FangSong", "HarmonyOS", "MiSans", "Alibaba",
    )

    def __init__(
        self,
        label: str,
        current: str = "",
        theme: Theme | None = None,
        on_change: Callable[[str], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self._on_change = on_change

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACE["sm"])

        self._label = QLabel(label)
        layout.addWidget(self._label)
        layout.addStretch(1)

        self._combo = QComboBox()
        self._combo.setFixedHeight(LAYOUT["control_height"])
        self._combo.setMinimumWidth(200)
        self._populate(current)
        self._combo.currentTextChanged.connect(self._handle_change)
        layout.addWidget(self._combo)

    def _populate(self, current: str) -> None:
        """填充字体列表。"""
        self._combo.addItem("跟随系统默认", "")

        try:
            families = sorted(set(QFontDatabase.families()))
        except Exception:  # noqa: BLE001 - 无 GUI 环境下无法枚举字体
            families = []

        cjk_families = [
            family
            for family in families
            if any(keyword.lower() in family.lower() for keyword in self.CJK_KEYWORDS)
        ]

        for family in cjk_families:
            self._combo.addItem(family, family)

        # 若当前配置的字体不在白名单内（用户手工编辑过配置），
        # 也一并加入，避免界面无法反映真实配置。
        if current and self._combo.findData(current) < 0:
            self._combo.addItem(f"{current}（自定义）", current)

        index = self._combo.findData(current)
        self._combo.setCurrentIndex(index if index >= 0 else 0)

    def _handle_change(self, _text: str) -> None:
        """字体变更。"""
        if self._on_change is not None:
            self._on_change(self.font_family)

    @property
    def font_family(self) -> str:
        """当前字体族；空字符串表示跟随系统默认。"""
        data = self._combo.currentData()
        return str(data) if data else ""

    def set_font_family(self, family: str) -> None:
        """设置字体（不触发回调）。"""
        index = self._combo.findData(family)
        if index < 0:
            self._combo.addItem(f"{family}（自定义）", family)
            index = self._combo.count() - 1
        blocked = self._combo.blockSignals(True)
        self._combo.setCurrentIndex(index)
        self._combo.blockSignals(blocked)

    def set_theme(self, theme: Theme) -> None:
        """应用主题。"""
        self._theme = theme
