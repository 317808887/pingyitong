"""主题：由设计令牌生成 Qt 样式表。

界面外观完全由 :mod:`~pingyitong.ui.design.tokens` 驱动，
切换主题只需替换 :class:`~pingyitong.ui.design.tokens.ColorScheme`，
不存在第二处硬编码颜色。

为什么用样式表而不是逐控件设置属性
    Qt 提供两种换肤方式：逐个 ``setStyleSheet`` 或统一在
    ``QApplication`` 上设置一份样式表。后者天然保证了"一致性"——
    所有控件共享同一套取值，不会出现某个按钮忘了改颜色而与其他控件
    不一致的情况。这正对应设计规范中"聚合的、不被察觉的正确性"。
"""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtGui import QColor, QFont, QFontDatabase

from ...utils.logger import get_logger
from . import tokens
from .tokens import DARK_SCHEME, LIGHT_SCHEME, ColorScheme

logger = get_logger(__name__)


@dataclass
class Theme:
    """一套完整主题：配色 + 字体族。"""

    name: str
    scheme: ColorScheme
    #: 界面字体族（已解析出系统中实际可用的那一项）
    font_family: str = ""
    #: 等宽字体族
    mono_family: str = ""
    #: 基础字号
    base_font_size: int = tokens.FONT_SIZE["body"]

    @property
    def is_dark(self) -> bool:
        """是否深色主题。"""
        return self.name == "dark"

    def color(self, token_name: str) -> QColor:
        """按令牌名取颜色对象。"""
        value = getattr(self.scheme, token_name, None)
        if value is None:
            logger.debug("未知颜色令牌：%s", token_name)
            return QColor("#FF00FF")  # 品红：明显暴露拼写错误
        return QColor(value)


def resolve_font_family(candidates: tuple[str, ...]) -> str:
    """返回系统中第一个可用的字体族。

    中文界面最忌讳因缺字而回退到点阵字体，因此显式按优先级探测，
    而不是把整个候选列表塞给 Qt（Qt 的字体回退行为在不同版本间不一致）。
    """
    try:
        available = set(QFontDatabase.families())
    except Exception:  # noqa: BLE001 - 无 GUI 环境下无法查询字体
        return candidates[0] if candidates else ""

    for family in candidates:
        if family in available:
            return family
    return candidates[-1] if candidates else ""


def build_theme(name: str = "dark") -> Theme:
    """构造主题对象。

    :param name: ``"dark"`` / ``"light"``。
    """
    scheme = LIGHT_SCHEME if name == "light" else DARK_SCHEME
    return Theme(
        name=name,
        scheme=scheme,
        font_family=resolve_font_family(tokens.FONT_FAMILIES),
        mono_family=resolve_font_family(tokens.MONO_FONT_FAMILIES),
    )


def base_font(theme: Theme, size_key: str = "body", weight: str = "regular") -> QFont:
    """构造界面基础字体。"""
    font = QFont(theme.font_family)
    font.setPixelSize(tokens.FONT_SIZE.get(size_key, tokens.FONT_SIZE["body"]))
    font.setWeight(_qt_weight(tokens.FONT_WEIGHT.get(weight, 400)))
    return font


def _qt_weight(weight: int) -> QFont.Weight:
    """把数值字重映射为 Qt 的 ``QFont.Weight`` 枚举。"""
    mapping = {
        400: QFont.Weight.Normal,
        500: QFont.Weight.Medium,
        600: QFont.Weight.DemiBold,
        700: QFont.Weight.Bold,
    }
    return mapping.get(weight, QFont.Weight.Normal)


def stylesheet(theme: Theme) -> str:
    """生成全局样式表。

    :param theme: 主题对象。
    :return: 可直接传给 ``QApplication.setStyleSheet`` 的字符串。
    """
    c = theme.scheme
    radius_md = tokens.RADIUS["md"]

    # 实时译文文本框：表面色 + 35% 不透明度（用户指定）。
    # 面板悬在桌面内容之上，过实的底色会遮住大片屏幕；
    # 但完全不透明度太低会影响文字可读性，35% 是用户选择的平衡点。
    _panel = QColor(c.surface_raised)
    transcript_bg = (
        f"rgba({_panel.red()}, {_panel.green()}, {_panel.blue()}, "
        f"{round(0.35 * 255)})"
    )
    radius_sm = tokens.RADIUS["sm"]
    control_h = tokens.LAYOUT["control_height"]

    return f"""
/* ===================== 基础 ===================== */

QWidget {{
    background-color: transparent;
    color: {c.text_primary};
    font-family: "{theme.font_family}";
    font-size: {theme.base_font_size}px;
}}

QMainWindow, QDialog {{
    background-color: {c.background};
}}

QWidget#Surface {{
    background-color: {c.surface};
    border: 1px solid {c.border};
    border-radius: {radius_md}px;
}}

QWidget#Page {{
    background-color: {c.background};
}}

/* ===================== 文字层级 ===================== */

QLabel {{
    background: transparent;
    color: {c.text_primary};
}}

QLabel#Title {{
    font-size: {tokens.FONT_SIZE["title"]}px;
    font-weight: 600;
}}

QLabel#Subtitle {{
    font-size: {tokens.FONT_SIZE["subtitle"]}px;
    font-weight: 600;
}}

QLabel#Secondary {{
    color: {c.text_secondary};
    font-size: {tokens.FONT_SIZE["small"]}px;
}}

QLabel#Muted {{
    color: {c.text_muted};
    font-size: {tokens.FONT_SIZE["caption"]}px;
}}

QLabel#Danger {{
    color: {c.danger};
    font-size: {tokens.FONT_SIZE["small"]}px;
}}

QLabel#Accent {{
    color: {c.accent_text};
    font-size: {tokens.FONT_SIZE["small"]}px;
}}

QLabel#SectionHeader {{
    color: {c.text_secondary};
    font-size: {tokens.FONT_SIZE["caption"]}px;
    font-weight: 600;
    padding: {tokens.SPACE["sm"]}px {tokens.SPACE["lg"]}px {tokens.SPACE["xs"]}px;
}}

/* ===================== 按钮 ===================== */

QPushButton {{
    background-color: {c.surface_raised};
    color: {c.text_primary};
    border: 1px solid {c.border};
    border-radius: {radius_sm}px;
    padding: 0 {tokens.SPACE["md"]}px;
    min-height: {control_h}px;
    font-size: {tokens.FONT_SIZE["body"]}px;
}}

QPushButton:hover {{
    background-color: {c.surface_hover};
}}

QPushButton:pressed {{
    background-color: {c.surface_active};
}}

QPushButton:disabled {{
    color: {c.text_muted};
    background-color: {c.surface};
}}

QPushButton:focus {{
    border: {tokens.BORDER_WIDTH["focus"]}px solid {c.border_focus};
}}

QPushButton#Primary {{
    background-color: {c.accent};
    color: {c.text_inverse};
    border: none;
    font-weight: 500;
}}

QPushButton#Primary:hover {{
    background-color: {c.accent_hover};
}}

QPushButton#Primary:pressed {{
    background-color: {c.accent_active};
}}

QPushButton#Primary:disabled {{
    background-color: {c.surface_active};
    color: {c.text_muted};
}}

QPushButton#Ghost {{
    background-color: transparent;
    border: none;
    color: {c.text_secondary};
}}

QPushButton#Ghost:hover {{
    background-color: {c.surface_hover};
    color: {c.text_primary};
}}

QPushButton#Danger {{
    background-color: transparent;
    border: 1px solid {c.danger};
    color: {c.danger};
}}

QPushButton#Danger:hover {{
    background-color: {c.danger_subtle};
}}

/* ===================== 输入控件 ===================== */

QLineEdit, QPlainTextEdit, QTextEdit {{
    background-color: {c.surface_raised};
    color: {c.text_primary};
    border: 1px solid {c.border};
    border-radius: {radius_sm}px;
    padding: 0 {tokens.SPACE["sm"]}px;
    min-height: {control_h}px;
    selection-background-color: {c.accent};
    selection-color: {c.text_inverse};
}}

QPlainTextEdit, QTextEdit {{
    padding: {tokens.SPACE["sm"]}px;
}}

QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus {{
    border: {tokens.BORDER_WIDTH["focus"]}px solid {c.border_focus};
}}

QLineEdit:disabled, QPlainTextEdit:disabled {{
    color: {c.text_muted};
    background-color: {c.surface};
}}

/* ===================== 实时译文文本框 ===================== */

#TranscriptPanel {{
    background-color: {transcript_bg};
    border: 1px solid {c.border};
    border-radius: {radius_md}px;
}}

#TranscriptText {{
    background: transparent;
    color: {c.text_primary};
    border: none;
}}

QComboBox {{
    background-color: {c.surface_raised};
    color: {c.text_primary};
    border: 1px solid {c.border};
    border-radius: {radius_sm}px;
    padding: 0 {tokens.SPACE["sm"]}px;
    min-height: {control_h}px;
}}

QComboBox:hover {{
    background-color: {c.surface_hover};
}}

QComboBox:focus {{
    border: {tokens.BORDER_WIDTH["focus"]}px solid {c.border_focus};
}}

QComboBox::drop-down {{
    border: none;
    width: 22px;
}}

QComboBox QAbstractItemView {{
    background-color: {c.surface_raised};
    color: {c.text_primary};
    border: 1px solid {c.border};
    border-radius: {radius_sm}px;
    selection-background-color: {c.accent_subtle};
    selection-color: {c.text_primary};
    outline: none;
    padding: {tokens.SPACE["xxs"]}px;
}}

/* ===================== 数值输入 ===================== */

QSpinBox, QDoubleSpinBox {{
    background-color: {c.surface_raised};
    color: {c.text_primary};
    border: 1px solid {c.border};
    border-radius: {radius_sm}px;
    padding: 0 {tokens.SPACE["xs"]}px;
    min-height: {control_h}px;
}}

QSpinBox:focus, QDoubleSpinBox:focus {{
    border: {tokens.BORDER_WIDTH["focus"]}px solid {c.border_focus};
}}

/* ===================== 滑块 ===================== */

QSlider::groove:horizontal {{
    height: 4px;
    background: {c.surface_active};
    border-radius: 2px;
}}

QSlider::sub-page:horizontal {{
    background: {c.accent};
    border-radius: 2px;
}}

QSlider::handle:horizontal {{
    background: {c.text_primary};
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}

QSlider::handle:horizontal:hover {{
    background: {c.accent_hover};
}}

QSlider::handle:horizontal:disabled {{
    background: {c.text_muted};
}}

/* ===================== 复选 / 单选 ===================== */

QCheckBox, QRadioButton {{
    spacing: {tokens.SPACE["sm"]}px;
    color: {c.text_primary};
}}

QCheckBox::indicator, QRadioButton::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {c.border};
    background-color: {c.surface_raised};
}}

QCheckBox::indicator {{
    border-radius: 4px;
}}

QRadioButton::indicator {{
    border-radius: 8px;
}}

QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
    border-color: {c.accent};
}}

QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background-color: {c.accent};
    border-color: {c.accent};
}}

QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{
    background-color: {c.surface};
    border-color: {c.border};
}}

/* ===================== 滚动条 ===================== */

QScrollArea {{
    background: transparent;
    border: none;
}}

QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 0;
}}

QScrollBar::handle:vertical {{
    background: {c.surface_active};
    border-radius: 5px;
    min-height: 32px;
}}

QScrollBar::handle:vertical:hover {{
    background: {c.text_muted};
}}

QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 0;
}}

QScrollBar::handle:horizontal {{
    background: {c.surface_active};
    border-radius: 5px;
    min-width: 32px;
}}

QScrollBar::add-line, QScrollBar::sub-line {{
    height: 0;
    width: 0;
}}

QScrollBar::add-page, QScrollBar::sub-page {{
    background: transparent;
}}

/* ===================== 标签页 ===================== */

QTabWidget::pane {{
    border: none;
    background: transparent;
}}

QTabBar {{
    background: transparent;
}}

QTabBar::tab {{
    background: transparent;
    color: {c.text_secondary};
    padding: {tokens.SPACE["sm"]}px {tokens.SPACE["md"]}px;
    border: none;
    border-bottom: 2px solid transparent;
    font-size: {tokens.FONT_SIZE["body"]}px;
}}

QTabBar::tab:hover {{
    color: {c.text_primary};
}}

QTabBar::tab:selected {{
    color: {c.text_primary};
    border-bottom: 2px solid {c.accent};
    font-weight: 500;
}}

/* ===================== 列表 / 表格 ===================== */

QListWidget, QTreeWidget, QTableWidget {{
    background-color: {c.surface};
    color: {c.text_primary};
    border: 1px solid {c.border};
    border-radius: {radius_sm}px;
    outline: none;
}}

QListWidget::item, QTreeWidget::item {{
    padding: {tokens.SPACE["sm"]}px;
    border-radius: {tokens.RADIUS["sm"]}px;
}}

QListWidget::item:hover, QTreeWidget::item:hover {{
    background-color: {c.surface_hover};
}}

QListWidget::item:selected, QTreeWidget::item:selected {{
    background-color: {c.accent_subtle};
    color: {c.text_primary};
}}

QHeaderView::section {{
    background-color: {c.surface_raised};
    color: {c.text_secondary};
    border: none;
    border-bottom: 1px solid {c.border};
    padding: {tokens.SPACE["xs"]}px {tokens.SPACE["sm"]}px;
    font-size: {tokens.FONT_SIZE["small"]}px;
}}

QTableWidget {{
    gridline-color: {c.border};
}}

QTableWidget::item {{
    padding: {tokens.SPACE["xs"]}px;
}}

/* ===================== 菜单 ===================== */

QMenu {{
    background-color: {c.surface_raised};
    color: {c.text_primary};
    border: 1px solid {c.border};
    border-radius: {radius_sm}px;
    padding: {tokens.SPACE["xs"]}px;
}}

QMenu::item {{
    padding: {tokens.SPACE["sm"]}px {tokens.SPACE["xl"]}px;
    border-radius: {tokens.RADIUS["sm"]}px;
}}

QMenu::item:selected {{
    background-color: {c.accent_subtle};
}}

QMenu::separator {{
    height: 1px;
    background: {c.border};
    margin: {tokens.SPACE["xs"]}px {tokens.SPACE["sm"]}px;
}}

/* ===================== 分组框 ===================== */

QGroupBox {{
    background-color: {c.surface};
    border: 1px solid {c.border};
    border-radius: {radius_md}px;
    margin-top: {tokens.SPACE["md"]}px;
    padding: {tokens.SPACE["lg"]}px;
    font-size: {tokens.FONT_SIZE["small"]}px;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: {tokens.SPACE["md"]}px;
    padding: 0 {tokens.SPACE["xs"]}px;
    color: {c.text_secondary};
    background-color: {c.background};
}}

/* ===================== 进度条 ===================== */

QProgressBar {{
    background-color: {c.surface_active};
    border: none;
    border-radius: 3px;
    height: 6px;
    text-align: center;
}}

QProgressBar::chunk {{
    background-color: {c.accent};
    border-radius: 3px;
}}

/* ===================== 提示框 ===================== */

QToolTip {{
    background-color: {c.surface_raised};
    color: {c.text_primary};
    border: 1px solid {c.border};
    border-radius: {tokens.RADIUS["sm"]}px;
    padding: {tokens.SPACE["xs"]}px {tokens.SPACE["sm"]}px;
    font-size: {tokens.FONT_SIZE["small"]}px;
}}

/* ===================== 分隔线 ===================== */

QFrame#Divider {{
    background-color: {c.border};
    max-height: 1px;
    border: none;
}}

QFrame#VDivider {{
    background-color: {c.border};
    max-width: 1px;
    border: none;
}}

/* ===================== 状态徽标 ===================== */

QLabel#Badge {{
    background-color: {c.accent_subtle};
    color: {c.accent_text};
    border-radius: {tokens.RADIUS["sm"]}px;
    padding: 1px {tokens.SPACE["xs"]}px;
    font-size: {tokens.FONT_SIZE["caption"]}px;
}}

QLabel#BadgeSuccess {{
    background-color: {c.accent_subtle};
    color: {c.success};
    border-radius: {tokens.RADIUS["sm"]}px;
    padding: 1px {tokens.SPACE["xs"]}px;
    font-size: {tokens.FONT_SIZE["caption"]}px;
}}

QLabel#BadgeWarning {{
    background-color: {c.accent_subtle};
    color: {c.warning};
    border-radius: {tokens.RADIUS["sm"]}px;
    padding: 1px {tokens.SPACE["xs"]}px;
    font-size: {tokens.FONT_SIZE["caption"]}px;
}}
"""
