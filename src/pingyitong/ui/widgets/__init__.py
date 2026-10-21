"""界面组件库。

全部组件由 :mod:`~pingyitong.ui.design.tokens` 的设计令牌驱动，
不含任何硬编码颜色或尺寸。

- :class:`~pingyitong.ui.widgets.button.Button` 按钮（按下缩放反馈）
- :class:`~pingyitong.ui.widgets.controls.Switch` 开关
- :class:`~pingyitong.ui.widgets.controls.SegmentedControl` 分段控件
- :class:`~pingyitong.ui.widgets.containers.Card` / ``SettingRow`` 容器
- :class:`~pingyitong.ui.widgets.toast.ToastManager` 通知系统
- :class:`~pingyitong.ui.widgets.tooltip.TooltipController` 工具提示
- :class:`~pingyitong.ui.widgets.slider_row.SliderRow` 数值滑块行
- :class:`~pingyitong.ui.widgets.language_combo.LanguageCombo` 可搜索的语言选择框
"""

from __future__ import annotations

from .button import Button, IconButton
from .containers import Card, Divider, SectionHeader, SettingRow, StatusPill
from .controls import SegmentedControl, Switch
from .language_combo import LanguageCombo, LanguageRow
from .slider_row import ColorRow, FontRow, SliderRow
from .toast import ToastKind, ToastManager, ToastRequest, ToastWidget
from .tooltip import TooltipController, TooltipMixin

__all__ = [
    "Button",
    "Card",
    "ColorRow",
    "Divider",
    "FontRow",
    "IconButton",
    "LanguageCombo",
    "LanguageRow",
    "SectionHeader",
    "SegmentedControl",
    "SettingRow",
    "SliderRow",
    "StatusPill",
    "Switch",
    "ToastKind",
    "ToastManager",
    "ToastRequest",
    "ToastWidget",
    "TooltipController",
    "TooltipMixin",
]
