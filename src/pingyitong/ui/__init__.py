"""UI 交互层。

对应《项目全栈开发文档》一·（五），由四个部分组成：

- :class:`~pingyitong.ui.tray_icon.TrayIcon` 系统托盘图标
- :class:`~pingyitong.ui.main_panel.MainPanel` 主设置面板
- :class:`~pingyitong.ui.floating_bar.FloatingBar` 悬浮控制条
- :class:`~pingyitong.components.render.OverlayWindow` 原位翻译透明层

辅助界面：

- :class:`~pingyitong.ui.region_selector.RegionSelector` 自定义翻译区域框选
- :class:`~pingyitong.ui.history_dialog.HistoryDialog` 翻译历史浏览

设计系统位于 :mod:`pingyitong.ui.design` 与 :mod:`pingyitong.ui.widgets`。
"""

from __future__ import annotations

from .app import Application, run

__all__ = ["Application", "run"]
