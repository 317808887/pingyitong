"""窗口渲染引擎（原位叠加渲染）。

- :class:`~pingyitong.components.render.overlay_window.OverlayWindow` 单屏透明层
- :class:`~pingyitong.components.render.overlay_window.OverlayManager` 多屏统一管理
- :mod:`~pingyitong.components.render.layout` 文本自适应排版
- :class:`~pingyitong.components.render.render_item.RenderItem` 渲染数据模型
"""

from __future__ import annotations

from .layout import TextLayout, build_font, elide_line, layout_text, wrap_text
from .overlay_window import OverlayManager, OverlayWindow
from .render_item import RenderFrame, RenderItem

__all__ = [
    "OverlayManager",
    "OverlayWindow",
    "RenderFrame",
    "RenderItem",
    "TextLayout",
    "build_font",
    "elide_line",
    "layout_text",
    "wrap_text",
]
