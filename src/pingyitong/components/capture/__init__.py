"""屏幕捕获模块。

职责：按可配置帧率抓取桌面画面，通过画面差分仅保留变化区域，
降低后续 OCR 与推理的负载。

开源组件：``python-mss``（BoboTiG/python-mss，MIT License）
自研内容：画面差分触发策略、区域裁剪、帧率动态控制
"""

from __future__ import annotations

from .frame_diff import DiffResult, FrameDiffer
from .geometry import (
    Rect,
    ScreenInfo,
    list_screens,
    logical_to_physical,
    physical_to_logical,
    region_to_monitor,
    virtual_desktop_rect,
)
from .screen_capture import CaptureFrame, ScreenCapturer

__all__ = [
    "CaptureFrame",
    "DiffResult",
    "FrameDiffer",
    "Rect",
    "ScreenCapturer",
    "ScreenInfo",
    "list_screens",
    "logical_to_physical",
    "physical_to_logical",
    "region_to_monitor",
    "virtual_desktop_rect",
]
