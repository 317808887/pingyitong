"""图像处理组件：画面差分后的区域裁剪、OCR 前处理、背景明暗分析。"""

from __future__ import annotations

from .preprocess import (
    RegionAppearance,
    analyse_region,
    dominant_background_color,
    downscale_boxes,
    downscale_for_detection,
    enhance_for_ocr,
    is_available,
)

__all__ = [
    "RegionAppearance",
    "analyse_region",
    "dominant_background_color",
    "downscale_boxes",
    "downscale_for_detection",
    "enhance_for_ocr",
    "is_available",
]
