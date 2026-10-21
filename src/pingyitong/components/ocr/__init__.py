"""文字识别组件。

- :class:`~pingyitong.components.ocr.ocr_engine.OcrEngine` 识别引擎
- :class:`~pingyitong.components.ocr.text_block.TextBlock` 文本块数据结构
"""

from __future__ import annotations

from .ocr_engine import (
    NullOcrBackend,
    OcrBackend,
    OcrEngine,
    RapidOcrBackend,
    create_ocr_engine,
    resolve_threads,
)
from .text_block import RecognizedFrame, TextBlock

__all__ = [
    "NullOcrBackend",
    "OcrBackend",
    "OcrEngine",
    "RapidOcrBackend",
    "RecognizedFrame",
    "TextBlock",
    "create_ocr_engine",
    "resolve_threads",
]
