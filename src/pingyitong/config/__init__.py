"""配置持久化层。

对外暴露 :class:`SettingsManager` 与各配置节的数据模型。
"""

from __future__ import annotations

from .schema import (
    AppConfig,
    CaptureConfig,
    HistoryConfig,
    HotkeyConfig,
    ModelConfig,
    OcrConfig,
    PreprocessConfig,
    ScheduleConfig,
    SingleModelConfig,
    StyleConfig,
    TranslationConfig,
    UiConfig,
    from_dict,
    merge_into,
    to_dict,
)
from .settings import SettingsManager

__all__ = [
    "AppConfig",
    "CaptureConfig",
    "HistoryConfig",
    "HotkeyConfig",
    "ModelConfig",
    "OcrConfig",
    "PreprocessConfig",
    "ScheduleConfig",
    "SettingsManager",
    "SingleModelConfig",
    "StyleConfig",
    "TranslationConfig",
    "UiConfig",
    "from_dict",
    "merge_into",
    "to_dict",
]
