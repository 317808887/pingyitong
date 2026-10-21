"""业务功能层。

对应《项目全栈开发文档》一·（四），实现：

- :class:`~pingyitong.services.translation_service.TranslationService` 实时翻译编排
- :class:`~pingyitong.services.text_preprocess.TextPreprocessor` OCR 后处理（自研算法）
- :class:`~pingyitong.services.glossary_service.Glossary` 专业术语库
- :class:`~pingyitong.services.history_service.HistoryService` 翻译历史回溯
- :class:`~pingyitong.services.hotkey_service.HotkeyService` 全局热键管理
- :mod:`~pingyitong.services.language` 源语言自动检测
"""

from __future__ import annotations

from .glossary_service import Glossary, GlossaryEntry, create_glossary
from .history_service import HistoryRecord, HistoryService
from .hotkey_service import (
    ACTION_COPY_RESULT,
    ACTION_LABELS,
    ACTION_SWITCH_MODEL,
    ACTION_TOGGLE_OVERLAY,
    ACTION_TOGGLE_TRANSLATION,
    HotkeyBinding,
    HotkeyService,
    default_bindings,
    format_sequence,
    parse_sequence,
)
from .language import LanguageGuess, detect_language, is_probably_translatable
from .text_preprocess import Paragraph, PreprocessStats, TextPreprocessor
from .translation_service import TranslationService, TranslationStats

__all__ = [
    "ACTION_COPY_RESULT",
    "ACTION_LABELS",
    "ACTION_SWITCH_MODEL",
    "ACTION_TOGGLE_OVERLAY",
    "ACTION_TOGGLE_TRANSLATION",
    "Glossary",
    "GlossaryEntry",
    "HistoryRecord",
    "HistoryService",
    "HotkeyBinding",
    "HotkeyService",
    "LanguageGuess",
    "Paragraph",
    "PreprocessStats",
    "TextPreprocessor",
    "TranslationService",
    "TranslationStats",
    "create_glossary",
    "default_bindings",
    "detect_language",
    "format_sequence",
    "is_probably_translatable",
    "parse_sequence",
]
