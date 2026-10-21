"""Offline recognizer selection; UI language choices are not an OCR coverage claim.

All paths resolve inside the shipped ``assets/ocr`` directory. Model absence is
an explicit error; this module never downloads or substitutes another alphabet.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path

from ...languages import LANGUAGES
from ...utils.paths import asset_path


@dataclass(frozen=True)
class RecognizerProfile:
    key: str
    filename: str
    rtl: bool = False

    def local_path(self) -> Path:
        path = asset_path("ocr", self.filename)
        if not path.is_file():
            raise FileNotFoundError(f"缺少离线 OCR 模型：{path.name}（assets/ocr）")
        return path


CHINESE = RecognizerProfile("ch", "ch_PP-OCRv5_rec_mobile.onnx")
PROFILES = {
    "han": CHINESE,
    "japanese": CHINESE,
    "hangul": RecognizerProfile("korean", "korean_PP-OCRv5_rec_mobile.onnx"),
    "cyrillic": RecognizerProfile("cyrillic", "cyrillic_PP-OCRv5_rec_mobile.onnx"),
    "arabic": RecognizerProfile("arabic", "arabic_PP-OCRv5_rec_mobile.onnx", rtl=True),
    "latin": RecognizerProfile("latin", "latin_PP-OCRv5_rec_mobile.onnx"),
    "devanagari": RecognizerProfile("devanagari", "devanagari_PP-OCRv5_rec_mobile.onnx"),
    "greek": RecognizerProfile("greek", "el_PP-OCRv5_rec_mobile.onnx"),
    "thai": RecognizerProfile("thai", "th_PP-OCRv5_rec_mobile.onnx"),
    "tamil": RecognizerProfile("tamil", "ta_PP-OCRv5_rec_mobile.onnx"),
    "telugu": RecognizerProfile("telugu", "te_PP-OCRv5_rec_mobile.onnx"),
    # Paddle's filename 'ka' denotes Kannada, not ISO 639-1 Georgian 'ka'.
    "kannada": RecognizerProfile("kannada", "ka_PP-OCRv4_rec_mobile.onnx"),
}


def profile_for_source(source_language: str) -> RecognizerProfile:
    """Auto uses one Chinese/English/Japanese model; it is not a 142-way detector."""
    if source_language == "auto":
        return CHINESE
    language = LANGUAGES.get(source_language)
    if language is None:
        raise ValueError(f"无法识别源语言：{source_language}")
    profile = PROFILES.get(language.script)
    if profile is None:
        raise ValueError(f"尚无离线 OCR 模型：{language.name}（{language.script}）")
    return profile


def source_profile_key(source_language: str) -> str:
    try:
        return profile_for_source(source_language).key
    except ValueError:
        return f"unsupported:{source_language}"


def reverse_rtl_prediction(text: str) -> str:
    """PaddleOCR's ``pred_reverse`` token rule, preserving ASCII number runs.

Applied to each complete visual line before paragraph assembly. Reversing an
already assembled paragraph would reverse its line order as well.
See ppocr/postprocess/rec_postprocess.py (Apache-2.0).
    Adapted from PaddleOCR, Copyright (c) PaddlePaddle Authors.
    Modified on 2026-10-06 to preserve Unicode EN/AN numeric runs and separators.
    Full license and attribution are shipped in assets/ocr/LICENSES.
"""
    # Preserve the upstream ASCII token rule, extending its numeric runs to
    # Unicode bidi EN/AN. Arabic-Indic and Eastern Arabic-Indic digits read
    # left-to-right inside Arabic; reversing individual digits corrupts values.
    ascii_run = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 :*./%+-")
    numeric_separators = frozenset("\u066a\u066b\u066c,")
    tokens: list[str] = []
    run: list[str] = []
    for index, char in enumerate(text):
        numeric = unicodedata.bidirectional(char) in {"EN", "AN"}
        numeric_separator = (
            char in numeric_separators or unicodedata.bidirectional(char) in {"CS", "ES", "ET"}
        ) and (
            (index > 0 and unicodedata.bidirectional(text[index - 1]) in {"EN", "AN"})
            or (index + 1 < len(text) and unicodedata.bidirectional(text[index + 1]) in {"EN", "AN"})
        )
        if char in ascii_run or numeric or numeric_separator:
            run.append(char)
        else:
            if run:
                tokens.append("".join(run))
                run.clear()
            tokens.append(char)
    if run:
        tokens.append("".join(run))
    return "".join(reversed(tokens))
