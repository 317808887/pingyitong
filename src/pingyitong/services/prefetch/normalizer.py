"""归一化与分句（联网预翻译，自研）。

**OCR 通道与字幕行必须同分布**：字幕文件解析出的原文与屏幕 OCR 读到的
文字要能对上，前提是两边走**同一套归一化**。v1.1 R3 的教训是匹配失败
多半不是阈值问题，而是两边归一化不一致。

归一化分两级：

- ``normalize_for_key``——生成缓存键，较严格（可改变键稳定性的一律折叠）；
- ``normalize_for_match``——fuzzy 匹配用，更宽松（去标点/语气符号），
  只用于相似度计算，不用于键。
"""

from __future__ import annotations

import re
import unicodedata

#: OCR 常见形近字符映射（可配置语义由上层覆盖；此处为安全子集：
#: 只折叠"形状相同或极近"的字符，不做可能改变语义的替换）
_SHAPE_MAP = str.maketrans({
    "，": ",", "。": ".", "！": "!", "？": "?", "：": ":", "；": ";",
    "（": "(", "）": ")", "“": '"', "”": '"', "‘": "'", "’": "'",
})

#: 零宽与不可见字符
_INVISIBLE = re.compile(r"[\u200b\u200c\u200d\ufeff\u00ad]")

#: 多余空白
_WHITESPACE = re.compile(r"\s+")

#: 句末标点（中英）：分句依据
_SENTENCE_END = re.compile(
    r"(?<=[.!?。！？；;])\s*|(?<=[。！？])"
)

_strip_control = dict.fromkeys(range(32))
_WIDTH_MAP = str.maketrans({
    **{chr(code): chr(code - 0xFEE0) for code in range(0xFF01, 0xFF5F)},
    "\u3000": " ",
})
_NUMBERS = re.compile(r"(?<![A-Za-z0-9])\d+(?:[.,:/-]\d+)*(?![A-Za-z0-9])")
_NEGATION = re.compile(
    r"\b(?:no|not|never|without|cannot|dont|doesnt|didnt|isnt|arent|wasnt|werent|"
    r"cant|couldnt|wont|wouldnt|shouldnt|mustnt|hasnt|havent|hadnt|neednt)\b|"
    r"[不未无没勿]|禁止"
)


def normalize_for_key(text: str) -> str:
    """生成缓存键用的归一化文本。

    NFC 归一化、全角 ASCII 转半角、去零宽字符、去控制字符、空白折叠。
    保留数字与大小写，避免 10/l0、US/us 等不同语义复用译文。
    """
    if not text:
        return ""
    cleaned = _INVISIBLE.sub("", text)
    # 兼容归一化 NFKC 会把 X² 折叠成 X2，改变数学与代码含义。
    cleaned = unicodedata.normalize("NFC", cleaned).translate(_WIDTH_MAP)
    cleaned = _WHITESPACE.sub(" ", cleaned).translate(_strip_control).strip()
    return cleaned


def normalize_for_match(text: str) -> str:
    """fuzzy 相似度用的归一化：比 key 更宽松——额外去标点。

    OCR 对标点的识别最不稳定（引号、省略号、破折号形态繁多），
    相似度计算去掉标点可以显著降低这类噪声的权重。
    """
    cleaned = normalize_for_key(text).translate(_SHAPE_MAP).lower()
    # 数字间的小数点/日期分隔符必须保留，否则 1.5 与 15 会被当成同一数值。
    return re.sub(
        r"[^\w\s\u4e00-\u9fff]",
        lambda match: match.group() if (
            match.group() in ".,:/-"
            and match.start() > 0 and match.end() < len(cleaned)
            and cleaned[match.start() - 1].isdigit() and cleaned[match.end()].isdigit()
        ) else "",
        cleaned,
    )


def fuzzy_semantics_match(observed: str, cached: str) -> bool:
    """OCR 容错不能复用不同数值或相反肯否语义的字幕译文。"""
    observed_norm = normalize_for_match(observed)
    cached_norm = normalize_for_match(cached)
    return (
        min(len(observed_norm), len(cached_norm)) >= 0.65 * max(len(observed_norm), len(cached_norm))
        and _NUMBERS.findall(observed_norm) == _NUMBERS.findall(cached_norm)
        and bool(_NEGATION.search(observed_norm)) == bool(_NEGATION.search(cached_norm))
    )


def split_sentences(text: str, lang_hint: str = "") -> list[str]:
    """按句切分（中英标点规则）。

    字幕行**不做**二次分句（以字幕行为单位保时间轴完整，见 v1.1 §3.2），
    本函数服务于网页正文等长文本的句级切分。
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return []
    pieces = re.split(r"(?<=[.!?。！？])\s*", cleaned)
    return [piece.strip() for piece in pieces if piece.strip()]
