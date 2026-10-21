"""源语言自动检测。

对应产品功能规划二·（四）："软件自动检测屏幕文字的源语言，用户仅需选择目标语言"。

实现方式为**字符脚本统计**而非神经网络模型，原因有三：

1. 零依赖、零延迟——每帧都要调用，任何额外模型都会拖慢流水线；
2. 屏幕场景的语种集合很小（8 种），且不同语种使用的字符集区分度极高
   （汉字、假名、谚文、西里尔字母互不重叠）；
3. 无需训练数据，也无需联网，符合全离线约束。

对拉丁字母语言（英/法/德/西）无法仅靠字符集区分，因此进一步使用
高频停用词与特征字符（如德语 ß、西班牙语 ñ、法语 ç）做加权判定。
置信度不足时交由调用方回退到用户配置的兜底语言。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from ..constants import AUTO_LANGUAGE
from ..languages import LANGUAGES

# Only source tokens with explicit technical syntax are exempt from output
# language checks. Ordinary words such as "close the window" are never masked.
# ASCII identifier boundaries deliberately permit adjacent CJK prose. Unicode
# \b treats both scripts as word characters and misses "检查closeEvent和app.py".
_TECHNICAL_TOKEN = re.compile(
    r"\[TERM_\d+\]|https?://[^\s<>，。！？]+|www\.[^\s<>，。！？]+|"
    r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|"
    r"[A-Za-z]:[\\/][^\s<>，。！？]+|(?:\.{1,2}/|/)[\w.-]+(?:/[\w.-]+)+|"
    r"(?<![A-Za-z0-9_])[\w.-]+\.(?:py|js|ts|json|yaml|yml|toml|md|txt|log|exe|dll|bin|"
    r"cpp|hpp|java|go|rs|html|css|xml|csv|ini|bat|ps1|sh)(?![A-Za-z0-9_])|"
    r"(?<![A-Za-z0-9_])(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}(?:/[^\s<>，。！？]*)?|"
    r"(?<![A-Za-z0-9_])--?[A-Za-z][\w-]*|"
    r"(?<![A-Za-z0-9_])(?:[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+|_[A-Za-z][A-Za-z0-9_]*)(?![A-Za-z0-9_])|"
    r"(?<![A-Za-z0-9_])[a-z]+[A-Z][A-Za-z0-9]*(?![A-Za-z0-9_])|"
    r"(?<![A-Za-z0-9_])[A-Z][a-z]+(?:[A-Z][A-Za-z0-9]*)+(?![A-Za-z0-9_])|"
    r"(?<![A-Za-z0-9_])[A-Za-z]+\d+[A-Za-z0-9]*(?![A-Za-z0-9_])|"
    r"(?<![A-Za-z0-9_])[0-9a-fA-F]{8,}(?![A-Za-z0-9_])|"
    r"(?<![A-Za-z0-9_])\d+(?:\.\d+)*(?:ms|fps|Hz|B|KB|MB|GB|TB|px|%)?(?![A-Za-z0-9_])"
)
_KNOWN_BRANDS = frozenset({
    "Windows", "Linux", "Chrome", "Firefox", "Python", "Java", "Codex", "Spark",
    "Microsoft", "Google", "Apple",
})
_TECHNICAL_ACRONYMS = frozenset({
    "API", "CPU", "GPU", "OCR", "AI", "UI", "UX", "RAM", "ROM", "URL",
    "HTTP", "HTTPS", "TCP", "UDP", "DNS", "JSON", "HTML", "CSS", "CSV", "PDF",
    "USB", "SSD", "HDD", "FPS", "SSH", "SSL", "TLS", "IP", "RGB", "DPI", "QR",
    "SQL", "DLL", "EXE", "ID", "GB", "MB", "KB", "TB", "Hz", "ms", "fps", "px",
})
_LATIN_WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+")
_TOKEN_TRAILING_PUNCTUATION = ".,:;?!)]}\"'，。！？；：）】》」』"

#: 源文里的单位符号与孤立大写字母。译文原样保留它们是正确的翻译结果
#: （"65 C"、"5 W"、"Level B"），把它们算成"未翻译的外来散文"会让整段
#: 正确译文被守卫误杀、退回英文原文（实测 t10：`65 C` 剥掉数字后只剩
#: 单位 `C`，has_foreign_prose=True）。
#: 只从**源文**收集：译文里凭空多出的拉丁词元永远不豁免，因此模型
#: 整段输出英文的场景（每个词都不在源文豁免集里）仍会被照常拦截。
_UNIT_TOKEN = re.compile(
    r"°[A-Za-z]{1,2}"
    r"|\b(?:kHz|MHz|GHz|Hz|kWh|rpm|dB|kg|mg|mm|cm|km|ml|lb|oz|ft|ms|fps|px)\b"
    r"|(?<![A-Za-z0-9_])[A-Z](?![A-Za-z0-9_])"
)

#: 标记与模板语法：HTML/XML 标签（含全部属性）、``{{ }}`` / ``{% %}`` /
#: ``<% %>`` 模板表达式。守卫只对**可见文本**判定——标签属性
#: （``action=``、``placeholder=``）与模板表达式（``{{ user.name }}``）
#: 本就要求原样保留，其英文不是"未翻译的外来散文"（实测 t12/t14：
#: 可见文本已正确译出，仍因属性词被整段判为外来散文而退回原文）。
_MARKUP = re.compile(r"\{\{.*?\}\}|\{%.*?%\}|<%.*?%>|</?[A-Za-z][^<>]*>")


def _visible_text(text: str) -> str:
    """剥离标记与模板语法，只留下用户能看见的文本。"""
    return _MARKUP.sub(" ", text)


_ZH_COMPLETION_AFTER = r"(?:完成|结束|結束|完毕|完畢)(?:之后|之後|以后|以後|后|後)"
_ZH_AFTER_PREFIX = r"[\s，,]*(?:请|請|仍然|依然|仍|再|先|务必|務必)?"
_JA_UNTIL_FORBIDDEN_ACTIONS = (
    (r"削除しない", r"(?:删除|刪除|移除)"),
    (r"閉じない", r"(?:关闭|關閉|关掉|關掉)"),
    (r"(?:電源を)?切らない", r"(?:关机|關機|关闭电源|關閉電源|断电|斷電)"),
)
_EN_BEFORE_FORBIDDEN_ACTIONS = {
    "close": r"(?:关闭|關閉|关掉|關掉)",
    "delete": r"(?:删除|刪除|移除)",
    "disconnect": r"(?:断开|斷開|切断|切斷)",
    "stop": r"(?:停止|停下|终止|終止)",
}


def has_reversed_until_condition(source: str, translated: str, target_language: str) -> bool:
    """Find explicit, high-confidence time-condition reversals in Chinese output.

    This is deliberately not a general semantic evaluator. Require a single
    source sentence and a known completion condition, so unrelated later
    prohibitions and other language pairs are not rejected on a keyword alone.
    """
    if target_language not in {"zh", "zh-TW"}:
        return False
    clauses = [part.strip() for part in re.split(r"[。！？\n]|(?<=[.!?])\s+", source) if part.strip()]
    if len(clauses) != 1:
        return False
    english_before = re.search(
        r"\b(?:do\s+not|don't|never)\s+"
        r"(?P<action>close|delete|disconnect|stop)\b"
        r"[^.!?\n]*\bbefore\b[^.!?\n]*"
        r"\b(?:finishes?|finished|completes?|completed|ends?|ended)\b",
        source, re.IGNORECASE,
    )
    if english_before:
        action = _EN_BEFORE_FORBIDDEN_ACTIONS[english_before.group("action").lower()]
        if re.search(
            _ZH_COMPLETION_AFTER + _ZH_AFTER_PREFIX
            + r"(?:不要|请勿|請勿|勿|不得|不能|不可|不应|不應|别|別|禁止|不)"
            + r"[^。！？.!?；;，,\n]{0,12}?" + action,
            translated,
        ):
            return True
    until = re.search(r"(?:完了する|終わる|済む)まで", source)
    if until and not re.search(r"(?:後|あと)", source[until.end():]):
        remainder = source[until.end():]
        for japanese, chinese in _JA_UNTIL_FORBIDDEN_ACTIONS:
            if re.search(japanese, remainder) and re.search(
                _ZH_COMPLETION_AFTER + _ZH_AFTER_PREFIX
                + r"(?:不要|请勿|請勿|勿|不得|不能|不可|不应|不應|别|別|禁止|不)"
                + r"[^。！？.!?；;，,\n]{0,12}?" + chinese,
                translated,
            ):
                return True
    # A positive instruction to maintain a state UNTIL the event ends must
    # not become an instruction to start maintaining it AFTER the event ends.
    if re.match(r"\s*(?:keep|remain|continue)\b", source, re.IGNORECASE) and re.search(
        r"\buntil\b[^.!?\n]*\b(?:ends?|finishes?|completes?|completed|done|over)\b",
        source, re.IGNORECASE,
    ) and not re.search(r"\b(?:after|then|once)\b", source, re.IGNORECASE):
        return bool(re.search(
            _ZH_COMPLETION_AFTER + _ZH_AFTER_PREFIX + r"(?:保持|维持|維持|继续|繼續)",
            translated,
        ))
    return False


def _without_source_technical_tokens(
    text: str, source: str, protected_terms: tuple[str, ...] = (),
) -> str:
    # OCR / prose includes sentence punctuation immediately after a URL or
    # path. It is not part of the protected identifier: the translated sentence
    # may move that punctuation after a Korean particle instead.
    # A Unicode filename may include Chinese characters. Keep that original
    # match, but also find ASCII names after surrounding Chinese prose:
    # "检查app.py" protects app.py even if the translation says "运行app.py".
    technical_tokens = _TECHNICAL_TOKEN.findall(source)
    technical_tokens.extend(_TECHNICAL_TOKEN.findall(re.sub(r"[\u3400-\u9fff]", " ", source)))
    tokens = {
        token.rstrip(_TOKEN_TRAILING_PUNCTUATION)
        for token in technical_tokens
    }
    tokens.discard("")
    # Supplied only for glossary entries actually matched in this source.
    # These explicit terminology exceptions do not exempt arbitrary prose.
    tokens.update(term for term in protected_terms if term)
    tokens.update(
        word for word in _LATIN_WORD.findall(source)
        if word in _KNOWN_BRANDS or word in _TECHNICAL_ACRONYMS
    )
    tokens.update(unit for unit in _UNIT_TOKEN.findall(source) if unit)
    for token in sorted(tokens, key=len, reverse=True):
        # Unicode \w includes CJK letters: ASCII identifier boundaries must
        # still permit a preserved token next to Korean particles / Chinese.
        text = re.sub(
            r"(?<![A-Za-z0-9_])" + re.escape(token) + r"(?![A-Za-z0-9_])",
            " ", text,
        )
    return text


def has_foreign_prose(
    text: str, target_language: str, source: str = "",
    protected_terms: tuple[str, ...] = (),
) -> bool:
    """Find foreign prose after removing unchanged technical source tokens.

    This checks every script rather than only the most common script, so a
    Korean sentence followed by Chinese instructions is still rejected.
    Script checks cannot distinguish two languages sharing Latin / Cyrillic
    writing; their names remain explicit in the translation prompt.
    """
    info = LANGUAGES.get(target_language)
    if info is None:
        return False
    expected = info.script
    allowed = {expected}
    if expected == "japanese":
        allowed = {"han", "kana"}
    if expected in {"cree", "inuktitut"}:
        allowed = {"canadian"}
    # Only visible text carries the language obligation: tags, attributes and
    # template expressions must stay verbatim, so their words are not prose.
    residual = _without_source_technical_tokens(
        _visible_text(text), source, protected_terms
    )
    counts: dict[str, int] = {}
    for ch in residual:
        # Common orthographic apostrophes used by Uzbek, Guarani and Navajo
        # are Unicode modifier letters, not evidence of a foreign alphabet.
        # Keep this neutral exception narrow; unknown scripts still fail.
        if ch in {"\u02bc", "\u02bb"}:
            continue
        if ch.isalpha():
            script = _script_of(ch)
            counts[script] = counts.get(script, 0) + 1
    if any(count for script, count in counts.items() if script not in allowed and script != "latin"):
        return True
    if "latin" in allowed:
        return (
            target_language not in {"en", "sco"}
            and _is_obvious_english_prose(residual)
        )
    if not counts.get("latin"):
        return False
    target_letters = sum(counts.get(script, 0) for script in allowed)
    if not target_letters:
        return True
    # Brands / identifiers have already been exempted from the original.
    # A lone untranslated UI word such as "close" is foreign prose too.
    residual_words = _LATIN_WORD.findall(residual)
    if len(residual_words) == 1 and _preserved_source_word(residual_words[0], source):
        # 孤立拉丁词元若与源文词一致（允许单复数等词形变化），说明模型
        # 保留了源文术语（如 `65 C` 的单位、"8 workers" 的 worker），
        # 译文主体已是目标语言——这不是"整段外来散文"。多个残留词或
        # 源文里没有的词（如中文源文里混进 "close"）仍照常拦截。
        return False
    return bool(residual_words)


def _stem(word: str) -> str:
    """极简词形归一：只折叠复数与 -ing/-ed 后缀，不做完整词干还原。

    用于判断"译文里的孤立拉丁词元是否就是源文词"。规则越简单越好：
    这里只需要覆盖模型保留源文名词时最常见的单复数差异
    （``workers`` ↔ ``worker``），不追求语言学上的完备。
    """
    folded = word.casefold()
    for suffix in ("ing", "ed", "es", "s"):
        if folded.endswith(suffix) and len(folded) - len(suffix) >= 3:
            return folded[: -len(suffix)]
    return folded


def _preserved_source_word(word: str, source: str) -> bool:
    """孤立拉丁词元是否是源文里出现过的词（允许词形变化）。"""
    stem = _stem(word)
    if not stem:
        return False
    return any(_stem(found) == stem for found in _LATIN_WORD.findall(source))


def _is_obvious_english_prose(text: str) -> bool:
    """Conservatively find long English prose returned for another Latin target.

    Short UI labels, proper names, close-language fragments and uncertain
    guesses remain inconclusive. Technical identifiers and explicit glossary
    exceptions are already removed by ``has_foreign_prose``. This is an output
    language guard, not an assertion of translation accuracy.
    """
    if sum(char.isalpha() for char in text) < 50:
        return False
    words = re.findall(r"[A-Za-z]+", text.lower())
    if len(words) < 8:
        return False
    hits = [word for word in words if word in _STOPWORDS["en"]]
    distinct = set(hits)
    if (
        len(hits) < 5
        or len(distinct) < 3
        or "and" not in distinct
        or not {"is", "are"}.intersection(distinct)
        or not {"in", "of", "to"}.intersection(distinct)
        or len(hits) / len(words) < 0.25
    ):
        return False
    guess = detect_language(text, "en")
    return guess.code == "en" and guess.confidence >= 0.95

#: 各语种的高频停用词。用于区分同为拉丁字母的语言。
_STOPWORDS: dict[str, set[str]] = {
    "en": {
        "the", "and", "of", "to", "in", "is", "you", "that", "it", "for",
        "on", "with", "as", "are", "this", "be", "at", "or", "by", "not",
    },
    "fr": {
        "le", "la", "les", "de", "et", "un", "une", "des", "du", "est",
        "que", "qui", "pour", "dans", "sur", "avec", "pas", "vous", "nous",
    },
    "de": {
        "der", "die", "das", "und", "ist", "von", "den", "mit", "auf", "für",
        "nicht", "ein", "eine", "dem", "des", "sich", "im", "zu", "als",
    },
    "es": {
        "el", "la", "los", "las", "de", "que", "y", "en", "un", "una",
        "es", "por", "con", "para", "no", "se", "su", "al", "como", "más",
    },
}

#: 各语种的特征字符（重音符号等），命中加分。
_DISTINCTIVE_CHARS: dict[str, set[str]] = {
    "de": set("ßäöüÄÖÜ"),
    "es": set("ñÑ¿¡"),
    "fr": set("çÇœŒàèùâêîôûëïü"),
    "ru": set("ёЁыЫэЭъЪ"),
}


@dataclass
class LanguageGuess:
    """语言检测结果。"""

    #: 推测的语言代码，无法判断时为 ``auto``
    code: str
    #: 置信度，0-1
    confidence: float
    #: 判定依据（脚本名或规则名），用于日志与调试
    basis: str = ""

    @property
    def is_confident(self) -> bool:
        """是否有足够把握。"""
        return self.code != AUTO_LANGUAGE and self.confidence >= 0.5


def detect_language(text: str, fallback: str = "en") -> LanguageGuess:
    """检测文本的源语言。

    :param text: 待检测文本。
    :param fallback: 无法判断时返回的兜底语言代码。
    :return: :class:`LanguageGuess`。
    """
    stripped = (text or "").strip()
    if not stripped:
        return LanguageGuess(code=fallback, confidence=0.0, basis="空文本")

    letters = [ch for ch in stripped if ch.isalpha()]
    if not letters:
        return LanguageGuess(code=fallback, confidence=0.0, basis="无字母字符")

    total = len(letters)

    # 第一层：按 Unicode 脚本统计。假名、谚文、汉字、西里尔字母的区分度是决定性的。
    counts = {"han": 0, "kana": 0, "hangul": 0, "cyrillic": 0, "latin": 0}
    for ch in letters:
        name = _script_of(ch)
        if name in counts:
            counts[name] += 1

    han_ratio = counts["han"] / total
    kana_ratio = counts["kana"] / total
    hangul_ratio = counts["hangul"] / total
    cyrillic_ratio = counts["cyrillic"] / total
    latin_ratio = counts["latin"] / total

    # 日语包含汉字 + 假名；只要出现假名即可判定为日语，
    # 因为中文文本中不会出现平假名/片假名。
    if kana_ratio > 0.05:
        return LanguageGuess(code="ja", confidence=min(1.0, 0.6 + kana_ratio), basis="假名")
    if hangul_ratio > 0.10:
        return LanguageGuess(code="ko", confidence=min(1.0, 0.6 + hangul_ratio), basis="谚文")
    if cyrillic_ratio > 0.30:
        return LanguageGuess(code="ru", confidence=min(1.0, 0.6 + cyrillic_ratio), basis="西里尔字母")
    if han_ratio > 0.60:
        return LanguageGuess(code="zh", confidence=min(1.0, 0.55 + han_ratio), basis="汉字")

    # 中英混排：按"词"而不是"字符"比较。
    #
    # 按字符比较会系统性偏向拉丁字母——一个汉字通常就是一个词（或半个词），
    # 而一个英文单词平均要 5 个字母。中文界面里嵌入英文术语
    # （如"打开 Settings 面板"）按字符算是英文占多数，但实际语义主体是中文。
    # 因此此处把汉字按 2 字折 1 词、拉丁字母按空格切分为词来比较，
    # 与人的直觉判断一致。
    if counts["han"] > 0:
        cjk_words = counts["han"] / 2.0
        latin_words = _count_latin_words(stripped)
        if cjk_words >= latin_words:
            ratio = counts["han"] / total
            return LanguageGuess(
                code="zh",
                confidence=min(0.95, 0.5 + ratio),
                basis=f"中英混排，中文约占 {cjk_words:.0f} 词、英文 {latin_words:.0f} 词",
            )

    # 纯拉丁字母语言，用特征字符与停用词区分英/法/德/西。
    if latin_ratio > 0.80:
        return _detect_latin(stripped, fallback)

    return LanguageGuess(code=fallback, confidence=0.2, basis="无法判定")


def decisive_script(text: str) -> str | None:
    """返回文本的"决定性脚本"类别（与 :func:`detect_language` 同阈值同优先级）。

    返回 ``"zh"``（汉字为主体）/ ``"ja"``（假名）/ ``"ko"``（谚文）/
    ``"ru"``（西里尔字母）；拉丁字母、混合或无字母文本无法给出决定性
    判定时返回 ``None``。

    与 :func:`detect_language` 的区别：这里只做**严格比例**判定，不走
    "中英混排按词折算"分支——那个分支允许汉字占比低至约 0.3 仍判为中文
    （面向源语言检测的召回），而本函数的调用方是**译文校验**与"同语言
    跳过前检查外文夹杂物"，必须宁可漏判也不误伤合法译文：译文里合法地
    混有标识符、路径、数字与代码，汉字占比被稀释是常态。
    """
    stripped = (text or "").strip()
    if not stripped:
        return None
    letters = [ch for ch in stripped if ch.isalpha()]
    if not letters:
        return None
    total = len(letters)
    counts = {"han": 0, "kana": 0, "hangul": 0, "cyrillic": 0}
    for ch in letters:
        name = _script_of(ch)
        if name in counts:
            counts[name] += 1
    if counts["kana"] / total > 0.05:
        return "ja"
    if counts["hangul"] / total > 0.10:
        return "ko"
    if counts["cyrillic"] / total > 0.30:
        return "ru"
    if counts["han"] / total > 0.60:
        return "zh"
    return None


def _count_latin_words(text: str) -> int:
    """统计文本中的拉丁字母单词数量。"""
    return len(re.findall(r"[A-Za-z]+", text))


def _detect_latin(text: str, fallback: str) -> LanguageGuess:
    """在拉丁字母语言之间做判定。"""
    lower = text.lower()
    words = [word.strip(".,;:!?()[]{}\"'`-") for word in lower.split()]
    words = [word for word in words if word]
    if not words:
        return LanguageGuess(code=fallback, confidence=0.2, basis="无有效词")

    scores: dict[str, float] = dict.fromkeys(("en", "fr", "de", "es"), 0.0)

    # 停用词命中
    for code, stopwords in _STOPWORDS.items():
        hits = sum(1 for word in words if word in stopwords)
        scores[code] += hits / len(words) * 2.0

    # 特征字符命中（权重更高，因为这些字符基本只出现在一种语言中）
    for code, distinctive in _DISTINCTIVE_CHARS.items():
        if code not in scores:
            continue
        hits = sum(1 for ch in text if ch in distinctive)
        if hits:
            scores[code] += 1.5

    best = max(scores, key=lambda key: scores[key])
    best_score = scores[best]
    if best_score <= 0.05:
        # 没有明显特征（例如短标识符、代码片段），英语是最安全的默认值
        return LanguageGuess(code="en", confidence=0.3, basis="拉丁字母，无语言特征")

    total_score = sum(scores.values()) or 1.0
    confidence = min(0.95, best_score / total_score)
    return LanguageGuess(code=best, confidence=confidence, basis="停用词与特征字符")


def _script_of(ch: str) -> str:
    """返回字符所属的脚本名称（仅关注本项目需要的几类）。"""
    code = ord(ch)

    # 汉字（含扩展区）
    if (
        0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF
        or 0xF900 <= code <= 0xFAFF or 0x20000 <= code <= 0x323AF
    ):
        return "han"
    # 平假名与片假名
    if 0x3040 <= code <= 0x30FF or 0x31F0 <= code <= 0x31FF:
        return "kana"
    # 谚文
    if 0xAC00 <= code <= 0xD7AF or 0x1100 <= code <= 0x11FF or 0x3130 <= code <= 0x318F:
        return "hangul"
    # 西里尔字母
    if 0x0400 <= code <= 0x04FF:
        return "cyrillic"
    # 拉丁字母（含扩展）
    if code < 0x0250 and "LATIN" in unicodedata.name(ch, ""):
        return "latin"
    if ch.isascii() and ch.isalpha():
        return "latin"

    # Unicode character names cover the remaining scripts selectable in the
    # UI without a language model / third-party detector on the hot path.
    name = unicodedata.name(ch, "")
    if name.startswith("LATIN "):
        return "latin"
    for prefix, script in (
        ("HANGUL", "hangul"), ("HIRAGANA", "kana"), ("KATAKANA", "kana"),
        ("CYRILLIC", "cyrillic"),
        ("COPTIC", "coptic"), ("ARABIC", "arabic"), ("HEBREW", "hebrew"),
        ("DEVANAGARI", "devanagari"), ("BENGALI", "bengali"),
        ("GURMUKHI", "gurmukhi"), ("GUJARATI", "gujarati"), ("ORIYA", "oriya"),
        ("TAMIL", "tamil"), ("TELUGU", "telugu"), ("KANNADA", "kannada"),
        ("MALAYALAM", "malayalam"), ("SINHALA", "sinhala"),
        ("THAI", "thai"), ("LAO", "lao"), ("TIBETAN", "tibetan"),
        ("MYANMAR", "myanmar"), ("GEORGIAN", "georgian"),
        ("ARMENIAN", "armenian"), ("ETHIOPIC", "ethiopic"),
        ("KHMER", "khmer"), ("GREEK", "greek"), ("THAANA", "thaana"),
        ("SYRIAC", "syriac"), ("BUGINESE", "lontara"),
        ("CHEROKEE", "cherokee"), ("YI", "yi"),
        ("CANADIAN SYLLABICS", "canadian"),
    ):
        if name.startswith(prefix + " "):
            return script

    return "other"


def is_probably_translatable(text: str) -> bool:
    """判断文本是否值得送去翻译。

    过滤掉纯数字、纯符号、以及无意义的单字符碎片，
    这些内容翻译后通常与原文相同，白白消耗推理时间。

    例外：**单个汉字/假名/谚文保留**。中文的"是/否"、日文的"閉じる"这类
    单字按钮在界面上很常见，且一个字就是一个完整词，翻译后有意义；
    而单个拉丁字母（图标残影、列表标记）几乎总是噪声。
    """
    stripped = (text or "").strip()
    if not stripped:
        return False

    letters = sum(1 for ch in stripped if ch.isalpha())
    if letters == 0:
        return False

    if len(stripped) == 1:
        # 单个 CJK 字符有意义，单个拉丁字母没有
        return _is_cjk_character(stripped)

    # 字母占比过低（例如 ``===>`` 或 ``12.5%``）不翻译
    return letters / len(stripped) >= 0.3


def _is_cjk_character(text: str) -> bool:
    """判断字符串是否由 CJK 字符构成。"""
    return any(
        start <= ord(ch) <= end
        for ch in text
        for start, end in (
            (0x4E00, 0x9FFF),   # 汉字
            (0x3040, 0x30FF),   # 平假名 / 片假名
            (0xAC00, 0xD7AF),   # 谚文
        )
    )
