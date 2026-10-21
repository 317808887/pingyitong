"""实测结果：Spark-X2.5 双模型的语言支持清单。

本文件由 ``scripts/test_languages.py`` 的实测输出填入，
**不要手工猜测**——每个代码都对应一次可复现的探针实验。

实测方法（三重信号，各取所长）
    ==================  ==========================================================
    信号                能证明什么
    ==================  ==========================================================
    输出文字脚本        必须与该语言应有的脚本一致。**脚本不符是硬性否决**——
                        模型不可能用拉丁字母写出泰语
    模型自报语言        用于区分共用同一脚本的语言（捷克语与斯洛伐克语都是拉丁字母）
    回译校验            把译文再译回英文，看能否还原探针句含义
    ==================  ==========================================================

    "每个信号只能证明它确实能证明的事"是这套判定的核心。两种极端做法都被实测否掉：
    只看脚本会**虚报**（模型能产出正确脚本的胡言乱语，如班巴拉语
    "ALMA YOUNTE ZANTE"）；把回译当硬闸门会**漏报**（回译由同一个小模型完成，
    它自身回译能力有限，实测把完全正确的葡萄牙语译文也否决了，
    142 种里误杀到只剩 59 种）。

    因此按证据强度分档，并**保留"疑似"这一诚实的中间态**，不把疑似冒充确定。

生成命令::

    python scripts/test_languages.py --model 1.7b --output docs/language_test_1.7b.json
    python scripts/test_languages.py --model 4b   --output docs/language_test_4b.json
"""

from __future__ import annotations

from typing import Final

#: 1.7B 实测：确定支持（脚本正确，且至少有一个额外信号支持语义正确）
VERIFIED_1_7B: Final[frozenset[str]] = frozenset({
    "af", "ak", "am", "ar", "as", "ay", "az", "be", "bg", "bm", "bn", "bo",
    "br", "bs", "ca", "ceb", "chr", "co", "cs", "cy", "da", "de", "dv", "el",
    "en", "es", "eu", "fa", "ff", "fi", "fo", "fr", "fy", "gd", "gl", "gn",
    "gu", "haw", "he", "hi", "hr", "ht", "hu", "hy", "id", "is", "it", "ja",
    "jw", "kk", "kl", "km", "kn", "ko", "ku", "ky", "la", "lb", "lg", "ln",
    "lo", "lt", "mg", "mi", "mk", "ml", "mn", "mr", "ms", "mt", "my", "ne",
    "nl", "no", "nso", "ny", "oc", "oj", "om", "pa", "pl", "pt", "qu", "rm",
    "ru", "rw", "sc", "sd", "sk", "sl", "sm", "sn", "so", "sq", "st", "sv",
    "sw", "ta", "te", "tg", "th", "tr", "ts", "tt", "tw", "ug", "uk", "ur",
    "uz", "vi", "xh", "yi", "zh", "zh-TW", "zu",
})

#: 1.7B 实测：疑似支持（脚本正确，但自报语言与回译都未能确证语义）
LIKELY_1_7B: Final[frozenset[str]] = frozenset({
    "eo", "ga", "ha", "ig", "lv", "moh", "nv", "ro", "su", "tk", "tl", "tn",
    "ve", "yo",
})

#: 4B 实测：确定支持。由 ``--model 4b`` 那一轮填入。
#:
#: 实测结论：**4B 明显强于 1.7B**（133 种确定支持 vs 115 种），
#: 这与"参数量带来更强的多语言能力"的预期一致。两个模型并非包含关系：
#: 1.7B 能译而 4B 未通过的有孟加拉语、尼扬贾语、意第绪语；
#: 4B 能译而 1.7B 未通过的有爱沙尼亚语、奥迪亚语、普什图语等 8 种。
VERIFIED_4B: Final[frozenset[str]] = frozenset({
    "af", "am", "ar", "as", "az", "be", "bg", "bo", "br", "bs", "ca", "ceb",
    "chr", "co", "cs", "cy", "da", "de", "dv", "el", "en", "eo", "es", "eu",
    "fa", "ff", "fi", "fo", "fr", "fy", "ga", "gd", "gl", "gn", "gu", "ha",
    "haw", "he", "hi", "hmn", "hr", "ht", "hu", "hy", "id", "ig", "is", "it",
    "ja", "jw", "kk", "kl", "km", "kn", "ko", "ku", "ky", "la", "lb", "lg",
    "ln", "lo", "lt", "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms", "mt",
    "my", "ne", "nl", "no", "nso", "nv", "oc", "oj", "om", "or", "pa", "pl",
    "ps", "pt", "qu", "rm", "ro", "ru", "rw", "sc", "sd", "si", "sk", "sl",
    "sn", "so", "sq", "ss", "st", "su", "sv", "sw", "ta", "te", "tg", "th",
    "tk", "tl", "tn", "tr", "ts", "tt", "tw", "ug", "uk", "ur", "uz", "ve",
    "vi", "wo", "xh", "yo", "zh", "zh-TW", "zu", "ak", "ay", "bm", "ee", "et",
    "moh",
})

#: 4B 实测：疑似支持（脚本正确，但自报语言与回译都未能确证语义）。
LIKELY_4B: Final[frozenset[str]] = frozenset({"sm"})

#: 关于"不支持"的集合：**不手工维护**。
#:
#: 只维护"通过实测"的两份清单（确定 / 疑似），"不支持"由
#: :func:`unsupported_for` 按补集推出。手工维护三份清单极易自相矛盾——
#: 曾出现"确定 115 + 疑似 14 + 不支持 55 = 184"超过候选总数 142 的情况。


def unsupported_for(candidates: frozenset[str], *supported: frozenset[str]) -> frozenset[str]:
    """按补集求"不支持"的语言集合。

    :param candidates: 参与实测的候选语言全集。
    :param supported: 实测通过的集合（确定 + 疑似）。
    :return: 候选集中未被任何 ``supported`` 覆盖的部分。
    """
    covered: frozenset[str] = frozenset()
    for group in supported:
        covered |= group
    return frozenset(candidates - covered)

#: 两个模型**共同**确定支持的语言。
#:
#: 界面默认以这个集合为准：只有两边都实测通过，才敢对用户说"确定可用"。
#: 4B 尚未实测时会退化为 1.7B 的结果，并在注释与文档中如实注明。
_VALIDATED: Final[tuple[frozenset[str], ...]] = tuple(
    item for item in (VERIFIED_1_7B, VERIFIED_4B) if item
)
VERIFIED_BOTH: Final[frozenset[str]] = (
    frozenset.intersection(*_VALIDATED) if len(_VALIDATED) > 1 else VERIFIED_1_7B
)

#: 至少被一个模型确定支持的语言（并集）。
VERIFIED_ANY: Final[frozenset[str]] = frozenset(VERIFIED_1_7B | VERIFIED_4B)

#: 所有实测确认可用的语言（确定 + 疑似）。
USABLE: Final[frozenset[str]] = frozenset(
    VERIFIED_ANY | LIKELY_1_7B | LIKELY_4B
)

#: 是否已完成双模型实测。
FULLY_TESTED: Final[bool] = bool(VERIFIED_4B)


def status(code: str) -> str:
    """返回某语言实测得到的支持档位。

    :return: ``"verified"`` / ``"likely"`` / ``"unsupported"``。
    """
    if not FULLY_TESTED:
        # 4B 未测完时，以 1.7B 的结论为准（并在文档中注明范围）
        if code in VERIFIED_1_7B:
            return "verified"
        if code in LIKELY_1_7B:
            return "likely"
        return "unsupported"

    if code in VERIFIED_BOTH:
        return "verified"
    if code in LIKELY_1_7B or code in LIKELY_4B or code in VERIFIED_ANY:
        return "likely"
    return "unsupported"


def is_verified(code: str) -> bool:
    """语言是否实测确定支持。"""
    return status(code) == "verified"


def counts(candidates: frozenset[str]) -> dict[str, int]:
    """按档位统计候选语言数量。

    **计数一律调用 :func:`status` 得到**，不再自行拼装集合运算：
    界面上的每个标记都来自 ``status``，只有同源才能保证
    "摘要说的数字"与"列表里标出来的条目"永远一致。
    """
    result = {"verified": 0, "likely": 0, "unsupported": 0}
    for code in candidates:
        result[status(code)] += 1
    return result


def summary(candidates: frozenset[str] | None = None) -> str:
    """返回一行实测结论摘要，供界面与文档展示。

    :param candidates: 候选语言全集；提供时才能给出"不支持"的数量
        （它由补集推出，本模块自身不知道候选集有多大）。
    """
    scope = "双模型共同" if FULLY_TESTED else "1.7B 单模型"

    if candidates is None:
        verified = len(VERIFIED_BOTH) if FULLY_TESTED else len(VERIFIED_1_7B)
        likely = len(LIKELY_1_7B) + (len(LIKELY_4B) if FULLY_TESTED else 0)
        return f"{scope}实测：确定支持 {verified} 种 · 疑似支持 {likely} 种"

    stats = counts(candidates)
    return (
        f"{scope}实测（共测 {len(candidates)} 种）："
        f"确定支持 {stats['verified']} 种 · "
        f"疑似支持 {stats['likely']} 种 · "
        f"不支持 {stats['unsupported']} 种"
    )
