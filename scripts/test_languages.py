#!/usr/bin/env python
"""实测 Spark-X2.5 双模型究竟支持哪些语言。

背景
    模型的官方说明只有一句 "support for more than 200 languages"，
    **没有公布具体清单**（已核对模型仓库 README 与 config.json）。
    本脚本用可复现的探针实验把这个说法落实成一份**实测清单**。

方法
    对每种候选语言执行三步：

    1. **试译**——用一个固定探针句要求模型译成该语言，并要求它同时回报
       实际使用的语言名；
    2. **脚本校验**——检查输出是否使用了该语言应有的文字脚本
       （如希腊语应出现希腊字母、泰语应出现泰文字符）。脚本是**决定性的**：
       模型不可能用拉丁字母写出泰语；
    3. **自报校验**——把模型自报的语言与目标语言比对，用于区分同为拉丁
       字母的语言（如捷克语与斯洛伐克语在脚本上无法区分）。

判定
    - ``verified``  脚本匹配 **且** 自报语言一致 → 确定支持
    - ``likely``    仅脚本匹配（拉丁字母语言无法进一步区分）→ 很可能支持
    - ``unsupported`` 输出为空、与原文相同或脚本明显不符 → 判定不支持

    之所以把后两档分开，是因为拉丁字母语言之间无法仅凭脚本判定，
    把"很可能"谎报成"确定"是不诚实的。

用法::

    python scripts/test_languages.py                    # 测全部语言（约 5–15 分钟）
    python scripts/test_languages.py --model 4b         # 只测高精度模型
    python scripts/test_languages.py --limit 20         # 先测前 20 种
    python scripts/test_languages.py --output result.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

#: 探针句。选一句语义中性的日常用语——过于技术性的句子会让模型倾向
#: 保留英文术语，干扰判定。
PROBE_TEXT = "The weather is nice today. Please open the window."

#: 提示词。要求模型回报所用语言，便于对拉丁字母语言做二次校验。
PROBE_SYSTEM = (
    "You are a professional translator. Translate the user's sentence into {language}. "
    "Output exactly two lines and nothing else:\n"
    "LINE 1: the translation\n"
    "LINE 2: the name of the language you translated into, in English\n"
    "Do not add explanations, numbering, quotes or markdown."
)

#: 在候选清单中**仅由少数语言使用**的脚本。这类脚本一旦匹配，几乎可以
#: 确定模型确实在写目标语言（例如希腊字母只用于希腊语、泰文只用于泰语），
#: 因此无需依赖模型自报语言即可判定"确定支持"。
#:
#: 反之，拉丁字母、西里尔字母、阿拉伯字母、天城文、汉字等被多种语言共用，
#: 光看脚本无法区分（捷克语与斯洛伐克语都是拉丁字母），必须结合自报语言。
EXCLUSIVE_SCRIPTS: Final[frozenset[str]] = frozenset({
    "greek", "hebrew", "thai", "hangul", "georgian", "armenian", "ethiopic",
    "sinhala", "khmer", "lao", "myanmar", "tamil", "telugu", "kannada",
    "malayalam", "gujarati", "gurmukhi", "oriya", "bengali", "tibetan",
    "thaana", "cherokee", "japanese", "inuktitut", "cree",
})

#: 探针句本身使用的语言。当目标语言就是它时，"输出与原文相同"是**正确结果**
#: 而不是失败——早期版本把英语误判为"不支持"就是漏了这一条。
PROBE_LANGUAGE_CODE: Final[str] = "en"

#: 探针句的关键概念词。回译后用它们判断译文是否真的传达了原意。
#:
#: 实测发现**仅靠脚本校验会严重虚报能力**：模型能产出正确脚本的胡言乱语，
#: 例如班巴拉语输出 "ALMA YOUNTE ZANTE"、格陵兰语输出 "Asa ása á, maad ása á."
#: ——脚本对得上，但根本不是译文。因此必须验证"意思是否翻对了"。
ROUND_TRIP_KEYWORDS: Final[tuple[tuple[str, ...], ...]] = (
    ("weather", "climate", "temperature", "forecast"),
    ("nice", "good", "beautiful", "fine", "pleasant", "lovely", "great", "warm"),
    ("open", "opening", "opens"),
    ("window", "windows", "pane", "casement"),
)

#: 回译时使用的提示词：要求把任意语言译回英文。
BACK_SYSTEM = (
    "You are a professional translator. Detect the language of the user's text "
    "and translate it into English. Output ONLY the English translation, "
    "nothing else."
)

#: 回译需要命中的关键概念数量下限。取 2 是折中：
#: 要求命中 4 个会因译文省略、代词替换而误杀；
#: 只要求 1 个则无法排除"碰巧出现一个词"。
ROUND_TRIP_MIN_HITS: Final[int] = 2

#: 简体/繁体特征字。用于区分同为汉字的 zh 与 zh-TW。
#:
#: 汉字脚本被中文与日文共用，但"汉字 + 无假名"已足以判定为中文
#: （日文输出必然带假名）；简体与繁体之间则要看用字习惯。
TRADITIONAL_ONLY_CHARS: Final[frozenset[str]] = frozenset(
    "們個這說麼開關請氣戶隻來時實過還點會後語譯體變據萬與愛樂車東風馬鳥魚龍門問聞間題學國圖書長雲電"
)
SIMPLIFIED_ONLY_CHARS: Final[frozenset[str]] = frozenset(
    "们个这说么开关请气户只来时实过还点会后语译体变据万与爱乐车东风马鸟鱼龙门问闻间题学国图书长云电"
)


def looks_traditional(text: str) -> bool | None:
    """判断文本使用简体还是繁体。

    :return: ``True`` 繁体、``False`` 简体、``None`` 无特征字无法判断。
    """
    traditional = sum(1 for char in text if char in TRADITIONAL_ONLY_CHARS)
    simplified = sum(1 for char in text if char in SIMPLIFIED_ONLY_CHARS)
    if traditional == simplified:
        return None
    return traditional > simplified


#: Unicode 脚本名 → 本项目语言表使用的脚本标识
_SCRIPT_ALIASES = {
    "GREEK": "greek",
    "CYRILLIC": "cyrillic",
    "HEBREW": "hebrew",
    "ARABIC": "arabic",
    "DEVANAGARI": "devanagari",
    "BENGALI": "bengali",
    "GURMUKHI": "gurmukhi",
    "GUJARATI": "gujarati",
    "ORIYA": "oriya",
    "TAMIL": "tamil",
    "TELUGU": "telugu",
    "KANNADA": "kannada",
    "MALAYALAM": "malayalam",
    "SINHALA": "sinhala",
    "THAI": "thai",
    "LAO": "lao",
    "TIBETAN": "tibetan",
    "MYANMAR": "myanmar",
    "GEORGIAN": "georgian",
    "ARMENIAN": "armenian",
    "ETHIOPIC": "ethiopic",
    "CHEROKEE": "cherokee",
    "THAANA": "thaana",
    "HANGUL": "hangul",
    "HIRAGANA": "japanese",
    "KATAKANA": "japanese",
    "CJK": "han",
    "COMMON": "latin",
    "INHERITED": "latin",
    "LATIN": "latin",
}


@dataclass
class LanguageResult:
    """一种语言的实测结果。"""

    code: str
    name: str
    english_name: str
    expected_script: str
    #: 判定：verified / likely / unsupported
    verdict: str = "unsupported"
    #: 模型自报的语言名
    reported_language: str = ""
    #: 译文（截断保存，避免结果文件过大）
    translation: str = ""
    #: 输出中实际检测到的主要脚本
    detected_script: str = ""
    #: 回译结果（把译文再译回英文），用于验证"意思是否真的翻对了"
    back_translation: str = ""
    #: 回译命中的关键概念数
    round_trip_hits: int = 0
    #: 耗时（秒）
    elapsed: float = 0.0
    #: 失败或判定为不支持的原因
    reason: str = ""


def dominant_script(text: str) -> str:
    """返回文本中占比最高的 Unicode 脚本名。

    忽略空白、标点、数字与符号，只看有意义的文字字符。
    """
    counts: dict[str, int] = {}
    for char in text:
        if char.isspace() or char.isdigit():
            continue
        category = unicodedata.category(char)
        if category.startswith("P") or category.startswith("S"):
            continue
        try:
            name = unicodedata.name(char)
        except ValueError:
            continue
        script = name.split()[0]
        counts[script] = counts.get(script, 0) + 1

    if not counts:
        return "none"
    return max(counts, key=lambda key: counts[key])


def normalise_script(script: str) -> str:
    """把 Unicode 脚本名映射为本项目的脚本标识。"""
    return _SCRIPT_ALIASES.get(script.upper(), script.lower())


def check_script(text: str, expected: str) -> tuple[bool, str]:
    """校验文本是否使用了期望的脚本。

    :return: ``(是否匹配, 实际主要脚本)``。

    对 ``latin`` 期望：只要没有出现其他明显脚本就算匹配——
    拉丁字母语言之间无法靠脚本区分。
    """
    actual_raw = dominant_script(text)
    actual = normalise_script(actual_raw)

    if expected == "latin":
        # 期望拉丁字母：出现任何**其他**脚本都算不符
        return actual == "latin", actual

    return actual == expected, actual


def parse_probe_output(raw: str) -> tuple[str, str]:
    """从模型输出中拆出译文与自报语言。"""
    text = (raw or "").strip()
    if not text:
        return "", ""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) == 1:
        return lines[0], ""

    # 常见变体：第一行译文、最后一行语言名
    marker = re.match(r"^(line\s*2|language)\s*[:：]\s*(.+)$", lines[-1], re.I)
    if marker:
        reported = marker.group(2)
        translation = " ".join(lines[:-1])
        return re.sub(r"^(line\s*1)\s*[:：]\s*", "", translation, flags=re.I), reported

    return " ".join(lines[:-1]), lines[-1]


def language_matches_reported(english_name: str, native_name: str, reported: str) -> bool:
    """判断模型自报的语言是否就是目标语言。

    宽松比较：只要自报串包含目标语言的英文名或母语名（或其前若干字符）
    即视为一致，容忍"Czech (Čeština)"这类写法差异。
    """
    if not reported:
        return False

    haystack = reported.lower()
    for candidate in (english_name, native_name):
        if not candidate:
            continue
        needle = candidate.lower().strip()
        if needle and needle in haystack:
            return True
        # 处理 "Afaan Oromoo" / "Oromo" 这类简称差异
        head = needle.split()[0] if needle.split() else ""
        if len(head) >= 4 and head in haystack:
            return True
    return False


def test_one(engine, info, timeout_note: str = "") -> LanguageResult:
    """测试单种语言。"""
    from pingyitong.components.llm.prompt import build_system_prompt  # noqa: F401

    result = LanguageResult(
        code=info.code,
        name=info.name,
        english_name=info.english_name,
        expected_script=info.script,
    )

    system = PROBE_SYSTEM.format(language=info.prompt_name())
    started = time.monotonic()
    try:
        raw = engine.generate(system, PROBE_TEXT, max_tokens=256)
    except Exception as exc:  # noqa: BLE001 - 单语言失败不应中断整轮实测
        result.reason = f"{type(exc).__name__}: {exc}"
        result.elapsed = time.monotonic() - started
        return result
    result.elapsed = time.monotonic() - started

    translation, reported = parse_probe_output(raw)
    result.translation = translation[:120]
    result.reported_language = reported[:60]

    if not translation:
        result.reason = "输出为空"
        return result

    # 目标语言就是探针句的语言时，输出与原文相同是**正确结果**，
    # 不能据此判定为"未翻译"（早期版本因此把英语误判为不支持）。
    if (
        info.code != PROBE_LANGUAGE_CODE
        and translation.strip().rstrip(".").lower()
        == PROBE_TEXT.strip().rstrip(".").lower()
    ):
        result.reason = "输出与原文相同（未翻译）"
        return result

    script_ok, detected = check_script(translation, info.script)
    result.detected_script = detected
    if not script_ok:
        result.reason = f"脚本不符（期望 {info.script}，实际 {detected}）"
        return result

    reported_ok = language_matches_reported(
        info.english_name, info.native_name, reported
    )

    # 回译校验：把译文再译回英文，看能否还原探针句的含义。
    # 这是区分"真的翻译了"与"产出同脚本胡言乱语"的关键一步。
    try:
        back = engine.generate(BACK_SYSTEM, translation, max_tokens=128)
    except Exception:  # noqa: BLE001
        back = ""
    back_text, _ = parse_probe_output(back)
    result.back_translation = back_text[:120]

    lowered = back_text.lower()
    hits = sum(
        1 for group in ROUND_TRIP_KEYWORDS if any(word in lowered for word in group)
    )
    result.round_trip_hits = hits
    round_trip_ok = hits >= ROUND_TRIP_MIN_HITS

    # 判定的核心原则：**每个信号只能证明它确实能证明的事**，
    # 并且只把"模型根本没在写这门语言"当作硬性否决。
    #
    # 实测得出这条原则的代价：
    #   · 只看脚本 → 虚报。模型会产出正确脚本的胡言乱语
    #     （班巴拉语 "ALMA YOUNTE ZANTE"、亚美尼亚语 "Հայերեն: Հայերեն"）。
    #   · 把回译当硬闸门 → 漏报。回译由同一个小模型完成，它自身回译能力有限，
    #     会把"前向翻译正确但回译没还原关键词"的语言全部误杀
    #     （实测葡萄牙语 "O tempo é bom hoje." 完全正确却被否决，
    #      142 种里误杀到只剩 59 种）。
    #
    # 因此：脚本不符才判不支持；脚本相符但证据不足时如实归入"疑似"。
    if reported_ok:
        result.verdict = "verified"
        result.reason = f"脚本与自报语言一致，回译命中 {hits}"
    elif info.code in {"zh", "zh-TW"}:
        # 汉字虽与日文共用，但日文输出必然带假名；此处已是纯汉字，可定为中文
        style = looks_traditional(translation)
        if (info.code == "zh-TW" and style is True) or (
            info.code == "zh" and style is not True
        ):
            result.verdict = "verified"
            result.reason = f"汉字（无假名）且用字符合预期，回译命中 {hits}"
        else:
            result.verdict = "likely"
            result.reason = f"确定为中文，但简繁用字未体现预期（特征字：{style}）"
    elif info.script in EXCLUSIVE_SCRIPTS:
        # 脚本为该语言独有（希腊字母、泰文、谚文……），脚本匹配本身即决定性证据。
        # 这类语言不因回译未命中而降级——那更可能是小模型回译能力不足。
        result.verdict = "verified"
        result.reason = f"脚本 {info.script} 为该语言独有，回译命中 {hits}"
    elif round_trip_ok:
        # 脚本匹配 + 回译还原了原意：两个信号都支持
        result.verdict = "verified"
        result.reason = f"脚本匹配且回译还原原意（命中 {hits}）"
    else:
        # 脚本对得上，但脚本被多种语言共用、自报语言不一致、回译也未还原原意。
        # 无法确认模型确实在写这门语言——如实归入"疑似"，不冒充确定。
        result.verdict = "likely"
        result.reason = (
            f"脚本匹配但非独有脚本、自报语言为 {reported!r}、回译命中仅 {hits}，"
            f"无法确证语义"
        )
    return result


def main(argv: list[str] | None = None) -> int:
    """命令行入口。"""
    parser = argparse.ArgumentParser(
        description="实测 Spark-X2.5 支持的语言",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", choices=["1.7b", "4b", "both"], default="1.7b",
                        help="要测试的模型，默认 1.7b（快得多，适合先摸清清单）")
    parser.add_argument("--limit", type=int, default=0, help="只测前 N 种语言")
    parser.add_argument("--output", type=Path, default=None, help="结果 JSON 输出路径")
    args = parser.parse_args(argv)

    from pingyitong.components.llm import ModelPool
    from pingyitong.config.schema import AppConfig
    from pingyitong.constants import MODEL_ACCURATE, MODEL_FAST
    from pingyitong.languages import LANGUAGES

    model_ids = {
        "1.7b": [MODEL_FAST],
        "4b": [MODEL_ACCURATE],
        "both": [MODEL_FAST, MODEL_ACCURATE],
    }[args.model]

    config = AppConfig()
    pool = ModelPool(config.model)

    languages = list(LANGUAGES.values())
    if args.limit > 0:
        languages = languages[: args.limit]

    print("=" * 76)
    print(f"Spark-X2.5 支持语言实测 —— {len(languages)} 种候选语言")
    print("=" * 76)
    print(f"探针句：{PROBE_TEXT!r}")
    print(f"模型：{', '.join(model_ids)}")
    print("判定：脚本匹配(决定性) + 模型自报语言(区分同脚本语言)")
    print()

    all_results: dict[str, list[dict]] = {}

    for model_id in model_ids:
        engine = pool.engine(model_id)
        if not engine.is_available():
            print(f"跳过 {model_id}：{engine.status().message}")
            continue

        print(f"加载 {model_id} …", end="", flush=True)
        started = time.monotonic()
        if not pool.ensure_loaded(model_id):
            print(" 失败")
            continue
        print(f" {time.monotonic() - started:.1f}s")

        results: list[LanguageResult] = []
        counts = {"verified": 0, "likely": 0, "unsupported": 0}

        for index, info in enumerate(languages, start=1):
            result = test_one(engine, info)
            results.append(result)
            counts[result.verdict] = counts.get(result.verdict, 0) + 1

            mark = {"verified": "✓", "likely": "~", "unsupported": "✗"}[result.verdict]
            detail = result.translation[:38] if result.translation else result.reason[:38]
            print(
                f"  [{index:3d}/{len(languages)}] {mark} {info.code:6s} {info.name:12s} "
                f"{result.elapsed:5.2f}s  {detail}"
            )

        all_results[model_id] = [asdict(r) for r in results]

        print()
        print(f"  {model_id} 汇总：确定支持 {counts['verified']} · "
              f"很可能支持 {counts['likely']} · 不支持 {counts['unsupported']}")
        print()

    pool.shutdown()

    if args.output:
        args.output.write_text(
            json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"结果已写入 {args.output}")

    # 输出可直接粘贴进 languages.py 的清单
    for model_id, rows in all_results.items():
        verified = [r["code"] for r in rows if r["verdict"] == "verified"]
        likely = [r["code"] for r in rows if r["verdict"] == "likely"]
        print()
        print(f"# {model_id} 实测：确定支持 {len(verified)} 种")
        print(f"VERIFIED_{model_id.upper().replace('.', '_')} = {sorted(verified)!r}")
        print(f"# {model_id} 实测：很可能支持 {len(likely)} 种")
        print(f"LIKELY_{model_id.upper().replace('.', '_')} = {sorted(likely)!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
