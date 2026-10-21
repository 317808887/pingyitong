"""翻译服务：编排预处理、术语库、缓存与模型池。

这是业务功能层的主入口，负责把一帧 OCR 结果变成可直接渲染的译文条目。
调用顺序（每一步都有明确目的）：

1. **文本后处理**：把碎片文本块还原为通顺段落（原创算法）；
2. **语言判定与跳过**：源语言等于目标语言时不做无谓推理；
3. **术语替换**：命中术语替换为占位符，保证译文稳定一致；
4. **缓存查询**：命中直接返回，跳过推理；
5. **模型调度**：按手动/资源/内容三级策略选定模型；
6. **批量推理**：多个片段合并为一次请求，显著降低单片段开销；
7. **解析与还原**：切分模型输出，还原术语占位符，失败时降级逐条翻译；
8. **组装渲染条目**：把译文与原文包围盒绑定，交给渲染层。

降级策略
    批量解析失败（模型没按分隔符输出）时会自动退化为逐条翻译。
    这是必要的兜底：批量请求虽快，但只要模型少输出一个分隔符，
    整批结果就会错位，而此时逐条翻译仍能给出正确结果。
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections import Counter, OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from copy import copy, deepcopy
from dataclasses import dataclass, field, replace

from ..components.capture.geometry import Rect
from ..components.llm import (
    ModelPool,
    TranslationCache,
    build_system_prompt,
    build_user_prompt,
    parse_response,
)
from ..components.llm.prompt import _strip_source_suffix
from ..components.ocr import RecognizedFrame
from ..components.render import RenderFrame, RenderItem
from ..constants import MODEL_ACCURATE, MODEL_FAST, MODEL_LABELS, SEGMENT_DELIMITER
from ..core.model_scheduler import ModelScheduler
from ..core.resource_monitor import ResourceSnapshot
from ..utils.logger import get_logger
from .code_filter import looks_like_code
from .glossary_service import Glossary
from .history_service import HistoryService
from .language import detect_language, has_foreign_prose, has_reversed_until_condition
from .semantic_review import (
    REVIEW_POLICY_VERSION,
    ResolutionDecision,
    ResolutionInput,
    ReviewDecision,
    ReviewedTranslation,
    ReviewInput,
    build_resolution_prompt,
    build_review_prompt,
    compact_review_digest,
    parse_resolutions,
    parse_reviews,
    resolution_digest,
    source_digest,
    translations_agree,
)
from .text_preprocess import Paragraph, TextPreprocessor

logger = get_logger(__name__)

#: 批量对位失败后允许逐条重试的最大条数。
#:
#: 逐条重试会把一次请求放大成 N 次（每次都重传系统提示），实测能把单帧
#: 耗时从数秒推到 49 秒。实时翻译等不起，因此设上限：
#: 超出的片段保留原文，宁可少译几条也不让界面卡住。
MAX_INDIVIDUAL_RETRIES = 4

#: 单帧纠错总预算：允许追加的纠错批请求数上限。
#:
#: 正常帧一次纠错都不触发，预算与延迟无关；异常帧最多再补两批——
#: 批量纠错可能把输入拆成多个子批、多批帧的逐条降级也会逐批发生，
#: 不设总账时最坏一帧会追加近十次串行请求。超限的片段直接放弃纠错，
#: 保留原文或已有译文。
MAX_FRAME_CORRECTION_BATCHES = 2

#: 4B（高精度）模型本地单请求的片段条数上限。
#:
#: 批量条数越多，单请求的解码时间越长，单帧延迟随之恶化；4B 比
#: 1.7B 更重，装批上限收紧到 8 条。1.7B 维持 ``model.batch_size``
#: 配置不变，在线条批量也不受此限（云端上下文更长）。
MAX_ACCURATE_BATCH_ITEMS = 8

#: 历史写入的短窗口去重参数。
#:
#: 静止画面上每帧都会重新产出同一批段落：缓存命中的条目已直接跳过，
#: 但缓存关闭、配置变更后重译等路径仍会让相同原文在连续帧里反复
#: 出现。同一原文在窗口期内只记一条；记录持续刷新时间戳，内容
#: 连续不变就持续去重，间隔超过窗口后才允许再次记录。
HISTORY_DEDUP_WINDOW_S = 15.0
HISTORY_DEDUP_MAX_ENTRIES = 256

#: 语言纠错追加语（批量纠错与单段重试共用）。
#:
#: 模型纠正语种时会顺手把 HTML 标签、属性和模板语法也"翻译"掉
#: （``<div class="用户资料">``），译文渲染后直接损坏结构；反过来
#: 也禁止把标签间的可见文本当作标记的一部分原样吐回。因此除技术
#: 标识符外，必须显式声明标签/属性/模板原样保留、只译可见文本。
_LANGUAGE_CORRECTION_SUFFIX = (
    ". Preserve only unchanged technical identifiers, paths and URLs. "
    "Keep HTML/XML tags, their attributes and template syntax "
    "({{ }}, {% %}, <% %>) exactly as they are, and translate ONLY the "
    "visible text between them."
)
_TIME_CONDITION_CORRECTION_SUFFIX = (
    "\nCorrect the time condition. Japanese 'Yまで ... ない' prohibits the action BEFORE Y finishes, "
    "not after: use '...完成前请勿...' or an equivalent faithful translation. "
    "English 'Do not X before Y finishes' also prohibits X BEFORE Y finishes, "
    "not after: use 'Y完成前请勿X' or an equivalent faithful translation. "
    "'Keep/remain/continue X until Y ends' means maintaining X BEFORE and UNTIL Y ends, "
    "never starting X AFTER Y ends. Preserve the action and its polarity. Output only the translation."
)
_NUMBER_CORRECTION_SUFFIX = (
    "\nCorrect the missing quantities: copy every numeral from the source exactly, "
    "including totals, denominators, dates and repeated values. Do not summarize them. "
    "Output only the complete translation."
)
_NUMERIC_TOKEN_RE = re.compile(r"(?<!\d)\d+(?:[.,]\d+)*(?!\d)")
_INTERNAL_TERM_RE = re.compile(r"\[TERM_\d+\]")


def _missing_numerals(original: str, translated: str) -> list[str]:
    """Return source numerals absent from a translation, preserving multiplicity."""
    # [TERM_0] is an internal lookup key, not a user-visible quantity. Its
    # omission is handled by the glossary guard, and must not spend the
    # separate numeric-recovery budget before placeholder restoration.
    original = _INTERNAL_TERM_RE.sub("", original)
    translated = _INTERNAL_TERM_RE.sub("", translated)
    remaining = Counter(_NUMERIC_TOKEN_RE.findall(translated))
    missing = []
    for numeral in _NUMERIC_TOKEN_RE.findall(original):
        if remaining[numeral]:
            remaining[numeral] -= 1
        else:
            missing.append(numeral)
    return missing


def _parse_single_response(raw: str, source: str, preserve_source: bool) -> list[str] | None:
    """Keep all clauses when a model inserts batch markers into one input.

    No cross-block mapping is involved in a single request. Remove only exact
    protocol markers absent from its source; damaged markers and foreign prose
    still go through the usual quality gate. Literal source markers stay intact.
    """
    parsed = parse_response(raw, 1, preserve_source, source_texts=[source])
    if parsed and SEGMENT_DELIMITER not in source and SEGMENT_DELIMITER in parsed[0]:
        parts = parsed[0].split(SEGMENT_DELIMITER)
        if all(part.strip() for part in parts):
            # A model may append each sentence's own source instead of the full
            # input. Strip only exact, ordered source clauses when counts agree;
            # unmatched suffixes remain subject to the usual language gate.
            clauses = [part.strip() for part in re.split(
                r"(?<=[。！？])\s*|(?<=[.!?])\s+|\n+", source.strip()
            ) if part.strip()]
            aligned_clauses = len(clauses) == len(parts)
            translated_parts: list[str] = []
            for index, part in enumerate(parts):
                normalized = parse_response(part, 1)
                if not normalized:
                    return None
                candidate = normalized[0]
                if preserve_source:
                    stripped = _strip_source_suffix(candidate, source)
                    if stripped == candidate and aligned_clauses:
                        stripped = _strip_source_suffix(candidate, clauses[index])
                    candidate = stripped
                if not candidate:
                    return None
                translated_parts.append(candidate)
            return [" ".join(translated_parts)]
    return parsed

# A small model can omit protected tokens, especially at sentence start.
# Bound the extra requests so glossary recovery cannot stall a busy frame.
MAX_GLOSSARY_RECOVERIES_PER_FRAME = 4
MAX_GLOSSARY_RECOVERY_TERMS = 8
MAX_GLOSSARY_RECOVERY_CHARS = 400
_ENGLISH_SCAFFOLD = re.compile(
    r"\b(?:the|has|have|is|are|was|were|a|an|of|to|for|with)\b",
    re.IGNORECASE,
)

#: 单次请求输入预算的安全余量（token），覆盖 chat 模板与特殊占位符的开销。
_PROMPT_TOKEN_MARGIN = 96

#: system 提示词是纯英文文本，BPE 分词约 4 字符/token。按 3 字符/token
#: 保守折算 token 数（保留 1/4 余量）；待翻译的用户文本不受此影响，
#: 仍按 CJK 最坏情况 1 字符 1 token 计。此前直接把 system 提示词的
#: 字符数当 token 数扣减（1400 字符当 1400 token），实际只有约 350 token，
#: 导致输入预算被压到 256 字符下限，内容较多的帧被拆成远多于必要的
#: 批次——每多一批就是一次串行请求往返，徒增翻译延迟。
_SYSTEM_PROMPT_CHARS_PER_TOKEN = 3

#: 输入预算的下限。极小上下文的模型也至少保证两三个短句能进同一批。
_MIN_INPUT_CHAR_BUDGET = 256

#: 批内记账用的分隔符开销（字符）：user 提示词里相邻片段之间实际插入
#: ``\n<<<SEG>>>\n`` 共 11 字符，按每片段 12 字符记账（n 片段 ≤ n×12）。
_SEGMENT_SEPARATOR_CHARS = 12

#: 在线 AI 模式下的输入字符预算下限。云端模型上下文普遍 ≥8k token
#: （system 提示词约 470 token，扣掉输出与余量后仍有数千 token 空间）；
#: 取 4000 字符在"减少请求往返"与"单请求耗时"之间平衡，且远低于
#: 触发服务商上下文限制的量级。
_API_INPUT_CHAR_BUDGET = 4000

#: 动态输出上限的系数与余量。译文长度极少超过源文本的 2 倍（中英互译
#: 均 < 2×；中文是最紧凑的书写语言，译成英文按 token 计约 1.5–2×），
#: 取 2.5× 再加 32 token 常量余量，正常翻译**永不触顶**；它只在模型
#: 跑飞（复读、失控生成）时截断，把最坏情况从 max_tokens（384 token
#: ≈ 1.5 秒）压到与输入长度成比例的量级。仅显式请求模型重复原文的
#: 调用需要双倍预算；生产由 RenderItem.source_text 保存原文，只生成译文。
_OUTPUT_CAP_RATIO = 2.5
_OUTPUT_CAP_MARGIN = 32
_OUTPUT_CAP_FLOOR = 48

#: 在线模式的输出上限硬顶。云端模型上下文远大于本地（≥8k token），
#: ``api.max_tokens`` 默认 768 且未在界面暴露——若仍以它钳制输出，
#: 长段落（剪贴板翻译）与密集帧的译文会在半途被截断：截断本身是
#: 用户可见的精度事故，批量模式下还会连带对位失败触发逐条重试。
#: 因此在线模式由动态估计（2.5×源 token + 分隔符记账）直接决定上限，
#: 本常量只防"跑飞"最坏情况（正常译文始终与输入成比例，触不到它；
#: 病态输出另有 timeout_seconds 兜底延迟）。取值与 settings 对
#: ``api.max_tokens`` 的钳制上限一致。
_API_OUTPUT_HARD_LIMIT = 8192

#: 输出里每个片段分隔符的 token 记账开销。批量响应必须在段间复述
#: ``<<<SEG>>>``（本地与云端 tokenizer 通常 5–12 token/个），而输出上限
#: 只按源文本长度估算、感知不到分隔符——一帧塞满短 UI 标签时（屏译
#: 最常见场景）上限可能掐断分隔符，解析对位失败触发逐条串行重试，
#: 一帧反而慢一个数量级。按 16 token/个从宽计入，宁可放宽也不触顶。
_SEGMENT_DELIMITER_TOKEN_ALLOWANCE = 16
_CJK_CHAR_RE = re.compile(r"[\u3400-\u9fff]")

#: 模型"拒答"特征。实测 1.7B 与 4B 都会偶发：一句完全正常的英文产品文案
#: 被判成需要行使内容安全策略，返回"…我无法完成该翻译…"。这段文字会
#: 原样画到屏幕上——用户看到的不是译文而是致歉，属可见的质量事故。
#:
#: 判定要求**同时**命中"拒绝动词"与"政策措辞/道歉式开头"：仅凭
#: "无法翻译"字面会误伤正常译文（例如 "该软件无法翻译此语言"）。
#: 动词部分放宽到"同一句内相隔不超过 12 字"，覆盖实测的多种措辞：
#: "无法完成该翻译"、"无法进行翻译"、"无法为你提供相关翻译"。
#: 真正的误伤防护由下面的政策措辞/开头要求承担。
_REFUSAL_VERB = re.compile(r"(无法|不能|拒绝)[^。！？\n]{0,12}(翻译|译文|译)")
_REFUSAL_POLICY = re.compile(
    r"(不符合|违反|违规|不当|不恰当|不适宜|不良信息|公序良俗|法律法规|内容政策|安全政策|健康|规范|合法|正向)"
)
_REFUSAL_OPENER = re.compile(
    r"^\s*(很抱歉|抱歉|对不起|我无法|我不能|你所提供的内容|您所提供的内容|作为(一个)?(AI|人工智能))"
)
_REFUSAL_ENGLISH = re.compile(
    r"(I (?:cannot|can't|can not|am unable to) (?:translate|assist|help)|"
    r"I'm sorry, but I|As an AI)",
    re.IGNORECASE,
)
_REFUSAL_OPERATION = re.compile(
    r"^\s*(?:(?:很抱歉|抱歉|对不起)[，,]\s*)?我(?:无法|不能|不会)[^。！？\n]{0,30}(?:执行|响应)"
)
_REFUSAL_ADVICE = re.compile(
    r"(?:请遵守|请通过合法|请提出|如果你有其他|任何要求.*不被允许|"
    r"请[^。！？\n]{0,35}(?:法律法规|安全规范|合规|平台规则))"
)
#: 拒答文本都很短（一段致歉）。超过此长度不再按拒答判定，避免在长文档
#: 译文里误伤"违反法律法规"这类正常表述。
_REFUSAL_MAX_CHARS = 400

#: 被拒答后的拆句重试上限。拒答常由整段里的某个组合触发，拆开后各句
#: 通常都能正常译出（实测：整句被拒 3/3 的段落，拆成两句后 2/2 通过）；
#: 但拆句会放大请求数，因此只对短段生效并且封顶。
MAX_REFUSAL_SPLIT_PARTS = 4
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。！？])\s+")

#: 分隔符残片标记。模型偶发把用户提示词连同分隔符一起复读并残缺化
#: （实测屏幕出现 "Lverify_i<<SEG>>>"——少一个尖括号的 `<<<SEG>>>`），
#: 精确切分对这种变体无效，必须在输出侧按子串拦截。
_SEGMENT_RESIDUE = "<SEG"

#: 确定性 UI 短词译名（仅 zh 目标、整词精确匹配、含术语占位符不短路）。
#: 1.7B 对独立成段的 "OK" 在精简/完整两种提示词下均稳定回显（探针
#: 30 个常见 UI 词仅此一例，compact 8/8、full 4/4 复现），语言守卫只能
#: 按设计保留原文——屏幕上 "OK" 按钮保持原样，等于每帧白付两次注定
#: 失败的推理（~200-280ms）。"Redo" 则是另一类：compact 提示词下
#: 稳定错译成"重新"（8/8），完整提示词才能译对"重做"——直接收编
#: 保证全路径一致的标准 UI 译名。首次命中即入缓存，后续帧走缓存快路径。
_UI_TOKEN_FALLBACKS_ZH = {
    "ok": "确定",
    "redo": "重做",
}

#: "保留原文"判定的重试抑制窗口（秒）。段落最终被判"无可行译文"（语言
#: 守卫拦下回显且批量纠错失败、拒答/术语恢复失败等）后，同键段落在窗口
#: 内不再送模型：保留原文渲染的结果不会入缓存（入缓存会把失败固化），
#: 而屏幕内容帧间高度重复——无抑制时同一段每帧重复整轮注定失败的推理
#: （第 75 轮实机忙屏 6-8s/帧的另一半根因）。窗口过期自动恢复重试；
#: 显式清除缓存时一并清空。
_WITHHELD_RETRY_SECONDS = 60.0


def _wrong_script(text: str, target_language: str, source: str = "") -> bool:
    """Reject foreign prose while preserving unchanged source technical tokens."""
    return has_foreign_prose(text, target_language, source)


def _has_foreign_residue(text: str, target_language: str) -> bool:
    """Check every foreign script before skipping a same-language paragraph."""
    return has_foreign_prose(text, target_language, text)


def _split_sentences(text: str) -> list[str]:
    """按句末标点切句（保留标点）。切不动或切得过碎时返回空列表。"""
    parts = [part.strip() for part in _SENTENCE_SPLIT.split(text.strip()) if part.strip()]
    if not (2 <= len(parts) <= MAX_REFUSAL_SPLIT_PARTS):
        return []
    return parts if all(len(part) >= 8 for part in parts) else []


def _looks_like_refusal(text: str) -> bool:
    """判断模型是否把"拒绝作答"当成译文返回（保守判定，宁可漏判）。"""
    if not text or len(text) > _REFUSAL_MAX_CHARS:
        return False
    if _REFUSAL_ENGLISH.search(text):
        return True
    # Some chat-tuned models treat a quoted desktop command as an instruction
    # and refuse to *execute* it. No word for "translation" appears in that
    # response, so the older refusal predicate missed it entirely.
    if _REFUSAL_OPERATION.search(text) and _REFUSAL_ADVICE.search(text):
        return True
    if _REFUSAL_VERB.search(text) is None:
        return False
    return _REFUSAL_POLICY.search(text) is not None or _REFUSAL_OPENER.search(text) is not None


def _estimate_source_tokens(text: str) -> int:
    """保守估计源文本的 token 数。

    CJK 按 1 字符 1 token（BPE 最坏情况）；其余文本按 3 字符 1 token
    （英文实际约 4 字符/token，取 3 保留余量）。
    """
    cjk = len(_CJK_CHAR_RE.findall(text))
    return cjk + (len(text) - cjk) // 3


def _output_cap(
    text: str,
    model_max_tokens: int,
    preserve_source: bool,
    segments: int = 1,
) -> int:
    """按源文本长度收紧单次请求的输出 token 上限。

    桌面翻译的输入绝大多数是短语和短句，但请求一直按模型上限
    （fast 384 / accurate 768）放开——模型一旦复读或跑飞，最坏延迟
    与输入无关、恒为上限值。正常译文长度与源文本成比例，因此
    ``min(模型上限, 2.5×源 token + 32)`` 对正常输出零影响，只封顶
    病态输出。短于下限的按 :data:`_OUTPUT_CAP_FLOOR` 兜底。

    :param segments: 本请求包含的片段数。批量响应要在段间复述分隔符，
        按 :data:`_SEGMENT_DELIMITER_TOKEN_ALLOWANCE` 逐个计入，防止
        塞满短标签的帧被上限截断（截断 → 对位失败 → 逐条串行重试）。
    """
    return max(_OUTPUT_CAP_FLOOR, min(
        int(model_max_tokens), _estimated_output_tokens(text, preserve_source, segments)
    ))


def _estimated_output_tokens(text: str, preserve_source: bool, segments: int = 1) -> int:
    """Unclipped output allowance, also used before packing a dense request."""
    cap = int(_estimate_source_tokens(text) * _OUTPUT_CAP_RATIO) + _OUTPUT_CAP_MARGIN
    cap += max(0, segments - 1) * _SEGMENT_DELIMITER_TOKEN_ALLOWANCE
    if preserve_source:
        cap *= 2
    return max(_OUTPUT_CAP_FLOOR, cap)


@dataclass
class TranslationStats:
    """一次翻译任务的统计，用于状态面板与性能压测。"""

    paragraphs: int = 0
    translated: int = 0
    from_cache: int = 0
    skipped: int = 0
    batches: int = 0
    failed_batches: int = 0
    elapsed: float = 0.0
    model_id: str = ""
    schedule_reason: str = ""
    #: 术语命中数量
    glossary_hits: int = 0
    reviewed: int = 0
    uncertain: int = 0
    generation_elapsed: float = 0.0
    review_elapsed: float = 0.0

    @property
    def cache_ratio(self) -> float:
        total = self.translated or 1
        return self.from_cache / total

    def as_text(self) -> str:
        return (
            f"{self.paragraphs} 段 / 译 {self.translated} / 缓存 {self.from_cache} "
            f"/ 跳过 {self.skipped} / 不确定 {self.uncertain} · {self.elapsed * 1000:.0f}ms"
        )


@dataclass
class _ReviewedPending:
    paragraph: Paragraph
    key: str
    generation_text: str
    replacements: dict[str, str]
    candidate: str = ""
    result: ReviewedTranslation | None = None
    reason: str = "translation_failed"
    compact_candidate: str = ""


@dataclass
class _TranslationRequest:
    """State shared by one translation and its online worker threads."""

    config: object
    glossary: Glossary
    live_glossary: Glossary
    revision: int
    namespace: str
    api_active: bool
    cache_entries: dict[str, str] = field(default_factory=dict)
    clear_cache: bool = False
    discarded_cache_keys: set[str] = field(default_factory=set)
    cache_identity: str | None = None
    stats: TranslationStats | None = None
    output: list[tuple[str, str]] | None = None
    history: tuple[list[RenderItem], str, TranslationStats] | None = None
    language_recovery_used: bool = False
    refusal_recovery_used: bool = False
    refusal_recoveries: int = 0
    defer_frame_recovery: bool = False
    retry_withheld: bool = False
    recovery_lock: threading.Lock = field(default_factory=threading.Lock)
    protected_terms: dict[str, tuple[str, ...]] = field(default_factory=dict)
    #: 帧过期回调（由 pipeline 按帧注入）：返回 True 表示该帧已被取代。
    frame_stale: Callable[[], bool] | None = None
    #: 本帧已消耗的纠错批预算（见 MAX_FRAME_CORRECTION_BATCHES）。
    correction_batches: int = 0
    #: 本帧被质量守卫**主动拒绝**的段文本（语言守卫纠错失败、拒答恢复
    #: 失败、纠错预算耗尽）。与"引擎失败/解析失败"的 ``None`` 区分：
    #: 前者是确定性失败（同输入必然复现），入抑制表防止每帧重复整轮
    #: 注定失败的推理；后者是瞬时失败，下一帧重试语义不变。
    rejected_texts: set[str] = field(default_factory=set)


class TranslationService:
    """翻译业务逻辑。

    :param config: :class:`~pingyitong.config.schema.AppConfig` 配置对象。
        **注意不是** ``SettingsManager``——本类只需读取配置值，
        配置变更后由 :class:`~pingyitong.ui.app.Application` 通过
        :meth:`apply_config` 推送新的配置对象。
    """

    def __init__(
        self,
        config: object,
        model_pool: ModelPool,
        scheduler: ModelScheduler,
        cache: TranslationCache,
        preprocessor: TextPreprocessor,
        glossary: Glossary,
        history: HistoryService | None = None,
    ) -> None:
        self._config = config
        self._pool = model_pool
        self._scheduler = scheduler
        self._cache = cache
        self._preprocessor = preprocessor
        self._glossary = glossary
        self._history = history
        self._last_stats = TranslationStats()
        #: 最近一次翻译的原文/译文，供"一键复制"使用
        self._last_output: list[tuple[str, str]] = []
        #: 接缝 A：持久预翻译缓存查询（默认 None = 未接入，行为等价现状）
        self._prefetch: object | None = None
        self._request_lock = threading.Lock()
        self._request_revision = 0
        self._request_context: ContextVar[_TranslationRequest | None] = ContextVar(
            f"translation_request_{id(self)}", default=None
        )
        #: 历史短窗口去重的 LRU（source_text → 上次写入时刻），见
        #: :meth:`_record_history`。
        self._history_seen: OrderedDict[str, float] = OrderedDict()
        self._history_seen_lock = threading.Lock()
        #: "保留原文"重试抑制表（cache_key → 过期时刻，monotonic），
        #: 见 :data:`_WITHHELD_RETRY_SECONDS`。帧线程与在线纠错线程都会
        #: 读写，须持锁。
        self._withheld_keys: dict[str, float] = {}
        self._withheld_lock = threading.Lock()

    # ------------------------------------------------------------------
    # 配置
    # ------------------------------------------------------------------

    def apply_config(self, config: object) -> None:
        """应用最新配置。"""
        with self._request_lock:
            self._config = config
            self._request_revision += 1

    def set_glossary(self, glossary: Glossary) -> None:
        """替换术语库（用户导入新术语表后调用）。

        新实例的 version 可能与旧实例相同，必须同时清空推理缓存。
        """
        with self._request_lock:
            self._glossary = glossary
            self._request_revision += 1
            self._cache.clear()

    def clear_output(self) -> None:
        """Forget copyable output and prevent an in-flight request restoring it."""
        with self._request_lock:
            self._request_revision += 1
            self._last_output = []
            self._last_stats = TranslationStats()

    @property
    def _request_config(self) -> object:
        request = self._request_context.get()
        return request.config if request is not None else self._config

    @property
    def _request_glossary(self) -> Glossary:
        request = self._request_context.get()
        return request.glossary if request is not None else self._glossary

    def _live_namespace(self) -> str:
        if getattr(self._config.translation, "semantic_review", True):
            return "local-reviewed"
        return getattr(self._pool, "translation_cache_namespace", lambda: "local")()

    def _api_mode_active(self) -> bool:
        request = self._request_context.get()
        if request is not None:
            return request.api_active
        return bool(getattr(self._pool, "api_mode_active", lambda: False)())

    def _frame_is_stale(self) -> bool:
        """当前翻译帧是否已被更新的帧取代。

        pipeline 在捕获时给帧打上过期回调；新帧到来（或用户操作使
        ``invalidate()`` 推进代数）后，继续为旧帧追加纠错请求只会
        拖慢新帧的响应，因此每轮纠错/重试前都要先问一次。
        """
        request = self._request_context.get()
        stale = request.frame_stale if request is not None else None
        if stale is None:
            return False
        try:
            return bool(stale())
        except Exception:  # noqa: BLE001 - 过期检查自身失败不能中断纠错
            logger.debug("帧过期检查失败，按未过期处理", exc_info=True)
            return False

    def _acquire_correction_batches(self, wanted: int) -> int:
        """申请纠错批预算，返回实际获批的批数（0 表示预算已耗尽）。

        预算按请求作用域记账（一帧一次翻译请求），并发纠错线程通过
        ``recovery_lock`` 共享。没有请求作用域时（测试直接调用内部
        方法）不做限制。
        """
        request = self._request_context.get()
        if request is None or wanted <= 0:
            return wanted
        with request.recovery_lock:
            remaining = MAX_FRAME_CORRECTION_BATCHES - request.correction_batches
            granted = min(wanted, max(0, remaining))
            request.correction_batches += granted
            return granted

    @contextmanager
    def _request_scope(self):
        # Only capture/commit holds this lock. OCR and network/model calls never
        # do, so a settings change can invalidate a blocked request immediately.
        with self._request_lock:
            glossary = copy(self._glossary)
            # set_entries replaces its lists, but clears the language-pair cache
            # in place. Give this request a separate dictionary for that cache.
            target = self._config.translation.target_language
            # An old unscoped English->Chinese custom glossary must not inject
            # Chinese when the user selects Korean or another target language.
            glossary._compiled = [
                (entry, pattern) for entry, pattern in self._glossary._compiled
                if self._explicit_term(entry, target)
                or not has_foreign_prose(entry.target, target, entry.source)
            ]
            glossary._pair_cache = {}
            request = _TranslationRequest(
                config=deepcopy(self._config),
                glossary=glossary,
                live_glossary=self._glossary,
                revision=self._request_revision,
                namespace=self._live_namespace(),
                api_active=bool(getattr(self._pool, "api_mode_active", lambda: False)()),
            )
        token = self._request_context.set(request)
        try:
            yield
            self._commit_request(request)
        finally:
            self._request_context.reset(token)

    def _request_is_current(self, request: _TranslationRequest) -> bool:
        # Called under _request_lock; checking the value also covers callers
        # that mutate an AppConfig before passing that same object to us.
        return (
            request.revision == self._request_revision
            and request.config == self._config
            and request.live_glossary is self._glossary
            and request.glossary.version == self._glossary.version
            and request.namespace == self._live_namespace()
        )

    def _request_superseded(self, request: _TranslationRequest) -> bool:
        """请求是否已被设置/术语表变更取代（乱序保护，需整体丢弃）。

        只看修订号、配置与术语表，不含命名空间：在线帧自身失败会推进
        命名空间代数，那是本帧产生的回退结果，不属于"被取代"。
        """
        return (
            request.revision != self._request_revision
            or request.config != self._config
            or request.live_glossary is not self._glossary
            or request.glossary.version != self._glossary.version
        )

    def _commit_request(self, request: _TranslationRequest) -> None:
        with self._request_lock:
            if self._request_superseded(request):
                return
            if getattr(request.config.translation, "semantic_review", True) and self._frame_is_stale():
                return
            # 缓存条目只在命名空间仍与当前一致时写入：在线帧内失败会推进
            # 命名空间代数，本帧的本地回退结果写入旧命名空间将永远无法被
            # 在线帧命中，写入新命名空间则会让下一帧跳过在线重试——两条
            # 路都会污染在线缓存，故整体不入缓存。
            cacheable = request.namespace == self._live_namespace()
            if cacheable and request.clear_cache:
                self._cache.clear()
                with self._withheld_lock:
                    self._withheld_keys.clear()
            if cacheable:
                for key in request.discarded_cache_keys:
                    self._cache.discard(key)
                for key, value in request.cache_entries.items():
                    self._cache.put(key, value)
            # 统计/输出/历史如实记录本帧：屏幕渲染的就是这份结果，
            # 剪贴板复制与历史查看器必须与屏幕保持一致。
            if request.stats is not None:
                self._last_stats = request.stats
            if request.output is not None:
                self._last_output = request.output

        if request.history is not None:
            items, model_id, stats = request.history
            self._record_history(items, model_id, stats, request=request)

    def _cache_put(self, key: str, value: str) -> None:
        request = self._request_context.get()
        if request is None:
            self._cache.put(key, value)
        else:
            request.cache_entries[key] = value

    def _cache_clear(self) -> None:
        request = self._request_context.get()
        if request is None:
            self._cache.clear()
        else:
            request.clear_cache = True

    def _cache_discard(self, key: str) -> None:
        request = self._request_context.get()
        if request is None:
            self._cache.discard(key)
        else:
            request.discarded_cache_keys.add(key)
            request.cache_entries.pop(key, None)

    def _suppress_withheld(self, cache_key: str) -> None:
        """记录"保留原文"判定：窗口期内同键段落不再送模型。"""
        expiry = time.monotonic() + _WITHHELD_RETRY_SECONDS
        with self._withheld_lock:
            self._withheld_keys[cache_key] = expiry

    def _is_withheld(self, cache_key: str) -> bool:
        now = time.monotonic()
        with self._withheld_lock:
            expiry = self._withheld_keys.get(cache_key)
            if expiry is None:
                return False
            if expiry > now:
                return True
            # 惰性清理过期项，避免长期运行下字典无限增长
            del self._withheld_keys[cache_key]
            return False

    def _publish_output(
        self, stats: TranslationStats, output: list[tuple[str, str]] | None = None
    ) -> None:
        request = self._request_context.get()
        if request is None:
            self._last_stats = stats
            if output is not None:
                self._last_output = output
        else:
            request.stats = stats
            request.output = output

    @property
    def last_stats(self) -> TranslationStats:
        """最近一次翻译统计。"""
        return self._last_stats

    @property
    def cache(self) -> TranslationCache:
        """推理缓存实例。"""
        return self._cache

    @property
    def glossary(self) -> Glossary:
        """当前术语库。"""
        return self._glossary

    def last_output(self) -> list[tuple[str, str]]:
        """返回最近一次翻译的 ``[(原文, 译文), ...]``。"""
        with self._request_lock:
            return list(self._last_output)

    def last_output_text(self) -> str:
        """返回最近一次翻译的纯文本，供剪贴板使用。"""
        return "\n".join(target for _, target in self.last_output())

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    def translate_frame(
        self,
        frame: RecognizedFrame,
        snapshot: ResourceSnapshot | None = None,
        cpu_sustained_high: bool = False,
    ) -> RenderFrame:
        """Translate a frame using one consistent request snapshot."""
        with self._request_scope():
            # pipeline 用帧对象携带过期回调（不改本方法签名，见
            # core/pipeline._translate_loop），挂到请求上供纠错阶梯查询。
            stale = frame.__dict__.get("_frame_stale")
            request = self._request_context.get()
            if request is not None and callable(stale):
                request.frame_stale = stale
            if request is not None:
                request.retry_withheld = bool(frame.__dict__.get("_translation_retry", False))
            return self._translate_frame_request(frame, snapshot, cpu_sustained_high)

    def _translate_frame_request(
        self,
        frame: RecognizedFrame,
        snapshot: ResourceSnapshot | None = None,
        cpu_sustained_high: bool = False,
    ) -> RenderFrame:
        """把一帧 OCR 结果翻译为渲染指令。

        :param frame: OCR 结果。
        :param snapshot: 当前资源快照，供调度器决策。
        :param cpu_sustained_high: CPU 是否持续高负载。
        :return: :class:`RenderFrame`。
        """
        if getattr(self._request_config.translation, "semantic_review", True):
            return self._translate_reviewed_frame(frame, snapshot, cpu_sustained_high)
        started = time.perf_counter()
        stats = TranslationStats()

        # 把来源显示器序号一并传下去：渲染阶段按它把译文分发到各块屏幕，
        # 缺失时条目会被整体丢弃（见 OverlayManager.apply）
        preprocess_started = time.perf_counter()
        paragraphs, preprocess_stats = self._preprocessor.process(
            frame.blocks, monitor_index=frame.monitor_index
        )
        preprocess_cost = time.perf_counter() - preprocess_started
        active_model = self._scheduler.current_model

        if not paragraphs:
            stats.model_id = active_model
            elapsed = time.perf_counter() - started
            stats.elapsed = elapsed
            self._publish_output(stats)
            return RenderFrame(
                seq=frame.seq,
                origin=frame.origin,
                items=[],
                is_full_frame=frame.is_full_frame,
                changed_regions=list(getattr(frame, "changed_regions", [])),
                source_items=[],
                retry_regions=[],
                timings={
                    "preprocess": preprocess_cost,
                    "routing": max(0.0, elapsed - preprocess_cost),
                    "inference": 0.0,
                    "translate": elapsed,
                },
            )

        stats.paragraphs = len(paragraphs)

        # 分流提前到调度之前：先跳过无需翻译的段落并完成术语替换，
        # 调度器的内容自适应才有**真实的**术语命中数可用——
        # 之前在调度时硬编码 0（术语替换发生在分流里，当时还没算），
        # "术语密度高 → 使用高精度模型"的分支因此永远不会触发。
        routing_started = time.perf_counter()
        pending, items, output_pairs, glossary_hits = self._prepare_paragraphs(
            paragraphs, stats
        )
        stats.glossary_hits = glossary_hits
        routing_cost = time.perf_counter() - routing_started

        inference_cost = 0.0
        if pending:
            schedule_started = time.perf_counter()
            decision = self._scheduler.decide(
                snapshot=snapshot,
                texts=[paragraph.text for paragraph, _, _ in pending],
                glossary_term_count=glossary_hits,
                cpu_sustained_high=cpu_sustained_high,
            )
            active_model = decision.model_id
            stats.model_id = active_model
            stats.schedule_reason = decision.reason
            routing_cost += time.perf_counter() - schedule_started

            inference_started = time.perf_counter()
            batch_items, batch_pairs = self._translate_pending(
                pending, active_model, stats
            )
            inference_cost = time.perf_counter() - inference_started
            items.extend(batch_items)
            output_pairs.extend(batch_pairs)

        elapsed = time.perf_counter() - started
        stats.elapsed = elapsed
        self._publish_output(stats, output_pairs)

        self._write_history(items, active_model, stats)

        # Keep the actual OCR source alongside successful translations. A
        # transient failure must not erase a previous translation of unchanged
        # text, and the capture pipeline needs explicit regions to retry even
        # when the screen's pixels have stopped changing.
        produced = {(item.box, item.source_text) for item in items}
        source_items = [self._build_item(paragraph, paragraph.text, False) for paragraph in paragraphs]
        retry_regions = [
            paragraph.box for paragraph in paragraphs
            if paragraph.text.strip() and (paragraph.box, paragraph.text) not in produced
        ]

        logger.debug(
            "翻译完成：%s（模型 %s，%s）",
            stats.as_text(),
            MODEL_LABELS.get(active_model, active_model),
            preprocess_stats.as_text(),
        )

        result = RenderFrame(
            seq=frame.seq,
            origin=frame.origin,
            items=items,
            is_full_frame=frame.is_full_frame,
            changed_regions=list(getattr(frame, "changed_regions", [])),
            source_items=source_items,
            retry_regions=retry_regions,
            timings={
                "preprocess": preprocess_cost,
                "routing": routing_cost,
                "inference": inference_cost,
                "translate": elapsed,
            },
        )
        return self._attach_processing_decisions(result, paragraphs, preprocess_stats)

    # ------------------------------------------------------------------
    # 段落翻译
    # ------------------------------------------------------------------

    def _translate_reviewed_frame(
        self, frame: RecognizedFrame, snapshot: ResourceSnapshot | None,
        cpu_sustained_high: bool,
    ) -> RenderFrame:
        started = time.perf_counter()
        paragraphs, preprocess_stats = self._preprocessor.process(frame.blocks, monitor_index=frame.monitor_index)
        preprocess_cost = time.perf_counter() - started
        stats = TranslationStats(paragraphs=len(paragraphs))
        items = self._translate_reviewed_units(paragraphs, stats, snapshot, cpu_sustained_high)
        stats.elapsed = time.perf_counter() - started
        self._publish_output(stats, [(item.source_text, item.text) for item in items])
        self._write_history(
            [item for item in items if item.quality_status == "accepted"], stats.model_id, stats,
        )
        # Uncertain items explicitly retain the source. They must not disappear
        # or invite the old automatic correction ladder on every static frame.
        result = RenderFrame(
            seq=frame.seq, origin=frame.origin, items=items,
            is_full_frame=frame.is_full_frame,
            changed_regions=list(getattr(frame, "changed_regions", [])),
            source_items=[self._build_item(paragraph, paragraph.text, False) for paragraph in paragraphs],
            retry_regions=[],
            diagnostics={
                **getattr(frame, "diagnostics", {}),
                "filtered": preprocess_stats.noise_removed,
                "duplicates_removed": preprocess_stats.duplicate_removed,
                "source_unchanged": stats.skipped,
                "review_skipped": sum(item.quality_reason == "review_skipped_by_policy" for item in items),
                "accepted": stats.translated,
                "review_rejected": sum(
                    item.quality_status == "uncertain" and item.quality_reason.startswith(("review_", "semantic_"))
                    for item in items
                ),
                "translation_failed": sum(
                    item.quality_status == "uncertain" and not item.quality_reason.startswith(("review_", "semantic_"))
                    for item in items
                ),
                "trusted_cache_reuse": stats.from_cache,
            },
            timings={
                "preprocess": preprocess_cost,
                "routing": max(0.0, stats.elapsed - preprocess_cost - stats.generation_elapsed - stats.review_elapsed),
                "inference": stats.generation_elapsed,
                "semantic_review": stats.review_elapsed,
                "translate": stats.elapsed,
            },
        )
        return self._attach_processing_decisions(result, paragraphs, preprocess_stats)

    @staticmethod
    def _attach_processing_decisions(frame, paragraphs, preprocess_stats):
        """Retain source/decision provenance in memory; only test homes persist it."""
        frame.__dict__["_diagnostic_events"] = [
            *getattr(preprocess_stats, "decision_events", []),
            *({"stage": "translation", "action": "reject" if item.quality_status == "uncertain" else "deliver",
               "reason": item.quality_reason or "accepted_full_text",
               "inputs": [{"text": item.source_text,
                           "rect": [item.box.x, item.box.y, item.box.width, item.box.height]}],
               "quality_status": item.quality_status}
              for item in frame.items),
        ]
        frame.__dict__["_paragraph_provenance"] = [
            {"text": paragraph.text,
             "rect": [paragraph.box.x, paragraph.box.y, paragraph.box.width, paragraph.box.height],
             "source_lines": list(getattr(paragraph, "source_lines", []))}
            for paragraph in paragraphs
        ]
        return frame

    def _translate_reviewed_units(
        self, paragraphs: list[Paragraph], stats: TranslationStats,
        snapshot: ResourceSnapshot | None = None, cpu_sustained_high: bool = False,
    ) -> list[RenderItem]:
        """The single publish boundary for text, screen, cached and fresh results.

        Only the current paragraph supplies context. There is no mutable chat
        history or previous-frame prompt; identical independently scoped units
        may share one reviewed result but retain all original screen positions.
        """
        translation = self._request_config.translation
        source = translation.source_language
        target = translation.target_language
        fallback = translation.fallback_source_language
        translatable = [
            paragraph for paragraph in paragraphs
            if paragraph.text.strip() and self._needs_translation(paragraph, source, target, fallback)
        ]
        if not translatable:
            stats.skipped = len(paragraphs)
            return [self._build_item(paragraph, paragraph.text, False) for paragraph in paragraphs]
        decision = self._scheduler.decide(
            snapshot=snapshot, texts=[paragraph.text for paragraph in translatable],
            glossary_term_count=0, cpu_sustained_high=cpu_sustained_high,
        )
        model_id = decision.model_id
        # A second independent request uses the selected local model. This is
        # never an implicit switch to another model or an external provider.
        stats.model_id = model_id
        stats.schedule_reason = decision.reason
        unit_by_key: dict[str, _ReviewedPending] = {}
        paragraph_units: dict[int, _ReviewedPending] = {}
        cached_keys: set[str] = set()
        for paragraph in translatable:
            detected = source if source != "auto" else detect_language(paragraph.text, fallback).code
            key = f"{self._cache_key(paragraph.text, detected, target)}|review-model:{model_id}"
            existing = unit_by_key.get(key)
            if existing is not None:
                paragraph_units[id(paragraph)] = existing
                continue
            substitution = self._substitute(paragraph.text, detected, target)
            stats.glossary_hits += substitution.hit_count
            unit = _ReviewedPending(paragraph, key, substitution.text, substitution.replacements)
            unit_by_key[key] = unit
            paragraph_units[id(paragraph)] = unit
            cached = self._cache.get_candidate(key)
            evidence = ReviewedTranslation.from_json(cached, paragraph.text) if cached is not None else None
            valid = bool(
                evidence is not None
                and evidence.generation_model == model_id and evidence.review_model == model_id
                and self._review_evidence_valid(paragraph.text, evidence, substitution.replacements)
            )
            self._cache.record_verified_lookup(valid, rejected=cached is not None and not valid)
            if valid:
                unit.result = evidence
                cached_keys.add(key)
            elif cached is not None:
                self._cache_discard(key)
            if unit.result is None and self._is_withheld(key):
                unit.reason = "review_rejected_recently"

        pending = [
            unit for unit in unit_by_key.values()
            if unit.result is None and unit.reason != "review_rejected_recently"
        ]
        local_translate = getattr(self._pool, "translate_local", None)
        if not callable(local_translate):
            for unit in pending:
                unit.reason = "local_model_unavailable"
        else:
            self._generate_review_candidates(pending, model_id, stats)
            if self._review_policy == "risk":
                # 问答 #9 的分级可靠性：含否定/条件/数字/高危动作词的段落
                # 必须模型二审；低风险段落由确定性守卫（生成层的结构、
                # 语言、数字、术语完整性检查）把关后直接发布。小字 UI
                # 文案大多低风险——这是延迟过大与"整批扣留只显示原文"
                # 两类问题的共同出口。
                risky: list[_ReviewedPending] = []
                for unit in pending:
                    if not unit.candidate:
                        continue
                    if self._is_low_risk_for_review(unit.paragraph.text, unit.candidate):
                        self._publish_deterministic(unit, model_id)
                    else:
                        risky.append(unit)
                self._review_candidates(risky, model_id, stats)
            else:
                self._review_candidates([unit for unit in pending if unit.candidate], model_id, stats)

        items = []
        for paragraph in paragraphs:
            unit = paragraph_units.get(id(paragraph))
            if unit is None:
                stats.skipped += 1
                items.append(self._build_item(paragraph, paragraph.text, False))
                continue
            evidence = unit.result
            item = self._build_item(paragraph, evidence.full_text if evidence else paragraph.text, unit.key in cached_keys)
            item.ocr_uncertain = paragraph.confidence < 0.65 or any(
                getattr(block, "ocr_uncertain", False) for block in paragraph.blocks
            )
            if evidence is not None:
                stats.translated += 1
                stats.from_cache += int(item.from_cache)
                stats.reviewed += int(not item.from_cache and evidence.review_model != "deterministic")
                if evidence.review_model == "deterministic":
                    item.quality_reason = "review_skipped_by_policy"
                item.display_text = evidence.display_text
            else:
                stats.uncertain += 1
                item.quality_status = "uncertain"
                item.quality_reason = unit.reason
            items.append(item)
        return items

    def _review_evidence_valid(
        self, source: str, evidence: ReviewedTranslation, replacements: dict[str, str],
    ) -> bool:
        """Recheck non-model invariants when reading evidence from disk."""
        if source_digest(source) != evidence.source_hash or not evidence.evidence_consistent():
            return False
        if any(
            not self._valid_output(source, text) or self._unresolved_terms(text, list(replacements), replacements)
            for text in (evidence.candidate, evidence.independent, evidence.full_text)
        ):
            return False
        return not (evidence.display_text and (
            not self._valid_output(source, evidence.display_text)
            or self._unresolved_terms(evidence.display_text, list(replacements), replacements)
            or len(evidence.display_text) >= len(evidence.full_text)
        ))

    def _local_infer(self, model_id: str, prompt: str, system_prompt: str, max_tokens: int) -> str:
        if self._frame_is_stale():
            return ""
        try:
            return self._pool.translate_local(
                model_id=model_id, prompt=prompt, system_prompt=system_prompt, max_tokens=max_tokens,
            ) or ""
        except Exception:  # noqa: BLE001 - errors become visible uncertainty, never an API fallback
            logger.warning("本地翻译或语义审查失败，保留原文", exc_info=True)
            return ""

    def _generate_review_candidates(self, units: list[_ReviewedPending], model_id: str, stats: TranslationStats) -> None:
        from ..components.llm.prompt import build_compact_prompt

        config = self._request_config.model.for_model(model_id)
        system = build_compact_prompt(
            self._request_config.translation.target_language,
            self._request_config.translation.source_language, concise=False,
        )
        # The first pass is always complete. Display shortening belongs to the
        # second pass and can never overwrite this canonical translation.
        system += (
            f"\nPreserve {SEGMENT_DELIMITER} separators exactly, including count and order. "
            "Translate each segment independently. Preserve every [TERM_n] placeholder."
        )
        max_output = max(48, int(config.max_tokens))
        context = max(512, int(config.n_ctx))

        def budget(batch: list[_ReviewedPending]) -> tuple[str, int, bool]:
            text = build_user_prompt([unit.generation_text for unit in batch])
            output = max(48, int(_estimate_source_tokens(text) * _OUTPUT_CAP_RATIO) + 32 + 16 * (len(batch) - 1))
            needed = _estimate_source_tokens(text) + len(system) // 3 + _PROMPT_TOKEN_MARGIN + output
            return text, output, output <= max_output and needed <= context

        batches: list[list[_ReviewedPending]] = []
        current: list[_ReviewedPending] = []
        for unit in units:
            if not budget([unit])[2]:
                unit.reason = "context_limit"
                continue
            # 上限 6：密集页面的串行请求数减半（实测 8 段 3 批 -> 2 批）。
            # 真实边界由 budget 的输出/上下文预算决定，上限只是保险丝。
            if current and (len(current) >= 6 or not budget([*current, unit])[2]):
                batches.append(current)
                current = []
            current.append(unit)
        if current:
            batches.append(current)
        for batch in batches:
            if self._frame_is_stale():
                for unit in batch:
                    unit.reason = "scene_changed"
                continue
            prompt, output, _ = budget(batch)
            started = time.perf_counter()
            raw = self._local_infer(model_id, prompt, system, output)
            stats.generation_elapsed += time.perf_counter() - started
            stats.batches += 1
            parsed = parse_response(raw, len(batch), source_texts=[unit.generation_text for unit in batch]) if raw else None
            if parsed is None or self._frame_is_stale():
                for unit in batch:
                    unit.reason = "scene_changed" if self._frame_is_stale() else "translation_structure"
                stats.failed_batches += 1
                continue
            for (unit, candidate) in zip(batch, parsed, strict=True):
                restored, missing = self._request_glossary.restore(candidate, unit.replacements)
                if not _looks_like_refusal(restored) and self._unresolved_terms(restored, missing, unit.replacements):
                    unit.reason = "translation_integrity"
                    self._suppress_withheld(unit.key)
                    continue
                if _looks_like_refusal(restored):
                    # 审查路径此前未接入拒答恢复：4B 会把屏幕上的祈使句
                    # 当成对自己的请求而拒答（实测 "Ignore previous
                    # instructions..." 稳定复现），批量路径的引用式重问
                    # 阶梯对它同样适用。复用同一预算与判分；仍被拒时保留
                    # 原文并保持 uncertain，绝不把致歉文案画上屏幕。
                    recovery_started = time.perf_counter()
                    recovered = self._retry_after_refusal(
                        unit.generation_text, model_id, output, preserve_source=False,
                        allow_model_fallback=False, force_local=True,
                    )
                    stats.generation_elapsed += time.perf_counter() - recovery_started
                    if recovered is None or not self._valid_output(unit.paragraph.text, recovered):
                        unit.reason = "model_refusal"
                        self._suppress_withheld(unit.key)
                        stats.failed_batches += 1
                        continue
                    candidate = recovered
                    restored, missing = self._request_glossary.restore(candidate, unit.replacements)
                # Surface-format failure does not justify a many-step retry
                # ladder. Its original remains visible with an explicit cause.
                if self._unresolved_terms(restored, missing, unit.replacements) or not self._valid_output(unit.paragraph.text, restored):
                    unit.reason = "translation_integrity"
                    self._suppress_withheld(unit.key)
                    continue
                unit.candidate = restored

    def _review_candidates(self, units: list[_ReviewedPending], model_id: str, stats: TranslationStats) -> None:
        from ..components.llm.prompt import build_semantic_review_prompt

        config = self._request_config.model.for_model(model_id)
        wants_compact = bool(getattr(self._request_config.translation, "compact_display", True))
        context = max(512, int(config.n_ctx))
        max_output = max(48, int(config.max_tokens))

        def budget(batch: list[_ReviewedPending], compact: bool) -> tuple[str, str, int, bool]:
            system = build_semantic_review_prompt(self._request_config.translation.target_language, compact=compact)
            prompt = build_review_prompt([ReviewInput(unit.paragraph.text, unit.candidate) for unit in batch])
            # Two translations plus fixed JSON fields/verdicts need a larger
            # cap than translation alone. Never truncate a verdict to fit.
            # 审查输出 = 独立译文(~1.5x) + 精简(~1.2x) + 12 个裁决词 +
            # JSON 语法开销。旧估算(2x+92)实测在批尾被 max_tokens 截断，
            # 整批 3 条 review_invalid 且白付约 2 秒；按实测形态重估。
            # 批内条数由预算自约束，此处只抬高单条开销上限。
            output = sum(
                (3 if compact else 2) * max(_estimate_source_tokens(unit.candidate), 20)
                + (60 if compact else 55) for unit in batch
            )
            needed = _estimate_source_tokens(prompt) + len(system) // 3 + _PROMPT_TOKEN_MARGIN + output
            return prompt, system, output, output <= max_output and needed <= context

        batches: list[tuple[list[_ReviewedPending], bool]] = []
        current: list[_ReviewedPending] = []
        current_compact = wants_compact
        for unit in units:
            compact = wants_compact
            if not budget([unit], compact)[3]:
                compact = False
            if not budget([unit], compact)[3]:
                unit.reason = "review_context_limit"
                continue
            if current and (compact != current_compact or len(current) >= 6 or not budget([*current, unit], compact)[3]):
                batches.append((current, current_compact))
                current = []
            current_compact = compact
            current.append(unit)
        if current:
            batches.append((current, current_compact))
        conflicts: list[tuple[_ReviewedPending, ReviewDecision, bool]] = []
        for batch, compact in batches:
            if self._frame_is_stale():
                for unit in batch:
                    unit.reason = "scene_changed"
                continue
            prompt, system, output, _ = budget(batch, compact)
            started = time.perf_counter()
            raw = self._local_infer(model_id, prompt, system, output)
            stats.review_elapsed += time.perf_counter() - started
            if self._frame_is_stale():
                for unit in batch:
                    unit.reason = "scene_changed"
                continue
            decisions = parse_reviews(raw, len(batch))
            for unit, decision in zip(batch, decisions, strict=True):
                if not decision.accepted or any(
                    not self._valid_output(unit.paragraph.text, text)
                    or self._unresolved_terms(text, list(unit.replacements), unit.replacements)
                    for text in (unit.candidate, decision.independent)
                ):
                    unit.reason = decision.reason or "review_independent_integrity"
                    self._suppress_withheld(unit.key)
                    continue
                if not translations_agree(unit.candidate, decision.independent):
                    conflicts.append((unit, decision, compact))
                    continue
                self._publish_review_evidence(unit, decision, model_id, compact)
        self._resolve_review_conflicts(conflicts, model_id, stats)
        self._review_compact_candidates(units, model_id, stats)

    def _publish_review_evidence(
        self, unit: _ReviewedPending, decision: ReviewDecision, model_id: str, compact: bool,
        resolution: ResolutionDecision | None = None,
    ) -> None:
        if self._frame_is_stale():
            unit.reason = "scene_changed"
            return
        choice = resolution.choice if resolution is not None else "agreement"
        full = decision.independent if choice == "B" else unit.candidate
        # Pass two only proposes a compact for A. No self-rating can publish
        # it, and this proposal is never transferable to a selected B.
        proposal = decision.compact if compact and choice != "B" else ""
        if proposal and (
            len(proposal) >= len(full) or not self._valid_output(unit.paragraph.text, proposal)
            or self._unresolved_terms(proposal, list(unit.replacements), unit.replacements)
        ):
            proposal = ""
        source_hash = source_digest(unit.paragraph.text)
        resolution_checks = resolution.checks if resolution is not None else ()
        binding = resolution_digest(
            source_hash, unit.candidate, decision.independent, choice, resolution_checks, model_id,
        ) if resolution is not None else ""
        evidence = ReviewedTranslation(
            source_hash=source_hash, full_text=full, display_text="",
            independent=decision.independent, generation_model=model_id, review_model=model_id,
            checks=decision.checks,
            candidate=unit.candidate, selection=choice, resolution_checks=resolution_checks,
            resolution_model=model_id if resolution is not None else "", resolution_binding=binding,
        )
        if not self._review_evidence_valid(unit.paragraph.text, evidence, unit.replacements):
            unit.reason = "review_evidence_integrity"
            self._suppress_withheld(unit.key)
            return
        unit.result = evidence
        unit.compact_candidate = proposal
        unit.reason = ""
        self._cache_put(unit.key, evidence.to_json())

    def _resolve_review_conflicts(
        self, conflicts: list[tuple[_ReviewedPending, ReviewDecision, bool]],
        model_id: str, stats: TranslationStats,
    ) -> None:
        """At most one explicit third audit per conflict, within the frame budget.

        A rejected/malformed second review is never eligible for this recovery.
        The chosen local model may share mistakes across passes; a resolver
        verdict is recorded evidence, not a guarantee of semantic correctness.
        """
        if not conflicts:
            return
        from ..components.llm.prompt import build_semantic_resolution_prompt

        config = self._request_config.model.for_model(model_id)
        system = build_semantic_resolution_prompt(self._request_config.translation.target_language)
        context, max_output = max(512, int(config.n_ctx)), max(48, int(config.max_tokens))

        def budget(batch: list[tuple[_ReviewedPending, ReviewDecision, bool]]) -> tuple[str, int, bool]:
            prompt = build_resolution_prompt([
                ResolutionInput(unit.paragraph.text, unit.candidate, decision.independent)
                for unit, decision, _compact in batch
            ])
            output = 48 * len(batch) + 16
            needed = _estimate_source_tokens(prompt) + len(system) // 3 + _PROMPT_TOKEN_MARGIN + output
            return prompt, output, output <= max_output and needed <= context

        batches: list[list[tuple[_ReviewedPending, ReviewDecision, bool]]] = []
        current: list[tuple[_ReviewedPending, ReviewDecision, bool]] = []
        for entry in conflicts:
            if not budget([entry])[2]:
                entry[0].reason = "review_resolution_context_limit"
                self._suppress_withheld(entry[0].key)
                continue
            if current and (len(current) >= 6 or not budget([*current, entry])[2]):
                batches.append(current)
                current = []
            current.append(entry)
        if current:
            batches.append(current)
        granted = self._acquire_correction_batches(len(batches))
        for batch_index, batch in enumerate(batches):
            if self._frame_is_stale():
                for unit, _decision, _compact in batch:
                    unit.reason = "scene_changed"
                continue
            if batch_index >= granted:
                for unit, _decision, _compact in batch:
                    unit.reason = "review_resolution_budget"
                    self._suppress_withheld(unit.key)
                continue
            prompt, output, _ = budget(batch)
            started = time.perf_counter()
            raw = self._local_infer(model_id, prompt, system, output)
            stats.review_elapsed += time.perf_counter() - started
            if self._frame_is_stale():
                for unit, _decision, _compact in batch:
                    unit.reason = "scene_changed"
                continue
            resolutions = parse_resolutions(raw, len(batch))
            for (unit, decision, compact), resolution in zip(batch, resolutions, strict=True):
                if not resolution.accepted:
                    unit.reason = resolution.reason
                    self._suppress_withheld(unit.key)
                    continue
                self._publish_review_evidence(unit, decision, model_id, compact, resolution)

    def _review_compact_candidates(
        self, units: list[_ReviewedPending], model_id: str, stats: TranslationStats,
    ) -> None:
        """Approve B only explicitly; uncertainty retains full, neither revokes it.

        Only a generated compact belonging to the accepted full candidate is
        eligible. This stage shares the correction ledger with full resolution;
        optional compaction failure retains full unless both texts are explicitly
        rejected by a valid neither verdict.
        """
        pending = [unit for unit in units if unit.result is not None and unit.compact_candidate]
        if not pending:
            return
        from ..components.llm.prompt import build_semantic_compact_review_prompt

        config = self._request_config.model.for_model(model_id)
        system = build_semantic_compact_review_prompt(self._request_config.translation.target_language)
        context, max_output = max(512, int(config.n_ctx)), max(48, int(config.max_tokens))

        def budget(batch: list[_ReviewedPending]) -> tuple[str, int, bool]:
            prompt = build_resolution_prompt([
                ResolutionInput(unit.paragraph.text, unit.result.full_text, unit.compact_candidate)
                for unit in batch
            ])
            output = 48 * len(batch) + 16
            needed = _estimate_source_tokens(prompt) + len(system) // 3 + _PROMPT_TOKEN_MARGIN + output
            return prompt, output, output <= max_output and needed <= context

        def discard_stale() -> bool:
            if not self._frame_is_stale():
                return False
            for unit in units:
                unit.result = None
                unit.reason = "scene_changed"
            return True

        batches: list[list[_ReviewedPending]] = []
        current: list[_ReviewedPending] = []
        for unit in pending:
            if not budget([unit])[2]:
                continue  # the reviewed full result remains publishable
            if current and (len(current) >= 6 or not budget([*current, unit])[2]):
                batches.append(current)
                current = []
            current.append(unit)
        if current:
            batches.append(current)
        granted = self._acquire_correction_batches(len(batches))
        for batch in batches[:granted]:
            if discard_stale():
                return
            prompt, output, _ = budget(batch)
            started = time.perf_counter()
            raw = self._local_infer(model_id, prompt, system, output)
            stats.review_elapsed += time.perf_counter() - started
            if discard_stale():
                return
            decisions = parse_resolutions(raw, len(batch))
            for unit, decision in zip(batch, decisions, strict=True):
                if decision.choice == "neither":
                    unit.result = None
                    unit.reason = "semantic_full_rejected_by_compact_review"
                    self._cache_discard(unit.key)
                    self._suppress_withheld(unit.key)
                    continue
                if decision.choice != "B" or not decision.accepted:
                    continue
                full = unit.result
                evidence = replace(
                    full, display_text=unit.compact_candidate, compact_checks=decision.checks,
                    compact_review_choice="B", compact_review_model=model_id,
                    compact_review_binding=compact_review_digest(
                        full.source_hash, full.full_text, unit.compact_candidate, decision.checks, model_id,
                    ),
                )
                if not self._review_evidence_valid(unit.paragraph.text, evidence, unit.replacements):
                    continue
                unit.result = evidence
                self._cache_put(unit.key, evidence.to_json())

    _REVIEW_RISK_RE = re.compile(
        r"\d"
        r"|(?<![A-Za-z])(?:not|no|never|none|without|until|unless|before|after|only"
        r"|must|should|shall|warn|warning|caution|danger|delete|remove|disable|enable"
        r"|stop|start|fail|failed|don't|doesn't|didn't|isn't|aren't|won't|can't|cannot)(?![A-Za-z])"
        r"|[\u4e00-\u9fff]",
        re.IGNORECASE,
    )

    @property
    def _review_policy(self) -> str:
        policy = getattr(self._request_config.translation, "review_policy", "always")
        return str(policy or "always").strip().lower()

    def _is_low_risk_for_review(self, source: str, candidate: str) -> bool:
        """风险分级（问答 #9）：否定/条件/时间/数字/高危动作词必须二审。

        源文含 CJK 字符一律视为高风险：中文源文里的否定（不/没/未）、
        条件（如果/除非/前/后）与数字的形态过于多样，保守全审。
        """
        if candidate.strip() == source.strip():
            return False
        if self._REVIEW_RISK_RE.search(source):
            return False
        # 生成层已过术语与完整性守卫；这里再防英文骨架残留，
        # 保证"跳过二审直接发布"的段落不会把混杂译文画上屏。
        return not self._has_english_scaffolding(source, candidate)

    def _publish_deterministic(self, unit: _ReviewedPending, model_id: str) -> None:
        """低风险段直接发布。

        证据对象仅存在于本次请求内存中，**不写任何缓存**——未经过模型
        二审的译文不得冒充可信缓存复用（问答 #15）。display_text 留空，
        原位排版退回完整译文 + 字号收缩，文字栏始终有完整数据。
        """
        unit.result = ReviewedTranslation(
            source_hash=source_digest(unit.paragraph.text),
            full_text=unit.candidate,
            display_text="",
            independent=unit.candidate,
            generation_model=model_id,
            review_model="deterministic",
        )
        unit.reason = ""

    @staticmethod
    def _explicit_term(entry, target_language: str) -> bool:
        """Respect declared invariant identifiers and explicit language pairs."""
        return (
            entry.source == entry.target
            or bool(entry.source_language and entry.target_language == target_language)
        )

    def _protected_target_terms(self, original: str) -> tuple[str, ...]:
        request = self._request_context.get()
        if request is not None and original in request.protected_terms:
            return request.protected_terms[original]
        translation = self._request_config.translation
        if not getattr(translation, "glossary_enabled", True):
            return ()
        source = translation.source_language
        if source == "auto":
            source = detect_language(original, translation.fallback_source_language).code
        target = translation.target_language
        terms = tuple(dict.fromkeys(
            entry.target for entry, pattern in self._request_glossary._compiled
            if self._explicit_term(entry, target)
            and (not entry.source_language or entry.source_language == source)
            and (not entry.target_language or entry.target_language == target)
            and pattern.search(original)
        ))
        if request is not None:
            request.protected_terms[original] = terms
        return terms

    def _valid_output(self, original: str, translated: str) -> bool:
        """One quality gate for inferred, restored and cached translations."""
        return bool(translated.strip()) and not (
            has_foreign_prose(
                translated, self._request_config.translation.target_language, original,
                self._protected_target_terms(original),
            )
            or _SEGMENT_RESIDUE in translated
            or _looks_like_refusal(translated)
            or self._has_english_scaffolding(original, translated)
            or has_reversed_until_condition(original, translated, self._request_config.translation.target_language)
            or _missing_numerals(original, translated)
        )

    def _prepare_paragraphs(
        self,
        paragraphs: list[Paragraph],
        stats: TranslationStats,
    ) -> tuple[
        list[tuple[Paragraph, str, dict[str, str]]],
        list[RenderItem],
        list[tuple[str, str]],
        int,
    ]:
        """分流：跳过无需翻译的段落，其余做术语替换并建立缓存键。

        :return: ``(待翻译段落, 渲染条目, 输出对, 术语命中数)``。
            渲染条目已包含"无需翻译而原样保留"的段落；待翻译段落的
            顺序与输入一致，缓存键与术语占位符一并带出。
        """
        translation_config = self._request_config.translation
        source_language = translation_config.source_language
        target_language = translation_config.target_language
        fallback_source = translation_config.fallback_source_language

        items: list[RenderItem] = []
        output_pairs: list[tuple[str, str]] = []
        pending: list[tuple[Paragraph, str, dict[str, str]]] = []
        glossary_hits = 0

        for paragraph in paragraphs:
            if not paragraph.text.strip():
                stats.skipped += 1
                continue

            if not self._needs_translation(paragraph, source_language, target_language, fallback_source):
                stats.skipped += 1
                items.append(
                    self._build_item(paragraph, paragraph.text, from_cache=False)
                )
                continue

            detected_source = (
                source_language if source_language != "auto"
                else paragraph.language or detect_language(paragraph.text, fallback_source).code
            )
            substitution = self._substitute(paragraph.text, detected_source, target_language)
            glossary_hits += substitution.hit_count
            cache_key = self._cache_key(
                substitution.text, detected_source, target_language
            )

            request = self._request_context.get()
            if self._is_withheld(cache_key) and not (request and request.retry_withheld):
                # 近期已被判"无可行译文"（保留原文）：窗口内不再送模型，
                # 也不生成条目——与被扣留时的呈现一致（原屏文字保持可见）。
                stats.skipped += 1
                continue

            cached = self._cache.get(cache_key)
            if cached is not None:
                restored, missing = self._request_glossary.restore(cached, substitution.replacements)
                if (
                    not self._unresolved_terms(restored, missing, substitution.replacements)
                    and self._valid_output(paragraph.text, restored)
                ):
                    stats.translated += 1
                    stats.from_cache += 1
                    items.append(self._build_item(paragraph, restored, from_cache=True))
                    output_pairs.append((paragraph.text, restored))
                    continue
                # Old/incomplete entries must not repeatedly suppress a term.
                self._cache_discard(cache_key)

            # 接缝 A（联网预翻译）：内存缓存 miss 后查一次持久预翻译缓存。
            # 未配置/未开启/异常一律返回 None，等价现状（见 prefetch/lookup）。
            # 源语言用段落检测值：预翻译条目按检测语言入库，
            # 用配置值（常为 auto）会查不到键。
            prefetch_hit = self._prefetch_lookup(
                substitution.text,
                detected_source,
                target_language,
            )
            if prefetch_hit is not None:
                restored, missing = self._request_glossary.restore(prefetch_hit, substitution.replacements)
                if (
                    not self._unresolved_terms(restored, missing, substitution.replacements)
                    and self._valid_output(paragraph.text, restored)
                ):
                    stats.translated += 1
                    stats.from_cache += 1
                    self._cache_put(cache_key, prefetch_hit)
                    items.append(self._build_item(paragraph, restored, from_cache=True))
                    output_pairs.append((paragraph.text, restored))
                    continue

            # 确定性 UI 短词短路（精度+延迟）：模型对这类词必然回显，
            # 送进去只会白付推理再被语言守卫退回原文。首帧直接映射标准
            # 译名并写入缓存，后续帧由上方缓存快路径接管。
            if target_language == "zh" and not substitution.replacements:
                fallback = _UI_TOKEN_FALLBACKS_ZH.get(substitution.text.strip().casefold())
                if fallback is not None:
                    self._cache_put(cache_key, fallback)
                    stats.translated += 1
                    items.append(self._build_item(paragraph, fallback, from_cache=False))
                    output_pairs.append((paragraph.text, fallback))
                    continue

            pending.append((paragraph, cache_key, substitution.replacements))
            paragraph.glossary_text = substitution.text

        return pending, items, output_pairs, glossary_hits

    def _prefetch_lookup(self, text: str, source_language: str, target_language: str) -> str | None:
        """持久预翻译缓存查询（接缝 A）。

        lookup 对象由应用层注入（``set_prefetch_lookup``）；这里只依赖
        :class:`~pingyitong.services.prefetch.lookup.PrefetchLookup` 协议，
        不 import 字幕/网页实现模块。
        """
        # Persistent prefetch entries are not scoped to the current external
        # provider or Key.  They must not intercept its first real request.
        if self._api_mode_active() or getattr(
            self._request_config.translation, "concise_output", False
        ):
            return None
        lookup = getattr(self, "_prefetch", None)
        if lookup is None:
            return None
        try:
            from .prefetch.lookup import LookupContext

            return lookup.lookup(
                text, LookupContext(target_lang=target_language, src_lang=source_language)
            )
        except Exception:  # noqa: BLE001 - 接缝异常等价 miss
            logger.debug("预翻译缓存查询失败，按 miss 处理", exc_info=True)
            return None

    def set_prefetch_lookup(self, lookup: object) -> None:
        """注入预翻译缓存查询（应用启动时装配；传 None 即关闭接缝）。"""
        self._prefetch = lookup

    def _translate_pending(
        self,
        pending: list[tuple[Paragraph, str, dict[str, str]]],
        model_id: str,
        stats: TranslationStats,
    ) -> tuple[list[RenderItem], list[tuple[str, str]]]:
        """批量翻译待处理段落并组装渲染条目。"""
        translation_config = self._request_config.translation
        model_config = self._request_config.model

        original_position = {id(item[0]): index for index, item in enumerate(pending)}
        unique: list[tuple[Paragraph, str, dict[str, str]]] = []
        first_by_key: dict[tuple[str, tuple[tuple[str, str], ...]], Paragraph] = {}
        duplicates: dict[int, list[Paragraph]] = {}
        for paragraph, key, replacements in pending:
            # Identical screen labels at different coordinates need one model
            # result. Include the restoration map so independently assigned
            # glossary placeholders cannot alias different protected terms.
            identity = (key, tuple(sorted(replacements.items())))
            first = first_by_key.get(identity)
            if first is None:
                first_by_key[identity] = paragraph
                unique.append((paragraph, key, replacements))
            else:
                duplicates.setdefault(id(first), []).append(paragraph)
        pending = unique

        source_language = translation_config.source_language
        target_language = translation_config.target_language
        preserve_source = bool(getattr(translation_config, "keep_source_text", False))

        items: list[RenderItem] = []
        output_pairs: list[tuple[str, str]] = []
        # 条目按原始段落顺序回填（装批前可能重排过外文段优先），见下方排序。
        assembled: list[tuple[int, RenderItem, tuple[str, str]]] = []

        concise = bool(getattr(translation_config, "concise_output", False))
        from ..components.llm.prompt import build_compact_prompt

        glossary_tokens = sorted({
            token for _, _, replacements in pending for token in replacements
        })
        full_prompt = build_system_prompt(
            source_language=source_language,
            target_language=target_language,
            glossary_terms=glossary_tokens,
            # RenderItem.source_text already stores the original. Asking the
            # model to repeat it adds ambiguous pipes/markers and extra tokens.
            # Keep the flag only for defensive parsing of provider responses.
            preserve_source=False,
            concise=concise,
        )
        compact_prompt = build_compact_prompt(target_language, source_language, concise=concise)

        batch_size = max(1, int(getattr(model_config, "batch_size", 6)))
        max_tokens = getattr(model_config.for_model(model_id), "max_tokens", 512)
        # 按较长的完整提示词装批，避免低估上下文占用。
        char_budget = self._input_char_budget(model_id, full_prompt, max_tokens)

        # 在线 AI 模式：云端模型上下文长（≥8k），单批可携带更多片段以减少
        # 请求往返；输入预算同步放大但仍设上限——单批过大只会拉长单请求
        # 耗时，并不会让总吞吐更高。若云端在本帧中途失效（冷却触发），
        # 池会逐请求回退本地模型，超出本地上下文的批次将走逐条重试兜底，
        # 结果仍正确，只是该帧更慢——这是罕见路径，接受该代价。
        api_config = getattr(self._request_config, "api", None)
        api_active = bool(
            api_config is not None
            and getattr(api_config, "enabled", False)
            and self._api_mode_active()
        )
        if api_active:
            batch_size = max(1, int(getattr(api_config, "batch_size", 10)))
            # 输出上限改由动态估计主导（见 _API_OUTPUT_HARD_LIMIT）：
            # 云端上下文长，正常译文不应被本地模型的输出量级截断
            max_tokens = _API_OUTPUT_HARD_LIMIT

            char_budget = max(char_budget, _API_INPUT_CHAR_BUDGET)
        elif model_id == MODEL_ACCURATE:
            # 本地 4B 单请求条数上限（见 MAX_ACCURATE_BATCH_ITEMS）。
            batch_size = min(batch_size, MAX_ACCURATE_BATCH_ITEMS)
        recoveries = 0

        # 实机根因（probe_mixed_batch：实机 8 段批 5/5 复现）：同一批里
        # 同语言段在前、外文段在后时，1.7B 会顺着前面"原样返回"的复读
        # 惯性把外文段也原样回显，每帧因此多付一次批量纠错请求。把外文段
        # （检测语言≠目标语言或含外文夹杂）整体排到同语言段之前再装批：
        # 任何批内外文段都先于同语言段出现，复读惯性没有机会吞掉它们；
        # 同语言段照旧送模型（繁简/日文汉字歧义仍由模型裁决），提示词与
        # 纠错路径均不变。实测探针 D/E 两种排布均 0 坏段，原顺序 10/40 坏段。
        foreign: list[tuple[Paragraph, str, dict[str, str]]] = []
        same_language: list[tuple[Paragraph, str, dict[str, str]]] = []
        for item in pending:
            text = item[0].glossary_text or item[0].text
            detected = detect_language(
                text, translation_config.fallback_source_language
            ).code
            if detected != target_language or _has_foreign_residue(
                text, target_language
            ):
                foreign.append(item)
            else:
                same_language.append(item)
        if foreign and same_language:
            pending = foreign + same_language

        batches = _pack_batches(pending, batch_size, char_budget, max_output_tokens=max_tokens)
        dispatch_batches: list[list[tuple[Paragraph, str, dict[str, str]]]] = []
        dispatch_prompts: list[str] = []
        for batch in batches:
            # 装批后可能剩下单独一段。仅当这一段没有术语占位符、对照或
            # 精简模式时，才省去批量规则，避免额外 prefill 与错误约束。
            # 在线模式除外：云端 prefill 廉价（单请求耗时由网络往返主导），
            # 精简提示词少掉的"UI 标准译法 / 已译回退 / 无寒暄"三条规则
            # 正是外部模型该赚的质量分；且术语单段本来就走完整提示词，
            # 冷却回退本地时也不会遇到没走过的路径。
            single_plain = (
                len(batch) == 1
                and not batch[0][2]
                and not preserve_source
                and not api_active
            )
            prompt = compact_prompt if single_plain else full_prompt
            dispatch_batches.append(batch)
            dispatch_prompts.append(prompt)
        batches, batch_prompts = dispatch_batches, dispatch_prompts

        request = self._request_context.get()
        # 并发分发（在线多批、本地拆分、本地超批拆包）时，语言/拒答纠错
        # 必须收集完整帧所有坏段再统一执行：否则先完成的工作线程会独占
        # 每帧一次的纠错预算，把其它可恢复的段落挤成"保留原文"。
        defer_recovery = request is not None and (api_active or len(batches) > 1)
        if defer_recovery:
            request.defer_frame_recovery = True
        try:
            if api_active and len(batches) > 1:
                batch_results = self._translate_batches_concurrently(
                    batches, batch_prompts, model_id, max_tokens, preserve_source,
                    max_concurrency=int(getattr(api_config, "max_concurrency", 4)),
                )
            else:
                batch_results = [
                    self._translate_batch(
                        batch=batch,
                        model_id=model_id,
                        system_prompt=system_prompt,
                        max_tokens=max_tokens,
                        preserve_source=preserve_source,
                    )
                    for batch, system_prompt in zip(batches, batch_prompts, strict=True)
                ]
        finally:
            if defer_recovery:
                request.defer_frame_recovery = False

        if defer_recovery:
            # Gather every worker's bad segments before spending the frame's
            # corrective budget. A worker must not consume it before another
            # batch arrives, leaving validly recoverable text untranslated.
            texts = [p.glossary_text or p.text for batch in batches for p, _, _ in batch]
            flat = [result for results in batch_results for result in results]
            flat = self._fix_refused_segments(flat, texts, model_id, max_tokens, preserve_source)
            flat = self._fix_wrong_language_segments(
                flat, texts, model_id, full_prompt, max_tokens, preserve_source
            )
            offset = 0
            for index, batch in enumerate(batches):
                batch_results[index] = flat[offset:offset + len(batch)]
                offset += len(batch)

        stats.batches += len(batches)
        for results in batch_results:
            if any(result is None for result in results):
                stats.failed_batches += 1

        for batch, results in zip(batches, batch_results, strict=True):
            for (paragraph, cache_key, replacements), translated in zip(batch, results, strict=True):
                if translated is None:
                    # Leave the real source screen visible. Drawing the source
                    # again falsely presented it as a target-language result.
                    # 仅"守卫主动拒绝"（确定性失败）入抑制表；引擎/解析失败
                    # 是瞬时的，下一帧照常重试。
                    if (
                        request is not None
                        and (paragraph.glossary_text or paragraph.text) in request.rejected_texts
                    ):
                        self._suppress_withheld(cache_key)
                    stats.skipped += 1 + len(duplicates.get(id(paragraph), ()))
                    continue

                restored, missing = self._request_glossary.restore(translated, replacements)
                unresolved = self._unresolved_terms(restored, missing, replacements)
                mixed = not self._valid_output(paragraph.text, restored)
                recovered = None
                if unresolved or mixed:
                    if recoveries < MAX_GLOSSARY_RECOVERIES_PER_FRAME:
                        recoveries += 1
                        recovered = self._retry_without_placeholders(
                            paragraph.text,
                            list(dict.fromkeys(replacements.values())) if mixed else unresolved,
                            model_id, max_tokens,
                            preserve_source, concise,
                        )
                        restored = recovered if recovered is not None else ""
                    else:
                        restored = ""
                    logger.warning(
                        "模型输出异常（缺失术语 %s 个，英文残留 %s），已%s",
                        len(unresolved), mixed,
                        "重新翻译" if restored else "保留原屏文字",
                    )

                if not self._valid_output(paragraph.text, restored):
                    # "输出语言不可行"类失败（外来散文/分隔符残片）是确定性
                    # 的——模型对同语言文本只会回显，纠错也救不回——入抑制
                    # 表；术语缺失类失败可能是采样波动，下一帧照常重试
                    # （既有语义，见 test_dropped_placeholder_never_caches）。
                    if (
                        has_foreign_prose(
                            restored, target_language, paragraph.text,
                            self._protected_target_terms(paragraph.text),
                        )
                        or _SEGMENT_RESIDUE in restored
                    ):
                        self._suppress_withheld(cache_key)
                    stats.skipped += 1 + len(duplicates.get(id(paragraph), ()))
                    continue

                stats.translated += 1
                if not unresolved and not mixed:
                    self._cache_put(cache_key, translated)
                elif recovered is not None:
                    # A successful literal retry is safe to reuse: cache lookup
                    # also verifies that all required target terms are present.
                    self._cache_put(cache_key, restored)
                assembled.append((
                    original_position[id(paragraph)],
                    self._build_item(paragraph, restored, from_cache=False),
                    (paragraph.text, restored),
                ))
                for duplicate in duplicates.get(id(paragraph), ()):
                    stats.translated += 1
                    assembled.append((
                        original_position[id(duplicate)],
                        self._build_item(duplicate, restored, from_cache=False),
                        (duplicate.text, restored),
                    ))

        # 装批重排只影响请求内的段落顺序；条目按原始 OCR 顺序回填，
        # 文字栏语序（reading_order 之外的直接消费方）与历史顺序不变。
        assembled.sort(key=lambda entry: entry[0])
        items.extend(entry[1] for entry in assembled)
        output_pairs.extend(entry[2] for entry in assembled)
        return items, output_pairs

    @staticmethod
    def _unresolved_terms(
        restored: str, missing: list[str], replacements: dict[str, str]
    ) -> list[str]:
        """Return target terms genuinely absent, not merely tokens omitted.

        Models sometimes output the correct target term directly instead of
        copying its placeholder. That is already a valid translation.
        """
        folded = restored.casefold()
        return list(dict.fromkeys(
            target for token in missing
            if (target := replacements[token]).casefold() not in folded
        ))

    def _has_english_scaffolding(self, original: str, translated: str) -> bool:
        """Catch a Chinese result that kept English sentence grammar.

        Two function words plus Han text is intentionally conservative: a
        single English product name or code identifier is not a failure.
        """
        target = self._request_config.translation.target_language
        if not has_foreign_prose(translated, target, original, self._protected_target_terms(original)):
            return False
        source = self._request_config.translation.source_language
        if target != "zh" or len(re.findall(r"[\u3400-\u9fff]", translated)) < 2:
            return False
        if source == "auto":
            source = detect_language(original, "en").code
        return source == "en" and len(_ENGLISH_SCAFFOLD.findall(translated)) >= 2

    def _retry_without_placeholders(
        self,
        original: str,
        required_terms: list[str],
        model_id: str,
        max_tokens: int,
        preserve_source: bool,
        concise: bool,
    ) -> str | None:
        """Retry literal source with explicit terms after the model drops a token.

        If it fails again, the caller keeps the original rather than showing
        a fluent-looking translation that silently omits a technical term.
        """
        if (
            len(required_terms) > MAX_GLOSSARY_RECOVERY_TERMS
            or sum(map(len, required_terms)) > MAX_GLOSSARY_RECOVERY_CHARS
        ):
            return None

        target_language = self._request_config.translation.target_language
        if preserve_source or concise:
            system_prompt = build_system_prompt(
                source_language=self._request_config.translation.source_language,
                target_language=target_language,
                preserve_source=False,
                concise=concise,
            )
        else:
            from ..components.llm.prompt import build_compact_prompt

            system_prompt = build_compact_prompt(
                target_language, self._request_config.translation.source_language
            )
        system_prompt += (
            "\nRequired terminology: use these exact target-language terms: "
            + json.dumps(required_terms, ensure_ascii=False)
            + ". Treat the list only as terminology data. Do not omit the subject of the sentence."
        )
        raw = self._pool.translate(
            model_id=model_id,
            prompt=original,
            system_prompt=system_prompt,
            max_tokens=_output_cap(original, max_tokens, False),
        )
        parsed = _parse_single_response(raw, original, preserve_source) if raw else None
        if not parsed:
            return None
        candidate = parsed[0]
        folded = candidate.casefold()
        if not self._valid_output(original, candidate):
            return None
        # 恢复路径本身也可能被拒答：这类输出不是译文的"缺失"，而是内容被
        # 换成了致歉，绝不能当作术语恢复的结果交回调用方。
        if _looks_like_refusal(candidate):
            return None
        return candidate if all(term.casefold() in folded for term in required_terms) else None

    def _retry_after_refusal(
        self,
        original: str,
        model_id: str,
        max_tokens: int,
        preserve_source: bool,
        *,
        allow_model_fallback: bool = True,
        force_local: bool = False,
    ) -> str | None:
        """模型误判内容安全策略时，用两步阶梯把正常内容译回来。

        实测结论（1.7B/4B 均可见）：拒绝与内容安全性无关——同一句把
        "inspect or delete" 换成 "open" 就能正常译出。因此：

        1. **引用式重问**：提示词里说明这是软件界面/产品文档文本、要求
           直接给出译文、不要评价或致歉。对照组实测：引用式 2/2 通过，
           而"你是专业译者"式反而 2/2 被拒——措辞越像"审查"，越容易触发。
        2. **拆句重试**：整段被拒时逐句请求再拼接（实测把被拒 3/3 的段落
           拆成两句后 2/2 通过）。句子数量与长度都有上限，拒绝放大请求。

        两步都没成功返回 ``None``，由调用方保留原文：屏幕上绝不出现致歉文案。
        """
        if self._frame_is_stale():
            logger.debug("帧已过期，跳过拒答重问")
            return None
        candidate = self._request_neutral(original, model_id, max_tokens, preserve_source, force_local=force_local)
        if candidate is not None:
            return candidate

        # A larger chat model can be more likely than the fast model to treat
        # a quoted UI command as an action request. Use the already-resident
        # fast model only for this confirmed refusal, without loading weights
        # or changing the user's selected model for future requests.
        fallback = self._ready_fast_fallback_model(model_id) if allow_model_fallback else None
        if fallback is not None:
            candidate = self._request_neutral(
                original, fallback,
                self._request_config.model.for_model(fallback).max_tokens,
                preserve_source, force_local=force_local,
            )
            if candidate is not None:
                return candidate

        parts = _split_sentences(original) if not preserve_source else []
        if not parts:
            return None
        logger.debug("整段重问仍被拒答，拆成 %s 句分别翻译", len(parts))
        pieces: list[str] = []
        for part in parts:
            if self._frame_is_stale():
                # 拆句是串行请求阶梯：过期即中止，已有句子结果作废。
                logger.debug("帧已过期，中止拆句重试")
                return None
            piece = self._request_neutral(part, model_id, max_tokens, preserve_source, force_local=force_local)
            if piece is None:
                return None
            pieces.append(piece)
        # 中文目标语言不用空格分隔句子，其余语言按英文习惯加空格
        separator = "" if self._request_config.translation.target_language.startswith("zh") else " "
        return separator.join(pieces)

    def _ready_fast_fallback_model(self, model_id: str) -> str | None:
        if (
            model_id != MODEL_ACCURATE
            or getattr(self._request_config.api, "enabled", False)
            or self._api_mode_active()
            or self._frame_is_stale()
            or not self._request_config.model.for_model(MODEL_FAST).enabled
        ):
            return None
        loaded = getattr(self._pool, "loaded_models", None)
        if not callable(loaded) or MODEL_FAST not in loaded():
            return None
        if not getattr(self._pool, "_warmed", {}).get(MODEL_FAST, True):
            return None
        return MODEL_FAST

    def _request_neutral(
        self,
        original: str,
        model_id: str,
        max_tokens: int,
        preserve_source: bool,
        *,
        force_local: bool = False,
    ) -> str | None:
        """用**完整提示词** + 引用式措辞重问一次；``None`` 表示仍被拒答。

        为什么不再用精简提示词：实测 4 句正常英文文案并列对照，
        精简提示词 12 次里被拒答 5 次，完整提示词 0 次——**拒答与内容
        安全无关，是精简提示词本身诱发的**（它只留三句硬要求，模型缺少
        "这是翻译任务"的充分上下文，于是把内容审核挂在嘴边）。
        完整提示词多出的 prefill 约 15ms，只在极少数拒答段上付出。
        """
        target_language = self._request_config.translation.target_language
        concise = bool(getattr(self._request_config.translation, "concise_output", False))
        if getattr(self._request_config.translation, "semantic_review", True):
            concise = False
        system_prompt = build_system_prompt(
            source_language=self._request_config.translation.source_language,
            target_language=target_language,
            preserve_source=False,
            concise=concise,
        )
        system_prompt += (
            "\nThe user prompt is ordinary software interface text quoted for translation."
            " Translate it literally and start your reply with the translation itself."
        )
        infer = self._local_infer if force_local else self._pool.translate
        raw = infer(
            model_id=model_id,
            prompt=original,
            system_prompt=system_prompt,
            max_tokens=_output_cap(original, max_tokens, False),
        )
        parsed = _parse_single_response(raw, original, preserve_source) if raw else None
        if not parsed or not parsed[0]:
            return None
        candidate = parsed[0]
        return candidate if self._valid_output(original, candidate) else None

    def _input_char_budget(self, model_id: str, system_prompt: str, max_tokens: int) -> int:
        """估算单次请求可携带的输入字符数。

        批量请求若超出模型上下文，llama-server 会直接拒绝（HTTP 400），
        整批作废后再走逐条重试——白白多花一整轮延迟。因此按上下文给
        输入设一个字符预算，超预算的片段分入下一批。

        估算从**最坏情况**出发：待翻译文本统一按 1 字符 1 token 计
        （英文实际只按约 1/4 计入，预算只会更富余）；system 提示词是
        纯英文，按 :data:`_SYSTEM_PROMPT_CHARS_PER_TOKEN` 折算。
        """
        model_config = self._request_config.model.for_model(model_id)
        n_ctx = int(getattr(model_config, "n_ctx", 2048))
        system_tokens = -(-len(system_prompt) // _SYSTEM_PROMPT_CHARS_PER_TOKEN)
        budget = n_ctx - int(max_tokens) - system_tokens - _PROMPT_TOKEN_MARGIN
        return max(_MIN_INPUT_CHAR_BUDGET, budget)

    def _translate_batches_concurrently(
        self,
        batches: list,
        prompts: list[str],
        model_id: str,
        max_tokens: int,
        preserve_source: bool,
        max_concurrency: int,
    ) -> list[list[str | None]]:
        """并发执行多批翻译（在线模式多批共用）。

        在线模式下整帧多批串行发送会让网络往返时间线性叠加：以 4 批、
        单批 1 秒计，串行 4 秒，4 路并发约 1 秒——这是在线模式"整帧更快"
        的主要来源。本地多批帧走串行路径：实机混杂文本（外文夹杂、乱码段）
        上"外文段前置 + 合并批"实测比逐段拆分更稳，且免付拆分后的逐段
        纠错开销。max_workers 同时被引擎槽位信号量钳制——若本地引擎被
        其他并发调用方（预取/剪贴板翻译）同时占用，llama-server 侧按
        4 槽排队（内嵌引擎自动退化为串行）。
        批间无依赖，结果按提交顺序回填；单个批次异常只影响该批
        （按全部失败处理，调用方保留原文），不会拖垮整帧。
        """
        max_workers = max(1, min(int(max_concurrency), len(batches)))
        results: list[list[str | None]] = []
        with ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="BatchTranslate"
        ) as executor:
            futures = [
                executor.submit(
                    copy_context().run,
                    self._translate_batch,
                    batch=batch,
                    model_id=model_id,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    preserve_source=preserve_source,
                )
                for batch, system_prompt in zip(batches, prompts, strict=True)
            ]
            for future, batch in zip(futures, batches, strict=True):
                try:
                    results.append(future.result())
                except Exception:  # noqa: BLE001 - 单批异常不拖垮整帧
                    logger.warning("在线批量翻译线程异常，该批按失败处理", exc_info=True)
                    results.append([None] * len(batch))
        return results

    def _translate_batch(
        self,
        batch: list[tuple[Paragraph, str, dict[str, str]]],
        model_id: str,
        system_prompt: str,
        max_tokens: int,
        preserve_source: bool,
    ) -> list[str | None]:
        """翻译一个批次。

        批次计数与失败计数由调用方统计（并发路径中本方法在
        工作线程里执行，直接改共享计数会竞态）。

        :return: 与 ``batch`` 等长的结果列表，元素为 ``None`` 表示该条翻译失败。
        """
        paragraphs = [item[0] for item in batch]
        texts = [paragraph.glossary_text or paragraph.text for paragraph in paragraphs]

        # 单条无需分隔符，直接请求即可
        if len(texts) == 1:
            raw = self._pool.translate(
                model_id=model_id,
                prompt=texts[0],
                system_prompt=system_prompt,
                max_tokens=_output_cap(texts[0], max_tokens, False),
            )
            if not raw:
                return [None]
            parsed = _parse_single_response(raw, texts[0], preserve_source)
            if parsed is None:
                return [None]
            return self._fix_wrong_language_segments(
                self._fix_refused_segments(
                    self._fix_echoed_segments(
                        list(parsed),
                        texts,
                        model_id=model_id,
                        system_prompt=system_prompt,
                        max_tokens=max_tokens,
                        preserve_source=preserve_source,
                    ),
                    texts,
                    model_id=model_id,
                    max_tokens=max_tokens,
                    preserve_source=preserve_source,
                ),
                texts,
                model_id=model_id,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                preserve_source=preserve_source,
            )

        user_prompt = build_user_prompt(texts)

        raw = self._pool.translate(
            model_id=model_id,
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=_output_cap(
                "\n".join(texts), max_tokens, False, segments=len(texts)
            ),
        )

        if raw:
            parsed = parse_response(raw, len(texts), preserve_source, source_texts=texts)
            if parsed is not None:
                return self._fix_wrong_language_segments(
                    self._fix_refused_segments(
                        self._fix_echoed_segments(
                            list(parsed),
                            texts,
                            model_id=model_id,
                            system_prompt=system_prompt,
                            max_tokens=max_tokens,
                            preserve_source=preserve_source,
                        ),
                        texts,
                        model_id=model_id,
                        max_tokens=max_tokens,
                        preserve_source=preserve_source,
                    ),
                    texts,
                    model_id=model_id,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    preserve_source=preserve_source,
                )

            logger.debug(
                "批量翻译输出无法对位（%s 段），降级为逐条翻译", len(texts)
            )

        # 批量失败或对位失败：逐条重试，保证结果正确性优于吞吐
        return self._fix_wrong_language_segments(
            self._fix_refused_segments(
                self._retry_individually(
                    texts, model_id, system_prompt, max_tokens, preserve_source
                ),
                texts,
                model_id=model_id,
                max_tokens=max_tokens,
                preserve_source=preserve_source,
            ),
            texts,
            model_id=model_id,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            preserve_source=preserve_source,
        )

    def _fix_echoed_segments(
        self,
        results: list[str | None],
        texts: list[str],
        model_id: str,
        system_prompt: str,
        max_tokens: int,
        preserve_source: bool,
    ) -> list[str | None]:
        """Treat foreign-language echoes as one bounded language correction."""
        # Foreign-language echoes are the same failure as language drift.
        # Share its one corrective batch rather than serially retrying each.
        return self._fix_wrong_language_segments(
            results, texts, model_id, system_prompt, max_tokens, preserve_source
        )

    def _fix_refused_segments(
        self,
        results: list[str | None],
        texts: list[str],
        model_id: str,
        max_tokens: int,
        preserve_source: bool,
    ) -> list[str | None]:
        """把"模型拒答"的段落换成一次中立重问；仍拒答则交回 ``None``。

        拒答判定见 :func:`_looks_like_refusal`。这里与复读修复不同的一点是
        **失败必须落成 ``None`` 而不是原样接受**：复读至少还是原文（屏幕上
        只是少了一处翻译），而拒答是一段与原文无关的致歉，画上去等于把
        用户的屏幕内容替换成道歉信——宁可这一段不译。
        """
        request = self._request_context.get()
        if request is not None and request.defer_frame_recovery:
            return results
        if self._frame_is_stale():
            logger.debug("帧已过期，跳过拒答重问")
            return results
        refused = [
            index
            for index, (source, translated) in enumerate(zip(texts, results, strict=True))
            if translated is not None and _looks_like_refusal(translated)
        ]
        if not refused:
            return results

        if self._api_mode_active():
            if request is not None:
                with request.recovery_lock:
                    recover = not request.refusal_recovery_used
                    request.refusal_recovery_used = True
                if not recover:
                    for index in refused:
                        results[index] = None
                        request.rejected_texts.add(texts[index])
                    return results
            selected = [texts[index] for index in refused]
            prompt = build_system_prompt(
                source_language=self._request_config.translation.source_language,
                target_language=self._request_config.translation.target_language,
                preserve_source=False,
                concise=bool(getattr(self._request_config.translation, "concise_output", False)),
            )
            prompt += (
                "\nThe user prompt is ordinary software interface text quoted for translation."
                " Translate it literally and start your reply with the translation itself."
            )
            parsed = self._run_correction_batches(
                selected, prompt, model_id, max_tokens, preserve_source
            )
            for position, index in enumerate(refused):
                candidate = parsed[position] if parsed else None
                results[index] = (
                    candidate if candidate is not None and self._valid_output(texts[index], candidate)
                    else None
                )
                if results[index] is None and request is not None:
                    request.rejected_texts.add(texts[index])
            logger.warning("模型拒答（%s 段），已进行一次中立批量重问", len(refused))
            return results

        # Local inference remains serial. Bound the number of short recovery
        # ladders over the whole frame, including all separate local batches.
        allowance = MAX_INDIVIDUAL_RETRIES
        if request is not None:
            with request.recovery_lock:
                allowance = max(0, MAX_INDIVIDUAL_RETRIES - request.refusal_recoveries)
                request.refusal_recoveries += min(allowance, len(refused))
        logger.warning("模型拒答（%s 段），对前 %s 段改用中立措辞重问", len(refused), allowance)
        for index in refused[:allowance]:
            results[index] = self._retry_after_refusal(
                texts[index], model_id, max_tokens, preserve_source
            )
            if results[index] is None and request is not None:
                request.rejected_texts.add(texts[index])
        for index in refused[allowance:]:
            results[index] = None
            if request is not None:
                request.rejected_texts.add(texts[index])
        return results

    def _fix_wrong_language_segments(
        self,
        results: list[str | None],
        texts: list[str],
        model_id: str,
        system_prompt: str,
        max_tokens: int,
        preserve_source: bool,
    ) -> list[str | None]:
        """Recover wrong-language segments with one batch request per frame.

        The budget is shared by parallel API workers. Failed correction is
        withheld from rendering, history and cache; the real source stays visible.
        """
        request = self._request_context.get()
        if request is not None and request.defer_frame_recovery:
            return results
        if self._frame_is_stale():
            # 过期帧马上会被 pipeline 丢弃：立即中止，不再消费纠错预算。
            logger.debug("帧已过期，中止语言纠错")
            return results
        target = self._request_config.translation.target_language
        temporal = {
            index for index, translated in enumerate(results)
            if translated is not None and has_reversed_until_condition(texts[index], translated, target)
        }
        numeric = {
            index for index, translated in enumerate(results)
            if translated is not None and _missing_numerals(texts[index], translated)
        }
        flagged = [
            index
            for index, translated in enumerate(results)
            if translated is not None
            and (
                has_foreign_prose(translated, target, texts[index], self._protected_target_terms(texts[index]))
                or _SEGMENT_RESIDUE in translated
                or has_reversed_until_condition(texts[index], translated, target)
                or index in numeric
            )
        ]
        if not flagged:
            return results

        if request is not None:
            with request.recovery_lock:
                recover = not request.language_recovery_used
                request.language_recovery_used = True
            if not recover:
                for index in flagged:
                    results[index] = None
                    request.rejected_texts.add(texts[index])
                return results

        # One corrective request for all bad segments in this batch replaces
        # N serial requests. The request-wide budget is shared by API workers.
        retry_texts = [texts[index] for index in flagged]
        correction = (
            system_prompt
            + "\nCorrect the output language: translate every natural-language word into "
            + self._request_config.translation.target_language
            + _LANGUAGE_CORRECTION_SUFFIX
        )
        if temporal:
            correction += _TIME_CONDITION_CORRECTION_SUFFIX
        if numeric:
            correction += _NUMBER_CORRECTION_SUFFIX
        parsed = self._run_correction_batches(
            retry_texts, correction, model_id, max_tokens, preserve_source
        )
        for position, index in enumerate(flagged):
            candidate = parsed[position] if parsed else None
            results[index] = (
                candidate if candidate is not None and self._valid_output(texts[index], candidate)
                else None
            )
            if results[index] is None and request is not None:
                request.rejected_texts.add(texts[index])
        remaining = [index for index in flagged if index in temporal and results[index] is None]
        fallback = self._condition_fallback_model(model_id) if remaining else None
        if fallback is not None:
            logger.warning("时间条件纠错仍失败，仅对此异常段尝试已就绪的高精度模型")
            accurate_tokens = self._request_config.model.for_model(fallback).max_tokens
            accurate = self._run_correction_batches(
                [texts[index] for index in remaining], correction, fallback, accurate_tokens, preserve_source
            )
            for position, index in enumerate(remaining):
                candidate = accurate[position]
                if candidate is not None and self._valid_output(texts[index], candidate):
                    results[index] = candidate
                    if request is not None:
                        request.rejected_texts.discard(texts[index])
        remaining_numeric = [index for index in flagged if index in numeric and results[index] is None]
        fallback = self._ready_fast_fallback_model(model_id) if remaining_numeric else None
        if fallback is not None:
            logger.warning("数字纠错仍失败，仅对此异常段尝试已就绪的极速模型")
            fast_tokens = self._request_config.model.for_model(fallback).max_tokens
            retried = self._run_correction_batches(
                [texts[index] for index in remaining_numeric], correction,
                fallback, fast_tokens, preserve_source,
            )
            for position, index in enumerate(remaining_numeric):
                candidate = retried[position]
                if candidate is not None and self._valid_output(texts[index], candidate):
                    results[index] = candidate
                    if request is not None:
                        request.rejected_texts.discard(texts[index])
        logger.warning("模型输出语言/条件/数字异常（%s 段），已进行批量纠错", len(flagged))
        return results

    def _run_correction_batches(
        self, texts: list[str], system_prompt: str, model_id: str,
        max_tokens: int, preserve_source: bool,
    ) -> list[str | None]:
        """One corrective phase, respecting context limits and bounded RTTs.

        Normal frames fit a single request. Large online frames use at most
        four chunks concurrently; local inference keeps its context and serial
        execution. Every returned segment still passes the caller's quality gate.
        """
        online = self._api_mode_active()
        budget = (_API_INPUT_CHAR_BUDGET if online else
                  self._input_char_budget(model_id, system_prompt, max_tokens))
        groups: list[list[int]] = []
        current: list[int] = []
        size = 0
        for index, text in enumerate(texts):
            cost = len(text) + _SEGMENT_SEPARATOR_CHARS
            exceeds_output = _estimated_output_tokens(
                "\n".join([*(texts[item] for item in current), text]), False, len(current) + 1
            ) > max_tokens
            if current and (size + cost > budget or exceeds_output):
                groups.append(current)
                current, size = [], 0
            current.append(index)
            size += cost
        if current:
            groups.append(current)
        selected = groups[:MAX_INDIVIDUAL_RETRIES]
        if self._frame_is_stale():
            logger.debug("帧已过期，中止本轮纠错批")
            return [None] * len(texts)
        granted = self._acquire_correction_batches(len(selected))
        if granted < len(selected):
            dropped = sum(len(group) for group in selected[granted:])
            logger.warning(
                "单帧纠错预算（%s 批）已用尽，其余 %s 段保留原文",
                MAX_FRAME_CORRECTION_BATCHES,
                dropped,
            )
            selected = selected[:granted]
        if not selected:
            return [None] * len(texts)
        outputs: list[str | None] = [None] * len(texts)

        def run(indices: list[int]) -> list[str | None]:
            parts = [texts[index] for index in indices]
            try:
                raw = self._pool.translate(
                    model_id=model_id, prompt=build_user_prompt(parts), system_prompt=system_prompt,
                    max_tokens=_output_cap(
                        "\n".join(parts), max_tokens, False, segments=len(parts)
                    ),
                )
                parsed = (
                    _parse_single_response(raw, parts[0], preserve_source)
                    if len(parts) == 1 else parse_response(
                        raw, len(parts), preserve_source, source_texts=parts
                    )
                ) if raw else None
                return list(parsed) if parsed is not None else [None] * len(parts)
            except Exception:  # noqa: BLE001 - a correction cannot discard the good segments
                logger.warning("纠错请求失败，保留原屏文字", exc_info=True)
                return [None] * len(parts)

        workers = min(self._api_concurrency(), len(selected))
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ApiCorrection") as executor:
                futures = [executor.submit(copy_context().run, run, indices) for indices in selected]
                corrected = [future.result() for future in futures]
        else:
            corrected = [run(indices) for indices in selected]
        for indices, values in zip(selected, corrected, strict=True):
            for index, value in zip(indices, values, strict=True):
                outputs[index] = value
        if len(groups) > len(selected):
            logger.warning("纠错输入超过 %s 批预算，其余异常译文不发布", len(selected))
        return outputs

    def _retry_wrong_language_single(
        self,
        prompt: str,
        system_prompt: str,
        target_language: str,
        model_id: str,
        max_tokens: int,
        preserve_source: bool,
        condition_correction: bool = False,
        number_correction: bool = False,
    ) -> str | None:
        """对"语言错误/带残片"的输出附带纠正提示做一次单段重试。

        批量路径的语种漂移在单段请求里大多能恢复（漂移与上下文里的
        其他片段相关），但"整段未翻译"（模型把代码注释、纯英文短语
        原样吐回）与措辞漂移不同——同一提示词盲重试通常复现同样结果。
        因此与批量纠错对齐，追加"翻译所有自然语言词、只保留技术标识符"
        的纠正提示，只在失败路径生效，不增加正常翻译的 prefill。
        重试产物必须通过同一套校验，否则返回 ``None`` 由调用方保留原文。
        """
        correction = (
            system_prompt
            + "\nCorrect the output language: translate every natural-language word into "
            + target_language
            + _LANGUAGE_CORRECTION_SUFFIX
        )
        if condition_correction:
            correction += _TIME_CONDITION_CORRECTION_SUFFIX
        if number_correction:
            correction += _NUMBER_CORRECTION_SUFFIX
        raw = self._pool.translate(
            model_id=model_id,
            prompt=prompt,
            system_prompt=correction,
            max_tokens=_output_cap(prompt, max_tokens, False),
        )
        parsed = _parse_single_response(raw, prompt, preserve_source) if raw else None
        candidate = parsed[0] if parsed and parsed[0] else ""
        if not candidate or not self._valid_output(prompt, candidate):
            fallback = self._condition_fallback_model(model_id) if condition_correction else None
            if fallback is not None and self._acquire_correction_batches(1) > 0:
                logger.warning("时间条件纠错仍失败，仅对此异常段尝试已就绪的高精度模型")
                return self._retry_wrong_language_single(
                    prompt, system_prompt, target_language, fallback,
                    self._request_config.model.for_model(fallback).max_tokens,
                    preserve_source, condition_correction=True,
                    number_correction=number_correction,
                )
            fallback = self._ready_fast_fallback_model(model_id) if number_correction else None
            if fallback is not None and self._acquire_correction_batches(1) > 0:
                logger.warning("数字纠错仍失败，仅对此异常段尝试已就绪的极速模型")
                return self._retry_wrong_language_single(
                    prompt, system_prompt, target_language, fallback,
                    self._request_config.model.for_model(fallback).max_tokens,
                    preserve_source, number_correction=True,
                )
            return None
        if _looks_like_refusal(candidate):
            return None
        return candidate

    def _condition_fallback_model(self, model_id: str) -> str | None:
        """Only a confirmed local fast-model reversal may use an already-ready 4B."""
        if (
            model_id != MODEL_FAST
            or getattr(self._request_config.api, "enabled", False)
            or self._api_mode_active()
            or self._frame_is_stale()
        ):
            return None
        if not self._request_config.model.for_model(MODEL_ACCURATE).enabled:
            return None
        loaded = getattr(self._pool, "loaded_models", None)
        if not callable(loaded) or MODEL_ACCURATE not in loaded():
            return None
        if not getattr(self._pool, "_warmed", {}).get(MODEL_ACCURATE, True):
            return None
        return MODEL_ACCURATE

    def _retry_individually(
        self,
        texts: list[str],
        model_id: str,
        system_prompt: str,
        max_tokens: int,
        preserve_source: bool,
    ) -> list[str | None]:
        """批量对位失败后的逐条重试。

        **必须限制重试条数**：逐条重试会把一次请求放大成 N 次，每次都重新
        处理上百 token 的系统提示。实测批量失败时单帧耗时从数秒膨胀到
        49 秒——而实时翻译等不起这个时间。

        超出上限的片段直接返回 ``None``（调用方会保留原文），
        宁可这一帧少译几条，也不能让界面卡住十几秒。

        本方法整体计入单帧纠错预算（多批帧可能逐批降级），且帧过期
        后立即中止——为一帧已经没人看的译文继续发请求没有意义。
        """
        if self._frame_is_stale():
            logger.debug("帧已过期，跳过逐条重试")
            return [None] * len(texts)
        if texts and self._acquire_correction_batches(1) <= 0:
            logger.warning("单帧纠错预算已用尽，批量对位失败的片段保留原文")
            return [None] * len(texts)
        limit = min(len(texts), MAX_INDIVIDUAL_RETRIES)
        if limit < len(texts):
            logger.debug(
                "批量对位失败，仅逐条重试前 %s / %s 条以限制延迟",
                limit,
                len(texts),
            )

        def _one(text: str) -> str | None:
            single_raw = self._pool.translate(
                model_id=model_id,
                prompt=text,
                system_prompt=system_prompt,
                max_tokens=_output_cap(text, max_tokens, False),
            )
            if not single_raw:
                return None
            single_parsed = _parse_single_response(single_raw, text, preserve_source)
            return single_parsed[0] if single_parsed else None

        # 在线模式：逐条请求的网络往返线性叠加（N 条 × 每条数百毫秒起），
        # 与多批并发同理由线程池压缩整帧延迟；本地模式保持串行——
        # llama-server 对并发请求只能排队，并行无收益还多占线程。
        concurrency = self._api_concurrency()
        if concurrency > 1:
            results: list[str | None] = []
            with ThreadPoolExecutor(
                max_workers=min(limit, concurrency),
                thread_name_prefix="ApiRetry",
            ) as executor:
                futures = [executor.submit(copy_context().run, _one, text) for text in texts[:limit]]
                for future in futures:
                    try:
                        results.append(future.result())
                    except Exception:  # noqa: BLE001 - 单条异常按该条失败处理
                        logger.warning("逐条重试线程异常，该条保留原文", exc_info=True)
                        results.append(None)
            results.extend([None] * (len(texts) - limit))
            return results

        results = []
        for index, text in enumerate(texts):
            if index >= limit:
                results.append(None)
                continue
            if self._frame_is_stale():
                # 串行逐条重试可能持续数秒，期间新帧已到：立即停手。
                results.append(None)
                continue
            results.append(_one(text))
        return results

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    def _api_concurrency(self) -> int:
        """在线模式可用的并发请求数；离线/未启用返回 1（串行）。

        与 :meth:`_translate_pending` 的 ``api_active`` 判定同口径：
        配置启用 **且** 池当前确实由在线引擎承担（不在冷却期）。
        """
        api_config = getattr(self._request_config, "api", None)
        if api_config is None or not getattr(api_config, "enabled", False):
            return 1
        if not self._api_mode_active():
            return 1
        return max(1, int(getattr(api_config, "max_concurrency", 4)))

    def _needs_translation(
        self,
        paragraph: Paragraph,
        source_language: str,
        target_language: str,
        fallback_source: str,
    ) -> bool:
        """判断段落是否需要翻译。"""
        if not getattr(self._request_config.preprocess, "skip_same_language", True):
            # 即使关掉了"同语言跳过"，命令行/代码片段也依然不送模型：
            # 翻译它们只会产出乱码，原样保留才是正确结果。
            return not looks_like_code(paragraph.text)

        if source_language and source_language != "auto":
            detected = source_language
        else:
            # Paragraph.language stores only the guessed code, discarding its
            # confidence and basis. Re-evaluate before skipping. Latin and
            # Cyrillic scripts are shared by many selectable languages; Han
            # also cannot distinguish Simplified/Traditional Chinese or
            # Japanese text without kana. Only kana/hangul are decisive here.
            guess = detect_language(paragraph.text, fallback_source)
            detected = guess.code
            reliable_same_language = guess.is_confident and (
                (guess.code == "ja" and guess.basis == "假名")
                or (guess.code == "ko" and guess.basis == "谚文")
            )
            if detected == target_language and not reliable_same_language:
                # Even a high score from the four-language Latin heuristic
                # cannot prove an already-target-language result. Let the
                # model decide and retain its result in the ordinary cache.
                # （已知代价：夹零星拉丁词的同语言段回显会被输出守卫拦下，
                # 纠错也救不回——由下方 rejected_texts 抑制表兜底，窗口内
                # 不再重复整轮注定失败的推理，见 _suppress_withheld。）
                return not looks_like_code(paragraph.text)

        if detected == target_language:
            # 同语言跳过前先检查"夹杂物"：主体是目标语言但嵌着成块的
            # 外文脚本（如中文界面里嵌着一句韩文），这些片段同样需要
            # 翻译——直接跳过会让用户看到"选了翻译却有一块没被覆盖"。
            # 无夹杂物时维持原行为：跳过（不需要翻译）。
            return _has_foreign_residue(paragraph.text, target_language)
        # 命令行、代码、路径占主体的段落不翻译：模型对这些内容的
        # "译文"基本是乱码（实测把 shell 命令翻成"宫廷所AD"一类），
        # 而提示词本来就要求这部分原样保留——跳过等于直接给出正确结果。
        return not looks_like_code(paragraph.text)

    def _substitute(
        self, text: str, source_language: str | None = None,
        target_language: str | None = None,
    ):
        """术语替换。术语库为空或未启用时返回原文本。"""
        if not getattr(self._request_config.translation, "glossary_enabled", True):
            from .glossary_service import GlossarySubstitution

            return GlossarySubstitution(text=text)
        translation = self._request_config.translation
        source_language = source_language or translation.source_language
        target_language = target_language or translation.target_language
        if source_language == "auto":
            source_language = detect_language(
                text, translation.fallback_source_language
            ).code
        substitution = self._request_glossary.substitute(text, source_language, target_language)
        request = self._request_context.get()
        if request is not None:
            # Before restoration, inference receives placeholders rather than
            # original terms. Carry the same narrow exceptions to that input.
            request.protected_terms[substitution.text] = tuple(dict.fromkeys((
                *request.protected_terms.get(substitution.text, ()),
                *self._protected_target_terms(text),
            )))
        return substitution

    def _cache_key(self, text: str, source_language: str, target_language: str) -> str:
        """构造推理缓存键。

        键包含：来源命名空间、文本、语言对、术语表版本。不含坐标——
        同一句话出现在屏幕不同位置时应命中同一缓存。精简模式会改变
        输出约束，因此与普通模式隔离，避免切换设置后复用风格不匹配的译文。
        """
        from hashlib import sha256
        from unicodedata import normalize

        if source_language == "auto":
            source_language = detect_language(
                text, self._request_config.translation.fallback_source_language
            ).code

        # Long documents often share the same opening paragraphs while their
        # final conditions / negations differ. A prefix key reuses the wrong
        # translation; hashing the complete input also keeps keys bounded.
        digest = sha256(normalize("NFC", text).encode("utf-8")).hexdigest()
        request = self._request_context.get()
        namespace = request.namespace if request is not None else self._live_namespace()
        if request is not None and request.cache_identity is not None:
            identity = request.cache_identity
        else:
            identity = self._cache_identity(namespace)
            if request is not None:
                request.cache_identity = identity
        concise = bool(getattr(self._request_config.translation, "concise_output", False))
        return (
            f"{namespace}|{source_language}>{target_language}"
            f"|g{self._request_glossary.version}|c{int(concise and not getattr(self._request_config.translation, 'semantic_review', True))}"
            f"|i{identity}|sha256:{digest}"
        )

    def _cache_identity(self, namespace: str) -> str:
        """Stable output policy identity, including providers across restarts.

        A process-local API revision resets to 1 on restart. It cannot identify
        the provider, model or account behind a persisted translation. Hash the
        effective output settings; neither a Key nor an endpoint is stored in
        cache keys. Performance-only settings do not invalidate good results.
        """
        from hashlib import sha256

        from ..components.llm.prompt import PROMPT_POLICY_VERSION

        settings = self._request_config
        if namespace.startswith("api:"):
            api = settings.api
            route = (
                "api", api.base_url.strip().rstrip("/"), api.model_name.strip(),
                api.api_key, api.temperature,
            )
        else:
            schedule = settings.schedule
            models = (schedule.active_model,) if schedule.mode == "manual" else (
                MODEL_FAST, MODEL_ACCURATE,
            )
            if (MODEL_FAST in models and settings.model.for_model(MODEL_ACCURATE).enabled
                    and not getattr(settings.translation, "semantic_review", True)):
                # A verified temporal-error fallback can produce the cached
                # result even in manual fast mode. Its policy must share the
                # same persistence isolation as ordinary model output.
                models = tuple(dict.fromkeys((*models, MODEL_ACCURATE)))
            model_fingerprint = getattr(self._pool, "model_content_fingerprint", None)

            def content_identity(model_id: str) -> str:
                if callable(model_fingerprint):
                    return model_fingerprint(model_id)
                # Test doubles do not load weights. Real ModelPool always
                # supplies a verified digest or a session-isolated nonce.
                pool_type = type(self._pool)
                return f"test-double:{pool_type.__module__}.{pool_type.__qualname__}"

            route = (
                "local", schedule.mode, schedule.active_model,
                schedule.content_adaptive, schedule.resource_adaptive,
                tuple((
                    model_id,
                    content_identity(model_id),
                    tuple(getattr(settings.model.for_model(model_id), name, None)
                          for name in ("file_name", "enabled", "n_ctx", "temperature", "top_p", "max_tokens")),
                ) for model_id in models),
            )
        glossary_enabled = getattr(settings.translation, "glossary_enabled", True)
        glossary = [
            (entry.source, entry.target, entry.case_sensitive, entry.whole_word,
             entry.source_language, entry.target_language)
            for entry, _ in self._request_glossary._compiled
        ] if glossary_enabled else []
        review_policy = (
            bool(getattr(settings.translation, "semantic_review", True)),
            REVIEW_POLICY_VERSION,
            bool(getattr(settings.translation, "compact_display", True)),
            str(getattr(settings.translation, "scene_mode", "auto")),
            "paragraph-only-context-v1",
        )
        payload = json.dumps(
            (PROMPT_POLICY_VERSION, route, glossary_enabled, glossary, review_policy),
            ensure_ascii=False, separators=(",", ":"),
        )
        return sha256(payload.encode("utf-8")).hexdigest()[:24]

    def _build_item(
        self,
        paragraph: Paragraph,
        translated: str,
        from_cache: bool,
    ) -> RenderItem:
        """把段落与译文组装为渲染条目。"""
        monitor_index = getattr(paragraph, "monitor_index", None)
        return RenderItem(
            box=paragraph.box,
            text=translated,
            source_text=paragraph.text,
            is_dark_background=False,
            from_cache=from_cache,
            monitor_index=monitor_index,
        )

    def _write_history(
        self,
        items: list[RenderItem],
        model_id: str,
        stats: TranslationStats,
    ) -> None:
        """写入翻译历史。"""
        request = self._request_context.get()
        if request is not None:
            request.history = (items, model_id, stats)
            return
        self._record_history(items, model_id, stats)

    def _record_history(
        self,
        items: list[RenderItem],
        model_id: str,
        stats: TranslationStats,
        request: _TranslationRequest | None = None,
    ) -> None:
        # _commit_request authorizes this write before releasing its short
        # state lock. Disk I/O may complete later, but its language labels and
        # privacy options must still come from that authorized request.
        settings = request.config if request is not None else self._request_config
        if self._history is None or not getattr(settings.history, "enabled", True):
            return

        store_source = bool(getattr(settings.history, "store_source_text", True))
        source_language = settings.translation.source_language
        target_language = settings.translation.target_language

        now = time.monotonic()
        for item in items:
            if not item.text.strip():
                continue
            if item.from_cache:
                # 缓存命中不新增历史：静止画面每帧都产出一批命中条目，
                # 逐帧写库正是"历史被同一屏内容刷爆"的来源。
                continue
            if self._history_recently_written(item.source_text, now):
                continue
            self._history.record(
                source_text=item.source_text,
                target_text=item.text,
                source_lang=source_language,
                target_lang=target_language,
                model_id=model_id,
                from_cache=item.from_cache,
                store_source=store_source,
            )

    def _history_recently_written(self, source_text: str, now: float) -> bool:
        """同一原文在去重窗口内是否已写过历史（LRU 记账）。

        每次查询都刷新该原文的时间戳：内容连续不变时持续去重，
        间隔超过 :data:`HISTORY_DEDUP_WINDOW_S` 后才允许再次记录。
        """
        with self._history_seen_lock:
            last = self._history_seen.get(source_text)
            self._history_seen[source_text] = now
            self._history_seen.move_to_end(source_text)
            while len(self._history_seen) > HISTORY_DEDUP_MAX_ENTRIES:
                self._history_seen.popitem(last=False)
            return last is not None and now - last < HISTORY_DEDUP_WINDOW_S

    # ------------------------------------------------------------------
    # 单条翻译（供术语库测试、剪贴板翻译等场景使用）
    # ------------------------------------------------------------------

    def translate_text(
        self,
        text: str,
        snapshot: ResourceSnapshot | None = None,
    ) -> str:
        """Translate text using one consistent request snapshot."""
        with self._request_scope():
            return self._translate_text_request(text, snapshot)

    def _translate_text_request(
        self,
        text: str,
        snapshot: ResourceSnapshot | None = None,
    ) -> str:
        """翻译一段文本，返回译文。用于非屏幕场景的即时翻译。"""
        text = (text or "").strip()
        if not text:
            return ""
        if getattr(self._request_config.translation, "semantic_review", True):
            started = time.perf_counter()
            stats = TranslationStats(paragraphs=1)
            paragraph = Paragraph(text=text, box=Rect(0, 0, 1, 1))
            items = self._translate_reviewed_units([paragraph], stats, snapshot)
            stats.elapsed = time.perf_counter() - started
            self._publish_output(stats, [(item.source_text, item.text) for item in items])
            return items[0].text if items else text

        # 与批量路径同口径：命令行/代码为主体的文本原样保留，
        # 不送模型——送了也只是把原样返回，白付一次推理延迟。
        if looks_like_code(text):
            return text

        settings = self._request_config
        source_language = settings.translation.source_language
        target_language = settings.translation.target_language

        detected_source = source_language if source_language != "auto" else detect_language(
            text, settings.translation.fallback_source_language
        ).code
        substitution = self._substitute(text, detected_source, target_language)
        cache_key = self._cache_key(substitution.text, detected_source, target_language)

        # 确定性 UI 短词短路（同 _collect_pending）：模型对 "OK" 必然回显。
        if target_language == "zh" and not substitution.replacements:
            fallback = _UI_TOKEN_FALLBACKS_ZH.get(text.casefold())
            if fallback is not None:
                self._cache_put(cache_key, fallback)
                return fallback

        cached = self._cache.get(cache_key)
        if cached is not None:
            restored, missing = self._request_glossary.restore(cached, substitution.replacements)
            if (
                not self._unresolved_terms(restored, missing, substitution.replacements)
                and self._valid_output(text, restored)
            ):
                return restored
            self._cache_discard(cache_key)

        decision = self._scheduler.decide(
            snapshot=snapshot, texts=[text], glossary_term_count=substitution.hit_count
        )
        model_config = settings.model.for_model(decision.model_id)
        max_tokens = model_config.max_tokens
        # 在线模式：输出上限跟随在线配置（云端上下文长），
        # 长段落译文不被本地模型的输出上限截断
        api_config = getattr(settings, "api", None)
        api_active = bool(
            api_config is not None
            and getattr(api_config, "enabled", False)
            and self._api_mode_active()
        )
        if api_active:
            # 输出上限由动态估计主导（同 _translate_pending：剪贴板等
            # 长文翻译不被 768 默认上限截断）
            max_tokens = _API_OUTPUT_HARD_LIMIT

        preserve_source = bool(getattr(settings.translation, "keep_source_text", False))
        concise = bool(getattr(settings.translation, "concise_output", False))

        # 精简提示词只为本地小模型省 prefill；在线模式 prefill 廉价
        # （延迟由网络往返主导），完整提示词多出 UI 标准译法、已译回退、
        # 无寒暄三条质量规则，直接换外部模型的译文质量。
        if (
            not substitution.replacements
            and not preserve_source
            and not api_active
        ):
            from ..components.llm.prompt import build_compact_prompt

            system_prompt = build_compact_prompt(
                target_language, source_language, concise=concise
            )
        else:
            system_prompt = build_system_prompt(
                source_language=source_language,
                target_language=target_language,
                glossary_terms=list(substitution.replacements),
                # Original text is available separately; request only the
                # translation while retaining defensive comparison parsing.
                preserve_source=False,
                concise=concise,
            )
        raw = self._pool.translate(
            model_id=decision.model_id,
            prompt=substitution.text,
            system_prompt=system_prompt,
            max_tokens=_output_cap(substitution.text, max_tokens, False),
        )
        if not raw:
            return ""

        parsed = _parse_single_response(raw, substitution.text, preserve_source)
        if not parsed:
            return ""

        restored, missing = self._request_glossary.restore(parsed[0], substitution.replacements)
        unresolved = self._unresolved_terms(restored, missing, substitution.replacements)
        mixed = self._has_english_scaffolding(text, restored)
        # 进入缓存的是"术语还原前"的模型输出（命中时调用方会再做还原）。
        # 各恢复分支替换模型输出后必须同步更新，否则缓存里留的是坏输出
        # （实测：拒答恢复后缓存仍存致歉文案，同文本第二次翻译命中缓存
        # 直接返回致歉——缓存投毒）。
        cacheable = parsed[0]
        if (
            not unresolved
            and not mixed
            and not _looks_like_refusal(restored)
            and (
                has_foreign_prose(restored, target_language, text, self._protected_target_terms(text))
                or _SEGMENT_RESIDUE in restored
                or has_reversed_until_condition(text, restored, target_language)
                or _missing_numerals(text, restored)
            )
        ):
            # 语言错误/分隔符残片与拒答同理：替换的是"内容"而不是"措辞"，
            # 用同一提示词单段重试一次，仍失败则保留原文。
            recovered = self._retry_wrong_language_single(
                substitution.text, system_prompt, target_language,
                decision.model_id, max_tokens, preserve_source,
                condition_correction=has_reversed_until_condition(text, restored, target_language),
                number_correction=bool(_missing_numerals(text, restored)),
            )
            if recovered is None:
                logger.warning("单段翻译输出语言错误或带分隔符残片，已保留原文")
                self._suppress_withheld(cache_key)
                return text
            cacheable = recovered
            restored, missing = self._request_glossary.restore(recovered, substitution.replacements)
            unresolved = self._unresolved_terms(restored, missing, substitution.replacements)
            mixed = self._has_english_scaffolding(text, restored)
        if not unresolved and not mixed and _looks_like_refusal(restored):
            # 拒答替换的是"内容"而不是"措辞"：即便术语齐全也算异常，
            # 因此放在术语/英文残留检查之外单独处理，失败则返回原文。
            recovered = self._retry_after_refusal(
                substitution.text, decision.model_id, max_tokens, preserve_source
            )
            if recovered is None:
                logger.warning("单段翻译被模型拒答，已保留原文")
                self._suppress_withheld(cache_key)
                return text
            cacheable = recovered
            restored, missing = self._request_glossary.restore(recovered, substitution.replacements)
            unresolved = self._unresolved_terms(restored, missing, substitution.replacements)
            mixed = self._has_english_scaffolding(text, restored)
        if unresolved or mixed:
            recovered = self._retry_without_placeholders(
                text,
                list(dict.fromkeys(substitution.replacements.values())) if mixed else unresolved,
                decision.model_id, max_tokens,
                preserve_source, concise,
            )
            if recovered is None:
                logger.warning(
                    "单段翻译异常（缺失术语 %s 个，英文残留 %s），已保留原文",
                    len(unresolved), mixed,
                )
                return text
            self._cache_put(cache_key, recovered)
            return recovered
        if not self._valid_output(text, restored):
            return text
        self._cache_put(cache_key, cacheable)
        return restored


def build_render_frame_from_pairs(
    pairs: list[tuple[Rect, str]],
    seq: int = 0,
    origin: Rect | None = None,
) -> RenderFrame:
    """便捷函数：由 ``(包围盒, 译文)`` 列表构造渲染帧（供测试使用）。"""
    return RenderFrame(
        seq=seq,
        origin=origin or Rect(0, 0, 0, 0),
        items=[RenderItem(box=box, text=text) for box, text in pairs],
    )


def _pack_batches(
    pending: list,
    batch_size: int,
    char_budget: int,
    max_output_tokens: int | None = None,
) -> list[list]:
    """把待翻译片段按条数、输入预算与完整译文的输出容量装批。

    条数上限来自配置（``model.batch_size``）；字符预算来自模型上下文
    （见 :meth:`TranslationService._input_char_budget`）。两个限制先到
    为准。输出容量不足会截断整批并消耗有限的降级重试条数，因此必须
    在正常请求前拆批。单个片段本身就超预算时仍独立成批——截断内容会丢译文，
    宁可让这一条按原有路径处理。
    """
    batches: list[list] = []
    current: list = []
    current_chars = 0
    for item in pending:
        text = item[0].glossary_text or item[0].text
        exceeds_output = (
            max_output_tokens is not None
            and _estimated_output_tokens(
                "\n".join([*(part[0].glossary_text or part[0].text for part in current), text]),
                False, len(current) + 1,
            ) > max_output_tokens
        )
        if current and (
            len(current) >= batch_size
            or current_chars + len(text) + _SEGMENT_SEPARATOR_CHARS > char_budget
            or exceeds_output
        ):
            batches.append(current)
            current = []
            current_chars = 0
        current.append(item)
        current_chars += len(text) + _SEGMENT_SEPARATOR_CHARS
    if current:
        batches.append(current)
    return batches
