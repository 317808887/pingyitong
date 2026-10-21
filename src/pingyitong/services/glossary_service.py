"""专业术语库（自研）。

对应产品功能规划三·（四）：

    "支持用户导入自定义术语表（CSV 格式，包含原文术语与译文对照）。
    翻译前，系统先对术语表中的词汇进行占位替换，翻译完成后再还原，
    确保行业专有名词、品牌名、人名等翻译结果稳定一致。"

为什么用"占位符替换"而不是"写在 Prompt 里"
    把术语表写进提示词有两个问题：一是术语一多就会挤占上下文，
    二是大模型并不保证严格遵守提示词，长文本下尤其容易漂移。
    替换为占位符后，术语在送入模型前就已从待翻译文本中"消失"，
    模型只需原样搬运占位符；即使模型漏抄，我们在还原阶段也能检测到
    （占位符缺失）并回退到该术语的直译，保证结果可控。

占位符字符选择
    使用 ``[TERM_0]`` 形式的可读占位符
    （见 :data:`~pingyitong.constants.GLOSSARY_TOKEN_TEMPLATE`）。
    实测极速模型会把 Unicode 私用区占位符误写为数字，导致术语丢失；
    可读占位符在同一组真实模型测试中能被完整保留。
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..constants import GLOSSARY_TOKEN_PATTERN, GLOSSARY_TOKEN_TEMPLATE
from ..utils.logger import get_logger

logger = get_logger(__name__)

#: CSV 中建议使用的表头名称（大小写不敏感）
_SOURCE_HEADERS = {"原文", "source", "src", "from", "term", "术语"}
_TARGET_HEADERS = {"译文", "target", "tgt", "to", "translation", "翻译"}


@dataclass
class GlossaryEntry:
    """一条术语对照。"""

    source: str
    target: str
    case_sensitive: bool = False
    #: 是否作为独立词匹配（避免 ``cat`` 命中 ``category``）
    whole_word: bool = True
    note: str = ""
    #: 为空表示对所有语言适用；预装库会明确限定语言对。
    source_language: str = ""
    target_language: str = ""

    def pattern(self) -> re.Pattern[str]:
        """构造匹配该术语的正则。"""
        escaped = re.escape(self.source)
        if self.whole_word:
            # 仅当术语首尾不是单词字符时才需要词边界，否则 \b 会误判
            prefix = r"(?<!\w)" if not self.source[:1].isalnum() else r"\b"
            suffix = r"(?!\w)" if not self.source[-1:].isalnum() else r"\b"
            escaped = f"{prefix}{escaped}{suffix}"
        flags = 0 if self.case_sensitive else re.IGNORECASE
        return re.compile(escaped, flags)


@dataclass
class GlossarySubstitution:
    """一次术语替换的结果。"""

    #: 替换后的文本
    text: str
    #: 出现过的占位符 → 术语译文
    replacements: dict[str, str] = field(default_factory=dict)
    #: 命中的术语原文列表
    matched_terms: list[str] = field(default_factory=list)

    @property
    def hit_count(self) -> int:
        return len(self.matched_terms)


class Glossary:
    """术语库：加载、匹配、替换与还原。"""

    def __init__(
        self,
        entries: list[GlossaryEntry] | None = None,
        case_sensitive: bool = False,
    ) -> None:
        self._entries: list[GlossaryEntry] = []
        self._compiled: list[tuple[GlossaryEntry, re.Pattern[str]]] = []
        self._pair_cache: dict[tuple[str, str], list[tuple[GlossaryEntry, re.Pattern[str]]]] = {}
        self.label = "自定义术语库"
        self.covered_languages = 0
        self.localized_languages = 0
        self.concept_count = 0
        self.invariant_count = 0
        self.case_sensitive = bool(case_sensitive)
        #: 术语表版本，用于推理缓存键——术语变更后缓存必须失效
        self.version = 0
        if entries:
            self.set_entries(entries)

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------

    def set_entries(self, entries: list[GlossaryEntry]) -> None:
        """替换全部术语，并按长度倒序排列。

        长度倒序很关键：先匹配长术语，避免 ``machine learning`` 被
        ``machine`` 提前命中而拆成两段。
        """
        # 应用全局大小写设置（条目自身未显式指定时）
        normalised = [
            GlossaryEntry(
                source=entry.source,
                target=entry.target,
                case_sensitive=entry.case_sensitive or self.case_sensitive,
                whole_word=entry.whole_word,
                note=entry.note,
                source_language=entry.source_language,
                target_language=entry.target_language,
            )
            for entry in entries
            if entry.source and entry.target
        ]
        normalised.sort(key=lambda entry: len(entry.source), reverse=True)
        self._entries = normalised
        self._compiled = [(entry, entry.pattern()) for entry in normalised]
        self._pair_cache.clear()
        self.version += 1
        logger.info("术语库已加载 %s 条术语", len(normalised))

    def load_from_csv(self, path: Path, case_sensitive: bool | None = None) -> int:
        """从 CSV 文件加载术语表。

        支持两种格式：

        - 带表头（``原文,译文`` 或 ``source,target``）
        - 无表头（前两列即为原文与译文）

        :return: 成功加载的条目数。
        """
        if case_sensitive is not None:
            self.case_sensitive = bool(case_sensitive)

        path = Path(path)
        if not path.exists():
            logger.warning("术语表文件不存在：%s", path)
            return 0

        entries: list[GlossaryEntry] = []
        try:
            # 尝试 UTF-8-SIG（Excel 导出的 CSV 通常带 BOM）
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                entries = self._parse_rows(list(csv.reader(handle)))
        except UnicodeDecodeError:
            logger.debug("UTF-8 解码失败，尝试 GBK")
            try:
                with path.open("r", encoding="gbk", newline="") as handle:
                    entries = self._parse_rows(list(csv.reader(handle)))
            except (OSError, UnicodeDecodeError) as exc:
                logger.error("术语表解析失败：%s", exc)
                return 0
        except OSError as exc:
            logger.error("术语表读取失败：%s", exc)
            return 0

        self.set_entries(entries)
        return len(entries)

    def save_to_csv(self, path: Path) -> bool:
        """把当前术语表导出为 CSV。"""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["原文", "译文", "区分大小写", "整词匹配", "源语言", "目标语言"])
                for entry in self._entries:
                    writer.writerow(
                        [
                            entry.source,
                            entry.target,
                            "是" if entry.case_sensitive else "否",
                            "是" if entry.whole_word else "否",
                            entry.source_language,
                            entry.target_language,
                        ]
                    )
            return True
        except OSError as exc:
            logger.error("术语表导出失败：%s", exc)
            return False

    @staticmethod
    def _parse_rows(rows: list[list[str]]) -> list[GlossaryEntry]:
        """把 CSV 行解析为术语条目。"""
        entries: list[GlossaryEntry] = []
        if not rows:
            return entries

        start = 0
        first = [cell.strip().lower() for cell in rows[0]]
        if first and (
            first[0] in {header.lower() for header in _SOURCE_HEADERS}
            or (len(first) > 1 and first[1] in {header.lower() for header in _TARGET_HEADERS})
        ):
            start = 1  # 跳过表头

        for row in rows[start:]:
            if len(row) < 2:
                continue
            source = row[0].strip()
            target = row[1].strip()
            if not source or not target or source.startswith("#"):
                continue

            case_sensitive = False
            whole_word = True
            if len(row) > 2 and row[2].strip():
                case_sensitive = row[2].strip().lower() in {"是", "true", "1", "yes", "y"}
            if len(row) > 3 and row[3].strip():
                whole_word = row[3].strip().lower() in {"是", "true", "1", "yes", "y"}

            entries.append(
                GlossaryEntry(
                    source=source,
                    target=target,
                    case_sensitive=case_sensitive,
                    whole_word=whole_word,
                    source_language=row[4].strip() if len(row) > 4 else "",
                    target_language=row[5].strip() if len(row) > 5 else "",
                )
            )

        return entries

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    @property
    def entries(self) -> list[GlossaryEntry]:
        """全部术语条目。"""
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def is_empty(self) -> bool:
        return not self._entries

    def add(self, source: str, target: str, **kwargs: object) -> None:
        """追加一条术语。"""
        entry = GlossaryEntry(source=source, target=target, **kwargs)  # type: ignore[arg-type]
        self.set_entries(self._entries + [entry])

    def find(self, source: str) -> GlossaryEntry | None:
        """按原文精确查找术语。"""
        for entry in self._entries:
            if entry.case_sensitive:
                if entry.source == source:
                    return entry
            elif entry.source.lower() == source.lower():
                return entry
        return None

    # ------------------------------------------------------------------
    # 替换与还原
    # ------------------------------------------------------------------

    def substitute(
        self, text: str, source_language: str = "", target_language: str = ""
    ) -> GlossarySubstitution:
        """把文本中命中的术语替换为占位符。

        同一个术语在文本中多次出现时复用同一个占位符编号，
        减少模型需要搬运的不同符号数量。
        """
        if not self._entries or not text:
            return GlossarySubstitution(text=text)

        result = text
        token_to_target: dict[str, str] = {}
        matched: list[str] = []
        seen_terms: dict[str, str] = {}  # 术语原文 -> 占位符

        pair = (source_language, target_language)
        patterns = self._pair_cache.get(pair)
        if patterns is None:
            patterns = [
                (entry, pattern)
                for entry, pattern in self._compiled
                if (not entry.source_language or entry.source_language == source_language)
                and (not entry.target_language or entry.target_language == target_language)
            ]
            self._pair_cache[pair] = patterns

        for entry, pattern in patterns:

            def _replace(match: re.Match[str], _entry: GlossaryEntry = entry) -> str:
                key = _entry.source if _entry.case_sensitive else _entry.source.lower()
                token = seen_terms.get(key)
                if token is None:
                    token = GLOSSARY_TOKEN_TEMPLATE.format(len(seen_terms))
                    seen_terms[key] = token
                    token_to_target[token] = _entry.target
                    matched.append(_entry.source)
                return token

            result = pattern.sub(_replace, result)

        if not token_to_target:
            return GlossarySubstitution(text=text)

        return GlossarySubstitution(
            text=result,
            replacements=token_to_target,
            matched_terms=matched,
        )

    def restore(self, text: str, replacements: dict[str, str]) -> tuple[str, list[str]]:
        """把译文中的占位符还原为术语译文。

        :param text: 模型输出的译文。
        :param replacements: :attr:`GlossarySubstitution.replacements`。
        :return: ``(还原后的文本, 未被模型保留的占位符列表)``。

        第二种返回值用于监控模型是否遵守了占位符约束。缺失的占位符
        会被清除残留标记，避免未还原的占位符出现在用户屏幕上。
        """
        if not replacements or not text:
            return text, []

        missing: list[str] = []
        result = text

        for token, target in replacements.items():
            if token in result:
                result = result.replace(token, target)
            else:
                missing.append(token)

        # 清理模型中可能产生的占位符变体（例如被拆分或补了空格）
        result = _strip_residual_tokens(result)

        return result, missing

    def stats_text(self) -> str:
        """返回术语库状态描述。"""
        if self.is_empty:
            return "未加载术语"
        if self.covered_languages:
            scope = (
                f"{self.covered_languages} 种实测语言保留标准名词"
                if self.invariant_count
                else f"{self.covered_languages} 种实测语言范围"
            )
            return (
                f"{self.label}：{self.concept_count} 个概念 · "
                f"{scope} · {self.localized_languages} 种语言有本地化词条"
            )
        return f"{self.label}：已加载 {len(self._entries)} 条术语"


def load_builtin_glossary(
    target_language: str, case_sensitive: bool = False, domain: str = "software_ai"
) -> Glossary:
    """Load bundled terms for the chosen target; missing translations are skipped.

    A shared technical identifier is preserved for every verified language.
    Localized concepts only apply where both language labels are available.
    """
    from ..glossary_catalog import BUILTIN_GLOSSARY_DOMAINS, DEFAULT_GLOSSARY_DOMAIN
    from ..utils import paths

    glossary = Glossary(case_sensitive=case_sensitive)
    if domain not in BUILTIN_GLOSSARY_DOMAINS:
        logger.warning("未知术语领域 %s，改用默认领域", domain)
        domain = DEFAULT_GLOSSARY_DOMAIN
    glossary.label = f"预装{BUILTIN_GLOSSARY_DOMAINS[domain]}术语库"
    path = paths.builtin_glossary_path(domain)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.error("预装术语库读取失败：%s", exc)
        return glossary
    if data.get("schema_version") != 1:
        logger.error("预装术语库版本不兼容：%s", data.get("schema_version"))
        return glossary
    if data.get("domain") != domain:
        logger.error("预装术语库领域不符：期望 %s，得到 %s", domain, data.get("domain"))
        return glossary

    supported = set(data.get("supported_languages", []))
    glossary.covered_languages = len(supported)
    glossary.concept_count = len(data.get("concepts", []))
    glossary.invariant_count = len(data.get("invariant_terms", []))
    entries = [
        GlossaryEntry(name, name, case_sensitive=True)
        for name in data.get("invariant_terms", [])
        if isinstance(name, str) and name.strip()
    ]
    if target_language in supported:
        # On scripts without word separators, Python's \b rejects terms in
        # normal sentences. Match the full localized phrase instead.
        unsegmented = {"zh", "zh-TW", "ja", "ko", "th", "lo", "km", "my", "bo"}
        seen: set[tuple[str, str]] = set()
        localized: set[str] = set()
        for concept in data.get("concepts", []):
            forms = concept.get("forms", {})
            target = forms.get(target_language, "").strip()
            if not target:
                continue
            for acronym in concept.get("acronyms", []):
                entries.append(GlossaryEntry(
                    acronym, target, case_sensitive=True,
                    target_language=target_language,
                    note=f"Wikidata {concept.get('wikidata_id', '')}",
                ))
            aliases = concept.get("aliases", {})
            for source_language, source in forms.items():
                variants = [source, *aliases.get(source_language, [])]
                for variant in variants:
                    variant = variant.strip()
                    key = (source_language, variant.casefold())
                    if (
                        not variant
                        or source_language == target_language
                        or source_language not in supported
                        # A single CJK/Thai glyph embedded in a longer word
                        # has no safe boundary; do not force a medical or
                        # scientific substitution from that fragment.
                        or (source_language in unsegmented and len(variant) < 2)
                        or key in seen
                    ):
                        continue
                    seen.add(key)
                    localized.add(source_language)
                    entries.append(GlossaryEntry(
                        source=variant,
                        target=target,
                        whole_word=source_language not in unsegmented,
                        source_language=source_language,
                        target_language=target_language,
                        note=f"Wikidata {concept.get('wikidata_id', '')}",
                    ))
        glossary.localized_languages = len(localized)
    glossary.set_entries(entries)
    return glossary


def _strip_residual_tokens(text: str) -> str:
    """清除残留的占位符（含被模型插入了空格的变体）。"""
    # 完全匹配的残留
    text = re.sub(GLOSSARY_TOKEN_PATTERN, "", text)
    # 清理历史私用区占位符被拆开或与空格混排的情况
    text = re.sub("[\ue000\ue001]+", "", text)
    return text


def create_glossary(config: object) -> Glossary:
    """依据翻译配置节创建并加载术语库。

    :param config: :class:`~pingyitong.config.schema.TranslationConfig`。
    """
    if not getattr(config, "glossary_enabled", False):
        return Glossary()

    path_value = getattr(config, "glossary_path", "") or ""
    mode = getattr(config, "glossary_mode", "") or ("custom" if path_value else "builtin")
    if mode != "custom":
        return load_builtin_glossary(
            getattr(config, "target_language", "zh"),
            case_sensitive=getattr(config, "glossary_case_sensitive", False),
            domain=getattr(config, "glossary_domain", "software_ai"),
        )

    glossary = Glossary(case_sensitive=getattr(config, "glossary_case_sensitive", False))
    if path_value:
        glossary.load_from_csv(Path(path_value))
    return glossary
