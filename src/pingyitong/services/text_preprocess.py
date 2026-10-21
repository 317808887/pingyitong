"""桌面场景 OCR 后处理（自研）。

对应《项目原创性声明》二·（五）"桌面场景 OCR 后处理算法"。
声明中列出的六项能力在此逐一实现：

============================  ==========================================
声明内容                        实现位置
============================  ==========================================
基于坐标位置的文本行合并          :meth:`TextPreprocessor._cluster_lines`
段落划分                        :meth:`TextPreprocessor._group_paragraphs`
重复文本去重                    :meth:`TextPreprocessor._deduplicate`
上下文拼接                      :meth:`TextPreprocessor._assemble_text`
断句修复                        :func:`repair_hyphenation` / :func:`join_segments`
============================  ==========================================

为什么必须做后处理
    OCR 是针对"图像中的文字区域"工作的，它并不理解版面结构。
    一个完整的句子若跨越两行，会被切成两个甚至更多文本块；
    一个按钮上的双行文案也会被拆开。若逐块送进大模型，模型看到的是
    "碎片"，译文会出现断句错误、缺少主语、术语被拆开等问题。
    本模块的作用就是把碎片按版面关系还原成"通顺完整的句子"。

为什么按段落翻译、按段落坐标渲染
    段落内各行在屏幕上是连续的，把它们合成一个矩形包围盒后，
    译文绘制在该包围盒内即可自然覆盖原文，视觉上保持"逐行对应"。
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

from ..components.capture.geometry import Rect
from ..components.ocr import TextBlock
from ..utils.logger import get_logger
from .language import detect_language, is_probably_translatable

logger = get_logger(__name__)

#: 句子结束标点。用于判断"上一行是否已经说完了"。
_TERMINAL_PUNCTUATION = set("。！？!?.;；:：…")

#: 不宜出现在行首的标点（中文排版规则），用于决定拼接时是否补空格。
_LEADING_PUNCTUATION = set("，。！？、；：）】》」』”’%,.!?;:)]}>")

#: CJK 字符范围，用于判断拼接时是否需要空格分隔。
_CJK_RANGES = (
    (0x4E00, 0x9FFF), (0x3400, 0x4DBF),
    (0x3040, 0x30FF), (0xAC00, 0xD7AF),
    (0x3000, 0x303F), (0xFF00, 0xFFEF),
)


@dataclass
class Paragraph:
    """后处理得到的段落，是翻译与渲染的基本单位。"""

    #: 拼接清洗后的完整文本
    text: str
    #: 段落包围盒（物理坐标），用于渲染定位
    box: Rect
    #: 构成该段落的原始文本块
    blocks: list[TextBlock] = field(default_factory=list)
    #: 各块置信度的加权平均
    confidence: float = 1.0
    #: 检测到的源语言
    language: str = ""
    #: 行数，用于判断是否为多行段落
    line_count: int = 1
    #: 术语替换后的文本（由翻译服务填充）。原文保留在 :attr:`text` 中，
    #: 用于界面展示与历史记录；送入模型的是本字段。
    glossary_text: str = ""
    #: 来源显示器序号（``mss`` 下标）。渲染阶段按它把译文分发到各块屏幕，
    #: 缺失时条目会被判为"不属于任何被捕获的显示器"而整体丢弃——
    #: 表现为翻译在跑、翻译层却一片空白。因此必须从捕获帧一路带下来。
    monitor_index: int | None = None
    #: Accepted raw OCR fragments before inline/wrapped assembly. Text,
    #: punctuation and each physical rectangle retain their source ownership.
    source_lines: list[dict] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def is_multiline(self) -> bool:
        return self.line_count > 1

    def __str__(self) -> str:  # pragma: no cover - 调试用
        preview = self.text if len(self.text) <= 30 else f"{self.text[:30]}…"
        return f"Paragraph({preview!r} lines={self.line_count} @{self.box.to_ltrb()})"


@dataclass
class PreprocessStats:
    """后处理统计，用于状态面板展示"清洗效果"。"""

    input_blocks: int = 0
    output_paragraphs: int = 0
    merged_blocks: int = 0
    duplicate_removed: int = 0
    noise_removed: int = 0
    decision_events: list[dict] = field(default_factory=list)

    def as_text(self) -> str:
        return (
            f"{self.input_blocks} 块 → {self.output_paragraphs} 段"
            f"（合并 {self.merged_blocks}、去重 {self.duplicate_removed}、"
            f"滤除 {self.noise_removed}）"
        )


class TextPreprocessor:
    """把 OCR 的碎片文本块还原为通顺段落。"""

    def __init__(
        self,
        line_merge_overlap_ratio: float = 0.55,
        same_line_gap_ratio: float = 1.6,
        paragraph_gap_ratio: float = 1.7,
        line_max_horizontal_gap_ratio: float = 4.0,
        merge_broken_lines: bool = True,
        repair_hyphenation_enabled: bool = True,
        deduplicate_enabled: bool = True,
        filter_noise_enabled: bool = True,
        skip_pure_numeric: bool = True,
        fallback_language: str = "en",
    ) -> None:
        self.line_merge_overlap_ratio = float(line_merge_overlap_ratio)
        self.same_line_gap_ratio = float(same_line_gap_ratio)
        self.paragraph_gap_ratio = float(paragraph_gap_ratio)
        self.line_max_horizontal_gap_ratio = float(line_max_horizontal_gap_ratio)
        self.merge_broken_lines = bool(merge_broken_lines)
        self.repair_hyphenation_enabled = bool(repair_hyphenation_enabled)
        self.deduplicate_enabled = bool(deduplicate_enabled)
        self.filter_noise_enabled = bool(filter_noise_enabled)
        self.skip_pure_numeric = bool(skip_pure_numeric)
        self.fallback_language = fallback_language
        self._decision_events: list[dict] = []
        self._source_records: dict[int, list[dict]] = {}

    # ------------------------------------------------------------------
    # 配置
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: object) -> TextPreprocessor:
        """按 :class:`~pingyitong.config.schema.AppConfig` 构造，与真实应用完全一致。

        直接 ``TextPreprocessor()`` 会落回本类的构造默认值，而其中
        ``line_max_horizontal_gap_ratio``（4.0）与 schema 默认（1.5）并不相同，
        会导致验证脚本复现出与真实应用不同的段落划分。所有脚本与界面
        都应经由本方法构造。
        """
        return cls(
            line_merge_overlap_ratio=config.ocr.line_merge_overlap_ratio,
            same_line_gap_ratio=config.ocr.same_line_gap_ratio,
            paragraph_gap_ratio=config.ocr.paragraph_gap_ratio,
            line_max_horizontal_gap_ratio=config.ocr.line_max_horizontal_gap_ratio,
            merge_broken_lines=config.preprocess.merge_broken_lines,
            repair_hyphenation_enabled=config.preprocess.repair_hyphenation,
            deduplicate_enabled=config.preprocess.deduplicate,
            filter_noise_enabled=config.preprocess.filter_noise,
            skip_pure_numeric=config.preprocess.skip_pure_numeric,
            fallback_language=config.translation.fallback_source_language,
        )

    def configure(self, config: object) -> None:
        """应用 :class:`~pingyitong.config.schema.PreprocessConfig` 与 OCR 几何参数。"""
        self.merge_broken_lines = bool(getattr(config, "merge_broken_lines", True))
        self.repair_hyphenation_enabled = bool(getattr(config, "repair_hyphenation", True))
        self.deduplicate_enabled = bool(getattr(config, "deduplicate", True))
        self.filter_noise_enabled = bool(getattr(config, "filter_noise", True))
        self.skip_pure_numeric = bool(getattr(config, "skip_pure_numeric", True))

        self.line_merge_overlap_ratio = float(
            getattr(config, "line_merge_overlap_ratio", self.line_merge_overlap_ratio)
        )
        self.same_line_gap_ratio = float(
            getattr(config, "same_line_gap_ratio", self.same_line_gap_ratio)
        )
        self.paragraph_gap_ratio = float(
            getattr(config, "paragraph_gap_ratio", self.paragraph_gap_ratio)
        )
        self.line_max_horizontal_gap_ratio = float(
            getattr(
                config,
                "line_max_horizontal_gap_ratio",
                self.line_max_horizontal_gap_ratio,
            )
        )

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    def process(
        self,
        blocks: list[TextBlock],
        monitor_index: int | None = None,
    ) -> tuple[list[Paragraph], PreprocessStats]:
        """执行完整的后处理流程。

        :param blocks: OCR 输出的文本块（全局物理坐标）。
        :param monitor_index: 这批文本块来自哪块显示器（``mss`` 下标）。
            会写入每个段落并最终带到渲染条目上，供渲染阶段分发译文。
        :return: ``(段落列表, 统计信息)``。
        """
        stats = PreprocessStats(input_blocks=len(blocks))
        self._decision_events = stats.decision_events
        self._source_records = {
            id(block): [{"source_id": ordinal, **_source_snapshot(block)}]
            for ordinal, block in enumerate(blocks)
        }

        working = list(blocks)
        if self.filter_noise_enabled:
            working, removed = self._filter_noise(working)
            stats.noise_removed += removed

        if self.deduplicate_enabled:
            working, removed = self._deduplicate(working)
            stats.duplicate_removed += removed

        if not working:
            return [], stats

        lines = self._cluster_lines(working)
        merged_lines = self._merge_within_lines(lines)
        # 统计被吸收掉的块数（合并前总块数 − 合并后总块数）
        stats.merged_blocks += sum(len(line) for line in lines) - sum(
            len(line) for line in merged_lines
        )

        paragraphs = self._group_paragraphs(merged_lines)
        stats.output_paragraphs = len(paragraphs)

        # 补上来源显示器序号：渲染阶段按它分发译文，缺了会被整体丢弃
        if monitor_index is not None:
            for paragraph in paragraphs:
                paragraph.monitor_index = monitor_index

        return paragraphs, stats

    def _sources(self, block: TextBlock) -> list[dict]:
        records = self._source_records.get(id(block), [_source_snapshot(block)])
        return [{**item, "rect": list(item["rect"])} for item in records]

    def _record(self, action: str, reason: str, blocks: list[TextBlock],
                output: TextBlock | None = None) -> None:
        self._decision_events.append({
            "stage": "preprocess", "action": action, "reason": reason,
            "inputs": [item for block in blocks for item in self._sources(block)],
            "output": _source_snapshot(output) if output is not None else None,
        })

    # ------------------------------------------------------------------
    # 1. 噪声过滤
    # ------------------------------------------------------------------

    def _filter_noise(self, blocks: list[TextBlock]) -> tuple[list[TextBlock], int]:
        """过滤难以构成有意义翻译的碎片。

        屏幕截图中常见的噪声包括：进度条上的数字、图标旁的单个符号、
        行号、纯装饰性字符。这些内容翻译后与原文相同，只会浪费推理时间。
        """
        accepted: list[TextBlock] = []
        removed = 0

        for block in blocks:
            text = block.text.strip()

            if not text:
                self._record("filter", "empty_text", [block])
                removed += 1
                continue

            # 单个字符且不是 CJK（单个汉字/假名有实际含义，单个拉丁字母通常没有）
            if len(text) == 1 and not _is_cjk(text):
                self._record("filter", "single_non_cjk_character", [block])
                removed += 1
                continue

            if self.skip_pure_numeric and _is_pure_numeric(text):
                self._record("filter", "pure_numeric", [block])
                removed += 1
                continue

            if not is_probably_translatable(text):
                self._record("filter", "no_translatable_script", [block])
                removed += 1
                continue

            accepted.append(block)

        return accepted, removed

    # ------------------------------------------------------------------
    # 2. 去重
    # ------------------------------------------------------------------

    def _deduplicate(self, blocks: list[TextBlock]) -> tuple[list[TextBlock], int]:
        """去除重复文本块。

        差分区域重叠会导致同一文字被识别两次；某些界面（如任务栏、
        状态栏）也会在相近位置重复渲染同一文案。
        """
        accepted: list[TextBlock] = []
        removed = 0

        for block in blocks:
            duplicate_index = -1
            for index, existing in enumerate(accepted):
                if existing.text != block.text:
                    continue
                overlap = existing.box.intersection(block.box)
                smaller = min(existing.box.area, block.box.area) or 1
                if overlap.area / smaller > 0.5:
                    duplicate_index = index
                    break

            if duplicate_index >= 0:
                # 保留置信度更高的那个，保证识别质量不退化
                previous = accepted[duplicate_index]
                if block.confidence > previous.confidence:
                    accepted[duplicate_index] = block
                self._record("deduplicate", "same_text_overlap", [previous, block],
                             accepted[duplicate_index])
                removed += 1
            else:
                accepted.append(block)

        return accepted, removed

    # ------------------------------------------------------------------
    # 3. 文本行聚类
    # ------------------------------------------------------------------

    def _cluster_lines(self, blocks: list[TextBlock]) -> list[list[TextBlock]]:
        """按垂直重叠把文本块聚成视觉行，再按水平间隙切分。

        分两步，第二步是关键：

        **第一步：按垂直重叠聚成"视觉行"**
        同一行的文字即使字号不同、基线略有差异，垂直范围也高度重合；
        相邻两行即使行距很小，垂直重叠也通常不足行高一半。

        **第二步：按相邻块之间的水平间隙把视觉行切成多段**
        桌面截图里常常并排开着多个窗口、侧栏或面板，它们在同一水平带上
        各有文字。若不做切分，这些毫不相干的元素会被当成同一行，
        进而合并成跨窗口的巨型段落，译文彻底错乱
        （实测出现过 2280 像素宽、把侧栏与终端内容混在一起的段落）。

        .. note::
           切分必须比较**相邻两个块之间的实际间隙**，而不是"块与整行的
           累积包围盒"之间的间隙。早期实现用的是后者，导致**链式合并**：
           每并入一个块，包围盒就向右扩张一点，下一个块只需离扩张后的
           边界足够近即可加入——于是一串每步只隔 70px 的块最终横跨 2283px，
           把整屏文字连成一段。
        """
        if not blocks:
            return []

        ordered = sorted(blocks, key=lambda b: (b.box.top, b.box.left))

        # 第一步：按垂直重叠聚成视觉行
        rows: list[list[TextBlock]] = []
        row_bounds: list[Rect] = []
        for block in ordered:
            placed = False
            for index, bounds in enumerate(row_bounds):
                if _vertical_overlap_ratio(bounds, block.box) >= self.line_merge_overlap_ratio:
                    rows[index].append(block)
                    row_bounds[index] = bounds.union(block.box)
                    placed = True
                    break
            if not placed:
                rows.append([block])
                row_bounds.append(block.box)

        # 第二步：行内按水平位置排序后，在每个超阈值间隙处切分
        lines: list[list[TextBlock]] = []
        for row in rows:
            row.sort(key=lambda b: b.box.left)
            current: list[TextBlock] = [row[0]]
            for previous, block in zip(row, row[1:], strict=False):
                if self._horizontally_adjacent(previous.box, block.box):
                    current.append(block)
                else:
                    lines.append(current)
                    current = [block]
            lines.append(current)

        # 按垂直位置排序并回填行号
        lines.sort(key=lambda line: (min(b.box.top for b in line), line[0].box.left))
        for line_id, line in enumerate(lines):
            for block in line:
                block.line_id = line_id
                for source in self._source_records.get(id(block), []):
                    source["line_id"] = line_id

        return lines

    def _horizontally_adjacent(self, previous: Rect, current: Rect) -> bool:
        """判断两个**相邻**文本块之间是否足够接近（可视为同一行内连续文字）。

        间隙超过字高的若干倍即视为不同栏（并排的窗口/侧栏），应在该处断开。

        注意入参是两个块各自的包围盒，而不是其中之一的累积包围盒——
        后者会引起链式合并（详见 :meth:`_cluster_lines` 的说明）。
        """
        gap = max(current.left - previous.right, previous.left - current.right)
        if gap <= 0:
            return True  # 水平有重叠，必然连续

        reference_height = min(previous.height, current.height) or 1
        return gap <= reference_height * self.line_max_horizontal_gap_ratio

    # ------------------------------------------------------------------
    # 4. 行内合并
    # ------------------------------------------------------------------

    def _merge_within_lines(self, lines: list[list[TextBlock]]) -> list[list[TextBlock]]:
        """合并同一行内水平相邻的文本块。

        OCR 常把 "Save" 和 "As" 识别为两块（中间有明显字距），
        但它们其实是同一个按钮文案。合并依据是水平间隙与字高的比例关系。
        """
        merged_lines: list[list[TextBlock]] = []

        for line in lines:
            if len(line) <= 1:
                merged_lines.append(line)
                continue

            current_group = [line[0]]
            result: list[TextBlock] = []

            for block in line[1:]:
                previous = current_group[-1]
                gap = block.box.left - previous.box.right
                reference_height = min(previous.height, block.height) or 1
                threshold = reference_height * self.same_line_gap_ratio

                if gap <= threshold:
                    current_group.append(block)
                else:
                    result.append(self._merge_group(current_group))
                    current_group = [block]

            result.append(self._merge_group(current_group))
            merged_lines.append(result)

        return merged_lines

    def _merge_group(self, group: list[TextBlock]) -> TextBlock:
        """把一个组内的文本块合并为单个文本块。

        合并前先解决"矛盾识别"：位置高度重合但文本不同的块，是 OCR 对
        同一处文字的两种读法（典型如 ``Cancel`` / ``Cancer``、``l`` / ``1``）。
        直接把两者拼接会在屏幕上渲染出"Cancel Cancer"这种明显的错误，
        因此只保留置信度更高的那个读法。
        """
        if len(group) == 1:
            return group[0]

        resolved = self._resolve_contradictions(group)
        kept_ids = {id(block) for block in resolved}
        for rejected in group:
            if id(rejected) in kept_ids:
                continue
            kept = next((block for block in resolved
                         if block.box.intersection(rejected.box).area
                         / (min(block.area, rejected.area) or 1) >= .5), None)
            if kept is not None:
                self._record("conflict" if kept.text != rejected.text else "deduplicate",
                             "higher_confidence_overlap", [rejected, kept], kept)
        if len(resolved) == 1:
            return resolved[0]

        box = resolved[0].box
        for block in resolved[1:]:
            box = box.union(block.box)

        text = _assemble_inline([block.text for block in _order_line_blocks(resolved)])
        confidence = sum(block.confidence for block in resolved) / len(resolved)

        result = TextBlock(
            text=text,
            box=box,
            confidence=confidence,
            index=resolved[0].index,
            line_id=resolved[0].line_id,
            merged_from=[block.index for block in resolved],
            ocr_uncertain=any(block.ocr_uncertain for block in resolved),
        )
        self._source_records[id(result)] = [
            item for block in resolved for item in self._sources(block)
        ]
        self._record("merge", "same_visual_row", resolved, result)
        return result

    @staticmethod
    def _resolve_contradictions(group: list[TextBlock]) -> list[TextBlock]:
        """剔除位置重合但读法冲突的文本块，每组只保留置信度最高者。"""
        resolved: list[TextBlock] = []

        for block in group:
            merged = False
            for index, kept in enumerate(resolved):
                overlap = kept.box.intersection(block.box)
                smaller = min(kept.box.area, block.box.area) or 1
                # 显著重叠即视为"同一处文字"
                if overlap.area / smaller < 0.5:
                    continue

                if kept.text == block.text:
                    # 同一读法：保留置信度更高的
                    if block.confidence > kept.confidence:
                        resolved[index] = block
                    merged = True
                    break

                # 同一处文字的不同读法：保留置信度更高的，不拼接
                if block.confidence > kept.confidence:
                    resolved[index] = block
                merged = True
                break

            if not merged:
                resolved.append(block)

        return resolved

    # ------------------------------------------------------------------
    # 5. 段落划分
    # ------------------------------------------------------------------

    def _group_paragraphs(self, lines: list[list[TextBlock]]) -> list[Paragraph]:
        """把文本行按垂直间距与语义连贯性划分为段落。

        判定为同一段落需要同时满足三个条件，缺一不可。这些条件共同回答
        一个问题："这两行是被排版折行的同一句话，还是界面上两个独立的元素？"

        1. **垂直间距足够小**——折行的行距是紧凑的；而界面上两个独立元素
           （列表项、菜单项、设置行）的间距明显更大；
        2. **上一行没有句末标点**——以句号/问号/感叹号结尾说明语义已完整；
        3. **上一行的宽度与下一行相当**——这条最关键也最容易被忽略：
           折行的上一行几乎占满行宽，而"Settings" 这种短标签后面跟着
           一整句说明文字时，两行宽度差异极大，显然不是同一句话。

        第 3 条是实测补充的：缺少它时，"Settings" 会被与下面一整句
        "Battery level is low..." 合并成一个段落，导致译文变为一个覆盖
        两处位置的大块，**丢失了逐元素的原位对照**——而原位对照正是本
        产品的核心价值。
        """
        if not lines:
            return []

        # 同时维护**多条**未闭合的段落链，而不是只有一条。
        #
        # 为什么：桌面常有并排的多栏，它们的文本行在垂直方向交替出现。
        # 若只与"上一条处理过的行"比较，右栏的行会插在左栏两行之间，
        # 把左栏的段落链打断（实测侧栏列表的相邻两行被拆成两段）。
        # 因此为每个仍在延续的段落保留"最后一行"，新行与**任意**一条链
        # 匹配就并入该链，从而实现按栏独立成段。
        chains: list[list[TextBlock]] = []
        chain_tails: list[TextBlock] = []

        for line in lines:
            if not line:
                continue

            target = -1
            for index, tail in enumerate(chain_tails):
                extend, reason = self._paragraph_decision(tail, line[0])
                if extend:
                    self._record("merge", "wrapped_rows", [tail, line[0]])
                    target = index
                    break
                # Only nearby candidates are meaningful rejected merge
                # decisions; do not create quadratic full-page audit payloads.
                gap = line[0].box.top - tail.box.bottom
                if (0 <= gap <= self.paragraph_gap_ratio * max(tail.height, line[0].height)
                        and _horizontally_overlapping(tail.box, line[0].box)):
                    self._record("retain", reason, [tail, line[0]])

            if target >= 0:
                chains[target].extend(line)
                chain_tails[target] = line[-1]
            else:
                chains.append(list(line))
                chain_tails.append(line[-1])

        # 按首行垂直位置排序，恢复阅读顺序
        ordered = sorted(chains, key=lambda chain: (chain[0].box.top, chain[0].box.left))
        paragraphs = [self._build_paragraph(chain) for chain in ordered]

        for index, paragraph in enumerate(paragraphs):
            for block in paragraph.blocks:
                block.paragraph_id = index

        return paragraphs

    def _should_extend_paragraph(self, previous_line: TextBlock, next_line: TextBlock) -> bool:
        """判断新的一行是否应并入 ``previous_line`` 所在的段落。

        需要同时满足：

        1. **垂直间距足够小**——折行的行距紧凑；界面上两个独立元素的间距明显更大；
        2. **水平同栏**——并排窗口的行水平上互不相交（见
           :func:`_horizontally_overlapping`）；
        3. **语义连贯**——上一行不以句末标点结尾（见
           :const:`_TERMINAL_PUNCTUATION`）；
        4. **宽度相当**——折行的上一行几乎占满行宽，而短标签后面跟着
           一整句说明文字时两行宽度差异极大（见 :func:`_looks_like_wrapped_line`）。
        """
        return self._paragraph_decision(previous_line, next_line)[0]

    def _paragraph_decision(self, previous_line: TextBlock,
                            next_line: TextBlock) -> tuple[bool, str]:
        gap = next_line.box.top - previous_line.box.bottom
        reference_height = max(previous_line.height, next_line.height) or 1
        if next_line.box.top <= previous_line.box.top:
            return False, "same_or_previous_visual_row"
        if gap / reference_height > self.paragraph_gap_ratio:
            return False, "paragraph_gap"

        if not _horizontally_overlapping(previous_line.box, next_line.box):
            return False, "different_column"

        if not self.merge_broken_lines:
            return False, "wrap_merge_disabled"

        # Missing a period is not evidence that two independently spaced
        # labels form a wrapped sentence. Require the same tight geometry for
        # comparable-width rows as for a short final wrapped row.
        if gap / reference_height > TIGHT_GAP_RATIO:
            return False, "visual_row_gap"

        previous_text = previous_line.text.rstrip()
        if previous_text and previous_text[-1] in _TERMINAL_PUNCTUATION:
            return False, "terminal_punctuation"

        extends = _looks_like_wrapped_line(previous_line, next_line)
        return extends, "wrapped_rows" if extends else "incompatible_row_geometry"

    def _build_paragraph(self, blocks: list[TextBlock]) -> Paragraph:
        """把一组文本块组装为 :class:`Paragraph`。"""
        box = blocks[0].box
        for block in blocks[1:]:
            box = box.union(block.box)

        line_ids = sorted({block.line_id for block in blocks})
        text = self._assemble_text(blocks)

        # 按面积加权，长块对整体置信度影响更大
        total_area = sum(block.area for block in blocks) or 1
        confidence = (
            sum(block.confidence * block.area for block in blocks) / total_area
        )

        guess = detect_language(text, self.fallback_language)

        return Paragraph(
            text=text,
            box=box,
            blocks=list(blocks),
            confidence=confidence,
            language=guess.code,
            line_count=max(1, len(line_ids)),
            source_lines=[item for block in blocks for item in self._sources(block)],
        )

    # ------------------------------------------------------------------
    # 6. 上下文拼接与断句修复
    # ------------------------------------------------------------------

    def _assemble_text(self, blocks: list[TextBlock]) -> str:
        """把段落内各文本块拼接为通顺文本。

        拼接规则区分同行与跨行：

        - **同行**：按水平顺序直接连接，拉丁字母之间补空格；
        - **跨行**：若上一行以连字符结尾，去除连字符直接连接（英文断词修复）；
          若上一行以句末标点结尾，换行处按句号处理；
          否则拉丁字母之间补空格，CJK 之间不补空格。
        """
        by_line: dict[int, list[TextBlock]] = {}
        for block in blocks:
            by_line.setdefault(block.line_id, []).append(block)
        ordered = [block for line_id in sorted(by_line)
                   for block in _order_line_blocks(by_line[line_id])]
        if not ordered:
            return ""

        result = ordered[0].text.strip()

        for index in range(1, len(ordered)):
            previous_block = ordered[index - 1]
            block = ordered[index]
            text = block.text.strip()
            if not text:
                continue

            if block.line_id == previous_block.line_id:
                result = _assemble_inline([result, text])
            else:
                result = self._join_lines(result, text)

        return _collapse_whitespace(result)

    def _join_lines(self, previous: str, current: str) -> str:
        """跨行拼接。"""
        if not previous:
            return current

        # 英文断词修复：``inter-`` + ``face`` → ``interface``
        # 仅当连字符前是字母时才视为断词；``--`` 或 ``- `` 等保持原样
        if (
            self.repair_hyphenation_enabled
            and previous.endswith("-")
            and len(previous) >= 2
            and previous[-2].isalpha()
        ):
            return previous[:-1] + current

        # 当前行以行首禁排标点开头（如中文逗号句号），直接连接不补空格
        if current and current[0] in _LEADING_PUNCTUATION:
            return previous + current

        if _needs_space_between(previous, current):
            return previous + " " + current

        return previous + current


# ----------------------------------------------------------------------
# 模块级辅助函数
# ----------------------------------------------------------------------


def _source_snapshot(block: TextBlock) -> dict:
    return {"index": block.index, "line_id": block.line_id, "text": block.text,
            "rect": list(block.box.to_sequence()), "confidence": block.confidence}


def _order_line_blocks(blocks: list[TextBlock]) -> list[TextBlock]:
    """Restore geometric word order for predominantly RTL visual lines.

    OCR already returns logical characters inside each block. Only block order
    changes here. Latin and numeric runs keep their left-to-right block order,
    and distinct visual lines remain top-to-bottom. The direction is inferred
    from the recognized Unicode text, never from the translation target.
    """
    visual = sorted(blocks, key=lambda block: block.box.left)
    counts = {"rtl": 0, "ltr": 0}
    for block in visual:
        for char in block.text:
            bidi = unicodedata.bidirectional(char)
            if bidi in {"R", "AL"}:
                counts["rtl"] += 1
            elif bidi == "L":
                counts["ltr"] += 1
    if counts["rtl"] <= counts["ltr"]:
        return visual

    runs: list[list[TextBlock]] = []
    directions: list[bool] = []
    for block in visual:
        # R/AL blocks follow the RTL base. A number/Latin-only block is an
        # embedded LTR run, including Arabic-Indic EN/AN digit blocks.
        rtl = any(unicodedata.bidirectional(char) in {"R", "AL"} for char in block.text)
        if runs and directions[-1] == rtl:
            runs[-1].append(block)
        else:
            runs.append([block])
            directions.append(rtl)
    ordered: list[TextBlock] = []
    for run, rtl in reversed(list(zip(runs, directions, strict=True))):
        ordered.extend(reversed(run) if rtl else run)
    return ordered


def repair_hyphenation(text: str) -> str:
    """修复被换行连字符拆开的英文单词。

    用于对已经拼接好的整段文本做二次修正，例如模型输出或外部导入的文本。
    """
    import re

    # 行内 ``word- word`` 形式（连字符后紧跟换行被替换为空格的情况）
    return re.sub(r"(?<=[A-Za-z])-\s+(?=[a-z])", "", text)


def join_segments(segments: list[str], cjk: bool = False) -> str:
    """按语义拼接多个片段。

    :param segments: 片段列表。
    :param cjk: 是否为 CJK 文本（决定是否补空格）。
    """
    cleaned = [segment.strip() for segment in segments if segment and segment.strip()]
    if not cleaned:
        return ""
    if cjk:
        return "".join(cleaned)

    result = cleaned[0]
    for segment in cleaned[1:]:
        if _needs_space_between(result, segment):
            result += " " + segment
        else:
            result += segment
    return result


def _assemble_inline(parts: list[str]) -> str:
    """同一行内按顺序连接文本。

    若前一段以空格结尾、后一段以空格开头，则只保留一个空格。
    """
    if not parts:
        return ""

    result = parts[0]
    for part in parts[1:]:
        if not result:
            result = part
            continue
        if not part:
            continue

        if result.endswith(" ") or part.startswith(" "):
            result += part
        elif _needs_space_between(result, part):
            result += " " + part
        else:
            result += part

    return result


def _needs_space_between(left: str, right: str) -> bool:
    """判断连接两段文本时是否需要补空格。

    规则：两侧都是拉丁字母/数字时需要空格；只要涉及 CJK 就不需要。
    """
    if not left or not right:
        return False

    left_char = left[-1]
    right_char = right[0]

    if left_char.isspace() or right_char.isspace():
        return False

    # CJK 与任何字符之间都不加空格
    if _is_cjk(left_char) or _is_cjk(right_char):
        return False

    # 标点后不加空格（句首标点等）
    if right_char in _LEADING_PUNCTUATION:
        return False

    left_is_word = left_char.isalnum()
    right_is_word = right_char.isalnum()
    return left_is_word and right_is_word


def _is_cjk(text: str) -> bool:
    """判断字符串是否包含 CJK 字符。"""
    return any(
        start <= ord(ch) <= end for ch in text for start, end in _CJK_RANGES
    )


def _is_pure_numeric(text: str) -> bool:
    """判断是否为纯数字（允许千分位、小数点、百分号、单位符号）。"""
    stripped = text.strip()
    if not stripped:
        return False

    allowed = set("0123456789.,:%+-*/()[]{}#$￥¥€°'\" ")
    return all(ch in allowed for ch in stripped)


def _vertical_overlap_ratio(a: Rect, b: Rect) -> float:
    """计算两个矩形在垂直方向的重叠比例（相对较矮者）。"""
    overlap_top = max(a.top, b.top)
    overlap_bottom = min(a.bottom, b.bottom)
    overlap = overlap_bottom - overlap_top
    if overlap <= 0:
        return 0.0
    shorter = min(a.height, b.height) or 1
    return overlap / shorter


def _horizontally_overlapping(a: Rect, b: Rect) -> bool:
    """判断两个矩形在水平方向是否属于同一栏（x 区间投影有显著重叠）。

    :param a: 上一行的包围盒。
    :param b: 下一行的包围盒。
    :return: 重叠宽度达到较窄一方宽度的 20% 以上时返回 ``True``。

    用于防止把并排窗口的文字行拼成同一段：真正的折行续行与上一行共享
    同一栏宽，水平投影必然大幅重叠；而分属不同窗口的行在水平上互不相交。
    阈值取 20% 而不是 0，是为了容忍行末标点、缩进等造成的轻微错位。

    .. note::
       这里必须**只比较 x 区间**，不能复用 :meth:`Rect.intersection`——
       那是二维求交，而上下相邻的两行在竖直方向本来就不重叠，
       二维求交会返回宽度 0，导致"同栏"判断恒为假、折行永远拼不回去。
    """
    overlap = min(a.right, b.right) - max(a.left, b.left)
    if overlap <= 0:
        return False
    narrower = min(a.width, b.width) or 1
    return overlap / narrower >= 0.2


#: 上一行宽度达到下一行的该倍数以上时，视为"宽行在上"的强不对齐。
#: 真正的折行续行不会比上一行窄太多（段落右边界参差有限），而横贯整窗的
#: 标题/横幅下方挂着另一栏的短元素（侧栏菜单项、按钮）时比例轻松超过 2 倍
#: （实测跨栏标题 580px 对右栏首行 263px 即达 2.2 倍——阈值取 3.0 时保护
#: 不触发，标题被并入右栏段，文字栏语序错乱）。
MUCH_WIDER_RATIO = 2.0

#: "宽行在上"时，续行允许超出上一行左右边缘的容差（相对续行自身宽度）。
#: 左对齐排版的续行绝不会伸到上一行左边缘之外；容差只用于容忍 OCR 包围盒抖动。
CONTAINMENT_TOLERANCE = 0.1

#: "宽行在上"时要求的行距上限（相对行高）。折行的行距是紧凑的；
#: 宽行与窄行之间留出明显空隙，说明窄行是界面上的另一个独立元素。
TIGHT_GAP_RATIO = 0.8


def _looks_like_wrapped_line(previous_line: TextBlock, next_line: TextBlock) -> bool:
    """判断"下一行"是否像是"上一行"被排版折行后的续行。

    依据是宽度：折行时上一行几乎占满可用行宽，因此它的宽度通常**大于等于**
    下一行；反过来，短标签（"Settings"、"确定"）后面跟着一整句说明文字时，
    两行宽度差异极大，属于界面上两个独立元素。

    阈值取 0.6：只要上一行宽度达到下一行的 60%，就认为它可能在同一次
    折行里。这个值刻意偏宽松——**误合并的代价（丢失原位对照）比漏合并的
    代价（多一次翻译请求）大得多**，因为前者用户直接看得出来，
    后者只影响速度。

    **宽行在上的反向保护**：上一行比下一行宽得多（≥ :data:`MUCH_WIDER_RATIO`
    倍）时，单看"上一行更宽"已无法区分两种局面——真折行的末行可以很短
    （"…over the lazy" / "dog."），而横贯整窗的标题下方挂着另一栏的短元素
    时上一行同样远宽于下一行（实测"Settings - General Preferences"标题与
    左侧栏的"Dashboard"被并成一段，原位译文变成横跨两者的大块）。此时
    额外要求续行的两个特征，缺一不可：

    1. **行距紧凑**（≤ :data:`TIGHT_GAP_RATIO` 倍行高）——折行的行距紧凑，
       独立元素之间通常留有明显空隙；
    2. **左右边缘对齐**（容差 :data:`CONTAINMENT_TOLERANCE`）——左对齐排版的
       续行不会伸到上一行左边缘之外；侧栏项往往比横贯标题的起点更靠左。
    """
    previous_width = previous_line.width or 1
    next_width = next_line.width or 1
    ratio = previous_width / next_width
    if ratio < MUCH_WIDER_RATIO:
        return ratio >= 0.6

    gap = next_line.box.top - previous_line.box.bottom
    reference_height = max(previous_line.height, next_line.height) or 1
    if gap / reference_height > TIGHT_GAP_RATIO:
        return False

    tolerance = next_width * CONTAINMENT_TOLERANCE
    return (
        next_line.box.left >= previous_line.box.left - tolerance
        and next_line.box.right <= previous_line.box.right + tolerance
    )


def _previous_line_id(ordered: list[TextBlock], block: TextBlock) -> int:
    """返回 ``block`` 之前那个块的行号。

    仅供外部调试与测试使用；主流程使用下标遍历以避免按值比较带来的歧义。
    """
    for index, candidate in enumerate(ordered):
        if candidate is block:
            if index == 0:
                return block.line_id
            return ordered[index - 1].line_id
    return block.line_id


def _collapse_whitespace(text: str) -> str:
    """压缩连续空白，去除首尾空白。"""
    return " ".join(text.split())
