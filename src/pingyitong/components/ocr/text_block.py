"""文本块数据结构。

OCR 输出的最小单元，承载文本内容、包围盒坐标与置信度，
并记录后续被合并/分段的结果（行号、段落号、来源块）。

坐标一律使用**物理坐标**（相对虚拟桌面左上角），
渲染前再由坐标映射模块统一折算为逻辑坐标。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from ..capture.geometry import Rect


@dataclass
class TextBlock:
    """一个被识别出的文本块。"""

    #: 识别出的文字内容（原始，未清洗）
    text: str
    #: 包围盒，物理坐标
    box: Rect
    #: 识别置信度，0-1
    confidence: float = 1.0
    #: 帧内序号，用于稳定排序
    index: int = 0
    #: 文本行分组标识，由后处理阶段填充
    line_id: int = -1
    #: 段落分组标识，由后处理阶段填充
    paragraph_id: int = -1
    #: 是否为多个原始块合并而成，记录来源序号
    merged_from: list[int] = field(default_factory=list)
    ocr_uncertain: bool = False

    # ------------------------------------------------------------------
    # 派生属性
    # ------------------------------------------------------------------

    @property
    def height(self) -> int:
        return self.box.height

    @property
    def width(self) -> int:
        return self.box.width

    @property
    def center(self) -> tuple[int, int]:
        return self.box.center

    @property
    def area(self) -> int:
        return self.box.area

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    # ------------------------------------------------------------------
    # 坐标变换
    # ------------------------------------------------------------------

    def to_local(self, origin: Rect) -> TextBlock:
        """转换为相对 ``origin`` 的局部坐标副本。"""
        return TextBlock(
            text=self.text,
            box=self.box.to_local(origin),
            confidence=self.confidence,
            index=self.index,
            line_id=self.line_id,
            paragraph_id=self.paragraph_id,
            merged_from=list(self.merged_from),
            ocr_uncertain=self.ocr_uncertain,
        )

    def to_global(self, origin: Rect) -> TextBlock:
        """将局部坐标还原为全局物理坐标副本。"""
        return TextBlock(
            text=self.text,
            box=self.box.to_global(origin),
            confidence=self.confidence,
            index=self.index,
            line_id=self.line_id,
            paragraph_id=self.paragraph_id,
            merged_from=list(self.merged_from),
            ocr_uncertain=self.ocr_uncertain,
        )

    def with_text(self, text: str, box: Rect | None = None) -> TextBlock:
        """返回替换文字（可同时替换包围盒）后的副本。"""
        return TextBlock(
            text=text,
            box=box if box is not None else self.box,
            confidence=self.confidence,
            index=self.index,
            line_id=self.line_id,
            paragraph_id=self.paragraph_id,
            merged_from=list(self.merged_from),
            ocr_uncertain=self.ocr_uncertain,
        )

    # ------------------------------------------------------------------
    # 缓存键
    # ------------------------------------------------------------------

    def cache_key(self, extra: str = "") -> str:
        """生成用于推理缓存的键。

        仅与文本内容（及可选附加标识）有关，与坐标无关——
        界面中相同文字出现在不同位置时应命中同一缓存。
        """
        payload = f"{self.text}\x1f{extra}"
        return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()

    def __str__(self) -> str:  # pragma: no cover - 仅用于调试输出
        preview = self.text if len(self.text) <= 24 else f"{self.text[:24]}…"
        return (
            f"TextBlock({preview!r} @{self.box.to_ltrb()} "
            f"conf={self.confidence:.2f} line={self.line_id})"
        )


@dataclass
class RecognizedFrame:
    """一帧的 OCR 结果，在流水线中由 OCR 阶段传递给预处理与翻译阶段。"""

    #: 对应捕获帧的序号
    seq: int
    #: 该帧在虚拟桌面中的物理位置
    origin: Rect
    #: 识别出的原始文本块（全局物理坐标）
    blocks: list[TextBlock] = field(default_factory=list)
    #: 识别耗时（秒）
    elapsed: float = 0.0
    #: 是否为整帧识别
    is_full_frame: bool = True
    #: 来源显示器索引
    monitor_index: int | None = None
    #: 本次重新识别区域（全局物理坐标），空白区域也需要传给渲染层。
    changed_regions: list[Rect] = field(default_factory=list)
    diagnostics: dict[str, int] = field(default_factory=dict)
    #: 当前识别中涉及合并、过滤、低对比恢复等操作的逐项原因。
    decision_events: list[dict] = field(default_factory=list)

    @property
    def texts(self) -> list[str]:
        """返回全部文本内容。"""
        return [block.text for block in self.blocks]

    def __len__(self) -> int:
        return len(self.blocks)
