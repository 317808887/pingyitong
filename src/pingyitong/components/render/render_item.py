"""渲染数据模型。

渲染层只消费本模块定义的数据结构，不感知 OCR、翻译与调度的存在，
这样各层可以独立替换与测试（见开发文档一、分层解耦要求）。

坐标约定
    :class:`RenderItem` 的 ``box`` 使用**物理坐标**；
    渲染时由 :class:`~pingyitong.core.coordinate.CoordinateMapper`
    折算为对应显示器的逻辑坐标。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from ..capture.geometry import Rect


@dataclass
class RenderItem:
    """一个待渲染的译文条目。"""

    #: 原文包围盒（物理坐标，全局）
    box: Rect
    #: 译文内容
    text: str
    #: 原文内容，用于对照模式与日志
    source_text: str = ""
    #: 渲染字号（逻辑像素）。``None`` 表示由渲染层按包围盒自动匹配
    font_px: int | None = None
    #: 原文区域是否为深色背景，用于浅色/深色界面自适应
    is_dark_background: bool = False
    #: 是否命中推理缓存
    from_cache: bool = False
    #: 该条目的来源显示器序号，用于多屏分发
    monitor_index: int | None = None
    #: 仅原位显示使用的保义简写；text 始终保存完整译文。
    display_text: str = ""
    quality_status: str = "accepted"
    quality_reason: str = ""
    ocr_uncertain: bool = False

    @property
    def width(self) -> int:
        return self.box.width

    @property
    def height(self) -> int:
        return self.box.height

    @property
    def is_renderable(self) -> bool:
        """内容与几何都有效才渲染。"""
        return bool(self.text.strip()) and not self.box.is_empty


@dataclass
class RenderFrame:
    """一帧完整的渲染指令。"""

    #: 对应捕获帧序号
    seq: int
    #: 本帧在虚拟桌面中的物理位置
    origin: Rect
    #: 待渲染条目（物理坐标）
    items: list[RenderItem] = field(default_factory=list)
    #: 是否来自整帧识别。整帧结果可以整体替换上一帧内容
    is_full_frame: bool = True
    #: 本帧各阶段的耗时统计（秒），用于状态面板与性能压测
    timings: dict[str, float] = field(default_factory=dict)
    #: 本次实际重新识别的区域（全局物理坐标），包括已变为空白的区域。
    changed_regions: list[Rect] = field(default_factory=list)
    #: 本轮真实识别到的段落原文；None 为旧调用方未提供此证据。
    source_items: list[RenderItem] | None = None
    #: 有原文但尚无合格译文的区域，允许流水线有限次数补译。
    retry_regions: list[Rect] = field(default_factory=list)
    #: 各处理阶段的处置计数，不能由最终条目数反推 OCR 完整率。
    diagnostics: dict[str, int] = field(default_factory=dict)

    @property
    def item_count(self) -> int:
        return len(self.items)

    @property
    def is_empty(self) -> bool:
        return not any(item.is_renderable for item in self.items)

    def deduplicate(self) -> None:
        """就地去除几何完全重合的条目。

        差分区域重叠时可能产生重复渲染，出现文字重影。
        """
        seen: set[tuple[str, int, int, int, int]] = set()
        unique: list[RenderItem] = []
        for item in self.items:
            signature = (item.text, item.box.x, item.box.y, item.box.width, item.box.height)
            if signature in seen:
                continue
            seen.add(signature)
            unique.append(item)
        self.items = unique

    def visible_items(self) -> list[RenderItem]:
        """返回可渲染的条目。"""
        return [item for item in self.items if item.is_renderable]


def merge_items(
    existing: list[RenderItem],
    incoming: list[RenderItem],
    dirty_regions: list[Rect] | None = None,
) -> list[RenderItem]:
    """把增量识别到的条目并入已有条目。

    画面差分只把"变化的区域"送去识别，因此增量帧里的条目只覆盖屏幕的一小块。
    与重新识别区域或新条目几何相交的旧条目丢弃，其余原样保留。
    重新识别区域即使已经变为空白，也必须清掉该处的旧译文。

    **这是合并规则的唯一实现**：翻译层绘制与实时文本框都调用它。
    两处各写一遍迟早会漂移——本项目已经因为"同一条规则写两份"出过问题
    （坐标换算、区域换算各有一份实现，只在原点为 0 时才碰巧一致）。
    """
    if not existing:
        return list(incoming)
    if not incoming and not dirty_regions:
        return list(existing)

    kept = [
        old
        for old in existing
        if not any(old.box.intersects(region) for region in dirty_regions or [])
        and not any(old.box.intersects(new.box) for new in incoming)
    ]
    kept.extend(incoming)
    return kept


def retain_confirmed_items(
    existing: list[RenderItem],
    incoming: list[RenderItem],
    source_items: list[RenderItem] | None,
) -> list[RenderItem]:
    """补译失败时，仅保留本轮再次确认原文相同且原位的已有译文。

    空白、变字、移动和旧调用方都不能靠时间保留旧文字，避免残影。
    """
    result = list(incoming)
    if not source_items:
        return result
    available = list(existing)
    for source in source_items:
        if any(item.box.intersects(source.box) for item in result):
            continue
        source_text = " ".join((source.source_text or source.text).split())
        for index, old in enumerate(available):
            if not old.source_text or " ".join(old.source_text.split()) != source_text:
                continue
            overlap = old.box.intersection(source.box).area
            if overlap < max(old.box.area, source.box.area) * 0.75:
                continue
            result.append(replace(old, box=source.box))
            available.pop(index)
            break
    return result
