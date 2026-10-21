"""阅读顺序重排（自研）。

对应需求："文本框里显示的文字按界面中**从上到下、从左到右**的顺序排列，
以确保语义流畅"。

**为什么不能简单地按 y 坐标排序**：屏幕上同时开着侧边栏、主内容区、
悬浮面板时，同一水平带上的元素分属不同区域。若只按 ``(y, x)`` 排，
侧边栏的一行会和主内容区的一行交替出现，读起来是支离破碎的
（例如"设置 / The quick brown fox / 历史 / jumps over"）。

因此这里采用经典的 **XY-cut** 思路（一种长期公开的版面分析算法，
非某个库的实现）的改进版，每一层分三步：

1. **先剥离横贯元素**——独处顶端/底端、x 跨度恰好堵住剩余条目全部
   栏空隙的条目（顶部横贯标题、底部页脚）先摘出来：不摘的话 x 向
   永远没有空隙，"侧栏 + 主区"的栏结构永远切不出来；
2. 剩余部分在"竖切分栏"与"横切分段"之间竞争——**栏结构成立时优先
   分栏**，行间隙达到 :data:`ROW_GAP_PRIORITY_RATIO` 倍栏间隙才改按
   行切（页脚与正文之间的巨大空隙靠这条胜出）；
3. 两种切分都不成立时，退化为行带分组，行内按左到右。

只做"先分栏、后分行"会在两类极常见的桌面版面上翻车（均实测踩过）：

1. **页脚**：横贯底部的状态栏文字与左栏在同一竖直区间内，纯竖切会把它
   并进左栏，文本框里页脚出现在正文中间；
2. **顶部横贯标题**：与上方所有内容之间的大空隙是最强的结构信号，
   应先把标题单独切成一段，再对剩余部分分栏。

桌面界面最常见的"左侧栏 + 右侧主区"结构则仍由竖切胜出（栏间距远大于
行间距），行为与旧版一致。

关于实现来源
    调研过 GitHub 上的现成方案，均不适合本项目：

    - ``axa-group/Parsr``（Apache-2.0）的 ReadingOrderDetectionModule
      是 TypeScript 写的**独立服务**，要跑 Docker 容器；为一个排序算法
      给离线桌面程序塞一个 Node 服务不合理；
    - ``layoutreader``（LayoutLMv3）需要 ``transformers`` 与约 500MB 模型，
      与"全离线、低延迟"的目标冲突；
    - ``kraken`` 是完整 OCR 引擎，与项目已有的 RapidOCR 重复。

    故按经典算法自行实现。**因为不是取自 GitHub 的组件，
    项目文档（PPT/Word）无需新增来源标注。**
"""

from __future__ import annotations

from dataclasses import dataclass

from ..components.capture.geometry import Rect
from .text_preprocess import _order_line_blocks

#: 判定"两栏之间"的空隙下限，相对行高。屏幕上的栏间距通常远大于字距，
#: 取 1.4 倍行高可以避开正常的词间空隙。
DEFAULT_COLUMN_GAP_RATIO = 1.4

#: 判定"上下两段之间"的空隙下限，相对行高。与栏间距同量级：
#: 段落间距（预处理里约 1.7 倍行高）可能达到这个值，但只有当它
#: **同时宽于任何栏间距**时才会赢下方向选择，因此不会拆散并排的栏。
DEFAULT_ROW_GAP_RATIO = 1.4

#: 同一行内判定"垂直重叠"的比例下限（相对较矮一方的高度）。
DEFAULT_ROW_OVERLAP_RATIO = 0.5

#: 存在栏结构时，行间隙必须达到栏间隙的该倍数，才能改按行切分。
#: 依据：桌面界面的"侧栏 + 主区"双栏结构极其稳定（栏间隙 ≥ 1.4 倍行高
#: 就足以成立），而栏内的垂直间隙（菜单项间距、段落间距）与栏间隙同量级
#: 是常态——若两个方向空隙相近时仍按"谁宽听谁的"，稀疏侧栏会被主区的
#: 段落间隙反复打断，文字框里侧栏项与正文交错出现（实测复现）。
#: 只有页脚/横幅这类**远超**栏间隙的水平空隙（实测中 ≥ 3 倍以上）
#: 才应该赢过栏结构。因此行间隙需达到栏间隙的 2.5 倍才改按行切。
ROW_GAP_PRIORITY_RATIO = 2.5


@dataclass
class _Boxed:
    """带位置的条目包装，使算法与具体的条目类型解耦。"""

    box: Rect
    payload: object

    @property
    def text(self) -> str:
        # Translate into English without changing the source layout's reading
        # direction. Never infer layout from the user's selected target.
        source = str(getattr(self.payload, "source_text", "") or "")
        return source if source.strip() else str(getattr(self.payload, "text", "") or "")


@dataclass
class _Column:
    entries: list[_Boxed]

    @property
    def text(self) -> str:
        return " ".join(entry.text for entry in self.entries)


def order_for_reading(
    items: list[object],
    column_gap_ratio: float = DEFAULT_COLUMN_GAP_RATIO,
    row_gap_ratio: float = DEFAULT_ROW_GAP_RATIO,
    row_overlap_ratio: float = DEFAULT_ROW_OVERLAP_RATIO,
) -> list[object]:
    """把条目按阅读顺序排列。

    :param items: 任何具有 ``box`` 属性（:class:`Rect`）的对象。
    :param column_gap_ratio: 判定分栏的空隙下限，相对行高。
    :param row_gap_ratio: 判定分段的空隙下限，相对行高。
    :param row_overlap_ratio: 判定同行的垂直重叠比例下限。
    :return: 重排后的新列表；输入为空时返回空列表。
    """
    if not items:
        return []

    boxed = [
        _Boxed(box=item.box, payload=item)
        for item in items
        if getattr(item, "box", None) is not None
    ]
    if len(boxed) <= 1:
        return [entry.payload for entry in boxed]

    ordered = _order_recursive(boxed, column_gap_ratio, row_gap_ratio, row_overlap_ratio)
    return [entry.payload for entry in ordered]


def reading_text(
    items: list[object],
    separator: str = "\n",
    strip: bool = True,
) -> str:
    """按阅读顺序把条目的文本拼成一段可读文字。

    只取 ``text`` 字段（译文）。空条目会被跳过——文本框里出现空行既无意义
    又会让行距忽大忽小。
    """
    texts: list[str] = []
    for item in order_for_reading(items):
        text = str(getattr(item, "text", "") or "")
        if strip:
            text = text.strip()
        if text:
            if getattr(item, "quality_status", "accepted") == "uncertain":
                text = "[翻译不确定，保留原文] " + text
            elif getattr(item, "ocr_uncertain", False):
                text = "[识别待核对] " + text
            texts.append(text)
    return separator.join(texts)


# ----------------------------------------------------------------------
# XY-cut
# ----------------------------------------------------------------------


def _order_recursive(
    entries: list[_Boxed],
    column_gap_ratio: float,
    row_gap_ratio: float,
    row_overlap_ratio: float,
) -> list[_Boxed]:
    """递归地做 XY-cut：先剥离横贯元素，再按"空隙更宽"的方向切。

    每层分三步：

    1. **剥离横贯元素**——独处顶端/底端、且x 向跨度恰好堵住剩余条目
       全部栏空隙的条目（横贯标题、页脚），先摘出来。它们不摘，
       x 向永远没有空隙，"侧栏 + 主区"的栏结构就永远切不出来；
    2. 剩余条目在"分栏"与"分行"之间竞争：栏结构成立时优先分栏，
       行间隙达到 :data:`ROW_GAP_PRIORITY_RATIO` 倍栏间隙才改按行
       （页脚与正文之间的巨大空隙靠这条胜出）；
    3. 两种切分都不成立时，退化为行带分组。
    """
    if len(entries) <= 1:
        return list(entries)

    # A widely spaced RTL visual line can look like three "columns" to XY-cut.
    # Its words are still one reading row; preserve LTR numeric/Latin subruns.
    if all(
        _same_row(entries[:index], entry, row_overlap_ratio)
        for index, entry in enumerate(entries[1:], start=1)
    ):
        return _order_line_blocks(entries)

    top, bottom, middle = _peel_spanning(entries, column_gap_ratio, row_gap_ratio)
    if len(middle) <= 1:
        ordered_middle: list[_Boxed] = list(middle)
    else:
        columns = _split_columns(middle, column_gap_ratio)
        rows = _split_rows(middle, row_gap_ratio)

        if _columns_preferred(columns, rows):
            ordered_middle = []
            column_entries = []
            for column in columns[0]:
                bounds = column[0].box
                for entry in column[1:]:
                    bounds = bounds.union(entry.box)
                # Reuse the same text-driven bidi rule for multirow columns.
                # Each column's lines retain their existing top-to-bottom order.
                column_entries.append(_Boxed(bounds, _Column(column)))
            for column_entry in _order_line_blocks(column_entries):
                column = column_entry.payload.entries
                ordered_middle.extend(
                    _order_recursive(column, column_gap_ratio, row_gap_ratio, row_overlap_ratio)
                )
        elif rows is not None:
            ordered_middle = []
            for band in rows[0]:
                ordered_middle.extend(
                    _order_recursive(band, column_gap_ratio, row_gap_ratio, row_overlap_ratio)
                )
        else:
            ordered_middle = _order_rows(middle, row_overlap_ratio)

    return top + ordered_middle + bottom


def _columns_preferred(
    columns: tuple[list[list[_Boxed]], float] | None,
    rows: tuple[list[list[_Boxed]], float] | None,
) -> bool:
    """给定两种切分候选，判断是否应当分栏。"""
    if columns is None:
        return False
    if rows is None:
        return True
    return columns[1] * ROW_GAP_PRIORITY_RATIO >= rows[1]


def _peel_spanning(
    entries: list[_Boxed],
    column_gap_ratio: float,
    row_gap_ratio: float,
) -> tuple[list[_Boxed], list[_Boxed], list[_Boxed]]:
    """反复剥离横贯元素，返回 ``(顶端条目, 底端条目, 剩余条目)``。

    "横贯元素"需同时满足：

    1. **独处边缘**——顶端条目在所有剩余条目之上（bottom ≤ 其余 top），
       底端条目在所有剩余条目之下；正因如此，把它放到最前/最后
       在任何正确的阅读顺序里都不会错；
    2. **堵住栏空隙**——把它拿掉之后剩余条目存在栏空隙，且它的 x 跨度
       恰好覆盖该空隙（标题横贯左右栏之间的间隙就是这个形态）；
    3. **剥离后剩余部分确实优先分栏**——否则维持原状走正常竞争
       （横贯元素被它下方的大空隙自然切开，行为与旧版一致）。
    """
    top: list[_Boxed] = []
    bottom: list[_Boxed] = []
    remaining = list(entries)

    changed = True
    while changed and len(remaining) > 1:
        changed = False
        peeled = _peel_one_edge(remaining, column_gap_ratio, row_gap_ratio, from_top=True)
        if peeled is not None:
            top.append(peeled)
            remaining.remove(peeled)
            changed = True
            continue
        peeled = _peel_one_edge(remaining, column_gap_ratio, row_gap_ratio, from_top=False)
        if peeled is not None:
            # 后剥离的底端条目位置更高，应排在先剥离者之前
            bottom.insert(0, peeled)
            remaining.remove(peeled)
            changed = True

    return top, bottom, remaining


def _peel_one_edge(
    entries: list[_Boxed],
    column_gap_ratio: float,
    row_gap_ratio: float,
    from_top: bool,
) -> _Boxed | None:
    """从顶端（或底端）找出一个可剥离的横贯条目；找不到返回 ``None``。"""
    if from_top:
        candidates = [
            entry
            for entry in entries
            if all(other.box.top >= entry.box.bottom for other in entries if other is not entry)
        ]
        candidates.sort(key=lambda entry: entry.box.top)
    else:
        candidates = [
            entry
            for entry in entries
            if all(other.box.bottom <= entry.box.top for other in entries if other is not entry)
        ]
        candidates.sort(key=lambda entry: -entry.box.bottom)

    for candidate in candidates:
        rest = [entry for entry in entries if entry is not candidate]
        if len(rest) <= 1:
            continue
        columns = _split_columns(rest, column_gap_ratio)
        if columns is None:
            continue
        if not _covers_widest_gap(candidate, rest, column_gap_ratio):
            continue
        rows = _split_rows(rest, row_gap_ratio)
        if _columns_preferred(columns, rows):
            return candidate
    return None


def _covers_widest_gap(
    candidate: _Boxed,
    rest: list[_Boxed],
    column_gap_ratio: float,
) -> bool:
    """判断候选条目的 x 跨度是否覆盖剩余条目之间的最宽栏空隙。"""
    line_height = _median_height(rest)
    if line_height <= 0:
        return False
    gap = _widest_gap_interval(rest, line_height * column_gap_ratio, axis="x")
    if gap is None:
        return False
    gap_start, gap_end = gap
    return candidate.box.left <= gap_start and candidate.box.right >= gap_end


def _widest_span_gap(
    entries: list[_Boxed],
    min_gap: float,
    axis: str,
) -> tuple[list[list[_Boxed]], float] | None:
    """在指定轴上找最宽空隙并把条目分成两侧；分不开时返回 ``None``。

    :param axis: ``"x"`` 按竖直空隙分栏（比较 ``left``/``right``），
        ``"y"`` 按水平空隙分段（比较 ``top``/``bottom``）。

    判据是"存在一段区间，没有任何条目跨越它"。为避免把同一行/同一栏里
    正常的间隙误判成结构分界，空隙宽度需达到行高的若干倍。
    """
    gap = _widest_gap_interval(entries, min_gap, axis)
    if gap is None:
        return None
    gap_start, gap_end = gap

    if axis == "x":
        starts = [entry.box.center[0] for entry in entries]
    else:
        starts = [entry.box.center[1] for entry in entries]

    cut = (gap_start + gap_end) / 2.0
    before = [
        entry for entry, center in zip(entries, starts, strict=False) if center < cut
    ]
    after = [
        entry for entry, center in zip(entries, starts, strict=False) if center >= cut
    ]
    if not before or not after:
        return None

    return [before, after], gap_end - gap_start


def _widest_gap_interval(
    entries: list[_Boxed],
    min_gap: float,
    axis: str,
) -> tuple[float, float] | None:
    """在指定轴上找最宽的结构空隙，返回 ``(起点, 终点)``；找不到返回 ``None``。

    判据与 :func:`_widest_span_gap` 相同：存在一段长度不小于 ``min_gap``
    的区间，没有任何条目跨越它。
    """
    if axis == "x":
        spans = sorted((entry.box.left, entry.box.right) for entry in entries)
    else:
        spans = sorted((entry.box.top, entry.box.bottom) for entry in entries)

    gaps: list[tuple[float, float]] = []
    current_end = spans[0][1]
    for start, end in spans[1:]:
        if start - current_end >= min_gap:
            gaps.append((current_end, start))
        current_end = max(current_end, end)

    if not gaps:
        return None

    return max(gaps, key=lambda pair: pair[1] - pair[0])


def _split_columns(
    entries: list[_Boxed],
    column_gap_ratio: float,
) -> tuple[list[list[_Boxed]], float] | None:
    """按竖直空隙把条目分成若干栏；分不开时返回 ``None``。

    判据是"存在一段横向区间，没有任何条目跨越它"。为避免把同一行里
    正常的词间空隙误判成栏间距，空隙宽度需达到行高的若干倍。
    """
    line_height = _median_height(entries)
    if line_height <= 0:
        return None
    min_gap = line_height * column_gap_ratio
    return _widest_span_gap(entries, min_gap, axis="x")


def _split_rows(
    entries: list[_Boxed],
    row_gap_ratio: float,
) -> tuple[list[list[_Boxed]], float] | None:
    """按水平空隙把条目分成上下若干段；分不开时返回 ``None``。

    顶部横贯的标题、底部横贯的页脚与正文之间的大空隙是最强的
    版面结构信号：先沿它们切开，剩下的部分再交给分栏逻辑。
    """
    line_height = _median_height(entries)
    if line_height <= 0:
        return None
    min_gap = line_height * row_gap_ratio
    return _widest_span_gap(entries, min_gap, axis="y")


def _order_rows(
    entries: list[_Boxed],
    row_overlap_ratio: float,
) -> list[_Boxed]:
    """把同一栏内的条目按行带分组，行内从左到右。"""
    by_top = sorted(entries, key=lambda entry: (entry.box.top, entry.box.left))

    rows: list[list[_Boxed]] = []
    for entry in by_top:
        placed = False
        for row in rows:
            if _same_row(row, entry, row_overlap_ratio):
                row.append(entry)
                placed = True
                break
        if not placed:
            rows.append([entry])

    rows.sort(key=lambda row: min(entry.box.top for entry in row))

    result: list[_Boxed] = []
    for row in rows:
        result.extend(_order_line_blocks(row))
    return result


def _same_row(row: list[_Boxed], entry: _Boxed, overlap_ratio: float) -> bool:
    """判断条目是否属于当前行带。

    与行内每个条目分别比较，而不是与所有条目的并集比较。
    高标题或逐步偏移的 OCR 框会把并集不断拉高，若只比较并集，
    不相交的上下两行也会被串成一行，随后按 x 排序就可能逆序。
    """
    for item in row:
        overlap = min(item.box.bottom, entry.box.bottom) - max(item.box.top, entry.box.top)
        shorter = min(item.box.height, entry.box.height) or 1
        if overlap <= 0 or overlap < shorter * overlap_ratio:
            return False
    return True


def _median_height(entries: list[_Boxed]) -> float:
    """取行高的中位数。

    用中位数而非均值：标题、图标一类元素的高度会明显偏离正文，
    均值容易被它们带偏。
    """
    heights = sorted(entry.box.height for entry in entries if entry.box.height > 0)
    if not heights:
        return 0.0
    middle = len(heights) // 2
    if len(heights) % 2:
        return float(heights[middle])
    return (heights[middle - 1] + heights[middle]) / 2.0
