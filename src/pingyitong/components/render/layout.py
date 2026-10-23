"""文本自适应排版（自研）。

对应《项目原创性声明》二·（三）"文本自适应换行与字号匹配"。

核心问题：译文的字符数往往与原文差异很大（中英互译通常 1:2），
若直接按原文包围盒绘制，会出现译文溢出遮挡相邻元素、或字号过小不可读。
本模块通过"按原文行高推定字号 → 按可读性下限约束 → 按包围盒宽度折行
→ 超出最大行数时回退缩小字号"的流程，在有限空间内取得可读性最优解。

排版全部使用 Qt 的 :class:`QTextLayout`，与最终绘制使用同一套断行规则，
因此"预估"与"实绘"结果一致，不会出现预估通过但实际溢出的情况。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QFont, QFontMetricsF, QTextLayout, QTextOption

from ...utils.logger import get_logger

logger = get_logger(__name__)

#: 自动字号模式下，字号相对原文行高的比例。
#: 略小于 1.0 是因为文字的视觉高度（x-height 到 cap-height）小于行高。
DEFAULT_HEIGHT_RATIO = 0.82

#: 每次迭代缩小的字号步长（像素）。
_FONT_STEP = 1


@dataclass
class TextLayout:
    """排版结果。"""

    #: 最终字号（像素）
    font_px: int
    #: 折行后的各行文本
    lines: list[str] = field(default_factory=list)
    #: 单行高度
    line_height: float = 0.0
    #: 全部行占用的总高度
    total_height: float = 0.0
    #: 最长一行的实测宽度（像素）。译文长度常与原文相差很大，
    #: 背景遮罩必须按它来收缩，否则短译文会被一个巨大的黑框包住。
    content_width: float = 0.0
    #: 是否因空间不足被截断
    truncated: bool = False
    #: 是否因为放不下而被缩小过字号
    shrunk: bool = False
    #: 连一整行也放不下时，只显示省略提示，不裁半截正常文字
    marker_only: bool = False
    #: Full candidate requirements, before the diagnostic-only elision below.
    needed_line_count: int = 0
    needed_height: float = 0.0
    overflow_reasons: list[str] = field(default_factory=list)

    @property
    def line_count(self) -> int:
        return len(self.lines)


def build_font(
    family: str,
    pixel_size: int,
    bold: bool = False,
) -> QFont:
    """构造用于排版与绘制的字体对象。

    ``family`` 为空时使用系统默认无衬线字体，避免中文缺字。
    """
    font = QFont()
    if family:
        font.setFamily(family)
    else:
        font.setStyleHint(QFont.StyleHint.SansSerif)
    font.setPixelSize(max(1, int(pixel_size)))
    font.setBold(bold)
    # 关闭字距微调，保证不同 DP 下的排版稳定
    font.setHintingPreference(QFont.HintingPreference.PreferFullHinting)
    return font


def wrap_text(text: str, font: QFont, max_width: float) -> list[str]:
    """按指定宽度对文本折行。

    :param text: 待排版文本。
    :param font: 使用的字体。
    :param max_width: 可用宽度（像素）。
    :return: 折行后的各行文本。
    """
    if not text:
        return []
    lines: list[str] = []
    # QTextLayout handles one paragraph; literal newlines must be split first.
    for paragraph in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if not paragraph or max_width <= 1:
            lines.append(paragraph)
            continue
        layout = QTextLayout(paragraph, font)
        option = QTextOption()
        option.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        layout.setTextOption(option)
        layout.beginLayout()
        # Qt indexes UTF-16 units, whereas Python indexes Unicode characters.
        encoded = paragraph.encode("utf-16-le")
        try:
            while True:
                line = layout.createLine()
                if not line.isValid():
                    break
                line.setLineWidth(max_width)
                start = line.textStart() * 2
                end = start + line.textLength() * 2
                lines.append(encoded[start:end].decode("utf-16-le"))
        finally:
            layout.endLayout()
    return lines


def line_height_for(font: QFont) -> float:
    """返回该字体在 Qt 度量下的行高。

    行高取 ascent + descent，而不是 ``lineSpacing()``：
    桌面界面文字往往需要压缩行距以避免遮挡相邻元素。
    """
    from PyQt6.QtGui import QFontMetricsF

    metrics = QFontMetricsF(font)
    return metrics.ascent() + metrics.descent()


def layout_text(
    text: str,
    box_width: float,
    box_height: float,
    style: object,
    colour_scheme_dark: bool = False,
    font_px: int | None = None,
    readable_min_px: int = 0,
) -> TextLayout:
    """为一个文本块计算排版方案。

    :param text: 译文内容。
    :param box_width: 可用宽度（逻辑像素）。
    :param box_height: 可用高度（逻辑像素），用于自动推定字号。
    :param style: :class:`~pingyitong.config.schema.StyleConfig`。
    :param colour_scheme_dark: 原文区域是否为深色背景（当前仅用于预留扩展）。
    :param font_px: 可选的条目固定字号；断行与实绘都使用该字号。
    :return: :class:`TextLayout`。
    """
    text = (text or "").strip()
    if not text:
        return TextLayout(font_px=0, lines=[], line_height=0.0, total_height=0.0)

    family = getattr(style, "font_family", "") or ""
    bold = bool(getattr(style, "bold", False))
    min_size = max(1, int(readable_min_px), int(getattr(style, "min_font_size", 12)))
    max_size = max(min_size, int(getattr(style, "max_font_size", 44)))
    max_lines = max(1, int(getattr(style, "max_lines", 6)))
    line_spacing = max(1.0, float(getattr(style, "line_spacing", 1.08)))
    mode = getattr(style, "font_size_mode", "auto")
    scale = float(getattr(style, "font_scale", 1.0))

    available_width = max(1.0, float(box_width))
    available_height = max(1.0, float(box_height))

    start_size = _initial_font_size(mode, style, box_height, scale, min_size, max_size)
    if font_px is not None:
        start_size = font_px
        mode = "fixed"
    start_size = max(min_size, min(max_size, int(start_size)))

    font = build_font(family, start_size, bold)
    lines = [line.rstrip() for line in wrap_text(text, font, available_width)]
    line_height = line_height_for(font) * line_spacing

    # 自动模式：若行数或总高超出可用空间，逐步缩小字号重排。
    shrunk = False
    size = start_size
    while mode != "fixed" and size > min_size:
        needed_height = line_height * len(lines)
        if (
            len(lines) <= max_lines and needed_height <= available_height
            and measure_lines(lines, font) <= available_width
        ):
            break

        size -= _FONT_STEP
        font = build_font(family, size, bold)
        lines = [line.rstrip() for line in wrap_text(text, font, available_width)]
        line_height = line_height_for(font) * line_spacing
        shrunk = True

    # The readable minimum and an explicit fixed font are constraints, not a
    # licence to draw extra rows over adjacent UI. Signal omitted content.
    needed_line_count = len(lines)
    needed_height = line_height * needed_line_count
    overflow_reasons = []
    if needed_line_count > max_lines:
        overflow_reasons.append("max_lines_limit")
    if needed_height > available_height + 1e-6:
        overflow_reasons.append("insufficient_height")
    if measure_lines(lines, font) > available_width:
        overflow_reasons.append("insufficient_width")
    visible_lines = min(max_lines, int((available_height + 1e-6) / line_height))
    marker_only = visible_lines == 0
    if marker_only:
        overflow_reasons.append("no_complete_line")
    truncated = len(lines) > visible_lines or measure_lines(lines, font) > available_width
    if marker_only:
        lines = ["…"]
        line_height = max(1.0, QFontMetricsF(font).tightBoundingRect("…").height())
    else:
        lines = lines[:visible_lines]
        lines = [elide_line(line, font, available_width) for line in lines]
        if truncated:
            lines[-1] = elide_line(lines[-1] + "…", font, available_width)

    return TextLayout(
        font_px=size,
        lines=lines,
        line_height=line_height,
        total_height=line_height * len(lines),
        content_width=measure_lines(lines, font),
        truncated=truncated,
        shrunk=shrunk,
        marker_only=marker_only,
        needed_line_count=needed_line_count,
        needed_height=needed_height,
        overflow_reasons=overflow_reasons,
    )


def _initial_font_size(
    mode: str,
    style: object,
    box_height: float,
    scale: float,
    min_size: int,
    max_size: int,
) -> int:
    """按模式推算起始字号。"""
    if mode == "fixed":
        return int(getattr(style, "font_size_fixed", 14))

    # 自动模式：以原文包围盒高度为基准，乘以行高比例与用户缩放系数。
    candidate = box_height * DEFAULT_HEIGHT_RATIO * max(0.2, scale)
    return int(round(max(min_size, min(max_size, candidate))))


def measure_lines(lines: list[str], font: QFont) -> float:
    """返回多行文本中最长一行的像素宽度。

    用 :class:`QFontMetricsF` 实测量取，而不是按字符数估算——中文、拉丁
    与混合文本的宽度差异极大，估算法会让背景框与文字明显不贴合。
    """
    if not lines:
        return 0.0
    metrics = QFontMetricsF(font)
    return max(metrics.horizontalAdvance(line) for line in lines)


def elide_line(text: str, font: QFont, max_width: float) -> str:
    """对单行文本加省略号，使其不超过给定宽度。

    用于固定字号模式下被截断的行，让用户明确知道内容被裁掉了。
    """
    from PyQt6.QtGui import QFontMetricsF

    metrics = QFontMetricsF(font)
    if metrics.horizontalAdvance(text) <= max_width:
        return text
    return metrics.elidedText(text, Qt.TextElideMode.ElideRight, max_width)


def fit_rect(box_width: int, box_height: int, style: object) -> QRectF:
    """返回考虑内边距后的可用绘制矩形。"""
    padding_x = float(getattr(style, "padding_x", 3))
    padding_y = float(getattr(style, "padding_y", 1))
    return QRectF(
        0.0,
        0.0,
        max(1.0, box_width - 2 * padding_x),
        max(1.0, box_height - 2 * padding_y),
    )
