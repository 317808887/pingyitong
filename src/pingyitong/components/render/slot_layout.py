"""Allocate readable translations inside observed capture space.

The OCR rectangle is an anchor and a protected source region, rather than the
maximum translation size. Other sources (including untranslated ones) and
already allocated translations remain obstacles. Allocation is deterministic
and deliberately reports overflow instead of shrinking below readable type.
"""

from __future__ import annotations

import statistics
import unicodedata
from dataclasses import dataclass

from PyQt6.QtCore import QRectF
from PyQt6.QtGui import QFontMetricsF

from .layout import DEFAULT_HEIGHT_RATIO, build_font, layout_text
from .render_item import RenderItem

READABLE_FONT_PX = 14
_GAP = 1.0


@dataclass(frozen=True)
class SlotAllocation:
    rect: QRectF
    font_px: int
    reason: str


def readable_font_size(items: list[RenderItem], style: object) -> int:
    """One body size per frame; source height never licenses 9px translation."""
    minimum = max(READABLE_FONT_PX, int(getattr(style, "min_font_size", 9)))
    maximum = max(minimum, int(getattr(style, "max_font_size", 44)))
    if getattr(style, "font_size_mode", "auto") == "fixed":
        return max(minimum, min(maximum, int(getattr(style, "font_size_fixed", 14))))
    heights = [item.height for item in items if item.height > 0]
    median = statistics.median(heights) if heights else minimum
    candidate = round(median * DEFAULT_HEIGHT_RATIO * max(
        0.2, float(getattr(style, "font_scale", 1.0)),
    ))
    size = max(minimum, min(maximum, 28, candidate))
    padding = max(0.0, float(getattr(style, "padding_y", 1))) * 2
    spacing = max(1.0, float(getattr(style, "line_spacing", 1.08)))
    while size > minimum:
        font = build_font(
            getattr(style, "font_family", "") or "", size,
            bool(getattr(style, "bold", False)),
        )
        if QFontMetricsF(font).height() * spacing + padding <= median:
            break
        size -= 1
    return size


def right_to_left(text: str) -> bool:
    """Use the first strong character, allowing numbers before Arabic/Hebrew."""
    for char in text:
        direction = unicodedata.bidirectional(char)
        if direction in ("R", "AL"):
            return True
        if direction == "L":
            return False
    return False


def allocate_slots(
    items: list[RenderItem],
    source_rects: list[QRectF],
    seeds: dict[int, QRectF],
    bounds: QRectF,
    excluded: list[QRectF],
    style: object,
    space_obstacles: list[QRectF] | None = None,
) -> dict[int, SlotAllocation]:
    """Fit complete text at a shared size without painting over other sources.

    ``seeds`` have already partitioned overlapping OCR geometry. Extra source
    rectangles represent identified content that has no render item. A slot
    contains its seed, is inside observed bounds and avoids every other seed,
    source and prior allocation. This permits safe use of whitespace while
    giving no assurance that an arbitrarily dense page will fit in place.

    ``space_obstacles`` are barriers measured from the captured pixels, i.e.
    contrasting content that OCR never returned as a source. They restrict
    expansion only: they never invalidate a seed, because a one-pixel
    difference between OCR geometry and pixel tiles must not disable the
    translation of an otherwise valid anchor.
    """
    shared_font = readable_font_size(items, style)
    result: dict[int, SlotAllocation] = {}
    occupied = {id(item): QRectF(seeds[id(item)]).intersected(bounds) for item in items}
    blockers = [rect.intersected(bounds) for rect in [*source_rects, *excluded]]
    blockers = [rect for rect in blockers if not rect.isEmpty()]
    pixel_barriers = [
        rect.intersected(bounds) for rect in (space_obstacles or [])
    ]
    pixel_barriers = [rect for rect in pixel_barriers if not rect.isEmpty()]
    order = sorted(items, key=lambda item: (
        item.box.top, item.box.left, item.box.bottom, item.box.right,
        item.source_text, item.text,
    ))
    for item in order:
        seed = occupied[id(item)]
        minimum = max(READABLE_FONT_PX, int(getattr(style, "min_font_size", 9)))
        maximum = max(minimum, int(getattr(style, "max_font_size", 44)))
        size = max(minimum, min(maximum, int(item.font_px or shared_font)))
        if seed.isEmpty():
            result[id(item)] = SlotAllocation(seed, size, "outside_capture_bounds")
            continue
        if item.quality_status == "uncertain" or _source_copy(item):
            result[id(item)] = SlotAllocation(seed, size, "source_preserved")
            continue
        obstacles = [rect for key, rect in occupied.items() if key != id(item)] + blockers
        if any(seed.intersects(rect) for rect in obstacles):
            result[id(item)] = SlotAllocation(seed, size, "source_or_window_obstructs_anchor")
            continue
        text = item.display_text or item.text
        slot = _fit(
            text, seed, bounds, obstacles, pixel_barriers, style, size,
        )
        if slot is None:
            result[id(item)] = SlotAllocation(seed, size, "no_readable_space")
        else:
            occupied[id(item)] = slot
            expanded = not seed.contains(slot)
            result[id(item)] = SlotAllocation(
                slot, size, "expanded_into_observed_space" if expanded else "source_slot_fits",
            )
    return result


def _source_copy(item: RenderItem) -> bool:
    return bool(item.source_text.strip()) and (
        " ".join(item.text.split()) == " ".join(item.source_text.split())
    )


def _fit(
    text: str, seed: QRectF, bounds: QRectF, obstacles: list[QRectF],
    barriers: list[QRectF], style: object, size: int,
) -> QRectF | None:
    """Grow the seed as far as sources and observed ink allow, else keep the seed.

    ``obstacles`` are structural claims (other seeds, extra sources, app
    windows); a seed intersecting them is already rejected by the caller.
    ``barriers`` are pixel evidence extracted from the capture and are only
    allowed to shrink the growth options, never to reject the seed itself.
    """
    blockers = [*obstacles, *barriers]
    padding_x = max(0.0, float(getattr(style, "padding_x", 3)))
    padding_y = max(0.0, float(getattr(style, "padding_y", 1)))
    font = build_font(
        getattr(style, "font_family", "") or "", size, bool(getattr(style, "bold", False)),
    )
    metrics = QFontMetricsF(font)
    line_height = metrics.height() * max(1.0, float(getattr(style, "line_spacing", 1.08)))
    natural_width = max(metrics.horizontalAdvance(line) for line in text.split("\n"))
    left_to_right = not right_to_left(text)
    top, bottom = bounds.top(), bounds.bottom()
    for obstacle in blockers:
        if not _axis_overlap(seed.left(), seed.right(), obstacle.left(), obstacle.right()):
            continue
        if obstacle.bottom() <= seed.top():
            top = max(top, min(seed.top(), obstacle.bottom() + _GAP))
        elif obstacle.top() >= seed.bottom():
            bottom = min(bottom, max(seed.bottom(), obstacle.top() - _GAP))
    # At most max_lines candidates; every one is remeasured with the actual
    # font and available width rather than estimating by character count.
    for row_count in range(1, max(1, int(getattr(style, "max_lines", 6))) + 1):
        height = max(seed.height(), row_count * line_height + padding_y * 2)
        low = max(top, seed.bottom() - height)
        high = min(seed.top(), bottom - height)
        if low > high + 1e-6:
            break  # More wrapped lines cannot recover missing vertical space.
        y = max(low, min(high, seed.center().y() - height / 2))
        left, right = bounds.left(), bounds.right()
        for obstacle in blockers:
            if not _axis_overlap(y, y + height, obstacle.top(), obstacle.bottom()):
                continue
            if obstacle.right() <= seed.left():
                left = max(left, min(seed.left(), obstacle.right() + _GAP))
            elif obstacle.left() >= seed.right():
                right = min(right, max(seed.right(), obstacle.left() - _GAP))
        width = max(seed.width(), min(right - left, natural_width + padding_x * 2 + 0.5))
        if left_to_right:
            x = max(left, min(seed.left(), right - width))
        else:
            x = min(right - width, max(left, seed.right() - width))
        candidate = QRectF(x, y, width, height)
        if not bounds.contains(candidate) or any(candidate.intersects(rect) for rect in blockers):
            continue
        layout = layout_text(
            text, max(1.0, width - 2 * padding_x), max(1.0, height - 2 * padding_y),
            style, font_px=size, readable_min_px=READABLE_FONT_PX,
        )
        if not layout.truncated:
            return candidate
    # Conservative pixel barriers may forbid every growth direction. The anchor
    # itself stays legitimate: it is exactly the region OCR assigned to this
    # text, and barriers never cover identified sources. Painting the full text
    # there is honest; stretching it over neighbouring ink would not be.
    if _accepts(seed, bounds, obstacles):
        layout = layout_text(
            text, max(1.0, seed.width() - 2 * padding_x),
            max(1.0, seed.height() - 2 * padding_y),
            style, font_px=size, readable_min_px=READABLE_FONT_PX,
        )
        if not layout.truncated:
            return QRectF(seed)
    return None


def _accepts(candidate: QRectF, bounds: QRectF, obstacles: list[QRectF]) -> bool:
    if candidate.isEmpty() or not bounds.contains(candidate):
        return False
    return not any(candidate.intersects(rect) for rect in obstacles)


def _axis_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return min(a_end, b_end) > max(a_start, b_start) + 1e-6
