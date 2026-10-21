"""原位渲染与叠加模块。

对应《项目原创性声明》二·（三）"桌面原位渲染引擎"。

实现要点：

- **无边框透明顶层窗口**：``FramelessWindowHint + WindowStaysOnTopHint +
  WindowTransparentForInput + WA_TranslucentBackground``，窗口置顶且鼠标穿透，
  不影响用户正常操作；
- **逐文本块渲染**：以原文框为锚点，在捕获边界和相邻内容之间分配可读译文槽；
- **高 DPI 适配**：窗口几何使用逻辑坐标，绘制前把物理坐标折算为逻辑坐标；
- **多显示器**：每块屏幕一个独立窗口，由 :class:`OverlayManager` 统一管理。

关于"不添加动画"的设计决策
    译文层最高以 15fps 刷新。按动效设计规范（见 ``docs/UI设计规范.md``），
    高频刷新的数据展示区域**不应加入入场/位移动画**：一方面动画会让
    正在阅读的文字持续位移，破坏原位对照；另一方面每秒十余次的动画
    重建会直接拖垮渲染性能。因此译文层的更新是瞬时替换，只在
    "整层显示/隐藏"这类低频操作上使用淡入淡出。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

from PyQt6.QtCore import QObject, QPoint, QPointF, QRect, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetricsF,
    QGuiApplication,
    QImage,
    QPainter,
    QPen,
    QRegion,
)
from PyQt6.QtWidgets import QWidget

from ...utils.logger import get_logger
from ..capture.geometry import Rect, ScreenInfo, list_screens
from .layout import build_font, layout_text
from .render_item import RenderFrame, RenderItem, merge_items
from .slot_layout import READABLE_FONT_PX, allocate_slots, right_to_left

logger = get_logger(__name__)

#: 排版缓存容量。屏幕文字的重复率很高，缓存可显著降低绘制开销。
_LAYOUT_CACHE_SIZE = 256


@dataclass
class _CachedLayout:
    """缓存的一次排版结果。"""

    font: QFont
    lines: list[str]
    line_height: float
    truncated: bool
    elided: list[str] | None = None
    #: 完整排版结果，供背景遮罩取用实测宽度与总行高
    layout: object | None = None


class OverlayWindow(QWidget):
    """单块显示器上的透明译文渲染层。"""

    #: 用户在本窗口上请求"复制当前翻译结果"（保留信号以便悬浮菜单扩展）
    copy_requested = pyqtSignal()
    presented = pyqtSignal(object, object, float)

    def __init__(
        self,
        screen: ScreenInfo,
        style: object,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._screen = screen
        self._style = style
        self._items: list[RenderItem] = []
        self._paint_rects: dict[int, QRectF] = {}
        self._layout_bounds: QRectF | None = None
        self._source_context: list[RenderItem] = []
        self._source_owners: dict[tuple, tuple] = {}
        self._font_sizes: dict[int, int] = {}
        self._layout_reasons: dict[int, str] = {}
        self._slot_cache_signature: tuple | None = None
        self._slot_cache: list[tuple[QRectF, int | None, str | None]] = []
        self._excluded_regions: list[Rect] = []
        #: Pixel barriers observed in the captured frame (window-local logical).
        #: Empty with ``_space_observed`` false means "no pixel evidence", which
        #: is the legacy behaviour and does not claim blank space.
        self._space_obstacles: list[QRectF] = []
        self._space_observed = False
        self._layout_cache: dict[tuple, _CachedLayout] = {}
        self._style_generation = 0
        self._visible = True
        self._pending_frame: RenderFrame | None = None

        self._setup_window()

    # ------------------------------------------------------------------
    # 窗口属性
    # ------------------------------------------------------------------

    def _setup_window(self) -> None:
        """配置无边框、置顶、鼠标穿透的透明窗口。"""
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowTransparentForInput
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        # 不参与系统合成器的窗口阴影与动画，减少桌面组合开销
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setWindowTitle("屏译通翻译层")

        self._apply_geometry()

    def _apply_geometry(self) -> None:
        """把窗口铺满所属显示器。"""
        rect: Rect = self._screen.logical
        self.setGeometry(rect.x, rect.y, max(1, rect.width), max(1, rect.height))

    @property
    def screen_info(self) -> ScreenInfo:
        """本窗口覆盖的显示器信息。"""
        return self._screen

    @property
    def style_config(self) -> object:
        """当前样式配置。"""
        return self._style

    # ------------------------------------------------------------------
    # 内容更新
    # ------------------------------------------------------------------

    def set_items(
        self, items: list[RenderItem], replace: bool = True,
        dirty_regions: list[Rect] | None = None,
    ) -> None:
        """更新本窗口的译文条目（逻辑坐标换算在此完成）。

        :param items: 属于本显示器的条目，``box`` 为全局物理坐标。
        :param replace: ``True`` 表示整帧识别，直接替换全部内容；
            ``False`` 表示增量帧，只覆盖本次识别到的区域，
            其余条目的译文保持不动。

        **增量帧必须合并而不是替换**：画面差分只把"变化的区域"送去识别，
        因此增量帧里的条目只覆盖屏幕的一小块。若直接替换，屏幕上其余位置
        的译文会在每一帧被清空、下一帧再画回来，肉眼看就是整屏文字疯狂
        闪烁——用户反馈"字幕出现的动态效果太差，根本没法看"。
        """
        logical = self._to_window_coords(items) if items else []
        logical_dirty = self._to_window_regions(dirty_regions or [])

        if not logical and not logical_dirty:
            # 增量帧没有新内容时保持现状；整帧识别结果为空才清空
            if replace and self._items:
                self._items = []
                self._paint_rects.clear()
                self._font_sizes.clear()
                self._layout_reasons.clear()
                self._slot_cache_signature = None
                self._slot_cache = []
                self._layout_cache.clear()
                self.update()
            elif self._items:
                self._rebuild_slots()
                self.update()
            return

        if replace:
            self._items = logical
        else:
            self._items = merge_items(self._items, logical, dirty_regions=logical_dirty)
        self._rebuild_slots()

        # 条目变化频繁，缓存按键包含文本与尺寸，命中率仍可保持较高。
        if len(self._layout_cache) > _LAYOUT_CACHE_SIZE:
            self._layout_cache.clear()
        self.update()

    def clear(self) -> None:
        """清空译文层内容。"""
        self._slot_cache_signature = None
        self._slot_cache = []
        self._source_context.clear()
        self._source_owners.clear()
        self._layout_bounds = None
        if self._items:
            self._items = []
            self._paint_rects.clear()
            self._font_sizes.clear()
            self._layout_reasons.clear()
            self._source_context.clear()
            self._source_owners.clear()
            self._layout_cache.clear()
            self.update()

    def set_excluded_regions(self, regions: list[Rect]) -> None:
        """Keep the app's interactive windows free of translated paint."""
        updated = list(dict.fromkeys(region for region in regions if not region.is_empty))
        if updated != self._excluded_regions:
            self._excluded_regions = updated
            self._rebuild_slots()
            self.update()

    def set_layout_context(
        self, bounds: Rect, sources: list[RenderItem] | None,
        replace: bool = True, dirty_regions: list[Rect] | None = None,
        provenance: list[dict] | None = None,
        space_obstacles: list[Rect] | None = None,
    ) -> None:
        """Use only observed capture pixels, including untranslated source blockers.

        ``space_obstacles`` are physical-coordinate barriers derived from the
        captured pixels themselves (contrasting ink that OCR did not return).
        Passing ``None`` records "no pixel evidence", which keeps the legacy
        source-only basis instead of pretending the remaining area is blank.
        """
        clipped = bounds.clamp(self._screen.physical)
        ratio = self._screen.device_pixel_ratio or 1.0
        self._layout_bounds = QRectF(
            (clipped.x - self._screen.physical.x) / ratio,
            (clipped.y - self._screen.physical.y) / ratio,
            clipped.width / ratio, clipped.height / ratio,
        ).intersected(QRectF(self.rect())) if not clipped.is_empty else QRectF()
        self._space_observed = space_obstacles is not None
        self._space_obstacles = [
            QRectF(_to_qrect(rect)).adjusted(1, 1, -1, -1)
            for rect in self._to_window_regions(space_obstacles or [])
            if not rect.is_empty
        ]
        self._space_obstacles = [rect for rect in self._space_obstacles if not rect.isEmpty()]
        logical_sources = self._to_window_coords(sources or [])
        self._source_context = logical_sources if replace else merge_items(
            self._source_context, logical_sources, self._to_window_regions(dirty_regions or []),
        )
        # A paragraph owns exactly the OCR lines consumed to produce it.
        # Containment alone is insufficient: a filtered number can sit inside
        # a paragraph rectangle and must remain a separate source obstacle.
        owners = {}
        for paragraph in provenance or []:
            if not paragraph.get("text") or not paragraph.get("rect"):
                continue
            owner = self._to_window_coords([RenderItem(
                Rect.from_sequence(paragraph["rect"]), paragraph["text"],
            )])[0]
            for line in paragraph.get("source_lines", []):
                if not line.get("text") or not line.get("rect"):
                    continue
                source = self._to_window_coords([RenderItem(
                    Rect.from_sequence(line["rect"]), line["text"],
                )])[0]
                owners[(source.box, source.text)] = (owner.box, owner.text)
        alive = {(item.box, item.text) for item in self._source_context}
        self._source_owners = {} if replace else {
            key: owner for key, owner in self._source_owners.items() if key in alive
        }
        self._source_owners.update(owners)

    def _rebuild_slots(self) -> None:
        render_boxes = {item.box for item in self._items}
        render_owners = {(item.box, item.source_text) for item in self._items}
        extras = [source for source in self._source_context if source.box not in render_boxes
                  and self._source_owners.get((source.box, source.text)) not in render_owners]
        order = sorted(self._items, key=lambda item: (
            item.box.top, item.box.left, item.box.bottom, item.box.right,
            item.text, item.source_text, item.display_text, item.font_px or -1, item.quality_status,
        ))
        signature = (
            self._style_generation,
            tuple((item.box, item.text, item.source_text, item.display_text,
                   item.font_px, item.quality_status) for item in order),
            tuple(sorted({source.box.to_ltrb() for source in extras})),
            tuple(self._excluded_regions),
            tuple(sorted(
                (round(rect.x(), 2), round(rect.y(), 2),
                 round(rect.width(), 2), round(rect.height(), 2))
                for rect in self._space_obstacles
            )),
            self._space_observed,
            (self._layout_bounds.x(), self._layout_bounds.y(), self._layout_bounds.width(),
             self._layout_bounds.height()) if self._layout_bounds is not None else None,
        )
        if signature == self._slot_cache_signature:
            self._paint_rects = {
                id(item): QRectF(slot[0]) for item, slot in zip(order, self._slot_cache, strict=True)
            }
            self._font_sizes = {id(item): slot[1] for item, slot in
                                zip(order, self._slot_cache, strict=True) if slot[1] is not None}
            self._layout_reasons = {id(item): slot[2] for item, slot in
                                    zip(order, self._slot_cache, strict=True) if slot[2] is not None}
            return
        seeds = _collision_free_rects(self._items)
        self._font_sizes = {}
        self._layout_reasons = {}
        if self._layout_bounds is None:
            # Older callers without observed bounds cannot claim blank space.
            self._paint_rects = {id(item): seeds[id(item)] for item in self._items}
            self._remember_slots(order, signature)
            return
        allocations = allocate_slots(
            self._items, [QRectF(_to_qrect(source.box)) for source in extras], seeds,
            self._layout_bounds,
            [QRectF(_to_qrect(rect)) for rect in self._to_window_regions(self._excluded_regions)],
            self._style,
            space_obstacles=self._space_obstacles if self._space_observed else None,
        )
        self._paint_rects = {key: slot.rect for key, slot in allocations.items()}
        self._font_sizes = {key: slot.font_px for key, slot in allocations.items()}
        self._layout_reasons = {key: slot.reason for key, slot in allocations.items()}
        self._remember_slots(order, signature)

    def _remember_slots(self, order: list[RenderItem], signature: tuple) -> None:
        self._slot_cache_signature = signature
        self._slot_cache = [(
            QRectF(self._paint_rects[id(item)]), self._font_sizes.get(id(item)),
            self._layout_reasons.get(id(item)),
        ) for item in order]

    def apply_style(self, style: object) -> None:
        """应用新的样式配置并重绘。"""
        self._style = style
        self._style_generation += 1
        self._layout_cache.clear()
        self._rebuild_slots()
        self.update()

    def set_layer_visible(self, visible: bool) -> None:
        """显示或隐藏整层（低频操作，由外部决定是否加淡入淡出）。"""
        self._visible = visible
        if visible:
            self.show()
            self.raise_()
        else:
            self.hide()

    @property
    def is_layer_visible(self) -> bool:
        return self._visible

    def refresh_screen(self, screen: ScreenInfo) -> None:
        """显示器几何变化（分辨率或缩放变更）后重新铺满。"""
        self._screen = screen
        self._apply_geometry()
        self._layout_cache.clear()
        # 旧条目的窗口局部坐标按旧 origin/ratio 换算，几何变化后已失效；
        # 保留会把旧译文持续画在错位位置（渲染审查 R2，中），清空等待
        # 流水线对当前画面重新识别后自愈。
        self._items = []
        self._paint_rects.clear()
        self._font_sizes.clear()
        self._layout_reasons.clear()
        self._source_context.clear()
        self._source_owners.clear()
        self._layout_bounds = None
        self._slot_cache_signature = None
        self._slot_cache = []
        self.update()

    # ------------------------------------------------------------------
    # 坐标转换
    # ------------------------------------------------------------------

    def _to_window_coords(self, items: list[RenderItem]) -> list[RenderItem]:
        """把物理坐标条目转换为本窗口的局部逻辑坐标。

        换算方式与 Qt 的 ``QHighDpiScaling`` 保持一致：窗口的逻辑原点
        与其所属显示器的**物理**原点重合，只有尺寸按缩放比例折算。
        因此"窗口内局部坐标"即物理坐标相对该屏物理原点的偏移除以缩放比例。

        这里刻意不使用 ``round(物理坐标 / 缩放) - 逻辑原点`` 的写法：
        在 150% 缩放的副屏上，后者会把已经对齐的原点重复扣减一次，
        使译文被推出窗口（表现为"翻译层没有任何内容"）。
        """
        ratio = self._screen.device_pixel_ratio or 1.0
        origin = self._screen.physical

        converted: list[RenderItem] = []
        for item in items:
            if not item.is_renderable:
                continue
            box = Rect(
                round((item.box.x - origin.x) / ratio),
                round((item.box.y - origin.y) / ratio),
                max(1, round(item.box.width / ratio)),
                max(1, round(item.box.height / ratio)),
            )
            converted.append(
                replace(item, box=box)
            )
        return converted

    def _to_window_regions(self, regions: list[Rect]) -> list[Rect]:
        """Convert dirty physical rectangles using the same screen origin."""
        import math

        ratio = self._screen.device_pixel_ratio or 1.0
        origin = self._screen.physical
        converted: list[Rect] = []
        for region in regions:
            clipped = region.clamp(origin)
            if clipped.is_empty:
                continue
            converted.append(Rect.from_ltrb(
                math.floor((clipped.left - origin.x) / ratio),
                math.floor((clipped.top - origin.y) / ratio),
                math.ceil((clipped.right - origin.x) / ratio),
                math.ceil((clipped.bottom - origin.y) / ratio),
            ))
        return converted

    # ------------------------------------------------------------------
    # 绘制
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt 命名约定
        """绘制后确认；验收模式保存这次真正送绘的缓冲。"""
        from ...utils.acceptance_evidence import evidence_enabled, save_paint

        frame = self._pending_frame
        capture = frame is not None and evidence_enabled()
        buffer = None
        if capture:
            ratio = self.devicePixelRatioF()
            buffer = QImage(round(self.width() * ratio), round(self.height() * ratio),
                            QImage.Format.Format_ARGB32_Premultiplied)
            buffer.setDevicePixelRatio(ratio)
            buffer.fill(Qt.GlobalColor.transparent)
        painter = QPainter(buffer if buffer is not None else self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        try:
            clip = QRegion(self.rect() if capture else event.rect())
            for region in self._to_window_regions(self._excluded_regions):
                clip = clip.subtracted(QRegion(_to_qrect(region)))
            painter.setClipRegion(clip)
            visible_rect = QRectF(self.rect() if capture else event.rect())
            for item in self._items:
                if not visible_rect.intersects(self._paint_rect_for(item)):
                    continue  # 视口外条目直接跳过，降低无谓绘制
                self._paint_item(painter, item)
        finally:
            painter.end()
        if buffer is not None:
            screen_painter = QPainter(self)
            screen_painter.drawImage(QPoint(0, 0), buffer)
            screen_painter.end()
        painted_at = time.perf_counter()
        if frame is not None:
            self._pending_frame = None
            self.presented.emit(self, frame, painted_at)
            if buffer is not None:
                save_paint(frame, self._screen.name, buffer, self.paint_evidence(), painted_at)

    def paint_evidence(self) -> list[dict]:
        """实际使用的排版、处置和互斥绘制区域，不推断未识别的文字数量。"""
        result = []
        for item in self._items:
            rect = self._paint_rect_for(item)
            layout = self._get_layout(item)
            disposition = (
                "uncertain" if item.quality_status == "uncertain" else
                "source_unchanged" if _is_source_copy(item) else
                "full_text_in_panel" if layout is None or layout.truncated
                or self._slot_blocked(item) else "translated"
            )
            translated = disposition == "translated"
            default_reason = {
                "uncertain": "translation_uncertain", "source_unchanged": "source_preserved",
                "full_text_in_panel": "no_readable_space",
                "translated": "source_slot_fits_without_expansion",
            }[disposition]
            result.append({
                "source": item.source_text, "full_text": item.text,
                "display_text": item.display_text or item.text,
                "compact_text": item.display_text or "",
                "compact_generated": bool(item.display_text),
                "quality_status": item.quality_status, "quality_reason": item.quality_reason,
                "ocr_uncertain": item.ocr_uncertain, "disposition": disposition,
                "from_cache": item.from_cache,
                "rect": [rect.x(), rect.y(), rect.width(), rect.height()],
                "source_rect": list(item.box.to_sequence()),
                "lines": layout.lines if layout else [],
                "font_px": layout.font.pixelSize() if layout else 0,
                "truncated": (layout.truncated or self._slot_blocked(item)) if layout else True,
                "rendered_lines": layout.lines if translated and layout else [],
                "inline_delivery": (
                    "compact" if item.display_text else "full"
                ) if translated else "none",
                "layout_reason": self._layout_reasons.get(id(item), default_reason),
                "readable_font_floor_px": READABLE_FONT_PX,
                "space_basis": (
                    "capture_roi_minus_detected_sources_excluded_windows_and_observed_ink"
                    if self._space_observed else
                    "capture_roi_minus_detected_sources_and_excluded_windows"
                ),
                "observed_ink_barriers": len(self._space_obstacles),
                "layout_overflow_reasons": (
                    list(layout.layout.overflow_reasons) if layout and layout.layout else []
                ),
                "layout_needed_lines": (
                    layout.layout.needed_line_count if layout and layout.layout else 0
                ),
                "layout_needed_height": (
                    layout.layout.needed_height if layout and layout.layout else 0.0
                ),
                "capture_bounds": (
                    [self._layout_bounds.x(), self._layout_bounds.y(),
                     self._layout_bounds.width(), self._layout_bounds.height()]
                    if self._layout_bounds is not None else None
                ),
            })
        return result

    def _paint_item(self, painter: QPainter, item: RenderItem) -> None:
        """绘制单个译文条目：背景遮罩 + 文字（可选阴影）。"""
        if item.quality_status == "uncertain":
            self._paint_uncertain(painter, item)
            return
        if _is_source_copy(item):
            return
        style = self._style
        rect = self._paint_rect_for(item)

        cached = self._get_layout(item)
        if cached is None or cached.truncated or self._slot_blocked(item):
            self._paint_uncertain(painter, item)
            return

        inner = self._inner_rect(rect)

        # 对照叠加模式贴合译文；完全覆盖模式必须遮住整个原文区域。
        painter.save()
        try:
            painter.setClipRect(rect, Qt.ClipOperation.IntersectClip)
            background = (
                rect if getattr(style, "mode", "cover") == "cover"
                else self._background_rect(
                    inner, cached, right_to_left(item.display_text or item.text),
                )
            )
            self._paint_background(painter, background, item)
            self._paint_text(painter, inner, item, cached)
        finally:
            painter.restore()

    def _paint_uncertain(self, painter: QPainter, item: RenderItem) -> None:
        # 保留原文可见；空间不足时不截断条件/否定制造另一种意思。
        rect = self._paint_rect_for(item).adjusted(1, 1, -1, -1)
        if rect.isEmpty():
            return
        painter.save()
        painter.setPen(QPen(QColor("#E9AD49"), 1, Qt.PenStyle.DotLine))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(rect)
        painter.restore()

    def _paint_rect_for(self, item: RenderItem) -> QRectF:
        """Return this item's disjoint slot without changing the full source item."""
        return self._paint_rects.get(id(item), QRectF(_to_qrect(item.box)))

    def _slot_blocked(self, item: RenderItem) -> bool:
        return self._layout_reasons.get(id(item)) in (
            "outside_capture_bounds", "source_or_window_obstructs_anchor", "no_readable_space",
        )

    def _inner_rect(self, rect: QRectF) -> QRectF:
        """Use the same padding for font measurement and actual painting."""
        padding_x = min(
            max(0.0, float(getattr(self._style, "padding_x", 3))),
            max(0.0, (rect.width() - 1) / 2),
        )
        padding_y = min(
            max(0.0, float(getattr(self._style, "padding_y", 1))),
            max(0.0, (rect.height() - 1) / 2),
        )
        return rect.adjusted(padding_x, padding_y, -padding_x, -padding_y)

    def _background_rect(
        self, inner: QRectF, cached: object, align_right: bool = False,
    ) -> QRectF:
        """返回贴合译文的背景矩形（左上角与文字对齐）。

        宽度取实测量的最长行宽，高度取排版后的总行高，再各自留出一点
        内边距；两者都不超过可用区域。文字仍按原位置绘制，
        因此译文不会因为背景收缩而移动。
        """
        layout = getattr(cached, "layout", None)
        if layout is None or not getattr(layout, "lines", None):
            return inner

        padding_x = float(getattr(self._style, "padding_x", 3))
        padding_y = float(getattr(self._style, "padding_y", 1))

        width = min(inner.width(), float(getattr(layout, "content_width", 0.0)) + padding_x * 2)
        height = min(
            inner.height(),
            float(getattr(layout, "total_height", 0.0)) + padding_y * 2,
        )
        if width <= 0 or height <= 0:
            return inner

        left = inner.right() - width if align_right else inner.left()
        return QRectF(left, inner.top(), width, height)

    def _paint_background(self, painter: QPainter, rect: QRectF, item: RenderItem) -> None:
        """绘制译文背景遮罩。

        - ``cover``（完全覆盖）：使用不透明遮罩，确保原文不可见；
          若原文背景为深色，使用深色遮罩以避免与界面割裂。
        - ``overlay``（半透明叠加）：按配置的不透明度叠加，便于原文对照。
        """
        style = self._style
        mode = getattr(style, "mode", "cover")
        radius = float(getattr(style, "bg_radius", 4))

        base = QColor(getattr(style, "bg_color", "#101418"))
        if not base.isValid():
            base = QColor("#101418")

        opacity = float(getattr(style, "bg_opacity", 0.62))
        alpha = 255 if mode == "cover" else int(max(0.0, min(1.0, opacity)) * 255)

        if alpha <= 0:
            return

        base.setAlpha(alpha)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(base))

        # 圆角半径不超过短边的一半，避免退化为异常形状
        # Source glyphs can touch the OCR box corners. Cover mode must mask the
        # whole box; a rounded translucent edge would leave original strokes.
        effective_radius = (
            0.0 if mode == "cover"
            else min(radius, rect.width() / 2, rect.height() / 2)
        )
        if effective_radius >= 1.0:
            painter.drawRoundedRect(rect, effective_radius, effective_radius)
        else:
            painter.drawRect(rect)

    def _paint_text(
        self,
        painter: QPainter,
        inner: QRectF,
        item: RenderItem,
        cached: _CachedLayout,
    ) -> None:
        """绘制译文文字，可选阴影。"""
        style = self._style
        text_color = QColor(getattr(style, "text_color", "#FFFFFF"))
        if not text_color.isValid():
            text_color = QColor("#FFFFFF")

        shadow = bool(getattr(style, "shadow", False))
        lines = cached.elided if cached.elided is not None else cached.lines
        font = cached.font
        painter.setFont(font)

        line_height = cached.line_height
        total_height = line_height * len(lines)
        # 垂直居中：译文块通常比原文行高略高，居中后与原文视觉基线更接近
        start_y = inner.top() + max(0.0, (inner.height() - total_height) / 2)
        ascent = _ascent(font)
        if getattr(cached.layout, "marker_only", False):
            # Ellipsis ink is much shorter than the full font line. Keep the
            # marker visible when the original box cannot hold one normal row.
            ink = QFontMetricsF(font).tightBoundingRect("…")
            start_y = inner.top() + max(0.0, (inner.height() - ink.height()) / 2)
            ascent = -ink.top()

        for index, line in enumerate(lines):
            baseline = start_y + ascent + index * line_height
            x = inner.left()
            if right_to_left(item.display_text or item.text):
                x = inner.right() - QFontMetricsF(font).horizontalAdvance(line)

            if shadow:
                shadow_color = QColor(getattr(style, "shadow_color", "#000000"))
                if shadow_color.isValid():
                    shadow_color.setAlpha(200)
                    offset = float(getattr(style, "shadow_offset", 1))
                    painter.setPen(QPen(shadow_color))
                    painter.drawText(
                        QPointF(x + offset, baseline + offset),
                        line,
                    )

            painter.setPen(QPen(text_color))
            painter.drawText(QPointF(x, baseline), line)

    # ------------------------------------------------------------------
    # 排版缓存
    # ------------------------------------------------------------------

    def _get_layout(self, item: RenderItem) -> _CachedLayout | None:
        """取得（并缓存）条目的排版结果。"""
        style = self._style
        inner = self._inner_rect(self._paint_rect_for(item))
        size = self._font_sizes.get(id(item))
        if size is None and item.font_px is not None:
            size = max(READABLE_FONT_PX, int(item.font_px))
        key = (
            item.display_text or item.text,
            inner.width(),
            inner.height(),
            self._style_generation,
            size,
        )
        cached = self._layout_cache.get(key)
        if cached is not None:
            return cached

        result = layout_text(
            item.display_text or item.text,
            inner.width(),
            inner.height(),
            style,
            colour_scheme_dark=item.is_dark_background,
            font_px=size,
            readable_min_px=READABLE_FONT_PX,
        )
        if not result.lines:
            return None

        font = build_font(
            getattr(style, "font_family", "") or "",
            result.font_px,
            bool(getattr(style, "bold", False)),
        )

        cached = _CachedLayout(
            layout=result,
            font=font,
            lines=result.lines,
            line_height=result.line_height,
            truncated=result.truncated,
            elided=list(result.lines) if result.truncated else None,
        )
        self._layout_cache[key] = cached
        return cached


class OverlayManager(QObject):
    """多显示器译文层的统一管理。

    为每块显示器维护一个 :class:`OverlayWindow`，按条目所属显示器分发，
    并处理显示器热插拔（分辨率、缩放、增删屏）后的重建。
    """

    presented = pyqtSignal(object, float)

    def __init__(self, style: object) -> None:
        super().__init__()
        self._pending_presentation: tuple[RenderFrame, set[str]] | None = None
        self._style = style
        self._windows: dict[str, OverlayWindow] = {}
        self._screens: list[ScreenInfo] = []
        self._excluded_regions: list[Rect] = []
        self._visible = True
        self.sync_screens()

    # ------------------------------------------------------------------
    # 显示器同步
    # ------------------------------------------------------------------

    def sync_screens(self) -> list[ScreenInfo]:
        """重新枚举显示器，为新屏幕创建窗口，清理已移除的屏幕窗口。

        :return: 当前显示器列表。
        """
        screens = list_screens()
        self._screens = screens
        current_names = {screen.name for screen in screens}

        for name in list(self._windows.keys()):
            if name not in current_names:
                window = self._windows.pop(name)
                window.close()
                window.deleteLater()
                logger.info("显示器已断开，移除对应翻译层：%s", name)

        for screen in screens:
            existing = self._windows.get(screen.name)
            if existing is None:
                window = OverlayWindow(screen, self._style)
                window.presented.connect(self._window_presented)
                window.set_excluded_regions(self._excluded_regions)
                self._windows[screen.name] = window
                # 热插拔新建的窗口必须按当前开关状态 show，否则新显示器的
                # 翻译层永远空白、直到用户手动关开翻译层（渲染审查 R1，高）。
                window.set_layer_visible(self._visible)
                logger.info("为显示器创建翻译层：%s", screen.label())
            else:
                # 几何变化（分辨率或缩放调整）后重新铺满
                if (
                    existing.screen_info.logical != screen.logical
                    or existing.screen_info.device_pixel_ratio != screen.device_pixel_ratio
                ):
                    existing.refresh_screen(screen)
                    logger.info("显示器几何变化，翻译层已重新适配：%s", screen.label())

        return screens

    @property
    def screens(self) -> list[ScreenInfo]:
        """当前显示器列表。"""
        return self._screens

    @property
    def window_count(self) -> int:
        """当前翻译层窗口数量。"""
        return len(self._windows)

    # ------------------------------------------------------------------
    # 内容分发
    # ------------------------------------------------------------------

    def apply_frame(
        self, frame: RenderFrame, active_monitor_indices: set[int] | None = None,
        replace: bool = True, dirty_regions: list[Rect] | None = None,
    ) -> int:
        source_lines = getattr(frame, "_source_lines", None)
        sources = frame.source_items
        if source_lines is not None:
            sources = [RenderItem(
                Rect.from_sequence(line["rect"]), str(line["text"]),
                source_text=str(line["text"]),
            ) for line in source_lines if line.get("text") and line.get("rect")]
        space_obstacles = getattr(frame, "_space_obstacles", None)
        for window in self._windows.values():
            window.set_layout_context(
                frame.origin, sources, replace, dirty_regions,
                getattr(frame, "_paragraph_provenance", None), space_obstacles,
            )
        count = self.apply(frame.items, active_monitor_indices, replace, dirty_regions)
        details = [entry for window in self._windows.values() for entry in window.paint_evidence()]
        frame.diagnostics["display_overflow"] = sum(
            entry["disposition"] == "full_text_in_panel" for entry in details
        )
        frame.diagnostics["display_uncertain"] = sum(
            entry["disposition"] == "uncertain" for entry in details
        )
        targets = {name for name, window in self._windows.items() if window.isVisible()}
        self._pending_presentation = (frame, targets) if targets else None
        for name in targets:
            self._windows[name]._pending_frame = frame
            self._windows[name].update()
        return count

    def _window_presented(self, window: OverlayWindow, frame: RenderFrame, at: float) -> None:
        pending = self._pending_presentation
        if pending is None or pending[0] is not frame:
            return
        pending[1].discard(window.screen_info.name)
        if not pending[1]:
            self._pending_presentation = None
            self.presented.emit(frame, at)

    def apply(
        self,
        items: list[RenderItem],
        active_monitor_indices: set[int] | None = None,
        replace: bool = True,
        dirty_regions: list[Rect] | None = None,
    ) -> int:
        """把条目按显示器分发到各窗口。

        :param items: 全部译文条目（物理坐标）。
        :param active_monitor_indices: 允许渲染的显示器序号集合；
            ``None`` 表示全部允许。
        :param replace: ``True`` 表示整帧识别，替换整层内容；
            ``False`` 表示增量帧，只更新识别到的那部分，其余译文保持不动
            （否则未变化区域会逐帧闪空再画回）。
        :return: 实际渲染的条目数量。

        序号**未知**（``None``）的条目一律保留，不参与过滤。这一条很关键：
        框选区域捕获曾经没有记录显示器序号，条目因此全是 ``None``，
        被这里当成"不属于任何被捕获的显示器"**整体丢弃**——翻译一直在跑
        （状态栏的帧率与延迟都正常），翻译层却始终空白，极难定位。
        宁可多画一条，也不要静默丢掉全部。
        """
        self._pending_presentation = None
        for window in self._windows.values():
            window._pending_frame = None
        if not self._visible:
            return 0

        buckets: dict[str, list[RenderItem]] = {screen.name: [] for screen in self._screens}
        rendered = 0

        for item in items:
            if not item.is_renderable:
                continue
            if (
                active_monitor_indices
                and item.monitor_index is not None
                and item.monitor_index not in active_monitor_indices
            ):
                continue
            screen = self._screen_for(item)
            if screen is None:
                continue
            buckets[screen.name].append(item)
            rendered += 1

        for name, window in self._windows.items():
            if dirty_regions:
                window.set_items(buckets.get(name, []), replace=replace, dirty_regions=dirty_regions)
            else:
                window.set_items(buckets.get(name, []), replace=replace)

        return rendered

    def clear(self) -> None:
        """清空全部翻译层。"""
        for window in self._windows.values():
            window.clear()

    def set_excluded_regions(self, regions: list[Rect]) -> None:
        """Clip translated paint around visible app windows on every screen."""
        self._excluded_regions = list(regions)
        for window in self._windows.values():
            window.set_excluded_regions(self._excluded_regions)

    def show_all(self, visible: bool = True) -> None:
        """显示或隐藏全部翻译层（全局热键"一键隐藏"使用）。

        这是低频操作，调用方可对其施加淡入淡出；窗口内容本身不做动画。
        """
        self._visible = visible
        for window in self._windows.values():
            window.set_layer_visible(visible)
        if not visible:
            self.clear()

    @property
    def is_visible(self) -> bool:
        """翻译层当前是否显示。"""
        return self._visible

    def apply_style(self, style: object) -> None:
        """把新样式应用到全部窗口。"""
        self._style = style
        for window in self._windows.values():
            window.apply_style(style)

    def set_click_through(self, enabled: bool = True) -> None:
        """开启/关闭鼠标穿透。

        默认穿透（不干扰用户操作）；区域框选等交互场景需临时关闭。
        """
        for window in self._windows.values():
            # 必须在 setWindowFlags **之前**记录可见性：该方法对可见窗口
            # 的语义是"先隐藏再应用新 flags"，之后 isVisible() 恒为 False，
            # show() 永远走不到——翻译层会就此消失且不再恢复
            # （_render_tick 只做 update()，不会重新 show）。
            was_visible = window.isVisible()
            window.setAttribute(
                Qt.WidgetAttribute.WA_TransparentForMouseEvents, enabled
            )
            flags = window.windowFlags()
            if enabled:
                flags |= Qt.WindowType.WindowTransparentForInput
            else:
                flags &= ~Qt.WindowType.WindowTransparentForInput
            window.setWindowFlags(flags)
            if was_visible:
                window.show()

    def shutdown(self) -> None:
        """关闭全部翻译层。"""
        for window in self._windows.values():
            window.close()
            window.deleteLater()
        self._windows.clear()

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _screen_for(self, item: RenderItem) -> ScreenInfo | None:
        """查找条目所属显示器。"""
        if not self._screens:
            return None

        cx, cy = item.box.center
        for screen in self._screens:
            physical = screen.physical
            if physical.left <= cx < physical.right and physical.top <= cy < physical.bottom:
                return screen

        # 坐标落在显示器之间的空隙（虚拟桌面拼接缝隙）时，归入最近的显示器
        if item.monitor_index is not None:
            for screen in self._screens:
                if screen.mss_index == item.monitor_index:
                    return screen

        return self._screens[0]


# ----------------------------------------------------------------------
# 辅助函数
# ----------------------------------------------------------------------


def _to_qrect(rect: Rect) -> QRect:
    """把自定义 :class:`Rect` 转换为 Qt 的 :class:`QRect`。"""
    return QRect(rect.x, rect.y, max(1, rect.width), max(1, rect.height))


def _is_source_copy(item: RenderItem) -> bool:
    """Unchanged source text occupies geometry but needs no second paint layer."""
    return bool(item.source_text.strip()) and (
        " ".join(item.text.split()) == " ".join(item.source_text.split())
    )


def _collision_free_rects(items: list[RenderItem]) -> dict[int, QRectF]:
    """Partition overlapping OCR boxes while keeping every slot inside its box.

    Only shrink; this preserves original alignment and cannot create a new
    collision with a previously processed pair. Geometry/text sorting makes
    the allocation independent of incremental input order.
    """
    rects = [QRectF(_to_qrect(item.box)) for item in items]
    order = sorted(range(len(items)), key=lambda index: (
        items[index].box.top, items[index].box.left,
        items[index].box.bottom, items[index].box.right, items[index].text,
    ))
    for position, index in enumerate(order):
        first = rects[index]
        for other_index in order[position + 1:]:
            if items[other_index].box.top >= first.bottom():
                break
            second = rects[other_index]
            overlap = first.intersected(second)
            if overlap.isEmpty():
                continue
            vertical_fraction = overlap.height() / min(first.height(), second.height())
            horizontal_fraction = overlap.width() / min(first.width(), second.width())
            if vertical_fraction <= horizontal_fraction:
                cut = (overlap.top() + overlap.bottom()) / 2
                if first.center().y() <= second.center().y():
                    first.setBottom(cut)
                    second.setTop(cut)
                else:
                    second.setBottom(cut)
                    first.setTop(cut)
            else:
                cut = (overlap.left() + overlap.right()) / 2
                if first.center().x() <= second.center().x():
                    first.setRight(cut)
                    second.setLeft(cut)
                else:
                    second.setRight(cut)
                    first.setLeft(cut)
    return {id(item): rect for item, rect in zip(items, rects, strict=True)}


def _ascent(font: QFont) -> float:
    """返回字体的上升高度（基线到顶部）。"""
    return QFontMetricsF(font).ascent()


def refresh_screen_cache() -> None:
    """清空 Qt 的屏幕缓存，便于在显示器变化后立即重新枚举。"""
    app = QGuiApplication.instance()
    if app is not None and hasattr(app, "screens"):
        # 触发一次屏幕枚举，使 QGuiApplication 更新内部缓存
        list(app.screens())
