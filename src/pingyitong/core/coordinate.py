"""坐标映射与对齐。

对应《项目原创性声明》二·（三）"原文坐标到屏幕坐标的映射、高 DPI 缩放适配"。

映射链路（全链路均为显式换算，不依赖隐式假设）：

``OCR 局部坐标``
    → 加上该帧 ``origin``，得到**虚拟桌面物理坐标**
    → 除以所属显示器的 ``devicePixelRatio``，得到**虚拟桌面逻辑坐标**
    → 减去窗口原点，得到**窗口内局部逻辑坐标**（由渲染层完成）

本模块还负责"渲染条目归属显示器"的判定，这是多显示器场景下
译文画到正确屏幕上的前提。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..components.capture.geometry import (
    Rect,
    ScreenInfo,
    list_screens,
    screen_at,
)
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class MappedBox:
    """一次映射的结果，同时保留两套坐标以便排查对齐问题。"""

    #: 物理坐标（虚拟桌面）
    physical: Rect
    #: 逻辑坐标（虚拟桌面）
    logical: Rect
    #: 所属显示器名称
    screen_name: str
    #: 该显示器的缩放比例
    device_pixel_ratio: float


class CoordinateMapper:
    """物理坐标与逻辑坐标之间的双向映射器。

    显示器配置（分辨率、缩放、排列）发生变化时必须调用
    :meth:`refresh`，否则会沿用过期的缩放比例导致译文偏移。
    """

    def __init__(self) -> None:
        self._screens: list[ScreenInfo] = []
        self.refresh()

    # ------------------------------------------------------------------
    # 显示器配置
    # ------------------------------------------------------------------

    def refresh(self) -> list[ScreenInfo]:
        """重新枚举显示器并刷新缓存。"""
        self._screens = list_screens()
        if self._screens:
            summary = ", ".join(
                f"{s.name}@{s.scale_percent}%" for s in self._screens
            )
            logger.debug("坐标映射已刷新：%s", summary)
        return self._screens

    @property
    def screens(self) -> list[ScreenInfo]:
        """当前显示器列表。"""
        return self._screens

    def set_screens(self, screens: list[ScreenInfo]) -> None:
        """直接注入显示器列表，便于单元测试。"""
        self._screens = list(screens)

    def screen_count(self) -> int:
        """显示器数量。"""
        return len(self._screens)

    def is_single_screen(self) -> bool:
        """是否只有一块屏幕（单屏时可跳过部分映射计算）。"""
        return len(self._screens) <= 1

    # ------------------------------------------------------------------
    # 映射
    # ------------------------------------------------------------------

    def screen_for_physical(self, rect: Rect) -> ScreenInfo | None:
        """查找物理坐标矩形所属显示器。"""
        return screen_at(rect.center, self._screens, logical=False)

    def screen_for_logical(self, rect: Rect) -> ScreenInfo | None:
        """查找逻辑坐标矩形所属显示器。"""
        return screen_at(rect.center, self._screens, logical=True)

    def physical_to_logical(self, rect: Rect) -> Rect:
        """物理坐标 → 逻辑坐标。

        换算公式遵循 Qt 的 ``QHighDpiScaling`` 规则：

        .. code-block:: text

            逻辑坐标 = 屏幕物理原点 + (物理坐标 - 屏幕物理原点) / 缩放比例

        注意**屏幕原点本身不做缩放**——只缩放尺寸。这一点极易写错：
        若把原点也除以缩放比例，副屏（缩放 150%）上的译文会整体偏移
        数百像素，甚至落到屏幕之外。原因见 :meth:`to_window_coords`。
        """
        screen = self.screen_for_physical(rect)
        return self._scale_within_screen(rect, screen, forward=False)

    def logical_to_physical(self, rect: Rect) -> Rect:
        """逻辑坐标 → 物理坐标，是 :meth:`physical_to_logical` 的逆运算。"""
        screen = self.screen_for_logical(rect)
        return self._scale_within_screen(rect, screen, forward=True)

    @staticmethod
    def _scale_within_screen(
        rect: Rect,
        screen: ScreenInfo | None,
        forward: bool,
    ) -> Rect:
        """在单块显示器内做物理/逻辑换算。

        :param forward: ``True`` 表示逻辑 → 物理，``False`` 表示物理 → 逻辑。
        """
        ratio = CoordinateMapper._ratio_of(screen)
        if ratio == 1.0 or screen is None:
            return rect

        origin = screen.physical
        factor = ratio if forward else 1.0 / ratio

        return Rect(
            origin.x + round((rect.x - origin.x) * factor),
            origin.y + round((rect.y - origin.y) * factor),
            max(1, round(rect.width * factor)),
            max(1, round(rect.height * factor)),
        )

    def map_box(self, physical: Rect) -> MappedBox:
        """一次性得到两套坐标与所属显示器信息。"""
        screen = self.screen_for_physical(physical)
        logical = self._scale_within_screen(physical, screen, forward=False)
        return MappedBox(
            physical=physical,
            logical=logical,
            screen_name=screen.name if screen else "",
            device_pixel_ratio=self._ratio_of(screen),
        )

    def to_window_coords(self, physical: Rect, screen: ScreenInfo) -> Rect:
        """物理坐标 → 指定窗口内的局部逻辑坐标。

        渲染层直接使用该方法，因此窗口移动或屏幕重排后无需改动渲染代码。

        由于 Qt 把每块屏幕的逻辑原点放在其物理原点上（只缩放尺寸），
        "窗口内局部坐标"就是把物理坐标相对**该屏物理原点**的偏移按比例
        折算——不需要再去减逻辑原点，否则会把已经对齐的原点重复扣减
        一次，导致译文被推出窗口。
        """
        ratio = screen.device_pixel_ratio or 1.0
        origin = screen.physical

        return Rect(
            round((physical.x - origin.x) / ratio),
            round((physical.y - origin.y) / ratio),
            max(1, round(physical.width / ratio)),
            max(1, round(physical.height / ratio)),
        )

    def virtual_desktop_physical(self) -> Rect:
        """返回虚拟桌面的物理包围盒。"""
        if not self._screens:
            return Rect(0, 0, 0, 0)
        result = self._screens[0].physical
        for screen in self._screens[1:]:
            result = result.union(screen.physical)
        return result

    def logical_bounds_of(self, screen_name: str) -> Rect | None:
        """返回指定显示器的逻辑几何。"""
        for screen in self._screens:
            if screen.name == screen_name:
                return screen.logical
        return None

    # ------------------------------------------------------------------
    # 捕获区域换算
    # ------------------------------------------------------------------

    def capture_region_physical(self, region: Rect) -> Rect:
        """把用户在界面上框选的逻辑区域换算为捕获所需的物理区域。

        捕获区域必须按显示器对齐，否则跨屏框选会得到非矩形位图，
        导致后续坐标换算出错。

        **换算直接复用 :meth:`logical_to_physical`**，不另写一份：
        这里曾经自己写了一遍 ``x * ratio``，而正确规则是"屏幕原点不缩放、
        只缩放尺寸"（见 :meth:`logical_to_physical`）。两份实现只在
        物理原点为 0 的屏幕上结果相同，接非主屏或副屏就会偏移。
        """
        screen = self.screen_for_logical(region)
        physical = self.logical_to_physical(region)
        if screen is None:
            return physical
        # 裁剪到该显示器范围内，保证捕获区域是单屏内的矩形
        return physical.clamp(screen.physical)

    def region_for_monitor(self, monitor_index: int | None) -> Rect | None:
        """返回指定显示器（mss 序号）对应的物理矩形。"""
        if monitor_index is None:
            return None
        for screen in self._screens:
            if screen.mss_index == monitor_index:
                return screen.physical
        return None

    def monitor_index_for_physical(self, rect: Rect) -> int | None:
        """返回物理矩形所在显示器的 ``mss`` 序号。

        框选区域捕获时必须知道这块区域属于哪块屏幕：显示器序号会一路带到
        渲染阶段，而渲染阶段按显示器分发条目。缺少它会让条目被判为
        "不属于任何被捕获的显示器"而被丢弃，表现为翻译层永远空白。
        """
        screen = screen_at(rect.center, self._screens, logical=False)
        return screen.mss_index if screen is not None else None

    def monitor_indices(self) -> list[int | None]:
        """返回全部显示器的 mss 序号。"""
        return [screen.mss_index for screen in self._screens]

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    @staticmethod
    def _ratio_of(screen: ScreenInfo | None) -> float:
        """取缩放比例，无效值回落为 1.0。"""
        if screen is None:
            return 1.0
        ratio = screen.device_pixel_ratio or 1.0
        return ratio if ratio > 0 else 1.0
