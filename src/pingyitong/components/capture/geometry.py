"""屏幕几何与高 DPI 坐标映射。

系统中有两套坐标系，必须显式区分，否则译文会错位：

**物理坐标**
    ``mss`` 捕获到的位图坐标，以虚拟桌面左上角为原点，单位为物理像素。
    多显示器时原点可能为负值（副屏位于主屏左侧或上方）。

**逻辑坐标**
    Qt 窗口与鼠标事件使用的坐标，已按 Windows 缩放比例折算，
    单位为设备无关像素。``devicePixelRatio`` 是两个坐标系之间的比例。

本模块提供两者之间的转换，是"高 DPI 屏幕坐标精确对齐"的实现基础。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ...utils.logger import get_logger

logger = get_logger(__name__)


def _new_mss(mss_module: object) -> object:
    """创建 mss 实例，兼容新旧 API。

    ``mss`` 10.x 起把工厂 ``mss.mss()`` 改名为 ``mss.MSS()``，
    旧名会在未来版本移除。这里优先用新名，同时保留对旧版本的兼容。
    """
    factory = getattr(mss_module, "MSS", None) or mss_module.mss
    return factory()


@dataclass(frozen=True)
class Rect:
    """矩形区域，左上角原点，含宽高。"""

    x: int
    y: int
    width: int
    height: int

    # ------------------------------------------------------------------
    # 派生属性
    # ------------------------------------------------------------------

    @property
    def left(self) -> int:
        return self.x

    @property
    def top(self) -> int:
        return self.y

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    @property
    def is_empty(self) -> bool:
        return self.width <= 0 or self.height <= 0

    # ------------------------------------------------------------------
    # 构造
    # ------------------------------------------------------------------

    @classmethod
    def from_ltrb(cls, left: int, top: int, right: int, bottom: int) -> Rect:
        """由左上右下四点构造。"""
        return cls(int(left), int(top), int(right) - int(left), int(bottom) - int(top))

    @classmethod
    def from_sequence(cls, values: Sequence[int]) -> Rect:
        """由 ``[x, y, w, h]`` 序列构造。"""
        if len(values) != 4:
            raise ValueError(f"矩形需要 4 个数值，收到 {len(values)} 个")
        return cls(int(values[0]), int(values[1]), int(values[2]), int(values[3]))

    # ------------------------------------------------------------------
    # 运算
    # ------------------------------------------------------------------

    def to_sequence(self) -> list[int]:
        """转换为 ``[x, y, w, h]`` 列表，便于 JSON 持久化。"""
        return [self.x, self.y, self.width, self.height]

    def to_ltrb(self) -> tuple[int, int, int, int]:
        """转换为 ``(left, top, right, bottom)`` 元组。"""
        return (self.left, self.top, self.right, self.bottom)

    def intersects(self, other: Rect) -> bool:
        """判断与另一矩形是否相交。"""
        return not (
            other.left >= self.right
            or other.right <= self.left
            or other.top >= self.bottom
            or other.bottom <= self.top
        )

    def intersection(self, other: Rect) -> Rect:
        """返回与另一矩形的交集，无交集时返回空矩形。"""
        left = max(self.left, other.left)
        top = max(self.top, other.top)
        right = min(self.right, other.right)
        bottom = min(self.bottom, other.bottom)
        if right <= left or bottom <= top:
            return Rect(left, top, 0, 0)
        return Rect.from_ltrb(left, top, right, bottom)

    def union(self, other: Rect) -> Rect:
        """返回能同时包含两个矩形的最小矩形。"""
        left = min(self.left, other.left)
        top = min(self.top, other.top)
        right = max(self.right, other.right)
        bottom = max(self.bottom, other.bottom)
        return Rect.from_ltrb(left, top, right, bottom)

    def expand(self, dx: int, dy: int | None = None) -> Rect:
        """向外扩展。``dy`` 省略时与 ``dx`` 相同。"""
        dy = dx if dy is None else dy
        return Rect(self.x - dx, self.y - dy, self.width + 2 * dx, self.height + 2 * dy)

    def clamp(self, bounds: Rect) -> Rect:
        """裁剪到给定边界内。"""
        return self.intersection(bounds)

    def translate(self, dx: int, dy: int) -> Rect:
        """平移。"""
        return Rect(self.x + dx, self.y + dy, self.width, self.height)

    def scale(self, factor: float) -> Rect:
        """按比例缩放（宽高与位置同时缩放）。"""
        return Rect(
            round(self.x * factor),
            round(self.y * factor),
            round(self.width * factor),
            round(self.height * factor),
        )

    def to_local(self, origin: Rect) -> Rect:
        """转换为相对于 ``origin`` 左上角的局部坐标。"""
        return Rect(self.x - origin.x, self.y - origin.y, self.width, self.height)

    def to_global(self, origin: Rect) -> Rect:
        """将局部坐标还原为相对虚拟桌面的全局坐标。"""
        return Rect(self.x + origin.x, self.y + origin.y, self.width, self.height)


@dataclass(frozen=True)
class ScreenInfo:
    """单个显示器的几何与缩放信息。"""

    #: Qt 屏幕对象名（如 ``\\\\.\\DISPLAY1``）
    name: str
    #: 逻辑坐标下的屏幕矩形
    logical: Rect
    #: 物理坐标下的屏幕矩形
    physical: Rect
    #: 缩放比例，1.0 表示 100%
    device_pixel_ratio: float
    #: 是否为系统主显示器
    is_primary: bool
    #: ``mss`` 的显示器索引（``monitors[1..n]``），未匹配到时为 ``None``
    mss_index: int | None = None

    @property
    def scale_percent(self) -> int:
        """缩放百分比，用于界面展示（如 150）。"""
        return round(self.device_pixel_ratio * 100)

    def label(self) -> str:
        """用于界面展示的名称。"""
        suffix = "（主屏）" if self.is_primary else ""
        return f"{self.name} · {self.logical.width}×{self.logical.height} @{self.scale_percent}%{suffix}"


def list_screens(app: object | None = None) -> list[ScreenInfo]:
    """枚举全部显示器。

    优先使用 Qt 获取逻辑几何与缩放比例；``mss`` 索引通过几何匹配补全，
    以便捕获时按显示器取图。

    :param app: 预留参数，接受 ``QApplication`` 实例以加快屏幕枚举；
        为 ``None`` 时使用 ``QGuiApplication`` 的全局实例。
    :return: 按系统顺序排列的显示器列表。
    """
    try:
        from PyQt6.QtGui import QGuiApplication
    except ImportError:  # pragma: no cover - PyQt6 为硬依赖，此处仅防御
        logger.error("未安装 PyQt6，无法枚举显示器")
        return []

    screens = QGuiApplication.screens()
    if not screens:
        logger.warning("未检测到任何显示器")
        return []

    primary = QGuiApplication.primaryScreen()

    # 延迟导入避免无 mss 环境下的导入错误影响几何模块使用
    mss_monitors: list[dict[str, int]] = []
    try:
        import mss

        with _new_mss(mss) as sct:
            mss_monitors = list(sct.monitors[1:])
    except Exception:  # noqa: BLE001 - mss 缺失或平台不支持时降级为无索引
        logger.debug("mss 不可用，显示器索引映射被跳过")

    result: list[ScreenInfo] = []
    used_indices: set[int] = set()

    for screen in screens:
        geo = screen.geometry()
        ratio = float(screen.devicePixelRatio() or 1.0)
        logical = Rect(geo.x(), geo.y(), geo.width(), geo.height())
        # 与 Qt 的 QHighDpiScaling 一致：屏幕原点**不参与缩放**，
        # 只缩放尺寸。Qt 6 在逐屏缩放模式下把每块屏幕的逻辑原点放在
        # 它的物理原点上（mss 报告的也是同一个原点），因此物理矩形的
        # x/y 必须原样保留。若把原点也乘上缩放比例，非 100% 缩放的
        # 副屏会整体偏移 origin × (ratio-1) 像素——mss 索引匹配失败、
        # 译文被画出窗口。本模块的 logical_to_physical /
        # physical_to_logical 也以此为前提（origin = screen.physical）。
        physical = Rect(
            geo.x(),
            geo.y(),
            round(geo.width() * ratio),
            round(geo.height() * ratio),
        )

        index = _match_mss_index(physical, mss_monitors, used_indices)
        if index is not None:
            used_indices.add(index)

        result.append(
            ScreenInfo(
                name=screen.name(),
                logical=logical,
                physical=physical,
                device_pixel_ratio=ratio,
                is_primary=screen is primary,
                mss_index=index,
            )
        )

    return result


def _match_mss_index(
    physical: Rect,
    monitors: Iterable[dict[str, int]],
    used: set[int],
) -> int | None:
    """在 ``mss`` 显示器列表中查找与给定物理几何一致的项。

    ``mss`` 的索引顺序不保证与 Qt 一致，因此按几何尺寸匹配而不是按下标对应。
    允许 ±2 像素误差，以容忍个别驱动的取整差异。
    """
    for position, monitor in enumerate(monitors, start=1):
        if position in used:
            continue
        same_size = (
            abs(monitor.get("width", 0) - physical.width) <= 2
            and abs(monitor.get("height", 0) - physical.height) <= 2
        )
        same_origin = (
            abs(monitor.get("left", 0) - physical.x) <= 2
            and abs(monitor.get("top", 0) - physical.y) <= 2
        )
        if same_size and same_origin:
            return position

    # 原点不一致但尺寸唯一匹配时（部分虚拟显示驱动会规整原点），退一步按尺寸匹配
    candidates = [
        position
        for position, monitor in enumerate(monitors, start=1)
        if position not in used
        and abs(monitor.get("width", 0) - physical.width) <= 2
        and abs(monitor.get("height", 0) - physical.height) <= 2
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


def virtual_desktop_rect(screens: Sequence[ScreenInfo] | None = None) -> Rect:
    """返回覆盖全部显示器的虚拟桌面矩形（物理坐标）。

    多显示器且副屏位于主屏左侧或上方时，返回值原点可能为负。
    """
    screens = screens if screens is not None else list_screens()
    if not screens:
        return Rect(0, 0, 0, 0)

    result = screens[0].physical
    for screen in screens[1:]:
        result = result.union(screen.physical)
    return result


def logical_to_physical(rect: Rect, screens: Sequence[ScreenInfo]) -> Rect:
    """将逻辑坐标矩形转换为物理坐标。

    换算规则与 Qt ``QHighDpiScaling`` 一致：**屏幕原点不缩放，只缩放尺寸**。

    .. code-block:: text

        物理坐标 = 屏幕物理原点 + (逻辑坐标 - 屏幕物理原点) * 缩放比例

    之所以不使用 ``逻辑坐标 * 缩放比例``：那会把屏幕原点也乘一次缩放，
    在 150% 缩放的副屏上造成数百像素的偏移，译文会画到错误的位置甚至
    屏幕之外。参见 :class:`~pingyitong.core.coordinate.CoordinateMapper`。
    """
    screen = screen_at(rect.center, screens, logical=True)
    if screen is None:
        return rect

    ratio = screen.device_pixel_ratio or 1.0
    if ratio == 1.0:
        return rect

    origin = screen.physical
    return Rect(
        origin.x + round((rect.x - origin.x) * ratio),
        origin.y + round((rect.y - origin.y) * ratio),
        max(1, round(rect.width * ratio)),
        max(1, round(rect.height * ratio)),
    )


def physical_to_logical(rect: Rect, screens: Sequence[ScreenInfo]) -> Rect:
    """将物理坐标矩形转换为逻辑坐标（:func:`logical_to_physical` 的逆运算）。"""
    screen = screen_at(rect.center, screens, logical=False)
    if screen is None:
        return rect

    ratio = screen.device_pixel_ratio or 1.0
    if ratio <= 1e-6:
        ratio = 1.0
    if ratio == 1.0:
        return rect

    origin = screen.physical
    return Rect(
        origin.x + round((rect.x - origin.x) / ratio),
        origin.y + round((rect.y - origin.y) / ratio),
        max(1, round(rect.width / ratio)),
        max(1, round(rect.height / ratio)),
    )


def screen_at(
    point: tuple[int, int],
    screens: Sequence[ScreenInfo],
    logical: bool = True,
) -> ScreenInfo | None:
    """查找包含指定点的显示器。

    :param point: ``(x, y)`` 坐标。
    :param logical: 该点使用逻辑坐标还是物理坐标。
    """
    x, y = point
    for screen in screens:
        rect = screen.logical if logical else screen.physical
        if rect.left <= x < rect.right and rect.top <= y < rect.bottom:
            return screen

    # 边界情况：点落在所有显示器之外（例如多屏之间的空隙），取最近的显示器
    if not screens:
        return None
    best = screens[0]
    best_distance = float("inf")
    for screen in screens:
        rect = screen.logical if logical else screen.physical
        cx, cy = rect.center
        distance = (cx - x) ** 2 + (cy - y) ** 2
        if distance < best_distance:
            best_distance = distance
            best = screen
    return best


def region_to_monitor(region: Rect, screens: Sequence[ScreenInfo]) -> ScreenInfo | None:
    """判断区域主要落在哪个显示器上，用于选择捕获目标。"""
    best: ScreenInfo | None = None
    best_area = 0
    for screen in screens:
        overlap = region.intersection(screen.logical)
        if overlap.area > best_area:
            best_area = overlap.area
            best = screen
    return best


def normalise_region(start: tuple[int, int], end: tuple[int, int]) -> Rect:
    """由拖拽的起止点构造规范化矩形（支持任意拖拽方向）。"""
    left, right = sorted((start[0], end[0]))
    top, bottom = sorted((start[1], end[1]))
    return Rect.from_ltrb(left, top, right, bottom)
