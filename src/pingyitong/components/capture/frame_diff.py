"""画面差分算法（自研）。

对应《项目原创性声明》二·（一）"基于画面差分的变化区域增量处理机制"。

算法流程：

1. 比较相邻两帧各颜色通道的逐像素绝对差；
2. 按阈值二值化，得到变化掩膜；
3. 形态学膨胀把相邻文字笔画连成整体，避免把一行文字切成碎片；
4. 提取轮廓包围盒，过滤过小噪点；
5. 迭代合并相距较近的包围盒，减少送入 OCR 的区域数量；
6. 用有固定像素上限的噪声阈值，避免大屏的小字变化被面积比例吞掉。

相比"整屏 OCR"，该机制把静态界面的识别开销降到接近零。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...utils.logger import get_logger

logger = get_logger(__name__)

try:  # pragma: no cover - 依赖缺失时在下方降级
    import cv2
    import numpy as np

    _CV_AVAILABLE = True
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    _CV_AVAILABLE = False

#: 单个区域最多合并出的变化块数量。
#:
#: 这个上限直接决定每帧要重新识别多少文字块，而 OCR 的耗时与**文本块数量**
#: 成正比（实测约 31ms/块）。因此区域划分得越细，每帧重识别的内容越少、
#: 延迟越低——这正是"跟随滚动"场景的关键。
#:
#: 早期取值偏小（12），超出时会把剩余小区域全部并成一个整屏大块，
#: 导致每帧都近乎整屏重识别（实测单帧 OCR 达 5–9 秒）。
MAX_REGIONS = 24


@dataclass
class DiffResult:
    """一次帧差分的结果。"""

    #: 是否存在值得处理的画面变化
    changed: bool
    #: 变化区域（局部坐标，相对于传入图像左上角）
    regions: list[tuple[int, int, int, int]] = field(default_factory=list)
    #: 变化像素占比，用于日志与调优
    change_ratio: float = 0.0

    @property
    def region_count(self) -> int:
        return len(self.regions)

    @classmethod
    def full_frame(cls, width: int, height: int) -> DiffResult:
        """构造"整帧变化"结果，用于强制全量识别。"""
        return cls(changed=True, regions=[(0, 0, width, height)], change_ratio=1.0)


class FrameDiffer:
    """帧间差分器。

    实例可复用，内部不保留帧数据（由调用方持有上一帧），
    因此同一实例可在多个显示器之间共享。
    """

    def __init__(
        self,
        pixel_threshold: int = 24,
        min_area_ratio: float = 0.0004,
        block_size: int = 16,
        max_regions: int = MAX_REGIONS,
    ) -> None:
        self.pixel_threshold = max(0, min(255, int(pixel_threshold)))
        self.min_area_ratio = max(0.0, min(1.0, float(min_area_ratio)))
        self.block_size = max(2, int(block_size))
        self.max_regions = max(1, int(max_regions))

    # ------------------------------------------------------------------
    # 参数
    # ------------------------------------------------------------------

    def configure(
        self,
        pixel_threshold: int | None = None,
        min_area_ratio: float | None = None,
        block_size: int | None = None,
    ) -> None:
        """运行时更新参数（用户在设置面板调整后立即生效）。"""
        if pixel_threshold is not None:
            self.pixel_threshold = max(0, min(255, int(pixel_threshold)))
        if min_area_ratio is not None:
            self.min_area_ratio = max(0.0, min(1.0, float(min_area_ratio)))
        if block_size is not None:
            self.block_size = max(2, int(block_size))

    # ------------------------------------------------------------------
    # 计算
    # ------------------------------------------------------------------

    def compute(self, previous, current) -> DiffResult:
        """比较两帧，返回变化区域。

        :param previous: 上一帧图像（BGR 或灰度 numpy 数组）。
        :param current: 当前帧图像，尺寸需与 ``previous`` 一致。
        :return: :class:`DiffResult`。依赖缺失或尺寸不一致时保守返回"整帧变化"。
        """
        if not _CV_AVAILABLE:
            return DiffResult.full_frame(_width(current), _height(current))

        if previous is None or current is None:
            return DiffResult.full_frame(_width(current), _height(current))

        height, width = current.shape[:2]
        if previous.shape[:2] != (height, width):
            # 分辨率变化（切换显示器、缩放变更）时无法差分，回退整帧。
            return DiffResult.full_frame(width, height)

        # Equal-luminance colours can carry different text. Comparing only
        # grayscale silently misses those edits, including warning colours.
        delta = cv2.absdiff(previous, current)
        if delta.ndim == 3:
            delta = np.max(delta[:, :, :3], axis=2)
        # Preserve faint source edits too. The legacy 24-level threshold can
        # hide a whole light-grey word; a small fixed area rejects isolated
        # pixel noise without discarding readable low-contrast strokes.
        _, mask = cv2.threshold(delta, min(self.pixel_threshold, 2), 255, cv2.THRESH_BINARY)

        changed_pixels = int(cv2.countNonZero(mask))
        total_pixels = max(1, width * height)
        ratio = changed_pixels / total_pixels

        # A single digit must remain detectable on a large/multi-monitor
        # desktop. Bound the noise floor in pixels, not just screen area.
        minimum_pixels = max(1, min(4, int(total_pixels * self.min_area_ratio)))
        if changed_pixels < minimum_pixels:
            return DiffResult(changed=False, regions=[], change_ratio=ratio)

        # 膨胀：把一行文字中断裂的笔画连成整体，减少后续轮廓数量。
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.block_size, self.block_size)
        )
        dilated = cv2.dilate(mask, kernel, iterations=1)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes: list[tuple[int, int, int, int]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((x, y, w, h))

        if not boxes:
            return DiffResult(changed=False, regions=[], change_ratio=ratio)

        # 合并容差取 1 个块大小而不是 2 个：容差越大区域越少但越大，
        # 而每帧重识别的文字块数量随区域面积增长，直接推高延迟。
        boxes = self._merge_boxes(boxes, gap=self.block_size)
        boxes = self._limit_regions(boxes)
        return DiffResult(changed=True, regions=boxes, change_ratio=ratio)

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _merge_boxes(
        self,
        boxes: list[tuple[int, int, int, int]],
        gap: int,
    ) -> list[tuple[int, int, int, int]]:
        """迭代合并彼此靠近或重叠的包围盒。

        以 ``gap`` 为容差，反复扫描直到没有可合并的矩形为止。
        矩形数量通常很小（十几个），因此二次复杂度可以接受。
        """
        pending = list(boxes)
        merged_any = True

        while merged_any and len(pending) > 1:
            merged_any = False
            result: list[tuple[int, int, int, int]] = []

            while pending:
                current = pending.pop()
                cx, cy, cw, ch = current
                absorbed = False

                for index, other in enumerate(pending):
                    ox, oy, ow, oh = other
                    if _within_gap(cx, cy, cw, ch, ox, oy, ow, oh, gap):
                        union = _union(cx, cy, cw, ch, ox, oy, ow, oh)
                        pending[index] = union
                        absorbed = True
                        merged_any = True
                        break

                if not absorbed:
                    result.append(current)

            pending = result

        return pending

    def _limit_regions(
        self,
        boxes: list[tuple[int, int, int, int]],
    ) -> list[tuple[int, int, int, int]]:
        """限制区域数量上限，把多余的小块并入整体包围盒。

        区域过多意味着屏幕大面积刷新（例如滚动整页），此时逐块 OCR
        反而比整屏 OCR 更慢，因此合并为一个整块。
        """
        if len(boxes) <= self.max_regions:
            return sorted(boxes, key=lambda b: (b[1], b[0]))

        if len(boxes) > self.max_regions * 3:
            # 极端情况：直接返回所有矩形的并集，等价于整屏处理。
            xs = [b[0] for b in boxes]
            ys = [b[1] for b in boxes]
            rights = [b[0] + b[2] for b in boxes]
            bottoms = [b[1] + b[3] for b in boxes]
            return [(
                min(xs),
                min(ys),
                max(rights) - min(xs),
                max(bottoms) - min(ys),
            )]

        # 保留面积最大的若干个，其余并入其并集。
        boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
        kept = boxes[: self.max_regions - 1]
        rest = boxes[self.max_regions - 1 :]

        merged = rest[0]
        for other in rest[1:]:
            merged = _union(*merged, *other)
        kept.append(merged)

        return sorted(kept, key=lambda b: (b[1], b[0]))


def is_available() -> bool:
    """返回差分所需的 OpenCV / NumPy 是否可用。"""
    return _CV_AVAILABLE


# ----------------------------------------------------------------------
# 模块级辅助函数
# ----------------------------------------------------------------------


def _to_gray(image):
    """转为单通道灰度图，输入本就是灰度时原样返回。"""
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _width(image) -> int:
    if image is None:
        return 0
    return int(image.shape[1]) if getattr(image, "ndim", 0) >= 2 else 0


def _height(image) -> int:
    if image is None:
        return 0
    return int(image.shape[0]) if getattr(image, "ndim", 0) >= 2 else 0


def _union(
    ax: int, ay: int, aw: int, ah: int,
    bx: int, by: int, bw: int, bh: int,
) -> tuple[int, int, int, int]:
    """返回两个矩形的并集。"""
    left = min(ax, bx)
    top = min(ay, by)
    right = max(ax + aw, bx + bw)
    bottom = max(ay + ah, by + bh)
    return (left, top, right - left, bottom - top)


def _within_gap(
    ax: int, ay: int, aw: int, ah: int,
    bx: int, by: int, bw: int, bh: int,
    gap: int,
) -> bool:
    """判断两矩形在容差 ``gap`` 内是否相交或相邻。"""
    separated_x = ax + aw + gap < bx or bx + bw + gap < ax
    separated_y = ay + ah + gap < by or by + bh + gap < ay
    return not separated_x and not separated_y
