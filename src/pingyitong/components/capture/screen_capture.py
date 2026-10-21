"""桌面实时捕获模块。

基于 ``python-mss`` 抓取桌面画面。选择 mss 的原因（见开发文档二·（一））：
纯 Python 实现、截图速度远超 PIL 等传统方案、支持多显示器。

自研部分：捕获节流与背压、区域裁剪、帧率动态控制。

线程约束
    mss 的 ``MSS`` 对象内部持有线程相关句柄，**必须在捕获线程内创建与销毁**。
    因此本模块使用线程局部存储，允许在一个线程中打开、在另一个线程中关闭。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from ...utils.logger import get_logger
from .geometry import Rect

logger = get_logger(__name__)

try:  # pragma: no cover
    import mss as _mss

    _MSS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _mss = None  # type: ignore[assignment]
    _MSS_AVAILABLE = False

try:  # pragma: no cover
    import cv2

    _CV_AVAILABLE = True
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]
    _CV_AVAILABLE = False


@dataclass
class CaptureFrame:
    """一帧桌面画面及其元数据。

    坐标约定：``origin`` 与 ``changed_regions`` 均使用**物理坐标**
    （相对虚拟桌面左上角），以便与 OCR 输出的包围盒直接相加。
    """

    #: 递增帧序号，用于丢弃过期结果
    seq: int
    #: 捕获时刻（``time.perf_counter()``），用于帧率统计与延迟计算
    timestamp: float
    #: BGR 图像，``numpy.ndarray``，形状 ``(h, w, 3)``
    image: Any
    #: 该帧在虚拟桌面中的物理位置
    origin: Rect
    #: 来源显示器索引（``mss`` 的 monitor 序号），区域捕获时为 ``None``
    monitor_index: int | None = None
    #: 本帧需要识别的区域（局部坐标，相对于 ``origin``）。
    #: 为空表示整帧都需要识别。
    changed_regions: list[Rect] = field(default_factory=list)
    #: 是否整帧处理（首帧、分辨率变化、差分关闭时为 ``True``）
    is_full_frame: bool = True
    #: 本帧自身捕获耗时；不可在异步阶段读取捕获器的“最近一次”耗时。
    capture_cost: float = 0.0

    @property
    def width(self) -> int:
        return int(self.image.shape[1]) if self.image is not None else 0

    @property
    def height(self) -> int:
        return int(self.image.shape[0]) if self.image is not None else 0

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)

    def effective_regions(self) -> list[Rect]:
        """返回实际需要 OCR 的区域（局部坐标）。

        整帧处理时返回覆盖全图的单个区域；否则返回差分得到的变化区域。
        区域会被裁剪到图像边界内，避免越界读取。
        """
        bounds = Rect(0, 0, self.width, self.height)
        if self.is_full_frame or not self.changed_regions:
            return [bounds]

        clamped: list[Rect] = []
        for region in self.changed_regions:
            clipped = region.clamp(bounds)
            if not clipped.is_empty:
                clamped.append(clipped)
        return clamped or [bounds]

    def crop(self, region: Rect) -> Any:
        """裁剪出指定区域（局部坐标）的视图。

        返回的是 numpy 视图而非副本；调用方若需长期持有请自行 ``copy()``。
        """
        left = max(0, region.left)
        top = max(0, region.top)
        right = min(self.width, region.right)
        bottom = min(self.height, region.bottom)
        if right <= left or bottom <= top:
            return None
        return self.image[top:bottom, left:right]


class ScreenCapturer:
    """屏幕捕获器。

    典型用法（在捕获线程内）::

        capturer = ScreenCapturer()
        try:
            capturer.open()
            while running:
                frame = capturer.capture_monitor(index)
        finally:
            capturer.close()
    """

    def __init__(self, monitor_index: int | None = None) -> None:
        self.monitor_index = monitor_index
        self._local = threading.local()
        self._seq = 0
        #: 最近一次捕获耗时（秒），供帧率动态控制参考
        self.last_capture_cost = 0.0

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    @staticmethod
    def is_available() -> bool:
        """返回捕获依赖是否就绪。"""
        return _MSS_AVAILABLE

    def open(self) -> None:
        """在当前线程初始化 mss 句柄。重复调用无副作用。"""
        if not _MSS_AVAILABLE:
            raise RuntimeError(
                "未安装 python-mss，无法进行屏幕捕获。请执行 pip install mss"
            )
        if getattr(self._local, "sct", None) is None:
            # mss 10.x 起工厂从 mss() 改名为 MSS()，优先用新名以兼容未来版本
            factory = getattr(_mss, "MSS", None) or _mss.mss
            self._local.sct = factory()
            logger.debug("屏幕捕获句柄已初始化（线程 %s）", threading.current_thread().name)

    def close(self) -> None:
        """释放当前线程的 mss 句柄。"""
        sct = getattr(self._local, "sct", None)
        if sct is not None:
            try:
                sct.close()
            except Exception:  # noqa: BLE001 - 句柄可能已被系统回收
                logger.debug("关闭捕获句柄时出现异常（可忽略）")
            self._local.sct = None

    def __enter__(self) -> ScreenCapturer:
        self.open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # 显示器信息
    # ------------------------------------------------------------------

    @property
    def _sct(self):
        sct = getattr(self._local, "sct", None)
        if sct is None:
            self.open()
            sct = self._local.sct
        return sct

    def monitors(self) -> list[dict[str, int]]:
        """返回 mss 检测到的显示器列表（下标 0 为全部显示器的并集）。"""
        return list(self._sct.monitors)

    def monitor_count(self) -> int:
        """返回物理显示器数量（不含并集项）。"""
        return max(0, len(self._sct.monitors) - 1)

    # ------------------------------------------------------------------
    # 捕获
    # ------------------------------------------------------------------

    def capture_monitor(self, monitor_index: int | None = None) -> CaptureFrame:
        """捕获指定显示器。

        :param monitor_index: ``mss`` 的显示器序号（从 1 开始）；省略时使用
            实例默认值，仍为空则取主显示器（序号 1）。
        """
        index = monitor_index or self.monitor_index or 1
        monitors = self._sct.monitors
        if index >= len(monitors):
            logger.warning("显示器序号 %s 不存在，回退到主显示器", index)
            index = 1
        return self._grab(monitors[index], monitor_index=index)

    def capture_all(self) -> CaptureFrame:
        """捕获全部显示器合成的虚拟桌面（``mss`` 下标 0）。"""
        return self._grab(self._sct.monitors[0], monitor_index=None)

    def capture_region(
        self,
        region: Rect,
        monitor_index: int | None = None,
    ) -> CaptureFrame:
        """捕获指定物理区域。

        :param region: 物理坐标矩形，相对虚拟桌面左上角。
        :param monitor_index: 该区域所属显示器的 ``mss`` 序号。**务必传入**：
            它会被逐级带到渲染阶段，而渲染阶段按显示器分发条目——
            索引缺失时条目会被判定为"不属于任何被捕获的显示器"而被丢弃，
            表现就是"翻译在跑，翻译层却一片空白"。
            调用方（流水线）持有坐标映射，由它判定区域落在哪块屏幕上。
        """
        monitor = {
            "left": int(region.left),
            "top": int(region.top),
            "width": max(1, int(region.width)),
            "height": max(1, int(region.height)),
        }
        return self._grab(monitor, monitor_index=monitor_index)

    def _grab(self, monitor: dict[str, int], monitor_index: int | None) -> CaptureFrame:
        """执行一次拍摄并封装为 :class:`CaptureFrame`。"""
        started = time.perf_counter()
        shot = self._sct.grab(monitor)
        image = _to_bgr(shot)
        cost = time.perf_counter() - started
        self.last_capture_cost = cost

        self._seq += 1
        origin = Rect(
            int(monitor["left"]), int(monitor["top"]),
            int(monitor["width"]), int(monitor["height"]),
        )

        return CaptureFrame(
            seq=self._seq,
            timestamp=started,
            image=image,
            origin=origin,
            monitor_index=monitor_index,
            changed_regions=[],
            is_full_frame=True,
            capture_cost=cost,
        )


def _to_bgr(shot: Any) -> Any:
    """把 mss 的截图对象转换为 BGR 数组。

    必须先经过 ``numpy.asarray`` 再做色彩转换：mss 返回的是 ``ScreenShot``
    对象（带缓冲协议），OpenCV 4.x 容忍这种输入，但 **OpenCV 5.0 起不再接受
    非 ndarray 参数**，直接传会抛::

        cv2.error: (-5:Bad argument) in function 'cvtColor'
        Overload resolution failed: src is not a numpy array, neither a scalar

    这是实测踩到的坑：`pip install -r requirements.txt` 会装上
    ``opencv-python 5.x``，导致屏幕捕获在第一次抓帧时就失败，
    表现为"程序起来了但读不到屏幕"。

    :param shot: ``mss`` 的 ScreenShot 对象或已是 ndarray 的帧。
    :return: 内存连续的 BGR ndarray（OCR 后端要求连续）。
    """
    import numpy as np

    array = np.asarray(shot)

    # 灰度或已经是 3 通道的输入直接返回连续副本
    if array.ndim == 2:
        return np.ascontiguousarray(array)
    if array.shape[2] == 3:
        return np.ascontiguousarray(array)
    if array.shape[2] != 4:
        raise ValueError(f"不支持的图像通道数：{array.shape}")

    if _CV_AVAILABLE:
        # cvtColor 的输出本身就是新分配的连续数组
        return cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)

    return np.ascontiguousarray(array[:, :, :3])
