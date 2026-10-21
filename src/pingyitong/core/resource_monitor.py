"""系统资源监控。

开源组件：``psutil``（giampaolo/psutil，BSD 3-Clause License）
用途：采集 CPU、内存、GPU 占用数据，为双模型调度提供依据。

对应开发文档二·（六）："监控系统资源占用，动态调整运行策略"。

GPU 显存采集为可选增强：优先使用 ``pynvml``（NVIDIA），
不可用时返回 ``None``，调度器会退化为仅依据 CPU 与帧率决策。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from ..utils.logger import get_logger

logger = get_logger(__name__)

try:  # pragma: no cover
    import psutil

    _PSUTIL_AVAILABLE = True
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]
    _PSUTIL_AVAILABLE = False

_pynvml = None
_NVML_READY: bool | None = None


@dataclass
class ResourceSnapshot:
    """一次资源采样结果。"""

    #: 采样时刻
    timestamp: float = 0.0
    #: 全系统 CPU 占用率（0-100）
    cpu_percent: float = 0.0
    #: 本进程 CPU 占用率（0-100，多核可超过 100）
    process_cpu_percent: float = 0.0
    #: 全系统内存占用率（0-100）
    memory_percent: float = 0.0
    #: 本进程内存占用（MB）
    process_memory_mb: float = 0.0
    #: GPU 显存空闲（MB）；无法获取时为 ``None``
    gpu_free_memory_mb: float | None = None
    #: GPU 利用率（0-100）；无法获取时为 ``None``
    gpu_utilization: float | None = None
    #: 渲染帧率（由渲染层回填）
    render_fps: float = 0.0
    #: 捕获帧率（由捕获层回填）
    capture_fps: float = 0.0
    #: 端到端延迟（秒）：从捕获到译文绘制完成。由流水线回填。
    #:
    #: 这是判断"用户是否在等"的**正确指标**。早期用渲染帧率做降级依据，
    #: 但静态屏幕上没有新内容可渲染，帧率天然趋近 0，于是被误判为"界面卡顿"
    #: 而反复降级模型——与内容自适应升级形成死循环，每次切换还要重载模型。
    end_to_end_latency: float = 0.0
    #: 逻辑 CPU 核心数
    cpu_count: int = 0

    @property
    def gpu_available(self) -> bool:
        """是否成功获取到 GPU 信息。"""
        return self.gpu_free_memory_mb is not None

    def as_text(self) -> str:
        """生成简洁的中文状态描述。"""
        parts = [f"CPU {self.cpu_percent:.0f}%", f"内存 {self.memory_percent:.0f}%"]
        if self.gpu_utilization is not None:
            parts.append(f"GPU {self.gpu_utilization:.0f}%")
        if self.render_fps:
            parts.append(f"渲染 {self.render_fps:.0f}fps")
        return " · ".join(parts)


class ResourceMonitor:
    """资源采样器。

    采样需要两次 ``cpu_percent`` 调用之间有时间间隔，因此
    :meth:`sample` 依赖于上一次调用；首次调用会返回 0，
    这符合 psutil 的语义而不是缺陷。
    """

    def __init__(self, interval: float = 1.0) -> None:
        self.interval = max(0.2, float(interval))
        self._last_sample = 0.0
        self._last_snapshot = ResourceSnapshot()
        self._cpu_history: list[tuple[float, float]] = []
        self._process = psutil.Process(os.getpid()) if _PSUTIL_AVAILABLE else None
        self._gpu_checked = False

    # ------------------------------------------------------------------
    # 采样
    # ------------------------------------------------------------------

    def sample(self, force: bool = False) -> ResourceSnapshot:
        """采集一次资源数据。

        :param force: 为 ``True`` 时忽略采样间隔立即采样。
        :return: :class:`ResourceSnapshot`。
        """
        now = time.perf_counter()
        if not force and (now - self._last_sample) < self.interval:
            return self._last_snapshot

        elapsed = now - self._last_sample if self._last_sample else 0.0
        self._last_sample = now

        if not _PSUTIL_AVAILABLE:
            return ResourceSnapshot(timestamp=now, cpu_count=os.cpu_count() or 0)

        cpu_percent = float(psutil.cpu_percent(interval=None))
        memory = psutil.virtual_memory()

        process_cpu = 0.0
        process_memory_mb = 0.0
        if self._process is not None:
            try:
                process_cpu = float(self._process.cpu_percent(interval=None))
                process_memory_mb = float(self._process.memory_info().rss) / (1024 * 1024)
            except Exception:  # noqa: BLE001 - 进程可能已退出或权限不足
                process_cpu = 0.0

        gpu_free, gpu_util = self._sample_gpu()

        snapshot = ResourceSnapshot(
            timestamp=now,
            cpu_percent=cpu_percent,
            process_cpu_percent=process_cpu,
            memory_percent=float(memory.percent),
            process_memory_mb=process_memory_mb,
            gpu_free_memory_mb=gpu_free,
            gpu_utilization=gpu_util,
            cpu_count=os.cpu_count() or 0,
        )
        # 保留上一轮的帧率回填值，避免被采样覆盖为 0
        snapshot.render_fps = self._last_snapshot.render_fps
        snapshot.capture_fps = self._last_snapshot.capture_fps

        self._last_snapshot = snapshot
        self._record_cpu(now, cpu_percent, elapsed)
        return snapshot

    def _sample_gpu(self) -> tuple[float | None, float | None]:
        """采集 GPU 显存与利用率。不可用时返回 ``(None, None)``。"""
        global _pynvml, _NVML_READY

        if _NVML_READY is None:
            try:
                import pynvml

                pynvml.nvmlInit()
                _pynvml = pynvml
                _NVML_READY = True
                logger.info("已启用 NVIDIA GPU 监控（pynvml）")
            except Exception:  # noqa: BLE001 - 无 NVIDIA 驱动或无 pynvml 均走此分支
                _pynvml = None
                _NVML_READY = False
                logger.debug("未检测到可用的 NVIDIA GPU 监控接口，调度将仅依据 CPU 与帧率")

        if not _NVML_READY or _pynvml is None:
            return None, None

        try:
            handle = _pynvml.nvmlDeviceGetHandleByIndex(0)
            memory = _pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = _pynvml.nvmlDeviceGetUtilizationRates(handle)
            return (
                float(memory.free) / (1024 * 1024),
                float(utilization.gpu),
            )
        except Exception:  # noqa: BLE001
            return None, None

    # ------------------------------------------------------------------
    # 帧率回填
    # ------------------------------------------------------------------

    def report_render_fps(self, fps: float) -> None:
        """由渲染层回填实际渲染帧率。"""
        self._last_snapshot.render_fps = max(0.0, float(fps))

    def report_capture_fps(self, fps: float) -> None:
        """由捕获层回填实际捕获帧率。"""
        self._last_snapshot.capture_fps = max(0.0, float(fps))

    def report_latency(self, seconds: float) -> None:
        """由渲染层回填端到端延迟。

        调度器用它判断"用户是否在等"——比渲染帧率可靠得多。
        """
        self._last_snapshot.end_to_end_latency = max(0.0, float(seconds))

    # ------------------------------------------------------------------
    # CPU 趋势
    # ------------------------------------------------------------------

    def _record_cpu(self, now: float, percent: float, elapsed: float) -> None:
        """记录 CPU 采样点，用于判断"持续高负载"。"""
        self._cpu_history.append((now, percent))
        # 只保留最近 60 秒的数据
        cutoff = now - 60.0
        self._cpu_history = [item for item in self._cpu_history if item[0] >= cutoff]

    def cpu_above_threshold_for(self, threshold: float, duration: float) -> bool:
        """判断 CPU 占用是否已持续超过阈值 ``duration`` 秒。

        单点瞬时值容易误判（编译器、浏览器渲染等短时占满 CPU），
        因此调度器使用"持续高负载"而不是瞬时高负载作为降级依据。

        :param threshold: 阈值（百分比）。
        :param duration: 需要的持续时长（秒）。
        """
        if not self._cpu_history:
            return False

        now = self._cpu_history[-1][0]
        window_start = now - duration
        window = [percent for timestamp, percent in self._cpu_history if timestamp >= window_start]

        if not window:
            return False
        # 需要有足够长的观察窗口，否则启动初期会误判
        if (now - self._cpu_history[0][0]) < duration * 0.8:
            return False

        return all(percent >= threshold for percent in window)

    def reset(self) -> None:
        """清空历史采样。"""
        self._cpu_history.clear()
        self._last_sample = 0.0
        self._last_snapshot = ResourceSnapshot()

    # ------------------------------------------------------------------
    # 静态信息
    # ------------------------------------------------------------------

    def cpu_count(self) -> int:
        """返回逻辑 CPU 核心数。"""
        return os.cpu_count() or 1

    def recommended_threads(self) -> int:
        """推荐推理线程数。

        取物理核心数级别而非全部逻辑核心：超线程核心在推理任务上
        收益有限，占满全部逻辑核反而会挤压 OCR 与界面线程。
        """
        count = self.cpu_count()
        if count <= 2:
            return count
        if count <= 8:
            return max(2, count - 2)
        return max(4, int(count * 0.7))

    def memory_total_mb(self) -> float:
        """返回系统总内存（MB）。"""
        if not _PSUTIL_AVAILABLE:
            return 0.0
        return float(psutil.virtual_memory().total) / (1024 * 1024)

    def is_available(self) -> bool:
        """psutil 是否可用。"""
        return _PSUTIL_AVAILABLE

    @staticmethod
    def recommend_model_for_memory() -> str:
        """依据系统内存给出模型推荐。

        8GB 内存的机器同时常驻 1.7B 与 4B（Q4 量化合计约 3.5GB）
        会挤压 OCR 与系统的可用内存，因此推荐仅使用 1.7B。
        """
        from ..constants import MODEL_ACCURATE, MODEL_FAST

        if not _PSUTIL_AVAILABLE:
            return MODEL_FAST
        total_gb = float(psutil.virtual_memory().total) / (1024**3)
        return MODEL_ACCURATE if total_gb >= 15 else MODEL_FAST
