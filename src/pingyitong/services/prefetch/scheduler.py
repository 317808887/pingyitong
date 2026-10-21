"""预翻译调度器（联网预翻译 P2，自研）。

对应方案 v1.2 §3.6。核心约束（R1/R2 的闭环）：

- **优先级**：播放位置当前行 > 前向推进 > 下一章 > 全量余量；
- **空闲出队**：全部出队条件满足才工作（网关允许、源健康、CPU 空闲、
  翻译队列浅、网页过盈亏门）；
- **翻译 worker 固定 1**，与实时翻译共用同一模型实例；实时任务到达后
  在句边界让出（本调度器只负责"何时出队下一句"，让出由翻译队列
  深度条件自然实现——队列深了就不再出队）；
- **取消**：关开关、注销轨、离开页面 → 对应任务取消。

本模块只做调度决策与任务簿记，不直接调用翻译：取句动作通过
``pull`` 回调（由装配层提供：从缓存查、未命中则调模型并落库），
保持纯逻辑可测。
"""

from __future__ import annotations

import enum
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from ...utils.logger import get_logger

logger = get_logger(__name__)


class Priority(enum.IntEnum):
    PLAYBACK_CURRENT = 0   # 活跃轨当前行 ± 半径
    PLAYBACK_AHEAD = 1     # 随播放位置前向推进
    NEXT_CHAPTER = 2       # 网页下一章（盈亏门控）
    BULK_REST = 3          # 余量，仅深度空闲


@dataclass(order=True)
class PrefetchJob:
    """一个待预翻译单元（一句字幕或一个网页）。"""

    priority: int
    seq: int                                   # 同优先级保持 FIFO
    kind: str = field(compare=False)           # subtitle_line | web_page
    track_or_site: str = field(compare=False)
    payload: object = field(compare=False)     # 行号 / URL
    page_key: str = field(default="", compare=False)   # 取消粒度：页/轨
    generation: int = field(default=0, compare=False)  # 同页重导时排除旧在途任务


class PrefetchScheduler:
    """优先级任务队列 + 出队条件门控。线程安全。"""

    def __init__(
        self,
        idle_check: Callable[[], bool],
        gate_allow: Callable[[str], bool],
        pull: Callable[[PrefetchJob], bool],
        *,
        model_queue_depth_max: int = 0,
        queue_depth_probe: Callable[[], int] | None = None,
    ) -> None:
        self._idle_check = idle_check
        self._gate_allow = gate_allow
        self._pull = pull
        self._model_queue_depth_max = model_queue_depth_max
        self._queue_depth_probe = queue_depth_probe
        self._lock = threading.Lock()
        self._jobs: list[PrefetchJob] = []
        self._seq = 0
        self._cancelled_pages: set[str] = set()
        self._generations: dict[str, int] = {}
        self._totals: dict[str, int] = {}   # page_key → 入队总数（进度分母）
        self._completed: dict[str, int] = {}
        self._failed: dict[str, int] = {}
        self._stop = False
        self._worker: threading.Thread | None = None
        self._wakeup = threading.Event()
        self.progress: Callable[[str, int, int], None] | None = None

    # ------------------------------------------------------------------
    # 任务入队 / 取消
    # ------------------------------------------------------------------

    def enqueue(self, job: PrefetchJob) -> None:
        """入队一个任务；已取消的页/轨直接丢弃。"""
        with self._lock:
            if job.page_key in self._cancelled_pages:
                return
            self._seq += 1
            job.seq = self._seq
            job.generation = self._generations.get(job.page_key, 0)
            self._jobs.append(job)
            self._jobs.sort()
            self._totals[job.page_key] = self._totals.get(job.page_key, 0) + 1
        self._wakeup.set()

    def cancel_page(self, page_key: str) -> None:
        """取消某页/某轨的全部待办（含未来同名入队）。"""
        with self._lock:
            self._cancelled_pages.add(page_key)
            self._generations[page_key] = self._generations.get(page_key, 0) + 1
            before = len(self._jobs)
            self._jobs = [
                job for job in self._jobs if job.page_key != page_key
            ]
            removed = before - len(self._jobs)
        if removed:
            logger.info("预翻译任务取消：%s（%s 个待办）", page_key, removed)

    def clear_cancelled(self, page_key: str) -> None:
        with self._lock:
            if page_key in self._cancelled_pages:
                self._totals.pop(page_key, None)
                self._completed.pop(page_key, None)
                self._failed.pop(page_key, None)
            self._cancelled_pages.discard(page_key)

    def pending_count(self) -> int:
        with self._lock:
            return len(self._jobs)

    def page_progress(self, page_key: str) -> tuple[int, int]:
        """返回 (已成功完成, 入队总数)；失败与取消不能冒充译文已就绪。"""
        with self._lock:
            total = self._totals.get(page_key, 0)
            done = self._completed.get(page_key, 0)
        return done, total

    def page_failures(self, page_key: str) -> int:
        """该页/字幕轨未产生译文的任务数，供界面报告部分失败。"""
        with self._lock:
            return self._failed.get(page_key, 0)

    # ------------------------------------------------------------------
    # worker
    # ------------------------------------------------------------------

    def start(self) -> None:
        """启动单个后台 worker（低优先级语义由出队条件保证）。"""
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop = False

        def _loop() -> None:
            while not self._stop:
                self._wakeup.wait(timeout=1.0)
                self._wakeup.clear()
                worked = self._drain_one()
                if not worked:
                    time.sleep(0.5)

        self._worker = threading.Thread(target=_loop, name="prefetch-worker", daemon=True)
        self._worker.start()

    def stop(self) -> None:
        """停止 worker；在途任务作废（句边界让出由模型调用层保证）。"""
        self._stop = True
        self._wakeup.set()

    def _drain_one(self) -> bool:
        """尝试出队并执行一个任务。返回是否有动作。

        队头任务被门控挡住时**跳过它继续找**，而不是放弃整轮——
        否则一个冷却中的源会饿死后面健康的源（每源冷却语义要求
        "只影响该源"）。
        """
        if not self._idle_check():
            return False

        with self._lock:
            job = next(
                (candidate for candidate in self._jobs
                 if candidate.page_key not in self._cancelled_pages
                 and self._gate_allow(candidate.track_or_site)),
                None,
            )
        if job is None:
            return False

        if (
            self._queue_depth_probe is not None
            and self._model_queue_depth_max >= 0
            and self._queue_depth_probe() > self._model_queue_depth_max
        ):
            return False  # 实时任务排队中，后台让出

        with self._lock:
            try:
                self._jobs.remove(job)
            except ValueError:
                return False  # 已被取消

        ok = False
        try:
            ok = bool(self._pull(job))
        except Exception:  # noqa: BLE001 - 单任务失败不终止 worker
            logger.debug("预翻译任务失败：%s", job.page_key, exc_info=True)
        with self._lock:
            if job.generation != self._generations.get(job.page_key, 0):
                return ok  # 被取消/重导的旧在途请求不能计入新一轮进度
            counter = self._completed if ok else self._failed
            counter[job.page_key] = counter.get(job.page_key, 0) + 1
        if self.progress is not None:
            done, total = self.page_progress(job.page_key)
            try:
                self.progress(job.page_key, done, total)
            except Exception:  # noqa: BLE001 - 通知失败不得终止后台 worker
                logger.debug("预翻译进度回调失败：%s", job.page_key, exc_info=True)
        return ok
