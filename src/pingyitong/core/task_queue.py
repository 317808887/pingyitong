"""流水线队列与背压策略。

对应开发文档二·（六）"高负载时自动丢弃过期任务帧，保证翻译结果始终
对应当前屏幕内容"。

设计要点
    流水线各阶段通过有界队列解耦，队列满时**丢弃最旧数据**而不是阻塞上游。
    屏幕翻译场景下，一帧翻译结果的时效性远超完整性：延迟 2 秒的译文
    即使准确也毫无价值，而且会覆盖掉更准确的新结果。因此"丢旧保新"
    是唯一正确的背压策略。

    :class:`LatestQueue` 进一步只保留最新一条，用于渲染队列——
    渲染层只需要画出当前状态，中间帧没有意义。
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from ..utils.logger import get_logger

T = TypeVar("T")

logger = get_logger(__name__)


@dataclass
class QueueStats:
    """队列运行统计，用于性能压测与状态展示。"""

    #: 累计入队
    enqueued: int = 0
    #: 累计出队
    dequeued: int = 0
    #: 因队列满被丢弃的数量
    dropped: int = 0
    #: 因数据过期（序号落后于最新）被丢弃的数量
    stale_dropped: int = 0

    @property
    def drop_rate(self) -> float:
        total = self.enqueued or 1
        return self.dropped / total

    def as_text(self) -> str:
        return f"入队 {self.enqueued} / 出队 {self.dequeued} / 丢弃 {self.dropped}"


class LatestQueue(Generic[T]):
    """有界队列：满时丢弃最旧元素，可选只保留最新一个元素。

    线程安全，可作为流水线阶段之间的唯一连接点。
    """

    def __init__(
        self,
        maxsize: int = 2,
        keep_latest_only: bool = False,
        name: str = "",
        coalesce: Callable[[T, T], T] | None = None,
    ) -> None:
        """
        :param maxsize: 容量上限，至少为 1。
        :param keep_latest_only: 为 ``True`` 时入队即清空旧数据，
            队列中永远只有最新一条（适用于渲染指令队列）。
        :param name: 队列名称，仅用于日志。
        """
        self.maxsize = max(1, int(maxsize))
        self.keep_latest_only = bool(keep_latest_only)
        self.name = name or "queue"
        self._coalesce = coalesce
        self._items: deque[tuple[int, T]] = deque()
        self._condition = threading.Condition()
        self._stats = QueueStats()

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def put(self, item: T, seq: int = 0) -> bool:
        """入队。

        :param item: 数据。
        :param seq: 帧序号，用于过期判定；传 0 表示不参与过期判断。
        :return: 数据被保留返回 ``True``；被立即丢弃返回 ``False``。
        """
        with self._condition:
            self._stats.enqueued += 1

            if self.keep_latest_only:
                if self._items:
                    # 差分帧包含互不相同的变化，直接丢旧会永久漏掉区域。
                    # 可选合并器把待处理变化并入最新快照，保留最新时间戳。
                    if self._coalesce is not None:
                        for _, previous in self._items:
                            item = self._coalesce(previous, item)
                    self._stats.dropped += len(self._items)
                    self._items.clear()
            elif len(self._items) >= self.maxsize:
                # 丢弃最旧的一条，为新数据腾位
                self._items.popleft()
                self._stats.dropped += 1

            self._items.append((seq, item))
            self._condition.notify()
            return True

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------

    def get(self, timeout: float | None = 0.1) -> T | None:
        """取出最早入队的数据，队列为空时等待。

        :param timeout: 等待秒数；``None`` 表示无限等待。
        :return: 数据；超时返回 ``None``。
        """
        with self._condition:
            if not self._items:
                self._condition.wait(timeout)
            if not self._items:
                return None

            _, item = self._items.popleft()
            self._stats.dequeued += 1
            return item

    def get_latest(self) -> T | None:
        """取出最新一条数据并清空队列其余内容。

        用于"只关心当前状态"的场景：拿到最新的同时丢掉所有过期中间态。
        """
        with self._condition:
            if not self._items:
                return None

            seq, item = self._items[-1]
            self._stats.dropped += max(0, len(self._items) - 1)
            self._items.clear()
            self._stats.dequeued += 1
            return item

    def get_after(self, min_seq: int) -> T | None:
        """取出序号不小于 ``min_seq`` 的最新数据，丢弃更早的过期数据。

        这是"保证结果对应当前屏幕内容"的具体实现：若队列中只有落后于
        ``min_seq`` 的数据，说明它们已经过期，直接丢弃而不做无谓的推理。
        """
        with self._condition:
            while self._items and self._items[0][0] < min_seq:
                self._items.popleft()
                self._stats.stale_dropped += 1

            if not self._items:
                return None

            _, item = self._items.popleft()
            self._stats.dequeued += 1
            return item

    def peek(self) -> T | None:
        """查看最早一条数据但不移除。"""
        with self._condition:
            return self._items[0][1] if self._items else None

    def peek_latest(self) -> T | None:
        """查看最新一条数据但不移除。"""
        with self._condition:
            return self._items[-1][1] if self._items else None

    def peek_seq(self) -> int:
        """返回最新一条数据的序号，队列为空时返回 0。"""
        with self._condition:
            return self._items[-1][0] if self._items else 0

    # ------------------------------------------------------------------
    # 维护
    # ------------------------------------------------------------------

    def clear(self) -> int:
        """清空队列，返回被清除的数量。"""
        with self._condition:
            count = len(self._items)
            self._items.clear()
            if count:
                self._stats.dropped += count
            return count

    def wake_all(self) -> None:
        """唤醒所有等待中的消费者，用于停止流水线时解除阻塞。"""
        with self._condition:
            self._condition.notify_all()

    def resize(self, maxsize: int) -> None:
        """调整容量；缩小时立即淘汰多余的最旧数据。"""
        with self._condition:
            self.maxsize = max(1, int(maxsize))
            while len(self._items) > self.maxsize:
                self._items.popleft()
                self._stats.dropped += 1

    def stats(self) -> QueueStats:
        """返回统计快照。"""
        with self._condition:
            return QueueStats(
                enqueued=self._stats.enqueued,
                dequeued=self._stats.dequeued,
                dropped=self._stats.dropped,
                stale_dropped=self._stats.stale_dropped,
            )

    def reset_stats(self) -> None:
        """清零统计。"""
        with self._condition:
            self._stats = QueueStats()

    def __len__(self) -> int:
        with self._condition:
            return len(self._items)

    @property
    def empty(self) -> bool:
        """队列是否为空。"""
        return len(self) == 0


class RateLimiter:
    """按目标帧率节流。

    捕获阶段使用：以目标帧率匀速产出，而不是"能跑多快跑多快"，
    否则会白白占满 CPU 并挤压 OCR 与推理线程。
    """

    def __init__(self, fps: float = 15.0) -> None:
        self._interval = 1.0 / max(0.1, float(fps))
        self._next_deadline = 0.0

    def configure(self, fps: float) -> None:
        """更新目标帧率。"""
        self._interval = 1.0 / max(0.1, float(fps))

    def reset(self) -> None:
        """重置计时基准，避免长时间暂停后出现补偿性突发。"""
        self._next_deadline = 0.0

    def wait(self, stop_event: threading.Event | None = None) -> bool:
        """阻塞到下一个时间片。

        :return: 正常等待到时间返回 ``True``；等待期间收到停止信号返回 ``False``。
        """
        now = time.perf_counter()
        if self._next_deadline <= 0:
            self._next_deadline = now + self._interval
            return True

        sleep_for = self._next_deadline - now
        if sleep_for > 0:
            if stop_event is not None:
                if stop_event.wait(sleep_for):
                    return False
            else:
                time.sleep(sleep_for)

        # 以固定步长推进而非重置为 now，避免累积漂移
        self._next_deadline += self._interval
        now = time.perf_counter()
        if self._next_deadline < now - self._interval:
            # 落后过多（例如系统休眠）时重新对齐
            self._next_deadline = now + self._interval
        return True
