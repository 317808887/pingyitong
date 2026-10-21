"""双层状态机（联网预翻译，自研）。

对应方案 v1.1 §4。v1.0 的单一 FALLBACK 全局态会让一个坏网页源
误伤字幕预热，因此 v1.1 拆成两层：

- **全局网关态**（:class:`GateState`）：只回答"当前是否允许产生
  网络请求"。DISABLED / OFFLINE / ONLINE / QUOTA_PAUSED。
- **每源健康态**（:class:`SourceHealth`）：站点或字幕轨各自维护
  OK / COOLDOWN，一个源连续失败只冷却该源，不波及其他源。
  字幕轨是本地文件（无网络请求），完全不经过网络冷却。

设计约束：状态机**不依赖 Qt**（UI 层通过 :meth:`snapshot` 轮询/包装
成信号即可），便于纯逻辑单测与后台线程调用。
每次迁移写 INFO 日志，日志文案同时是界面徽标文案的唯一来源。
"""

from __future__ import annotations

import enum
import threading
import time
from dataclasses import dataclass

from ...utils.logger import get_logger

logger = get_logger(__name__)


class GateState(enum.Enum):
    """全局网关态。``value`` 即界面徽标文案。"""

    DISABLED = "已关闭"
    OFFLINE = "离线·屏幕翻译中"
    ONLINE = "预翻译中"
    QUOTA_PAUSED = "配额已用尽"


class SourceHealth(enum.Enum):
    """每源健康态。"""

    OK = "ok"
    COOLDOWN = "cooldown"


@dataclass
class _SourceRecord:
    health: SourceHealth = SourceHealth.OK
    failures: int = 0
    cooldown_until: float = 0.0
    last_reason: str = ""


@dataclass(frozen=True)
class GateSnapshot:
    """对外一致快照（UI 徽标与调度器共用）。"""

    gate: GateState
    allowed: bool          # 是否允许产生网络请求
    badge: str             # 界面徽标文案（与日志同源）


class PrefetchGate:
    """双层状态机。线程安全。"""

    def __init__(
        self,
        enabled: bool = False,
        online: bool = False,
        quota_exceeded: bool = False,
        health_failure_n: int = 3,
        cooldown_s: float = 120.0,
    ) -> None:
        self._lock = threading.Lock()
        self._enabled = enabled
        self._online = online
        self._quota_exceeded = quota_exceeded
        self._failure_n = max(1, int(health_failure_n))
        self._cooldown_s = max(1.0, float(cooldown_s))
        self._sources: dict[str, _SourceRecord] = {}
        self._gate = self._compute_gate()

    # ------------------------------------------------------------------
    # 派生
    # ------------------------------------------------------------------

    def _compute_gate(self) -> GateState:
        if not self._enabled:
            return GateState.DISABLED
        if self._quota_exceeded:
            return GateState.QUOTA_PAUSED
        if not self._online:
            return GateState.OFFLINE
        return GateState.ONLINE

    def _transition_to(self, new_gate: GateState) -> None:
        if new_gate != self._gate:
            logger.info("预翻译网关：%s → %s", self._gate.value, new_gate.value)
        self._gate = new_gate

    # ------------------------------------------------------------------
    # 全局输入
    # ------------------------------------------------------------------

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._enabled = bool(enabled)
            self._transition_to(self._compute_gate())

    def set_online(self, online: bool) -> None:
        with self._lock:
            self._online = bool(online)
            self._transition_to(self._compute_gate())

    def set_quota_exceeded(self, exceeded: bool) -> None:
        with self._lock:
            self._quota_exceeded = bool(exceeded)
            self._transition_to(self._compute_gate())

    # ------------------------------------------------------------------
    # 每源健康
    # ------------------------------------------------------------------

    def notify_fetch_failed(self, source_id: str, reason: str) -> None:
        """某源抓取失败：计数，达到阈值进 COOLDOWN（只影响该源）。"""
        now = time.time()
        with self._lock:
            record = self._sources.setdefault(source_id, _SourceRecord())
            record.failures += 1
            record.last_reason = reason
            if record.failures >= self._failure_n:
                if record.health != SourceHealth.COOLDOWN:
                    logger.info(
                        "预翻译源健康：%s OK → COOLDOWN（连续 %s 次：%s）",
                        source_id, record.failures, reason,
                    )
                record.health = SourceHealth.COOLDOWN
                record.cooldown_until = now + self._cooldown_s

    def notify_fetch_ok(self, source_id: str) -> None:
        """抓取成功：该源恢复 OK（COOLDOWN 期的自动重试成功也走这里）。"""
        with self._lock:
            record = self._sources.get(source_id)
            if record is None:
                return
            if record.health != SourceHealth.OK:
                logger.info("预翻译源健康：%s COOLDOWN → OK（探测恢复）", source_id)
            record.health = SourceHealth.OK
            record.failures = 0
            record.cooldown_until = 0.0

    def source_allowed(self, source_id: str) -> bool:
        """该源当前是否可产生网络任务。

        全局网关为 ONLINE 且源不在冷却中。冷却期满时放行**一次**轻量
        探测重试（再次失败会重新冷却）——FALLBACK 自动恢复的实现点。
        """
        with self._lock:
            record = self._sources.get(source_id)
            if record is not None and record.health == SourceHealth.COOLDOWN:
                if time.time() >= record.cooldown_until:
                    record.cooldown_until = time.time() + self._cooldown_s
                    logger.info("预翻译源健康：%s 冷却期满，允许自动重试", source_id)
                    return True
                return False
            if self._gate != GateState.ONLINE:
                return False
        return True

    # ------------------------------------------------------------------
    # 快照
    # ------------------------------------------------------------------

    def snapshot(self) -> GateSnapshot:
        with self._lock:
            gate = self._gate
        return GateSnapshot(
            gate=gate,
            allowed=gate == GateState.ONLINE,
            badge=gate.value,
        )
