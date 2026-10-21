"""盈亏门控（联网预翻译 P3，自研）。

对应方案 v1.2 §3.9。**只作用于网页下一章预取**；字幕注册即预热，
不经过本门。

公式（需求第七节）：预取经济盈亏线 `a* = C / (ΔL·V + C)`——
只有站点历史预取准确率 `a`（预取后被真正阅读的比例，7 天窗）
达到 `a*` 才继续为该站点预取下一章。

**冷启动**：新站点没有历史 `a`，给每日试探配额（默认 20 次/站/天）；
配额耗尽即停止，等真实 `a` 形成。这样既不会无限浪费，也不会
因为"没有历史所以永远不预取"而死锁。
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from ...utils.logger import get_logger

logger = get_logger(__name__)


def threshold(C: float, delta_L_s: float, V: float) -> float:
    """盈亏线 a* = C / (ΔL·V + C)。"""
    denominator = delta_L_s * V + C
    if denominator <= 0:
        return 1.0
    return C / denominator


@dataclass
class SiteStats:
    """站点预取统计（7 天窗）。"""

    prefetched: int = 0
    used: int = 0

    @property
    def accuracy(self) -> float:
        return self.used / self.prefetched if self.prefetched else 0.0


class EconGate:
    """网页下一章盈亏门控。"""

    def __init__(
        self,
        stats_provider: Callable[[str], SiteStats],
        C: float = 1.0,
        delta_L_s: float = 2.0,
        V: float = 1.0,
        coldstart_per_site_day: int = 20,
        coldstart_window_s: float = 86400.0,
    ) -> None:
        self._stats_provider = stats_provider
        self._C = max(0.01, float(C))
        self._delta_L = max(0.0, float(delta_L_s))
        self._V = max(0.0, float(V))
        self._coldstart_quota = max(0, int(coldstart_per_site_day))
        self._coldstart_window = max(60.0, float(coldstart_window_s))
        self._coldstart_used: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def threshold(self) -> float:
        return threshold(self._C, self._delta_L, self._V)

    def allow_prefetch(self, site: str) -> bool:
        """是否允许为该站点预取下一章。"""
        a_star = self.threshold()
        stats = self._stats_provider(site)
        if stats.prefetched >= 10:
            # 历史足够：准确率达标才放行
            if stats.accuracy >= a_star:
                return True
            logger.debug(
                "盈亏门控拦截 %s：准确率 %.1f%% < 门槛 %.1f%%",
                site, stats.accuracy * 100, a_star * 100,
            )
            return False
        # 冷启动：给试探配额
        return self._consume_coldstart(site)

    def _consume_coldstart(self, site: str) -> bool:
        now = time.time()
        with self._lock:
            stamps = [ts for ts in self._coldstart_used.get(site, [])
                      if now - ts < self._coldstart_window]
            if len(stamps) >= self._coldstart_quota:
                return False
            stamps.append(now)
            self._coldstart_used[site] = stamps
            return True


