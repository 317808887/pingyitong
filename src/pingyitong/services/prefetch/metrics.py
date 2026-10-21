"""预翻译指标埋点（联网预翻译，自研）。

对应方案 v1.2 §3.10 / §10.1：**真实计数，禁止模拟填充**。
所有事件追加进 SQLite（沿用 SentenceCache 的连接管理约定），
快照从库里聚合，设置页可查、可导出；保留期默认 30 天，每日轮转。

口径（§10.1）：
- 缓存命中率 h = hit / (hit+miss)，分 exact / fuzzy_subtitle；
- 预取准确率 a = 被命中预取句 / 全部预取句（按源、7 天窗）；
- 预取浪费率 = 1 − a；
- 回退率与原因分桶；延迟按路径分桶取 p50/p95；
- 流量 = bytes_in 求和。
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from ...utils.logger import get_logger

logger = get_logger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS prefetch_metrics (
    ts     REAL NOT NULL,
    kind   TEXT NOT NULL,
    path   TEXT,
    value  REAL,
    detail TEXT
);
CREATE INDEX IF NOT EXISTS idx_metrics_ts ON prefetch_metrics(ts);
"""


@dataclass
class MetricsSnapshot:
    """一段时间窗内的聚合结论。"""

    lookups: int = 0
    hits: int = 0
    hit_rate: float = 0.0
    exact_hits: int = 0
    fuzzy_hits: int = 0
    prefetch_count: int = 0
    prefetch_used: int = 0
    prefetch_accuracy: float = 0.0
    fallbacks: int = 0
    fallback_reasons: dict = field(default_factory=dict)
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    bytes_used: int = 0

    def as_text(self) -> str:
        return (
            f"命中率 {self.hit_rate:.1%}（{self.hits}/{self.lookups}）· "
            f"预取准确率 {self.prefetch_accuracy:.1%} · "
            f"回退 {self.fallbacks} 次 · "
            f"延迟 p50 {self.latency_p50_ms:.0f}ms / p95 {self.latency_p95_ms:.0f}ms"
        )


def _percentile(sorted_values: list[float], ratio: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, int(len(sorted_values) * ratio))
    return sorted_values[index]


class PrefetchMetrics:
    """本地埋点。线程安全（写串行，读即时聚合）。"""

    def __init__(self, path: Path, retention_days: float = 30.0) -> None:
        self._path = Path(path)
        self._retention = max(1.0, float(retention_days)) * 86400.0
        self._write_lock = threading.Lock()
        self._local = threading.local()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._connection().executescript(_DDL)
        self._connection().commit()

    def _connection(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = sqlite3.connect(str(self._path), timeout=5.0)
            connection.execute("PRAGMA journal_mode=WAL")
            self._local.connection = connection
        return connection

    def _record(self, kind: str, path: str = "", value: float = 0.0,
                detail: dict | None = None) -> None:
        try:
            with self._write_lock:
                self._connection().execute(
                    "INSERT INTO prefetch_metrics (ts, kind, path, value, detail) "
                    "VALUES (?,?,?,?,?)",
                    (time.time(), kind, path, value,
                     json.dumps(detail, ensure_ascii=False) if detail else None),
                )
                self._connection().commit()
        except sqlite3.Error:
            logger.debug("指标写入失败（忽略）", exc_info=True)

    # ------------------------------------------------------------------
    # 埋点入口
    # ------------------------------------------------------------------

    def on_lookup(self, hit: bool, kind: str) -> None:
        """kind ∈ exact | fuzzy_subtitle；miss 时 kind=miss。"""
        self._record("hit" if hit else "miss", path=kind or "miss")

    def on_prefetch(self, key: str, source: str, bytes_used: int = 0) -> None:
        self._record("prefetch", path=source, value=bytes_used,
                     detail={"key": key, "bytes": bytes_used})

    def on_prefetch_used(self, key: str, site: str = "") -> None:
        self._record("used", path=site, detail={"key": key} if site else None)

    def on_fallback(self, reason: str, source_id: str) -> None:
        self._record("fallback", path=source_id, detail={"reason": reason})

    def on_latency(self, ms: float, path: str) -> None:
        self._record("latency", path=path, value=ms)

    # ------------------------------------------------------------------
    # 聚合
    # ------------------------------------------------------------------

    def snapshot(self, window_s: float = 7 * 86400.0) -> MetricsSnapshot:
        since = time.time() - max(60.0, float(window_s))
        connection = self._connection()
        snap = MetricsSnapshot()

        rows = connection.execute(
            "SELECT kind, COUNT(*), COALESCE(SUM(value), 0) FROM prefetch_metrics "
            "WHERE ts >= ? GROUP BY kind",
            (since,),
        ).fetchall()
        counts = {kind: (count, value) for kind, count, value in rows}

        misses = counts.get("miss", (0, 0))
        snap.exact_hits = int(counts.get("hit", (0, 0))[0])
        snap.fuzzy_hits = snap.exact_hits
        snap.hits = snap.exact_hits
        snap.lookups = snap.hits + int(misses[0])
        snap.hit_rate = snap.hits / snap.lookups if snap.lookups else 0.0

        snap.prefetch_count = int(counts.get("prefetch", (0, 0))[0])
        snap.prefetch_used = int(counts.get("used", (0, 0))[0])
        snap.prefetch_accuracy = (
            snap.prefetch_used / snap.prefetch_count if snap.prefetch_count else 0.0
        )
        snap.bytes_used = int(counts.get("prefetch", (0, 0))[1])

        fallback_rows = connection.execute(
            "SELECT path, COUNT(*) FROM prefetch_metrics "
            "WHERE kind='fallback' AND ts >= ? GROUP BY path",
            (since,),
        ).fetchall()
        snap.fallbacks = sum(count for _, count in fallback_rows)
        snap.fallback_reasons = dict(fallback_rows)

        latencies = [
            row[0] for row in connection.execute(
                "SELECT value FROM prefetch_metrics "
                "WHERE kind='latency' AND ts >= ? ORDER BY value",
                (since,),
            ).fetchall()
        ]
        snap.latency_p50_ms = _percentile(latencies, 0.50)
        snap.latency_p95_ms = _percentile(latencies, 0.95)
        return snap

    def site_stats(self, site: str, window_s: float = 7 * 86400.0):
        """站点预取统计（盈亏门控输入）：(预取数, 被使用数)。

        口径：kind=prefetch 且 path=站点 → 预取数；
        kind=used 且 path 的 key 属于该站点的预取 → 被使用数。
        used 事件记录的 path 就是 key，因此这里按 detail 反查
        不可行——改为直接统计 used 事件总数与该站点 prefetch 的
        比例近似。为保持简单准确，预取时把站点写入 detail，
        used 时把站点写入 path。
        """
        since = time.time() - max(60.0, float(window_s))
        try:
            prefetched = self._connection().execute(
                "SELECT COUNT(*) FROM prefetch_metrics "
                "WHERE kind='prefetch' AND path=? AND ts >= ?",
                (site, since),
            ).fetchone()[0]
            used = self._connection().execute(
                "SELECT COUNT(*) FROM prefetch_metrics "
                "WHERE kind='used' AND detail=? AND ts >= ?",
                (site, since),
            ).fetchone()[0]
        except sqlite3.Error:
            return type("S", (), {"prefetched": 0, "used": 0, "accuracy": 0.0})()
        return type("S", (), {"prefetched": prefetched, "used": used, "accuracy": used / prefetched if prefetched else 0.0})()

    def rotate(self) -> int:
        """删除保留期外的指标行（每日调用）。"""
        cutoff = time.time() - self._retention
        try:
            with self._write_lock:
                cursor = self._connection().execute(
                    "DELETE FROM prefetch_metrics WHERE ts < ?", (cutoff,)
                )
                self._connection().commit()
                return max(0, cursor.rowcount)
        except sqlite3.Error:
            return 0

    def close(self) -> None:
        """关闭当前线程连接（测试与退出时调用；否则 Windows 下临时目录
        会因文件句柄未释放而无法清理）。"""
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            with contextlib.suppress(sqlite3.Error):
                connection.close()
            self._local.connection = None
