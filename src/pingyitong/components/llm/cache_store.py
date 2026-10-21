"""推理缓存持久化存储（SQLite）。

:class:`~pingyitong.components.llm.translation_cache.TranslationCache`
本身是进程内 LRU：重启后为空，而桌面界面文案（菜单、按钮、状态栏）
跨会话高度重复，重启后同样的文字要重新推理一遍。本模块把缓存条目
落盘到应用数据目录，让跨会话重复文本直接命中：

- 键即内存缓存的键（namespace|方向|术语表版本|sha256），语义不变；
- 翻译帧线程只入队，磁盘写入由后台线程批量完成，帧延迟零增加；
- 启动时按写入时间倒序加载至多 ``limit`` 条，时间戳重置为本进程
  时钟（TTL 重新起算，磁盘条目跨会话仍然有效）；
- 数据库损坏或不可写时静默降级为纯内存缓存，不影响翻译功能。
"""

from __future__ import annotations

import contextlib
import queue
import sqlite3
import threading
from pathlib import Path

from ...utils.logger import get_logger

logger = get_logger(__name__)

#: 磁盘保留的最大条目数（启动时裁剪最旧的；远大于内存 LRU 容量）
MAX_DISK_ENTRIES = 20000

_DDL = """
CREATE TABLE IF NOT EXISTS inference_cache (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    created_at REAL NOT NULL
)
"""


class PersistentCacheStore:
    """推理缓存的 SQLite 后备存储；磁盘写入全部在后台线程完成。"""

    def __init__(
        self,
        path: str | Path,
        *,
        max_disk_entries: int = MAX_DISK_ENTRIES,
    ) -> None:
        self._path = Path(path)
        self._max_disk_entries = max(1, int(max_disk_entries))
        self._db_lock = threading.Lock()
        self._queue_lock = threading.Lock()
        self._generation = 0
        self._key_generations: dict[str, int] = {}
        self._queue: queue.SimpleQueue[tuple[int, int, str, str, float] | None] = queue.SimpleQueue()
        self._enabled = True
        self._conn: sqlite3.Connection | None = None
        self._writer: threading.Thread | None = None
        connection: sqlite3.Connection | None = None
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # check_same_thread=False：连接由 _db_lock 串行化，writer 线程也要使用
            connection = sqlite3.connect(
                str(self._path), timeout=5.0, check_same_thread=False
            )
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.executescript(_DDL)
            connection.commit()
            self._conn = connection
        except (sqlite3.Error, OSError):
            # 半打开的连接必须显式关掉，否则 Windows 上文件句柄会拖住目录清理
            if connection is not None:
                with contextlib.suppress(Exception):
                    connection.close()
            logger.warning("推理缓存持久化不可用（%s），降级为纯内存缓存", self._path)
            self._enabled = False
            return
        self._writer = threading.Thread(
            target=self._write_loop, name="CacheStoreWriter", daemon=True
        )
        self._writer.start()
        self._prune()

    # ------------------------------------------------------------------
    # 对外接口（除 close/load 外均不阻塞调用线程）
    # ------------------------------------------------------------------

    def load(self, limit: int) -> list[tuple[str, str, float]]:
        """按写入时间从新到旧返回至多 ``limit`` 条 ``(key, value, created_at)``。"""
        if not self._enabled or self._conn is None:
            return []
        try:
            with self._db_lock:
                rows = self._conn.execute(
                    "SELECT key, value, created_at FROM inference_cache "
                    "ORDER BY created_at DESC LIMIT ?",
                    (max(0, int(limit)),),
                ).fetchall()
        except sqlite3.Error:
            return []
        return [(str(k), str(v), float(c)) for k, v, c in rows]

    def put(self, key: str, value: str, created_at: float) -> None:
        """入队一条写入；由后台线程落盘。"""
        with self._queue_lock:
            if not self._enabled:
                return
            self._queue.put((
                self._generation, self._key_generations.get(key, 0),
                key, value, float(created_at),
            ))

    def clear(self) -> None:
        """清空磁盘条目（含尚未落盘的排队写入）。"""
        if not self._enabled or self._conn is None:
            return
        # The writer may already hold a batch outside the queue. Invalidating
        # its generation prevents that batch from restoring deleted entries.
        # Never drain the queue: doing so can consume close()'s stop sentinel.
        with self._queue_lock:
            self._generation += 1
            self._key_generations.clear()
        try:
            with self._db_lock:
                self._conn.execute("DELETE FROM inference_cache")
                self._conn.commit()
        except sqlite3.Error:
            pass

    def discard(self, key: str) -> None:
        """Delete one invalid translation, including its queued older writes."""
        if not self._enabled or self._conn is None:
            return
        with self._queue_lock:
            self._key_generations[key] = self._key_generations.get(key, 0) + 1
        try:
            with self._db_lock:
                self._conn.execute("DELETE FROM inference_cache WHERE key = ?", (key,))
                self._conn.commit()
        except sqlite3.Error:
            pass

    def close(self) -> None:
        """刷尽排队写入并停止后台线程；幂等。"""
        if self._writer is None:
            return
        with self._queue_lock:
            self._enabled = False
            self._queue.put(None)
        self._writer.join(timeout=5.0)
        self._writer = None
        if self._conn is not None:
            try:
                with self._db_lock:
                    self._conn.close()
            except sqlite3.Error:
                pass
            self._conn = None
        self._enabled = False

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _write_loop(self) -> None:
        """批量消费写入队列；SQLite 异常时放弃后续写入（内存缓存不受影响）。"""
        while True:
            item = self._queue.get()
            if item is None:
                return
            batch = [item]
            while True:
                try:
                    nxt = self._queue.get_nowait()
                except queue.Empty:
                    break
                if nxt is None:
                    self._flush(batch)
                    return
                batch.append(nxt)
            if not self._flush(batch):
                return

    def _flush(self, batch: list[tuple[int, int, str, str, float]]) -> bool:
        if self._conn is None:
            return False
        try:
            with self._db_lock:
                with self._queue_lock:
                    rows = [
                        (key, value, created_at)
                        for generation, key_generation, key, value, created_at in batch
                        if generation == self._generation
                        and key_generation == self._key_generations.get(key, 0)
                    ]
                self._conn.executemany(
                    "INSERT OR REPLACE INTO inference_cache(key, value, created_at) "
                    "VALUES (?, ?, ?)",
                    rows,
                )
                self._conn.commit()
        except sqlite3.Error:
            logger.warning("推理缓存写入磁盘失败，本会话放弃持久化", exc_info=True)
            self._enabled = False
            return False
        return True

    def _prune(self) -> None:
        """启动时只保留最新的 ``max_disk_entries`` 条。"""
        if self._conn is None:
            return
        try:
            with self._db_lock:
                self._conn.execute(
                    "DELETE FROM inference_cache WHERE key NOT IN ("
                    "SELECT key FROM inference_cache ORDER BY created_at DESC LIMIT ?"
                    ")",
                    (self._max_disk_entries,),
                )
                self._conn.commit()
        except sqlite3.Error:
            pass
