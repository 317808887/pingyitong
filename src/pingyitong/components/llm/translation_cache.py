"""推理缓存（自研）。

对应《项目原创性声明》二·（二）中"推理缓存机制"部分。

桌面界面的文字具有高度重复性：菜单项、按钮、状态栏文案在滚动与
窗口切换时会反复出现。缓存命中可直接跳过推理，是实时性的关键。

缓存键 = 文本内容 + 源语言 + 目标语言 + 术语表版本，与坐标无关，
因此同一句话出现在屏幕不同位置时同样命中。

可选传入 :class:`~pingyitong.components.llm.cache_store.PersistentCacheStore`
把条目落盘（SQLite，后台线程写入）：重启后加载既有条目（TTL 重新
起算），跨会话重复的界面文案无需再次推理。
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ...utils.logger import get_logger

if TYPE_CHECKING:
    from .cache_store import PersistentCacheStore

logger = get_logger(__name__)

@dataclass
class CacheStats:
    """缓存统计，用于状态面板展示。"""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expired: int = 0
    verified_hits: int = 0
    verified_requests: int = 0
    rejected_candidates: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0

    def as_text(self) -> str:
        """生成简洁的中文描述。"""
        if self.verified_requests:
            rate = self.verified_hits / self.verified_requests
            return (
                f"可信复用 {rate * 100:.0f}%（{self.verified_hits}/{self.verified_requests}）"
                f" · 无效缓存 {self.rejected_candidates}"
            )
        if not self.total:
            return "暂无请求"
        return f"命中率 {self.hit_rate * 100:.0f}%（{self.hits}/{self.total}）"


class TranslationCache:
    """带容量上限与过期时间的线程安全翻译缓存。

    使用 ``OrderedDict`` 实现 LRU：命中后把条目移到末尾，
    淘汰时从头部（最久未使用）移除。
    """

    def __init__(
        self,
        max_entries: int = 4096,
        ttl_seconds: float = 1800.0,
        enabled: bool = True,
        store: PersistentCacheStore | None = None,
    ) -> None:
        self.max_entries = max(0, int(max_entries))
        self.ttl_seconds = max(0.0, float(ttl_seconds))
        self.enabled = bool(enabled) and self.max_entries > 0
        self._store: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self._lock = threading.Lock()
        self._stats = CacheStats()
        self._disk = store if self.enabled else None
        if self._disk is not None:
            self._load_from_disk()

    def _load_from_disk(self) -> None:
        """把持久层条目装入内存 LRU；时间戳重置为本进程时钟（TTL 重新起算）。"""
        try:
            rows = self._disk.load(self.max_entries)
        except Exception:  # pragma: no cover - 持久层自身已兜底，防御性兜底
            logger.warning("读取推理缓存持久层失败，按空缓存启动", exc_info=True)
            rows = []
        now = time.perf_counter()
        # load 返回从新到旧；按最旧→最新插入，使最新条目位于 LRU 尾部
        for key, value, _created_at in reversed(rows):
            self._store[key] = (value, now)

    # ------------------------------------------------------------------
    # 配置
    # ------------------------------------------------------------------

    def configure(
        self,
        max_entries: int | None = None,
        ttl_seconds: float | None = None,
        enabled: bool | None = None,
    ) -> None:
        """运行时更新缓存参数。容量调小时立即执行淘汰。"""
        with self._lock:
            if max_entries is not None:
                self.max_entries = max(0, int(max_entries))
            if ttl_seconds is not None:
                self.ttl_seconds = max(0.0, float(ttl_seconds))
            if enabled is not None:
                self.enabled = bool(enabled) and self.max_entries > 0
            self._evict_if_needed()

    @property
    def persistent_store(self) -> PersistentCacheStore | None:
        """当前挂接的持久层；启动时缓存被禁用（未接持久层）时为 ``None``。"""
        return self._disk

    def attach_store(self, store: PersistentCacheStore | None) -> bool:
        """运行时补接持久层并加载既有条目，返回是否完成补接。

        供"启动时缓存关闭、设置中稍后开启"的场景使用：不补接的话该会话
        既不落盘也读不到历史条目。已有持久层或传入空对象时不做任何事。
        """
        if store is None:
            return False
        with self._lock:
            if self._disk is not None:
                return False
            self._disk = store
            self._load_from_disk()
        return True

    # ------------------------------------------------------------------
    # 读写
    # ------------------------------------------------------------------

    def get(self, key: str) -> str | None:
        """读取缓存。命中返回译文，未命中或已过期返回 ``None``。"""
        return self._get(key, account=True)

    def get_candidate(self, key: str) -> str | None:
        """Read evidence without counting it as successful translation reuse."""
        return self._get(key, account=False)

    def record_verified_lookup(self, accepted: bool, *, rejected: bool = False) -> None:
        """Count only after the service has validated the cached evidence."""
        if not self.enabled:
            return
        with self._lock:
            self._stats.verified_requests += 1
            self._stats.verified_hits += int(accepted)
            self._stats.rejected_candidates += int(rejected)
            self._stats.hits += int(accepted)
            self._stats.misses += int(not accepted)

    def _get(self, key: str, *, account: bool) -> str | None:
        if not self.enabled:
            return None

        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._stats.misses += int(account)
                return None

            value, stored_at = entry
            if self._is_expired(stored_at):
                del self._store[key]
                self._stats.expired += 1
                self._stats.misses += int(account)
                return None

            self._store.move_to_end(key)
            self._stats.hits += int(account)
            return value

    def put(self, key: str, value: str) -> None:
        """写入缓存。"""
        if not self.enabled or value is None:
            return

        with self._lock:
            self._store[key] = (value, time.perf_counter())
            self._store.move_to_end(key)
            self._evict_if_needed()
            if self._disk is not None:
                # Enqueue before a concurrent clear can invalidate the entry;
                # the persistent store's put only locks its short queue state.
                self._disk.put(key, value, time.time())

    def discard(self, key: str) -> bool:
        """Remove one invalid entry without losing unrelated verified results."""
        with self._lock:
            removed = self._store.pop(key, None) is not None
            if self._disk is not None:
                self._disk.discard(key)
            return removed

    def get_many(self, keys: list[str]) -> dict[str, str]:
        """批量读取，返回命中的 ``{键: 译文}``。"""
        result: dict[str, str] = {}
        for key in keys:
            value = self.get(key)
            if value is not None:
                result[key] = value
        return result

    def put_many(self, items: dict[str, str]) -> None:
        """批量写入。"""
        for key, value in items.items():
            self.put(key, value)

    # ------------------------------------------------------------------
    # 维护
    # ------------------------------------------------------------------

    def clear(self) -> int:
        """清空缓存，返回被清除的条目数。"""
        with self._lock:
            count = len(self._store)
            self._store.clear()
            self._stats = CacheStats()
            if self._disk is not None:
                self._disk.clear()
        return count

    def close(self) -> None:
        """刷尽持久层的排队写入并停止后台线程；未接持久层时为空操作。"""
        if self._disk is not None:
            self._disk.close()

    def purge_expired(self) -> int:
        """主动清理过期条目，返回清理数量。

        供空闲时的定时任务调用，避免缓存被长期占位。
        """
        if not self.enabled or self.ttl_seconds <= 0:
            return 0

        removed = 0
        with self._lock:
            for key in [k for k, (_, ts) in self._store.items() if self._is_expired(ts)]:
                del self._store[key]
                removed += 1
            self._stats.expired += removed
        return removed

    def stats(self) -> CacheStats:
        """返回统计数据副本。"""
        with self._lock:
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                expired=self._stats.expired,
                verified_hits=self._stats.verified_hits,
                verified_requests=self._stats.verified_requests,
                rejected_candidates=self._stats.rejected_candidates,
            )

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _is_expired(self, stored_at: float) -> bool:
        if self.ttl_seconds <= 0:
            return False
        return (time.perf_counter() - stored_at) > self.ttl_seconds

    def _evict_if_needed(self) -> None:
        """超出容量时淘汰最久未使用的条目。调用方需持有锁。"""
        if not self.enabled:
            return
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)
            self._stats.evictions += 1
