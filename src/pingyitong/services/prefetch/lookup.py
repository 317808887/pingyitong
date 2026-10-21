"""接缝 A：PrefetchLookup（联网预翻译 → 现有翻译链路）。

对应方案 v1.1 §1.1 接缝 A。翻译服务在内存缓存 miss 后调用
:meth:`PrefetchLookup.lookup` 一次；翻译服务只依赖本模块的协议，
**不得 import 字幕/网页模块**（依赖方向：核心 ← prefetch 实现）。

**常开语义（§12-D1）**：读缓存不产生流量，因此 lookup 不受总开关
控制——总开关只管网络抓取与预热。旧译文失效由 TTL/版本机制独立保证。
任何异常、超时一律返回 ``None``，等价现状（接缝开销预算 p95 ≤ 5ms，
超出即返回 None，绝不阻塞实时链路）。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Protocol

from ...utils.logger import get_logger
from .sentence_cache import SentenceCache

logger = get_logger(__name__)

#: 接缝单次调用耗时预算（秒）。P0 建议 p95 ≤ 5ms，超预算直接放弃。
_LOOKUP_BUDGET_S = 0.05


@dataclass(frozen=True)
class LookupContext:
    """查询上下文：决定走精确键通道还是字幕 fuzzy 通道。"""

    target_lang: str
    src_lang: str = "auto"
    #: 字幕场景：活跃轨 id + 当前确认行号（供 ±N 窗口匹配）
    track_id: str | None = None
    line_no: int | None = None
    #: 网页场景：站点主机名（domain），走精确键
    domain: str = "screen"


class PrefetchLookup(Protocol):
    """翻译服务依赖的查询协议（依赖倒置，核心层不 import prefetch 实现）。"""

    def lookup(self, text: str, context: LookupContext) -> str | None: ...


class SentenceCacheLookup:
    """基于 :class:`SentenceCache` 的默认实现。

    - ``context.track_id`` 存在且行号可得 → fuzzy 通道（字幕）；
    - 否则 → 精确键通道（网页/阅读器/屏幕文本）；
    - 全程持预算超时保护；任何异常返回 ``None``。
    """

    def __init__(
        self,
        cache: SentenceCache,
        fuzzy_ratio: float = 0.80,
        fuzzy_window: int = 4,
        sequence_check: int = 2,
        row_lock_misses: int = 2,
    ) -> None:
        self._cache = cache
        self._fuzzy_ratio = fuzzy_ratio
        self._window = fuzzy_window
        self._sequence_check = sequence_check
        #: 每轨最近确认行号（fuzzy 跟踪；由上层在确认命中后回写）
        self._last_line: dict[str, int] = {}
        #: 接缝 C 行锁：track_id -> (line_no, translated, 连续 miss 计数)。
        #: 同一行字幕在屏幕上会停留数秒，期间 OCR 每帧都读——某一帧
        #: 识别质量差导致 fuzzy miss 时，若直接放行走实时推理，译文会
        #: 在"缓存直出"与"实时输出"之间跳变（闪烁）。行锁让该行在
        #: 连续 row_lock_misses 次 miss 内保持已确认译文，之后释放。
        self._row_locks: dict[str, dict] = {}
        self._language_pairs: dict[str, tuple[str, str]] = {}
        self._row_lock_misses = max(1, int(row_lock_misses))
        self._active_track_id: str | None = None
        self._lock = threading.Lock()

    def _lookup_subtitle(
        self, text: str, track_id: str, context: LookupContext,
    ) -> str | None:
        """字幕 fuzzy 通道：首定位 / ±N 跟踪 / 序列校验 / 行锁（接缝 C）。"""
        from .normalizer import fuzzy_semantics_match, normalize_for_match

        text_norm = normalize_for_match(text)
        with self._lock:
            pair = (context.src_lang, context.target_lang)
            if self._language_pairs.get(track_id) != pair:
                self._last_line.pop(track_id, None)
                self._row_locks.pop(track_id, None)
                self._language_pairs[track_id] = pair
            last = self._last_line.get(track_id)
            locked = self._row_locks.get(track_id)

        if last is None:
            # 首定位（无锚点时的全轨一次性定位）
            hit = self._cache.locate_line(
                text_norm, track_id,
                src_lang=context.src_lang, tgt_lang=context.target_lang,
            )
            if hit is None:
                return None
            self.confirm_line(track_id, hit.line_no, hit.translated, source_text=text)
            return hit.translated

        hit = self._cache.get_fuzzy(
            text_norm, track_id, last,
            window=self._window,
            min_ratio=self._fuzzy_ratio,
            sequence_check=self._sequence_check,
            last_confirmed_line=last,
            src_lang=context.src_lang, tgt_lang=context.target_lang,
        )
        if hit is not None:
            self.confirm_line(track_id, hit.line_no, hit.translated, source_text=text)
            return hit.translated

        # fuzzy miss：行锁保持已确认译文（防闪烁），连续超限才释放
        if locked is not None:
            with self._lock:
                entry = self._row_locks.get(track_id)
                if entry is None:
                    return None
                if not fuzzy_semantics_match(text, entry.get("source_text", "")):
                    self._row_locks.pop(track_id, None)
                    return None
                entry["misses"] += 1
                if entry["misses"] <= self._row_lock_misses:
                    return entry["translated"]
                self._row_locks.pop(track_id, None)
        return None

    def confirm_line(self, track_id: str, line_no: int,
                     translated: str | None = None, source_text: str = "") -> None:
        """fuzzy 命中确认后回写行号（跟踪窗口锚点）并上行锁。"""
        with self._lock:
            self._last_line[track_id] = line_no
            if translated is not None:
                self._row_locks[track_id] = {
                    "line_no": line_no, "translated": translated, "misses": 0,
                    "source_text": source_text,
                }

    def release_lock(self, track_id: str) -> None:
        """解除行锁（目标语言变更 / 轨注销时调用）。"""
        with self._lock:
            self._row_locks.pop(track_id, None)

    def set_active_track(self, track_id: str | None) -> None:
        """设置活跃轨：此后 lookup 自动走字幕 fuzzy 通道。

        由应用层在活跃轨切换时调用；传 None 回到精确键通道。
        """
        with self._lock:
            previous = self._active_track_id
            self._active_track_id = track_id
            if previous is not None:
                self._last_line.pop(previous, None)
                self._row_locks.pop(previous, None)
                self._language_pairs.pop(previous, None)

    def lookup(self, text: str, context: LookupContext) -> str | None:
        started = time.perf_counter()
        try:
            return self._lookup_within_budget(text, context, started)
        except Exception:  # noqa: BLE001 - 接缝永不抛异常
            logger.debug("预翻译缓存查询异常，按 miss 处理", exc_info=True)
            return None

    def _lookup_within_budget(
        self, text: str, context: LookupContext, started: float
    ) -> str | None:
        if not (text or "").strip():
            return None

        with self._lock:
            track_id = context.track_id or self._active_track_id

        if track_id is not None:
            result = self._lookup_subtitle(text, track_id, context)
        else:
            key = SentenceCache.cache_key(
                text, context.src_lang, context.target_lang,
                context.domain, self._cache._model_version,
            )
            result = self._cache.get_exact(key)

        if time.perf_counter() - started > _LOOKUP_BUDGET_S:
            logger.debug("预翻译缓存查询超出接缝预算（%.0fms），按 miss 处理",
                         (time.perf_counter() - started) * 1000)
            # 模块契约：超预算一律返回 None。即便结果已在手上，放行它
            # 等于默许实时链路可以无限期等待缓存——预算从此只是装饰。
            return None
        return result
