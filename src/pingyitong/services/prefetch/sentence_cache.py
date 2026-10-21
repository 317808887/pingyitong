"""持久化句级翻译缓存（联网预翻译，自研）。

对应方案 v1.1 §3.4 / §5：

- SQLite 存储，``WAL`` + ``synchronous=NORMAL``（读多写少且要求不阻塞
  实时链路）；schema 变更走 ``user_version`` 迁移；
- 键 = sha256(归一化原文 | 源语言 | 目标语言 | 领域标识 | 模型版本)；
- 失效：TTL 到期、``model_version`` 变化、网页 ``etag`` 变化；
- 容量：按文本字节估算上限，超限按 ``last_used_at`` LRU 淘汰；
- fuzzy：字幕场景按 ``(track_id, 行号窗口)`` 用 rapidfuzz 匹配，
  **窗口与序列校验参数由 P0 标定**（±4 行 / 0.80 / 校验 2 行）。

线程约束（v1.1 §1.3）：读连接归属调用线程（线程局部连接），
写操作持同一把进程内锁串行；任何异常向上抛由调用方决定降级。
"""

from __future__ import annotations

import contextlib
import hashlib
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from rapidfuzz import fuzz

from .normalizer import fuzzy_semantics_match, normalize_for_key, normalize_for_match

#: schema 版本：变更结构时 +1 并在 _migrate 补迁移
_USER_VERSION = 2

_DDL = """
CREATE TABLE IF NOT EXISTS prefetch_sentence (
    key           TEXT PRIMARY KEY,
    source_kind   TEXT NOT NULL,
    track_or_site TEXT NOT NULL,
    source_norm   TEXT NOT NULL,
    source_raw    TEXT NOT NULL,
    translated    TEXT NOT NULL,
    src_lang      TEXT NOT NULL,
    tgt_lang      TEXT NOT NULL,
    model_version TEXT NOT NULL,
    created_at    REAL NOT NULL,
    expires_at    REAL NOT NULL,
    etag          TEXT,
    last_used_at  REAL NOT NULL,
    used_count    INTEGER NOT NULL DEFAULT 0,
    bytes_in      INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_prefetch_lru
    ON prefetch_sentence(track_or_site, last_used_at);
CREATE INDEX IF NOT EXISTS idx_prefetch_version
    ON prefetch_sentence(model_version, expires_at);

CREATE TABLE IF NOT EXISTS prefetch_subtitle_line (
    track_id    TEXT NOT NULL,
    line_no     INTEGER NOT NULL,
    start_ms    INTEGER NOT NULL,
    end_ms      INTEGER NOT NULL,
    source_norm TEXT NOT NULL,
    key         TEXT NOT NULL,
    PRIMARY KEY (track_id, line_no)
);

CREATE TABLE IF NOT EXISTS prefetch_meta (
    site_or_track TEXT PRIMARY KEY,
    kind          TEXT NOT NULL,
    etag          TEXT,
    updated_at    REAL,
    title         TEXT
);
"""


@dataclass
class SentenceEntry:
    """一条待写入的句级译文。"""

    source: str
    translated: str
    src_lang: str
    tgt_lang: str
    domain: str                  # 字幕轨 id 或站点主机名
    model_version: str
    source_kind: str             # subtitle | web
    ttl_days: float
    etag: str | None = None
    bytes_in: int = 0
    #: 字幕行索引字段（source_kind="subtitle" 时必填）
    line_no: int | None = None
    start_ms: int | None = None
    end_ms: int | None = None


@dataclass(frozen=True)
class FuzzyHit:
    """fuzzy 命中结果。"""

    key: str
    translated: str
    line_no: int
    confidence: float


class SentenceCache:
    """持久化句级缓存。线程安全：读线程局部连接、写串行。"""

    def __init__(
        self,
        path: Path,
        model_version: str,
        max_bytes: float = 64.0,
    ) -> None:
        self._path = Path(path)
        self._model_version = model_version
        # max_bytes 历史参数名实际接收 MB，与 PrefetchConfig.cache_max_mb 一致。
        # 容量下限保护作用在**字节数**上而不是 MB 参数上：
        # 写成 max(1.0, mb) 会把小于 1MB 的合法配置（测试/受限环境）钳成 1MB
        self._max_bytes = max(1.0, float(max_bytes) * 1024 * 1024)
        self._write_lock = threading.Lock()
        self._local = threading.local()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ------------------------------------------------------------------
    # 连接与 schema
    # ------------------------------------------------------------------

    def _connection(self) -> sqlite3.Connection:
        """线程局部连接：SQLite 连接不可跨线程复用。"""
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = sqlite3.connect(str(self._path), timeout=5.0)
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute("PRAGMA foreign_keys=ON")
            self._local.connection = connection
        return connection

    def _init_schema(self) -> None:
        with self._write_lock:
            connection = self._connection()
            connection.executescript(_DDL)
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version < _USER_VERSION:
                # 旧键折叠数字/大小写，可能包含语义碰撞；升级后重新预热。
                connection.execute("DELETE FROM prefetch_subtitle_line")
                connection.execute("DELETE FROM prefetch_sentence")
                connection.execute(f"PRAGMA user_version={_USER_VERSION}")
            connection.commit()

    # ------------------------------------------------------------------
    # 键
    # ------------------------------------------------------------------

    @staticmethod
    def cache_key(
        source_norm: str,
        src_lang: str,
        tgt_lang: str,
        domain: str,
        model_version: str,
    ) -> str:
        """缓存键：五个字段组合后 sha256。

        归一化原文进哈希而不是原文：同一段话 OCR 两次读到略有差异的文本，
        归一化后应命中同一条缓存。
        """
        source_norm = normalize_for_key(source_norm)
        material = "|".join((source_norm, src_lang, tgt_lang, domain, model_version))
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def get_exact(self, key: str) -> str | None:
        """精确键查询；命中即刷新 LRU 与命中计数。"""
        now = time.time()
        try:
            row = self._connection().execute(
                "SELECT translated, expires_at, model_version "
                "FROM prefetch_sentence WHERE key = ?",
                (key,),
            ).fetchone()
        except sqlite3.Error:
            return None
        if row is None:
            return None
        translated, expires_at, row_version = row
        # 版本不一致视为失效（模型升级后旧译文不可显示）
        if row_version != self._model_version:
            return None
        if expires_at < now:
            return None
        self._touch(key, now)
        return str(translated)

    def get_fuzzy(
        self,
        text_norm: str,
        track_id: str,
        center_line: int,
        window: int = 4,
        min_ratio: float = 0.80,
        sequence_check: int = 2,
        last_confirmed_line: int | None = None,
        src_lang: str | None = None,
        tgt_lang: str | None = None,
    ) -> FuzzyHit | None:
        """字幕 fuzzy 查询：±window 行窗口内匹配，带序列一致性校验。

        匹配参数由 P0 标定（±4 / 0.80 / 校验 2 → 命中 99.0%、错配 0.0%）。
        低置信返回 ``None``——**宁可不命中，不可错配**（错配会把甲句的
        译文显示给乙句）。
        """
        if window <= 0:
            return None
        lo, hi = center_line - window, center_line + window
        try:
            rows = self._connection().execute(
                "SELECT l.line_no, l.source_norm, l.key FROM prefetch_subtitle_line l "
                "JOIN prefetch_sentence s ON s.key = l.key "
                "WHERE l.track_id = ? AND l.line_no BETWEEN ? AND ? "
                "AND (? IS NULL OR s.tgt_lang = ?) "
                "AND (? IS NULL OR s.src_lang IN (?, 'auto')) "
                "AND s.model_version = ? AND s.expires_at >= ?",
                (track_id, lo, hi, tgt_lang, tgt_lang,
                 None if src_lang == "auto" else src_lang, src_lang,
                 self._model_version, time.time()),
            ).fetchall()
        except sqlite3.Error:
            return None

        best_no, best_key, best_score = None, None, 0.0
        for line_no, source_norm, key in rows:
            if not fuzzy_semantics_match(text_norm, source_norm):
                continue
            score = fuzz.partial_ratio(text_norm, normalize_for_match(source_norm))
            if score > best_score:
                best_no, best_key, best_score = line_no, key, score

        if best_no is None or best_score / 100.0 < min_ratio:
            return None

        # 序列一致性：与最近确认行的跳变超出容许范围 → 不可信
        if last_confirmed_line is not None and sequence_check > 0:
            drift = abs(best_no - last_confirmed_line)
            if drift > sequence_check + window:
                return None

        translated = self.get_exact(best_key)
        if translated is None:
            return None
        return FuzzyHit(key=best_key, translated=translated,
                        line_no=best_no, confidence=best_score / 100.0)

    def locate_line(
        self, text_norm: str, track_id: str,
        src_lang: str | None = None, tgt_lang: str | None = None,
    ) -> FuzzyHit | None:
        """全轨首定位：在整条字幕轨里找与 OCR 文本最匹配的行。

        方案 v1.2 §3.3：首定位只在"没有任何位置锚点"时发生（如导入后
        第一帧），且应避免出现在高频帧路径上——2400 行 × rapidfuzz 约
        3~8ms，定位成功后由调用方 ``confirm_line`` 固定 ±N 行窗口。
        低置信返回 ``None``（宁可不命中，不可错配）。
        """
        if not text_norm:
            return None
        try:
            rows = self._connection().execute(
                "SELECT l.line_no, l.source_norm, l.key FROM prefetch_subtitle_line l "
                "JOIN prefetch_sentence s ON s.key = l.key WHERE l.track_id = ? "
                "AND (? IS NULL OR s.tgt_lang = ?) "
                "AND (? IS NULL OR s.src_lang IN (?, 'auto')) "
                "AND s.model_version = ? AND s.expires_at >= ?",
                (track_id, tgt_lang, tgt_lang,
                 None if src_lang == "auto" else src_lang, src_lang,
                 self._model_version, time.time()),
            ).fetchall()
        except sqlite3.Error:
            return None
        from rapidfuzz import fuzz

        best_no, best_key, best_score = None, None, 0.0
        for line_no, source_norm, key in rows:
            if not fuzzy_semantics_match(text_norm, source_norm):
                continue
            score = fuzz.partial_ratio(text_norm, normalize_for_match(source_norm))
            if score > best_score:
                best_no, best_key, best_score = line_no, key, score
        if best_no is None or best_score / 100.0 < 0.60:
            # 首定位阈值低于跟踪阈值：首帧 OCR 噪声更大，宁可低置信
            # 重定位也不直接错配；仍不过则走实时推理
            return None
        translated = self.get_exact(best_key)
        if translated is None:
            return None
        return FuzzyHit(key=best_key, translated=translated,
                        line_no=best_no, confidence=best_score / 100.0)

    def get_subtitle_line(self, track_id: str, line_no: int) -> dict | None:
        """取字幕行索引（调度器 pull 时需要原文与时间轴）。"""
        try:
            row = self._connection().execute(
                "SELECT l.source_norm, s.source_raw, l.start_ms, l.end_ms "
                "FROM prefetch_subtitle_line l "
                "JOIN prefetch_sentence s ON s.key = l.key "
                "WHERE l.track_id = ? AND l.line_no = ?",
                (track_id, int(line_no)),
            ).fetchone()
        except sqlite3.Error:
            return None
        if row is None:
            return None
        return {
            "source_norm": row[0], "source_raw": row[1],
            "start_ms": row[2], "end_ms": row[3],
        }

    def _touch(self, key: str, now: float) -> None:
        try:
            with self._write_lock:
                self._connection().execute(
                    "UPDATE prefetch_sentence SET last_used_at = ?, "
                    "used_count = used_count + 1 WHERE key = ?",
                    (now, key),
                )
                self._connection().commit()
        except sqlite3.Error:
            pass  # 计数失败不影响主链路

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def put_many(self, entries: list[SentenceEntry]) -> int:
        """批量写入（预翻译结果落库），返回写入条数。"""
        if not entries:
            return 0
        now = time.time()
        rows = []
        line_rows = []
        for entry in entries:
            source_norm = normalize_for_key(entry.source)
            if not source_norm:
                continue
            key = self.cache_key(
                source_norm, entry.src_lang, entry.tgt_lang,
                entry.domain, entry.model_version,
            )
            rows.append((
                key, entry.source_kind, entry.domain, source_norm,
                entry.source, entry.translated, entry.src_lang, entry.tgt_lang,
                entry.model_version, now, now + entry.ttl_days * 86400.0,
                entry.etag, now, 0, entry.bytes_in,
            ))
        if not rows:
            return 0
        try:
            with self._write_lock:
                connection = self._connection()
                connection.executemany(
                    "INSERT OR REPLACE INTO prefetch_sentence VALUES "
                    "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    rows,
                )
                # 字幕行索引：把 (track_id, line_no) 指到对应键
                for entry in entries:
                    if entry.line_no is not None and entry.start_ms is not None:
                        source_norm = normalize_for_key(entry.source)
                        key = self.cache_key(
                            source_norm, entry.src_lang, entry.tgt_lang,
                            entry.domain, entry.model_version,
                        )
                        line_rows.append((
                            entry.domain, entry.line_no,
                            int(entry.start_ms), int(entry.end_ms or 0),
                            source_norm, key,
                        ))
                if line_rows:
                    connection.executemany(
                        "INSERT OR REPLACE INTO prefetch_subtitle_line VALUES "
                        "(?,?,?,?,?,?)",
                        line_rows,
                    )
                connection.commit()
        except sqlite3.Error:
            return 0
        return len(rows)

    def replace_subtitle_lines(self, track_id: str, lines: list[SubtitleLineEntry]) -> None:
        """重建一条字幕轨的行索引（文件热替换 / 版本更新）。"""
        try:
            with self._write_lock:
                connection = self._connection()
                connection.execute(
                    "DELETE FROM prefetch_subtitle_line WHERE track_id = ?",
                    (track_id,),
                )
                connection.executemany(
                    "INSERT OR REPLACE INTO prefetch_subtitle_line VALUES (?,?,?,?,?,?)",
                    [
                        (track_id, line.line_no, int(line.start_ms),
                         int(line.end_ms), line.source_norm, line.key)
                        for line in lines
                    ],
                )
                connection.commit()
        except sqlite3.Error:
            pass

    # ------------------------------------------------------------------
    # 维护
    # ------------------------------------------------------------------

    def usage_bytes(self) -> int:
        """缓存内容字节数（不含索引与指标表）。"""
        try:
            row = self._connection().execute(
                "SELECT COALESCE(SUM(LENGTH(CAST(source_raw AS BLOB)) "
                "+ LENGTH(CAST(translated AS BLOB))), 0) "
                "FROM prefetch_sentence"
            ).fetchone()
            return int(row[0])
        except sqlite3.Error:
            return 0

    def prune(self) -> int:
        """清理：TTL 到期、旧模型版本。返回删除行数。"""
        now = time.time()
        try:
            with self._write_lock:
                connection = self._connection()
                cursor = connection.execute(
                    "DELETE FROM prefetch_sentence "
                    "WHERE expires_at < ? OR model_version != ?",
                    (now, self._model_version),
                )
                deleted = cursor.rowcount
                deleted += connection.execute(
                    "DELETE FROM prefetch_subtitle_line WHERE key NOT IN "
                    "(SELECT key FROM prefetch_sentence)"
                ).rowcount
                connection.commit()
        except sqlite3.Error:
            return 0

        # 容量上限：超出按 LRU 淘汰
        if self.usage_bytes() > self._max_bytes:
            self._evict_lru()
        return deleted

    def _evict_lru(self) -> int:
        """按 last_used_at 淘汰最久未用条目，直到回到容量内。"""
        evicted = 0
        try:
            with self._write_lock:
                connection = self._connection()
                rows = connection.execute(
                    "SELECT key FROM prefetch_sentence ORDER BY last_used_at ASC"
                ).fetchall()
                for (key,) in rows:
                    if self.usage_bytes() <= self._max_bytes:
                        break
                    connection.execute(
                        "DELETE FROM prefetch_sentence WHERE key = ?", (key,)
                    )
                    evicted += 1
                connection.commit()
        except sqlite3.Error:
            pass
        return evicted

    def invalidate_source(self, site_or_track: str) -> None:
        """源内容更新（etag 变化 / 字幕文件替换）时删除该源全部条目。"""
        try:
            with self._write_lock:
                connection = self._connection()
                connection.execute(
                    "DELETE FROM prefetch_sentence WHERE track_or_site = ?",
                    (site_or_track,),
                )
                connection.execute(
                    "DELETE FROM prefetch_subtitle_line WHERE track_id = ?",
                    (site_or_track,),
                )
                connection.commit()
        except sqlite3.Error:
            pass

    def clear(self) -> None:
        """用户点"清除缓存"。"""
        try:
            with self._write_lock:
                connection = self._connection()
                connection.execute("DELETE FROM prefetch_sentence")
                connection.execute("DELETE FROM prefetch_subtitle_line")
                connection.execute("DELETE FROM prefetch_meta")
                connection.commit()
        except sqlite3.Error:
            pass

    def close(self) -> None:
        """释放当前线程连接（测试与退出用）。"""
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            with contextlib.suppress(sqlite3.Error):
                connection.close()
            self._local.connection = None


@dataclass
class SubtitleLineEntry:
    """字幕行索引条目（写入 prefetch_subtitle_line）。"""

    line_no: int
    start_ms: int
    end_ms: int
    source_norm: str
    key: str
