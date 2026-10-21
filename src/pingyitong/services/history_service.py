"""翻译历史回溯（自研）。

对应产品功能规划三·（五）：

    "软件自动将每次翻译的原文与译文按时间戳保存到本地数据库，
    支持按时间、关键词检索，支持批量导出为 TXT/CSV 文件。"

隐私设计
    历史数据库完全存放在本机应用数据目录，不参与任何同步或上报。
    用户可在设置中关闭"记录原文"，此时仅保存译文——因为原文来自
    屏幕，可能包含比译文更敏感的信息（如内部系统路径、账号名）。
    关闭记录后，历史检索只能按译文关键词进行。

并发
    OCR/翻译在后台线程中产生历史记录，UI 线程中执行检索与导出，
    因此使用每线程独立的 SQLite 连接（``threading.local``），
    并启用 WAL 模式以允许读写并发。
"""

from __future__ import annotations

import csv
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from ..utils import paths
from ..utils.logger import get_logger

logger = get_logger(__name__)


def _literal_keyword_pattern(keyword: str) -> str:
    """Build a substring LIKE pattern without treating user text as wildcards."""
    escaped = keyword.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS translation_history (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT    NOT NULL,
    source_lang   TEXT    NOT NULL DEFAULT '',
    target_lang   TEXT    NOT NULL DEFAULT '',
    source_text   TEXT    NOT NULL DEFAULT '',
    target_text   TEXT    NOT NULL DEFAULT '',
    model_id      TEXT    NOT NULL DEFAULT '',
    from_cache    INTEGER NOT NULL DEFAULT 0,
    confidence    REAL    NOT NULL DEFAULT 1.0
);
CREATE INDEX IF NOT EXISTS idx_history_created ON translation_history(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_history_target  ON translation_history(target_text);
"""


@dataclass
class HistoryRecord:
    """一条翻译历史。"""

    id: int = 0
    created_at: str = ""
    source_lang: str = ""
    target_lang: str = ""
    source_text: str = ""
    target_text: str = ""
    model_id: str = ""
    from_cache: bool = False
    confidence: float = 1.0

    @property
    def created_at_display(self) -> str:
        """用于界面展示的时间字符串。"""
        try:
            moment = datetime.fromisoformat(self.created_at)
            return moment.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            return self.created_at

    @property
    def preview(self) -> str:
        """译文预览（用于列表）。"""
        text = self.target_text.replace("\n", " ")
        return text if len(text) <= 60 else f"{text[:60]}…"


class HistoryService:
    """翻译历史的持久化与检索。"""

    def __init__(self, db_path: Path | None = None, enabled: bool = True) -> None:
        self._path = Path(db_path) if db_path else paths.history_db_path()
        self.enabled = bool(enabled)
        self._local = threading.local()
        #: 写操作的互斥锁。**注意：这是非可重入锁**，绝不能在已持有的
        #: 情况下再次获取（见 _init_lock 的说明）。
        self._write_lock = threading.Lock()
        #: 建表专用锁，与 _write_lock 分开。
        #:
        #: 为什么必须分开：`record` / `prune` / `clear` 都会先持有
        #: `_write_lock` 再调用 `_connection()`，而 `_connection()` 首次
        #: 需要建表也要加锁。若两处共用同一把非可重入锁，就会**自我死锁**
        #: ——实测后果是主线程永久卡死、程序"未响应"（py-spy 堆栈显示
        #: MainThread 停在 _connection）。
        #: 拆成两把锁从结构上消除该问题，而非依赖锁的可重入语义。
        self._init_lock = threading.Lock()
        self._initialised = False

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------

    def _connection(self) -> sqlite3.Connection:
        """取得当前线程的连接，首次访问时初始化表结构。"""
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            return connection

        self._path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self._path), timeout=5.0)
        connection.row_factory = sqlite3.Row
        try:
            # WAL 允许"写入的同时检索"，避免后台写库阻塞界面查询
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            if not self._initialised:
                # 用独立的 _init_lock：调用方可能已持有 _write_lock，
                # 共用同一把非可重入锁会导致自我死锁。
                with self._init_lock:
                    if not self._initialised:
                        connection.executescript(_SCHEMA)
                        connection.commit()
                        self._initialised = True
        except sqlite3.DatabaseError as exc:
            connection.close()
            self._quarantine_corrupt_database(exc)
            raise
        # 连接必须在建表成功后才缓存：建表失败就把连接放进 thread-local，
        # 该线程此后永远命中"已缓存"分支，全部查询报 no such table（审查 H1）。
        self._local.connection = connection
        return connection

    def _quarantine_corrupt_database(self, exc: Exception) -> None:
        """数据库损坏时隔离旧文件，给历史功能一条重建生路（审查 H2）。

        仅在初始化失败时进入；同一异常链只隔离一次，避免并发线程
        重复改名。原文件保留为 .corrupt-<时间戳> 供事后取证。
        """
        import time as _time

        corrupt = self._path.with_name(f"{self._path.name}.corrupt-{_time.strftime('%Y%m%d-%H%M%S')}")
        try:
            self._path.rename(corrupt)
            logger.error("历史数据库损坏（%s），旧文件已隔离为 %s，将重建新库", exc, corrupt.name)
        except OSError:
            logger.error("历史数据库损坏（%s），且旧文件无法隔离，历史功能保持不可用", exc)
        with self._init_lock:
            self._initialised = False

    def close(self) -> None:
        """关闭当前线程的连接。"""
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            try:
                connection.close()
            except sqlite3.Error:
                logger.debug("关闭历史数据库连接时出现异常（可忽略）")
            self._local.connection = None

    def is_available(self) -> bool:
        """数据库是否可访问。"""
        try:
            self._connection()
        except (OSError, sqlite3.Error) as exc:
            logger.error("历史数据库不可用：%s", exc)
            return False
        return True

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def record(
        self,
        source_text: str,
        target_text: str,
        source_lang: str = "",
        target_lang: str = "",
        model_id: str = "",
        from_cache: bool = False,
        confidence: float = 1.0,
        store_source: bool = True,
    ) -> int:
        """记录一条翻译。

        :param store_source: 是否保存原文。为 ``False`` 时原文列留空，
            仅保留译文，用于隐私敏感场景。
        :return: 新记录的行号；失败返回 0。
        """
        if not self.enabled or not target_text.strip():
            return 0

        payload = (
            datetime.now().isoformat(timespec="seconds"),
            source_lang,
            target_lang,
            source_text if store_source else "",
            target_text,
            model_id,
            1 if from_cache else 0,
            float(confidence),
        )

        try:
            with self._write_lock:
                connection = self._connection()
                cursor = connection.execute(
                    "INSERT INTO translation_history "
                    "(created_at, source_lang, target_lang, source_text, target_text,"
                    " model_id, from_cache, confidence) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    payload,
                )
                connection.commit()
                return int(cursor.lastrowid or 0)
        except (OSError, sqlite3.Error) as exc:
            logger.error("写入翻译历史失败：%s", exc)
            return 0

    def record_batch(self, records: list[HistoryRecord]) -> int:
        """批量写入，返回成功写入的条数。"""
        if not self.enabled or not records:
            return 0

        rows = [
            (
                record.created_at or datetime.now().isoformat(timespec="seconds"),
                record.source_lang,
                record.target_lang,
                record.source_text,
                record.target_text,
                record.model_id,
                1 if record.from_cache else 0,
                float(record.confidence),
            )
            for record in records
            if record.target_text.strip()
        ]
        if not rows:
            return 0

        try:
            with self._write_lock:
                connection = self._connection()
                connection.executemany(
                    "INSERT INTO translation_history "
                    "(created_at, source_lang, target_lang, source_text, target_text,"
                    " model_id, from_cache, confidence) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )
                connection.commit()
                return len(rows)
        except (OSError, sqlite3.Error) as exc:
            logger.error("批量写入翻译历史失败：%s", exc)
            return 0

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------

    def search(
        self,
        keyword: str = "",
        limit: int = 200,
        offset: int = 0,
        days: int = 0,
    ) -> list[HistoryRecord]:
        """检索历史记录。

        :param keyword: 关键词，同时匹配原文与译文（子串匹配）。
        :param limit: 返回条数上限。
        :param offset: 偏移量，用于分页。
        :param days: 仅检索最近 N 天；``0`` 表示不限。
        """
        conditions: list[str] = []
        params: list[object] = []

        if keyword:
            conditions.append("(source_text LIKE ? ESCAPE '\\' OR target_text LIKE ? ESCAPE '\\')")
            pattern = _literal_keyword_pattern(keyword)
            params.extend([pattern, pattern])

        if days > 0:
            since = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
            conditions.append("created_at >= ?")
            params.append(since)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = (
            "SELECT * FROM translation_history "
            f"{where} ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?"
        )
        params.extend([max(1, limit), max(0, offset)])

        try:
            connection = self._connection()
            rows = connection.execute(sql, params).fetchall()
        except (OSError, sqlite3.Error) as exc:
            logger.error("检索翻译历史失败：%s", exc)
            return []

        return [_row_to_record(row) for row in rows]

    def recent(self, limit: int = 50) -> list[HistoryRecord]:
        """返回最近的翻译记录。"""
        return self.search(limit=limit)

    def count(self, keyword: str = "") -> int:
        """统计记录总数（可带关键词）。"""
        try:
            connection = self._connection()
            if keyword:
                pattern = _literal_keyword_pattern(keyword)
                row = connection.execute(
                    "SELECT COUNT(*) AS n FROM translation_history "
                    "WHERE source_text LIKE ? ESCAPE '\\' OR target_text LIKE ? ESCAPE '\\'",
                    (pattern, pattern),
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT COUNT(*) AS n FROM translation_history"
                ).fetchone()
            return int(row["n"]) if row else 0
        except (OSError, sqlite3.Error) as exc:
            logger.error("统计翻译历史失败：%s", exc)
            return 0

    # ------------------------------------------------------------------
    # 导出
    # ------------------------------------------------------------------

    def export_csv(self, target: Path, keyword: str = "", days: int = 0) -> int:
        """导出为 CSV，返回导出的记录数。"""
        records = self.search(keyword=keyword, limit=1_000_000, days=days)
        if not records:
            return 0

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            # utf-8-sig 便于 Excel 直接识别中文
            with target.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    ["时间", "源语言", "目标语言", "原文", "译文", "模型", "命中缓存"]
                )
                for record in records:
                    writer.writerow(
                        [
                            record.created_at_display,
                            record.source_lang,
                            record.target_lang,
                            record.source_text,
                            record.target_text,
                            record.model_id,
                            "是" if record.from_cache else "否",
                        ]
                    )
            return len(records)
        except OSError as exc:
            logger.error("导出历史 CSV 失败：%s", exc)
            return 0

    def export_txt(self, target: Path, keyword: str = "", days: int = 0) -> int:
        """导出为纯文本，返回导出的记录数。"""
        records = self.search(keyword=keyword, limit=1_000_000, days=days)
        if not records:
            return 0

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(f"[{record.created_at_display}] "
                                 f"{record.source_lang}→{record.target_lang}\n")
                    if record.source_text:
                        handle.write(f"原文：{record.source_text}\n")
                    handle.write(f"译文：{record.target_text}\n")
                    handle.write("-" * 60 + "\n")
            return len(records)
        except OSError as exc:
            logger.error("导出历史 TXT 失败：%s", exc)
            return 0

    # ------------------------------------------------------------------
    # 维护
    # ------------------------------------------------------------------

    def clear(self) -> int:
        """清空全部历史，返回删除的条数。"""
        try:
            with self._write_lock:
                connection = self._connection()
                cursor = connection.execute("DELETE FROM translation_history")
                connection.commit()
                deleted = int(cursor.rowcount or 0)
                # 回收磁盘空间：VACUUM 可能因并发读卡顿（SQLITE_BUSY），
                # 失败只记日志——删除已成功，不能吞掉结果误导调用方（审查 H3）。
                try:
                    connection.execute("VACUUM")
                except sqlite3.Error:
                    logger.debug("清空后 VACUUM 失败（可忽略，磁盘空间稍后回收）")
            logger.info("已清空翻译历史：%s 条", deleted)
            return deleted
        except (OSError, sqlite3.Error) as exc:
            logger.error("清空翻译历史失败：%s", exc)
            return 0

    def prune(self, max_entries: int) -> int:
        """裁剪历史，仅保留最近的 ``max_entries`` 条。

        :param max_entries: 保留上限；``0`` 表示不限制。
        :return: 删除的条数。
        """
        if max_entries <= 0:
            return 0

        try:
            with self._write_lock:
                connection = self._connection()
                total = self.count()
                if total <= max_entries:
                    return 0

                cursor = connection.execute(
                    "DELETE FROM translation_history WHERE id NOT IN ("
                    "  SELECT id FROM translation_history"
                    "  ORDER BY created_at DESC, id DESC LIMIT ?"
                    ")",
                    (max_entries,),
                )
                connection.commit()
                return int(cursor.rowcount or 0)
        except (OSError, sqlite3.Error) as exc:
            logger.error("裁剪翻译历史失败：%s", exc)
            return 0

    def database_size_mb(self) -> float:
        """返回数据库文件大小（MB），用于设置面板展示。"""
        try:
            return self._path.stat().st_size / (1024 * 1024)
        except OSError:
            return 0.0


def _row_to_record(row: sqlite3.Row) -> HistoryRecord:
    """把数据库行转换为 :class:`HistoryRecord`。"""
    return HistoryRecord(
        id=int(row["id"]),
        created_at=str(row["created_at"]),
        source_lang=str(row["source_lang"]),
        target_lang=str(row["target_lang"]),
        source_text=str(row["source_text"]),
        target_text=str(row["target_text"]),
        model_id=str(row["model_id"]),
        from_cache=bool(row["from_cache"]),
        confidence=float(row["confidence"]),
    )
