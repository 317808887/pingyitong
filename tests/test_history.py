"""翻译历史服务测试。

**本文件最先要保证的是一件具体的事：不能死锁。**

实测发生过一次严重缺陷：`HistoryService` 用非可重入的 ``threading.Lock``
同时保护"写操作"和"建表初始化"，而 ``record`` / ``prune`` / ``clear`` 会
先持有写锁再调用 ``_connection()``，后者建表时又去拿同一把锁——
**自我死锁**。后果是主线程永久阻塞、程序显示"未响应"，而且窗口会保留
最后一帧画面，从截图上看一切正常，极难察觉。

因此这里的每个公开方法调用都用**看门狗**包裹：超过 10 秒未返回即判定为
死锁并让测试失败，而不是让整个测试套件挂住。
"""

from __future__ import annotations

import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from pingyitong.services.history_service import HistoryRecord, HistoryService

#: 单个调用的最长容忍时间（秒）。正常调用都在毫秒级；
#: 超过这个值说明发生了死锁或异常阻塞。
WATCHDOG_SECONDS = 10.0


class Watchdog:
    """在超时后中断被监视代码的上下文管理器。

    用守护线程 + 事件实现：超时则记录错误并置位标志，被监视的调用返回后
    由调用方断言。注意它**不会**强制打断已死锁的线程（Python 无法安全地
    做到），但能让测试快速失败并给出明确信息，而不是挂住整个测试进程。
    """

    def __init__(self, timeout: float = WATCHDOG_SECONDS, label: str = "") -> None:
        self.timeout = timeout
        self.label = label
        self.timed_out = False
        self._done = threading.Event()
        self._timer = threading.Timer(timeout, self._on_timeout)
        self._timer.daemon = True

    def _on_timeout(self) -> None:
        if not self._done.is_set():
            self.timed_out = True

    def __enter__(self) -> Watchdog:
        self._timer.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._done.set()
        self._timer.cancel()

    def assert_completed(self, test: unittest.TestCase, action: str) -> None:
        """断言调用在超时前完成。"""
        test.assertFalse(
            self.timed_out,
            f"{action} 超过 {self.timeout:.0f} 秒未返回 —— 极可能是死锁"
            f"（历史：_write_lock 曾被用于建表初始化，导致自我死锁）。"
            f"标签：{self.label}",
        )


class HistoryServiceTestCase(unittest.TestCase):
    """提供隔离的临时数据库。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "history.db"
        self.service = HistoryService(self.db_path)

    def tearDown(self) -> None:
        self.service.close()
        self._tmp.cleanup()

    def guarded(self, action: str, func, *args, **kwargs):
        """带看门狗地调用一个方法。"""
        with Watchdog(label=action) as watchdog:
            result = func(*args, **kwargs)
        watchdog.assert_completed(self, action)
        return result


class TestNoDeadlock(HistoryServiceTestCase):
    """死锁回归测试（本文件最重要的部分）。

    这些调用在修复前会永久阻塞。每个方法都单独测一遍，因为死锁路径取决于
    **哪一个方法先被调用**——先调 ``record()`` 会先完成建表，之后 ``prune()``
    就不再走进建表分支，反而不会卡。所以必须逐方法验证。
    """

    def test_unwritable_directory_does_not_interrupt_translation(self) -> None:
        from unittest import mock

        with mock.patch.object(Path, "mkdir", side_effect=PermissionError("history unavailable")):
            self.assertFalse(self.service.is_available())
            self.assertEqual(self.service.record("Hello", "你好"), 0)
            self.assertEqual(self.service.record_batch([HistoryRecord(target_text="你好")]), 0)
            self.assertEqual(self.service.search(), [])
            self.assertEqual(self.service.count(), 0)
            self.assertEqual(self.service.clear(), 0)
            self.assertEqual(self.service.prune(1), 0)

    def test_prune_on_fresh_service(self) -> None:
        """全新实例上直接调用 prune 不应死锁。

        这是实际崩溃的路径：启动 15 秒后维护定时器首次触发 prune，
        此时数据库尚未建表，于是走进建表分支并死锁。
        """
        self.guarded("prune", self.service.prune, 100)

    def test_record_on_fresh_service(self) -> None:
        """全新实例上直接调用 record 不应死锁。"""
        self.guarded("record", self.service.record, "原文", "译文")

    def test_clear_on_fresh_service(self) -> None:
        """全新实例上直接调用 clear 不应死锁。"""
        self.guarded("clear", self.service.clear)

    def test_count_on_fresh_service(self) -> None:
        """全新实例上直接调用 count 不应死锁。"""
        self.guarded("count", self.service.count)

    def test_search_on_fresh_service(self) -> None:
        """全新实例上直接调用 search 不应死锁。"""
        self.guarded("search", self.service.search, "关键词")

    def test_record_batch_on_fresh_service(self) -> None:
        """全新实例上直接调用 record_batch 不应死锁。"""
        self.guarded("record_batch", self.service.record_batch, [])

    def test_export_on_fresh_service(self) -> None:
        """全新实例上直接调用 export_csv 不应死锁。"""
        self.service.record("a", "b")
        target = Path(self._tmp.name) / "out.csv"
        self.guarded("export_csv", self.service.export_csv, target)

    def test_sequential_full_lifecycle(self) -> None:
        """完整生命周期（建表 → 写 → 查 → 裁剪 → 导出 → 清空）全程不阻塞。"""
        self.guarded("record", self.service.record, "Hello", "你好")
        self.guarded("record", self.service.record, "World", "世界")
        self.assertGreaterEqual(self.service.count(), 1)
        self.guarded("search", self.service.search, "Hello")
        self.guarded("prune", self.service.prune, 1)

        target = Path(self._tmp.name) / "life.csv"
        self.guarded("export_csv", self.service.export_csv, target)
        self.assertTrue(target.exists())

        self.guarded("export_txt", self.service.export_txt, Path(self._tmp.name) / "life.txt")
        self.guarded("clear", self.service.clear)

    def test_repeated_prune_does_not_block(self) -> None:
        """反复调用 prune（模拟维护定时器）不应阻塞。

        第一次调用会建表；后续调用走已初始化分支。两条路径都要验证。
        """
        for index in range(5):
            with self.subTest(call=index):
                self.guarded(f"prune #{index}", self.service.prune, 100)

    def test_concurrent_access_does_not_block(self) -> None:
        """多线程并发写入不应死锁（WAL 允许读写并发）。"""
        errors: list[str] = []
        completed: list[int] = []

        def worker(offset: int) -> None:
            try:
                for index in range(20):
                    self.service.record(f"src-{offset}-{index}", f"dst-{offset}-{index}")
                completed.append(offset)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{type(exc).__name__}: {exc}")
            finally:
                # 每个线程持有自己的连接（threading.local），必须显式关闭，
                # 否则临时目录里的 db 文件会一直被占用而无法删除。
                self.service.close()

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
        for thread in threads:
            thread.start()

        deadline = time.monotonic() + WATCHDOG_SECONDS
        for thread in threads:
            thread.join(timeout=max(0.1, deadline - time.monotonic()))

        alive = [t for t in threads if t.is_alive()]
        self.assertFalse(alive, "并发写入有线程未在超时前结束 —— 可能是死锁")
        self.assertEqual(errors, [])
        self.assertEqual(sorted(completed), [0, 1, 2])


class TestRecording(HistoryServiceTestCase):
    """记录写入。"""

    def test_record_and_read_back(self) -> None:
        """写入后应能读回。"""
        self.service.record("Hello world", "你好世界", source_lang="en", target_lang="zh")
        records = self.service.recent(limit=10)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].source_text, "Hello world")
        self.assertEqual(records[0].target_text, "你好世界")

    def test_store_source_can_be_disabled(self) -> None:
        """关闭记录原文时应只保留译文。

        原文来自屏幕，可能比译文更敏感（内部系统路径、账号名等），
        因此必须支持只存译文。
        """
        self.service.record("敏感原文", "译文", store_source=False)
        records = self.service.recent(limit=10)
        self.assertEqual(records[0].source_text, "")
        self.assertEqual(records[0].target_text, "译文")

    def test_empty_target_ignored(self) -> None:
        """空译文不应产生记录。"""
        self.assertEqual(self.service.record("原文", ""), 0)
        self.assertEqual(self.service.count(), 0)

    def test_disabled_service_ignores_writes(self) -> None:
        """禁用时不写入。"""
        service = HistoryService(self.db_path, enabled=False)
        try:
            self.assertEqual(service.record("a", "b"), 0)
            self.assertEqual(service.count(), 0)
        finally:
            service.close()

    def test_batch_record(self) -> None:
        """批量写入。"""
        records = [
            HistoryRecord(source_text=f"src{i}", target_text=f"dst{i}", model_id="m")
            for i in range(5)
        ]
        self.assertEqual(self.service.record_batch(records), 5)
        self.assertEqual(self.service.count(), 5)

    def test_batch_skips_empty_targets(self) -> None:
        """批量写入时跳过空译文。"""
        records = [
            HistoryRecord(source_text="a", target_text=""),
            HistoryRecord(source_text="b", target_text="有效"),
        ]
        self.assertEqual(self.service.record_batch(records), 1)


class TestSearchAndExport(HistoryServiceTestCase):
    """检索与导出。"""

    def setUp(self) -> None:
        super().setUp()
        for index in range(6):
            self.service.record(f"keyword-{index}", f"译文-{index}")

    def test_search_by_keyword(self) -> None:
        """按关键词检索原文与译文。"""
        self.assertEqual(len(self.service.search("keyword-3")), 1)
        self.assertEqual(len(self.service.search("译文-3")), 1)

    def test_search_no_match(self) -> None:
        """无匹配时返回空列表。"""
        self.assertEqual(self.service.search("绝对不存在的词"), [])

    def test_limit_and_offset(self) -> None:
        """分页参数生效。"""
        first = self.service.search(limit=2)
        self.assertEqual(len(first), 2)
        second = self.service.search(limit=2, offset=2)
        self.assertEqual(len(second), 2)
        self.assertNotEqual(first[0].id, second[0].id)

    def test_export_csv(self) -> None:
        """导出 CSV 应包含全部记录与表头。"""
        target = Path(self._tmp.name) / "export.csv"
        count = self.service.export_csv(target)
        self.assertEqual(count, 6)

        content = target.read_text(encoding="utf-8-sig")
        self.assertIn("时间", content)
        self.assertIn("keyword-1", content)

    def test_export_txt(self) -> None:
        """导出纯文本。"""
        target = Path(self._tmp.name) / "export.txt"
        self.assertEqual(self.service.export_txt(target), 6)
        self.assertIn("原文：", target.read_text(encoding="utf-8"))

    def test_export_empty_returns_zero(self) -> None:
        """无记录时导出返回 0 而不是创建空文件。"""
        empty = HistoryService(Path(self._tmp.name) / "empty.db")
        try:
            target = Path(self._tmp.name) / "none.csv"
            self.assertEqual(empty.export_csv(target), 0)
        finally:
            empty.close()


class TestLiteralKeyword(HistoryServiceTestCase):
    """Search and export use literal text, including LIKE metacharacters."""

    def test_source_and_target_keywords_are_literal(self) -> None:
        cases = [
            ("item_1", "item_1", "itemA1", False),
            ("10%", "10%", "100", False),
            (r"C:\logs\trace", r"C:\logs\trace", "C:logs:trace", False),
            (r"folder\100%_done", r"folder\100%_done", r"folder\100XAdone", False),
            ("target_1", "target_1", "targetA1", True),
        ]
        for keyword, matching, unrelated, in_target in cases:
            with self.subTest(keyword=keyword, in_target=in_target):
                self.service.clear()
                if in_target:
                    expected = self.service.record("source-a", matching)
                    self.service.record("source-b", unrelated)
                else:
                    expected = self.service.record(matching, "译文-a")
                    self.service.record(unrelated, "译文-b")
                self.assertEqual([record.id for record in self.service.search(keyword)], [expected])
                self.assertEqual(self.service.count(keyword), 1)

    def test_exports_have_the_same_literal_keyword_scope(self) -> None:
        self.service.record("file_10%.txt", "保留此记录")
        self.service.record("fileA100.txt", "排除此记录")
        keyword = "file_10%"
        self.assertEqual(self.service.count(keyword), 1)
        for suffix, exporter in (("csv", self.service.export_csv), ("txt", self.service.export_txt)):
            with self.subTest(format=suffix):
                target = Path(self._tmp.name) / f"filtered.{suffix}"
                self.assertEqual(exporter(target, keyword=keyword), 1)
                content = target.read_text(encoding="utf-8-sig")
                self.assertIn("file_10%.txt", content)
                self.assertNotIn("fileA100.txt", content)

    def test_literal_search_preserves_pagination_and_time_filter(self) -> None:
        now = datetime.now()
        records = [
            HistoryRecord(
                source_text=f"file_%_{index}",
                target_text=f"译文-{index}",
                created_at=(now - timedelta(days=days)).isoformat(timespec="seconds"),
            )
            for index, days in ((0, 30), (1, 2), (2, 1))
        ]
        records.append(HistoryRecord(source_text="fileAA_3", target_text="排除"))
        self.service.record_batch(records)
        matches = self.service.search("file_%", days=7, limit=1, offset=1)
        self.assertEqual([record.source_text for record in matches], ["file_%_1"])
        target = Path(self._tmp.name) / "recent.txt"
        self.assertEqual(self.service.export_txt(target, keyword="file_%", days=7), 2)
        self.assertNotIn("file_%_0", target.read_text(encoding="utf-8"))


class TestMaintenance(HistoryServiceTestCase):
    """清空与裁剪。"""

    def test_prune_keeps_newest(self) -> None:
        """裁剪应保留最新记录。"""
        for index in range(10):
            self.service.record(f"src-{index}", f"dst-{index}")
        self.service.prune(3)

        self.assertEqual(self.service.count(), 3)
        remaining = {r.source_text for r in self.service.search(limit=10)}
        self.assertIn("src-9", remaining)
        self.assertNotIn("src-0", remaining)

    def test_prune_noop_when_under_limit(self) -> None:
        """未超上限时不删除。"""
        self.service.record("a", "b")
        self.assertEqual(self.service.prune(100), 0)

    def test_prune_zero_means_unlimited(self) -> None:
        """上限为 0 表示不限制，不做任何删除。"""
        self.service.record("a", "b")
        self.assertEqual(self.service.prune(0), 0)
        self.assertEqual(self.service.count(), 1)

    def test_clear_removes_all(self) -> None:
        """清空全部记录。"""
        for index in range(4):
            self.service.record(f"s{index}", f"t{index}")
        self.assertEqual(self.service.clear(), 4)
        self.assertEqual(self.service.count(), 0)

    def test_database_size_reported(self) -> None:
        """应能报告数据库文件大小（供设置面板展示）。"""
        self.service.record("a", "b")
        self.assertGreater(self.service.database_size_mb(), 0.0)


class TestRecordDisplay(unittest.TestCase):
    """记录对象的展示属性。"""

    def test_created_at_display(self) -> None:
        """时间应被格式化为可读字符串。"""
        record = HistoryRecord(created_at=datetime(2026, 9, 18, 20, 30, 0).isoformat())
        self.assertEqual(record.created_at_display, "2026-09-18 20:30:00")

    def test_invalid_created_at_falls_back(self) -> None:
        """时间格式非法时原样返回而不是抛异常。"""
        record = HistoryRecord(created_at="不是时间")
        self.assertEqual(record.created_at_display, "不是时间")

    def test_preview_truncates(self) -> None:
        """长译文预览应被截断。"""
        record = HistoryRecord(target_text="长" * 100)
        self.assertLessEqual(len(record.preview), 61)
        self.assertTrue(record.preview.endswith("…"))

    def test_preview_short_text_unchanged(self) -> None:
        """短译文预览不截断。"""
        record = HistoryRecord(target_text="短文本")
        self.assertEqual(record.preview, "短文本")

    def test_preview_strips_newlines(self) -> None:
        """预览应把换行替换为空格，避免列表行高跳动。"""
        record = HistoryRecord(target_text="第一行\n第二行")
        self.assertNotIn("\n", record.preview)


if __name__ == "__main__":
    unittest.main(verbosity=2)
