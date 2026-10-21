"""Release-audit regression: rapid stop/start reset consistency + capture_failed shutdown.

接线纪律（r0005 教训，勿回退）：
生产中 app.py 把 capture_failed 连到主线程槽，AutoConnection → queued，
stop() 永远在主线程执行。测试必须复刻该接线：setUpClass 先建
QApplication，等待循环用 processEvents 驱动 queued 信号。禁止
DirectConnection 把 stop() 拉到捕获线程——stop() 内含 thread.join() 与
QTimer 操作，join(自己) 抛的 RuntimeError 穿过 PyQt6 C++ 信号边界会
qFatal → abort → 0xC0000409（进程级崩溃，1419+ 测试全灭）。
"""
from __future__ import annotations

import threading
import time
import unittest

from tests.test_pipeline_lifecycle import _ensure_app, _make_pipeline

try:
    from PyQt6.QtCore import QCoreApplication

    PYQT_AVAILABLE = True
except ImportError:  # pragma: no cover
    PYQT_AVAILABLE = False


@unittest.skipUnless(PYQT_AVAILABLE, "需要 PyQt6")
class TestRapidToggleReset(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # 必须把 QApplication 存到类属性上：返回值一旦无人引用，
        # PyQt 会销毁 C++ 对象与事件分发器，processEvents() 就成了
        # 无环可转的空操作，queued 信号永远无法送达（r0005 实测）。
        cls.app = _ensure_app()
        if cls.app is None:
            raise unittest.SkipTest("无法创建 QApplication")

    def test_rapid_stop_start_cycles_keep_pipeline_functional(self) -> None:
        """快速 stop->start 切换：重置在 _differ_lock 内完成，状态一致可继续。"""
        pipeline, translator, overlay = _make_pipeline()
        for _ in range(5):
            pipeline.stop()
            self.assertTrue(pipeline.start())
        # 状态一致性：新一代计数已推进，重试/未解析区清空
        self.assertGreaterEqual(pipeline._retry_generation, 5)
        self.assertEqual(pipeline._pending_retry_regions, [])
        self.assertEqual(pipeline._unresolved_regions, [])
        self.assertEqual(pipeline._retry_counts, {})
        self.assertTrue(pipeline._status.running)
        pipeline.stop()


@unittest.skipUnless(PYQT_AVAILABLE, "需要 PyQt6")
class TestCaptureFailedShutdown(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # 必须把 QApplication 存到类属性上：返回值一旦无人引用，
        # PyQt 会销毁 C++ 对象与事件分发器，processEvents() 就成了
        # 无环可转的空操作，queued 信号永远无法送达（r0005 实测）。
        cls.app = _ensure_app()
        if cls.app is None:
            raise unittest.SkipTest("无法创建 QApplication")

    def test_repeated_capture_failures_emit_capture_failed(self) -> None:
        """捕获连续失败：capture_failed 发出并在主线程触发整条停机。"""
        received = {"capture_failed": False, "error": ""}
        pipeline, translator, overlay = _make_pipeline()
        pipeline.capture_failed.connect(lambda: received.__setitem__("capture_failed", True))
        pipeline.error_occurred.connect(lambda msg: received.__setitem__("error", msg))

        def _boom(*a, **k):
            raise RuntimeError("boom")

        class _ExplodingCapturer:
            last_capture_cost = 0.0

            def open(self) -> None:
                return None

            def close(self) -> None:
                return None

            capture_region = _boom
            capture_monitor = _boom
            capture_all = _boom

        pipeline._capturer = _ExplodingCapturer()

        stopped = threading.Event()
        original_stop = pipeline.stop

        def stopping_stop(*a, **k):
            stopped.set()
            return original_stop(*a, **k)

        pipeline.stop = stopping_stop
        # 与 app.py 一致：不指定连接类型。发射线程=捕获线程、接收方活在
        # 主线程 → AutoConnection 走 queued，槽在主线程执行。
        pipeline.capture_failed.connect(stopping_stop)

        pipeline.start()
        # queued 信号需要事件循环分发：主线程边等边驱动。
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            QCoreApplication.processEvents()
            if stopped.is_set() and received["capture_failed"]:
                break
            time.sleep(0.02)
        self.assertTrue(received["capture_failed"], "capture_failed 未在 15s 内发出")
        self.assertTrue(stopped.is_set(), "capture_failed 未触发停机")
        self.assertIn("连续失败", received["error"])
        # 恢复真实 stop 并收尾（此时已停机，幂等早退）
        pipeline.stop = original_stop
        pipeline.stop()


if __name__ == "__main__":
    unittest.main()
