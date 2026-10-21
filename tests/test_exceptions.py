"""全局异常钩子测试。

**为什么需要这个测试文件**

PyQt6 对"Python 重写的槽/虚函数里抛出未捕获异常"的处理是调用 ``qFatal()``：
进程立即终止，退出码 ``0xC0000409``，**不打印堆栈、不留日志、界面无提示**。
项目因此出现过两次"用户点一下按钮程序就消失"的问题，排查时几乎没有线索。

装上 :func:`pingyitong.utils.install_exception_hooks` 之后，PyQt6 会改为调用
``sys.excepthook``，程序得以存活并报告问题。这些测试守住这条防线：

- 事件处理器里抛异常时，进程必须活下来，且异常被记录、被上报界面；
- 后台线程里抛异常同样要被记录（否则功能会静默失效）；
- 钩子不能把 ``KeyboardInterrupt`` 一起吞掉。

未安装 PyQt6 时整个模块跳过。
"""

from __future__ import annotations

import sys
import threading
import unittest

try:
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication, QWidget

    PYQT_AVAILABLE = True
except ImportError:  # pragma: no cover - 无 PyQt6 环境
    PYQT_AVAILABLE = False


@unittest.skipUnless(PYQT_AVAILABLE, "需要 PyQt6")
class TestUncaughtExceptionHandling(unittest.TestCase):
    """未捕获异常的存活与上报。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance()
        if cls.app is None:
            try:
                cls.app = QApplication([])
            except Exception:  # noqa: BLE001
                raise unittest.SkipTest("无法创建 QApplication") from None

    def setUp(self) -> None:
        from pingyitong.utils import exceptions

        exceptions.reset_for_tests()
        self.reports: list[tuple[str, str, str]] = []

    def tearDown(self) -> None:
        from pingyitong.utils import exceptions

        exceptions.reset_for_tests()

    def test_install_sets_both_hooks(self) -> None:
        """安装后主线程与后台线程钩子都必须被替换。"""
        from pingyitong.utils import install_exception_hooks

        default_main = sys.excepthook
        default_thread = threading.excepthook

        install_exception_hooks()
        try:
            self.assertIsNot(sys.excepthook, default_main, "sys.excepthook 未被替换")
            self.assertIsNot(
                threading.excepthook, default_thread, "threading.excepthook 未被替换"
            )
        finally:
            sys.excepthook = default_main
            threading.excepthook = default_thread

    def test_exception_in_paint_does_not_kill_process(self) -> None:
        """绘制回调抛异常后进程必须存活。

        这是最贴近真实崩溃场景的一条：用户看到"一点框选程序就没了"，
        正是 ``paintEvent`` 抛 ``AttributeError`` 触发了 ``qFatal()``。
        装上钩子后，异常应当被交给 excepthook，程序继续跑完事件循环。
        """
        from pingyitong.utils import install_exception_hooks, report_count

        install_exception_hooks(lambda t, d, k: self.reports.append((t, d, k)))

        class Boom(QWidget):
            def paintEvent(self, event) -> None:  # noqa: N802
                raise AttributeError("模拟绘制代码里的笔误")

        widget = Boom()
        survived = {"value": False}
        try:
            from PyQt6.QtGui import QPixmap

            widget.resize(120, 60)
            # render() 会同步派发 paintEvent，不依赖窗口是否可见
            widget.render(QPixmap(120, 60))
            survived["value"] = True
        finally:
            widget.deleteLater()

        self.assertTrue(survived["value"], "绘制抛异常后不应终止进程")
        self.assertGreaterEqual(report_count(), 1, "异常必须被计数上报")
        self.assertTrue(self.reports, "异常必须被上报到界面")
        title, description, kind = self.reports[0]
        self.assertEqual(kind, "error")
        self.assertIn("AttributeError", description)

    def test_exception_inside_timer_slot_keeps_loop_running(self) -> None:
        """定时器槽里抛异常后事件循环仍应继续处理后续事件。

        模拟"界面回调出错，程序却照常可用"这一期望行为。
        """
        from pingyitong.utils import install_exception_hooks

        install_exception_hooks(lambda t, d, k: self.reports.append((t, d, k)))

        ticks: list[int] = []

        def bad_slot() -> None:
            ticks.append(-1)
            raise RuntimeError("模拟槽函数出错")

        def good_slot() -> None:
            ticks.append(1)
            self.app.quit()

        QTimer.singleShot(10, bad_slot)
        QTimer.singleShot(60, good_slot)
        # 其他测试留下的 deleteLater 窗口可能在这里才被销毁；测试只需
        # 验证槽异常后的事件处理，不应因最后一个窗口关闭而提前退循环。
        quit_on_close = self.app.quitOnLastWindowClosed()
        self.app.setQuitOnLastWindowClosed(False)
        try:
            self.app.exec()
        finally:
            self.app.setQuitOnLastWindowClosed(quit_on_close)

        self.assertIn(-1, ticks, "出错的槽应当被调用过")
        self.assertIn(1, ticks, "出错之后事件循环应当继续运行")
        self.assertTrue(self.reports, "异常应当被上报")

    def test_thread_exception_is_reported(self) -> None:
        """后台线程抛异常必须被记录。

        否则模型加载线程之类的后台任务挂掉后，界面会永远停在"加载中"，
        而用户得不到任何解释。
        """
        from pingyitong.utils import install_exception_hooks, report_count

        install_exception_hooks(lambda t, d, k: self.reports.append((t, d, k)))

        def worker() -> None:
            raise ValueError("模拟后台线程错误")

        thread = threading.Thread(target=worker, name="测试线程")
        thread.start()
        thread.join(timeout=5)

        self.assertGreaterEqual(report_count(), 1, "线程异常必须被计数上报")
        self.assertTrue(self.reports)
        self.assertIn("测试线程", self.reports[0][1])

    def test_keyboard_interrupt_is_not_swallowed(self) -> None:
        """``Ctrl+C`` 必须仍能中断程序，不能被钩子悄悄吞掉。"""
        from pingyitong.utils import install_exception_hooks

        install_exception_hooks(lambda t, d, k: self.reports.append((t, d, k)))

        try:
            raise KeyboardInterrupt()
        except KeyboardInterrupt:
            # 钩子只处理非 KeyboardInterrupt；这里直接调用钩子验证其分支
            sys.excepthook(*sys.exc_info())

        self.assertEqual(self.reports, [], "KeyboardInterrupt 不应被当作错误上报")

    def test_reporter_failure_does_not_recurse(self) -> None:
        """上报回调自身出错时不能再次抛异常（否则会无限递归）。"""

        def broken_reporter(title: str, description: str, kind: str) -> None:
            raise RuntimeError("提示弹不出来")

        from pingyitong.utils import install_exception_hooks, report_count

        install_exception_hooks(broken_reporter)

        try:
            raise RuntimeError("第一个错误")
        except RuntimeError:
            sys.excepthook(*sys.exc_info())  # 不应抛出

        self.assertGreaterEqual(report_count(), 1)


if __name__ == "__main__":
    unittest.main()
