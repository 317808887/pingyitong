"""安装自检测试。

背景（见架构说明 6.13）：只验证 import 会漏掉"ONNX 模型文件没打进去"
这类打包问题——那种情况下导入成功、``is_ready()`` 返回 True，
但一识别就报错。``--self-check`` 对合成图像做真实推理，本模块固定
自检的**报告组装逻辑**（标记、失败收集、异常兜底、弹窗判定），
不依赖任何真实硬件与模型。
"""

from __future__ import annotations

import io
import unittest
from unittest import mock

from pingyitong import selfcheck


def _fake_check(ok: bool, detail: str = "ok"):
    """构造一个返回固定结论的检查函数。"""
    def check() -> tuple[bool, str]:
        return ok, detail
    return check


def _checks(*items: tuple[str, bool, bool, str]):
    """由 ``(名称, 是否通过, 是否必过, 详情)`` 构造检查项元组。"""
    return tuple(
        (label, _fake_check(ok, detail), required)
        for label, ok, required, detail in items
    )


class TestCollectReport(unittest.TestCase):
    """报告组装逻辑。"""

    def test_all_pass_no_failures(self) -> None:
        checks = _checks(
            ("屏幕捕获", True, True, "可用"),
            ("文字识别", True, True, "可用"),
            ("数据目录", True, False, "可写"),
        )
        with mock.patch.object(selfcheck, "_collect_report_checks", return_value=checks):
            lines, failures = selfcheck._collect_report()
        self.assertEqual(failures, [])
        self.assertTrue(any("自检通过" in line for line in lines))
        self.assertTrue(any("✓ 屏幕捕获：可用" in line for line in lines))

    def test_required_failure_collected_with_fail_mark(self) -> None:
        """必检项失败：计入 failures、标 ✗、结论为未通过。"""
        checks = _checks(
            ("屏幕捕获", True, True, "可用"),
            ("文字识别", False, True, "RapidOCR 未就绪"),
            ("数据目录", False, False, "不可写"),  # 非必检：只警告
        )
        with mock.patch.object(selfcheck, "_collect_report_checks", return_value=checks):
            lines, failures = selfcheck._collect_report()
        self.assertEqual(failures, ["文字识别"])
        self.assertTrue(any("✗ 文字识别：RapidOCR 未就绪" in line for line in lines))
        # 非必检失败用 "!" 标记，不计入失败
        self.assertTrue(any("! 数据目录：不可写" in line for line in lines))
        self.assertTrue(any("自检未通过：文字识别" in line for line in lines))

    def test_check_exception_does_not_crash(self) -> None:
        """单项检查抛异常不得让自检崩溃，按失败项处理。"""
        def boom() -> tuple[bool, str]:
            raise RuntimeError("意外")

        checks = (("屏幕捕获", boom, True),)
        with mock.patch.object(selfcheck, "_collect_report_checks", return_value=checks):
            lines, failures = selfcheck._collect_report()
        self.assertEqual(failures, ["屏幕捕获"])
        self.assertTrue(any("自检异常：RuntimeError" in line for line in lines))


class TestWindowedDetection(unittest.TestCase):
    """打包后的窗口子系统（无控制台）判定。"""

    def test_frozen_without_stdout_is_windowed(self) -> None:
        with mock.patch.object(selfcheck.sys, "frozen", True, create=True), \
             mock.patch.object(selfcheck.sys, "stdout", None):
            self.assertTrue(selfcheck._is_windowed())

    def test_console_process_is_not_windowed(self) -> None:
        with mock.patch.object(selfcheck.sys, "frozen", True, create=True), \
             mock.patch.object(selfcheck.sys, "stdout", io.StringIO()):
            self.assertFalse(selfcheck._is_windowed())

    def test_dev_process_without_frozen_is_not_windowed(self) -> None:
        with mock.patch.object(selfcheck.sys, "frozen", False, create=True), \
             mock.patch.object(selfcheck.sys, "stdout", None):
            self.assertFalse(selfcheck._is_windowed())


if __name__ == "__main__":
    unittest.main(verbosity=2)
