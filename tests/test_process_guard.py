"""子进程生命周期守卫测试。

**这个文件守的是一条很贵的教训**

``llama-server`` 是程序拉起的独立进程，每个占 1–6 GB 内存，还要在显存里
分配权重与计算缓冲区。程序被强制结束（任务管理器"结束任务"、崩溃、断电）
时清理代码没有机会执行，它们就变成孤儿。实测攒到过 **8 个孤儿、23.3 GB**：
机器内存占用 89%，显存被挤到使用共享内存（走 PCIe），模型推理慢到超时，
界面上一个字都翻不出来——而 GPU 利用率显示 99%，看上去"在忙"。

因此这里验证的是**内核级保障**：父进程无论怎么死，子进程都会被一起终止。
效果不能靠"代码看起来对"来判断，必须真的验证一次。
"""

from __future__ import annotations

import contextlib
import ctypes
import subprocess
import sys
import time
import unittest
from ctypes import wintypes
from unittest.mock import MagicMock, patch

from pingyitong.utils import process_guard

_WINDOWS = sys.platform == "win32"


class _ChildProcess:
    """用于测试的长睡眠子进程。"""

    def __init__(self) -> None:
        self.process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])

    @property
    def pid(self) -> int:
        return self.process.pid

    def is_alive(self) -> bool:
        return self.process.poll() is None

    def cleanup(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
            with contextlib.suppress(Exception):
                self.process.wait(timeout=5)


@unittest.skipUnless(_WINDOWS, "作业对象是 Windows 机制")
class TestKillOnCloseJob(unittest.TestCase):
    """作业对象：父进程消失时子进程必须一并被终止。"""

    def setUp(self) -> None:
        process_guard.reset_for_tests()
        self._closed = False

    def tearDown(self) -> None:
        if not self._closed:
            self._close_job()
        process_guard.reset_for_tests()

    def _close_job(self) -> None:
        """关闭作业对象句柄，模拟父进程消失（内核会终止其中所有进程）。"""
        handle = process_guard._job_handle
        if handle:
            process_guard._kernel32().CloseHandle(handle)
        self._closed = True

    def test_job_can_be_created(self) -> None:
        """应当能创建作业对象。创建失败会退化为"没有这层保护"，必须能察觉。"""
        self.assertIsNotNone(
            process_guard.create_kill_on_close_job(),
            "作业对象创建失败，子进程将无法随程序退出而终止",
        )

    def test_job_creation_is_idempotent(self) -> None:
        """重复调用返回同一个句柄，不应反复创建。"""
        first = process_guard.create_kill_on_close_job()
        second = process_guard.create_kill_on_close_job()
        self.assertEqual(first, second)

    def test_assigned_child_is_killed_when_job_closes(self) -> None:
        """把子进程加入作业后，作业关闭时子进程必须被终止。

        这是整个防护的核心：子进程不能比父进程活得久。
        """
        self.assertIsNotNone(process_guard.create_kill_on_close_job())

        child = _ChildProcess()
        try:
            self.assertTrue(process_guard.assign_to_job(child.pid), "子进程加入作业失败")
            time.sleep(0.5)
            self.assertTrue(child.is_alive(), "加入作业不应立即杀死子进程")

            self._close_job()

            for _ in range(20):
                time.sleep(0.25)
                if not child.is_alive():
                    break
            self.assertFalse(
                child.is_alive(),
                "作业关闭后子进程仍然存活——防护未生效，会重新产生孤儿进程",
            )
        finally:
            child.cleanup()

    def test_unassigned_child_survives_job_close(self) -> None:
        """没有加入作业的进程不受影响（不能误伤无关进程）。"""
        self.assertIsNotNone(process_guard.create_kill_on_close_job())

        child = _ChildProcess()
        try:
            time.sleep(0.3)
            self._close_job()
            time.sleep(0.5)
            self.assertTrue(
                child.is_alive(),
                "未加入作业的进程被误杀——防护范围过宽",
            )
        finally:
            child.cleanup()

    def test_assign_failure_is_not_fatal(self) -> None:
        """加入作业失败时返回 False，且不得抛异常。

        某些环境不允许嵌套作业，此时程序必须照常运行，
        只是失去这层保护（调用方会记日志）。
        """
        process_guard.create_kill_on_close_job()
        # 用一个几乎不可能存在的 PID
        self.assertFalse(process_guard.assign_to_job(0x7FFFFFF0))


class TestReapStaleServers(unittest.TestCase):
    """启动时清理孤儿进程：必须精准，不能误伤。"""

    def test_missing_executable_yields_nothing(self) -> None:
        """路径不存在时不报错、不清理任何东西。"""
        self.assertEqual(
            process_guard.reap_stale_servers("Z:/不存在的目录/llama-server.exe"), []
        )

    def test_does_not_touch_unrelated_processes(self) -> None:
        """同名但不在本程序运行时目录的进程不得被清理。

        用户可能自己也在用 llama.cpp。只有完整应用归属记录与进程身份一致，
        且启动者已经消失时才可以清理。
        """
        child = _ChildProcess()
        try:
            time.sleep(0.3)
            killed = process_guard.reap_stale_servers("Z:/本程序的运行时目录/llama-server.exe")
            self.assertEqual(killed, [])
            self.assertTrue(child.is_alive(), "无关进程被误杀")
        finally:
            child.cleanup()

    def test_only_matches_server_image_name(self) -> None:
        """判据里的进程名必须是推理后端，不能是别的程序。"""
        self.assertEqual(process_guard.SERVER_IMAGE_NAME, "llama-server.exe")


class TestJobObjectAbi(unittest.TestCase):
    """64 位句柄不能按 ctypes 默认 int 传递，创建失败不能泄漏句柄。"""

    def tearDown(self) -> None:
        process_guard.reset_for_tests()

    def test_kernel32_declares_pointer_sized_handles(self) -> None:
        kernel32 = MagicMock()
        with patch.object(ctypes, "WinDLL", return_value=kernel32, create=True):
            process_guard._kernel32()
        self.assertIs(kernel32.CreateJobObjectW.restype, wintypes.HANDLE)
        self.assertIs(kernel32.OpenProcess.restype, wintypes.HANDLE)
        self.assertEqual(
            kernel32.AssignProcessToJobObject.argtypes, [wintypes.HANDLE, wintypes.HANDLE]
        )
        self.assertEqual(kernel32.CloseHandle.argtypes, [wintypes.HANDLE])

    def test_limit_configuration_failure_closes_created_handle(self) -> None:
        process_guard.reset_for_tests()
        kernel32 = MagicMock()
        handle = 0x1234567887654321
        kernel32.CreateJobObjectW.return_value = handle
        kernel32.SetInformationJobObject.return_value = False
        with (
            patch.object(process_guard.sys, "platform", "win32"),
            patch.object(process_guard, "_kernel32", return_value=kernel32),
        ):
            self.assertIsNone(process_guard.create_kill_on_close_job())
            self.assertIsNone(process_guard.create_kill_on_close_job())
        kernel32.CloseHandle.assert_called_once_with(handle)
        kernel32.CreateJobObjectW.assert_called_once()


if __name__ == "__main__":
    unittest.main()
