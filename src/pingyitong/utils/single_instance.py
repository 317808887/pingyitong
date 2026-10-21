"""单实例控制。

通过 Windows 命名互斥体（Named Mutex）确保同一时间只运行一个实例。
用户重复启动时，不新开进程，而是唤醒已有实例打开主面板。

该实现只使用本机内核对象，不涉及任何网络通信，符合全离线约束。
"""

from __future__ import annotations

import ctypes
import sys
import threading
from collections.abc import Callable
from ctypes import wintypes

from .logger import get_logger

logger = get_logger(__name__)

_ERROR_ALREADY_EXISTS = 183
_WAIT_OBJECT_0 = 0
_INFINITE = 0xFFFFFFFF

_IS_WINDOWS = sys.platform == "win32"


class SingleInstance:
    """基于命名互斥体的单实例守卫。

    用法::

        guard = SingleInstance("PingYiTong")
        if not guard.acquire():
            guard.notify_existing()   # 唤醒已有实例后退出
            raise SystemExit(0)
        guard.start_listener(lambda: print("已有实例被再次启动"))
    """

    def __init__(self, name: str) -> None:
        self._mutex_name = f"Global\\{name}.Mutex"
        self._event_name = f"Global\\{name}.ShowEvent"
        self._mutex: int | None = None
        self._event: int | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def acquire(self) -> bool:
        """尝试成为主实例。

        :return: 成功取得所有权返回 ``True``；已有实例在运行返回 ``False``。
        """
        if not _IS_WINDOWS:
            # 非 Windows 平台不做限制，功能开发以 Windows 为目标平台。
            return True

        kernel32 = ctypes.windll.kernel32
        kernel32.CreateMutexW.restype = wintypes.HANDLE
        handle = kernel32.CreateMutexW(None, True, self._mutex_name)
        last_error = kernel32.GetLastError()

        if not handle:
            logger.warning("创建互斥体失败，跳过单实例检查")
            return True

        if last_error == _ERROR_ALREADY_EXISTS:
            kernel32.CloseHandle(handle)
            return False

        self._mutex = handle
        return True

    def notify_existing(self) -> None:
        """唤醒正在运行的主实例（用于第二次启动时前置窗口）。"""
        if not _IS_WINDOWS:
            return
        kernel32 = ctypes.windll.kernel32
        kernel32.OpenEventW.restype = wintypes.HANDLE
        handle = kernel32.OpenEventW(0x0002, False, self._event_name)  # EVENT_MODIFY_STATE
        if handle:
            kernel32.SetEvent(handle)
            kernel32.CloseHandle(handle)

    def start_listener(self, callback: Callable[[], None]) -> None:
        """启动后台线程，等待其他实例发来的唤醒信号。

        :param callback: 收到信号时调用的回调，将在后台线程中执行。
            调用方若需操作界面，应通过 Qt 信号转发至主线程。
        """
        if not _IS_WINDOWS:
            return

        kernel32 = ctypes.windll.kernel32
        kernel32.CreateEventW.restype = wintypes.HANDLE
        self._event = kernel32.CreateEventW(None, False, False, self._event_name)

        def _loop() -> None:
            while not self._stop.is_set():
                result = kernel32.WaitForSingleObject(self._event, 500)
                if result == _WAIT_OBJECT_0:
                    try:
                        callback()
                    except Exception:  # noqa: BLE001 - 回调异常不得终止监听线程
                        logger.exception("单实例唤醒回调执行失败")

        self._thread = threading.Thread(
            target=_loop, name="SingleInstanceListener", daemon=True
        )
        self._thread.start()

    def release(self) -> None:
        """释放内核对象。进程退出时操作系统会自动回收，此处用于显式清理。"""
        self._stop.set()
        if not _IS_WINDOWS:
            return
        kernel32 = ctypes.windll.kernel32
        if self._event:
            kernel32.CloseHandle(self._event)
            self._event = None
        if self._mutex:
            kernel32.ReleaseMutex(self._mutex)
            kernel32.CloseHandle(self._mutex)
            self._mutex = None
