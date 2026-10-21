"""Keep the translator's own windows out of the OCR capture input."""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from functools import lru_cache

from PyQt6.QtCore import QEvent, QObject, pyqtSignal
from PyQt6.QtWidgets import QWidget

from .logger import get_logger

logger = get_logger(__name__)
_EXCLUDE_FROM_CAPTURE = 0x11


@lru_cache(maxsize=1)
def _window_api():
    api = ctypes.WinDLL("user32", use_last_error=True)
    api.SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]
    api.SetWindowDisplayAffinity.restype = wintypes.BOOL
    api.GetWindowDisplayAffinity.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    api.GetWindowDisplayAffinity.restype = wintypes.BOOL
    api.IsWindow.argtypes = [wintypes.HWND]
    api.IsWindow.restype = wintypes.BOOL
    return api


def exclude_window_from_capture(hwnd: int) -> bool:
    """Exclude a top-level window owned by this process, preserving its display.

    WDA_EXCLUDEFROMCAPTURE requires Windows 10 build 19041 or newer. Do not
    use its older black-window fallback: OCR must see the original below it.
    """
    if sys.platform != "win32" or not hwnd or sys.getwindowsversion().build < 19041:
        return False
    try:
        api = _window_api()
        if not api.IsWindow(hwnd):
            return False
        affinity = wintypes.DWORD()
        if api.GetWindowDisplayAffinity(hwnd, ctypes.byref(affinity)) and affinity.value == _EXCLUDE_FROM_CAPTURE:
            return True
        if not api.SetWindowDisplayAffinity(hwnd, _EXCLUDE_FROM_CAPTURE):
            logger.warning("无法排除本程序窗口的捕获（Windows 错误 %s）", ctypes.get_last_error())
            return False
        return bool(api.GetWindowDisplayAffinity(hwnd, ctypes.byref(affinity))) and affinity.value == _EXCLUDE_FROM_CAPTURE
    except (AttributeError, OSError):
        logger.debug("当前窗口后端不支持捕获排除", exc_info=True)
        return False


class CaptureExclusionFilter(QObject):
    """Apply exclusion when an app window opens or its native handle changes."""

    windows_changed = pyqtSignal()

    def __init__(self, parent: QObject) -> None:
        super().__init__(parent)
        self._applying: set[int] = set()
        parent.installEventFilter(self)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() not in (
            QEvent.Type.Show, QEvent.Type.WinIdChange, QEvent.Type.Hide,
            QEvent.Type.Move, QEvent.Type.Resize,
        ):
            return False
        if not isinstance(watched, QWidget) or not watched.isWindow():
            return False
        if not type(watched).__module__.startswith("pingyitong."):
            return False
        identity = id(watched)
        if identity in self._applying:
            return False
        if event.type() in (QEvent.Type.Show, QEvent.Type.WinIdChange):
            self._applying.add(identity)
            try:
                exclude_window_from_capture(int(watched.winId()))
            finally:
                self._applying.discard(identity)
        # Overlay geometry is handled by OverlayManager itself. Interactive
        # windows must also exclude their visible area from overlay painting.
        if not type(watched).__module__.startswith("pingyitong.components.render."):
            self.windows_changed.emit()
        return False
