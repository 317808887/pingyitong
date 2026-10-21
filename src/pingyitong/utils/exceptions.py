"""全局异常钩子。

**为什么必须有这个模块**

PyQt6 对"Python 重写的虚函数/槽里抛出未捕获异常"的处理方式是调用
``qFatal()``：进程立刻终止，**不打印堆栈、不给日志留痕、界面不提示**。
实测退出码为 ``0xC0000409``。

对本项目而言这是致命的：用户看到的只是"点了一下某个按钮，程序就没了"，
既不知道发生了什么，也没有任何可供排查的线索。项目里已经发生过两次——
一次是绘制代码把 Qt ``QRect`` 当成本项目的 ``Rect`` 用（``is_empty``），
一次是同一处把 ``rect.width`` 当成属性而不是方法。两次都表现为
"程序无声消失"。

装上 :func:`install_exception_hooks` 后，PyQt6 会改为调用 ``sys.excepthook``，
进程不再自杀，异常被记入日志并通过界面提示用户——这样"出错"和
"崩溃"就分开了。

**不吞掉错误**：钩子只负责让程序活下来并把问题讲清楚，不做静默恢复。
提示里会明确写出"该操作未生效"，避免用户误以为操作成功。
"""

from __future__ import annotations

import sys
import threading
import traceback
from collections.abc import Callable

from .logger import get_logger

logger = get_logger(__name__)

#: 异常摘要的最大长度，防止把超长信息塞进界面提示
_MAX_SUMMARY = 160

#: 界面回调，接收 ``(标题, 描述, 级别)``。由 UI 层注册。
_ui_reporter: Callable[[str, str, str], None] | None = None

#: 已经报告过的异常次数，供状态栏/诊断使用
_report_count = 0


def _summarise(exc: BaseException) -> str:
    """把异常压成一行人类可读的摘要。"""
    text = f"{type(exc).__name__}: {exc}".strip()
    if len(text) > _MAX_SUMMARY:
        text = text[: _MAX_SUMMARY - 1] + "…"
    return text


def _describe_origin(module: str, where: str) -> str:
    """描述异常发生在哪个模块的哪一类操作里。"""
    short = module.split("pingyitong.")[-1] if "pingyitong" in module else module
    return f"{short} · {where}" if where else short


def _report(exc: BaseException, origin: str) -> None:
    """记录异常，并按需通知界面。"""
    global _report_count
    _report_count += 1

    logger.error("未捕获的异常（%s）：%s", origin, _summarise(exc))
    logger.error("完整堆栈：\n%s", "".join(traceback.format_exception(exc)))

    if _ui_reporter is None:
        return
    try:
        _ui_reporter(
            "操作未完成",
            f"{_summarise(exc)}（{origin}）· 详情见日志",
            "error",
        )
    except Exception:  # noqa: BLE001 - 报告异常时不能再抛异常
        logger.debug("异常提示的界面回调自身失败，已忽略")


def _make_excepthook() -> Callable:
    """构造主线程异常钩子。"""

    def _hook(exc_type, exc_value, exc_tb) -> None:
        if exc_value is None:
            exc_value = exc_type()
        # KeyboardInterrupt 属于用户主动中断，交回默认处理
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return

        frame = exc_tb.tb_frame if exc_tb is not None else None
        origin = _describe_origin(
            frame.f_globals.get("__name__", "") if frame else "",
            frame.f_code.co_name if frame else "",
        )
        _report(exc_value, origin)

    return _hook


def _make_thread_hook() -> Callable:
    """构造后台线程异常钩子。

    后台线程里抛异常默认只打印到 stderr 且不影响主线程，
    但同样会让功能静默失效（例如模型加载线程挂掉后界面永远停在"加载中"）。
    """

    def _hook(args: threading.ExceptHookArgs) -> None:
        if args.exc_type is SystemExit:
            return
        exc_value = args.exc_value if args.exc_value is not None else args.exc_type()
        thread_name = args.thread.name if args.thread is not None else "未知线程"
        _report(exc_value, f"后台线程 {thread_name}")

    return _hook


def install_exception_hooks(
    reporter: Callable[[str, str, str], None] | None = None,
) -> None:
    """安装主线程与后台线程的异常钩子。

    :param reporter: 界面提示回调 ``(标题, 描述, 级别)``。
        传入 ``None`` 时只记日志。
    """
    global _ui_reporter
    _ui_reporter = reporter

    sys.excepthook = _make_excepthook()
    threading.excepthook = _make_thread_hook()

    # 析构函数里抛出的异常无法被 excepthook 捕获，单独接管
    def _unraisable(args: object) -> None:
        exc = getattr(args, "exc_value", None)
        if exc is not None:
            _report(exc, "对象析构")

    if hasattr(sys, "unraisablehook"):
        sys.unraisablehook = _unraisable  # type: ignore[assignment]

    logger.debug("全局异常钩子已安装")


def report_count() -> int:
    """返回已报告的异常次数（诊断用）。"""
    return _report_count


def reset_for_tests() -> None:
    """重置内部状态，仅供测试使用。"""
    global _ui_reporter, _report_count
    _ui_reporter = None
    _report_count = 0
    sys.excepthook = sys.__excepthook__
    threading.excepthook = threading.__excepthook__  # type: ignore[attr-defined]
