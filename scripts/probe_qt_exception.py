"""验证 PyQt6 中事件处理器抛异常时的行为。

这决定了"要不要装全局 excepthook"这个决策：
若 PyQt6 在未处理异常时直接 ``qFatal()`` 终止进程，那么任何一处界面
回调里的笔误都会让整个程序**无声消失**（用户看不到任何提示、
日志也留不下痕迹）。这正是"框选屏幕时崩溃"表现出来
的现象。

实测结论（PyQt6 6.11 / Windows）：**默认确实会终止进程**。
未装钩子时退出码为 ``0xC0000409``（``qFatal``），且不打印任何堆栈；
装上 ``sys.excepthook`` 后同一个异常被交给钩子，进程继续运行。

这个结论是 :mod:`pingyitong.utils.exceptions` 的设计前提，
因此本脚本保留下来：**PyQt 升级后应重新运行一次**，
确认前提仍然成立（若 PyQt 未来改为可配置，这里的做法也可简化）。

运行：
    python scripts/probe_qt_exception.py
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHILD = textwrap.dedent(
    """
    import sys
    sys.path.insert(0, r"{src}")

    {hook}

    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication, QWidget

    app = QApplication(sys.argv)

    class Boom(QWidget):
        def paintEvent(self, event):
            raise RuntimeError("模拟界面回调里的笔误")

    w = Boom()
    w.show()

    def trigger():
        # 直接触发一次真实绘制路径
        w.repaint()
        print("绘制调用已返回", flush=True)
        app.quit()

    QTimer.singleShot(100, trigger)
    code = app.exec()
    print("事件循环正常结束，退出码", code, flush=True)
    """
)


def run_case(name: str, hook: str) -> None:
    """运行一个子进程用例并报告结果。"""
    script = CHILD.format(src=str(ROOT / "src"), hook=hook)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    print(f"--- {name} ---")
    print(f"退出码: {proc.returncode}")
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    if out:
        print("stdout:", out.replace("\n", "\n        "))
    if err:
        print("stderr:", err.replace("\n", "\n        "))
    print()


def main() -> int:
    run_case("默认（无自定义 excepthook）", "")
    run_case(
        "自定义 excepthook",
        textwrap.dedent(
            """
            import traceback
            def _hook(t, v, tb):
                print("excepthook 捕获到：", t.__name__, v, flush=True)
            sys.excepthook = _hook
            """
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
