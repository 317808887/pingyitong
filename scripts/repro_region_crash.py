"""复现"框选屏幕时崩溃"。

用真实 Qt 事件驱动 :class:`RegionSelector` 走完整拖拽流程，并把
``region_selected`` 接到一个**尽量接近真实处理器**的槽上
（写配置 + 更新面板文字 + 重新识别 + 弹提示），以复现用户报告的崩溃。

崩溃根因见 ``docs/架构说明.md`` 六·6.4：选区是 Qt 的 ``QRect``，
宽高要用方法 ``selection.width()``；把它当属性 ``selection.width``
传给本项目的 ``Rect`` 后，到后续算术才抛 ``TypeError``，
而 PyQt6 会把事件处理器里的未捕获异常变成 ``qFatal()`` 直接结束进程。

修复前：进程在 ``release`` 之后消失，无任何 Python 栈（退出码非 0）。
修复后：完整走完拖拽、取消、空绘制三条路径，退出码 0。

运行：
    python scripts/repro_region_crash.py
"""

from __future__ import annotations

import faulthandler
import os
import sys
import traceback
from pathlib import Path

# 段错误（Qt 层崩溃）不会抛 Python 异常，必须打开 faulthandler 才能在
# 进程被信号杀死时拿到 Python 栈。
faulthandler.enable()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("QT_QPA_PLATFORM", "windows") if sys.platform == "win32" else None


def main() -> int:
    from PyQt6.QtCore import QPoint, Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication

    _app = QApplication.instance() or QApplication(sys.argv)

    from pingyitong.components.capture.geometry import Rect, list_screens
    from pingyitong.ui.design.theme import build_theme
    from pingyitong.ui.region_selector import RegionSelector
    from pingyitong.utils import install_exception_hooks

    # 与真实程序一致：先装钩子，再看界面回调里出错时会发生什么
    install_exception_hooks()

    theme = build_theme("dark")

    print("=== 显示器 ===")
    for i, s in enumerate(list_screens()):
        print(f"  [{i}] {s.label()} logical={s.logical} physical={s.physical} mss={s.mss_index}")

    received: list[tuple] = []

    def on_selected(logical: Rect, monitor) -> None:
        # 模拟真实处理器 ``Application._on_region_selected`` 的每一步。
        # 注意这里**故意**保留一次会抛异常的动作：装上全局异常钩子后，
        # 它应当被记录下来并提示用户，而不是把进程带走。
        print(f"  [槽] 收到 region_selected logical={logical} monitor={monitor}")
        print("  [槽] 1/4 to_sequence")
        seq = logical.to_sequence()
        print(f"  [槽]     -> {seq}")

        print("  [槽] 2/4 触发一次异常（验证钩子能兜住）")
        from pingyitong.utils import report_count

        try:
            raise RuntimeError("模拟界面回调里的笔误")
        except RuntimeError:
            # 走真实的钩子路径：handler 抛异常时 PyQt 会调用 sys.excepthook
            sys.excepthook(*sys.exc_info())
        print(f"  [槽]     已上报次数={report_count()}")

        print("  [槽] 3/4 写配置（模拟）")
        from pingyitong.config.schema import AppConfig
        from pingyitong.config.settings import SettingsManager

        cfg = AppConfig()
        print(f"  [槽]     capture.mode={cfg.capture.mode}")
        print(f"  [槽]     SettingsManager 可用={SettingsManager is not None}")

        print("  [槽] 4/4 完成")
        received.append((logical, monitor))

    selector = RegionSelector(theme)
    selector.region_selected.connect(on_selected)
    selector.cancelled.connect(lambda: print("  [槽] cancelled"))

    print("=== 开始框选 ===")
    selector.start()
    QTest.qWait(300)

    geo = selector.geometry()
    print(f"覆盖层几何: {geo.x()},{geo.y()} {geo.width()}x{geo.height()}")

    # 起点与终点都取窗口内部，避免落在别的显示器之外
    start = QPoint(200, 200)
    end = QPoint(700, 560)

    print("press")
    QTest.mousePress(selector, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
    QTest.qWait(50)

    # 分几步移动，模拟真实拖拽
    for step in range(1, 6):
        x = start.x() + (end.x() - start.x()) * step // 5
        y = start.y() + (end.y() - start.y()) * step // 5
        QTest.mouseMove(selector, QPoint(x, y))
        QTest.qWait(30)
    print("move done")

    print("release")
    QTest.mouseRelease(selector, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, end)
    QTest.qWait(300)

    print(f"=== 结果：收到 {len(received)} 次选择 ===")

    # 收尾：走一遍取消路径
    print("=== 取消路径 ===")
    selector.start()
    QTest.qWait(100)
    QTest.keyClick(selector, Qt.Key.Key_Escape)
    QTest.qWait(100)

    # 再走一遍"仅移动不按下"的绘制路径
    print("=== 空绘制路径 ===")
    selector.start()
    QTest.qWait(100)
    QTest.mouseMove(selector, QPoint(300, 300))
    QTest.qWait(100)
    selector.cancel()
    QTest.qWait(100)

    print("=== 全部通过，未崩溃 ===")
    selector.deleteLater()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001
        print("!!! 捕获到异常 !!!", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1) from None
