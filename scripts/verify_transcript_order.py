"""端到端验证：文字栏模式的语序排列。

模拟真实应用把译文快照写入文字栏（文本框）的路径——
``Application._on_transcript_updated`` 拿到渲染条目后调用
:func:`pingyitong.services.reading_order.reading_text`——用若干个
典型桌面版面实测"文字栏里的输出是否按 从上到下、从左到右 且
分栏不交错 的顺序排列"。

每个场景都构造 :class:`~pingyitong.components.render.RenderItem`
（与流水线真实交付的类型一致，只有 ``box`` 与 ``text`` 参与排序），
先给出期望顺序再实际重排比对。任何场景失败即退出码非零。

运行：
    python scripts/verify_transcript_order.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def build_cases() -> list[tuple[str, list[object], list[str]]]:
    """构造 ``(场景名, 乱序条目, 期望语序)`` 列表。"""
    from pingyitong.components.capture.geometry import Rect
    from pingyitong.components.render import RenderItem

    def item(text: str, left: int, top: int, width: int, height: int) -> RenderItem:
        return RenderItem(box=Rect(left, top, width, height), text=text)

    return [
        (
            "侧栏 + 主区（最易翻车的交替混排）",
            [
                item("新建任务", 20, 100, 120, 20),
                item("The quick brown fox", 400, 96, 400, 22),
                item("历史", 20, 200, 80, 20),
                item("jumps over the lazy dog", 400, 196, 460, 22),
                item("设置", 20, 300, 80, 20),
                item("Segmentation fault", 400, 296, 300, 22),
            ],
            [
                "新建任务",
                "历史",
                "设置",
                "The quick brown fox",
                "jumps over the lazy dog",
                "Segmentation fault",
            ],
        ),
        (
            "完整桌面版面（横贯标题最先、侧栏连续、正文连续、页脚最后）",
            [
                item("通知设置", 40, 520, 240, 32),
                item("仪表盘", 40, 120, 240, 32),
                item("正文第二段", 400, 480, 1400, 60),
                item("用户资料", 40, 320, 240, 32),
                item("正文第一段", 400, 200, 1400, 60),
                item("标题", 100, 20, 1700, 40),
                item("页脚", 40, 1020, 1760, 28),
            ],
            ["标题", "仪表盘", "用户资料", "通知设置", "正文第一段", "正文第二段", "页脚"],
        ),
        (
            "双栏文档（左栏整体读完再读右栏）",
            [
                item("右栏甲", 900, 100, 300, 20),
                item("左栏甲", 100, 104, 300, 20),
                item("右栏乙", 900, 300, 300, 20),
                item("左栏乙", 100, 304, 300, 20),
                item("右栏丙", 900, 500, 300, 20),
                item("左栏丙", 100, 504, 300, 20),
            ],
            ["左栏甲", "左栏乙", "左栏丙", "右栏甲", "右栏乙", "右栏丙"],
        ),
        (
            "字幕单栏（严格从上到下，基线略有抖动）",
            [
                item("第三行", 100, 182, 400, 20),
                item("第一行", 100, 100, 400, 22),
                item("第二行", 102, 141, 398, 20),
            ],
            ["第一行", "第二行", "第三行"],
        ),
        (
            "同一行内从左到右",
            [
                item("右侧", 500, 100, 100, 20),
                item("左侧", 100, 102, 100, 20),
            ],
            ["左侧", "右侧"],
        ),
    ]


def main() -> int:
    from pingyitong.services.reading_order import reading_text

    print("=" * 70)
    print("文字栏语序实测（reading_text，与主程序同一调用路径）")
    print("=" * 70)

    failures = 0
    for name, items, expected in build_cases():
        text = reading_text(list(items))
        actual = [line for line in text.split("\n") if line]
        ok = actual == expected
        failures += 0 if ok else 1

        verdict = "通过" if ok else "失败"
        print(f"\n【{verdict}】{name}")
        print("  文字栏输出：")
        for line in actual:
            print(f"    {line}")
        if not ok:
            print("  期望顺序：")
            for line in expected:
                print(f"    {line}")

    print()
    if failures:
        print(f"结论：{failures} 个场景语序异常，需要修复")
        return 1
    print("结论：全部场景语序正常（分栏不交错、从上到下、从左到右）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
