"""阅读顺序重排测试。

对应需求："文本框里显示的文字按界面中从上到下、从左到右的顺序排列，
以确保语义流畅"。

**核心要防的是"按 (y, x) 直接排序"这种看似正确实则不通的做法**：
屏幕上同时有侧边栏与主内容区时，同一水平带的元素分属不同区域，
只按 y 排会让两边的内容交替出现，读起来支离破碎。
"""

from __future__ import annotations

import unittest

from pingyitong.components.capture.geometry import Rect
from pingyitong.services.reading_order import order_for_reading, reading_text


class _Item:
    """最小条目：只需要 box 与 text。"""

    def __init__(self, text: str, box: Rect) -> None:
        self.text = text
        self.box = box

    def __repr__(self) -> str:  # pragma: no cover - 便于失败时阅读
        return f"<{self.text}>"


def _order(items: list[_Item]) -> list[str]:
    return [item.text for item in order_for_reading(items)]


class TestColumnDetection(unittest.TestCase):
    """先分栏，再在栏内按行读取。"""

    def test_sidebar_and_main_are_not_interleaved(self) -> None:
        """左侧栏与右侧主内容区必须分组读取。

        这是"按 y 排序"会失败、而分栏能救回来的典型场景：
        左右两边的第一行在同一水平带，但语义上毫不相干。
        """
        items = [
            _Item("新建任务", Rect(20, 100, 120, 20)),
            _Item("The quick brown fox", Rect(400, 96, 400, 22)),
            _Item("历史", Rect(20, 200, 80, 20)),
            _Item("jumps over the lazy dog", Rect(400, 196, 460, 22)),
            _Item("设置", Rect(20, 300, 80, 20)),
            _Item("Segmentation fault", Rect(400, 296, 300, 22)),
        ]
        self.assertEqual(
            _order(items),
            [
                "新建任务",
                "历史",
                "设置",
                "The quick brown fox",
                "jumps over the lazy dog",
                "Segmentation fault",
            ],
        )

    def test_three_columns_read_left_to_right(self) -> None:
        """三栏时也按栏从左到右。"""
        items = [
            _Item("右栏", Rect(1200, 100, 100, 20)),
            _Item("左栏", Rect(50, 100, 100, 20)),
            _Item("中栏", Rect(600, 100, 100, 20)),
        ]
        self.assertEqual(_order(items), ["左栏", "中栏", "右栏"])

    def test_small_gaps_do_not_split_columns(self) -> None:
        """同一行内正常的词间空隙不得被误判成栏间距。"""
        items = [
            _Item("Hello", Rect(100, 100, 60, 20)),
            _Item("world", Rect(170, 100, 60, 20)),  # 间隙 10px，远小于行高
        ]
        self.assertEqual(_order(items), ["Hello", "world"])


class TestRowBandSplitting(unittest.TestCase):
    """横切与竖切的方向选择：空隙更宽的方向先切。

    回归背景：旧实现只会"先分栏、栏内分行"，遇到底部横贯页脚时
    页脚会被并进左栏，文本框里页脚出现在正文中间。
    """

    def test_footer_is_read_last(self) -> None:
        """底部页脚必须排在最后，而不是被并进左侧栏。"""
        items = [
            _Item("主区第一行", Rect(880, 120, 500, 24)),
            _Item("页脚", Rect(40, 1040, 600, 22)),
            _Item("侧栏一项", Rect(40, 120, 200, 24)),
            _Item("主区第二行", Rect(880, 160, 500, 24)),
            _Item("侧栏二项", Rect(40, 160, 200, 24)),
        ]
        self.assertEqual(
            _order(items),
            [
                "侧栏一项",
                "侧栏二项",
                "主区第一行",
                "主区第二行",
                "页脚",
            ],
        )

    def test_full_width_title_is_read_first(self) -> None:
        """横贯全宽的顶部标题先于两侧内容被读到。"""
        items = [
            _Item("侧栏", Rect(40, 120, 200, 24)),
            _Item("标题", Rect(100, 40, 1200, 30)),
            _Item("页脚", Rect(40, 1040, 600, 22)),
            _Item("正文", Rect(880, 120, 500, 24)),
        ]
        self.assertEqual(
            _order(items),
            ["标题", "侧栏", "正文", "页脚"],
        )

    def test_paragraph_gap_does_not_break_column_flow(self) -> None:
        """栏内段落间距较大时不得把并排的栏拆散成交替行。

        主栏内部两段之间有 1.8 倍行高的空隙，但栏间距远大于它，
        因此仍应整体按"先左栏、后右栏"读取。
        """
        items = [
            _Item("侧栏甲", Rect(40, 100, 200, 20)),
            _Item("侧栏乙", Rect(40, 220, 200, 20)),
            _Item("正文第一段", Rect(880, 100, 400, 20)),
            _Item("正文第二段", Rect(880, 236, 400, 20)),
        ]
        self.assertEqual(
            _order(items),
            ["侧栏甲", "侧栏乙", "正文第一段", "正文第二段"],
        )

    def test_title_sidebar_content_footer_full_layout(self) -> None:
        """完整桌面版面：横贯标题最先、侧栏连续、主区连续、页脚最后。

        回归背景（两个缺陷叠加，文字栏语序错乱实测复现）：

        1. 横贯全宽的标题（1700px）堵住了侧栏与主区之间的栏空隙，
           旧算法只按"最宽空隙"选方向，稀疏侧栏（项间距 168px）与
           主区段落间距（220px）都大于栏间隙（120px），于是层层横切，
           侧栏项与正文交替出现；
        2. 修复方式：每层先把"独处顶端/底端且堵住栏空隙"的横贯元素
           （标题、页脚）剥离，剩余部分存在栏结构时优先分栏——行间隙
           需达到栏间隙的 2.5 倍（页脚级的大空隙）才能改按行切。
        """
        items = [
            _Item("通知设置", Rect(40, 520, 240, 32)),
            _Item("仪表盘", Rect(40, 120, 240, 32)),
            _Item("正文第二段", Rect(400, 480, 1400, 60)),
            _Item("用户资料", Rect(40, 320, 240, 32)),
            _Item("正文第一段", Rect(400, 200, 1400, 60)),
            _Item("标题", Rect(100, 20, 1700, 40)),
            _Item("页脚", Rect(40, 1020, 1760, 28)),
        ]
        self.assertEqual(
            _order(items),
            [
                "标题",
                "仪表盘",
                "用户资料",
                "通知设置",
                "正文第一段",
                "正文第二段",
                "页脚",
            ],
        )


class TestRowOrdering(unittest.TestCase):
    """单栏内按行带从上到下，行内从左到右。"""

    def test_lines_top_to_bottom(self) -> None:
        items = [
            _Item("第三行", Rect(100, 180, 400, 20)),
            _Item("第一行", Rect(100, 100, 400, 20)),
            _Item("第二行", Rect(100, 140, 400, 20)),
        ]
        self.assertEqual(_order(items), ["第一行", "第二行", "第三行"])

    def test_same_row_left_to_right(self) -> None:
        items = [
            _Item("右侧", Rect(500, 100, 100, 20)),
            _Item("左侧", Rect(100, 102, 100, 20)),
        ]
        self.assertEqual(_order(items), ["左侧", "右侧"])

    def test_slightly_offset_baseline_is_same_row(self) -> None:
        """基线略有高低（真实 OCR 常见）仍应视为同一行。"""
        items = [
            _Item("后", Rect(300, 104, 80, 20)),
            _Item("前", Rect(100, 100, 80, 20)),
        ]
        self.assertEqual(_order(items), ["前", "后"])

    def test_clearly_separate_rows_are_ordered_by_top(self) -> None:
        """垂直几乎不重叠的两块属于不同行，按上到下落。"""
        items = [
            _Item("下", Rect(100, 200, 300, 20)),
            _Item("上", Rect(100, 100, 300, 20)),
        ]
        self.assertEqual(_order(items), ["上", "下"])


class TestReadingText(unittest.TestCase):
    """拼装成可读文字。"""

    def test_skips_blank_and_keeps_order(self) -> None:
        items = [
            _Item("B", Rect(400, 100, 200, 20)),
            _Item("   ", Rect(20, 100, 60, 20)),
            _Item("A", Rect(20, 200, 60, 20)),
        ]
        self.assertEqual(reading_text(items), "A\nB")

    def test_empty_input(self) -> None:
        self.assertEqual(reading_text([]), "")

    def test_custom_separator(self) -> None:
        items = [_Item("一", Rect(100, 100, 40, 20)), _Item("二", Rect(100, 200, 40, 20))]
        self.assertEqual(reading_text(items, separator=" "), "一 二")


class TestRobustness(unittest.TestCase):
    """异常输入不得崩溃。"""

    def test_single_item(self) -> None:
        items = [_Item("唯一", Rect(100, 100, 60, 20))]
        self.assertEqual(_order(items), ["唯一"])

    def test_items_without_box_are_dropped(self) -> None:
        """缺几何信息的条目无法定位，直接跳过而不是抛异常。"""

        class _NoBox:
            text = "无坐标"

        items = [_Item("有坐标", Rect(100, 100, 60, 20)), _NoBox()]
        self.assertEqual(_order(items), ["有坐标"])

    def test_zero_height_items_do_not_divide_by_zero(self) -> None:
        items = [_Item("零高", Rect(100, 100, 60, 0)), _Item("正常", Rect(100, 200, 60, 20))]
        self.assertEqual(len(_order(items)), 2)


if __name__ == "__main__":
    unittest.main()
