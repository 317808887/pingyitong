"""桌面场景 OCR 后处理测试。

对应《项目原创性声明》二·（五）声明的六项原创能力。这些算法把 OCR 输出的
碎片文本还原为通顺段落，是翻译质量的关键前置环节，因此逐项覆盖：

- 基于坐标位置的文本行合并
- 段落划分
- 重复文本去重
- 上下文拼接
- 断句修复（断行、连字符断词）
"""

from __future__ import annotations

import unittest

from pingyitong.components.capture.geometry import Rect
from pingyitong.components.ocr import TextBlock
from pingyitong.services.reading_order import reading_text
from pingyitong.services.text_preprocess import (
    TextPreprocessor,
    join_segments,
    repair_hyphenation,
)


def block(text: str, x: int, y: int, width: int = 100, height: int = 16, confidence: float = 0.95) -> TextBlock:
    """构造一个位于指定位置的文本块。"""
    return TextBlock(
        text=text,
        box=Rect(x, y, width, height),
        confidence=confidence,
        index=0,
    )


class TestNoiseFiltering(unittest.TestCase):
    """噪声过滤。"""

    def setUp(self) -> None:
        self.pre = TextPreprocessor()

    def test_pure_numbers_removed(self) -> None:
        """纯数字（行号、进度值）应被过滤。"""
        paragraphs, stats = self.pre.process([block("12345", 0, 0)])
        self.assertEqual(len(paragraphs), 0)
        self.assertEqual(stats.noise_removed, 1)

    def test_single_latin_char_removed(self) -> None:
        """单个拉丁字母通常是图标残影，应被过滤。"""
        paragraphs, _ = self.pre.process([block("A", 0, 0)])
        self.assertEqual(len(paragraphs), 0)

    def test_single_cjk_char_kept(self) -> None:
        """单个汉字有实际含义，必须保留。"""
        paragraphs, _ = self.pre.process([block("确", 0, 0)])
        self.assertEqual(len(paragraphs), 1)

    def test_symbols_removed(self) -> None:
        """纯符号应被过滤。"""
        paragraphs, _ = self.pre.process([block("===>", 0, 0)])
        self.assertEqual(len(paragraphs), 0)

    def test_normal_text_kept(self) -> None:
        """正常文本必须保留。"""
        paragraphs, _ = self.pre.process([block("Open settings", 0, 0)])
        self.assertEqual(len(paragraphs), 1)
        self.assertEqual(paragraphs[0].text, "Open settings")


class TestDeduplication(unittest.TestCase):
    """重复文本去重。"""

    def test_identical_overlapping_blocks_collapsed(self) -> None:
        """位置高度重叠的相同文本只保留一条。

        差分区域重叠会让同一处文字被识别两次，不去重会出现文字重影。
        """
        pre = TextPreprocessor()
        blocks = [
            block("Save", 10, 10, confidence=0.90),
            block("Save", 11, 11, confidence=0.98),
        ]
        paragraphs, stats = pre.process(blocks)
        self.assertEqual(len(paragraphs), 1)
        self.assertEqual(stats.duplicate_removed, 1)
        # 应保留置信度更高的那个
        self.assertAlmostEqual(paragraphs[0].blocks[0].confidence, 0.98)

    def test_same_text_at_different_positions_kept(self) -> None:
        """不同位置的相同文本（如两处"确定"按钮）不应被去重。"""
        pre = TextPreprocessor()
        blocks = [block("OK", 10, 10, 40), block("OK", 500, 400, 40)]
        paragraphs, _ = pre.process(blocks)
        self.assertEqual(len(paragraphs), 2)

    def test_contradictory_readings_keep_best_confidence(self) -> None:
        """同一位置的两种读法应择优选一，而不是拼接。

        OCR 对同一处文字可能给出两种读法（典型如 ``Cancel`` / ``Cancer``、
        ``l`` / ``1``）。若直接拼接，屏幕上会出现"Cancel Cancer"这种
        明显错误，比识别错一个字更糟。
        """
        pre = TextPreprocessor()
        blocks = [
            block("Cancel", 10, 10, confidence=0.72),
            block("Cancer", 10, 10, confidence=0.94),
        ]
        paragraphs, _ = pre.process(blocks)
        self.assertEqual(len(paragraphs), 1)
        self.assertEqual(paragraphs[0].text, "Cancer")

    def test_different_text_at_different_position_kept(self) -> None:
        """位置不同的文本即使内容相近也必须分别处理。"""
        pre = TextPreprocessor()
        blocks = [
            block("Cancel", 10, 10, confidence=0.9),
            block("Cancer", 10, 400, confidence=0.9),
        ]
        paragraphs, _ = pre.process(blocks)
        self.assertEqual(len(paragraphs), 2)


class TestLineClustering(unittest.TestCase):
    """文本行聚类。"""

    def setUp(self) -> None:
        self.pre = TextPreprocessor()

    def test_blocks_on_same_baseline_form_one_line(self) -> None:
        """垂直高度重合的块应归为同一行。"""
        blocks = [
            block("Hello", 0, 100, 60),
            block("world", 70, 101, 60),
        ]
        paragraphs, _ = self.pre.process(blocks)
        self.assertEqual(len(paragraphs), 1)
        self.assertEqual(paragraphs[0].line_count, 1)
        self.assertIn("Hello", paragraphs[0].text)
        self.assertIn("world", paragraphs[0].text)

    def test_different_rows_form_separate_lines(self) -> None:
        """垂直不重合的块应归为不同行。"""
        blocks = [block("Line one", 0, 100, 100, 16), block("Line two", 0, 140, 100, 16)]
        paragraphs, _ = self.pre.process(blocks)
        self.assertEqual(len(paragraphs), 2)
        self.assertEqual([p.line_count for p in paragraphs], [1, 1])

    def test_slight_baseline_offset_still_same_line(self) -> None:
        """基线轻微错位（不同字号混排）仍应视为同一行。

        判定依据是垂直重叠比例而不是中心点距离，正是为了应对这种情况。
        """
        blocks = [block("Title", 0, 100, 80, 20), block("subtitle", 90, 104, 60, 12)]
        paragraphs, _ = self.pre.process(blocks)
        self.assertEqual(len(paragraphs), 1)
        self.assertEqual(paragraphs[0].line_count, 1)


class TestWithinLineMerging(unittest.TestCase):
    """行内相邻文本块合并。"""

    def test_close_blocks_merged(self) -> None:
        """水平间距小于阈值的块应合并（例如被拆开的按钮文案）。"""
        pre = TextPreprocessor()
        blocks = [block("Save", 0, 100, 40, 16), block("As", 44, 100, 20, 16)]
        paragraphs, stats = pre.process(blocks)
        self.assertEqual(len(paragraphs), 1)
        self.assertGreaterEqual(stats.merged_blocks, 1)
        self.assertEqual(paragraphs[0].text, "Save As")

    def test_distant_blocks_become_separate_paragraphs(self) -> None:
        """水平间距过大的块属于不同元素，应各自成段。

        这是保守取舍的结果：屏幕翻译优先保证**位置正确**，因此只在真正
        紧邻的文字之间合并。把"Cancel"与几百像素外的"Apply"拼成一段，
        会让译文变成一个横跨两者的巨块、彻底错位；
        各自成段则两段译文分别落在各自位置上。
        """
        pre = TextPreprocessor()
        blocks = [block("Cancel", 0, 100, 60, 16), block("Apply", 400, 100, 50, 16)]
        paragraphs, _ = pre.process(blocks)
        self.assertEqual(len(paragraphs), 2)
        texts = {p.text for p in paragraphs}
        self.assertEqual(texts, {"Cancel", "Apply"})

    def test_side_by_side_windows_are_not_merged(self) -> None:
        """并排窗口的文字不得被合并到同一段。

        实测缺陷：桌面并排开着侧栏与窗口时，只按垂直相邻分组会把
        分属不同窗口的文字拼成一段（最宽达 2283 像素），译文完全错乱。
        """
        pre = TextPreprocessor()
        # 左栏（侧栏）与右栏（另一窗口）在同一水平带上各有文字
        blocks = [
            block("Sidebar text here", 20, 500, 200, 20),
            block("Another pane text", 800, 505, 220, 20),
            block("Sidebar second row", 20, 530, 210, 20),
        ]
        paragraphs, _ = pre.process(blocks)
        # 侧栏的两行应合成一段；右栏那段必须独立
        sidebar = [p for p in paragraphs if "Sidebar" in p.text]
        other = [p for p in paragraphs if "Another pane" in p.text]
        self.assertEqual(len(other), 1)
        self.assertEqual(len(sidebar), 1, "侧栏自身的两行应合成一段")
        self.assertIn("second row", sidebar[0].text)
        # 跨栏合并的段落宽度不应出现
        for p in paragraphs:
            self.assertLess(p.box.width, 600, f"段落过宽，疑似跨栏合并：{p.text!r}")


class TestParagraphGrouping(unittest.TestCase):
    """段落划分。"""

    def setUp(self) -> None:
        self.pre = TextPreprocessor()

    def test_near_lines_join_into_paragraph(self) -> None:
        """紧邻的行合并为一段（句子被换行截断的情况）。"""
        blocks = [
            block("This is a sentence that was", 0, 100, 200, 16),
            block("wrapped by the renderer", 0, 118, 150, 16),
        ]
        paragraphs, _ = self.pre.process(blocks)
        self.assertEqual(len(paragraphs), 1)
        self.assertEqual(paragraphs[0].line_count, 2)

    def test_distant_lines_split_into_paragraphs(self) -> None:
        """间距较大的行拆分为不同段落。"""
        blocks = [
            block("First paragraph here.", 0, 100, 200, 16),
            block("Second paragraph here.", 0, 200, 200, 16),
        ]
        paragraphs, _ = self.pre.process(blocks)
        self.assertEqual(len(paragraphs), 2)

    def test_terminal_punctuation_splits_paragraph(self) -> None:
        """上一行以句末标点结束说明语义完整，应另起一段。

        即使两行垂直间距接近，也不应把两句独立的话拼成一句——
        这会让译文出现主语错配。
        """
        blocks = [
            block("Something已完成。", 0, 100, 200, 16),
            block("Next item starts here", 0, 118, 200, 16),
        ]
        paragraphs, _ = self.pre.process(blocks)
        self.assertEqual(len(paragraphs), 2)

    def test_full_width_title_not_merged_with_sidebar_item(self) -> None:
        """横贯整窗的标题不得与下方另一栏的短项并成一段。

        实测缺陷："Settings - General Preferences" 标题（1700px 宽）与左侧栏
        "Dashboard"（240px 宽）被并成一段，原位译文变成横跨两者的大块，
        文字栏里出现"设置 - 通用偏好 仪表盘"的跨栏混排。

        上一行远宽于下一行（超过 :data:`~pingyitong.services.text_preprocess.MUCH_WIDER_RATIO`
        倍）时必须是真续行才允许合并：行距紧凑
        且左右边缘对齐。此处行距 60px（1.5 倍行高）且下一行左缘超出
        上一行左缘，两个续行特征都不满足，必须各自成段。
        """
        pre = TextPreprocessor(line_max_horizontal_gap_ratio=1.5)
        blocks = [
            block("Settings - General Preferences", 100, 20, 1700, 40),
            block("Dashboard", 40, 120, 240, 32),
        ]
        paragraphs, _ = pre.process(blocks)
        self.assertEqual(len(paragraphs), 2)
        texts = {p.text for p in paragraphs}
        self.assertEqual(texts, {"Settings - General Preferences", "Dashboard"})

    def test_extremely_short_last_line_still_joins(self) -> None:
        """上一行远宽于下一行、但行距紧凑且左对齐时仍是折行续行。

        与 :meth:`test_full_width_title_not_merged_with_sidebar_item` 相对：
        宽行在上的**真折行**（末行可以很短）必须照常合并，反向保护
        不得误伤正常段落。
        """
        pre = TextPreprocessor()
        blocks = [
            block("The quick brown fox jumps over the lazy", 0, 100, 400, 16),
            block("dog.", 0, 118, 40, 16),
        ]
        paragraphs, _ = pre.process(blocks)
        self.assertEqual(len(paragraphs), 1)
        self.assertEqual(paragraphs[0].line_count, 2)

    def test_paragraph_ids_assigned(self) -> None:
        """段落号应被回填到文本块，便于调试与追溯。"""
        blocks = [
            block("First paragraph.", 0, 0, 200, 16),
            block("Second paragraph.", 0, 200, 200, 16),
        ]
        paragraphs, _ = self.pre.process(blocks)
        self.assertEqual(paragraphs[0].blocks[0].paragraph_id, 0)
        self.assertEqual(paragraphs[1].blocks[0].paragraph_id, 1)


class TestTextAssembly(unittest.TestCase):
    """上下文拼接规则。"""

    def test_latin_words_separated_by_space(self) -> None:
        """拉丁字母词之间补空格。"""
        pre = TextPreprocessor()
        blocks = [block("Hello", 0, 100, 40), block("World", 44, 100, 40)]
        paragraphs, _ = pre.process(blocks)
        self.assertEqual(paragraphs[0].text, "Hello World")

    def test_cjk_not_separated_by_space(self) -> None:
        """中文之间不补空格。"""
        pre = TextPreprocessor()
        blocks = [block("打开", 0, 100, 40), block("设置", 44, 100, 40)]
        paragraphs, _ = pre.process(blocks)
        self.assertEqual(paragraphs[0].text, "打开设置")

    def test_cjk_and_latin_no_extra_space(self) -> None:
        """中英之间不额外补空格。"""
        pre = TextPreprocessor()
        blocks = [block("打开", 0, 100, 40), block("Settings", 44, 100, 60)]
        paragraphs, _ = pre.process(blocks)
        self.assertEqual(paragraphs[0].text, "打开Settings")

    def test_leading_punctuation_no_space(self) -> None:
        """行首禁排标点前不补空格。"""
        self.assertEqual(join_segments(["他说", "，然后离开"], cjk=False), "他说，然后离开")

    def test_join_segments_cjk_mode(self) -> None:
        """CJK 模式直接连接。"""
        self.assertEqual(join_segments(["第一句", "第二句"], cjk=True), "第一句第二句")

    def test_join_segments_skips_empty(self) -> None:
        """空片段应被跳过。"""
        self.assertEqual(join_segments(["Hello", "", "  ", "World"]), "Hello World")


class TestHyphenationRepair(unittest.TestCase):
    """断词修复。"""

    def test_repair_hyphenated_word(self) -> None:
        """行尾连字符断词应被拼接。

        ``inter-`` + ``face`` 应还原为 ``interface``，而不是 ``inter-face``。
        """
        self.assertEqual(repair_hyphenation("inter- face"), "interface")

    def test_keep_real_hyphen(self) -> None:
        """真正的连字符词不应被改动。"""
        self.assertEqual(repair_hyphenation("state-of-the-art"), "state-of-the-art")

    def test_cross_line_hyphen_joined(self) -> None:
        """跨行断词在段落拼接时被修复。"""
        pre = TextPreprocessor(repair_hyphenation_enabled=True)
        blocks = [
            block("The inter-", 0, 100, 100, 16),
            block("face is ready", 0, 118, 100, 16),
        ]
        paragraphs, _ = pre.process(blocks)
        self.assertEqual(len(paragraphs), 1)
        self.assertIn("interface", paragraphs[0].text)
        self.assertNotIn("inter-", paragraphs[0].text)

    def test_repair_can_be_disabled(self) -> None:
        """关闭修复时不改动原文。"""
        pre = TextPreprocessor(repair_hyphenation_enabled=False)
        blocks = [
            block("The inter-", 0, 100, 100, 16),
            block("face is ready", 0, 118, 100, 16),
        ]
        paragraphs, _ = pre.process(blocks)
        self.assertIn("-", paragraphs[0].text)


class TestParagraphGeometry(unittest.TestCase):
    """段落包围盒（渲染定位依据）。"""

    def test_box_is_union_of_blocks(self) -> None:
        """段落包围盒应为内部各块的并集。"""
        pre = TextPreprocessor()
        blocks = [
            block("Line one", 10, 100, 100, 16),
            block("Line two", 10, 120, 80, 16),
        ]
        paragraphs, _ = pre.process(blocks)
        self.assertEqual(len(paragraphs), 1)
        box = paragraphs[0].box
        self.assertEqual(box.left, 10)
        self.assertEqual(box.top, 100)
        self.assertEqual(box.bottom, 136)

    def test_confidence_is_area_weighted(self) -> None:
        """段落置信度按面积加权，长块影响更大。"""
        pre = TextPreprocessor()
        blocks = [
            block("Long reliable text here", 0, 100, 300, 16, confidence=0.99),
            block("x", 310, 100, 10, 16, confidence=0.55),
        ]
        paragraphs, _ = pre.process(blocks)
        self.assertGreater(paragraphs[0].confidence, 0.9)

    def test_language_detected(self) -> None:
        """段落应带上检测到的源语言。"""
        pre = TextPreprocessor()
        paragraphs, _ = pre.process([block("Hello world how are you", 0, 0)])
        self.assertEqual(paragraphs[0].language, "en")


class TestStatistics(unittest.TestCase):
    """统计信息（用于状态面板展示清洗效果）。"""

    def test_stats_reported(self) -> None:
        """统计应如实反映输入与输出数量。"""
        pre = TextPreprocessor()
        blocks = [
            block("Hello", 0, 100, 60),
            block("world", 70, 101, 60),
            block("123", 0, 300, 30),
        ]
        paragraphs, stats = pre.process(blocks)
        self.assertEqual(stats.input_blocks, 3)
        self.assertEqual(stats.output_paragraphs, len(paragraphs))
        self.assertEqual(stats.noise_removed, 1)
        self.assertIn("块", stats.as_text())

    def test_empty_input(self) -> None:
        """空输入不应报错。"""
        pre = TextPreprocessor()
        paragraphs, stats = pre.process([])
        self.assertEqual(paragraphs, [])
        self.assertEqual(stats.input_blocks, 0)


class TestFromConfig(unittest.TestCase):
    """from_config 工厂：脚本与界面必须经由它构造，保证与真实应用一致。"""

    def test_matches_app_construction(self) -> None:
        """from_config 的各参数应与 AppConfig 中的对应配置一致。"""
        from pingyitong.config.schema import AppConfig

        config = AppConfig()
        pre = TextPreprocessor.from_config(config)

        self.assertEqual(pre.line_merge_overlap_ratio, config.ocr.line_merge_overlap_ratio)
        self.assertEqual(pre.same_line_gap_ratio, config.ocr.same_line_gap_ratio)
        self.assertEqual(pre.paragraph_gap_ratio, config.ocr.paragraph_gap_ratio)
        self.assertEqual(
            pre.line_max_horizontal_gap_ratio,
            config.ocr.line_max_horizontal_gap_ratio,
        )
        self.assertEqual(pre.merge_broken_lines, config.preprocess.merge_broken_lines)
        self.assertEqual(
            pre.repair_hyphenation_enabled, config.preprocess.repair_hyphenation
        )
        self.assertEqual(pre.deduplicate_enabled, config.preprocess.deduplicate)
        self.assertEqual(pre.filter_noise_enabled, config.preprocess.filter_noise)
        self.assertEqual(pre.skip_pure_numeric, config.preprocess.skip_pure_numeric)
        self.assertEqual(
            pre.fallback_language, config.translation.fallback_source_language
        )

    def test_bare_default_differs_from_schema(self) -> None:
        """裸构造的默认值与 schema 不同——这正是 from_config 存在的原因。

        ``line_max_horizontal_gap_ratio`` 构造默认 4.0、schema 默认 1.5；
        脚本若用裸构造，复现结果会与真实应用不一致。
        """
        from pingyitong.config.schema import AppConfig

        self.assertNotEqual(TextPreprocessor().line_max_horizontal_gap_ratio,
                            AppConfig().ocr.line_max_horizontal_gap_ratio)


class _OrderedItem:
    """阅读顺序测试的最小条目：只需要 box 与 text。"""

    def __init__(self, box: Rect, text: str) -> None:
        self.box = box
        self.text = text

    def __repr__(self) -> str:  # pragma: no cover - 便于失败时阅读
        return f"<{self.text!r}>"


class TestSpanningTitleRegression(unittest.TestCase):
    """跨栏大标题 + 双栏正文的语序回归。

    实测缺陷：跨栏标题（580px 宽，box=(296,40,580×51)）与右栏首行
    （263px 宽，box=(696,177,263×40)）的间距比 86/51≈1.69 恰好压线通过
    ``paragraph_gap_ratio``（1.7），水平重叠 180/263≈0.68 也通过，而宽度比
    580/263≈2.2 低于旧的 ``MUCH_WIDER_RATIO``（3.0）——宽标题反向保护
    不触发，标题被当成右栏首行的折行续行合并。合并出的巨盒横跨两栏，
    XY-cut 切不开栏，文字栏语序变成"左栏→标题→右栏"。

    修复后标题必须独立成段，交给 ``reading_order._peel_spanning`` 按
    "标题→左栏→右栏" 输出。
    """

    def setUp(self) -> None:
        self.pre = TextPreprocessor()

    @staticmethod
    def _spanning_title_layout() -> list[TextBlock]:
        """合成复现几何：跨栏标题 + 左右双栏各两行（数据取自实测 OCR）。"""
        return [
            block("ZULU spanning main title here", 296, 40, 580, 51),
            block("ALPHA body left one", 60, 177, 240, 40),
            block("BRAVO body left two", 60, 277, 240, 40),
            block("DELTA body right one", 696, 177, 263, 40),
            block("ECHO body right two", 696, 277, 263, 40),
        ]

    def test_title_stays_independent_after_preprocess(self) -> None:
        """标题不得被并入栏内段落，应作为独立条目输出。"""
        paragraphs, _ = self.pre.process(self._spanning_title_layout())
        # Original captured geometry has 60px blank gaps between 40px rows.
        # Preserve the independent rows rather than inferring a wrap from
        # similar widths and absent punctuation alone.
        self.assertEqual(len(paragraphs), 5, f"段落数错乱：{paragraphs}")

        title = [p for p in paragraphs if "ZULU" in p.text]
        self.assertEqual(len(title), 1, "标题应恰好成一段")
        self.assertEqual(title[0].text, "ZULU spanning main title here")
        self.assertEqual(title[0].box.width, 580, "标题段落盒不应并入他栏")
        self.assertEqual(title[0].line_count, 1)

    def test_full_chain_reading_order_is_title_left_right(self) -> None:
        """完整链路：合成块 → process → reading_text，语序为标题→左栏→右栏。"""
        paragraphs, _ = self.pre.process(self._spanning_title_layout())
        items = [_OrderedItem(p.box, p.text) for p in paragraphs]
        lines = reading_text(items).splitlines()
        self.assertEqual(len(lines), 5)
        self.assertTrue(lines[0].startswith("ZULU"), f"标题应最先：{lines}")
        self.assertTrue(lines[1].startswith("ALPHA"), f"左栏应第二：{lines}")
        self.assertTrue(lines[2].startswith("BRAVO"), f"左栏续项应第三：{lines}")
        self.assertTrue(lines[3].startswith("DELTA"), f"右栏应随后：{lines}")

    def test_single_column_wrapped_lines_still_merge(self) -> None:
        """守卫：单栏折行段仍必须合并为一段（修复不得误伤正常折行）。"""
        blocks = [
            block("ALPHA the first line of the paragraph wraps", 60, 177, 500, 40),
            block("BRAVO and continues here on the second line", 60, 219, 480, 40),
        ]
        paragraphs, _ = self.pre.process(blocks)
        self.assertEqual(len(paragraphs), 1)
        self.assertEqual(paragraphs[0].line_count, 2)

    def test_two_columns_without_title_keep_order(self) -> None:
        """守卫：双栏无标题版面语序不变（先左栏后右栏）。"""
        blocks = [
            block("ALPHA left one", 60, 100, 240, 40),
            block("BRAVO left two", 60, 200, 240, 40),
            block("CHARLIE right one", 696, 100, 263, 40),
            block("DELTA right two", 696, 200, 263, 40),
        ]
        paragraphs, _ = self.pre.process(blocks)
        items = [_OrderedItem(p.box, p.text) for p in paragraphs]
        lines = reading_text(items).splitlines()
        self.assertEqual(len(lines), 4)
        self.assertTrue(lines[0].startswith("ALPHA"), f"左栏应先：{lines}")
        self.assertTrue(lines[1].startswith("BRAVO"), f"左栏续项应第二：{lines}")
        self.assertTrue(lines[2].startswith("CHARLIE"), f"右栏应后：{lines}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
