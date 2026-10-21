"""专业术语库测试。

对应产品功能规划三·（四）与《项目原创性声明》。

术语库是"翻译结果稳定一致"的保证机制，其核心是
**替换 → 翻译 → 还原** 三步。测试重点：

- 占位符机制是否可靠（含模型不遵守约定时的兜底）；
- 匹配顺序（长术语优先）与大小写/整词匹配语义；
- CSV 导入的格式兼容性（表头、无表头、GBK、注释行）。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pingyitong.constants import GLOSSARY_TOKEN_TEMPLATE
from pingyitong.services.glossary_service import (
    Glossary,
    GlossaryEntry,
    create_glossary,
    load_builtin_glossary,
)


class TestSubstitution(unittest.TestCase):
    """术语替换。"""

    def setUp(self) -> None:
        self.glossary = Glossary(
            [
                GlossaryEntry("machine learning", "机器学习"),
                GlossaryEntry("neural network", "神经网络"),
                GlossaryEntry("GPU", "图形处理器"),
            ]
        )

    def test_basic_substitution(self) -> None:
        """命中的术语应被替换为占位符。"""
        result = self.glossary.substitute("We use GPU here")
        self.assertNotIn("GPU", result.text)
        self.assertEqual(result.hit_count, 1)
        self.assertIn(GLOSSARY_TOKEN_TEMPLATE.format(0), result.text)

    def test_token_uses_model_readable_characters(self) -> None:
        """极速模型会丢失私用区字符，占位符必须使用可读字符。"""
        self.assertTrue(GLOSSARY_TOKEN_TEMPLATE.format(0).isascii())

    def test_text_unchanged_when_no_match(self) -> None:
        """无命中时原文不变。"""
        original = "nothing to match here"
        result = self.glossary.substitute(original)
        self.assertEqual(result.text, original)
        self.assertEqual(result.hit_count, 0)

    def test_repeated_term_reuses_token(self) -> None:
        """同一术语多次出现复用同一占位符。

        这减少了模型需要搬运的不同符号数量，降低漏抄概率。
        """
        result = self.glossary.substitute("GPU and GPU again")
        self.assertEqual(result.hit_count, 1)
        token = GLOSSARY_TOKEN_TEMPLATE.format(0)
        self.assertEqual(result.text.count(token), 2)

    def test_longest_term_matched_first(self) -> None:
        """长术语优先匹配，避免被短术语拆开。

        若 ``machine`` 先命中，``machine learning`` 就会被拆成
        "machine 的译文 + learning"，术语一致性随之失效。
        """
        glossary = Glossary(
            [
                GlossaryEntry("machine", "机器"),
                GlossaryEntry("machine learning", "机器学习"),
            ]
        )
        result = glossary.substitute("machine learning is useful")
        self.assertEqual(result.matched_terms, ["machine learning"])
        self.assertEqual(result.replacements[GLOSSARY_TOKEN_TEMPLATE.format(0)], "机器学习")

    def test_multiple_terms_get_distinct_tokens(self) -> None:
        """不同术语分配不同占位符。"""
        result = self.glossary.substitute("GPU runs machine learning")
        self.assertEqual(result.hit_count, 2)
        self.assertEqual(len(result.replacements), 2)

    def test_case_insensitive_by_default(self) -> None:
        """默认不区分大小写。"""
        result = self.glossary.substitute("gpu and Gpu and GPU")
        self.assertEqual(result.hit_count, 1)

    def test_case_sensitive_when_requested(self) -> None:
        """显式要求区分大小写时只匹配原样写法。"""
        glossary = Glossary(
            [GlossaryEntry("GPU", "图形处理器", case_sensitive=True)]
        )
        self.assertEqual(glossary.substitute("GPU").hit_count, 1)
        self.assertEqual(glossary.substitute("gpu").hit_count, 0)

    def test_whole_word_matching_avoids_partial_hits(self) -> None:
        """整词匹配避免 ``cat`` 命中 ``category``。"""
        glossary = Glossary([GlossaryEntry("cat", "猫", whole_word=True)])
        self.assertEqual(glossary.substitute("the cat sat").hit_count, 1)
        self.assertEqual(glossary.substitute("category list").hit_count, 0)

    def test_substring_matching_when_not_whole_word(self) -> None:
        """关闭整词匹配后允许子串命中。"""
        glossary = Glossary([GlossaryEntry("cat", "猫", whole_word=False)])
        self.assertEqual(glossary.substitute("category").hit_count, 1)


class TestRestoration(unittest.TestCase):
    """术语还原。"""

    def setUp(self) -> None:
        self.glossary = Glossary([GlossaryEntry("GPU", "图形处理器")])

    def test_restore_replaces_token(self) -> None:
        """占位符应被还原为术语译文。"""
        substitution = self.glossary.substitute("Use GPU")
        token = next(iter(substitution.replacements))
        restored, missing = self.glossary.restore(
            f"请使用 {token} 进行计算", substitution.replacements
        )
        self.assertEqual(restored, "请使用 图形处理器 进行计算")
        self.assertEqual(missing, [])

    def test_missing_token_reported_and_removed(self) -> None:
        """模型漏抄占位符时应报告并清除残留。

        宁可少一个术语，也不能让 ``\\ue0000\\ue001`` 这种私用区字符
        直接显示在用户屏幕上——那看起来像程序崩溃。
        """
        substitution = self.glossary.substitute("Use GPU")
        restored, missing = self.glossary.restore("请使用图形处理器", substitution.replacements)
        self.assertEqual(len(missing), 1)
        self.assertEqual(restored, "请使用图形处理器")

    def test_residual_token_cleanup(self) -> None:
        """即使模型把占位符拆散，残留字符也应被清除。"""
        substitution = self.glossary.substitute("Use GPU")
        # 模拟模型在占位符中间插入了空格
        restored, _ = self.glossary.restore(
            "使用 \ue000 0 \ue001 计算", substitution.replacements
        )
        self.assertNotIn("\ue000", restored)
        self.assertNotIn("\ue001", restored)

    def test_restore_no_replacements(self) -> None:
        """无替换表时原样返回。"""
        restored, missing = self.glossary.restore("任意文本", {})
        self.assertEqual(restored, "任意文本")
        self.assertEqual(missing, [])


class TestCsvLoading(unittest.TestCase):
    """CSV 导入。"""

    def _write(self, content: str, encoding: str = "utf-8") -> Path:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding=encoding
        ) as handle:
            handle.write(content)
        return Path(handle.name)

    def test_load_with_header(self) -> None:
        """带中文表头的文件应正确解析。"""
        path = self._write("原文,译文\nGPU,图形处理器\nCPU,中央处理器\n")
        glossary = Glossary()
        count = glossary.load_from_csv(path)
        self.assertEqual(count, 2)
        self.assertEqual(len(glossary), 2)

    def test_load_with_english_header(self) -> None:
        """带英文表头的文件应正确解析。"""
        path = self._write("source,target\nGPU,graphics processor\n")
        glossary = Glossary()
        self.assertEqual(glossary.load_from_csv(path), 1)

    def test_load_without_header(self) -> None:
        """无表头文件的两列即为原文与译文。"""
        path = self._write("GPU,图形处理器\nCPU,中央处理器\n")
        glossary = Glossary()
        count = glossary.load_from_csv(path)
        self.assertEqual(count, 2)
        # 无表头时不应把首行当作说明丢失
        self.assertIsNotNone(glossary.find("GPU"))

    def test_comment_lines_ignored(self) -> None:
        """以 # 开头的注释行应被忽略。"""
        path = self._write("# 这是注释\nGPU,图形处理器\n")
        glossary = Glossary()
        self.assertEqual(glossary.load_from_csv(path), 1)

    def test_empty_lines_skipped(self) -> None:
        """空行应被跳过。"""
        path = self._write("GPU,图形处理器\n\n\nCPU,中央处理器\n")
        glossary = Glossary()
        self.assertEqual(glossary.load_from_csv(path), 2)

    def test_incomplete_rows_skipped(self) -> None:
        """列数不足的行应被跳过而不是产生空术语。"""
        path = self._write("GPU,图形处理器\n只有一列\n")
        glossary = Glossary()
        self.assertEqual(glossary.load_from_csv(path), 1)

    def test_gbk_encoding_fallback(self) -> None:
        """GBK 编码（中文 Excel 常见导出格式）应能正常读取。"""
        path = self._write("GPU,图形处理器\n", encoding="gbk")
        glossary = Glossary()
        self.assertEqual(glossary.load_from_csv(path), 1)
        self.assertIsNotNone(glossary.find("GPU"))

    def test_bom_encoding(self) -> None:
        """带 BOM 的 UTF-8（Excel 默认导出）应能正常读取。"""
        path = self._write("原文,译文\nGPU,图形处理器\n", encoding="utf-8-sig")
        glossary = Glossary()
        count = glossary.load_from_csv(path)
        self.assertEqual(count, 1)
        # BOM 不应残留在首个术语里
        self.assertIsNotNone(glossary.find("GPU"))

    def test_missing_file_returns_zero(self) -> None:
        """文件不存在时返回 0 而不是抛异常。"""
        glossary = Glossary()
        self.assertEqual(glossary.load_from_csv(Path("不存在的文件.csv")), 0)

    def test_optional_columns_parsed(self) -> None:
        """第三、四列的大小写与整词设置应被解析。"""
        path = self._write("GPU,图形处理器,是,否\n")
        glossary = Glossary()
        glossary.load_from_csv(path)
        entry = glossary.find("GPU")
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertTrue(entry.case_sensitive)
        self.assertFalse(entry.whole_word)

    def test_language_columns_are_scoped(self) -> None:
        path = self._write("原文,译文,区分大小写,整词匹配,源语言,目标语言\nGPU,图形处理器,是,是,en,zh\n")
        glossary = Glossary()
        self.assertEqual(glossary.load_from_csv(path), 1)
        self.assertEqual(glossary.substitute("GPU", "en", "zh").hit_count, 1)
        self.assertEqual(glossary.substitute("GPU", "fr", "zh").hit_count, 0)
        self.assertEqual(glossary.substitute("GPU", "en", "ja").hit_count, 0)


class TestCsvExport(unittest.TestCase):
    """CSV 导出。"""

    def test_roundtrip_through_csv(self) -> None:
        """导出后再导入应得到等价的术语表。"""
        original = Glossary(
            [
                GlossaryEntry("GPU", "图形处理器"),
                GlossaryEntry("machine learning", "机器学习"),
            ]
        )
        target = Path(tempfile.mkdtemp()) / "glossary.csv"
        self.assertTrue(original.save_to_csv(target))

        restored = Glossary()
        count = restored.load_from_csv(target)
        self.assertEqual(count, 2)
        self.assertIsNotNone(restored.find("GPU"))
        self.assertIsNotNone(restored.find("machine learning"))


class TestGlossaryMisc(unittest.TestCase):
    """其他行为。"""

    def test_version_increments_on_change(self) -> None:
        """术语表变更应递增版本号。

        版本号参与推理缓存键，因此递增是缓存自动失效的机制保障。
        """
        glossary = Glossary()
        first = glossary.version
        glossary.set_entries([GlossaryEntry("A", "甲")])
        self.assertGreater(glossary.version, first)

    def test_add_appends(self) -> None:
        """追加术语。"""
        glossary = Glossary()
        glossary.add("GPU", "图形处理器")
        self.assertEqual(len(glossary), 1)

    def test_empty_entries_rejected(self) -> None:
        """原文或译文为空的条目应被丢弃。"""
        glossary = Glossary()
        glossary.set_entries([GlossaryEntry("", "空"), GlossaryEntry("空", "")])
        self.assertEqual(len(glossary), 0)

    def test_stats_text(self) -> None:
        """状态描述文本。"""
        self.assertIn("未加载", Glossary().stats_text())
        glossary = Glossary([GlossaryEntry("A", "甲")])
        self.assertIn("1", glossary.stats_text())

    def test_create_glossary_disabled(self) -> None:
        """配置关闭时返回空术语表。"""

        class Config:
            glossary_enabled = False
            glossary_path = ""
            glossary_case_sensitive = False

        self.assertTrue(create_glossary(Config()).is_empty)

    def test_create_glossary_with_path(self) -> None:
        """配置指定路径时应自动加载。"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as handle:
            handle.write("GPU,图形处理器\n")

        class Config:
            glossary_enabled = True
            glossary_path = handle.name
            glossary_case_sensitive = False

        glossary = create_glossary(Config())
        self.assertEqual(len(glossary), 1)

    def test_builtin_loaded_by_default(self) -> None:
        class Config:
            glossary_enabled = True
            glossary_mode = ""
            glossary_path = ""
            glossary_case_sensitive = False
            target_language = "zh"

        glossary = create_glossary(Config())
        result = glossary.substitute("machine learning and API", "en", "zh")
        self.assertEqual(result.hit_count, 2)
        self.assertIn("机器学习", result.replacements.values())
        self.assertIn("应用程序接口", result.replacements.values())

    def test_builtin_language_pair_isolated(self) -> None:
        glossary = load_builtin_glossary("zh")
        french = glossary.substitute("intelligence artificielle", "fr", "zh")
        english = glossary.substitute("intelligence artificielle", "en", "zh")
        self.assertEqual(french.hit_count, 1)
        self.assertIn("人工智能", french.replacements.values())
        self.assertEqual(english.hit_count, 0)
        self.assertEqual(glossary.substitute("人工智能", "zh", "zh").hit_count, 0)

    def test_builtin_unsegmented_source_matches_in_sentence(self) -> None:
        glossary = load_builtin_glossary("en")
        result = glossary.substitute("这是人工智能技术。", "zh", "en")
        self.assertEqual(result.hit_count, 1)
        self.assertIn("artificial intelligence", result.replacements.values())

    def test_builtin_invariant_terms_cover_low_resource_languages(self) -> None:
        glossary = load_builtin_glossary("zh")
        result = glossary.substitute("Python", "ak", "zh")
        self.assertEqual(result.hit_count, 1)
        self.assertEqual(list(result.replacements.values()), ["Python"])

    def test_existing_custom_path_remains_selected(self) -> None:
        path = Path(tempfile.mkdtemp()) / "customer.csv"
        path.write_text("GPU,图形处理器\n", encoding="utf-8")

        class Config:
            glossary_enabled = True
            glossary_mode = ""  # Existing settings have no mode field.
            glossary_path = str(path)
            glossary_case_sensitive = False
            target_language = "zh"

        glossary = create_glossary(Config())
        self.assertEqual(glossary.label, "自定义术语库")
        self.assertEqual(glossary.substitute("GPU", "en", "zh").hit_count, 1)

    def test_all_shipped_subject_packs_translate_to_chinese(self) -> None:
        cases = {
            "software_ai": ("machine learning", "机器学习"),
            "medicine": ("hypertension", "高血压"),
            "biology": ("cell membrane", "细胞膜"),
            "history": ("Industrial Revolution", "工业革命"),
            "physics": ("quantum mechanics", "量子力学"),
            "chemistry": ("covalent bond", "共价键"),
            "mathematics": ("linear algebra", "线性代数"),
        }
        for domain, (source, target) in cases.items():
            with self.subTest(domain=domain):
                glossary = load_builtin_glossary("zh", domain=domain)
                result = glossary.substitute(source, "en", "zh")
                self.assertEqual(result.hit_count, 1)
                self.assertEqual(list(result.replacements.values()), [target])

    def test_subject_pack_isolation_and_config_selection(self) -> None:
        medical = load_builtin_glossary("zh", domain="medicine")
        self.assertEqual(medical.substitute("cell membrane", "en", "zh").hit_count, 0)
        self.assertEqual(medical.substitute("hypertension", "fr", "zh").hit_count, 0)

        class Config:
            glossary_enabled = True
            glossary_mode = "builtin"
            glossary_domain = "biology"
            glossary_path = ""
            glossary_case_sensitive = False
            target_language = "zh"

        selected = create_glossary(Config())
        self.assertEqual(selected.substitute("cell membrane", "en", "zh").hit_count, 1)
        self.assertEqual(selected.substitute("hypertension", "en", "zh").hit_count, 0)

    def test_invalid_domain_cannot_escape_bundled_directory(self) -> None:
        from pingyitong.utils.paths import builtin_glossary_path

        with self.assertRaises(ValueError):
            builtin_glossary_path("../other")

    def test_medical_surgery_refers_to_procedure_not_specialty(self) -> None:
        import json

        from pingyitong.utils.paths import builtin_glossary_path

        data = json.loads(builtin_glossary_path("medicine").read_text(encoding="utf-8"))
        surgery = next(c for c in data["concepts"] if c["id"] == "surgery")
        self.assertEqual(surgery["wikidata_id"], "Q600236")
        self.assertIn("medical procedure", surgery["description_en"])
        result = load_builtin_glossary("zh", domain="medicine").substitute(
            "surgery", "en", "zh"
        )
        self.assertEqual(list(result.replacements.values()), ["外科手术"])

    def test_single_cjk_glyph_does_not_replace_part_of_longer_word(self) -> None:
        glossary = load_builtin_glossary("en", domain="chemistry")
        self.assertEqual(glossary.substitute("核酸", "zh", "en").hit_count, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
