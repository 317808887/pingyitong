"""语言清单与实测结果的一致性测试。

这个文件的职责是**守住"代码里的实测结论"与"实测原始记录"的一致性**。

为什么有必要：``language_test_results.py`` 里的语言代码是**手工**从实测
脚本输出的 JSON 抄进来的。手工抄写已经出过一次错——漏掉了 ``af``（南非荷兰语）
和 ``am``（阿姆哈拉语）两种确定支持的语言，导致界面少标了两项"确定支持"，
而且因为集合本身仍然自洽（只是少了两个元素），任何自洽性检查都发现不了。

因此这里直接拿 ``docs/language_test_*.json``（实测原始证据）做对照，
让抄写错误无法通过测试。
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import pingyitong.language_test_results as results
from pingyitong.languages import LANGUAGES

#: 仓库根目录：tests/ 的上一级
_ROOT = Path(__file__).resolve().parents[1]

#: 实测原始记录的位置
_EVIDENCE_4B = _ROOT / "docs" / "language_test_4b.json"
_EVIDENCE_1_7B = _ROOT / "docs" / "language_test_1.7b.json"


def _load_verdicts(path: Path) -> dict[str, set[str]]:
    """读取实测 JSON，按档位归集语言代码。

    JSON 结构为 ``{模型名: [ {code, verdict, ...}, ... ]}``。

    :return: ``{"verified": {...}, "likely": {...}, "unsupported": {...}}``
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[str, set[str]] = {"verified": set(), "likely": set(), "unsupported": set()}
    for rows in data.values():
        for row in rows:
            verdict = row.get("verdict", "")
            code = row.get("code", "")
            if code and verdict in grouped:
                grouped[verdict].add(code)
    return grouped


class TestLanguageTable(unittest.TestCase):
    """语言表自身的基本性质。"""

    def test_codes_are_unique(self) -> None:
        """语言代码不得重复（字典键天然唯一，这里防的是表内重复行）。"""
        codes = list(LANGUAGES)
        self.assertEqual(len(codes), len(set(codes)))

    def test_every_entry_is_complete(self) -> None:
        """每种语言都必须有中文名、英文名与脚本信息。"""
        for code, info in LANGUAGES.items():
            with self.subTest(code=code):
                self.assertTrue(info.name, "缺少中文名")
                self.assertTrue(info.english_name, "缺少英文名")
                self.assertTrue(info.script, "缺少文字脚本")

    def test_language_count_matches_documentation(self) -> None:
        """语言数量应与文档中宣称的一致。"""
        self.assertEqual(len(LANGUAGES), 142)


class TestRecordedResultsMatchEvidence(unittest.TestCase):
    """代码里记录的实测结论必须与原始 JSON 完全一致。"""

    def setUp(self) -> None:
        if not _EVIDENCE_4B.exists():
            self.skipTest(f"缺少 4B 实测原始记录：{_EVIDENCE_4B}")
        self.evidence_4b = _load_verdicts(_EVIDENCE_4B)

    def test_verified_4b_matches_evidence(self) -> None:
        """``VERIFIED_4B`` 必须与 JSON 里判为 verified 的完全一致。

        正是这条断言能发现"抄写时漏掉几个代码"的问题。
        """
        recorded = set(results.VERIFIED_4B)
        actual = self.evidence_4b["verified"]
        self.assertEqual(
            recorded - actual,
            set(),
            "代码中记录为确定支持、但实测 JSON 里没有的语言（抄写多出）",
        )
        self.assertEqual(
            actual - recorded,
            set(),
            "实测 JSON 判为确定支持、但代码里漏掉的语言（抄写遗漏）",
        )

    def test_likely_4b_matches_evidence(self) -> None:
        """``LIKELY_4B`` 同样必须与 JSON 一致。"""
        self.assertEqual(set(results.LIKELY_4B), self.evidence_4b["likely"])

    def test_verified_1_7b_matches_evidence(self) -> None:
        """1.7B 的结论也必须有原始记录支撑。"""
        if not _EVIDENCE_1_7B.exists():
            self.skipTest(f"缺少 1.7B 实测原始记录：{_EVIDENCE_1_7B}")
        evidence = _load_verdicts(_EVIDENCE_1_7B)
        self.assertEqual(set(results.VERIFIED_1_7B), evidence["verified"])
        self.assertEqual(set(results.LIKELY_1_7B), evidence["likely"])

    def test_all_recorded_codes_exist_in_language_table(self) -> None:
        """实测结论里出现的代码都必须在语言表中存在。"""
        known = set(LANGUAGES)
        for name in ("VERIFIED_1_7B", "LIKELY_1_7B", "VERIFIED_4B", "LIKELY_4B"):
            group = getattr(results, name)
            with self.subTest(group=name):
                self.assertEqual(
                    set(group) - known,
                    set(),
                    f"{name} 中存在语言表里没有的代码",
                )

    def test_no_code_is_both_verified_and_likely(self) -> None:
        """同一档位集合之间不得重叠，否则计数会重复。"""
        for verified, likely in (
            (results.VERIFIED_1_7B, results.LIKELY_1_7B),
            (results.VERIFIED_4B, results.LIKELY_4B),
        ):
            with self.subTest(verified=len(verified)):
                self.assertEqual(set(verified) & set(likely), set())


class TestStatusAndSummary(unittest.TestCase):
    """档位判定与摘要的一致性。"""

    def test_counts_always_sum_to_candidates(self) -> None:
        """三档计数之和必须等于候选总数。

        这是最容易出错的地方：早先手工维护"不支持"清单时，
        出现过三个数字加起来 184 > 候选 142 的矛盾。
        """
        candidates = frozenset(LANGUAGES)
        stats = results.counts(candidates)
        self.assertEqual(sum(stats.values()), len(candidates))

    def test_summary_numbers_match_per_language_status(self) -> None:
        """摘要里的数字必须与逐语言查询的结果一致。

        界面上的标记来自 :func:`status`，摘要若另行计算就可能出现
        "摘要说 109 种确定支持，但列表里数出来是 111 个勾"。
        """
        candidates = frozenset(LANGUAGES)
        stats = results.counts(candidates)
        text = results.summary(candidates)

        self.assertIn(f"确定支持 {stats['verified']} 种", text)
        self.assertIn(f"疑似支持 {stats['likely']} 种", text)
        self.assertIn(f"不支持 {stats['unsupported']} 种", text)

    def test_dual_model_takes_effect(self) -> None:
        """4B 结果已记录后，判定必须考虑两个模型。"""
        self.assertTrue(results.FULLY_TESTED, "4B 实测结果应已记录")
        # 只有两个模型都确定支持，才对外称"确定支持"
        both = set(results.VERIFIED_1_7B) & set(results.VERIFIED_4B)
        self.assertEqual(set(results.VERIFIED_BOTH), both)

    def test_status_is_total(self) -> None:
        """每种候选语言都必须能给出一个档位，不得漏判。"""
        valid = {"verified", "likely", "unsupported"}
        for code in LANGUAGES:
            with self.subTest(code=code):
                self.assertIn(results.status(code), valid)

    def test_languages_failing_both_models_are_unsupported(self) -> None:
        """两个模型都未通过的语言才被标为"不支持"。

        这是诚实标注的核心：只要有一个模型实测可用，就不该对用户说
        "不支持"——那会让人误以为软件不支持这门语言。
        """
        candidates = frozenset(LANGUAGES)
        unsupported = results.unsupported_for(
            candidates,
            results.VERIFIED_ANY,
            results.LIKELY_1_7B,
            results.LIKELY_4B,
        )
        for code in LANGUAGES:
            with self.subTest(code=code):
                if code in unsupported:
                    self.assertEqual(results.status(code), "unsupported")
                else:
                    self.assertIn(results.status(code), {"verified", "likely"})


if __name__ == "__main__":
    unittest.main()
