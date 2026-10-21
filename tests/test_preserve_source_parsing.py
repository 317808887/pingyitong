"""Comparison mode must preserve translated menus and every later clause."""

from __future__ import annotations

import unittest

from pingyitong.components.llm.prompt import parse_response
from pingyitong.config.schema import AppConfig
from pingyitong.constants import SEGMENT_DELIMITER
from pingyitong.services.translation_service import _parse_single_response
from tests.test_cache_and_code_routing_regression import _OfflinePool
from tests.test_translation_service import block, make_frame, make_service


def _comparison_service(*responses):
    config = AppConfig()
    config.translation.semantic_review = False  # legacy single-pass contract
    config.translation.source_language = "en"
    config.translation.target_language = "ko"
    config.translation.keep_source_text = True
    pool = _OfflinePool(*responses)
    return make_service(config, pool=pool)[0], pool


class TestKnownSourceSuffix(unittest.TestCase):
    def test_menus_with_pipes_in_both_translation_and_source(self):
        for source, translated in (
            ("Save | Cancel", "저장 | 취소"),
            ("File | Edit | View", "파일 | 편집 | 보기"),
            ("保存 | 取消", "Save | Cancel"),
        ):
            with self.subTest(source=source):
                self.assertEqual(
                    parse_response(f"{translated} | {source}", 1, True, [source]),
                    [translated],
                )
                self.assertEqual(parse_response(translated, 1, True, [source]), [translated])

    def test_only_complete_known_source_is_removed(self):
        for raw in ("저장 | 취소 | Save", "저장 | 취소 | Save | Exit", "저장 | 취소"):
            with self.subTest(raw=raw):
                self.assertEqual(parse_response(raw, 1, True, ["Save | Cancel"]), [raw])
        self.assertEqual(parse_response("저장 | 취소", 1, True), ["저장 | 취소"])
        self.assertEqual(parse_response("저장 | 취소 | Save | Cancel", 1, False,
                                        ["Save | Cancel"]), ["저장 | 취소 | Save | Cancel"])

    def test_known_suffix_tolerates_separator_spacing_without_guessing(self):
        self.assertEqual(parse_response("저장 | 취소|Save | Cancel", 1, True,
                                        ["Save | Cancel"]), ["저장 | 취소"])
        self.assertEqual(parse_response("저장 | 취소 |  Save | Cancel  ", 1, True,
                                        ["Save | Cancel"]), ["저장 | 취소"])
        self.assertEqual(parse_response("저장 | 취소 Save | Cancel", 1, True,
                                        ["Save | Cancel"]), ["저장 | 취소 Save | Cancel"])
        self.assertIsNone(parse_response("| Save | Cancel", 1, True, ["Save | Cancel"]))

    def test_real_provider_segment_marker_before_exact_comparison_source(self):
        for translated in ("保存 | 取消", "저장 | 취소"):
            raw = translated + " <<<SEG>>> Save | Cancel"
            with self.subTest(translated=translated):
                self.assertEqual(_parse_single_response(raw, "Save | Cancel", True), [translated])
                self.assertEqual(parse_response(raw, 1, False, ["Save | Cancel"]), [raw])
        raw = "저장 | 취소 <<<SEG>>> Save | Exit"
        self.assertEqual(parse_response(raw, 1, True, ["Save | Cancel"]), [raw])
        raw = "저장 | 취소<<<SEG>>>Save | Cancel<<<SEG>>>열기 | 닫기<<<SEG>>>Open | Close"
        self.assertIsNone(parse_response(raw, 2, True, ["Save | Cancel", "Open | Close"]))

    def test_batch_suffixes_use_corresponding_source_and_strict_count(self):
        sources = ["Save | Cancel", "Open | Close"]
        raw = f"저장 | 취소 | {sources[0]}{SEGMENT_DELIMITER}열기 | 닫기 | {sources[1]}"
        self.assertEqual(parse_response(raw, 2, True, sources), ["저장 | 취소", "열기 | 닫기"])
        # A source from another block must not be mistaken for this block's suffix.
        swapped = f"저장 | 취소 | {sources[1]}{SEGMENT_DELIMITER}열기 | 닫기 | {sources[0]}"
        self.assertEqual(parse_response(swapped, 2, True, sources),
                         ["저장 | 취소 | Open | Close", "열기 | 닫기 | Save | Cancel"])
        self.assertIsNone(parse_response(raw + SEGMENT_DELIMITER + "추가", 2, True, sources))
        self.assertIsNone(parse_response(raw, 2, True, sources[:1]))

    def test_per_segment_fences_do_not_hide_the_known_suffix(self):
        raw = f"```저장 | 취소 | Save | Cancel```{SEGMENT_DELIMITER}```열기 | 닫기 | Open | Close```"
        self.assertEqual(parse_response(raw, 2, True, ["Save | Cancel", "Open | Close"]),
                         ["저장 | 취소", "열기 | 닫기"])


class TestSingleSourceClauseSuffixes(unittest.TestCase):
    def test_each_clause_has_its_exact_source_and_internal_menu_pipes(self):
        source = "Save | Cancel. Open | Close."
        raw = "저장 | 취소. | Save | Cancel.<<<SEG>>>열기 | 닫기. | Open | Close."
        self.assertEqual(_parse_single_response(raw, source, True), ["저장 | 취소. 열기 | 닫기."])

    def test_full_source_after_last_clause_does_not_discard_that_clause(self):
        source = "Save | Cancel. Open | Close."
        raw = "저장 | 취소.<<<SEG>>>열기 | 닫기. | " + source
        self.assertEqual(_parse_single_response(raw, source, True), ["저장 | 취소. 열기 | 닫기."])

    def test_each_part_can_copy_full_source_or_its_corresponding_clause(self):
        source = "Save | Cancel. Open | Close."
        for first_source, second_source in (
            (source, source),
            (source, "Open | Close."),
            ("Save | Cancel.", source),
        ):
            with self.subTest(first=first_source, second=second_source):
                raw = f"저장 | 취소. | {first_source}<<<SEG>>>열기 | 닫기. | {second_source}"
                self.assertEqual(_parse_single_response(raw, source, True),
                                 ["저장 | 취소. 열기 | 닫기."])

    def test_cjk_source_clauses_without_whitespace_are_exactly_matched(self):
        source = "保存文件。关闭窗口。"
        raw = "파일을 저장하세요. | 保存文件。<<<SEG>>>창을 닫으세요. | 关闭窗口。"
        self.assertEqual(_parse_single_response(raw, source, True),
                         ["파일을 저장하세요. 창을 닫으세요."])

    def test_unmatched_clause_suffix_is_kept_for_language_gate(self):
        raw = "저장 | 취소. | Open | Close.<<<SEG>>>열기 | 닫기. | Save | Cancel."
        self.assertEqual(_parse_single_response(raw, "Save | Cancel. Open | Close.", True),
                         ["저장 | 취소. | Open | Close. 열기 | 닫기. | Save | Cancel."])
        # The literal protocol token in a source is not a sentence split.
        source = "Save<<<SEG>>>Cancel"
        self.assertEqual(_parse_single_response("저장<<<SEG>>>취소 | " + source, source, True),
                         ["저장<<<SEG>>>취소"])


class TestComparisonModeProductionRoutes(unittest.TestCase):
    def test_single_text_and_frame_cache_keep_all_menu_items(self):
        for frame_mode in (False, True):
            with self.subTest(frame=frame_mode):
                service, pool = _comparison_service("저장 | 취소 | Save | Cancel")
                source = "Save | Cancel"
                for _ in range(2):
                    if frame_mode:
                        rendered = service.translate_frame(make_frame(block(source, 0, 0)))
                        self.assertEqual([item.text for item in rendered.items], ["저장 | 취소"])
                        self.assertEqual(rendered.items[0].source_text, source)
                    else:
                        self.assertEqual(service.translate_text(source), "저장 | 취소")
                self.assertEqual(len(pool.calls), 1)
                self.assertNotIn("append the original", pool.calls[0][1])

    def test_multi_frame_and_cache_keep_each_block_and_source(self):
        raw = "저장 | 취소 | Save | Cancel<<<SEG>>>열기 | 닫기 | Open | Close"
        service, pool = _comparison_service(raw)
        frame = make_frame(block("Save | Cancel", 0, 0), block("Open | Close", 0, 100))
        for _ in range(2):
            rendered = service.translate_frame(frame)
            self.assertEqual([(item.text, item.source_text) for item in rendered.items],
                             [("저장 | 취소", "Save | Cancel"), ("열기 | 닫기", "Open | Close")])
        self.assertEqual(len(pool.calls), 1)
        self.assertNotIn("append the original", pool.calls[0][1])

    def test_misaligned_batch_retries_individual_blocks_without_truncation(self):
        service, pool = _comparison_service(
            "저장<<<SEG>>>취소<<<SEG>>>열기",
            "저장 | 취소 | Save | Cancel", "열기 | 닫기 | Open | Close",
        )
        rendered = service.translate_frame(make_frame(
            block("Save | Cancel", 0, 0), block("Open | Close", 0, 100),
        ))
        self.assertEqual([item.text for item in rendered.items], ["저장 | 취소", "열기 | 닫기"])
        self.assertEqual(len(pool.calls), 3)

    def test_single_wrong_language_correction_keeps_menu_and_all_clauses(self):
        source = "Save | Cancel. Open | Close."
        service, pool = _comparison_service(
            "Save | Cancel. Open | Close.",
            "저장 | 취소. | Save | Cancel.<<<SEG>>>열기 | 닫기. | Open | Close.",
        )
        self.assertEqual(service.translate_text(source), "저장 | 취소. 열기 | 닫기.")
        self.assertEqual(service.translate_text(source), "저장 | 취소. 열기 | 닫기.")
        self.assertEqual(len(pool.calls), 2)

    def test_single_and_multi_correction_batches_use_known_sources(self):
        for sources, raw, expected in (
            (["Save | Cancel. Open | Close."],
             "저장 | 취소. | Save | Cancel.<<<SEG>>>열기 | 닫기. | Open | Close.",
             ["저장 | 취소. 열기 | 닫기."]),
            (["Save | Cancel", "Open | Close"],
             "저장 | 취소 | Save | Cancel<<<SEG>>>열기 | 닫기 | Open | Close",
             ["저장 | 취소", "열기 | 닫기"]),
        ):
            with self.subTest(sources=sources):
                service, pool = _comparison_service(raw)
                self.assertEqual(service._run_correction_batches(sources, "Translate to Korean.",
                                                                 "fast", 128, True), expected)
                self.assertEqual(len(pool.calls), 1)
        service, _ = _comparison_service("저장<<<SEG>>>취소<<<SEG>>>열기")
        self.assertEqual(service._run_correction_batches(["Save", "Open"],
                                                         "Translate to Korean.", "fast", 128, True),
                         [None, None])

    def test_inexact_source_suffix_cannot_hide_foreign_prose_or_enter_cache(self):
        service, _ = _comparison_service("저장 | 취소 | Save | Exit", "저장 | 취소 | Save | Exit")
        rendered = service.translate_frame(make_frame(block("Save | Cancel", 0, 0)))
        self.assertEqual(rendered.items, [])
        self.assertEqual(len(service.cache), 0)


if __name__ == "__main__":
    unittest.main()
