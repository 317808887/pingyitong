"""Mixed-language clauses must remain one valid translation without extra RTT."""
from __future__ import annotations

import unittest

from pingyitong.config.schema import AppConfig
from pingyitong.services.translation_service import _parse_single_response
from tests.test_cache_and_code_routing_regression import _OfflinePool
from tests.test_translation_service import block, make_frame, make_service


class TestSingleResponseMarkers(unittest.TestCase):
    def test_complete_clauses_are_kept_in_text_frame_and_cache(self):
        for target, source, raw, expected in [
            ("ko", "安装完成后关闭窗口。Click the Settings button.",
             "설치 완료 후 창을 닫습니다.<<<SEG>>>설정 버튼을 클릭합니다.",
             "설치 완료 후 창을 닫습니다. 설정 버튼을 클릭합니다."),
            ("ja", "请保存文件。Then close the window.",
             "ファイルを保存してください。<<<SEG>>>その後、ウィンドウを閉じてください。",
             "ファイルを保存してください。 その後、ウィンドウを閉じてください。"),
        ]:
            for frame_mode in (False, True):
                with self.subTest(target=target, frame=frame_mode):
                    config = AppConfig()
                    config.translation.semantic_review = False  # legacy single-pass contract
                    config.translation.target_language = target
                    pool = _OfflinePool(raw)
                    service = make_service(config, pool=pool)[0]
                    def run(frame_mode=frame_mode, service=service, source=source):
                        if frame_mode:
                            return service.translate_frame(make_frame(block(source, 0, 0))).items[0].text
                        return service.translate_text(source)
                    self.assertEqual(run(), expected)
                    self.assertEqual(run(), expected)
                    self.assertEqual(len(pool.calls), 1)

    def test_foreign_clause_is_still_rejected(self):
        config = AppConfig()
        config.translation.semantic_review = False  # legacy single-pass contract
        config.translation.target_language = "ko"
        pool = _OfflinePool("설치 완료.<<<SEG>>>Close the window.")
        service = make_service(config, pool=pool)[0]
        rendered = service.translate_frame(make_frame(block("Installation complete. Close the window.", 0, 0)))
        self.assertEqual(rendered.items, [])
        self.assertEqual(len(service.cache), 0)

    def test_source_suffix_is_removed_from_each_clause_without_losing_translation(self):
        self.assertEqual(
            _parse_single_response(
                "파일을 저장하세요. | Save the file.<<<SEG>>>창을 닫으세요. | Close the window.",
                "Save the file. Close the window.", True,
            ),
            ["파일을 저장하세요. 창을 닫으세요."],
        )

    def test_single_correction_keeps_all_clauses(self):
        pool = _OfflinePool("설치 완료.<<<SEG>>>창을 닫으세요.")
        service = make_service(pool=pool)[0]
        self.assertEqual(
            service._run_correction_batches(["Installation complete. Close the window."],
                                            "Translate to Korean.", "fast", 128, False),
            ["설치 완료. 창을 닫으세요."],
        )

    def test_literal_source_and_damaged_markers_are_not_removed(self):
        self.assertEqual(_parse_single_response("a<<<SEG>>>b", "a<<<SEG>>>b", False), ["a<<<SEG>>>b"])
        self.assertEqual(_parse_single_response("설치<<<SEG>>", "Install", False), ["설치<<<SEG>>"])
