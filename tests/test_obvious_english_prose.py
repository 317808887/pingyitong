"""Bounded wrong-language recovery for Latin targets without a new dependency."""
from __future__ import annotations

import unittest

from pingyitong.config.schema import AppConfig
from pingyitong.services.language import detect_language, has_foreign_prose
from tests.test_mixed_target_language import _Responses
from tests.test_translation_service import block, make_frame, make_service

ENGLISH = "Everyone is born free, and is equally entitled in dignity and rights."


class TestObviousEnglishProse(unittest.TestCase):
    def test_long_english_is_rejected_for_other_latin_targets(self):
        for code in ("sv", "nl", "fy", "ny", "af", "hmn", "xh", "wo"):
            with self.subTest(code=code):
                self.assertTrue(has_foreign_prose(ENGLISH, code))

    def test_real_entitled_to_english_variant_is_rejected(self):
        text = "Everyone is born free, and is equally entitled to respect and rights."
        self.assertTrue(has_foreign_prose(text, "fy"))
        self.assertFalse(has_foreign_prose(text, "en"))
        self.assertFalse(has_foreign_prose(text, "sco"))

    def test_english_and_scots_remain_exempt(self):
        self.assertFalse(has_foreign_prose(ENGLISH, "en"))
        self.assertFalse(has_foreign_prose(ENGLISH, "sco"))

    def test_short_ui_and_proper_names_stay_inconclusive(self):
        for text in ("Open Settings", "Microsoft Windows and Apple", "Save", "FPS 60 CPU 12%", "1234.56"):
            with self.subTest(text=text):
                self.assertFalse(has_foreign_prose(text, "nl"))

    def test_dutch_detected_as_english_is_not_rejected(self):
        text = "Dit is een mooi huis."
        self.assertEqual(detect_language(text).code, "en")
        self.assertTrue(detect_language(text).is_confident)
        self.assertFalse(has_foreign_prose(text, "nl"))
        long_dutch = "Dit is een mooi huis. Dit is een mooi huis. Dit is een mooi huis. Dit is een mooi huis."
        self.assertFalse(has_foreign_prose(long_dutch, "nl"))

    def test_long_near_language_prose_does_not_use_confidence_alone(self):
        for code, text in (
            ("nl", "Alle mensen worden vrij en gelijk in waardigheid en rechten geboren."),
            ("fy", "Alle minsken wurde frij en gelyk yn weardichheid en rjochten berne."),
            ("sv", "Alla människor är födda fria och lika i värde och rättigheter."),
            ("af", "Alle mense word vry en gelyk in waardigheid en regte gebore."),
        ):
            with self.subTest(code=code):
                self.assertFalse(has_foreign_prose(text, code))

    def test_glossary_exception_removes_only_explicit_matched_term(self):
        protected = (ENGLISH,)
        self.assertFalse(has_foreign_prose("Voorbeeld: " + ENGLISH, "nl", "示例：" + ENGLISH, protected))
        self.assertTrue(has_foreign_prose(ENGLISH, "nl", "示例：" + ENGLISH))

    def test_long_source_identifiers_are_exempt_but_natural_english_is_not(self):
        identifiers = "C:/example/long_directory/Everyone_is_born_free_and_is_equally_entitled_in_dignity_and_rights/file.py"
        self.assertFalse(has_foreign_prose(identifiers, "nl", identifiers))
        self.assertTrue(has_foreign_prose(ENGLISH, "nl", ENGLISH))

    def test_sparse_english_tokens_in_long_text_remain_inconclusive(self):
        text = "and is in " + "abcdefghijklmnopqrstuvwx " * 10
        self.assertFalse(has_foreign_prose(text, "nl"))

    def test_existing_other_script_gate_still_rejects(self):
        self.assertTrue(has_foreign_prose("Всі люди народжуються вільними", "sv"))
        self.assertTrue(has_foreign_prose("닫으세요. Click the Settings button.", "ko"))

    def test_common_orthographic_apostrophes_are_neutral_not_foreign_script(self):
        for code, text in (
            ("uz", "Har bir inson, qayerda boʻlishidan qatʼi nazar, oʻzining huquq subyekti sifatida tan olinishi huquqiga egadir."),
            ("gn", "Mayma yvypóra oguereko derecho ojehechakuaávo ichupe opaite handárupi haʼeha peteĩ máva leiduéra renondépe."),
            ("nv", "Tsídí tʼáá ájítʼé bee háhazʼą́ iináʼázhdoolííłgo áádóó ʼałdó tʼáá yiní kʼehgo hołdzilgo hołhazʼą́ądoo."),
            ("haw", "Hānau kūʻokoʻa ʻia nā kānaka a pau, a ua kau like nā pono a me ka hanohano."),
        ):
            with self.subTest(code=code):
                self.assertFalse(has_foreign_prose(text, code))
        self.assertTrue(has_foreign_prose("ʼʻქართული ტექსტი", "nl"))
        self.assertTrue(has_foreign_prose("ʼʻ한국어 텍스트", "sv"))

    def test_service_recovers_latin_wrong_language_before_publishing(self):
        config = AppConfig()
        config.translation.semantic_review = False  # legacy single-pass contract
        config.translation.source_language = "zh"
        config.translation.target_language = "sv"
        config.schedule.mode = "manual"
        corrected = "Alla människor är födda fria och lika i värde och rättigheter."
        pool = _Responses(ENGLISH, corrected)
        service, _ = make_service(config, pool=pool)
        self.assertEqual(service.translate_text("人人生而自由，在尊严和权利上一律平等。"), corrected)
        self.assertEqual(len(pool.calls), 2)

    def test_repeated_english_is_withheld_after_bounded_recovery(self):
        config = AppConfig()
        config.translation.semantic_review = False  # legacy single-pass contract
        config.translation.source_language = "zh"
        config.translation.target_language = "sv"
        config.schedule.mode = "manual"
        pool = _Responses(ENGLISH, ENGLISH)
        service, _ = make_service(config, pool=pool)
        frame = service.translate_frame(make_frame(block("人人生而自由，在尊严和权利上一律平等。", 0, 0)))
        self.assertEqual(frame.items, [])
        self.assertEqual(len(pool.calls), 2)


class TestPreservedSourceWordExemption(unittest.TestCase):
    """译文里孤立保留的源文词不算"整段外来散文"。

    模型保留源文名词（``8 workers`` → ``8 个 worker``）是正确译文，
    此前被守卫整段误杀；而源文里没有的词（中文段里混进 ``close``）
    与多个残留词仍照常拦截。
    """

    def test_single_preserved_word_passes(self):
        self.assertFalse(has_foreign_prose("8 个 worker", "zh", "8 workers online"))
        self.assertFalse(has_foreign_prose("8 个 workers 都在线", "zh", "8 workers online"))

    def test_foreign_word_not_in_source_is_still_rejected(self):
        self.assertTrue(has_foreign_prose("这是一个 close 操作", "zh", "这是一个操作"))
        self.assertTrue(has_foreign_prose("这是一个 close the window 操作", "zh", "这是一个操作"))

    def test_multiple_residual_words_are_still_rejected(self):
        self.assertTrue(has_foreign_prose("请 close the window", "zh", "请关闭窗口"))


if __name__ == "__main__":
    unittest.main()
