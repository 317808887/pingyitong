"""Uncertain automatic guesses must not suppress real translation requests.

The recording pool tests routing only, not OCR or model translation quality.
"""
from __future__ import annotations

import unittest

from pingyitong.components.capture.geometry import Rect
from pingyitong.config.schema import AppConfig
from pingyitong.services.language import detect_language
from pingyitong.services.text_preprocess import Paragraph
from tests.test_translation_request_context import _RequestPool
from tests.test_translation_service import block, make_frame, make_service


class TestAutomaticSameLanguageSkip(unittest.TestCase):
    def _service(self, target="en", source="auto", reply="Welcome"):
        config = AppConfig()
        config.translation.semantic_review = False  # legacy single-pass contract
        config.translation.source_language = source
        config.translation.target_language = target
        config.translation.glossary_enabled = False
        pool = _RequestPool()
        pool.reply = reply
        service, _ = make_service(config, pool=pool)
        return service, pool

    def test_low_confidence_latin_fallback_does_not_skip_english_target(self):
        for text in ("Selamat datang", "Bom dia", "Watu wote"):
            with self.subTest(text=text):
                guess = detect_language(text)
                self.assertEqual(guess.code, "en")
                self.assertFalse(guess.is_confident)
                service, pool = self._service()
                frame = service.translate_frame(make_frame(block(text, 0, 0)))
                self.assertEqual(len(pool.calls), 1)
                self.assertEqual(len(frame.items), 1)
                self.assertEqual(service.last_stats.skipped, 0)

    def test_stored_paragraph_code_cannot_hide_guess_confidence(self):
        service, _ = self._service()
        paragraph = Paragraph(text="Selamat datang", box=Rect(0, 0, 200, 32), language="en")
        self.assertTrue(service._needs_translation(paragraph, "auto", "en", "en"))

    def test_cyrillic_script_confidence_is_not_russian_language_evidence(self):
        for text in ("Відкрийте вікно.", "Адчыніце акно."):
            with self.subTest(text=text):
                guess = detect_language(text)
                self.assertEqual(guess.code, "ru")
                self.assertTrue(guess.is_confident)
                service, pool = self._service(target="ru", reply="Откройте окно.")
                service.translate_frame(make_frame(block(text, 0, 0)))
                self.assertEqual(len(pool.calls), 1)
                self.assertEqual(service.last_stats.skipped, 0)

    def test_confident_latin_guess_still_uses_model(self):
        text = "The window is closed."
        service, pool = self._service()
        service.translate_frame(make_frame(block(text, 0, 0)))
        self.assertTrue(detect_language(text).is_confident)
        self.assertEqual(len(pool.calls), 1)
        self.assertEqual(service.last_stats.skipped, 0)

    def test_han_guess_still_uses_model_for_script_and_language_ambiguity(self):
        for text in ("請關閉視窗。", "図書館", "请关闭窗口。"):
            with self.subTest(text=text):
                service, pool = self._service(target="zh", reply="请关闭窗口。")
                service.translate_frame(make_frame(block(text, 0, 0)))
                self.assertEqual(len(pool.calls), 1)
                self.assertEqual(service.last_stats.skipped, 0)

    def test_unique_kana_and_hangul_same_language_can_still_skip(self):
        for language, text in (("ja", "ウィンドウを閉じてください。"), ("ko", "창을 닫으세요.")):
            with self.subTest(language=language):
                service, pool = self._service(target=language)
                frame = service.translate_frame(make_frame(block(text, 0, 0)))
                self.assertEqual(pool.calls, [])
                self.assertEqual(frame.items[0].text, text)

    def test_explicit_same_language_keeps_user_selection(self):
        for language, text in (("en", "Save"), ("id", "Selamat datang"), ("ru", "Откройте окно.")):
            with self.subTest(language=language):
                service, pool = self._service(target=language, source=language)
                frame = service.translate_frame(make_frame(block(text, 0, 0)))
                self.assertEqual(pool.calls, [])
                self.assertEqual(frame.items[0].text, text)

    def test_code_still_skips_uncertain_guess(self):
        service, pool = self._service()
        service.translate_frame(make_frame(block("cd data && cat file.txt", 0, 0)))
        self.assertEqual(pool.calls, [])


if __name__ == "__main__":
    unittest.main()
