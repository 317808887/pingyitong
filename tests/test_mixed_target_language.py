"""Target-language gates and bounded recovery; no external service or weights."""

from __future__ import annotations

import unittest

from pingyitong.config.schema import AppConfig
from pingyitong.constants import SEGMENT_DELIMITER
from pingyitong.services.glossary_service import Glossary, GlossaryEntry
from pingyitong.services.language import has_foreign_prose
from tests.test_translation_service import block, make_frame, make_service


class _Responses:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def translate(self, model_id, prompt, system_prompt, max_tokens=None):
        self.calls.append((prompt, system_prompt))
        return self.responses.pop(0) if self.responses else ""

    def is_available(self, model_id):
        return True

    def any_available(self):
        return True


def _korean_service(pool, glossary=None):
    config = AppConfig()
    config.translation.semantic_review = False  # legacy single-pass contract
    config.translation.source_language = "auto"
    config.translation.target_language = "ko"
    config.schedule.mode = "manual"
    return make_service(config, glossary=glossary, pool=pool)[0]


class TestTargetScripts(unittest.TestCase):
    def test_korean_does_not_hide_foreign_prose_behind_its_majority_script(self):
        for text in (
            "설치가 완료되었습니다，请等待程序完成。",
            "설치가 완료되었습니다. Please close the window.",
            "설치가 완료되었습니다 close",
            "The file is open.", "Файл открыт", "الملف مفتوح", "これは日本語です",
        ):
            with self.subTest(text=text):
                self.assertTrue(has_foreign_prose(text, "ko", "文件已打开。"))

    def test_source_technical_tokens_and_brands_are_preserved(self):
        source = (
            "OpenAI GPU SHA256 saves myFunction foo_bar to "
            "C:/用户/报告.json at https://example.com/help in 300ms and 3.84GB."
        )
        candidate = (
            "OpenAI GPU SHA256에서 myFunction foo_bar를 C:/用户/报告.json에 저장합니다. "
            "https://example.com/help 300ms 3.84GB"
        )
        self.assertFalse(has_foreign_prose(candidate, "ko", source))
        self.assertTrue(has_foreign_prose("파일을 닫으세요. close the window", "ko", source))

    def test_uppercase_ui_words_are_not_exempted_as_acronyms(self):
        for source in ("OPEN FILE", "SAVE NOW"):
            with self.subTest(source=source):
                self.assertTrue(has_foreign_prose(source, "ko", source))
        self.assertFalse(has_foreign_prose("GPU API OCR가 준비되었습니다.", "ko", "GPU API OCR"))

    def test_path_or_url_sentence_punctuation_is_not_part_of_its_name(self):
        for token in (
            "C:/reports/result.json", "C:\\reports\\result.json",
            "https://example.com/status", "https://example.com/status?q=1",
            "www.example.com/status", "../reports/result.json",
        ):
            for punctuation in (".", ",", ":", ";", "?", "!", ")", "]", "}", "\"", "'"):
                with self.subTest(token=token, punctuation=punctuation):
                    source = "Save the file at " + token + punctuation
                    candidate = "파일을 " + token + "에 저장하세요."
                    self.assertFalse(has_foreign_prose(candidate, "ko", source))
        self.assertTrue(has_foreign_prose(
            "파일을 C:/reports/other.json에 저장하세요.", "ko",
            "Save the file at C:/reports/result.json.",
        ))

    def test_language_table_scripts_are_used_not_a_four_language_allowlist(self):
        for target, text in (
            ("zh-TW", "這是繁體中文。"), ("uk", "Файл відкрито"),
            ("ar", "الملف مفتوح"), ("hi", "फ़ाइल खुली है"),
            ("th", "เปิดไฟล์แล้ว"), ("el", "Το αρχείο είναι ανοιχτό"),
        ):
            with self.subTest(target=target):
                self.assertFalse(has_foreign_prose(text, target))
                self.assertTrue(has_foreign_prose(text + " 한국어 미번역", target))


class TestMixedLanguagePublishing(unittest.TestCase):
    def test_mixed_korean_source_must_be_translated(self):
        pool = _Responses("설치가 완료되었습니다. 기다려 주세요.")
        service = _korean_service(pool)
        frame = service.translate_frame(make_frame(block("설치가 완료되었습니다，请等待程序完成。", 0, 0)))
        self.assertEqual(len(pool.calls), 1)
        self.assertEqual(frame.items[0].text, "설치가 완료되었습니다. 기다려 주세요.")

    def test_legitimate_korean_source_is_still_skipped(self):
        pool = _Responses()
        service = _korean_service(pool)
        frame = service.translate_frame(make_frame(block("파일이 열려 있습니다.", 0, 0)))
        self.assertEqual(pool.calls, [])
        self.assertEqual(frame.items[0].text, "파일이 열려 있습니다.")

    def test_foreign_prose_is_recovered_before_publication_and_cache(self):
        for bad in (
            "파일이 열려 있습니다，请等待。", "The file is open.",
            "الملف مفتوح", "Файл открыт", "파일이 열려 있습니다. close the window.",
        ):
            with self.subTest(bad=bad):
                pool = _Responses(bad, "파일이 열려 있습니다.")
                service = _korean_service(pool)
                source = "文件已打开。"
                frame = service.translate_frame(make_frame(block(source, 0, 0)))
                self.assertEqual(frame.items[0].text, "파일이 열려 있습니다.")
                second = service.translate_frame(make_frame(block(source, 0, 0)))
                self.assertEqual(second.items[0].text, frame.items[0].text)
                self.assertEqual(len(pool.calls), 2)

    def test_stale_memory_cache_is_a_miss_and_retranslated(self):
        pool = _Responses("파일이 열려 있습니다.")
        service = _korean_service(pool)
        service.cache.put(service._cache_key("文件已打开。", "zh", "ko"), "文件已打开。")
        frame = service.translate_frame(make_frame(block("文件已打开。", 0, 0)))
        self.assertEqual(frame.items[0].text, "파일이 열려 있습니다.")
        self.assertEqual(len(pool.calls), 1)
        self.assertFalse(frame.items[0].from_cache)

    def test_stale_prefetch_is_a_miss_and_retranslated(self):
        class Lookup:
            def lookup(self, text, context):
                return "The file is open."

        pool = _Responses("파일이 열려 있습니다.")
        service = _korean_service(pool)
        service.set_prefetch_lookup(Lookup())
        frame = service.translate_frame(make_frame(block("文件已打开。", 0, 0)))
        self.assertEqual(frame.items[0].text, "파일이 열려 있습니다.")
        self.assertEqual(len(pool.calls), 1)

    def test_text_translation_checks_cached_and_recovered_content(self):
        pool = _Responses("The file is open.", "파일이 열려 있습니다.")
        service = _korean_service(pool)
        source = "文件已打开。"
        service.cache.put(service._cache_key(source, "auto", "ko"), "الملف مفتوح")
        self.assertEqual(service.translate_text(source), "파일이 열려 있습니다.")
        self.assertEqual(service.translate_text(source), "파일이 열려 있습니다.")
        self.assertEqual(len(pool.calls), 2)

    def test_literal_glossary_recovery_cannot_bypass_language_gate(self):
        pool = _Responses("준비되었습니다.", "GPU is ready.")
        glossary = Glossary([GlossaryEntry("GPU", "GPU")])
        service = _korean_service(pool, glossary)
        frame = service.translate_frame(make_frame(block("The GPU is ready.", 0, 0)))
        self.assertEqual(frame.items, [])
        self.assertEqual(service.last_output(), [])
        self.assertEqual(len(service.cache), 0)

    def test_explicit_source_comparison_is_removed_before_language_gate(self):
        pool = _Responses("파일이 열렸습니다. | 文件已打开。")
        service = _korean_service(pool)
        service._config.translation.keep_source_text = True
        frame = service.translate_frame(make_frame(block("文件已打开。", 0, 0)))
        self.assertEqual(frame.items[0].text, "파일이 열렸습니다.")
        self.assertEqual(frame.items[0].source_text, "文件已打开。")

    def test_unscoped_old_glossary_cannot_inject_chinese_into_korean(self):
        glossary = Glossary([GlossaryEntry("GPU", "图形处理器")])
        pool = _Responses("GPU가 준비되었습니다.")
        service = _korean_service(pool, glossary)
        frame = service.translate_frame(make_frame(block("The GPU is ready.", 0, 0)))
        self.assertEqual(frame.items[0].text, "GPU가 준비되었습니다.")
        self.assertNotIn("[TERM_", pool.calls[0][0])

    def test_failed_wrong_language_never_draws_or_records_a_source_copy(self):
        pool = _Responses("Файл открыт", "Файл открыт")
        service = _korean_service(pool)
        frame = service.translate_frame(make_frame(block("文件已打开。", 0, 0)))
        self.assertEqual(frame.items, [])
        self.assertEqual(service.last_output(), [])
        self.assertEqual(len(service.cache), 0)

    def test_many_bad_segments_need_one_corrective_batch_not_serial_retries(self):
        sources = [f"文件已打开，第{i}行。" for i in range(6)]
        pool = _Responses(
            SEGMENT_DELIMITER.join("The file is open." for _ in sources),
            SEGMENT_DELIMITER.join(f"파일이 열렸습니다. {i}" for i in range(6)),
        )
        service = _korean_service(pool)
        frame = service.translate_frame(make_frame(*[
            block(source, 0, i * 400) for i, source in enumerate(sources)
        ]))
        self.assertEqual(len(pool.calls), 2)
        self.assertEqual(len(frame.items), 6)

    def test_online_refusal_recovery_is_one_batch_not_six_round_trips(self):
        sources = [f"The file is open. {i}" for i in range(6)]
        refusal = "抱歉，我无法完成该翻译，因为违反内容政策。"
        pool = _Responses(
            SEGMENT_DELIMITER.join(refusal for _ in sources),
            SEGMENT_DELIMITER.join(f"文件已打开。{i}" for i in range(6)),
        )
        pool.api_mode_active = lambda: True
        config = AppConfig()
        config.translation.semantic_review = False  # legacy single-pass contract
        config.api.enabled = True
        service, _ = make_service(config, pool=pool)
        frame = service.translate_frame(make_frame(*[
            block(source, 0, i * 400) for i, source in enumerate(sources)
        ]))
        self.assertEqual(len(pool.calls), 2)
        self.assertEqual(len(frame.items), 6)

    def test_wrong_language_across_online_batches_is_corrected_together(self):
        sources = [f"文件已打开，第{i}行。" for i in range(6)]
        wrong = SEGMENT_DELIMITER.join("The file is open." for _ in range(3))
        corrected = [f"파일이 열렸습니다. {i}" for i in range(6)]
        pool = _Responses(wrong, wrong, SEGMENT_DELIMITER.join(corrected))
        pool.api_mode_active = lambda: True
        config = AppConfig()
        config.translation.semantic_review = False  # legacy single-pass contract
        config.translation.target_language = "ko"
        config.api.enabled = True
        config.api.batch_size = 3
        service, _ = make_service(config, pool=pool)
        frame = service.translate_frame(make_frame(*[
            block(source, 0, i * 400) for i, source in enumerate(sources)
        ]))
        self.assertEqual(len(pool.calls), 3)  # two initial batches, one corrective request
        self.assertEqual([item.text for item in frame.items], corrected)
        self.assertEqual(pool.calls[-1][0].split("\n" + SEGMENT_DELIMITER + "\n"), sources)

    def test_refusals_across_online_batches_are_corrected_together(self):
        sources = [f"The file is open. {i}" for i in range(6)]
        refusal = "抱歉，我无法完成该翻译，因为违反内容政策。"
        wrong = SEGMENT_DELIMITER.join(refusal for _ in range(3))
        corrected = [f"文件已打开。{i}" for i in range(6)]
        pool = _Responses(wrong, wrong, SEGMENT_DELIMITER.join(corrected))
        pool.api_mode_active = lambda: True
        config = AppConfig()
        config.translation.semantic_review = False  # legacy single-pass contract
        config.api.enabled = True
        config.api.batch_size = 3
        service, _ = make_service(config, pool=pool)
        frame = service.translate_frame(make_frame(*[
            block(source, 0, i * 400) for i, source in enumerate(sources)
        ]))
        self.assertEqual(len(pool.calls), 3)
        self.assertEqual([item.text for item in frame.items], corrected)

    def test_path_at_end_of_source_sentence_survives_and_is_cached(self):
        source = "Save the file at C:/reports/result.json."
        translated = "파일을 C:/reports/result.json에 저장하세요."
        pool = _Responses(translated)
        service = _korean_service(pool)
        frame = service.translate_frame(make_frame(block(source, 0, 0)))
        self.assertEqual(frame.items[0].text, translated)
        again = service.translate_frame(make_frame(block(source, 0, 0)))
        self.assertEqual(again.items[0].text, translated)
        self.assertTrue(again.items[0].from_cache)
        self.assertEqual(len(pool.calls), 1)


if __name__ == "__main__":
    unittest.main()
