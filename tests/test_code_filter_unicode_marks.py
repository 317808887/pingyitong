"""Combining vowels in prose must not route a paragraph around translation."""

from __future__ import annotations

import unicodedata
import unittest

from pingyitong.config.schema import AppConfig
from pingyitong.services.code_filter import looks_like_code
from tests.test_translation_service import block, make_frame, make_service

_TAMIL_OCR = (
    "மனிதப் பிறிவியினர் சகலரும் சுதந்திரமாகவேபிறக்கின்றனர்; அவர்கள் மதிப்பிலும்,"
    "உரிமைகளிலும் சமமானவர்கள், அவர்கள்நியாயத்தையும் மனச்சாட்சியையும்இயற்பண்பாகப் பெற்றவர்கள்."
)


class _RecordingPool:
    def __init__(self):
        self.calls = []

    def translate(self, model_id, prompt, system_prompt, max_tokens=None):
        self.calls.append((prompt, system_prompt))
        return "The text is translated."

    def is_available(self, model_id):
        return True

    def any_available(self):
        return True


class TestCombiningMarksInProse(unittest.TestCase):
    def test_real_tamil_ocr_is_sent_to_translation_in_manual_and_auto_source(self):
        self.assertFalse(looks_like_code(_TAMIL_OCR))
        for source_language in ("ta", "auto"):
            with self.subTest(source_language=source_language):
                config = AppConfig()
                config.translation.semantic_review = False  # legacy single-pass contract
                config.translation.source_language = source_language
                config.translation.target_language = "en"
                pool = _RecordingPool()
                service, _ = make_service(config, pool=pool)
                frame = service.translate_frame(make_frame(block(_TAMIL_OCR, 0, 0)))
                self.assertEqual(len(pool.calls), 1)
                self.assertEqual(frame.items[0].text, "The text is translated.")

    def test_independent_prose_with_vowels_in_other_scripts_is_not_code(self):
        # Literal excerpts from the independently sourced UDHR audit corpus.
        for source in (
            "সমস্ত মানুষ স্বাধীনভাবে সমান মর্যাদা এবং অধিকার নিয়ে জন্মগ্রহণ করে।",
            "ಎಲ್ಲಾ ಮಾನವರೂ ಸ್ವತಂತ್ರರಾಗಿಯೇ ಜನಿಸಿದ್ದಾರೆ.",
            "सभी मनुष्यों को गौरव और अधिकारों के मामले में जन्मजात स्वतन्त्रता और समानता प्राप्त है ।",
            "ހުރިހާ އިންސާނުންވެސް ދުނިޔެއަށް އުފަންވަނީ، މިނިވަންކަމުގައި،",
            "འགྲོ་བ་མིའི་རིགས་རྒྱུད་ཡོངས་ལ་སྐྱེས་ཙམ་ཉིད་ནས་ཆེ་མཐོངས་དང༌།",
        ):
            with self.subTest(source=source):
                self.assertFalse(looks_like_code(source))

    def test_decomposed_latin_prose_is_not_code(self):
        source = unicodedata.normalize("NFD", "Ça, été, élevé, oublié, déjà, café, élève.")
        self.assertFalse(looks_like_code(source))

    def test_unattached_marks_and_symbol_soup_are_still_code(self):
        for source in (
            "\u0301 " * 10,
            "}\u0301]\u0301)\u0301 {\u0301[\u0301(\u0301 " * 3,
            "a" + "\u0301" * 100 + "}]){[[}}",
        ):
            with self.subTest(source=source):
                self.assertTrue(looks_like_code(source))

    def test_commands_with_unicode_identifiers_are_still_code(self):
        for source in (
            "python தமிழ்.py --input source.json --output result.json",
            "git diff --cached --name-only -- தமிழ்.txt",
            "cat Café.txt | grep --color=always --line-number file",
        ):
            with self.subTest(source=source):
                self.assertTrue(looks_like_code(source))


if __name__ == "__main__":
    unittest.main()
