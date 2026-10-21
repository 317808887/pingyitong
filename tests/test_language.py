"""源语言自动检测测试。

覆盖产品功能规划二·（四）的要求："软件自动检测屏幕文字的源语言"。
重点验证各语种的区分能力，以及中英混排界面（最常见的真实场景）的判定。
"""

from __future__ import annotations

import unittest

from pingyitong.services.language import (
    detect_language,
    is_probably_translatable,
)


class TestScriptDetection(unittest.TestCase):
    """按字符脚本判定的语种。"""

    def test_chinese(self) -> None:
        """纯汉字判定为中文。"""
        guess = detect_language("这是一个中文句子，用于测试语言检测功能。")
        self.assertEqual(guess.code, "zh")
        self.assertTrue(guess.is_confident)

    def test_japanese_with_kana(self) -> None:
        """含假名判定为日语。

        这是与中文的关键区分点：中文文本中不会出现平假名或片假名。
        """
        guess = detect_language("これはテストです。")
        self.assertEqual(guess.code, "ja")

    def test_japanese_with_kanji_and_kana(self) -> None:
        """汉字 + 假名混排仍判定为日语。"""
        guess = detect_language("設定画面を開いてください。")
        self.assertEqual(guess.code, "ja")

    def test_korean(self) -> None:
        """谚文判定为韩语。"""
        guess = detect_language("안녕하세요 세계")
        self.assertEqual(guess.code, "ko")

    def test_russian(self) -> None:
        """西里尔字母判定为俄语。"""
        guess = detect_language("Привет мир, это тест.")
        self.assertEqual(guess.code, "ru")

    def test_english(self) -> None:
        """常见英文判定为英语。"""
        guess = detect_language("The quick brown fox jumps over the lazy dog.")
        self.assertEqual(guess.code, "en")


class TestLatinDisambiguation(unittest.TestCase):
    """同为拉丁字母的语种区分。"""

    def test_german_by_distinctive_chars(self) -> None:
        """德语特征字符 ß / ü 应触发德语判定。"""
        guess = detect_language("Die Straße ist groß und schön.")
        self.assertEqual(guess.code, "de")

    def test_spanish_by_distinctive_chars(self) -> None:
        """西班牙语特征字符 ñ 应触发西语判定。"""
        guess = detect_language("El niño está en la casa con su madre.")
        self.assertEqual(guess.code, "es")

    def test_french_by_stopwords(self) -> None:
        """法语停用词应触发法语判定。"""
        guess = detect_language("Le chat est dans la maison avec les enfants.")
        self.assertEqual(guess.code, "fr")

    def test_short_identifier_defaults_to_english(self) -> None:
        """无语言特征的短标识符回退为英语。

        屏幕上的 ``Save``、``OK`` 这类单词没有足够特征，
        回退到英语是最安全的选择（覆盖面最广）。
        """
        guess = detect_language("Save")
        self.assertEqual(guess.code, "en")
        # 但置信度应当较低，以便调用方决定是否改用兜底语言
        self.assertLess(guess.confidence, 0.6)


class TestEdgeCases(unittest.TestCase):
    """边界情况。"""

    def test_empty_text_returns_fallback(self) -> None:
        """空文本返回兜底语言。"""
        guess = detect_language("", fallback="ja")
        self.assertEqual(guess.code, "ja")
        self.assertEqual(guess.confidence, 0.0)

    def test_whitespace_only(self) -> None:
        """纯空白同样回退。"""
        guess = detect_language("   \n\t ", fallback="de")
        self.assertEqual(guess.code, "de")

    def test_symbols_only(self) -> None:
        """纯符号回退。"""
        guess = detect_language("===>>> ***", fallback="fr")
        self.assertEqual(guess.code, "fr")

    def test_mixed_cjk_latin_dominant_chinese(self) -> None:
        """中英混排以占比最高的脚本为准。"""
        guess = detect_language("打开 Settings 面板进行 configuration 设置")
        self.assertEqual(guess.code, "zh")

    def test_mixed_cjk_latin_dominant_english(self) -> None:
        """英文为主、含少量汉字时判定为英语。"""
        guess = detect_language(
            "The configuration panel allows you to adjust 设置 options"
        )
        self.assertEqual(guess.code, "en")

    def test_fallback_is_respected(self) -> None:
        """兜底语言参数应被采纳。"""
        guess = detect_language("12345", fallback="es")
        self.assertEqual(guess.code, "es")


class TestTranslatableFilter(unittest.TestCase):
    """可翻译性判断。"""

    def test_normal_sentence(self) -> None:
        """正常句子应当被翻译。"""
        self.assertTrue(is_probably_translatable("Hello world"))

    def test_pure_number_rejected(self) -> None:
        """纯数字不翻译。"""
        self.assertFalse(is_probably_translatable("12345"))

    def test_percentage_rejected(self) -> None:
        """带符号的数字不翻译。"""
        self.assertFalse(is_probably_translatable("98.5%"))

    def test_pure_symbols_rejected(self) -> None:
        """纯符号不翻译。"""
        self.assertFalse(is_probably_translatable("===>"))

    def test_single_char_rejected(self) -> None:
        """单字符不翻译。"""
        self.assertFalse(is_probably_translatable("A"))

    def test_short_word_accepted(self) -> None:
        """短单词应当被翻译。"""
        self.assertTrue(is_probably_translatable("OK"))

    def test_mostly_digits_rejected(self) -> None:
        """字母占比过低时不翻译（例如版本号与计量数值）。"""
        self.assertFalse(is_probably_translatable("v1.2.3456"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
