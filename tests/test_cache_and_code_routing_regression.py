"""Offline regressions for full-input cache identity and ordinary UI prose."""

from __future__ import annotations

import unittest
from copy import deepcopy

from pingyitong.config.schema import AppConfig
from pingyitong.services.code_filter import looks_like_code
from tests.test_translation_service import block, make_frame, make_service


class _OfflinePool:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []
        self.namespace = "local"

    def translate(self, model_id, prompt, system_prompt, max_tokens=None):
        self.calls.append((prompt, system_prompt))
        return self.responses.pop(0) if self.responses else ""

    def translation_cache_namespace(self):
        return self.namespace

    def is_available(self, model_id):
        return True

    def any_available(self):
        return True


class TestCompleteInputCacheKey(unittest.TestCase):
    def test_long_sources_with_opposite_final_meaning_do_not_share_a_result(self):
        prefix = "The translation must preserve the meaning of each sentence. " * 40
        positive = prefix + "Access is permitted."
        negative = prefix + "Access is prohibited."
        self.assertGreater(len(prefix), 2000)
        pool = _OfflinePool("允许访问。", "禁止访问。")
        service, _ = make_service(pool=pool)
        self.assertEqual(service.translate_text(positive), "允许访问。")
        self.assertEqual(service.translate_text(negative), "禁止访问。")
        self.assertEqual(len(pool.calls), 2)
        self.assertEqual(service.translate_text(negative), "禁止访问。")
        self.assertEqual(service.translate_text(positive), "允许访问。")
        self.assertEqual(len(pool.calls), 2)

    def test_full_digest_keeps_key_bounded_and_preserves_route_and_direction(self):
        pool = _OfflinePool()
        service, _ = make_service(pool=pool)
        source = "Meaningful text. " * 1000
        key = service._cache_key(source, "en", "zh")
        self.assertLess(len(key), 200)
        self.assertNotEqual(key, service._cache_key(source + "Not permitted.", "en", "zh"))
        self.assertNotEqual(key, service._cache_key(source, "en", "ko"))
        self.assertNotEqual(key, service._cache_key(source, "fr", "zh"))
        pool.namespace = "api:2:0"
        self.assertNotEqual(key, service._cache_key(source, "en", "zh"))
        self.assertTrue(service._cache_key(source, "en", "zh").startswith("api:2:0|"))

    def test_concise_mode_does_not_reuse_standard_translation(self):
        """Changing the output style must miss the cache and ask the model again."""
        config = AppConfig()
        config.translation.semantic_review = False  # legacy single-pass contract
        config.translation.source_language = "en"
        pool = _OfflinePool("常规译文。", "精简译文。")
        service, _ = make_service(config, pool=pool)

        self.assertEqual(service.translate_text("The report is very important."), "常规译文。")

        concise_config = deepcopy(config)
        concise_config.translation.concise_output = True
        service.apply_config(concise_config)
        self.assertEqual(service.translate_text("The report is very important."), "精简译文。")
        self.assertEqual(len(pool.calls), 2)


class TestOrdinaryProseCodeRouting(unittest.TestCase):
    def test_url_navigation_and_command_named_english_verbs_are_translated(self):
        sources = (
            "Home | Settings | https://example.com/help",
            "Open the file at C:/reports/result.json | use the Save button.",
            "Use /usr/local/bin || /usr/bin as defaults.",
            "Make the output --compact and --concise for easier reading.",
            "Export the report with --json and --pretty.",
            "Kill the process with --force and --recursive if it hangs.",
        )
        for source in sources:
            with self.subTest(source=source):
                self.assertFalse(looks_like_code(source))
                config = AppConfig()
                config.translation.semantic_review = False  # legacy single-pass contract
                config.translation.target_language = "ko"
                pool = _OfflinePool("설정 창을 여세요.")
                service, _ = make_service(config, pool=pool)
                frame = service.translate_frame(make_frame(block(source, 0, 0)))
                self.assertEqual(len(pool.calls), 1)
                self.assertEqual(frame.items[0].text, "설정 창을 여세요.")
                self.assertNotEqual(frame.items[0].text, source)

    def test_real_shell_commands_remain_preserved(self):
        for source in (
            "make --debug --jobs=2",
            "git commit -m 'fix' --author=author",
            "python scripts/check.py --model fast --output report.json",
            "taskkill /F /PID 38176 2>&1 | head -1",
            '正在执行 cd "F:/项目" && cat > /tmp/repro.py <<EOF',
        ):
            with self.subTest(source=source):
                self.assertTrue(looks_like_code(source))
                config = AppConfig()
                config.translation.semantic_review = False  # legacy single-pass contract
                config.translation.target_language = "ko"
                pool = _OfflinePool()
                service, _ = make_service(config, pool=pool)
                frame = service.translate_frame(make_frame(block(source, 0, 0)))
                self.assertEqual(pool.calls, [])
                self.assertEqual(frame.items[0].text, source)


if __name__ == "__main__":
    unittest.main()
