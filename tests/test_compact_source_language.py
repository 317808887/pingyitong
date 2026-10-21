"""Source-selection routing regressions with a recording stand-in, not real translation."""
from __future__ import annotations

import unittest
from copy import deepcopy

from pingyitong.config.schema import AppConfig
from pingyitong.constants import MODEL_FAST
from tests.test_translation_request_context import _RequestPool
from tests.test_translation_service import block, make_frame, make_service


class TestCompactSourceLanguage(unittest.TestCase):
    def _service(self, source="fr"):
        config = AppConfig()
        config.translation.semantic_review = False  # legacy single-pass contract
        config.translation.source_language = source
        config.translation.target_language = "zh"
        config.translation.glossary_enabled = False
        pool = _RequestPool()
        pool.reply = "面包"
        service, _ = make_service(config, pool=pool)
        return config, service, pool

    def test_plain_single_screen_paragraph_sends_selected_source(self):
        _, service, pool = self._service()
        frame = service.translate_frame(make_frame(block("pain", 0, 0)))
        self.assertEqual(len(frame.items), 1)
        self.assertIn("Translate from French", pool.calls[0][1])
        self.assertNotIn("separator", pool.calls[0][1])

    def test_clipboard_text_sends_selected_source(self):
        _, service, pool = self._service()
        service.translate_text("pain")
        self.assertIn("Translate from French", pool.calls[0][1])
        self.assertNotIn("separator", pool.calls[0][1])

    def test_required_terms_recovery_sends_selected_source(self):
        _, service, pool = self._service()
        output = service._retry_without_placeholders(
            "pain", ["面包"], MODEL_FAST, 128, False, False,
        )
        self.assertEqual(output, "面包")
        self.assertIn("Translate from French", pool.calls[0][1])
        self.assertIn("Required terminology", pool.calls[0][1])

    def test_compact_source_uses_request_snapshot_during_direction_change(self):
        config, service, pool = self._service()
        switched = deepcopy(config)
        switched.translation.source_language = "en"
        original_decide = service._scheduler.decide

        def change_source_before_prompt(**kwargs):
            service.apply_config(switched)
            return original_decide(**kwargs)

        service._scheduler.decide = change_source_before_prompt
        service.translate_text("pain")
        self.assertIn("Translate from French", pool.calls[0][1])
        self.assertEqual(len(service.cache), 0)
        service._scheduler.decide = original_decide
        pool.reply = "疼痛"
        service.translate_text("pain")
        self.assertIn("Translate from English", pool.calls[-1][1])
        self.assertEqual(len(service.cache), 1)


if __name__ == "__main__":
    unittest.main()
