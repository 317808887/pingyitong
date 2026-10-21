"""Explicit terminology stays intact without exempting surrounding prose."""

from __future__ import annotations

import unittest

from pingyitong.config.schema import AppConfig
from pingyitong.services.glossary_service import Glossary, GlossaryEntry, load_builtin_glossary
from tests.test_cache_and_code_routing_regression import _OfflinePool
from tests.test_translation_service import block, make_frame, make_service


def _service(pool, glossary):
    config = AppConfig()
    config.translation.semantic_review = False  # legacy single-pass contract
    config.translation.source_language = "en"
    config.translation.target_language = "ko"
    return make_service(config, pool=pool, glossary=glossary)[0]


class TestExplicitGlossaryExceptions(unittest.TestCase):
    def test_curated_invariant_is_substituted_restored_and_cached(self):
        glossary = load_builtin_glossary("ko")
        pool = _OfflinePool("[TERM_0]를 여세요.")
        service = _service(pool, glossary)
        frame = make_frame(block("Open Docker.", 0, 0))
        first = service.translate_frame(frame)
        self.assertEqual(first.items[0].text, "Docker를 여세요.")
        self.assertIn("[TERM_0]", pool.calls[0][0])
        self.assertEqual(service.translate_frame(frame).items[0].text, "Docker를 여세요.")
        self.assertEqual(len(pool.calls), 1)

    def test_explicit_pair_term_is_honored_in_frame_and_text(self):
        glossary = Glossary([GlossaryEntry("tensor", "Tensor", source_language="en", target_language="ko")])
        pool = _OfflinePool("[TERM_0]를 사용하세요.", "[TERM_0]를 사용하세요.")
        service = _service(pool, glossary)
        frame = service.translate_frame(make_frame(block("Use the tensor.", 0, 0)))
        self.assertEqual(frame.items[0].text, "Tensor를 사용하세요.")
        self.assertEqual(service.translate_text("Please use the tensor."), "Tensor를 사용하세요.")
        self.assertTrue(all("[TERM_0]" in prompt for prompt, _ in pool.calls))

    def test_literal_target_term_also_passes_without_placeholder(self):
        glossary = Glossary([GlossaryEntry("tensor", "Tensor", source_language="en", target_language="ko")])
        pool = _OfflinePool("Tensor를 사용하세요.")
        service = _service(pool, glossary)
        self.assertEqual(service.translate_text("Use the tensor."), "Tensor를 사용하세요.")
        self.assertEqual(len(pool.calls), 1)

    def test_exemption_does_not_allow_english_around_a_protected_term(self):
        glossary = Glossary([GlossaryEntry("tensor", "Tensor", source_language="en", target_language="ko")])
        pool = _OfflinePool("[TERM_0] is ready.", "[TERM_0]가 준비되었습니다.")
        service = _service(pool, glossary)
        frame = service.translate_frame(make_frame(block("The tensor is ready.", 0, 0)))
        self.assertEqual(frame.items[0].text, "Tensor가 준비되었습니다.")
        self.assertEqual(len(pool.calls), 2)

    def test_unmatched_or_wrong_pair_terms_cannot_exempt_arbitrary_english(self):
        glossary = Glossary([GlossaryEntry("other", "close", source_language="fr", target_language="ko")])
        pool = _OfflinePool("설치 완료 close", "설치 완료 close")
        service = _service(pool, glossary)
        frame = service.translate_frame(make_frame(block("Installation is complete.", 0, 0)))
        self.assertEqual(frame.items, [])
        self.assertEqual(len(service.cache), 0)


if __name__ == "__main__":
    unittest.main()
