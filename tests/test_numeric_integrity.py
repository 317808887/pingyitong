"""A fluent translation must not silently drop a source quantity."""

from pingyitong.config.schema import AppConfig
from pingyitong.constants import MODEL_ACCURATE, MODEL_FAST
from pingyitong.services.translation_service import _missing_numerals
from tests.test_mixed_target_language import _Responses
from tests.test_translation_service import block, make_frame, make_service

SOURCE = "Only 3 of the 15 samples were positive."
BAD = "仅3个样本为阳性。"
GOOD = "15个样本中仅3个为阳性。"


def test_number_comparison_preserves_multisets_and_decimal_tokens():
    assert _missing_numerals(SOURCE, BAD) == ["15"]
    assert _missing_numerals(SOURCE, GOOD) == []
    assert _missing_numerals("Version 1.7 contains 2 fixes", "版本1.7包含2项修复") == []
    assert _missing_numerals("Two 5 mg doses plus another 5 mg dose", "两次5毫克剂量") == ["5"]
    assert _missing_numerals("[TERM_0] studies particles", "研究粒子") == []


def test_single_omitted_total_is_retried_and_bad_output_not_cached():
    config = AppConfig()
    config.translation.semantic_review = False  # legacy single-pass contract
    config.translation.source_language = "en"
    pool = _Responses(BAD, GOOD)
    service, _ = make_service(config, pool=pool)
    assert service.translate_text(SOURCE) == GOOD
    assert len(pool.calls) == 2
    assert "every numeral" in pool.calls[1][1]
    assert service.translate_text(SOURCE) == GOOD
    assert len(pool.calls) == 2


def test_frame_omitted_total_is_retried_before_rendering():
    config = AppConfig()
    config.translation.semantic_review = False  # legacy single-pass contract
    config.translation.source_language = "en"
    pool = _Responses(BAD, GOOD)
    service, _ = make_service(config, pool=pool)
    result = service.translate_frame(make_frame(block(SOURCE, 0, 0)))
    assert [item.text for item in result.items] == [GOOD]
    assert len(pool.calls) == 2


def test_persistent_missing_total_never_published_or_cached():
    config = AppConfig()
    config.translation.semantic_review = False  # legacy single-pass contract
    config.translation.source_language = "en"
    pool = _Responses(BAD, BAD)
    service, _ = make_service(config, pool=pool)
    assert service.translate_text(SOURCE) == SOURCE
    assert len(service.cache) == 0


def test_accurate_numeric_failure_can_use_ready_fast_model():
    class ReadyPool(_Responses):
        def __init__(self):
            super().__init__(BAD, BAD, GOOD)
            self.models = []

        def translate(self, model_id, prompt, system_prompt, max_tokens=None):
            self.models.append(model_id)
            return super().translate(model_id, prompt, system_prompt, max_tokens)

        def loaded_models(self):
            return [MODEL_ACCURATE, MODEL_FAST]

    config = AppConfig()
    config.translation.semantic_review = False  # legacy single-pass contract
    config.translation.source_language = "en"
    config.schedule.mode = "manual"
    config.schedule.active_model = MODEL_ACCURATE
    pool = ReadyPool()
    service, _ = make_service(config, pool=pool)
    assert service.translate_text(SOURCE) == GOOD
    assert pool.models == [MODEL_ACCURATE, MODEL_ACCURATE, MODEL_FAST]


def test_frame_numeric_failure_can_use_ready_fast_model():
    class ReadyPool(_Responses):
        def __init__(self):
            super().__init__(BAD, BAD, GOOD)
            self.models = []

        def translate(self, model_id, prompt, system_prompt, max_tokens=None):
            self.models.append(model_id)
            return super().translate(model_id, prompt, system_prompt, max_tokens)

        def loaded_models(self):
            return [MODEL_ACCURATE, MODEL_FAST]

    config = AppConfig()
    config.translation.semantic_review = False  # legacy single-pass contract
    config.translation.source_language = "en"
    config.schedule.mode = "manual"
    config.schedule.active_model = MODEL_ACCURATE
    pool = ReadyPool()
    service, _ = make_service(config, pool=pool)
    result = service.translate_frame(make_frame(block(SOURCE, 0, 0)))
    assert [item.text for item in result.items] == [GOOD]
    assert pool.models == [MODEL_ACCURATE, MODEL_ACCURATE, MODEL_FAST]
