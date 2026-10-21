"""Dense screen translations must fit output capacity before bounded recovery."""

from pingyitong.config.schema import AppConfig
from pingyitong.constants import MODEL_FAST, SEGMENT_DELIMITER
from tests.test_mixed_target_language import _Responses
from tests.test_translation_service import block, make_frame, make_service


class _CapacityPool:
    """Simulate the local server withholding a token-limited complete response."""

    def __init__(self):
        self.calls = []
        self.truncated = 0

    def translate(self, model_id, prompt, system_prompt, max_tokens=None):
        parts = [part.strip() for part in prompt.split(SEGMENT_DELIMITER)]
        translations = [
            "请在保存完成前保持此窗口打开，不要关闭程序或关闭计算机，以免丢失尚未保存的更改。"
            "保存完成后再继续下一步操作并查看状态。" + part.rsplit(" ", 1)[-1]
            for part in parts
        ]
        # CJK output characters each need a token, and batch markers have
        # their own cost. Return no partial text, as llama_server does.
        needed = sum(len(text) for text in translations) + 12 * (len(parts) - 1)
        self.calls.append((parts, max_tokens, needed))
        if needed > max_tokens:
            self.truncated += 1
            return ""
        return SEGMENT_DELIMITER.join(translations)

    def is_available(self, model_id):
        return True

    def any_available(self):
        return True


def _dense_service():
    config = AppConfig()
    config.translation.semantic_review = False  # legacy single-pass contract
    config.schedule.mode = "manual"
    config.schedule.active_model = MODEL_FAST
    config.model.batch_size = 16  # The user's current dense-screen setting.
    pool = _CapacityPool()
    return make_service(config, pool=pool)[0], pool


def test_dense_output_is_split_before_truncation_and_keeps_every_position():
    service, pool = _dense_service()
    sources = [
        "Please keep this window open until saving has finished. Do not close the computer before completion. "
        + str(index)
        for index in range(6)
    ]
    frame = make_frame(*[block(text, 0, index * 400) for index, text in enumerate(sources)])
    result = service.translate_frame(frame)
    assert len(result.items) == len(sources)
    assert [item.source_text for item in result.items] == sources
    assert pool.truncated == 0
    assert all(needed <= limit for _, limit, needed in pool.calls)
    assert service.last_stats.skipped == 0
    assert [item.source_text for item in result.source_items] == sources
    assert result.retry_regions == []
    # A repeated frame reuses every successful result, including the tail.
    calls = len(pool.calls)
    repeated = service.translate_frame(frame)
    assert len(repeated.items) == len(sources)
    assert all(item.from_cache for item in repeated.items)
    assert len(pool.calls) == calls


def test_correction_batches_respect_output_capacity_and_restore_all_segments():
    service, pool = _dense_service()
    texts = ["Please keep this window open until saving finishes. " * 2 + str(i) for i in range(6)]
    with service._request_scope():
        result = service._run_correction_batches(texts, "Translate accurately into Chinese.", MODEL_FAST, 384, False)
    assert len(result) == len(texts)
    assert all(result)
    assert pool.truncated == 0


def test_mixed_chinese_and_english_prose_still_reaches_translation():
    service, pool = _dense_service()
    source = "保存已开始。Please do not close the window until saving finishes."
    result = service.translate_frame(make_frame(block(source, 0, 0)))
    assert len(result.items) == 1
    assert pool.calls
    assert result.items[0].source_text == source


def test_transient_failure_keeps_source_and_marks_only_untranslated_region():
    config = AppConfig()
    config.translation.semantic_review = False  # legacy single-pass contract
    config.translation.source_language = "zh"
    pool = _Responses("", "")
    service, _ = make_service(config, pool=pool)
    same = block("保存完成。", 0, 0)
    foreign = block("保存已开始。Please do not close the window.", 0, 400)
    result = service.translate_frame(make_frame(same, foreign))
    assert len(result.source_items) == 2
    assert [item.source_text for item in result.source_items] == [same.text, foreign.text]
    assert result.retry_regions == [foreign.box]
    assert [item.source_text for item in result.items] == [same.text]


def test_controlled_retry_bypasses_suppression_without_publishing_bad_output():
    source = "Please do not close the window until saving finishes."
    pool = _Responses(source, source, "保存完成前请勿关闭窗口。")
    service, _ = make_service(pool=pool)
    frame = make_frame(block(source, 0, 0))
    failed = service.translate_frame(frame)
    assert failed.items == []
    assert failed.retry_regions == [frame.blocks[0].box]
    calls = len(pool.calls)
    suppressed = service.translate_frame(frame)
    assert suppressed.items == []
    assert len(pool.calls) == calls
    frame.__dict__["_translation_retry"] = True
    recovered = service.translate_frame(frame)
    assert len(recovered.items) == 1
    assert recovered.items[0].text == "保存完成前请勿关闭窗口。"
    assert recovered.retry_regions == []
    assert len(pool.calls) == calls + 1


def test_empty_recognition_has_explicit_empty_source_and_retry_regions():
    service, pool = _dense_service()
    result = service.translate_frame(make_frame())
    assert result.source_items == []
    assert result.retry_regions == []
    assert pool.calls == []

