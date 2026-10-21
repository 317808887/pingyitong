"""Reject explicit before/after reversals without rewriting model translations."""

import pytest

from pingyitong.config.schema import AppConfig
from pingyitong.constants import MODEL_ACCURATE, MODEL_FAST
from pingyitong.services.language import has_reversed_until_condition
from tests.test_mixed_target_language import _Responses
from tests.test_semantic_review import cache_key, review_response, reviewed_service
from tests.test_translation_service import block, make_frame, make_service

JA = "確認が終わるまで、ファイルを削除しないでください。"
EN = "Keep the cable connected until transfer ends."
EN_BEFORE = "Do not close before the backup finishes."
NOMINALIZED_BEFORE = "不要关闭备份完成之前的操作。"


@pytest.mark.parametrize("source,output", [
    (EN_BEFORE, NOMINALIZED_BEFORE),
    (EN_BEFORE, "请勿关闭备份完成前的窗口。"),
    ("Don't close the window before installation finishes.", "不要关闭安装完成以前的窗口。"),
    ("Do not delete files before the transfer completes.", "请勿删除传输完成之前的文件。"),
    ("Do not stop before the upload ends.", "不得停止上传结束前的任务。"),
])
def test_completion_condition_cannot_become_the_forbidden_actions_object(source, output):
    # The general second pass owns semantic roles; do not add another pattern
    # for every possible action/object to the old narrow temporal guard.
    checks = ["ok", "wrong", "ok", "wrong", "ok", "wrong"]
    service, pool = reviewed_service(output, [review_response(checks=checks, compact="")])
    result = service.translate_frame(make_frame(block(source, 0, 0)))
    assert result.items[0].text == source
    assert result.items[0].quality_status == "uncertain"
    assert len(service.cache) == 0
    assert len(pool.calls) == 2


@pytest.mark.parametrize("output", [
    "备份完成之前不要关闭。", "不要在备份完成前关闭。",
    "请勿在备份完成以前关闭窗口。", "不要关闭，直到备份完成。",
    "请等备份完成后再关闭。", "在备份完成之前，请不要关闭窗口。",
    "备份完成前不要关闭当前窗口中的操作。",
])
def test_completion_condition_placement_does_not_reject_faithful_chinese(output):
    assert not has_reversed_until_condition(EN_BEFORE, output, "zh")


def test_source_relative_clause_and_other_targets_are_not_nominalization_errors():
    assert not has_reversed_until_condition(
        "Do not close the task that started before the backup finished.",
        "不要关闭备份完成之前的任务。", "zh",
    )
    assert not has_reversed_until_condition(EN_BEFORE, NOMINALIZED_BEFORE, "ja")


@pytest.mark.parametrize("as_frame", [False, True])
def test_nominalized_condition_is_not_replaced_by_unreviewed_second_translation(as_frame):
    good = "备份完成之前不要关闭。"
    checks = ["ok", "wrong", "ok", "wrong", "ok", "wrong"]
    service, pool = reviewed_service(NOMINALIZED_BEFORE, [review_response(good, checks=checks)])
    if as_frame:
        result = service.translate_frame(make_frame(block(EN_BEFORE, 0, 0)))
        assert [item.text for item in result.items] == [EN_BEFORE]
        assert result.items[0].quality_status == "uncertain"
        assert not result.retry_regions
    else:
        assert service.translate_text(EN_BEFORE) == EN_BEFORE
    assert len(pool.calls) == 2
    assert len(service.cache) == 0
    assert service.translate_text(EN_BEFORE) == EN_BEFORE
    assert len(pool.calls) == 2


@pytest.mark.parametrize("as_frame", [False, True])
def test_repeated_nominalization_is_withheld_and_never_cached(as_frame):
    service, pool = reviewed_service(NOMINALIZED_BEFORE, NOMINALIZED_BEFORE)
    if as_frame:
        result = service.translate_frame(make_frame(block(EN_BEFORE, 0, 0)))
        assert result.items[0].text == EN_BEFORE
        assert result.items[0].quality_status == "uncertain"
    else:
        assert service.translate_text(EN_BEFORE) == EN_BEFORE
    assert len(pool.calls) == 2
    assert len(service.cache) == 0


def test_old_nominalized_condition_cache_is_revalidated_and_replaced():
    service, pool = reviewed_service("备份完成前请勿关闭。", [review_response("备份完成前请勿关闭。", compact="")])
    service.cache.put(cache_key(service, EN_BEFORE), NOMINALIZED_BEFORE)
    assert service.translate_text(EN_BEFORE) == "备份完成前请勿关闭。"
    assert len(pool.calls) == 2
    assert service.translate_text(EN_BEFORE) == "备份完成前请勿关闭。"
    assert len(pool.calls) == 2


@pytest.mark.parametrize("source,output", [
    (JA, "请在确认完成后不要删除文件。"),
    (JA, "确认结束后，请勿删除文件。"),
    (JA, "请在确认完毕以后不删除文件。"),
    ("保存が完了するまで、この画面を閉じないでください。", "保存完成后不要关闭此画面。"),
    (EN, "在传输结束后保持线缆连接。"),
    (EN, "传输完成之后，请继续保持线缆连接。"),
    ("Remain quiet until the meeting ends.", "会议结束后保持安静。"),
    (EN_BEFORE, "在备份完成后不要关闭。"),
    ("Do not delete the files before transfer completes.", "传输完成后不要删除文件。"),
])
def test_explicit_until_reversals_are_rejected(source, output):
    assert has_reversed_until_condition(source, output, "zh")


@pytest.mark.parametrize("source,output", [
    (JA, "确认完成前请勿删除文件。"),
    (JA, "请等确认结束后再删除文件。"),
    (JA, "确认完成前请勿删除文件。完成后不需要重启。"),
    (JA, "确认完成后无需删除文件，完成前也请勿删除。"),
    (EN, "保持线缆连接直到传输结束。"),
    (EN, "请在传输完成前保持线缆连接，完成后再断开。"),
    (EN, "传输完成后不再需要保持连接。"),
    ("確認が終わった後、ファイルを削除しないでください。", "确认结束后不要删除文件。"),
    ("確認が終わるまで削除しないでください、その後も削除しないでください。", "确认结束后也请勿删除。"),
    ("Close the window after saving completes.", "保存完成后不要再次打开窗口。"),
    (EN_BEFORE, "备份完成前不要关闭。"),
    (EN_BEFORE, "请等备份完成后再关闭。"),
])
def test_before_negation_after_permission_and_unrelated_after_are_accepted(source, output):
    assert not has_reversed_until_condition(source, output, "zh")


def test_other_targets_are_outside_the_narrow_guard():
    assert not has_reversed_until_condition(JA, "请在确认完成后不要删除文件。", "en")
    assert not has_reversed_until_condition("Attendez la fin du transfert.", "结束后保持连接。", "zh")


@pytest.mark.parametrize("source,bad,good", [
    (JA, "请在确认完成后不要删除文件。", "确认完成前请勿删除文件。"),
    (EN, "在传输结束后保持线缆连接。", "请保持线缆连接直到传输结束。"),
])
def test_single_reversal_is_retranslated_and_only_good_output_is_cached(source, bad, good):
    config = AppConfig()
    config.translation.semantic_review = False  # legacy single-pass contract
    config.translation.source_language = "ja" if source == JA else "en"
    pool = _Responses(bad, good)
    service, _ = make_service(config, pool=pool)
    assert service.translate_text(source) == good
    assert len(pool.calls) == 2
    assert "Correct the time condition" in pool.calls[1][1]
    assert service.translate_text(source) == good
    assert len(pool.calls) == 2


def test_english_before_reversal_is_retranslated_and_not_cached():
    config = AppConfig()
    config.translation.semantic_review = False  # legacy single-pass contract
    config.translation.source_language = "en"
    pool = _Responses("在备份完成后不要关闭。", "备份完成前不要关闭。")
    service, _ = make_service(config, pool=pool)
    assert service.translate_text(EN_BEFORE) == "备份完成前不要关闭。"
    assert len(pool.calls) == 2
    assert "Correct the time condition" in pool.calls[1][1]
    assert service.translate_text(EN_BEFORE) == "备份完成前不要关闭。"
    assert len(pool.calls) == 2


def test_frame_reversal_uses_formal_correction_and_repeated_failure_stays_unpublished():
    config = AppConfig()
    config.translation.semantic_review = False  # legacy single-pass contract
    config.translation.source_language = "ja"
    bad = "请在确认完成后不要删除文件。"
    pool = _Responses(bad, bad)
    service, _ = make_service(config, pool=pool)
    result = service.translate_frame(make_frame(block(JA, 0, 0)))
    assert result.items == []
    assert len(service.cache) == 0
    assert len(pool.calls) == 2
    assert "Correct the time condition" in pool.calls[1][1]
    assert result.retry_regions


class _ReadyFallbackPool(_Responses):
    def __init__(self, *responses, ready=True, online=False):
        super().__init__(*responses)
        self.models = []
        self.ready = ready
        self.online = online

    def translate(self, **kwargs):
        self.models.append(kwargs["model_id"])
        return super().translate(**kwargs)

    def loaded_models(self):
        return [MODEL_FAST, MODEL_ACCURATE] if self.ready else [MODEL_FAST]

    def api_mode_active(self):
        return self.online


@pytest.mark.parametrize("as_frame", [False, True])
def test_only_failed_temporal_correction_uses_ready_accurate_fallback(as_frame):
    config = AppConfig()
    config.translation.semantic_review = False  # legacy single-pass contract
    config.translation.source_language = "ja"
    bad = "请在确认完成后不要删除文件。"
    good = "确认完成前请勿删除文件。"
    pool = _ReadyFallbackPool(bad, bad, good)
    service, _ = make_service(config, pool=pool)
    if as_frame:
        result = service.translate_frame(make_frame(block(JA, 0, 0)))
        assert [item.text for item in result.items] == [good]
        assert result.retry_regions == []
    else:
        assert service.translate_text(JA) == good
    assert pool.models == [MODEL_FAST, MODEL_FAST, MODEL_ACCURATE]
    assert config.schedule.active_model == MODEL_FAST
    assert service.translate_text(JA) == good
    assert len(pool.models) == 3


@pytest.mark.parametrize("disabled,ready,online,api_enabled", [
    (True, True, False, False), (False, False, False, False),
    (False, True, True, True), (False, True, False, True),
])
def test_disabled_unready_or_online_model_never_activates_local_fallback(disabled, ready, online, api_enabled):
    config = AppConfig()
    config.translation.semantic_review = False  # legacy single-pass contract
    config.translation.source_language = "ja"
    config.model.accurate.enabled = not disabled
    config.api.enabled = api_enabled
    bad = "请在确认完成后不要删除文件。"
    pool = _ReadyFallbackPool(bad, bad, "确认完成前请勿删除文件。", ready=ready, online=online)
    service, _ = make_service(config, pool=pool)
    assert service.translate_text(JA) == JA
    assert pool.models == [MODEL_FAST, MODEL_FAST]
    assert config.model.accurate.enabled is not disabled


def test_unrelated_language_failure_does_not_use_temporal_fallback():
    pool = _ReadyFallbackPool("Файл открыт", "Файл открыт", "文件已打开。")
    service, _ = make_service(pool=pool)
    assert service.translate_text("The file is open.") == "The file is open."
    assert pool.models == [MODEL_FAST, MODEL_FAST]


def test_failed_accurate_temporal_fallback_is_not_published_or_cached():
    config = AppConfig()
    config.translation.semantic_review = False  # legacy single-pass contract
    config.translation.source_language = "ja"
    bad = "请在确认完成后不要删除文件。"
    pool = _ReadyFallbackPool(bad, bad, bad)
    service, _ = make_service(config, pool=pool)
    assert service.translate_text(JA) == JA
    assert pool.models == [MODEL_FAST, MODEL_FAST, MODEL_ACCURATE]
    assert len(service.cache) == 0

