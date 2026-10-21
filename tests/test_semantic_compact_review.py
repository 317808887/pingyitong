"""A compact is untrusted until a distinct local request audits its exact text."""

from __future__ import annotations

import json
import time
from copy import deepcopy

import pytest

from pingyitong.components.llm import prompt as prompt_module
from pingyitong.constants import MODEL_FAST, SEGMENT_DELIMITER
from pingyitong.services import translation_service as service_module
from pingyitong.services.semantic_review import (
    REVIEW_POLICY_VERSION,
    ReviewedTranslation,
    source_digest,
)
from tests.test_semantic_review import (
    FULL,
    SHORT,
    SOURCE,
    cache_key,
    review_response,
    reviewed_service,
)
from tests.test_translation_service import block, make_frame


def verdict(choice="B", checks="111111"):
    return [{"id": 0, "choice": choice, "checks": checks}]


def compact_service(response, *, short=SHORT, self_checks="111111"):
    review = review_response(compact=short, compact_checks=self_checks)
    return reviewed_service(FULL, [review], response)


@pytest.mark.parametrize("self_checks", ["111111", "000000", "invalid", None])
def test_only_distinct_compact_review_can_enable_display_and_its_evidence_is_cached(self_checks):
    review = {"id": 0, "independent": FULL, "checks": "111111", "compact": SHORT}
    if self_checks is not None:
        review["compact_checks"] = self_checks
    service, pool = reviewed_service(FULL, [review], verdict())
    item = service.translate_frame(make_frame(block(SOURCE, 0, 0))).items[0]
    assert item.text == FULL and item.display_text == SHORT and item.quality_status == "accepted"
    assert len(pool.calls) == 3
    assert "semantic_compact_review:" in pool.calls[2]["system_prompt"]
    assert "semantic_resolution:" not in pool.calls[2]["system_prompt"]
    assert json.loads(pool.calls[2]["prompt"]) == [{"id": 0, "source": SOURCE, "A": FULL, "B": SHORT}]
    evidence = ReviewedTranslation.from_json(service.cache.get_candidate(cache_key(service)), SOURCE)
    assert evidence is not None
    assert evidence.compact_review_choice == "B"
    assert evidence.compact_review_model == MODEL_FAST
    assert evidence.compact_checks == ("ok",) * 6
    assert len(evidence.compact_review_binding) == 64
    assert service.last_output_text() == FULL
    assert service.translate_text(SOURCE) == FULL
    assert len(pool.calls) == 3, "reviewed compact replay must use its cached evidence"


def test_old_compact_self_approval_without_new_verdict_keeps_full_translation():
    service, pool = compact_service("")
    first = service.translate_frame(make_frame(block(SOURCE, 0, 0))).items[0]
    assert first.text == FULL and first.display_text == "" and first.quality_status == "accepted"
    assert len(pool.calls) == 3
    assert len(service.cache) == 1
    assert service.translate_text(SOURCE) == FULL
    assert len(pool.calls) == 3


@pytest.mark.parametrize("response", [
    verdict("A"), verdict("uncertain", "??????"),
    verdict("B", "111011"), verdict("B", "11111?"), "yes",
    '[{"id":0,"choice":"B","checks":"111111"}',
    '[{"id":0,"choice":"A","choice":"B","checks":"111111"}]',
    verdict("neither", "11111"), '[{"id":0,"choice":"neither"}]',
])
def test_rejected_or_invalid_compact_audit_never_withholds_reviewed_full_translation(response):
    service, pool = compact_service(response)
    item = service.translate_frame(make_frame(block(SOURCE, 0, 0))).items[0]
    assert item.text == FULL and item.quality_status == "accepted"
    assert item.display_text == "" and len(service.cache) == 1
    assert len(pool.calls) == 3


def test_explicit_valid_neither_retracts_full_and_its_staged_cache():
    service, pool = compact_service(verdict("neither", "000000"))
    result = service.translate_frame(make_frame(block(SOURCE, 0, 0)))
    item = result.items[0]
    assert item.text == SOURCE and item.display_text == ""
    assert item.quality_status == "uncertain"
    assert item.quality_reason == "semantic_full_rejected_by_compact_review"
    assert result.diagnostics["review_rejected"] == 1
    assert service.cache.get_candidate(cache_key(service)) is None
    assert len(service.cache) == 0 and service.last_output_text() == SOURCE
    assert service.translate_text(SOURCE) == SOURCE
    assert len(pool.calls) == 3, "retracted full must not be republished from cache or retried immediately"


def test_neither_retracts_only_its_item_and_does_not_discard_another_reviewed_full():
    other_source, other_full, other_short = "The report contains 2 pages.", "报告包含2页。", "报告2页。"
    service, pool = reviewed_service(
        FULL + SEGMENT_DELIMITER + other_full,
        [review_response(compact=SHORT, index=0), review_response(other_full, compact=other_short, index=1)],
        [{"id": 0, "choice": "neither", "checks": "000000"},
         {"id": 1, "choice": "A", "checks": "111111"}],
    )
    result = service.translate_frame(make_frame(block(SOURCE, 0, 0), block(other_source, 0, 160)))
    assert result.items[0].quality_reason == "semantic_full_rejected_by_compact_review"
    assert result.items[0].text == SOURCE
    assert result.items[1].text == other_full and result.items[1].quality_status == "accepted"
    assert result.items[1].display_text == ""
    assert len(service.cache) == 1 and len(pool.calls) == 3
    assert service.cache.get_candidate(cache_key(service)) is None
    assert ReviewedTranslation.from_json(service.cache.get_candidate(cache_key(service, other_source)), other_source)


def test_shortening_completion_condition_to_event_is_a_candidate_not_trusted_output():
    source = "Do not disconnect the cable until the transfer is complete."
    full, bad_short = "传输完成前请勿断开电缆。", "传输前勿断开电缆。"
    service, pool = reviewed_service(
        full, [{"id": 0, "independent": full, "checks": "111111", "compact": bad_short}],
        verdict("B", "111011"),
    )
    item = service.translate_frame(make_frame(block(source, 0, 0))).items[0]
    assert len(pool.calls) == 3
    assert json.loads(pool.calls[2]["prompt"])[0]["B"] == bad_short
    assert item.text == full and item.display_text == "" and item.quality_status == "accepted"


def test_foreign_word_in_french_compact_is_reviewed_separately_and_full_is_preserved():
    source = "Save the file before closing the window."
    full = "Enregistrez le fichier avant de fermer la fenêtre."
    bad_short = "Enregistrez la datei avant fermeture."
    service, pool = reviewed_service(
        full, [{"id": 0, "independent": full, "checks": "111111", "compact": bad_short}],
        verdict("A"),
    )
    service._config.translation.target_language = "fr"
    item = service.translate_frame(make_frame(block(source, 0, 0))).items[0]
    assert len(pool.calls) == 3
    assert json.loads(pool.calls[2]["prompt"])[0]["B"] == bad_short
    assert item.text == full and item.display_text == "" and item.quality_status == "accepted"


@pytest.mark.parametrize("proposal", ["", FULL, FULL + "额外内容。"])
def test_no_actual_shortening_does_not_add_an_inference(proposal):
    service, pool = compact_service(verdict(), short=proposal)
    item = service.translate_frame(make_frame(block(SOURCE, 0, 0))).items[0]
    assert item.text == FULL and item.display_text == ""
    assert len(pool.calls) == 2


def test_full_resolution_selected_b_cannot_inherit_original_a_compact():
    other_full = "备份结束之前请勿关闭窗口。"
    service, pool = reviewed_service(
        FULL, [review_response(other_full, compact=SHORT)], verdict("B"), verdict("B"),
    )
    item = service.translate_frame(make_frame(block(SOURCE, 0, 0))).items[0]
    assert item.text == other_full and item.display_text == ""
    assert len(pool.calls) == 3
    assert "semantic_resolution:" in pool.calls[2]["system_prompt"]
    assert all("semantic_compact_review:" not in call["system_prompt"] for call in pool.calls)


def test_full_resolution_and_compact_audit_share_ledger_with_full_priority(monkeypatch):
    monkeypatch.setattr(service_module, "MAX_FRAME_CORRECTION_BATCHES", 1)
    other_full = "备份结束之前请勿关闭窗口。"
    service, pool = reviewed_service(
        FULL, [review_response(other_full, compact=SHORT)], verdict("A"), verdict("B"),
    )
    item = service.translate_frame(make_frame(block(SOURCE, 0, 0))).items[0]
    assert item.text == FULL and item.quality_status == "accepted" and item.display_text == ""
    assert len(pool.calls) == 3 and len(service.cache) == 1


def test_compact_context_limit_keeps_full_and_does_not_call_model(monkeypatch):
    monkeypatch.setattr(prompt_module, "build_semantic_compact_review_prompt", lambda _target: "semantic_compact_review:" + "x" * 12000)
    service, pool = compact_service(verdict())
    item = service.translate_frame(make_frame(block(SOURCE, 0, 0))).items[0]
    assert item.text == FULL and item.quality_status == "accepted" and item.display_text == ""
    assert len(pool.calls) == 2 and len(service.cache) == 1


def test_stale_frame_during_compact_audit_never_publishes_or_caches_old_results():
    service, pool = compact_service(verdict())
    stale = False
    frame = make_frame(block(SOURCE, 0, 0))
    frame.__dict__["_frame_stale"] = lambda: stale

    def change_scene(_call):
        nonlocal stale
        if len(pool.calls) == 3:
            stale = True

    pool.on_call = change_scene
    item = service.translate_frame(frame).items[0]
    assert item.text == SOURCE and item.quality_status == "uncertain"
    assert len(service.cache) == 0 and service.last_output_text() == ""


def test_settings_change_during_compact_audit_discards_staged_cache():
    service, pool = compact_service(verdict())
    updated = deepcopy(service._config)
    updated.translation.compact_display = False
    pool.on_call = lambda _call: service.apply_config(updated) if len(pool.calls) == 3 else None
    service.translate_text(SOURCE)
    assert len(service.cache) == 0 and service.last_output_text() == ""


def test_compact_review_time_is_included_in_review_and_total_elapsed():
    service, pool = compact_service(verdict())
    pool.on_call = lambda _call: time.sleep(0.01) if len(pool.calls) == 3 else None
    result = service.translate_frame(make_frame(block(SOURCE, 0, 0)))
    assert service.last_stats.review_elapsed >= 0.009
    assert result.timings["semantic_review"] >= 0.009
    assert service.last_stats.elapsed >= service.last_stats.review_elapsed


@pytest.mark.parametrize("field,value", [
    ("full_text", FULL + "改动"), ("display_text", "传输前勿断开电缆。"),
    ("compact_checks", "111011"), ("compact_review_choice", "A"),
    ("compact_review_model", "another-model"), ("compact_review_binding", "0" * 64),
    ("source_hash", "0" * 64),
])
def test_compact_cache_cannot_mix_source_full_compact_model_or_verdict(field, value):
    service, _ = compact_service(verdict())
    assert service.translate_text(SOURCE) == FULL
    raw = json.loads(service.cache.get_candidate(cache_key(service)))
    raw[field] = value
    assert ReviewedTranslation.from_json(json.dumps(raw), SOURCE) is None


def test_v4_compact_self_evidence_cannot_skip_new_v5_independent_audit():
    service, pool = compact_service(verdict())
    old = json.loads(ReviewedTranslation(
        source_digest(SOURCE),
        FULL, SHORT, FULL, MODEL_FAST, MODEL_FAST, compact_checks=("ok",) * 6,
    ).to_json())
    old.update(format="reviewed-translation-v4", policy=REVIEW_POLICY_VERSION - 1)
    service.cache.put(cache_key(service), json.dumps(old))
    assert service.translate_text(SOURCE) == FULL
    assert len(pool.calls) == 3
    assert service.cache.stats().rejected_candidates == 1
