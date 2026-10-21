"""Offline evidence gates; these tests do not operate or validate a desktop."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_native_acceptance.py"
spec = importlib.util.spec_from_file_location("native_acceptance_summary", SCRIPT)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)

SHA = "a" * 64


def evidence(*, blank=False, reviewer_type="ai", captured=1.1, png=True):
    items = [] if blank else [{"source": "Hello", "full_text": "Bonjour", "display_text": "Bonjour",
                              "quality_status": "accepted", "disposition": "translated",
                              "truncated": False, "rect": [1, 1, 100, 20], "from_cache": True}]
    rows = [] if blank else [{"id": "row-1", "source": "Hello"}]
    corpus = {"run_id": "run", "seed": 17, "corpus_sha256": "corpus",
              "scenes": [{"scene_id": "scene-01", "category": "blank" if blank else "subtitle",
                          "source_language": "en", "target_language": "fr", "rows": rows,
                          "latency_target_ms": 600}]}
    events = [{"run_id": "run", "scene_id": "scene-01", "generation": 1,
               "paint_completed_monotonic_ns": 1_000_000_000,
               "blocks": [] if blank else [{"text": "Hello"}]}]
    paints = [{"run_id": "run", "evidence_kind": "application_submitted_paint_buffer",
               "painted_at": 1.4, "captured_at": captured, "items": items,
               "seq": 7, "_png_verified": png}]
    identity = {"executable": "F:/qa/PingYiTong.exe", "executable_sha256": SHA}
    review = {"instance_id": "scene-01@1", "output_sha256": summary.output_hash(items),
              "full_meaning": True, "compact_meaning": True, "readable": True}
    if reviewer_type is not None:
        review["reviewer_type"] = reviewer_type
    annotations = {"native_ui": {"run_id": "run", "performed": True,
                                  "installed_executable_sha256": SHA, "evidence_reference": "native mouse log"},
                   "reviews": [] if blank else [review]}
    return corpus, events, paints, identity, annotations


def run(data):
    return summary.summarize(*data, installed_sha256=SHA)


def test_independent_ai_review_is_named_ai_and_never_human_approval():
    result = run(evidence())
    assert result["outcome"] == "passed"
    row = result["scenes"][0]
    assert row["meaning_readability_review_passed"] and row["ai_review_passed"]
    assert not row["human_review_passed"]
    assert result["metrics"]["ai_review_passed"] == 1
    assert result["metrics"]["human_review_passed"] == 0
    assert result["review_template"]["reviews"][0]["reviewer_type"] is None
    assert any("not human acceptance" in item for item in result["limitations"])


@pytest.mark.parametrize("reviewer_type", [None, "unknown"])
def test_legacy_untyped_true_flags_do_not_implicitly_become_human_review(reviewer_type):
    result = run(evidence(reviewer_type=reviewer_type))
    assert result["outcome"] == "failed"
    assert not result["scenes"][0]["human_review_passed"]
    assert not result["scenes"][0]["meaning_readability_review_passed"]


def test_explicit_human_review_preserves_compatibility_field():
    result = run(evidence(reviewer_type="human"))
    assert result["outcome"] == "passed"
    assert result["scenes"][0]["human_review_passed"]
    assert not result["scenes"][0]["ai_review_passed"]


def test_blank_requires_post_appearance_capture_empty_items_and_actual_buffer():
    result = run(evidence(blank=True))
    assert result["outcome"] == "passed"
    row = result["scenes"][0]
    assert row["blank_scene"] and not row["review_required"]
    assert row["meaning_readability_review_passed"] is None
    assert row["scene_to_complete_ms"] == pytest.approx(400)
    assert result["metrics"]["blank_scenes_verified"] == 1
    assert result["review_template"]["reviews"] == []
    assert not summary.ready([]), "The empty case has its own explicit gate"


@pytest.mark.parametrize("failure", ["old_capture", "nonempty", "missing_png", "wrong_identity", "no_mouse"])
def test_blank_cannot_pass_from_old_or_unbound_evidence(failure):
    data = evidence(blank=True)
    if failure == "old_capture":
        data[2][0]["captured_at"] = 0.9
    elif failure == "nonempty":
        data[2][0]["items"] = evidence()[2][0]["items"]
    elif failure == "missing_png":
        data[2][0]["_png_verified"] = False
    elif failure == "wrong_identity":
        data[3]["executable_sha256"] = "wrong"
    else:
        data[4]["native_ui"]["performed"] = False
    result = run(data)
    assert result["outcome"] != "passed"
    assert result["metrics"]["blank_scenes_verified"] == 0


def test_all_cache_flags_do_not_claim_zero_native_model_calls():
    result = run(evidence())
    assert result["scenes"][0]["cache_route"] == "full_cache"
    assert result["scenes"][0]["cache_route_basis"] == "item.from_cache flags only"
    assert result["scenes"][0]["model_zero_calls_proven"] is None
    assert result["model_zero_calls_proven"] is None


def test_legacy_effective_inline_text_is_unknown_compact_coverage():
    result = run(evidence())
    assert result["scenes"][0]["items"][0]["display_text"] == "Bonjour"
    compact = result["scenes"][0]["compact_generation_evidence"]
    assert compact["generated_items"] == 0 and compact["unknown_items"] == 1
    assert result["metrics"]["generated_compact_items"] == 0
    assert any("nonempty display_text" in item for item in result["limitations"])


def test_explicit_compact_evidence_is_preserved_and_bound_to_review_hash():
    data = evidence()
    item = data[2][0]["items"][0]
    old_hash = summary.output_hash([item])
    item.update(compact_generated=False, compact_text="")
    no_compact_hash = summary.output_hash([item])
    assert old_hash != no_compact_hash
    item.update(compact_generated=True, compact_text="Salut")
    new_hash = summary.output_hash([item])
    assert new_hash != no_compact_hash
    result = run(data)
    assert result["scenes"][0]["compact_generation_evidence"]["generated_items"] == 1
    assert result["scenes"][0]["items"][0]["compact_text"] == "Salut"
    assert not result["scenes"][0]["meaning_readability_review_passed"], "Old output hash cannot approve changed compact evidence"


def v2_evidence():
    data = evidence()
    data[0]["protocol_version"] = summary.PROTOCOL_V2
    paint = data[2][0]
    paint["process_id"] = 123
    paint["items"][0]["font_px"] = 14
    paint["source_lines"] = [{"text": "Hello", "rect": [1, 1, 100, 20], "line_id": 0}]
    paint["refresh_kind"] = "initial"
    paint["delivery_evidence"] = {
        "delivery_route": "trusted_cache_restore", "confirmed_retained_items": 0,
        "work": {"protocol_version": summary.PROTOCOL_V2, "coverage": "model_pool_local_and_api_calls",
                 "completed": True, "events": [], "counts": {
                     "model_load": 0, "model_warmup": 0, "local_inference": 0, "api_inference": 0,
                 }},
    }
    return data


def test_v2_separates_literal_source_lines_and_observed_cache_restore():
    result = run(v2_evidence())
    row = result["scenes"][0]
    assert result["outcome"] == "passed"
    assert row["literal_recognized_rows"] == 1
    assert row["source_line_evidence"]["distinct_line_structure_preserved"]
    assert row["model_zero_calls_proven"] is True
    assert row["runtime_observation"]["delivery_route"] == "trusted_cache_restore"
    assert result["metrics"]["stage_cohorts"]["initial/trusted_cache_restore"]["count"] == 1


def test_v2_punctuation_loss_is_not_hidden_by_legacy_content_match():
    data = v2_evidence()
    data[0]["scenes"][0]["rows"][0]["source"] = "Hello."
    data[1][0]["blocks"][0]["text"] = "Hello."
    result = run(data)
    row = result["scenes"][0]
    assert row["recognized_rows"] == 1 and row["literal_recognized_rows"] == 0
    assert result["outcome"] == "failed" and row["scene_to_complete_ms"] is None


def test_v2_two_source_rows_cannot_be_claimed_from_one_combined_line():
    result = summary.source_line_matches(
        [{"id": "a", "source": "Hello"}, {"id": "b", "source": "Goodbye"}],
        [{"text": "Hello Goodbye", "line_id": 0, "rect": [1, 1, 100, 20]}],
    )
    assert not result["distinct_line_structure_preserved"]
    assert result["matched_rows"] == 0


def test_v2_tiny_font_fails_even_if_annotation_says_readable():
    data = v2_evidence()
    data[2][0]["items"][0]["font_px"] = 9
    result = run(data)
    assert result["outcome"] == "failed"
    assert result["scenes"][0]["scene_to_complete_ms"] is None
    assert any("pixel floor" in issue for issue in result["scenes"][0]["issues"])


@pytest.mark.parametrize("fault", ["counter_mismatch", "foreign_process", "incomplete"])
def test_v2_unbound_or_inconsistent_work_never_proves_zero_calls(fault):
    data = v2_evidence()
    work = data[2][0]["delivery_evidence"]["work"]
    if fault == "incomplete":
        work["completed"] = False
    else:
        work["events"] = [{"kind": "local_inference", "process_id": 999 if fault == "foreign_process" else 123}]
    result = run(data)
    assert result["scenes"][0]["model_zero_calls_proven"] is None
    assert result["outcome"] == "failed"


def test_blank_clear_is_never_included_in_translation_latency_percentiles():
    result = run(evidence(blank=True))
    assert result["metrics"]["scene_latency_p50_ms"] is None
    assert result["metrics"]["scene_latency_p95_ms"] is None
    assert result["metrics"]["blank_clear_p50_ms"] == pytest.approx(400)
