"""Evidence tooling must not confuse publication, zero work and cache reuse."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from pingyitong.config.schema import AppConfig
from pingyitong.constants import MODEL_FAST, SEGMENT_DELIMITER
from pingyitong.core import ModelScheduler

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = script("audit_random_translation")
semantic_probe = script("audit_translation_semantics")


class Pool:
    def __init__(self, *responses):
        self.responses = list(responses)

    def translate_local(self, model_id, prompt, system_prompt, max_tokens=None):
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def translate(self, **kwargs):
        return self.translate_local(**kwargs)

    def api_mode_active(self):
        return False


def published(source="source", full="translation", *, compact="", status="accepted", reason="", cache=False):
    return {"source": source, "full_text": full, "display_text": compact,
            "source_retained": source == full, "quality_status": status,
            "quality_reason": reason, "from_cache": cache}


def step(*items, calls=0):
    return {"items": list(items), "model_call_count": calls}


def test_traces_real_local_boundary_once_for_direct_and_legacy_routes():
    pool = Pool("full generation", '[{"id":0,"checks":"111111"}]')
    trace = probe.LocalCallTrace(pool)
    pool.translate_local("model", "private source", "generation system", 45)
    pool.translate(model_id="model", prompt="private source plus candidate",
                   system_prompt="independent checks review", max_tokens=123)
    assert len(trace.calls) == 2
    assert [call["stage"] for call in trace.calls] == ["generation", "semantic_review"]
    assert trace.calls[0]["raw"] == "full generation"
    assert trace.calls[1]["raw"] == '[{"id":0,"checks":"111111"}]'
    assert all(call["elapsed_ms"] >= 0 for call in trace.calls)
    assert trace.calls[0]["max_tokens"] == 45
    assert len(trace.calls[0]["prompt_sha256"]) == 64
    assert "private source" not in json.dumps(trace.calls)
    trace.restore()
    assert pool.translate_local == trace.original


def test_local_failure_is_retained_as_a_call_instead_of_a_false_zero():
    pool = Pool(RuntimeError("local inference failed"))
    trace = probe.LocalCallTrace(pool)
    with pytest.raises(RuntimeError, match="local inference failed"):
        pool.translate_local("model", "source", "system")
    assert len(trace.calls) == 1
    assert trace.calls[0]["raw"] is None
    assert trace.calls[0]["error"]["type"] == "RuntimeError"


def test_synthetic_review_inputs_preserve_local_ids_and_candidate_mapping():
    pool = Pool('[{"id":0,"independent":"target","checks":"111111"}]')
    trace = probe.LocalCallTrace(pool)
    inputs = [{"id": 0, "source": "authored source", "candidate": "authored translation"}]
    pool.translate_local("model", json.dumps(inputs), "independent checks review")
    assert trace.calls[0]["review_inputs"] == inputs


def test_semantic_resolution_has_its_own_stage_and_bound_synthetic_inputs():
    pool = Pool('[{"id":0,"choice":"A","checks":"111111"}]')
    trace = probe.LocalCallTrace(pool)
    inputs = [{"id": 0, "source": "authored source", "A": "first", "B": "second"}]
    pool.translate_local("model", json.dumps(inputs), "semantic_resolution choice checks independent")
    call = trace.calls[0]
    assert call["stage"] == "semantic_resolution"
    assert call["resolution_inputs"] == inputs
    assert "review_inputs" not in call
    assert len(call["prompt_sha256"]) == 64
    assert call["raw"] == '[{"id":0,"choice":"A","checks":"111111"}]'


def test_compact_review_is_a_separate_call_with_exact_full_and_short_inputs():
    pool = Pool('[{"id":0,"choice":"B","checks":"111111"}]')
    trace = probe.LocalCallTrace(pool)
    inputs = [{"id": 0, "source": "authored source", "A": "reviewed complete text", "B": "short candidate"}]
    pool.translate_local("model", json.dumps(inputs),
                         "semantic_compact_review semantic_resolution independent checks")
    call = trace.calls[0]
    assert call["stage"] == "semantic_compact_review"
    assert call["compact_review_inputs"] == inputs
    assert "resolution_inputs" not in call and "review_inputs" not in call
    assert call["raw"] == '[{"id":0,"choice":"B","checks":"111111"}]'
    assert len(call["prompt_sha256"]) == 64 and call["elapsed_ms"] >= 0


@pytest.mark.parametrize("item", [
    published(full="source", status="uncertain", reason="review_invalid"),
    published(full="source"),
    published(reason="review_skipped_by_policy"),
])
def test_zero_model_calls_without_reviewed_translation_never_prove_a_hit(item):
    evidence = probe.cache_evidence(step(item), step(item))
    assert evidence["status"] == "not_applicable_no_reviewed_translation"
    assert not evidence["whole_frame_zero_model_calls_proven"]


def test_cache_proof_checks_full_and_compact_content_not_just_hit_count():
    first = step(published(compact="short"), calls=2)
    replay = step(published(compact="changed meaning", cache=True))
    evidence = probe.cache_evidence(first, replay)
    assert evidence["status"] == "failed_reviewed_output_not_reused"
    assert not evidence["whole_frame_zero_model_calls_proven"]
    correct = probe.cache_evidence(first, step(published(compact="short", cache=True)))
    assert correct["status"] == "verified_zero_model_calls"
    assert correct["whole_frame_zero_model_calls_proven"]


def test_scene_probe_starts_empty_runs_real_service_and_preserves_review_evidence():
    sources = ["Save the file before closing the window.", "The report contains 12 pages and 3 tables."]
    outputs = ["关闭窗口前，请保存文件。", "报告包含12页和3张表格。"]
    pool = Pool(SEGMENT_DELIMITER.join(outputs), json.dumps([
        {"id": index, "independent": output, "checks": "111111"}
        for index, output in enumerate(outputs)
    ]))
    config = AppConfig()
    config.schedule.mode = "manual"
    config.schedule.active_model = MODEL_FAST
    config.translation.glossary_enabled = False
    config.translation.compact_display = False
    scene = {"scene_id": "example", "category": "subtitle", "source_language": "en", "target_language": "zh",
             "rows": [{"id": str(index), "source": source, "reference": output,
                       "invariants": "Human relation review required"}
                      for index, (source, output) in enumerate(zip(sources, outputs, strict=True))]}
    result = probe.run_scene(scene, config, pool, ModelScheduler(config.schedule), probe.LocalCallTrace(pool))
    assert result["cache_entries_before_fresh"] == 0
    first, replay = result["passes"]
    assert first["model_call_count"] == 2
    assert [call["stage"] for call in first["model_calls"]] == ["generation", "semantic_review"]
    assert [item["full_text"] for item in first["items"]] == outputs
    assert all(item["quality_status"] == "accepted" for item in first["items"])
    assert first["human_semantic_review"] == "pending"
    assert first["stage_timings_ms"]["translate"] >= 0
    assert first["unmapped_row_ids"] == []
    assert replay["model_call_count"] == 0 and replay["model_calls"] == []
    assert result["cache_proof"]["whole_frame_zero_model_calls_proven"]


def test_auto_language_corpus_is_rejected_before_any_model_load(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema": "pingyitong-random-scenes-v1", "scenes": [{
        "scene_id": "bad", "source_language": "auto", "target_language": "zh", "rows": [],
    }]}), encoding="utf-8")
    with pytest.raises(ValueError, match="explicitly"):
        probe.load_corpus(path)


@pytest.mark.parametrize("verified", [True, False])
def test_identity_wait_is_explicit_before_measurement_and_unverified_is_visible(verified):
    class IdentityPool:
        def wait_for_model_fingerprints(self, models, timeout):
            assert models == ["model"] and timeout == 30
            return verified

        def model_content_fingerprint(self, model_id):
            assert model_id == "model"
            return "sha256:weights" if verified else "pending:session"

    identity = semantic_probe.wait_for_model_identity(IdentityPool(), "model")
    assert identity["identity_status"] == ("verified" if verified else "identity_unverified")
    assert identity["model_content_fingerprint"] == ("sha256:weights" if verified else "pending:session")
    assert identity["identity_wait_ms"] >= 0


def test_serial_measurement_unloads_only_previous_owned_model_before_next_load():
    class MeasurementPool:
        def __init__(self):
            self.events = []

        def unload(self, model_id):
            self.events.append(("unload", model_id))

        def ensure_loaded(self, model_id):
            self.events.append(("load", model_id))
            return True

        def wait_for_model_fingerprints(self, models, timeout):
            self.events.append(("identity_wait", models[0], timeout))
            return True

        def model_content_fingerprint(self, model_id):
            return "sha256:" + model_id

    pool = MeasurementPool()
    first = probe.prepare_model_measurement(pool, "fast", None)
    second = probe.prepare_model_measurement(pool, "accurate", "fast")
    assert pool.events == [("load", "fast"), ("identity_wait", "fast", 30),
                           ("unload", "fast"), ("load", "accurate"),
                           ("identity_wait", "accurate", 30)]
    assert first["previous_model_unloaded"] is None
    assert first["previous_model_unload_ms"] == 0
    assert second["previous_model_unloaded"] == "fast"
    assert second["previous_model_unload_ms"] >= 0
    assert first["model_load_ms"] >= 0 and second["model_load_ms"] >= 0
    assert second["measurement_policy"] == "serial_owned_pool_single_model_v1"
