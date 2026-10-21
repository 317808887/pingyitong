"""Backend-call observation gates; no real model, screen or installed-app claims."""
from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest

from pingyitong.components.capture.geometry import Rect
from pingyitong.components.llm.model_pool import ModelPool
from pingyitong.components.render import RenderFrame, RenderItem
from pingyitong.config.schema import AppConfig
from pingyitong.constants import MODEL_FAST
from pingyitong.utils import execution_evidence as evidence


@pytest.fixture
def observed(monkeypatch):
    events = []
    monkeypatch.setattr(evidence, "evidence_enabled", lambda: True)
    monkeypatch.setattr(evidence, "save_execution_event", lambda event: events.append(dict(event)))
    return events


class Backend:
    def __init__(self):
        self.loaded = False
        self.calls = 0

    def is_loaded(self):
        return self.loaded

    def is_available(self):
        return True

    def load(self):
        self.loaded = True
        return True

    def warm_up(self):
        return True

    def translate_batch(self, prompts, system_prompt, max_tokens=None):
        self.calls += 1
        return ["private generated text"]


def test_actual_pool_call_sites_separate_load_warmup_and_inference(observed, tmp_path):
    pool = ModelPool(AppConfig().model)
    backend = Backend()
    pool._engines[MODEL_FAST] = backend
    weights = tmp_path / "fake.gguf"
    weights.write_bytes(b"GGUF-test-only")
    pool._model_weight_path = lambda _model: weights
    frame = SimpleNamespace(seq=11, _translation_epoch=3)
    with evidence.trace_translation(frame) as first:
        assert pool.translate_local(MODEL_FAST, "private source text", "private prompt") == "private generated text"
    assert first.snapshot()["counts"] == {
        "model_load": 1, "model_warmup": 1, "local_inference": 1, "api_inference": 0,
    }
    with evidence.trace_translation(frame) as second:
        pool.translate_local(MODEL_FAST, "private source text", "private prompt")
    assert second.snapshot()["counts"]["local_inference"] == 1
    assert second.snapshot()["counts"]["model_load"] == second.snapshot()["counts"]["model_warmup"] == 0
    assert backend.calls == 2
    assert all(event["successful"] is True for event in observed)
    assert all(event["elapsed_seconds"] >= 0 for event in observed)
    serialized = json.dumps([first.snapshot(), second.snapshot(), observed])
    assert "private source text" not in serialized
    assert "private generated text" not in serialized
    assert "private prompt" not in serialized


def test_disabled_personal_session_collects_nothing(monkeypatch):
    monkeypatch.setattr(evidence, "evidence_enabled", lambda: False)
    monkeypatch.setattr(evidence, "save_execution_event", lambda _: pytest.fail("personal evidence must not be written"))
    with (evidence.trace_translation(SimpleNamespace(seq=1)) as trace,
          evidence.model_work("local_inference", "model") as event):
        assert event is None
    assert trace is None


def test_concurrent_preload_is_not_attributed_to_frame(observed):
    def preload():
        with evidence.model_work("model_load", "other-model") as event:
            event["successful"] = True
    with evidence.trace_translation(SimpleNamespace(seq=1)) as trace:
        worker = threading.Thread(target=preload)
        worker.start()
        worker.join()
    assert trace.snapshot()["counts"]["model_load"] == 0
    assert len(observed) == 1 and observed[0]["trace_id"] is None


def test_failed_call_is_counted_and_exception_text_is_not_persisted(observed):
    with (pytest.raises(RuntimeError),
          evidence.trace_translation(SimpleNamespace(seq=2)) as failed,
          evidence.model_work("local_inference", "model")):
        raise RuntimeError("private source in an exception")
    assert failed.snapshot()["completed"] is False
    assert failed.snapshot()["counts"]["local_inference"] == 1
    assert observed[0]["exception_type"] == "RuntimeError"
    assert observed[0]["successful"] is False
    assert "private source" not in json.dumps(observed)
    with evidence.model_work("model_load", "startup-model"):
        pass
    assert observed[-1]["trace_id"] is None, "failed scope must reset context"


def cached_frame():
    frame = RenderFrame(4, Rect(0, 0, 300, 100), [
        RenderItem(Rect(10, 10, 200, 20), "Bonjour", "Hello", from_cache=True),
    ], diagnostics={"trusted_cache_reuse": 1})
    frame._execution_evidence = {
        "completed": True, "counts": {"local_inference": 0, "api_inference": 0},
    }
    return frame


def test_true_cache_restore_requires_observed_zero_calls_and_trusted_reuse():
    frame = cached_frame()
    result = evidence.delivery_evidence(frame)
    assert result["delivery_route"] == "trusted_cache_restore"
    assert result["zero_inference_calls_proven"] is True
    frame.diagnostics["trusted_cache_reuse"] = 0
    assert evidence.delivery_evidence(frame)["delivery_route"] == "unobserved"
    frame.diagnostics["trusted_cache_reuse"] = 1
    frame._execution_evidence["counts"]["local_inference"] = 1
    assert evidence.delivery_evidence(frame)["delivery_route"] == "model_with_cache"


def test_retained_cached_flag_is_never_called_a_new_cache_restore():
    result = evidence.delivery_evidence(cached_frame(), retained_count=1)
    assert result["delivery_route"] == "retained_confirmed"
    assert result["zero_inference_calls_proven"] is True


@pytest.mark.parametrize("trace", [None, {"completed": False}, {"completed": True, "coalesced_unknown": True}])
def test_missing_or_partial_observation_cannot_prove_zero_calls(trace):
    frame = cached_frame()
    frame._execution_evidence = trace
    result = evidence.delivery_evidence(frame)
    assert result["zero_inference_calls_proven"] is None
    assert result["delivery_route"] != "trusted_cache_restore"
