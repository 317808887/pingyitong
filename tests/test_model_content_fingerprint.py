"""Weight-content cache isolation without starting an inference backend."""

from __future__ import annotations

import hashlib
import os
import threading
from types import SimpleNamespace

import pytest

from pingyitong.components.llm import model_pool as pool_module
from pingyitong.components.llm.model_pool import ModelPool
from pingyitong.config.schema import AppConfig
from pingyitong.constants import MODEL_FAST
from tests.test_semantic_review import SOURCE, reviewed_service


@pytest.fixture
def weights(tmp_path, monkeypatch):
    model = tmp_path / "same-name.gguf"
    model.write_bytes(b"GGUF-first-weights")
    monkeypatch.setattr(pool_module.paths, "resolve_model_path", lambda _name: model)
    return model


def make_pool():
    config = AppConfig()
    config.model.accurate.enabled = False
    return ModelPool(config.model)


def replace_weights(path, content=b"GGUF-other-weights"):
    previous = path.stat()
    path.write_bytes(content)
    os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns + 1_000_000))


def test_ready_identity_hashes_complete_contents_and_is_portable(weights, monkeypatch):
    original = pool_module._hash_model_weights
    calls = []

    def counted(path, signature):
        calls.append(path)
        return original(path, signature)

    monkeypatch.setattr(pool_module, "_hash_model_weights", counted)
    first = make_pool()
    assert first.wait_for_model_fingerprints([MODEL_FAST], timeout=2)
    expected = "sha256:" + hashlib.sha256(weights.read_bytes()).hexdigest()
    assert first.model_content_fingerprint(MODEL_FAST) == expected
    for _ in range(20):
        assert first.model_content_fingerprint(MODEL_FAST) == expected
    assert len(calls) == 1, "a stable file must be hashed once per pool, not per text"
    second = make_pool()
    assert second.wait_for_model_fingerprints([MODEL_FAST], timeout=2)
    assert second.model_content_fingerprint(MODEL_FAST) == expected


def test_pending_identity_returns_without_waiting_and_never_crosses_pools(weights, monkeypatch):
    original = pool_module._hash_model_weights
    release = threading.Event()
    started = threading.Event()

    def blocked(path, signature):
        started.set()
        assert release.wait(3)
        return original(path, signature)

    monkeypatch.setattr(pool_module, "_hash_model_weights", blocked)
    first, second = make_pool(), make_pool()
    try:
        identity = first.model_content_fingerprint(MODEL_FAST)
        assert started.wait(1)
        other = second.model_content_fingerprint(MODEL_FAST)
        assert identity.startswith("pending:") and other.startswith("pending:")
        assert identity != other
        assert not first.wait_for_model_fingerprints([MODEL_FAST], timeout=0)
        assert not release.is_set(), "identity lookup must not wait for GGUF reads"
    finally:
        release.set()
        assert first.wait_for_model_fingerprints([MODEL_FAST], timeout=2)
        assert second.wait_for_model_fingerprints([MODEL_FAST], timeout=2)


def test_same_filename_replacement_changes_service_cache_identity(weights):
    pool = make_pool()
    assert pool.wait_for_model_fingerprints([MODEL_FAST], timeout=2)
    service, _ = reviewed_service()
    service._config.model.accurate.enabled = False
    service._pool = pool
    before = service._cache_key(SOURCE, "en", "zh")
    original = pool.model_content_fingerprint(MODEL_FAST)
    replace_weights(weights)
    assert pool.wait_for_model_fingerprints([MODEL_FAST], timeout=2)
    assert pool.model_content_fingerprint(MODEL_FAST) != original
    assert service._cache_key(SOURCE, "en", "zh") != before


def test_missing_weights_have_session_isolation_and_no_portable_identity(weights):
    weights.unlink()
    first, second = make_pool(), make_pool()
    identity = first.model_content_fingerprint(MODEL_FAST)
    assert identity.startswith("unavailable:")
    assert second.model_content_fingerprint(MODEL_FAST) != identity
    assert not first.wait_for_model_fingerprints([MODEL_FAST], timeout=0)


def test_file_changed_during_background_read_cannot_publish_old_hash(weights, monkeypatch):
    original = pool_module._hash_model_weights
    entered, release = threading.Event(), threading.Event()

    def blocked(path, signature):
        entered.set()
        assert release.wait(3)
        return original(path, signature)

    monkeypatch.setattr(pool_module, "_hash_model_weights", blocked)
    pool = make_pool()
    pool.model_content_fingerprint(MODEL_FAST)
    assert entered.wait(1)
    record = pool._weight_fingerprints[MODEL_FAST]
    try:
        replace_weights(weights)
    finally:
        release.set()
    assert record.finished.wait(2)
    assert not record.digest, "a checksum for the old stat must not become portable"
    assert pool.wait_for_model_fingerprints([MODEL_FAST], timeout=2)
    assert pool.model_content_fingerprint(MODEL_FAST) == "sha256:" + hashlib.sha256(weights.read_bytes()).hexdigest()


def test_replacement_beneath_loaded_engine_stays_isolated_until_reload(weights):
    pool = make_pool()
    engine = SimpleNamespace(model_path=weights, loaded=False)
    engine.is_loaded = lambda: engine.loaded
    engine.is_available = lambda: True
    engine.load = lambda: setattr(engine, "loaded", True) or True
    engine.unload = lambda: setattr(engine, "loaded", False)
    engine.warm_up = lambda: True
    pool._engines[MODEL_FAST] = engine
    assert pool.ensure_loaded(MODEL_FAST)
    assert pool.wait_for_model_fingerprints([MODEL_FAST], timeout=2)
    replace_weights(weights)
    assert pool.model_content_fingerprint(MODEL_FAST).startswith("changed-loaded:")
    assert not pool.wait_for_model_fingerprints([MODEL_FAST], timeout=0)
    pool.unload(MODEL_FAST)
    assert pool.ensure_loaded(MODEL_FAST)
    assert pool.wait_for_model_fingerprints([MODEL_FAST], timeout=2)
    assert pool.model_content_fingerprint(MODEL_FAST) == "sha256:" + hashlib.sha256(weights.read_bytes()).hexdigest()
