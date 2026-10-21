"""Preload decisions follow the actual device budget, without running models."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import psutil
import pytest

from pingyitong.components.llm import model_pool as pool_module
from pingyitong.components.llm.backend import BackendKind, BackendProbe
from pingyitong.components.llm.model_pool import ModelPool
from pingyitong.config.schema import AppConfig
from pingyitong.constants import MODEL_ACCURATE, MODEL_FAST
from pingyitong.core import resource_monitor as monitor_module


@pytest.fixture
def resource_pool(monkeypatch):
    config = AppConfig()
    pool = ModelPool(config.model)
    sizes = {config.model.fast.file_name: 1024**3, config.model.accurate.file_name: 3 * 1024**3}
    files = {
        name: SimpleNamespace(stat=lambda size=size: SimpleNamespace(st_size=size))
        for name, size in sizes.items()
    }
    monkeypatch.setattr(pool_module.paths, "find_model_file", files.get)
    monkeypatch.setattr(pool_module, "_gpu_backend_available", lambda _executable: False)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=64 * 1024**3))
    monkeypatch.setattr(pool, "backend_probe", lambda: BackendProbe(
        BackendKind.LLAMA_SERVER, executable="llama-server.exe", cuda_available=False,
    ))
    return pool


@pytest.mark.parametrize("backend", ["cuda", "vulkan"])
@pytest.mark.parametrize("free_mb,expected", [(6000.0, True), (4000.0, False), (None, False)])
def test_gpu_preload_uses_vram_and_never_falls_back_to_large_system_ram(
    resource_pool, monkeypatch, backend, free_mb, expected,
):
    pool = resource_pool
    monkeypatch.setattr(pool, "backend_probe", lambda: BackendProbe(
        BackendKind.LLAMA_SERVER, executable="llama-server.exe", cuda_available=backend == "cuda",
    ))
    monkeypatch.setattr(pool_module, "_gpu_backend_available", lambda _executable: backend == "vulkan")
    monkeypatch.setattr(pool, "_gpu_free_memory_mb", lambda: free_mb)

    def unexpected_ram_query():
        pytest.fail("GPU residency must not use system RAM, even when VRAM is unknown")

    monkeypatch.setattr(psutil, "virtual_memory", unexpected_ram_query)
    assert pool._can_hold_both_models() is expected
    expected_targets = [MODEL_FAST, MODEL_ACCURATE] if expected else [MODEL_FAST]
    assert pool.preload_targets() == expected_targets


@pytest.mark.parametrize("available_mb,expected", [(8192.0, True), (4096.0, False)])
def test_cpu_backend_uses_system_ram_budget(resource_pool, monkeypatch, available_mb, expected):
    pool = resource_pool
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=available_mb * 1024**2))
    monkeypatch.setattr(pool, "_gpu_free_memory_mb", lambda: pytest.fail("CPU backend must not query VRAM"))
    assert pool._can_hold_both_models() is expected


def test_explicit_accurate_preload_does_not_automatically_add_fast(resource_pool, monkeypatch):
    pool = resource_pool
    pool._config.fast.preload = False
    pool._config.accurate.preload = True
    monkeypatch.setattr(pool, "_can_hold_both_models", lambda: pytest.fail("explicit selection needs no dual-model guess"))
    assert pool.preload_targets() == [MODEL_ACCURATE]
    assert not pool._both_models_resident


def test_explicit_dual_preload_is_respected_when_automatic_budget_would_fail(resource_pool, monkeypatch):
    pool = resource_pool
    pool._config.accurate.preload = True
    monkeypatch.setattr(pool, "_can_hold_both_models", lambda: pytest.fail("explicit user preload must be respected"))
    assert pool.preload_targets() == [MODEL_FAST, MODEL_ACCURATE]
    assert pool._both_models_resident


@pytest.mark.parametrize("fast_option", ["preload", "enabled"])
def test_automatic_accurate_is_not_started_without_selected_fast(resource_pool, monkeypatch, fast_option):
    pool = resource_pool
    setattr(pool._config.fast, fast_option, False)
    monkeypatch.setattr(pool, "_can_hold_both_models", lambda: pytest.fail("no first model means no automatic second model"))
    assert pool.preload_targets() == []


def test_dual_residency_flag_tracks_current_targets_instead_of_previous_budget(resource_pool, monkeypatch):
    pool = resource_pool
    monkeypatch.setattr(pool, "_can_hold_both_models", lambda: True)
    assert pool.preload_targets() == [MODEL_FAST, MODEL_ACCURATE]
    assert pool._both_models_resident
    monkeypatch.setattr(pool, "_can_hold_both_models", lambda: False)
    assert pool.preload_targets() == [MODEL_FAST]
    assert not pool._both_models_resident


class RefCountingNvml:
    def __init__(self, fail_stage=None):
        self.refs = 0
        self.calls = []
        self.fail_stage = fail_stage

    def nvmlInit(self):
        self.calls.append("init")
        if self.fail_stage == "init":
            raise RuntimeError("NVML initialization failed")
        self.refs += 1

    def nvmlShutdown(self):
        self.calls.append("shutdown")
        assert self.refs > 0
        self.refs -= 1

    def nvmlDeviceGetHandleByIndex(self, index):
        self.calls.append("handle")
        assert self.refs > 0 and index == 0
        if self.fail_stage == "handle":
            raise RuntimeError("no matching device")
        return "gpu-0"

    def nvmlDeviceGetMemoryInfo(self, handle):
        self.calls.append("memory")
        assert self.refs > 0 and handle == "gpu-0"
        if self.fail_stage == "memory":
            raise RuntimeError("memory query failed")
        return SimpleNamespace(free=6000 * 1024**2)

    def nvmlDeviceGetUtilizationRates(self, handle):
        assert self.refs > 0 and handle == "gpu-0"
        return SimpleNamespace(gpu=5.0)


def test_cold_start_initializes_and_releases_its_own_nvml_reference(monkeypatch):
    nvml = RefCountingNvml()
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    pool = ModelPool(AppConfig().model)
    assert pool._gpu_free_memory_mb() == 6000.0
    assert nvml.refs == 0
    assert nvml.calls == ["init", "handle", "memory", "shutdown"]


@pytest.mark.parametrize("failure", ["init", "handle", "memory"])
def test_nvml_failure_releases_only_a_successfully_acquired_reference(monkeypatch, failure):
    nvml = RefCountingNvml(fail_stage=failure)
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    pool = ModelPool(AppConfig().model)
    assert pool._gpu_free_memory_mb() is None
    assert nvml.refs == 0
    assert nvml.calls.count("shutdown") == (0 if failure == "init" else 1)


def test_preload_query_preserves_resource_monitors_existing_nvml_reference(monkeypatch):
    nvml = RefCountingNvml()
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    monkeypatch.setattr(monitor_module, "_pynvml", None)
    monkeypatch.setattr(monitor_module, "_NVML_READY", None)
    monitor = monitor_module.ResourceMonitor()
    try:
        assert monitor._sample_gpu() == (6000.0, 5.0)
        assert nvml.refs == 1
        assert monitor_module._NVML_READY is True
        assert ModelPool(AppConfig().model)._gpu_free_memory_mb() == 6000.0
        assert nvml.refs == 1, "preload must release only its own init, not the monitor's reference"
        assert monitor_module._pynvml is nvml
        assert monitor_module._NVML_READY is True
        assert monitor._sample_gpu() == (6000.0, 5.0)
        assert nvml.calls.count("init") == 2
        assert nvml.calls.count("shutdown") == 1
    finally:
        if nvml.refs:
            nvml.nvmlShutdown()
