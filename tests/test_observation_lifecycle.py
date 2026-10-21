"""Scene freshness and latency must describe real outstanding display work."""
from __future__ import annotations

import os
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6.QtWidgets import QApplication

from pingyitong.components.capture import CaptureFrame, FrameDiffer
from pingyitong.components.capture.geometry import Rect
from pingyitong.components.ocr import RecognizedFrame, TextBlock
from pingyitong.components.render import RenderFrame, RenderItem
from pingyitong.core.pipeline import (
    Pipeline,
    _carry_observation,
    _coalesce_capture,
    _coalesce_recognized,
    _coalesce_render,
)

APP = QApplication.instance() or QApplication([])
ORIGIN = Rect(-200, 100, 400, 160)
LEFT = Rect(-180, 120, 100, 30)
RIGHT = Rect(30, 120, 100, 30)


def pipeline():
    config = SimpleNamespace(capture=SimpleNamespace(
        fps=1, monitor_interval_ms=50, enable_diff=True, monitors=[],
        diff_pixel_threshold=24, diff_min_area_ratio=0.0004, diff_block_size=16,
    ))
    return Pipeline(config, SimpleNamespace(last_capture_cost=0.999), FrameDiffer(),
                    object(), object(), Mock(is_visible=True))


def capture(seq, image=None):
    return CaptureFrame(seq, 10.0 + seq / 20, np.zeros((160, 400, 3), dtype=np.uint8)
                        if image is None else image, ORIGIN, capture_cost=0.002)


def render(seq, image, items):
    frame = RenderFrame(seq, ORIGIN, items, source_items=[
        RenderItem(item.box, item.source_text, item.source_text) for item in items
    ])
    frame.__dict__["_capture_image"] = image
    frame.__dict__["_captured_at"] = 10.0
    return frame


def accept(p, frame):
    p._result_queue.put(frame, seq=frame.seq)
    p._render_tick()


def test_legacy_one_fps_does_not_delay_monitoring():
    p = pipeline()
    assert p._monitor_fps() == 20
    p._config.capture.fps = 30
    assert p._monitor_fps() == 20
    p._config.capture.monitor_interval_ms = 100
    assert p._monitor_fps() == 10


def test_stable_twenty_hz_observations_trigger_no_extra_ocr_and_do_not_starve_work():
    p = pipeline()
    first = capture(1)
    assert p._should_process(first)
    for seq in range(2, 22):
        assert not p._should_process(capture(seq))
        assert p._is_current_frame(first)
    assert len(p.latency_snapshot()["monitor_intervals"]) == 20
    assert p._status.monitor_interval == pytest.approx(0.05)


def test_partial_source_change_keeps_stable_inflight_sentence():
    p = pipeline()
    first = capture(1)
    p._should_process(first)
    current = capture(2)
    current.image[20:50, 20:120] = 255
    p._should_process(current)
    frame = render(1, first.image, [
        RenderItem(LEFT, "旧左", "old left"), RenderItem(RIGHT, "稳定右", "stable right"),
    ])
    assert p._is_current_frame(frame)
    p._prune_stale_regions(frame)
    assert [item.text for item in frame.items] == ["稳定右"]
    assert not frame.is_full_frame
    assert LEFT not in frame.changed_regions


def test_changed_only_source_cancels_old_model_work_without_waiting_for_new_translation():
    p = pipeline()
    first = capture(1)
    p._should_process(first)
    frame = render(1, first.image, [RenderItem(LEFT, "旧左", "old left")])
    current = capture(2)
    current.image[20:50, 20:120] = 255
    p._should_process(current)
    assert not p._is_current_frame(frame)


def test_big_scene_change_invalidates_inflight_work_even_if_a_small_patch_matches():
    p = pipeline()
    first = capture(1)
    p._should_process(first)
    current = capture(2, np.full_like(first.image, 255))
    current.image[20:50, 230:330] = 0
    assert p._should_process(current)
    assert current.is_full_frame
    assert not p._is_current_frame(first)


def test_changed_translation_disappears_before_slow_replacement_and_unchanged_stays():
    p = pipeline()
    first = capture(1)
    p._should_process(first)
    accept(p, render(1, first.image, [RenderItem(LEFT, "旧左", "old left"),
                                    RenderItem(RIGHT, "稳定右", "stable right")]))
    current = capture(2)
    current.image[20:50, 20:120] = 255
    p._should_process(current)
    p._render_tick()
    assert [item.text for item in p.snapshot_items()] == ["稳定右"]


def test_unrelated_cursor_does_not_clear_or_cancel_confirmed_text():
    p = pipeline()
    first = capture(1)
    p._should_process(first)
    frame = render(1, first.image, [RenderItem(LEFT, "保留", "keep")])
    accept(p, frame)
    current = capture(2)
    current.image[130:140, 350:360] = 255
    p._should_process(current)
    p._render_tick()
    assert [item.text for item in p.snapshot_items()] == ["保留"]
    assert p._is_current_frame(frame)


@pytest.mark.parametrize("stage", ["capture", "ocr", "render"])
def test_coalescing_preserves_oldest_outstanding_wait(stage):
    if stage == "capture":
        a, b = capture(1), capture(2)
        a.is_full_frame = b.is_full_frame = False
        merge = _coalesce_capture
    elif stage == "ocr":
        a = RecognizedFrame(1, ORIGIN, [TextBlock("left", LEFT)], is_full_frame=False)
        b = RecognizedFrame(2, ORIGIN, [TextBlock("right", RIGHT)], is_full_frame=False)
        merge = _coalesce_recognized
    else:
        a, b = RenderFrame(1, ORIGIN, [], is_full_frame=False), RenderFrame(2, ORIGIN, [], is_full_frame=False)
        merge = _coalesce_render
    a.__dict__.update(_change_started_at=10.0, _observation_lower_bound=9.95)
    b.__dict__.update(_change_started_at=12.0, _observation_lower_bound=11.95)
    result = merge(a, b)
    assert result.__dict__["_change_started_at"] == 10.0
    assert result.__dict__["_observation_lower_bound"] == 9.95


def test_submission_is_not_misreported_as_actual_paint_and_routes_are_separate(monkeypatch):
    p = pipeline()
    monkeypatch.setattr("pingyitong.core.pipeline.time.perf_counter", lambda: 10.4)
    frame = render(1, capture(1).image, [
        RenderItem(LEFT, "缓存", "cached", from_cache=True),
        RenderItem(RIGHT, "模型", "model", from_cache=False),
    ])
    frame.__dict__.update(_change_started_at=10.0, _observation_lower_bound=9.95)
    accept(p, frame)
    assert p._status.submit_latency == pytest.approx(0.4)
    assert not p._status.presentation_confirmed
    assert p._status.end_to_end_latency == 0
    assert frame.__dict__["_latency_route"] == "mixed"
    p._on_presented(frame, 10.5)
    assert p._status.presentation_confirmed
    assert p._status.observation_latency == pytest.approx(0.5)
    assert p._status.end_to_end_latency == pytest.approx(0.55)
    assert p.latency_snapshot()["samples"][0]["scope"] == "initial"
    second = render(2, capture(2).image, [RenderItem(LEFT, "缓存", "cached", from_cache=True)])
    second.is_full_frame = False
    accept(p, second)
    p._on_presented(second, 10.6)
    assert p.latency_snapshot()["samples"][-1]["scope"] == "partial"
    assert p.latency_snapshot()["samples"][-1]["route"] == "cache"


def test_equal_luminance_colour_edit_is_detected():
    old = np.zeros((100, 100, 3), dtype=np.uint8)
    new = old.copy()
    old[20:30, 20:50] = (255, 0, 0)
    new[20:30, 20:50] = (0, 49, 0)
    assert FrameDiffer().compute(old, new).changed


def test_confirmed_previous_translation_has_distinct_retention_route():
    p = pipeline()
    image = capture(1).image
    first = render(1, image, [RenderItem(LEFT, "Bonjour", "Hello", from_cache=True)])
    accept(p, first)
    missed = render(2, image, [])
    missed._execution_evidence = {"completed": True, "counts": {"local_inference": 0, "api_inference": 0}}
    accept(p, missed)
    assert len(missed.items) == 1
    assert missed._delivery_evidence["delivery_route"] == "retained_confirmed"
    assert missed._delivery_evidence["confirmed_retained_items"] == 1
    assert missed._delivery_evidence["zero_inference_calls_proven"] is True
    assert missed._refresh_kind == "full_refresh", "An invalidated full scan is not a first session paint"


def test_coalescing_source_evidence_replaces_changed_line_and_keeps_other_line():
    image = capture(1).image
    first, second = render(1, image, []), render(2, image, [])
    first.is_full_frame = second.is_full_frame = False
    first._source_lines = [
        {"text": "old", "rect": [LEFT.x, LEFT.y, LEFT.width, LEFT.height]},
        {"text": "stable", "rect": [RIGHT.x, RIGHT.y, RIGHT.width, RIGHT.height]},
    ]
    second.changed_regions = [LEFT]
    second._source_lines = [{"text": "new", "rect": [LEFT.x, LEFT.y, LEFT.width, LEFT.height]}]
    second._execution_evidence = {"completed": True, "counts": {"local_inference": 0, "api_inference": 0}, "events": []}
    result = _coalesce_render(first, second)
    assert [line["text"] for line in result._source_lines] == ["stable", "new"]
    assert result._execution_evidence["coalesced_unknown"]
    assert not result._execution_evidence["completed"], "Unknown older work cannot prove zero inference"


def test_tiny_faint_edit_on_large_desktop_is_not_lost_by_area_ratio():
    old = np.full((2000, 3000, 3), 250, dtype=np.uint8)
    new = old.copy()
    new[0:2, 0:3] = 245
    result = FrameDiffer().compute(old, new)
    assert result.changed
    assert any(x == 0 and y == 0 for x, y, _w, _h in result.regions)
    assert result.change_ratio < 0.0004


def test_uncertain_old_source_is_not_promoted_by_pixel_confirmation():
    p = pipeline()
    image = capture(1).image
    uncertain = RenderItem(LEFT, "original", "original", quality_status="uncertain")
    accept(p, render(1, image, [uncertain]))
    missed = render(2, image, [])
    accept(p, missed)
    assert p.snapshot_items() == []


def test_retry_does_not_reset_wait_for_the_same_untranslated_source(monkeypatch):
    p = pipeline()
    first = capture(1)
    p._should_process(first)
    monkeypatch.setattr("pingyitong.core.pipeline.time.perf_counter", lambda: 10.1)
    frame = render(1, first.image, [])
    frame.source_items = [RenderItem(LEFT, "source", "source")]
    frame.retry_regions = [LEFT]
    frame.__dict__.update(_change_started_at=10.05, _observation_lower_bound=10.0)
    accept(p, frame)
    monkeypatch.setattr("pingyitong.core.pipeline.time.perf_counter", lambda: 12.0)
    retry = capture(40)
    assert p._should_process(retry)
    assert retry.__dict__["_change_started_at"] == 10.05
    assert retry.__dict__["_observation_lower_bound"] == 10.0


@pytest.mark.parametrize("clear_output", [False, True])
@pytest.mark.parametrize("capture_delay", [0.05, 1.25])
def test_invalidation_after_long_session_owns_its_discovery_window(
    monkeypatch, clear_output, capture_delay,
):
    p = pipeline()
    p._session_started_at = 10.0
    assert p._should_process(capture(1))
    for seq in range(2, 22):
        assert not p._should_process(capture(seq))

    monkeypatch.setattr("pingyitong.core.pipeline.time.perf_counter", lambda: 200.0)
    p.invalidate(clear_output=clear_output)
    fresh = capture(22)
    fresh.timestamp = 200.0 + capture_delay
    fresh.__dict__["_translation_epoch"] = p._translation_epoch
    assert p._should_process(fresh)
    assert fresh.__dict__["_change_started_at"] == fresh.timestamp
    assert fresh.__dict__["_observation_lower_bound"] == 200.0
    assert p._session_started_at == 10.0

    frame = render(fresh.seq, fresh.image, [RenderItem(LEFT, "缓存", "cached")])
    _carry_observation(fresh, frame)
    frame.__dict__.update(
        _captured_at=fresh.timestamp,
        _translation_epoch=p._translation_epoch,
        _latency_route="cache",
    )
    p._on_presented(frame, fresh.timestamp + 0.3)
    assert p._status.observation_latency == pytest.approx(0.3)
    assert p._status.detection_wait_bound == pytest.approx(capture_delay)
    assert p._status.end_to_end_latency == pytest.approx(capture_delay + 0.3)
    # A delayed capture is a real wait, even when longer than the 50ms monitor.
    assert p.latency_snapshot()["samples"][-1]["appearance_lower_bound"] == 200.0


def test_invalidation_does_not_erase_later_same_source_retry_wait(monkeypatch):
    p = pipeline()
    p._session_started_at = 10.0
    monkeypatch.setattr("pingyitong.core.pipeline.time.perf_counter", lambda: 200.0)
    p.invalidate()
    first = capture(1)
    first.timestamp = 200.05
    first.__dict__["_translation_epoch"] = p._translation_epoch
    assert p._should_process(first)

    frame = render(first.seq, first.image, [])
    _carry_observation(first, frame)
    frame.__dict__.update(
        _captured_at=first.timestamp, _translation_epoch=p._translation_epoch,
    )
    frame.source_items = [RenderItem(LEFT, "source", "source")]
    frame.retry_regions = [LEFT]
    monkeypatch.setattr("pingyitong.core.pipeline.time.perf_counter", lambda: 200.1)
    accept(p, frame)

    monkeypatch.setattr("pingyitong.core.pipeline.time.perf_counter", lambda: 202.0)
    retry = capture(40)
    retry.timestamp = 202.0
    retry.__dict__["_translation_epoch"] = p._translation_epoch
    assert p._should_process(retry)
    assert retry.__dict__["_change_started_at"] == 200.05
    assert retry.__dict__["_observation_lower_bound"] == 200.0

    result = render(retry.seq, retry.image, [RenderItem(LEFT, "译文", "source")])
    _carry_observation(retry, result)
    result.__dict__.update(
        _captured_at=retry.timestamp, _translation_epoch=p._translation_epoch,
    )
    p._on_presented(result, 203.0)
    assert p._status.end_to_end_latency == pytest.approx(3.0)


def test_old_epoch_submission_and_paint_cannot_overwrite_latency_status(monkeypatch):
    p = pipeline()
    first = capture(1)
    first.__dict__["_translation_epoch"] = p._translation_epoch
    assert p._should_process(first)
    old = render(first.seq, first.image, [RenderItem(LEFT, "旧译文", "source")])
    _carry_observation(first, old)
    old.__dict__.update(
        _captured_at=first.timestamp, _translation_epoch=p._translation_epoch,
    )
    p._on_presented(old, 10.5)
    samples = p.latency_snapshot()["samples"]
    completed_latency = p._status.end_to_end_latency

    monkeypatch.setattr("pingyitong.core.pipeline.time.perf_counter", lambda: 200.0)
    p.invalidate()
    p._on_presented(old, 200.2)
    accept(p, old)
    assert p.latency_snapshot()["samples"] == samples
    assert p._status.end_to_end_latency == completed_latency
    assert p._last_render_seq == 0
    assert p.snapshot_items() == []


def test_ocr_timing_owns_its_capture_cost_even_after_new_capture():
    p = pipeline()
    p._ocr = Mock()
    p._ocr.recognize_frame.return_value = RecognizedFrame(1, ORIGIN, [])
    p._frame_queue.put(capture(1), seq=1)
    stop = threading.Event()
    worker = threading.Thread(target=p._ocr_loop, args=(stop,))
    worker.start()
    try:
        recognized = p._ocr_queue.get(timeout=2)
        assert recognized is not None
        assert recognized.__dict__["_capture_cost"] == 0.002
    finally:
        stop.set()
        p._frame_queue.wake_all()
        worker.join(timeout=2)
    assert not worker.is_alive()
