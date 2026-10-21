"""Dense overlay updates retain confirmed text and retry incomplete translation."""
from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6.QtWidgets import QApplication

from pingyitong.components.capture import CaptureFrame
from pingyitong.components.capture.geometry import Rect
from pingyitong.components.render import RenderFrame, RenderItem
from pingyitong.core.pipeline import Pipeline, _coalesce_render

_APP = QApplication.instance() or QApplication([])
ORIGIN = Rect(-100, 200, 800, 600)
PARAGRAPH = Rect(-80, 230, 600, 75)


def make_pipeline():
    config = SimpleNamespace(capture=SimpleNamespace(
        monitors=[], enable_diff=True, diff_pixel_threshold=24,
        diff_min_area_ratio=0.0004, diff_block_size=16,
    ))
    differ = Mock()
    differ.compute.return_value = SimpleNamespace(changed=True, regions=[(24, 38, 12, 12)])
    return Pipeline(config, SimpleNamespace(last_capture_cost=0), differ, object(), object(), Mock(is_visible=True))


def capture(seq):
    return CaptureFrame(seq, float(seq), np.zeros((600, 800, 3), dtype=np.uint8), ORIGIN)


def accept(p, frame):
    p._result_queue.put(frame, seq=frame.seq)
    p._render_tick()


def test_tiny_changed_roi_recognizes_entire_previous_paragraph():
    p = make_pipeline()
    p._screen_items = [RenderItem(PARAGRAPH, "完整译文", "The complete paragraph contains three lines.")]
    p._previous_frame = capture(1)
    newest = capture(2)
    assert p._should_process(newest)
    assert not newest.is_full_frame
    regions = [area.to_global(ORIGIN) for area in newest.changed_regions]
    assert any(area.intersection(PARAGRAPH).area == PARAGRAPH.area for area in regions)


def test_failed_refresh_retains_translation_only_for_confirmed_same_source():
    p = make_pipeline()
    old = RenderItem(PARAGRAPH, "已验译文", "Save the file before closing.")
    accept(p, RenderFrame(1, ORIGIN, [old]))
    source = RenderItem(PARAGRAPH, old.source_text, old.source_text)
    accept(p, RenderFrame(2, ORIGIN, [], source_items=[source], retry_regions=[PARAGRAPH]))
    assert [item.text for item in p.snapshot_items()] == ["已验译文"]
    assert p._overlay.apply.call_args.args[0][0].text == "已验译文"
    changed = RenderItem(PARAGRAPH, "Do not save the file.", "Do not save the file.")
    accept(p, RenderFrame(3, ORIGIN, [], source_items=[changed], retry_regions=[PARAGRAPH]))
    assert p.snapshot_items() == []


def test_blank_refresh_clears_instead_of_sticking_old_translation():
    p = make_pipeline()
    accept(p, RenderFrame(1, ORIGIN, [RenderItem(PARAGRAPH, "旧译文", "Old text.")]))
    accept(p, RenderFrame(2, ORIGIN, [], source_items=[]))
    assert p.snapshot_items() == []


def test_pending_full_result_survives_same_source_failed_increment():
    old = RenderItem(PARAGRAPH, "已验译文", "Save the file.")
    previous = RenderFrame(1, ORIGIN, [old], source_items=[RenderItem(PARAGRAPH, old.source_text, old.source_text)])
    latest = RenderFrame(2, ORIGIN, [], is_full_frame=False, changed_regions=[PARAGRAPH],
                         source_items=[RenderItem(PARAGRAPH, old.source_text, old.source_text)], retry_regions=[PARAGRAPH])
    result = _coalesce_render(previous, latest)
    assert [item.text for item in result.items] == ["已验译文"]
    assert result.retry_regions == [PARAGRAPH]


def test_static_screen_retries_missing_region_three_times_then_stops(monkeypatch):
    p = make_pipeline()
    p._previous_frame = capture(1)
    p._differ.compute.return_value = SimpleNamespace(changed=False, regions=[])
    now = [10.0]
    monkeypatch.setattr("pingyitong.core.pipeline.time.perf_counter", lambda: now[0])
    source = RenderItem(PARAGRAPH, "Save the file.", "Save the file.")
    for index in range(3):
        accept(p, RenderFrame(index + 2, ORIGIN, [], source_items=[source], retry_regions=[PARAGRAPH]))
        assert not p._should_process(capture(index + 10))
        now[0] += 10
        retry = capture(index + 20)
        assert p._should_process(retry)
        assert retry.__dict__["_translation_retry"] is True
        assert retry.is_full_frame is False
        assert retry.changed_regions == [PARAGRAPH.to_local(ORIGIN)]
    accept(p, RenderFrame(40, ORIGIN, [], source_items=[source], retry_regions=[PARAGRAPH]))
    now[0] += 100
    assert not p._should_process(capture(41))
    assert "暂未译出 1 处" in p._status.as_text()
    accept(p, RenderFrame(42, ORIGIN, [RenderItem(PARAGRAPH, "保存文件。", source.text)],
                         source_items=[source]))
    assert p._status.pending_translations == 0
    assert "待补译" not in p._status.as_text()


def test_old_retry_is_discarded_after_real_screen_change():
    p = make_pipeline()
    p._previous_frame = capture(1)
    old_retry = capture(2)
    old_retry.__dict__["_retry_generation"] = 0
    assert p._should_process(capture(3))
    assert not p._is_current_frame(old_retry)


def test_same_source_pixels_keep_old_translation_during_one_ocr_miss():
    p = make_pipeline()
    pixels = np.full((600, 800, 3), 170, dtype=np.uint8)
    initial = RenderFrame(1, ORIGIN, [RenderItem(PARAGRAPH, "稳定译文", "Visible source.")], source_items=[])
    initial.__dict__["_capture_image"] = pixels
    accept(p, initial)
    missed = RenderFrame(2, ORIGIN, [], source_items=[])
    missed.__dict__["_capture_image"] = pixels.copy()
    accept(p, missed)
    assert [item.text for item in p.snapshot_items()] == ["稳定译文"]
    blank = RenderFrame(3, ORIGIN, [], source_items=[])
    blank.__dict__["_capture_image"] = np.zeros_like(pixels)
    accept(p, blank)
    assert p.snapshot_items() == []
    assert p._source_fingerprints == {}


def test_pixel_confirmation_cannot_preserve_old_language_after_invalidation():
    p = make_pipeline()
    frame = RenderFrame(1, ORIGIN, [RenderItem(PARAGRAPH, "旧语言", "Visible source.")])
    frame.__dict__["_capture_image"] = np.ones((600, 800, 3), dtype=np.uint8)
    accept(p, frame)
    assert p._source_fingerprints
    p.invalidate(clear_output=True)
    assert p._source_fingerprints == {}
    assert p.snapshot_items() == []


def test_corrected_negation_source_outranks_identical_pixel_fallback():
    p = make_pipeline()
    pixels = np.ones((600, 800, 3), dtype=np.uint8)
    initial = RenderFrame(1, ORIGIN, [RenderItem(PARAGRAPH, "请保存文件。", "Save the file.")])
    initial.__dict__["_capture_image"] = pixels
    accept(p, initial)
    corrected = RenderFrame(2, ORIGIN, [], source_items=[
        RenderItem(PARAGRAPH, "Do not save the file.", "Do not save the file."),
    ], retry_regions=[PARAGRAPH])
    corrected.__dict__["_capture_image"] = pixels
    accept(p, corrected)
    assert p.snapshot_items() == []


def test_unrelated_refresh_keeps_and_later_completes_pending_translation(monkeypatch):
    p = make_pipeline()
    now = [10.0]
    monkeypatch.setattr("pingyitong.core.pipeline.time.perf_counter", lambda: now[0])
    p._previous_frame = capture(1)
    source = RenderItem(PARAGRAPH, "Save the file.", "Save the file.")
    accept(p, RenderFrame(2, ORIGIN, [], source_items=[source], retry_regions=[PARAGRAPH]))
    p._differ.compute.return_value = SimpleNamespace(changed=True, regions=[(700, 500, 20, 20)])
    unrelated = capture(3)
    assert p._should_process(unrelated)
    accept(p, RenderFrame(3, ORIGIN, [], is_full_frame=False,
                         changed_regions=[Rect(600, 700, 20, 20)], source_items=[]))
    assert p._pending_retry_regions == [PARAGRAPH.to_local(ORIGIN)]
    now[0] += 10
    newest = capture(4)
    assert p._should_process(newest)
    assert newest.__dict__["_translation_retry"] is True
    assert any(area.intersection(PARAGRAPH.to_local(ORIGIN)).area == PARAGRAPH.area
               for area in newest.changed_regions)


def test_unrelated_screen_change_does_not_invalidate_inflight_region_retry():
    p = make_pipeline()
    p._previous_frame = capture(1)
    p._unresolved_regions = [PARAGRAPH.to_local(ORIGIN)]
    p._differ.compute.return_value = SimpleNamespace(changed=True, regions=[(700, 500, 20, 20)])
    old_retry = capture(2)
    old_retry.__dict__["_retry_generation"] = 0
    assert p._should_process(capture(3))
    assert p._is_current_frame(old_retry)


def test_new_failed_region_gets_own_budget_after_another_region_exhausts(monkeypatch):
    p = make_pipeline()
    now = [10.0]
    monkeypatch.setattr("pingyitong.core.pipeline.time.perf_counter", lambda: now[0])
    p._previous_frame = capture(1)
    source = RenderItem(PARAGRAPH, "Save the file.", "Save the file.")
    p._retry_counts[PARAGRAPH.to_local(ORIGIN)] = 3
    accept(p, RenderFrame(2, ORIGIN, [], source_items=[source], retry_regions=[PARAGRAPH]))
    assert p._pending_retry_regions == []
    new_region = Rect(600, 700, 20, 20)
    new_source = RenderItem(new_region, "New words.", "New words.")
    accept(p, RenderFrame(3, ORIGIN, [], is_full_frame=False, changed_regions=[new_region],
                         source_items=[new_source], retry_regions=[new_region]))
    assert p._pending_retry_regions == [new_region.to_local(ORIGIN)]
    assert "待补译 1 处" in p._status.as_text()
    assert "暂未译出 1 处" in p._status.as_text()
    p._differ.compute.return_value = SimpleNamespace(changed=False, regions=[])
    now[0] += 10
    assert p._should_process(capture(4))
    assert p._retry_counts[PARAGRAPH.to_local(ORIGIN)] == 3
    assert p._retry_counts[new_region.to_local(ORIGIN)] == 1
