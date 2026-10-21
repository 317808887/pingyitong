"""Dropped incremental frames must preserve pending changes and text deletions."""

from __future__ import annotations

import os
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt6.QtWidgets import QApplication

from pingyitong.components.capture import CaptureFrame
from pingyitong.components.capture.geometry import Rect
from pingyitong.components.ocr import RecognizedFrame, TextBlock
from pingyitong.components.render import RenderFrame, RenderItem
from pingyitong.components.render.render_item import merge_items
from pingyitong.core.pipeline import (
    Pipeline,
    _coalesce_capture,
    _coalesce_recognized,
    _coalesce_render,
)
from pingyitong.core.task_queue import LatestQueue

ORIGIN = Rect(-100, 200, 200, 100)
LEFT = Rect(-90, 210, 50, 20)
RIGHT = Rect(20, 210, 50, 20)


def _capture(seq: int, *, full: bool = False, regions=None) -> CaptureFrame:
    return CaptureFrame(
        seq=seq, timestamp=float(seq), origin=ORIGIN,
        image=np.full((100, 200, 3), seq, dtype=np.uint8),
        changed_regions=list(regions or []), is_full_frame=full, monitor_index=2,
    )


def _recognized(seq: int, blocks, *, full: bool = False, regions=None):
    return RecognizedFrame(
        seq=seq, origin=ORIGIN, blocks=blocks,
        is_full_frame=full, changed_regions=list(regions or []), monitor_index=2,
    )


def _render(seq: int, items, *, full: bool = False, regions=None):
    return RenderFrame(
        seq=seq, origin=ORIGIN, items=items,
        is_full_frame=full, changed_regions=list(regions or []),
    )


class TestIncrementalCoalescing(unittest.TestCase):
    def test_capture_preserves_disjoint_changes_using_latest_pixels_and_time(self):
        queue = LatestQueue(keep_latest_only=True, coalesce=_coalesce_capture)
        first = _capture(1, regions=[Rect(10, 10, 50, 20)])
        latest = _capture(2, regions=[Rect(120, 10, 50, 20)])
        queue.put(first, seq=1)
        queue.put(latest, seq=2)

        result = queue.get_latest()
        self.assertIs(result, latest)
        self.assertEqual(result.timestamp, 2)
        self.assertTrue(np.all(result.image == 2))
        self.assertEqual(result.changed_regions, [*first.changed_regions, Rect(120, 10, 50, 20)])
        self.assertEqual(queue.stats().dropped, 1)

    def test_pending_first_full_capture_cannot_be_lost(self):
        latest = _capture(2, regions=[Rect(10, 10, 50, 20)])
        result = _coalesce_capture(_capture(1, full=True), latest)
        self.assertTrue(result.is_full_frame)
        self.assertEqual(result.changed_regions, [])
        self.assertTrue(np.all(result.image == 2))

    def test_capture_rejects_changed_epoch_monitor_origin_or_dimensions(self):
        for changed in ("epoch", "monitor", "origin", "dimensions"):
            with self.subTest(changed=changed):
                previous, latest = _capture(1, full=True), _capture(2)
                if changed == "epoch":
                    latest.__dict__["_translation_epoch"] = 1
                elif changed == "monitor":
                    latest.monitor_index = 1
                elif changed == "origin":
                    latest.origin = Rect(0, 0, 200, 100)
                else:
                    latest.image = np.zeros((80, 200, 3), dtype=np.uint8)
                result = _coalesce_capture(previous, latest)
                self.assertFalse(result.is_full_frame)

    def test_ocr_pending_full_snapshot_preserves_unchanged_text_and_deletion(self):
        previous = _recognized(1, [TextBlock("left", LEFT), TextBlock("right", RIGHT)], full=True)
        latest = _recognized(2, [], regions=[LEFT])
        result = _coalesce_recognized(previous, latest)
        self.assertTrue(result.is_full_frame)
        self.assertEqual(result.texts, ["right"])
        self.assertEqual(result.seq, 2)

    def test_ocr_pending_increments_keep_both_disjoint_changes(self):
        previous = _recognized(1, [TextBlock("left", LEFT)], regions=[LEFT])
        latest = _recognized(2, [TextBlock("right", RIGHT)], regions=[RIGHT])
        result = _coalesce_recognized(previous, latest)
        self.assertFalse(result.is_full_frame)
        self.assertEqual(result.texts, ["left", "right"])
        self.assertEqual(result.changed_regions, [LEFT, RIGHT])

    def test_render_pending_deletion_is_preserved(self):
        previous = _render(1, [RenderItem(LEFT, "旧译文"), RenderItem(RIGHT, "未改变")], full=True)
        latest = _render(2, [], regions=[LEFT])
        result = _coalesce_render(previous, latest)
        self.assertTrue(result.is_full_frame)
        self.assertEqual([item.text for item in result.items], ["未改变"])

    def test_render_pending_increments_keep_delete_region_even_without_text(self):
        previous = _render(1, [RenderItem(RIGHT, "新译文")], regions=[RIGHT])
        latest = _render(2, [], regions=[LEFT])
        result = _coalesce_render(previous, latest)
        self.assertFalse(result.is_full_frame)
        self.assertEqual(result.changed_regions, [RIGHT, LEFT])
        self.assertEqual([item.text for item in result.items], ["新译文"])

    def test_merge_blank_region_removes_old_translation(self):
        old = [RenderItem(LEFT, "已删除"), RenderItem(RIGHT, "保留")]
        result = merge_items(old, [], dirty_regions=[LEFT])
        self.assertEqual([item.text for item in result], ["保留"])

    def test_merge_shorter_text_clears_other_old_words_in_dirty_region(self):
        old = [RenderItem(LEFT, "旧左"), RenderItem(RIGHT, "旧右")]
        dirty = Rect(-95, 205, 170, 30)
        result = merge_items(old, [RenderItem(LEFT, "新短句")], dirty_regions=[dirty])
        self.assertEqual([item.text for item in result], ["新短句"])


class TestPipelineDeletion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _pipeline(self, ocr=None, translator=None):
        overlay = Mock(is_visible=True)
        pipeline = Pipeline(
            config=SimpleNamespace(capture=SimpleNamespace(monitors=[])),
            capturer=SimpleNamespace(last_capture_cost=0), differ=object(),
            ocr_engine=ocr or object(), translator=translator or object(), overlay=overlay,
        )
        return pipeline, overlay

    def test_blank_ocr_result_reaches_translation_and_keeps_global_dirty_region(self):
        entered = threading.Event()
        ocr = Mock()
        ocr.recognize_frame.side_effect = lambda frame: (
            entered.set() or _recognized(frame.seq, [], regions=[])
        )
        pipeline, _ = self._pipeline(ocr=ocr)
        pipeline._frame_queue.put(_capture(1, regions=[Rect(10, 10, 50, 20)]), seq=1)
        stop = threading.Event()
        worker = threading.Thread(target=pipeline._ocr_loop, args=(stop,))
        worker.start()
        try:
            self.assertTrue(entered.wait(timeout=2))
            result = pipeline._ocr_queue.get(timeout=2)
            self.assertIsNotNone(result)
            self.assertEqual(result.blocks, [])
            self.assertEqual(result.changed_regions, [LEFT])
        finally:
            stop.set()
            pipeline._frame_queue.wake_all()
            worker.join(timeout=2)
        self.assertFalse(worker.is_alive())

    def test_blank_increment_clears_transcript_and_dispatches_dirty_region(self):
        pipeline, overlay = self._pipeline()
        pipeline.set_transcript_enabled(True)
        pipeline._update_transcript(_render(
            1, [RenderItem(LEFT, "已删除"), RenderItem(RIGHT, "保留")], full=True,
        ))
        pipeline._result_queue.put(_render(2, [], regions=[LEFT]), seq=2)
        pipeline._render_tick()
        self.assertEqual([item.text for item in pipeline._transcript_items], ["保留"])
        overlay.apply.assert_called_once_with([], None, replace=False, dirty_regions=[LEFT])


if __name__ == "__main__":
    unittest.main()
