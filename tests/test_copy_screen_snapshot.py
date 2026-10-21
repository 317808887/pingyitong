"""Copy uses accepted full-screen state, including unchanged incremental lines."""
from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from pingyitong.components.capture.geometry import Rect
from pingyitong.components.render import RenderFrame, RenderItem
from pingyitong.core.pipeline import Pipeline
from pingyitong.services.reading_order import reading_text

_APP = QApplication.instance() or QApplication([])
TOP = Rect(20, 20, 300, 30)
BOTTOM = Rect(20, 80, 300, 30)
ORIGIN = Rect(0, 0, 800, 600)


def pipeline():
    return Pipeline(
        config=SimpleNamespace(capture=SimpleNamespace(monitors=[])),
        capturer=SimpleNamespace(last_capture_cost=0), differ=object(),
        ocr_engine=object(), translator=object(), overlay=Mock(is_visible=True),
    )


def render(p, seq, items, *, full=False, dirty=None, epoch=None):
    frame = RenderFrame(seq, ORIGIN, items, is_full_frame=full, changed_regions=dirty or [])
    if epoch is not None:
        frame.__dict__["_translation_epoch"] = epoch
    p._result_queue.put(frame, seq=seq)
    p._render_tick()


def test_copy_retains_unchanged_lines_in_reading_order_without_text_panel():
    p = pipeline()
    render(p, 1, [RenderItem(BOTTOM, "下行"), RenderItem(TOP, "旧上行")], full=True)
    render(p, 2, [RenderItem(TOP, "新上行")], dirty=[TOP])
    assert reading_text(p.snapshot_items()) == "新上行\n下行"
    assert p._transcript_items == []
    snapshot = p.snapshot_items()
    snapshot.clear()
    assert len(p.snapshot_items()) == 2


def test_deleted_region_removed_from_copy_and_panel_reopen_has_full_snapshot():
    p = pipeline()
    render(p, 1, [RenderItem(TOP, "上行"), RenderItem(BOTTOM, "下行")], full=True)
    render(p, 2, [], dirty=[TOP])
    received = []
    p.transcript_updated.connect(received.append)
    p.set_transcript_enabled(True)
    assert reading_text(received[-1]) == "下行"
    assert reading_text(p.snapshot_items()) == "下行"


def test_pending_and_old_language_results_cannot_be_copied_after_invalidation():
    p = pipeline()
    p.set_transcript_enabled(True)
    render(p, 1, [RenderItem(TOP, "旧语言")], full=True)
    p._result_queue.put(RenderFrame(2, ORIGIN, [RenderItem(TOP, "尚未渲染")]), seq=2)
    assert reading_text(p.snapshot_items()) == "旧语言"
    received = []
    p.transcript_updated.connect(received.append)
    p.invalidate(clear_output=True)
    assert received == [[]]
    assert p.snapshot_items() == []
    render(p, 3, [RenderItem(TOP, "旧在途结果")], full=True, epoch=0)
    assert p.snapshot_items() == []
    render(p, 4, [RenderItem(TOP, "新语言")], full=True, epoch=1)
    assert reading_text(p.snapshot_items()) == "新语言"
