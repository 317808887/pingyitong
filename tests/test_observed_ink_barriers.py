"""Observed-ink barriers: pixel evidence restricts growth, never the anchor.

These are offline component regressions. They prove that unread contrasting
pixels stop translation expansion, and that a conservative barrier does not
disable the translation of a source box OCR actually identified.
"""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt6.QtCore import QRectF
from PyQt6.QtWidgets import QApplication

from pingyitong.components.capture.geometry import Rect, ScreenInfo
from pingyitong.components.render.overlay_window import OverlayManager, OverlayWindow
from pingyitong.components.render.render_item import RenderFrame, RenderItem
from pingyitong.components.render.slot_layout import allocate_slots
from pingyitong.components.render.space_occupancy import ObservedSpaceDetector
from pingyitong.config.schema import StyleConfig

SCREEN = ScreenInfo("INK", Rect(0, 0, 640, 400), Rect(0, 0, 640, 400), 1.0, True)
FIXED = StyleConfig(font_size_mode="fixed", font_size_fixed=14, shadow=False)
LONG_TEXT = "请勿在传输完成之前断开线缆，否则数据会丢失。"
DEFAULT_BOUNDS = Rect(10, 10, 600, 120)

# Font metrics need a QGuiApplication; the allocator is exercised without a
# widget tree, so one process-wide instance is enough.
APP = QApplication.instance() or QApplication([])


def allocation(
    item: RenderItem, barriers: list[Rect] | None, bounds: Rect | None = None,
):
    bounds = bounds or DEFAULT_BOUNDS
    seeds = {id(item): QRectF(item.box.x, item.box.y, item.box.width, item.box.height)}
    result = allocate_slots(
        [item], [], seeds,
        QRectF(bounds.x, bounds.y, bounds.width, bounds.height), [], FIXED,
        space_obstacles=None if barriers is None else [
            QRectF(rect.x, rect.y, rect.width, rect.height) for rect in barriers
        ],
    )
    return result[id(item)]


class TestObservedSpaceDetector(unittest.TestCase):
    def scene(self) -> np.ndarray:
        image = np.full((120, 400, 3), 245, dtype=np.uint8)
        # Contrasting mark that OCR never returned as a source.
        image[40:58, 250:310] = 35
        return image

    def test_unread_ink_becomes_a_barrier_outside_source_boxes(self) -> None:
        origin = Rect(0, 0, 400, 120)
        source = Rect(20, 30, 60, 18)
        observation = ObservedSpaceDetector().analyze(self.scene(), origin, [source])
        self.assertEqual(observation.evidence["mode"], "dominant_uniform_background")
        self.assertTrue(observation.obstacles)
        self.assertTrue(any(rect.intersects(Rect(250, 40, 60, 18)) for rect in observation.obstacles))
        self.assertFalse(any(rect.intersects(source) for rect in observation.obstacles))

    def test_source_pixels_are_not_reported_as_unread_ink(self) -> None:
        origin = Rect(0, 0, 400, 120)
        source = Rect(250, 40, 60, 18)
        observation = ObservedSpaceDetector().analyze(self.scene(), origin, [source])
        self.assertFalse(any(rect.intersects(source) for rect in observation.obstacles))
        self.assertEqual(observation.evidence["source_exclusion_padding_px"], 1)

    def test_missing_or_mismatched_pixels_fall_back_to_conservative_barriers(self) -> None:
        origin = Rect(0, 0, 400, 120)
        source = Rect(20, 30, 60, 18)
        for image in (None, np.zeros((10, 10, 3), dtype=np.uint8)):
            with self.subTest(shape=getattr(image, "shape", None)):
                observation = ObservedSpaceDetector().analyze(image, origin, [source])
                self.assertEqual(
                    observation.evidence["mode"], "missing_or_mismatched_capture_pixels",
                )
                self.assertTrue(observation.obstacles)

    def test_repeat_analysis_is_cached_without_changing_barriers(self) -> None:
        origin = Rect(0, 0, 400, 120)
        source = Rect(20, 30, 60, 18)
        detector = ObservedSpaceDetector()
        first = detector.analyze(self.scene(), origin, [source])
        second = detector.analyze(self.scene(), origin, [source])
        self.assertFalse(first.evidence["cache_hit"])
        self.assertTrue(second.evidence["cache_hit"])
        self.assertEqual(first.obstacles, second.obstacles)


class TestBarriersRestrictGrowthOnly(unittest.TestCase):
    def test_growth_into_whitespace_happens_without_pixel_evidence(self) -> None:
        item = RenderItem(Rect(20, 30, 60, 20), LONG_TEXT)
        slot = allocation(item, None)
        self.assertGreater(slot.rect.width(), item.box.width + 20)
        self.assertEqual(slot.reason, "expanded_into_observed_space")

    def test_whitespace_is_not_used_when_pixels_there_are_unread_ink(self) -> None:
        item = RenderItem(Rect(20, 30, 60, 20), LONG_TEXT)
        barrier = Rect(100, 20, 220, 40)
        slot = allocation(item, [barrier])
        self.assertFalse(slot.rect.intersects(QRectF(100, 20, 220, 40)))
        self.assertLess(slot.rect.width(), allocation(item, None).rect.width())

    def test_barrier_over_anchor_never_disables_identified_source(self) -> None:
        item = RenderItem(Rect(20, 30, 60, 20), LONG_TEXT)
        # A tile boundary that brushes the anchor must not reject the anchor.
        slot = allocation(item, [Rect(18, 28, 4, 4)])
        self.assertNotEqual(slot.reason, "source_or_window_obstructs_anchor")
        self.assertFalse(slot.rect.isEmpty())

    def test_structural_obstacles_still_reject_the_anchor(self) -> None:
        item = RenderItem(Rect(20, 30, 60, 20), LONG_TEXT)
        blocker = RenderItem(Rect(30, 30, 20, 20), "8")
        seeds = {
            id(item): QRectF(20, 30, 60, 20),
            id(blocker): QRectF(30, 30, 20, 20),
        }
        result = allocate_slots(
            [item, blocker], [], seeds, QRectF(10, 10, 600, 120), [], FIXED,
            space_obstacles=[QRectF(0, 0, 1, 1)],
        )
        self.assertEqual(
            result[id(item)].reason, "source_or_window_obstructs_anchor",
        )


class TestOverlayReportsItsSpaceBasis(unittest.TestCase):
    def setUp(self) -> None:
        self.window = OverlayWindow(SCREEN, FIXED)

    def tearDown(self) -> None:
        self.window.close()
        self.window.deleteLater()
        APP.processEvents()

    def test_basis_changes_only_when_pixel_evidence_is_supplied(self) -> None:
        item = RenderItem(Rect(20, 30, 60, 20), LONG_TEXT)
        self.window.set_layout_context(Rect(10, 10, 400, 120), [item])
        self.window.set_items([item])
        legacy = self.window.paint_evidence()[0]
        self.assertEqual(
            legacy["space_basis"],
            "capture_roi_minus_detected_sources_and_excluded_windows",
        )
        self.assertEqual(legacy["observed_ink_barriers"], 0)

        self.window.set_layout_context(
            Rect(10, 10, 400, 120), [item], space_obstacles=[Rect(100, 20, 220, 40)],
        )
        self.window.set_items([item])
        observed = self.window.paint_evidence()[0]
        self.assertEqual(
            observed["space_basis"],
            "capture_roi_minus_detected_sources_excluded_windows_and_observed_ink",
        )
        self.assertGreaterEqual(observed["observed_ink_barriers"], 1)
        self.assertFalse(
            self.window._paint_rect_for(self.window._items[0]).intersects(
                QRectF(100, 20, 220, 40)
            )
        )

    def test_manager_forwards_frame_barriers_to_every_window(self) -> None:
        item = RenderItem(Rect(20, 30, 60, 20), LONG_TEXT)
        frame = RenderFrame(1, Rect(0, 0, 640, 400), [item])
        frame._space_obstacles = [Rect(100, 20, 220, 40)]
        manager = SimpleNamespace(
            _windows={SCREEN.name: self.window},
            apply=lambda items, *args: self.window.set_items(items) or len(items),
        )
        OverlayManager.apply_frame(manager, frame)
        self.assertEqual(self.window._space_obstacles[0], QRectF(101, 21, 218, 38))

    def test_manager_without_pixel_evidence_keeps_legacy_basis(self) -> None:
        item = RenderItem(Rect(20, 30, 60, 20), LONG_TEXT)
        frame = RenderFrame(1, Rect(0, 0, 640, 400), [item])
        manager = SimpleNamespace(
            _windows={SCREEN.name: self.window},
            apply=lambda items, *args: self.window.set_items(items) or len(items),
        )
        OverlayManager.apply_frame(manager, frame)
        self.assertFalse(self.window._space_observed)
        self.assertEqual(self.window._space_obstacles, [])


class TestPipelineAttachesSpaceEvidence(unittest.TestCase):
    """Unbound call: the pipeline stage is exercised without threads or models."""

    def attach(self, image, sources, bounds=Rect(0, 0, 400, 120)):
        from pingyitong.core.pipeline import Pipeline

        recognized = SimpleNamespace(
            origin=bounds, blocks=[SimpleNamespace(box=box) for box in sources],
        )
        frame = SimpleNamespace(
            timings={}, source_items=[], diagnostics={},
        )
        frame.__dict__["_capture_image"] = image
        pipeline = SimpleNamespace(_space_detector=ObservedSpaceDetector())
        cost = Pipeline._attach_space_evidence(pipeline, recognized, frame)
        return frame, cost

    def test_evidence_is_attached_with_cost_accounting(self) -> None:
        image = np.full((120, 400, 3), 245, dtype=np.uint8)
        image[40:58, 250:310] = 35
        frame, cost = self.attach(image, [Rect(20, 30, 60, 18)])
        self.assertGreaterEqual(cost, 0.0)
        self.assertEqual(frame.timings["space_observe"], cost)
        self.assertTrue(frame.__dict__["_space_obstacles"])
        self.assertEqual(frame.__dict__["_space_evidence"]["constraint"], "expansion_only")

    def test_missing_image_records_no_evidence_instead_of_guessing(self) -> None:
        frame, cost = self.attach(None, [Rect(20, 30, 60, 18)])
        self.assertEqual(cost, 0.0)
        self.assertIsNone(frame.__dict__["_space_obstacles"])
        self.assertNotIn("space_observe", frame.timings)


if __name__ == "__main__":
    unittest.main()
