"""The selected language pair and every visible translation must agree."""

from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from pingyitong.components.capture.geometry import Rect
from pingyitong.components.ocr import RecognizedFrame
from pingyitong.components.render import RenderFrame
from pingyitong.config.settings import SettingsManager
from pingyitong.core.pipeline import Pipeline
from pingyitong.ui.app import Application
from pingyitong.ui.design.theme import build_theme
from pingyitong.ui.main_panel import MainPanel
from tests.test_translation_service import make_service


class _Overlay:
    is_visible = True

    def __init__(self) -> None:
        self.applied: list[list] = []
        self.cleared = 0

    def apply(self, items, _monitors=None, *, replace=True) -> int:
        self.applied.append(list(items))
        return len(items)

    def clear(self) -> None:
        self.cleared += 1


class _BlockingTranslator:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def translate_frame(self, recognized, **_kwargs):
        self.entered.set()
        self.release.wait(timeout=5)
        return RenderFrame(seq=recognized.seq, origin=recognized.origin, items=[])


def _pipeline(translator=None):
    overlay = _Overlay()
    pipeline = Pipeline(
        config=SimpleNamespace(capture=SimpleNamespace(monitors=[])),
        capturer=SimpleNamespace(last_capture_cost=0.0),
        differ=object(),
        ocr_engine=object(),
        translator=translator or object(),
        overlay=overlay,
    )
    return pipeline, overlay


class TestLanguageDirectionSync(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_same_language_selection_is_kept_and_roundtrips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = SettingsManager(Path(directory) / "config.json")
            controller = SimpleNamespace(theme=build_theme(), settings=settings)
            panel = MainPanel(controller)
            try:
                panel._source_language.setCurrentIndex(
                    panel._source_language.findData("en")
                )
                panel._target_language.setCurrentIndex(
                    panel._target_language.findData("en")
                )

                self.assertEqual(settings.config.translation.source_language, "en")
                self.assertEqual(panel._source_language.currentData(), "en")
                self.assertEqual(panel._target_language.currentData(), "en")
                restored = SettingsManager(settings.path)
                restored.load()
                self.assertEqual(restored.config.translation.source_language, "en")
                self.assertEqual(restored.config.translation.target_language, "en")
            finally:
                panel.close()

    def test_source_then_target_matches_target_then_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = SettingsManager(Path(directory) / "config.json")
            controller = SimpleNamespace(theme=build_theme(), settings=settings)
            panel = MainPanel(controller)
            try:
                for order in (("source", "target"), ("target", "source")):
                    settings.update({"translation": {
                        "source_language": "auto", "target_language": "zh",
                    }})
                    panel.load_from_settings()
                    for field in order:
                        combo = getattr(panel, f"_{field}_language")
                        combo.setCurrentIndex(combo.findData("zh" if field == "source" else "en"))
                    self.assertEqual(settings.config.translation.source_language, "zh")
                    self.assertEqual(settings.config.translation.target_language, "en")
                    self.assertEqual(panel._source_language.currentData(), "zh")
                    self.assertEqual(panel._target_language.currentData(), "en")
            finally:
                panel.close()

    def test_source_first_change_invalidates_old_request_and_visible_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = SettingsManager(Path(directory) / "config.json")
            service, _ = make_service(settings.config)
            pipeline, overlay = _pipeline(service)
            pipeline._ocr = mock.Mock()
            pipeline.set_transcript_enabled(True)
            pipeline._transcript_items = ["old translation"]
            service._last_output = [("original", "old translation")]
            controller = SimpleNamespace(
                theme=build_theme(), settings=settings,
                _last_translation_direction=("auto", "zh"),
                _last_api_settings=None, overlay=mock.Mock(),
                ocr_engine=mock.Mock(), pipeline=pipeline, preprocessor=mock.Mock(),
                differ=mock.Mock(), scheduler=mock.Mock(), model_pool=mock.Mock(),
                cache=mock.Mock(), translator=service, history=mock.Mock(),
                _refresh_api_runtime_status=mock.Mock(),
                prefetch_gate=SimpleNamespace(
                    set_enabled=mock.Mock(),
                    snapshot=lambda: SimpleNamespace(badge=""),
                ),
                network_monitor=mock.Mock(), reload_glossary=mock.Mock(),
                transcript_panel=SimpleNamespace(is_expanded=False),
            )
            panel = MainPanel(controller)
            controller.panel = panel
            settings.changed.connect(lambda config: Application._on_config_changed(controller, config))
            try:
                with service._request_scope():
                    old_request = service._request_context.get()
                    old_key = service._cache_key("Hello", "auto", "zh")
                    old_request.cache_entries[old_key] = "old translation"
                    panel._source_language.setCurrentIndex(panel._source_language.findData("zh"))
                    panel._target_language.setCurrentIndex(panel._target_language.findData("en"))
                    self.assertFalse(service._request_is_current(old_request))
                    self.assertEqual(old_request.config.translation.source_language, "auto")
                    self.assertEqual(old_request.config.translation.target_language, "zh")
                self.assertIsNone(service.cache.get(old_key))
                with service._request_scope():
                    current_request = service._request_context.get()
                    self.assertEqual(current_request.config.translation.source_language, "zh")
                    self.assertEqual(current_request.config.translation.target_language, "en")
                self.assertEqual(service.last_output_text(), "")
                self.assertEqual(pipeline._transcript_items, [])
                self.assertGreaterEqual(pipeline._translation_epoch, 2)
                self.assertGreaterEqual(overlay.cleared, 2)
            finally:
                panel.close()

    def test_direction_change_discards_queued_results_and_transcript(self) -> None:
        pipeline, overlay = _pipeline()
        pipeline.set_transcript_enabled(True)
        pipeline._transcript_items = ["previous language"]
        previous = RenderFrame(seq=1, origin=Rect(0, 0, 10, 10), items=[])
        pipeline._result_queue.put(previous, seq=1)

        pipeline.invalidate(clear_output=True)
        pipeline._render_tick()

        self.assertEqual(overlay.applied, [])
        self.assertEqual(overlay.cleared, 1)
        self.assertEqual(pipeline._transcript_items, [])

    def test_inflight_old_language_translation_is_discarded(self) -> None:
        translator = _BlockingTranslator()
        pipeline, _overlay = _pipeline(translator)
        recognized = RecognizedFrame(
            seq=1, origin=Rect(0, 0, 10, 10), blocks=[], is_full_frame=True
        )
        recognized.__dict__["_translation_epoch"] = 0
        pipeline._ocr_queue.put(recognized, seq=1)

        stopped = threading.Event()
        worker = threading.Thread(target=pipeline._translate_loop, args=(stopped,))
        worker.start()
        try:
            self.assertTrue(translator.entered.wait(timeout=2))
            pipeline.invalidate(clear_output=True)
        finally:
            translator.release.set()
            stopped.set()
            pipeline._ocr_queue.wake_all()
            worker.join(timeout=2)

        self.assertFalse(worker.is_alive())
        self.assertTrue(pipeline._result_queue.empty)

    def test_new_frame_survives_queue_cleanup_during_direction_change(self) -> None:
        pipeline, _overlay = _pipeline()
        pipeline._config.capture.enable_diff = True
        queued = threading.Event()
        frame = SimpleNamespace(
            seq=2,
            is_full_frame=False,
            changed_regions=[],
            timestamp=time.monotonic(),
            origin=Rect(0, 0, 10, 10),
        )

        def capture_new_frame() -> None:
            frame.__dict__["_translation_epoch"] = pipeline._translation_epoch
            if pipeline._should_process(frame):
                pipeline._frame_queue.put(frame, seq=frame.seq)
                queued.set()

        worker = threading.Thread(target=capture_new_frame)
        original_clear = pipeline._frame_queue.clear

        def cleanup_with_capture_race() -> None:
            worker.start()
            queued.wait(timeout=0.05)
            original_clear()

        with mock.patch.object(
            pipeline._frame_queue, "clear", side_effect=cleanup_with_capture_race
        ):
            pipeline.invalidate(clear_output=True)
        worker.join(timeout=2)

        self.assertFalse(worker.is_alive())
        self.assertTrue(queued.is_set())
        self.assertIs(pipeline._frame_queue.get_latest(), frame)
        self.assertTrue(frame.is_full_frame)


if __name__ == "__main__":
    unittest.main()
