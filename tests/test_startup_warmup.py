"""Startup cleanup must precede inference; OCR warming is independent and safe."""

from __future__ import annotations

import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from pingyitong.components.capture import CaptureFrame
from pingyitong.components.capture.geometry import Rect
from pingyitong.config.schema import AppConfig
from pingyitong.ui.app import Application, _SerializedOcrEngine


class _IsolatedAppLogTest(unittest.TestCase):
    def setUp(self):
        logger_patch = patch("pingyitong.ui.app.logger")
        logger_patch.start()
        self.addCleanup(logger_patch.stop)


class TestStartupServerCleanup(_IsolatedAppLogTest):
    def _controller(self):
        return SimpleNamespace(
            _stale_servers_reaped=False, _pending_reaped_server_count=0,
            model_pool=SimpleNamespace(backend_probe=Mock(return_value=SimpleNamespace(executable="synthetic-runtime.exe"))),
            notify=Mock(),
        )

    def test_cleanup_runs_once_before_prefetch_can_start_and_defers_notice(self):
        controller = self._controller()
        with patch("pingyitong.ui.app.reap_stale_servers", return_value=[1, 2]) as reap:
            Application._reap_stale_servers(controller)
            Application._reap_stale_servers(controller)
        reap.assert_called_once_with("synthetic-runtime.exe")
        controller.notify.assert_not_called()
        self.assertEqual(controller._pending_reaped_server_count, 2)

    def test_probe_failure_never_allows_late_sweep_of_live_servers(self):
        controller = self._controller()
        controller.model_pool.backend_probe.side_effect = OSError("missing backend")
        with patch("pingyitong.ui.app.reap_stale_servers") as reap:
            Application._reap_stale_servers(controller)
            controller.model_pool.backend_probe.side_effect = None
            Application._reap_stale_servers(controller)
        self.assertEqual(controller.model_pool.backend_probe.call_count, 1)
        reap.assert_not_called()

    def test_layer_construction_cleans_before_any_worker_can_be_constructed(self):
        events = []
        config = AppConfig()
        config.schedule.preload_on_start = False
        pool = SimpleNamespace(apply_api_config=lambda _cfg: events.append("api"))
        controller = SimpleNamespace(_apply_reduced_motion_setting=Mock(), _reap_stale_servers=lambda: events.append("reap"))

        class StopConstructionError(Exception):
            pass

        def create_pool(*_args, **_kwargs):
            events.append("pool")
            return pool

        # Stop before caches/UI/prefetch workers; none can start before cleanup.
        with (
            patch("pingyitong.ui.app.SettingsManager", return_value=SimpleNamespace(load=lambda: config)),
            patch("pingyitong.ui.app.build_theme"),
            patch("pingyitong.ui.app.ScreenCapturer"),
            patch("pingyitong.ui.app.FrameDiffer"),
            patch("pingyitong.ui.app._SerializedOcrEngine"),
            patch("pingyitong.ui.app.ModelPool", side_effect=create_pool),
            patch("pingyitong.ui.app.TranslationCache", side_effect=StopConstructionError),
            self.assertRaises(StopConstructionError),
        ):
            Application._build_layers(controller)
        self.assertEqual(events, ["pool", "reap", "api"])

    def test_duplicate_begin_preload_does_not_sweep_or_start_more_work(self):
        controller = SimpleNamespace(
            model_pool=SimpleNamespace(is_preloading=True, preload_async=Mock()),
            _reap_stale_servers=Mock(), _warm_ocr_async=Mock(),
        )
        Application._begin_preload(controller)
        controller._reap_stale_servers.assert_not_called()
        controller._warm_ocr_async.assert_not_called()
        controller.model_pool.preload_async.assert_not_called()

    def test_later_preload_never_sweeps_existing_loaded_servers(self):
        controller = SimpleNamespace(
            model_pool=SimpleNamespace(is_preloading=False, preload_targets=Mock(return_value=[]), preload_async=Mock(return_value=False)),
            _reap_stale_servers=Mock(), _warm_ocr_async=Mock(),
            preload_progress=Mock(), preload_finished=Mock(), notify=Mock(),
        )
        Application._begin_preload(controller)
        controller._reap_stale_servers.assert_not_called()


class TestIndependentOcrWarmup(_IsolatedAppLogTest):
    def _controller(self, warm):
        return SimpleNamespace(
            ocr_engine=SimpleNamespace(_backend=object(), warm_up=warm),
            _ocr_warm_lock=threading.Lock(), _ocr_warm_running=False,
            _ocr_warmed_backend=None,
        )

    def _wait_finished(self, controller):
        deadline = time.perf_counter() + 2
        while controller._ocr_warm_running and time.perf_counter() < deadline:
            time.sleep(0.001)
        self.assertFalse(controller._ocr_warm_running)

    def test_disabling_local_model_preload_still_starts_ocr_warmup(self):
        config = AppConfig()
        config.schedule.preload_on_start = False
        controller = SimpleNamespace(settings=SimpleNamespace(config=config), _warm_ocr_async=Mock(), model_pool=Mock())
        Application._start_preload(controller)
        controller._warm_ocr_async.assert_called_once()
        controller.model_pool.preload_targets.assert_not_called()

    def test_api_only_install_without_local_weights_still_warms_ocr(self):
        config = AppConfig()
        config.api.enabled = True
        controller = SimpleNamespace(settings=SimpleNamespace(config=config), _warm_ocr_async=Mock(), model_pool=SimpleNamespace(preload_targets=Mock(return_value=[])))
        with patch("pingyitong.ui.app.QTimer.singleShot") as later:
            Application._start_preload(controller)
        controller._warm_ocr_async.assert_called_once()
        later.assert_not_called()

    def test_warmup_returns_without_waiting_and_does_not_reenter(self):
        entered, release = threading.Event(), threading.Event()

        def warm():
            entered.set()
            release.wait(timeout=2)
            return True

        callback = Mock(side_effect=warm)
        controller = self._controller(callback)
        started = time.perf_counter()
        Application._warm_ocr_async(controller)
        self.assertLess(time.perf_counter() - started, 0.5)
        try:
            self.assertTrue(entered.wait(timeout=1))
            Application._warm_ocr_async(controller)
            self.assertEqual(callback.call_count, 1)
        finally:
            release.set()
            self._wait_finished(controller)
        Application._warm_ocr_async(controller)
        self.assertEqual(callback.call_count, 1)

    def test_failed_warmup_can_retry_and_rebuilt_backend_warms_again(self):
        callback = Mock(side_effect=[False, True, True])
        controller = self._controller(callback)
        for _ in range(2):
            Application._warm_ocr_async(controller)
            self._wait_finished(controller)
        self.assertEqual(callback.call_count, 2)
        controller.ocr_engine._backend = object()
        Application._warm_ocr_async(controller)
        self._wait_finished(controller)
        self.assertEqual(callback.call_count, 3)

    def test_exception_does_not_leave_warmup_permanently_running(self):
        callback = Mock(side_effect=[RuntimeError("transient"), True])
        controller = self._controller(callback)
        for _ in range(2):
            Application._warm_ocr_async(controller)
            self._wait_finished(controller)
        self.assertEqual(callback.call_count, 2)

    def test_live_frame_cannot_enter_mutable_backend_during_warmup(self):
        entered, release, live_started = threading.Event(), threading.Event(), threading.Event()

        class Backend:
            name = "synthetic"
            calls = 0

            def is_ready(self):
                return True

            def recognize(self, _image):
                self.calls += 1
                if self.calls == 1:
                    entered.set()
                    release.wait(timeout=2)
                return []

        backend = Backend()
        engine = _SerializedOcrEngine(backend=backend)
        frame = CaptureFrame(seq=1, timestamp=0, image=np.zeros((50, 100, 3), dtype=np.uint8), origin=Rect(0, 0, 100, 50))
        warmer = threading.Thread(target=engine.warm_up)

        def recognize():
            live_started.set()
            engine.recognize_frame(frame)

        live = threading.Thread(target=recognize)
        warmer.start()
        try:
            self.assertTrue(entered.wait(timeout=1))
            live.start()
            self.assertTrue(live_started.wait(timeout=1))
            self.assertEqual(backend.calls, 1)
        finally:
            release.set()
            warmer.join(timeout=2)
            if live.ident is not None:
                live.join(timeout=2)
        self.assertFalse(warmer.is_alive())
        self.assertFalse(live.is_alive())
        self.assertEqual(backend.calls, 5)


if __name__ == "__main__":
    unittest.main()
