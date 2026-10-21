"""Real Qt wiring from imported subtitles to guarded pretranslation output."""

from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PyQt6.QtCore import QObject, QThread
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from pingyitong.config.schema import AppConfig
from pingyitong.services.prefetch.active_track import ActiveTrackRegistry, SubtitleTrack
from pingyitong.services.prefetch.lookup import LookupContext, SentenceCacheLookup
from pingyitong.services.prefetch.normalizer import normalize_for_key
from pingyitong.services.prefetch.scheduler import PrefetchJob, PrefetchScheduler
from pingyitong.services.prefetch.sentence_cache import SentenceCache
from pingyitong.services.prefetch.subtitle_source import parse_subtitle_file
from pingyitong.ui.app import Application
from tests.test_translation_service import make_service


class TestPrefetchProductWiring(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.qt_app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        config = AppConfig()
        config.translation.semantic_review = False  # legacy single-pass contract
        config.translation.local_only = False  # This suite exercises explicitly enabled network prefetch.
        config.prefetch.enabled = True
        config.prefetch.subtitle_dirs = [str(self.root)]
        config.translation.target_language = "zh"
        self.controller = Application.__new__(Application)
        QObject.__init__(self.controller)
        self.controller.settings = SimpleNamespace(config=config)
        self.controller.translator, self.pool = make_service(config)
        self.controller.prefetch_cache = SentenceCache(self.root / "cache.db", "test-policy")
        self.controller.prefetch_lookup = SentenceCacheLookup(self.controller.prefetch_cache)
        self.controller.translator.set_prefetch_lookup(self.controller.prefetch_lookup)
        self.controller.track_registry = ActiveTrackRegistry()
        self.controller._subtitle_lines_lock = threading.Lock()
        self.controller._subtitle_lines = {}
        self.controller._prefetch_shutdown = False
        self.controller._subtitle_watcher = None
        self.controller._subtitle_watch_dirs = ()
        self.controller.notify = mock.Mock()
        self.ui_threads = []
        self.progress_text = []

        def record_progress(text):
            self.ui_threads.append(QThread.currentThread())
            self.progress_text.append(text)

        self.controller.panel = SimpleNamespace(set_prefetch_progress=record_progress)
        self.controller.prefetch_scheduler = PrefetchScheduler(
            idle_check=lambda: True, gate_allow=lambda source: True,
            pull=self.controller._pull_prefetch_job,
        )
        # Drive the real scheduler deterministically without starting its timer loop.
        self.controller.prefetch_scheduler.start = mock.Mock()
        self.controller._wire_prefetch_progress()

    def tearDown(self) -> None:
        self.controller._prefetch_shutdown = True
        self.controller.prefetch_scheduler.stop()
        self.controller._subtitle_watcher = None
        self.controller.prefetch_cache.close()
        self.controller.translator._cache.close()
        self.controller.deleteLater()
        self.qt_app.processEvents()
        self.directory.cleanup()

    def _subtitle(self) -> Path:
        path = self.root / "Episode.srt"
        path.write_text(
            "1\n00:00:01,000 --> 00:00:02,000\nHello world\n\n"
            "2\n00:00:03,000 --> 00:00:04,000\nSave the file\n",
            encoding="utf-8",
        )
        return path

    def _register(self) -> str:
        path = self._subtitle()
        lines = parse_subtitle_file(path, normalize_for_key)
        track = SubtitleTrack(str(path), str(path), path.name, len(lines))
        self.controller.register_subtitle_track(track, lines)
        return track.track_id

    def test_existing_srt_is_parsed_queued_translated_and_reused(self) -> None:
        path = self._subtitle()
        self.controller._start_subtitle_watch(self.controller.settings.config)
        QTest.qWait(600)
        self.assertEqual(self.controller.prefetch_scheduler.pending_count(), 2)
        self.assertEqual(self.controller.prefetch_lookup._active_track_id, str(path))
        self.assertTrue(self.controller.prefetch_scheduler._drain_one())
        self.assertTrue(self.controller.prefetch_scheduler._drain_one())
        first = self.controller.prefetch_cache.get_subtitle_line(str(path), 0)
        self.assertEqual(first["source_raw"], "Hello world")
        self.assertEqual((first["start_ms"], first["end_ms"]), (1000, 2000))
        cached = self.controller.prefetch_lookup.lookup(
            "Hello world", LookupContext(src_lang="en", target_lang="zh")
        )
        self.assertTrue(cached)
        self.assertEqual(len(self.pool.calls), 2)

    def test_progress_from_worker_runs_on_ui_thread(self) -> None:
        track_id = self._register()

        def run():
            try:
                self.controller.prefetch_scheduler._drain_one()
            finally:
                self.controller.prefetch_cache.close()

        worker = threading.Thread(target=run)
        worker.start()
        worker.join(timeout=3)
        self.assertFalse(worker.is_alive())
        self.assertFalse(self.progress_text)
        self.qt_app.processEvents()
        self.assertTrue(self.progress_text)
        self.assertEqual(self.ui_threads, [self.qt_app.thread()])
        self.assertEqual(self.controller.prefetch_scheduler.page_progress(f"track:{track_id}"), (1, 2))

    def test_failed_translation_does_not_count_as_ready_or_enter_cache(self) -> None:
        track_id = self._register()
        self.pool.fail = True
        self.assertFalse(self.controller.prefetch_scheduler._drain_one())
        self.qt_app.processEvents()
        self.assertEqual(self.controller.prefetch_scheduler.page_progress(f"track:{track_id}"), (0, 2))
        self.assertEqual(self.controller.prefetch_scheduler.page_failures(f"track:{track_id}"), 1)
        self.assertIsNone(self.controller.prefetch_cache.get_subtitle_line(track_id, 0))
        self.assertIn("1 行失败", self.progress_text[-1])

    def test_settings_change_during_translation_does_not_cache_wrong_target(self) -> None:
        track_id = self._register()

        def changed(_text):
            self.controller.settings.config.translation.target_language = "en"
            return "Hello world"

        self.controller.translator.translate_text = changed
        self.assertFalse(self.controller.prefetch_scheduler._drain_one())
        self.assertIsNone(self.controller.prefetch_cache.get_subtitle_line(track_id, 0))

    def test_reimport_while_old_line_is_in_flight_cannot_restore_old_source(self) -> None:
        track_id = self._register()
        entered, release = threading.Event(), threading.Event()
        results = []

        def blocked_translation(text):
            self.assertEqual(text, "Hello world")
            entered.set()
            self.assertTrue(release.wait(timeout=3))
            return "旧字幕的译文"

        self.controller.translator.translate_text = blocked_translation

        def run():
            try:
                results.append(self.controller.prefetch_scheduler._drain_one())
            finally:
                self.controller.prefetch_cache.close()

        worker = threading.Thread(target=run)
        worker.start()
        try:
            self.assertTrue(entered.wait(timeout=2))
            path = Path(track_id)
            path.write_text(
                "1\n00:00:01,000 --> 00:00:02,000\nUpdated subtitle line\n\n"
                "2\n00:00:03,000 --> 00:00:04,000\nSave the file\n", encoding="utf-8",
            )
            lines = parse_subtitle_file(path, normalize_for_key)
            self.controller.register_subtitle_track(
                SubtitleTrack(track_id, str(path), path.name, len(lines)), lines
            )
        finally:
            release.set()
            worker.join(timeout=3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(results, [False])
        self.assertIsNone(self.controller.prefetch_cache.get_subtitle_line(track_id, 0))
        self.assertEqual(
            self.controller.prefetch_scheduler.page_progress(f"track:{track_id}"), (0, 2)
        )

    def test_disable_pretranslation_prevents_model_request(self) -> None:
        track_id = self._register()
        self.controller.settings.config.prefetch.enabled = False
        job = PrefetchJob(0, 0, "subtitle_line", track_id, 0, f"track:{track_id}")
        self.assertFalse(self.controller._pull_prefetch_job(job))
        self.assertEqual(self.pool.calls, [])

    def test_concise_output_is_not_written_into_standard_subtitle_cache(self) -> None:
        track_id = self._register()
        self.controller.settings.config.translation.concise_output = True
        self.assertTrue(self.controller.prefetch_scheduler._drain_one())
        self.assertIsNone(self.controller.prefetch_cache.get_subtitle_line(track_id, 0))
        self.assertEqual(len(self.pool.calls), 1)

    def test_online_provider_output_is_not_written_into_shared_subtitle_cache(self) -> None:
        track_id = self._register()
        self.controller.translator._pool.translation_cache_namespace = lambda: "api:1:0"
        self.assertTrue(self.controller.prefetch_scheduler._drain_one())
        self.assertIsNone(self.controller.prefetch_cache.get_subtitle_line(track_id, 0))

    def test_web_pretranslation_uses_quality_gate_and_detected_language(self) -> None:
        self.controller.prefetch_gate = SimpleNamespace(
            snapshot=lambda: SimpleNamespace(gate=SimpleNamespace(value="预翻译中"))
        )
        self.controller.prefetch_metrics = mock.Mock()
        article = SimpleNamespace(text_blocks=["Save the file"], etag="v1")
        result = SimpleNamespace(kind="ok", html="<p>Save the file</p>", final_url="https://test.invalid", etag="v1")
        with mock.patch("pingyitong.services.prefetch.web_fetcher.fetch", return_value=result), mock.patch(
            "pingyitong.services.prefetch.web_fetcher.extract_article", return_value=article
        ):
            self.assertTrue(self.controller._prefetch_web_page("https://test.invalid/page"))
        key = SentenceCache.cache_key("Save the file", "en", "zh", "test.invalid", "test-policy")
        cached = self.controller.prefetch_cache.get_exact(key)
        self.assertTrue(cached)
        self.assertNotIn("<<<SEG>>>", cached)
        self.assertEqual(len(self.pool.calls), 1)


if __name__ == "__main__":
    unittest.main()
