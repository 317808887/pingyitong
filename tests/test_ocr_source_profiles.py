"""Source alphabet routing, bounded resident models and nonblocking setup."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PyQt6.QtWidgets import QApplication

from pingyitong.components.ocr.ocr_engine import (
    OcrEngine,
    _join_segment_results,
    _Segment,
    create_ocr_engine,
)
from pingyitong.components.ocr.recognizer_profiles import (
    profile_for_source,
    reverse_rtl_prediction,
)
from pingyitong.config.schema import AppConfig, OcrConfig
from pingyitong.core.pipeline import Pipeline
from pingyitong.ui.app import Application


class _Backend:
    name = "synthetic"
    error = ""
    def __init__(self, *, threads=0, profile=None):
        self.profile, self.threads = profile, threads
        self._crop_cache = {"unchanged": ("correct text", .99)}
    def is_ready(self):
        return True


class TestSourceRouting(unittest.TestCase):
    def test_all_official_script_profiles(self):
        expected = {"auto":"ch", "zh":"ch", "zh-TW":"ch", "ja":"ch",
                    "en":"latin", "fr":"latin", "ko":"korean", "ru":"cyrillic",
                    "ar":"arabic", "fa":"arabic", "hi":"devanagari", "el":"greek",
                    "th":"thai", "ta":"tamil", "te":"telugu", "kn":"kannada"}
        for source, key in expected.items():
            with self.subTest(source=source):
                self.assertEqual(profile_for_source(source).key, key)

    def test_georgian_not_mapped_from_kannada_filename(self):
        with self.assertRaisesRegex(ValueError, "格鲁吉亚语.*georgian"):
            profile_for_source("ka")

    def test_unknown_and_missing_model_do_not_fallback(self):
        with self.assertRaisesRegex(ValueError, "bad-source"):
            profile_for_source("bad-source")
        with tempfile.TemporaryDirectory() as folder, patch("pingyitong.components.ocr.recognizer_profiles.asset_path", return_value=Path(folder)/"absent.onnx"):
            engine = OcrEngine(source_language="ko")
            self.assertIn("韩语", engine.preflight_error())
            self.assertIn("absent.onnx", engine.preflight_error())
            self.assertIsNone(engine._backend)

    def test_factory_explicit_source(self):
        engine = create_ocr_engine(OcrConfig(), source_language="ja")
        self.assertEqual(engine.source_language, "ja")
        self.assertEqual(engine._profile_key, "ch")

    def test_unsupported_never_constructs_chinese_backend(self):
        with patch("pingyitong.components.ocr.ocr_engine.RapidOcrBackend") as factory:
            engine = OcrEngine(source_language="he")
            self.assertFalse(engine.is_ready())
            self.assertIn("希伯来语", engine.error_message())
            factory.assert_not_called()


class TestProfileLifecycle(unittest.TestCase):
    def test_source_switch_and_status_inspection_do_not_load(self):
        with patch("pingyitong.components.ocr.ocr_engine.RapidOcrBackend", side_effect=AssertionError("GUI load")):
            engine = OcrEngine(source_language="en")
            engine.configure(source_language="ko")
            _ = engine.backend_name
            _ = engine.error_message()
            self.assertIsNone(engine._backend)

    def test_same_script_and_target_parameter_update_reuse_cache(self):
        with patch("pingyitong.components.ocr.ocr_engine.RapidOcrBackend", side_effect=_Backend) as factory:
            engine = OcrEngine(source_language="en")
            original = engine.backend
            engine.configure(source_language="fr", min_confidence=.6)
            self.assertIs(engine.backend, original)
            self.assertIn("unchanged", original._crop_cache)
            self.assertEqual(factory.call_count, 1)

    def test_profile_cache_is_lru_and_bounded_to_three(self):
        with patch("pingyitong.components.ocr.ocr_engine.RapidOcrBackend", side_effect=_Backend) as factory:
            engine = OcrEngine(source_language="en")
            latin = engine.backend
            for source in ("ko", "ru", "en"):
                engine.configure(source_language=source)
                _ = engine.backend
            self.assertIs(engine.backend, latin)
            self.assertEqual(factory.call_count, 3)
            engine.configure(source_language="ar")
            _ = engine.backend
            self.assertEqual(len(engine._profile_backends), 3)
            self.assertNotIn("korean", engine._profile_backends)
            engine.configure(source_language="ko")
            self.assertIsNot(engine.backend, latin)
            self.assertEqual(factory.call_count, 5)

    def test_thread_change_clears_all_profiles(self):
        with patch("pingyitong.components.ocr.ocr_engine.RapidOcrBackend", side_effect=_Backend):
            engine = OcrEngine(source_language="en", threads=4)
            _ = engine.backend
            engine.configure(source_language="ko")
            _ = engine.backend
            engine.configure(threads=2)
            self.assertFalse(engine._profile_backends)
            self.assertIsNone(engine._backend)
            self.assertEqual(engine.backend.threads, 2)

    def test_switch_during_load_never_publishes_old_profile(self):
        entered, release = threading.Event(), threading.Event()
        def factory(**kwargs):
            if kwargs["profile"].key == "latin":
                entered.set()
                release.wait(3)
            return _Backend(**kwargs)
        results = []
        with patch("pingyitong.components.ocr.ocr_engine.RapidOcrBackend", side_effect=factory):
            engine = OcrEngine(source_language="en")
            worker = threading.Thread(target=lambda: results.append(engine.backend))
            worker.start()
            try:
                self.assertTrue(entered.wait(2))
                engine.configure(source_language="ko")
            finally:
                release.set()
                worker.join(3)
            self.assertFalse(worker.is_alive())
            self.assertEqual(results[0].profile.key, "korean")

    def test_away_and_back_during_load_reuses_valid_loaded_profile(self):
        entered, release = threading.Event(), threading.Event()
        def factory(**kwargs):
            entered.set()
            release.wait(3)
            return _Backend(**kwargs)
        with patch("pingyitong.components.ocr.ocr_engine.RapidOcrBackend", side_effect=factory) as create:
            engine = OcrEngine(source_language="en")
            worker = threading.Thread(target=lambda: engine.backend)
            worker.start()
            try:
                self.assertTrue(entered.wait(2))
                engine.configure(source_language="ko")
                engine.configure(source_language="en")
            finally:
                release.set()
                worker.join(3)
            self.assertEqual(create.call_count, 1)
            self.assertEqual(engine.backend.profile.key, "latin")


class TestPreflightOnGui(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_start_preflights_without_model_creation(self):
        ocr = SimpleNamespace(preflight_error=Mock(return_value=""), is_ready=Mock(side_effect=AssertionError("GUI load")))
        pipe = Pipeline(AppConfig(), object(), object(), ocr,
                        SimpleNamespace(last_stats=SimpleNamespace(model_id="test")), Mock())
        with patch.object(pipe, "_spawn"):
            self.assertTrue(pipe.start())
        pipe.stop()
        ocr.preflight_error.assert_called_once()
        ocr.is_ready.assert_not_called()

    def test_source_changes_during_warmup_schedule_current_backend(self):
        entered, release = threading.Event(), threading.Event()
        calls=[]
        old, new = object(),object()
        engine=SimpleNamespace(_backend=old,_backend_generation=0)
        def warm():
            calls.append(engine._backend)
            if len(calls)==1:
                entered.set()
                release.wait(3)
            return True
        engine.warm_up=warm
        controller=SimpleNamespace(ocr_engine=engine,_ocr_warm_lock=threading.Lock(),
                                   _ocr_warm_running=False,_ocr_warmed_backend=None)
        Application._warm_ocr_async(controller)
        try:
            self.assertTrue(entered.wait(2))
            engine._backend=new
            engine._backend_generation=1
        finally:
            release.set()
        deadline=time.perf_counter()+3
        while len(calls)<2 and time.perf_counter()<deadline:
            time.sleep(.001)
        self.assertEqual(calls,[old,new])

    def test_dependencies_preflight_without_model_creation(self):
        ocr = SimpleNamespace(preflight_error=Mock(return_value=""), is_ready=Mock(side_effect=AssertionError("GUI load")))
        controller = SimpleNamespace(ocr_engine=ocr, model_pool=SimpleNamespace(any_available=lambda:True))
        with patch("pingyitong.ui.app.ScreenCapturer.is_available", return_value=True):
            Application._check_dependencies(controller)
        ocr.preflight_error.assert_called_once()
        ocr.is_ready.assert_not_called()


class TestArabicDecoder(unittest.TestCase):
    def test_arabic_indic_digits_remain_in_logical_numeric_order(self):
        self.assertEqual(reverse_rtl_prediction("نآلا حاتم ١٢٣٤٥ بلطلا مقر"), "رقم الطلب ١٢٣٤٥ متاح الآن")

    def test_ascii_and_persian_digits_preserve_runs(self):
        for digits in ("12345", "۱۲۳۴۵", "١٢٣٫٤٥", "١٢٣٬٤٥٦", "١٢٣،٤٥", "2026/10/06"):
            with self.subTest(digits=digits):
                self.assertIn(digits, reverse_rtl_prediction("نآلا " + digits + " مقر"))

    def test_latin_only_decoder_is_identity(self):
        self.assertEqual(reverse_rtl_prediction("Order 12345"), "Order 12345")


class TestMultilingualSegmentSpaces(unittest.TestCase):
    def test_measured_word_gap_preserved_for_non_ascii_words(self):
        for left,right in (("рождаются","свободными"),("γεννιούνται","ελεύθεροι"),
                           ("때부터","자유로우며"),("பிறிவியினர்","சகலரும்")):
            with self.subTest(left=left):
                self.assertEqual(_join_segment_results([(left,.99),(right,.99)],
                    [_Segment(None,False),_Segment(None,True)])[0], left+" "+right)

    def test_hard_cut_never_adds_word_space(self):
        self.assertEqual(_join_segment_results([("рожда",.99),("ются",.99)],
            [_Segment(None,False),_Segment(None,False)])[0],"рождаются")

    def test_han_and_kana_do_not_acquire_spaces(self):
        for left,right in (("中文","文本"),("すべて","の人間")):
            self.assertEqual(_join_segment_results([(left,.99),(right,.99)],
                [_Segment(None,False),_Segment(None,True)])[0],left+right)


if __name__ == "__main__":
    unittest.main()
