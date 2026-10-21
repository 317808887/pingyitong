"""Self-check must honor its data path and verify the known OCR literal."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pingyitong import selfcheck
from pingyitong.__main__ import main
from pingyitong.utils import paths


class TestSelfCheckCliOcr(unittest.TestCase):
    def setUp(self):
        self.previous_portable = paths.is_portable()
        self.addCleanup(paths.set_portable, self.previous_portable)

    def test_home_selfcheck_writes_only_requested_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            expected=Path(folder)/"selected home"
            def check():
                self.assertEqual(paths.app_data_dir(),expected)
                self.assertEqual(selfcheck._write_log(["isolated self-check"]), expected/"selfcheck.log")
                return 0
            with patch.dict(os.environ,{},clear=False), patch.object(selfcheck,"run_self_check",side_effect=check):
                os.environ.pop("PINGYITONG_HOME",None)
                self.assertEqual(main(["--self-check","--home",str(expected)]),0)
            self.assertEqual((expected/"selfcheck.log").read_text(encoding="utf-8"),"isolated self-check\n")

    def test_portable_selfcheck_resolves_program_data_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            def check():
                self.assertTrue(paths.is_portable())
                self.assertEqual(paths.app_data_dir(),root/"data")
                return 0
            with patch.dict(os.environ,{},clear=False), patch.object(paths,"project_root",return_value=root), patch.object(selfcheck,"run_self_check",side_effect=check):
                os.environ.pop("PINGYITONG_HOME",None)
                self.assertEqual(main(["--portable","--self-check"]),0)

    def test_ocr_unready_returns_specific_model_error(self):
        engine=Mock()
        engine.is_ready.return_value=False
        engine.error_message.return_value="源语言 auto：缺少离线 OCR 模型 ch-v5.onnx"
        with patch("pingyitong.components.ocr.ocr_engine.OcrEngine",return_value=engine):
            ok,detail=selfcheck._check_ocr()
        self.assertFalse(ok)
        self.assertIn("ch-v5.onnx",detail)
        engine.recognize_image.assert_not_called()

    def test_nonempty_wrong_ocr_cannot_pass_and_literal_normalizes(self):
        engine=Mock()
        engine.is_ready.return_value=True
        for value,expected in (("Self test 12345",True),("SELF\n test 12345",True),
                               ("Self test 1234",False),("Self test 54321",False),
                               ("Hello world",False),("识别出其他文字12345",False)):
            with self.subTest(value=value):
                engine.recognize_image.return_value=[SimpleNamespace(text=value)]
                with patch("pingyitong.components.ocr.ocr_engine.OcrEngine",return_value=engine):
                    ok,detail=selfcheck._check_ocr()
                self.assertEqual(ok,expected)
                if not expected:
                    self.assertIn("测试图识别不符",detail)


if __name__ == "__main__":
    unittest.main()
