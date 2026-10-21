"""Web reading must use the same verified translations as screen reading."""

from __future__ import annotations

import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest import mock

from pingyitong.config.schema import AppConfig
from pingyitong.ui.web_reader_dialog import WebReaderDialog
from tests.test_cache_and_code_routing_regression import _OfflinePool
from tests.test_translation_service import make_service


class TestWebReaderTranslationIntegrity(unittest.TestCase):
    def _dialog(self, pool):
        config = AppConfig()
        config.translation.semantic_review = False  # legacy single-pass contract
        config.translation.source_language = "en"
        translator = make_service(config, pool=pool)[0]
        app = SimpleNamespace(
            settings=SimpleNamespace(config=config), translator=translator,
            prefetch_cache=SimpleNamespace(_model_version="v", put_many=mock.Mock()),
            scheduler=SimpleNamespace(current_model="fast"), model_pool=pool,
        )
        dialog = SimpleNamespace(
            _app=app, _config=config, _blocks=["Save the file."], _host="example.test",
            _reader=mock.Mock(), _status=mock.Mock(), _go_button=mock.Mock(),
            _busy=False, isVisible=lambda: True,
            _post_to_ui=lambda fn: fn(), _done=mock.Mock(), _enqueue_next_chapters=mock.Mock(),
        )
        return dialog

    def _run(self, dialog):
        article = SimpleNamespace(etag="etag", next_candidates=[])
        with mock.patch("pingyitong.ui.web_reader_dialog.threading.Thread") as thread:
            thread.return_value.start.side_effect = (
                lambda: thread.call_args.kwargs["target"]()
            )
            WebReaderDialog._translate_blocks(dialog, article)

    def test_bad_output_is_neither_shown_nor_written(self):
        for bad in ("Save the file.", "请保存<SEG文件。", "很抱歉，我无法为你提供翻译。"):
            with self.subTest(bad=bad):
                pool = _OfflinePool(*([bad] * 8))
                dialog = self._dialog(pool)
                self._run(dialog)
                dialog._reader.update_translation.assert_not_called()
                dialog._app.prefetch_cache.put_many.assert_not_called()
                self.assertEqual(len(dialog._app.translator.cache), 0)

    def test_success_uses_official_service_and_captured_language(self):
        dialog = self._dialog(_OfflinePool("请保存文件。"))
        with mock.patch.object(dialog._app.translator, "translate_text",
                               wraps=dialog._app.translator.translate_text) as translate:
            self._run(dialog)
        translate.assert_called_once_with("Save the file.")
        dialog._reader.update_translation.assert_called_once_with(0, "请保存文件。")
        entry = dialog._app.prefetch_cache.put_many.call_args.args[0][0]
        self.assertEqual((entry.src_lang, entry.tgt_lang, entry.domain),
                         ("en", "zh", "example.test"))

    def test_target_switch_during_translation_drops_old_result(self):
        dialog = self._dialog(_OfflinePool("请保存文件。"))
        translator = dialog._app.translator

        def switch_during_inference(**kwargs):
            updated = deepcopy(dialog._app.settings.config)
            updated.translation.target_language = "ko"
            dialog._app.settings.config = updated
            translator.apply_config(updated)
            return "请保存文件。"

        dialog._app.model_pool.translate = switch_during_inference
        self._run(dialog)
        dialog._reader.update_translation.assert_not_called()
        dialog._app.prefetch_cache.put_many.assert_not_called()
        self.assertEqual(len(translator.cache), 0)
        self.assertFalse(dialog._busy)

    def test_old_dialog_config_cannot_override_current_direction(self):
        dialog = self._dialog(_OfflinePool("파일을 저장하세요."))
        latest = deepcopy(dialog._app.settings.config)
        latest.translation.target_language = "ko"
        dialog._app.settings.config = latest
        dialog._app.translator.apply_config(latest)
        self._run(dialog)
        entry = dialog._app.prefetch_cache.put_many.call_args.args[0][0]
        self.assertEqual(entry.tgt_lang, "ko")
        dialog._reader.update_translation.assert_called_once_with(0, "파일을 저장하세요.")

    def test_reader_close_cancels_result_without_worker_touching_qt(self):
        dialog = self._dialog(_OfflinePool("请保存文件。"))
        dialog.isVisible = mock.Mock(side_effect=AssertionError("Qt queried by worker"))
        original = dialog._app.translator.translate_text

        def close_during_translation(block):
            result = original(block)
            dialog._translation_cancel.set()
            return result

        dialog._app.translator.translate_text = close_during_translation
        self._run(dialog)
        dialog._reader.update_translation.assert_not_called()
        dialog._app.prefetch_cache.put_many.assert_not_called()
        dialog.isVisible.assert_not_called()


if __name__ == "__main__":
    unittest.main()
