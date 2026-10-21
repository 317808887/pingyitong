"""Deterministic request-boundary regressions; no model or external API calls."""

from __future__ import annotations

import threading
import unittest
from copy import deepcopy

from pingyitong.config.schema import AppConfig
from pingyitong.constants import GLOSSARY_TOKEN_TEMPLATE, SEGMENT_DELIMITER
from pingyitong.services.glossary_service import Glossary, GlossaryEntry
from tests.test_translation_service import block, make_frame, make_service


class _RequestPool:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.namespace = "local"
        self.online = False
        self.on_call = None
        self.reply = "欢迎使用此应用程序。"

    def translate(self, model_id, prompt, system_prompt, max_tokens=None):
        self.calls.append((prompt, system_prompt))
        if self.on_call is not None:
            self.on_call(prompt, system_prompt)
        return self.reply

    def translation_cache_namespace(self) -> str:
        return self.namespace

    def api_mode_active(self) -> bool:
        return self.online

    def any_available(self) -> bool:
        return True

    def is_available(self, model_id) -> bool:
        return True


class _History:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def record(self, **values):
        self.records.append(values)


def _config() -> AppConfig:
    config = AppConfig()
    config.translation.semantic_review = False  # legacy single-pass contract
    config.translation.source_language = "en"
    config.translation.target_language = "zh"
    config.schedule.mode = "manual"
    return config


class TestTranslationRequestContext(unittest.TestCase):
    def test_direction_switch_between_prepare_and_inference_cannot_pollute_old_key(self):
        config = _config()
        updated = deepcopy(config)
        updated.translation.source_language = "auto"
        updated.translation.target_language = "en"
        pool = _RequestPool()
        service, _ = make_service(config, pool=pool)
        history = _History()
        service._history = history
        service._last_output = [("Previous", "先前译文")]
        original_decide = service._scheduler.decide

        def switch_during_frame(**kwargs):
            service.apply_config(updated)
            return original_decide(**kwargs)

        service._scheduler.decide = switch_during_frame
        first = service.translate_frame(make_frame(block("Hello world", 0, 0)))
        self.assertEqual(first.items[0].text, pool.reply)
        self.assertIn("Chinese", pool.calls[0][1])
        old_key = service._cache_key("Hello world", "en", "zh")
        self.assertIsNone(service.cache.get(old_key))
        self.assertEqual(service.last_output_text(), "先前译文")
        self.assertEqual(history.records, [])

        service._scheduler.decide = original_decide
        service.apply_config(config)
        second = service.translate_frame(make_frame(block("Hello world", 0, 0), seq=2))
        self.assertFalse(second.items[0].from_cache)
        self.assertEqual(len(pool.calls), 2)
        self.assertEqual(history.records[0]["target_lang"], "zh")

    def test_in_place_config_change_is_detected_and_next_request_reads_latest_value(self):
        config = _config()
        pool = _RequestPool()
        service, _ = make_service(config, pool=pool)

        def mutate_during_inference(prompt, system):
            config.translation.target_language = "en"
            pool.on_call = None

        pool.on_call = mutate_during_inference
        self.assertEqual(service.translate_text("Hello world"), pool.reply)
        self.assertEqual(len(service.cache), 0)
        pool.reply = "Welcome to this application."
        self.assertEqual(service.translate_text("Hello world"), pool.reply)
        self.assertIn("English", pool.calls[-1][1])
        self.assertEqual(service.translate_text("Hello world"), pool.reply)
        self.assertEqual(len(pool.calls), 2)

    def test_same_version_glossary_replacement_invalidates_inflight_request(self):
        config = _config()
        original = Glossary([GlossaryEntry("model", "模型")])
        updated = Glossary([GlossaryEntry("model", "模组")])
        self.assertEqual(original.version, updated.version)
        pool = _RequestPool()
        pool.reply = GLOSSARY_TOKEN_TEMPLATE.format(0) + "已准备就绪。"
        service, _ = make_service(config, glossary=original, pool=pool)

        def replace_during_inference(prompt, system):
            service.set_glossary(updated)
            pool.on_call = None

        pool.on_call = replace_during_inference
        first = service.translate_frame(make_frame(block("The model is ready", 0, 0)))
        self.assertEqual(first.items[0].text, "模型已准备就绪。")
        self.assertEqual(len(service.cache), 0)
        self.assertEqual(service.last_output_text(), "")
        second = service.translate_frame(make_frame(block("The model is ready", 0, 0), seq=2))
        self.assertEqual(second.items[0].text, "模组已准备就绪。")
        self.assertFalse(second.items[0].from_cache)
        self.assertEqual(len(pool.calls), 2)

    def test_route_recovery_during_request_cannot_store_online_result_as_local(self):
        config = _config()
        config.api.enabled = True
        pool = _RequestPool()
        service, _ = make_service(config, pool=pool)

        def recover_during_inference(prompt, system):
            pool.namespace = "api:1:1"
            pool.online = True
            pool.on_call = None

        pool.on_call = recover_during_inference
        self.assertEqual(service.translate_text("Hello world"), pool.reply)
        self.assertEqual(len(service.cache), 0)
        self.assertEqual(service.translate_text("Hello world"), pool.reply)
        self.assertEqual(service.translate_text("Hello world"), pool.reply)
        self.assertEqual(len(pool.calls), 2)
        self.assertEqual(len(service.cache), 1)

    def test_online_batch_workers_share_original_config_and_recovery_language(self):
        config = _config()
        config.api.enabled = True
        config.api.batch_size = 1
        updated = deepcopy(config)
        updated.translation.target_language = "ko"
        updated.api.enabled = False
        pool = _RequestPool()
        pool.online = True
        pool.namespace = "api:1:0"
        service, _ = make_service(config, pool=pool)
        contexts = []
        seen: dict[str, int] = {}
        call_lock = threading.Lock()

        def reply(model_id, prompt, system_prompt, max_tokens=None):
            context = service._request_context.get()
            with call_lock:
                contexts.append(context)
                seen[prompt] = seen.get(prompt, 0) + 1
                first = seen[prompt] == 1
            if "Correct the output language" in system_prompt:
                return SEGMENT_DELIMITER.join(
                    "这是正确的中文译文。" for _ in prompt.split(SEGMENT_DELIMITER)
                )
            return "정확한 한국어 번역입니다" if first else "这是正确的中文译文。"

        pool.translate = reply
        original_decide = service._scheduler.decide

        def switch_before_workers(**kwargs):
            service.apply_config(updated)
            return original_decide(**kwargs)

        service._scheduler.decide = switch_before_workers
        result = service.translate_frame(make_frame(
            block("Hello world", 0, 0), block("Save changes", 0, 100)
        ))
        # One frame-wide recovery combines both workers' bad results, and
        # retains the captured configuration even after the UI changes it.
        self.assertEqual([item.text for item in result.items], ["这是正确的中文译文。"] * 2)
        self.assertEqual(len(contexts), 3)
        self.assertTrue(all(context is contexts[0] and context is not None for context in contexts))
        self.assertEqual(contexts[0].config.translation.target_language, "zh")
        self.assertEqual(len(service.cache), 0)

    def test_parallel_individual_retries_inherit_request_context(self):
        config = _config()
        config.api.enabled = True
        config.api.batch_size = 2
        updated = deepcopy(config)
        updated.translation.target_language = "ko"
        pool = _RequestPool()
        pool.online = True
        pool.namespace = "api:1:0"
        service, _ = make_service(config, pool=pool)
        contexts = []

        def reply(model_id, prompt, system_prompt, max_tokens=None):
            contexts.append(service._request_context.get())
            if SEGMENT_DELIMITER in prompt:
                service.apply_config(updated)
                return "Unparseable batch output"
            return "这是正确的中文译文。"

        pool.translate = reply
        result = service.translate_frame(make_frame(
            block("Hello world", 0, 0), block("Save changes", 0, 100)
        ))
        self.assertEqual([item.text for item in result.items], ["这是正确的中文译文。"] * 2)
        self.assertEqual(len(contexts), 3)
        self.assertTrue(all(context is contexts[0] and context is not None for context in contexts))
        self.assertEqual(len(service.cache), 0)

    def test_clear_output_cannot_be_undone_by_inflight_request(self):
        config = _config()
        pool = _RequestPool()
        service, _ = make_service(config, pool=pool)
        history = _History()
        service._history = history
        service._last_output = [("Previous", "先前译文")]
        pool.on_call = lambda prompt, system: service.clear_output()
        service.translate_frame(make_frame(block("Hello world", 0, 0)))
        self.assertEqual(service.last_output_text(), "")
        self.assertEqual(service.last_stats.paragraphs, 0)
        self.assertEqual(history.records, [])
        self.assertEqual(len(service.cache), 0)

    def test_config_change_does_not_wait_for_blocked_inference(self):
        config = _config()
        pool = _RequestPool()
        service, _ = make_service(config, pool=pool)
        entered = threading.Event()
        release = threading.Event()
        updated = threading.Event()
        errors = []

        def block_request(prompt, system):
            entered.set()
            if not release.wait(3):
                raise AssertionError("model request was not released")

        def translate():
            try:
                service.translate_text("Hello world")
            except Exception as exc:
                errors.append(exc)

        pool.on_call = block_request
        worker = threading.Thread(target=translate)
        worker.start()
        updater = None
        try:
            self.assertTrue(entered.wait(2))

            def update():
                service.apply_config(deepcopy(config))
                updated.set()

            updater = threading.Thread(target=update)
            updater.start()
            self.assertTrue(updated.wait(0.5), "configuration waited for model inference")
        finally:
            release.set()
            worker.join(3)
            if updater is not None:
                updater.join(3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(len(service.cache), 0)

    def test_history_io_does_not_block_config_and_uses_authorized_snapshot(self):
        config = _config()
        pool = _RequestPool()
        service, _ = make_service(config, pool=pool)
        entered = threading.Event()
        release = threading.Event()
        updated = threading.Event()
        records = []
        errors = []

        class BlockedHistory:
            def record(self, **values):
                entered.set()
                if not release.wait(3):
                    raise AssertionError("history write was not released")
                records.append(values)

        service._history = BlockedHistory()

        def translate():
            try:
                service.translate_frame(make_frame(block("Hello world", 0, 0)))
            except Exception as exc:
                errors.append(exc)

        worker = threading.Thread(target=translate)
        worker.start()
        updater = None
        try:
            self.assertTrue(entered.wait(2))
            changed = deepcopy(config)
            changed.translation.source_language = "auto"
            changed.translation.target_language = "en"
            changed.history.store_source_text = False

            def update():
                service.apply_config(changed)
                updated.set()

            updater = threading.Thread(target=update)
            updater.start()
            self.assertTrue(updated.wait(0.5), "configuration waited for history disk I/O")
        finally:
            release.set()
            worker.join(3)
            if updater is not None:
                updater.join(3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["source_lang"], "en")
        self.assertEqual(records[0]["target_lang"], "zh")
        self.assertTrue(records[0]["store_source"])


if __name__ == "__main__":
    unittest.main()
