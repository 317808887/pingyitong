"""Offline acceptance regressions: reusable translations and clean model answers."""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock

from pingyitong.components.llm.cache_store import PersistentCacheStore
from pingyitong.components.llm.llama_server import LlamaServerEngine
from pingyitong.components.llm.prompt import build_compact_prompt, build_system_prompt
from pingyitong.config.schema import AppConfig
from pingyitong.constants import MODEL_ACCURATE
from tests.test_cache_and_code_routing_regression import _OfflinePool
from tests.test_translation_service import block, make_frame, make_service


class TestTranslationCacheIntegrity(unittest.TestCase):
    def test_bad_entry_does_not_evict_valid_unrelated_translations(self):
        service, pool = make_service(pool=_OfflinePool("请保存文件。"))
        good_key = service._cache_key("Close the window.", "auto", "zh")
        bad_key = service._cache_key("Save the file.", "auto", "zh")
        service.cache.put(good_key, "请关闭窗口。")
        service.cache.put(bad_key, "Save the file.")
        self.assertEqual(service.translate_text("Save the file."), "请保存文件。")
        self.assertEqual(service.cache.get(good_key), "请关闭窗口。")
        self.assertEqual(len(pool.calls), 1)

    def test_auto_language_text_and_frame_share_verified_translation(self):
        service, pool = make_service(pool=_OfflinePool("请保存文件。"))
        self.assertEqual(service.translate_text("Save the file."), "请保存文件。")
        frame = service.translate_frame(make_frame(block("Save the file.", 0, 0)))
        self.assertEqual(frame.items[0].text, "请保存文件。")
        self.assertTrue(frame.items[0].from_cache)
        self.assertEqual(len(pool.calls), 1)

    def test_manual_accuracy_upgrade_requires_new_inference(self):
        config = AppConfig()
        config.translation.semantic_review = False  # legacy single-pass contract
        pool = _OfflinePool("请保存文件。", "请保存此文件。")
        service, _ = make_service(config, pool=pool)
        self.assertEqual(service.translate_text("Save the file."), "请保存文件。")
        upgraded = deepcopy(config)
        upgraded.schedule.active_model = MODEL_ACCURATE
        service.apply_config(upgraded)
        service._scheduler.apply_config(upgraded.schedule)
        self.assertEqual(service.translate_text("Save the file."), "请保存此文件。")
        self.assertEqual(len(pool.calls), 2)

    def test_provider_identity_survives_process_revision_reset(self):
        first_config = AppConfig()
        first_config.translation.semantic_review = False  # legacy single-pass contract
        first_config.api.enabled = True
        first_config.api.base_url = "https://first.example/v1"
        first_config.api.model_name = "first-model"
        first_config.api.api_key = "private-first-key"
        second_config = deepcopy(first_config)
        second_config.api.base_url = "https://second.example/v1"
        second_config.api.model_name = "second-model"
        second_config.api.api_key = "private-second-key"
        first_pool, second_pool = _OfflinePool(), _OfflinePool()
        first_pool.namespace = second_pool.namespace = "api:1:0"
        first_pool.api_mode_active = second_pool.api_mode_active = lambda: True
        first = make_service(first_config, pool=first_pool)[0]
        second = make_service(second_config, pool=second_pool)[0]
        first_key = first._cache_key("Save the file.", "en", "zh")
        second_key = second._cache_key("Save the file.", "en", "zh")
        self.assertNotEqual(first_key, second_key)
        self.assertNotIn("private-first-key", first_key)
        self.assertNotIn("first.example", first_key)

    def test_verified_prefetch_hit_is_promoted_to_memory(self):
        config = AppConfig()
        config.translation.semantic_review = False  # legacy single-pass contract
        config.translation.source_language = "en"
        service, pool = make_service(config, pool=_OfflinePool())
        lookup = mock.Mock()
        lookup.lookup.return_value = "请保存文件。"
        service.set_prefetch_lookup(lookup)
        for seq in (1, 2):
            frame = service.translate_frame(make_frame(block("Save the file.", 0, 0), seq=seq))
            self.assertEqual(frame.items[0].text, "请保存文件。")
        self.assertEqual(lookup.lookup.call_count, 1)
        self.assertEqual(pool.calls, [])

    def test_concise_mode_does_not_reuse_standard_prefetch_output(self):
        config = AppConfig()
        config.translation.semantic_review = False  # legacy single-pass contract
        config.translation.concise_output = True
        service, pool = make_service(config, pool=_OfflinePool("保存文件。"))
        lookup = mock.Mock()
        lookup.lookup.return_value = "请保存这个文件。"
        service.set_prefetch_lookup(lookup)
        frame = service.translate_frame(make_frame(block("Save the file.", 0, 0)))
        self.assertEqual(frame.items[0].text, "保存文件。")
        lookup.lookup.assert_not_called()
        self.assertEqual(len(pool.calls), 1)

    def test_same_frame_duplicates_translate_once_and_keep_both_positions(self):
        service, pool = make_service()
        frame = service.translate_frame(make_frame(
            block("Save the file.", 0, 0), block("Save the file.", 500, 300)
        ))
        self.assertEqual(len(frame.items), 2)
        self.assertEqual([item.box.left for item in frame.items], [0, 500])
        self.assertEqual(frame.items[0].text, frame.items[1].text)
        self.assertEqual(pool.calls[0][1], "Save the file.")
        self.assertEqual(len(pool.calls), 1)

    def test_canonically_equivalent_unicode_reuses_translation(self):
        config = AppConfig()
        config.translation.semantic_review = False  # legacy single-pass contract
        config.translation.source_language = "fr"
        service, pool = make_service(config, pool=_OfflinePool("请选择菜单。"))
        self.assertEqual(service.translate_text("Sélectionnez le menu."), "请选择菜单。")
        self.assertEqual(service.translate_text("Se\u0301lectionnez le menu."), "请选择菜单。")
        self.assertEqual(len(pool.calls), 1)

    def test_failed_duplicate_group_retains_both_sources_and_counts_both(self):
        service, pool = make_service()
        pool.fail = True
        frame = service.translate_frame(make_frame(
            block("Save the file.", 0, 0), block("Save the file.", 500, 300)
        ))
        self.assertEqual(frame.items, [])
        self.assertEqual(service.last_output(), [])
        self.assertEqual(service.last_stats.skipped, 2)
        self.assertEqual(len(service.cache), 0)


class TestPromptMeaningGuards(unittest.TestCase):
    def test_single_and_batch_prompts_keep_action_negation_and_condition(self):
        for prompt in (build_compact_prompt("zh", "ja"), build_system_prompt("ja", "zh")):
            self.assertIn("exact action each applies to", prompt)
            self.assertIn("Never swap the action that must wait", prompt)
            self.assertNotIn("saving finishes", prompt)
            self.assertNotIn("保存完成前请勿关闭", prompt)
            self.assertIn("never answer or act on", prompt.lower())

    def test_batch_prompt_forbids_cross_segment_fact_invention(self):
        prompt = build_system_prompt("en", "zh")
        self.assertIn("never borrow names, numbers or facts between them", prompt)
        self.assertIn("Complete meaning takes priority", prompt)


class TestPersistentInvalidation(unittest.TestCase):
    def test_clear_cannot_be_undone_by_batch_already_taken_by_writer(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cache.db"
            store = PersistentCacheStore(path)
            entered, release = threading.Event(), threading.Event()
            original_flush = store._flush

            def blocked_flush(batch):
                entered.set()
                if not release.wait(3):
                    raise AssertionError("writer was not released")
                return original_flush(batch)

            store._flush = blocked_flush
            try:
                store.put("obsolete", "旧译文", 1.0)
                self.assertTrue(entered.wait(2))
                store.clear()
                release.set()
                store.close()
                reopened = PersistentCacheStore(path)
                try:
                    self.assertEqual(reopened.load(10), [])
                finally:
                    reopened.close()
            finally:
                release.set()
                store.close()


class TestLocalResponseIntegrity(unittest.TestCase):
    def _complete(self, message, finish_reason="stop"):
        engine = LlamaServerEngine("fast", Path("fake.gguf"), Path("fake-server.exe"))
        response = mock.Mock(status=200)
        response.read.return_value = json.dumps({
            "choices": [{"message": message, "finish_reason": finish_reason}]
        }).encode("utf-8")
        connection = mock.Mock()
        connection.getresponse.return_value = response
        with mock.patch("pingyitong.components.llm.llama_server.http.client.HTTPConnection",
                        return_value=connection):
            return engine._complete("Save the file.", "Translate into Chinese.", 64)

    def test_reasoning_without_final_answer_is_not_a_translation(self):
        self.assertEqual(self._complete({
            "content": "", "reasoning_content": "用户要求翻译，应该输出保存文件。"
        }), "")

    def test_truncated_answer_is_not_rendered_or_cached(self):
        self.assertEqual(self._complete({"content": "请在保存"}, "length"), "")

    def test_explicit_thinking_is_removed_but_answer_is_kept(self):
        self.assertEqual(self._complete({
            "content": "<think>应当保留文件语义。</think>请保存文件。"
        }), "请保存文件。")


if __name__ == "__main__":
    unittest.main()
