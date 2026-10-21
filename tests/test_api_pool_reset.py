"""A local model rescan must preserve the online connection pool."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from pingyitong.components.llm import ModelPool
from pingyitong.config.schema import ApiConfig, AppConfig
from pingyitong.constants import MODEL_FAST
from tests.test_api_engine import MockChatServer


class TestApiPoolReset(unittest.TestCase):
    def test_explicit_local_translation_never_uses_available_online_provider(self):
        server = MockChatServer()
        pool = ModelPool(AppConfig().model)
        self.addCleanup(server.close)
        self.addCleanup(pool.shutdown)
        api = ApiConfig(enabled=True, base_url=server.base_url, model_name="test-model")
        with patch("pingyitong.components.llm.api_engine.ApiEngine.preconnect"):
            pool.apply_api_config(api)
        with patch.object(pool._api_engine, "translate_batch") as online, \
             patch.object(pool, "ensure_loaded", return_value=False):
            assert pool.translate_local(MODEL_FAST, "private words", "s") == ""
            online.assert_not_called()

    def test_rescan_retains_working_api_and_reuses_connection(self):
        server = MockChatServer()
        pool = ModelPool(AppConfig().model)
        self.addCleanup(server.close)
        self.addCleanup(pool.shutdown)
        api = ApiConfig(enabled=True, base_url=server.base_url, api_key="test-key",
                        model_name="test-model", timeout_seconds=2)
        with patch("pingyitong.components.llm.api_engine.ApiEngine.preconnect"):
            pool.apply_api_config(api)
        engine = pool._api_engine
        self.assertEqual(pool.translate(MODEL_FAST, "first", "s"), "译:first")
        connections = server.connections
        pool.reset()
        pool.apply_api_config(api)
        self.assertIs(pool._api_engine, engine)
        self.assertFalse(engine._closed)
        self.assertEqual(pool.translate(MODEL_FAST, "second", "s"), "译:second")
        self.assertEqual(server.connections, connections)
        pool.shutdown()
        self.assertTrue(engine._closed)
