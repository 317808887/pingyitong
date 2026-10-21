"""在线 AI 引擎（OpenAI 兼容接口）测试。

覆盖六块：

1. ``<think>`` 思考过程剥离（推理模型的正文污染防护）；
2. 引擎对本地 mock 服务的请求/响应全链路（载荷结构、鉴权头、
   分隔符拼装、错误处理、**连接复用**）；
3. 连续失败后的冷却与自动恢复（保证断网时请求直接走本地，
   不白付超时）；
4. :class:`ModelPool` 的路由与回退（在线优先、失败回落本地）；
5. 翻译服务在线模式的并发批量路径（结果对位、顺序保持）；
6. 配置兼容（旧配置文件无 ``api`` 节、非法值回落）与设置面板冒烟。

mock 服务是本文件自带的 :class:`MockChatServer`（标准库 ``http.server``），
不发任何真实网络请求。
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from pingyitong.components.capture.geometry import Rect
from pingyitong.components.llm import ModelPool, TranslationCache
from pingyitong.components.llm import api_engine as api_engine_module
from pingyitong.components.llm.api_engine import (
    MAX_OUTPUT_TOKENS,
    ApiEngine,
    strip_thinking,
)
from pingyitong.components.llm.base import EngineStatus
from pingyitong.components.ocr import RecognizedFrame, TextBlock
from pingyitong.config.schema import ApiConfig, AppConfig, from_dict, to_dict
from pingyitong.config.settings import SettingsManager
from pingyitong.constants import MODEL_FAST, SEGMENT_DELIMITER
from pingyitong.core.model_scheduler import ModelScheduler
from pingyitong.services.glossary_service import Glossary
from pingyitong.services.text_preprocess import TextPreprocessor
from pingyitong.services.translation_service import TranslationService
from tests.test_translation_service import fake_translation


def make_frame(*blocks: TextBlock, seq: int = 1) -> RecognizedFrame:
    """构造一帧 OCR 结果（与 test_translation_service 相同的约定）。"""
    return RecognizedFrame(
        seq=seq,
        origin=Rect(0, 0, 1920, 1080),
        blocks=list(blocks),
        is_full_frame=True,
    )


def block(text: str, x: int, y: int, width: int = 200, height: int = 16) -> TextBlock:
    """构造文本块。"""
    return TextBlock(text=text, box=Rect(x, y, width, height), confidence=0.95)


# ---------------------------------------------------------------------------
# mock 服务
# ---------------------------------------------------------------------------


class _MockHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self) -> None:
        self.server.connections += 1
        super().setup()

    def do_POST(self) -> None:  # noqa: N802 - http.server 命名约定
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        request = {
            "path": self.path,
            "headers": {key.lower(): value for key, value in self.headers.items()},
            "payload": json.loads(body.decode("utf-8")),
        }
        self.server.requests.append(request)
        status, content = self.server.respond(request)
        if status == 200:
            if isinstance(content, dict):
                # 完整响应体形态：模拟 finish_reason、分片 content 等
                # 需要精确控制顶层结构的场景
                data = json.dumps(content, ensure_ascii=False).encode("utf-8")
            else:
                data = json.dumps(
                    {"choices": [{"message": {"content": content}}]}, ensure_ascii=False
                ).encode("utf-8")
        else:
            # 错误响应按真实服务商形态原样返回（顶层 error 结构），
            # 让 _error_detail 走 error.message 提取路径而非 str(整个包)
            data = content.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args) -> None:
        pass


class MockChatServer:
    """进程内 OpenAI 兼容 mock 服务。

    :param responder: ``(request) -> (status, content)``；默认把
        user 内容原样加 ``译:`` 前缀返回（保留分隔符结构）。
        ``content`` 为 ``dict`` 时按完整响应体原样序列化（可携带
        ``finish_reason``、分片 ``content`` 等字段）；为 ``str`` 时
        包装成标准 ``choices[0].message.content`` 结构。
    """

    def __init__(self, responder=None) -> None:
        self.requests: list[dict] = []
        self._responder = responder
        self._http = ThreadingHTTPServer(("127.0.0.1", 0), _MockHandler)
        self._http.connections = 0
        self._http.requests = self.requests
        self._http.respond = self._respond
        self._thread = threading.Thread(
            target=self._http.serve_forever, name="MockChat", daemon=True
        )
        self._thread.start()

    def _respond(self, request: dict) -> tuple[int, str]:
        if self._responder is not None:
            return self._responder(request)
        user = request["payload"]["messages"][1]["content"]
        return 200, "译:" + user

    @property
    def base_url(self) -> str:
        host, port = self._http.server_address[:2]
        return f"http://{host}:{port}/v1"

    @property
    def connections(self) -> int:
        """已建立的 TCP 连接数（HTTP 层计数，keep-alive 复用不增加）。"""
        return self._http.connections

    def close(self) -> None:
        self._http.shutdown()
        self._http.server_close()


def make_engine(server: MockChatServer, **overrides) -> ApiEngine:
    fields = {
        "enabled": True,
        "base_url": server.base_url,
        "api_key": "sk-test",
        "model_name": "mock-model",
        "timeout_seconds": 5.0,
    }
    fields.update(overrides)
    return ApiEngine(ApiConfig(**fields))


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


# ---------------------------------------------------------------------------
# 1. 思考过程剥离
# ---------------------------------------------------------------------------


class TestStripThinking(unittest.TestCase):
    def test_removes_closed_block(self) -> None:
        self.assertEqual(strip_thinking("<think>让我想想</think>你好"), "你好")

    def test_removes_unclosed_block(self) -> None:
        self.assertEqual(strip_thinking("<think>思考被截断"), "")

    def test_plain_text_untouched(self) -> None:
        self.assertEqual(strip_thinking("普通译文"), "普通译文")

    def test_multiline_block(self) -> None:
        text = "<think>\nline1\nline2\n</think>\n结果"
        self.assertEqual(strip_thinking(text), "结果")


# ---------------------------------------------------------------------------
# 2. 引擎与 mock 服务的全链路
# ---------------------------------------------------------------------------


class TestApiEngineAgainstMock(unittest.TestCase):
    def setUp(self) -> None:
        self.server = MockChatServer()
        self.engine = make_engine(self.server)

    def tearDown(self) -> None:
        self.engine.unload()
        self.server.close()

    def test_translate_returns_content(self) -> None:
        self.assertTrue(self.engine.is_available())
        self.assertEqual(
            self.engine.translate_batch(["Hello"], "system"), ["译:Hello"]
        )

    def test_request_payload_structure(self) -> None:
        self.engine.translate_batch(["Hello"], "system", max_tokens=64)
        request = self.server.requests[0]
        self.assertEqual(request["path"], "/v1/chat/completions")
        self.assertEqual(request["payload"]["model"], "mock-model")
        self.assertEqual(request["payload"]["max_tokens"], 64)
        self.assertFalse(request["payload"]["stream"])
        self.assertEqual(request["payload"]["messages"][0]["role"], "system")
        self.assertEqual(request["payload"]["messages"][1]["content"], "Hello")
        self.assertEqual(request["headers"]["authorization"], "Bearer sk-test")

    def test_no_key_omits_authorization(self) -> None:
        engine = make_engine(self.server, api_key="")
        self.assertEqual(engine.translate_batch(["Hi"], "s"), ["译:Hi"])
        self.assertNotIn("authorization", self.server.requests[0]["headers"])

    def test_multi_segment_joined_with_delimiter(self) -> None:
        self.engine.translate_batch(["A", "B", "C"], "system")
        user = self.server.requests[0]["payload"]["messages"][1]["content"]
        self.assertEqual(user, f"A\n{SEGMENT_DELIMITER}\nB\n{SEGMENT_DELIMITER}\nC")

    def test_thinking_stripped_from_content(self) -> None:
        server = MockChatServer(
            lambda request: (200, "<think>先想一想</think>正确译文")
        )
        try:
            engine = make_engine(server)
            self.assertEqual(engine.translate_batch(["Hi"], "s"), ["正确译文"])
        finally:
            server.close()

    def test_error_status_returns_empty(self) -> None:
        server = MockChatServer(
            lambda request: (
                401,
                json.dumps({"error": {"message": "Invalid API key"}}),
            )
        )
        try:
            engine = make_engine(server)
            self.assertEqual(engine.translate_batch(["Hi"], "s"), [""])
            ok, message = engine.test()
            self.assertFalse(ok)
            self.assertIn("401", message)
        finally:
            server.close()

    def test_connection_reused_across_requests(self) -> None:
        """连接复用是延迟优化的核心：第二次请求不重新握手。"""
        self.engine.translate_batch(["one"], "s")
        self.engine.translate_batch(["two"], "s")
        self.engine.translate_batch(["three"], "s")
        self.assertEqual(self.server.connections, 1)

    def test_status_exposes_model_name(self) -> None:
        status = self.engine.status()
        self.assertTrue(status.available)
        self.assertEqual(status.message, "mock-model")


# ---------------------------------------------------------------------------
# 3. 冷却与自动恢复
# ---------------------------------------------------------------------------


class TestCooldown(unittest.TestCase):
    def make_dead_engine(self) -> ApiEngine:
        return ApiEngine(
            ApiConfig(
                enabled=True,
                base_url=f"http://127.0.0.1:{free_port()}/v1",
                model_name="m",
                timeout_seconds=1.0,
            )
        )

    def test_consecutive_failures_trip_cooldown(self) -> None:
        engine = self.make_dead_engine()
        for _ in range(api_engine_module.FAILURE_THRESHOLD):
            self.assertEqual(engine.translate_batch(["Hi"], "s"), [""])
        self.assertFalse(engine.is_available())

    def test_cooldown_expires_and_recovers(self) -> None:
        engine = self.make_dead_engine()
        for _ in range(api_engine_module.FAILURE_THRESHOLD):
            engine.translate_batch(["Hi"], "s")
        self.assertFalse(engine.is_available())
        with mock.patch.object(api_engine_module, "COOLDOWN_SECONDS", 0.05):
            time.sleep(0.1)
            self.assertTrue(engine.is_available())

    def test_partial_failures_do_not_trip_cooldown(self) -> None:
        engine = self.make_dead_engine()
        engine.translate_batch(["Hi"], "s")
        engine.translate_batch(["Hi"], "s")
        self.assertTrue(engine.is_available())

    def test_auth_error_trips_cooldown_on_first_failure(self) -> None:
        """401 重试必然复现：首次失败即冷却，不必凑满 3 次。

        冷却的门在 :class:`ModelPool.translate`（先查 ``is_available``
        再调引擎），池级行为由 TestModelPoolRouting 的同名场景覆盖。
        """
        server = MockChatServer(
            lambda request: (401, json.dumps({"error": {"message": "bad key"}}))
        )
        try:
            engine = make_engine(server)
            self.assertEqual(engine.translate_batch(["Hi"], "s"), [""])
            self.assertFalse(engine.is_available())
            status = engine.status()
            self.assertIn("401", status.message)
        finally:
            server.close()

    def test_bad_request_trips_cooldown_on_first_failure(self) -> None:
        """400（请求被拒，模型名/参数写错最常见）重试必然复现：首次即冷却。

        与 401/403/404 同属确定性失败；池级行为由 TestModelPoolRouting
        的同名场景覆盖。
        """
        server = MockChatServer(
            lambda request: (400, json.dumps({"error": {"message": "unknown model"}}))
        )
        try:
            engine = make_engine(server)
            self.assertEqual(engine.translate_batch(["Hi"], "s"), [""])
            self.assertFalse(engine.is_available())
            status = engine.status()
            self.assertIn("400", status.message)
        finally:
            server.close()

    def test_payment_required_trips_cooldown_on_first_failure(self) -> None:
        """402（余额不足/配额耗尽）重试必然复现：首次即冷却。

        与 400/401 同属确定性失败——欠费账户在分钟尺度内不会自愈，
        逐段重试只会白付往返；池级行为由 TestModelPoolRouting 覆盖。
        """
        server = MockChatServer(
            lambda request: (402, json.dumps({"error": {"message": "insufficient balance"}}))
        )
        try:
            engine = make_engine(server)
            self.assertEqual(engine.translate_batch(["Hi"], "s"), [""])
            self.assertFalse(engine.is_available())
            status = engine.status()
            self.assertIn("402", status.message)
        finally:
            server.close()

    def test_server_error_keeps_normal_failure_counting(self) -> None:
        """5xx/429 可能瞬时：仍按连续失败计数，单次不进冷却。"""
        server = MockChatServer(lambda request: (500, "boom"))
        try:
            engine = make_engine(server)
            engine.translate_batch(["Hi"], "s")
            self.assertTrue(engine.is_available())
        finally:
            server.close()

    def test_cooldown_message_reports_auth_error(self) -> None:
        server = MockChatServer(lambda request: (401, "{}"))
        try:
            engine = make_engine(server)
            engine.translate_batch(["Hi"], "s")
            status = engine.status()
            self.assertFalse(status.available)
            self.assertIn("401", status.message)
        finally:
            server.close()


# ---------------------------------------------------------------------------
# 4. ModelPool 路由与回退
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 3b. 空内容按失败记账
# ---------------------------------------------------------------------------


class TestEmptyContentFailure(unittest.TestCase):
    """空内容响应按失败而非成功记账。

    推理类模型（DeepSeek-R1、Qwen-thinking 等）可能把全部输出额度耗在
    思考上：HTTP 200 但 content 为空。若记成功，冷却永不触发，每个
    片段都要白付一次云端往返再回退本地。
    """

    @staticmethod
    def make_engine_with(responder) -> tuple[ApiEngine, MockChatServer]:
        server = MockChatServer(responder)
        return make_engine(server), server

    def test_empty_content_returns_empty_and_counts_failure(self) -> None:
        engine, server = self.make_engine_with(lambda request: (200, ""))
        try:
            self.assertEqual(engine.translate_batch(["Hello"], "s"), [""])
            # 单次空响应不冷却（与瞬时失败同策略），连续 3 次才冷却
            self.assertTrue(engine.is_available())
        finally:
            engine.unload()
            server.close()

    def test_thinking_only_content_counts_as_failure(self) -> None:
        engine, server = self.make_engine_with(
            lambda request: (200, "<think>思考未完成")
        )
        try:
            self.assertEqual(engine.translate_batch(["Hello"], "s"), [""])
        finally:
            engine.unload()
            server.close()

    def test_three_empty_responses_trip_cooldown(self) -> None:
        engine, server = self.make_engine_with(lambda request: (200, ""))
        try:
            for _ in range(api_engine_module.FAILURE_THRESHOLD):
                engine.translate_batch(["Hello"], "s")
            self.assertFalse(engine.is_available())
        finally:
            engine.unload()
            server.close()

    def test_test_button_reports_empty_reply(self) -> None:
        engine, server = self.make_engine_with(lambda request: (200, ""))
        try:
            ok, message = engine.test()
            self.assertFalse(ok)
            self.assertIn("为空", message)
            for _ in range(api_engine_module.FAILURE_THRESHOLD - 1):
                engine.test()
            self.assertFalse(engine.is_available())
        finally:
            engine.unload()
            server.close()


class _FakeLocalEngine:
    """替身本地引擎：记录调用，返回带前缀的伪译文。"""

    model_id = MODEL_FAST
    busy = False

    def __init__(self) -> None:
        self.calls = 0

    def is_available(self) -> bool:
        return True

    def is_loaded(self) -> bool:
        return True

    def load(self) -> bool:
        return True

    def unload(self) -> None:
        return None

    def warm_up(self) -> None:
        return None

    def translate_batch(self, segments, system_prompt, max_tokens=None):
        self.calls += 1
        if "Translate " in system_prompt:
            # Product service tests require an actual target-script stand-in;
            # raw pool transport tests still use their literal "system" prompt.
            return [fake_translation(segment, "本地:") for segment in segments]
        return [f"本地:{segments[0]}"]

    def status(self) -> EngineStatus:
        return EngineStatus(model_id=self.model_id, available=True, loaded=True)


class TestModelPoolRouting(unittest.TestCase):
    def setUp(self) -> None:
        self.server = MockChatServer()
        self.pool = ModelPool(AppConfig().model)
        self.fake_local = _FakeLocalEngine()
        self.pool._engines[MODEL_FAST] = self.fake_local

    def tearDown(self) -> None:
        self.pool.shutdown()
        self.server.close()

    def test_routes_to_api_when_enabled(self) -> None:
        self.pool.apply_api_config(
            ApiConfig(enabled=True, base_url=self.server.base_url,
                      api_key="k", model_name="m")
        )
        self.assertTrue(self.pool.api_mode_active())
        self.assertEqual(
            self.pool.translate(MODEL_FAST, "Hello", "system"), "译:Hello"
        )
        self.assertEqual(self.fake_local.calls, 0)

    def test_falls_back_to_local_on_api_failure(self) -> None:
        server = MockChatServer(lambda request: (500, "boom"))
        try:
            self.pool.apply_api_config(
                ApiConfig(enabled=True, base_url=server.base_url, model_name="m")
            )
            result = self.pool.translate(MODEL_FAST, "Hello", "system")
            self.assertEqual(result, "本地:Hello")
            self.assertEqual(self.fake_local.calls, 1)
        finally:
            server.close()

    def test_auth_failure_sends_next_frame_straight_to_local(self) -> None:
        """401 首次失败即冷却：下一帧不再触碰在线引擎，直接本地。"""
        server = MockChatServer(lambda request: (401, "bad key"))
        try:
            self.pool.apply_api_config(
                ApiConfig(enabled=True, base_url=server.base_url, model_name="m")
            )
            self.assertEqual(
                self.pool.translate(MODEL_FAST, "Hello", "system"), "本地:Hello"
            )
            self.assertEqual(
                self.pool.translate(MODEL_FAST, "World", "system"), "本地:World"
            )
            self.assertEqual(self.fake_local.calls, 2)
            self.assertEqual(len(server.requests), 1)
        finally:
            server.close()

    def test_bad_request_sends_next_frame_straight_to_local(self) -> None:
        """400（如模型名写错）首次失败即冷却：下一帧不再触碰在线引擎。"""
        server = MockChatServer(lambda request: (400, "unknown model"))
        try:
            self.pool.apply_api_config(
                ApiConfig(enabled=True, base_url=server.base_url, model_name="m")
            )
            self.assertEqual(
                self.pool.translate(MODEL_FAST, "Hello", "system"), "本地:Hello"
            )
            self.assertEqual(
                self.pool.translate(MODEL_FAST, "World", "system"), "本地:World"
            )
            self.assertEqual(self.fake_local.calls, 2)
            self.assertEqual(len(server.requests), 1)
        finally:
            server.close()

    def test_disabled_goes_straight_to_local(self) -> None:
        self.pool.apply_api_config(ApiConfig(enabled=False))
        self.assertFalse(self.pool.api_mode_active())
        self.assertEqual(
            self.pool.translate(MODEL_FAST, "Hello", "system"), "本地:Hello"
        )

    def test_same_config_does_not_rebuild_engine(self) -> None:
        config = ApiConfig(enabled=True, base_url=self.server.base_url,
                           api_key="k", model_name="m")
        self.pool.apply_api_config(config)
        first = self.pool._api_engine
        self.pool.apply_api_config(
            ApiConfig(enabled=True, base_url=self.server.base_url,
                      api_key="k", model_name="m")
        )
        self.assertIs(self.pool._api_engine, first)

    def test_changed_config_rebuilds_engine(self) -> None:
        self.pool.apply_api_config(
            ApiConfig(enabled=True, base_url=self.server.base_url, model_name="a")
        )
        first = self.pool._api_engine
        self.pool.apply_api_config(
            ApiConfig(enabled=True, base_url=self.server.base_url, model_name="b")
        )
        self.assertIsNot(self.pool._api_engine, first)
        self.assertTrue(first._closed)

    def test_connection_check_recovers_live_engine_from_cooldown(self) -> None:
        attempts = {"count": 0}

        def respond(request: dict) -> tuple[int, str]:
            attempts["count"] += 1
            if attempts["count"] == 1:
                return 401, json.dumps({"error": {"message": "bad key"}})
            return 200, "中文译文"

        server = MockChatServer(respond)
        try:
            config = ApiConfig(
                enabled=True, base_url=server.base_url, api_key="new-key", model_name="m"
            )
            self.pool.apply_api_config(config)
            self.assertEqual(
                self.pool.translate(MODEL_FAST, "Hello", "system"), "本地:Hello"
            )
            self.assertFalse(self.pool.api_mode_active())

            ok, message = self.pool.test_api_connection(config)
            self.assertTrue(ok, message)
            self.assertTrue(self.pool.api_mode_active())
            self.assertEqual(
                self.pool.translate(MODEL_FAST, "Again", "system"), "中文译文"
            )
            self.assertEqual(len(server.requests), 3)
            self.assertTrue(all(
                request["headers"]["authorization"] == "Bearer new-key"
                for request in server.requests
            ))
        finally:
            server.close()

    def test_connection_check_before_enabling_uses_temporary_engine(self) -> None:
        config = ApiConfig(
            enabled=False, base_url=self.server.base_url,
            api_key="preview-key", model_name="m",
        )
        ok, message = self.pool.test_api_connection(config)
        self.assertTrue(ok, message)
        self.assertIsNone(self.pool.api_status())
        self.assertEqual(len(self.server.requests), 1)
        self.assertEqual(
            self.server.requests[0]["headers"]["authorization"], "Bearer preview-key"
        )

    def test_api_status_exposed(self) -> None:
        self.assertIsNone(self.pool.api_status())
        self.pool.apply_api_config(
            ApiConfig(enabled=True, base_url=self.server.base_url, model_name="m")
        )
        self.assertEqual(self.pool.api_status().message, "m")

    def test_existing_local_cache_does_not_hide_new_api_key(self) -> None:
        """Enabling or changing a key must send a new request to that provider."""
        self.server._responder = lambda request: (200, "云端译文")
        config = AppConfig()
        config.translation.semantic_review = False  # legacy single-pass contract
        config.schedule.mode = "manual"
        config.schedule.active_model = MODEL_FAST
        config.translation.source_language = "en"
        config.translation.target_language = "zh"
        service = TranslationService(
            config=config,
            model_pool=self.pool,
            scheduler=ModelScheduler(config.schedule),
            cache=TranslationCache(enabled=True),
            preprocessor=TextPreprocessor.from_config(config),
            glossary=Glossary(),
            history=None,
        )
        self.assertEqual(service.translate_text("Hello world"), fake_translation("Hello world", "本地:"))

        config.api = ApiConfig(
            enabled=True, base_url=self.server.base_url, api_key="first-key", model_name="m"
        )
        self.pool.apply_api_config(config.api)
        service.apply_config(config)
        self.assertEqual(service.translate_text("Hello world"), "云端译文")
        self.assertEqual(len(self.server.requests), 1)
        self.assertEqual(self.server.requests[-1]["headers"]["authorization"], "Bearer first-key")

        # A successful API translation may be reused until the provider changes.
        self.assertEqual(service.translate_text("Hello world"), "云端译文")
        self.assertEqual(len(self.server.requests), 1)
        config.api.api_key = "second-key"
        self.pool.apply_api_config(config.api)
        service.apply_config(config)
        self.assertEqual(service.translate_text("Hello world"), "云端译文")
        self.assertEqual(len(self.server.requests), 2)
        self.assertEqual(self.server.requests[-1]["headers"]["authorization"], "Bearer second-key")

    def test_failed_api_fallback_is_not_cached_as_online_translation(self) -> None:
        attempts = {"count": 0}

        def respond(request: dict) -> tuple[int, str]:
            attempts["count"] += 1
            if attempts["count"] == 1:
                return 500, "temporary error"
            return 200, "云端译文"

        server = MockChatServer(respond)
        try:
            config = AppConfig()
            config.translation.semantic_review = False  # legacy single-pass contract
            config.schedule.mode = "manual"
            config.schedule.active_model = MODEL_FAST
            config.translation.source_language = "en"
            config.translation.target_language = "zh"
            config.api = ApiConfig(enabled=True, base_url=server.base_url, model_name="m")
            self.pool.apply_api_config(config.api)
            service = TranslationService(
                config=config,
                model_pool=self.pool,
                scheduler=ModelScheduler(config.schedule),
                cache=TranslationCache(enabled=True),
                preprocessor=TextPreprocessor.from_config(config),
                glossary=Glossary(),
                history=None,
            )
            self.assertEqual(service.translate_text("Hello world"), fake_translation("Hello world", "本地:"))
            self.assertEqual(service.translate_text("Hello world"), "云端译文")
            self.assertEqual(attempts["count"], 2)
            self.assertEqual(service.translate_text("Hello world"), "云端译文")
            self.assertEqual(attempts["count"], 2)
        finally:
            server.close()

    def test_online_frame_skips_local_prefetch_cache(self) -> None:
        self.server._responder = lambda request: (200, "云端译文")
        config = AppConfig()
        config.translation.semantic_review = False  # legacy single-pass contract
        config.schedule.mode = "manual"
        config.schedule.active_model = MODEL_FAST
        config.translation.source_language = "en"
        config.translation.target_language = "zh"
        config.api = ApiConfig(enabled=True, base_url=self.server.base_url, model_name="m")
        self.pool.apply_api_config(config.api)
        service = make_service(self.pool, config)
        lookup = mock.Mock()
        lookup.lookup.return_value = "本地预译"
        service.set_prefetch_lookup(lookup)

        render = service.translate_frame(make_frame(block("Hello world", 10, 10)))
        self.assertEqual(render.items[0].text, "云端译文")
        self.assertEqual(len(self.server.requests), 1)
        lookup.lookup.assert_not_called()


# ---------------------------------------------------------------------------
# 5. 翻译服务在线模式（并发批量）
# ---------------------------------------------------------------------------


class _ApiFakePool:
    """带在线模式标记的假池：沿用 FakeModelPool 的行为约定。"""

    def __init__(self, prefix: str = "译:") -> None:
        self.prefix = prefix
        self.calls: list[tuple[str, str]] = []
        self.api_active = False

    def api_mode_active(self) -> bool:
        return self.api_active

    def translate(self, model_id, prompt, system_prompt, max_tokens=None) -> str:
        self.calls.append((model_id, prompt))
        parts = prompt.split(SEGMENT_DELIMITER)
        return SEGMENT_DELIMITER.join(fake_translation(part.strip(), self.prefix) for part in parts)

    def is_available(self, model_id: str) -> bool:
        return True

    def any_available(self) -> bool:
        return True


def make_service(pool, config):
    return TranslationService(
        config=config,
        model_pool=pool,
        scheduler=ModelScheduler(config.schedule),
        cache=TranslationCache(enabled=False),
        preprocessor=TextPreprocessor.from_config(config),
        glossary=Glossary(),
        history=None,
    )


class TestServiceApiMode(unittest.TestCase):
    def test_concurrent_batches_preserve_order(self) -> None:
        config = AppConfig()
        config.translation.semantic_review = False  # legacy single-pass contract
        config.api.enabled = True
        config.api.batch_size = 5
        config.api.max_concurrency = 3
        pool = _ApiFakePool()
        pool.api_active = True
        service = make_service(pool, config)

        texts = [f"Source sentence {i}" for i in range(12)]
        # 垂直间距足够大，预处理不会把相邻块合并成同一段落
        frame = make_frame(
            *[block(text, 10, index * 80) for index, text in enumerate(texts)]
        )
        render = service.translate_frame(frame)

        # 12 段 / 每批 5 → 3 批并发（本地批量 6 与在线批量 5 互不干扰）
        self.assertEqual(len(pool.calls), 3)
        self.assertEqual(len(render.items), 12)
        for item, source in zip(render.items, texts, strict=True):
            self.assertEqual(item.source_text, source)
            self.assertEqual(item.text, fake_translation(source))

    def test_api_disabled_uses_local_batching(self) -> None:
        config = AppConfig()
        config.translation.semantic_review = False  # legacy single-pass contract
        pool = _ApiFakePool()
        pool.api_active = False
        service = make_service(pool, config)

        texts = [f"Source sentence {i}" for i in range(12)]
        frame = make_frame(
            *[block(text, 10, index * 80) for index, text in enumerate(texts)]
        )
        render = service.translate_frame(frame)

        # 本地批量大小 6 → 2 批串行
        self.assertEqual(len(pool.calls), 2)
        self.assertEqual(len(render.items), 12)

    def test_api_mode_raises_output_cap(self) -> None:
        """在线模式下输出上限跟随在线配置：长段落不被本地 384 上限截断。"""
        config = AppConfig()
        config.translation.semantic_review = False  # legacy single-pass contract
        config.api.enabled = True
        config.api.max_tokens = 1024
        pool = _ApiFakePool()
        pool.api_active = True
        seen: dict[str, int | None] = {}

        def record(model_id, prompt, system_prompt, max_tokens=None):
            seen["max_tokens"] = max_tokens
            return "译:好"

        pool.translate = record
        service = make_service(pool, config)
        # 200 个汉字 ≈ 200 token，动态上限 2.5×200+32=532；
        # 本地 fast 模型上限 384 会被 min() 压住，在线配置 1024 则放行 532
        service.translate_text("好" * 200)
        self.assertEqual(seen["max_tokens"], 532)

    def test_concurrent_path_with_broken_batch_keeps_source(self) -> None:
        """并发路径下未恢复片段保持原屏可见，其余译文正确发布。"""
        config = AppConfig()
        config.translation.semantic_review = False  # legacy single-pass contract
        config.api.enabled = True
        config.api.batch_size = 5
        pool = _ApiFakePool()
        pool.api_active = True
        service = make_service(pool, config)

        original_translate = pool.translate
        state = {"calls": 0}

        def flaky(model_id, prompt, system_prompt, max_tokens=None):
            state["calls"] += 1
            if state["calls"] == 2:
                return ""  # 第二批失败
            return original_translate(model_id, prompt, system_prompt, max_tokens)

        pool.translate = flaky
        texts = [f"Source sentence {i}" for i in range(12)]
        frame = make_frame(
            *[block(text, 10, index * 80) for index, text in enumerate(texts)]
        )
        render = service.translate_frame(frame)

        self.assertTrue(render.items)
        # Failed translations do not duplicate the source on the overlay. The
        # unchanged screen remains visible at those positions.
        for item in render.items:
            self.assertEqual(item.text, fake_translation(item.source_text))
            self.assertNotEqual(item.text, item.source_text)
        self.assertLess(len(render.items), len(texts))
        rendered_sources = {item.source_text for item in render.items}
        self.assertTrue(rendered_sources.issubset(set(texts)))


# ---------------------------------------------------------------------------
# 6. 配置兼容
# ---------------------------------------------------------------------------


class TestApiConfigCompat(unittest.TestCase):
    def test_legacy_config_without_api_section(self) -> None:
        raw = to_dict(AppConfig())
        raw.pop("api")
        config = from_dict(AppConfig, raw)
        self.assertEqual(config.api, ApiConfig())
        self.assertFalse(config.api.enabled)

    def test_round_trip_preserves_api(self) -> None:
        config = AppConfig()
        config.translation.semantic_review = False  # legacy single-pass contract
        config.api.enabled = True
        config.api.base_url = "https://api.example.com/v1"
        config.api.model_name = "test-model"
        restored = from_dict(AppConfig, to_dict(config))
        self.assertEqual(restored.api, config.api)

    def test_settings_sanitize_disables_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = SettingsManager(Path(tmp) / "config.json")
            settings.update(
                {"api": {"enabled": True, "base_url": "", "model_name": ""}}
            )
            self.assertFalse(settings.config.api.enabled)

    def test_settings_sanitize_normalizes_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = SettingsManager(Path(tmp) / "config.json")
            settings.update({"api": {"base_url": "https://api.example.com/v1/"}})
            self.assertEqual(settings.config.api.base_url, "https://api.example.com/v1")

    def test_settings_sanitize_clamps_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = SettingsManager(Path(tmp) / "config.json")
            settings.update(
                {
                    "api": {
                        "batch_size": 999,
                        "max_concurrency": 0,
                        "timeout_seconds": 0.01,
                    }
                }
            )
            api = settings.config.api
            self.assertEqual(api.batch_size, 24)
            self.assertEqual(api.max_concurrency, 1)
            self.assertEqual(api.timeout_seconds, 1.0)


# ---------------------------------------------------------------------------
# 7. 设置面板冒烟
# ---------------------------------------------------------------------------


class TestApiPanel(unittest.TestCase):
    """设置面板在线 AI 卡的持久化与预设行为。"""

    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])
        from pingyitong.ui.design.theme import build_theme
        from pingyitong.ui.main_panel import MainPanel

        cls._tmp = tempfile.TemporaryDirectory()
        settings = SettingsManager(Path(cls._tmp.name) / "config.json")
        controller = SimpleNamespace(
            theme=build_theme(),
            settings=settings,
            notify=lambda *args, **kwargs: None,
        )
        cls.settings = settings
        cls.panel = MainPanel(controller)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.panel.close()
        cls._tmp.cleanup()

    def test_fields_persist_to_config(self) -> None:
        self.panel._api_base_url.setText("https://api.example.com/v1")
        self.panel._api_model.setText("test-model")
        self.panel._persist_api()
        api = self.settings.config.api
        self.assertEqual(api.base_url, "https://api.example.com/v1")
        self.assertEqual(api.model_name, "test-model")

    def test_provider_preset_fills_fields(self) -> None:
        self.panel._api_provider.setCurrentIndex(1)  # DeepSeek
        api = self.settings.config.api
        self.assertEqual(api.base_url, "https://api.deepseek.com/v1")
        self.assertEqual(api.model_name, "deepseek-chat")

    def test_incomplete_enable_falls_back(self) -> None:
        self.panel._api_base_url.setText("")
        self.panel._api_model.setText("")
        self.panel._api_switch.setChecked(True)
        # 配置层 sanitize 强制回落，界面开关同步回未勾选
        self.assertFalse(self.settings.config.api.enabled)
        self.assertFalse(self.panel._api_switch.isChecked())

    def test_key_is_password_masked(self) -> None:
        self.assertEqual(
            self.panel._api_key.echoMode(),
            self.panel._api_key.EchoMode.Password,
        )


class TestAdaptiveParamRetry(unittest.TestCase):
    """400 参数名自适应重试：OpenAI 推理系怪癖自愈。

    o 系列/gpt-5 等模型拒绝 ``max_tokens``（提示改用
    ``max_completion_tokens``），部分模型只接受默认温度。这类 400
    属确定性失败，按原样重试只会进冷却——"填了新模型名"会被误报成
    "配置或认证错误"。引擎从错误信息里学到正确参数后当次请求内
    重试，之后直接按正确参数发出，不再触发该 400。
    """

    @staticmethod
    def _hint_400(hint: str) -> tuple[int, str]:
        return (
            400,
            json.dumps({"error": {"message": f"Unsupported parameter. Use {hint} instead."}}),
        )

    def test_400_hint_switches_to_completion_tokens(self) -> None:
        """400 点名 max_completion_tokens：当次请求内改参数重试并记住。"""
        seen_keys: list[str] = []

        def responder(request: dict) -> tuple[int, str]:
            payload = request["payload"]
            if "max_tokens" in payload:
                seen_keys.append("max_tokens")
                return self._hint_400("'max_completion_tokens'")
            seen_keys.append("max_completion_tokens")
            return 200, "译:Hello"

        server = MockChatServer(responder)
        try:
            engine = make_engine(server)
            self.assertEqual(engine.translate_batch(["Hi"], "s"), ["译:Hello"])
            self.assertEqual(seen_keys, ["max_tokens", "max_completion_tokens"])
            # 第二批直接用学到的参数名，不再触发 400
            self.assertEqual(engine.translate_batch(["Hi"], "s"), ["译:Hello"])
            self.assertEqual(
                seen_keys,
                ["max_tokens", "max_completion_tokens", "max_completion_tokens"],
            )
            self.assertTrue(engine.is_available())
        finally:
            server.close()

    def test_400_hint_omits_temperature(self) -> None:
        """400 点名 temperature：移除该参数重试并记住。"""
        seen_temp: list[bool] = []

        def responder(request: dict) -> tuple[int, str]:
            has_temp = "temperature" in request["payload"]
            seen_temp.append(has_temp)
            if has_temp:
                return self._hint_400("default 'temperature'")
            return 200, "译:Hello"

        server = MockChatServer(responder)
        try:
            engine = make_engine(server)
            self.assertEqual(engine.translate_batch(["Hi"], "s"), ["译:Hello"])
            self.assertEqual(seen_temp, [True, False])
            self.assertEqual(engine.translate_batch(["Hi"], "s"), ["译:Hello"])
            self.assertEqual(seen_temp, [True, False, False])
        finally:
            server.close()

    def test_both_quirks_learned_within_one_request(self) -> None:
        """两家怪癖同现（o 系/gpt-5）：同一次请求内两连自适应，不进冷却。

        只重试一次的旧实现会把第二处怪癖（temperature）留到下一次
        冷却期满才学到，首个请求必然失败并白白冷却 60 秒。
        """
        seen: list[str] = []

        def responder(request: dict) -> tuple[int, str]:
            payload = request["payload"]
            if "max_tokens" in payload:
                seen.append("max_tokens")
                return self._hint_400("'max_completion_tokens'")
            if "temperature" in payload:
                seen.append("temperature")
                return self._hint_400("default 'temperature'")
            seen.append("ok")
            return 200, "译:Hello"

        server = MockChatServer(responder)
        try:
            engine = make_engine(server)
            self.assertEqual(engine.translate_batch(["Hi"], "s"), ["译:Hello"])
            self.assertEqual(seen, ["max_tokens", "temperature", "ok"])
            # 两处怪癖均已学到：第二批一次成功且不再触发 400
            self.assertEqual(engine.translate_batch(["Hi"], "s"), ["译:Hello"])
            self.assertEqual(seen, ["max_tokens", "temperature", "ok", "ok"])
            self.assertTrue(engine.is_available())
        finally:
            server.close()

    def test_adaptation_is_bounded(self) -> None:
        """自适应上限 2 次：之后仍 400 则抛出照常冷却，不无限重试。"""
        calls: list[int] = []

        def responder(request: dict) -> tuple[int, str]:
            calls.append(1)
            # 永远点名 max_completion_tokens：第一次学到后提示仍在，
            # 第二次自适应因标志已置而无事可做，必须到此为止
            return self._hint_400("'max_completion_tokens'")

        server = MockChatServer(responder)
        try:
            engine = make_engine(server)
            self.assertEqual(engine.translate_batch(["Hi"], "s"), [""])
            self.assertEqual(len(calls), 2)
            self.assertFalse(engine.is_available())
            self.assertIn("400", engine.status().message)
        finally:
            server.close()

    def test_unrelated_400_no_retry_and_cooldowns(self) -> None:
        """与参数名无关的 400（如上下文超长）不重试，照常进冷却。"""
        calls: list[int] = []

        def responder(request: dict) -> tuple[int, str]:
            calls.append(1)
            return (
                400,
                json.dumps(
                    {"error": {"message": "maximum context length is 8192 tokens"}}
                ),
            )

        server = MockChatServer(responder)
        try:
            engine = make_engine(server)
            self.assertEqual(engine.translate_batch(["Hi"], "s"), [""])
            self.assertEqual(len(calls), 1)
            self.assertFalse(engine.is_available())
            self.assertIn("400", engine.status().message)
        finally:
            server.close()

    def test_successful_test_clears_cooldown(self) -> None:
        """"测试连接"成功即解除冷却，不必等剩余冷却期走完。"""
        state = {"fail": True}

        def responder(request: dict) -> tuple[int, str]:
            if state["fail"]:
                return (500, json.dumps({"error": {"message": "upstream error"}}))
            return 200, "OK"

        server = MockChatServer(responder)
        try:
            engine = make_engine(server)
            for _ in range(3):
                self.assertEqual(engine.translate_batch(["Hi"], "s"), [""])
            self.assertFalse(engine.is_available())
            state["fail"] = False
            ok, message = engine.test()
            self.assertTrue(ok, message)
            self.assertTrue(engine.is_available())
        finally:
            server.close()


# ---------------------------------------------------------------------------
# 7. 预连接与陈旧连接检出（首帧延迟优化）
# ---------------------------------------------------------------------------


class TestPreconnect(unittest.TestCase):
    """预连接：首帧免付 TCP 握手；纯连接不产生请求、不消耗配额。"""

    def setUp(self) -> None:
        self.server = MockChatServer()
        self.engine = make_engine(self.server)

    def tearDown(self) -> None:
        self.engine.unload()
        self.server.close()

    def test_preconnect_reused_by_first_request(self) -> None:
        self.engine.preconnect()
        target = self.engine._preconnect_target
        # 预连接数量与批并发对齐：整帧多批并发时每条都要独立握手
        self.assertGreaterEqual(target, 1)
        # 服务端 accept 到 handler 线程计数存在调度窗口，轮询等待落地
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and self.server.connections < target:
            time.sleep(0.02)
        self.assertEqual(self.server.connections, target)
        self.assertEqual(len(self.engine._idle_connections), target)
        # 纯连接无 HTTP 请求：配额零消耗
        self.assertEqual(len(self.server.requests), 0)
        results = self.engine.translate_batch(["Hello"], "s")
        self.assertEqual(results, ["译:Hello"])
        # 首个请求直接复用预连接，未新建连接
        self.assertEqual(self.server.connections, target)
        self.assertEqual(len(self.server.requests), 1)

    def test_preconnect_failure_is_silent(self) -> None:
        """端口无人监听时预连接不得抛出，也不得污染连接池。"""
        engine = make_engine(
            self.server, base_url=f"http://127.0.0.1:{free_port()}/v1"
        )
        try:
            engine.preconnect()
            self.assertEqual(engine._idle_connections, [])
        finally:
            engine.unload()

    def test_apply_api_config_preconnects_in_background(self) -> None:
        """启用在线 AI 即后台预连接：等待落地后首请求零新建连接。"""
        pool = ModelPool(AppConfig().model)
        pool._engines[MODEL_FAST] = _FakeLocalEngine()
        try:
            pool.apply_api_config(
                ApiConfig(enabled=True, base_url=self.server.base_url,
                          api_key="k", model_name="m")
            )
            engine = pool._api_engine
            target = engine._preconnect_target
            # 等预连接线程完成入池（服务端 accept 计数早于客户端入池，
            # 轮询服务端计数会与首请求竞态）
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                with engine._pool_lock:
                    if len(engine._idle_connections) >= target:
                        break
                time.sleep(0.02)
            with engine._pool_lock:
                self.assertEqual(len(engine._idle_connections), target)
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and self.server.connections < target:
                time.sleep(0.02)
            self.assertEqual(self.server.connections, target)
            self.assertEqual(len(self.server.requests), 0)
            self.assertEqual(pool.translate(MODEL_FAST, "Hi", "s"), "译:Hi")
            self.assertEqual(len(self.server.requests), 1)
            self.assertEqual(self.server.connections, target)
        finally:
            pool.shutdown()


class TestTruncationGuard(unittest.TestCase):
    """``finish_reason=length`` 截断防护：翻倍预算重问，仍截断按失败回退。"""

    @staticmethod
    def _body(content: str, finish_reason: str) -> dict:
        return {
            "choices": [
                {"message": {"content": content}, "finish_reason": finish_reason}
            ]
        }

    def test_length_truncation_retries_with_double_budget(self) -> None:
        seen: list[int] = []

        def responder(request: dict) -> tuple[int, object]:
            seen.append(int(request["payload"].get("max_tokens", 0)))
            if len(seen) == 1:
                return 200, self._body("半截译", "length")
            return 200, self._body("完整译文", "stop")

        server = MockChatServer(responder)
        try:
            engine = make_engine(server)
            try:
                self.assertEqual(
                    engine.translate_batch(["Hello"], "s", max_tokens=100),
                    ["完整译文"],
                )
                self.assertEqual(seen, [100, 200])
                # 两次请求本身都成功，截断重试不得额外记失败
                self.assertTrue(engine.is_available())
                self.assertEqual(engine.status().total_calls, 1)
            finally:
                engine.unload()
        finally:
            server.close()

    def test_persistent_truncation_returns_failure(self) -> None:
        server = MockChatServer(
            lambda request: (200, self._body("半截译", "length"))
        )
        try:
            engine = make_engine(server)
            try:
                before = engine.failure_generation
                self.assertEqual(
                    engine.translate_batch(["Hello"], "s", max_tokens=100), [""]
                )
                # 半截译文绝不上屏：按失败记账，由调用方回退本地模型
                self.assertGreater(engine.failure_generation, before)
            finally:
                engine.unload()
        finally:
            server.close()

    def test_empty_content_with_length_finish_retries(self) -> None:
        """推理模型耗尽预算时 content 为空 + length：翻倍预算同样要救。"""
        seen: list[int] = []

        def responder(request: dict) -> tuple[int, object]:
            seen.append(int(request["payload"].get("max_tokens", 0)))
            if len(seen) == 1:
                return 200, self._body("", "length")
            return 200, self._body("完整译文", "stop")

        server = MockChatServer(responder)
        try:
            engine = make_engine(server)
            try:
                self.assertEqual(
                    engine.translate_batch(["Hello"], "s", max_tokens=100),
                    ["完整译文"],
                )
                self.assertEqual(seen, [100, 200])
                self.assertTrue(engine.is_available())
            finally:
                engine.unload()
        finally:
            server.close()

    def test_truncation_at_hard_ceiling_skips_retry(self) -> None:
        seen: list[int] = []

        def responder(request: dict) -> tuple[int, object]:
            seen.append(int(request["payload"].get("max_tokens", 0)))
            return 200, self._body("半截译", "length")

        server = MockChatServer(responder)
        try:
            engine = make_engine(server)
            try:
                self.assertEqual(
                    engine.translate_batch(
                        ["Hello"], "s", max_tokens=MAX_OUTPUT_TOKENS
                    ),
                    [""],
                )
                # 已在硬顶，翻倍必越界（服务商会 400）：只发一次
                self.assertEqual(seen, [MAX_OUTPUT_TOKENS])
            finally:
                engine.unload()
        finally:
            server.close()


class TestContentExtraction(unittest.TestCase):
    """响应正文提取的兼容形态。"""

    def test_list_content_parts_are_joined(self) -> None:
        """部分网关把 content 返回为分片数组，拼接后正常使用、不误触冷却。"""

        def responder(request: dict) -> tuple[int, object]:
            return 200, {
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "text", "text": "第一段，"},
                                {"type": "text", "text": "第二段。"},
                            ]
                        },
                        "finish_reason": "stop",
                    }
                ]
            }

        server = MockChatServer(responder)
        try:
            engine = make_engine(server)
            try:
                self.assertEqual(
                    engine.translate_batch(["Hello"], "s"), ["第一段，第二段。"]
                )
                self.assertTrue(engine.is_available())
            finally:
                engine.unload()
        finally:
            server.close()


class _FakePooledConn:
    """最小连接替身：检出/取用/丢弃路径只触 ``.sock`` 与 ``close()``。"""

    def __init__(self) -> None:
        left, right = socket.socketpair()
        self.sock = left
        self._peer = right

    def close(self) -> None:
        for sock in (self.sock, self._peer):
            with contextlib.suppress(OSError):
                sock.close()


class TestStaleCheckout(unittest.TestCase):
    """取用前零开销检出服务端已关闭的陈旧连接，省一次失败重试往返。"""

    @staticmethod
    def _engine_with(*conns) -> ApiEngine:
        engine = ApiEngine(
            ApiConfig(enabled=True, base_url="http://mock.local/v1", model_name="m")
        )
        with engine._pool_lock:
            engine._idle_connections.extend(conns)
        return engine

    def test_dropped_connection_skipped(self) -> None:
        alive = _FakePooledConn()
        dead = _FakePooledConn()
        dead._peer.close()  # 对端关闭 → EOF 可读 → 判定陈旧
        # _checkout 从栈顶弹出：dead 在前，检出后跳过再取 alive
        engine = self._engine_with(alive, dead)
        self.assertIs(engine._checkout(), alive)

    def test_healthy_connection_returned(self) -> None:
        conn = _FakePooledConn()
        engine = self._engine_with(conn)
        self.assertIs(engine._checkout(), conn)

    def test_all_dropped_returns_none(self) -> None:
        dead = _FakePooledConn()
        dead._peer.close()
        engine = self._engine_with(dead)
        self.assertIsNone(engine._checkout())


class TestSslContextReuse(unittest.TestCase):
    """TLS 上下文全引擎复用：创建一次要加载系统 CA（实测 ~10ms/次），
    不得退化为每条新连接现建。"""

    def test_https_engine_shares_one_context(self) -> None:
        engine = ApiEngine(
            ApiConfig(
                enabled=True, base_url="https://mock.local/v1", model_name="m"
            )
        )
        self.assertIsNotNone(engine._ssl_context)
        first = engine._new_connection()
        second = engine._new_connection()
        self.assertIs(first._context, engine._ssl_context)
        self.assertIs(second._context, engine._ssl_context)

    def test_http_engine_has_no_context(self) -> None:
        engine = ApiEngine(
            ApiConfig(enabled=True, base_url="http://mock.local/v1", model_name="m")
        )
        self.assertIsNone(engine._ssl_context)


# ---------------------------------------------------------------------------
# 8. 逐条降级的并发化（在线模式）
# ---------------------------------------------------------------------------


class TestParallelIndividualRetry(unittest.TestCase):
    """批量对位失败后的逐条重试：在线模式并发压缩往返，本地合并批保持串行。

    本地纯文本多段帧默认拆成并发单段请求（走不到对位失败路径）；
    受约束的合并批（精简/对照/术语）仍会进入"对位失败 → 逐条重试"，
    且必须串行——本地引擎的槽位之外不应再叠加请求级并发。
    """

    @staticmethod
    def _frame(count: int):
        texts = [f"Source sentence {i}" for i in range(count)]
        frame = make_frame(
            *[block(text, 10, index * 80) for index, text in enumerate(texts)]
        )
        return frame, texts

    @staticmethod
    def _slow_misaligned_pool(pool: _ApiFakePool, state: dict) -> None:
        """批量请求返回错位输出；单条请求正常回译，并记录并发峰值。"""
        lock = threading.Lock()

        def translate(model_id, prompt, system_prompt, max_tokens=None):
            with lock:
                state["current"] += 1
                state["max"] = max(state["max"], state["current"])
            time.sleep(0.3)  # 模拟一次网络往返
            with lock:
                state["current"] -= 1
            if SEGMENT_DELIMITER in prompt:
                return "对位失败输出"  # 段数不匹配 → 触发逐条降级
            return fake_translation(prompt)

        pool.translate = translate

    def test_api_mode_retries_concurrently(self) -> None:
        config = AppConfig()
        config.translation.semantic_review = False  # legacy single-pass contract
        config.api.enabled = True
        config.api.max_concurrency = 4
        pool = _ApiFakePool()
        pool.api_active = True
        state = {"current": 0, "max": 0}
        self._slow_misaligned_pool(pool, state)
        service = make_service(pool, config)
        frame, texts = self._frame(4)

        started = time.perf_counter()
        render = service.translate_frame(frame)
        elapsed = time.perf_counter() - started

        # 4 段单批：1 次批量失败 + 4 条并发重试 ≈ 2×往返；串行需 5×0.3=1.5s
        self.assertLess(elapsed, 1.0)
        self.assertGreaterEqual(state["max"], 2)
        self.assertEqual(len(render.items), 4)
        for item, source in zip(render.items, texts, strict=True):
            self.assertEqual(item.text, fake_translation(source))

    def test_local_mode_retries_serially(self) -> None:
        """本地引擎的批量失败逐条重试保持串行语义：槽位之外不再叠加并发。"""
        config = AppConfig()
        config.translation.semantic_review = False  # legacy single-pass contract
        pool = _ApiFakePool()
        pool.api_active = False
        state = {"current": 0, "max": 0}
        self._slow_misaligned_pool(pool, state)
        service = make_service(pool, config)
        frame, texts = self._frame(4)

        render = service.translate_frame(frame)

        self.assertEqual(state["max"], 1)
        self.assertEqual(len(render.items), 4)
        for item, source in zip(render.items, texts, strict=True):
            self.assertEqual(item.text, fake_translation(source))


if __name__ == "__main__":
    unittest.main()
