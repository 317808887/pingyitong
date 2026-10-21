"""端到端验证：在线 AI（OpenAI 兼容）路径与本地自动回退。

验证四件事，全部走**产品真实装配**（ModelPool 路由 + TranslationService）：

1. **单段**：翻译路由到本地 mock OpenAI 服务并返回预期结果；
2. **批量与并发**：12 段帧按在线批量大小拆 3 批并发发送，
   译文与原文严格对位；
3. **连接复用**：预连接完成后，连续两次请求零新建 TCP 连接
   （在预连接基线之上不增长，复用连接池里的现成连接）
   （省去每次 TCP+TLS 握手的 100–300ms，是在线模式延迟优化的核心）；
4. **自动回退**：把路由指向不可达端口，翻译仍能由本地模型完成
   （本机需有 llama-server 运行时，脚本会自动拉起）。

默认全部使用本地 mock 服务，不发真实外网请求。若要实测真实服务商：
    python scripts/verify_api_translation.py --real \
        --base-url https://api.deepseek.com/v1 --api-key sk-xxx --model deepseek-chat
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def mock_translation(text: str) -> str:
    """Return real target-language fixtures while retaining segment boundaries."""
    from pingyitong.constants import SEGMENT_DELIMITER

    fixed = {
        "Hello world, this is a test.": "你好世界，这是一个测试。",
        "Second request.": "第二次请求。",
    }
    outputs = []
    for part in text.split(SEGMENT_DELIMITER):
        part = part.strip()
        match = re.fullmatch(r"Source sentence number (\d+) for translation\.", part)
        outputs.append(f"待翻译的第 {match[1]} 个句子。" if match else fixed.get(part, "模拟服务未配置此句。"))
    return f"\n{SEGMENT_DELIMITER}\n".join(outputs)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self) -> None:
        self.server.connections += 1
        super().setup()

    def do_POST(self) -> None:  # noqa: N802 - http.server 命名约定
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        user = payload["messages"][1]["content"]
        content = mock_translation(user)
        data = json.dumps(
            {"choices": [{"message": {"content": content}}]}, ensure_ascii=False
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args) -> None:
        pass


def start_mock_server() -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.connections = 0
    threading.Thread(target=server.serve_forever, name="MockApi", daemon=True).start()
    return server


def build_translator(config):
    from pingyitong.components.llm import ModelPool, TranslationCache
    from pingyitong.core import ModelScheduler
    from pingyitong.services.glossary_service import Glossary
    from pingyitong.services.text_preprocess import TextPreprocessor
    from pingyitong.services.translation_service import TranslationService

    pool = ModelPool(config.model, idle_timeout=600)
    pool.apply_api_config(config.api)
    translator = TranslationService(
        config=config,
        model_pool=pool,
        scheduler=ModelScheduler(config.schedule),
        cache=TranslationCache(enabled=False),
        preprocessor=TextPreprocessor.from_config(config),
        glossary=Glossary(),
        history=None,
    )
    return pool, translator


def run_frame(translator, texts: list[str], y_step: float = 80):
    from pingyitong.components.capture.geometry import Rect
    from pingyitong.components.ocr import RecognizedFrame, TextBlock

    blocks = [
        TextBlock(
            text=text,
            box=Rect(10, index * y_step + 10, 300, 16),
            confidence=0.95,
        )
        for index, text in enumerate(texts)
    ]
    frame = RecognizedFrame(
        seq=1, origin=Rect(0, 0, 1920, 1080), blocks=blocks, is_full_frame=True
    )
    return translator.translate_frame(frame)


def main() -> int:
    parser = argparse.ArgumentParser(description="验证在线 AI 路径与本地回退")
    parser.add_argument("--real", action="store_true", help="用真实服务商实测单段翻译")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default="")
    args = parser.parse_args()

    from pingyitong.config.schema import ApiConfig, AppConfig
    from pingyitong.constants import MODEL_FAST

    config = AppConfig()
    config.schedule.mode = "manual"
    config.schedule.active_model = MODEL_FAST
    config.translation.source_language = "auto"
    config.translation.target_language = "zh"

    mock_server = None
    if args.real:
        if not (args.base_url and args.model):
            print("!! --real 需要 --base-url 与 --model")
            return 2
        config.api = ApiConfig(
            enabled=True,
            base_url=args.base_url,
            api_key=args.api_key,
            model_name=args.model,
        )
        print(f"[真实服务] {args.base_url} · {args.model}")
    else:
        mock_server = start_mock_server()
        config.api = ApiConfig(
            enabled=True,
            base_url=f"http://127.0.0.1:{mock_server.server_address[1]}/v1",
            model_name="mock-translator",
        )

    pool, translator = build_translator(config)
    failures = 0

    if not args.real:
        # apply_api_config 会后台并行预建连接（数量与 max_concurrency 对齐，
        # 首帧整批免握手）。预连接完成前不能开始计数——否则预连接线程与
        # 请求竞态会多出连接。等全部预连接入池（或超时）后取基线：
        # 之后两次请求若零新建连接，即证明复用生效。
        engine = pool._api_engine
        target = getattr(engine, "_preconnect_target", 1)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            with engine._pool_lock:
                if len(engine._idle_connections) >= target:
                    break
            time.sleep(0.02)
        # 再等一拍让并行预连接的 accept 全部落账，基线才稳定
        time.sleep(0.2)
        baseline_connections = mock_server.connections

    print("=" * 72)

    # 1) 单段
    started = time.perf_counter()
    result = translator.translate_text("Hello world, this is a test.")
    single_ms = (time.perf_counter() - started) * 1000
    print(f"[单段]  {result!r}  ({single_ms:.0f}ms)")
    if args.real:
        if not result.strip():
            print("  !! 单段翻译为空")
            failures += 1
    elif result != mock_translation("Hello world, this is a test."):
        print("  !! 未得到 mock 服务的预期中文译文")
        failures += 1

    if not args.real:
        # 2) 连接复用：两次请求不再新建 TCP 连接（预连接基线之上零增长）
        translator.translate_text("Second request.")
        new_connections = mock_server.connections - baseline_connections
        print(
            f"[连接复用] 预连接 {baseline_connections} 条 · "
            f"两次请求新建 {new_connections} 条"
        )
        if new_connections != 0:
            print("  !! 连接未被复用（延迟优化失效）")
            failures += 1

        # 3) 批量 + 并发：12 段 → 3 批（batch_size=5）并发
        texts = [f"Source sentence number {i} for translation." for i in range(12)]
        started = time.perf_counter()
        render = run_frame(translator, texts)
        frame_ms = (time.perf_counter() - started) * 1000
        translated = [item.text for item in render.items]
        aligned = translated == [mock_translation(text) for text in texts]
        print(
            f"[批量并发] {len(translated)}/12 段 · 对位"
            f"{'正确' if aligned else '错误'} · 整帧 {frame_ms:.0f}ms · "
            f"TCP 连接数 {mock_server.connections}"
        )
        if not aligned or len(translated) != 12:
            print(f"  !! 译文: {translated}")
            failures += 1

        # 4) 自动回退：指向不可达端口，本地模型接住
        print("[本地回退] 指向不可达端口，验证回退…")
        config.api = ApiConfig(
            enabled=True,
            base_url="http://127.0.0.1:9/v1",
            model_name="unreachable",
            timeout_seconds=3.0,
        )
        pool.apply_api_config(config.api)
        pool.ensure_loaded(MODEL_FAST)  # 预热本地模型，回退耗时只算翻译本身
        started = time.perf_counter()
        result = translator.translate_text("The weather is nice today.")
        fallback_ms = (time.perf_counter() - started) * 1000
        has_cjk = any(0x4E00 <= ord(ch) <= 0x9FFF for ch in result)
        print(f"  译文: {result!r}  ({fallback_ms:.0f}ms)")
        if not result.strip() or not has_cjk:
            print("  !! 回退后未得到中文译文")
            failures += 1

    pool.shutdown()
    if mock_server is not None:
        mock_server.shutdown()
        mock_server.server_close()

    print("=" * 72)
    if failures:
        print(f"结论：{failures} 项失败，需要修复")
        return 3
    print("结论：在线 AI 路由、批量并发、连接复用与本地回退全部正常")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
