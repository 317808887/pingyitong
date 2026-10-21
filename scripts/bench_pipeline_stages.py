"""端到端延迟分解测量（真实模型 + 真实 OCR，逐阶段计时）。

模拟典型字幕场景：屏幕出现 1~3 行新英文文本 → 流水线各阶段耗时。
目的：找到 300ms 目标下的瓶颈，为优化提供方向。

运行：python scripts/bench_pipeline_stages.py
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from PyQt6.QtGui import QGuiApplication
    _app = QGuiApplication([])

    from pingyitong.components.llm.model_pool import ModelPool
    from pingyitong.components.llm.translation_cache import TranslationCache
    from pingyitong.config.schema import AppConfig
    from pingyitong.core.model_scheduler import ModelScheduler
    from pingyitong.services.glossary_service import Glossary
    from pingyitong.services.history_service import HistoryService
    from pingyitong.services.text_preprocess import TextPreprocessor
    from pingyitong.services.translation_service import TranslationService

    config = AppConfig()
    config.translation.target_language = "zh"
    config.schedule.mode = "manual"
    config.schedule.active_model = "spark-x2.5-1.7b"
    config.model.fast.preload = True

    # ---- 模型 ----
    print("加载 1.7B 模型…")
    pool = ModelPool(config.model, idle_timeout=600)
    loaded = pool.preload(["spark-x2.5-1.7b"])
    if not any(loaded.values()):
        print("模型加载失败")
        return 1

    scheduler = ModelScheduler(config.schedule)
    translator = TranslationService(
        config=config, model_pool=pool, scheduler=scheduler,
        cache=TranslationCache(enabled=True, max_entries=4096, ttl_seconds=1800),
        preprocessor=TextPreprocessor.from_config(config), glossary=Glossary(),
        history=HistoryService(),
    )

    # ---- 场景：1~3 行短句（模拟字幕逐行出现） ----
    scenarios = [
        ("单句", ["Stay behind me."]),
        ("双句", ["Stay behind me.", "We don't have much time."]),
        ("三句", ["Stay behind me.", "We don't have much time.", "Open the door!"]),
    ]

    print("\n=== 端到端延迟分解（1.7B，翻译到中文）===\n")

    # 先预热：让模型加载 + KV cache 就位
    from pingyitong.components.capture.geometry import Rect
    from pingyitong.components.ocr.text_block import TextBlock

    warmup_blocks = [
        TextBlock(text="Warm up the model with this sentence.",
                  box=Rect(100, 100, 400, 30), confidence=0.95),
    ]
    translator.translate_frame(type("R", (), {
        "seq": 0, "origin": Rect(0, 0, 1920, 1080),
        "blocks": warmup_blocks, "monitor_index": 1,
        "is_full_frame": True, "elapsed": 0,
    })())
    print("预热完成\n")

    for name, texts in scenarios:
        # 构造 OCR 块
        blocks = [
            TextBlock(text=text, box=Rect(100, 100 + i * 40, 400, 30),
                      confidence=0.95)
            for i, text in enumerate(texts)
        ]

        # 阶段计时
        # OCR 已经完成了（blocks 是模拟的），这里只测翻译
        t_pre_start = time.perf_counter()
        preprocessor = TextPreprocessor.from_config(config)
        paragraphs, _ = preprocessor.process(blocks)
        t_pre = time.perf_counter() - t_pre_start

        # 翻译（逐次计时）
        t_trans_start = time.perf_counter()
        translator.translate_frame(type("R", (), {
            "seq": 1, "origin": Rect(0, 0, 1920, 1080),
            "blocks": blocks, "monitor_index": 1,
            "is_full_frame": True, "elapsed": 0,
        })())
        t_trans = time.perf_counter() - t_trans_start

        stats = translator.last_stats
        e2e = t_pre + t_trans
        print(f"{name} ({len(texts)} 段):")
        print(f"  预处理: {t_pre*1000:.0f}ms")
        print(f"  翻译:   {t_trans*1000:.0f}ms")
        print(f"  端到端: {e2e*1000:.0f}ms")
        print(f"  统计:   {stats.as_text()}")
        print()

    # ---- 优化试验：短 prompt 对比 ----
    print("=== 优化对比：完整 prompt vs 精简 prompt ===\n")
    from pingyitong.components.llm.prompt import build_system_prompt

    full_prompt = build_system_prompt("auto", "zh")
    short_prompt = (
        "Translate to Chinese. Output only the translation. "
        "Keep numbers, paths, code as-is."
    )

    test_text = "Stay behind me."
    for label, prompt in (("完整", full_prompt), ("精简", short_prompt)):
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            pool.translate(model_id="spark-x2.5-1.7b", prompt=test_text,
                           system_prompt=prompt, max_tokens=128)
            times.append((time.perf_counter() - t0) * 1000)
        print(f"  {label} prompt: 中位 {statistics.median(times):.0f}ms "
              f"最快 {min(times):.0f}ms")

    # ---- 逐行 vs 批量（单句场景）----
    print("\n=== 1.7B 单行请求延迟（5 次采样）===\n")
    for i in range(5):
        t0 = time.perf_counter()
        pool.translate(model_id="spark-x2.5-1.7b",
                       prompt=f"Test line {i}: adjust the setting",
                       system_prompt=full_prompt, max_tokens=128)
        print(f"  #{i+1}: {(time.perf_counter()-t0)*1000:.0f}ms")

    pool.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
