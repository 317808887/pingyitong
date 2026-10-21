"""端到端验证：三种翻译方向（中→英 / 英→中 / 自动→中文）。

用**真实**的模型、Prompt 与翻译服务逐方向"真的译一遍"，回答一个具体问题：
**切换目标语言后，三种翻译方向是否都能得到正确的译文**。

同时测量每方向的推理耗时（预热后连续两次，取第二次为稳态值），
用于回归"翻译延迟"指标。缓存被刻意禁用：这里要测的是真实推理，
不是缓存命中。

运行：
    python scripts/verify_translation_directions.py
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def run_direction(translator, source: str, target: str, text: str) -> tuple[str, float]:
    """按给定语言方向翻译一段文本，返回 ``(译文, 耗时秒)``。"""
    translator._config.translation.source_language = source
    translator._config.translation.target_language = target

    started = time.monotonic()
    result = translator.translate_text(text)
    elapsed = time.monotonic() - started
    return result, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description="验证三种翻译方向")
    parser.add_argument("--model", choices=("fast", "accurate"), default="fast")
    args = parser.parse_args()

    from pingyitong.components.llm import ModelPool, TranslationCache
    from pingyitong.config.schema import AppConfig
    from pingyitong.constants import MODEL_ACCURATE, MODEL_FAST
    from pingyitong.core import ModelScheduler
    from pingyitong.services.glossary_service import Glossary
    from pingyitong.services.text_preprocess import TextPreprocessor
    from pingyitong.services.translation_service import TranslationService

    config = AppConfig()
    config.schedule.mode = "manual"
    config.schedule.active_model = MODEL_FAST if args.model == "fast" else MODEL_ACCURATE

    pool = ModelPool(config.model, idle_timeout=600)
    scheduler = ModelScheduler(config.schedule)
    translator = TranslationService(
        config=config,
        model_pool=pool,
        scheduler=scheduler,
        cache=TranslationCache(enabled=False),
        preprocessor=TextPreprocessor.from_config(config),
        glossary=Glossary(),
        history=None,
    )

    print("加载模型…")
    if not pool.ensure_loaded(config.schedule.active_model):
        print("模型加载失败，无法继续验证")
        pool.shutdown()
        return 2

    # 预热：排除首次推理的固定开销，使计时反映稳态延迟
    warm_started = time.monotonic()
    pool.translate(
        model_id=config.schedule.active_model,
        prompt=".",
        system_prompt="Reply with a single dot.",
        max_tokens=8,
    )
    print(f"预热完成（{time.monotonic() - warm_started:.1f}s）")
    print()

    cases = [
        ("英→中", "en", "zh", "The quick brown fox jumps over the lazy dog.", (r"狐狸", r"狗")),
        ("中→英", "zh", "en", "今天天气很好，我们去公园散步吧。", (r"weather", r"park")),
        ("自动→中文", "auto", "zh", "Click the Settings button to open the preferences dialog.", (r"设置", r"对话框")),
    ]

    print("=" * 72)
    failures = 0
    for label, source, target, text, semantic_patterns in cases:
        # 每个方向跑两次：第一次含分隔符等一次性开销，第二次为稳态
        first, first_ms = run_direction(translator, source, target, text)
        second, second_ms = run_direction(translator, source, target, text)

        print(f"[{label}]  source={source}  target={target}")
        print(f"  原文: {text}")
        print(f"  译文: {first}")
        print(f"  耗时: 首次 {first_ms * 1000:.0f}ms / 稳态 {second_ms * 1000:.0f}ms")

        # 温度 0.2 下两次采样的措辞可能略有差异，属正常现象；
        # 方向正确性用脚本判定：中英互译时译文必须落在目标文字系统里。
        if first.strip() != second.strip():
            print("  （两次措辞略有差异：采样随机性，正常）")
        for label_run, translated in (("首次", first), ("稳态", second)):
            if not translated.strip():
                print(f"  !! {label_run}译文为空 —— 视为失败")
                failures += 1
                continue
            has_cjk = any(0x4E00 <= ord(ch) <= 0x9FFF for ch in translated)
            letters = [ch for ch in translated if ch.isalpha()]
            latin = sum(1 for ch in letters if ord(ch) < 0x0250)
            mostly_latin = bool(letters) and latin / len(letters) > 0.8
            if target == "zh" and not has_cjk:
                print(f"  !! {label_run}译文不含汉字 —— 方向错误")
                failures += 1
            elif target == "en" and not mostly_latin:
                print(f"  !! {label_run}译文不是拉丁字母 —— 方向错误")
                failures += 1
            missing = [pattern for pattern in semantic_patterns if not re.search(pattern, translated, re.I)]
            if missing:
                print(f"  !! {label_run}译文缺失关键语义: {missing}")
                failures += 1
        print()

    # 批量对位验证：流水线的真实路径是多段合并单次请求
    from pingyitong.components.llm import build_prompt, parse_response

    segments = ["Save file", "打开设置", "Cancel operation"]
    pair = build_prompt(segments, "auto", "zh")
    raw = pool.translate(
        model_id=config.schedule.active_model,
        prompt=pair.user,
        system_prompt=pair.system,
        max_tokens=384,
    )
    parsed = parse_response(raw, len(segments)) if raw else None
    print("[批量对位] 3 段合并单次请求")
    print(f"  原文: {segments}")
    print(f"  译文: {parsed}")
    if parsed is None:
        print("  !! 批量输出无法对位 —— 降级逻辑会生效，但应排查")
        failures += 1
    print()

    pool.shutdown()
    print("=" * 72)
    if failures:
        print(f"结论：{failures} 项失败，需要修复")
        return 3
    print("结论：中→英、英→中、自动→中文 三种方向全部正常")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
