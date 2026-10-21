"""端到端延迟基准（含真实 OCR + 翻译，逐阶段计时）。

模拟典型字幕场景：一块 600×80 像素的小区域内出现 1~3 行英文文本，
流水线完整跑一遍（截屏 → OCR → 预处理 → 翻译），测每阶段耗时。

测试文字用 OpenCV 绘制到合成帧上，不依赖屏幕当前内容——
若直接截取实时屏幕，截到空白区域时 OCR 为 0 块、翻译被跳过，
基准就变成空转（曾导致"OCR 50ms / 翻译 0ms / 渲染 0 条"的假读数）。

运行：python scripts/bench_e2e.py
"""

from __future__ import annotations

import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def draw_texts(image, texts: list[str]):
    """把测试文字用 OpenCV 画到 BGR 图像上（白底黑字，同 selfcheck 风格）。"""
    import cv2

    image = image.copy()
    image[:] = 255
    line_height = image.shape[0] // (len(texts) + 1)
    for i, text in enumerate(texts):
        y = line_height * (i + 1)
        cv2.putText(image, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 0), 2, cv2.LINE_AA)
    return image


def main() -> int:
    from PyQt6.QtGui import QGuiApplication

    from pingyitong.config.schema import AppConfig
    _app = QGuiApplication([])

    from pingyitong.components.capture.geometry import Rect
    from pingyitong.components.llm.model_pool import ModelPool
    from pingyitong.components.llm.translation_cache import TranslationCache
    from pingyitong.components.ocr.ocr_engine import OcrEngine
    from pingyitong.core.model_scheduler import ModelScheduler
    from pingyitong.services.glossary_service import Glossary
    from pingyitong.services.text_preprocess import TextPreprocessor
    from pingyitong.services.translation_service import TranslationService

    config = AppConfig()
    config.translation.target_language = "zh"
    config.schedule.mode = "manual"
    config.model.fast.preload = True

    # 加载模型
    print("加载 1.7B…")
    pool = ModelPool(config.model, idle_timeout=600)
    pool.preload(["spark-x2.5-1.7b"])

    scheduler = ModelScheduler(config.schedule)
    translator = TranslationService(
        config=config, model_pool=pool, scheduler=scheduler,
        cache=TranslationCache(enabled=True), preprocessor=TextPreprocessor.from_config(config),
        glossary=Glossary(),
    )

    # 真实截屏（600×80 逻辑 → 900×120 物理），验证捕获管线可用；
    # 随后把像素替换为绘制的测试文字，保证 OCR 输入内容确定。
    from pingyitong.core.coordinate import CoordinateMapper
    mapper = CoordinateMapper()
    region = Rect(100, 200, 600, 80)
    physical = mapper.logical_to_physical(region)

    from pingyitong.components.capture.screen_capture import ScreenCapturer
    capturer = ScreenCapturer()
    frame = capturer.capture_region(physical)
    print(f"截屏区域: {physical.width}×{physical.height} 物理 px")

    ocr = OcrEngine()
    ocr.warm_up()

    test_texts = [
        ["Stay behind me."],
        ["Stay behind me.", "We don't have much time."],
        ["The quick brown fox jumps over the lazy dog near the river bank."],
    ]

    print("\n=== 端到端延迟基准（含真实 OCR + 翻译）===\n")

    all_e2e: list[float] = []
    all_cold_e2e: list[float] = []

    for si, texts in enumerate(test_texts):
        e2e_times = []
        ocr_times = []
        trans_times = []

        # 同一合成帧重复 5 次：第 1 次翻译走完整推理（冷），其余命中缓存（稳态）
        synth_image = draw_texts(frame.image, texts)
        synth_frame = replace(frame, image=synth_image)

        for _ in range(5):
            t0 = time.perf_counter()

            # 阶段 1：OCR（合成文字帧）
            t_ocr = time.perf_counter()
            recognized = ocr.recognize_frame(synth_frame)
            ocr_cost = time.perf_counter() - t_ocr

            # 阶段 2：预处理 + 翻译
            t_trans = time.perf_counter()
            render_frame = translator.translate_frame(recognized)
            trans_cost = time.perf_counter() - t_trans

            total = time.perf_counter() - t0
            e2e_times.append(total * 1000)
            ocr_times.append(ocr_cost * 1000)
            trans_times.append(trans_cost * 1000)

        n_items = len(render_frame.items)
        cold_trans = trans_times[0]
        steady_trans = statistics.median(trans_times[1:]) if len(trans_times) > 1 else cold_trans
        cold_e2e = e2e_times[0]
        all_e2e.append(statistics.median(e2e_times))
        all_cold_e2e.append(cold_e2e)
        print(f"场景 {si+1}（{len(texts)} 行英文）:")
        print(f"  OCR:     中位 {statistics.median(ocr_times):.0f}ms")
        print(f"  翻译:    首次 {cold_trans:.0f}ms / 稳态(缓存) {steady_trans:.0f}ms")
        print(f"  端到端:  首次 {cold_e2e:.0f}ms / 中位 {statistics.median(e2e_times):.0f}ms"
              f"  最快 {min(e2e_times):.0f}ms  最慢 {max(e2e_times):.0f}ms")
        print(f"  渲染条目: {n_items}")
        if n_items == 0:
            print("  [警告] 渲染条目为 0：OCR 未识别到文字，本场景为空转，需排查")
        else:
            preview = " | ".join(item.text for item in render_frame.items[:3])
            print(f"  译文预览: {preview}")
        print()

    # 汇总
    print("=== 300ms 门槛判定 ===")
    worst = max(all_e2e)
    cold_median = statistics.median(all_cold_e2e)
    print(f"  各场景端到端中位数(稳态): {', '.join(f'{v:.0f}ms' for v in all_e2e)}")
    print(f"  最慢场景: 稳态 {worst:.0f}ms / 首次中位 {cold_median:.0f}ms"
          f"（逐场景首次: {', '.join(f'{v:.0f}ms' for v in all_cold_e2e)}）")
    if worst <= 300 and cold_median <= 300:
        print("  结论: 达标（全部场景稳态中位数与首次中位数均 ≤ 300ms）")
    elif worst <= 300:
        print("  结论: 稳态达标，但首次翻译中位数超过 300ms，需排查冷启动")
    else:
        print("  结论: 未达标（存在中位数 > 300ms 的场景，需排查瓶颈阶段）")

    pool.shutdown()
    capturer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
