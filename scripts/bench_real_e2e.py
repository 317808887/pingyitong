"""真实端到端延迟基准（自造字幕画面 → 真实 OCR → 真实模型翻译）。

与 ``bench_e2e.py`` 的区别（那个脚本的测量是不可信的，见下）：
``bench_e2e.py`` 把屏幕上**当前显示的内容**当输入，跑完一圈却没有断言
识别/翻译到了什么——屏幕上恰好没有文字或内容已被缓存时，数值无法
代表翻译延迟。本脚本把**已知英文台词**画进一张图，再识别和翻译，
逐条核对译文非空，测量画面就绪至译文可渲染的处理耗时。

测量口径
--------
- **端到端**：从"这一帧画面已就绪"到"该帧译文可渲染"，即 OCR + 预处理 + 模型推理。
  不含捕获等待和实际显示刷新，不能视为用户看到译文的完整延迟。
- **冷启动一次性成本**（模型加载、OCR 预热）单独测量，不计入稳态延迟。
- 每个样本用**互不相同的台词**，避免命中翻译缓存把成本测没。

运行::

    python scripts/bench_real_e2e.py                # 1.7B + 4B 全测
    python scripts/bench_real_e2e.py --model fast   # 只测极速模型
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

#: 台词库：每次取一组互不相同的内容，覆盖字幕常见的长度分布
_LINES = [
    "Stay behind me.",
    "We don't have much time.",
    "Open the door before they see us.",
    "The reactor will overload in three minutes.",
    "I have been waiting for this moment my whole life.",
    "Nobody leaves this room until we find the truth.",
    "Check the readings on the second console.",
    "That signal is coming from somewhere deep underground.",
    "If we cut the power now, the whole city goes dark.",
    "You were never supposed to find out about this.",
    "Tell me everything you know about the missing convoy.",
    "The storm will hit the coast by sunrise tomorrow.",
    "We need to move the supplies before the roads flood.",
    "Every decision you make from now on changes the outcome.",
    "Keep your voice down, the walls have ears in this building.",
    "There is a faster route through the old service tunnel.",
    "I counted at least six guards on the northern gate.",
    "The data we recovered points to a facility in the mountains.",
    "Give me one good reason to trust anything you say.",
    "Time is the only resource we cannot manufacture.",
    "She has been tracking that frequency for eleven days.",
    "Whatever happens next, remember why we started this.",
    "The last transmission ended in the middle of a sentence.",
    "Load the cargo and do not stop for anything on the way.",
]


def _render(texts: list[str], width: int = 1600, line_height: int = 64) -> object:
    """把台词渲染成一张模拟字幕的画面（白字黑底，与播放器字幕一致）。"""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    height = line_height * len(texts) + 24
    image = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(image)

    font = None
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            font = ImageFont.truetype(name, 40)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()

    for index, line in enumerate(texts):
        draw.text((16, 12 + index * line_height), line, fill=(255, 255, 255), font=font)

    # 流水线与 OpenCV 全程使用 BGR，PIL 是 RGB——必须翻转，否则
    # 颜色通道错位（对识别影响不大，但预处理/背景分析与真实捕获不一致）。
    return np.array(image)[:, :, ::-1].copy()


def _percentiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    def pick(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return {
        "p50": pick(0.50),
        "p95": pick(0.95),
        "max": ordered[-1],
        "min": ordered[0],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="真实端到端延迟基准")
    parser.add_argument("--model", choices=("fast", "accurate", "both"), default="both")
    parser.add_argument("--samples", type=int, default=12, help="每个场景的样本数")
    parser.add_argument("--out", default="", help="把结果写入 JSON 文件")
    args = parser.parse_args(argv)

    from PyQt6.QtGui import QGuiApplication

    app = QGuiApplication([])  # noqa: F841 - Qt 环境初始化

    from pingyitong.components.capture.geometry import Rect
    from pingyitong.components.llm.model_pool import ModelPool
    from pingyitong.components.llm.translation_cache import TranslationCache
    from pingyitong.components.ocr.ocr_engine import OcrEngine
    from pingyitong.components.ocr.text_block import RecognizedFrame
    from pingyitong.config.schema import AppConfig
    from pingyitong.core.model_scheduler import ModelScheduler
    from pingyitong.services.glossary_service import Glossary
    from pingyitong.services.text_preprocess import TextPreprocessor
    from pingyitong.services.translation_service import TranslationService

    config = AppConfig()
    config.translation.target_language = "zh"
    config.schedule.mode = "manual"

    models = {
        "fast": ("spark-x2.5-1.7b", config.model.fast),
        "accurate": ("spark-x2.5-4b", config.model.accurate),
    }
    selected = ["fast", "accurate"] if args.model == "both" else [args.model]

    print("=" * 72)
    print("屏译通 真实端到端延迟基准（自造字幕画面 + 真实 OCR + 真实模型）")
    print("=" * 72)

    # ---- 一次性成本（不计入稳态延迟）----
    ocr = OcrEngine()
    if not ocr.is_ready():
        print("OCR 未就绪，无法测量", file=sys.stderr)
        return 1

    warm_image = _render(["Warm up the recognition pipeline."])
    t0 = time.perf_counter()
    ocr.warm_up()
    blocks = ocr.recognize_image(warm_image)
    ocr_cold = time.perf_counter() - t0
    print(f"OCR 冷启动（含预热）：{ocr_cold * 1000:.0f}ms，"
          f"预热帧识别出 {len(blocks)} 块：{[b.text for b in blocks]}")

    if not blocks:
        print("预热帧未识别出文字，OCR 链路异常", file=sys.stderr)
        return 1

    report: dict[str, object] = {
        "measured_at": dt.datetime.now().astimezone().isoformat(),
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "method": (
            "synthetic known English subtitle -> real RapidOCR -> real local GGUF "
            "translation; cache disabled; p95 SLA applies to standard 1-2 line subtitles"
        ),
        "ocr_cold_ms": ocr_cold * 1000,
        "models": {},
    }

    for key in selected:
        model_id, model_config = models[key]
        model_config.preload = True

        print(f"\n--- 模型 {model_id} ---")
        pool = ModelPool(config.model, idle_timeout=600)
        t0 = time.perf_counter()
        loaded = pool.preload([model_id])
        load_cost = time.perf_counter() - t0
        if not any(loaded.values()):
            print(f"模型加载失败，跳过：{model_id}", file=sys.stderr)
            continue
        print(f"模型冷加载：{load_cost:.1f}s")

        scheduler = ModelScheduler(config.schedule)
        translator = TranslationService(
            config=config,
            model_pool=pool,
            scheduler=scheduler,
            # 每轮清缓存，确保测的是"真要翻译"的延迟
            cache=TranslationCache(enabled=False),
            preprocessor=TextPreprocessor.from_config(config),
            glossary=Glossary(),
        )

        # 模型热身：llama-server 刚启动的头几秒，CUDA 上下文/显存分配仍在
        # 后台进行，此时测 OCR 会与模型加载争 CPU，得出虚高的数字。
        # 与真实运行一致：流水线启动时就完成一次预热翻译，稳态从那之后开始。
        warmup = RecognizedFrame(
            seq=999,
            origin=Rect(0, 0, 1600, 100),
            blocks=ocr.recognize_image(_render(["Warm the translation model up now."])),
            elapsed=0,
            is_full_frame=True,
            monitor_index=1,
        )
        t0 = time.perf_counter()
        translator.translate_frame(warmup)
        print(f"模型热身翻译：{(time.perf_counter() - t0) * 1000:.0f}ms（不计入统计）")

        model_report: dict[str, object] = {"load_s": load_cost, "scenarios": {}}

        for line_count in (1, 2, 4, 8):
            e2e: list[float] = []
            ocr_ms: list[float] = []
            trans_ms: list[float] = []
            preprocess_ms: list[float] = []
            inference_ms: list[float] = []
            failed_batches = 0
            missing = 0
            ocr_exact = 0

            for sample in range(args.samples):
                start = (sample * line_count) % len(_LINES)
                texts = [_LINES[(start + i) % len(_LINES)] for i in range(line_count)]
                image = _render(texts)

                frame_started = time.perf_counter()

                t_ocr = time.perf_counter()
                recognized = ocr.recognize_image(image)
                ocr_cost = time.perf_counter() - t_ocr

                if not recognized:
                    missing += 1
                    continue
                ocr_exact += [block.text for block in recognized] == texts

                recognized_frame = RecognizedFrame(
                    seq=sample,
                    origin=Rect(0, 0, image.shape[1], image.shape[0]),
                    blocks=recognized,
                    elapsed=ocr_cost,
                    is_full_frame=True,
                    monitor_index=1,
                )
                # 与流水线一致：端到端从"画面就绪"开始计时
                recognized_frame.__dict__["_captured_at"] = frame_started

                t_trans = time.perf_counter()
                render_frame = translator.translate_frame(recognized_frame)
                trans_cost = time.perf_counter() - t_trans

                e2e.append((time.perf_counter() - frame_started) * 1000)
                ocr_ms.append(ocr_cost * 1000)
                trans_ms.append(trans_cost * 1000)
                preprocess_ms.append(render_frame.timings.get("preprocess", 0.0) * 1000)
                inference_ms.append(render_frame.timings.get("inference", 0.0) * 1000)

                # 核对：确实产出了译文，而不是空跑
                stats = translator.last_stats
                failed_batches += stats.failed_batches
                if stats.translated < line_count or any(
                    not any("\u4e00" <= char <= "\u9fff" for char in item.text)
                    for item in render_frame.items
                ):
                    missing += 1

            if not e2e:
                print(f"  {line_count} 行：无有效样本")
                continue

            entry = {
                "lines": line_count,
                "samples": len(e2e),
                "incomplete": missing,
                "ocr_exact": ocr_exact,
                "e2e": _percentiles(e2e),
                "ocr": _percentiles(ocr_ms),
                "preprocess": _percentiles(preprocess_ms),
                "inference": _percentiles(inference_ms),
                "translate": _percentiles(trans_ms),
                "failed_batches": failed_batches,
                "raw_ms": {
                    "e2e": e2e,
                    "ocr": ocr_ms,
                    "preprocess": preprocess_ms,
                    "inference": inference_ms,
                    "translate": trans_ms,
                },
            }
            model_report["scenarios"][str(line_count)] = entry
            missing_label = f"，未完整翻译 {missing} 次" if missing else ""
            print(
                f"  {line_count} 行字幕：端到端 p50 {entry['e2e']['p50']:.0f}ms / "
                f"p95 {entry['e2e']['p95']:.0f}ms / max {entry['e2e']['max']:.0f}ms"
                f"（OCR p50 {entry['ocr']['p50']:.0f}ms，"
                f"翻译 p50 {entry['translate']['p50']:.0f}ms"
                f"，OCR 原文完全一致 {ocr_exact}/{len(e2e)}"
                f"{missing_label}）"
            )

        report["models"][model_id] = model_report
        pool.shutdown()

    # ---- 300ms 门槛判定 ----
    print("\n" + "=" * 72)
    print("300ms 门槛判定（端到端 p95，标准字幕场景 1~2 行）")
    print("=" * 72)
    verdict: dict[str, object] = {}
    for model_id, model_report in report["models"].items():  # type: ignore[union-attr]
        for line_count in ("1", "2"):
            entry = model_report["scenarios"].get(line_count)  # type: ignore[union-attr]
            if entry is None:
                continue
            p95 = entry["e2e"]["p95"]
            ok = (
                p95 < 300
                and entry["incomplete"] == 0
                and entry["ocr_exact"] == entry["samples"]
            )
            verdict[f"{model_id}:{line_count}"] = {"p95": p95, "pass": ok}
            print(f"  {model_id} {line_count} 行：p95 {p95:.0f}ms → "
                  f"{'通过' if ok else '未达标'}")
    report["verdict"] = verdict

    if args.out:
        Path(args.out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n结果已写入：{args.out}")

    return 0 if verdict and all(item["pass"] for item in verdict.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
