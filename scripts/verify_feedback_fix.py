"""Verify the selected online service using synthetic text, never screenshots.

Reads the saved API configuration without changing it or printing credentials.
Records real direction, mixed-document, cache and two-line OCR timings.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    from bench_real_e2e import _LINES, _percentiles, _render
    from PyQt6.QtGui import QGuiApplication

    from pingyitong.components.capture.geometry import Rect
    from pingyitong.components.llm import ModelPool, TranslationCache
    from pingyitong.components.ocr import OcrEngine, RecognizedFrame, TextBlock
    from pingyitong.config.schema import AppConfig, from_dict
    from pingyitong.constants import MODEL_ACCURATE
    from pingyitong.core import ModelScheduler
    from pingyitong.services.glossary_service import Glossary
    from pingyitong.services.language import has_foreign_prose
    from pingyitong.services.text_preprocess import TextPreprocessor
    from pingyitong.services.translation_service import TranslationService
    from pingyitong.utils.paths import config_path
    from pingyitong.version import __version__

    app = QGuiApplication.instance() or QGuiApplication([])
    config = from_dict(AppConfig, json.loads((args.config or config_path()).read_text(encoding="utf-8")))
    if not config.api.enabled or not config.api.base_url or not config.api.api_key:
        print("No configured online service; no real requests made.")
        return 2
    config.schedule.mode = "manual"
    config.schedule.active_model = MODEL_ACCURATE
    pool = ModelPool(config.model, idle_timeout=600)
    pool.apply_api_config(config.api)
    translator = TranslationService(
        config=config, model_pool=pool, scheduler=ModelScheduler(config.schedule),
        cache=TranslationCache(), preprocessor=TextPreprocessor.from_config(config),
        glossary=Glossary(), history=None,
    )
    engine = pool._api_engine
    report = {
        "version": __version__, "provider_host": urlsplit(config.api.base_url).hostname,
        "model": config.api.model_name, "directions": [], "comparison": [], "dense_document": {},
        "two_line_ocr": [], "timing_scope": "image ready through OCR and translation, excludes display refresh",
    }
    failures = []
    try:
        cases = [
            ("zh", "en", "请保存文件，然后关闭窗口。"),
            ("en", "zh", "The door is open."),
            ("auto", "zh", "Click the Settings button."),
            ("auto", "ko", "安装完成后关闭窗口。Click the Settings button."),
            ("auto", "ja", "请保存文件。Then close the window."),
        ]
        for source, target, text in cases:
            config.translation.source_language = source
            config.translation.target_language = target
            calls = engine.status().total_calls
            started = time.perf_counter()
            output = translator.translate_text(text)
            elapsed = (time.perf_counter() - started) * 1000
            after = engine.status().total_calls
            repeated = translator.translate_text(text)
            ok = bool(output.strip()) and not has_foreign_prose(output, target, text)
            cached = repeated == output and engine.status().total_calls == after
            report["directions"].append({
                "source": source, "target": target, "input": text, "output": output,
                "elapsed_ms": round(elapsed, 1), "api_calls": after - calls,
                "target_language_ok": ok, "repeat_cached": cached,
            })
            if not ok or not cached or after <= calls:
                failures.append(f"direction {source}->{target}")

        original_comparison = config.translation.keep_source_text
        config.translation.keep_source_text = True
        for target, expected_terms in [("zh", ("保存", "取消")), ("ko", ("저장", "취소"))]:
            config.translation.source_language = "en"
            config.translation.target_language = target
            source = "Save | Cancel"
            before = engine.status().total_calls
            output = translator.translate_text(source)
            after = engine.status().total_calls
            repeated = translator.translate_text(source)
            ok = all(term in output for term in expected_terms) and not has_foreign_prose(output, target, source)
            cached = repeated == output and engine.status().total_calls == after
            report["comparison"].append({"target": target, "input": source, "output": output,
                                         "all_menu_items": ok, "repeat_cached": cached,
                                         "api_calls": after - before})
            if not ok or not cached or after <= before:
                failures.append(f"comparison en->{target}")
        config.translation.keep_source_text = original_comparison
        config.translation.source_language = "auto"
        config.translation.target_language = "ko"
        texts = [
            "安装完成后，请关闭窗口。",
            "Click the Settings button to open the preferences dialog.",
            "저장을 마쳤습니다。随后关闭设置窗口。",
            "The online service is connected and ready to translate.",
            "译文应使用目标语言，不要夹杂其他自然语言。",
            "Keep the GPU temperature below 80 degrees.",
            "Check https://example.com/status before retrying.",
            "Save the report to C:/reports/result.json.",
            "Run build_exe.py with --installer to create the package.",
            "字幕按从上到下的顺序排列。",
            "There are 2 subtitle lines and 12 fps capture.",
            "프로그램이 준비되었습니다.",
            "OPEN FILE",
            "SAVE NOW",
        ]
        frame = RecognizedFrame(
            seq=100, origin=Rect(0, 0, 1600, 1000), is_full_frame=True,
            blocks=[TextBlock(text=t, box=Rect(20, 20 + i * 65, 1400, 24), confidence=0.99)
                    for i, t in enumerate(texts)],
        )
        calls = engine.status().total_calls
        started = time.perf_counter()
        rendered = translator.translate_frame(frame)
        elapsed = (time.perf_counter() - started) * 1000
        pairs = translator.last_output()
        valid = all(bool(o.strip()) and not has_foreign_prose(o, "ko", s) for s, o in pairs)
        complete = len(rendered.items) == len(texts)
        after = engine.status().total_calls
        started = time.perf_counter()
        repeat = translator.translate_frame(frame)
        cached_ms = (time.perf_counter() - started) * 1000
        # Cache hits and already-target items can be assembled in a different
        # order. Compare their actual text-to-screen-position mapping.
        def signature(items):
            return sorted((i.box.to_ltrb(), i.text) for i in items)

        cached = engine.status().total_calls == after and signature(repeat.items) == signature(rendered.items)
        report["dense_document"] = {
            "source_blocks": len(texts), "rendered_blocks": len(rendered.items),
            "outputs": [{"source": s, "output": o} for s, o in pairs],
            "elapsed_ms": round(elapsed, 1), "cached_ms": round(cached_ms, 1),
            "api_calls": after - calls, "all_target_language": valid,
            "complete": complete, "repeat_cached": cached,
        }
        if not valid or not complete or not cached:
            failures.append("mixed document")

        ocr = OcrEngine(threads=config.ocr.threads, min_confidence=config.ocr.min_confidence)
        started = time.perf_counter()
        ocr.warm_up()
        ocr.recognize_image(_render(["Warm up the recognition pipeline."]))
        report["ocr_initialization_ms"] = round((time.perf_counter() - started) * 1000, 1)
        for sample in range(6):
            lines = _LINES[sample * 2:sample * 2 + 2]
            image = _render(lines)
            calls = engine.status().total_calls
            started = time.perf_counter()
            blocks = ocr.recognize_image(image)
            ocr_ms = (time.perf_counter() - started) * 1000
            found = " ".join(b.text for b in blocks)
            recognized_ok = " ".join(lines).lower() == found.lower()
            frame = RecognizedFrame(
                seq=200 + sample, origin=Rect(0, 0, image.shape[1], image.shape[0]),
                blocks=blocks, is_full_frame=True,
            )
            rendered = translator.translate_frame(frame)
            total_ms = (time.perf_counter() - started) * 1000
            pairs = translator.last_output()
            valid = bool(rendered.items) and all(not has_foreign_prose(o, "ko", s) and o.strip() for s, o in pairs)
            report["two_line_ocr"].append({
                "sample": sample, "input": lines, "ocr_ms": round(ocr_ms, 1),
                "total_ms": round(total_ms, 1), "api_calls": engine.status().total_calls - calls,
                "recognized_exactly": recognized_ok, "outputs": [o for _, o in pairs],
                "valid_target_language": valid,
            })
            if not valid or not recognized_ok:
                failures.append(f"two-line sample {sample}")
        values = [sample["total_ms"] for sample in report["two_line_ocr"]]
        report["two_line_percentiles_ms"] = {k: round(v, 1) for k, v in _percentiles(values).items()}
        report["api_only"] = not pool._engines
        report["failures"] = failures
        report["passed"] = not failures and report["api_only"] and all(math.isfinite(v) for v in values)
    finally:
        pool.shutdown()
        app.processEvents()
    output_path = args.output or ROOT / "artifacts" / "api_feedback_fix_20261001.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "failures": failures,
                      "dense_ms": report["dense_document"]["elapsed_ms"],
                      "two_line_ms": report["two_line_percentiles_ms"],
                      "record": str(output_path)}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
