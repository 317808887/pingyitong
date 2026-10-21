"""Reproducible real-model quality, cache and context acceptance evidence.

Mechanical checks are guardrails, not a claim of universal semantic accuracy.
The JSON keeps every source and output for human review. Cache tests measure
repeated workload hit rate separately from uncached model latency.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CASES = [
    ("negation", "en", "zh", "Do not close this window until the backup is complete.",
     [("不要", "勿", "不能", "不可"), ("备份",)]),
    ("numbers_units", "en", "zh", "The timeout is 15 seconds, not 50 seconds.",
     [("15",), ("50",), ("不", "而非")]),
    ("decimal_percent", "en", "zh", "The error rate fell from 12.5% to 0.8% in 2026.",
     [("12.5",), ("0.8",), ("2026",)]),
    ("condition", "en", "zh", "Only administrators can delete these files.",
     [("只有", "仅",), ("管理员",), ("删除",)]),
    ("technical", "en", "zh", "The queue preserves the order of messages when the network reconnects.",
     [("队列",), ("顺序",), ("网络",)]),
    ("path", "en", "zh", "Please save the report to C:/Reports/result.csv before exiting.",
     [("C:/Reports/result.csv",), ("保存",)]),
    ("context_a", "en", "zh", "Alice owns three red bicycles in Berlin.",
     [("三", "3"), ("红",), ("柏林",)]),
    ("context_b", "en", "zh", "The temperature is below zero today.",
     [("零", "0"), ("温度", "气温")]),
    ("context_c", "en", "zh", "Bob has no bicycle and lives in Paris.",
     [("没有", "无"), ("自行车",), ("巴黎",)]),
    ("zh_en", "zh", "en", "请勿删除这三个文件，备份尚未完成。",
     [("not", "Don't", "don't"), ("three", "3"), ("backup",)]),
    ("japanese", "ja", "zh", "保存が完了するまで、この画面を閉じないでください。",
     [("保存",), ("完成", "完毕", "结束"), ("关闭",), ("不要", "勿", "不能", "不可")]),
    ("japanese_variant", "ja", "zh", "ダウンロードが完了するまで、電源を切らないでください。",
     [("下载",), ("完成", "完毕", "结束"), ("电源", "关机"), ("不要", "勿", "不能", "不可")]),
    ("japanese_until", "ja", "zh", "確認が終わるまで、ファイルを削除しないでください。",
     [("确认", "检查", "核实"), ("结束", "完成", "完毕"), ("删除",), ("不要", "勿", "不能", "不可")]),
    ("english_until", "en", "zh", "Do not disconnect the cable until the transfer finishes.",
     [("传输",), ("完成", "结束", "完毕"), ("电缆", "线缆", "连接线", "缆线", "断开线"), ("不要", "勿", "不能", "不可")]),
    ("german", "de", "zh", "Bitte schließen Sie dieses Fenster nicht.",
     [("不要", "勿", "不能", "不可"), ("窗口",)]),
    ("french", "fr", "zh", "Le fichier contient trois pages et deux tableaux.",
     [("三", "3"), ("两", "二", "2"), ("表",)]),
    ("ui", "en", "zh", "OK", [("确定",)]),
]


def check_output(source, output, requirements, target):
    problems = []
    if not output.strip() or output == source:
        problems.append("empty or unchanged output")
    if "<<<SEG>>>" in output or "__TERM_" in output:
        problems.append("internal marker leaked")
    for alternatives in requirements:
        if not any(token.casefold() in output.casefold() for token in alternatives):
            problems.append("missing semantic anchor: " + "/".join(alternatives))
    if target == "zh" and not any("\u4e00" <= ch <= "\u9fff" for ch in output):
        problems.append("wrong writing system")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ready-accurate-backup", action="store_true",
                        help="Prewarm enabled 4B backup for fast requests, matching the desktop product policy.")
    args = parser.parse_args()
    from pingyitong.components.llm import ModelPool, PersistentCacheStore, TranslationCache
    from pingyitong.config.schema import AppConfig
    from pingyitong.constants import MODEL_ACCURATE, MODEL_FAST
    from pingyitong.core import ModelScheduler
    from pingyitong.services.glossary_service import Glossary
    from pingyitong.services.text_preprocess import TextPreprocessor
    from pingyitong.services.translation_service import TranslationService

    report = {"scope": "real local models; anchors plus recorded outputs for semantic review",
              "accurate_backup_prewarmed": args.ready_accurate_backup,
              "models": {}, "cache": {}}
    for model_id in (MODEL_FAST, MODEL_ACCURATE):
        config = AppConfig()
        config.api.enabled = False
        config.schedule.mode = "manual"
        config.schedule.active_model = model_id
        pool = ModelPool(config.model)
        actual_calls = []
        real_translate = pool.translate
        def record_translate(_calls=actual_calls, _translate=real_translate, **kwargs):
            _calls.append(kwargs["model_id"])
            return _translate(**kwargs)
        pool.translate = record_translate
        def service(cache, config=config, pool=pool):
            return TranslationService(config=config, model_pool=pool,
                scheduler=ModelScheduler(config.schedule), cache=cache,
                preprocessor=TextPreprocessor.from_config(config), glossary=Glossary(), history=None)
        translator = service(TranslationCache(enabled=False))
        rows = []
        try:
            if not pool.ensure_loaded(model_id):
                raise RuntimeError("model failed to load: " + model_id)
            if args.ready_accurate_backup and model_id == MODEL_FAST and not pool.ensure_loaded(MODEL_ACCURATE):
                raise RuntimeError("enabled accurate backup failed to load")
            for label, src, tgt, source, requirements in CASES:
                config.translation.source_language = src
                config.translation.target_language = tgt
                started = time.perf_counter()
                calls_before = len(actual_calls)
                output = translator.translate_text(source)
                elapsed = (time.perf_counter() - started) * 1000
                problems = check_output(source, output, requirements, tgt)
                if label == "context_b" and any(t in output for t in ("Alice", "爱丽", "柏林", "自行车")):
                    problems.append("previous request context leaked")
                if label == "japanese" and any(t in output for t in ("不要保存", "勿保存", "不能保存")):
                    problems.append("negation attached to wrong action: source forbids closing, not saving")
                rows.append({"case": label, "source_language": src, "target_language": tgt,
                             "source": source, "output": output, "uncached_ms": elapsed,
                             "requested_model": model_id, "actual_models": actual_calls[calls_before:],
                             "problems": problems})
                print(model_id, label, f"{elapsed:.0f}ms", output, problems, flush=True)
            config.translation.source_language = "en"
            config.translation.target_language = "zh"
            cache_path = args.output.with_name("cache_" + model_id + ".sqlite3")
            cache = TranslationCache(store=PersistentCacheStore(cache_path))
            cache.clear()
            cached_translator = service(cache)
            timings = []
            outputs = []
            source = "The backup is complete. You can safely close this window."
            for _ in range(6):
                started = time.perf_counter()
                outputs.append(cached_translator.translate_text(source))
                timings.append((time.perf_counter() - started) * 1000)
            stats = asdict(cache.stats())
            cache.close()
            restored = TranslationCache(store=PersistentCacheStore(cache_path))
            restored_translator = service(restored)
            started = time.perf_counter()
            restart_output = restored_translator.translate_text(source)
            restart_ms = (time.perf_counter() - started) * 1000
            disk_stats = asdict(restored.stats())
            restored.close()
            passed = bool(outputs[0]) and len(set(outputs + [restart_output])) == 1 and stats["hits"] >= 5 and disk_stats["hits"] >= 1
            report["cache"][model_id] = {"source": source, "output": outputs[0], "first_ms": timings[0],
                "repeat_median_ms": statistics.median(timings[1:]), "stats": stats,
                "restart_ms": restart_ms, "restart_stats": disk_stats,
                "workload_hit_rate": stats["hits"] / max(1, stats["hits"] + stats["misses"]), "passed": passed}
        finally:
            pool.shutdown()
        report["models"][model_id] = rows
    report["passed"] = all(not row["problems"] for rows in report["models"].values() for row in rows) and all(r["passed"] for r in report["cache"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
