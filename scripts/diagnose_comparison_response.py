"""Trace two synthetic comparison-mode requests using the saved API config.

Never prints credentials, changes saved configuration or starts local models.
Only the synthetic menu and optionally two specified mixed-language sentences
are sent; repeated valid translations must use cache.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", action="store_true", help="Also verify the synthetic menu hits cache.")
    parser.add_argument("--mixed", action="store_true", help="Verify both clauses of the Korean/Japanese cases.")
    args = parser.parse_args()

    from pingyitong.components.llm import ModelPool, TranslationCache
    from pingyitong.config.schema import AppConfig, from_dict
    from pingyitong.constants import MODEL_ACCURATE
    from pingyitong.core import ModelScheduler
    from pingyitong.services.glossary_service import Glossary
    from pingyitong.services.language import has_foreign_prose
    from pingyitong.services.text_preprocess import TextPreprocessor
    from pingyitong.services.translation_service import TranslationService
    from pingyitong.utils.paths import config_path

    config = from_dict(AppConfig, json.loads((args.config or config_path()).read_text(encoding="utf-8")))
    if not config.api.enabled or not config.api.base_url or not config.api.api_key:
        print("No configured online service; no requests made.")
        return 2
    config.schedule.mode = "manual"
    config.schedule.active_model = MODEL_ACCURATE
    config.translation.source_language = "en"
    config.translation.keep_source_text = True
    pool = ModelPool(config.model, idle_timeout=600)
    pool.apply_api_config(config.api)

    def refuse_local_engine(*args, **kwargs):
        raise RuntimeError("The diagnostic cannot start local inference.")

    pool.engine = refuse_local_engine
    translator = TranslationService(
        config=config, model_pool=pool, scheduler=ModelScheduler(config.schedule),
        cache=TranslationCache(), preprocessor=TextPreprocessor.from_config(config),
        glossary=Glossary(), history=None,
    )
    real_translate = pool.translate
    report = {"provider_host": urlsplit(config.api.base_url).hostname,
              "model": config.api.model_name, "comparison": []}
    active_trace = []
    secret = config.api.api_key

    def safe(value):
        return value.replace(secret, "[redacted]") if isinstance(value, str) and secret else value

    allowed_sources = {"Save | Cancel", "安装完成后关闭窗口。Click the Settings button.",
                       "请保存文件。Then close the window."}

    def trace_translate(model_id, prompt, system_prompt, max_tokens=None):
        if prompt not in allowed_sources:
            raise RuntimeError("The diagnostic only permits its synthetic menu input.")
        started = time.perf_counter()
        raw = real_translate(model_id, prompt, system_prompt, max_tokens)
        active_trace.append({"prompt": safe(prompt), "system_prompt": safe(system_prompt),
                             "max_tokens": max_tokens, "raw": safe(raw),
                             "elapsed_ms": round((time.perf_counter() - started) * 1000, 1)})
        return raw

    pool.translate = trace_translate
    failures = []
    try:
        cases = [("en", "zh", "Save | Cancel", ("保存", "取消")),
                 ("en", "ko", "Save | Cancel", ("저장", "취소"))]
        if args.mixed:
            cases.extend([("auto", "ko", "安装完成后关闭窗口。Click the Settings button.",
                           ("설치", "창", "설정", "버튼")),
                          ("auto", "ja", "请保存文件。Then close the window.",
                           ("保存", "閉", "ウィンドウ"))])
        for source_language, target, source, expected in cases:
            config.translation.source_language = source_language
            config.translation.target_language = target
            active_trace = []
            started = time.perf_counter()
            output = translator.translate_text(source)
            before_repeat = len(active_trace)
            repeated = translator.translate_text(source) if args.repeat else None
            valid = all(word in output for word in expected) and not has_foreign_prose(
                output, target, source
            )
            cached = repeated == output and len(active_trace) == before_repeat if args.repeat else None
            report["comparison"].append({"source_language": source_language, "target": target,
                                         "input": source, "keep_source_text": True,
                                         "output": safe(output), "valid_menu": valid,
                                         "repeat_cached": cached, "trace": active_trace,
                                         "elapsed_ms": round((time.perf_counter() - started) * 1000, 1)})
            if not valid or (args.repeat and not cached):
                failures.append(target)
        report["api_only"] = not pool._engines
    finally:
        pool.shutdown()
    report["passed"] = not failures and report["api_only"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
