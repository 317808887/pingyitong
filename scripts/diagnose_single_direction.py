"""Record only synthetic requests and raw translation responses, never credentials."""
from __future__ import annotations

# ruff: noqa: E402
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
config_file = Path(os.environ["LOCALAPPDATA"]) / "PingYiTong" / "config.json"
os.environ["PINGYITONG_HOME"] = str(ROOT / "artifacts" / "diagnostic_home_0612")

from pingyitong.components.llm import ModelPool, TranslationCache
from pingyitong.config.schema import AppConfig, from_dict
from pingyitong.core import ModelScheduler
from pingyitong.services.glossary_service import Glossary
from pingyitong.services.language import has_foreign_prose
from pingyitong.services.text_preprocess import TextPreprocessor
from pingyitong.services.translation_service import TranslationService

config = from_dict(AppConfig, json.loads(config_file.read_text(encoding="utf-8")))
pool = ModelPool(config.model)
pool.apply_api_config(config.api)
records = []
original_translate = pool.translate


def trace(**kwargs):
    raw = original_translate(**kwargs)
    records.append({"input": kwargs["prompt"], "system": kwargs["system_prompt"], "raw": raw})
    return raw


pool.translate = trace
service = TranslationService(config=config, model_pool=pool,
                             scheduler=ModelScheduler(config.schedule), cache=TranslationCache(),
                             preprocessor=TextPreprocessor.from_config(config), glossary=Glossary(), history=None)
report = []
try:
    for target, text in [("ko", "安装完成后关闭窗口。Click the Settings button."),
                         ("ja", "请保存文件。Then close the window.")]:
        config.translation.source_language = "auto"
        config.translation.target_language = target
        start = len(records)
        output = service.translate_text(text)
        calls = records[start:]
        report.append({"target": target, "output": output, "calls": calls,
                       "raw_checks": [has_foreign_prose(c["raw"], target, text) for c in calls]})
finally:
    pool.shutdown()
output = ROOT / "artifacts" / "single_direction_diagnostic_0612_20261001.json"
output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
