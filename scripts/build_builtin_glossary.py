"""Snapshot CC0 Wikidata labels for the bundled software/AI glossary.

Run only when intentionally updating the shipped terminology. The application
reads the checked-in JSON offline and never contacts Wikidata at runtime.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pingyitong.language_test_results import VERIFIED_1_7B, VERIFIED_4B  # noqa: E402
from pingyitong.languages import LANGUAGES  # noqa: E402

# Each ID was checked against its English Wikidata description to avoid an
# unrelated item with the same name. Labels are community-maintained terms,
# not a certification of every translation.
CONCEPTS = {
    "artificial_intelligence": "Q11660",
    "machine_learning": "Q2539",
    "deep_learning": "Q197536",
    "artificial_neural_network": "Q192776",
    "large_language_model": "Q115305900",
    "natural_language_processing": "Q30642",
    "computer_vision": "Q844240",
    "optical_character_recognition": "Q167555",
    "generative_ai": "Q117246174",
    "speech_recognition": "Q189436",
    "speech_synthesis": "Q16346",
    "application_programming_interface": "Q165194",
    "software": "Q7397",
    "operating_system": "Q9135",
    "database": "Q8513",
    "algorithm": "Q8366",
    "cloud_computing": "Q483639",
    "encryption": "Q141090",
    "computer_security": "Q3510521",
    "source_code": "Q128751",
    "open_source_software": "Q1130645",
    "dataset": "Q1172284",
    "training_set": "Q3997298",
    "machine_learning_model": "Q115215420",
    "data_science": "Q2374463",
    "supervised_learning": "Q334384",
    "unsupervised_learning": "Q1152135",
    "reinforcement_learning": "Q830687",
    "computer_program": "Q40056",
    "programming_language": "Q9143",
    "software_engineering": "Q80993",
    "user_interface": "Q47146",
    "computer_network": "Q1301371",
    "computer_graphics": "Q150971",
}

# Brand names and formal protocol/file-format identifiers are preserved as-is.
# They work for every verified language without fabricating local translations.
INVARIANT_TERMS = (
    "Spark-X2.5", "Python", "PyTorch", "TensorFlow", "CUDA", "ONNX",
    "LoRA", "GGUF", "JSON", "HTTP", "HTTPS", "SQL", "UTF-8", "Git",
    "Docker", "Kubernetes", "Linux", "Windows",
)

# Wikidata and application language tags differ in these cases.
LANGUAGE_TAGS = {"zh-TW": ("zh-tw", "zh-hant"), "jw": ("jv",), "no": ("nb", "nn")}

# Product terminology in Simplified Chinese differs from some generic zh
# labels, which are Traditional or regional variants.
CURATED_FORMS = {
    "artificial_neural_network": {"zh": "人工神经网络"},
    "large_language_model": {"zh": "大语言模型"},
    "generative_ai": {"zh": "生成式人工智能"},
    "database": {"zh": "数据库"},
    "training_set": {"zh": "训练集"},
    "cloud_computing": {"zh": "云计算"},
}

CURATED_ACRONYMS = {
    "artificial_intelligence": ["AI"],
    "machine_learning": ["ML"],
    "large_language_model": ["LLM"],
    "natural_language_processing": ["NLP"],
    "optical_character_recognition": ["OCR"],
    "application_programming_interface": ["API"],
}


def _read_entity(qid: str) -> dict:
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
    request = urllib.request.Request(
        url, headers={"User-Agent": "PingYiTongGlossaryBuilder/1.0 (offline terminology snapshot)"}
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)["entities"][qid]
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError(qid)


def build() -> dict:
    supported = sorted((VERIFIED_1_7B | VERIFIED_4B) & LANGUAGES.keys())
    concepts = []
    for slug, qid in CONCEPTS.items():
        labels = _read_entity(qid)["labels"]
        forms = {}
        for code in supported:
            tags = LANGUAGE_TAGS.get(code, (code,))
            for tag in tags:
                label = labels.get(tag)
                if label and label.get("value", "").strip():
                    forms[code] = label["value"].strip()
                    break
        if "en" not in forms:
            raise ValueError(f"{qid} has no English label")
        forms.update(CURATED_FORMS.get(slug, {}))
        concepts.append({
            "id": slug, "wikidata_id": qid, "forms": forms,
            "acronyms": CURATED_ACRONYMS.get(slug, []),
        })
        time.sleep(0.25)
    return {
        "schema_version": 1,
        "domain": "software_ai",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "Wikidata entity labels (CC0) plus curated invariant identifiers",
        "source_url": "https://www.wikidata.org/wiki/Wikidata:Licensing",
        "supported_languages": supported,
        "invariant_terms": list(INVARIANT_TERMS),
        "concepts": concepts,
    }


if __name__ == "__main__":
    output = ROOT / "config" / "glossaries" / "software_ai.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)
