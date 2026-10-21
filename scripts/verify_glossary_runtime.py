"""Real-model smoke test for every shipped glossary domain.

This is intentionally separate from unit tests: it loads GGUF models and
checks that placeholder-based terminology survives actual generation.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CASES = (
    ("software_ai", "The team uses machine learning.", "机器学习"),
    ("medicine", "The patient has hypertension.", "高血压"),
    ("biology", "The cell membrane protects the cell.", "细胞膜"),
    ("history", "The Industrial Revolution changed production.", "工业革命"),
    ("physics", "Quantum mechanics studies particles.", "量子力学"),
    ("chemistry", "The molecule has a covalent bond.", "共价键"),
    ("mathematics", "Linear algebra studies vectors.", "线性代数"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("fast", "accurate", "both"), default="both")
    parser.add_argument("--domain", choices=tuple(case[0] for case in CASES))
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="pingyitong-glossary-", ignore_cleanup_errors=True) as home:
        old_home = os.environ.get("PINGYITONG_HOME")
        os.environ["PINGYITONG_HOME"] = home
        try:
            return _run(args.model, args.domain)
        finally:
            if old_home is None:
                os.environ.pop("PINGYITONG_HOME", None)
            else:
                os.environ["PINGYITONG_HOME"] = old_home


def _run(model_choice: str, domain_choice: str | None) -> int:
    from pingyitong.components.llm import ModelPool, TranslationCache
    from pingyitong.config.schema import AppConfig
    from pingyitong.constants import MODEL_ACCURATE, MODEL_FAST
    from pingyitong.core import ModelScheduler
    from pingyitong.services.glossary_service import load_builtin_glossary
    from pingyitong.services.text_preprocess import TextPreprocessor
    from pingyitong.services.translation_service import TranslationService

    ids = {"fast": MODEL_FAST, "accurate": MODEL_ACCURATE}
    chosen = list(ids) if model_choice == "both" else [model_choice]
    cases = [case for case in CASES if domain_choice is None or case[0] == domain_choice]
    config = AppConfig()
    config.schedule.mode = "manual"
    config.translation.source_language = "en"
    config.translation.target_language = "zh"
    config.model.fast.temperature = 0.0
    config.model.accurate.temperature = 0.0
    pool = ModelPool(config.model, idle_timeout=600)
    model_calls: list[tuple[str, str, str]] = []
    original_translate = pool.translate

    def traced_translate(**kwargs):
        output = original_translate(**kwargs)
        model_calls.append((kwargs["prompt"], kwargs["system_prompt"], output or ""))
        return output

    pool.translate = traced_translate
    scheduler = ModelScheduler(config.schedule)
    translator = TranslationService(
        config=config,
        model_pool=pool,
        scheduler=scheduler,
        cache=TranslationCache(enabled=False),
        preprocessor=TextPreprocessor.from_config(config),
        glossary=load_builtin_glossary("zh"),
        history=None,
    )
    failures = 0
    try:
        for choice in chosen:
            config.schedule.active_model = ids[choice]
            scheduler.apply_config(config.schedule)
            if not pool.ensure_loaded(ids[choice]):
                print(f"{choice}: model failed to load")
                failures += len(cases)
                continue
            for domain, source, expected in cases:
                translator.set_glossary(load_builtin_glossary("zh", domain=domain))
                model_calls.clear()
                output = translator.translate_text(source)
                source_language_leaked = bool(re.search(
                    r"\b(?:the|has|is|are|a|an|was|were|of|to|for|with)\b",
                    output, re.IGNORECASE,
                ))
                passed = expected in output and not source_language_leaked
                print(f"{choice}/{domain}: {'PASS' if passed else 'FAIL'} | {output}")
                if not passed:
                    for prompt, system_prompt, raw in model_calls:
                        print(f"  prompt={prompt!r}")
                        print(f"  raw={raw!r}")
                        print(f"  system={system_prompt[:500]!r}")
                    from pingyitong.components.llm.prompt import build_compact_prompt

                    probe = original_translate(
                        model_id=ids[choice],
                        prompt=source,
                        system_prompt=(
                            build_compact_prompt("zh")
                            + f"\nRequired terminology: use the exact Chinese term {expected}. "
                            "Do not omit the subject of the sentence."
                        ),
                        max_tokens=192,
                    )
                    print(f"  fallback_probe={probe!r}")
                    failures += 1
    finally:
        pool.shutdown()
    print(f"result: {len(chosen) * len(cases) - failures}/{len(chosen) * len(cases)} passed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
