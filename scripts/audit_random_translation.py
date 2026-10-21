"""Probe fresh multilingual scene text through the real local product translator.

Input is generate_acceptance_scenes.py JSON. This deliberately bypasses OCR,
capture, painting and installation. It cannot establish screen latency or a
semantic accuracy pass: every original/reference/output remains pending human
review. Each model/scene starts with a new empty in-memory cache and translator,
then repeats the exact frame to measure actual evidence-bearing cache reuse.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_translation_semantics import (  # noqa: E402
    MEASUREMENT_POLICY,
    LocalCallTrace,
    frame_result,
    prepare_model_measurement,
    recognized_frame,
)


def load_corpus(path: Path) -> dict:
    corpus = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(corpus, dict) or corpus.get("schema") != "pingyitong-random-scenes-v1":
        raise ValueError("Expected generate_acceptance_scenes.py corpus")
    scenes = corpus.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("The corpus must contain scenes")
    seen = set()
    for scene in scenes:
        if not isinstance(scene, dict) or not isinstance(scene.get("scene_id"), str):
            raise ValueError("Each scene needs an explicit scene_id")
        if scene["scene_id"] in seen:
            raise ValueError("Duplicate scene_id")
        seen.add(scene["scene_id"])
        source, target = scene.get("source_language"), scene.get("target_language")
        if not source or not target or source == "auto" or source == target:
            raise ValueError("Each scene must explicitly specify distinct source/target languages")
        rows = scene.get("rows")
        if not isinstance(rows, list) or not rows:
            raise ValueError("Each scene needs source/reference rows")
        for row in rows:
            if not isinstance(row, dict) or any(
                not isinstance(row.get(field), str) or not row[field].strip()
                for field in ("id", "source", "reference", "invariants")
            ):
                raise ValueError("Each row requires id, source, reference and invariants")
    return corpus


def cache_evidence(first: dict, replay: dict) -> dict:
    """Zero calls from withheld/same-language output cannot prove a cache hit."""
    eligible = [item for item in first["items"]
                if item["quality_status"] == "accepted" and not item["source_retained"]
                and item["quality_reason"] != "review_skipped_by_policy"]
    reused = [item for item in replay["items"]
              if item["quality_status"] == "accepted" and item["from_cache"]
              and not item["source_retained"]]
    # Source plus complete/compact output must match. Counting hits alone could
    # conceal a shifted row or stale output from another language direction.
    def signature(item):
        return item["source"], item["full_text"], item["display_text"]

    expected = sorted(signature(item) for item in eligible)
    actual = sorted(signature(item) for item in reused)
    unchanged = expected == actual
    zero_calls = replay["model_call_count"] == 0
    if not eligible:
        status = "not_applicable_no_reviewed_translation"
    elif not unchanged:
        status = "failed_reviewed_output_not_reused"
    elif zero_calls:
        status = "verified_zero_model_calls"
    else:
        status = "verified_reviewed_subset_other_items_required_inference"
    return {
        "status": status, "eligible_reviewed_items": len(eligible),
        "reused_reviewed_items": len(reused), "reviewed_outputs_unchanged": unchanged,
        "replay_model_calls": replay["model_call_count"],
        "whole_frame_zero_model_calls_proven": bool(eligible) and unchanged and zero_calls,
    }


def run_scene(scene: dict, config, pool, scheduler, trace: LocalCallTrace) -> dict:
    from pingyitong.components.llm import TranslationCache
    from pingyitong.services.glossary_service import Glossary
    from pingyitong.services.text_preprocess import TextPreprocessor
    from pingyitong.services.translation_service import TranslationService

    config.translation.source_language = scene["source_language"]
    config.translation.target_language = scene["target_language"]
    config.translation.fallback_source_language = scene["source_language"]
    config.translation.scene_mode = {
        "technical_document": "document", "dense_page": "dense", "subtitle": "subtitle",
    }.get(scene["category"], "auto")
    cache = TranslationCache()
    translator = TranslationService(
        config=config, model_pool=pool, scheduler=scheduler, cache=cache,
        preprocessor=TextPreprocessor.from_config(config), glossary=Glossary(), history=None,
    )
    before = len(cache)
    passes = []
    for seq, label in enumerate(("fresh_empty_cache", "identical_frame_replay"), 1):
        trace.calls.clear()
        started = time.perf_counter()
        rendered = translator.translate_frame(recognized_frame([row["source"] for row in scene["rows"]], seq))
        result = frame_result(rendered, (time.perf_counter() - started) * 1000, trace.calls)
        result.update({"pass": label, "cache_entries": len(cache), "cache_stats": asdict(cache.stats())})
        for item in result["items"]:
            item["row_ids"] = [row["id"] for row in scene["rows"] if row["source"] == item["source"]]
        result["unmapped_row_ids"] = [row["id"] for row in scene["rows"]
                                      if not any(row["id"] in item["row_ids"] for item in result["items"])]
        passes.append(result)
    return {
        "scene_id": scene["scene_id"], "category": scene["category"],
        "source_language": scene["source_language"], "target_language": scene["target_language"],
        "rows": scene["rows"], "cache_entries_before_fresh": before,
        "passes": passes, "cache_proof": cache_evidence(*passes),
        "human_semantic_review": "pending",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=("fast", "accurate", "both"), default="both")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--review-policy", choices=("always", "risk"), default="always")
    parser.add_argument("--scene-id", action="append", help="Run only selected scene IDs (repeatable)")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite a previous result; use a fresh output path")
    corpus = load_corpus(args.scenes)
    scenes = [scene for scene in corpus["scenes"] if not args.scene_id or scene["scene_id"] in args.scene_id]
    if not scenes or (args.scene_id and set(args.scene_id) - {scene["scene_id"] for scene in scenes}):
        parser.error("Selected scene IDs are absent from the corpus")

    from pingyitong.components.llm import ModelPool
    from pingyitong.config.schema import AppConfig
    from pingyitong.constants import MODEL_ACCURATE, MODEL_FAST
    from pingyitong.core import ModelScheduler
    from pingyitong.version import __version__

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "pingyitong-random-local-probe-v1", "source_version": __version__,
        "scope": "Actual local ModelPool plus TranslationService.translate_frame; authored known text bypasses OCR/capture/paint/native installation. Elapsed time is translation-only and cannot establish the 300–600ms screen target.",
        "corpus_sha256": hashlib.sha256(args.scenes.read_bytes()).hexdigest(),
        "generated_content_sha256": corpus.get("corpus_sha256"),
        "seed": corpus.get("seed"), "run_id": corpus.get("run_id"),
        "temperature": args.temperature, "review_policy": args.review_policy,
        "measurement_policy": MEASUREMENT_POLICY,
        "human_semantic_review": "pending", "models": {}, "results": [], "operational_errors": [],
    }
    ids = {"fast": MODEL_FAST, "accurate": MODEL_ACCURATE}
    choices = ids if args.model == "both" else {args.model: ids[args.model]}
    old_home = os.environ.get("PINGYITONG_HOME")
    with tempfile.TemporaryDirectory(prefix="pingyitong-random-probe-") as home:
        os.environ["PINGYITONG_HOME"] = home
        pool = None
        trace = None
        try:
            config = AppConfig()
            config.schedule.mode = "manual"
            config.api.enabled = False
            config.prefetch.enabled = False
            config.translation.local_only = True
            config.translation.semantic_review = True
            config.translation.review_policy = args.review_policy
            config.model.fast.temperature = args.temperature
            config.model.accurate.temperature = args.temperature
            pool = ModelPool(config.model, idle_timeout=600)
            trace = LocalCallTrace(pool)
            scheduler = ModelScheduler(config.schedule)
            previous_model_id = None
            for model_name, model_id in choices.items():
                config.schedule.active_model = model_id
                scheduler.apply_config(config.schedule)
                measurement = prepare_model_measurement(pool, model_id, previous_model_id)
                previous_model_id = model_id
                report["models"][model_name] = measurement
                for scene in scenes:
                    result = run_scene(scene, config, pool, scheduler, trace)
                    result.update({"model": model_name, **measurement})
                    if measurement["identity_status"] != "verified":
                        proof = result["cache_proof"]
                        proof["unverified_identity_observation_status"] = proof["status"]
                        proof["status"] = "identity_unverified"
                        proof["whole_frame_zero_model_calls_proven"] = False
                    report["results"].append(result)
                    # Checkpoint the complete outputs before progressing to the
                    # next scene, so an interruption does not erase evidence.
                    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    print(json.dumps({"model": model_name, "scene_id": scene["scene_id"],
                                      "direction": f"{scene['source_language']}->{scene['target_language']}",
                                      "calls": [step["model_call_count"] for step in result["passes"]],
                                      "statuses": [item["quality_status"] for item in result["passes"][0]["items"]],
                                      "cache_proof": result["cache_proof"]["status"],
                                      "human_semantic_review": "pending"}), flush=True)
        except Exception as error:
            report["operational_errors"].append({"type": type(error).__name__, "message": str(error)})
        finally:
            if trace is not None:
                trace.restore()
            if pool is not None:
                pool.shutdown()
            if old_home is None:
                os.environ.pop("PINGYITONG_HOME", None)
            else:
                os.environ["PINGYITONG_HOME"] = old_home
    first_items = [item for result in report["results"] for item in result["passes"][0]["items"]]
    report["summary"] = {
        "scenes": len(report["results"]),
        "accepted_translations": sum(item["quality_status"] == "accepted" and not item["source_retained"] for item in first_items),
        "uncertain_source_retained": sum(item["quality_status"] == "uncertain" for item in first_items),
        "other_source_retained": sum(item["quality_status"] != "uncertain" and item["source_retained"] for item in first_items),
        "unmapped_rows": sum(len(result["passes"][0]["unmapped_row_ids"]) for result in report["results"]),
        "zero_call_cache_frames": sum(result["cache_proof"]["whole_frame_zero_model_calls_proven"] for result in report["results"]),
        "cache_reuse_regressions": sum(result["cache_proof"]["status"].startswith("failed_") for result in report["results"]),
        "identity_unverified_models": [name for name, model in report["models"].items()
                                       if model["identity_status"] != "verified"],
        "semantic_accuracy": "pending_human_review",
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))
    return int(bool(report["operational_errors"]) or bool(report["summary"]["cache_reuse_regressions"]))


if __name__ == "__main__":
    raise SystemExit(main())
