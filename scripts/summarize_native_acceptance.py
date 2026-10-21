"""Join actual installed-app paint evidence with source paints and explicit reviews.

No model calls, screenshot simulation, mouse automation or implicit semantic
approval. A source match and capture after scene appearance are both required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import unicodedata
from pathlib import Path

PROTOCOL_V2 = "pingyitong-native-stages-v2"
READABLE_FONT_PX = 14


def literal_text(text):
    """Only NFC and whitespace; punctuation, sign, case and order remain."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def literal_matched_rows(rows, items):
    return {row["id"]: [index for index, item in enumerate(items)
                        if literal_text(row["source"]) in literal_text(item.get("source", ""))]
            for row in rows if literal_text(row["source"]) and
            any(literal_text(row["source"]) in literal_text(item.get("source", "")) for item in items)}


def source_line_matches(rows, lines):
    groups = {}
    for index, line in enumerate(lines):
        line_id = line.get("line_id", -1)
        key = ("line", line_id) if isinstance(line_id, int) and line_id >= 0 else ("block", index)
        groups.setdefault(key, []).append(line)
    texts = {key: literal_text(" ".join(line.get("text", "") for line in
             sorted(values, key=lambda value: value.get("rect", [0])[0])))
             for key, values in groups.items()}
    # One recognized line may not stand in for two distinct source rows.
    matches = {row["id"]: [str(key) for key, value in texts.items() if literal_text(row["source"]) == value]
               for row in rows}
    complete = all(matches.values()) and len({key for values in matches.values() for key in values}) >= len(rows)
    return {"matched_rows": sum(bool(value) for value in matches.values()),
            "distinct_line_structure_preserved": complete,
            "row_line_matches": matches}


def observed_runtime(paint):
    """Validate recorded call counters against same-process operation events."""
    unknown = {"observed": False, "delivery_route": "unobserved", "inference_calls": None,
               "zero_inference_calls_proven": None, "counts": None}
    if not paint:
        return unknown
    delivery = paint.get("delivery_evidence") or {}
    work = delivery.get("work") or {}
    kinds = ("model_load", "model_warmup", "local_inference", "api_inference")
    if (work.get("protocol_version") != PROTOCOL_V2 or work.get("coverage") != "model_pool_local_and_api_calls"
            or work.get("completed") is not True or work.get("coalesced_unknown")
            or not isinstance(work.get("events"), list) or type(paint.get("process_id")) is not int):
        return unknown
    events = work["events"]
    if any(not isinstance(event, dict) or event.get("kind") not in kinds
           or event.get("process_id") != paint["process_id"] for event in events):
        return unknown
    counts = {kind: sum(event["kind"] == kind for event in events) for kind in kinds}
    if any(type(work.get("counts", {}).get(kind)) is not int or work["counts"][kind] != count
           for kind, count in counts.items()):
        return unknown
    inferences = counts["local_inference"] + counts["api_inference"]
    route = delivery.get("delivery_route", "unobserved")
    if route == "trusted_cache_restore" and (inferences or delivery.get("confirmed_retained_items", 0)):
        return unknown
    return {"observed": True, "delivery_route": route, "counts": counts,
            "inference_calls": inferences, "zero_inference_calls_proven": inferences == 0,
            "trace_id": work.get("trace_id"),
            "confirmed_retained_items": delivery.get("confirmed_retained_items", 0)}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(text):
    return "".join(char for char in unicodedata.normalize("NFKC", text).casefold()
                   if not char.isspace() and not unicodedata.category(char).startswith("P"))


def output_hash(items):
    values = [{**{key: item.get(key, "") for key in ("source", "full_text", "display_text")},
               "compact_text": item.get("compact_text"),
               "compact_generated": item.get("compact_generated")} for item in items]
    return hashlib.sha256(json.dumps(values, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def compact_generation_evidence(items):
    known = [item for item in items if type(item.get("compact_generated")) is bool
             and isinstance(item.get("compact_text"), str)
             and bool(item["compact_text"]) == item["compact_generated"]]
    return {
        "generated_items": sum(item["compact_generated"] for item in known),
        "not_generated_items": sum(not item["compact_generated"] for item in known),
        "unknown_items": len(items) - len(known),
        "basis": "Explicit compact_generated and compact_text fields; display_text alone is effective inline text, not generated-compact evidence.",
    }


def matched_rows(rows, items):
    matched = {}
    for row in rows:
        source = normalize(row["source"])
        matches = [index for index, item in enumerate(items) if source and source in normalize(item.get("source", ""))]
        if matches:
            matched[row["id"]] = matches
    return matched


def overlap_pairs(items):
    overlaps = []
    for index, first in enumerate(items):
        if len(first.get("rect", [])) != 4:
            continue
        x, y, width, height = first["rect"]
        for other_index, other in enumerate(items[index + 1:], index + 1):
            if len(other.get("rect", [])) != 4:
                continue
            ox, oy, ow, oh = other["rect"]
            area = max(0, min(x + width, ox + ow) - max(x, ox)) * max(0, min(y + height, oy + oh) - max(y, oy))
            if area > 0.01:
                overlaps.append({"indices": [index, other_index], "area": area})
    return overlaps


def ready(items):
    return bool(items) and all(item.get("quality_status") == "accepted" and
                               item.get("disposition") == "translated" and
                               not item.get("truncated", True) and
                               bool(item.get("full_text", "").strip()) for item in items)


def percentile(values, percent):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * percent / 100) - 1)] if ordered else None


def summarize(corpus: dict, fixture_events: list[dict], paints: list[dict], identity: dict,
              annotations: dict, installed_sha256: str, model_events: list[dict] | None = None) -> dict:
    run_id = corpus["run_id"]
    strict_protocol = corpus.get("protocol_version") == PROTOCOL_V2
    scene_map = {scene["scene_id"]: scene for scene in corpus["scenes"]}
    events = sorted((event for event in fixture_events if event.get("run_id") == run_id),
                    key=lambda event: event["paint_completed_monotonic_ns"])
    reviews = {entry["instance_id"]: entry for entry in annotations.get("reviews", [])}
    rows = []
    for index, event in enumerate(events):
        scene = scene_map.get(event["scene_id"])
        if scene is None:
            continue
        blank_scene = not scene["rows"]
        instance_id = f"{event['scene_id']}@{event['generation']}"
        appeared = event["paint_completed_monotonic_ns"] / 1e9
        next_appearance = events[index + 1]["paint_completed_monotonic_ns"] / 1e9 if index + 1 < len(events) else math.inf
        candidates = []
        missing_capture = 0
        for paint in paints:
            if paint.get("run_id") != run_id or paint.get("evidence_kind") != "application_submitted_paint_buffer":
                continue
            painted = paint.get("painted_at", 0)
            if not appeared <= painted < next_appearance:
                continue
            captured = paint.get("captured_at")
            if captured is None:
                missing_capture += 1
                continue
            if captured < appeared or captured > painted:
                continue
            matches = matched_rows(scene["rows"], paint.get("items", []))
            candidates.append((paint, matches))
        candidates.sort(key=lambda pair: pair[0]["painted_at"])
        complete = [(paint, matches) for paint, matches in candidates
                    if len(matches) == len(scene["rows"]) and (not blank_scene or paint.get("items") == [])
                    and (not strict_protocol or blank_scene or
                         len(literal_matched_rows(scene["rows"], paint.get("items", []))) == len(scene["rows"]))]
        translated = complete if blank_scene else [(paint, matches) for paint, matches in complete
                    if ready(paint["items"]) and (not strict_protocol or
                       all(item.get("font_px", 0) >= READABLE_FONT_PX for item in paint["items"]))]
        chosen = translated[0] if translated else complete[0] if complete else max(candidates, key=lambda pair: len(pair[1]), default=(None, {}))
        paint, matches = chosen
        issues = []
        if paint is None:
            issues.append("No post-scene capture with matching source evidence")
        elif len(matches) != len(scene["rows"]):
            issues.append("Not all source rows were recognized in an eligible paint")
        if not translated:
            issues.append("No eligible post-scene empty paint" if blank_scene else
                          "No complete, accepted, untruncated original-position translation")
        expected_sources = [normalize(row["source"]) for row in scene["rows"]]
        fixture_sources = [normalize(block["text"]) for block in event.get("blocks", [])]
        if fixture_sources != expected_sources:
            issues.append("Fixture content differs from the recorded corpus; use a separate scene id for edits")
        items = paint.get("items", []) if paint else []
        literal_matches = literal_matched_rows(scene["rows"], items)
        line_matches = source_line_matches(scene["rows"], paint.get("source_lines", []) if paint else [])
        runtime = observed_runtime(paint)
        if strict_protocol and not blank_scene:
            if len(literal_matches) != len(scene["rows"]):
                issues.append("Source punctuation/case/order differs from literal source rows")
            if not line_matches["distinct_line_structure_preserved"]:
                issues.append("Distinct original source-line evidence is missing or changed")
            if any(item.get("disposition") == "translated" and item.get("font_px", 0) < READABLE_FONT_PX for item in items):
                issues.append("Inline text is below the readable logical-pixel floor")
            if not runtime["observed"]:
                issues.append("Bound per-frame model operation evidence is missing or inconsistent")
        extra_items = [item.get("source", "") for item in items
                       if not any(source in normalize(item.get("source", "")) for source in expected_sources)]
        if extra_items:
            issues.append("Unexpected source items remain in the painted scene")
        overlaps = overlap_pairs(items)
        if overlaps:
            issues.append("Allocated paint rectangles overlap")
        digest = output_hash(items)
        review = reviews.get(instance_id, {})
        review_matches = review.get("output_sha256") == digest
        reviewer_type = review.get("reviewer_type")
        reviewed = (not blank_scene and review_matches and reviewer_type in {"human", "ai"}
                    and all(review.get(key) is True for key in ("full_meaning", "compact_meaning", "readable")))
        if not blank_scene and not reviewed:
            issues.append("Meaning/readability review missing, failed, untyped, or bound to different output")
        png_verified = bool(paint and paint.get("_png_verified"))
        if not png_verified:
            issues.append("Actual submitted PNG buffer is missing")
        latency = (translated[0][0]["painted_at"] - appeared) * 1000 if translated else None
        target = scene.get("latency_target_ms")
        if target is not None and (latency is None or latency > target):
            issues.append("Scene-to-complete-translation latency target not met")
        cached = sum(bool(item.get("from_cache")) for item in items)
        route = "not_applicable_blank" if blank_scene else "full_cache" if items and cached == len(items) else "mixed_cache" if cached else "uncached"
        rows.append({"instance_id": instance_id, "scene_id": scene["scene_id"], "category": scene["category"],
                     "source_language": scene["source_language"], "target_language": scene["target_language"],
                     "source_rows": len(scene["rows"]), "recognized_rows": len(matches),
                     "literal_recognized_rows": len(literal_matches), "source_line_evidence": line_matches,
                     "source_complete_ms": (complete[0][0]["painted_at"] - appeared) * 1000 if complete else None,
                     "scene_to_complete_ms": latency, "latency_target_ms": target,
                     "cache_route": route, "from_cache_items": cached, "items": items,
                     "output_sha256": digest, "paint_file": paint.get("_file") if paint else None,
                     "seq": paint.get("seq") if paint else None, "overlap_pairs": overlaps,
                     "unexpected_sources": extra_items, "blank_scene": blank_scene,
                     "review_required": not blank_scene, "reviewer_type": reviewer_type if not blank_scene else None,
                     "meaning_readability_review_passed": reviewed if not blank_scene else None,
                     "human_review_passed": bool(reviewed and reviewer_type == "human"),
                     "ai_review_passed": bool(reviewed and reviewer_type == "ai"),
                     "cache_route_basis": "item.from_cache flags only",
                     "model_zero_calls_proven": runtime["zero_inference_calls_proven"] if strict_protocol else None,
                     "runtime_observation": runtime,
                     "refresh_kind": paint.get("refresh_kind") if paint else None,
                     "scenario_kind": scene.get("scenario_kind"),
                     "diagnostic_events": paint.get("diagnostic_events", []) if paint else [],
                     "compact_generation_evidence": compact_generation_evidence(items),
                     "missing_capture_evidence": missing_capture, "issues": issues})
    native = annotations.get("native_ui", {})
    identity_verified = (bool(installed_sha256) and identity.get("executable_sha256", "").casefold() == installed_sha256.casefold()
                         and Path(identity.get("executable", "")).name.casefold() == "pingyitong.exe")
    native_verified = (identity_verified and native.get("run_id") == run_id and native.get("performed") is True
                       and native.get("installed_executable_sha256", "").casefold() == installed_sha256.casefold()
                       and bool(native.get("evidence_reference")))
    limitations = ["Timing ends at application paint submission; compositor/monitor presentation delay is not observable here.",
                   "Rectangle checks and saved paint pixels accompany an explicitly typed readability review; darkness alone is not a universal overlap detector.",
                   "Legacy cache_route uses item.from_cache flags and does not prove zero model calls; any v2 proof is separately recorded in runtime_observation."]
    if any(row["ai_review_passed"] for row in rows):
        limitations.append("Some meaning/readability judgments are independent AI reviews, not human acceptance; reviewer_type is retained per scene.")
    if any(row["compact_generation_evidence"]["unknown_items"] for row in rows):
        limitations.append("Some legacy/inconsistent paint items lack explicit compact-generation evidence; nonempty display_text does not establish generated-compact coverage.")
    if not native_verified:
        limitations.append("Installed identity or native mouse-operation annotation is missing/unmatched.")
    missing_scenes = sorted(set(scene_map) - {row["scene_id"] for row in rows})
    if missing_scenes:
        limitations.append("Unexecuted scenes: " + ", ".join(missing_scenes))
    latencies = [row["scene_to_complete_ms"] for row in rows
                 if not row["blank_scene"] and row["scene_to_complete_ms"] is not None]
    blank_latencies = [row["scene_to_complete_ms"] for row in rows
                       if row["blank_scene"] and row["scene_to_complete_ms"] is not None]
    cohorts = {}
    for row in rows:
        key = f"{row['category']}/{row['cache_route']}"
        cohorts.setdefault(key, []).append(row)
    metrics = {"scenes_executed": len(rows), "scenes_expected": len(scene_map),
               "scenes_passed": sum(not row["issues"] for row in rows),
               "semantic_review_passed": sum(row["meaning_readability_review_passed"] is True for row in rows),
               "meaning_readability_review_passed": sum(row["meaning_readability_review_passed"] is True for row in rows),
               "human_review_passed": sum(row["human_review_passed"] for row in rows),
               "ai_review_passed": sum(row["ai_review_passed"] for row in rows),
               "blank_scenes_verified": sum(row["blank_scene"] and native_verified and not row["issues"] for row in rows),
               "generated_compact_items": sum(row["compact_generation_evidence"]["generated_items"] for row in rows),
               "compact_generation_unknown_items": sum(row["compact_generation_evidence"]["unknown_items"] for row in rows),
               "source_rows": sum(row["source_rows"] for row in rows),
               "recognized_rows": sum(row["recognized_rows"] for row in rows),
               "literal_recognized_rows": sum(row["literal_recognized_rows"] for row in rows),
               "scene_latency_p50_ms": statistics.median(latencies) if latencies else None,
               "scene_latency_p95_ms": percentile(latencies, 95),
               "blank_clear_p50_ms": statistics.median(blank_latencies) if blank_latencies else None,
               "blank_clear_p95_ms": percentile(blank_latencies, 95),
               "cohorts": {key: {"count": len(values), "passed": sum(not value["issues"] for value in values),
                                  "p95_ms": percentile([value["scene_to_complete_ms"] for value in values if value["scene_to_complete_ms"] is not None], 95)}
                           for key, values in cohorts.items()}}
    passed = bool(rows) and native_verified and not missing_scenes and all(not row["issues"] for row in rows)
    stage_cohorts = {}
    for row in rows:
        if row["blank_scene"]:
            key = "blank_clear"
        else:
            runtime = row["runtime_observation"]
            key = "/".join((row["scenario_kind"] or row["refresh_kind"] or "unknown_refresh", runtime["delivery_route"]))
        stage_cohorts.setdefault(key, []).append(row)
    metrics["stage_cohorts"] = {
        key: {"count": len(values), "completed_count": sum(value["scene_to_complete_ms"] is not None for value in values),
              "p50_ms": statistics.median([value["scene_to_complete_ms"] for value in values
                                           if value["scene_to_complete_ms"] is not None])
                        if any(value["scene_to_complete_ms"] is not None for value in values) else None,
              "p95_ms": percentile([value["scene_to_complete_ms"] for value in values
                                     if value["scene_to_complete_ms"] is not None], 95)}
        for key, values in stage_cohorts.items()
    }
    load_events = [event for event in model_events or []
                   if event.get("run_id") == run_id and event.get("process_id") == identity.get("process_id")
                   and event.get("evidence_kind") == "observed_model_pool_operation"
                   and event.get("kind") in {"model_load", "model_warmup"}
                   and isinstance(event.get("elapsed_seconds"), (float, int))]
    metrics["model_load_and_warmup"] = [
        {key: event.get(key) for key in ("kind", "model_id", "started_at", "finished_at", "elapsed_seconds", "successful", "trace_id")}
        for event in load_events
    ]
    metrics["model_load_observation_available"] = model_events is not None
    if strict_protocol:
        limitations.append("Only v2 runtime_observation proves per-frame ModelPool calls; legacy cache_route still describes item flags. Preload/warmup events are separate and are not subtracted from scene waiting time.")
    return {"run_id": run_id, "seed": corpus["seed"], "corpus_sha256": corpus["corpus_sha256"],
            "protocol_version": PROTOCOL_V2 if strict_protocol else "legacy-paint-content-v1",
            "outcome": "passed" if passed else "failed" if rows else "partial", "native_installed_verified": native_verified,
            "cache_route_basis": "item.from_cache flags only", "model_zero_calls_proven": None,
            "identity": identity, "metrics": metrics, "limitations": limitations, "scenes": rows,
            "review_template": {"native_ui": {"run_id": run_id, "performed": False,
                                               "installed_executable_sha256": installed_sha256, "evidence_reference": ""},
                                "reviews": [{"instance_id": row["instance_id"], "output_sha256": row["output_sha256"],
                                             "reviewer_type": None, "full_meaning": None,
                                             "compact_meaning": None, "readable": None, "note": ""}
                                            for row in rows if row["review_required"]]}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--fixture-paints", type=Path, required=True)
    parser.add_argument("--paint-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--installed-exe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args()
    corpus = read_json(args.corpus)
    events = [json.loads(line) for line in args.fixture_paints.read_text(encoding="utf-8").splitlines() if line.strip()]
    paints = []
    for path in args.paint_dir.glob("paint_*.json"):
        paint = read_json(path)
        paint["_file"] = str(path)
        paint["_png_verified"] = path.with_suffix(".png").is_file() and path.with_suffix(".png").stat().st_size > 0
        paints.append(paint)
    result = summarize(corpus, events, paints, read_json(args.paint_dir / "identity.json"),
                       read_json(args.annotations) if args.annotations else {}, file_hash(args.installed_exe),
                       model_events=[json.loads(line) for line in
                                     (args.paint_dir / "model_events.jsonl").read_text(encoding="utf-8").splitlines()
                                     if line.strip()] if (args.paint_dir / "model_events.jsonl").exists() else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.summary_output:
        # Keep permanent comparisons small; raw source/output and paint paths stay
        # in the two retained raw rounds, not the long-lived metric registry.
        summary = {key: value for key, value in result.items() if key not in {"scenes", "review_template"}}
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("outcome", "native_installed_verified", "metrics")}, ensure_ascii=False))
    return 0 if result["outcome"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
