"""Change our controlled fixture; observe actual installed-app paint files.

This never operates desktop controls or fabricates OCR/model/paint output.
Exit success means structural, readable delivery, not independent semantic
approval. A timeout and all partial observations are preserved for review.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

_spec = importlib.util.spec_from_file_location("native_summary", Path(__file__).with_name("summarize_native_acceptance.py"))
summary = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "control", "state", "paint-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--screen", required=True)
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Observation output exists; do not overwrite previous evidence")
    corpus = summary.read_json(args.corpus)
    scene = next(value for value in corpus["scenes"] if value["scene_id"] == args.scene)
    command = json.dumps(scene["control"], ensure_ascii=False, indent=2)
    pending = args.control.with_suffix(".pending")
    pending.write_text(command, encoding="utf-8")
    pending.replace(args.control)
    start = time.perf_counter()
    seen = set()
    observations = []
    fixture = None
    first_complete = None
    while time.perf_counter() - start < args.timeout:
        try:
            fixture = summary.read_json(args.state)
            if fixture.get("scene_id") != args.scene:
                time.sleep(0.1)
                continue
            appeared = fixture["paint_completed_monotonic_ns"] / 1e9
            for path in sorted(args.paint_dir.glob("paint_*.json")):
                if str(path) in seen:
                    continue
                try:
                    paint = summary.read_json(path)
                except (OSError, ValueError):
                    continue
                if paint.get("screen") != args.screen or paint.get("captured_at", 0) < appeared:
                    continue
                if not path.with_suffix(".png").is_file():
                    continue
                seen.add(str(path))
                items = paint.get("items", [])
                matched = summary.literal_matched_rows(scene["rows"], items)
                lines = summary.source_line_matches(scene["rows"], paint.get("source_lines", []))
                complete = (not items) if not scene["rows"] else (
                    len(matched) == len(scene["rows"]) and lines["distinct_line_structure_preserved"]
                    and summary.ready(items)
                    and all(item.get("font_px", 0) >= summary.READABLE_FONT_PX for item in items)
                )
                entry = {"file": str(path), "seq": paint.get("seq"),
                         "literal_source_rows": len(matched), "source_line_evidence": lines,
                         "scene_to_paint_ms": (paint["painted_at"] - appeared) * 1000,
                         "structural_readable_delivery": complete,
                         "runtime_observation": summary.observed_runtime(paint),
                         "dispositions": [item.get("disposition") for item in items],
                         "font_px": [item.get("font_px") for item in items]}
                observations.append(entry)
                if complete and first_complete is None:
                    first_complete = entry
            if first_complete is not None:
                break
        except (OSError, ValueError):
            pass
        time.sleep(0.1)
    result = {"run_id": corpus["run_id"], "scene_id": args.scene, "scenario_kind": scene.get("scenario_kind"),
              "fixture_generation": fixture.get("generation") if fixture else None,
              "first_structural_readable_delivery": first_complete,
              "independent_semantics_passed": None, "wait_seconds": time.perf_counter() - start,
              "observations": sorted(observations, key=lambda entry: entry["scene_to_paint_ms"])}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("scene_id", "first_structural_readable_delivery", "wait_seconds")}, ensure_ascii=False))
    return 0 if first_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
