"""Prepare a fresh test home and controlled scene; never operate desktop UI."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from pingyitong.config.schema import AppConfig
    from pingyitong.constants import MODEL_ACCURATE, MODEL_FAST

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--model", choices=("fast", "accurate"), default="accurate")
    args = parser.parse_args()
    if args.home.exists():
        raise SystemExit("Test home already exists; use a fresh directory to keep prior evidence")
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    scene = next(item for item in corpus["scenes"] if item["scene_id"] == args.scene)
    config = AppConfig()
    config.ui.start_minimized = False
    config.ui.first_run_done = True
    config.ui.floating_bar_pos = [1320, 40]
    config.capture.mode = "region"
    config.capture.region = scene["control"]["geometry"]
    config.capture.monitors = [1]
    config.capture.monitor_interval_ms = 50
    config.translation.source_language = scene["source_language"]
    config.translation.target_language = scene["target_language"]
    config.translation.review_policy = "always"
    config.translation.local_only = True
    config.schedule.mode = "manual"
    config.schedule.active_model = MODEL_FAST if args.model == "fast" else MODEL_ACCURATE
    config.model.fast.preload = args.model == "fast"
    config.model.accurate.preload = args.model == "accurate"
    config.model.fast.temperature = config.model.accurate.temperature = 0
    config.prefetch.enabled = config.api.enabled = False
    config.hotkey.toggle_translation = "Ctrl+Alt+F8"
    config.hotkey.switch_model = "Ctrl+Alt+F9"
    config.hotkey.copy_result = "Ctrl+Alt+F10"
    config.hotkey.toggle_overlay = "Ctrl+Alt+F11"
    args.home.mkdir(parents=True)
    (args.home / "config.json").write_text(json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8")
    (args.home / "control.json").write_text(json.dumps(scene["control"], ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"home": str(args.home), "run_id": corpus["run_id"],
                      "scene": args.scene, "source": scene["source_language"],
                      "target": scene["target_language"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
