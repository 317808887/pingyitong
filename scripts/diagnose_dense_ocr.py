"""真实 OCR 密集小字召回诊断；原始 OCR 与上限过滤分开记账。"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _normalize(text: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]", "", text).casefold()


def dense_fixture(font_size: int = 14, rows: int = 50, columns: int = 4):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (1920, 1080), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", font_size)
    lines = []
    for column in range(columns):
        for row in range(rows):
            text = f"Panel {column + 1:02d} item {row + 1:03d}: save backup file."
            lines.append(text)
            draw.text((20 + column * (1920 // columns), 12 + row * 20), text, font=font, fill=(35, 35, 35))
    return np.asarray(image)[:, :, ::-1].copy(), lines


def run_case(backend, image, expected: list[str], name: str, output: Path):
    det_ms = 0.0
    detector = backend._engine.text_det
    def timed_det(patch):
        nonlocal det_ms
        started = time.perf_counter()
        result = detector(patch)
        det_ms += (time.perf_counter() - started) * 1000
        return result
    backend._engine.text_det = timed_det
    started = time.perf_counter()
    try:
        blocks = backend.recognize(image)
    finally:
        backend._engine.text_det = detector
    elapsed = (time.perf_counter() - started) * 1000
    normalized = {_normalize(block.text) for block in blocks}
    exact = [text for text in expected if _normalize(text) in normalized]
    joined = _normalize(" ".join(block.text for block in blocks))
    anchors = [text for text in expected if _normalize(text) in joined]
    result = {
        "name": name, "shape": list(image.shape), "elapsed_ms": elapsed,
        "det_ms": det_ms, "raw_blocks": len(blocks), "expected_lines": len(expected),
        "exact_lines": len(exact), "anchor_matches": len(anchors),
        "missing_exact": [text for text in expected if text not in exact],
        "blocks": [{"text": block.text, "confidence": block.confidence,
                    "box": [block.box.x, block.box.y, block.box.width, block.box.height]}
                   for block in blocks],
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in {"blocks", "missing_exact"}}, ensure_ascii=True), flush=True)
    return blocks, result


def main() -> int:
    import cv2

    from pingyitong.components.ocr.ocr_engine import OcrEngine, RapidOcrBackend

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/refresh_fix_20261012")
    parser.add_argument("--phase", default="before")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--prototype-tiles", action="store_true")
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    backend = RapidOcrBackend(threads=8)
    if not backend.is_ready():
        raise RuntimeError(backend.error)
    reports = []
    cases = []
    for font_size in (12, 14, 18):
        image, expected = dense_fixture(font_size)
        cv2.imwrite(str(args.output_dir / f"dense_font_{font_size}.png"), image)
        cases.append((f"dense_{font_size}", image, expected))
    if args.image:
        image = cv2.imread(str(args.image))
        if image is None:
            raise RuntimeError("用户截图不可读取")
        cases.append(("user_screenshot", image, [
            "屏译通 0.6.28 验收与迭代报告", "最终完整回归", "1204 passed",
            "1074 subtests passed", "72.77", "已修复的问题", "translation_final.json",
            "final_real_e2e.json", "静态检查", "打包模块",
        ]))

    if args.production:
        ocr = OcrEngine(backend=backend, enable_preprocess=True)
        original_recognize = backend.recognize
        raw_count = 0
        def tapped_recognize(image):
            nonlocal raw_count
            blocks = original_recognize(image)
            raw_count = len(blocks)
            return blocks
        backend.recognize = tapped_recognize
        production_reports = []
        for name, image, expected in cases:
            backend._crop_cache.clear()
            samples = []
            for attempt in range(max(1, args.repeat)):
                started = time.perf_counter()
                blocks = ocr.recognize_image(image)
                elapsed = (time.perf_counter() - started) * 1000
                texts = {_normalize(block.text) for block in blocks}
                joined = _normalize(" ".join(block.text for block in blocks))
                samples.append({
                    "attempt": attempt, "raw_blocks": raw_count, "kept": len(blocks),
                    "dropped": raw_count - len(blocks), "elapsed_ms": elapsed,
                    "exact_lines": sum(_normalize(text) in texts for text in expected),
                    "anchor_matches": sum(_normalize(text) in joined for text in expected),
                })
            result = {
                "name": name, "shape": list(image.shape), "expected_lines": len(expected),
                "preprocess": True, "upscale": 1.0, "min_confidence": .5,
                "min_box_height": 6, "max_blocks_budget": 120, "samples": samples,
                "missing_exact": [text for text in expected if _normalize(text) not in texts],
                "blocks": [{"text": block.text, "confidence": block.confidence,
                            "box": [block.box.x, block.box.y, block.box.width, block.box.height]}
                           for block in blocks],
            }
            production_reports.append(result)
            print(json.dumps({key: value for key, value in result.items()
                              if key not in {"blocks", "missing_exact"}}, ensure_ascii=True), flush=True)
        (args.output_dir / f"ocr_{args.phase}_production_summary.json").write_text(
            json.dumps(production_reports, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        return 0

    default_side = backend.DET_LONG_SIDE
    original_detect = backend._detect
    def tiled_detect(engine, image):
        import numpy as np
        height, width = image.shape[:2]
        boxes = []
        for top in range(0, height, 448):
            for left in (0,):
                tile = image[top:min(top + 512, height), left:width]
                detected, _ = engine.text_det(tile)
                if detected is None:
                    continue
                for box in detected:
                    shifted = np.asarray(box, dtype="float32") + [left, top]
                    x1, y1 = shifted[:, 0].min(), shifted[:, 1].min()
                    x2, y2 = shifted[:, 0].max(), shifted[:, 1].max()
                    if (top > 0 and y1 < top + 24) or (
                        top + 512 < height and y2 > top + 512 - 24
                    ):
                        continue
                    # overlapping tiles keep the larger complete detection.
                    replaced = False
                    for index, prior in enumerate(boxes):
                        px1, py1 = prior[:, 0].min(), prior[:, 1].min()
                        px2, py2 = prior[:, 0].max(), prior[:, 1].max()
                        overlap = max(0, min(x2, px2) - max(x1, px1)) * max(0, min(y2, py2) - max(y1, py1))
                        area, parea = (x2-x1)*(y2-y1), (px2-px1)*(py2-py1)
                        if overlap / max(1, min(area, parea)) > .6:
                            if area > parea:
                                boxes[index] = shifted
                            replaced = True
                            break
                    if not replaced:
                        boxes.append(shifted)
        return np.asarray(engine.sorted_boxes(np.asarray(boxes)), dtype="float32") if boxes else None
    for name, image, expected in cases:
        modes = [("default", default_side), ("native", 4000)]
        if args.prototype_tiles:
            modes.append(("tiled", 4000))
        for mode, side in modes:
            backend.DET_LONG_SIDE = side
            backend._detect = tiled_detect if mode == "tiled" else original_detect
            backend._crop_cache.clear()
            blocks, result = run_case(
                backend, image, expected, f"{name}_{mode}",
                args.output_dir / f"ocr_{args.phase}_{name}_{mode}.json",
            )
            filtered = OcrEngine(backend=backend)._filter(blocks)
            result["accepted_default_filter"] = len(filtered)
            result["filter_dropped"] = len(blocks) - len(filtered)
            filtered_texts = {_normalize(block.text) for block in filtered}
            result["accepted_exact_lines"] = sum(_normalize(text) in filtered_texts for text in expected)
            result["filter_parameters"] = {"min_confidence": .5, "min_box_height": 6,
                                           "max_blocks_budget": 120}
            print(json.dumps({key: result[key] for key in (
                "name", "raw_blocks", "accepted_default_filter", "filter_dropped",
                "accepted_exact_lines", "filter_parameters",
            )}, ensure_ascii=True), flush=True)
            reports.append(result)
    backend.DET_LONG_SIDE = default_side
    (args.output_dir / f"ocr_{args.phase}_summary.json").write_text(
        json.dumps([{key: value for key, value in item.items() if key != "blocks"}
                    for item in reports], ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
