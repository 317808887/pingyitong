"""真实 Qt 字体/OCR/本地模型/差分/原位绘制的同步 QA 驱动。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SENTENCES = [
    "Do not close before the backup finishes.",
    "The timeout is 15 seconds, not 50 seconds.",
    "Only administrators can delete these files.",
    "Save the report before you exit.",
    "Alice owns three red bicycles in Berlin.",
    "Bob has no bicycle and lives in Paris.",
    "The queue preserves the order of messages.",
    "The temperature is below zero today.",
    "Keep the cable connected until transfer ends.",
    "The report contains three pages.",
    "Close this dialog when saving is complete.",
    "Check the numbers before you continue.",
]
CHANGED_SENTENCE = "You can close after the backup finishes."


def normalized(text: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]", "", text).casefold()


def semantic_review(steps):
    """检查已知语料的条件、否定、数字和术语，保留失败实际输出。"""
    problems = []
    initial = {normalized(item["source"]): item["translation"] for item in steps[0]["snapshot"]}
    changed = {normalized(item["source"]): item["translation"] for item in steps[1]["snapshot"]}
    negative = initial.get(normalized(SENTENCES[0]), "")
    positive = changed.get(normalized(CHANGED_SENTENCE), "")
    numbers = initial.get(normalized(SENTENCES[1]), "")
    until = initial.get(normalized(SENTENCES[8]), "")
    from pingyitong.services.language import has_reversed_until_condition
    if has_reversed_until_condition(SENTENCES[0], negative, "zh"):
        problems.append({"case": "backup_before_direction", "source": SENTENCES[0],
                         "output": negative, "problem": "before became after"})
    if not any(token in negative for token in ("不要", "勿", "不能", "不可")):
        problems.append({"case": "negation", "source": SENTENCES[0], "output": negative,
                         "problem": "source forbids closing before backup completes"})
    if "备份" not in negative:
        problems.append({"case": "backup_term", "source": SENTENCES[0], "output": negative,
                         "problem": "backup was not preserved as the backup action"})
    if not all(token in numbers for token in ("15", "50")) or not any(token in numbers for token in ("不", "而非")):
        problems.append({"case": "15_not_50", "source": SENTENCES[1], "output": numbers,
                         "problem": "numbers or their negated comparison changed"})
    if any(token in until for token in ("结束后", "完成后", "结束之后", "完成之后")):
        problems.append({"case": "until_condition", "source": SENTENCES[8], "output": until,
                         "problem": "keep connected until transfer ends was changed to after transfer ends"})
    if positive == negative or "备份" not in positive or not any(token in positive for token in ("可以", "可")):
        problems.append({"case": "changed_permission", "source": CHANGED_SENTENCE, "output": positive,
                         "problem": "changed permission did not receive a new affirmative translation"})
    return {"checked": ["negation", "backup_term", "backup_before_direction", "15_not_50", "until_condition", "changed_permission"],
            "problems": problems, "passed": not problems,
            "scope": "known critical semantic relations; no universal accuracy claim"}


def qimage_array(image):
    import numpy as np

    raw = image.convertToFormat(image.Format.Format_RGBA8888)
    pointer = raw.bits()
    pointer.setsize(raw.sizeInBytes())
    return np.frombuffer(pointer, dtype="uint8").reshape(raw.height(), raw.bytesPerLine() // 4, 4)[:, :raw.width()].copy()


def fixture(sentences, marker: bool = False):
    from PyQt6.QtCore import QPointF, QRect
    from PyQt6.QtGui import QColor, QFont, QImage, QPainter

    image = QImage(960, 300, QImage.Format.Format_ARGB32)
    image.fill(QColor(245, 245, 245))
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    font = QFont("Segoe UI")
    font.setPixelSize(16)
    painter.setFont(font)
    painter.setPen(QColor(25, 25, 25))
    for index, sentence in enumerate(sentences):
        if not sentence:
            continue
        column, row = divmod(index, 6)
        painter.drawText(QPointF(20 + column * 480, 28 + row * 42), sentence)
    if marker:
        painter.fillRect(QRect(935, 276, 16, 16), QColor(20, 20, 20))
    painter.end()
    return image, qimage_array(image)[:, :, :3][:, :, ::-1].copy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/refresh_fix_20261012")
    parser.add_argument("--model", choices=("fast", "accurate"), default="fast")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"integrated_{args.model}"
    os.environ["PINGYITONG_HOME"] = str(args.output_dir / f"{prefix}_home")
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    import numpy as np
    from PyQt6.QtCore import QPoint, Qt
    from PyQt6.QtGui import QFont, QFontDatabase, QFontInfo, QImage, QPainter, QRawFont
    from PyQt6.QtWidgets import QApplication

    from pingyitong.components.capture import CaptureFrame, FrameDiffer
    from pingyitong.components.capture.geometry import Rect, ScreenInfo
    from pingyitong.components.llm import ModelPool, TranslationCache
    from pingyitong.components.ocr import create_ocr_engine
    from pingyitong.components.render.overlay_window import OverlayManager, OverlayWindow
    from pingyitong.config.schema import AppConfig
    from pingyitong.constants import MODEL_ACCURATE, MODEL_FAST
    from pingyitong.core import ModelScheduler
    from pingyitong.core.pipeline import Pipeline
    from pingyitong.services.glossary_service import Glossary
    from pingyitong.services.text_preprocess import TextPreprocessor
    from pingyitong.services.translation_service import TranslationService

    app = QApplication.instance() or QApplication([])
    model_id = MODEL_FAST if args.model == "fast" else MODEL_ACCURATE
    def source_hashes():
        return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (ROOT / "src/pingyitong/services/translation_service.py",
                             ROOT / "src/pingyitong/services/language.py",
                             ROOT / "src/pingyitong/core/pipeline.py",
                             ROOT / "src/pingyitong/components/ocr/ocr_engine.py")}
    font_loads = []
    for font_path in (Path(r"C:\Windows\Fonts\segoeui.ttf"), Path(r"C:\Windows\Fonts\msyh.ttc")):
        font_id = QFontDatabase.addApplicationFont(str(font_path))
        families = QFontDatabase.applicationFontFamilies(font_id) if font_id >= 0 else []
        font_loads.append({"path": str(font_path), "id": font_id, "families": families})
    fixture_font = QFont("Segoe UI")
    fixture_font.setPixelSize(16)
    if not QRawFont.fromFont(fixture_font).supportsCharacter("A"):
        raise RuntimeError("offscreen Qt could not load actual Windows Latin glyphs")
    config = AppConfig()
    config.api.enabled = False
    config.schedule.mode = "manual"
    config.schedule.active_model = model_id
    config.translation.source_language = "en"
    config.translation.target_language = "zh"
    config.translation.glossary_enabled = False
    # Keep complete sentences and their terminal punctuation under the normal
    # product paragraph rules; no QA stub replaces the preprocessor.
    ocr = create_ocr_engine(config.ocr, source_language="en")
    pool = ModelPool(config.model)
    cache = TranslationCache()
    cache.clear()
    translator = TranslationService(
        config=config, model_pool=pool, scheduler=ModelScheduler(config.schedule),
        cache=cache, preprocessor=TextPreprocessor.from_config(config), glossary=Glossary(), history=None,
    )
    bounds = Rect(0, 0, 960, 300)
    screen = ScreenInfo("QA synthetic Windows font", bounds, bounds, 1.0, True, 1)
    window = OverlayWindow(screen, config.style)
    # Real manager.apply and real window.paintEvent are used, while the QA
    # fixture's screen geometry avoids reading or controlling desktop apps.
    overlay = object.__new__(OverlayManager)
    overlay._style = config.style
    overlay._windows = {screen.name: window}
    overlay._screens = [screen]
    overlay._excluded_regions = []
    overlay._visible = True
    pipeline = Pipeline(config, SimpleNamespace(last_capture_cost=0.0), FrameDiffer(), ocr, translator, overlay)
    pipeline.set_transcript_enabled(True)
    report = {
        "scope": "synchronous QA driver; real FrameDiffer/_should_process, OCR, local model, _render_tick and OverlayWindow paint; no capture thread or desktop app automation",
        "model_id": model_id, "model_mode": args.model,
        "source_hashes_before": source_hashes(),
        "source_geometry": [960, 300], "requested_font": "Segoe UI", "font_pixels": 16,
        "resolved_font": QFontInfo(QFont("Segoe UI")).family(),
        "font_loads": font_loads,
        "steps": [], "passed": False,
    }
    model_calls = []
    original_translate = pool.translate
    def tracked_translate(*args, **kwargs):
        model_calls.append({"model_id": kwargs.get("model_id", args[0] if args else None),
                            "prompt": kwargs.get("prompt", args[1] if len(args) > 1 else "")})
        return original_translate(*args, **kwargs)
    pool.translate = tracked_translate

    steps = [
        ("initial", SENTENCES, False, 12),
        ("change_first", [CHANGED_SENTENCE, *SENTENCES[1:]], False, 12),
        ("unrelated_pixels", [CHANGED_SENTENCE, *SENTENCES[1:]], True, 12),
        ("delete_line", [CHANGED_SENTENCE, *SENTENCES[1:5], "", *SENTENCES[6:]], True, 11),
        ("all_blank", [""] * 12, False, 0),
    ]
    previous_outputs = {}
    try:
        loaded_at = time.perf_counter()
        if not pool.ensure_loaded(model_id):
            raise RuntimeError("real local model did not load")
        report["model_load_seconds"] = time.perf_counter() - loaded_at
        report["engine_class"] = type(pool.engine(model_id)).__name__
        if report["engine_class"] == "LayoutPreviewEngine":
            raise RuntimeError("preview engine is not a real translation model")
        report["ocr_ready"] = ocr.is_ready()
        if not report["ocr_ready"]:
            raise RuntimeError(ocr.error_message())

        for sequence, (label, sentences, marker, expected_count) in enumerate(steps, 1):
            step_started = time.perf_counter()
            source_image, pixels = fixture(sentences, marker)
            source_image.save(str(args.output_dir / f"{prefix}_{sequence}_{label}_source.png"))
            capture = CaptureFrame(sequence, time.perf_counter(), pixels, bounds, monitor_index=1)
            capture.__dict__["_translation_epoch"] = pipeline._translation_epoch
            before_calls = len(model_calls)
            should_process = pipeline._should_process(capture)
            result = {
                "label": label, "expected_count": expected_count, "should_process": should_process,
                "full_frame": capture.is_full_frame,
                "dirty_regions": [area.to_sequence() for area in capture.changed_regions],
                "ocr": [], "retry_regions": [], "problems": [],
            }
            if should_process:
                recognized = ocr.recognize_frame(capture)
                recognized.changed_regions = [area.to_global(bounds) for area in capture.changed_regions]
                old_boxes = [] if recognized.is_full_frame else [
                    box for box in pipeline._ocr_context_boxes
                    if not any(box.intersects(area) for area in recognized.changed_regions)
                ]
                pipeline._ocr_context_boxes = list(dict.fromkeys([
                    *old_boxes, *(block.box for block in recognized.blocks),
                ]))
                for key in ("_translation_retry", "_retry_generation"):
                    if key in capture.__dict__:
                        recognized.__dict__[key] = capture.__dict__[key]
                recognized.__dict__["_translation_epoch"] = pipeline._translation_epoch
                recognized.__dict__["_frame_stale"] = lambda frame=recognized: not pipeline._is_current_frame(frame)
                recognized.__dict__["_captured_at"] = capture.timestamp
                recognized.__dict__["_capture_image"] = pixels
                result["ocr"] = [{"text": block.text, "box": block.box.to_sequence(),
                                  "confidence": block.confidence} for block in recognized.blocks]
                rendered = translator.translate_frame(recognized)
                rendered.__dict__["_translation_epoch"] = pipeline._translation_epoch
                rendered.__dict__["_capture_image"] = pixels
                rendered.__dict__["_captured_at"] = capture.timestamp
                rendered.__dict__["_result_enqueued_at"] = time.perf_counter()
                for key in ("_translation_retry", "_retry_generation"):
                    if key in recognized.__dict__:
                        rendered.__dict__[key] = recognized.__dict__[key]
                rendered.changed_regions = list(recognized.changed_regions)
                result["retry_regions"] = [area.to_sequence() for area in rendered.retry_regions]
                result["source_items"] = [{"source": item.source_text, "box": item.box.to_sequence()}
                                          for item in rendered.source_items or []]
                pipeline._result_queue.put(rendered, seq=sequence)
                pipeline._render_tick()
                app.processEvents()

            snapshot = pipeline.snapshot_items()
            outputs = {normalized(item.source_text): item.text for item in snapshot}
            expected_sources = {normalized(sentence) for sentence in sentences if sentence}
            result["snapshot_count"] = len(snapshot)
            result["window_item_count"] = len(window._items)
            result["real_model_calls"] = len(model_calls) - before_calls
            result["snapshot"] = [{"source": item.source_text, "translation": item.text,
                                   "box": item.box.to_sequence(), "from_cache": item.from_cache}
                                  for item in snapshot]
            if len(snapshot) != expected_count or len(window._items) != expected_count:
                result["problems"].append("snapshot/window count is incomplete or has ghost items")
            if set(outputs) != expected_sources:
                result["problems"].append("OCR source set differs from complete expected sentences")
            if previous_outputs:
                unchanged = set(previous_outputs) & set(outputs)
                if any(previous_outputs[source] != outputs[source] for source in unchanged):
                    result["problems"].append("unchanged source translation changed")
            if label == "initial" and result["real_model_calls"] == 0:
                result["problems"].append("first frame did not invoke the real model")
            if label == "change_first":
                if normalized(SENTENCES[0]) in outputs or normalized(CHANGED_SENTENCE) not in outputs:
                    result["problems"].append("old first sentence survived or new sentence missing")
                if result["real_model_calls"] == 0:
                    result["problems"].append("changed sentence did not invoke the model")
                old_output = previous_outputs.get(normalized(SENTENCES[0]))
                if old_output == outputs.get(normalized(CHANGED_SENTENCE)):
                    result["problems"].append("changed sentence kept the old translation")
            if label == "unrelated_pixels" and outputs != previous_outputs:
                result["problems"].append("unrelated pixels changed translations")
            if label == "delete_line" and normalized(SENTENCES[5]) in outputs:
                result["problems"].append("deleted line ghost survived")

            layer = QImage(window.size(), QImage.Format.Format_ARGB32_Premultiplied)
            layer.fill(Qt.GlobalColor.transparent)
            window.render(layer)
            layer.save(str(args.output_dir / f"{prefix}_{sequence}_{label}_layer.png"))
            composite = source_image.copy()
            painter = QPainter(composite)
            painter.drawImage(QPoint(0, 0), layer)
            painter.end()
            composite.save(str(args.output_dir / f"{prefix}_{sequence}_{label}_overlay.png"))
            layer_pixels = qimage_array(layer)
            result["painted_pixels"] = int(np.count_nonzero(layer_pixels[:, :, 3]))
            if expected_count and result["painted_pixels"] == 0:
                result["problems"].append("GUI did not paint translated glyphs")
            if not expected_count and result["painted_pixels"]:
                result["problems"].append("blank frame retained painted ghost pixels")
            result["elapsed_ms"] = (time.perf_counter() - step_started) * 1000
            result["passed"] = not result["problems"]
            report["steps"].append(result)
            print(json.dumps({key: result[key] for key in (
                "label", "snapshot_count", "window_item_count", "real_model_calls",
                "painted_pixels", "elapsed_ms", "problems", "passed",
            )}, ensure_ascii=True), flush=True)
            previous_outputs = outputs

        report["model_calls"] = model_calls
        report["cache_stats"] = asdict(cache.stats())
        report["structure_passed"] = all(step["passed"] for step in report["steps"])
        report["semantic_review"] = semantic_review(report["steps"])
        report["passed"] = report["structure_passed"] and report["semantic_review"]["passed"]
    except Exception as error:
        report["exception"] = {"type": type(error).__name__, "message": str(error)}
        print(json.dumps(report["exception"], ensure_ascii=True), flush=True)
    finally:
        pool.shutdown()
        cache.close()
        window.close()
        pipeline._render_timer.stop()
        pipeline._status_timer.stop()
        report["source_hashes_after"] = source_hashes()
        report["source_unchanged_during_run"] = report["source_hashes_before"] == report["source_hashes_after"]
        (args.output_dir / f"{prefix}_dense_overlay.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
        )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
