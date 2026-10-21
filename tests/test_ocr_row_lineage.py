"""Source-row ownership and audit decisions, including missed punctuation."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pingyitong.components.capture.geometry import Rect
from pingyitong.components.ocr.ocr_engine import (
    OcrEngine,
    RapidOcrBackend,
    _prepare_ocr_image,
    _stretch_low_contrast,
)
from pingyitong.components.ocr.text_block import TextBlock
from pingyitong.services.text_preprocess import TextPreprocessor


def _block(text, y, *, x=20, width=280, height=16, confidence=.98, index=0):
    return TextBlock(text, Rect(x, y, width, height), confidence=confidence, index=index)


def test_small_faint_ink_is_not_left_in_narrow_dynamic_range():
    image = np.full((100, 300, 3), 248, dtype=np.uint8)
    image[20:35, 20:210:5] = 228
    processed = _prepare_ocr_image(image, 1)
    assert int(processed.max()) - int(processed.min()) > 128


def test_low_contrast_stretch_does_not_amplify_one_level_blank_noise():
    image = np.full((100, 300, 3), 248, dtype=np.uint8)
    image[::2, ::2] = 247
    assert _stretch_low_contrast(image) is image


def test_low_contrast_stretch_keeps_clear_pixels_unchanged():
    image = np.full((100, 300, 3), 248, dtype=np.uint8)
    image[20:35, 20:210:5] = 18
    assert _stretch_low_contrast(image) is image


def test_missing_stop_does_not_join_spaced_russian_instructions():
    original = [_block("Сохраняйте журнал до конца проверки", 100),
                _block("Откройте клапан после сигнала.", 134, width=260)]
    paragraphs, stats = TextPreprocessor().process(original)
    assert [p.text for p in paragraphs] == [b.text for b in original]
    assert any(e["reason"] == "visual_row_gap" for e in stats.decision_events)


def test_explicitly_disabled_wrap_merge_keeps_close_japanese_rows():
    original = [_block("青い表示が消えるまで", 100),
                _block("赤い弁を開けないでください。", 118)]
    paragraphs, stats = TextPreprocessor(merge_broken_lines=False).process(original)
    assert [p.text for p in paragraphs] == [b.text for b in original]
    assert any(e["reason"] == "wrap_merge_disabled" for e in stats.decision_events)


def test_true_wrapped_text_preserves_each_original_row_before_hyphen_repair():
    original = [_block("Vérifiez le trans-", 100),
                _block("fert avant de quitter.", 118, width=240)]
    paragraphs, stats = TextPreprocessor().process(original)
    assert len(paragraphs) == 1
    assert paragraphs[0].text == "Vérifiez le transfert avant de quitter."
    assert [line["text"] for line in paragraphs[0].source_lines] == [b.text for b in original]
    assert [line["rect"] for line in paragraphs[0].source_lines] == [list(b.box.to_sequence()) for b in original]
    assert [line["line_id"] for line in paragraphs[0].source_lines] == [0, 1]
    assert any(e["action"] == "merge" and e["reason"] == "wrapped_rows" for e in stats.decision_events)


def test_inline_merge_keeps_distinct_raw_sources_even_when_indexes_are_equal():
    original = [_block("Fermer", 100, width=40),
                _block("maintenant.", 100, x=64, width=75)]
    paragraphs, _ = TextPreprocessor().process(original)
    assert len(paragraphs) == 1
    assert [line["text"] for line in paragraphs[0].source_lines] == [b.text for b in original]
    assert len({line["source_id"] for line in paragraphs[0].source_lines}) == 2


def test_low_confidence_filter_reason_is_retained_with_original_coordinates():
    engine = OcrEngine(retain_uncertain=False, box_padding=0)
    candidate = _block("не удалять", 14, confidence=.15)
    assert engine._filter([candidate]) == []
    event = engine.decision_events[-1]
    assert event["action"] == "filter"
    assert event["reason"] == "low_confidence"
    assert event["inputs"][0]["text"] == candidate.text
    assert event["inputs"][0]["rect"] == list(candidate.box.to_sequence())


def test_empty_observation_cannot_reuse_the_previous_calls_decisions():
    engine = OcrEngine(retain_uncertain=False, box_padding=0)
    engine._filter([_block("не удалять", 14, confidence=.15)])
    assert engine.decision_events
    assert engine.recognize_image(None) == []
    assert engine.decision_events == []


def test_backend_empty_text_has_an_explicit_reason_and_local_source_box(monkeypatch):
    backend = RapidOcrBackend()
    backend._engine = SimpleNamespace(get_crop_img_list=lambda image, boxes: [image])
    monkeypatch.setattr(backend, "_detect", lambda engine, image: np.array([
        [[34, 34], [94, 34], [94, 48], [34, 48]],
    ], dtype=np.float32))
    monkeypatch.setattr(backend, "_recognize_crops", lambda engine, crops: [("", .3)])
    assert backend.recognize(np.full((100, 200, 3), 248, dtype=np.uint8)) == []
    event = backend.decision_events[0]
    assert event["reason"] == "empty_recognizer_text"
    assert event["inputs"][0]["rect"] == [10, 10, 60, 14]


def test_filtered_region_decision_restores_global_physical_coordinates():
    class Backend:
        def is_ready(self):
            return True

        def recognize(self, image):
            return [_block("Отмена", 7, x=5, confidence=.1)]

    engine = OcrEngine(backend=Backend(), enable_preprocess=False,
                       retain_uncertain=False, box_padding=0)
    frame = SimpleNamespace(origin=Rect(-100, 200, 400, 300), seq=17,
                            is_full_frame=False, monitor_index=2,
                            effective_regions=lambda: [Rect(20, 30, 300, 100)],
                            crop=lambda region: np.full((100, 300, 3), 248, dtype=np.uint8))
    result = engine.recognize_frame(frame)
    assert result.blocks == []
    assert result.decision_events[0]["inputs"][0]["rect"] == [-75, 237, 280, 16]


def test_noise_and_duplicate_removals_have_per_source_decisions():
    original = [_block("42", 10), _block("Relancer", 100, confidence=.8),
                _block("Relancer", 100, confidence=.99)]
    paragraphs, stats = TextPreprocessor().process(original)
    assert len(paragraphs) == 1
    assert stats.noise_removed == 1 and stats.duplicate_removed == 1
    assert any(e["action"] == "filter" and e["reason"] == "pure_numeric" for e in stats.decision_events)
    assert any(e["action"] == "deduplicate" and e["reason"] == "same_text_overlap" for e in stats.decision_events)


def test_contradictory_readings_are_audited_instead_of_silently_disappearing():
    original = [_block("Relaunch", 100, confidence=.7),
                _block("Relunch", 100, confidence=.91)]
    paragraphs, stats = TextPreprocessor().process(original)
    assert paragraphs[0].text == "Relunch"
    conflicts = [e for e in stats.decision_events if e["action"] == "conflict"]
    assert conflicts
    assert {i["text"] for e in conflicts for i in e["inputs"]} == {"Relaunch", "Relunch"}
    assert [line["text"] for line in paragraphs[0].source_lines] == ["Relunch"]
