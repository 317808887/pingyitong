"""密集小字的检测覆盖、裁剪边距与完整文字保留回归。"""
from __future__ import annotations

from unittest import mock

import numpy as np

from pingyitong.components.capture.geometry import Rect
from pingyitong.components.ocr.ocr_engine import (
    OcrEngine,
    RapidOcrBackend,
    _expand_recognition_boxes,
    _prepare_ocr_image,
    _prepare_recognition_crop,
)
from pingyitong.components.ocr.text_block import TextBlock


class _Detector:
    def __init__(self):
        self.shapes = []

    def text_det(self, image):
        self.shapes.append(image.shape)
        count = 2 if image.shape[1] > 1000 else 1
        boxes = np.array([[[10, 10 + 30 * index], [200, 10 + 30 * index],
                           [200, 24 + 30 * index], [10, 24 + 30 * index]]
                          for index in range(count)], dtype="float32")
        return boxes, 0.0

    @staticmethod
    def sorted_boxes(boxes):
        return list(boxes)


def test_dense_document_keeps_detection_resolution_when_preview_would_be_nonempty():
    detector = _Detector()
    backend = RapidOcrBackend()
    boxes = backend._detect(detector, np.zeros((1080, 1920, 3), dtype="uint8"))
    assert detector.shapes == [(1080, 1920, 3)]
    assert len(boxes) == 2


def test_thin_subtitle_retains_fast_detection_and_original_coordinates():
    detector = _Detector()
    backend = RapidOcrBackend()
    boxes = backend._detect(detector, np.zeros((88, 1600, 3), dtype="uint8"))
    assert detector.shapes == [(44, 800, 3)]
    assert boxes[0][0].tolist() == [20.0, 20.0]


def test_document_resolution_remains_bounded_for_large_desktop():
    detector = _Detector()
    backend = RapidOcrBackend()
    backend._detect(detector, np.zeros((2160, 3840, 3), dtype="uint8"))
    assert max(detector.shapes[0][:2]) <= backend.DOCUMENT_DET_LONG_SIDE


def test_small_text_crop_margin_does_not_double_height_or_capture_adjacent_rows():
    boxes = np.array([[[10, 10], [200, 10], [200, 24], [10, 24]]], dtype="float32")
    grown = _expand_recognition_boxes(boxes, 4)
    assert grown[0, :, 1].min() == 8
    assert grown[0, :, 1].max() == 26


def test_large_subtitle_retains_four_pixel_tail_padding():
    boxes = np.array([[[10, 10], [200, 10], [200, 58], [10, 58]]], dtype="float32")
    grown = _expand_recognition_boxes(boxes, 4)
    assert np.isclose(grown[0, :, 0].min(), 6)
    assert np.isclose(grown[0, :, 1].min(), 6)
    assert np.isclose(grown[0, :, 1].max(), 62)


def test_200_credible_lines_survive_old_120_budget_in_original_order():
    engine = OcrEngine(max_blocks_per_frame=120, box_padding=0)
    blocks = [TextBlock(f"line {index}", Rect(0, index * 18, 200, 14),
                        confidence=.99 if index % 2 else .90, index=index)
              for index in range(200)]
    accepted = engine._filter(blocks)
    assert len(accepted) == 200
    assert [block.text for block in accepted] == [f"line {index}" for index in range(200)]


def test_low_confidence_candidate_is_retained_and_marked_without_reordering():
    engine = OcrEngine(max_blocks_per_frame=1, box_padding=0)
    blocks = [TextBlock("first", Rect(0, 0, 200, 14), confidence=.9),
              TextBlock("noise", Rect(0, 18, 200, 14), confidence=.2),
              TextBlock("last", Rect(0, 36, 200, 14), confidence=.8)]
    result = engine._filter(blocks)
    assert [block.text for block in result] == ["first", "noise", "last"]
    assert [block.ocr_uncertain for block in result] == [False, True, False]


def test_high_contrast_tiny_glyph_pixels_are_preserved():
    image = np.full((100, 200, 3), 245, dtype="uint8")
    image[8:24, 20:180] = 35
    with mock.patch("pingyitong.components.ocr.ocr_engine.enhance_for_ocr") as enhance:
        assert _prepare_ocr_image(image, 1.0) is image
        enhance.assert_not_called()


def test_low_contrast_image_still_gets_existing_enhancement():
    image = np.full((100, 200, 3), 245, dtype="uint8")
    image[8:24, 20:180] = 235
    sentinel = object()
    with mock.patch("pingyitong.components.ocr.ocr_engine.enhance_for_ocr", return_value=sentinel) as enhance:
        assert _prepare_ocr_image(image, 1.5) is sentinel
        enhance.assert_called_once_with(image, upscale=1.5)


def test_high_contrast_explicit_upscale_still_applies():
    image = np.full((100, 200, 3), 245, dtype="uint8")
    image[8:24, 20:180] = 35
    assert _prepare_ocr_image(image, 2.0).shape == (200, 400, 3)


def test_tiny_recognition_crop_upscales_without_changing_aspect_ratio():
    crop = np.zeros((18, 180, 3), dtype="uint8")
    assert _prepare_recognition_crop(crop).shape == (54, 540, 3)


def test_large_subtitle_crop_is_not_upscaled_again():
    crop = np.zeros((48, 400, 3), dtype="uint8")
    assert _prepare_recognition_crop(crop) is crop


def test_tiny_crop_cache_hit_skips_resampling_and_model_call():
    class Engine:
        def text_rec(self, crops, _return_word_box):
            return [("small text", .99)] * len(crops), 0.0
    backend = RapidOcrBackend()
    crop = np.zeros((18, 100, 3), dtype="uint8")
    with mock.patch("pingyitong.components.ocr.ocr_engine._prepare_recognition_crop",
                    wraps=_prepare_recognition_crop) as prepare:
        assert backend._recognize_crops(Engine(), [crop]) == [("small text", .99)]
        assert backend._recognize_crops(Engine(), [crop.copy()]) == [("small text", .99)]
        assert prepare.call_count == 1
