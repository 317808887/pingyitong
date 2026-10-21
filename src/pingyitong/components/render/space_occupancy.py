"""Conservative local-pixel barriers for in-place translation expansion.

Only source rectangles are known to contain text that may be covered. A
dominant near-uniform background permits whitespace outside them; unknown
contrasting pixels protect their whole tiles. A photograph/gradient without
a dominant background keeps all unknown pixels protected. This is a bounded
heuristic, not recognition of arbitrary images or proof of empty content.
"""

from __future__ import annotations

import math
import time
import zlib
from dataclasses import dataclass

import numpy as np

from ..capture.geometry import Rect

TILE_PX = 16
COLOUR_DIFFERENCE = 8
MIN_BACKGROUND_SHARE = 0.60


@dataclass(frozen=True)
class SpaceObservation:
    obstacles: list[Rect]
    evidence: dict


class ObservedSpaceDetector:
    """One-frame local memory cache; pixels are never written to disk here."""

    def __init__(self) -> None:
        self._key: tuple | None = None
        self._image: np.ndarray | None = None
        self._result: SpaceObservation | None = None

    def analyze(
        self, image: np.ndarray | None, origin: Rect, source_rects: list[Rect],
    ) -> SpaceObservation:
        started = time.perf_counter()
        sources = sorted({
            (rect.expand(1).clamp(origin)).to_ltrb()
            for rect in source_rects if not rect.clamp(origin).is_empty
        })
        local_sources = [Rect.from_ltrb(*rect).translate(-origin.x, -origin.y) for rect in sources]
        valid_image = (
            isinstance(image, np.ndarray) and image.dtype == np.uint8 and image.ndim == 3
            and image.shape == (origin.height, origin.width, 3)
            and not origin.is_empty
        )
        if not valid_image:
            result = self._fallback(origin, local_sources, "missing_or_mismatched_capture_pixels")
            return SpaceObservation(result.obstacles, {
                **result.evidence, "cache_hit": False,
                "analysis_ms": (time.perf_counter() - started) * 1000,
            })
        pixels = np.ascontiguousarray(image)
        # CRC is an inexpensive first filter, never a proof of equal pixels.
        # Keeping a private copy also detects in-place mutation of a caller's
        # array; an equal checksum still requires exact array equality.
        checksum = zlib.crc32(memoryview(pixels))
        key = (origin, tuple(sources), pixels.shape, checksum)
        if (key == self._key and self._image is not None and self._result is not None
                and np.array_equal(pixels, self._image)):
            return SpaceObservation(self._result.obstacles, {
                **self._result.evidence, "cache_hit": True,
                "analysis_ms": (time.perf_counter() - started) * 1000,
            })

        valid = np.ones(pixels.shape[:2], dtype=bool)
        for source in local_sources:
            valid[source.top:source.bottom, source.left:source.right] = False
        step = max(1, math.ceil(math.sqrt(valid.size / 60000)))
        samples = pixels[::step, ::step][valid[::step, ::step]]
        background, share = _background(samples)
        if background is None or share < MIN_BACKGROUND_SHARE:
            result = self._fallback(origin, local_sources, "nonuniform_background_conservative")
            result = SpaceObservation(result.obstacles, {**result.evidence, "background_share": share})
        else:
            foreground = np.zeros(valid.shape, dtype=bool)
            for channel in range(3):
                values = pixels[:, :, channel]
                colour = int(background[channel])
                foreground |= (values.astype(np.int16) < colour - COLOUR_DIFFERENCE)
                foreground |= (values.astype(np.int16) > colour + COLOUR_DIFFERENCE)
            foreground &= valid
            tile_rows = math.ceil(origin.height / TILE_PX)
            tile_columns = math.ceil(origin.width / TILE_PX)
            padded = np.zeros((tile_rows * TILE_PX, tile_columns * TILE_PX), dtype=bool)
            padded[:origin.height, :origin.width] = foreground
            blocked = padded.reshape(tile_rows, TILE_PX, tile_columns, TILE_PX).any(axis=(1, 3))
            merged = _tile_rectangles(blocked, origin.width, origin.height)
            barriers = _subtract_sources(merged, local_sources)
            result = SpaceObservation(
                [rect.translate(origin.x, origin.y) for rect in barriers],
                {"mode": "dominant_uniform_background", "background_share": share,
                 "background_bgr": [int(value) for value in background],
                 "tile_px": TILE_PX, "colour_difference": COLOUR_DIFFERENCE,
                 "unknown_foreground_pixels": int(foreground.sum()),
                 "blocked_tiles": int(blocked.sum()), "obstacle_rectangles": len(barriers),
                 "source_exclusion_padding_px": 1},
            )
        self._key = key
        self._image = pixels.copy()
        self._result = result
        return SpaceObservation(result.obstacles, {
            **result.evidence, "cache_hit": False,
            "analysis_ms": (time.perf_counter() - started) * 1000,
        })

    @staticmethod
    def _fallback(origin: Rect, sources: list[Rect], reason: str) -> SpaceObservation:
        barriers = _subtract_sources([Rect(0, 0, origin.width, origin.height)], sources)
        return SpaceObservation(
            [rect.translate(origin.x, origin.y) for rect in barriers],
            {"mode": reason, "background_share": None, "tile_px": TILE_PX,
             "colour_difference": COLOUR_DIFFERENCE, "obstacle_rectangles": len(barriers),
             "source_exclusion_padding_px": 1},
        )


def _background(samples: np.ndarray) -> tuple[np.ndarray | None, float]:
    if not len(samples):
        return None, 0.0
    quantized = samples.astype(np.uint32) >> 3
    keys = quantized[:, 0] | (quantized[:, 1] << 5) | (quantized[:, 2] << 10)
    counts = np.bincount(keys, minlength=32768)
    key = int(counts.argmax())
    matching = keys == key
    return np.median(samples[matching], axis=0), float(counts[key] / len(samples))


def _tile_rectangles(blocked: np.ndarray, width: int, height: int) -> list[Rect]:
    """Merge adjacent tile runs so a large photograph is not 1000 obstacles."""
    active: dict[tuple[int, int], Rect] = {}
    result = []
    for row_index, row in enumerate(blocked):
        padded = np.pad(row.astype(np.int8), (1, 1))
        edges = np.diff(padded)
        spans = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True))
        current = {}
        for left_tile, right_tile in spans:
            key = (int(left_tile), int(right_tile))
            previous = active.pop(key, None)
            if previous is None:
                previous = Rect(int(left_tile) * TILE_PX, row_index * TILE_PX,
                                min(width, int(right_tile) * TILE_PX) - int(left_tile) * TILE_PX,
                                min(TILE_PX, height - row_index * TILE_PX))
            else:
                previous = Rect(previous.x, previous.y, previous.width,
                                min(height, (row_index + 1) * TILE_PX) - previous.y)
            current[key] = previous
        result.extend(active.values())
        active = current
    result.extend(active.values())
    return result


def _subtract_sources(rectangles: list[Rect], sources: list[Rect]) -> list[Rect]:
    result = rectangles
    for source in sources:
        next_result = []
        for rectangle in result:
            overlap = rectangle.intersection(source)
            if overlap.is_empty:
                next_result.append(rectangle)
                continue
            for piece in (
                Rect.from_ltrb(rectangle.left, rectangle.top, rectangle.right, overlap.top),
                Rect.from_ltrb(rectangle.left, overlap.bottom, rectangle.right, rectangle.bottom),
                Rect.from_ltrb(rectangle.left, overlap.top, overlap.left, overlap.bottom),
                Rect.from_ltrb(overlap.right, overlap.top, rectangle.right, overlap.bottom),
            ):
                if not piece.is_empty:
                    next_result.append(piece)
        result = next_result
    return result
