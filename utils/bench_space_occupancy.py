"""Ad-hoc microbenchmark for ObservedSpaceDetector (not part of test suite)."""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402

from pingyitong.components.capture.geometry import Rect  # noqa: E402
from pingyitong.components.render.space_occupancy import ObservedSpaceDetector  # noqa: E402


def scene(height: int, width: int) -> np.ndarray:
    image = np.full((height, width, 3), 248, dtype=np.uint8)
    for row in range(20, height - 20, 24):
        image[row:row + 12, 30:width - 30] = 40
    return image


def measure(size: tuple[int, int], rounds: int = 5) -> None:
    height, width = size
    image = scene(height, width)
    origin = Rect(0, 0, width, height)
    sources = [Rect(30, 20 + i * 24, 200, 12) for i in range(0, (height - 40) // 24, 2)]
    detector = ObservedSpaceDetector()
    cold = []
    warm = []
    for _ in range(rounds):
        detector = ObservedSpaceDetector()
        started = time.perf_counter()
        result = detector.analyze(image, origin, sources)
        cold.append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        detector.analyze(image, origin, sources)
        warm.append((time.perf_counter() - started) * 1000)
    print(
        f"{width}x{height} cold p50={np.median(cold):6.1f}ms min={min(cold):6.1f}ms "
        f"max={max(cold):6.1f}ms warm p50={np.median(warm):6.1f}ms "
        f"obstacles={len(result.obstacles)} mode={result.evidence['mode']}"
    )


for size in ((720, 1280), (1080, 1920), (1440, 2560), (1080, 3840)):
    measure(size)
