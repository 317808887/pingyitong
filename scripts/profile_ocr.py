"""OCR 成本分解与优化验证（真实 RapidOCR，逐项 A/B）。

目的：找到"1600px 宽字幕行"这类长宽比很大的输入上，det 与 rec 各占多少，
以及以下候选优化各能省多少：

1. ``det_limit_side_len`` 736 → 640 / 512（检测降采样更狠）
2. ``rec_img_shape`` 高度 48 → 48（基准）/ 32（更快但偏离训练分布）
3. 按长宽比分组做 rec（避免一个超宽行把整批都拉宽）
4. crop 级 OCR 缓存（同一行像素不变时直接复用上次结果）

运行：python scripts/profile_ocr.py
"""

from __future__ import annotations

import hashlib
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _render(lines: list[str], width: int, font_size: int = 40):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype("arial.ttf", font_size)
    line_height = int(font_size * 1.6)
    height = line_height * len(lines) + 20
    image = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    for index, line in enumerate(lines):
        draw.text((16, 10 + index * line_height), line, fill=(255, 255, 255), font=font)
    return np.array(image)


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def main() -> int:
    from rapidocr_onnxruntime import RapidOCR

    scenarios = {
        "字幕 1 行 1600px": _render(["Stay behind me."], 1600),
        "字幕 2 行 1600px": _render(["Stay behind me.", "We don't have much time."], 1600),
        "字幕 2 行 1280px": _render(["Stay behind me.", "We don't have much time."], 1280),
        "正文 4 行 900px": _render(
            [
                "The quick brown fox jumps over the lazy dog.",
                "Pack my box with five dozen liquor jugs.",
                "How vexingly quick daft zebras jump!",
                "Sphinx of black quartz, judge my vow.",
            ],
            900,
            24,
        ),
    }

    print("=" * 78)
    print("OCR 成本分解（det / rec 分离，真实 RapidOCR）")
    print("=" * 78)

    for label, image in scenarios.items():
        print(f"\n--- {label}：{image.shape[1]}×{image.shape[0]} px ---")
        for limit in (736, 640, 512):
            engine = RapidOCR(
                intra_op_num_threads=8, use_cls=False,
                det_limit_side_len=limit, det_limit_type="max",
            )
            # 预热
            engine(image)

            totals, dets, recs, boxes_count = [], [], [], []
            for _ in range(5):
                t0 = time.perf_counter()
                result, elapse = engine(image)
                totals.append((time.perf_counter() - t0) * 1000)
                if elapse and len(elapse) >= 2:
                    dets.append(elapse[0] * 1000)
                    recs.append(elapse[-1] * 1000)
                boxes_count.append(len(result) if result else 0)

            print(
                f"  limit={limit}: 总 {_median(totals):.0f}ms "
                f"(det {_median(dets):.0f}ms, rec {_median(recs):.0f}ms, "
                f"{boxes_count[-1]} 块)"
            )

    # ---- rec 长宽比分组的效果 ----
    print("\n" + "=" * 78)
    print("rec 分组对比：混合长短行时，是否按长宽比分组")
    print("=" * 78)
    mixed = _render(
        [
            "A very long subtitle line that spans the entire width of the screen area.",
            "Short.",
            "Another long line that also spans most of the available horizontal space.",
            "Tiny one.",
        ],
        1600,
    )
    engine = RapidOCR(intra_op_num_threads=8, use_cls=False,
                      det_limit_side_len=736, det_limit_type="max")
    engine(mixed)
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        engine(mixed)
        times.append((time.perf_counter() - t0) * 1000)
    print(f"  默认（按检测顺序、每 6 条一批）：总 {_median(times):.0f}ms")

    # 手动分组：先 det，再按长宽比排序后调用 text_rec
    times_grouped, times_plain = [], []
    for _ in range(5):
        boxes, _ = engine.text_det(mixed)
        crops = engine.get_crop_img_list(mixed, boxes)
        t0 = time.perf_counter()
        engine.text_rec(crops)
        times_plain.append((time.perf_counter() - t0) * 1000)

        order = sorted(range(len(crops)), key=lambda i: crops[i].shape[1] / crops[i].shape[0])
        sorted_crops = [crops[i] for i in order]
        t0 = time.perf_counter()
        engine.text_rec(sorted_crops)
        times_grouped.append((time.perf_counter() - t0) * 1000)
    print(f"  rec 直接调用（原顺序）：{_median(times_plain):.0f}ms")
    print(f"  rec 按长宽比分组后调用：{_median(times_grouped):.0f}ms")

    # ---- crop 级缓存收益 ----
    print("\n" + "=" * 78)
    print("crop 级缓存：两行字幕只改一行时的收益")
    print("=" * 78)
    frame_a = _render(["Stay behind me.", "We don't have much time."], 1600)
    frame_b = _render(["Stay behind me.", "Open the door before they see us."], 1600)
    engine(frame_a)

    def ocr_with_crop_cache(image, cache: dict[str, str]) -> tuple[float, int, int]:
        started = time.perf_counter()
        boxes, _ = engine.text_det(image)
        if boxes is None:
            return (time.perf_counter() - started) * 1000, 0, 0
        crops = engine.get_crop_img_list(image, boxes)
        hits = 0
        misses = []
        miss_index = []
        for crop in crops:
            digest = hashlib.blake2b(crop.tobytes(), digest_size=8).hexdigest()
            if digest in cache:
                hits += 1
            else:
                misses.append(crop)
                miss_index.append(digest)
        if misses:
            engine.text_rec(misses)
        for digest in miss_index:
            cache[digest] = "1"
        return (time.perf_counter() - started) * 1000, hits, len(crops)

    cache: dict[str, str] = {}
    for label, image in (("首帧（全 miss）", frame_a), ("次帧（改了一行）", frame_b)):
        values = []
        for _ in range(3):
            cost, hits, total = ocr_with_crop_cache(image, cache)
            values.append(cost)
        print(f"  {label}: {_median(values):.0f}ms（命中 {hits}/{total} 行）")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
