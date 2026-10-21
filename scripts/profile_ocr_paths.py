"""OCR 检测/识别优化 A/B（真实 RapidOCR，带**文本正确性**核对）。

背景（读 rapidocr_onnxruntime 源码得到的事实）：

- ``limit_type='max'`` 时库会**忽略**我们传的 ``limit_side_len``，
  改按 ``get_preprocess`` 的分档（<960→960，<1500→1500，其余 2000）取值；
- 更关键的是 ``maybe_add_letterbox``：图像 ``宽/高 > width_height_ratio``（默认 8）
  时会**上下补边**到 ``max(w/8, 30)*2`` 高。字幕条（如 1600×84，比例 19）
  因此被补成 1600×400，检测面积膨胀到 4.7 倍——这就是"1600px 宽字幕
  检测耗时 68ms"的真正原因。

本脚本对比四种方案，并且**逐条核对识别文本**（省时间不能以识别错误为代价）：

A. 现状：默认补边 + ``limit_type='max'``
B. 关掉补边（``width_height_ratio`` 设为很大）
C. 检测前自行降采样到长边 ~800，识别框从**原分辨率**图裁剪
D. B + C

运行：python scripts/profile_ocr_paths.py
"""

from __future__ import annotations

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


def _run_plain(engine, image):
    result, elapse = engine(image)
    texts = [item[1] for item in result] if result else []
    return texts, (elapse[-1] if elapse else 0.0) * 1000


def _run_split(engine, image, det_long_side: int):
    """检测在降采样图上做，识别从原分辨率裁剪——识别精度不吃降采样。"""
    import cv2

    height, width = image.shape[:2]
    scale = min(1.0, det_long_side / float(max(height, width)))
    if scale < 1.0:
        small = cv2.resize(
            image, (max(32, int(round(width * scale))), max(32, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = image

    boxes, _ = engine.text_det(small)
    if boxes is None or len(boxes) == 0:
        return [], 0.0

    if scale < 1.0:
        boxes = boxes / scale

    crops = engine.get_crop_img_list(image, boxes)
    rec_res, _ = engine.text_rec(crops, False)
    return [text for text, _score in rec_res], 0.0


def main() -> int:
    from rapidocr_onnxruntime import RapidOCR

    cases = {
        "字幕 1 行 1600": (_render(["Stay behind me."], 1600), ["Stay behind me."]),
        "字幕 2 行 1600": (
            _render(["Stay behind me.", "We don't have much time."], 1600),
            ["Stay behind me.", "We don't have much time."],
        ),
        "字幕 2 行 1280": (
            _render(["Open the door before they see us.", "The reactor is overloading."], 1280),
            ["Open the door before they see us.", "The reactor is overloading."],
        ),
        "正文 4 行 900": (
            _render(
                [
                    "The quick brown fox jumps over the lazy dog.",
                    "Pack my box with five dozen liquor jugs.",
                    "How vexingly quick daft zebras jump!",
                    "Sphinx of black quartz, judge my vow.",
                ],
                900,
                24,
            ),
            [
                "The quick brown fox jumps over the lazy dog.",
                "Pack my box with five dozen liquor jugs.",
                "How vexingly quick daft zebras jump!",
                "Sphinx of black quartz, judge my vow.",
            ],
        ),
    }

    engines = {
        "A 现状（默认补边）": RapidOCR(
            intra_op_num_threads=8, use_cls=False,
            det_limit_side_len=736, det_limit_type="max",
        ),
        "B 关补边": RapidOCR(
            intra_op_num_threads=8, use_cls=False,
            det_limit_side_len=736, det_limit_type="max",
            width_height_ratio=10000,
        ),
    }

    print("=" * 84)
    print("OCR 方案对比（含识别文本核对，✓ 表示与渲染文本一致）")
    print("=" * 84)

    for label, (image, truth) in cases.items():
        print(f"\n--- {label}：{image.shape[1]}×{image.shape[0]} px ---")
        for engine_label, engine in engines.items():
            for split in (False, True):
                if split and engine_label.startswith("A"):
                    continue  # A+C 组合意义不大，跳过
                times, texts = [], []
                for _ in range(5):
                    started = time.perf_counter()
                    if split:
                        found, _ = _run_split(engine, image, 800)
                    else:
                        found, _ = _run_plain(engine, image)
                    times.append((time.perf_counter() - started) * 1000)
                    texts = found
                ok = [t.strip() for t in texts] == [t.strip() for t in truth]
                name = engine_label + (" + 检测降采样 800" if split else "")
                print(
                    f"  {name}: {statistics.median(times):.0f}ms  "
                    f"{'✓ 文本一致' if ok else '✗ 文本不符'}"
                    f"{'' if ok else '  识别到：' + repr(texts[:4])}"
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
