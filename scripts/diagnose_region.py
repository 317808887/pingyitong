"""诊断：框选区域模式下，捕获范围与渲染坐标是否正确。

背景：``ScreenCapturer.capture_region`` 的文档说参数是**物理坐标**，
而 ``RegionSelector`` 给出并存入配置的是**逻辑坐标**（已按 Windows
缩放比例折算）。若二者直接对接，捕获到的就不是用户框选的那块屏幕。

本脚本用当前配置各跑一遍"按逻辑坐标捕获"与"按物理坐标捕获"，
打印两者实际覆盖的屏幕位置与识别到的文字，从而判断偏差。

运行：
    python scripts/diagnose_region.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    # 显示器枚举依赖 QGuiApplication，必须先建一个（无需显示窗口）
    from PyQt6.QtGui import QGuiApplication

    _app = QGuiApplication.instance() or QGuiApplication([])

    from pingyitong.components.capture.geometry import Rect, list_screens
    from pingyitong.components.capture.screen_capture import ScreenCapturer
    from pingyitong.components.ocr.ocr_engine import OcrEngine
    from pingyitong.utils import paths

    config_file = Path(paths.config_path())
    data = json.loads(config_file.read_text(encoding="utf-8"))
    capture = data.get("capture", {})
    stored = capture.get("region")

    print("配置文件:", config_file)
    print("capture.mode =", capture.get("mode"))
    print("capture.region（存入的是逻辑坐标）= ", stored)
    print()

    screens = list_screens()
    for i, s in enumerate(screens):
        print(f"  显示器[{i}] {s.label()}  逻辑={s.logical}  物理={s.physical}")
    print()

    if not stored:
        print("未设置区域，退出")
        return 1

    logical = Rect.from_sequence(stored)

    # 用户框选区域换算到物理坐标（正确做法）
    from pingyitong.core.coordinate import CoordinateMapper

    mapper = CoordinateMapper()
    physical = mapper.capture_region_physical(logical)
    print("逻辑坐标 =", logical)
    print("换算后的物理坐标 =", physical)
    print()

    ocr = OcrEngine()
    ocr.warm_up()

    for label, rect in (("按逻辑坐标直接捕获（当前实现）", logical), ("按物理坐标捕获（用户实际框选）", physical)):
        print("=" * 68)
        print(f"{label}: {rect}")
        with ScreenCapturer() as capturer:
            frame = capturer.capture_region(rect)
            print(f"  帧 origin = {frame.origin}   图像尺寸 = {frame.image.shape[1]}x{frame.image.shape[0]}")
            recognized = ocr.recognize_frame(frame)
            blocks = recognized.blocks
            print(f"  识别到 {len(blocks)} 个文本块")
            for block in blocks[:6]:
                box = block.box
                print(f"    - 物理({box.x},{box.y},{box.width}x{box.height}) "
                      f"逻辑({round(box.x / 1.5)},{round(box.y / 1.5)})  {block.text[:38]!r}")
            # 渲染端把物理坐标折算到逻辑坐标
            screen = [s for s in screens if s.is_primary][0]
            ratio = screen.device_pixel_ratio
            origin = screen.physical
            print("  渲染端将画在（逻辑坐标）:")
            for block in blocks[:3]:
                lx = origin.x + round((block.box.x - origin.x) / ratio)
                ly = origin.y + round((block.box.y - origin.y) / ratio)
                lw = round(block.box.width / ratio)
                lh = round(block.box.height / ratio)
                print(f"    - ({lx},{ly}) {lw}x{lh}")
        print()

    ocr.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
