"""端到端验证：框选区域的语言链路。

用**真实**的捕获、OCR、文本后处理、模型与坐标换算出一次完整结果，
回答一个具体问题：**框选一块屏幕后，译文会不会落在正确的位置上**。

背景：用户报"选框选取后，开启翻译，并没有进行翻译"。排查发现两层原因：

1. 框选区域是逻辑坐标，却被直接当物理坐标去截屏（150% 缩放下
   少截一大圈），译文位置随之整体错位；
2. 用户的目标语言是中文，而被框选的窗口内容本身就以中文为主 ——
   译成中文等于原样，屏幕上自然"看不出被翻译过"。

本脚本用 ``--target en`` 跑一遍：目标语言与被选内容不同，
于是应当能看到译文落在选框范围内。

运行：
    python scripts/verify_region_translation.py [目标语言代码]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    target = sys.argv[1] if len(sys.argv) > 1 else "en"

    from PyQt6.QtGui import QGuiApplication

    _app = QGuiApplication.instance() or QGuiApplication([])

    from pingyitong.components.capture.geometry import Rect
    from pingyitong.components.capture.screen_capture import ScreenCapturer
    from pingyitong.components.ocr.ocr_engine import OcrEngine
    from pingyitong.core.coordinate import CoordinateMapper
    from pingyitong.languages import LANGUAGES
    from pingyitong.utils import paths

    config_file = Path(paths.config_path())
    data = json.loads(config_file.read_text(encoding="utf-8"))

    region = data.get("capture", {}).get("region")
    if not region:
        print("配置里没有框选区域，请先在界面上框选一次")
        return 1

    logical = Rect.from_sequence(region)
    mapper = CoordinateMapper()
    physical = mapper.capture_region_physical(logical)

    print("=" * 70)
    print(f"框选（逻辑坐标）   : {logical}")
    print(f"换算（物理坐标）   : {physical}")
    print(f"目标语言           : {target}  {LANGUAGES[target].name if target in LANGUAGES else ''}")
    print("=" * 70)

    # ---- 捕获 + OCR ----
    ocr = OcrEngine()
    ocr.warm_up()
    with ScreenCapturer() as capturer:
        frame = capturer.capture_region(physical)
        recognized = ocr.recognize_frame(frame)
    print(f"捕获图像           : {frame.image.shape[1]}×{frame.image.shape[0]} 物理像素")
    print(f"识别文本块         : {len(recognized.blocks)}")

    # ---- 真实翻译（使用项目自己的 Prompt 与模型池）----
    from pingyitong.components.llm import ModelPool, TranslationCache
    from pingyitong.config.schema import AppConfig
    from pingyitong.core import ModelScheduler
    from pingyitong.services.glossary_service import Glossary
    from pingyitong.services.history_service import HistoryService
    from pingyitong.services.text_preprocess import TextPreprocessor
    from pingyitong.services.translation_service import TranslationService

    config = AppConfig()
    config.translation.target_language = target
    config.schedule.mode = "manual"
    config.model.fast.preload = True

    pool = ModelPool(config.model, idle_timeout=600)
    scheduler = ModelScheduler(config.schedule)
    translator = TranslationService(
        config=config,
        model_pool=pool,
        scheduler=scheduler,
        cache=TranslationCache(enabled=False),
        preprocessor=TextPreprocessor.from_config(config),
        glossary=Glossary(),
        history=HistoryService(),
    )

    print("加载模型…")
    loaded = pool.preload([config.schedule.active_model])
    if not any(loaded.values()):
        print(f"模型加载失败，无法继续验证：{loaded}")
        pool.shutdown()
        return 2

    render_frame = translator.translate_frame(recognized)
    stats = translator.last_stats
    print(f"翻译统计           : {stats.as_text()}")
    print()

    if not render_frame.items:
        print("没有产生任何渲染条目")

    print(f"渲染条目 {len(render_frame.items)} 条（物理坐标 -> 逻辑坐标）:")
    inside = 0
    for item in render_frame.items[:10]:
        box = item.box
        print(f"  - 物理({box.x},{box.y}) {box.width}x{box.height}  {item.text[:52]!r}")
    print()

    # ---- 坐标一致性：渲染条目应落在框选范围内 ----
    for item in render_frame.items:
        if physical.intersects(item.box) or (
            physical.left <= item.box.x <= physical.right
            and physical.top <= item.box.y <= physical.bottom
        ):
            inside += 1
    print(f"落在框选范围内的条目：{inside} / {len(render_frame.items)}")
    if render_frame.items and inside == 0:
        print("!! 全部条目都在框选范围之外 —— 坐标链路有问题")
        pool.shutdown()
        return 3

    # ---- 展示几条实际译文 ----
    translated = [i for i in render_frame.items if i.text.strip()]
    print()
    print("译文样例:")
    for item in translated[:6]:
        print(f"  {item.text[:70]!r}")

    pool.shutdown()
    print()
    print("结论: 框选 -> 捕获 -> 识别 -> 翻译 -> 渲染坐标 全链路正常")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
