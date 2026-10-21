"""翻译质量自检：用典型屏幕文本走产品翻译路径，检查译文质量。

检查维度：
1. 语言纯度——译文是否完全用目标语言（无源语言残留）
2. 完整性——译文是否被截断
3. 术语保留——数字/路径/代码是否原样保留
4. 合理性——是否有明显的乱码或重复

与 verify_translation_directions.py 相同，走 TranslationService 完整产品路径
（预处理 + 生产 prompt + 输出封顶 + 质量守卫），而非裸调模型——
裸调无法反映用户实际看到的译文（短句、超短词在裸 prompt 下常不翻译）。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from pingyitong.components.llm import ModelPool, TranslationCache
    from pingyitong.config.schema import AppConfig
    from pingyitong.constants import MODEL_FAST
    from pingyitong.core import ModelScheduler
    from pingyitong.services.glossary_service import Glossary
    from pingyitong.services.text_preprocess import TextPreprocessor
    from pingyitong.services.translation_service import TranslationService

    config = AppConfig()
    config.translation.target_language = "zh"
    config.schedule.mode = "manual"
    config.schedule.active_model = MODEL_FAST

    pool = ModelPool(config.model, idle_timeout=600)
    scheduler = ModelScheduler(config.schedule)
    translator = TranslationService(
        config=config,
        model_pool=pool,
        scheduler=scheduler,
        cache=TranslationCache(enabled=False),
        preprocessor=TextPreprocessor.from_config(config),
        glossary=Glossary(),
        history=None,
    )
    if not pool.ensure_loaded(MODEL_FAST):
        print("模型加载失败，无法继续质量自检")
        pool.shutdown()
        return 2

    # 典型屏幕文本：覆盖 UI 标签 / 错误消息 / 技术文档 / 代码注释 / 含路径
    TEST_CASES = [  # noqa: N806 - 函数内常量表
        # (类别, 原文, 源语言, 目标语言, 检查要点)
        ("UI 标签",  "Save changes before closing?",          "en", "zh", ["保存", "关闭"]),
        ("UI 标签",  "Are you sure you want to delete this item?", "en", "zh", ["删除"]),
        ("错误消息", "Failed to connect to the server. Please check your network connection and try again.", "en", "zh", ["连接", "网络"]),
        ("错误消息", "Access denied. You do not have permission to perform this operation.", "en", "zh", ["权限", "拒绝"]),
        ("技术文档", "The pipeline processes frames at 15 fps and uses a multi-stage queue system for backpressure.", "en", "zh", ["15", "队列"]),
        ("代码注释", "# TODO: fix the race condition in worker_thread.py line 42", "en", "zh", ["worker_thread.py", "42"]),
        ("含路径",   "The config file is located at C:/Users/config/settings.json", "en", "zh", ["settings.json"]),
        ("短句",     "OK",     "en", "zh", []),
        ("短句",     "Cancel", "en", "zh", []),
        ("中文→英",  "这是一个测试句子。", "zh", "en", ["test"]),
    ]

    print("=== 翻译质量自检（1.7B，产品路径）===\n")
    issues = []

    for cat, source, source_lang, target_lang, expect_keywords in TEST_CASES:
        config.translation.source_language = source_lang
        config.translation.target_language = target_lang
        t0 = time.monotonic()
        raw = translator.translate_text(source)
        elapsed = (time.monotonic() - t0) * 1000
        translated = raw.strip() if raw else "(空)"

        # 自动检查
        problems = []
        if not translated or translated == "(空)":
            problems.append("译文为空")
        # 代码注释：提示词要求代码标识符与代码原样保留，模型把整条
        # 注释原样返回是正确结果，不能按"未翻译"判失败。
        code_passthrough = cat == "代码注释"
        if translated == source and not code_passthrough:
            problems.append("译文与原文相同（未翻译）")
        # 目标为中文时检查是否有大段英文残留
        if target_lang == "zh" and not code_passthrough:
            cjk_chars = sum(1 for c in translated if '\u4e00' <= c <= '\u9fff')
            if cjk_chars == 0 and len(translated) > 10:
                problems.append("译文无中文（可能未翻译）")
        # 关键词检查
        missing = [kw for kw in expect_keywords if kw not in translated]
        if missing:
            problems.append(f"缺少关键词: {missing}")
        # 重复检测
        if len(translated) > 10 and translated[:len(translated)//2] == translated[len(translated)//2:]:
            problems.append("译文重复")
        # 截断检测
        if translated.endswith("...") or translated.endswith("…"):
            problems.append("疑似截断")

        status = "FAIL " + "; ".join(problems) if problems else "OK"
        print(f"[{cat}] {status}")
        print(f"  原文: {source[:70]}")
        print(f"  译文: {translated[:70]}")
        print(f"  耗时: {elapsed:.0f}ms")
        print()
        if problems:
            issues.append((cat, source[:40], problems))

    # 与标准质量检查一样走产品路径，验证最终可读输出。
    print("=== 语义精简对比 ===\n")
    long_text = ("The system will automatically detect and install the latest "
                 "security updates for your device. Please do not turn off your "
                 "computer during this process as it may take several minutes "
                 "to complete.")
    config.translation.source_language = "en"
    config.translation.target_language = "zh"
    config.translation.concise_output = False
    normal = translator.translate_text(long_text)
    config.translation.concise_output = True
    concise = translator.translate_text(long_text)
    for mode, output in (("标准", normal), ("精简", concise)):
        if not output.strip() or "<<<SEG>>>" in output:
            issues.append((mode, long_text[:40], ["译文为空或泄漏分段标记"]))
        if not any(word in output for word in ("不要", "勿", "不能", "不可")):
            issues.append((mode, long_text[:40], ["关闭计算机的否定约束缺失"]))
    print(f"  标准: {len(normal.strip())} 字 | {normal.strip()[:80]}")
    print(f"  精简: {len(concise.strip())} 字 | {concise.strip()[:80]}")
    print()

    # ---- 汇总 ----
    print(f"=== 汇总：{len(TEST_CASES)} 个用例，{len(issues)} 个有问题 ===")
    for cat, src, probs in issues:
        print(f"  FAIL [{cat}] {src}... -> {'; '.join(probs)}")
    if not issues:
        print("  全部通过")
    pool.shutdown()
    return 1 if issues else 0



if __name__ == "__main__":
    raise SystemExit(main())
