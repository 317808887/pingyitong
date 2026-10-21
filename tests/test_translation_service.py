"""翻译服务集成测试。

与 :mod:`tests.test_pipeline_components` 不同，本文件测试的是
**翻译服务的完整编排流程**：文本后处理 → 术语替换 → 缓存查询 →
模型调度 → 推理 → 输出解析 → 术语还原 → 组装渲染条目。

它使用假的推理引擎（:class:`FakeModelPool`）替代真实大模型，
因此不需要模型权重，也不需要 GPU，可以在任何环境运行。

**这个测试存在的直接原因**：开发过程中曾出现"向 Pipeline 与
TranslationService 传入了 ``SettingsManager`` 而非 ``AppConfig``"的缺陷——
两者都只需要读取配置值，但只有 ``AppConfig`` 具有
``capture`` / ``translation`` / ``model`` 等属性。
该缺陷无法被语法检查或静态导入检查发现，只有在真正跑一遍翻译流程时
才会暴露为 ``AttributeError``。本文件正是为了把它固定在测试里。
"""

from __future__ import annotations

import re
import unittest
from unittest import mock

from pingyitong.components.capture.geometry import Rect
from pingyitong.components.llm.translation_cache import TranslationCache
from pingyitong.components.ocr import RecognizedFrame, TextBlock
from pingyitong.config.schema import AppConfig
from pingyitong.constants import (
    GLOSSARY_TOKEN_TEMPLATE,
    MODEL_ACCURATE,
    MODEL_FAST,
    SEGMENT_DELIMITER,
)
from pingyitong.core.model_scheduler import ModelScheduler
from pingyitong.core.resource_monitor import ResourceSnapshot
from pingyitong.services import translation_service as translation_service_module
from pingyitong.services.glossary_service import Glossary, GlossaryEntry
from pingyitong.services.language import decisive_script
from pingyitong.services.text_preprocess import TextPreprocessor
from pingyitong.services.translation_service import (
    TranslationService,
    _looks_like_refusal,
    _split_sentences,
    _wrong_script,
)


def fake_translation(text: str, prefix: str = "译:") -> str:
    """Deterministic Chinese stand-in, preserving glossary placeholders.

    A prefix plus English was not a translation and cannot exercise a strict
    target-language gate. Encode source Latin words as Chinese+numeric markers
    to retain a distinct result for each segment without simulating mixed prose.
    """
    parts = re.split(r"(\[TERM_\d+\])", text)
    return prefix + "".join(
        part if part.startswith("[TERM_") else re.sub(
            r"[A-Za-z]+", lambda match: "词" + "".join(str(ord(ch)) for ch in match[0]), part,
        ) for part in parts
    )


class FakeModelPool:
    """假的模型池。

    不加载任何权重，把送入的片段原样加上语言标记返回，
    使测试可以精确断言"哪段文本被送去翻译了"以及"结果如何被还原"。
    """

    def __init__(self, prefix: str = "译:") -> None:
        self.prefix = prefix
        self.calls: list[tuple[str, str]] = []
        #: 设为 True 时模拟推理失败，用于验证降级路径
        self.fail = False

    def translate(
        self,
        model_id: str,
        prompt: str,
        system_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        """返回 ``译:`` 前缀的伪翻译，保留分隔符结构。"""
        self.calls.append((model_id, prompt))
        if self.fail:
            return ""
        parts = prompt.split(SEGMENT_DELIMITER)
        translated = [fake_translation(part.strip(), self.prefix) for part in parts]
        return SEGMENT_DELIMITER.join(translated)

    def is_available(self, model_id: str) -> bool:
        return True

    def any_available(self) -> bool:
        return True


def make_frame(*blocks: TextBlock, seq: int = 1) -> RecognizedFrame:
    """构造一帧 OCR 结果。"""
    return RecognizedFrame(
        seq=seq,
        origin=Rect(0, 0, 1920, 1080),
        blocks=list(blocks),
        is_full_frame=True,
    )


def block(text: str, x: int, y: int, width: int = 200, height: int = 16) -> TextBlock:
    """构造文本块。"""
    return TextBlock(text=text, box=Rect(x, y, width, height), confidence=0.95)


def make_service(
    config: AppConfig | None = None,
    glossary: Glossary | None = None,
    pool: FakeModelPool | None = None,
) -> tuple[TranslationService, FakeModelPool]:
    """装配一个使用假推理引擎的翻译服务。"""
    config = config or AppConfig()
    config.translation.semantic_review = False  # legacy single-pass contract
    pool = pool or FakeModelPool()
    service = TranslationService(
        config=config,
        model_pool=pool,  # type: ignore[arg-type]
        scheduler=ModelScheduler(config.schedule),
        cache=TranslationCache(
            max_entries=config.model.cache_max_entries,
            ttl_seconds=config.model.cache_ttl_seconds,
            enabled=config.model.enable_cache,
        ),
        preprocessor=TextPreprocessor(),
        glossary=glossary or Glossary(),
        history=None,
    )
    return service, pool


class TestConfigurationAccess(unittest.TestCase):
    """配置访问正确性（防止传入错误类型的配置对象）。

    这两条断言看似平淡，但它们对应的是一个真实发生过的缺陷：
    传入 ``SettingsManager`` 时会在翻译第一帧时抛 ``AttributeError``。
    """

    def test_service_reads_appconfig_attributes(self) -> None:
        """翻译服务必须能读取所有它需要的配置节。"""
        config = AppConfig()
        config.translation.semantic_review = False  # legacy single-pass contract
        service, _ = make_service(config)

        # 这些是 _translate_paragraphs / _needs_translation / _write_history
        # 实际访问的属性路径。任何一处不存在都会在这里失败。
        self.assertIsNotNone(service._config.capture)
        self.assertIsNotNone(service._config.translation)
        self.assertIsNotNone(service._config.model)
        self.assertIsNotNone(service._config.preprocess)
        self.assertIsNotNone(service._config.history)
        self.assertIsNotNone(service._config.model.for_model(MODEL_FAST))

    def test_apply_config_accepts_new_object(self) -> None:
        """配置变更后应能换上新的配置对象并生效。"""
        service, _ = make_service()
        updated = AppConfig()
        updated.translation.semantic_review = False  # legacy single-pass contract
        updated.translation.target_language = "en"
        service.apply_config(updated)
        service._pool.translate = lambda **kwargs: "Hello world."

        frame = make_frame(block("你好世界", 0, 0))
        render_frame = service.translate_frame(frame)
        # 目标语言变成英语后，中文内容应当被送去翻译（不再被"同语言跳过"）
        self.assertEqual(len(render_frame.items), 1)


class TestBasicTranslation(unittest.TestCase):
    """基本翻译流程。"""

    def test_translates_single_paragraph(self) -> None:
        """单个段落应被翻译并生成渲染条目。"""
        service, pool = make_service()
        render_frame = service.translate_frame(make_frame(block("Hello world", 0, 0)))

        self.assertEqual(len(render_frame.items), 1)
        self.assertEqual(render_frame.items[0].text, fake_translation("Hello world"))
        self.assertEqual(len(pool.calls), 1)

    def test_render_item_keeps_original_box(self) -> None:
        """渲染条目必须保留原文包围盒——这是原位渲染的前提。"""
        service, _ = make_service()
        render_frame = service.translate_frame(make_frame(block("Hi", 10, 20, 80, 18)))
        self.assertEqual(render_frame.items[0].box.to_sequence(), [10, 20, 80, 18])

    def test_render_item_records_source_text(self) -> None:
        """应记录原文，供对照模式与历史使用。"""
        service, _ = make_service()
        render_frame = service.translate_frame(make_frame(block("Hello", 0, 0)))
        self.assertEqual(render_frame.items[0].source_text, "Hello")

    def test_empty_frame_produces_no_items(self) -> None:
        """没有文本块的帧不应产生渲染条目。"""
        service, pool = make_service()
        render_frame = service.translate_frame(make_frame())
        self.assertEqual(len(render_frame.items), 0)
        self.assertEqual(len(pool.calls), 0)

    def test_noise_only_frame_produces_no_items(self) -> None:
        """仅含噪声（纯数字）的帧不应触发推理。"""
        service, pool = make_service()
        render_frame = service.translate_frame(make_frame(block("12345", 0, 0)))
        self.assertEqual(len(render_frame.items), 0)
        self.assertEqual(len(pool.calls), 0)


class TestSameLanguageSkip(unittest.TestCase):
    """同语言跳过。"""

    def test_skips_when_source_equals_target(self) -> None:
        """显式选择源和目标中文时，中文内容不应送去翻译。"""
        config = AppConfig()
        config.translation.semantic_review = False  # legacy single-pass contract
        config.translation.source_language = "zh"
        service, pool = make_service(config)
        render_frame = service.translate_frame(make_frame(block("这是中文内容", 0, 0)))

        self.assertEqual(len(pool.calls), 0)
        self.assertEqual(len(render_frame.items), 1)
        # 内容原样保留
        self.assertEqual(render_frame.items[0].text, "这是中文内容")

    def test_translates_when_source_differs(self) -> None:
        """目标语言为中文时，英文内容应被翻译。"""
        service, pool = make_service()
        render_frame = service.translate_frame(make_frame(block("This is English", 0, 0)))
        self.assertEqual(len(pool.calls), 1)
        self.assertEqual(render_frame.items[0].text, fake_translation("This is English"))

    def test_skip_can_be_disabled(self) -> None:
        """关闭跳过逻辑后，同语言内容也会被送去翻译。"""
        config = AppConfig()
        config.translation.semantic_review = False  # legacy single-pass contract
        config.preprocess.skip_same_language = False
        service, pool = make_service(config)
        service.translate_frame(make_frame(block("这是中文内容", 0, 0)))
        self.assertEqual(len(pool.calls), 1)

    def test_explicit_source_language_used(self) -> None:
        """显式指定源语言时以配置为准，不做自动检测。"""
        config = AppConfig()
        config.translation.semantic_review = False  # legacy single-pass contract
        config.translation.source_language = "en"
        service, pool = make_service(config)
        # 文本是中文，但配置说源语言是英语 → 应当翻译（因为 en != zh）
        service.translate_frame(make_frame(block("这是中文内容", 0, 0)))
        self.assertEqual(len(pool.calls), 1)


class TestCaching(unittest.TestCase):
    """推理缓存。"""

    def test_second_identical_frame_hits_cache(self) -> None:
        """相同文本第二次出现应命中缓存，不再推理。

        界面文字重复率很高（菜单、按钮、状态栏），这是实时性的关键优化。
        """
        service, pool = make_service()
        frame = make_frame(block("Settings", 0, 0))

        service.translate_frame(frame)
        self.assertEqual(len(pool.calls), 1)

        render_frame = service.translate_frame(make_frame(block("Settings", 0, 0)))
        self.assertEqual(len(pool.calls), 1, "第二次不应再推理")
        self.assertTrue(render_frame.items[0].from_cache)

    def test_same_text_different_position_hits_cache(self) -> None:
        """同一文本出现在不同位置应命中同一缓存（缓存键不含坐标）。

        样本不用 "OK"——它已被确定性 UI 短词短路收编（零推理），
        测不出"首次推理、二次缓存"的路径。
        """
        service, pool = make_service()
        service.translate_frame(make_frame(block("Settings", 10, 10)))
        service.translate_frame(make_frame(block("Settings", 800, 600)))
        self.assertEqual(len(pool.calls), 1)

    def test_cache_disabled_always_infers(self) -> None:
        """关闭缓存后每次都推理。"""
        config = AppConfig()
        config.translation.semantic_review = False  # legacy single-pass contract
        config.model.enable_cache = False
        service, pool = make_service(config)
        service.translate_frame(make_frame(block("Settings", 0, 0)))
        service.translate_frame(make_frame(block("Settings", 0, 0)))
        self.assertEqual(len(pool.calls), 2)

    def test_cache_invalidated_by_glossary_change(self) -> None:
        """术语表变更后缓存必须失效。

        否则新导入的术语不会生效——用户会觉得"导入了却没作用"。
        """
        service, pool = make_service()
        service.translate_frame(make_frame(block("GPU is fast", 0, 0)))
        self.assertEqual(len(pool.calls), 1)

        # 导入术语表（版本号变化 → 缓存键变化）
        service.set_glossary(Glossary([GlossaryEntry("GPU", "图形处理器")]))
        service.translate_frame(make_frame(block("GPU is fast", 0, 0)))
        self.assertEqual(len(pool.calls), 2, "术语表变更后应重新推理")


class TestGlossaryIntegration(unittest.TestCase):
    """术语库与翻译流程的集成。"""

    def test_glossary_applied_and_restored(self) -> None:
        """术语应被替换后送给模型，并在返回时还原。"""
        glossary = Glossary([GlossaryEntry("GPU", "图形处理器")])
        service, pool = make_service(glossary=glossary)

        render_frame = service.translate_frame(make_frame(block("Use GPU here", 0, 0)))

        # 送给模型的文本里不应含有 GPU（已被占位符替换）
        sent_text = pool.calls[0][1]
        self.assertNotIn("GPU", sent_text)
        self.assertIn(GLOSSARY_TOKEN_TEMPLATE.format(0), sent_text)

        # 还原后的译文应包含术语译文，且不含占位符
        result = render_frame.items[0].text
        self.assertIn("图形处理器", result)
        self.assertNotIn(GLOSSARY_TOKEN_TEMPLATE.format(0), result)

    def test_glossary_stats_counted(self) -> None:
        """术语命中数应被统计。"""
        glossary = Glossary([GlossaryEntry("GPU", "图形处理器")])
        service, _ = make_service(glossary=glossary)
        service.translate_frame(make_frame(block("Use GPU here", 0, 0)))
        self.assertGreaterEqual(service.last_stats.glossary_hits, 1)

    def test_glossary_disabled_by_config(self) -> None:
        """配置关闭术语库后不应替换。"""
        config = AppConfig()
        config.translation.semantic_review = False  # legacy single-pass contract
        config.translation.glossary_enabled = False
        glossary = Glossary([GlossaryEntry("GPU", "图形处理器")])
        service, pool = make_service(config, glossary=glossary)

        service.translate_frame(make_frame(block("Use GPU here", 0, 0)))
        # 术语未替换，因此原文完整送给模型
        self.assertIn("GPU", pool.calls[0][1])

    def test_dropped_placeholder_retries_literal_term_for_single_text(self) -> None:
        """1.7B may omit a leading token; the term must not disappear."""
        class DroppingPool(FakeModelPool):
            def __init__(self) -> None:
                super().__init__()
                self.responses = ["研究粒子。", "量子力学研究粒子。"]

            def translate(self, model_id, prompt, system_prompt, max_tokens=None):
                self.calls.append((model_id, prompt))
                return self.responses.pop(0)

        config = AppConfig()
        config.translation.semantic_review = False  # legacy single-pass contract
        config.translation.source_language = "en"
        config.schedule.mode = "manual"
        config.schedule.active_model = MODEL_FAST
        pool = DroppingPool()
        glossary = Glossary([GlossaryEntry("Quantum mechanics", "量子力学")])
        service, _ = make_service(config, glossary, pool)

        self.assertEqual(
            service.translate_text("Quantum mechanics studies particles."),
            "量子力学研究粒子。",
        )
        self.assertEqual(len(pool.calls), 2)
        self.assertIn("[TERM_0]", pool.calls[0][1])
        self.assertIn("Quantum mechanics", pool.calls[1][1])
        self.assertEqual(
            service.translate_text("Quantum mechanics studies particles."),
            "量子力学研究粒子。",
        )
        self.assertEqual(len(pool.calls), 2, "成功恢复后的译文应走缓存")

    def test_successful_frame_recovery_is_cached(self) -> None:
        class DroppingPool(FakeModelPool):
            def __init__(self) -> None:
                super().__init__()
                self.responses = ["研究粒子。", "量子力学研究粒子。"]

            def translate(self, model_id, prompt, system_prompt, max_tokens=None):
                self.calls.append((model_id, prompt))
                return self.responses.pop(0)

        config = AppConfig()
        config.translation.semantic_review = False  # legacy single-pass contract
        config.translation.source_language = "en"
        config.schedule.mode = "manual"
        config.schedule.active_model = MODEL_FAST
        pool = DroppingPool()
        service, _ = make_service(
            config, Glossary([GlossaryEntry("Quantum mechanics", "量子力学")]), pool,
        )
        frame = make_frame(block("Quantum mechanics studies particles.", 0, 0))

        for _ in range(2):
            self.assertEqual(
                service.translate_frame(frame).items[0].text,
                "量子力学研究粒子。",
            )
        self.assertEqual(len(pool.calls), 2)

    def test_english_scaffolding_retry_restores_full_chinese_sentence(self) -> None:
        class MixedPool(FakeModelPool):
            def __init__(self) -> None:
                super().__init__()
                self.responses = [
                    "The [TERM_1] has a [TERM_0].",
                    "该分子存在共价键。",
                ]

            def translate(self, model_id, prompt, system_prompt, max_tokens=None):
                self.calls.append((model_id, prompt))
                return self.responses.pop(0)

        config = AppConfig()
        config.translation.semantic_review = False  # legacy single-pass contract
        config.translation.source_language = "en"
        config.schedule.mode = "manual"
        config.schedule.active_model = MODEL_ACCURATE
        pool = MixedPool()
        glossary = Glossary([
            GlossaryEntry("molecule", "分子"),
            GlossaryEntry("covalent bond", "共价键"),
        ])
        service, _ = make_service(config, glossary, pool)
        source = "The molecule has a covalent bond."

        self.assertEqual(service.translate_text(source), "该分子存在共价键。")
        self.assertEqual(service.translate_text(source), "该分子存在共价键。")
        self.assertEqual(len(pool.calls), 2)

    def test_english_scaffolding_retry_rejects_second_mixed_result(self) -> None:
        class MixedPool(FakeModelPool):
            def __init__(self) -> None:
                super().__init__()
                self.responses = [
                    "The [TERM_1] has a [TERM_0].",
                    "The 分子 has a 共价键.",
                ]

            def translate(self, model_id, prompt, system_prompt, max_tokens=None):
                self.calls.append((model_id, prompt))
                return self.responses.pop(0)

        config = AppConfig()
        config.translation.semantic_review = False  # legacy single-pass contract
        config.translation.source_language = "en"
        config.schedule.mode = "manual"
        config.schedule.active_model = MODEL_ACCURATE
        pool = MixedPool()
        glossary = Glossary([
            GlossaryEntry("molecule", "分子"),
            GlossaryEntry("covalent bond", "共价键"),
        ])
        service, _ = make_service(config, glossary, pool)
        source = "The molecule has a covalent bond."
        self.assertEqual(service.translate_text(source), source)

    def test_dropped_placeholder_never_caches_incomplete_frame(self) -> None:
        class DroppingPool(FakeModelPool):
            def __init__(self) -> None:
                super().__init__()
                self.responses = ["研究粒子。", "研究粒子。", "量子力学研究粒子。"]

            def translate(self, model_id, prompt, system_prompt, max_tokens=None):
                self.calls.append((model_id, prompt))
                return self.responses.pop(0)

        config = AppConfig()
        config.translation.semantic_review = False  # legacy single-pass contract
        config.translation.source_language = "en"
        config.schedule.mode = "manual"
        config.schedule.active_model = MODEL_FAST
        pool = DroppingPool()
        glossary = Glossary([GlossaryEntry("Quantum mechanics", "量子力学")])
        service, _ = make_service(config, glossary, pool)
        text = "Quantum mechanics studies particles."

        first = service.translate_frame(make_frame(block(text, 0, 0)))
        self.assertEqual(first.items, [])  # Keep the real source screen visible.
        self.assertEqual(service.last_output(), [])
        second = service.translate_frame(make_frame(block(text, 0, 0))).items[0].text
        self.assertEqual(second, "量子力学研究粒子。")
        self.assertEqual(len(pool.calls), 3)


class TestBatchTranslation(unittest.TestCase):
    """批量翻译与对位。"""

    def test_pack_batches_respects_count_and_char_budget(self) -> None:
        """装批同时受条数上限与字符预算约束。

        批量请求超出模型上下文会被 llama-server 拒绝（HTTP 400），
        整批作废后还得逐条重试，白白多一轮延迟。
        """
        from pingyitong.services.text_preprocess import Paragraph
        from pingyitong.services.translation_service import _pack_batches

        def para(text: str):
            return (Paragraph(text=text, box=Rect(0, 0, 10, 10)), "key", {})

        items = [para("x" * 400) for _ in range(5)]
        # 条数上限 6 允许全进一批，但字符预算 900 只装得下两条
        batches = _pack_batches(items, batch_size=6, char_budget=900)
        self.assertEqual([len(b) for b in batches], [2, 2, 1])

        # 预算充足时仍按条数上限装批
        batches = _pack_batches(items, batch_size=3, char_budget=10_000)
        self.assertEqual([len(b) for b in batches], [3, 2])

    def test_long_batch_splits_before_context_overflow(self) -> None:
        """长段落装批应在溢出上下文前切分，而不是靠请求失败兜底。"""
        service, pool = make_service()
        long_text = "This is a fairly long English sentence. " * 12  # 480 字符
        # Distinct inputs exercise context packing; duplicate labels are now
        # safely translated once and reused at all original positions.
        frame = make_frame(*[block(long_text + f"Item {i}.", 0, i * 600) for i in range(6)])
        service.translate_frame(frame)

        # 极速模型 n_ctx=2048、max_tokens=384：按最坏情况 1 字符 1 token，
        # 6 段 × 480 字必然超出，应被拆成多批而不是一次塞满
        self.assertGreater(len(pool.calls), 1)
        for _, prompt in pool.calls:
            self.assertLess(len(prompt), 2048)

    def test_multiple_paragraphs_batched(self) -> None:
        """多个段落合并进单次批量请求（分隔符对位）。"""
        service, pool = make_service()
        frame = make_frame(
            block("First paragraph here.", 0, 0),
            block("Second paragraph here.", 0, 400),
        )
        render_frame = service.translate_frame(frame)

        self.assertEqual(len(render_frame.items), 2)
        self.assertEqual(len(pool.calls), 1)
        _, prompt = pool.calls[0]
        self.assertIn(SEGMENT_DELIMITER, prompt)

    def test_batch_results_mapped_back_correctly(self) -> None:
        """批量结果的顺序必须与输入段落一一对应。

        对位错误会导致译文张冠李戴——把第一个段落的译文画到第二个位置上。
        """
        service, _ = make_service()
        frame = make_frame(
            block("Alpha paragraph.", 0, 0),
            block("Beta paragraph.", 0, 400),
        )
        render_frame = service.translate_frame(frame)

        texts = [item.text for item in render_frame.items]
        self.assertIn(fake_translation("Alpha paragraph."), texts)
        self.assertIn(fake_translation("Beta paragraph."), texts)

    def test_batch_size_limits_request(self) -> None:
        """批量大小限制装批：5 段按批量 2 装成 [2, 2, 1]。"""
        config = AppConfig()
        config.translation.semantic_review = False  # legacy single-pass contract
        config.model.batch_size = 2
        service, pool = make_service(config)

        # 构造 5 个独立段落（垂直间距足够大）
        blocks = [block(f"Paragraph number {i} here.", 0, i * 300) for i in range(5)]
        service.translate_frame(make_frame(*blocks))

        self.assertEqual(len(pool.calls), 3)
        sizes = [len(prompt.split(SEGMENT_DELIMITER)) for _, prompt in pool.calls]
        self.assertEqual(sizes, [2, 2, 1])

    def test_single_tail_batch_uses_compact_prompt(self) -> None:
        """装批后的单段尾巴应支付精简提示词而非批量提示词。"""
        class PromptPool(FakeModelPool):
            def __init__(self) -> None:
                super().__init__()
                self.system_prompts: list[str] = []

            def translate(self, model_id, prompt, system_prompt, max_tokens=None):
                self.system_prompts.append(system_prompt)
                return super().translate(model_id, prompt, system_prompt, max_tokens)

        config = AppConfig()
        config.translation.semantic_review = False  # legacy single-pass contract
        config.model.batch_size = 2
        pool = PromptPool()
        config.translation.source_language = "fr"
        service, _ = make_service(config, pool=pool)
        frame = make_frame(*[
            block(f"English paragraph number {i}.", 0, i * 300)
            for i in range(3)
        ])
        service.translate_frame(frame)

        # 3 段装成 [2, 1]：批量段用完整提示词（含分隔符规则），
        # 单段尾巴用精简提示词（带手动源语言，比完整提示词短）
        from pingyitong.components.llm.prompt import build_system_prompt

        full = build_system_prompt("fr", "zh")
        self.assertEqual(len(pool.system_prompts), 2)
        batch_prompt, tail_prompt = pool.system_prompts
        self.assertIn("separator", batch_prompt)
        self.assertNotIn("separator", tail_prompt)
        self.assertIn("Translate from French", tail_prompt)
        self.assertLess(len(tail_prompt), len(full))

    def test_glossary_single_batch_keeps_placeholder_rule(self) -> None:
        """单段命中术语时必须明确要求模型原样保留占位符。"""
        class PromptPool(FakeModelPool):
            system_prompt = ""

            def translate(self, model_id, prompt, system_prompt, max_tokens=None):
                self.system_prompt = system_prompt
                return super().translate(model_id, prompt, system_prompt, max_tokens)

        glossary = Glossary([GlossaryEntry("OpenAI", "开放人工智能")])
        pool = PromptPool()
        service, _ = make_service(glossary=glossary, pool=pool)
        service.translate_frame(make_frame(block("OpenAI provides useful tools.", 0, 0)))

        self.assertIn("protected glossary", pool.system_prompt)
        self.assertIn(GLOSSARY_TOKEN_TEMPLATE.format(0), pool.system_prompt)

    def test_input_char_budget_counts_system_prompt_as_english(self) -> None:
        """输入预算应把英文 system 提示词按英文密度折算 token。

        system 提示词约 1800 字符、实际约 600 token。若把字符数直接
        当 token 数扣减，极速模型的输入预算会掉到 256 字符下限，
        内容较多的帧被拆成远多于必要的批次——每多一批就是一次
        串行请求往返，徒增翻译延迟。

        门槛取 900：预算须远高于 256 下限、且稳稳装下现实帧
        （约 840 字符，见下一条测试）；提示词继续加长导致预算
        跌破 900 时，应优先精简提示词而非放宽本门槛。
        """
        from pingyitong.components.llm import build_system_prompt

        service, _ = make_service()
        system_prompt = build_system_prompt(
            source_language="auto", target_language="zh"
        )
        budget = service._input_char_budget(MODEL_FAST, system_prompt, 384)
        self.assertGreater(budget, 900)

    def test_realistic_paragraphs_fit_input_and_output_capacity(self) -> None:
        """装批同时保证完整译文容量，不能仅因输入未溢出就塞满一批。

        6 段约 840 字符虽在极速模型输入预算内，384 token 的输出容量
        无法覆盖完整批译文与分隔符。拆成三批仍优于原来的六次请求，
        同时避免截断后只重试前四段而丢失尾部文字。
        """
        config = AppConfig()
        config.translation.semantic_review = False  # legacy single-pass contract
        config.schedule.mode = "manual"
        config.schedule.active_model = MODEL_FAST
        service, pool = make_service(config)

        text = "This is a medium sized English paragraph. " * 3
        blocks = [block(text + f"Item {i}.", 0, i * 600) for i in range(6)]
        result = service.translate_frame(make_frame(*blocks))

        self.assertEqual(len(result.items), len(blocks))
        self.assertEqual(len(pool.calls), 3)

    def test_failure_falls_back_to_original_text(self) -> None:
        """推理失败时必须保留原文，绝不能留空白。

        若返回空字符串，渲染层会在原文位置画一个空遮罩——
        原文被盖住却什么也不显示，用户会以为屏幕坏了。
        """
        pool = FakeModelPool()
        pool.fail = True
        service, _ = make_service(pool=pool)

        render_frame = service.translate_frame(make_frame(block("Hello world", 0, 0)))
        self.assertEqual(render_frame.items, [])
        self.assertEqual(service.last_output(), [])


class _EchoingModelPool(FakeModelPool):
    """模拟 1.7B 模型的复读缺陷。

    批量请求时把指定片段**原样**返回（其余片段正常加前缀）；
    单段请求则正常翻译——与实测行为一致：著名例句在多段批量中
    被原样照抄，单独请求则翻译正常。
    """

    def __init__(self, echo_text: str) -> None:
        super().__init__()
        self.echo_text = echo_text

    def translate(
        self,
        model_id: str,
        prompt: str,
        system_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        self.calls.append((model_id, prompt))
        if self.fail:
            return ""
        parts = prompt.split(SEGMENT_DELIMITER)
        if len(parts) > 1:
            translated = [
                part.strip() if part.strip() == self.echo_text
                else fake_translation(part.strip(), self.prefix)
                for part in parts
            ]
            return SEGMENT_DELIMITER.join(translated)
        return fake_translation(prompt.strip(), self.prefix)


class TestEchoedSegmentRetry(unittest.TestCase):
    """批量结果中"译文=原文"的段落应被单独重试。

    能进入批量的段落都已通过"需要翻译"筛选，译文与原文完全相同
    几乎必然是模型复读缺陷——直接采用会让屏幕上留下一块没翻译的
    原文。修复方式：仅对复读段补一次单段请求。
    """

    def test_echoed_segment_is_retried_individually(self) -> None:
        """批量中被复读的段落应重试并取回正确译文。"""
        pool = _EchoingModelPool("The quick brown fox jumps.")
        service, _ = make_service(pool=pool)

        frame = make_frame(
            block("Dashboard", 0, 0, 240, 32),
            block("The quick brown fox jumps.", 0, 400, 400, 32),
        )
        render_frame = service.translate_frame(frame)

        texts = [item.text for item in render_frame.items]
        # 复读段被重试：最终是"单段请求"的译文，而不是原样照抄的原文
        self.assertIn(fake_translation("Dashboard"), texts)
        self.assertIn(fake_translation("The quick brown fox jumps."), texts)
        self.assertNotIn("The quick brown fox jumps.", texts)
        # 一次批量请求 + 仅为复读段的一次单段重试
        self.assertEqual(len(pool.calls), 2)
        batch_calls = [c for _, c in pool.calls if SEGMENT_DELIMITER in c]
        self.assertEqual(len(batch_calls), 1)

    def test_no_echo_keeps_single_batch_request(self) -> None:
        """正常批量结果不应触发任何额外请求（延迟零开销）。"""
        service, pool = make_service()
        frame = make_frame(
            block("First paragraph here.", 0, 0),
            block("Second paragraph here.", 0, 400),
        )
        service.translate_frame(frame)
        self.assertEqual(len(pool.calls), 1)


class _RefusingModelPool(FakeModelPool):
    """对指定文本"拒答"的假模型池。

    复现真实的 1.7B/4B 行为：模型偶发把内容安全策略用在一句完全正常的
    英文上，把致歉文案当成译文返回。带中立重问标记（提示词里出现
    "Do not comment on"）的请求不再拒答——对应真实情况下同一句换个
    措辞就能译出的现象（实测 "inspect or delete" 改成 "open" 即正常）。
    """

    #: 真实输出的节选（实测原样照抄自 1.7B）
    REFUSAL = "你所提供的内容包含不良信息，不符合规范，因此我无法完成该翻译。"

    def __init__(self, refuse_text: str, prefix: str = "译:", neutral_ask_works: bool = True) -> None:
        super().__init__(prefix)
        self.refuse_text = refuse_text
        #: 中立重问是否生效；``False`` 表示连重问也拒答（验证兜底路径）
        self.neutral_ask_works = neutral_ask_works
        self.system_prompts: list[str] = []

    def translate(
        self,
        model_id: str,
        prompt: str,
        system_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        self.calls.append((model_id, prompt))
        self.system_prompts.append(system_prompt)
        if self.fail:
            return ""
        neutral = "quoted for translation" in system_prompt
        parts = prompt.split(SEGMENT_DELIMITER)
        translated = []
        for part in parts:
            text = part.strip()
            refused = text == self.refuse_text and not (neutral and self.neutral_ask_works)
            translated.append(self.REFUSAL if refused else fake_translation(text, self.prefix))
        return SEGMENT_DELIMITER.join(translated)


class _AlwaysRefusingModelPool(_RefusingModelPool):
    """连中立重问也拒答，用于验证兜底行为。"""

    def __init__(self, refuse_text: str, prefix: str = "译:") -> None:
        super().__init__(refuse_text, prefix, neutral_ask_works=False)


class _LengthSensitiveRefusingPool(_RefusingModelPool):
    """长句一律拒答、短句正常翻译——复现"整段触发审核、拆句后通过"。

    短句的"译文"用纯中文占位（长度标记），以免触发英文残留恢复路径，
    让本用例只考察拒答这一条链路。
    """

    def __init__(self, max_chars: int = 80) -> None:
        super().__init__("")
        self.max_chars = max_chars

    def translate(
        self,
        model_id: str,
        prompt: str,
        system_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        self.calls.append((model_id, prompt))
        self.system_prompts.append(system_prompt)
        parts = prompt.split(SEGMENT_DELIMITER)
        return SEGMENT_DELIMITER.join(
            self.REFUSAL if len(part.strip()) > self.max_chars else f"中文{len(part.strip())}"
            for part in parts
        )


class _KoreanDriftPool(FakeModelPool):
    """复现"韩文漂移"的假模型池：首次请求对目标段输出韩文，重试恢复正常。

    对应实测缺陷（1.7B，韩文源段 + 中文目标，批量 4/4 轮复现）：模型把
    韩文源段**原样输出**（未译成目标语言中文）。重试（同文本第二次请求）
    恢复为中文——与"漂移与采样随机性相关"的实测一致。
    """

    KOREAN = "버전 0.6.8:검증 1veriyinstaller0lo-이 1개의 설치 패키지"
    #: 重试恢复后的"中文译文"（假池没有真实翻译能力，用固定中文占位）
    RECOVERED = "版本 0.6.8 验证与安装包一起下载"

    def __init__(self, drift_text: str, prefix: str = "译:") -> None:
        super().__init__(prefix)
        self.drift_text = drift_text
        self.system_prompts: list[str] = []
        self._drifted: set[str] = set()

    def translate(
        self,
        model_id: str,
        prompt: str,
        system_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        self.calls.append((model_id, prompt))
        self.system_prompts.append(system_prompt)
        if self.fail:
            return ""
        parts = prompt.split(SEGMENT_DELIMITER)
        if len(parts) == 1:
            text = prompt.strip()
            if text == self.drift_text and text not in self._drifted:
                self._drifted.add(text)
                return self.KOREAN
            return f"{self.prefix}{self.RECOVERED}"
        translated: list[str] = []
        for part in parts:
            text = part.strip()
            if text == self.drift_text and text not in self._drifted:
                self._drifted.add(text)
                translated.append(self.KOREAN)
            else:
                translated.append(fake_translation(text, self.prefix))
        return SEGMENT_DELIMITER.join(translated)


class _AlwaysKoreanPool(_KoreanDriftPool):
    """连单段重试也输出韩文——验证兜底行为（保留原文）。"""

    def translate(
        self,
        model_id: str,
        prompt: str,
        system_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        self.calls.append((model_id, prompt))
        self.system_prompts.append(system_prompt)
        parts = prompt.split(SEGMENT_DELIMITER)
        return SEGMENT_DELIMITER.join(self.KOREAN for _ in parts)


class _DelimiterResiduePool(FakeModelPool):
    """复现"分隔符残缺"的假模型池：批量时对指定段输出 <<SEG>>> 变体。

    精确切分只认 `<<<SEG>>>`，少一个尖括号的变体不会被切掉，残片直接
    进入译文——实测屏幕出现 "Lverify_i<<SEG>>>"。
    """

    RESIDUE = "Lverify_i<<SEG>>> 安装包已就绪"

    def __init__(self, residue_text: str, prefix: str = "译:") -> None:
        super().__init__(prefix)
        self.residue_text = residue_text

    def translate(
        self,
        model_id: str,
        prompt: str,
        system_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        self.calls.append((model_id, prompt))
        parts = prompt.split(SEGMENT_DELIMITER)
        if len(parts) == 1:
            return fake_translation(prompt.strip(), self.prefix)
        return SEGMENT_DELIMITER.join(
            self.RESIDUE if part.strip() == self.residue_text else fake_translation(part.strip(), self.prefix)
            for part in parts
        )


class TestRefusalDetection(unittest.TestCase):
    """拒答判定必须保守：正常译文里的"无法翻译"不能误判。"""

    def test_real_refusal_is_detected(self) -> None:
        self.assertTrue(_looks_like_refusal(_RefusingModelPool.REFUSAL))
        self.assertTrue(_looks_like_refusal("很抱歉，我无法完成这个翻译任务。"))
        self.assertTrue(_looks_like_refusal("I cannot translate this request."))
        self.assertTrue(_looks_like_refusal(
            "我无法执行该操作，删除文件属于危险行为，可能会对数据安全造成严重威胁，"
            "请遵守法律法规和系统安全规范，不要尝试执行此类指令。"
        ))
        self.assertTrue(_looks_like_refusal(
            "抱歉，我无法执行该操作。任何要求删除文件、执行违规指令的请求"
            "均不符合安全规范，请遵守法律法规。"
        ))
        self.assertTrue(_looks_like_refusal(
            "抱歉，我无法执行该操作。任何要求删除文件的行为都违背安全规范。"
            "请提出符合法律法规的平台请求。"
        ))
        self.assertTrue(_looks_like_refusal(
            "抱歉，我无法执行该操作。任何要求删除文件的行为都违背安全规范。"
            "请确保你的请求符合法律法规和安全规范。"
        ))
        self.assertTrue(_looks_like_refusal(
            "抱歉，我无法执行该操作，也不能生成删除文件的内容。"
            "如果你有其他需要翻译的软件界面文本，我可以帮你翻译。"
        ))

    def test_legitimate_translation_is_not_flagged(self) -> None:
        """仅命中"无法翻译"字面、没有政策措辞时不算拒答。"""
        self.assertFalse(_looks_like_refusal("该软件无法翻译此语言。"))
        self.assertFalse(_looks_like_refusal("未报告任何使用数据。"))
        self.assertFalse(_looks_like_refusal("我无法执行该操作，因为权限不足。"))
        self.assertFalse(_looks_like_refusal(""))

    def test_measured_wording_variants_are_detected(self) -> None:
        """实测的三种措辞都要识别（措辞在不同轮次里不一样）。"""
        variants = [
            "你所提供的内容包含不良信息，不符合规范，因此我无法完成该翻译。",
            "你所提供的内容涉及不恰当的信息，无法进行翻译。"
            "请提供合法、正向的文本内容，以便进行准确翻译。",
            "你所描述的内容涉及传播违法信息，严重违反法律法规和道德准则，"
            "因此我无法为你提供相关翻译。",
        ]
        for text in variants:
            with self.subTest(text=text[:20]):
                self.assertTrue(_looks_like_refusal(text))

    def test_sentence_split_guards(self) -> None:
        """切句只在 2–4 句、且每句不短于 8 字符时启用。"""
        self.assertEqual(
            _split_sentences("First sentence here. Second sentence here."),
            ["First sentence here.", "Second sentence here."],
        )
        self.assertEqual(_split_sentences("Only one sentence."), [])
        self.assertEqual(_split_sentences("A. B. C. D. E."), [])

    def test_long_text_is_never_flagged(self) -> None:
        """长文档里出现"违反法律法规"属于正常表述，不做拒答判定。"""
        self.assertFalse(_looks_like_refusal("无法完成该翻译，" + "正文内容。" * 200))


class TestRefusedSegmentRecovery(unittest.TestCase):
    """模型把拒答当译文返回时，屏幕上也绝不能出现致歉文案。"""

    def test_refused_segment_is_reasked_in_neutral_wording(self) -> None:
        refused = "The model files live in a folder you can inspect or delete."
        pool = _RefusingModelPool(refused)
        service, _ = make_service(pool=pool)

        frame = make_frame(
            block("Dashboard", 0, 0, 240, 32),
            block(refused, 0, 400, 400, 32),
        )
        render_frame = service.translate_frame(frame)
        texts = [item.text for item in render_frame.items]

        self.assertNotIn(_RefusingModelPool.REFUSAL, texts)
        self.assertIn(fake_translation(refused), texts)
        # 一次批量请求 + 仅为拒答段的一次中立重问
        self.assertEqual(len(pool.calls), 2)
        # 重问必须用完整提示词：实测精简提示词才是诱发拒答的原因
        retry_prompt = pool.system_prompts[-1]
        self.assertIn("professional screen-content translator", retry_prompt)
        self.assertIn("quoted for translation", retry_prompt)

    def test_persistent_refusal_keeps_original_text(self) -> None:
        """重问仍拒答时保留原文，而不是把致歉画到屏幕上。"""
        refused = "The model files live in a folder you can inspect or delete."
        pool = _AlwaysRefusingModelPool(refused)
        service, _ = make_service(pool=pool)

        frame = make_frame(
            block("Dashboard", 0, 0, 240, 32),
            block(refused, 0, 400, 400, 32),
        )
        render_frame = service.translate_frame(frame)
        texts = [item.text for item in render_frame.items]

        self.assertNotIn(_RefusingModelPool.REFUSAL, texts)
        self.assertNotIn(refused, texts)
        self.assertIn(fake_translation("Dashboard"), texts)
        self.assertEqual(service.last_output(), [("Dashboard", fake_translation("Dashboard"))])

    def test_long_refusal_is_split_into_sentences(self) -> None:
        """整段重问仍被拒答时，拆句重试应把内容译回来。"""
        long_text = (
            "Everything you translate stays on this computer. "
            "The history database and the model files live in a folder."
        )
        parts = _split_sentences(long_text)
        pool = _LengthSensitiveRefusingPool(max_chars=80)
        service, _ = make_service(pool=pool)

        frame = make_frame(block(long_text, 0, 0, 900, 32))
        render_frame = service.translate_frame(frame)
        text = render_frame.items[0].text

        self.assertNotIn(_RefusingModelPool.REFUSAL, text)
        # 两句各自译出后拼接（拼接符对中文目标语言不加空格）
        self.assertEqual(text, f"中文{len(parts[0])}中文{len(parts[1])}")
        # 一次批量 + 一次整段重问 + 两次拆句 = 4 次请求
        self.assertEqual(len(pool.calls), 4)

    def test_normal_batch_triggers_no_extra_request(self) -> None:
        """正常批量结果不触发任何额外请求（延迟零开销）。"""
        service, pool = make_service()
        frame = make_frame(
            block("First paragraph here.", 0, 0),
            block("Second paragraph here.", 0, 400),
        )
        service.translate_frame(frame)
        self.assertEqual(len(pool.calls), 1)

    def test_translate_text_refusal_falls_back_to_source(self) -> None:
        """剪贴板等单条路径同样不返回致歉文案。"""
        target = "The quick brown fox jumps."
        pool = _AlwaysRefusingModelPool(target)
        service, _ = make_service(pool=pool)
        result = service.translate_text(target)
        self.assertEqual(result, "The quick brown fox jumps.")

    def test_translate_text_refusal_is_retried_then_recovered(self) -> None:
        target = "The quick brown fox jumps."
        pool = _RefusingModelPool(target)
        service, _ = make_service(pool=pool)
        result = service.translate_text(target)
        self.assertEqual(result, fake_translation(target))
        self.assertEqual(len(pool.calls), 2)


class TestWrongLanguageGuard(unittest.TestCase):
    """模型把译文输出成未选择的语言（韩文漂移）时的拦截与恢复。

    实测（1.7B，韩文源段 + 中文目标，批量 4/4 轮复现）：模型把韩文源段
    原样输出。覆盖层直接画韩文 = "翻译成了未选择的语言"；而 kana/谚文
    与目标语言的脚本差异是决定性的，可以做廉价可靠的校验。
    """

    def test_decisive_script_verdicts(self) -> None:
        """严格比例判定：各脚本的决定性样本与放行样本。"""
        self.assertEqual(decisive_script("버전 검증입니다"), "ko")
        self.assertEqual(decisive_script("これは日本語です"), "ja")
        self.assertEqual(decisive_script("Это русский текст"), "ru")
        self.assertEqual(decisive_script("这是中文内容"), "zh")
        # 拉丁、数字、标识符、中英混排（汉字被稀释）→ 无决定性判定
        self.assertIsNone(decisive_script("Save As"))
        self.assertIsNone(decisive_script("0.6.8"))
        self.assertIsNone(decisive_script("PINGYITONG_MODELS_DIR"))
        self.assertIsNone(decisive_script("安装版优先 SHA256 172d2d89 c29ea86f 安装目录找权重"))

    def test_wrong_script_output_guard(self) -> None:
        """输出校验用宽松阈值：少量谚文/假名即拦截（≥2 字符且 >3%）。"""
        # 实测复现的残缺输出：2 个谚文字符混在中文+标识符里
        self.assertTrue(_wrong_script("버전 0.6.8:검증 1veriyinstaller0lo- 与 1 个", "zh"))
        self.assertTrue(_wrong_script("삼성", "zh"))
        self.assertTrue(_wrong_script("ソニー", "zh"))
        self.assertTrue(_wrong_script("Это текст", "zh"))
        # 目标语言本身的脚本不拦截
        self.assertFalse(_wrong_script("版本 0.6.8", "zh"))
        self.assertTrue(_wrong_script("Save As", "zh"))
        self.assertFalse(_wrong_script("한국어 텍스트", "ko"))
        self.assertFalse(_wrong_script("Это текст", "ru"))
        # 日文目标放行汉字（无法靠脚本与中文区分，保守不误伤）
        self.assertFalse(_wrong_script("这是中文输出", "ja"))
        # 非中日韩俄目标不允许出现任何决定性 CJK/西里尔输出
        self.assertTrue(_wrong_script("中文输出", "en"))
        self.assertFalse(_wrong_script("", "zh"))

    def test_wrong_language_segment_is_retried(self) -> None:
        """批量中被翻成韩文的段落应单段重试并取回正确译文。"""
        drifted = "버전 0.6.8 검증 설치 패키지와 함께 다운로드됩니다"
        pool = _KoreanDriftPool(drifted)
        service, _ = make_service(pool=pool)

        frame = make_frame(
            block("Dashboard", 0, 0, 240, 32),
            block(drifted, 0, 400, 400, 32),
            block("Save Changes", 0, 800, 240, 32),
        )
        render_frame = service.translate_frame(frame)
        texts = [item.text for item in render_frame.items]

        for text in texts:
            self.assertNotRegex(text, r"[가-힯]", f"覆盖层出现谚文: {text!r}")
        # 漂移段被重试恢复为中文译文
        self.assertIn(f"译:{_KoreanDriftPool.RECOVERED}", texts)
        # 一次批量请求 + 仅为漂移段的一次单段重试
        self.assertEqual(len(pool.calls), 2)

    def test_persistent_wrong_language_keeps_original(self) -> None:
        """重试仍输出韩文时保留原文——韩文'译文'比原文更糟。"""
        drifted = "The quick brown fox jumps."
        pool = _AlwaysKoreanPool(drifted)
        service, _ = make_service(pool=pool)

        frame = make_frame(
            block("Dashboard", 0, 0, 240, 32),
            block(drifted, 0, 400, 400, 32),
        )
        render_frame = service.translate_frame(frame)
        texts = [item.text for item in render_frame.items]

        self.assertNotIn(drifted, texts)  # 原屏自然保留，不复制成译文
        for text in texts:
            self.assertNotRegex(text, r"[\uac00-\ud7af]", f"覆盖层出现谚文: {text!r}")

    def test_delimiter_residue_is_retried(self) -> None:
        """分隔符残缺变体（<<SEG>>>）不应出现在覆盖层上。"""
        residue = "Installer 0.6.8 packages"
        pool = _DelimiterResiduePool(residue)
        service, _ = make_service(pool=pool)

        frame = make_frame(
            block("Dashboard", 0, 0, 240, 32),
            block(residue, 0, 400, 400, 32),
        )
        render_frame = service.translate_frame(frame)
        texts = [item.text for item in render_frame.items]

        for text in texts:
            self.assertNotIn("<SEG", text, f"覆盖层出现分隔符残片: {text!r}")
        self.assertIn(fake_translation(residue), texts)

    def test_foreign_residue_in_same_language_paragraph_is_translated(self) -> None:
        """主体是中文但嵌着成块韩文的段落不应被同语言跳过。

        对应"有些语言甚至不进行翻译覆盖"：此前 detected==target 直接跳过，
        夹在中文里的韩文块永远不被翻译。修复后该段落会被送模型
        （假池没有真实翻译能力，这里只断言"未被跳过"）。
        """
        mixed = "这是一个中文说明段落，其中嵌着一句韩文：버전 검증이 완료되었습니다 그리고 더 많은 내용"
        # 韩文字符占比超过 10%（严格阈值），判定为夹带外文
        service, pool = make_service()
        service.translate_frame(make_frame(block(mixed, 0, 0, 900, 32)))
        self.assertGreaterEqual(len(pool.calls), 1)

    def test_pure_same_language_paragraph_still_skipped(self) -> None:
        """显式同语中文段落含少量拉丁标识符仍跳过，夹杂物检查不误伤。"""
        text = "安装版优先从安装目录找权重 SHA256 前 8 位 172d2d89"
        config = AppConfig()
        config.translation.semantic_review = False  # legacy single-pass contract
        config.translation.source_language = "zh"
        service, pool = make_service(config)
        service.translate_frame(make_frame(block(text, 0, 0, 600, 32)))
        self.assertEqual(len(pool.calls), 0)

    def test_translate_text_wrong_language_is_retried(self) -> None:
        """剪贴板等单条路径同样拦截韩文漂移并重试恢复。"""
        target = "The quick brown fox jumps."
        pool = _KoreanDriftPool(target)
        service, _ = make_service(pool=pool)
        result = service.translate_text(target)
        self.assertEqual(result, f"译:{_KoreanDriftPool.RECOVERED}")
        self.assertNotRegex(result, r"[\uac00-\ud7af]")
        self.assertEqual(len(pool.calls), 2)

    def test_translate_text_persistent_wrong_language_keeps_source(self) -> None:
        target = "The quick brown fox jumps."
        pool = _AlwaysKoreanPool(target)
        service, _ = make_service(pool=pool)
        result = service.translate_text(target)
        self.assertEqual(result, target)

    def test_recovered_output_is_cached_not_the_raw_bad_output(self) -> None:
        """恢复后的译文进入缓存，坏输出不得投毒缓存。

        实测缺陷：拒答/语言错误恢复后 `cache.put` 仍写入第一次的原始坏
        输出，同文本第二次翻译命中缓存直接返回致歉/韩文。
        """
        target = "The quick brown fox jumps."
        pool = _KoreanDriftPool(target)
        service, _ = make_service(pool=pool)
        first = service.translate_text(target)
        self.assertNotRegex(first, r"[\uac00-\ud7af]")
        # 第二次命中缓存：返回的仍必须是恢复后的干净译文
        second = service.translate_text(target)
        self.assertEqual(second, first)
        self.assertNotRegex(second, r"[\uac00-\ud7af]")


class _DeadPool(FakeModelPool):
    """推理引擎不可达：一律返回空串（等价引擎失败/加载中）。"""

    def translate(
        self,
        model_id: str,
        prompt: str,
        system_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        self.calls.append((model_id, prompt))
        return ""


class TestWithheldSuppression(unittest.TestCase):
    """"保留原文"判定后的重试抑制（第 75 轮延迟修复的另一半）。

    守卫主动拒绝（语言纠错失败、拒答恢复失败）是确定性失败，抑制窗口
    内同键段落不再送模型；引擎失败与术语恢复失败是瞬时的，下一帧照常
    重试（既有语义，见 test_dropped_placeholder_never_caches_incomplete_frame）。
    """

    def test_guard_rejected_segment_suppressed_next_frame(self) -> None:
        """语言守卫拒绝且纠错失败后，同键段落下帧零推理。"""
        pool = _AlwaysKoreanPool("Please save the file.")
        service, _ = make_service(pool=pool)
        first = service.translate_frame(make_frame(block("Please save the file.", 0, 0)))
        self.assertEqual(len(pool.calls), 2)  # 批量 + 纠错
        self.assertEqual(len(first.items), 0)  # 保留原文，不出条目

        second = service.translate_frame(make_frame(block("Please save the file.", 0, 0)))
        self.assertEqual(len(pool.calls), 2)  # 抑制生效：零新推理
        self.assertEqual(len(second.items), 0)

    def test_suppression_expires_after_window(self) -> None:
        """抑制窗口过期后恢复重试（窗口置 0 = 记录即刻过期）。"""
        pool = _AlwaysKoreanPool("Please save the file.")
        service, _ = make_service(pool=pool)
        with mock.patch.object(
            translation_service_module, "_WITHHELD_RETRY_SECONDS", 0.0
        ):
            service.translate_frame(make_frame(block("Please save the file.", 0, 0)))
            self.assertEqual(len(pool.calls), 2)
            service.translate_frame(make_frame(block("Please save the file.", 0, 0)))
            self.assertEqual(len(pool.calls), 4)  # 窗口已过期：批量 + 纠错各再来一次

    def test_engine_failure_is_not_suppressed(self) -> None:
        """引擎失败（空响应）是瞬时失败：下一帧照常重试。"""
        pool = _DeadPool()
        service, _ = make_service(pool=pool)
        service.translate_frame(make_frame(block("Please save the file.", 0, 0)))
        service.translate_frame(make_frame(block("Please save the file.", 0, 0)))
        self.assertEqual(len(pool.calls), 2)

    def test_glossary_failure_is_not_suppressed(self) -> None:
        """术语恢复失败是瞬时失败：下一帧照常重试（既有语义）。"""

        class DroppingPool(FakeModelPool):
            def __init__(self) -> None:
                super().__init__()
                self.responses = ["研究粒子。", "研究粒子。", "量子力学研究粒子。"]

            def translate(self, model_id, prompt, system_prompt, max_tokens=None):
                self.calls.append((model_id, prompt))
                return self.responses.pop(0)

        config = AppConfig()
        config.translation.semantic_review = False  # legacy single-pass contract
        config.translation.source_language = "en"
        config.schedule.mode = "manual"
        config.schedule.active_model = MODEL_FAST
        pool = DroppingPool()
        glossary = Glossary([GlossaryEntry("Quantum mechanics", "量子力学")])
        service, _ = make_service(config, glossary, pool)
        text = "Quantum mechanics studies particles."
        service.translate_frame(make_frame(block(text, 0, 0)))
        result = service.translate_frame(make_frame(block(text, 0, 0)))
        self.assertEqual(result.items[0].text, "量子力学研究粒子。")
        self.assertEqual(len(pool.calls), 3)


class _RecordingScheduler(ModelScheduler):
    """记录 decide() 收到的术语命中数。

    用于回归一个真实缺陷：旧实现把 ``glossary_term_count`` 硬编码为 0
    （术语替换发生在调度之后，调度时还没有命中数），导致调度器的
    "术语密度高 → 使用高精度模型"内容自适应分支永远不会触发。
    """

    def __init__(self, config: object) -> None:
        super().__init__(config)
        self.seen_glossary_term_counts: list[int] = []

    def decide(
        self,
        snapshot: ResourceSnapshot | None,
        texts: list[str] | None = None,
        glossary_term_count: int = 0,
        cpu_sustained_high: bool = False,
    ):
        self.seen_glossary_term_counts.append(glossary_term_count)
        return super().decide(
            snapshot=snapshot,
            texts=texts,
            glossary_term_count=glossary_term_count,
            cpu_sustained_high=cpu_sustained_high,
        )


class TestUiTokenFallback(unittest.TestCase):
    """确定性 UI 短词短路：模型必然回显的 "OK" 直接映射标准译名。

    1.7B 对独立成段的 "OK" 在精简/完整两种提示词下均稳定回显（30 个
    常见 UI 词探针仅此一例），语言守卫只能保留原文；短路既修精度又
    省掉两次注定失败的推理（~200-280ms/帧）。
    """

    def test_ok_short_circuits_without_model_call(self) -> None:
        service, pool = make_service()
        render_frame = service.translate_frame(make_frame(block("OK", 0, 0)))
        self.assertEqual(len(render_frame.items), 1)
        self.assertEqual(render_frame.items[0].text, "确定")
        self.assertEqual(pool.calls, [])

    def test_redo_maps_to_standard_ui_term(self) -> None:
        # Redo 与 OK 同理：compact 提示词下模型稳定错译"重新"（8/8），
        # 完整提示词才是"重做"——映射保证全路径一致。
        service, pool = make_service()
        render_frame = service.translate_frame(make_frame(block("Redo", 0, 0)))
        self.assertEqual(render_frame.items[0].text, "重做")
        self.assertEqual(pool.calls, [])

    def test_case_insensitive_but_whole_token_only(self) -> None:
        # 大小写不敏感（OK/Ok/ok 同源同译）；非整词（OK Cancel）不命中，
        # 交给模型正常翻译。
        for text in ("OK", "Ok", "ok"):
            service, pool = make_service()
            render_frame = service.translate_frame(make_frame(block(text, 0, 0)))
            self.assertEqual(render_frame.items[0].text, "确定", text)
            self.assertEqual(pool.calls, [])
        service, pool = make_service()
        service.translate_frame(make_frame(block("OK Cancel", 0, 0)))
        self.assertEqual(len(pool.calls), 1)

    def test_fallback_fills_cache_for_later_frames(self) -> None:
        # 首帧短路并写入缓存，后续帧走缓存快路径，两条都出标准译名。
        service, pool = make_service()
        first = service.translate_frame(make_frame(block("OK", 0, 0), seq=1))
        second = service.translate_frame(make_frame(block("OK", 0, 0), seq=2))
        self.assertEqual(first.items[0].text, "确定")
        self.assertEqual(second.items[0].text, "确定")
        self.assertEqual(pool.calls, [])

    def test_translate_text_short_circuit(self) -> None:
        service, pool = make_service()
        self.assertEqual(service.translate_text("OK"), "确定")
        self.assertEqual(pool.calls, [])

    def test_non_zh_target_not_intercepted(self) -> None:
        # 映射值是中文译名，非 zh 目标必须放行给模型（en→ja 正常推理）。
        config = AppConfig()
        config.translation.semantic_review = False  # legacy single-pass contract
        config.translation.source_language = "en"
        config.translation.target_language = "ja"
        service, pool = make_service(config)
        service.translate_frame(make_frame(block("OK", 0, 0)))
        self.assertEqual(len(pool.calls), 1)


class TestSchedulingIntegration(unittest.TestCase):
    """调度的接入。"""

    def test_uses_scheduler_selected_model(self) -> None:
        """应使用调度器选定的模型发起推理。"""
        config = AppConfig()
        config.translation.semantic_review = False  # legacy single-pass contract
        config.schedule.mode = "manual"
        config.schedule.active_model = MODEL_FAST
        service, pool = make_service(config)

        service.translate_frame(make_frame(block("Hello world", 0, 0)))
        self.assertEqual(len(pool.calls), 1)
        self.assertEqual(pool.calls[0][0], MODEL_FAST)

    def test_scheduler_receives_real_glossary_hits(self) -> None:
        """调度器应收到真实的术语命中数，而不是恒为 0。"""
        config = AppConfig()
        config.translation.semantic_review = False  # legacy single-pass contract
        config.schedule.mode = "auto"
        config.schedule.content_adaptive = True
        glossary = Glossary([GlossaryEntry("GPU", "图形处理器")])
        service, pool = make_service(config, glossary=glossary)
        scheduler = _RecordingScheduler(config.schedule)
        service._scheduler = scheduler

        service.translate_frame(make_frame(block("Use GPU here", 0, 0)))

        self.assertEqual(len(pool.calls), 1)
        self.assertTrue(scheduler.seen_glossary_term_counts)
        self.assertGreaterEqual(scheduler.seen_glossary_term_counts[-1], 1)

    def test_snapshot_does_not_break_translation(self) -> None:
        """传入资源快照不应影响正常翻译（回归：快照曾导致内容自适应失效）。"""
        service, pool = make_service()
        snapshot = ResourceSnapshot(cpu_percent=10.0, render_fps=15.0)

        render_frame = service.translate_frame(
            make_frame(block("A short phrase", 0, 0)), snapshot=snapshot
        )
        self.assertEqual(len(render_frame.items), 1)
        self.assertEqual(len(pool.calls), 1)

    def test_stats_reported(self) -> None:
        """统计信息应被填充，供状态面板展示。"""
        service, _ = make_service()
        service.translate_frame(make_frame(block("Hello world", 0, 0)))

        stats = service.last_stats
        self.assertEqual(stats.paragraphs, 1)
        self.assertEqual(stats.translated, 1)
        self.assertEqual(stats.skipped, 0)
        self.assertEqual(stats.batches, 1)
        self.assertEqual(stats.failed_batches, 0)
        self.assertTrue(stats.model_id)
        self.assertIn("段", stats.as_text())
        # 注意：不断言 stats.elapsed > 0 —— 使用假推理引擎时整个流程在
        # 微秒级完成，而 Windows 上 time.monotonic() 的分辨率约为 15ms，
        # 断言会随机失败。真实推理耗时由性能压测单独度量。

    def test_stats_counts_skipped_same_language(self) -> None:
        """被跳过的段落应计入 skipped，与已翻译的段落区分开。"""
        config = AppConfig()
        config.translation.semantic_review = False  # legacy single-pass contract
        config.translation.source_language = "zh"
        service, _ = make_service(config)
        # 显式源和目标均为中文，中文内容会被跳过
        service.translate_frame(make_frame(block("这是中文内容", 0, 0)))

        stats = service.last_stats
        self.assertEqual(stats.skipped, 1)
        self.assertEqual(stats.translated, 0)


class TestTranslateText(unittest.TestCase):
    """单条文本翻译（供非屏幕场景使用）。"""

    def test_plain_text_uses_compact_prompt(self) -> None:
        class PromptPool(FakeModelPool):
            system_prompt = ""

            def translate(self, model_id, prompt, system_prompt, max_tokens=None):
                self.system_prompt = system_prompt
                return super().translate(model_id, prompt, system_prompt, max_tokens)

        pool = PromptPool()
        service, _ = make_service(pool=pool)
        service.translate_text("Open Settings")
        self.assertNotIn("separator", pool.system_prompt)

    def test_concise_plain_text_avoids_batch_separator_prompt(self) -> None:
        class PromptPool(FakeModelPool):
            system_prompt = ""

            def translate(self, model_id, prompt, system_prompt, max_tokens=None):
                self.system_prompt = system_prompt
                return super().translate(model_id, prompt, system_prompt, max_tokens)

        config = AppConfig()
        config.translation.semantic_review = False  # legacy single-pass contract
        config.translation.concise_output = True
        pool = PromptPool()
        service, _ = make_service(config, pool=pool)
        service.translate_text("Do not close before the backup finishes.")
        self.assertIn("CONCISE MODE", pool.system_prompt)
        self.assertNotIn(SEGMENT_DELIMITER, pool.system_prompt)

    def test_glossary_text_uses_protected_prompt(self) -> None:
        class PromptPool(FakeModelPool):
            system_prompt = ""

            def translate(self, model_id, prompt, system_prompt, max_tokens=None):
                self.system_prompt = system_prompt
                return super().translate(model_id, prompt, system_prompt, max_tokens)

        pool = PromptPool()
        glossary = Glossary([GlossaryEntry("OpenAI", "开放人工智能")])
        service, _ = make_service(glossary=glossary, pool=pool)
        service.translate_text("OpenAI provides useful tools.")
        self.assertIn("protected glossary", pool.system_prompt)
        self.assertIn(GLOSSARY_TOKEN_TEMPLATE.format(0), pool.system_prompt)

    def test_translate_text_returns_translation(self) -> None:
        """应返回译文。"""
        service, _ = make_service()
        self.assertEqual(service.translate_text("Hello"), fake_translation("Hello"))

    def test_empty_text_returns_empty(self) -> None:
        """空文本返回空字符串。"""
        service, _ = make_service()
        self.assertEqual(service.translate_text(""), "")
        self.assertEqual(service.translate_text("   "), "")

    def test_translate_text_uses_cache(self) -> None:
        """重复调用应命中缓存。"""
        service, pool = make_service()
        service.translate_text("Hello")
        service.translate_text("Hello")
        self.assertEqual(len(pool.calls), 1)

    def test_translate_text_applies_glossary(self) -> None:
        """单条翻译也应走术语替换与还原。"""
        glossary = Glossary([GlossaryEntry("GPU", "图形处理器")])
        service, _ = make_service(glossary=glossary)
        result = service.translate_text("Use GPU")
        self.assertIn("图形处理器", result)
        self.assertNotIn(GLOSSARY_TOKEN_TEMPLATE.format(0), result)

    def test_code_text_is_returned_without_model_call(self) -> None:
        """命令行文本原样返回，与批量路径同口径、不付推理延迟。"""
        service, pool = make_service()
        text = 'cd "F:/讯飞" && python run.py --debug'
        self.assertEqual(service.translate_text(text), text)
        self.assertEqual(pool.calls, [])


class TestClipboardOutput(unittest.TestCase):
    """复制译文所需的数据。"""

    def test_last_output_pairs(self) -> None:
        """应保留原文与译文的配对，供一键复制使用。"""
        service, _ = make_service()
        service.translate_frame(make_frame(block("Hello world", 0, 0)))

        pairs = service.last_output()
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][0], "Hello world")
        self.assertEqual(pairs[0][1], fake_translation("Hello world"))

    def test_last_output_text(self) -> None:
        """纯文本形式应可直接写入剪贴板。"""
        service, _ = make_service()
        service.translate_frame(make_frame(block("Hello world", 0, 0)))
        self.assertEqual(service.last_output_text(), fake_translation("Hello world"))


class _RecordingPool(FakeModelPool):
    """记录每次请求的 max_tokens，用于验证动态输出上限。"""

    def __init__(self, prefix: str = "译:") -> None:
        super().__init__(prefix)
        self.max_tokens_seen: list[int | None] = []

    def translate(
        self,
        model_id: str,
        prompt: str,
        system_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        self.max_tokens_seen.append(max_tokens)
        return super().translate(
            model_id=model_id,
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
        )


class TestDynamicOutputCap(unittest.TestCase):
    """按输入长度收紧输出上限：正常翻译不触顶，跑飞输出被封顶。"""

    def test_estimate_source_tokens(self) -> None:
        from pingyitong.services.translation_service import _estimate_source_tokens

        # CJK 按 1 字符 1 token（BPE 最坏情况）
        self.assertEqual(_estimate_source_tokens("今天天气很好"), 6)
        # 英文按 3 字符 1 token（保守折算）
        self.assertEqual(_estimate_source_tokens("abcdefghijk"), 11 // 3)
        # 混合文本分段计算
        self.assertEqual(_estimate_source_tokens("设置Settings"), 2 + 8 // 3)

    def test_output_cap_bounds(self) -> None:
        from pingyitong.services.translation_service import _output_cap

        # 短输入被显著收紧（8 字符 CJK → 2.5×8+32 = 52）
        self.assertEqual(_output_cap("设置按钮取消操作", 384, False), 52)
        # 不低于下限
        self.assertEqual(_output_cap("OK", 384, False), 48)
        # 长输入不超过模型上限
        self.assertEqual(_output_cap("a" * 4000, 384, False), 384)
        self.assertEqual(_output_cap("a" * 4000, 768, False), 768)
        # 对照模式输出翻倍（仍受模型上限约束）
        self.assertEqual(_output_cap("设置按钮", 384, True), 84)
        self.assertEqual(_output_cap("a" * 4000, 384, True), 384)

    def test_short_single_request_gets_capped_tokens(self) -> None:
        service, pool = make_service(pool=_RecordingPool())
        service.translate_frame(make_frame(block("Save file", 0, 0)))
        self.assertEqual(len(pool.max_tokens_seen), 1)
        # 9 字符英文 → est 3 token → cap 39 → 下限 48，远小于模型上限
        self.assertEqual(pool.max_tokens_seen[0], 48)

    def test_long_input_keeps_model_max_tokens(self) -> None:
        service, pool = make_service(pool=_RecordingPool())
        long_text = "This is a long English paragraph. " * 40
        service.translate_frame(make_frame(block(long_text, 0, 0)))
        self.assertEqual(pool.max_tokens_seen[0], 384)

    def test_batch_cap_covers_all_segments(self) -> None:
        service, pool = make_service(pool=_RecordingPool())
        # 行距取 45px（> 1.7 倍行高的段落合并阈值 ≈ 27px）：6 个短标签保持
        # 独立成段进入同一批请求；间距过近会被预处理按折行段落合并成单段，
        # 批量 cap 的分隔符记账路径就不会被触发
        blocks = [block(f"Menu item {i}", 0, i * 45) for i in range(6)]
        service.translate_frame(make_frame(*blocks))
        # 六段成批：joined 71 字符 → est 23 → 89，
        # 另加 5 个分隔符 × 16 记账 → 169，且仍小于模型上限
        self.assertEqual(pool.max_tokens_seen[0], 89 + 5 * 16)

    def test_batch_cap_accounts_for_delimiter_allowance(self) -> None:
        from pingyitong.services.translation_service import (
            _SEGMENT_DELIMITER_TOKEN_ALLOWANCE,
            _output_cap,
        )

        text = "\n".join(f"Menu item {i}" for i in range(6))
        single = _output_cap(text, 768, False)
        batched = _output_cap(text, 768, False, segments=6)
        # 同一源文本、仅片段数不同：差值恰为分隔符记账
        self.assertEqual(batched - single, 5 * _SEGMENT_DELIMITER_TOKEN_ALLOWANCE)
        # 上限仍受模型 max_tokens 钳制
        self.assertEqual(_output_cap("a" * 4000, 768, False, segments=20), 768)

    def test_batch_cap_delimiter_allowance_reaches_pool(self) -> None:
        """六段长短不一的文本真正成批：分隔符记账应体现在池收到的上限里。"""
        service, pool = make_service(pool=_RecordingPool())
        texts = [
            "Save your work",
            "Open the settings dialog",
            "Settings menu",
            "Cancel the operation",
            "Apply changes now",
            "Exit program",
        ]
        blocks = [block(t, 0, i * 60) for i, t in enumerate(texts)]
        service.translate_frame(make_frame(*blocks))
        # joined 105 字符 → est 35 → 119，加 5 个分隔符 × 16 记账 → 199
        self.assertEqual(pool.max_tokens_seen[0], 199)


class _ApiPromptPool(FakeModelPool):
    """记录 system_prompt 与 max_tokens 的模型池，可模拟在线模式激活。"""

    def __init__(self, api_active: bool = True) -> None:
        super().__init__()
        self.api_active_flag = api_active
        self.system_prompts: list[str] = []
        self.max_tokens_seen: list[int | None] = []

    def translate(self, model_id, prompt, system_prompt, max_tokens=None):
        self.system_prompts.append(system_prompt)
        self.max_tokens_seen.append(max_tokens)
        return super().translate(model_id, prompt, system_prompt, max_tokens)

    def api_mode_active(self) -> bool:
        return self.api_active_flag


class TestApiModePromptSelection(unittest.TestCase):
    """在线 AI 模式下单段也应使用完整提示词。

    精简提示词是为本地 1.7B 省 prefill 设计的（实测省 ~15ms）；
    外部模型的延迟由网络往返主导，prefill 近乎免费。精简版缺少
    "优先软件 UI 标准译法 / 已是目标语言则原样返回 / 不加寒暄"
    三条质量规则，在线模式下换用完整提示词是纯精度收益。
    """

    @staticmethod
    def _api_config() -> AppConfig:
        config = AppConfig()
        config.translation.semantic_review = False  # legacy single-pass contract
        config.api.enabled = True
        config.api.base_url = "http://127.0.0.1:9/v1"
        config.api.model_name = "test-model"
        return config

    def test_api_mode_single_frame_uses_full_prompt(self) -> None:
        """在线模式下单个普通段落用完整提示词。"""
        pool = _ApiPromptPool()
        service, _ = make_service(self._api_config(), pool=pool)
        service.translate_frame(make_frame(block("Open Settings", 0, 0)))

        self.assertEqual(len(pool.system_prompts), 1)
        # 完整提示词的两个标志：分隔符规则与 UI 标准译法要求
        self.assertIn("separator", pool.system_prompts[0])
        self.assertIn("standard localized term", pool.system_prompts[0])

    def test_local_mode_single_frame_keeps_compact_prompt(self) -> None:
        """本地模式单段仍用精简提示词（保留 prefill 优化）。"""
        pool = _ApiPromptPool(api_active=False)
        service, _ = make_service(pool=pool)
        service.translate_frame(make_frame(block("Open Settings", 0, 0)))

        self.assertEqual(len(pool.system_prompts), 1)
        self.assertNotIn("separator", pool.system_prompts[0])

    def test_api_mode_translate_text_uses_full_prompt(self) -> None:
        """在线模式下单条文本翻译同样用完整提示词。"""
        pool = _ApiPromptPool()
        service, _ = make_service(self._api_config(), pool=pool)
        service.translate_text("Open Settings")

        self.assertEqual(len(pool.system_prompts), 1)
        self.assertIn("separator", pool.system_prompts[0])

    def test_local_mode_translate_text_keeps_compact_prompt(self) -> None:
        """本地模式单条文本翻译仍用精简提示词。"""
        pool = _ApiPromptPool(api_active=False)
        service, _ = make_service(pool=pool)
        service.translate_text("Open Settings")

        self.assertEqual(len(pool.system_prompts), 1)
        self.assertNotIn("separator", pool.system_prompts[0])

    def test_api_mode_fallback_still_uses_safe_prompt(self) -> None:
        """在线配置完整但请求失败回退本地时，完整提示词对本地模型同样合法。

        术语命中的单段本来就走完整提示词，本地模型处理它没有问题；
        这里固定该行为，防止有人把回退路径改成精简提示词。
        """
        pool = _ApiPromptPool()
        glossary = Glossary([GlossaryEntry("OpenAI", "开放人工智能")])
        service, _ = make_service(self._api_config(), glossary=glossary, pool=pool)
        service.translate_text("OpenAI provides useful tools.")

        self.assertIn("protected glossary", pool.system_prompts[0])


class TestApiModeOutputCap(unittest.TestCase):
    """在线模式的输出上限由动态估计主导，长文不被 768 默认上限截断。

    ``api.max_tokens`` 默认 768 且未在界面暴露；旧逻辑把它当硬顶
    （``min(768, 动态估计)``），长段落与密集帧的译文会在半途被截断——
    截断是用户可见的精度事故，批量下还连带对位失败触发逐条重试。
    现行为：在线模式以动态估计（2.5×源 token + 分隔符记账）决定上限，
    8192 硬顶只防跑飞；本地模式行为不变。
    """

    @staticmethod
    def _api_config() -> AppConfig:
        config = AppConfig()
        config.translation.semantic_review = False  # legacy single-pass contract
        config.api.enabled = True
        config.api.base_url = "http://127.0.0.1:9/v1"
        config.api.model_name = "test-model"
        return config

    def test_api_mode_long_paragraph_not_clamped_to_768(self) -> None:
        """长段落在在线模式下按动态估计放开上限（>768）。"""
        service, pool = make_service(
            self._api_config(), pool=_ApiPromptPool()
        )
        long_text = "This is a long English paragraph. " * 40  # 1360 字符
        service.translate_frame(make_frame(block(long_text, 0, 0)))

        self.assertEqual(len(pool.max_tokens_seen), 1)
        # est = 1360/3 = 453 → int(453×2.5)+32 = 1164，不再被 768 钳制
        self.assertEqual(pool.max_tokens_seen[0], 1164)

    def test_api_mode_short_frame_keeps_dynamic_cap(self) -> None:
        """短帧仍由动态估计收紧（不是放开到 768/8192）。"""
        service, pool = make_service(
            self._api_config(), pool=_ApiPromptPool()
        )
        service.translate_frame(make_frame(block("Open Settings", 0, 0)))
        # 13 字符英文 → est 4 → 42 → 下限 48
        self.assertEqual(pool.max_tokens_seen[0], 48)

        # 六短标签成批：与本地模式同款断言（动态估计 + 分隔符记账主导）
        service2, pool2 = make_service(
            self._api_config(), pool=_ApiPromptPool()
        )
        texts = [
            "Save your work",
            "Open the settings dialog",
            "Settings menu",
            "Cancel the operation",
            "Apply changes now",
            "Exit program",
        ]
        blocks = [block(t, 0, i * 60) for i, t in enumerate(texts)]
        service2.translate_frame(make_frame(*blocks))
        self.assertEqual(pool2.max_tokens_seen[0], 199)

    def test_api_mode_translate_text_long_input(self) -> None:
        """剪贴板/单条翻译的长文同样不受 768 截断。"""
        service, pool = make_service(
            self._api_config(), pool=_ApiPromptPool()
        )
        service.translate_text("OpenAI provides useful tools. " * 60)
        # strip 后 1799 字符（末尾空格被去）→ est = 599 → int(599×2.5)+32 = 1529
        self.assertEqual(pool.max_tokens_seen[0], 1529)

    def test_api_mode_absurd_input_hits_hard_limit(self) -> None:
        """超长输入的动态估计触到 8192 硬顶（跑飞保护仍在）。"""
        service, pool = make_service(
            self._api_config(), pool=_ApiPromptPool()
        )
        service.translate_text("word " * 4000)  # 20000 字符 → est 6666
        self.assertEqual(pool.max_tokens_seen[0], 8192)


class _OnlineFailureBumpPool(FakeModelPool):
    """模拟在线池：前 N 次翻译触发失败并推进命名空间代数。

    真实流程中在线请求失败由 ModelPool 内部回退本地并递增
    failure_generation；这里用可变命名空间复现同一时序——
    帧开始捕获旧代数，帧内失败后代数已变化。
    """

    def __init__(self, failures: int = 1) -> None:
        super().__init__()
        self._generation = 0
        self._remaining_failures = failures

    def translate(self, model_id, prompt, system_prompt, max_tokens=None):
        result = super().translate(model_id, prompt, system_prompt, max_tokens)
        if self._remaining_failures > 0:
            self._remaining_failures -= 1
            self._generation += 1
        return result

    def api_mode_active(self) -> bool:
        return True

    def translation_cache_namespace(self) -> str:
        return f"api:1:{self._generation}"


class TestFailedFrameStillPublishes(unittest.TestCase):
    """在线帧内失败后：观察态照常发布，缓存整体不入。

    缓存命名空间含失败代数（防本地回退结果污染在线缓存）。旧实现把
    "命名空间已变化"等同于"请求被取代"，连统计/输出/历史一起丢弃——
    用户按复制键拿到的是上一帧的旧译文，历史查看器也缺这一帧。
    屏幕上渲染的就是这份（部分回退的）结果，观察态必须与屏幕一致；
    只有缓存写入需要随命名空间作废。
    """

    @staticmethod
    def _api_config() -> AppConfig:
        config = AppConfig()
        config.translation.semantic_review = False  # legacy single-pass contract
        config.api.enabled = True
        config.api.base_url = "http://127.0.0.1:9/v1"
        config.api.model_name = "test-model"
        return config

    def test_frame_with_inframe_failure_still_publishes_output(self) -> None:
        """帧内失败推进命名空间后，本帧译文仍进入复制/历史可见的输出。"""
        service, _ = make_service(self._api_config(), pool=_OnlineFailureBumpPool())
        render = service.translate_frame(make_frame(block("Open Settings", 0, 0)))

        self.assertEqual(len(render.items), 1)
        self.assertEqual(len(service.last_output()), 1)
        self.assertTrue(service.last_output_text().strip())

    def test_frame_with_inframe_failure_does_not_poison_cache(self) -> None:
        """回退结果不入缓存，下一帧在线重试；健康帧恢复正常缓存。"""
        service, pool = make_service(self._api_config(), pool=_OnlineFailureBumpPool())

        service.translate_frame(make_frame(block("Open Settings", 0, 0)))
        self.assertEqual(len(service.cache), 0)

        # 第二帧：代数已稳定，在线重试后结果应正常入缓存
        service.translate_frame(make_frame(block("Open Settings", 0, 0), seq=2))
        self.assertEqual(len(pool.calls), 2)  # 未被旧命名空间缓存截胡
        self.assertEqual(len(service.cache), 1)

        # 第三帧：同文本直接命中缓存，不再送模型
        service.translate_frame(make_frame(block("Open Settings", 0, 0), seq=3))
        self.assertEqual(len(pool.calls), 2)
        self.assertEqual(len(service.cache), 1)

    def test_superseded_frame_still_drops_everything(self) -> None:
        """被设置变更取代的请求仍整体丢弃（乱序保护不因本次修复放松）。"""
        holder: dict[str, object] = {}

        class _ConfigChurnPool(FakeModelPool):
            def translate(self, model_id, prompt, system_prompt, max_tokens=None):
                updated = AppConfig()
                updated.translation.semantic_review = False  # legacy single-pass contract
                updated.translation.target_language = "en"
                holder["service"].apply_config(updated)
                return super().translate(model_id, prompt, system_prompt, max_tokens)

        pool = _ConfigChurnPool()
        service, _ = make_service(self._api_config(), pool=pool)
        holder["service"] = service

        service.translate_frame(make_frame(block("Open Settings", 0, 0)))
        self.assertEqual(service.last_output(), [])
        self.assertEqual(len(service.cache), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
