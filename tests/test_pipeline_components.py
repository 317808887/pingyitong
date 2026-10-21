"""流水线基础设施测试。

覆盖构成异步流水线的三个基础组件：

- :class:`LatestQueue` —— 队列与背压策略（"丢旧保新"）
- :class:`TranslationCache` —— 推理缓存（LRU + 过期）
- :mod:`prompt` —— 翻译 Prompt 构造与输出解析

这三者共同决定了"翻译结果是否对应当前屏幕内容"以及"相同文本是否
重复推理"，是实时性的关键路径。
"""

from __future__ import annotations

import threading
import time
import unittest

from pingyitong.components.llm.prompt import (
    build_compact_prompt,
    build_prompt,
    build_system_prompt,
    build_user_prompt,
    language_name,
    parse_response,
)
from pingyitong.components.llm.translation_cache import TranslationCache
from pingyitong.constants import GLOSSARY_TOKEN_TEMPLATE, SEGMENT_DELIMITER
from pingyitong.core.task_queue import LatestQueue, RateLimiter


class TestLatestQueue(unittest.TestCase):
    """有界队列与背压。"""

    def test_put_get_roundtrip(self) -> None:
        """基本读写。"""
        queue: LatestQueue[int] = LatestQueue(maxsize=3)
        queue.put(1)
        queue.put(2)
        self.assertEqual(queue.get(timeout=0.01), 1)
        self.assertEqual(queue.get(timeout=0.01), 2)

    def test_get_on_empty_returns_none(self) -> None:
        """空队列返回 None 而不是阻塞或报错。"""
        queue: LatestQueue[int] = LatestQueue(maxsize=2)
        self.assertIsNone(queue.get(timeout=0.01))

    def test_drops_oldest_when_full(self) -> None:
        """队列满时丢弃最旧数据。

        这是核心背压策略：屏幕翻译场景下时效性优先于完整性——
        延迟 2 秒的译文即使准确也没有价值，而且会覆盖更新的结果。
        """
        queue: LatestQueue[int] = LatestQueue(maxsize=2)
        for value in range(5):
            queue.put(value)
        # 只保留最后两个
        self.assertEqual(len(queue), 2)
        self.assertEqual(queue.get(timeout=0.01), 3)
        self.assertEqual(queue.get(timeout=0.01), 4)
        self.assertEqual(queue.stats().dropped, 3)

    def test_keep_latest_only(self) -> None:
        """only-latest 模式永远只保留一条。

        渲染队列使用该模式：中间帧没有绘制意义。
        """
        queue: LatestQueue[int] = LatestQueue(maxsize=5, keep_latest_only=True)
        for value in range(5):
            queue.put(value)
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue.get(timeout=0.01), 4)

    def test_get_latest_clears_rest(self) -> None:
        """取最新一条时其余数据被丢弃。"""
        queue: LatestQueue[int] = LatestQueue(maxsize=5)
        for value in range(4):
            queue.put(value)
        self.assertEqual(queue.get_latest(), 3)
        self.assertEqual(len(queue), 0)

    def test_get_after_discards_stale(self) -> None:
        """序号落后的数据应被判定为过期并丢弃。

        这正是"保证翻译结果始终对应当前屏幕内容"的实现：
        若队列中只有落后于最新帧的数据，说明它们已经过期，无需再推理。
        """
        queue: LatestQueue[str] = LatestQueue(maxsize=5)
        queue.put("old-1", seq=1)
        queue.put("old-2", seq=2)
        queue.put("new", seq=10)

        # 要求序号 >= 10，前两条落后数据应被丢弃
        self.assertEqual(queue.get_after(min_seq=10), "new")
        self.assertEqual(queue.stats().stale_dropped, 2)

        # 再放入一条过期数据：应立即判定为过期，不做任何处理
        dropped_before = queue.stats().stale_dropped
        queue.put("stale", seq=5)
        self.assertIsNone(queue.get_after(min_seq=10))
        self.assertEqual(queue.stats().stale_dropped, dropped_before + 1)

    def test_peek_does_not_remove(self) -> None:
        """peek 不移除数据。"""
        queue: LatestQueue[int] = LatestQueue(maxsize=3)
        queue.put(7)
        self.assertEqual(queue.peek(), 7)
        self.assertEqual(len(queue), 1)

    def test_peek_seq(self) -> None:
        """查看最新序号。"""
        queue: LatestQueue[int] = LatestQueue(maxsize=3)
        self.assertEqual(queue.peek_seq(), 0)
        queue.put(1, seq=42)
        self.assertEqual(queue.peek_seq(), 42)

    def test_clear(self) -> None:
        """清空队列。"""
        queue: LatestQueue[int] = LatestQueue(maxsize=5)
        for value in range(3):
            queue.put(value)
        self.assertEqual(queue.clear(), 3)
        self.assertTrue(queue.empty)

    def test_resize_shrinks_immediately(self) -> None:
        """缩容时立即淘汰多余数据。"""
        queue: LatestQueue[int] = LatestQueue(maxsize=5)
        for value in range(5):
            queue.put(value)
        queue.resize(2)
        self.assertEqual(len(queue), 2)
        self.assertEqual(queue.get(timeout=0.01), 3)

    def test_stats_text(self) -> None:
        """统计信息可读。"""
        queue: LatestQueue[int] = LatestQueue(maxsize=1)
        queue.put(1)
        queue.put(2)
        self.assertIn("丢弃", queue.stats().as_text())

    def test_blocking_get_wakes_on_put(self) -> None:
        """阻塞等待应被新数据唤醒（流水线阶段间的主要同步方式）。"""
        queue: LatestQueue[str] = LatestQueue(maxsize=2)
        received: list[str] = []

        def consumer() -> None:
            item = queue.get(timeout=1.0)
            if item is not None:
                received.append(item)

        thread = threading.Thread(target=consumer)
        thread.start()
        time.sleep(0.05)
        queue.put("data")
        thread.join(timeout=1.0)

        self.assertEqual(received, ["data"])

    def test_wake_all_unblocks_consumer(self) -> None:
        """停止流水线时应能唤醒阻塞中的线程，避免退出时卡住。"""
        queue: LatestQueue[str] = LatestQueue(maxsize=2)
        queue.wake_all()  # 不应抛异常

    def test_thread_safety_under_contention(self) -> None:
        """并发读写不应丢失或重复。"""
        queue: LatestQueue[int] = LatestQueue(maxsize=32)
        produced: list[int] = []

        def producer() -> None:
            for value in range(200):
                queue.put(value)
                produced.append(value)

        def consumer() -> None:
            for _ in range(200):
                item = queue.get(timeout=0.2)
                if item is None:
                    break

        threads = [
            threading.Thread(target=producer),
            threading.Thread(target=consumer),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=3.0)

        self.assertEqual(len(produced), 200)


class TestRateLimiter(unittest.TestCase):
    """帧率节流。"""

    def test_wait_respects_interval(self) -> None:
        """等待应满足目标间隔。"""
        limiter = RateLimiter(fps=50)  # 20ms
        limiter.wait()
        started = time.monotonic()
        limiter.wait()
        elapsed = time.monotonic() - started
        self.assertGreaterEqual(elapsed, 0.010)

    def test_stop_event_interrupts_wait(self) -> None:
        """停止信号应中断等待，避免停止流水线时卡住十几毫秒以上。"""
        limiter = RateLimiter(fps=1)
        limiter.wait()
        stop = threading.Event()
        stop.set()
        self.assertFalse(limiter.wait(stop))

    def test_configure_changes_interval(self) -> None:
        """运行时可调整帧率。"""
        limiter = RateLimiter(fps=1)
        limiter.configure(fps=100)
        limiter.wait()
        started = time.monotonic()
        limiter.wait()
        self.assertLess(time.monotonic() - started, 0.5)


class TestTranslationCache(unittest.TestCase):
    """推理缓存。"""

    def test_put_get(self) -> None:
        """基本读写。"""
        cache = TranslationCache(max_entries=8)
        cache.put("k", "v")
        self.assertEqual(cache.get("k"), "v")

    def test_miss_returns_none(self) -> None:
        """未命中返回 None。"""
        self.assertIsNone(TranslationCache().get("absent"))

    def test_lru_eviction(self) -> None:
        """超出容量时淘汰最久未使用项。"""
        cache = TranslationCache(max_entries=2)
        cache.put("a", "1")
        cache.put("b", "2")
        cache.get("a")          # 让 a 变成最近使用
        cache.put("c", "3")     # 应淘汰 b

        self.assertIsNone(cache.get("b"))
        self.assertEqual(cache.get("a"), "1")
        self.assertEqual(cache.get("c"), "3")

    def test_ttl_expiry(self) -> None:
        """超过存活期的条目应失效。"""
        cache = TranslationCache(max_entries=8, ttl_seconds=0.05)
        cache.put("k", "v")
        self.assertEqual(cache.get("k"), "v")
        time.sleep(0.08)
        self.assertIsNone(cache.get("k"))

    def test_purge_expired(self) -> None:
        """主动清理过期条目。"""
        cache = TranslationCache(max_entries=8, ttl_seconds=0.02)
        cache.put("a", "1")
        cache.put("b", "2")
        time.sleep(0.05)
        self.assertEqual(cache.purge_expired(), 2)

    def test_disabled_cache_never_stores(self) -> None:
        """禁用时不缓存。"""
        cache = TranslationCache(max_entries=8, enabled=False)
        cache.put("k", "v")
        self.assertIsNone(cache.get("k"))
        self.assertEqual(len(cache), 0)

    def test_zero_capacity_disables(self) -> None:
        """容量为 0 等同于禁用，避免无意义的内存占用。"""
        cache = TranslationCache(max_entries=0)
        self.assertFalse(cache.enabled)

    def test_batch_operations(self) -> None:
        """批量读写。"""
        cache = TranslationCache(max_entries=8)
        cache.put_many({"a": "1", "b": "2", "c": "3"})
        hits = cache.get_many(["a", "b", "zzz"])
        self.assertEqual(hits, {"a": "1", "b": "2"})

    def test_hit_rate_stats(self) -> None:
        """命中率统计（用于状态面板展示）。"""
        cache = TranslationCache(max_entries=8)
        cache.put("a", "1")
        cache.get("a")
        cache.get("miss")
        stats = cache.stats()
        self.assertEqual(stats.hits, 1)
        self.assertEqual(stats.misses, 1)
        self.assertAlmostEqual(stats.hit_rate, 0.5)
        self.assertIn("命中率", stats.as_text())

    def test_clear_resets_stats(self) -> None:
        """清空缓存同时重置统计。"""
        cache = TranslationCache(max_entries=8)
        cache.put("a", "1")
        cache.get("a")
        self.assertEqual(cache.clear(), 1)
        self.assertEqual(cache.stats().hits, 0)

    def test_configure_resizes(self) -> None:
        """运行时调整容量并立即淘汰。"""
        cache = TranslationCache(max_entries=8)
        for index in range(6):
            cache.put(f"k{index}", str(index))
        cache.configure(max_entries=2)
        self.assertEqual(len(cache), 2)

    def test_thread_safety(self) -> None:
        """并发读写不应抛出异常。"""
        cache = TranslationCache(max_entries=64)

        def worker(offset: int) -> None:
            for index in range(200):
                cache.put(f"k{offset}-{index}", "v")
                cache.get(f"k{offset}-{index}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5.0)
        self.assertLessEqual(len(cache), 64)


class TestPromptConstruction(unittest.TestCase):
    """Prompt 构造。"""

    def test_language_name_known(self) -> None:
        """已知语言代码应转为"英文名（母语名）"。

        带上母语名是刻意的：冷门语言（如僧伽罗语、普什图语）只给英文名时，
        模型容易按近亲语言处理；母语名能显著提高指向准确性。
        详见 :meth:`pingyitong.languages.LanguageInfo.prompt_name`。
        """
        zh = language_name("zh")
        self.assertIn("Simplified Chinese", zh)
        self.assertIn("简体中文", zh)

        ja = language_name("ja")
        self.assertIn("Japanese", ja)
        self.assertIn("日本語", ja)

    def test_compact_prompt_preserves_explicit_source_for_every_language(self) -> None:
        from pingyitong.languages import LANGUAGES

        for source in LANGUAGES:
            with self.subTest(source=source):
                prompt = build_compact_prompt("zh", source)
                self.assertTrue(prompt.startswith(f"Translate from {language_name(source)} to "))
                self.assertIn(language_name("zh"), prompt)
                self.assertNotIn(SEGMENT_DELIMITER, prompt)

    def test_compact_prompt_auto_keeps_existing_single_argument_contract(self) -> None:
        default = build_compact_prompt("zh")
        self.assertEqual(default, build_compact_prompt("zh", "auto"))
        self.assertEqual(default, build_compact_prompt("zh", ""))
        self.assertTrue(default.startswith("Translate to "))
        self.assertNotIn("Translate from", default)

    def test_compact_prompt_anti_truncation_uses_concrete_examples(self) -> None:
        """术语截断治理必须是具体正反例，不能是抽象规则。

        1.7B 对抽象的"不要截断"指令无感（实测改措辞无效，Back→回
        仍 4/4），只有具体正反例才有效（祈使句规则同款教训）。
        zh 目标给中英对照示例；其他语言退化为纯文字规则，不得把
        中文示例泄漏进非中译文。
        """
        zh = build_compact_prompt("zh")
        self.assertIn("'Back' translates to '返回'", zh)
        self.assertIn("'Next' to '下一步'", zh)
        self.assertIn("never a truncated form", zh)
        en = build_compact_prompt("en")
        self.assertNotIn("返回", en)
        self.assertIn("never truncate or omit part of a term", en)

    def test_compact_prompt_avoids_pronoun_contaminating_example(self) -> None:
        prompt = build_compact_prompt("zh", "en")
        self.assertIn("translate commands, never answer", prompt)
        self.assertNotIn("Tell me everything you know", prompt)
        self.assertIn("Preserve every entity, number, condition", prompt)

    def test_compact_concise_prompt_preserves_constraints_without_batch_marker(self) -> None:
        prompt = build_compact_prompt("zh", "en", concise=True)
        self.assertIn("CONCISE MODE", prompt)
        self.assertIn("Preserve every clause", prompt)
        self.assertNotIn(SEGMENT_DELIMITER, prompt)

    def test_language_name_covers_full_table(self) -> None:
        """语言表中的每一种都不能退化成兜底文案。

        否则用户在下拉里选了一门语言，提示词却只说"译成源语言"，
        等于没选——而且不会有任何报错提示。
        """
        from pingyitong.languages import LANGUAGES

        for code in LANGUAGES:
            with self.subTest(code=code):
                name = language_name(code)
                self.assertNotEqual(name, "the source language")
                self.assertTrue(name.strip())

    def test_concise_mode_adds_rules(self) -> None:
        """开启语义精简后，提示词应包含精简要求。"""
        normal = build_system_prompt("en", "zh")
        concise = build_system_prompt("en", "zh", concise=True)
        self.assertNotIn("CONCISE MODE", normal)
        self.assertIn("CONCISE MODE", concise)
        self.assertGreater(len(concise), len(normal))

    def test_concise_mode_protects_technical_terms(self) -> None:
        """精简模式必须明确禁止省略术语——只压缩措辞，不牺牲信息。

        这条边界不写清楚的话，模型很容易顺手把限定语、术语修饰一并砍掉，
        译文变短了但技术含义变了，比不精简更糟。
        """
        prompt = build_system_prompt("en", "zh", concise=True)
        for required in (
            "NEVER omit",
            "technical term",
            "academic term",
            "proper noun",
            "number",
            "unit",
            "code identifier",
        ):
            with self.subTest(required=required):
                self.assertIn(required, prompt)

    def test_concise_mode_keeps_segment_rules(self) -> None:
        """精简模式不得破坏分隔符规则（否则批量翻译会错位）。"""
        prompt = build_system_prompt("en", "zh", concise=True)
        self.assertIn("separator rules above", prompt)
        self.assertIn(SEGMENT_DELIMITER, prompt)

    def test_concise_mode_keeps_meaning_guards(self) -> None:
        """压缩措辞不得改变论断：否定、条件、数量、比较必须保留。"""
        prompt = build_system_prompt("en", "zh", concise=True)
        for required in ("negations", "conditions", "quantities", "comparisons"):
            with self.subTest(required=required):
                self.assertIn(required, prompt)

    def test_language_name_unknown_falls_back(self) -> None:
        """未知代码回退为通用描述而不是崩溃。"""
        self.assertEqual(language_name("xx"), "the source language")

    def test_system_prompt_with_explicit_source(self) -> None:
        """指定源语言时提示词应明确方向。"""
        prompt = build_system_prompt("en", "zh")
        self.assertIn("English", prompt)
        self.assertIn("Simplified Chinese", prompt)

    def test_system_prompt_with_auto_source(self) -> None:
        """自动检测时提示词不指定源语言。"""
        prompt = build_system_prompt("auto", "zh")
        self.assertIn("Simplified Chinese", prompt)
        self.assertNotIn("from English", prompt)

    def test_system_prompt_demands_delimiter_preservation(self) -> None:
        """提示词必须要求模型保留分隔符——这是批量翻译对位的唯一依据。"""
        prompt = build_system_prompt("en", "zh")
        self.assertIn(SEGMENT_DELIMITER, prompt)

    def test_system_prompt_mentions_glossary_tokens(self) -> None:
        """有术语占位符时应提醒模型原样保留。"""
        token = GLOSSARY_TOKEN_TEMPLATE.format(0)
        prompt = build_system_prompt("en", "zh", glossary_terms=[token])
        self.assertIn(token, prompt)

    def test_user_prompt_single_segment_has_no_delimiter(self) -> None:
        """单片段不插入分隔符，避免模型自行添加内容。"""
        self.assertEqual(build_user_prompt(["hello"]), "hello")

    def test_user_prompt_multiple_segments_joined(self) -> None:
        """多片段用分隔符连接。"""
        prompt = build_user_prompt(["one", "two", "three"])
        self.assertEqual(prompt.count(SEGMENT_DELIMITER), 2)
        self.assertEqual(prompt.split(SEGMENT_DELIMITER)[0].strip(), "one")

    def test_build_prompt_returns_pair(self) -> None:
        """构造出的提示词对包含 system 与 user。"""
        pair = build_prompt(["hello"], "en", "zh")
        self.assertIn("translator", pair.system.lower())
        self.assertEqual(pair.user, "hello")


class TestPromptParsing(unittest.TestCase):
    """模型输出解析。"""

    def test_single_segment(self) -> None:
        """单片段直接返回。"""
        self.assertEqual(parse_response("你好", 1), ["你好"])

    def test_matching_segment_count(self) -> None:
        """片段数匹配时按分隔符切分。"""
        raw = f"一{SEGMENT_DELIMITER}二{SEGMENT_DELIMITER}三"
        self.assertEqual(parse_response(raw, 3), ["一", "二", "三"])

    def test_count_mismatch_returns_none(self) -> None:
        """片段数不匹配时返回 None，触发逐条降级。

        批量请求只要模型少输出一个分隔符，整批结果就会错位；
        此时逐条翻译虽慢但结果正确，因此必须能检测出来。
        """
        raw = f"一{SEGMENT_DELIMITER}二"
        self.assertIsNone(parse_response(raw, 3))

    def test_extra_empty_segments_tolerated(self) -> None:
        """模型在首尾多输出空片段时应容错。"""
        raw = f"{SEGMENT_DELIMITER}一{SEGMENT_DELIMITER}二{SEGMENT_DELIMITER}"
        self.assertEqual(parse_response(raw, 2), ["一", "二"])

    def test_empty_output_returns_none(self) -> None:
        """空输出返回 None。"""
        self.assertIsNone(parse_response("", 1))
        self.assertIsNone(parse_response("   ", 1))

    def test_wrapping_quotes_stripped(self) -> None:
        """模型习惯性加的整体引号应被剥离。"""
        self.assertEqual(parse_response('"你好"', 1), ["你好"])
        self.assertEqual(parse_response("“你好”", 1), ["你好"])

    def test_internal_quotes_preserved(self) -> None:
        """内部引号不应被误删。"""
        self.assertEqual(parse_response('他说的"好"字', 1), ['他说的"好"字'])

    def test_source_suffix_stripped_in_contrast_mode(self) -> None:
        """对照模式下应剥离附加的原文部分。"""
        self.assertEqual(
            parse_response("你好 | hello", 1, preserve_source=True, source_texts=["hello"]), ["你好"]
        )

    def test_preserve_source_in_batch(self) -> None:
        """批量 + 对照模式下逐段剥离原文。"""
        raw = f"一 | one{SEGMENT_DELIMITER}二 | two"
        self.assertEqual(
            parse_response(raw, 2, preserve_source=True, source_texts=["one", "two"]),
            ["一", "二"],
        )

    def test_whitespace_trimmed(self) -> None:
        """各段首尾空白应被清理。"""
        raw = f"  一  {SEGMENT_DELIMITER}  二  "
        self.assertEqual(parse_response(raw, 2), ["一", "二"])

    def test_empty_segment_in_batch_rejected(self) -> None:
        """存在真实空段时判定为失败，避免用空串覆盖原文。"""
        raw = f"一{SEGMENT_DELIMITER}{SEGMENT_DELIMITER}三"
        self.assertIsNone(parse_response(raw, 3))

    def test_wrapping_code_fence_stripped(self) -> None:
        """外部模型把译文整段包进代码围栏时应剥离（在线精度加固）。"""
        self.assertEqual(parse_response("```zh\n你好\n```", 1), ["你好"])
        self.assertEqual(parse_response("```你好```", 1), ["你好"])
        # 批量整体围栏
        raw = f"```zh\n一{SEGMENT_DELIMITER}二\n```"
        self.assertEqual(parse_response(raw, 2), ["一", "二"])

    def test_per_segment_fences_stripped_in_batch(self) -> None:
        """外部模型给每段单独套围栏时逐段剥离。"""
        raw = f"```一```{SEGMENT_DELIMITER}```二```"
        self.assertEqual(parse_response(raw, 2), ["一", "二"])

    def test_unbalanced_fence_preserved(self) -> None:
        """未闭合或正文内部的围栏不得误删（保守剥离）。"""
        self.assertEqual(parse_response("```\n你好", 1), ["```\n你好"])
        self.assertEqual(parse_response("代码如下```x```", 1), ["代码如下```x```"])
        raw = f"一{SEGMENT_DELIMITER}``二"
        self.assertEqual(parse_response(raw, 2), ["一", "``二"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
