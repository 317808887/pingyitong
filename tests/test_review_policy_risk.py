"""风险分级审查（review_policy="risk"）与完整 JSON 的行为契约。

always 语义由 test_semantic_review.py 锁定；本文件锁定：
1. 低风险段跳过模型二审，直接发布，且不写任何缓存；
2. 高风险段（否定/条件/数字/CJK 源文）仍走完整二重审查；
3. 审查输出被截断时整批保留原文，不以无歧义性保证的前缀作证据。
"""
from __future__ import annotations

import json

import pytest

from pingyitong.components.llm.translation_cache import TranslationCache
from pingyitong.config.schema import AppConfig
from pingyitong.constants import MODEL_FAST, SEGMENT_DELIMITER
from pingyitong.core import ModelScheduler
from pingyitong.services.glossary_service import Glossary
from pingyitong.services.semantic_review import REVIEW_DIMENSIONS
from pingyitong.services.text_preprocess import TextPreprocessor
from pingyitong.services.translation_service import TranslationService
from tests.test_translation_service import block, make_frame

LOW_RISK = "The report contains many interesting pages."
HIGH_RISK = "Do not close the window before the backup finishes."


class CallCountingPool:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def translate_local(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0) if self.responses else ""

    def translate(self, **kwargs):  # pragma: no cover - must never be used
        raise AssertionError("reviewed content must stay local")

    def api_mode_active(self):
        return False


def risk_service(*responses, policy: str = "risk"):
    config = AppConfig()
    config.translation.source_language = "en"
    config.translation.target_language = "zh"
    config.translation.review_policy = policy
    config.schedule.mode = "manual"
    config.schedule.active_model = MODEL_FAST
    pool = CallCountingPool(*responses)
    service = TranslationService(
        config=config,
        model_pool=pool,  # type: ignore[arg-type]
        scheduler=ModelScheduler(config.schedule),
        cache=TranslationCache(),
        preprocessor=TextPreprocessor(),
        glossary=Glossary(),
        history=None,
    )
    return service, pool


def review_response(independent: str, index: int = 0):
    return {
        "id": index,
        "independent": independent,
        "checks": ["ok"] * len(REVIEW_DIMENSIONS),
        "compact": "",
        "compact_checks": ["ok"] * len(REVIEW_DIMENSIONS),
    }


def test_low_risk_segment_skips_model_review_and_publishes():
    translated = "报告包含许多有趣的页面。"
    service, pool = risk_service(translated)
    result = service.translate_frame(make_frame(block(LOW_RISK, 0, 0)))
    assert len(pool.calls) == 1, "低风险段只允许一次生成调用"
    item = result.items[0]
    assert item.text == translated
    assert item.quality_status == "accepted"
    assert item.display_text == ""
    assert service.cache.stats().verified_hits == 0, "未审译文不得进入可信缓存"
    # 再次翻译同一文本：不读已审缓存，重新生成（每次仍是 1 次调用）
    service.translate_frame(make_frame(block(LOW_RISK, 0, 0)))
    assert len(pool.calls) == 2


def test_high_risk_segment_still_gets_full_review():
    full = "备份完成前请勿关闭窗口。"
    service, pool = risk_service(full, json.dumps([review_response(full)]))
    result = service.translate_frame(make_frame(block(HIGH_RISK, 0, 0)))
    assert len(pool.calls) == 2, "否定+条件段必须二重审查"
    item = result.items[0]
    assert item.text == full
    assert item.quality_status == "accepted"


@pytest.mark.parametrize("risky", [
    "The file has 3 parts.",
    "Save before exiting.",
    "Click Start to begin.",
    "Only admins may delete files.",
    "不要在传输完成前断开电缆。",
])
def test_risky_markers_force_review(risky):
    service, _pool = risk_service()
    assert not service._is_low_risk_for_review(risky, "候选"), f"{risky!r} 应判高风险"


def test_plain_prose_without_markers_is_low_risk():
    service, _pool = risk_service()
    assert service._is_low_risk_for_review(LOW_RISK, "报告包含许多有趣的页面。")


def test_truncated_review_output_retains_all_originals_without_prefix_approval():
    # 两条都含否定词 → 同入审查批；生成响应按批量协议用分隔符拼接
    gen = f"第一句译文。{SEGMENT_DELIMITER}第二句译文。"
    batch = json.dumps([
        review_response("第一句译文。", 0),
        review_response("第二句译文。", 1),
    ])
    truncated = batch[: len(batch) - 30]  # 尾部对象被截断
    service, pool = risk_service(gen, truncated)
    result = service.translate_frame(
        make_frame(block("Do not close the plain window.", 0, 0),
                   block("Do not save before exiting.", 0, 40))
    )
    # 不完整末对象可能仍含重复 id 或冲突字段，前缀也不能当作有效证据。
    statuses = [item.quality_status for item in result.items]
    assert statuses == ["uncertain", "uncertain"]
    assert [item.text for item in result.items] == [
        "Do not close the plain window.", "Do not save before exiting.",
    ]
    assert len(service.cache) == 0


def test_always_policy_keeps_two_calls_for_plain_prose():
    full = "报告包含许多有趣的页面。"
    service, pool = risk_service(full, json.dumps([review_response(full)]), policy="always")
    result = service.translate_frame(make_frame(block(LOW_RISK, 0, 0)))
    assert len(pool.calls) == 2
    assert result.items[0].quality_status == "accepted"


def test_default_policy_reviews_plain_prose_and_actor_pronouns():
    assert AppConfig().translation.review_policy == "always"
    service, pool = risk_service("告诉我我知道的一切。", json.dumps([{
        "id": 0, "independent": "告诉我你知道的一切。", "checks": "011111",
    }]), policy=AppConfig().translation.review_policy)
    frame = service.translate_frame(make_frame(block("Tell me everything you know.", 0, 0)))
    assert len(pool.calls) == 2
    assert frame.items[0].quality_status == "uncertain"
    assert frame.items[0].text == "Tell me everything you know."
    assert service.cache.stats().verified_hits == 0
