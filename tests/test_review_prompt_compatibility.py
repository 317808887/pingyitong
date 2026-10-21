"""r0006 frozen review prompt compatibility, with v5 publication gates retained."""

from __future__ import annotations

import hashlib
import json

import pytest

from pingyitong.components.llm import prompt as prompt_module
from pingyitong.services.semantic_review import parse_reviews
from tests.test_semantic_review import SOURCE, reviewed_service


# Output SHA256s generated from the exact frozen r0006 function. Keeping the
# hashes here avoids depending on an iteration directory that retention prunes.
@pytest.mark.parametrize("target,compact,expected", [
    ("zh", True, "a72b5810091a94beff949c62a546bb61b2e913aacebe6a33156f53abd85cb15c"),
    ("zh", False, "6eb8c046827c017c89e84c0094b4907123883f60b99c91838a81c8d03df6e752"),
    ("fr", True, "f625aecc25dbca9bffd2ca5e76b22f03e3aa7c239e44e837b7cec3ca35ccddcf"),
    ("fr", False, "0d0b5598318ac76e5cdb27f71f13f9b7eb6ab4eff8b38763a808003c723b8950"),
])
def test_review_prompt_matches_frozen_v4_output(target, compact, expected):
    actual = prompt_module.build_semantic_review_prompt(target, compact=compact)
    assert hashlib.sha256(actual.encode("utf-8")).hexdigest() == expected


def test_restored_self_rating_is_only_a_compatibility_field_not_compact_evidence():
    raw = json.dumps([{
        "id": 0, "independent": "完整译文。", "checks": "111111",
        "compact": "短译。", "compact_checks": "111111",
    }])
    decision = parse_reviews(raw, 1)[0]
    assert decision.accepted and decision.compact == "短译。"
    assert decision.compact_checks == ()


def test_prompt_policy_eight_isolates_cached_previous_prompt_behavior(monkeypatch):
    assert prompt_module.PROMPT_POLICY_VERSION == 8
    service, _pool = reviewed_service()
    current = service._cache_key(SOURCE, "en", "zh")
    monkeypatch.setattr(prompt_module, "PROMPT_POLICY_VERSION", 7)
    assert service._cache_key(SOURCE, "en", "zh") != current
