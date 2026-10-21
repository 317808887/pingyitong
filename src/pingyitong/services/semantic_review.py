"""Strict, local translation review protocol and reusable evidence.

The second pass must independently translate the source and compare six
meaning dimensions. A bare approval, repaired translation, or partial JSON is
never a publishable result. This protocol is a quality gate, not a proof that
a small model cannot make the same mistake twice.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

REVIEW_POLICY_VERSION = 5
CACHE_FORMAT = "reviewed-translation-v5"
REVIEW_DIMENSIONS = ("actors", "actions", "negation", "conditions", "quantities", "completeness")


class _DuplicateFieldError(ValueError):
    """A JSON object cannot carry conflicting facts under the same key."""


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateFieldError(key)
        value[key] = item
    return value


def translations_agree(candidate: str, independent: str) -> bool:
    """Only NFC and whitespace folding establish deterministic agreement.

    Different wording can be a valid paraphrase, but a self-reported approval
    cannot resolve it reliably. Preserve the source rather than pick a winner.
    Agreement is an additional guard, not proof against shared model errors.
    """
    from unicodedata import normalize

    first = " ".join(normalize("NFC", candidate).split())
    return bool(first) and first == " ".join(normalize("NFC", independent).split())


def source_digest(source: str) -> str:
    from unicodedata import normalize

    return hashlib.sha256(normalize("NFC", source).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ReviewInput:
    source: str
    candidate: str


@dataclass(frozen=True)
class ReviewDecision:
    accepted: bool = False
    independent: str = ""
    checks: tuple[str, ...] = ()
    compact: str = ""
    #: Pass-two compact is an unreviewed candidate; its self-rating is ignored.
    compact_checks: tuple[str, ...] = ()
    reason: str = "review_invalid"


@dataclass(frozen=True)
class ResolutionInput:
    source: str
    candidate: str
    independent: str


@dataclass(frozen=True)
class ResolutionDecision:
    choice: str = "uncertain"
    checks: tuple[str, ...] = ()
    reason: str = "review_resolution_invalid"

    @property
    def accepted(self) -> bool:
        return self.choice in ("A", "B") and self.checks == ("ok",) * len(REVIEW_DIMENSIONS)


def resolution_digest(
    source_hash: str, candidate: str, independent: str, choice: str,
    checks: tuple[str, ...], model_id: str,
) -> str:
    """Bind a decision to exact A/B texts; this is consistency, not authentication."""
    value = (REVIEW_POLICY_VERSION, source_hash, candidate, independent, choice, checks, model_id)
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def compact_review_digest(
    source_hash: str, full: str, compact: str, checks: tuple[str, ...], model_id: str,
) -> str:
    """A distinct request must audit this exact compact against source/full."""
    value = (REVIEW_POLICY_VERSION, "semantic_compact_review", source_hash, full, compact, "B", checks, model_id)
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ReviewedTranslation:
    """Only this evidence-bearing object may enter the reviewed cache."""

    source_hash: str
    full_text: str
    display_text: str
    independent: str
    generation_model: str
    review_model: str
    checks: tuple[str, ...] = field(default_factory=lambda: ("ok",) * len(REVIEW_DIMENSIONS))
    compact_checks: tuple[str, ...] = ()
    candidate: str = ""
    selection: str = "agreement"
    resolution_checks: tuple[str, ...] = ()
    resolution_model: str = ""
    resolution_binding: str = ""
    compact_review_choice: str = ""
    compact_review_model: str = ""
    compact_review_binding: str = ""

    def __post_init__(self) -> None:
        if not self.candidate:
            object.__setattr__(self, "candidate", self.full_text)

    def evidence_consistent(self) -> bool:
        """Check provenance and selected-text binding, not semantic truth."""
        all_ok = ("ok",) * len(REVIEW_DIMENSIONS)
        if self.checks != all_ok or not all((
            self.candidate.strip(), self.independent.strip(), self.full_text.strip(),
            self.generation_model.strip(), self.review_model.strip(),
        )):
            return False
        if self.display_text:
            if (
                self.selection == "B" or len(self.display_text) >= len(self.full_text)
                or self.compact_checks != all_ok or self.compact_review_choice != "B"
                or self.compact_review_model != self.review_model
                or self.compact_review_binding != compact_review_digest(
                    self.source_hash, self.full_text, self.display_text, self.compact_checks, self.compact_review_model,
                )
            ):
                return False
        elif self.compact_checks or self.compact_review_choice or self.compact_review_model or self.compact_review_binding:
            return False
        if self.selection == "agreement":
            return (
                self.full_text == self.candidate and translations_agree(self.candidate, self.independent)
                and not self.resolution_checks and not self.resolution_model and not self.resolution_binding
            )
        if self.selection not in ("A", "B") or self.resolution_checks != all_ok:
            return False
        if translations_agree(self.candidate, self.independent) or self.resolution_model != self.review_model:
            return False
        selected = self.candidate if self.selection == "A" else self.independent
        if self.full_text != selected or (self.selection == "B" and (self.display_text or self.compact_checks)):
            return False
        return self.resolution_binding == resolution_digest(
            self.source_hash, self.candidate, self.independent, self.selection,
            self.resolution_checks, self.resolution_model,
        )

    def to_json(self) -> str:
        return json.dumps({
            "format": CACHE_FORMAT,
            "policy": REVIEW_POLICY_VERSION,
            "source_hash": self.source_hash,
            "full_text": self.full_text,
            "display_text": self.display_text,
            "independent": self.independent,
            "generation_model": self.generation_model,
            "review_model": self.review_model,
            "checks": self.checks,
            "compact_checks": self.compact_checks,
            "candidate": self.candidate,
            "selection": self.selection,
            "resolution_checks": self.resolution_checks,
            "resolution_model": self.resolution_model,
            "resolution_binding": self.resolution_binding,
            "compact_review_choice": self.compact_review_choice,
            "compact_review_model": self.compact_review_model,
            "compact_review_binding": self.compact_review_binding,
        }, ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: str, source: str) -> ReviewedTranslation | None:
        try:
            value = json.loads(raw, object_pairs_hook=_unique_json_object)
        except (ValueError, TypeError):
            return None
        if not isinstance(value, dict) or value.get("format") != CACHE_FORMAT:
            return None
        if value.get("policy") != REVIEW_POLICY_VERSION or value.get("source_hash") != source_digest(source):
            return None
        fields = (
            "full_text", "display_text", "independent", "generation_model", "review_model",
            "candidate", "selection", "resolution_model", "resolution_binding",
            "compact_review_choice", "compact_review_model", "compact_review_binding",
        )
        if any(not isinstance(value.get(name), str) for name in fields):
            return None
        if any(not value[name].strip() for name in ("full_text", "candidate", "independent", "generation_model", "review_model")):
            return None
        checks = _checks(value.get("checks"))
        compact_checks = _checks(value.get("compact_checks"))
        if checks != ("ok",) * len(REVIEW_DIMENSIONS):
            return None
        if value["display_text"] and compact_checks != ("ok",) * len(REVIEW_DIMENSIONS):
            return None
        evidence = cls(
            source_hash=value["source_hash"], checks=checks, compact_checks=compact_checks,
            resolution_checks=_checks(value.get("resolution_checks")),
            **{name: value[name] for name in fields},
        )
        return evidence if evidence.evidence_consistent() else None


#: 小模型会把裁决写成中文同义词（实测 1.7B：无法确定/正确/错误）。
#: 归一化只是让"证据可解析"，不改 fail-closed 语义：只有明确的肯定裁决算 ok。
_VERDICT_SYNONYMS = {
    "ok": "ok",
    "pass": "ok",
    "正确": "ok",
    "一致": "ok",
    "无误": "ok",
    "wrong": "wrong",
    "错误": "wrong",
    "有误": "wrong",
    "不对": "wrong",
    "不一致": "wrong",
    "uncertain": "uncertain",
    "无法确定": "uncertain",
    "不确定": "uncertain",
    "无法判断": "uncertain",
}


def _normalize_verdict(item: object) -> str | None:
    if not isinstance(item, str):
        return None
    verdict = _VERDICT_SYNONYMS.get(item.strip())
    return verdict


def _checks(value: object) -> tuple[str, ...]:
    """接受六元列表或按维度命名的字典（1.7B 实际形态），其余一律拒绝。

    列表必须按 REVIEW_DIMENSIONS 顺序给全六个值；字典按维度名抽取，
    缺维度或多余非维度键都视为证据不完整。
    """
    if isinstance(value, str):
        if len(value) != len(REVIEW_DIMENSIONS) or any(ch not in "10?" for ch in value):
            return ()
        return tuple({"1": "ok", "0": "wrong", "?": "uncertain"}[ch] for ch in value)
    if isinstance(value, list):
        if len(value) != len(REVIEW_DIMENSIONS):
            return ()
        normalized = [_normalize_verdict(item) for item in value]
    elif isinstance(value, dict):
        if set(value) != set(REVIEW_DIMENSIONS):
            return ()
        normalized = [_normalize_verdict(value[name]) for name in REVIEW_DIMENSIONS]
    else:
        return ()
    if any(verdict is None for verdict in normalized):
        return ()
    return tuple(normalized)  # type: ignore[arg-type]


def build_review_prompt(inputs: list[ReviewInput]) -> str:
    """JSON-escape screen content so it cannot alter the outer protocol."""
    return json.dumps([
        {"id": index, "source": item.source, "candidate": item.candidate}
        for index, item in enumerate(inputs)
    ], ensure_ascii=False, separators=(",", ":"))


def build_resolution_prompt(inputs: list[ResolutionInput]) -> str:
    return json.dumps([
        {"id": index, "source": item.source, "A": item.candidate, "B": item.independent}
        for index, item in enumerate(inputs)
    ], ensure_ascii=False, separators=(",", ":"))


def parse_resolutions(raw: str, count: int) -> list[ResolutionDecision]:
    """A bounded third review needs complete, uniquely mapped explicit verdicts.

    No truncated-object salvage or extra repair is attempted. A/B are only
    choices, never replacements supplied in this reply.
    """
    rejected = [ResolutionDecision()] * count
    try:
        values = json.loads(raw, object_pairs_hook=_unique_json_object)
    except (ValueError, TypeError):
        return rejected
    if not isinstance(values, list) or len(values) != count:
        return rejected
    by_id: dict[int, dict] = {}
    for value in values:
        if not isinstance(value, dict) or set(value) != {"id", "choice", "checks"}:
            return rejected
        index = value["id"]
        if type(index) is not int or index < 0 or index >= count or index in by_id:
            return rejected
        by_id[index] = value
    decisions = []
    for index in range(count):
        value = by_id[index]
        choice, checks = value["choice"], _checks(value["checks"])
        if choice not in ("A", "B", "neither", "uncertain") or not checks:
            decisions.append(ResolutionDecision())
        elif choice in ("neither", "uncertain"):
            decisions.append(ResolutionDecision(choice, checks, "review_resolution_" + choice))
        elif checks != ("ok",) * len(REVIEW_DIMENSIONS):
            dimension = next(name for name, check in zip(REVIEW_DIMENSIONS, checks, strict=True) if check != "ok")
            decisions.append(ResolutionDecision(choice, checks, "semantic_resolution_" + dimension))
        else:
            decisions.append(ResolutionDecision(choice, checks, ""))
    return decisions


def parse_reviews(raw: str, count: int) -> list[ReviewDecision]:
    """Fail closed on missing/duplicate IDs, missing evidence and bare yes/no.

    A truncated tail cannot prove the absence of a conflicting ID or field.
    Require complete JSON; never publish a parsed prefix from a broken batch.
    """
    rejected = [ReviewDecision()] * count
    text = raw.strip()
    if text.startswith("```json\n") and text.endswith("\n```"):
        text = text[8:-4].strip()
    try:
        values: object = json.loads(text, object_pairs_hook=_unique_json_object)
    except (ValueError, TypeError):
        return rejected
    if not isinstance(values, list) or len(values) != count:
        return rejected
    by_id: dict[int, dict] = {}
    for value in values:
        if not isinstance(value, dict):
            return rejected
        index = value.get("id")
        if type(index) is not int or index < 0 or index >= count or index in by_id:
            return rejected
        by_id[index] = value
    decisions = []
    for index in range(count):
        value = by_id.get(index)
        if value is None:
            decisions.append(ReviewDecision())
            continue
        independent = value.get("independent")
        checks = _checks(value.get("checks"))
        if not isinstance(independent, str) or not independent.strip() or not checks:
            decisions.append(ReviewDecision())
            continue
        accepted = all(check == "ok" for check in checks)
        reason = "" if accepted else "semantic_" + next(
            dimension for dimension, check in zip(REVIEW_DIMENSIONS, checks, strict=True) if check != "ok"
        )
        compact = value.get("compact", "")
        if not isinstance(compact, str):
            compact = ""
        decisions.append(ReviewDecision(
            accepted=accepted, independent=independent.strip(), checks=checks,
            compact=compact.strip(), reason=reason,
        ))
    return decisions
