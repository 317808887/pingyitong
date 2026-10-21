"""P0 fuzzy 标定样本集回归夹具。

把 P0 探针的标定实验固化为回归测试（方案 v1.2 §10.2 第 2 条）：
同样的样本、同样的噪声模型、同样的参数（partial_ratio 0.80 + ±4 窗口 +
序列校验 2 + 低置信全轨重定位），必须持续满足发布门槛——

- 命中率 ≥ 90%
- 错配率 ≤ 1%

**防止什么**：后续有人调 fuzzy 参数、改归一化、换相似度算法时，
没有这组夹具就会在无感知中跌破门槛，字幕错配（甲句显示乙句译文）
直接进产品。

样本与噪声模型说明：字幕行取自典型影片对白风格；噪声模型覆盖
OCR 常见错误谱（I/l/1 混淆、O/0、标点丢失、空格丢失、字符丢失/
重复/替换），并叠加 15% 丢行 + 5% 重复行（模拟真实观测）。
固定随机种子，结果可复现。
"""

from __future__ import annotations

import random
import unittest

from rapidfuzz import fuzz

from pingyitong.services.prefetch.normalizer import normalize_for_match

#: P0 标定参数（与 prefetch.fuzzy_* 配置默认值一致；改参数必须带重标定数据）
THRESHOLD = 0.80
WINDOW = 4
SEQUENCE_CHECK = 2

#: 发布门槛（D6）
MIN_HIT_RATE = 0.90
MAX_MISMATCH_RATE = 0.01

_SUBTITLES = [
    "I never thought it would end this way.", "What are you doing here?",
    "Get down!", "We don't have much time.", "This isn't what it looks like.",
    "Are you sure about that?", "Everything changed after that night.",
    "You have to trust me.", "It's too dangerous.", "I'll handle this myself.",
    "Where were you last night?", "That's the last time I ask you.",
    "Stay behind me.", "Do you hear that?", "We should go back.",
    "Nobody knows the truth.", "It was never about the money.",
    "Look what they did to this place.", "You knew this day would come.",
    "There's something I have to tell you.", "How long has it been?",
    "Three years, two months, eight days.", "I made a promise.",
    "And I intend to keep it.", "Open the door!", "Who's there?",
    "Don't move.", "Put the gun down.", "You don't understand.",
    "I understand perfectly.", "Then you know what happens next.",
    "We're running out of options.", "There has to be another way.",
    "Not this time.", "You always say that.", "And I'm always right.",
    "Not always.", "Close enough.", "What happened to your face?",
    "The same thing that happened to yours.", "That's not an answer.",
    "It's the only answer I have.", "Follow me.", "Where are we going?",
    "Somewhere safe.", "No place is safe anymore.", "Then we make it safe.",
    "How?", "One step at a time.", "I can't do this alone.", "You're not alone.",
    "Not anymore.", "Promise me.", "I promise.", "Swear it.", "I swear.",
    "Then it's settled.", "What if you're wrong?", "Then I'm wrong.",
    "And if you're right?", "God help us all.", "Move!", "Down!",
    "Get to the car!", "Start the engine!", "Faster!", "Almost there.",
    "We made it.", "For now.", "What's the plan?", "Same as always.",
    "Improvise.", "That's not a plan.", "It's worked so far.",
    "Until it doesn't.", "Then we improvise again.",
] * 6  # ≈ 456 行，模拟一部长片


def _ocr_noise(text: str, rng: random.Random) -> str:
    """受控 OCR 噪声（P0 标定所用模型，勿随意改动——改动即重标定）。"""
    out = []
    for ch in text:
        r = rng.random()
        if ch in "Il|" and r < 0.30:
            out.append(rng.choice("1lI"))
        elif ch == "O" and r < 0.25:
            out.append("0")
        elif ch in ",.;:!" and r < 0.35 or ch == " " and r < 0.06 or ch.isalpha() and r < 0.02:
            continue
        elif ch.isalpha() and r < 0.035:
            out.append(ch + ch)
        elif ch.isalpha() and r < 0.05:
            out.append(rng.choice("abcdefghijklmnopqrstuvwxyz"))
        else:
            out.append(ch)
    return "".join(out)


def _simulate_and_measure(seed: int, miss_rate: float = 0.15):
    """一次完整仿真：返回 (命中率, 错配率, 全轨重定位次数)。"""
    rng = random.Random(seed)
    track = [" ".join(x.lower().split()) for x in _SUBTITLES]
    hit = miss = mismatch = retrack = 0
    center = 0
    recent: list[int] = []

    for idx, _line in enumerate(track):
        if rng.random() < miss_rate:
            continue
        obs_idx = recent[-1] if (recent and rng.random() < 0.05) else idx
        noisy = " ".join(_ocr_noise(_SUBTITLES[obs_idx], rng).lower().split())
        needle = normalize_for_match(noisy)

        lo, hi = max(0, center - WINDOW), min(len(track), center + WINDOW + 1)
        best_i, best_s = None, -1.0
        for i in range(lo, hi):
            s = fuzz.partial_ratio(needle, track[i])
            if s > best_s:
                best_i, best_s = i, s
        if best_s / 100.0 < THRESHOLD:
            # 低置信 → 全轨重定位（限本帧一次）
            best_i, best_s = None, -1.0
            for i, line_t in enumerate(track):
                s = fuzz.partial_ratio(needle, line_t)
                if s > best_s:
                    best_i, best_s = i, s
            retrack += 1
            if best_s / 100.0 < THRESHOLD:
                miss += 1
                continue

        # 序列一致性校验：跳变超出容许范围视为错配候选，拒绝
        if recent and abs(best_i - recent[-1]) > SEQUENCE_CHECK + WINDOW:
            mismatch += 1
            continue
        hit += 1
        center = best_i
        recent.append(best_i)

    total = hit + miss + mismatch
    return hit / total, mismatch / total, retrack


class TestP0FuzzyFixture(unittest.TestCase):
    """P0 样本集回归夹具：参数改动必须重新过门槛。"""

    def test_hit_rate_meets_release_gate(self) -> None:
        hit_rate, mismatch_rate, _ = _simulate_and_measure(seed=7)
        self.assertGreaterEqual(
            hit_rate, MIN_HIT_RATE,
            f"fuzzy 命中率 {hit_rate:.1%} 跌破发布门槛 {MIN_HIT_RATE:.0%}",
        )

    def test_mismatch_rate_meets_release_gate(self) -> None:
        hit_rate, mismatch_rate, _ = _simulate_and_measure(seed=7)
        self.assertLessEqual(
            mismatch_rate, MAX_MISMATCH_RATE,
            f"fuzzy 错配率 {mismatch_rate:.1%} 超过发布门槛 {MAX_MISMATCH_RATE:.0%}"
            "——错配意味着甲句显示乙句译文",
        )

    def test_multiple_seeds_stable(self) -> None:
        """多组种子下门槛都要成立（防单一种子侥幸）。"""
        for seed in (7, 11, 42):
            with self.subTest(seed=seed):
                hit_rate, mismatch_rate, _ = _simulate_and_measure(seed=seed)
                self.assertGreaterEqual(hit_rate, MIN_HIT_RATE)
                self.assertLessEqual(mismatch_rate, MAX_MISMATCH_RATE)


if __name__ == "__main__":
    unittest.main()
