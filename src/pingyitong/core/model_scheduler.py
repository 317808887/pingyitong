"""双模型动态调度（自研）。

对应《项目原创性声明》二·（二）"双模型池的调度策略"与开发文档三、
"双模型动态调度策略"的三级机制。

三级调度与优先级
    一级 **手动模式**：用户显式指定模型，系统完全遵从，不做任何自动切换。
    二级 **资源自适应**：CPU 持续高负载或渲染帧率过低时降级到 1.7B。
        这是"保护交互流畅度"的硬约束，**优先级高于内容自适应**——
        宁可译文质量略降，也不能让用户的电脑卡顿。
    三级 **内容自适应**：长句、术语密度高的文本升级到 4B 以提升质量。

防抖设计
    若不加约束，模型会在负载波动时反复"加载—卸载—再加载"，
    每次加载 4B 模型需要数秒，反而让体验更差。因此：

    - **切换冷却**：两次自动切换之间至少间隔 :data:`MIN_SWITCH_INTERVAL` 秒；
    - **持续判定**：负载是否过高需持续 :data:`CPU_HIGH_SUSTAIN_SECONDS` 秒才成立；
    - **不重复加载**：目标模型与当前已加载模型一致时直接跳过。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..constants import (
    CONTENT_LONG_TEXT_CHARS,
    CONTENT_TERM_DENSITY_THRESHOLD,
    CPU_HIGH_THRESHOLD,
    GPU_FREE_MEMORY_MIN_MB,
    LATENCY_HIGH_THRESHOLD,
    MODEL_ACCURATE,
    MODEL_FAST,
    MODEL_LABELS,
    SCHEDULE_MODE_AUTO,
)
from ..utils.logger import get_logger
from .resource_monitor import ResourceSnapshot

logger = get_logger(__name__)

#: 两次自动切换之间的最小间隔（秒），防止模型抖动。
#:
#: 取 30 秒而非更短：每次切换要卸载旧模型再加载新模型（含预热），
#: 实测 4B 约 3–10 秒。若冷却期短于切换成本本身，就会陷入
#: "刚切完又切回去"的抖动，用户看到的是一段段空窗期。
MIN_SWITCH_INTERVAL = 30.0

#: 值得为"内容自适应"切换模型的最短文本长度（字符）。
#:
#: 切换模型是有成本的：4B 的加载 + 预热约 3–10 秒，期间该帧翻译会明显延迟。
#: 因此只有当本次内容确实长到能体现模型差异时才值得切换。
#:
#: 实测问题：屏幕上一段 1 个字符的 OCR 碎片（如数字"1"）会让调度器判定
#: "短句文本（平均 1 字符）"而把已加载的 4B 换回 1.7B，白白付出一次
#: 模型卸载与加载。短到这种程度的文本用哪个模型都一样。
MIN_CONTENT_SWITCH_CHARS = 12


@dataclass
class ScheduleDecision:
    """一次调度决策及其理由。"""

    #: 选定的模型标识
    model_id: str
    #: 决策来源：``manual`` / ``resource`` / ``content`` / ``fallback``
    reason_code: str
    #: 面向用户的理由说明
    reason: str
    #: 是否发生了模型切换
    switched: bool = False

    def as_text(self) -> str:
        return f"{MODEL_LABELS.get(self.model_id, self.model_id)}（{self.reason}）"


class ModelScheduler:
    """双模型调度器。

    调用方在每次翻译请求前调用 :meth:`decide`，传入当前资源快照与
    待翻译文本，即可得到应使用的模型。
    """

    def __init__(self, config: object) -> None:
        self._config = config
        self._current_model = getattr(config, "active_model", MODEL_ACCURATE)
        self._last_switch_at = 0.0
        self._last_decision: ScheduleDecision | None = None
        self._switch_count = 0
        self._degrade_count = 0

    # ------------------------------------------------------------------
    # 配置
    # ------------------------------------------------------------------

    def apply_config(self, config: object) -> None:
        """应用新的调度配置。"""
        self._config = config
        if getattr(config, "mode", "") != SCHEDULE_MODE_AUTO:
            # 切回手动模式时，立即采纳用户选定的模型
            configured = getattr(config, "active_model", self._current_model)
            if configured != self._current_model:
                self._current_model = configured
                logger.info("已切换为手动模式，固定使用 %s", MODEL_LABELS.get(configured, configured))

    @property
    def current_model(self) -> str:
        """当前使用的模型。"""
        return self._current_model

    @property
    def last_decision(self) -> ScheduleDecision | None:
        """最近一次决策。"""
        return self._last_decision

    @property
    def switch_count(self) -> int:
        """累计自动切换次数，用于状态面板。"""
        return self._switch_count

    # ------------------------------------------------------------------
    # 决策
    # ------------------------------------------------------------------

    def decide(
        self,
        snapshot: ResourceSnapshot | None,
        texts: list[str] | None = None,
        glossary_term_count: int = 0,
        cpu_sustained_high: bool = False,
    ) -> ScheduleDecision:
        """决定本次请求使用的模型。

        :param snapshot: 当前资源快照；``None`` 时跳过资源自适应。
        :param texts: 本次待翻译文本，用于内容自适应。
        :param glossary_term_count: 本次命中的术语数量，参与术语密度计算。
        :param cpu_sustained_high: CPU 是否已**持续**高于阈值。
            由调用方基于 :class:`~pingyitong.core.resource_monitor.ResourceMonitor`
            的历史采样判定，避免瞬时峰值导致误降级。
        :return: :class:`ScheduleDecision`。
        """
        mode = getattr(self._config, "mode", "manual")

        # 一级：手动模式，完全遵从用户选择
        if mode != SCHEDULE_MODE_AUTO:
            manual_model = getattr(self._config, "active_model", self._current_model)
            decision = ScheduleDecision(
                model_id=manual_model,
                reason_code="manual",
                reason="手动模式",
                switched=manual_model != self._current_model,
            )
            self._current_model = manual_model
            self._last_decision = decision
            return decision

        # 二级：资源自适应（优先级高于内容自适应）
        resource_decision = self._decide_by_resource(snapshot, cpu_sustained_high)
        if resource_decision is not None:
            self._last_decision = resource_decision
            return resource_decision

        # 三级：内容自适应
        content_decision = self._decide_by_content(texts or [], glossary_term_count)
        self._last_decision = content_decision
        return content_decision

    # ------------------------------------------------------------------
    # 二级：资源自适应
    # ------------------------------------------------------------------

    def _decide_by_resource(
        self,
        snapshot: ResourceSnapshot | None,
        cpu_sustained_high: bool,
    ) -> ScheduleDecision | None:
        """依据资源负载决策。

        :return: 资源触发了调整时返回对应决策；**资源正常时返回 ``None``**，
            以便 :meth:`decide` 继续执行三级内容自适应。

            这里必须返回 ``None`` 而不是"沿用当前模型"的决策——否则
            只要调用方提供了资源快照（流水线总是会提供），内容自适应就
            永远不会被执行，三级调度会退化成二级。
        """
        if snapshot is None or not getattr(self._config, "resource_adaptive", True):
            return None

        # 端到端延迟过高：降级以保证响应速度。
        #
        # 这里刻意**不用渲染帧率**：静态屏幕上没有新内容可渲染，帧率天然
        # 趋近 0，用它做降级依据会把"屏幕没变"误判成"界面卡顿"，进而与
        # 内容自适应升级形成死循环（实测日志：4B→1.7B→4B→1.7B 反复切换，
        # 每次切换都要重载模型数秒）。延迟直接反映"用户在等多久"，是正确信号。
        latency_threshold = getattr(
            self._config, "latency_high_threshold", LATENCY_HIGH_THRESHOLD
        )
        if (
            snapshot.end_to_end_latency >= latency_threshold
            and self._current_model != MODEL_FAST
        ):
            if self._can_switch():
                return self._switch_to(
                    MODEL_FAST,
                    "resource",
                    f"端到端延迟 {snapshot.end_to_end_latency:.1f}s 超过 "
                    f"{latency_threshold:.1f}s，降级以保证响应速度",
                )
            return None

        # CPU 持续高负载：降级到 1.7B
        cpu_threshold = getattr(self._config, "cpu_high_threshold", CPU_HIGH_THRESHOLD)
        if cpu_sustained_high and self._current_model != MODEL_FAST and self._can_switch():
            return self._switch_to(
                MODEL_FAST,
                "resource",
                f"CPU 占用持续高于 {cpu_threshold:.0f}%，降级以降低负载",
            )

        # GPU 显存空闲充足且负载不高：升级到 4B
        gpu_min = getattr(self._config, "gpu_free_memory_min_mb", GPU_FREE_MEMORY_MIN_MB)
        if (
            snapshot.gpu_free_memory_mb is not None
            and snapshot.gpu_free_memory_mb >= gpu_min
            and not cpu_sustained_high
            and snapshot.cpu_percent < cpu_threshold
            and self._current_model != MODEL_ACCURATE
            and self._can_switch()
        ):
            return self._switch_to(
                MODEL_ACCURATE,
                "resource",
                f"GPU 显存空闲 {snapshot.gpu_free_memory_mb:.0f}MB、CPU 负载较低，"
                f"升级以提升翻译质量",
            )

        # 资源未触发任何调整，交由内容自适应决策
        return None

    # ------------------------------------------------------------------
    # 三级：内容自适应
    # ------------------------------------------------------------------

    def _decide_by_content(
        self,
        texts: list[str],
        glossary_term_count: int,
    ) -> ScheduleDecision:
        """依据文本特征决策。"""
        if not getattr(self._config, "content_adaptive", True) or not texts:
            return ScheduleDecision(
                model_id=self._current_model,
                reason_code="content",
                reason="默认模型",
                switched=False,
            )

        long_threshold = getattr(self._config, "long_text_chars", CONTENT_LONG_TEXT_CHARS)
        total_chars = sum(len(text) for text in texts)
        average_chars = total_chars / max(1, len(texts))
        # 以最长片段为准：只要有一段是长句，就值得用高精度模型
        longest = max(len(text) for text in texts)

        term_density = glossary_term_count / max(1, len(texts))

        # 内容过短时不值得切换：切换成本（卸载 + 加载 + 预热，数秒）
        # 远高于收益，而且短文本在两个模型上的表现没有可感差异。
        # 实测一个 1 字符的 OCR 碎片曾把已加载的 4B 换回 1.7B。
        if longest < MIN_CONTENT_SWITCH_CHARS:
            return ScheduleDecision(
                model_id=self._current_model,
                reason_code="content",
                reason=f"文本过短（最长 {longest} 字符），沿用当前模型",
                switched=False,
            )

        wants_accurate = False
        reason = ""

        if longest >= long_threshold:
            wants_accurate = True
            reason = f"检测到长句（{longest} 字符），使用高精度模型"
        elif term_density >= CONTENT_TERM_DENSITY_THRESHOLD:
            wants_accurate = True
            reason = f"专业术语密度较高（{term_density:.2f}），使用高精度模型"
        elif average_chars < long_threshold * 0.5:
            wants_accurate = False
            reason = f"短句文本（平均 {average_chars:.0f} 字符），使用极速模型"
        else:
            # 中等长度：保持当前模型，避免频繁切换
            return ScheduleDecision(
                model_id=self._current_model,
                reason_code="content",
                reason="文本长度适中，沿用当前模型",
                switched=False,
            )

        target = MODEL_ACCURATE if wants_accurate else MODEL_FAST
        if target == self._current_model:
            return ScheduleDecision(
                model_id=target,
                reason_code="content",
                reason=reason,
                switched=False,
            )

        if not self._can_switch():
            return ScheduleDecision(
                model_id=self._current_model,
                reason_code="content",
                reason="建议切换但处于冷却期，暂用当前模型",
                switched=False,
            )

        return self._switch_to(target, "content", reason)

    # ------------------------------------------------------------------
    # 切换
    # ------------------------------------------------------------------

    def _can_switch(self) -> bool:
        """是否已过冷却期。"""
        if self._last_switch_at == 0.0:
            return True
        return (time.perf_counter() - self._last_switch_at) >= MIN_SWITCH_INTERVAL

    def _switch_to(self, model_id: str, reason_code: str, reason: str) -> ScheduleDecision:
        """执行模型切换并记录。"""
        previous = self._current_model
        self._current_model = model_id
        self._last_switch_at = time.perf_counter()
        self._switch_count += 1
        if model_id == MODEL_FAST and previous == MODEL_ACCURATE:
            self._degrade_count += 1

        logger.info(
            "自动调度切换模型：%s → %s（%s）",
            MODEL_LABELS.get(previous, previous),
            MODEL_LABELS.get(model_id, model_id),
            reason,
        )
        return ScheduleDecision(
            model_id=model_id,
            reason_code=reason_code,
            reason=reason,
            switched=True,
        )

    def force_model(self, model_id: str) -> None:
        """强制指定当前模型（用户通过热键或悬浮条切换）。"""
        if model_id == self._current_model:
            return
        logger.info(
            "用户手动切换模型：%s → %s",
            MODEL_LABELS.get(self._current_model, self._current_model),
            MODEL_LABELS.get(model_id, model_id),
        )
        self._current_model = model_id
        # 手动切换后同样进入冷却期，避免自动调度立即把它切回去
        self._last_switch_at = time.perf_counter()

    def toggle_model(self) -> str:
        """在 1.7B 与 4B 之间切换，返回切换后的模型。"""
        target = MODEL_FAST if self._current_model == MODEL_ACCURATE else MODEL_ACCURATE
        self.force_model(target)
        return target

    def stats_text(self) -> str:
        """返回调度统计的中文描述。"""
        if getattr(self._config, "mode", "manual") != SCHEDULE_MODE_AUTO:
            return "手动模式"
        return f"自动调度 · 已切换 {self._switch_count} 次"
