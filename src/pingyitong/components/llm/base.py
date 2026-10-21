"""翻译引擎抽象与降级实现。

定义统一的推理接口，使核心调度层与具体推理后端解耦：
上层的双模型池、缓存、调度策略都只依赖 :class:`TranslationEngine` 协议。

同时提供 :class:`LayoutPreviewEngine`：在模型权重尚未就绪时，
它按原文长度生成等长的占位标记，使原位渲染链路仍可被完整验证
（用于核对坐标对齐、字号适配与多屏适配），**不会伪造任何翻译内容**。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ...utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EngineStatus:
    """引擎状态快照，供界面展示。"""

    #: 模型标识（见 constants.MODEL_*）
    model_id: str
    #: 是否已加载到内存
    loaded: bool = False
    #: 是否可用（依赖与权重均就绪）
    available: bool = False
    #: 状态说明，直接展示给用户
    message: str = ""
    #: 权重文件路径
    path: str = ""
    #: 最近一次推理耗时（秒）
    last_latency: float = 0.0
    #: 累计推理次数
    total_calls: int = 0

    def as_text(self) -> str:
        """生成简洁的中文状态描述。"""
        if not self.available:
            return self.message or "不可用"
        if not self.loaded:
            return "待加载"
        if self.last_latency:
            return f"已加载 · 上次 {self.last_latency * 1000:.0f}ms"
        return "已加载"


@runtime_checkable
class TranslationEngine(Protocol):
    """翻译引擎协议。

    实现方需保证 :meth:`translate_batch` 返回的列表长度与入参一致，
    且顺序严格对应。
    """

    model_id: str

    def is_available(self) -> bool:
        """依赖与权重是否就绪。"""

    def is_loaded(self) -> bool:
        """模型是否已在内存中。"""

    def load(self) -> bool:
        """加载模型，成功返回 ``True``。"""

    def unload(self) -> None:
        """卸载模型，释放内存。"""

    def translate_batch(
        self,
        segments: list[str],
        system_prompt: str,
        max_tokens: int | None = None,
    ) -> list[str]:
        """批量翻译，返回与 ``segments`` 等长且同序的结果。"""

    def status(self) -> EngineStatus:
        """返回状态快照。"""


class LayoutPreviewEngine:
    """占位引擎：用于在模型未就绪时验证原位渲染链路。

    输出为固定前缀加原文，长度与原文同量级，因此可以真实反映
    "译文撑不撑得下原文的包围盒"、"多行如何换行"等排版问题。
    界面会明确标注当前处于预览状态，不将其呈现为真实翻译。
    """

    #: 展示给用户的前缀，明确标识这不是翻译结果
    MARKER = "〔未加载模型〕"

    def __init__(self, model_id: str, reason: str = "", path: str = "") -> None:
        self.model_id = model_id
        self._reason = reason or "未找到模型权重文件"
        #: 权重文件的**期望路径**。即使文件不存在也要记录，
        #: 这样界面提示能告诉用户"应该放到哪里"，而不是只说"找不到"。
        self._path = path
        self._calls = 0
        self._last_latency = 0.0

    def is_available(self) -> bool:
        return False

    def is_loaded(self) -> bool:
        return False

    def load(self) -> bool:
        return False

    def unload(self) -> None:
        return None

    def translate_batch(
        self,
        segments: list[str],
        system_prompt: str,
        max_tokens: int | None = None,
    ) -> list[str]:
        started = time.perf_counter()
        self._calls += 1
        result = [f"{self.MARKER}{segment}" for segment in segments]
        self._last_latency = time.perf_counter() - started
        return result

    def status(self) -> EngineStatus:
        return EngineStatus(
            model_id=self.model_id,
            loaded=False,
            available=False,
            message=f"预览模式：{self._reason}",
            path=self._path,
            last_latency=self._last_latency,
            total_calls=self._calls,
        )


@dataclass
class EngineMetrics:
    """引擎运行指标，供资源面板展示。"""

    calls: int = 0
    failures: int = 0
    total_latency: float = 0.0
    last_latency: float = 0.0
    #: 最近若干次请求的平均耗时
    recent_latencies: list[float] = field(default_factory=list)

    def record(self, latency: float, success: bool = True) -> None:
        """记录一次推理。"""
        self.calls += 1
        if not success:
            self.failures += 1
        self.last_latency = latency
        self.total_latency += latency
        self.recent_latencies.append(latency)
        if len(self.recent_latencies) > 32:
            self.recent_latencies.pop(0)

    @property
    def average_latency(self) -> float:
        if not self.calls:
            return 0.0
        return self.total_latency / self.calls

    @property
    def recent_average(self) -> float:
        """近期平均耗时，比全量平均更能反映当前性能。"""
        if not self.recent_latencies:
            return 0.0
        return sum(self.recent_latencies) / len(self.recent_latencies)
