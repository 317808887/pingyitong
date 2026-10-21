"""本地大模型推理模块（llama.cpp 封装）。

开源组件：
    - ``llama.cpp``（ggerganov/llama.cpp，MIT License）GGUF 量化与轻量化推理
    - ``llama-cpp-python`` 作为 Python 绑定
    - 模型权重：Spark-X2.5-1.7B / 4B（iflytek/spark-llm，Apache-2.0）

自研内容：双模型加载与切换封装、批量文本翻译、Prompt 模板适配、
    CPU 指令集与 GPU 卸载参数调优。

隐私约束
    推理全程在本机完成，不发起任何网络请求。模型加载后即使断网，
    翻译功能依然完整可用（见原创性声明二·（四））。
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from ...utils.logger import get_logger
from .base import EngineMetrics, EngineStatus

logger = get_logger(__name__)


class LlamaEngine:
    """基于 llama.cpp 的单模型推理引擎。

    一个实例对应一个 GGUF 权重文件。线程安全：内部使用互斥锁串行化推理，
    因为 llama.cpp 的上下文对象不支持并发调用。
    """

    def __init__(
        self,
        model_id: str,
        model_path: Path,
        n_ctx: int = 2048,
        n_threads: int = 0,
        n_gpu_layers: int = 0,
        n_batch: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: int = 512,
        verbose: bool = False,
    ) -> None:
        self.model_id = model_id
        self.model_path = Path(model_path)
        self.n_ctx = max(256, int(n_ctx))
        self.n_threads = max(0, int(n_threads))
        self.n_gpu_layers = max(0, int(n_gpu_layers))
        self.n_batch = max(32, int(n_batch))
        self.temperature = max(0.0, min(2.0, float(temperature)))
        self.top_p = max(0.0, min(1.0, float(top_p)))
        self.max_tokens = max(32, int(max_tokens))
        self.verbose = verbose

        self._model: Any = None
        self._lock = threading.Lock()
        self._last_used = 0.0
        #: 模型就绪的时刻。用作"从未推理过"时计算空闲时长的基准——
        #: 刚加载完的模型不是空闲的，不能被空闲回收逻辑立刻卸载。
        self._loaded_at = 0.0
        self._unavailable_reason = ""
        self.metrics = EngineMetrics()

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Python 绑定与权重文件是否均就绪。"""
        if not _llama_cpp_available():
            self._unavailable_reason = (
                "未安装 llama-cpp-python，请执行 pip install llama-cpp-python"
            )
            return False

        if not self.model_path.exists():
            self._unavailable_reason = (
                f"未找到模型权重：{self.model_path.name}（请放置于 {self.model_path.parent}）"
            )
            return False

        self._unavailable_reason = ""
        return True

    def is_loaded(self) -> bool:
        """模型是否已常驻内存。"""
        return self._model is not None

    @property
    def unavailable_reason(self) -> str:
        """不可用原因，供界面提示。"""
        return self._unavailable_reason

    @property
    def idle_seconds(self) -> float:
        """距**上次使用或加载**的闲置时长（秒）。用于 4B 模型的空闲卸载。

        基准必须回落到加载时刻：模型刚加载完、还没推理过时它不算空闲。
        早期实现在未使用过时返回 ``inf``，导致维护定时器一发现模型加载完成
        就立刻把它卸载（甚至打断正在进行的首次推理）。
        """
        baseline = self._last_used or self._loaded_at
        if not baseline:
            return 0.0
        return time.perf_counter() - baseline

    def status(self) -> EngineStatus:
        """返回状态快照。"""
        return EngineStatus(
            model_id=self.model_id,
            loaded=self.is_loaded(),
            available=self.is_available(),
            message=self._unavailable_reason,
            path=str(self.model_path),
            last_latency=self.metrics.last_latency,
            total_calls=self.metrics.calls,
        )

    def estimate_memory_mb(self) -> float:
        """粗略估算加载所需内存（MB）。

        以文件体积的 1.15 倍估算：GGUF 权重加载后还需 KV 缓存与
        推理中间张量，经验上约为文件体积的 10%~20%。
        """
        try:
            size = self.model_path.stat().st_size
        except OSError:
            return 0.0
        return size / (1024 * 1024) * 1.15

    # ------------------------------------------------------------------
    # 加载 / 卸载
    # ------------------------------------------------------------------

    def load(self) -> bool:
        """加载模型到内存。已加载时直接返回成功。"""
        with self._lock:
            if self._model is not None:
                return True

            if not self.is_available():
                return False

            started = time.perf_counter()
            try:
                from llama_cpp import Llama

                self._model = Llama(
                    model_path=str(self.model_path),
                    n_ctx=self.n_ctx,
                    n_threads=self.n_threads or None,
                    n_gpu_layers=self.n_gpu_layers,
                    n_batch=self.n_batch,
                    verbose=self.verbose,
                    seed=-1,
                )
            except Exception:  # noqa: BLE001 - 加载失败原因多样（内存不足/文件损坏）
                self._unavailable_reason = "模型加载失败，详见日志"
                logger.exception("模型加载失败：%s", self.model_path.name)
                self._model = None
                return False

            elapsed = time.perf_counter() - started
            # 记录就绪时刻：空闲时长的基准。刚加载完的模型不算空闲。
            self._loaded_at = time.perf_counter()
            self._last_used = 0.0
            logger.info(
                "模型已加载：%s（耗时 %.2fs，估算占用 %.0fMB）",
                self.model_path.name,
                elapsed,
                self.estimate_memory_mb(),
            )
            return True

    def unload(self) -> None:
        """卸载模型并释放内存。"""
        with self._lock:
            if self._model is None:
                return
            model, self._model = self._model, None

        # 在锁外释放，避免析构耗时时阻塞其他线程的状态查询
        try:
            del model
        except Exception:  # noqa: BLE001
            logger.debug("释放模型对象时出现异常（可忽略）")
        logger.info("模型已卸载：%s", self.model_path.name)

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------

    def translate_batch(
        self,
        segments: list[str],
        system_prompt: str,
        max_tokens: int | None = None,
    ) -> list[str]:
        """执行一次批量翻译请求。

        :param segments: 已由 Prompt 模块拼接好的**单个**用户提示词
            （见 :func:`pingyitong.components.llm.prompt.build_user_prompt`）。
        :param system_prompt: system 提示词。
        :param max_tokens: 覆盖默认的生成上限。
        :return: 模型原始输出的单元素列表；失败时返回空列表。
        """
        if not segments:
            return []

        prompt = segments[0]
        started = time.perf_counter()

        if not self.load():
            self.metrics.record(time.perf_counter() - started, success=False)
            return []

        try:
            with self._lock:
                output = self._generate(prompt, system_prompt, max_tokens)
        except Exception:  # noqa: BLE001 - 单次推理失败不应终止整个流水线
            logger.exception("推理失败：%s", self.model_path.name)
            self.metrics.record(time.perf_counter() - started, success=False)
            return []

        latency = time.perf_counter() - started
        self.metrics.record(latency, success=bool(output))
        self._last_used = time.perf_counter()
        return [output] if output else []

    def _generate(self, prompt: str, system_prompt: str, max_tokens: int | None) -> str:
        """调用 llama.cpp 生成文本。

        优先使用对话接口（Spark-X2.5 为指令微调模型）；若该模型的
        GGUF 元数据缺少对话模板，则回退到补全接口并手工拼接提示词。
        """
        limit = max_tokens or self.max_tokens

        try:
            response = self._model.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=limit,
                stream=False,
            )
            choices = response.get("choices") or []
            if choices:
                content = choices[0].get("message", {}).get("content", "")
                if content:
                    return str(content).strip()
        except Exception:  # noqa: BLE001 - 缺少对话模板时回退
            logger.debug("对话接口不可用，回退到补全接口")

        fallback_prompt = f"{system_prompt}\n\n{prompt}\n"
        response = self._model(
            fallback_prompt,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=limit,
            stop=["</s>", "\n\n\n"],
        )
        choices = response.get("choices") or []
        if not choices:
            return ""
        return str(choices[0].get("text", "")).strip()

    def warm_up(self) -> bool:
        """预热：执行一次极短推理，把权重页换入内存。

        避免用户第一次触发翻译时出现明显卡顿。
        """
        if not self.load():
            return False
        try:
            self.translate_batch(["."], "Reply with a single dot.", max_tokens=8)
        except Exception:  # noqa: BLE001
            logger.debug("预热失败（不影响正常使用）")
            return False
        return True


def _llama_cpp_available() -> bool:
    """检测 llama-cpp-python 是否可导入。"""
    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        return False
    return True


def llama_cpp_available() -> bool:
    """对外暴露的绑定可用性检测。"""
    return _llama_cpp_available()
