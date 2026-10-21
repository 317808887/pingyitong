"""双模型池（自研）。

对应《项目原创性声明》二·（二）"双模型池的调度策略"与
开发文档二·（四）"模型按需加载、常驻与卸载机制"。

设计要点：

- **1.7B 常驻**：极速模型保持在内存中，保证毫秒级响应；
- **4B 懒加载**：高精度模型首次被调度到时才加载，避免启动即占用数 GB 内存；
- **空闲卸载**：4B 在超过 ``accurate_idle_timeout`` 未被使用后自动释放内存，
  缓解长时间后台运行导致的内存占用攀升；
- **加载串行化**：同一模型的并发加载请求合并为一次，避免重复占用内存。
"""

from __future__ import annotations

import contextlib
import hashlib
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from ...constants import MODEL_ACCURATE, MODEL_FAST, MODEL_LABELS, MODEL_REPOS
from ...utils import paths
from ...utils.execution_evidence import model_work
from ...utils.logger import get_logger
from ..llm.base import EngineStatus, LayoutPreviewEngine, TranslationEngine
from . import backend as backend_module
from .backend import BackendKind, BackendProbe
from .llama_engine import LlamaEngine, llama_cpp_available

logger = get_logger(__name__)


@dataclass
class _WeightFingerprint:
    signature: tuple
    digest: str = ""
    finished: threading.Event = field(default_factory=threading.Event)


def _weight_signature(path: Path) -> tuple:
    stat = path.stat()
    return (str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)


def _hash_model_weights(path: Path, expected: tuple) -> str:
    """Read complete weights once, rejecting a concurrent replacement/write."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        import os

        before = os.fstat(stream.fileno())
        signature = (str(path.resolve()), before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_ino)
        if signature != expected:
            return ""
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(stream.fileno())
    if (after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino) != expected[1:]:
        return ""
    return digest.hexdigest() if _weight_signature(path) == expected else ""


def _gpu_backend_available(executable: str) -> bool:
    """判断 llama-server 是否带有可用的 GPU 后端。

    有 CUDA 或 Vulkan 后端时可以让 llama.cpp 把层卸载到 GPU；
    纯 CPU 构建下传 ``--n-gpu-layers`` 也不会出错，但会多一次无谓的探测，
    因此这里先判断再决定。
    """
    if not executable:
        return False
    from .backend import _has_cuda_runtime, has_vulkan_runtime

    return _has_cuda_runtime(executable) or has_vulkan_runtime(executable)


@dataclass
class PreloadProgress:
    """模型预加载进度，供界面展示。"""

    #: 当前正在加载的模型标识
    model_id: str = ""
    #: 已完成数量 / 总数
    completed: int = 0
    total: int = 0
    #: 是否全部结束（无论成功与否）
    finished: bool = False
    #: 成功加载的模型列表
    loaded: list[str] = None  # type: ignore[assignment]
    #: 加载失败的模型 → 原因
    failed: dict[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.loaded is None:
            self.loaded = []
        if self.failed is None:
            self.failed = {}

    @property
    def ratio(self) -> float:
        """进度比例 0.0-1.0。"""
        return self.completed / self.total if self.total else 0.0

    def as_text(self) -> str:
        """生成中文进度描述。"""
        if self.finished:
            if self.failed:
                return f"模型加载完成（{len(self.failed)} 个未就绪）"
            return "模型已就绪"
        if not self.model_id:
            return "准备加载模型…"
        label = MODEL_LABELS.get(self.model_id, self.model_id)
        if self.total > 1:
            return f"正在加载 {label}（{self.completed + 1}/{self.total}）"
        return f"正在加载 {label}"


class ModelPool:
    """管理 1.7B / 4B 两个本地模型的生命周期。

    预加载在**后台线程**中执行：加载 1.7B 需要数秒、4B 需要十余秒，
    在主线程里做会让界面直接卡死（Qt 主线程被占用时窗口无法重绘、
    按钮无响应，用户会以为程序崩了）。
    """

    def __init__(self, model_config: object, idle_timeout: float = 180.0) -> None:
        """
        :param model_config: :class:`~pingyitong.config.schema.ModelConfig`。
        :param idle_timeout: 非常驻模型的空闲卸载超时（秒）。
        """
        self._config = model_config
        self.idle_timeout = max(30.0, float(idle_timeout))
        self._engines: dict[str, TranslationEngine] = {}
        self._locks: dict[str, threading.Lock] = {
            MODEL_FAST: threading.Lock(),
            MODEL_ACCURATE: threading.Lock(),
        }
        self._pool_lock = threading.Lock()
        self._preload_thread: threading.Thread | None = None
        self._preload_progress = PreloadProgress(total=0, finished=True)
        #: 最近一次后端探测结果，供界面展示"当前用的是哪个后端"
        self._backend_probe: BackendProbe | None = None
        #: 已预热的模型，避免重复预热（预热本身要花数秒）
        self._warmed: dict[str, bool] = {}
        #: 启动时预加载并保持常驻的模型。这些模型不参与空闲回收——
        #: 卸载后再次使用需要重新加载数秒，正是"双模型常驻"要消除的开销。
        self._preloaded: set[str] = set()
        #: 是否两个模型都常驻（决定切换是否零成本）
        self._both_models_resident = False
        #: 是否已请求最终清理（程序退出）。置位后预加载线程不再发起新的
        #: 加载：否则退出时的 unload 会与进行中的 load 竞态——unload 先
        #: 完成、load 随后拉起 llama-server，退出后便无人回收它。
        self._shutting_down = False
        #: 在线 AI 引擎（OpenAI 兼容接口）。非 None 且可用时，
        #: :meth:`translate` 的请求优先路由到它，失败回退本地模型。
        self._api_engine: object | None = None
        #: 上次应用的在线配置特征，用于判断是否需要重建引擎
        self._api_signature: tuple | None = None
        #: 在线服务配置版本；只用于进程内缓存命名空间，不含明文 Key。
        self._api_revision = 0
        #: Unverified weights never reuse translations across pool instances.
        #: Hashing is asynchronous; the translation hot path only checks stat.
        self._fingerprint_nonce = uuid4().hex
        self._fingerprint_lock = threading.Lock()
        self._weight_fingerprints: dict[str, _WeightFingerprint] = {}
        self._loaded_weight_signatures: dict[str, tuple] = {}

    def _model_weight_path(self, model_id: str) -> Path:
        with self._pool_lock:
            engine = self._engines.get(model_id)
        loaded_path = getattr(engine, "model_path", None)
        return Path(loaded_path) if loaded_path is not None else paths.resolve_model_path(
            self._config.for_model(model_id).file_name,
        )

    def model_content_fingerprint(self, model_id: str) -> str:
        """Return a content identity without reading GGUF bytes on this thread.

        Until the background SHA256 completes, the returned nonce only allows
        reuse within this pool. File stat changes invalidate the digest and
        schedule one new computation. A replaced file beneath a loaded engine
        stays session-scoped until the engine is explicitly reloaded.
        """
        path = self._model_weight_path(model_id)
        try:
            signature = _weight_signature(path)
        except OSError:
            return f"unavailable:{self._fingerprint_nonce}:{model_id}"
        with self._fingerprint_lock:
            loaded = self._loaded_weight_signatures.get(model_id)
            if loaded is not None and loaded != signature:
                stamp = hashlib.sha256(repr(signature).encode()).hexdigest()[:16]
                return f"changed-loaded:{self._fingerprint_nonce}:{model_id}:{stamp}"
            record = self._weight_fingerprints.get(model_id)
            if record is None or record.signature != signature:
                record = _WeightFingerprint(signature)
                self._weight_fingerprints[model_id] = record
                thread = threading.Thread(
                    target=self._compute_model_fingerprint, args=(model_id, path, record),
                    name=f"weight-sha256-{model_id}", daemon=True,
                )
                thread.start()
            if record.digest:
                return f"sha256:{record.digest}"
            stamp = hashlib.sha256(repr(signature).encode()).hexdigest()[:16]
            return f"pending:{self._fingerprint_nonce}:{model_id}:{stamp}"

    def _compute_model_fingerprint(self, model_id: str, path: Path, record: _WeightFingerprint) -> None:
        try:
            digest = _hash_model_weights(path, record.signature)
            with self._fingerprint_lock:
                if self._weight_fingerprints.get(model_id) is record:
                    record.digest = digest
        except OSError:
            logger.debug("模型内容指纹尚不可用：%s", model_id)
        finally:
            record.finished.set()

    def wait_for_model_fingerprints(
        self, model_ids: list[str] | None = None, timeout: float | None = 30.0,
    ) -> bool:
        """Explicit QA/startup wait; never called by the translation hot path.

        This only reads local files. It does not load a model or contact any
        inference backend. False means at least one identity remains isolated.
        """
        targets = model_ids if model_ids is not None else [
            model_id for model_id in (MODEL_FAST, MODEL_ACCURATE)
            if self._config.for_model(model_id).enabled
        ]
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        for model_id in targets:
            identity = self.model_content_fingerprint(model_id)
            if identity.startswith("sha256:"):
                continue
            if not identity.startswith("pending:"):
                return False
            with self._fingerprint_lock:
                record = self._weight_fingerprints[model_id]
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            if not record.finished.wait(remaining):
                return False
            if not self.model_content_fingerprint(model_id).startswith("sha256:"):
                return False
        return True

    # ------------------------------------------------------------------
    # 引擎获取
    # ------------------------------------------------------------------

    def engine(self, model_id: str) -> TranslationEngine:
        """取得指定模型的引擎实例（不触发加载）。"""
        with self._pool_lock:
            existing = self._engines.get(model_id)
            if existing is not None:
                return existing

        created = self._create_engine(model_id)
        with self._pool_lock:
            # 并发场景下可能已被其他线程创建，保留先创建的那个
            return self._engines.setdefault(model_id, created)

    def _create_engine(self, model_id: str) -> TranslationEngine:
        """按配置与可用后端创建引擎。

        后端选择顺序（详见 :mod:`.backend`）：

        1. ``llama-server`` 外部进程——Spark-X2.5 唯一可行的方式；
        2. ``llama-cpp-python`` 进程内——适用于上游支持的架构；
        3. 都不行时降级为预览引擎（界面如实标注，不伪装翻译结果）。
        """
        model_config = self._config.for_model(model_id)
        # 在全部候选目录中解析实际路径：项目内 models/ 优先，
        # 其次应用数据目录；并容忍文件名大小写与后缀差异。
        path = paths.resolve_model_path(model_config.file_name)

        prefer_server = bool(getattr(self._config, "prefer_server_backend", True))
        probe = backend_module.probe(
            allow_server=prefer_server,
            allow_python=True,
        )
        self._backend_probe = probe

        if probe.kind is BackendKind.LLAMA_SERVER:
            engine = self._create_server_engine(model_id, model_config, path, probe)
            if engine is not None and engine.is_available():
                logger.info(
                    "模型后端：llama-server（%s）→ %s",
                    MODEL_LABELS.get(model_id, model_id),
                    path,
                )
                return engine
            # llama-server 可用但该模型文件缺失：继续尝试进程内后端没有意义
            # （同一份权重文件），直接降级为预览引擎
            reason = engine.unavailable_reason if engine is not None else probe.detail
            return LayoutPreviewEngine(model_id, reason, path=str(path))

        if probe.kind is BackendKind.LLAMA_CPP_PYTHON:
            return self._create_python_engine(model_id, model_config, path)

        return LayoutPreviewEngine(model_id, probe.detail, path=str(path))

    def _create_server_engine(
        self,
        model_id: str,
        model_config: object,
        path: Path,
        probe: object,
    ) -> object | None:
        """创建 llama-server 外部进程引擎。"""
        try:
            from .llama_server import LlamaServerEngine, default_chat_template
        except ImportError:  # pragma: no cover - 模块缺失属异常情况
            logger.exception("无法导入 llama-server 引擎")
            return None

        # GPU 卸载层数：用户显式配置了就用配置值，否则在有 GPU 后端时
        # 让引擎自动卸载全部层（llama.cpp 会按显存容量自行裁剪）。
        #
        # 注意：这里必须把"配置值"与"是否自动"分开传递。
        # 曾经把自动值（999）直接写进 n_gpu_layers，导致 apply_config 的
        # 参数比较永远不相等（999 vs 配置 0），每次配置变更都会卸载重载模型。
        configured_layers = int(getattr(model_config, "n_gpu_layers", 0))
        auto_offload = configured_layers == 0 and _gpu_backend_available(
            str(probe.executable)
        )
        if auto_offload:
            logger.info(
                "%s 检测到 GPU 后端，将尝试自动卸载全部层",
                MODEL_LABELS.get(model_id, model_id),
            )

        return LlamaServerEngine(
            model_id=model_id,
            model_path=Path(path),
            executable=str(probe.executable),
            n_ctx=model_config.n_ctx,
            n_threads=model_config.n_threads,
            n_gpu_layers=configured_layers,
            auto_gpu_offload=auto_offload,
            temperature=model_config.temperature,
            top_p=model_config.top_p,
            max_tokens=model_config.max_tokens,
            # Spark-X2.5 的 GGUF 不含 chat template，必须外部提供，
            # 否则模型输出会退化成重复片段（实测）。
            chat_template=default_chat_template(),
            enable_thinking=bool(getattr(self._config, "enable_thinking", False)),
        )

    def _create_python_engine(
        self,
        model_id: str,
        model_config: object,
        path: Path,
    ) -> TranslationEngine:
        """创建 llama-cpp-python 进程内引擎。"""
        engine = LlamaEngine(
            model_id=model_id,
            model_path=Path(path),
            n_ctx=model_config.n_ctx,
            n_threads=model_config.n_threads,
            n_gpu_layers=model_config.n_gpu_layers,
            n_batch=model_config.n_batch,
            temperature=model_config.temperature,
            top_p=model_config.top_p,
            max_tokens=model_config.max_tokens,
        )

        if engine.is_available():
            logger.info("模型权重已定位：%s → %s", MODEL_LABELS.get(model_id, model_id), path)
            return engine

        reason = engine.unavailable_reason
        if not llama_cpp_available():
            logger.warning(
                "llama-cpp-python 未安装，%s 进入预览模式（仅验证渲染链路）",
                MODEL_LABELS.get(model_id, model_id),
            )
        else:
            logger.warning(
                "%s 权重缺失（%s），进入预览模式。可通过 scripts/download_models.py 下载，"
                "或从 Hugging Face 仓库 %s 获取后放入 %s",
                MODEL_LABELS.get(model_id, model_id),
                model_config.file_name,
                MODEL_REPOS.get(model_id, ""),
                paths.project_models_dir(),
            )
        return LayoutPreviewEngine(model_id, reason, path=str(path))

    # ------------------------------------------------------------------
    # 加载 / 卸载
    # ------------------------------------------------------------------

    def ensure_loaded(self, model_id: str) -> bool:
        """确保模型已加载**并完成预热**。

        预热是必要的：模型加载完只是把权重读进内存，第一次真正的前向计算
        仍要付出可观开销（实测首次翻译 11–21 秒，之后每句 0.06–0.11 秒）。
        若不做预热，这笔开销会落在用户点"开始翻译"后的第一帧上，
        用户会以为程序卡死。

        加载与预热紧挨着发生：``engine.load()`` 一旦成功就立刻预热，且
        **预热明确成功才登记** ``_warmed``。预热失败若也登记成功，下次
        请求会走"已热"快路径跳过预热，把 11–21 秒的首次前向开销甩给
        翻译请求——这正是"翻译路径冷加载"的来源之一。正常情况下启动
        预加载（:meth:`preload_async`）在后台线程完成同一套加载+预热，
        翻译请求只命中"已热"快路径；内存不足未预加载的模型走这里的
        同步回退（按需加载路径保留）。

        :return: 是否可用（预览模式返回 ``False``）。
        """
        engine = self.engine(model_id)
        if engine.is_loaded() and self._warmed.get(model_id, False):
            return True
        if not engine.is_available():
            return False

        lock = self._locks.setdefault(model_id, threading.Lock())
        with lock:
            # 双重检查：等待锁期间可能已被其他线程加载完成
            if engine.is_loaded() and self._warmed.get(model_id, False):
                return True

            if not engine.is_loaded():
                logger.info("正在加载模型：%s", MODEL_LABELS.get(model_id, model_id))
                # Record which on-disk revision the engine actually loads.
                # Replacing that path later must not bless old-engine output
                # with the replacement file's portable cache identity.
                try:
                    loaded_signature = _weight_signature(self._model_weight_path(model_id))
                except OSError:
                    loaded_signature = None
                with model_work("model_load", model_id) as load_event:
                    loaded_ok = engine.load()
                    if load_event is not None:
                        load_event["successful"] = bool(loaded_ok)
                if not loaded_ok:
                    return False
                with self._fingerprint_lock:
                    if loaded_signature is not None:
                        self._loaded_weight_signatures[model_id] = loaded_signature

            if not self._warmed.get(model_id, False):
                logger.info("正在预热模型：%s", MODEL_LABELS.get(model_id, model_id))
                with model_work("model_warmup", model_id) as warm_event:
                    warmed = engine.warm_up()
                    if warm_event is not None:
                        warm_event["successful"] = warmed is not False
                if warmed is False:
                    # 不登记 _warmed：下次 ensure_loaded 会重试预热，
                    # 避免冷模型被当成已热模型送进翻译请求。
                    logger.warning(
                        "模型预热失败：%s，将在下次使用时重试",
                        MODEL_LABELS.get(model_id, model_id),
                    )
                else:
                    # 只有明确 False 才算失败：部分替身引擎不返回布尔值。
                    self._warmed[model_id] = True

            return True

    def unload(self, model_id: str) -> None:
        """卸载指定模型。"""
        engine = self.engine(model_id)
        if engine.is_loaded():
            engine.unload()
        # 卸载后必须清掉预热标记：否则重新加载时会跳过预热，
        # 下一次翻译就要用户承担那十几秒的首次前向计算开销。
        self._warmed.pop(model_id, None)
        with self._fingerprint_lock:
            self._loaded_weight_signatures.pop(model_id, None)

    def unload_idle(self) -> list[str]:
        """卸载已超时闲置的非常驻模型。

        :return: 被卸载的模型标识列表。
        """
        unloaded: list[str] = []
        for model_id in (MODEL_FAST, MODEL_ACCURATE):
            model_config = self._config.for_model(model_id)
            if model_config.keep_resident:
                continue

            # 预加载常驻的模型不参与空闲回收：卸载后再用又要重新加载，
            # 用户会经历一次数秒的卡顿——这正是"同时常驻两个模型"要消除的开销。
            if model_id in self._preloaded:
                continue

            engine = self.engine(model_id)
            if not engine.is_loaded():
                continue

            # 正在推理的模型绝不能卸载：终止服务会让对方的请求连接被重置，
            # 那一帧翻译直接失败（实测出现过 ConnectionResetError）。
            if getattr(engine, "busy", False):
                logger.debug("%s 正在推理中，跳过本轮空闲回收", model_id)
                continue

            idle = getattr(engine, "idle_seconds", 0.0)
            if idle >= self.idle_timeout:
                logger.info(
                    "模型空闲 %.0fs 超过阈值 %.0fs，自动卸载释放内存",
                    idle,
                    self.idle_timeout,
                )
                # busy 检查与真正终止之间存在窗口：请求可能恰好在此刻拿到
                # 推理锁。llama-server 引擎用 try_unload 先取推理锁再终止，
                # 取不到锁说明请求正在进行，跳过本轮（下轮再试）。
                try_unload = getattr(engine, "try_unload", None)
                if try_unload is not None:
                    if not try_unload():
                        logger.debug("%s 正在推理中，跳过本轮空闲回收", model_id)
                        continue
                else:
                    engine.unload()
                # 同 unload()：清掉预热标记，使下次加载重新预热
                self._warmed.pop(model_id, None)
                unloaded.append(model_id)
        return unloaded

    def preload(self, model_ids: list[str] | None = None) -> dict[str, bool]:
        """**同步**预加载指定模型（会阻塞调用线程）。

        适用于脚本、测试与"用户手动点击加载"且已确认可接受阻塞的场景。
        界面启动路径请使用 :meth:`preload_async`。
        """
        targets = model_ids or [MODEL_FAST]
        return {model_id: self.ensure_loaded(model_id) for model_id in targets}

    # ------------------------------------------------------------------
    # 异步预加载
    # ------------------------------------------------------------------

    def preload_targets(self) -> list[str]:
        """返回按配置与可用内存应当预加载的模型列表。

        双重考虑：

        1. **配置开关**——各模型的 ``preload`` 字段；
        2. **资源是否允许**——若内存/显存足够同时容纳两个模型，则**主动
           把高精度模型也预加载**。

        为什么第 2 点至关重要：模型切换本身要卸载旧模型、加载新模型并预热，
        实测 3–10 秒。若两个模型都常驻，切换只是"换一个已加载好的引擎来用"，
        **耗时接近零**。对本项目这类实时工具，切换延迟直接决定可用性——
        用户看到的是翻译卡住十秒。

        预算必须对应实际推理设备：GPU 后端只看显存，CPU 后端看系统
        内存。GPU 显存不可测时，不以系统内存代替，不自动加载第二模型。
        """
        targets: list[str] = []

        for model_id in (MODEL_FAST, MODEL_ACCURATE):
            model_config = self._config.for_model(model_id)
            if not model_config.enabled:
                continue

            wanted = bool(model_config.preload)
            if model_id == MODEL_ACCURATE and not wanted:
                # 高精度模型默认不预加载，但资源够就应该加载——
                # 这样自动调度切换它时不必付出加载代价。
                # This is an automatic *second* model, not permission to load
                # accurate alone when the user disabled fast preloading.
                wanted = MODEL_FAST in targets and self._can_hold_both_models()
                if wanted:
                    logger.info(
                        "内存/显存充足，将同时预加载高精度模型，使模型切换无需重新加载"
                    )

            if not wanted:
                continue

            # 权重不存在时不预加载：避免无意义地等待一个必然失败的加载
            if paths.find_model_file(model_config.file_name) is None:
                logger.debug(
                    "%s 未启用预加载：未找到权重文件 %s",
                    MODEL_LABELS.get(model_id, model_id),
                    model_config.file_name,
                )
                continue

            targets.append(model_id)

        self._both_models_resident = MODEL_FAST in targets and MODEL_ACCURATE in targets
        return targets

    def _can_hold_both_models(self) -> bool:
        """判断资源是否足够同时常驻两个模型。

        估算方式：模型 Q4 量化后约为文件体积的 1.15 倍（含 KV 缓存与中间张量）。
        GPU 路径再保留约 1GB 显存，CPU 路径保留约 2GB 内存。
        已知 GPU 后端却无法读取显存时，保守不自动加载第二模型。
        """
        fast = self._config.for_model(MODEL_FAST)
        accurate = self._config.for_model(MODEL_ACCURATE)

        needed_mb = 0.0
        for model_config in (fast, accurate):
            path = paths.find_model_file(model_config.file_name)
            if path is None:
                return False  # 权重不全，谈不上同时常驻
            try:
                needed_mb += path.stat().st_size / (1024 * 1024) * 1.15
            except OSError:
                return False

        # Match _create_server_engine's actual GPU offload decision. A Vulkan
        # build uses GPU memory even when the CUDA runtime is unavailable.
        probe = self.backend_probe()
        uses_gpu = probe.kind is BackendKind.LLAMA_SERVER and (
            probe.cuda_available or _gpu_backend_available(str(probe.executable))
        )
        if uses_gpu:
            free_mb = self._gpu_free_memory_mb()
            if free_mb is None:
                logger.debug("GPU 显存预算不可用，不自动预加载第二模型")
                return False
            return free_mb >= needed_mb + 1024.0

        # CPU 路径：看系统可用内存
        try:
            import psutil

            available_mb = psutil.virtual_memory().available / (1024 * 1024)
        except Exception:  # noqa: BLE001 - psutil 缺失时保守不预加载
            return False

        # 这里用"可用内存"而不是"总内存"：总内存大但已被占满时预加载会更糟
        return available_mb >= needed_mb + 2048.0

    def _gpu_free_memory_mb(self) -> float | None:
        """返回 GPU 空闲显存（MB）；无法获取时返回 ``None``。"""
        nvml = None
        initialized = False
        try:
            import pynvml

            nvml = pynvml
            # Preloading happens before ResourceMonitor's first sample. Own a
            # separate NVML reference instead of depending on its global state.
            nvml.nvmlInit()
            initialized = True
            handle = nvml.nvmlDeviceGetHandleByIndex(0)
            return float(nvml.nvmlDeviceGetMemoryInfo(handle).free) / (1024 * 1024)
        except Exception:  # noqa: BLE001 - unknown VRAM forbids automatic dual GPU residency
            return None
        finally:
            if initialized and nvml is not None:
                # Release only this call's successful init. A monitor already
                # using NVML keeps its own reference and remains functional.
                with contextlib.suppress(Exception):
                    nvml.nvmlShutdown()

    @property
    def preload_progress(self) -> PreloadProgress:
        """返回预加载进度快照。"""
        return self._preload_progress

    @property
    def is_preloading(self) -> bool:
        """是否正在进行预加载。"""
        thread = self._preload_thread
        return thread is not None and thread.is_alive()

    def preload_async(
        self,
        model_ids: list[str] | None = None,
        on_progress: Callable[[PreloadProgress], None] | None = None,
        on_finished: Callable[[PreloadProgress], None] | None = None,
    ) -> bool:
        """在后台线程中预加载模型。

        :param model_ids: 要加载的模型；``None`` 时使用 :meth:`preload_targets`。
        :param on_progress: 进度回调，**在后台线程中被调用**。
            若回调需要更新界面，必须自行切换到主线程（例如通过 Qt 信号）。
        :param on_finished: 完成回调，同样在后台线程中调用。
        :return: 是否启动了预加载；已在加载中或无目标时返回 ``False``。
        """
        if self.is_preloading:
            logger.debug("预加载已在进行中，忽略重复请求")
            return False

        targets = model_ids if model_ids is not None else self.preload_targets()
        if not targets:
            self._preload_progress = PreloadProgress(total=0, finished=True)
            return False
        for model_id in targets:
            self.model_content_fingerprint(model_id)

        # 记录为常驻集合：预加载的模型不再参与空闲回收，
        # 使后续模型切换无需重新加载（切换延迟接近零）。
        self._preloaded.update(targets)
        # 新一轮预加载意味着程序仍在使用中（例如"重新检测模型"），
        # 撤销之前 shutdown() 置位的清理标记，否则预加载线程会立刻停手。
        self._shutting_down = False

        progress = PreloadProgress(total=len(targets))
        self._preload_progress = progress

        def _worker() -> None:
            for index, model_id in enumerate(targets):
                if self._shutting_down:
                    logger.info("程序正在退出，跳过剩余模型的预加载")
                    break

                progress.model_id = model_id
                progress.completed = index
                if on_progress is not None:
                    _safe_call(on_progress, progress, "进度回调")

                try:
                    ok = self.ensure_loaded(model_id)
                except Exception as exc:  # noqa: BLE001 - 加载失败不应终止整批
                    ok = False
                    progress.failed[model_id] = str(exc)
                    logger.exception("预加载 %s 时发生异常", model_id)

                if ok:
                    progress.loaded.append(model_id)
                elif model_id not in progress.failed:
                    # 未抛异常但返回 False：多为权重缺失或依赖未安装
                    engine = self.engine(model_id)
                    progress.failed[model_id] = (
                        engine.status().message or "模型不可用"
                    )

                progress.completed = index + 1
                if on_progress is not None:
                    _safe_call(on_progress, progress, "进度回调")

            progress.finished = True
            progress.model_id = ""
            if on_finished is not None:
                _safe_call(on_finished, progress, "完成回调")

        self._preload_thread = threading.Thread(
            target=_worker, name="ModelPreload", daemon=True
        )
        self._preload_thread.start()
        return True

    def wait_for_preload(self, timeout: float | None = None) -> bool:
        """等待预加载结束（主要用于测试与退出前的收尾）。"""
        thread = self._preload_thread
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()

    def shutdown(self, *, close_api: bool = True) -> None:
        """卸载全部模型。程序退出时调用。"""
        # 预加载线程可能正在拉起 llama-server：先让它停手并短暂等待，
        # 否则 unload 与进行中的 load 竞态——unload 先结束、load 随后
        # 拉起新进程，退出后便无人回收它（作业对象能兜底，但属于纯浪费）。
        # 只等 3 秒：预加载正常需 2-10 秒，用户"启动后立刻退出"时
        # 不值得为此拖住退出流程；剩余线程是守护线程，进程退出即终结。
        self._shutting_down = True
        self.wait_for_preload(timeout=3.0)

        with self._pool_lock:
            engines = list(self._engines.values())
            self._warmed.clear()
        with self._fingerprint_lock:
            self._loaded_weight_signatures.clear()

        for engine in engines:
            try:
                engine.unload()
            except Exception:  # noqa: BLE001 - 退出路径不因卸载异常而中断
                logger.debug("卸载模型时出现异常（可忽略）")

        api_engine = self._api_engine
        if close_api and api_engine is not None:
            with contextlib.suppress(Exception):
                api_engine.unload()

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------

    def translate(
        self,
        model_id: str,
        prompt: str,
        system_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        """使用指定模型执行一次翻译请求。

        在线 AI 引擎已启用且可用时，请求优先路由到云端模型；
        云端失败（网络异常、超时、限流、冷却中）**自动回退本地模型**，
        保证任何情况下译文条路都是通的。

        :return: 模型输出原文；失败或处于预览模式时返回空字符串。
        """
        api_engine = self._api_engine
        if api_engine is not None and api_engine.is_available():
            try:
                with model_work("api_inference", "configured_api") as api_event:
                    results = api_engine.translate_batch([prompt], system_prompt, max_tokens)
                    if api_event is not None:
                        api_event["successful"] = bool(results and results[0])
            except Exception:  # noqa: BLE001 - 引擎异常等价于请求失败
                logger.warning("在线 AI 请求异常，本次回退本地模型", exc_info=True)
                results = None
            raw = results[0] if results else ""
            if raw:
                self._last_inference_at = time.perf_counter()
                return raw
            if results is not None:
                logger.warning("在线 AI 未返回结果，本次回退本地模型")

        return self.translate_local(model_id, prompt, system_prompt, max_tokens)

    def translate_local(
        self, model_id: str, prompt: str, system_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        """仅访问本机模型；不得因在线配置或失败而改走网络。"""
        engine = self.engine(model_id)
        if not engine.is_available() or not self.ensure_loaded(model_id):
            return ""
        with model_work("local_inference", model_id) as inference_event:
            results = engine.translate_batch([prompt], system_prompt, max_tokens)
            if inference_event is not None:
                inference_event["successful"] = bool(results and results[0])
        # 记录最近一次推理结束时刻：预翻译调度器以"模型空闲秒数"作为
        # 出队门控之一——任何实时推理活动都会重置空闲窗，压制后台预取
        self._last_inference_at = time.perf_counter()
        return results[0] if results else ""

    # ------------------------------------------------------------------
    # 在线 AI（OpenAI 兼容接口）
    # ------------------------------------------------------------------

    def apply_api_config(self, api_config: object) -> None:
        """应用在线 AI 配置；有效字段变化时重建引擎。

        重建是廉价的（无权重、无进程），旧引擎的连接池立即关闭。
        """
        signature = self._api_config_signature(api_config)
        if signature == self._api_signature:
            return
        self._api_signature = signature
        self._api_revision += 1
        old_engine = self._api_engine
        self._api_engine = None
        if old_engine is not None:
            old_engine.unload()
        if not signature[0]:
            return

        from .api_engine import ApiEngine

        engine = ApiEngine(api_config)
        self._api_engine = engine
        logger.info("在线 AI 已启用：%s · %s", signature[1], signature[3])
        # 后台预建 TCP/TLS 连接：首个在线翻译帧免付 100–300ms 握手。
        # 纯连接不产生 HTTP 请求、不消耗配额；失败静默（首次请求再试）
        threading.Thread(
            target=engine.preconnect, name="ApiPreconnect", daemon=True
        ).start()

    @staticmethod
    def _api_config_signature(api_config: object) -> tuple:
        """Return all settings that change the wire request or active route."""
        return (
            bool(getattr(api_config, "enabled", False)),
            str(getattr(api_config, "base_url", "")).strip().rstrip("/"),
            str(getattr(api_config, "api_key", "")),
            str(getattr(api_config, "model_name", "")).strip(),
            round(float(getattr(api_config, "temperature", 0.2)), 4),
            round(float(getattr(api_config, "timeout_seconds", 10.0)), 3),
            int(getattr(api_config, "max_tokens", 768)),
        )

    def api_mode_active(self) -> bool:
        """当前是否由在线 AI 承担翻译（已启用、配置完整且不在冷却期）。

        翻译服务据此选择批量参数（在线模型上下文长、可并发）。
        """
        engine = self._api_engine
        return engine is not None and engine.is_available()

    def translation_cache_namespace(self) -> str:
        """Distinguish local and online translations without exposing the Key.

        A failed online request increments the engine generation before the
        local fallback is returned.  The fallback was prepared under the old
        namespace, so it cannot prevent the next online retry or recovery.
        """
        engine = self._api_engine
        if engine is None or not engine.is_available():
            return "local"
        return f"api:{self._api_revision}:{engine.failure_generation}"

    def test_api_connection(self, api_config: object) -> tuple[bool, str]:
        """Test the live online engine when enabled, clearing its cooldown.

        Before enabling online mode there is no live engine, so a temporary
        one lets the user validate address, model, and Key first.
        """
        engine = (
            self._api_engine
            if self._api_config_signature(api_config) == self._api_signature
            else None
        )
        if engine is not None:
            return engine.test()
        from .api_engine import ApiEngine

        temporary = ApiEngine(api_config)
        try:
            return temporary.test()
        finally:
            temporary.unload()

    def api_status(self) -> EngineStatus | None:
        """在线引擎状态快照；未启用时返回 ``None``。"""
        if self._api_engine is None:
            return None
        return self._api_engine.status()

    def model_idle_seconds(self) -> float:
        """距离最近一次模型推理结束的秒数；从未推理过返回 ``inf``。"""
        last = getattr(self, "_last_inference_at", None)
        if last is None:
            return float("inf")
        return time.perf_counter() - last

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------

    def statuses(self) -> dict[str, EngineStatus]:
        """返回两个模型的状态快照。"""
        return {model_id: self.engine(model_id).status() for model_id in (MODEL_FAST, MODEL_ACCURATE)}

    def backend_probe(self) -> BackendProbe:
        """返回当前推理后端的探测结果。

        首次调用会触发探测，之后复用结果；``reset()`` 后可重新探测。
        """
        if self._backend_probe is None:
            self._backend_probe = backend_module.probe(
                allow_server=bool(getattr(self._config, "prefer_server_backend", True)),
                allow_python=True,
            )
        return self._backend_probe

    def backend_summary(self) -> str:
        """返回后端与加速方式的中文描述，供界面与状态栏展示。"""
        probe = self.backend_probe()
        if not probe.usable:
            return "预览模式（无可用推理后端）"

        if probe.kind is BackendKind.LLAMA_SERVER:
            from .backend import has_vulkan_runtime

            accents: list[str] = []
            if probe.cuda_available:
                accents.append("CUDA")
            if has_vulkan_runtime(probe.executable):
                accents.append("Vulkan")
            suffix = f" · {'/'.join(accents)} 可选" if accents else " · CPU"
            return f"llama-server{suffix}"

        return probe.kind.label

    def is_backend_ready(self) -> bool:
        """推理后端是否可用（可用于真实翻译）。"""
        return self.backend_probe().usable

    @property
    def both_models_resident(self) -> bool:
        """两个模型是否都已常驻内存。

        为 ``True`` 时模型切换不需要加载，延迟接近零——调度器可以放心
        在极速与高精度之间按内容切换而不必担心卡顿。
        """
        return self._both_models_resident and all(
            self.engine(model_id).is_loaded() for model_id in (MODEL_FAST, MODEL_ACCURATE)
        )

    def switch_cost_hint(self) -> str:
        """返回切换模型的大致代价描述，供调度器与界面参考。"""
        if self.both_models_resident:
            return "两个模型均已常驻，切换无加载开销"
        return "切换需加载模型，可能造成数秒延迟"

    def status_text(self, model_id: str) -> str:
        """返回单个模型的状态描述文本。"""
        return self.engine(model_id).status().as_text()

    def is_available(self, model_id: str) -> bool:
        """模型是否可用（预览模式返回 ``False``）。"""
        return self.engine(model_id).is_available()

    def any_available(self) -> bool:
        """是否至少有一个真实模型可用。"""
        return any(self.is_available(mid) for mid in (MODEL_FAST, MODEL_ACCURATE))

    def loaded_models(self) -> list[str]:
        """当前已加载到内存的模型列表。"""
        return [
            model_id
            for model_id in (MODEL_FAST, MODEL_ACCURATE)
            if self.engine(model_id).is_loaded()
        ]

    def estimated_memory_mb(self) -> float:
        """已加载模型的估算内存占用（MB）。"""
        total = 0.0
        with self._pool_lock:
            engines = list(self._engines.values())
        for engine in engines:
            if engine.is_loaded() and hasattr(engine, "estimate_memory_mb"):
                total += engine.estimate_memory_mb()
        return total

    def apply_config(self, model_config: object, idle_timeout: float | None = None) -> None:
        """应用新的模型配置。

        参数变化需要重新加载才能生效。

        **不能在推理进行中强行卸载**：本方法由界面在主线程调用，而推理在
        流水线线程中进行；此刻终止 llama-server 会让对方的 HTTP 连接被
        重置（实测日志里的 ``ConnectionResetError`` 就是这么来的）。
        因此改为"忙时只打标记，下次使用前重载"，既避免连接重置，
        也避免在主线程等待推理结束（最多可能等 60 秒，界面会卡住）。
        """
        with self._pool_lock:
            engines = list(self._engines.items())

        for model_id, engine in engines:
            new_config = model_config.for_model(model_id)
            changed = (
                getattr(engine, "n_ctx", None) != new_config.n_ctx
                or getattr(engine, "n_gpu_layers", None) != new_config.n_gpu_layers
                or getattr(engine, "n_threads", None) != new_config.n_threads
            )
            if not (changed and engine.is_loaded()):
                continue

            if getattr(engine, "busy", False):
                # 有请求正在进行：只打标记，由 ensure_loaded 在下次使用前重载
                logger.info(
                    "模型参数已变更，但 %s 正在推理中，将在下次使用前重载", model_id
                )
                if hasattr(engine, "request_reload"):
                    engine.request_reload()
            else:
                # busy 检查与卸载之间仍有窗口（请求可能刚拿到推理锁），
                # llama-server 引擎用 try_unload 先取推理锁再终止；
                # 取不到锁说明请求刚开始，退回"忙时打标记"路径。
                try_unload = getattr(engine, "try_unload", None)
                if try_unload is not None:
                    if not try_unload():
                        logger.info(
                            "模型参数已变更，但 %s 刚开始推理，将在下次使用前重载",
                            model_id,
                        )
                        if hasattr(engine, "request_reload"):
                            engine.request_reload()
                        continue
                else:
                    engine.unload()
                logger.info("模型参数已变更，卸载 %s 以便按新参数重新加载", model_id)

        self._config = model_config
        if idle_timeout is not None:
            self.idle_timeout = max(30.0, float(idle_timeout))

    def reset(self) -> None:
        """丢弃全部引擎实例，下次访问时按当前配置重建。

        同时清除后端探测缓存——用户可能刚下载好推理运行时，
        重新探测才能发现它。
        """
        # A local weights/backend rescan does not change the online service.
        # Keep its live connection pool: unloading it marks the API engine
        # permanently closed, while an unchanged API signature will reuse it.
        self.shutdown(close_api=False)
        with self._pool_lock:
            self._engines.clear()
        self._backend_probe = None


def _safe_call(callback: Callable[[PreloadProgress], None], progress: PreloadProgress, label: str) -> None:
    """调用回调并吞掉异常。

    进度回调通常来自界面层；一次回调抛异常不应该让模型加载失败。
    """
    try:
        callback(progress)
    except Exception:  # noqa: BLE001
        logger.exception("%s执行失败", label)


def _default_model_path(file_name: str) -> Path:
    """便捷函数：返回模型文件的解析路径（供脚本与测试使用）。"""
    return paths.resolve_model_path(file_name)


__all__ = ["ModelPool", "PreloadProgress", "MODEL_FAST", "MODEL_ACCURATE"]
