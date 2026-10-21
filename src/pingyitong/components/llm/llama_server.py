"""llama-server 外部进程推理后端。

Spark-X2.5 的 ``spark2_5`` 架构只有 XHToken 维护的 llama.cpp 分支支持，
因此这个后端是让 Spark-X2.5 真正跑起来的**唯一途径**（详见 :mod:`.backend`）。

工作方式
    本模块拉起 ``llama-server.exe`` 子进程并把模型加载进该进程，
    然后通过**本机回环地址**（默认 ``127.0.0.1``，端口由系统分配以避免
    冲突）用 HTTP 调用它的 **OpenAI 兼容对话接口**
    ``/v1/chat/completions``。

    必须用对话接口而不是原始的 ``/completion``：Spark-X2.5 的 GGUF 权重里
    **不含 chat template**（``tokenizer.chat_template`` 缺失），直接走补全
    接口会让模型看到裸文本，输出退化为重复片段或自问自答。对话接口会套用
    我们通过 ``--chat-template-file`` 传入的官方模板。

关于"全离线"：这是与本程序自己拉起的子进程之间的进程间通信，
不涉及任何外部网络。程序**不发起任何对外网络请求**。

两个来自实测的坑（都已处理）
    1. **非 ASCII 路径**：llama.cpp 读取 chat template 文件用的是窄字符路径，
       在中文 Windows 上无法打开含中文的路径（而模型加载用的是宽字符路径，
       所以模型路径可以有中文）。因此本模块会把模板复制到一个纯 ASCII 的
       临时目录再传给子进程。
    2. **模型是推理模型**：默认会先输出思考过程。桌面翻译需要的是即时响应，
       因此默认关闭思考（``enable_thinking=False``）——实测开启思考会让单句
       延迟从约 0.1 秒升到约 0.6 秒。关闭后仍能处理大多数界面文案，
       用户可在设置中开启以换取更贴切的措辞。
"""

from __future__ import annotations

import contextlib
import http.client
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from ...utils.logger import get_logger
from ...utils.process_guard import assign_to_job
from .base import EngineMetrics, EngineStatus

logger = get_logger(__name__)

#: 启动后等待服务就绪的最长时间（秒）。
#: 4B 模型在机械硬盘上加载可能需要 20 秒以上，因此给足余量。
STARTUP_TIMEOUT = 180.0

#: 单次请求的超时（秒）。桌面翻译要保证响应性，过长的超时没有意义——
#: 与其等它超时，不如让流水线丢弃该帧。
REQUEST_TIMEOUT = 60.0

#: 就绪探测的间隔（秒）
_POLL_INTERVAL = 0.5

#: 项目内随附的 Spark-X2.5 官方 chat template 文件名
CHAT_TEMPLATE_NAME = "spark-x2.5.jinja"



class LlamaServerError(RuntimeError):
    """llama-server 启动或通信失败。"""


class LlamaServerEngine:
    """通过 llama-server 子进程执行推理的引擎。

    一个实例对应一个模型文件。线程安全：推理请求按槽位信号量限流
    （llama-server 默认 4 个槽位，能同时解码多个序列；把并发控制放在
    客户端既匹配服务端槽位数，也让"终止前等待在途请求"有确定语义）。
    """

    #: 后端类型标识，供界面展示
    backend_kind = "llama-server"

    def __init__(
        self,
        model_id: str,
        model_path: Path,
        executable: str,
        n_ctx: int = 2048,
        n_threads: int = 0,
        n_gpu_layers: int = 0,
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: int = 512,
        prefer_gpu: bool = True,
        enable_thinking: bool = False,
        chat_template: Path | None = None,
        auto_gpu_offload: bool = False,
    ) -> None:
        self.model_id = model_id
        self.model_path = Path(model_path)
        self.executable = executable
        self.n_ctx = max(256, int(n_ctx))
        self.n_threads = max(0, int(n_threads))
        #: **配置中的** GPU 卸载层数。
        #:
        #: 这里必须保存配置值而非实际生效值：``ModelPool.apply_config``
        #: 会用它与新配置比较来判断"参数是否变更"。若存生效值（自动模式下
        #: 是 999），就会与新配置的 0 永不相等，导致每次配置变更都误判为
        #: "参数已变"并卸载重载模型——实测表现为模型反复加载卸载。
        self.n_gpu_layers = max(0, int(n_gpu_layers))
        #: 是否让 llama.cpp 自动把全部层卸载到 GPU（配置为 0 但检测到 GPU 时启用）
        self.auto_gpu_offload = bool(auto_gpu_offload)
        self.temperature = max(0.0, min(2.0, float(temperature)))
        self.top_p = max(0.0, min(1.0, float(top_p)))
        self.max_tokens = max(32, int(max_tokens))
        self.prefer_gpu = bool(prefer_gpu)
        #: 是否启用模型的思考过程。桌面实时翻译默认关闭以换取响应速度。
        self.enable_thinking = bool(enable_thinking)
        self.chat_template = Path(chat_template) if chat_template else None

        self._process: subprocess.Popen[bytes] | None = None
        self._port: int = 0
        # 推理并发上限。llama-server 默认 4 槽位（实测启动日志 n_slots=4），
        # 客户端放行同样数量的在途请求：短字幕场景两段拆成两次单段请求
        # 并发发送，GPU 把两条序列合批解码，比合并成一条长请求省约 45%
        # 墙钟时间（实测 236.7ms → 129.9ms）。超出槽数的请求由服务端排队。
        self._max_concurrent = 4
        self._infer_slots = threading.BoundedSemaphore(self._max_concurrent)
        self._inflight = 0
        self._inflight_lock = threading.Lock()
        self._load_lock = threading.Lock()
        self._last_used = 0.0
        #: 服务就绪的时刻。用作"从未推理过"时计算空闲时长的基准——
        #: 刚加载完的模型不是空闲的，绝不能被空闲回收逻辑立刻卸载。
        self._loaded_at = 0.0
        self._unavailable_reason = ""
        self._log_path: Path | None = None
        self._log_handle: Any = None
        self._template_copy: Path | None = None
        self.metrics = EngineMetrics()
        #: 是否需要在下次使用前重新加载（参数变更但当时正在推理时置位）
        self._needs_reload = False

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """可执行文件与权重文件是否均就绪。"""
        if not self.executable or not Path(self.executable).is_file():
            self._unavailable_reason = (
                "未找到 llama-server 可执行文件。"
                "请运行 scripts/download_runtime.py 获取推理运行时"
            )
            return False
        if not self.model_path.is_file():
            self._unavailable_reason = (
                f"未找到模型权重：{self.model_path.name}（期望位置 {self.model_path.parent}）"
            )
            return False
        self._unavailable_reason = ""
        return True

    def is_loaded(self) -> bool:
        """服务进程是否已在运行并就绪。"""
        return self._process is not None and self._process.poll() is None

    @property
    def unavailable_reason(self) -> str:
        """不可用原因，供界面提示。"""
        return self._unavailable_reason

    @property
    def busy(self) -> bool:
        """是否有推理请求正在进行。

        读在途计数而非尝试获取许可——本属性会被界面线程读取，阻塞它
        等于冻结界面。计数只在申请/归还槽位许可时变更，读取无锁竞争。
        """
        with self._inflight_lock:
            return self._inflight > 0

    def request_reload(self) -> None:
        """请求在下次使用前重新加载。

        用于"参数已变但当前正在推理"的场景：立即终止服务会重置对方连接，
        而在主线程等待推理结束又可能卡住界面，因此推迟到下次使用前处理。
        """
        self._needs_reload = True

    def consume_reload_request(self) -> bool:
        """读取并清除重载请求。"""
        if not self._needs_reload:
            return False
        self._needs_reload = False
        return True

    @property
    def idle_seconds(self) -> float:
        """距**上次使用或加载**的闲置时长（秒）。

        基准必须回落到"加载时刻"而不是返回无穷大：模型刚加载完、一次推理
        都还没做过时，它显然**不是**空闲的。曾经这里在 ``_last_used`` 为 0 时
        返回 ``inf``，于是维护定时器（每 15 秒）在模型加载后立刻判定
        "空闲了无穷久"并把它卸载——而那一刻首次推理往往正在进行，
        结果是连接被重置、模型被反复加载。实测日志：

        .. code-block:: text

            20:58:00 模型空闲 infs 超过阈值 180s，自动卸载释放内存
            20:58:00 llama-server 推理失败（ConnectionResetError）
            20:58:00 正在加载模型：高精度模式 (Spark-X2.5-4B)
        """
        baseline = self._last_used or self._loaded_at
        if not baseline:
            # 既没加载过也没用过：交给调用方按"不空闲"处理更安全
            return 0.0
        return time.perf_counter() - baseline

    @property
    def port(self) -> int:
        """服务监听端口（0 表示尚未启动）。"""
        return self._port

    def status(self) -> EngineStatus:
        """返回状态快照。"""
        detail = self._unavailable_reason
        if self.is_loaded():
            detail = f"llama-server 运行中（127.0.0.1:{self._port}）"

        return EngineStatus(
            model_id=self.model_id,
            loaded=self.is_loaded(),
            available=self.is_available(),
            message=detail,
            path=str(self.model_path),
            last_latency=self.metrics.last_latency,
            total_calls=self.metrics.calls,
        )

    def estimate_memory_mb(self) -> float:
        """粗略估算加载所需内存（MB）。"""
        try:
            size = self.model_path.stat().st_size
        except OSError:
            return 0.0
        return size / (1024 * 1024) * 1.15

    def backend_description(self) -> str:
        """返回后端与加速方式的描述，供界面展示。"""
        from .backend import has_vulkan_runtime

        accelerators: list[str] = []
        if self.n_gpu_layers > 0 and self.prefer_gpu:
            accelerators.append(f"GPU 卸载 {self.n_gpu_layers} 层")
        if has_vulkan_runtime(self.executable):
            accelerators.append("Vulkan 可用")

        suffix = f"（{', '.join(accelerators)}）" if accelerators else "（CPU）"
        return f"llama-server 外部进程{suffix}"

    # ------------------------------------------------------------------
    # 进程生命周期
    # ------------------------------------------------------------------

    def load(self) -> bool:
        """启动 llama-server 子进程并等待服务就绪。"""
        with self._load_lock:
            # 参数变更有待重载时，先停掉旧进程再按新参数启动。
            #
            # 终止前必须收齐全部推理槽位许可（等价于等待所有在途请求结束）：
            # 否则可能杀掉正在处理请求的服务，对方会收到
            # ConnectionResetError（实测出现过）。
            # 这里阻塞等待是可以接受的——本方法由预加载线程或翻译线程调用，
            # 不在界面线程上。
            if self.consume_reload_request() and self.is_loaded():
                logger.info("按新的模型参数重新加载：%s", self.model_path.name)
                self._acquire_all_slots(timeout=None)
                try:
                    self._terminate()
                finally:
                    self._release_all_slots()

            if self.is_loaded():
                return True

            if not self.is_available():
                return False

            started = time.perf_counter()
            try:
                self._spawn()
                self._wait_until_ready()
            except LlamaServerError as exc:
                self._unavailable_reason = str(exc)
                logger.error("llama-server 启动失败：%s", exc)
                self._terminate()
                return False
            except Exception as exc:  # noqa: BLE001 - 启动失败原因多样
                self._unavailable_reason = f"llama-server 启动异常：{exc}"
                logger.exception("llama-server 启动异常")
                self._terminate()
                return False

            elapsed = time.perf_counter() - started
            # 记录就绪时刻：空闲时长的基准。刚加载完的模型不算空闲。
            self._loaded_at = time.perf_counter()
            self._last_used = 0.0
            logger.info(
                "llama-server 已就绪（端口 %s，耗时 %.1fs，日志 %s）",
                self._port,
                elapsed,
                self._log_path,
            )
            return True

    def _spawn(self) -> None:
        """拉起子进程。

        端口用 ``--port 0`` 让系统分配空闲端口，避免与用户机器上已有的
        服务（或本程序的另一个实例）冲突——固定端口是这类工具最常见的
        "启动失败"原因。
        """
        self._port = _pick_free_port()

        command = [
            self.executable,
            "--model", str(self.model_path),
            "--host", "127.0.0.1",
            "--port", str(self._port),
            "--ctx-size", str(self.n_ctx),
            "--n-predict", str(self.max_tokens),
            "--log-verbosity", "1",
            # FlashAttention：注意力计算省显存带宽。对 1.7B 短上下文实测
            # 收益持平（生成型负载瓶颈在输出 token），对 4B 与长提示
            # 有利无害，因此固定开启。
            "--flash-attn", "on",
            # KV 缓存复用：系统提示词每次请求都相同，开启后命中缓存段
            # 跳过重新预填充。复用窗口取 512，覆盖完整系统提示词
            # （约 400–600 token）并留出余量，跨片段前缀尽量命中。
            "--cache-reuse", "512",
        ]

        # 推理线程数。**不要占满全部核心**：OCR 与推理是流水线上相邻的两级，
        # 推理吃满 CPU 时 OCR 会明显变慢（实测整屏 OCR 从 1.0 秒涨到 5 秒以上），
        # 净效果反而更慢。自动值给 OCR 与界面留出余量。
        # 有 GPU 后端时推理主要占用显存，少量 CPU 线程即可。
        threads = self.n_threads or _default_threads()
        if threads > 0:
            command += ["--threads", str(threads)]

        # GPU 卸载层数。自动模式下给一个很大的值，让 llama.cpp 按显存容量
        # 自行决定实际卸载多少层（它会自动裁剪到能放下的层数）。
        gpu_layers = 0
        if self.prefer_gpu:
            gpu_layers = 999 if self.auto_gpu_offload else self.n_gpu_layers
        if gpu_layers > 0:
            command += ["--n-gpu-layers", str(gpu_layers)]

        # chat template：Spark-X2.5 的 GGUF 里不含模板，必须外部提供。
        template_path = _prepare_chat_template(self.chat_template)
        if template_path is not None:
            self._template_copy = template_path
            command += ["--chat-template-file", str(template_path)]
            logger.debug("使用 chat template：%s", template_path)
        else:
            logger.warning(
                "未找到 Spark-X2.5 的 chat template，模型输出可能退化为重复片段。"
                "模板应位于 assets/chat_templates/%s",
                CHAT_TEMPLATE_NAME,
            )

        # 不需要 Web UI，减少启动开销与端口暴露面
        command += ["--no-webui"]

        creationflags = 0
        startupinfo = None
        if sys.platform == "win32":
            # 不弹出控制台窗口；同时让子进程不继承我们的控制台
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        # 日志写入文件：llama.cpp 的加载信息对排查问题很有价值，
        # 但不能污染主程序的日志。
        self._log_path = _log_file_for(self.model_id)
        log_handle = open(self._log_path, "wb")  # noqa: SIM115 - 需长期持有

        logger.debug("启动 llama-server：%s", " ".join(command))
        self._process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
            startupinfo=startupinfo,
            cwd=str(Path(self.executable).parent),
            env=_child_env(),
        )
        self._log_handle = log_handle

        # 把子进程绑到"关闭即终止"的作业对象上：程序若被强制结束（任务管理器、
        # 崩溃、断电），内核会连带终止它。否则它会变成孤儿继续占着内存与显存
        # ——实测攒到过 8 个、23.3 GB，直接把整机拖垮（见 utils/process_guard）。
        assign_to_job(self._process.pid)

    def _wait_until_ready(self) -> None:
        """轮询 ``/health`` 直到服务就绪或超时。"""
        deadline = time.perf_counter() + STARTUP_TIMEOUT

        while time.perf_counter() < deadline:
            # 子进程提前退出说明加载失败（架构不支持、文件损坏、显存不足等）
            if self._process is None or self._process.poll() is not None:
                code = self._process.returncode if self._process else -1
                raise LlamaServerError(
                    f"llama-server 进程意外退出（退出码 {code}）。"
                    f"详见日志：{self._log_path}"
                )

            if self._health_ok():
                return

            time.sleep(_POLL_INTERVAL)

        raise LlamaServerError(
            f"等待 llama-server 就绪超时（{STARTUP_TIMEOUT:.0f} 秒）。"
            f"详见日志：{self._log_path}"
        )

    def _health_ok(self) -> bool:
        """探测服务是否已就绪。"""
        try:
            connection = http.client.HTTPConnection(
                "127.0.0.1", self._port, timeout=3.0
            )
            try:
                connection.request("GET", "/health")
                response = connection.getresponse()
                body = response.read()
                if response.status != 200:
                    return False
                # llama-server 就绪时返回 {"status":"ok"}
                try:
                    payload = json.loads(body.decode("utf-8", "replace"))
                    return str(payload.get("status", "")).lower() in {"ok", "no slot available"}
                except json.JSONDecodeError:
                    return True
            finally:
                connection.close()
        except (OSError, http.client.HTTPException):
            return False

    def unload(self) -> None:
        """终止子进程并释放资源。"""
        with self._load_lock:
            self._terminate()

        logger.info("llama-server 已停止：%s", self.model_path.name)

    def _acquire_all_slots(self, timeout: float | None) -> bool:
        """收齐全部推理槽位许可，即等待所有在途请求结束。

        :param timeout: 秒；``None`` 表示无限等待（参数重载路径用）。
        :return: 是否收齐；超时则归还已取得的许可并返回 ``False``。
        """
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        acquired = 0
        while acquired < self._max_concurrent:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            if not self._infer_slots.acquire(timeout=remaining):
                break
            acquired += 1
        if acquired == self._max_concurrent:
            return True
        for _ in range(acquired):
            self._infer_slots.release()
        return False

    def _release_all_slots(self) -> None:
        """归还由 :meth:`_acquire_all_slots` 收齐的全部许可。"""
        for _ in range(self._max_concurrent):
            self._infer_slots.release()

    def try_unload(self, timeout: float = 2.0) -> bool:
        """尝试卸载；收不齐推理槽位许可（正在推理）时**不**终止并返回 ``False``。

        :meth:`unload` 只取加载锁，与推理持有的槽位许可互不覆盖。空闲回收
        先用 :attr:`busy` 判定、再调用 ``unload``，两次操作之间存在窗口：
        请求可能恰好在此刻进入，进程被终止后对方收到
        ``ConnectionResetError``，那一帧翻译失败。先收齐全部许可再终止
        （与 :meth:`load` 的参数重载路径同一做法）即可关闭该窗口。
        """
        with self._load_lock:
            if not self._acquire_all_slots(timeout=max(0.0, float(timeout))):
                return False
            try:
                self._terminate()
            finally:
                self._release_all_slots()

        logger.info("llama-server 已停止：%s", self.model_path.name)
        return True

    def _terminate(self) -> None:
        """结束子进程（先温和请求，超时后强杀）。"""
        process, self._process = self._process, None
        self._port = 0

        if process is not None and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                logger.debug("llama-server 未响应终止请求，强制结束")
                try:
                    process.kill()
                    process.wait(timeout=3.0)
                except Exception:  # noqa: BLE001
                    pass
            except Exception:  # noqa: BLE001
                logger.debug("终止 llama-server 时出现异常（可忽略）")

        handle = getattr(self, "_log_handle", None)
        if handle is not None:
            with contextlib.suppress(Exception):
                handle.close()
            self._log_handle = None

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------

    def translate_batch(
        self,
        segments: list[str],
        system_prompt: str,
        max_tokens: int | None = None,
    ) -> list[str]:
        """执行一次翻译请求。

        接口与 :class:`~pingyitong.components.llm.llama_engine.LlamaEngine`
        保持一致，因此上层无需区分后端。

        :param segments: 单元素列表，元素为已拼装好的用户提示词。
        :return: 单元素结果列表；失败时返回空列表。
        """
        if not segments:
            return []

        prompt = segments[0]
        started = time.perf_counter()

        if not self.load():
            self.metrics.record(time.perf_counter() - started, success=False)
            return []

        try:
            self._infer_slots.acquire()
        except Exception:  # noqa: BLE001 - 许可申请失败等价于请求失败
            logger.exception("llama-server 推理失败")
            self.metrics.record(time.perf_counter() - started, success=False)
            return []
        try:
            with self._inflight_lock:
                self._inflight += 1
            try:
                output = self._complete(prompt, system_prompt, max_tokens)
            finally:
                with self._inflight_lock:
                    self._inflight -= 1
        except Exception:  # noqa: BLE001 - 单次推理失败不应终止流水线
            logger.exception("llama-server 推理失败")
            self.metrics.record(time.perf_counter() - started, success=False)
            return []
        finally:
            self._infer_slots.release()

        latency = time.perf_counter() - started
        self.metrics.record(latency, success=bool(output))
        self._last_used = time.perf_counter()
        return [output] if output else []

    def _complete(self, prompt: str, system_prompt: str, max_tokens: int | None) -> str:
        """调用 OpenAI 兼容的对话接口。

        用对话接口而非 ``/completion``：Spark-X2.5 的 GGUF 权重不含 chat
        template，走补全接口会让模型看到裸文本，实测输出会退化成
        ``"As As As As…"`` 这类重复片段或自问自答。对话接口会套用我们
        通过 ``--chat-template-file`` 传入的官方模板。
        """
        payload: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": max_tokens or self.max_tokens,
            "stream": False,
            # Spark-X2.5 是推理模型，默认会先输出思考过程。桌面实时翻译
            # 需要即时响应，因此默认关闭思考（实测单句延迟 0.1s vs 0.6s）。
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        connection = http.client.HTTPConnection("127.0.0.1", self._port, timeout=REQUEST_TIMEOUT)
        try:
            connection.request(
                "POST",
                "/v1/chat/completions",
                body=body,
                headers={"Content-Type": "application/json; charset=utf-8"},
            )
            response = connection.getresponse()
            raw = response.read()
        finally:
            connection.close()

        if response.status != 200:
            detail = raw.decode("utf-8", "replace")[:200]
            raise LlamaServerError(f"llama-server 返回 HTTP {response.status}：{detail}")

        try:
            result: dict[str, Any] = json.loads(raw.decode("utf-8", "replace"))
        except json.JSONDecodeError as exc:
            raise LlamaServerError(f"无法解析 llama-server 响应：{exc}") from exc

        choices = result.get("choices") or []
        if not choices:
            return ""

        if choices[0].get("finish_reason") == "length":
            # A plausible half-sentence passes language detection but changes
            # the source meaning. Preserve the real screen and retry next frame.
            logger.warning("本地模型输出被截断，本次不使用不完整译文")
            return ""
        message = choices[0].get("message") or {}
        from .api_engine import strip_thinking

        # Reasoning is not a final answer, even when written in the target
        # language. Returning it would leak internal analysis into the overlay
        # and poison the persistent translation cache.
        return strip_thinking(str(message.get("content") or "")).strip()

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        """执行一次通用生成，返回模型原始输出。

        与 :meth:`translate_batch` 的区别：这里直接接收 system/user 两段文本，
        不经过翻译批量逻辑。供语言能力实测、Prompt 调试等场景使用。
        """
        results = self.translate_batch([user_prompt], system_prompt, max_tokens)
        return results[0] if results else ""

    def warm_up(self) -> bool:
        """预热：启动服务并**执行一次极短推理**。

        只 ``load()`` 是不够的：模型加载完成后，第一次真正的前向计算仍要
        付出可观的固定开销（图构建、KV 缓存分配、后端内核初始化）。实测
        首次翻译要 11–21 秒，而之后每句只要 0.06–0.11 秒。

        若不预热，这笔开销会落在用户点"开始翻译"之后的第一帧上——
        用户会以为程序卡死了。放到启动预加载阶段执行，用户就感受不到。
        """
        if not self.load():
            return False

        try:
            # 极短输入 + 极小输出：目的是触发一次完整前向计算，
            # 而不是真的生成内容。
            self._complete(".", "Reply with a single dot.", max_tokens=8)
        except Exception:  # noqa: BLE001 - 预热失败不影响正常使用
            logger.debug("预热推理失败（不影响后续使用）")
            return False
        return True


# ----------------------------------------------------------------------
# 辅助函数
# ----------------------------------------------------------------------


def _child_env() -> dict[str, str]:
    """构造子进程环境变量。

    多 GPU 的机器上（例如独立显卡 + 核显），llama.cpp 的 Vulkan 后端会把
    模型层**拆分到所有设备**；核显（UMA 共享内存）远慢于独立显卡，拆分
    只会让解码变慢。实测限制到设备 0（几乎总是独立显卡）后，1.7B 解码
    由 235 t/s 升到 274 t/s，且延迟抖动明显变小。

    只在用户没有显式设置该变量时才写入，给特殊机器留出覆盖口子。
    """
    env = os.environ.copy()
    env.setdefault("GGML_VK_VISIBLE_DEVICES", "0")
    return env


def _default_threads() -> int:
    """返回推理的默认线程数。

    取 ``max(4, 物理核数 // 3)``，而不是全部核心：OCR 与推理在流水线上
    紧邻，推理占满 CPU 会让 OCR 明显变慢（实测整屏 OCR 由 1.0 秒涨到 5 秒
    以上），总延迟反而恶化。留出余量给 OCR 与界面后整体更快。
    """
    try:
        import psutil

        cores = psutil.cpu_count(logical=False) or 4
    except Exception:  # noqa: BLE001 - psutil 缺失时按逻辑核数估算
        import os

        cores = max(4, (os.cpu_count() or 8) // 2)

    return max(4, int(cores) // 3)


def _pick_free_port() -> int:
    """申请一个空闲端口。

    llama-server 需要显式端口号（``--port 0`` 不可靠），因此这里先绑定
    0 号端口取得系统分配的空闲端口号，再立即释放交给它使用。
    存在极小的竞态窗口，但比固定端口可靠得多——固定端口是这类工具
    "启动失败"最常见的原因。
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def default_chat_template() -> Path | None:
    """返回项目随附的 Spark-X2.5 chat template 路径。

    Spark-X2.5 的 GGUF 权重里**不含** ``tokenizer.chat_template``，必须由
    调用方提供，否则模型输出会退化成重复片段。
    """
    from ...utils import paths

    for candidate in (
        paths.asset_path("chat_templates", CHAT_TEMPLATE_NAME),
        paths.project_root() / "assets" / "chat_templates" / CHAT_TEMPLATE_NAME,
    ):
        if candidate.is_file():
            return candidate
    return None


def _prepare_chat_template(template: Path | None) -> Path | None:
    """把 chat template 复制到一个**纯 ASCII 路径**并返回它。

    为什么必须这样做：llama.cpp 读取 chat template 文件时使用窄字符
    （ANSI）路径，在中文 Windows 上无法打开含中文的目录——实测报
    ``failed to open file 'F:\\讯飞\\项目源码\\assets\\...'``。
    而模型权重走的加载路径使用宽字符，所以同样是中文路径却能正常工作。

    复制到临时目录可以无副作用地绕开这个限制。注意**目录名与文件名都必须
    是 ASCII**——llama.cpp 读取的是完整路径，只把目录换成英文而保留中文
    文件名同样会失败。因此这里固定使用英文文件名。
    """
    if template is None or not template.is_file():
        return None

    try:
        str(template).encode("ascii")
        return template  # 完整路径已是 ASCII，无需拷贝
    except UnicodeEncodeError:
        pass

    try:
        target_dir = Path(tempfile.gettempdir()) / "pingyitong"
        target_dir.mkdir(parents=True, exist_ok=True)
        # 文件名也必须是 ASCII，因此不使用 template.name
        target = target_dir / "spark-x2.5-chat-template.jinja"
        shutil.copyfile(template, target)
        return target
    except OSError as exc:
        logger.warning("无法复制 chat template 到临时目录（%s），将尝试直接使用原路径", exc)
        return template


def _log_file_for(model_id: str) -> Path:
    """返回该模型对应的 llama-server 日志文件路径。"""
    from ...utils import paths

    directory = paths.log_dir() / "llama-server"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{model_id}.log"


def is_running(engine: object) -> bool:
    """判断给定引擎是否为运行中的 llama-server（供测试与诊断使用）。"""
    return isinstance(engine, LlamaServerEngine) and engine.is_loaded()


def kill_orphans() -> int:
    """清理可能残留的 llama-server 进程。

    程序异常退出（例如被任务管理器结束）时子进程可能残留。
    启动时调用一次可以避免残留进程继续占用内存。

    :return: 是否结束了残留进程（1 是 / 0 否；taskkill 按映像名整体
        结束，无法给出精确个数）。启动路径已改用 utils.reap_stale_servers，
        本函数保留供外部脚本调用。
    """
    if sys.platform != "win32":
        return 0

    try:
        result = subprocess.run(
            ["taskkill", "/F", "/IM", "llama-server.exe"],
            capture_output=True,
            timeout=15,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception:  # noqa: BLE001
        return 0

    if result.returncode == 0:
        logger.info("已清理残留的 llama-server 进程")
        return 1
    return 0
