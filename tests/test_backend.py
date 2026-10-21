"""推理后端选择测试。

Spark-X2.5 的 ``spark2_5`` 架构只有 XHToken 维护的 llama.cpp 分支支持，
上游 llama.cpp（``llama-cpp-python`` 捆绑的那个）不认识它。因此后端选择
不是"优化"，而是**能不能跑起来**的分水岭：

- 选错后端 → 模型加载失败 → 用户看到预览模式而不是翻译；
- 探测顺序写反 → 明明装好了运行时却仍走进程内后端 → 同样失败。

本文件覆盖探测顺序、可执行文件查找、路径容错与降级行为。
"""

from __future__ import annotations

import contextlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pingyitong.components.llm import backend as backend_module
from pingyitong.components.llm.backend import BackendKind
from pingyitong.utils import paths


@contextlib.contextmanager
def fake_inflight(engine):
    """伪造一个在途推理请求。

    占住一个推理槽位许可并登记在途计数，等价于真实 ``translate_batch``
    在 ``_complete`` 期间持有的资源：``busy`` 应为真，``try_unload``
    应因收不齐全部许可而拒绝终止。
    """
    engine._infer_slots.acquire()
    with engine._inflight_lock:
        engine._inflight += 1
    try:
        yield
    finally:
        with engine._inflight_lock:
            engine._inflight -= 1
        engine._infer_slots.release()


class TestBackendProbe(unittest.TestCase):
    """后端探测与优先级。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.home = self.root / "home"
        self.home.mkdir()
        self._env = mock.patch.dict(os.environ, {"PINGYITONG_HOME": str(self.home)})
        self._env.start()
        # 环境变量会干扰探测，先清掉
        os.environ.pop("PINGYITONG_LLAMA_SERVER", None)

    def tearDown(self) -> None:
        self._env.stop()
        self._tmp.cleanup()

    def _fake_server(self) -> Path:
        """造一个假的 llama-server 可执行文件。"""
        server = self.root / "llama-server.exe"
        server.write_bytes(b"MZ-fake")
        return server

    def test_server_preferred_over_python_binding(self) -> None:
        """两者都可用时应优先选择 llama-server。

        它是 Spark-X2.5 唯一可行的方式；进程内后端只适用于架构受上游
        支持的其它模型。
        """
        server = self._fake_server()
        with mock.patch.object(
            backend_module, "discover_server_executable", return_value=str(server)
        ), mock.patch(
            "pingyitong.components.llm.llama_engine.llama_cpp_available",
            return_value=True,
        ):
            probe = backend_module.probe()
        self.assertIs(probe.kind, BackendKind.LLAMA_SERVER)
        self.assertTrue(probe.usable)

    def test_falls_back_to_python_binding(self) -> None:
        """没有 llama-server 时回退到进程内后端。"""
        with mock.patch.object(
            backend_module, "discover_server_executable", return_value=""
        ), mock.patch(
            "pingyitong.components.llm.llama_engine.llama_cpp_available",
            return_value=True,
        ):
            probe = backend_module.probe()
        self.assertIs(probe.kind, BackendKind.LLAMA_CPP_PYTHON)
        self.assertTrue(probe.usable)

    def test_none_when_nothing_available(self) -> None:
        """两者都不可用时应报告为不可用，并给出可操作的提示。"""
        with mock.patch.object(
            backend_module, "discover_server_executable", return_value=""
        ), mock.patch(
            "pingyitong.components.llm.llama_engine.llama_cpp_available",
            return_value=False,
        ):
            probe = backend_module.probe()
        self.assertIs(probe.kind, BackendKind.NONE)
        self.assertFalse(probe.usable)
        self.assertIn("download_runtime", probe.detail)

    def test_server_can_be_disabled(self) -> None:
        """用户禁用 llama-server 后不应再探测它。"""
        server = self._fake_server()
        with mock.patch.object(
            backend_module, "discover_server_executable", return_value=str(server)
        ), mock.patch(
            "pingyitong.components.llm.llama_engine.llama_cpp_available",
            return_value=True,
        ):
            probe = backend_module.probe(allow_server=False)
        self.assertIs(probe.kind, BackendKind.LLAMA_CPP_PYTHON)

    def test_probe_reports_executable_path(self) -> None:
        """探测结果应带上可执行文件路径，便于界面展示与排查。"""
        server = self._fake_server()
        with mock.patch.object(
            backend_module, "discover_server_executable", return_value=str(server)
        ):
            probe = backend_module.probe()
        self.assertEqual(probe.executable, str(server))
        self.assertIn(str(server), probe.detail)

    def test_backend_labels_distinct(self) -> None:
        """三种后端的展示名称应互不相同（界面用它告诉用户当前状态）。"""
        labels = {kind.label for kind in BackendKind}
        self.assertEqual(len(labels), len(list(BackendKind)))


class TestServerDiscovery(unittest.TestCase):
    """llama-server 可执行文件查找。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        os.environ.pop("PINGYITONG_LLAMA_SERVER", None)

    def tearDown(self) -> None:
        os.environ.pop("PINGYITONG_LLAMA_SERVER", None)
        self._tmp.cleanup()

    def test_env_override_wins(self) -> None:
        """环境变量指定的路径优先级最高。"""
        server = self.root / "custom-server.exe"
        server.write_bytes(b"MZ")
        with mock.patch.dict(os.environ, {"PINGYITONG_LLAMA_SERVER": str(server)}):
            self.assertEqual(
                backend_module.discover_server_executable(), str(server)
            )

    def test_env_override_ignored_when_missing(self) -> None:
        """环境变量指向不存在的文件时应忽略它并继续查找。"""
        with mock.patch.dict(
            os.environ, {"PINGYITONG_LLAMA_SERVER": str(self.root / "没有这个.exe")}
        ):
            found = backend_module.discover_server_executable()
        # 不应返回那个不存在的路径
        self.assertNotIn("没有这个.exe", found)

    def test_project_runtime_dir_is_searched(self) -> None:
        """项目内 runtime/llama/ 应在查找范围内。

        这是 ``scripts/download_runtime.py`` 的默认安装位置。
        """
        candidates = paths.llama_runtime_candidates()
        self.assertTrue(
            any("runtime" in str(c) and "llama" in str(c) for c in candidates),
            f"候选路径中应包含 runtime/llama：{candidates}",
        )

    def test_returns_empty_when_not_found(self) -> None:
        """找不到时返回空字符串而不是抛异常。"""
        with mock.patch.object(paths, "llama_runtime_candidates", return_value=[]), \
             mock.patch("shutil.which", return_value=None):
            self.assertEqual(backend_module.discover_server_executable(), "")

    def test_runtime_dir_creatable(self) -> None:
        """运行时目录应能按需创建。"""
        directory = paths.llama_runtime_dir()
        self.assertTrue(directory.exists())


class TestGpuBackendDetection(unittest.TestCase):
    """GPU 后端探测。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self._tmp.name)
        self.server = self.directory / "llama-server.exe"
        self.server.write_bytes(b"MZ")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_cuda_detected_by_dll(self) -> None:
        """同目录下有 CUDA 运行时 DLL 时应判定为可用。"""
        (self.directory / "cudart64_12.dll").write_bytes(b"")
        self.assertTrue(backend_module._has_cuda_runtime(str(self.server)))

    def test_cuda_absent(self) -> None:
        """没有 CUDA DLL 时不应误报为可用。"""
        with mock.patch.dict(os.environ, {"CUDA_PATH": ""}):
            self.assertFalse(backend_module._has_cuda_runtime(str(self.server)))

    def test_vulkan_detected_by_dll(self) -> None:
        """同目录下有 Vulkan 后端时应判定为可用。

        对 Blackwell 等新架构显卡，Vulkan 是 CUDA 之外的有效加速途径。
        """
        (self.directory / "ggml-vulkan.dll").write_bytes(b"")
        self.assertTrue(backend_module.has_vulkan_runtime(str(self.server)))

    def test_vulkan_absent(self) -> None:
        """没有 Vulkan 后端时不应误报。"""
        self.assertFalse(backend_module.has_vulkan_runtime(str(self.server)))

    def test_empty_path_is_safe(self) -> None:
        """空路径不应抛异常（未找到可执行文件时的常见状态）。"""
        self.assertFalse(backend_module._has_cuda_runtime(""))
        self.assertFalse(backend_module.has_vulkan_runtime(""))


class TestChatTemplateHandling(unittest.TestCase):
    """chat template 的准备。

    Spark-X2.5 的 GGUF 权重**不含** ``tokenizer.chat_template``，必须外部
    提供，否则模型输出会退化成 ``"As As As As…"`` 这类重复片段（实测）。
    另外 llama.cpp 读取模板文件用的是窄字符路径，无法打开中文路径，
    因此需要复制到纯 ASCII 目录。
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_default_template_exists_in_project(self) -> None:
        """项目内应随附官方 chat template。"""
        from pingyitong.components.llm.llama_server import default_chat_template

        template = default_chat_template()
        self.assertIsNotNone(template, "未找到 assets/chat_templates/spark-x2.5.jinja")
        assert template is not None
        self.assertTrue(template.is_file())
        # 模板应包含模型特有的对话标记，而不只是一个空文件
        content = template.read_text(encoding="utf-8", errors="replace")
        self.assertIn("<|System|>", content)
        self.assertIn("enable_thinking", content)

    def test_ascii_path_used_directly(self) -> None:
        """路径已是纯 ASCII 时不复制，直接用原路径。"""
        from pingyitong.components.llm.llama_server import _prepare_chat_template

        source = self.root / "ascii.jinja"
        source.write_text("{{ prompt }}", encoding="utf-8")
        self.assertEqual(_prepare_chat_template(source), source)

    def test_non_ascii_path_is_copied(self) -> None:
        """非 ASCII 路径应被复制到纯 ASCII 的临时位置。

        这是实测踩到的坑：中文目录下 llama-server 报
        ``failed to open file``，而模型权重却是同一个中文目录且加载正常
        ——因为模型加载走宽字符路径，模板读取走窄字符路径。
        """
        from pingyitong.components.llm.llama_server import _prepare_chat_template

        non_ascii_dir = self.root / "中文目录"
        non_ascii_dir.mkdir()
        source = non_ascii_dir / "模板.jinja"
        source.write_text("{{ prompt }}", encoding="utf-8")

        prepared = _prepare_chat_template(source)
        self.assertIsNotNone(prepared)
        assert prepared is not None
        self.assertIsNot(prepared, source)
        # 结果路径必须能被 ASCII 编码（llama.cpp 的前提）
        prepared_str = str(prepared)
        prepared_str.encode("ascii")  # 不抛异常即通过
        self.assertTrue(prepared.is_file())
        self.assertEqual(prepared.read_text(encoding="utf-8"), "{{ prompt }}")

    def test_missing_template_returns_none(self) -> None:
        """模板不存在时返回 None（调用方据此给出提示）。"""
        from pingyitong.components.llm.llama_server import _prepare_chat_template

        self.assertIsNone(_prepare_chat_template(None))
        self.assertIsNone(_prepare_chat_template(self.root / "不存在.jinja"))


class TestEngineConfiguration(unittest.TestCase):
    """引擎构造参数。"""

    def test_thinking_disabled_by_default(self) -> None:
        """思考模式默认关闭。

        桌面实时翻译需要即时响应：实测同一句话关闭思考约 0.1 秒、
        开启约 0.6 秒。默认值的选择直接影响产品可用性。
        """
        from pingyitong.config.schema import ModelConfig

        self.assertFalse(ModelConfig().enable_thinking)

    def test_server_backend_preferred_by_default(self) -> None:
        """默认优先使用 llama-server 后端。"""
        from pingyitong.config.schema import ModelConfig

        self.assertTrue(ModelConfig().prefer_server_backend)

    def test_engine_accepts_thinking_flag(self) -> None:
        """引擎应接受并保存思考开关。"""
        from pingyitong.components.llm.llama_server import LlamaServerEngine

        engine = LlamaServerEngine(
            model_id="spark-x2.5-1.7b",
            model_path=Path("不存在.gguf"),
            executable="llama-server.exe",
            enable_thinking=True,
        )
        self.assertTrue(engine.enable_thinking)

    def test_engine_reports_unavailable_when_files_missing(self) -> None:
        """权重或可执行文件缺失时应报告不可用并说明原因。"""
        from pingyitong.components.llm.llama_server import LlamaServerEngine

        engine = LlamaServerEngine(
            model_id="spark-x2.5-1.7b",
            model_path=Path("不存在的模型.gguf"),
            executable="不存在的可执行文件.exe",
        )
        self.assertFalse(engine.is_available())
        self.assertTrue(engine.unavailable_reason)
        status = engine.status()
        self.assertFalse(status.available)
        self.assertTrue(status.message)

    def test_free_port_is_in_valid_range(self) -> None:
        """端口探测应返回合法端口号。"""
        from pingyitong.components.llm.llama_server import _pick_free_port

        port = _pick_free_port()
        self.assertGreater(port, 1024)
        self.assertLess(port, 65536)


class TestDeferredReload(unittest.TestCase):
    """推理进行中时的参数变更处理。

    参数变更需要重新加载模型才能生效，但**不能在推理进行中强行终止服务**：
    那会让对方的 HTTP 连接被重置（实测日志出现过 ``ConnectionResetError``）。
    而换成"在主线程等推理结束"更糟——等一次推理最多 60 秒，界面会冻住。

    因此约定：忙时只打标记，由下次 ``load()`` 前按新参数重载。
    """

    def _engine(self):
        from pathlib import Path

        from pingyitong.components.llm.llama_server import LlamaServerEngine

        return LlamaServerEngine(
            model_id="spark-x2.5-1.7b",
            model_path=Path("不存在.gguf"),
            executable="llama-server.exe",
        )

    def test_not_busy_when_idle(self) -> None:
        """空闲时不应报告忙碌。"""
        engine = self._engine()
        self.assertFalse(engine.busy)

    def test_busy_when_lock_held(self) -> None:
        """在途请求存在时应报告忙碌。"""
        engine = self._engine()
        with fake_inflight(engine):
            self.assertTrue(engine.busy)
        self.assertFalse(engine.busy)

    def test_busy_check_does_not_block(self) -> None:
        """忙碌检测本身不得阻塞（它在界面线程被调用）。"""
        import time as time_module

        engine = self._engine()
        with fake_inflight(engine):
            started = time_module.monotonic()
            _ = engine.busy
            elapsed = time_module.monotonic() - started
        self.assertLess(elapsed, 0.5, "busy 属性阻塞了调用方")

    def test_reload_request_roundtrip(self) -> None:
        """重载请求应能被置位并消费一次。"""
        engine = self._engine()
        self.assertFalse(engine.consume_reload_request())
        engine.request_reload()
        self.assertTrue(engine.consume_reload_request())
        # 消费后应清零，避免每次加载都重载
        self.assertFalse(engine.consume_reload_request())

    def test_reload_request_default_off(self) -> None:
        """新建引擎不应带着重载标记。"""
        from pathlib import Path

        from pingyitong.components.llm.llama_server import LlamaServerEngine

        engine = LlamaServerEngine(
            model_id="m", model_path=Path("x.gguf"), executable="y.exe"
        )
        self.assertFalse(engine._needs_reload)

    def test_pool_skips_unload_while_busy(self) -> None:
        """模型池在推理进行中不应强行卸载引擎。

        这条断言直接对应实测的 ``ConnectionResetError``：界面切模型时
        终止了正在服务的 llama-server，导致那一帧翻译失败。
        """
        from pingyitong.components.llm.model_pool import ModelPool
        from pingyitong.config.schema import AppConfig

        config = AppConfig()
        pool = ModelPool(config.model)
        engine = pool.engine("spark-x2.5-1.7b")

        # 伪造"已加载且正在推理"的状态
        engine._process = mock.MagicMock()
        engine._process.poll.return_value = None  # 进程存活
        with fake_inflight(engine):
            changed = AppConfig()
            changed.model.fast.n_ctx = (engine.n_ctx or 0) + 1024  # 制造参数变更
            pool.apply_config(changed.model)

            # 关键一：服务未被终止（进程仍存活），因此不会有连接重置
            self.assertTrue(engine.is_loaded(), "推理进行中不应终止服务")
            # 关键二：已登记重载请求，供下次使用前按新参数重建
            self.assertTrue(
                engine.consume_reload_request(), "忙时应登记重载请求而非立即卸载"
            )
        engine._process = None

    def test_pool_unloads_when_idle(self) -> None:
        """空闲时的参数变更应立即卸载，不必等到下次使用。"""
        from pingyitong.components.llm.model_pool import ModelPool
        from pingyitong.config.schema import AppConfig

        config = AppConfig()
        pool = ModelPool(config.model)
        engine = pool.engine("spark-x2.5-1.7b")

        engine._process = mock.MagicMock()
        engine._process.poll.return_value = None
        engine._process.terminate.return_value = None
        engine._process.wait.return_value = 0

        changed = AppConfig()
        changed.model.fast.n_ctx = (engine.n_ctx or 0) + 1024
        pool.apply_config(changed.model)

        self.assertFalse(engine.is_loaded(), "空闲时应直接卸载以便按新参数重载")
        engine._process = None


class TestGpuLayerConfigComparison(unittest.TestCase):
    """GPU 卸载层数的"配置值 vs 生效值"。

    实测踩过一个坑：自动模式下把**生效值 999** 写进了引擎的 ``n_gpu_layers``，
    而 ``ModelPool.apply_config`` 用它与新配置的 ``0`` 比较来判断参数是否变更，
    于是永远判定"变了" → 每次配置变更都卸载重载模型。日志表现为
    llama-server 反复"已停止 → 已就绪 → 已停止"。

    因此约定：``n_gpu_layers`` 只保存**配置值**，自动卸载通过
    ``auto_gpu_offload`` 单独表达。
    """

    def _engine(self, **kwargs):
        from pathlib import Path

        from pingyitong.components.llm.llama_server import LlamaServerEngine

        params = {
            "model_id": "spark-x2.5-1.7b",
            "model_path": Path("不存在.gguf"),
            "executable": "llama-server.exe",
        }
        params.update(kwargs)
        return LlamaServerEngine(**params)

    def test_configured_layers_stored_verbatim(self) -> None:
        """配置的层数应原样保存，不被自动值覆盖。"""
        engine = self._engine(n_gpu_layers=0, auto_gpu_offload=True)
        self.assertEqual(engine.n_gpu_layers, 0, "n_gpu_layers 必须是配置值")
        self.assertTrue(engine.auto_gpu_offload)

    def test_auto_offload_does_not_alter_comparison(self) -> None:
        """自动卸载不应让"参数未变"被误判为"已变"。

        这正是模型反复重载的成因，必须固定住。
        """
        from pingyitong.config.schema import AppConfig

        engine = self._engine(n_gpu_layers=0, auto_gpu_offload=True)
        config = AppConfig().model.for_model("spark-x2.5-1.7b")

        # 用 apply_config 的比较条件逐一核对
        self.assertEqual(engine.n_ctx, config.n_ctx)
        self.assertEqual(engine.n_gpu_layers, config.n_gpu_layers)
        self.assertEqual(engine.n_threads, config.n_threads)

    def test_explicit_layers_preserved(self) -> None:
        """用户显式指定层数时不启用自动模式。"""
        engine = self._engine(n_gpu_layers=20, auto_gpu_offload=False)
        self.assertEqual(engine.n_gpu_layers, 20)
        self.assertFalse(engine.auto_gpu_offload)

    def test_pool_passes_configured_value(self) -> None:
        """模型池创建的引擎应保存配置值而非生效值。

        这条断言直接对应实测：有 GPU 时配置值是 0，引擎里不能变成 999。
        """
        from pingyitong.components.llm.model_pool import ModelPool
        from pingyitong.config.schema import AppConfig

        config = AppConfig()
        pool = ModelPool(config.model)
        engine = pool.engine("spark-x2.5-1.7b")

        model_config = config.model.for_model("spark-x2.5-1.7b")
        self.assertEqual(
            getattr(engine, "n_gpu_layers", None),
            model_config.n_gpu_layers,
            "引擎保存的 GPU 层数必须与配置一致，否则 apply_config 会反复重载模型",
        )

    def test_repeated_apply_config_is_noop(self) -> None:
        """连续多次应用相同配置不应触发任何重载。

        回归防线：若比较条件被改坏，这里会立刻暴露。
        """
        from pingyitong.components.llm.model_pool import ModelPool
        from pingyitong.config.schema import AppConfig

        config = AppConfig()
        pool = ModelPool(config.model)
        engine = pool.engine("spark-x2.5-1.7b")

        # 伪造"已加载"
        engine._process = mock.MagicMock()
        engine._process.poll.return_value = None
        engine._process.terminate.return_value = None
        engine._process.wait.return_value = 0

        for _ in range(3):
            pool.apply_config(config.model)

        self.assertTrue(engine.is_loaded(), "参数未变时不应卸载模型")
        self.assertFalse(engine.consume_reload_request(), "参数未变时不应登记重载")
        engine._process = None


class TestIdleBaseline(unittest.TestCase):
    """空闲时长的基准。

    实测严重缺陷：``idle_seconds`` 在"模型已加载但从未推理过"时返回 ``inf``，
    于是维护定时器（每 15 秒一次）在模型加载完成后**立刻**判定它空闲超时
    并卸载——而那一刻首次推理往往正在进行，日志表现为：

    .. code-block:: text

        模型空闲 infs 超过阈值 180s，自动卸载释放内存
        llama-server 推理失败（ConnectionResetError）
        正在加载模型：高精度模式 (Spark-X2.5-4B)

    结果是 4B 模型几乎无法使用：每次加载完就被杀、然后又要重载。
    正确语义是"刚加载完的模型不算空闲"。
    """

    def _server_engine(self):
        from pathlib import Path

        from pingyitong.components.llm.llama_server import LlamaServerEngine

        return LlamaServerEngine(
            model_id="m", model_path=Path("x.gguf"), executable="y.exe"
        )

    def _python_engine(self):
        from pathlib import Path

        from pingyitong.components.llm.llama_engine import LlamaEngine

        return LlamaEngine(model_id="m", model_path=Path("x.gguf"))

    def test_never_returns_infinity(self) -> None:
        """全新引擎的空闲时长不得为无穷大。

        这是缺陷的直接表现：``inf`` 会让任何空闲阈值判定立即成立。
        """
        for label, engine in (
            ("llama-server", self._server_engine()),
            ("llama-cpp-python", self._python_engine()),
        ):
            with self.subTest(backend=label):
                idle = engine.idle_seconds
                self.assertNotEqual(idle, float("inf"), "空闲时长不应为无穷大")
                self.assertGreaterEqual(idle, 0.0)

    def test_just_loaded_model_is_not_idle(self) -> None:
        """刚加载完的模型不应被判定为空闲超时。"""
        engine = self._server_engine()
        engine._loaded_at = 1000.0
        with mock.patch(
            "pingyitong.components.llm.llama_server.time.perf_counter",
            return_value=1000.0,
        ):
            self.assertEqual(engine.idle_seconds, 0.0)

    def test_idle_grows_after_load(self) -> None:
        """加载后经过的时间应计入空闲时长。"""
        engine = self._server_engine()
        engine._loaded_at = 970.0
        with mock.patch(
            "pingyitong.components.llm.llama_server.time.perf_counter",
            return_value=1000.0,
        ):
            self.assertEqual(engine.idle_seconds, 30.0)

    def test_last_used_takes_precedence(self) -> None:
        """推理过之后应以"上次使用"为基准，而不是加载时刻。"""
        engine = self._server_engine()
        engine._loaded_at = 400.0  # 很久前加载
        engine._last_used = 995.0  # 5 秒前用过
        with mock.patch(
            "pingyitong.components.llm.llama_server.time.perf_counter",
            return_value=1000.0,
        ):
            self.assertEqual(engine.idle_seconds, 5.0)

    def test_pool_skips_unload_while_busy(self) -> None:
        """正在推理的模型不应被空闲回收卸载。

        终止服务会重置对方连接，那一帧翻译直接失败。
        """
        from pingyitong.components.llm.model_pool import ModelPool
        from pingyitong.config.schema import AppConfig

        config = AppConfig()
        # 把 4B 设为非常驻，使它能进入空闲回收逻辑
        pool = ModelPool(config.model, idle_timeout=0.0)
        engine = pool.engine("spark-x2.5-4b")

        engine._process = mock.MagicMock()
        engine._process.poll.return_value = None
        engine._loaded_at = 0.0  # 伪造"很久没动"
        engine._last_used = 0.0
        with fake_inflight(engine):  # 伪造"正在推理"
            unloaded = pool.unload_idle()
            self.assertNotIn("spark-x2.5-4b", unloaded, "推理中不应被卸载")
        engine._process = None


class TestChildEnv(unittest.TestCase):
    """子进程环境：限制 Vulkan 设备枚举，避免核显拆层拖慢解码。"""

    def test_defaults_to_device_zero(self) -> None:
        import os
        from unittest import mock

        from pingyitong.components.llm.llama_server import _child_env

        clean = {k: v for k, v in os.environ.items() if k != "GGML_VK_VISIBLE_DEVICES"}
        with mock.patch.dict(os.environ, clean, clear=True):
            self.assertEqual(_child_env()["GGML_VK_VISIBLE_DEVICES"], "0")

    def test_respects_user_override(self) -> None:
        import os
        from unittest import mock

        from pingyitong.components.llm.llama_server import _child_env

        with mock.patch.dict(
            os.environ, {"GGML_VK_VISIBLE_DEVICES": "1"}, clear=True
        ):
            self.assertEqual(_child_env()["GGML_VK_VISIBLE_DEVICES"], "1")

    def test_env_is_a_copy(self) -> None:
        from pingyitong.components.llm.llama_server import _child_env

        env = _child_env()
        env["GGML_VK_VISIBLE_DEVICES"] = "9"
        self.assertNotEqual(_child_env().get("GGML_VK_VISIBLE_DEVICES"), "9")


if __name__ == "__main__":
    unittest.main(verbosity=2)
