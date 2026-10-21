"""模型权重路径解析与预加载测试。

对应的需求："把模型权重加入进去，程序开启时自动加载"。

这里要保证的核心行为：

1. **多位置查找**——权重可能放在项目内的 ``models/``（源码运行场景）
   或应用数据目录（安装场景），程序应两者都能找到，而不是要求
   用户把文件搬到某个固定位置；
2. **文件名容错**——官方发布的文件名大小写与配置不同时仍能识别；
   从 Ollama / LM Studio 下载的文件常带额外后缀，也不应因此加载失败；
3. **缺失时如实报告**——找不到权重时返回"期望路径"而非莫须有的路径，
   界面据此提示"应该放到哪里"；
4. **预加载只针对真实存在的权重**——权重不在时不应无意义地等待
   一个必然失败的加载；
5. **预加载不阻塞主线程**——界面启动路径必须走异步接口。
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pingyitong.components.llm.model_pool import ModelPool, PreloadProgress
from pingyitong.config.schema import AppConfig
from pingyitong.constants import MODEL_ACCURATE, MODEL_FAST, MODEL_FILE_NAMES
from pingyitong.utils import paths


class ModelPathsTestCase(unittest.TestCase):
    """提供完全隔离的模型查找环境。

    **必须接管查找目录列表**，否则测试结果会取决于开发机上是否已经下载了
    权重：交付目录里放了 3.45GB 的权重后，所有"应当找不到"的断言都会失败。
    这类依赖机器状态的测试是最糟糕的——CI 通过、开发机失败，反之亦然。

    因此这里直接替换 :func:`paths.models_search_dirs`，让查找范围完全由
    测试决定；真实查找顺序的验证放在 :class:`TestSearchDirectories` 中单独进行。
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

        self.models_dir = self.root / "project_models"
        self.appdata_dir = self.root / "appdata_models"
        self.home_dir = self.root / "home"
        for directory in (self.models_dir, self.appdata_dir, self.home_dir):
            directory.mkdir(parents=True)

        self._env = mock.patch.dict(
            os.environ, {"PINGYITONG_HOME": str(self.home_dir)}
        )
        self._env.start()

        self._dirs = mock.patch.object(
            paths,
            "models_search_dirs",
            return_value=[self.models_dir, self.appdata_dir],
        )
        self._dirs.start()

    def tearDown(self) -> None:
        self._dirs.stop()
        self._env.stop()
        self._tmp.cleanup()

    def touch_model(self, name: str, directory: Path | None = None) -> Path:
        """创建一个占位的模型文件（内容无关，只测路径解析）。"""
        target = (directory or self.models_dir) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"GGUF-placeholder")
        return target


class TestResolveModelPath(ModelPathsTestCase):
    """模型文件解析。"""

    def test_exact_name_match(self) -> None:
        """文件名与配置完全一致时直接命中。"""
        expected = self.touch_model(MODEL_FILE_NAMES[MODEL_FAST])
        self.assertEqual(paths.resolve_model_path(MODEL_FILE_NAMES[MODEL_FAST]), expected)

    def test_case_insensitive_match(self) -> None:
        """官方文件名大小写与配置不同时仍应命中。

        Hugging Face 上发布的是 ``Spark-X2.5-1.7B-Q4_K_M.gguf``，
        而用户在 Windows 上可能把文件改成小写形式；二者应等价。
        """
        lower = MODEL_FILE_NAMES[MODEL_FAST].lower()
        expected = self.touch_model(lower)
        self.assertEqual(paths.resolve_model_path(MODEL_FILE_NAMES[MODEL_FAST]), expected)

    def test_fuzzy_match_by_parameter_size(self) -> None:
        """文件名带额外后缀时按参数量特征匹配。

        从 Ollama / LM Studio 下载的文件常形如
        ``spark-x2.5-1.7b-q4_k_m-00001-of-00001.gguf``，
        不应因为名字不完全一致就加载失败。
        """
        expected = self.touch_model(
            "spark-x2.5-1.7b-q4_k_m-00001-of-00001.gguf"
        )
        self.assertEqual(paths.resolve_model_path(MODEL_FILE_NAMES[MODEL_FAST]), expected)

    def test_fuzzy_match_does_not_confuse_sizes(self) -> None:
        """特征匹配不得把 4B 的文件当成 1.7B。

        参数量标识不同，混用会导致加载错误的权重或内存耗尽。
        """
        self.touch_model("spark-x2.5-4b-q4_k_m-00001-of-00001.gguf")
        resolved = paths.resolve_model_path(MODEL_FILE_NAMES[MODEL_FAST])
        self.assertFalse(resolved.is_file())

    def test_exact_match_preferred_over_fuzzy(self) -> None:
        """精确匹配优先于模糊匹配。"""
        exact = self.touch_model(MODEL_FILE_NAMES[MODEL_FAST])
        self.touch_model("spark-x2.5-1.7b-q8_0-extra.gguf")
        self.assertEqual(paths.resolve_model_path(MODEL_FILE_NAMES[MODEL_FAST]), exact)

    def test_ignores_non_gguf_files(self) -> None:
        """非 GGUF 文件不参与匹配。"""
        self.touch_model("spark-x2.5-1.7b-notes.txt")
        self.assertFalse(paths.resolve_model_path(MODEL_FILE_NAMES[MODEL_FAST]).is_file())

    def test_missing_returns_expected_path(self) -> None:
        """未找到时返回期望路径，供错误提示告诉用户放哪里。

        不能返回一个随机的占位路径——用户会照着它去找一个根本不存在的位置。
        """
        resolved = paths.resolve_model_path(MODEL_FILE_NAMES[MODEL_FAST])
        self.assertFalse(resolved.is_file())
        self.assertEqual(resolved.name, MODEL_FILE_NAMES[MODEL_FAST])

    def test_find_model_file_returns_none_when_missing(self) -> None:
        """``find_model_file`` 在缺失时返回 None（不猜测）。"""
        self.assertIsNone(paths.find_model_file(MODEL_FILE_NAMES[MODEL_FAST]))

    def test_find_model_file_returns_path_when_present(self) -> None:
        """存在时返回真实路径。"""
        expected = self.touch_model(MODEL_FILE_NAMES[MODEL_ACCURATE])
        self.assertEqual(paths.find_model_file(MODEL_FILE_NAMES[MODEL_ACCURATE]), expected)

    def test_empty_dir_does_not_raise(self) -> None:
        """目录为空时不抛异常。"""
        self.assertFalse(paths.resolve_model_path(MODEL_FILE_NAMES[MODEL_FAST]).is_file())

    def test_nonexistent_dir_does_not_raise(self) -> None:
        """目录不存在时同样不抛异常（首次运行时的常见状态）。"""
        env = mock.patch.dict(
            os.environ, {"PINGYITONG_MODELS_DIR": str(self.root / "从来没有过")}
        )
        env.start()
        try:
            self.assertIsNone(paths.find_model_file(MODEL_FILE_NAMES[MODEL_FAST]))
        finally:
            env.stop()


class TestSearchDirectories(unittest.TestCase):
    """真实查找目录的组成与优先级。

    这个类**不替换** ``models_search_dirs``——它验证的正是该函数本身。
    断言只涉及"包含哪些目录、顺序如何"，与目录里是否有文件无关，
    因此不受开发机状态影响。
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_env_override_is_first(self) -> None:
        """环境变量指定的目录优先级最高。"""
        override = self.root / "custom_models"
        with mock.patch.dict(os.environ, {"PINGYITONG_MODELS_DIR": str(override)}):
            directories = paths.models_search_dirs()
        self.assertEqual(directories[0], override)

    def test_project_dir_is_included(self) -> None:
        """项目内 models/ 应在查找范围内。

        这是源码运行场景的主要位置——仓库自带权重后直接可用。
        """
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PINGYITONG_MODELS_DIR", None)
            directories = paths.models_search_dirs()
        self.assertIn(paths.project_models_dir(), directories)

    def test_appdata_dir_is_included(self) -> None:
        """应用数据目录也应在查找范围内（安装场景）。"""
        with mock.patch.dict(os.environ, {"PINGYITONG_HOME": str(self.root)}):
            directories = paths.models_search_dirs()
        expected = Path(self.root) / "models"
        self.assertIn(expected, directories)

    def test_no_duplicates(self) -> None:
        """目录列表不应有重复项。

        环境变量恰好指向项目内目录时最容易产生重复，
        重复会让同一目录被扫描两次。
        """
        with mock.patch.dict(
            os.environ, {"PINGYITONG_MODELS_DIR": str(paths.project_models_dir())}
        ):
            directories = paths.models_search_dirs()
        self.assertEqual(len(directories), len(set(directories)))

    def test_priority_order(self) -> None:
        """优先级顺序为：环境变量 > 项目内 > 应用数据目录。"""
        override = self.root / "custom"
        with mock.patch.dict(
            os.environ,
            {"PINGYITONG_MODELS_DIR": str(override), "PINGYITONG_HOME": str(self.root)},
        ):
            directories = paths.models_search_dirs()

        project_dir = paths.project_models_dir()
        appdata_dir = Path(self.root) / "models"

        if project_dir in directories:
            self.assertLess(directories.index(override), directories.index(project_dir))
        if appdata_dir in directories and project_dir in directories:
            self.assertLess(
                directories.index(project_dir), directories.index(appdata_dir)
            )


class TestMultiLocationLookup(ModelPathsTestCase):
    """多位置查找的具体行为。"""

    def test_appdata_model_found_when_not_in_project(self) -> None:
        """权重只在应用数据目录时也应能找到（安装场景）。"""
        expected = self.touch_model(MODEL_FILE_NAMES[MODEL_FAST], self.appdata_dir)
        self.assertEqual(paths.resolve_model_path(MODEL_FILE_NAMES[MODEL_FAST]), expected)

    def test_project_dir_takes_priority(self) -> None:
        """两个目录都有同名文件时，优先使用项目内的。"""
        project_model = self.touch_model(MODEL_FILE_NAMES[MODEL_FAST], self.models_dir)
        self.touch_model(MODEL_FILE_NAMES[MODEL_FAST], self.appdata_dir)
        self.assertEqual(
            paths.resolve_model_path(MODEL_FILE_NAMES[MODEL_FAST]), project_model
        )

    def test_falls_through_to_second_dir(self) -> None:
        """第一个目录里没有时继续查下一个目录。"""
        expected = self.touch_model(MODEL_FILE_NAMES[MODEL_ACCURATE], self.appdata_dir)
        self.assertEqual(
            paths.resolve_model_path(MODEL_FILE_NAMES[MODEL_ACCURATE]), expected
        )


class TestPreloadTargets(ModelPathsTestCase):
    """预加载目标的选择。"""

    def _pool(self, config: AppConfig | None = None) -> ModelPool:
        return ModelPool(config.model if config else AppConfig().model)

    def test_fast_model_targeted_when_weight_present(self) -> None:
        """权重存在且配置允许时，极速模型应被列为预加载目标。"""
        self.touch_model(MODEL_FILE_NAMES[MODEL_FAST])
        self.assertIn(MODEL_FAST, self._pool().preload_targets())

    def test_missing_weight_excluded(self) -> None:
        """权重缺失的模型不应被列为预加载目标。

        否则启动时会去等一个必然失败的加载，用户白等数秒。
        """
        # 目录为空，两个模型都缺权重
        self.assertEqual(self._pool().preload_targets(), [])

    def test_accurate_preloaded_when_resources_allow(self) -> None:
        """资源充足时应把高精度模型也预加载。

        这是降低翻译延迟的关键：模型切换要卸载旧模型、加载新模型并预热，
        实测 3–10 秒。两个模型都常驻后，切换只是换一个已加载好的引擎来用，
        耗时接近零。
        """
        self.touch_model(MODEL_FILE_NAMES[MODEL_FAST])
        self.touch_model(MODEL_FILE_NAMES[MODEL_ACCURATE])
        pool = self._pool()
        with mock.patch.object(pool, "_can_hold_both_models", return_value=True):
            targets = pool.preload_targets()
        self.assertIn(MODEL_ACCURATE, targets, "资源充足时应同时预加载高精度模型")
        self.assertIn(MODEL_FAST, targets)

    def test_accurate_not_preloaded_when_resources_tight(self) -> None:
        """资源不足时只预加载极速模型。

        内存吃满导致换页的卡顿比模型加载更严重，因此宁可保持按需加载。
        """
        self.touch_model(MODEL_FILE_NAMES[MODEL_FAST])
        self.touch_model(MODEL_FILE_NAMES[MODEL_ACCURATE])
        pool = self._pool()
        with mock.patch.object(pool, "_can_hold_both_models", return_value=False):
            targets = pool.preload_targets()
        self.assertNotIn(MODEL_ACCURATE, targets)
        self.assertIn(MODEL_FAST, targets)

    def test_accurate_can_be_forced_on(self) -> None:
        """用户显式开启预加载时，即使资源判定为紧张也要加载。"""
        self.touch_model(MODEL_FILE_NAMES[MODEL_FAST])
        self.touch_model(MODEL_FILE_NAMES[MODEL_ACCURATE])

        config = AppConfig()
        config.model.accurate.preload = True
        pool = self._pool(config)
        with mock.patch.object(pool, "_can_hold_both_models", return_value=False):
            self.assertIn(MODEL_ACCURATE, pool.preload_targets())

    def test_missing_weight_blocks_both_preload(self) -> None:
        """缺任一权重时不应判定为"可同时常驻"。"""
        self.touch_model(MODEL_FILE_NAMES[MODEL_FAST])
        # 故意不创建 4B 权重
        pool = self._pool()
        self.assertFalse(pool._can_hold_both_models())

    def test_disabled_model_excluded(self) -> None:
        """禁用或关闭预加载的模型应被排除。"""
        self.touch_model(MODEL_FILE_NAMES[MODEL_FAST])

        config = AppConfig()
        config.model.fast.preload = False
        self.assertEqual(self._pool(config).preload_targets(), [])

        config = AppConfig()
        config.model.fast.enabled = False
        self.assertEqual(self._pool(config).preload_targets(), [])

    def test_preloaded_models_exempt_from_idle_unload(self) -> None:
        """预加载常驻的模型不应被空闲回收卸载。

        卸载后再次使用又要重新加载数秒，正是"双模型常驻"要消除的开销。
        """
        from pingyitong.components.llm.model_pool import ModelPool

        self.touch_model(MODEL_FILE_NAMES[MODEL_FAST])
        self.touch_model(MODEL_FILE_NAMES[MODEL_ACCURATE])
        pool = ModelPool(AppConfig().model, idle_timeout=0.0)
        pool._preloaded.update({MODEL_FAST, MODEL_ACCURATE})

        for model_id in (MODEL_FAST, MODEL_ACCURATE):
            engine = pool.engine(model_id)
            engine._process = mock.MagicMock()
            engine._process.poll.return_value = None
            engine._loaded_at = 0.0
            engine._last_used = 0.0

        self.assertEqual(pool.unload_idle(), [], "常驻模型不应被空闲回收")

        for model_id in (MODEL_FAST, MODEL_ACCURATE):
            pool.engine(model_id)._process = None


class TestPreloadProgress(ModelPathsTestCase):
    """预加载进度对象。"""

    def test_ratio_computation(self) -> None:
        """进度比例计算。"""
        progress = PreloadProgress(total=4, completed=1)
        self.assertAlmostEqual(progress.ratio, 0.25)

    def test_ratio_with_zero_total(self) -> None:
        """总数为 0 时不应除零。"""
        self.assertEqual(PreloadProgress(total=0).ratio, 0.0)

    def test_text_while_loading(self) -> None:
        """加载中的描述应包含模型名与序号。"""
        progress = PreloadProgress(model_id=MODEL_FAST, completed=0, total=2)
        text = progress.as_text()
        self.assertIn("正在加载", text)
        self.assertIn("1/2", text)

    def test_text_when_finished(self) -> None:
        """完成后的描述。"""
        progress = PreloadProgress(total=1, completed=1, finished=True, loaded=[MODEL_FAST])
        self.assertIn("就绪", progress.as_text())

    def test_text_when_finished_with_failures(self) -> None:
        """有失败项时描述应体现出来。"""
        progress = PreloadProgress(
            total=1, completed=1, finished=True, failed={MODEL_FAST: "缺少依赖"}
        )
        self.assertIn("未就绪", progress.as_text())


class TestPreloadAsync(ModelPathsTestCase):
    """异步预加载行为。"""

    def test_returns_false_without_targets(self) -> None:
        """没有可加载目标时不启动线程。"""
        pool = ModelPool(AppConfig().model)
        self.assertFalse(pool.preload_async())

    def test_does_not_block_and_reports_completion(self) -> None:
        """应在后台线程中执行并汇报完成结果。

        这里刻意让权重文件存在但依赖缺失（测试环境没有 llama-cpp-python），
        于是会走到失败汇报分支——正好覆盖"不可用时如实报告"的行为。
        """
        self.touch_model(MODEL_FILE_NAMES[MODEL_FAST])
        pool = ModelPool(AppConfig().model)

        finished: list[PreloadProgress] = []
        started = pool.preload_async(on_finished=finished.append)

        self.assertTrue(started, "应启动预加载")
        self.assertTrue(pool.wait_for_preload(timeout=30.0), "预加载应在超时前结束")

        self.assertEqual(len(finished), 1)
        progress = finished[0]
        self.assertTrue(progress.finished)
        self.assertEqual(progress.completed, 1)
        # 依赖缺失时不应报告为已加载
        self.assertEqual(progress.loaded, [])
        self.assertTrue(progress.failed, "不可用时必须记录原因")

    def test_second_call_ignored_while_running(self) -> None:
        """重复调用不应启动第二个加载线程。"""
        self.touch_model(MODEL_FILE_NAMES[MODEL_FAST])
        pool = ModelPool(AppConfig().model)

        first = pool.preload_async()
        second = pool.preload_async()
        # 第一次启动后，第二次可能因为已完成而返回 True/False，但绝不应并发两个线程
        if first:
            self.assertFalse(second, "已在加载中时应忽略重复请求")

        pool.wait_for_preload(timeout=30.0)

    def test_progress_callback_invoked(self) -> None:
        """应至少回调一次进度（开始与结束各一次）。"""
        self.touch_model(MODEL_FILE_NAMES[MODEL_FAST])
        pool = ModelPool(AppConfig().model)

        snapshots: list[PreloadProgress] = []
        pool.preload_async(on_progress=snapshots.append)
        pool.wait_for_preload(timeout=30.0)

        self.assertGreaterEqual(len(snapshots), 2)

    def test_exception_in_callback_does_not_break_loading(self) -> None:
        """进度回调抛异常不应导致加载中断。

        回调通常来自界面层，界面出问题不该让模型加载失败。
        """
        self.touch_model(MODEL_FILE_NAMES[MODEL_FAST])
        pool = ModelPool(AppConfig().model)

        def bad_callback(progress: PreloadProgress) -> None:
            raise RuntimeError("界面回调炸了")

        finished: list[PreloadProgress] = []
        pool.preload_async(on_progress=bad_callback, on_finished=finished.append)
        self.assertTrue(pool.wait_for_preload(timeout=30.0))
        self.assertEqual(len(finished), 1, "回调异常不应阻止完成通知")


class TestEngineFallback(ModelPathsTestCase):
    """权重缺失时的降级行为。"""

    def test_missing_weight_yields_preview_engine(self) -> None:
        """权重缺失时引擎应降级为预览模式，而不是抛异常。

        这是"界面如实标注、不伪装翻译结果"的前提：程序必须能起来，
        并在预览模式下把〔未加载模型〕标记画在正确的位置上。
        """
        pool = ModelPool(AppConfig().model)
        engine = pool.engine(MODEL_FAST)

        self.assertFalse(engine.is_available())
        status = engine.status()
        self.assertFalse(status.available)

    def test_preview_engine_marks_output(self) -> None:
        """预览模式的输出必须带明确标记，不能伪装成翻译结果。"""
        pool = ModelPool(AppConfig().model)
        engine = pool.engine(MODEL_FAST)

        results = engine.translate_batch(["Hello world"], "system")
        self.assertEqual(len(results), 1)
        self.assertIn("未加载模型", results[0])
        self.assertIn("Hello world", results[0])

    def test_any_available_false_without_weights(self) -> None:
        """没有权重时不应报告任何模型可用。"""
        pool = ModelPool(AppConfig().model)
        self.assertFalse(pool.any_available())


if __name__ == "__main__":
    unittest.main(verbosity=2)
