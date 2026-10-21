"""配置与调度测试。

覆盖三类关键行为：

1. **配置的向前兼容**——旧版配置文件缺失新字段时必须能正常加载，
   否则用户升级程序后会遭遇"设置全丢"或启动失败；
2. **配置清洗**——手工编辑配置文件写入越界值时必须被纠正，
   不能让非法参数流入捕获/推理链路；
3. **三级调度决策**——手动模式必须完全遵从用户选择；
   资源约束必须**优先于**内容偏好（保护交互流畅度）；
   并且切换要有冷却，避免模型反复加载卸载。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pingyitong.config.schema import (
    AppConfig,
    StyleConfig,
    from_dict,
    merge_into,
    to_dict,
)
from pingyitong.constants import (
    MODEL_ACCURATE,
    MODEL_FAST,
    SCHEDULE_MODE_AUTO,
    SCHEDULE_MODE_MANUAL,
)
from pingyitong.core.model_scheduler import ModelScheduler
from pingyitong.core.resource_monitor import ResourceSnapshot


class TestSchemaRoundTrip(unittest.TestCase):
    """配置序列化。"""

    def test_default_roundtrip(self) -> None:
        """默认配置序列化后可完整还原。"""
        original = AppConfig()
        restored = from_dict(AppConfig, to_dict(original))
        self.assertEqual(to_dict(original), to_dict(restored))

    def test_nested_roundtrip(self) -> None:
        """嵌套节同样可还原。"""
        config = AppConfig()
        config.style.bg_opacity = 0.33
        config.capture.fps = 24
        restored = from_dict(AppConfig, to_dict(config))
        self.assertAlmostEqual(restored.style.bg_opacity, 0.33)
        self.assertEqual(restored.capture.fps, 24)

    def test_missing_fields_use_defaults(self) -> None:
        """缺失字段回落到默认值（向前兼容的关键）。

        用户升级程序后，旧配置文件不含新加入的字段；
        此时必须能正常启动，而不是报错。
        """
        restored = from_dict(AppConfig, {"style": {"bg_opacity": 0.5}})
        self.assertAlmostEqual(restored.style.bg_opacity, 0.5)
        # 未提供的字段保持默认
        self.assertEqual(restored.capture.fps, AppConfig().capture.fps)
        self.assertEqual(restored.style.max_lines, AppConfig().style.max_lines)

    def test_unknown_fields_ignored(self) -> None:
        """未知字段被忽略而不是报错（降级安装的场景）。"""
        restored = from_dict(AppConfig, {"未来版本的新节": {"x": 1}, "capture": {"fps": 20}})
        self.assertEqual(restored.capture.fps, 20)

    def test_wrong_type_uses_default(self) -> None:
        """类型非法时回落默认值。"""
        restored = from_dict(AppConfig, {"capture": {"fps": "不是数字"}})
        self.assertEqual(restored.capture.fps, AppConfig().capture.fps)

    def test_string_number_coerced(self) -> None:
        """数字被写成字符串时应宽松转换（手改配置文件的常见情形）。"""
        restored = from_dict(AppConfig, {"capture": {"fps": "20"}})
        self.assertEqual(restored.capture.fps, 20)

    def test_boolean_string_coerced(self) -> None:
        """布尔值写成字符串时应正确识别。"""
        restored = from_dict(AppConfig, {"capture": {"enable_diff": "false"}})
        self.assertFalse(restored.capture.enable_diff)
        restored = from_dict(AppConfig, {"capture": {"enable_diff": "true"}})
        self.assertTrue(restored.capture.enable_diff)

    def test_list_field_parsed(self) -> None:
        """列表字段正确解析（翻译区域坐标）。"""
        restored = from_dict(AppConfig, {"capture": {"region": [10, 20, 300, 400]}})
        self.assertEqual(restored.capture.region, [10, 20, 300, 400])

    def test_none_preserved(self) -> None:
        """None 应保持为 None（表示"未设置"而非 0）。"""
        restored = from_dict(AppConfig, {"capture": {"region": None}})
        self.assertIsNone(restored.capture.region)

    def test_null_required_values_use_defaults(self) -> None:
        restored = from_dict(AppConfig, {"capture": {"fps": None}, "hotkey": {"enabled": None}})
        self.assertEqual(restored.capture.fps, AppConfig().capture.fps)
        self.assertTrue(restored.hotkey.enabled)

    def test_nonfinite_numbers_use_defaults(self) -> None:
        for value in (float("inf"), float("nan"), "Infinity", "NaN", "-Infinity"):
            with self.subTest(value=value):
                restored = from_dict(
                    AppConfig, {"capture": {"fps": value}, "ocr": {"min_confidence": value}}
                )
                self.assertEqual(restored.capture.fps, AppConfig().capture.fps)
                self.assertEqual(restored.ocr.min_confidence, AppConfig().ocr.min_confidence)

    def test_invalid_list_items_use_default(self) -> None:
        for region in ([0, 0, "bad", 400], [0, 0, None, 400]):
            with self.subTest(region=region):
                restored = from_dict(AppConfig, {"capture": {"region": region}})
                self.assertIsNone(restored.capture.region)

    def test_partial_model_sections_keep_field_factory_defaults(self) -> None:
        restored = from_dict(
            AppConfig, {"model": {"fast": {"n_ctx": 4096}, "accurate": {"temperature": 0.1}}}
        )
        self.assertEqual(restored.model.fast.n_ctx, 4096)
        self.assertEqual(restored.model.fast.file_name, AppConfig().model.fast.file_name)
        self.assertEqual(restored.model.accurate.file_name, AppConfig().model.accurate.file_name)
        self.assertEqual(restored.model.fast.max_tokens, AppConfig().model.fast.max_tokens)
        self.assertTrue(restored.model.fast.preload)

    def test_invalid_model_section_keeps_field_factory_defaults(self) -> None:
        restored = from_dict(AppConfig, {"model": {"fast": None, "accurate": "bad"}})
        self.assertEqual(to_dict(restored.model), to_dict(AppConfig().model))

    def test_invalid_nested_model_values_keep_field_factory_defaults(self) -> None:
        restored = from_dict(AppConfig, {"model": {
            "fast": {"file_name": None, "max_tokens": None, "preload": "invalid"},
            "accurate": {"file_name": None, "max_tokens": "invalid"},
        }})
        self.assertEqual(to_dict(restored.model), to_dict(AppConfig().model))

    def test_non_dict_input_returns_default(self) -> None:
        """输入不是字典时返回默认配置而不是崩溃。"""
        self.assertEqual(to_dict(from_dict(AppConfig, "坏数据")), to_dict(AppConfig()))
        self.assertEqual(to_dict(from_dict(AppConfig, None)), to_dict(AppConfig()))


class TestMergeInto(unittest.TestCase):
    """配置补丁合并。"""

    def test_single_section_patch(self) -> None:
        """仅更新指定节，其余节保持不变。"""
        base = AppConfig()
        base.capture.fps = 10
        merged = merge_into(base, {"ui": {"theme": "light"}})

        self.assertEqual(merged.ui.theme, "light")
        self.assertEqual(merged.capture.fps, 10)

    def test_original_untouched(self) -> None:
        """合并不修改原对象（避免意外的共享状态）。"""
        base = AppConfig()
        merge_into(base, {"ui": {"theme": "light"}})
        self.assertNotEqual(base.ui.theme, "light")

    def test_nested_patch_keeps_sibling_fields(self) -> None:
        """两级补丁必须只改目标字段，**不得重置同节兄弟字段**。

        这是实际发生过的数据损坏：合并只做一层时，
        ``{"model": {"accurate": {"n_ctx": 2048}}}`` 会把整个 ``accurate``
        子节替换成 ``{"n_ctx": 2048}``，模型文件名回落为空字符串，
        高精度模型随之"消失"。设置面板大量使用这类局部补丁。
        """
        base = AppConfig()
        merged = merge_into(base, {"model": {"accurate": {"n_ctx": 2048}}})

        self.assertEqual(merged.model.accurate.n_ctx, 2048)
        self.assertEqual(
            merged.model.accurate.file_name,
            base.model.accurate.file_name,
            "两级补丁重置了同节的模型文件名",
        )
        self.assertEqual(
            merged.model.accurate.max_tokens,
            base.model.accurate.max_tokens,
        )
        # 快速模型的配置也不应受影响
        self.assertEqual(merged.model.fast.file_name, base.model.fast.file_name)

    def test_translation_partial_patch_keeps_fields(self) -> None:
        """翻译节局部补丁不重置其他翻译设置（面板开关走的路径）。"""
        base = AppConfig()
        base.translation.target_language = "ja"
        base.translation.fallback_source_language = "de"

        merged = merge_into(base, {"translation": {"concise_output": True}})

        self.assertTrue(merged.translation.concise_output)
        self.assertEqual(merged.translation.target_language, "ja")
        self.assertEqual(merged.translation.fallback_source_language, "de")


class TestSanitize(unittest.TestCase):
    """配置清洗（越界值纠正）。"""

    def _manager(self, patch: dict):
        from pingyitong.config import SettingsManager

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as handle:
            json.dump(patch, handle, ensure_ascii=False)

        manager = SettingsManager(Path(handle.name))
        manager.load()
        return manager

    def test_fps_clamped(self) -> None:
        """帧率被夹到合法区间，避免 0fps 或 9999fps 拖垮系统。"""
        manager = self._manager({"capture": {"fps": 9999}})
        from pingyitong.constants import MAX_CAPTURE_FPS

        self.assertLessEqual(manager.config.capture.fps, MAX_CAPTURE_FPS)

        manager = self._manager({"capture": {"fps": -5}})
        from pingyitong.constants import MIN_CAPTURE_FPS

        self.assertGreaterEqual(manager.config.capture.fps, MIN_CAPTURE_FPS)

    def test_opacity_clamped(self) -> None:
        """不透明度必须落在 0-1，否则遮罩会渲染异常。"""
        self.assertLessEqual(self._manager({"style": {"bg_opacity": 5.0}}).config.style.bg_opacity, 1.0)
        self.assertGreaterEqual(self._manager({"style": {"bg_opacity": -1.0}}).config.style.bg_opacity, 0.0)

    def test_invalid_region_reset(self) -> None:
        """区域格式非法时重置为全屏，避免捕获到非法矩形。"""
        manager = self._manager({"capture": {"region": [1, 2], "mode": "region"}})
        self.assertIsNone(manager.config.capture.region)
        self.assertEqual(manager.config.capture.mode, "fullscreen")

    def test_nonpositive_region_size_reset(self) -> None:
        for region in ([0, 0, 0, 400], [0, 0, 300, -1]):
            with self.subTest(region=region):
                manager = self._manager({"capture": {"region": region, "mode": "region"}})
                self.assertIsNone(manager.config.capture.region)
                self.assertEqual(manager.config.capture.mode, "fullscreen")

    def test_region_mode_without_valid_region_resets(self) -> None:
        manager = self._manager({"capture": {"region": [0, 0, "bad", 400], "mode": "region"}})
        self.assertEqual(manager.config.capture.mode, "fullscreen")

    def test_invalid_floating_bar_position_resets(self) -> None:
        manager = self._manager({"ui": {"floating_bar_pos": [10]}})
        self.assertIsNone(manager.config.ui.floating_bar_pos)

    def test_bom_config_keeps_user_settings(self) -> None:
        from pingyitong.config import SettingsManager

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text('{"capture": {"fps": 20}}', encoding="utf-8-sig")
            manager = SettingsManager(path)
            self.assertEqual(manager.load().capture.fps, 20)

    def test_undecodable_config_falls_back_and_import_returns_false(self) -> None:
        from pingyitong.config import SettingsManager

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_bytes(b"\xffinvalid")
            manager = SettingsManager(path)
            self.assertEqual(manager.load().capture.fps, AppConfig().capture.fps)
            self.assertTrue(path.with_suffix(".json.bak").exists())
            self.assertFalse(manager.import_from(path))

    def test_min_max_font_size_ordered(self) -> None:
        """最小字号不得大于最大字号，否则排版会退化。"""
        manager = self._manager({"style": {"min_font_size": 40, "max_font_size": 10}})
        self.assertLessEqual(
            manager.config.style.min_font_size, manager.config.style.max_font_size
        )

    def test_invalid_theme_reset(self) -> None:
        """未知主题名回退为深色，避免样式表生成失败。"""
        manager = self._manager({"ui": {"theme": "紫色主题"}})
        self.assertEqual(manager.config.ui.theme, "dark")

    def test_invalid_schedule_mode_reset(self) -> None:
        """未知调度模式回退为手动模式。"""
        manager = self._manager({"schedule": {"mode": "随机"}})
        self.assertEqual(manager.config.schedule.mode, SCHEDULE_MODE_MANUAL)

    def test_same_language_pair_preserves_explicit_source(self) -> None:
        """同语方向也保留显式源语言，让用户可以先改源再改目标。"""
        manager = self._manager(
            {"translation": {"source_language": "zh", "target_language": "zh"}}
        )
        self.assertEqual(manager.config.translation.source_language, "zh")

    def test_broken_json_falls_back(self) -> None:
        """配置文件损坏时应备份并回退默认配置，保证能启动。"""
        from pingyitong.config import SettingsManager

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as handle:
            handle.write("{ 这不是合法 JSON")

        manager = SettingsManager(Path(handle.name))
        config = manager.load()
        self.assertEqual(to_dict(config), to_dict(AppConfig()))


class TestSettingsPersistence(unittest.TestCase):
    """配置持久化。"""

    def test_save_and_reload(self) -> None:
        """保存后重新加载应得到相同配置。"""
        from pingyitong.config import SettingsManager

        directory = Path(tempfile.mkdtemp())
        path = directory / "config.json"

        manager = SettingsManager(path)
        manager.load()
        manager.update({"ui": {"theme": "light"}, "capture": {"fps": 20}})

        reloaded = SettingsManager(path)
        reloaded.load()
        self.assertEqual(reloaded.config.ui.theme, "light")
        self.assertEqual(reloaded.config.capture.fps, 20)

    def test_section_reset(self) -> None:
        """单节恢复默认。"""
        from pingyitong.config import SettingsManager

        directory = Path(tempfile.mkdtemp())
        manager = SettingsManager(directory / "config.json")
        manager.load()
        manager.update({"style": {"bg_opacity": 0.1}})
        manager.reset_section("style")
        self.assertAlmostEqual(manager.config.style.bg_opacity, StyleConfig().bg_opacity)

    def test_export_import(self) -> None:
        """导出后导入应还原配置（用于备份与迁移）。"""
        from pingyitong.config import SettingsManager

        directory = Path(tempfile.mkdtemp())
        manager = SettingsManager(directory / "config.json")
        manager.load()
        manager.update({"ui": {"notifications": False}})

        target = directory / "export.json"
        self.assertTrue(manager.export_to(target))

        other = SettingsManager(directory / "other.json")
        other.load()
        self.assertTrue(other.import_from(target))
        self.assertFalse(other.config.ui.notifications)


class TestModelScheduler(unittest.TestCase):
    """双模型三级调度。"""

    def _config(self, **overrides):
        config = AppConfig().schedule
        for key, value in overrides.items():
            setattr(config, key, value)
        return config

    def test_manual_mode_always_honours_user(self) -> None:
        """手动模式下即使用户选择"较慢但更准"的模型也必须遵从。

        用户显式选择优先于任何自动化策略——否则用户会觉得设置不生效。
        """
        scheduler = ModelScheduler(
            self._config(mode=SCHEDULE_MODE_MANUAL, active_model=MODEL_ACCURATE)
        )
        busy = ResourceSnapshot(cpu_percent=95.0, memory_percent=90.0, render_fps=3.0)
        decision = scheduler.decide(busy, ["a long sentence " * 20])
        self.assertEqual(decision.model_id, MODEL_ACCURATE)
        self.assertEqual(decision.reason_code, "manual")

    def test_high_latency_degrades_to_fast(self) -> None:
        """端到端延迟过高时降级到 1.7B，保证响应速度。"""
        scheduler = ModelScheduler(
            self._config(mode=SCHEDULE_MODE_AUTO, active_model=MODEL_ACCURATE)
        )
        snapshot = ResourceSnapshot(
            cpu_percent=20.0, render_fps=15.0, end_to_end_latency=5.0
        )
        decision = scheduler.decide(snapshot)
        self.assertEqual(decision.model_id, MODEL_FAST)
        self.assertEqual(decision.reason_code, "resource")
        self.assertIn("延迟", decision.reason)

    def test_acceptable_latency_does_not_degrade(self) -> None:
        """延迟在阈值内时不应降级。"""
        scheduler = ModelScheduler(
            self._config(mode=SCHEDULE_MODE_AUTO, active_model=MODEL_ACCURATE)
        )
        snapshot = ResourceSnapshot(
            cpu_percent=20.0, render_fps=15.0, end_to_end_latency=0.5
        )
        decision = scheduler.decide(snapshot, ["This is a fairly long sentence " * 5])
        self.assertEqual(decision.model_id, MODEL_ACCURATE)

    def test_low_render_fps_does_not_trigger_degrade(self) -> None:
        """渲染帧率低**不得**成为降级依据。

        这是实测踩到的死循环：静态屏幕上没有新内容可渲染，帧率天然趋近 0，
        用降级会把"屏幕没变"误判成"界面卡顿"，于是与内容自适应升级形成
        ``4B→1.7B→4B→1.7B`` 反复切换，每次切换都要重载模型数秒。

        正确的信号是端到端延迟（用户实际在等多久）。
        """
        scheduler = ModelScheduler(
            self._config(mode=SCHEDULE_MODE_AUTO, active_model=MODEL_ACCURATE)
        )
        # 帧率几乎为 0（静态屏幕），但延迟正常 → 不应降级
        snapshot = ResourceSnapshot(
            cpu_percent=20.0, render_fps=0.2, end_to_end_latency=0.3
        )
        decision = scheduler.decide(snapshot)
        self.assertEqual(
            decision.model_id, MODEL_ACCURATE,
            "渲染帧率低不是降级依据——静态屏幕天然帧率低",
        )

    def test_no_flip_flop_between_content_and_resource(self) -> None:
        """内容自适应与资源自适应不得形成来回切换。

        模拟实测场景：静态屏幕上延迟正常、帧率极低，同时来了一段长文本。
        调度器应稳定选择高精度模型，而不是被帧率拉回 1.7B。
        """
        scheduler = ModelScheduler(
            self._config(mode=SCHEDULE_MODE_AUTO, content_adaptive=True, active_model=MODEL_FAST)
        )
        snapshot = ResourceSnapshot(
            cpu_percent=20.0, render_fps=0.3, end_to_end_latency=0.4
        )
        long_text = "This is a fairly long sentence that should use the accurate model. " * 3

        decision = scheduler.decide(snapshot, [long_text])
        self.assertEqual(decision.model_id, MODEL_ACCURATE)

        # 紧接着再来一帧同样的输入：应保持 4B，不应翻回 1.7B
        decision = scheduler.decide(snapshot, [long_text])
        self.assertEqual(decision.model_id, MODEL_ACCURATE, "不应因低帧率翻回极速模型")
        self.assertFalse(decision.switched)

    def test_switch_cooldown_exceeds_switch_cost(self) -> None:
        """切换冷却期必须长于切换本身的成本。

        4B 的加载 + 预热实测 3–10 秒。若冷却期短于这个成本，就会出现
        "刚切完又切回去"的抖动，用户看到的是一段段空翻译窗。
        """
        from pingyitong.core.model_scheduler import MIN_SWITCH_INTERVAL

        self.assertGreaterEqual(
            MIN_SWITCH_INTERVAL, 30.0,
            "冷却期应不小于 30 秒，否则模型抖动比不切换更糟",
        )

    def test_sustained_cpu_degrades_to_fast(self) -> None:
        """CPU 持续高负载时降级。"""
        scheduler = ModelScheduler(
            self._config(mode=SCHEDULE_MODE_AUTO, active_model=MODEL_ACCURATE)
        )
        snapshot = ResourceSnapshot(cpu_percent=85.0, render_fps=15.0)
        decision = scheduler.decide(snapshot, cpu_sustained_high=True)
        self.assertEqual(decision.model_id, MODEL_FAST)

    def test_transient_cpu_spike_does_not_degrade(self) -> None:
        """瞬时 CPU 峰值不触发降级（避免误判）。"""
        scheduler = ModelScheduler(
            self._config(mode=SCHEDULE_MODE_AUTO, active_model=MODEL_ACCURATE)
        )
        snapshot = ResourceSnapshot(cpu_percent=85.0, render_fps=15.0)
        decision = scheduler.decide(snapshot, cpu_sustained_high=False)
        self.assertEqual(decision.model_id, MODEL_ACCURATE)

    def test_resource_constraint_beats_content_preference(self) -> None:
        """资源约束优先于内容偏好。

        长文本本应使用高精度模型，但 CPU 已持续高负载时必须降级——
        宁可译文质量略降，也不能让用户的电脑卡顿。
        """
        scheduler = ModelScheduler(
            self._config(mode=SCHEDULE_MODE_AUTO, active_model=MODEL_ACCURATE)
        )
        snapshot = ResourceSnapshot(cpu_percent=95.0, render_fps=15.0)
        decision = scheduler.decide(
            snapshot, ["a very long sentence " * 20], cpu_sustained_high=True
        )
        self.assertEqual(decision.model_id, MODEL_FAST)

    def test_long_text_upgrades_to_accurate(self) -> None:
        """长句在资源充足时升级到高精度模型。"""
        scheduler = ModelScheduler(
            self._config(mode=SCHEDULE_MODE_AUTO, content_adaptive=True, active_model=MODEL_FAST)
        )
        snapshot = ResourceSnapshot(cpu_percent=10.0, render_fps=15.0)
        long_text = "This is a fairly long sentence. " * 5
        decision = scheduler.decide(snapshot, [long_text])
        self.assertEqual(decision.model_id, MODEL_ACCURATE)

    def test_short_text_uses_fast_model(self) -> None:
        """达到长度门槛的短句使用极速模型。

        注意这里用的不是 ``"OK"`` 这类超短文本——那种长度不值得为切换
        付出模型加载成本，调度器会沿用当前模型（见
        :meth:`test_tiny_text_does_not_trigger_switch`）。
        """
        scheduler = ModelScheduler(
            self._config(mode=SCHEDULE_MODE_AUTO, content_adaptive=True, active_model=MODEL_ACCURATE)
        )
        snapshot = ResourceSnapshot(cpu_percent=10.0, render_fps=15.0)
        decision = scheduler.decide(snapshot, ["Download complete"])
        self.assertEqual(decision.model_id, MODEL_FAST)

    def test_switch_cooldown_prevents_thrashing(self) -> None:
        """切换后进入冷却期，避免模型反复加载卸载。

        每次加载 4B 模型需要数秒，抖动会让体验比不切换更差。
        """
        scheduler = ModelScheduler(
            self._config(mode=SCHEDULE_MODE_AUTO, active_model=MODEL_ACCURATE)
        )
        snapshot = ResourceSnapshot(cpu_percent=95.0, render_fps=15.0)
        first = scheduler.decide(snapshot, cpu_sustained_high=True)
        self.assertTrue(first.switched)

        # 立即再次请求反向切换，应被冷却期拦下
        calm = ResourceSnapshot(cpu_percent=5.0, render_fps=20.0)
        second = scheduler.decide(calm, ["very long text " * 30])
        self.assertFalse(second.switched)
        self.assertEqual(second.model_id, MODEL_FAST)

    def test_toggle_model(self) -> None:
        """两个模型之间互相切换。"""
        scheduler = ModelScheduler(self._config(active_model=MODEL_FAST))
        self.assertEqual(scheduler.toggle_model(), MODEL_ACCURATE)
        self.assertEqual(scheduler.toggle_model(), MODEL_FAST)

    def test_force_model(self) -> None:
        """强制指定模型。"""
        scheduler = ModelScheduler(self._config())
        scheduler.force_model(MODEL_FAST)
        self.assertEqual(scheduler.current_model, MODEL_FAST)

    def test_switch_count_tracked(self) -> None:
        """切换次数被统计（用于状态面板）。"""
        scheduler = ModelScheduler(
            self._config(mode=SCHEDULE_MODE_AUTO, active_model=MODEL_ACCURATE)
        )
        snapshot = ResourceSnapshot(cpu_percent=95.0, render_fps=15.0)
        scheduler.decide(snapshot, cpu_sustained_high=True)
        self.assertGreaterEqual(scheduler.switch_count, 1)
        self.assertIn("自动调度", scheduler.stats_text())

    def test_disabled_resource_adaptive_skips(self) -> None:
        """关闭资源自适应后不因负载降级。"""
        scheduler = ModelScheduler(
            self._config(
                mode=SCHEDULE_MODE_AUTO,
                active_model=MODEL_ACCURATE,
                resource_adaptive=False,
            )
        )
        snapshot = ResourceSnapshot(cpu_percent=99.0, render_fps=2.0)
        decision = scheduler.decide(snapshot, cpu_sustained_high=True)
        self.assertNotEqual(decision.reason_code, "resource")

    def test_decision_text_is_human_readable(self) -> None:
        """决策理由应可直接展示给用户。"""
        scheduler = ModelScheduler(self._config(mode=SCHEDULE_MODE_MANUAL))
        decision = scheduler.decide(None)
        self.assertIn("手动", decision.as_text())

    def test_tiny_text_does_not_trigger_switch(self) -> None:
        """过短的文本不应触发模型切换。

        实测问题：屏幕上一段 1 个字符的 OCR 碎片会让调度器判定
        "短句文本（平均 1 字符）"，把已加载的 4B 换回 1.7B——
        白白付出一次卸载与加载（数秒），而 1 个字符用哪个模型都没差别。
        """
        scheduler = ModelScheduler(
            self._config(mode=SCHEDULE_MODE_AUTO, content_adaptive=True, active_model=MODEL_ACCURATE)
        )
        snapshot = ResourceSnapshot(cpu_percent=10.0, render_fps=15.0)

        for text in ("1", "OK", "a", "42"):
            with self.subTest(text=text):
                decision = scheduler.decide(snapshot, [text])
                self.assertFalse(decision.switched, f"{text!r} 不应触发切换")
                self.assertEqual(decision.model_id, MODEL_ACCURATE)
                self.assertIn("过短", decision.reason)

    def test_short_but_meaningful_text_still_downgrades(self) -> None:
        """达到长度门槛的短句仍应能触发降级（门槛不能把功能废掉）。"""
        scheduler = ModelScheduler(
            self._config(mode=SCHEDULE_MODE_AUTO, content_adaptive=True, active_model=MODEL_ACCURATE)
        )
        snapshot = ResourceSnapshot(cpu_percent=10.0, render_fps=15.0)
        # 20 字符：超过门槛，且远低于长句阈值（80）
        decision = scheduler.decide(snapshot, ["Download complete now"])
        self.assertEqual(decision.model_id, MODEL_FAST)
        self.assertTrue(decision.switched)

    def test_long_text_still_upgrades(self) -> None:
        """长句仍应升级到高精度模型（门槛不影响原有能力）。"""
        scheduler = ModelScheduler(
            self._config(mode=SCHEDULE_MODE_AUTO, content_adaptive=True, active_model=MODEL_FAST)
        )
        snapshot = ResourceSnapshot(cpu_percent=10.0, render_fps=15.0)
        decision = scheduler.decide(snapshot, ["This is a fairly long sentence. " * 5])
        self.assertEqual(decision.model_id, MODEL_ACCURATE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
