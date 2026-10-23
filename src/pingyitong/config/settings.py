"""配置读写与变更通知。

配置以 JSON 文件持久化于应用数据目录，重启自动恢复。
所有修改通过 :class:`SettingsManager` 进行，变更后广播信号，
使各模块（调度器、渲染层、UI）能即时响应而无需轮询。
"""

from __future__ import annotations

import contextlib
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QObject, pyqtSignal

from ..constants import MODEL_ACCURATE, SCHEDULE_MODE_MANUAL
from ..utils import paths
from ..utils.logger import get_logger
from .schema import AppConfig, from_dict, merge_into, to_dict

logger = get_logger(__name__)


class SettingsManager(QObject):
    """配置管理器（单例式使用，由 :class:`~pingyitong.app.Application` 持有）。"""

    #: 任意配置变更后发出，携带新的 :class:`AppConfig`
    changed = pyqtSignal(object)
    #: 指定节变更，参数为节名（如 ``"style"``、``"ui"``）
    section_changed = pyqtSignal(str)

    def __init__(self, path: Path | None = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._path = path or paths.config_path()
        self._config = AppConfig()
        self._listeners: dict[str, list[Callable[[AppConfig], None]]] = {}
        self._suspend_notify = False

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def config(self) -> AppConfig:
        """当前配置对象。

        返回内部实例，调用方不应直接修改；请使用
        :meth:`update` / :meth:`set_section` 以便触发持久化与通知。
        """
        return self._config

    @property
    def path(self) -> Path:
        """配置文件路径。"""
        return self._path

    # ------------------------------------------------------------------
    # 读写
    # ------------------------------------------------------------------

    def load(self) -> AppConfig:
        """从磁盘加载配置。文件不存在或损坏时使用默认配置。

        :return: 加载后的配置对象。
        """
        if not self._path.exists():
            logger.info("未找到配置文件，使用默认配置：%s", self._path)
            self._config = AppConfig()
            self._apply_env_overrides()
            return self._config

        try:
            raw = json.loads(self._path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            logger.warning("配置文件解析失败（%s），已备份并回退默认配置", exc)
            self._backup_broken_file()
            self._config = AppConfig()
            self._apply_env_overrides()
            return self._config

        self._config = from_dict(AppConfig, raw)
        self._apply_env_overrides()
        self._sanitize()
        logger.info("配置加载完成：%s", self._path)
        return self._config

    def save(self) -> bool:
        """将当前配置写回磁盘。

        采用"写临时文件再替换"的方式，避免写入过程中断导致配置损坏。

        :return: 写入成功返回 ``True``。
        """
        self._sanitize()
        payload = to_dict(self._config)
        tmp = self._path.with_suffix(".json.tmp")
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp.replace(self._path)
        except OSError as exc:
            logger.error("配置保存失败：%s", exc)
            return False
        logger.debug("配置已保存：%s", self._path)
        return True

    # ------------------------------------------------------------------
    # 修改
    # ------------------------------------------------------------------

    def update(self, patch: dict[str, Any], notify: bool = True) -> AppConfig:
        """按节合并配置补丁并持久化。

        :param patch: 形如 ``{"ui": {"theme": "light"}}`` 的字典。
        :param notify: 是否广播变更信号。
        """
        self._config = merge_into(self._config, patch)
        self._sanitize()
        self.save()
        if notify:
            self._emit_changed(list(patch.keys()))
        return self._config

    def set_section(self, section: str, values: dict[str, Any]) -> AppConfig:
        """更新单个配置节。"""
        return self.update({section: values})

    def reset_section(self, section: str) -> AppConfig:
        """将指定配置节恢复为默认值。"""
        defaults = to_dict(AppConfig())
        if section not in defaults:
            logger.warning("未知配置节：%s", section)
            return self._config
        self._config = from_dict(AppConfig, {**to_dict(self._config), section: defaults[section]})
        self._sanitize()
        self.save()
        self._emit_changed([section])
        return self._config

    def reset_all(self) -> AppConfig:
        """恢复全部默认配置。"""
        self._config = AppConfig()
        self._apply_env_overrides()
        self.save()
        self._emit_changed(["*"])
        return self._config

    def export_to(self, target: Path) -> bool:
        """导出配置到指定文件，便于用户备份或在设备间迁移。"""
        try:
            target.write_text(
                json.dumps(to_dict(self._config), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return True
        except OSError as exc:
            logger.error("配置导出失败：%s", exc)
            return False

    def import_from(self, source: Path) -> bool:
        """从指定文件导入配置。"""
        try:
            raw = json.loads(source.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            logger.error("配置导入失败：%s", exc)
            return False
        self._config = from_dict(AppConfig, raw)
        self._sanitize()
        self.save()
        self._emit_changed(["*"])
        return True

    # ------------------------------------------------------------------
    # 订阅
    # ------------------------------------------------------------------

    def subscribe(self, section: str, callback: Callable[[AppConfig], None]) -> None:
        """订阅某个配置节的变更。

        :param section: 配置节名；``"*"`` 表示订阅全部变更。
        """
        self._listeners.setdefault(section, []).append(callback)

    def unsubscribe(self, section: str, callback: Callable[[AppConfig], None]) -> None:
        """取消订阅。"""
        listeners = self._listeners.get(section)
        if not listeners:
            return
        with contextlib.suppress(ValueError):
            listeners.remove(callback)

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _emit_changed(self, sections: list[str]) -> None:
        if self._suspend_notify:
            return
        self.changed.emit(self._config)
        for section in sections:
            self.section_changed.emit(section)
            for callback in list(self._listeners.get(section, ())):
                try:
                    callback(self._config)
                except Exception:  # noqa: BLE001 - 单个订阅者异常不应影响其他订阅者
                    logger.exception("配置订阅回调执行失败：section=%s", section)
        for callback in list(self._listeners.get("*", ())):
            try:
                callback(self._config)
            except Exception:  # noqa: BLE001
                logger.exception("配置全量订阅回调执行失败")

    def _backup_broken_file(self) -> None:
        with contextlib.suppress(OSError):
            shutil.copy2(self._path, self._path.with_suffix(".json.bak"))

    def _apply_env_overrides(self) -> None:
        """应用仅由环境变量控制的临时覆盖（不写回配置文件）。

        用于开发调试，例如强制使用模拟推理引擎。
        """
        import os

        if os.environ.get("PINGYITONG_MOCK_LLM") == "1":
            logger.warning("已通过环境变量启用模拟翻译引擎（调试模式）")

    def _sanitize(self) -> None:
        """纠正越界或非法取值，保证下游模块拿到安全参数。"""
        cfg = self._config

        from ..constants import MAX_CAPTURE_FPS, MIN_CAPTURE_FPS

        cfg.capture.fps = max(MIN_CAPTURE_FPS, min(MAX_CAPTURE_FPS, int(cfg.capture.fps)))
        cfg.capture.monitor_interval_ms = max(50, min(1000, int(cfg.capture.monitor_interval_ms)))
        if cfg.translation.scene_mode not in {"auto", "subtitle", "dense", "document", "dynamic"}:
            cfg.translation.scene_mode = "auto"
        if cfg.translation.review_policy not in {"always", "risk"}:
            cfg.translation.review_policy = "always"
        cfg.capture.diff_pixel_threshold = max(0, min(255, int(cfg.capture.diff_pixel_threshold)))
        cfg.capture.diff_min_area_ratio = max(0.0, min(1.0, float(cfg.capture.diff_min_area_ratio)))
        cfg.capture.diff_block_size = max(2, int(cfg.capture.diff_block_size))

        if cfg.capture.region is not None and (
            len(cfg.capture.region) != 4
            or cfg.capture.region[2] <= 0
            or cfg.capture.region[3] <= 0
        ):
            logger.warning("捕获区域格式非法，已重置为全屏")
            cfg.capture.region = None
            cfg.capture.mode = "fullscreen"

        if cfg.capture.mode not in {"fullscreen", "region"} or (
            cfg.capture.mode == "region" and cfg.capture.region is None
        ):
            cfg.capture.mode = "fullscreen"

        if cfg.ui.floating_bar_pos is not None and len(cfg.ui.floating_bar_pos) != 2:
            cfg.ui.floating_bar_pos = None

        cfg.ocr.min_confidence = max(0.0, min(1.0, float(cfg.ocr.min_confidence)))
        cfg.ocr.min_box_height = max(1, int(cfg.ocr.min_box_height))
        cfg.ocr.max_blocks_per_frame = max(1, int(cfg.ocr.max_blocks_per_frame))
        cfg.ocr.upscale_factor = max(0.5, min(3.0, float(cfg.ocr.upscale_factor)))
        # 线程数上限保守取 64：超过这个值必然是配置写错，
        # 否则会在识别时创建大量线程，反而严重拖慢（实测见 OcrConfig.threads）
        cfg.ocr.threads = max(0, min(64, int(cfg.ocr.threads)))

        for model_cfg in (cfg.model.fast, cfg.model.accurate):
            model_cfg.n_ctx = max(256, int(model_cfg.n_ctx))
            model_cfg.n_threads = max(0, int(model_cfg.n_threads))
            model_cfg.n_gpu_layers = max(0, int(model_cfg.n_gpu_layers))
            model_cfg.n_batch = max(32, int(model_cfg.n_batch))
            model_cfg.temperature = max(0.0, min(2.0, float(model_cfg.temperature)))
            model_cfg.top_p = max(0.0, min(1.0, float(model_cfg.top_p)))
            model_cfg.max_tokens = max(32, int(model_cfg.max_tokens))

        cfg.model.batch_size = max(1, int(cfg.model.batch_size))
        cfg.model.cache_max_entries = max(0, int(cfg.model.cache_max_entries))
        cfg.model.cache_ttl_seconds = max(0.0, float(cfg.model.cache_ttl_seconds))

        api = cfg.api
        api.base_url = str(api.base_url or "").strip().rstrip("/")
        api.api_key = str(api.api_key or "").strip()
        api.model_name = str(api.model_name or "").strip()
        api.temperature = max(0.0, min(2.0, float(api.temperature)))
        api.max_tokens = max(16, min(8192, int(api.max_tokens)))
        api.timeout_seconds = max(1.0, min(120.0, float(api.timeout_seconds)))
        api.batch_size = max(1, min(24, int(api.batch_size)))
        api.max_concurrency = max(1, min(8, int(api.max_concurrency)))
        # 信息不完整的在线配置不启用：与其让每个请求都失败再回退本地
        # （每帧白付一次超时），不如直接走本地路径
        if api.enabled and (not api.base_url or not api.model_name):
            if api.enabled:
                logger.warning("在线 AI 配置不完整（缺接口地址或模型名），已回退本地模型")
            api.enabled = False

        if cfg.schedule.mode not in {"manual", "auto"}:
            cfg.schedule.mode = SCHEDULE_MODE_MANUAL
        if cfg.schedule.active_model not in {"spark-x2.5-1.7b", "spark-x2.5-4b"}:
            cfg.schedule.active_model = MODEL_ACCURATE
        cfg.schedule.cpu_high_threshold = max(10.0, min(99.0, float(cfg.schedule.cpu_high_threshold)))
        cfg.schedule.long_text_chars = max(10, int(cfg.schedule.long_text_chars))
        cfg.schedule.accurate_idle_timeout = max(
            30.0, float(cfg.schedule.accurate_idle_timeout)
        )

        # Preserve an explicitly selected language, including a same-language
        # pair. Replacing it with auto here loses the user's source when they
        # select the source before changing the previous target (zh -> zh -> en).
        # The translation service already decides whether same-language content
        # needs translation, including mixed-language paragraphs.

        cfg.style.font_scale = max(0.4, min(3.0, float(cfg.style.font_scale)))
        cfg.style.min_font_size = max(12, int(cfg.style.min_font_size))
        cfg.style.max_font_size = max(cfg.style.min_font_size, int(cfg.style.max_font_size))
        cfg.style.font_size_fixed = max(12, int(cfg.style.font_size_fixed))
        cfg.style.bg_opacity = max(0.0, min(1.0, float(cfg.style.bg_opacity)))
        cfg.style.line_spacing = max(0.8, min(2.5, float(cfg.style.line_spacing)))
        cfg.style.max_lines = max(1, int(cfg.style.max_lines))
        if cfg.style.mode not in {"cover", "overlay"}:
            cfg.style.mode = "cover"

        if cfg.ui.theme not in {"dark", "light", "system"}:
            cfg.ui.theme = "dark"
